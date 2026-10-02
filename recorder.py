"""Simple voice recorder: record from the microphone, play it back, save as WAV,
and transcribe speech to text with faster-whisper (afterwards or live while speaking)."""

import glob
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# 16 kHz mono is the standard input format for speech-recognition models.
SAMPLE_RATE = 16000
CHANNELS = 1

# Whisper model sizes, smallest/fastest first. Bigger = slower but more accurate.
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
WHISPER_MODEL = "base"

# Language choices shown in the app; None means Whisper detects the language itself.
LANGUAGES = {"Auto-detect": None, "Danish": "da", "English": "en"}


def _add_cuda_dlls():
    """Make the CUDA libraries from the nvidia-* pip packages (requirements-gpu.txt) findable on Windows."""
    try:
        import nvidia
    except ImportError:
        return
    for package_dir in nvidia.__path__:
        for bin_dir in glob.glob(os.path.join(package_dir, "*", "bin")):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


def _best_device():
    """Use an NVIDIA GPU when there is one, otherwise the CPU."""
    _add_cuda_dlls()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class Transcriber:
    """Wraps faster-whisper. The model is loaded on first use (downloaded the first time)."""

    def __init__(self, model_size=WHISPER_MODEL):
        self.model_size = model_size
        self.language = None  # None = auto-detect
        self.vocabulary = ""  # names/terms that help recognition (Whisper's "initial prompt")
        self.device, self.compute_type = _best_device()
        self._model = None
        # Live and full transcription may run from different threads; use the model one at a time.
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._model is not None

    def configure(self, model_size, language, vocabulary):
        with self._lock:
            if model_size != self.model_size:
                self.model_size = model_size
                self._model = None  # the new model is loaded on next use
            self.language = language
            self.vocabulary = vocabulary

    def segments(self, audio, **options):
        """Transcribe audio and return (list of segments, info). Options go to WhisperModel.transcribe."""
        options.setdefault("language", self.language)
        if self.vocabulary:
            # Whisper copies the prompt's style; without a final period it may drop punctuation.
            options.setdefault("initial_prompt", self.vocabulary.rstrip(".") + ".")
        with self._lock:
            try:
                return self._transcribe(audio, options)
            except RuntimeError:
                if self.device == "cpu":
                    raise
                # CUDA libraries missing or GPU out of memory: fall back to the CPU.
                self.device, self.compute_type = "cpu", "int8"
                self._model = None
                return self._transcribe(audio, options)

    def _transcribe(self, audio, options):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        segments, info = self._model.transcribe(audio.flatten(), **options)
        return list(segments), info

    def transcribe(self, audio):
        segments, info = self.segments(audio, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)
        return text, info.language


class LiveTranscriber:
    """Transcribes a recording while it is in progress.

    About once a second the audio that has not been committed yet (the "window")
    is transcribed and reported as provisional text. When the speaker pauses, or
    the window gets long, the finished text is committed and the window moves
    forward, so each pass only has to transcribe a few seconds of audio.

    Progress is reported on the `updates` queue as tuples:
      ("text", (committed, partial)), ("error", exception) or ("done", None).
    """

    UPDATE_INTERVAL = 1.0  # seconds between transcription passes
    PAUSE = 1.0  # silence after speech (seconds) that finishes a sentence
    MAX_WINDOW = 15.0  # once the window is this long, commit all but the last segment
    HARD_LIMIT = 25.0  # Whisper handles at most 30 s at a time; commit everything here

    def __init__(self, recorder, transcriber):
        self.recorder = recorder
        self.transcriber = transcriber
        self.updates = queue.Queue()
        self._thread = None
        self._stop = threading.Event()

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self.updates = queue.Queue()
        self._committed = []
        self._offset = 0  # samples of the recording already committed
        self._language = self.transcriber.language  # None = detect from the first words
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the worker to finish. It transcribes the rest of the audio, then reports "done"."""
        self._stop.set()

    def _run(self):
        try:
            while not self._stop.wait(self.UPDATE_INTERVAL):
                self._step(final=False)
            self._step(final=True)
        except Exception as e:
            self.updates.put(("error", e))
        self.updates.put(("done", None))

    def _step(self, final):
        audio = self.recorder.snapshot()[self._offset:]
        window = len(audio) / SAMPLE_RATE
        segments = []
        if window >= 0.5:
            segments, info = self.transcriber.segments(
                audio,
                language=self._language,
                beam_size=1,  # greedy decoding: faster, good enough for live text
                vad_filter=True,  # skip silence, which also avoids made-up words
                condition_on_previous_text=False,
                word_timestamps=True,  # precise end of speech, to detect pauses
            )
            # Short clips make language detection jumpy, so keep the first confident guess.
            if self._language is None and segments and info.language_probability > 0.8:
                self._language = info.language

        speech_end = 0.0
        if segments:
            last = segments[-1]
            speech_end = last.words[-1].end if last.words else last.end

        if final or window > self.HARD_LIMIT:
            self._commit(segments, len(audio))
            segments = []
        elif segments and window - speech_end > self.PAUSE:
            # The speaker paused: everything so far is final.
            self._commit(segments, int((speech_end + self.PAUSE / 2) * SAMPLE_RATE))
            segments = []
        elif not segments and window > self.PAUSE:
            # Only silence: drop it, keeping a little in case speech is just starting.
            self._offset += len(audio) - int(self.PAUSE / 2 * SAMPLE_RATE)
        elif window > self.MAX_WINDOW and len(segments) > 1:
            # Long speech without a pause: commit the finished sentences.
            self._commit(segments[:-1], int(segments[-1].start * SAMPLE_RATE))
            segments = segments[-1:]

        partial = " ".join(segment.text.strip() for segment in segments)
        self.updates.put(("text", (" ".join(self._committed), partial)))

    def _commit(self, segments, samples):
        self._committed.extend(segment.text.strip() for segment in segments)
        self._offset += samples


class Recorder:
    def __init__(self):
        self._chunks = []
        self._lock = threading.Lock()
        self._stream = None
        self.audio = np.empty((0, CHANNELS), dtype=np.float32)

    @property
    def is_recording(self):
        return self._stream is not None

    def start(self):
        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time, status):
        # Runs on the audio thread: just hand the data off.
        with self._lock:
            self._chunks.append(indata.copy())

    def snapshot(self):
        """Return all audio recorded so far (safe to call while recording)."""
        with self._lock:
            if not self._chunks:
                return np.empty((0, CHANNELS), dtype=np.float32)
            audio = np.concatenate(self._chunks)
            self._chunks = [audio]  # keep it merged so the next snapshot is cheap
            return audio

    def stop(self):
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.audio = self.snapshot()

    @property
    def duration(self):
        return len(self.audio) / SAMPLE_RATE

    def play(self):
        sd.stop()
        sd.play(self.audio, SAMPLE_RATE)

    def save(self, path):
        pcm = np.int16(np.clip(self.audio, -1.0, 1.0) * 32767)
        wavfile.write(path, SAMPLE_RATE, pcm)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Recorder")
        self.resizable(False, False)
        self.recorder = Recorder()
        self.transcriber = Transcriber()
        self.live = LiveTranscriber(self.recorder, self.transcriber)
        self._results = queue.Queue()
        self._elapsed = 0

        self.status = tk.Label(self, text="Ready", font=("Segoe UI", 14), width=24)
        self.status.pack(padx=20, pady=(20, 10))

        buttons = tk.Frame(self)
        buttons.pack(padx=20, pady=(0, 5))
        self.record_btn = tk.Button(buttons, text="● Record", width=10, command=self.toggle_record)
        self.play_btn = tk.Button(buttons, text="▶ Play", width=10, command=self.play, state=tk.DISABLED)
        self.save_btn = tk.Button(buttons, text="💾 Save", width=10, command=self.save, state=tk.DISABLED)
        self.transcribe_btn = tk.Button(
            buttons, text="📝 Transcribe", width=12, command=self.transcribe, state=tk.DISABLED
        )
        for btn in (self.record_btn, self.play_btn, self.save_btn, self.transcribe_btn):
            btn.pack(side=tk.LEFT, padx=5)

        self.live_var = tk.BooleanVar(value=True)
        self.live_check = tk.Checkbutton(self, text="Live transcription while recording", variable=self.live_var)
        self.live_check.pack(pady=(0, 5))

        settings = tk.LabelFrame(self, text="Recognition settings")
        settings.pack(padx=20, pady=(0, 10), fill=tk.X)
        tk.Label(settings, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.model_var = tk.StringVar(value=WHISPER_MODEL)
        self.model_box = ttk.Combobox(
            settings, textvariable=self.model_var, values=WHISPER_MODELS, state="readonly", width=16
        )
        self.model_box.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        tk.Label(settings, text="Language:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.language_var = tk.StringVar(value="Auto-detect")
        self.language_box = ttk.Combobox(
            settings, textvariable=self.language_var, values=list(LANGUAGES), state="readonly", width=12
        )
        self.language_box.grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)
        tk.Label(settings, text="Vocabulary:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.vocabulary_entry = tk.Entry(settings)
        self.vocabulary_entry.grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=3)
        settings.columnconfigure(3, weight=1)
        self.device_label = tk.Label(settings, fg="gray")
        self.device_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 3))
        self._update_device_label()

        self.text = tk.Text(self, width=60, height=8, wrap=tk.WORD, font=("Segoe UI", 11))
        self.text.tag_configure("partial", foreground="gray")
        self.text.pack(padx=20, pady=(0, 20))

    def _set_audio_buttons(self, state):
        for btn in (self.play_btn, self.save_btn, self.transcribe_btn):
            btn.config(state=state)

    def _set_settings_state(self, enabled):
        # Settings are locked while the model is in use.
        self.model_box.config(state="readonly" if enabled else tk.DISABLED)
        self.language_box.config(state="readonly" if enabled else tk.DISABLED)
        self.vocabulary_entry.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self.live_check.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _apply_settings(self):
        self.transcriber.configure(
            self.model_var.get(),
            LANGUAGES[self.language_var.get()],
            self.vocabulary_entry.get().strip(),
        )

    def _update_device_label(self):
        device = "NVIDIA GPU" if self.transcriber.device == "cuda" else "CPU"
        self.device_label.config(text=f"Running on: {device}")

    def _show_text(self, committed, partial=""):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, committed)
        if partial:
            self.text.insert(tk.END, (" " if committed else "") + partial, "partial")
        self.text.see(tk.END)

    def toggle_record(self):
        if self.recorder.is_recording:
            self.recorder.stop()
            self.record_btn.config(text="● Record")
            if self.live.is_running:
                # Wait for the live transcriber to finish the last bit of audio.
                self.live.stop()
                self.record_btn.config(state=tk.DISABLED)
                self.status.config(text="Finishing transcription...", fg="blue")
            else:
                self._recording_finished()
        else:
            sd.stop()
            try:
                self.recorder.start()
            except sd.PortAudioError as e:
                messagebox.showerror("Microphone error", str(e))
                return
            self.record_btn.config(text="■ Stop")
            self._set_audio_buttons(tk.DISABLED)
            self._set_settings_state(False)
            self._show_text("")
            if self.live_var.get():
                self._apply_settings()
                self.live.start()
                self.after(100, self._check_live)
            self._elapsed = 0
            self._tick()

    def _recording_finished(self):
        self.record_btn.config(state=tk.NORMAL)
        self._set_settings_state(True)
        self._update_device_label()
        self.status.config(text=f"Recorded {self.recorder.duration:.1f} s", fg="black")
        has_audio = self.recorder.duration > 0
        self._set_audio_buttons(tk.NORMAL if has_audio else tk.DISABLED)

    def _tick(self):
        if self.recorder.is_recording:
            text = f"Recording... {self._elapsed} s"
            if self.live.is_running and not self.transcriber.is_loaded:
                text += " (loading model)"
            self.status.config(text=text, fg="red")
            self._elapsed += 1
            self.after(1000, self._tick)

    def _check_live(self):
        # Tkinter is not thread-safe, so live updates are applied here on the main thread.
        while True:
            try:
                kind, payload = self.live.updates.get_nowait()
            except queue.Empty:
                self.after(100, self._check_live)
                return
            if kind == "text":
                self._show_text(*payload)
            elif kind == "error":
                messagebox.showerror("Live transcription error", str(payload))
            elif kind == "done":
                if not self.recorder.is_recording:
                    self._recording_finished()
                return

    def play(self):
        self.status.config(text=f"Playing {self.recorder.duration:.1f} s", fg="green")
        self.recorder.play()

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV audio", "*.wav")]
        )
        if path:
            self.recorder.save(path)
            self.status.config(text="Saved", fg="black")

    def transcribe(self):
        self._apply_settings()
        if self.transcriber.is_loaded:
            self.status.config(text="Transcribing...", fg="blue")
        else:
            self.status.config(text=f"Loading {self.model_var.get()} model...", fg="blue")
        self.record_btn.config(state=tk.DISABLED)
        self.transcribe_btn.config(state=tk.DISABLED)
        self._set_settings_state(False)
        audio = self.recorder.audio
        threading.Thread(target=self._transcribe_worker, args=(audio,), daemon=True).start()
        self.after(100, self._check_results)

    def _transcribe_worker(self, audio):
        # Runs in a background thread so the window stays responsive.
        try:
            self._results.put(("ok", self.transcriber.transcribe(audio)))
        except Exception as e:
            self._results.put(("error", e))

    def _check_results(self):
        # Tkinter is not thread-safe, so the UI is updated here on the main thread.
        try:
            kind, result = self._results.get_nowait()
        except queue.Empty:
            self.after(100, self._check_results)
            return
        self.record_btn.config(state=tk.NORMAL)
        self.transcribe_btn.config(state=tk.NORMAL)
        self._set_settings_state(True)
        self._update_device_label()
        if kind == "error":
            self.status.config(text="Transcription failed", fg="red")
            messagebox.showerror("Transcription error", str(result))
            return
        text, language = result
        self.status.config(text=f"Transcribed (language: {language})", fg="black")
        self._show_text(text or "(no speech detected)")


if __name__ == "__main__":
    App().mainloop()
