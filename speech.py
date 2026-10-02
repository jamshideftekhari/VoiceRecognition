"""Speech recognition with faster-whisper, shared by the desktop recorder and the web app."""

import glob
import io
import os
import threading

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

    def transcribe(self, audio, **options):
        options.setdefault("beam_size", 5)
        segments, info = self.segments(audio, **options)
        text = " ".join(segment.text.strip() for segment in segments)
        return text, info.language


def decode_audio(data):
    """Decode a recording in any common format (WAV, WebM/Opus, Ogg, MP4...) to 16 kHz mono float32.

    Uses PyAV (installed with faster-whisper) directly; faster_whisper.decode_audio
    passes an option that newer PyAV versions no longer accept.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
    for resampled in resampler.resample(None):  # flush what the resampler still holds
        chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0
