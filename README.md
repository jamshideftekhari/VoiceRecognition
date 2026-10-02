# Voice Recognition

A Python voice app: record from the microphone, play it back, save it as a WAV
file, and transcribe the speech to text — live while you speak, or afterwards — with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

It also contains an AI ordering agent (`order_agent.py`): a Claude-powered
restaurant waiter that takes orders in a terminal chat or a web page. It will later be
connected to the voice recognition.

## Requirements

- Python 3.10 or newer (tested on 3.14, Windows 11)
- A microphone and speakers/headphones
- Internet connection the first time you use a Whisper model (each model is
  downloaded once and cached in `%USERPROFILE%\.cache\huggingface`)
- Optional: an NVIDIA GPU for much faster transcription

## Installation

```
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, also install the CUDA libraries (~1 GB):

```
python -m pip install -r requirements-gpu.txt
```

The app uses the GPU automatically when it finds one, and falls back to the CPU
if the GPU cannot be used. The settings panel shows which one is in use.

## Usage

```
python recorder.py
```

| Button     | Action                                          |
|------------|-------------------------------------------------|
| ● Record   | Start recording (the button changes to ■ Stop)  |
| ■ Stop     | Stop recording and show the length              |
| ▶ Play     | Play back the last recording                    |
| 💾 Save    | Save the recording as a `.wav` file             |
| 📝 Transcribe | Convert the recording to text (shown in the text box) |

Tick **Live transcription while recording** (on by default) to see the text
appear while you speak. Provisional text is shown in gray and turns black once
it is final, which happens when you pause for about a second (or after ~15 s of
continuous speech). When you press Stop, the last part is finished off before
the buttons are enabled again. Press **📝 Transcribe** afterwards to re-transcribe
the whole recording in one pass, which is slightly more accurate.

Transcription runs offline, on the GPU if available or else the CPU. The
detected language is shown in the status line. The first transcription with a
model is slower because the model has to be downloaded and loaded.

## Recognition settings

The **Recognition settings** panel is locked while recording or transcribing;
changes apply from the next recording or transcription.

- **Model**: which Whisper model to use (see below). A newly selected model is
  loaded (and, the first time, downloaded) on its first use.
- **Language**: *Auto-detect*, *Danish* or *English*. Choosing the language
  avoids wrong guesses on short clips and is slightly faster.
- **Vocabulary**: names and technical terms you use, e.g.
  `Tkinter, faster-whisper, Aarhus`. They are passed to Whisper as its *initial
  prompt*, which makes those spellings more likely. A period is added
  automatically, since Whisper copies the prompt's style and drops punctuation
  if the prompt has none.

### Choosing a Whisper model

| Model            | Size     | Notes                                                      |
|------------------|----------|------------------------------------------------------------|
| `tiny`           | ~75 MB   | Fastest, least accurate                                    |
| `base`           | ~150 MB  | Default; good balance for English on a CPU                 |
| `small`          | ~480 MB  | Noticeably better, especially for Danish                   |
| `medium`         | ~1.5 GB  | High accuracy, slow on CPU                                 |
| `large-v3-turbo` | ~1.6 GB  | Close to `large-v3` quality but much faster; best on a GPU |
| `large-v3`       | ~3 GB    | Best accuracy, very slow on CPU                            |

Measured on an NVIDIA RTX A1000 (6 GB), 23 s of speech: `base` 0.6 s,
`large-v3-turbo` 1.0 s. On the CPU, `base` is about 6× slower than on the GPU.
`WHISPER_MODEL` at the top of `recorder.py` sets the default.

## How live transcription works

`LiveTranscriber` runs in a background thread. About once a second it
transcribes the audio that has not been committed yet (the *window*), using
greedy decoding and Whisper's voice-activity filter so silence is skipped.

- When there is more than 1 s of silence after the last word, the text is
  committed and the window moves past it.
- If the window grows beyond 15 s without a pause, all finished segments except
  the last are committed.
- At 25 s everything is committed, since Whisper handles at most 30 s at a time.

This keeps each pass short, so the delay stays around 1–2 s no matter how long
the recording is. The language is detected from the first confident result and
then kept fixed for the rest of the recording (unless a language is chosen in
the settings). The timing constants are class
attributes on `LiveTranscriber` and can be tuned.

## Restaurant ordering agent

`order_agent.py` is a waiter for the example restaurant *Café Mølle*, chatting
in the terminal (Danish or English). It needs an Anthropic API key and an
internet connection.

```
set ANTHROPIC_API_KEY=sk-ant-...        (PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-...")
python order_agent.py
python order_agent.py --debug           (also shows the tool calls the agent makes)
```

Example (illustrative; the exact wording varies):

```
Waiter: Hi, welcome to Café Mølle! What can I get for you today?
You: to margherita og en fadøl
Waiter: To Pizza Margherita og en fadøl, det bliver 235 kr. Ellers andet?
```

When the customer confirms, the order is saved as JSON in the `orders` folder.

### Web interface

```
python web_app.py
```

Then open <http://127.0.0.1:5000>. The page shows the chat on the left and the
order being built (with total) and the menu on the right; on a phone they are
stacked. Clicking a dish adds its name to your message, and **New order**
starts over.

- Each browser tab is its own conversation and order. Conversations live in
  the server's memory and are forgotten after an hour of inactivity or when
  the server restarts.
- `python web_app.py --host 0.0.0.0` makes the page reachable from other
  devices on your network (e.g. a phone or a Raspberry Pi kiosk) at
  `http://<this-pc's-ip>:5000`. The API key stays on the server.
- This uses Flask's built-in server, which is fine for testing and a local
  network, but not for the open internet.

### How the computer sees the text (for teaching)

Tick **Show bytes (hex)** in the page header to show every chat message the way
the computer stores it: each character with its **UTF-8** bytes in
hexadecimal underneath. A live preview under the input box updates as you type.

```
t    o    ␣    f    a    d    ø         l
0x74 0x6F 0x20 0x66 0x61 0x64 0xC3 0xB8 0x6C
8 characters -> 9 bytes (UTF-8)
```

Every byte is written with the prefix `0x`, the usual way to mark a number as
hexadecimal (so `0x20` is 32, not twenty).

- Colors show how many bytes a character needs: **1 byte** for English
  letters, digits and punctuation (the same values as ASCII), **2 bytes** for
  `æ ø å`, **3 bytes** for `€`, **4 bytes** for emoji.
- Hover a character to see its Unicode code point (e.g. `ø` = `U+00F8`) and its
  bytes in binary (`11000011 10111000`).
- Spaces are shown as `␣` (byte `0x20`).
- The setting is remembered in the browser.

Ideas for class: compare `a`/`A` (`0x61`/`0x41`, differing by one bit), the
digits `0`–`9` (`0x30`–`0x39`), and why `fadøl` is 5 characters but 6 bytes.

In the terminal, `python order_agent.py --hex` prints the same view after each
message. Emoji are wider than one column in most terminals, so the columns may
shift slightly after an emoji.

### How it works

Claude handles the conversation, but the order itself lives in Python. Claude
can only change it through four **tools**:

| Tool | What it does |
|---|---|
| `add_item(item, quantity, notes)` | Add a dish; rejects items not on the menu and suggests close matches |
| `remove_item(line_number, quantity)` | Remove a line or reduce its quantity |
| `view_order()` | Show the order with line numbers and total |
| `place_order(customer_name)` | Save the order; only after the customer has confirmed |

Every tool returns the updated order, so prices and totals are always
calculated by Python, never by the AI. Dish names are matched loosely
(Danish/English names, misspellings such as "pepperonni" or "fadol"), which
helps with speech-recognition errors later.

- **Menu**: `menu.json` (dishes, prices, descriptions, allergens). It is put in
  the system prompt, so no extra tool call is needed to read it.
- **Model**: Claude Opus 5 (`claude-opus-5`) with `effort: low` for quick
  replies. Change `MODEL` at the top of the file to try another model.
- **Prompt caching** is enabled, so the menu and instructions are cheaper after
  the first turn.
- **Fallbacks**: if a request is declined by Claude's safety checks, it is
  automatically retried on Anthropic's recommended fallback model.
- **Cost**: roughly $0.05–0.15 per order on Claude Opus 5 ($5 / $25 per million
  input / output tokens), less with caching.

The `OrderAgent` class is independent of the terminal: `agent.send(text)`
returns the waiter's reply, which makes it easy to connect to voice input.

## Audio format

Audio is recorded at **16 kHz, mono**, the standard input format for
speech-recognition models such as Whisper and Vosk. Saved files are 16-bit PCM WAV.

## Project structure

| File               | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `recorder.py`      | The app: `Recorder` (audio), `Transcriber` (faster-whisper), `LiveTranscriber` (live text) and `App` (Tkinter UI) |
| `requirements.txt` | Python dependencies                                           |
| `requirements-gpu.txt` | Optional CUDA libraries for NVIDIA GPUs                   |
| `order_agent.py`   | Restaurant ordering agent: `Menu`, `Order`, tools and `OrderAgent` (Claude) |
| `menu.json`        | Example menu for the ordering agent                           |
| `web_app.py`       | Web interface for the ordering agent (Flask)                  |
| `static/index.html`| The web page: chat, live order and menu                       |
| `orders/`          | Placed orders, one JSON file each (created on first order)    |

The `Recorder` and `Transcriber` classes are independent of the UI, so they can
be reused from other scripts:

```python
import time
from recorder import Recorder, Transcriber

r = Recorder()
r.start()
time.sleep(3)
r.stop()
r.save("hello.wav")

text, language = Transcriber().transcribe(r.audio)
print(language, text)
```

## Troubleshooting

- **"Microphone error" when recording**: check that a microphone is connected and
  that Windows allows apps to use it (Settings → Privacy & security → Microphone).
- **Symlink warning from `huggingface_hub` on first run**: harmless; the model
  cache still works. Enable Windows Developer Mode to silence it.
- **Transcription is slow**: pick a smaller model, or install
  `requirements-gpu.txt` if you have an NVIDIA GPU. If the settings panel says
  *CPU* even though you have one, check that the NVIDIA driver is installed.
- **List audio devices**:
  `python -c "import sounddevice as sd; print(sd.query_devices())"`

## Roadmap

- [x] Record and play back voice
- [x] Save recordings as WAV
- [x] Transcribe speech to text with `faster-whisper`
- [x] Live transcription while speaking
- [x] Model, language and vocabulary settings; GPU support
- [ ] Try a Whisper model fine-tuned on Danish
- [x] Restaurant ordering agent (terminal chat)
- [x] Web interface for the ordering agent
- [x] Hexadecimal (UTF-8) view of the chat for teaching
- [ ] Order by voice: connect speech recognition to the ordering agent
- [ ] Spoken replies (text-to-speech)

## Changelog

### 0.7.0 – 2026-10-02: Hexadecimal view for teaching
- Web page: **Show bytes (hex)** switch that shows each chat message as
  characters with their UTF-8 bytes in hex (`0xC3 0xB8`), color-coded by byte count, with a
  live preview while typing and code point and binary on hover.
- CLI: `--hex` option prints the same view in the terminal.

### 0.6.0 – 2026-10-02: Web interface
- New `web_app.py` (Flask) and `static/index.html`: order through a web page
  with the chat, the live order with total, and a clickable menu. Works on
  phones too, and follows the system's light/dark mode.
- One conversation per browser tab; idle conversations are removed after an hour.
- `--host 0.0.0.0` to use it from other devices on the network.
- `Order.to_dict()` for the order as data; `OrderAgent` can share one API client.
- The terminal chat (`order_agent.py`) works as before.
- Added `flask` to `requirements.txt`.

### 0.5.0 – 2026-10-01: Restaurant ordering agent
- New `order_agent.py`: a Claude-powered waiter (Claude Opus 5, Anthropic SDK
  tool runner) that takes orders in a terminal chat, in Danish or English.
- Tools `add_item`, `remove_item`, `view_order` and `place_order`; the order
  and prices are kept in Python, and placed orders are saved in `orders/`.
- Loose matching of dish names (Danish/English names and misspellings).
- Example menu in `menu.json`.
- `--debug` flag to show tool calls; clear messages for a missing or invalid
  API key; a failed turn is rolled back so the customer can repeat it.
- Added `anthropic` to `requirements.txt`.

### 0.4.0 – 2026-10-01: Recognition settings and GPU
- Added a **Recognition settings** panel with a model dropdown (`tiny` to
  `large-v3`, including `large-v3-turbo`), a language choice (Auto-detect,
  Danish, English) and a vocabulary field.
- Whisper runs on an NVIDIA GPU automatically when available (`float16`), with
  automatic fallback to the CPU. The panel shows which one is in use.
- Added `requirements-gpu.txt` with the CUDA libraries; the app finds them
  without changing system settings.
- The vocabulary is passed as Whisper's initial prompt, with a period added so
  punctuation is kept.
- The status line shows when a model is being loaded.

### 0.3.0 – 2026-10-01: Live transcription
- Added live transcription while recording, toggled with the
  **Live transcription while recording** checkbox (on by default).
- Provisional text is shown in gray and turns black once committed (after a
  pause of about 1 s, or after ~15 s of continuous speech).
- Pressing Stop finishes transcribing the last part before the buttons are
  enabled again.
- New `LiveTranscriber` class that runs the live transcription in a background
  thread, using Whisper's voice-activity filter and word timestamps to detect
  pauses.
- `Recorder.snapshot()` returns the audio recorded so far, so it can be read
  while recording is still in progress.
- `Transcriber.segments()` returns the raw Whisper segments; a lock makes sure
  only one transcription uses the model at a time.

### 0.2.0 – 2026-10-01: Transcription
- Added the **📝 Transcribe** button, which converts the recording to text with
  faster-whisper (offline, on the CPU) and shows it in a text box.
- The language is detected automatically and shown in the status line.
- New `Transcriber` class; the model (`WHISPER_MODEL`, default `base`) is loaded
  on first use.
- Transcription runs in a background thread so the window stays responsive.
- Added `faster-whisper` to `requirements.txt`.

### 0.1.0 – 2026-10-01: Recorder
- First version: record from the microphone, play back, and save as a 16 kHz
  mono 16-bit WAV file.
- Tkinter window with Record/Stop, Play and Save buttons and a recording timer.
- `Recorder` class with the audio logic, kept separate from the UI.
- Added `README.md` and `requirements.txt`.
