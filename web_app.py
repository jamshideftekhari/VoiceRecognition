"""Web interface for the restaurant ordering agent, with typed or spoken (push-to-talk) orders.

Run `python web_app.py` and open http://127.0.0.1:5000 in a browser.
Each browser tab gets its own conversation and order; the CLI (order_agent.py) still works as before.
"""

import argparse
import logging
import sys
import threading
import time
import uuid

import anthropic
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from order_agent import NO_CREDENTIALS_MESSAGE, Menu, OrderAgent, has_credentials
from speech import LANGUAGES, SAMPLE_RATE, WHISPER_MODELS, Transcriber, decode_audio

SESSION_TIMEOUT = 60 * 60  # forget conversations that have been idle for an hour
MAX_MESSAGE_LENGTH = 1000
MAX_RECORDING_SECONDS = 60
MIN_RECORDING_SECONDS = 0.3

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # largest accepted upload (a recording), in bytes
log = logging.getLogger(__name__)

menu = Menu()
client = anthropic.Anthropic()  # shared by all conversations

# Speech recognition, shared by all conversations. The dish names help Whisper spell them right.
transcriber = Transcriber()
transcriber.vocabulary = ", ".join(
    dict.fromkeys(name for item in menu.items.values() for name in (item["name"], item["name_da"]))
)
_sessions = {}  # session id -> Session
_sessions_lock = threading.Lock()


class Session:
    def __init__(self):
        self.agent = OrderAgent(menu=menu, client=client)
        self.lock = threading.Lock()  # one message at a time per conversation
        self.last_used = time.time()


def _get_session(session_id):
    with _sessions_lock:
        now = time.time()
        for expired in [sid for sid, s in _sessions.items() if now - s.last_used > SESSION_TIMEOUT]:
            del _sessions[expired]
        session = _sessions.get(session_id)
        if session:
            session.last_used = now
        return session


def _error(message, status):
    return jsonify(error=message), status


@app.errorhandler(413)
def too_large(_):
    return _error("The recording is too large. Please keep each message short.", 413)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/session")
def new_session():
    """Start a new conversation and order."""
    session = Session()
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = session
    return jsonify(
        session_id=session_id,
        restaurant=menu.restaurant,
        greeting=session.agent.greeting,
        menu=list(menu.items.values()),
        currency=menu.currency,
        order=session.agent.order.to_dict(),
    )


@app.post("/api/voice")
def voice():
    """Turn a push-to-talk recording into text. The page then sends the text to /api/chat."""
    session = _get_session(request.form.get("session_id", ""))
    if session is None:
        return _error("This conversation has expired. Please start a new order.", 404)
    if session.agent.is_done:
        return _error("This order has already been placed. Start a new order to order again.", 409)
    upload = request.files.get("audio")
    if upload is None:
        return _error("No recording was received.", 400)
    data = upload.read()
    try:
        audio = decode_audio(data)
    except Exception:
        log.exception("Could not decode recording")
        return _error("Couldn't read the recording. Please try again.", 400)
    seconds = len(audio) / SAMPLE_RATE
    if seconds < MIN_RECORDING_SECONDS:
        return _error("That was too short. Hold the microphone button while you speak.", 400)
    if seconds > MAX_RECORDING_SECONDS:
        return _error(f"Please keep each message under {MAX_RECORDING_SECONDS} seconds.", 400)
    # The voice-activity filter skips silence, where Whisper would otherwise invent text.
    text, language = transcriber.transcribe(audio, vad_filter=True)
    text = text.strip()
    if not text:
        return _error("Sorry, I didn't catch that. Please try again.", 422)
    return jsonify(
        text=text,
        language=language,
        # For teaching: how much data the sound was, compared with the text it became.
        audio={"seconds": round(seconds, 2), "samples": len(audio), "sample_rate": SAMPLE_RATE,
               "upload_bytes": len(data), "pcm_bytes": len(audio) * 2},
    )


@app.post("/api/chat")
def chat():
    """Send what the customer said; returns the waiter's reply and the updated order."""
    data = request.get_json(silent=True) or {}
    session = _get_session(str(data.get("session_id", "")))
    if session is None:
        return _error("This conversation has expired. Please start a new order.", 404)
    text = str(data.get("text", "")).strip()[:MAX_MESSAGE_LENGTH]
    if not text:
        return _error("Please type a message.", 400)
    if session.agent.is_done:
        return _error("This order has already been placed. Start a new order to order again.", 409)
    if not session.lock.acquire(blocking=False):
        return _error("Please wait for the waiter's reply.", 409)
    try:
        reply = session.agent.send(text)
    except anthropic.RateLimitError:
        return _error("The waiter is busy right now. Please try again in a moment.", 429)
    except anthropic.APIConnectionError:
        return _error("Couldn't reach the ordering system. Please try again.", 502)
    except anthropic.APIError:
        log.exception("Claude API error")
        return _error("Something went wrong. Please try again.", 502)
    finally:
        session.lock.release()
    return jsonify(reply=reply, order=session.agent.order.to_dict(), done=session.agent.is_done)


def main():
    parser = argparse.ArgumentParser(description="Web interface for the restaurant ordering agent.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on; use 0.0.0.0 to allow other devices on the network")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--https", action="store_true",
                        help="serve over HTTPS with a self-signed certificate; browsers only allow the "
                             "microphone on localhost or HTTPS, so use this for phones and other devices")
    parser.add_argument("--whisper-model", choices=WHISPER_MODELS,
                        help="speech-recognition model (default: large-v3-turbo on an NVIDIA GPU, else base)")
    parser.add_argument("--language", choices=[code for code in LANGUAGES.values() if code],
                        help="language spoken to the waiter, e.g. da or en (default: detect automatically)")
    args = parser.parse_args()
    if not has_credentials(client):
        sys.exit(NO_CREDENTIALS_MESSAGE)
    logging.basicConfig(level=logging.INFO)

    transcriber.model_size = args.whisper_model or ("large-v3-turbo" if transcriber.device == "cuda" else "base")
    transcriber.language = args.language
    print(f"Loading speech recognition ({transcriber.model_size} on {transcriber.device.upper()})...")
    transcriber.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # load the model now, not on the first order

    scheme = "https" if args.https else "http"
    print(f"Open {scheme}://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port} in a browser.")
    app.run(host=args.host, port=args.port, threaded=True, ssl_context="adhoc" if args.https else None)


if __name__ == "__main__":
    main()
