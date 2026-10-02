"""Web interface for the restaurant ordering agent.

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
from flask import Flask, jsonify, request, send_from_directory

from order_agent import NO_CREDENTIALS_MESSAGE, Menu, OrderAgent, has_credentials

SESSION_TIMEOUT = 60 * 60  # forget conversations that have been idle for an hour
MAX_MESSAGE_LENGTH = 1000

app = Flask(__name__, static_folder="static")
log = logging.getLogger(__name__)

menu = Menu()
client = anthropic.Anthropic()  # shared by all conversations
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
    args = parser.parse_args()
    if not has_credentials(client):
        sys.exit(NO_CREDENTIALS_MESSAGE)
    logging.basicConfig(level=logging.INFO)
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port} in a browser.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
