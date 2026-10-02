"""Restaurant ordering agent: a Claude-powered waiter that takes orders in a terminal chat.

The AI handles the conversation; the menu, the basket and the prices live in Python.
Claude changes the order only through the tools defined in `make_tools`.
"""

import argparse
import difflib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic import beta_tool
from anthropic.lib.tools import ToolError

MODEL = "claude-opus-5"
MENU_FILE = Path(__file__).with_name("menu.json")
ORDERS_DIR = Path(__file__).with_name("orders")

SYSTEM_PROMPT = """\
You are the friendly waiter at {restaurant}, taking a customer's order. The customer \
talks to you through a speech-recognition system, and your replies will later be read \
aloud, so:
- Keep replies short and conversational: one to three sentences, no lists, no markdown.
- Speech recognition can mishear words. If a request is unclear or doesn't match the \
menu, ask instead of guessing.
- Reply in the language the customer uses (for example Danish or English).

Taking the order:
- Use the tools for every change to the order, and only order items from the menu below.
- Put special requests (e.g. "no onions") in the notes of the item they belong to.
- Only mention prices and totals that come from the menu or a tool result.
- Answer questions about dishes and allergens from the menu. If the menu doesn't say, \
tell the customer you'll ask the kitchen rather than guessing.
- When the customer is done, read back the full order with the total and ask them to \
confirm. Call place_order only after they have clearly confirmed, and ask for a name \
for the order if you don't have one.

The customer has already been greeted. Menu (prices in {currency}):
{menu}
"""

GREETING = "Hi, welcome to {restaurant}! What can I get for you today?"


def _normalize(name):
    """'Coca-Cola Zero', 'cola_zero' -> 'coca cola zero', 'cola zero'."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


class Menu:
    def __init__(self, path=MENU_FILE):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.restaurant = data["restaurant"]
        self.currency = data["currency"]
        self.items = {item["id"]: item for item in data["items"]}

    def find(self, query):
        """Find a menu item by id or (English/Danish) name, tolerating small misspellings."""
        key = _normalize(query)
        names = {}
        for item in self.items.values():
            for name in (item["id"], item["name"], item["name_da"]):
                names[_normalize(name)] = item
        if key in names:
            return names[key]
        matches = difflib.get_close_matches(key, names, n=5, cutoff=0.6)
        found = list({names[m]["id"]: names[m] for m in matches}.values())  # unique, best first
        if len(found) == 1:
            return found[0]
        suggestions = ", ".join(item["name"] for item in found[:3]) or "none"
        raise ToolError(f"'{query}' is not on the menu. Closest matches: {suggestions}.")

    def as_prompt_text(self):
        lines = []
        for item in self.items.values():
            allergens = ", ".join(item["allergens"]) or "none"
            veg = ", vegetarian" if item["vegetarian"] else ""
            lines.append(
                f"- {item['id']}: {item['name']} / {item['name_da']} ({item['category']}{veg}) "
                f"{item['price']} - {item['description']}. Allergens: {allergens}."
            )
        return "\n".join(lines)


@dataclass
class OrderLine:
    item: dict
    quantity: int
    notes: str = ""

    @property
    def total(self):
        return self.item["price"] * self.quantity


class Order:
    def __init__(self, menu):
        self.menu = menu
        self.lines = []
        self.placed_as = None  # order number once placed

    @property
    def total(self):
        return sum(line.total for line in self.lines)

    def add(self, query, quantity=1, notes=""):
        if quantity < 1:
            raise ToolError("Quantity must be at least 1.")
        item = self.menu.find(query)
        notes = notes.strip()
        for line in self.lines:
            if line.item["id"] == item["id"] and line.notes == notes:
                line.quantity += quantity
                break
        else:
            self.lines.append(OrderLine(item, quantity, notes))

    def remove(self, line_number, quantity=0):
        if not 1 <= line_number <= len(self.lines):
            raise ToolError(f"There is no line {line_number}. {self.summary()}")
        line = self.lines[line_number - 1]
        if quantity <= 0 or quantity >= line.quantity:
            self.lines.remove(line)
        else:
            line.quantity -= quantity

    def summary(self):
        if not self.lines:
            return "The order is empty."
        rows = []
        for number, line in enumerate(self.lines, start=1):
            notes = f" ({line.notes})" if line.notes else ""
            rows.append(f"{number}. {line.quantity} x {line.item['name']}{notes}: {line.total} {self.menu.currency}")
        return "Current order:\n" + "\n".join(rows) + f"\nTotal: {self.total} {self.menu.currency}"

    def to_dict(self):
        """The order as plain data, e.g. for a web page."""
        return {
            "lines": [
                {"number": number, "item_id": line.item["id"], "name": line.item["name"],
                 "quantity": line.quantity, "notes": line.notes, "line_total": line.total}
                for number, line in enumerate(self.lines, start=1)
            ],
            "total": self.total,
            "currency": self.menu.currency,
            "placed_as": self.placed_as,
        }

    def place(self, customer_name):
        if not self.lines:
            raise ToolError("The order is empty, so there is nothing to place.")
        if self.placed_as:
            raise ToolError(f"The order has already been placed as order {self.placed_as}.")
        with _orders_lock:
            return self._save(customer_name)

    def _save(self, customer_name):
        now = datetime.now()
        ORDERS_DIR.mkdir(exist_ok=True)
        # Order numbers start at 1 each day, which is easy for customers to remember.
        number = str(len(list(ORDERS_DIR.glob(f"order-{now:%Y%m%d}-*.json"))) + 1)
        placed = now.isoformat(timespec="seconds")
        record = {
            "order_number": number,
            "customer_name": customer_name,
            "time": placed,
            "lines": [
                {"item_id": l.item["id"], "name": l.item["name"], "quantity": l.quantity,
                 "notes": l.notes, "price": l.item["price"], "line_total": l.total}
                for l in self.lines
            ],
            "total": self.total,
            "currency": self.menu.currency,
            "status": ORDER_STATUSES[0],
            "status_history": [{"status": ORDER_STATUSES[0], "time": placed}],
        }
        # Never overwrite another order placed in the same second.
        for attempt in range(1, 100):
            suffix = "" if attempt == 1 else f"-{attempt}"
            try:
                with open(ORDERS_DIR / f"order-{now:%Y%m%d-%H%M%S}{suffix}.json", "x", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                break
            except FileExistsError:
                continue
        self.placed_as = number
        return number


# ---- Placed orders, as stored in the orders folder (used by the kitchen page) ----

# The stages an order goes through after the customer has placed it.
ORDER_STATUSES = ["received", "preparing", "ready", "delivered"]
_orders_lock = threading.Lock()


def _order_path(order_id):
    # Only accept ids that look like our own file names, so nobody can reach other files.
    if not re.fullmatch(r"order-\d{8}-\d{6}(-\d+)?", order_id):
        raise KeyError(order_id)
    path = ORDERS_DIR / f"{order_id}.json"
    if not path.exists():
        raise KeyError(order_id)
    return path


def _read_order(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    record["id"] = path.stem
    # Orders saved before statuses existed count as just received.
    record.setdefault("status", ORDER_STATUSES[0])
    record.setdefault("status_history", [{"status": ORDER_STATUSES[0], "time": record["time"]}])
    return record


def list_orders():
    """All placed orders, newest first."""
    if not ORDERS_DIR.exists():
        return []
    orders = []
    for path in ORDERS_DIR.glob("order-*.json"):
        try:
            orders.append(_read_order(path))
        except (OSError, ValueError, KeyError):
            continue  # skip a file that is being written or is damaged
    return sorted(orders, key=lambda o: (o["time"], o["id"]), reverse=True)


def update_order_status(order_id, status):
    """Move an order to `status` (one step forward or back) and record when it happened."""
    if status not in ORDER_STATUSES:
        raise ValueError(f"Unknown status '{status}'.")
    with _orders_lock:
        path = _order_path(order_id)
        record = _read_order(path)
        step = ORDER_STATUSES.index(status) - ORDER_STATUSES.index(record["status"])
        if abs(step) != 1:
            raise ValueError(f"Order {record['order_number']} is '{record['status']}' and can't move to '{status}'.")
        record["status"] = status
        record["status_history"].append({"status": status, "time": datetime.now().isoformat(timespec="seconds")})
        del record["id"]
        # Write to a temporary file first, so the kitchen page never reads a half-written order.
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
        record["id"] = path.stem
        return record


def make_tools(order):
    """The tools Claude can call. Each returns the updated order so Claude always sees the real state."""

    @beta_tool
    def add_item(item: str, quantity: int = 1, notes: str = "") -> str:
        """Add a menu item to the order.

        Args:
            item: The menu item id (preferred) or its name.
            quantity: How many to add.
            notes: Special requests for this item, e.g. "no onions". Empty if none.
        """
        order.add(item, quantity, notes)
        return order.summary()

    @beta_tool
    def remove_item(line_number: int, quantity: int = 0) -> str:
        """Remove an item from the order, or reduce its quantity.

        Args:
            line_number: The line number as shown in the current order (starting at 1).
            quantity: How many to remove. 0 removes the whole line.
        """
        order.remove(line_number, quantity)
        return order.summary()

    @beta_tool
    def view_order() -> str:
        """Show the current order with line numbers, prices and the total."""
        return order.summary()

    @beta_tool
    def place_order(customer_name: str) -> str:
        """Send the order to the kitchen. Only call this after the customer has confirmed
        the read-back order and total.

        Args:
            customer_name: The name the order is placed under.
        """
        number = order.place(customer_name)
        return f"Order placed as number {number} for {customer_name}. {order.summary()}"

    return [add_item, remove_item, view_order, place_order]


class OrderAgent:
    """One conversation with one customer. Call send() with what the customer said."""

    def __init__(self, menu=None, model=MODEL, on_tool_call=None, client=None):
        self.menu = menu or Menu()
        self.order = Order(self.menu)
        self.model = model
        self.on_tool_call = on_tool_call  # optional callback(name, input) for debugging
        self.client = client or anthropic.Anthropic()
        self.tools = make_tools(self.order)
        self.system = SYSTEM_PROMPT.format(
            restaurant=self.menu.restaurant, currency=self.menu.currency, menu=self.menu.as_prompt_text()
        )
        self.history = []

    @property
    def greeting(self):
        return GREETING.format(restaurant=self.menu.restaurant)

    @property
    def is_done(self):
        return self.order.placed_as is not None

    def send(self, text):
        """Send the customer's words to the waiter and return the waiter's reply."""
        start = len(self.history)
        try:
            return self._send(text)
        except Exception:
            del self.history[start:]  # forget the failed turn so the customer can just repeat it
            raise

    def _send(self, text):
        self.history.append({"role": "user", "content": text})
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=list(self.history),
            output_config={"effort": "low"},  # quick replies matter more than deep thinking here
            cache_control={"type": "ephemeral"},  # the menu and instructions are reused every turn
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a request is declined, retry it on Anthropic's recommended model
            max_iterations=10,
        )
        replies = []
        for message in runner:
            # Keep the full conversation, including tool calls and their results.
            self.history.append(message.to_param())
            if message.stop_reason == "refusal":
                replies.append("Sorry, I can't help with that. Is there anything else you'd like to order?")
                continue
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    replies.append(block.text.strip())
                elif block.type == "tool_use" and self.on_tool_call:
                    self.on_tool_call(block.name, block.input)
            tool_results = runner.generate_tool_call_response()  # cached: the runner won't run them twice
            if tool_results:
                self.history.append(tool_results)
        return "\n".join(replies)


def hex_view(text, width=72):
    """Show how the computer stores text: each character above its UTF-8 bytes in hex.

    Example: 'Hej ø' ->   H    e    j    ␣    ø
                          0x48 0x65 0x6A 0x20 0xC3 0xB8
    """
    char_row, byte_row, rows = "", "", []
    for ch in text:
        hex_bytes = " ".join(f"0x{b:02X}" for b in ch.encode("utf-8"))  # 0x marks a hexadecimal number
        column = len(hex_bytes) + 1
        if len(byte_row) + column > width:
            rows += [char_row, byte_row]
            char_row, byte_row = "", ""
        char_row += ("␣" if ch == " " else ch).ljust(column)
        byte_row += hex_bytes.ljust(column)
    rows += [char_row, byte_row]
    byte_count = len(text.encode("utf-8"))
    rows.append(f"{len(text)} characters -> {byte_count} bytes (UTF-8)")
    return "\n".join(rows)


NO_CREDENTIALS_MESSAGE = (
    "No Anthropic credentials found. Set the ANTHROPIC_API_KEY environment variable "
    "(create a key at https://platform.claude.com) or log in with `ant auth login`."
)


def has_credentials(client):
    return bool(client.api_key or client.auth_token or client.credentials)


def main():
    parser = argparse.ArgumentParser(description="Chat with the restaurant ordering agent.")
    parser.add_argument("--debug", action="store_true", help="show the tool calls the agent makes")
    parser.add_argument("--hex", action="store_true", help="show each message as UTF-8 bytes in hexadecimal")
    args = parser.parse_args()

    def show_tool_call(name, tool_input):
        print(f"  \033[90m[{name}] {json.dumps(tool_input, ensure_ascii=False)}\033[0m")

    agent = OrderAgent(on_tool_call=show_tool_call if args.debug else None)
    if not has_credentials(agent.client):
        sys.exit(NO_CREDENTIALS_MESSAGE)
    print(f"Waiter: {agent.greeting}  (type 'quit' to leave)")
    while not agent.is_done:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if args.hex:
            print(f"\033[90m{hex_view(text)}\033[0m")
        try:
            reply = agent.send(text)
        except anthropic.AuthenticationError:
            sys.exit("No valid Anthropic API key. Set the ANTHROPIC_API_KEY environment variable and try again.")
        except anthropic.RateLimitError:
            print("Waiter: Sorry, I'm a bit busy right now. Please say that again in a moment.")
            continue
        except anthropic.APIConnectionError:
            print("Waiter: Sorry, I couldn't reach the ordering system. Please check the internet connection.")
            continue
        print(f"Waiter: {reply}")
        if args.hex:
            print(f"\033[90m{hex_view(reply)}\033[0m")

    if agent.is_done:
        print(f"\n{agent.order.summary()}\nSaved in the '{ORDERS_DIR.name}' folder as order {agent.order.placed_as}.")


if __name__ == "__main__":
    main()
