"""Restaurant ordering agent: a Claude-powered waiter that takes orders in a terminal chat.

The AI handles the conversation; the menu, the basket and the prices live in Python.
Claude changes the order only through the tools defined in `make_tools`.
"""

import argparse
import difflib
import json
import sys
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

    def place(self, customer_name):
        if not self.lines:
            raise ToolError("The order is empty, so there is nothing to place.")
        if self.placed_as:
            raise ToolError(f"The order has already been placed as order {self.placed_as}.")
        now = datetime.now()
        number = now.strftime("%H%M%S")
        ORDERS_DIR.mkdir(exist_ok=True)
        record = {
            "order_number": number,
            "customer_name": customer_name,
            "time": now.isoformat(timespec="seconds"),
            "lines": [
                {"item_id": l.item["id"], "name": l.item["name"], "quantity": l.quantity,
                 "notes": l.notes, "price": l.item["price"], "line_total": l.total}
                for l in self.lines
            ],
            "total": self.total,
            "currency": self.menu.currency,
        }
        path = ORDERS_DIR / f"order-{now:%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        self.placed_as = number
        return number


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

    def __init__(self, menu=None, model=MODEL, on_tool_call=None):
        self.menu = menu or Menu()
        self.order = Order(self.menu)
        self.model = model
        self.on_tool_call = on_tool_call  # optional callback(name, input) for debugging
        self.client = anthropic.Anthropic()
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


def main():
    parser = argparse.ArgumentParser(description="Chat with the restaurant ordering agent.")
    parser.add_argument("--debug", action="store_true", help="show the tool calls the agent makes")
    args = parser.parse_args()

    def show_tool_call(name, tool_input):
        print(f"  \033[90m[{name}] {json.dumps(tool_input, ensure_ascii=False)}\033[0m")

    agent = OrderAgent(on_tool_call=show_tool_call if args.debug else None)
    client = agent.client
    if not (client.api_key or client.auth_token or client.credentials):
        sys.exit(
            "No Anthropic credentials found. Set the ANTHROPIC_API_KEY environment variable "
            "(create a key at https://platform.claude.com) or log in with `ant auth login`."
        )
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

    if agent.is_done:
        print(f"\n{agent.order.summary()}\nSaved in the '{ORDERS_DIR.name}' folder as order {agent.order.placed_as}.")


if __name__ == "__main__":
    main()
