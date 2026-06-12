#!/usr/bin/env python3
"""
CLI chat client for the Competitive Pricing Agent API.

Usage:
    python scripts/chat.py                        # connect to localhost:8000
    python scripts/chat.py --url http://host:8000 # custom server
    python scripts/chat.py --product-id <uuid>    # start with a known product
    python scripts/chat.py --resume <conv-id>     # continue an existing conversation

Commands during chat:
    /quit or /exit   — quit
    /new             — start a new conversation
    /id              — show current conversation_id
    /history         — show message history from the server
"""

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000"
WIDTH = 80


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def _wrap(text: str, indent: str = "") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent))
    return "\n".join(wrapped)


def _print_banner():
    print("=" * WIDTH)
    print("  Competitive Pricing Agent — Chat CLI")
    print("  Type your question. Commands: /new  /id  /history  /quit")
    print("=" * WIDTH)


def _print_assistant(answer: str, meta: dict):
    print(f"\n\033[36mAssistant\033[0m")
    print(_wrap(answer, indent="  "))
    intent = meta.get("intent", "")
    confidence = meta.get("confidence", "")
    product_id = meta.get("product_id") or ""
    actions = [a for a in (meta.get("actions_triggered") or []) if a not in ("user_message_saved",)]
    if intent or confidence:
        print(f"\n  \033[90m[intent={intent}  confidence={confidence}  product={product_id or '—'}]\033[0m")
    offers = meta.get("offers") or []
    if offers:
        print(f"\n  \033[33mOffers ({len(offers)}):\033[0m")
        for o in offers[:5]:
            stock = "✓" if o.get("in_stock") else "✗"
            print(f"    {stock} {o.get('merchant', '?'):20s}  €{o.get('price', 0):.2f}  {o.get('title', '')[:40]}")
    equivalents = meta.get("equivalents") or []
    if equivalents:
        print(f"\n  \033[35mEquivalents ({len(equivalents)}):\033[0m")
        for eq in equivalents[:5]:
            score = eq.get("score", 0)
            print(f"    score={score:.2f}  €{eq.get('price', 0):.2f}  {eq.get('title', '')[:50]}")
    missing = meta.get("missing_information") or []
    if missing:
        print(f"\n  \033[90mMissing: {', '.join(missing)}\033[0m")
    print()


def _show_history(base_url: str, conversation_id: str):
    try:
        data = _get(f"{base_url}/api/v1/chat/conversations/{conversation_id}")
    except RuntimeError as e:
        print(f"  Error fetching history: {e}")
        return
    msgs = data.get("messages", [])
    print(f"\n--- History for conversation {conversation_id} ({len(msgs)} messages) ---")
    for m in msgs:
        role = m["role"].upper()
        content = m["content"][:200]
        ts = m.get("created_at", "")[:19]
        print(f"  [{ts}] {role}: {content}")
    print("---")


def chat_loop(base_url: str, product_id: str | None, conversation_id: str | None):
    _print_banner()
    if conversation_id:
        print(f"  Resuming conversation: {conversation_id}")
    if product_id:
        print(f"  Product context: {product_id}")
    print()

    while True:
        try:
            user_input = input("\033[32mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print("Bye.")
            break

        if user_input.lower() == "/new":
            conversation_id = None
            product_id = None
            print("  Started a new conversation.\n")
            continue

        if user_input.lower() == "/id":
            print(f"  conversation_id: {conversation_id or '(none yet)'}\n")
            continue

        if user_input.lower() == "/history":
            if not conversation_id:
                print("  No active conversation yet.\n")
            else:
                _show_history(base_url, conversation_id)
            continue

        payload = {
            "message": user_input,
            "conversation_id": conversation_id,
            "product_id": product_id,
        }

        print("  \033[90m…thinking\033[0m")
        try:
            resp = _post(f"{base_url}/api/v1/chat", payload)
        except RuntimeError as e:
            print(f"\n  \033[31mError:\033[0m {e}\n")
            continue

        conversation_id = resp.get("conversation_id") or conversation_id
        product_id = resp.get("product_id") or product_id

        _print_assistant(resp.get("answer", ""), resp)


def main():
    parser = argparse.ArgumentParser(description="Chat CLI for Competitive Pricing Agent")
    parser.add_argument("--url", default=BASE_URL, help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--product-id", default=None, help="Start with a specific product_id")
    parser.add_argument("--resume", default=None, metavar="CONV_ID", help="Resume an existing conversation_id")
    args = parser.parse_args()

    # Quick health check
    try:
        _get(f"{args.url}/health")
    except Exception as e:
        print(f"Cannot reach API at {args.url}: {e}", file=sys.stderr)
        print("Make sure the server is running: uvicorn api.main:app --reload", file=sys.stderr)
        sys.exit(1)

    chat_loop(
        base_url=args.url,
        product_id=args.product_id,
        conversation_id=args.resume,
    )


if __name__ == "__main__":
    main()
