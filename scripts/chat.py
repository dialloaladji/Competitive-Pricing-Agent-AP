#!/usr/bin/env python3
"""
CLI chat client for the Competitive Pricing Agent API (streaming).

Usage:
    python scripts/chat.py                           # localhost:8000, streaming
    python scripts/chat.py --url http://host:8001    # custom server
    python scripts/chat.py --no-stream               # non-streaming fallback
    python scripts/chat.py --product-id <uuid>       # start with a known product
    python scripts/chat.py --resume <conv-id>        # continue an existing conversation

Commands during chat:
    /quit or /exit   — quit
    /new             — start a new conversation
    /id              — show current conversation_id
    /history         — show message history
"""

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000"
WIDTH = 88

# ANSI colours
C_RESET  = "\033[0m"
C_GREEN  = "\033[32m"
C_CYAN   = "\033[36m"
C_YELLOW = "\033[33m"
C_MAGENTA= "\033[35m"
C_GREY   = "\033[90m"
C_RED    = "\033[31m"
C_DIM    = "\033[2m"
C_BOLD   = "\033[1m"


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
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


def _wrap(text: str, indent: str = "  ") -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif line.startswith("•") or line.startswith("-"):
            out.append(f"{indent}{line}")
        else:
            out.extend(textwrap.wrap(line, width=WIDTH - len(indent),
                                     initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


def _print_banner():
    print("=" * WIDTH)
    print(f"  {C_BOLD}Competitive Pricing Agent — Chat CLI{C_RESET}  (streaming)")
    print(f"  Commands: {C_GREY}/new  /id  /history  /quit{C_RESET}")
    print("=" * WIDTH)


def _print_meta(meta: dict):
    intent = meta.get("intent", "")
    confidence = meta.get("confidence", "")
    product_id = meta.get("product_id") or ""
    if intent or confidence:
        print(f"\n  {C_GREY}[intent={intent}  confidence={confidence}  product={product_id or '—'}]{C_RESET}")
    offers = meta.get("offers") or []
    if offers:
        print(f"\n  {C_YELLOW}Offres ({len(offers)}):{C_RESET}")
        for o in offers[:5]:
            stock = "✓" if o.get("in_stock") else "✗"
            print(f"    {stock} {o.get('merchant','?'):22s}  €{o.get('price',0):.2f}  {o.get('title','')[:38]}")
    equivalents = meta.get("equivalents") or []
    if equivalents:
        print(f"\n  {C_MAGENTA}Équivalents ({len(equivalents)}):{C_RESET}")
        for eq in equivalents[:5]:
            print(f"    score={eq.get('score',0):.2f}  €{eq.get('price',0):.2f}  {eq.get('title','')[:48]}")
    missing = meta.get("missing_information") or []
    if missing:
        print(f"\n  {C_GREY}Manquant : {', '.join(missing)}{C_RESET}")
    print()


# ── streaming ─────────────────────────────────────────────────────────────────

def _stream_chat(base_url: str, payload: dict) -> dict | None:
    """
    Calls /chat/stream and renders events live.
    Returns the final response dict from the 'done' event, or None on error.
    """
    url = f"{base_url}/api/v1/chat/stream"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    last_thinking = ""
    answer_started = False
    final_data = None

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n\r")
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "thinking":
                    text = event.get("text", "")
                    # Overwrite the thinking line in-place
                    print(f"\r  {C_GREY}⟳ {text:<60}{C_RESET}", end="", flush=True)
                    last_thinking = text

                elif etype == "token":
                    if not answer_started:
                        # Clear the thinking line and start the answer block
                        print(f"\r{' ' * (WIDTH)}\r", end="", flush=True)
                        print(f"\n{C_CYAN}Assistant{C_RESET}")
                        answer_started = True
                    token = event.get("text", "")
                    print(f"  {token}", end="", flush=True)

                elif etype == "done":
                    if not answer_started:
                        # Thinking-only path (no product found), clear the line
                        print(f"\r{' ' * WIDTH}\r", end="", flush=True)
                    else:
                        print()  # newline after streamed answer
                    final_data = event.get("data", {})

                elif etype == "error":
                    print(f"\r  {C_RED}Erreur serveur : {event.get('text')}{C_RESET}")
                    return None

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n  {C_RED}HTTP {e.code}: {body}{C_RESET}")
        return None
    except Exception as e:
        print(f"\n  {C_RED}Erreur réseau : {e}{C_RESET}")
        return None

    return final_data


# ── non-streaming ──────────────────────────────────────────────────────────────

def _print_assistant_static(answer: str, meta: dict):
    print(f"\n{C_CYAN}Assistant{C_RESET}")
    print(_wrap(answer))
    _print_meta(meta)


def _show_history(base_url: str, conversation_id: str):
    try:
        data = _get(f"{base_url}/api/v1/chat/conversations/{conversation_id}")
    except RuntimeError as e:
        print(f"  Erreur : {e}")
        return
    msgs = data.get("messages", [])
    print(f"\n--- Historique conversation {conversation_id} ({len(msgs)} messages) ---")
    for m in msgs:
        role = m["role"].upper()
        content = m["content"][:200]
        ts = m.get("created_at", "")[:19]
        print(f"  [{ts}] {role}: {content}")
    print("---")


# ── main loop ──────────────────────────────────────────────────────────────────

def chat_loop(base_url: str, product_id: str | None, conversation_id: str | None, use_stream: bool):
    _print_banner()
    if conversation_id:
        print(f"  Reprise conversation : {conversation_id}")
    if product_id:
        print(f"  Contexte produit : {product_id}")
    print()

    while True:
        try:
            user_input = input(f"{C_GREEN}Vous>{C_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print("Au revoir.")
            break
        if user_input.lower() == "/new":
            conversation_id = None
            product_id = None
            print(f"  {C_GREY}Nouvelle conversation démarrée.{C_RESET}\n")
            continue
        if user_input.lower() == "/id":
            print(f"  conversation_id : {conversation_id or '(aucun)'}\n")
            continue
        if user_input.lower() == "/history":
            if not conversation_id:
                print("  Aucune conversation active.\n")
            else:
                _show_history(base_url, conversation_id)
            continue

        payload = {
            "message": user_input,
            "conversation_id": conversation_id,
            "product_id": product_id,
        }

        if use_stream:
            resp = _stream_chat(base_url, payload)
            if resp is None:
                continue
            conversation_id = resp.get("conversation_id") or conversation_id
            product_id = resp.get("product_id") or product_id
            _print_meta(resp)
        else:
            print(f"  {C_GREY}…réflexion{C_RESET}")
            try:
                resp = _post(f"{base_url}/api/v1/chat", payload)
            except RuntimeError as e:
                print(f"\n  {C_RED}Erreur :{C_RESET} {e}\n")
                continue
            conversation_id = resp.get("conversation_id") or conversation_id
            product_id = resp.get("product_id") or product_id
            _print_assistant_static(resp.get("answer", ""), resp)


def main():
    parser = argparse.ArgumentParser(description="Chat CLI for Competitive Pricing Agent")
    parser.add_argument("--url", default=BASE_URL, help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--product-id", default=None, help="Start with a specific product_id")
    parser.add_argument("--resume", default=None, metavar="CONV_ID", help="Resume an existing conversation_id")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (use blocking request)")
    args = parser.parse_args()

    try:
        _get(f"{args.url}/health")
    except Exception as e:
        print(f"Impossible de joindre l'API {args.url}: {e}", file=sys.stderr)
        print("Démarrez le serveur : uvicorn api.main:app --reload", file=sys.stderr)
        sys.exit(1)

    chat_loop(
        base_url=args.url,
        product_id=args.product_id,
        conversation_id=args.resume,
        use_stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
