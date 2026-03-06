#!/usr/bin/env python3
"""
Standalone Telegram notification helper for FRANKENSTALLM 3B training.

Usage:
    python3 scripts/telegram_notify.py "Your message here"
    python3 scripts/telegram_notify.py "<b>Bold</b> message" --parse-mode HTML

Function API:
    from scripts.telegram_notify import send_telegram
    send_telegram("message text")
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error
import logging
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
TIMEOUT   = 15  # seconds
MAX_MSG_LEN = 4096  # Telegram limit

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [telegram_notify] %(levelname)s: %(message)s",
)
log = logging.getLogger("telegram_notify")


def send_telegram(
    message: str,
    parse_mode: str = "HTML",
    token: str = BOT_TOKEN,
    chat_id: str = CHAT_ID,
    disable_web_page_preview: bool = True,
) -> bool:
    """
    Send a Telegram message via Bot API using urllib (curl-free).

    Args:
        message:  Text to send (HTML or Markdown depending on parse_mode).
        parse_mode: "HTML" or "Markdown" or "" (plain).
        token:    Bot token (defaults to module-level BOT_TOKEN).
        chat_id:  Recipient chat/channel ID.
        disable_web_page_preview: Suppress link previews.

    Returns:
        True on success, False on any error.
    """
    if not message:
        log.warning("Empty message — skipping send.")
        return False

    # Truncate if over Telegram limit, with notice
    if len(message) > MAX_MSG_LEN:
        truncated_notice = "\n\n<i>[message truncated]</i>" if parse_mode == "HTML" else "\n\n[message truncated]"
        message = message[: MAX_MSG_LEN - len(truncated_notice)] + truncated_notice

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload: dict = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    data = urllib.parse.urlencode(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            if result.get("ok"):
                return True
            else:
                log.error("Telegram API error: %s", result.get("description", result))
                return False

    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        log.error("HTTP %d from Telegram: %s", e.code, err_body)
        return False

    except urllib.error.URLError as e:
        log.error("Network error sending Telegram message: %s", e.reason)
        return False

    except json.JSONDecodeError as e:
        log.error("Failed to parse Telegram response: %s", e)
        return False

    except Exception as e:  # noqa: BLE001
        log.error("Unexpected error in send_telegram: %s", e)
        return False


def send_telegram_safe(message: str, **kwargs) -> bool:
    """
    Wrapper that catches ALL exceptions — guaranteed never to crash the caller.
    Suitable for embedding in training loops where stability is critical.
    """
    try:
        return send_telegram(message, **kwargs)
    except Exception as e:  # noqa: BLE001
        log.error("send_telegram_safe caught unhandled exception: %s", e)
        return False


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Send a Telegram message from the command line."
    )
    parser.add_argument("message", nargs="?", help="Message text to send")
    parser.add_argument(
        "--parse-mode",
        default="HTML",
        choices=["HTML", "Markdown", "MarkdownV2", ""],
        help="Telegram parse_mode (default: HTML)",
    )
    parser.add_argument(
        "--token", default=BOT_TOKEN, help="Override bot token"
    )
    parser.add_argument(
        "--chat-id", default=CHAT_ID, help="Override chat ID"
    )
    args = parser.parse_args()

    # Allow piped stdin if no positional arg given
    if args.message is None:
        if not sys.stdin.isatty():
            args.message = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    ok = send_telegram(
        args.message,
        parse_mode=args.parse_mode,
        token=args.token,
        chat_id=args.chat_id,
    )

    if ok:
        print("Telegram message sent successfully.")
        sys.exit(0)
    else:
        print("ERROR: Failed to send Telegram message.", file=sys.stderr)
        sys.exit(1)
