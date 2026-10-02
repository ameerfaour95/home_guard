"""Readiness probe: can the box's Telegram assistant actually hear the owner?

Run on the box:  python -m home_guard_project.box.telegram_check
Prints one line - READY / NOT_READY / SKIP - and exits 0 (ready or skip) or 1.
check_box.ps1 runs this and turns the line into a [PASS]/[WARN] row. Uses
getMe + getChatMember only (never getUpdates), so it never disturbs the running
inbox poller.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        import truststore  # noqa: PLC0415
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - the check should still try without it
        pass

    from dotenv import load_dotenv  # noqa: PLC0415

    from .boxconfig import MODE_INFERENCE, PROJECT_ROOT, get_option  # noqa: PLC0415
    from .telegram_notify import _as_chat_ids, check_group_readiness  # noqa: PLC0415

    load_dotenv(os.path.join(PROJECT_ROOT, "api_key.env"))

    if str(get_option("mode")) != MODE_INFERENCE:
        print("SKIP: data-collection mode - this box has no Telegram assistant")
        return 0
    if str(get_option("alert_channel") or "telegram") not in ("telegram", "both"):
        print("SKIP: alerts are not delivered over Telegram on this box")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_ids = _as_chat_ids(get_option("telegram_chat_ids"))
    if not token or not chat_ids:
        print("NOT_READY: Telegram bot token or chat id is missing - set them in setup")
        return 1

    bot = "bot"
    for chat in chat_ids:
        result = check_group_readiness(token, chat)
        bot = result.get("bot") or bot
        if not result.get("ready"):
            print(f"NOT_READY: @{bot} in {chat} - {result.get('reason')}")
            return 1
    print(f"READY: @{bot} can read the owner's messages in {len(chat_ids)} chat(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
