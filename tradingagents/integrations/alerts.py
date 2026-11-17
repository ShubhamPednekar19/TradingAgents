"""Where trade alerts go: the console, a Telegram chat, or a webhook.

Each notifier is configured from environment variables so no secret has to be
passed on the command line:

    TRADINGAGENTS_TELEGRAM_BOT_TOKEN   bot token from @BotFather
    TRADINGAGENTS_TELEGRAM_CHAT_ID     your chat id (message the bot, then read getUpdates)
    TRADINGAGENTS_ALERT_WEBHOOK_URL    Slack / Discord / any URL that accepts a JSON POST

A failing notifier is logged and skipped, so one broken channel never stops the
others or the watcher.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable

import requests

logger = logging.getLogger(__name__)

Notifier = Callable[[str], None]


def console_notifier(message: str) -> None:
    print(message, flush=True)


def telegram_notifier(token: str, chat_id: str) -> Notifier:
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(message: str) -> None:
        response = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
        response.raise_for_status()

    return send


def webhook_notifier(url: str) -> Notifier:
    def send(message: str) -> None:
        # "text" is read by Slack, "content" by Discord; other receivers get both.
        response = requests.post(url, json={"text": message, "content": message}, timeout=15)
        response.raise_for_status()

    return send


def notifiers_from_env(console: bool = True) -> list[Notifier]:
    found: list[Notifier] = [console_notifier] if console else []
    token = os.getenv("TRADINGAGENTS_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TRADINGAGENTS_TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        found.append(telegram_notifier(token, chat_id))
    webhook = os.getenv("TRADINGAGENTS_ALERT_WEBHOOK_URL", "").strip()
    if webhook:
        found.append(webhook_notifier(webhook))
    return found


def broadcast(message: str, notifiers: Iterable[Notifier]) -> None:
    for notify in notifiers:
        try:
            notify(message)
        except Exception as exc:  # noqa: BLE001 - one channel must not stop the rest
            logger.warning("alert channel failed: %s", exc)
