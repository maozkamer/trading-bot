"""
Chain-of-thought transparency helpers.
Streams agent thinking steps to Telegram so the user can see what's happening.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

_TOKEN = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("BOT_TOKEN")

_STEP_ICONS = {
    "think":  "🤔",
    "tool":   "🔧",
    "result": "📊",
    "done":   "💡",
    "error":  "⚠️",
}


def _send_sync(chat_id: int | str, text: str) -> None:
    """Fire-and-forget sync HTTP POST to Telegram sendMessage."""
    if not _TOKEN:
        log.debug("transparency: no token, skipping send")
        return
    url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
    try:
        httpx.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as exc:
        log.debug("transparency send failed: %s", exc)


async def _send_async(chat_id: int | str, text: str) -> None:
    await asyncio.get_event_loop().run_in_executor(None, _send_sync, chat_id, text)


def _format(icon: str, label: str, detail: str = "") -> str:
    msg = f"{icon} <b>{label}</b>"
    if detail:
        short = detail[:300] + ("…" if len(detail) > 300 else "")
        msg += f"\n<code>{short}</code>"
    return msg


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def think_step(chat_id: int | str, thought: str) -> None:
    """Send a thinking step to Telegram (sync, non-blocking best-effort)."""
    msg = _format(_STEP_ICONS["think"], "חושב…", thought)
    _send_sync(chat_id, msg)


def tool_call_step(chat_id: int | str, tool_name: str, tool_input: dict[str, Any]) -> None:
    """Announce a tool call."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
    msg = _format(_STEP_ICONS["tool"], f"מפעיל {tool_name}", args_str)
    _send_sync(chat_id, msg)


def tool_result_step(chat_id: int | str, tool_name: str, result: Any) -> None:
    """Show a summarised tool result."""
    if isinstance(result, dict):
        detail = str(result)
    elif isinstance(result, list):
        detail = f"[{len(result)} items]"
    else:
        detail = str(result)
    msg = _format(_STEP_ICONS["result"], f"תוצאה: {tool_name}", detail)
    _send_sync(chat_id, msg)


def conclusion_step(chat_id: int | str, conclusion: str) -> None:
    """Send the final conclusion step."""
    msg = _format(_STEP_ICONS["done"], "מסקנה", conclusion)
    _send_sync(chat_id, msg)


def error_step(chat_id: int | str, message: str) -> None:
    """Send an error notice."""
    msg = _format(_STEP_ICONS["error"], "שגיאה", message)
    _send_sync(chat_id, msg)


# Async variants for use inside async agent loop
async def think_step_async(chat_id: int | str, thought: str) -> None:
    await _send_async(chat_id, _format(_STEP_ICONS["think"], "חושב…", thought))


async def tool_call_step_async(chat_id: int | str, tool_name: str, tool_input: dict) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
    await _send_async(chat_id, _format(_STEP_ICONS["tool"], f"מפעיל {tool_name}", args_str))


async def tool_result_step_async(chat_id: int | str, tool_name: str, result: Any) -> None:
    detail = str(result)[:300]
    await _send_async(chat_id, _format(_STEP_ICONS["result"], f"תוצאה: {tool_name}", detail))
