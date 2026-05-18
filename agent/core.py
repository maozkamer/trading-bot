"""
Core agent loop: Claude API with tool use, transparency, and memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import anthropic

from agent.memory import init_memory_db, recall_memory, save_message, get_history
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.transparency import (
    conclusion_step,
    error_step,
    think_step,
    tool_call_step,
    tool_result_step,
)

log = logging.getLogger(__name__)

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_MODEL = "claude-opus-4-5"
_MAX_ITERATIONS = 15

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "agent_prompts" / "system.md"


def _load_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        log.error("Could not load system prompt: %s", exc)
        return "אתה Maoz Trading Agent. ענה בעברית. אל תמציא נתונים — השתמש תמיד בכלים."


def _inject_memory(user_message: str, system_prompt: str) -> str:
    """Prepend relevant long-term memories to the system prompt."""
    try:
        memories = recall_memory(user_message, limit=3)
        if not memories:
            return system_prompt
        mem_block = "\n\n## זיכרון ארוך טווח רלוונטי\n"
        for m in memories:
            mem_block += f"- [{m['category']}] {m['content']}\n"
        return system_prompt + mem_block
    except Exception as exc:
        log.debug("_inject_memory failed: %s", exc)
        return system_prompt


def _execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON-serialisable string result."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = fn(**tool_input)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as exc:
        log.error("Tool %s raised: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


def run_agent(user_message: str, chat_id: int | str) -> str:
    """
    Run the full Claude agent loop for a user message.
    Sends chain-of-thought steps to Telegram via transparency helpers.
    Returns the final text response.
    """
    if not _ANTHROPIC_KEY:
        msg = "❌ ANTHROPIC_API_KEY לא מוגדר."
        error_step(chat_id, msg)
        return msg

    init_memory_db()
    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    system_prompt = _inject_memory(user_message, _load_system_prompt())

    save_message(chat_id, "user", user_message)
    messages: list[dict] = get_history(chat_id, limit=10)
    final_text = ""

    for iteration in range(_MAX_ITERATIONS):
        log.info("Agent loop iteration %d", iteration + 1)
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except Exception as exc:
            log.error("Anthropic API error: %s", exc)
            error_step(chat_id, str(exc))
            return f"❌ שגיאת API: {exc}"

        # Collect text and tool_use blocks
        text_parts: list[str] = []
        tool_uses: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                if block.text.strip():
                    think_step(chat_id, block.text.strip())
            elif block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        if text_parts:
            final_text = "\n".join(text_parts).strip()

        # If no tool calls → Claude is done
        if response.stop_reason == "end_turn" or not tool_uses:
            if final_text:
                conclusion_step(chat_id, final_text[:400])
            break

        # Execute tools and build tool_result blocks
        tool_results: list[dict] = []
        for tu in tool_uses:
            tool_call_step(chat_id, tu["name"], tu["input"])
            result_str = _execute_tool(tu["name"], tu["input"])
            tool_result_step(chat_id, tu["name"], result_str)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })

        # Append assistant turn + tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    else:
        log.warning("Agent hit MAX_ITERATIONS (%d)", _MAX_ITERATIONS)
        error_step(chat_id, f"הגעתי למקסימום איטרציות ({_MAX_ITERATIONS}).")

    if final_text:
        save_message(chat_id, "assistant", final_text)
    return final_text or "אין תוצאה."


# ─────────────────────────────────────────────────────────────
#  Async wrapper (used by scheduler and bot handler)
# ─────────────────────────────────────────────────────────────

async def run_agent_async(user_message: str, chat_id: int | str) -> str:
    """Run the blocking agent loop in a thread so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_agent, user_message, chat_id)
