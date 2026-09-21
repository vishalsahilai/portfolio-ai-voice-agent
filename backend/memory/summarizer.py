"""
memory/summarizer.py

Background exchange summarization using Gemini.
Fires as asyncio.create_task() so it never blocks the WebSocket response.
"""

import asyncio
import json
import re
from typing import Optional

from google.genai import types

from memory.memory_manager import memory_manager
from utils.logger import get_logger

logger = get_logger(__name__)

_SUMMARIZE_PROMPT = """Summarize this exchange in strict JSON with exactly these keys:
user_intent, bot_response, context.
Respond ONLY with valid JSON. No markdown. No explanation. No preamble.

Exchange:
User: {user_text}
Assistant: {bot_text}

Expected format:
{{
  "user_intent": "what the user wanted (1 sentence)",
  "bot_response": "what the bot said (1-2 sentences)",
  "context": "important details to remember for future turns"
}}"""

# Regex to strip JSON code blocks from bot_response before summarizing
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\{.*?\}\s*```", re.DOTALL)


def _strip_json_blocks(text: str) -> str:
    """Remove embedded JSON/code blocks from bot response before summarization."""
    return _JSON_BLOCK_RE.sub("", text).strip()


def _normalize_smart_quotes(text: str) -> str:
    """Replace Unicode smart quotes with standard ASCII quotes for JSON parsing."""
    replacements = {
        "\u2018": "'",   # '
        "\u2019": "'",   # '
        "\u201c": '"',   # "
        "\u201d": '"',   # "
        "\u2032": "'",   # ′
        "\u2033": '"',   # ″
    }
    for smart, plain in replacements.items():
        text = text.replace(smart, plain)
    return text


def _extract_text_from_response(response_content) -> str:
    """
    Handle Gemini returning content as either a list of blocks or a plain string.
    """
    if isinstance(response_content, list):
        return " ".join(
            block.get("text", "")
            for block in response_content
            if isinstance(block, dict)
        )
    if isinstance(response_content, str):
        return response_content
    return str(response_content)


def _parse_summary(raw: str) -> Optional[dict]:
    """
    Safely parse Gemini's JSON summary response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Smart/curly quotes
    - Missing required keys
    - Malformed JSON
    """
    try:
        clean = raw.strip()

        # Strip markdown fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            # Remove first line (```json or ```) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            clean = "\n".join(inner).strip()

        # Normalize smart quotes
        clean = _normalize_smart_quotes(clean)

        # Remove any remaining control characters that break JSON
        clean = clean.replace("\x00", "")

        parsed = json.loads(clean)

        required_keys = ("user_intent", "bot_response", "context")
        if all(k in parsed for k in required_keys):
            return {k: str(parsed[k]) for k in required_keys}

        logger.warning(f"Summary JSON missing required keys — got: {list(parsed.keys())}")
        return None

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse summary JSON: {e} — raw: {raw[:300]}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing summary: {e}")
        return None


async def summarize_exchange_in_background(
    session_id: str,
    user_text: str,
    bot_text: str,
    message_count: int,
) -> None:
    """
    Decide whether to summarize and fire as a non-blocking background task.

    message_count: the PRE-increment count at the time of the turn.
      - message_count == 0 → Turn 1: no summarization
      - message_count == 1 → Turn 2: summarize the completed exchange
      - message_count >= 2 → Turn 3+: summarize the completed exchange

    Called from websocket_routes.py after every completed exchange.
    The WebSocket response is already sent before this is called.
    """
    if not memory_manager.is_configured:
        return

    # Turn 1 (message_count == 0 before increment) — no summarization needed
    if message_count < 1:
        logger.info(f"[{session_id}] Turn 1 — skipping summarization")
        return

    logger.info(
        f"[{session_id}] Firing background summarization "
        f"(message_count={message_count})"
    )
    asyncio.create_task(
        _run_summarization(session_id, user_text, bot_text)
    )


async def _run_summarization(
    session_id: str,
    user_text: str,
    bot_text: str,
) -> None:
    """
    Actual summarization — runs in background.
    Errors are caught and logged; they never surface to the WebSocket.
    """
    try:
        from llm.llm_service import llm_service  # local import avoids circular dependency

        # Strip any JSON blocks from bot_text before summarizing
        clean_bot_text = _strip_json_blocks(bot_text)

        prompt = _SUMMARIZE_PROMPT.format(
            user_text=user_text,
            bot_text=clean_bot_text,
        )

        logger.info(f"[{session_id}] Summarizing exchange with LLM provider...")

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        raw_response = await (
            llm_service.generate_reply_from_contents_async(
                contents
            )
        )

        if not raw_response:
            logger.warning(f"[{session_id}] LLM returned empty summary — using fallback")
            raw_text = ""
        else:
            # Handle list or string content
            raw_text = _extract_text_from_response(raw_response)

        summary = None
        if raw_text:
            summary = _parse_summary(raw_text)

        if not summary:
            # Graceful fallback — store truncated raw exchange
            summary = {
                "user_intent": user_text[:150],
                "bot_response": clean_bot_text[:150],
                "context": "Auto-generated fallback summary",
            }
            logger.warning(f"[{session_id}] Using fallback plain-text summary")

        await memory_manager.append_summary(session_id, summary)
        logger.info(f"[{session_id}] Exchange summarized and saved ✅")

    except Exception as e:
        logger.error(f"[{session_id}] Background summarization failed: {e}")
        # Do NOT re-raise — background task failures must never affect the WebSocket
