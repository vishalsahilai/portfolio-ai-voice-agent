"""
memory/context_builder.py

Builds the Gemini contents list using the exact 3-phase hybrid memory algorithm.

Phase 1 (message_count == 0): current message only.
Phase 2 (message_count == 1): previous turn + current message.
Phase 3 (message_count >= 2): all summaries + current message (no raw history).
"""

from typing import List, Optional

from google.genai import types

from memory.memory_manager import memory_manager
from utils.logger import get_logger

logger = get_logger(__name__)

_SUMMARY_BLOCK_HEADER = "[Conversation Summary]"
_EXCHANGE_TEMPLATE = (
    "[Exchange {n}]\n"
    "User Intent: {user_intent}\n"
    "Bot Response: {bot_response}\n"
    "Context: {context}"
)
_RAG_BLOCK_TEMPLATE = "[Relevant context from knowledge base:\n{context}]"


def _format_summaries(summaries: List[dict]) -> str:
    """Format structured summary dicts into a readable block."""
    lines = [_SUMMARY_BLOCK_HEADER, ""]
    for i, s in enumerate(summaries, 1):
        lines.append(
            _EXCHANGE_TEMPLATE.format(
                n=i,
                user_intent=s.get("user_intent", ""),
                bot_response=s.get("bot_response", ""),
                context=s.get("context", ""),
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _make_user_content(user_text: str, retrieved_context: str = "") -> types.Content:
    """Build the final user Content object, injecting RAG context if present."""
    if retrieved_context:
        rag_block = _RAG_BLOCK_TEMPLATE.format(context=retrieved_context)
        full_text = f"{rag_block}\n\n{user_text}"
    else:
        full_text = user_text

    return types.Content(
        role="user",
        parts=[types.Part(text=full_text)],
    )


def get_context_for_llm(
    message_count: int,
    current_message: str,
    last_messages: Optional[dict] = None,
    summaries: Optional[List[dict]] = None,
) -> str:
    """
    Pure (non-async) context builder used by prompt_builder.py.

    Returns a plain text string representing the memory portion of the prompt.

    message_count == 0  → Phase 1: current message only
    message_count == 1  → Phase 2: previous turn + current message
    message_count >= 2  → Phase 3: all summaries + current message
    """
    summaries = summaries or []
    last_messages = last_messages or {}

    if message_count == 0:
        # Phase 1 — no history
        return current_message

    if message_count == 1:
        # Phase 2 — include the one previous exchange
        prev_user = last_messages.get("user", "")
        prev_assistant = last_messages.get("assistant", "")
        parts = []
        if prev_user or prev_assistant:
            parts.append("[Previous Turn]")
            if prev_user:
                parts.append(f"User: {prev_user}")
            if prev_assistant:
                parts.append(f"Assistant: {prev_assistant}")
            parts.append("")
        parts.append("[Current Message]")
        parts.append(current_message)
        return "\n".join(parts)

    # Phase 3 — summaries only, no raw history
    parts = []
    if summaries:
        parts.append(_format_summaries(summaries))
        parts.append("")
    parts.append("[Current Message]")
    parts.append(current_message)
    return "\n".join(parts)


async def build_gemini_context(
    session_id: str,
    new_user_text: str,
    retrieved_context: str = "",
    message_count: int = 0,
) -> List[types.Content]:
    """
    Assemble the full contents list for a Gemini call.

    session_id      : used to load summaries / last_messages from MongoDB
    new_user_text   : what the user just said (transcribed by Whisper)
    retrieved_context: RAG chunks from retriever.retrieve() — "" if none
    message_count   : pre-increment count — drives which phase to use
                      (0 = first message, 1 = second message, 2+ = third+)

    Returns List[types.Content] ready to pass as `contents` to Gemini.
    """
    contents: List[types.Content] = []

    # ── Phase 1: First message — no history ───────────────────────────────────
    if message_count == 0:
        logger.info(f"[{session_id}] Phase 1 — first message, no history")
        contents.append(_make_user_content(new_user_text, retrieved_context))
        return contents

    # ── Phase 2: Second message — one raw previous exchange ───────────────────
    if message_count == 1:
        logger.info(f"[{session_id}] Phase 2 — including previous exchange")
        last_messages = await memory_manager.get_last_messages(session_id) or {}
        prev_user = last_messages.get("user", "")
        prev_assistant = last_messages.get("assistant", "")

        if prev_user:
            # Send previous turn as actual conversation turns
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=prev_user)],
                )
            )
        if prev_assistant:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=prev_assistant)],
                )
            )

        contents.append(_make_user_content(new_user_text, retrieved_context))
        return contents

    # ── Phase 3: Third message and beyond — summaries only ────────────────────
    logger.info(f"[{session_id}] Phase 3 — using summaries (message_count={message_count})")
    summaries = await memory_manager.get_summaries(session_id)

    if summaries:
        formatted = _format_summaries(summaries)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=formatted)],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text="Understood. I have the conversation context.")],
            )
        )
        logger.info(f"[{session_id}] Injected {len(summaries)} summaries")
    else:
        # Summaries not ready yet (background task still running) — graceful fallback
        logger.warning(
            f"[{session_id}] Phase 3 but no summaries found — responding with current message only"
        )

    contents.append(_make_user_content(new_user_text, retrieved_context))
    return contents
