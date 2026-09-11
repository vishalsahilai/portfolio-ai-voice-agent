import asyncio
import time
from typing import Any, Dict, Optional

from elevenlabs.core.api_error import ApiError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from audio.stt_deepgram import deepgram_stt
from call.call_state_machine import CallState
from call.session_manager import Session, session_manager
from llm.llm_service import AllLLMProvidersFailed, LLMApplicationError, llm_service
from memory.context_builder import build_gemini_context
from memory.memory_manager import memory_manager
from memory.summarizer import summarize_exchange_in_background
from rag.retriever import retriever
from tts.voice_manager import AllElevenLabsKeysExhausted, voice_manager
from utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter()

_MIN_TRANSCRIPT_LENGTH = 2
_JUNK_PHRASES = frozenset({"...", ". . ."})
MAX_QUESTIONS_PER_DAY = 7


def _valid_transcript(text: str) -> bool:
    text = text.strip()
    return len(text) >= _MIN_TRANSCRIPT_LENGTH and text.lower() not in _JUNK_PHRASES


def _plain_alignment(value: Any) -> Optional[Dict[str, list]]:
    if not value:
        return None

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            value = value.model_dump()
        except Exception:
            return None

    if not isinstance(value, dict):
        return None

    chars_raw = value.get("chars") or value.get("characters")

    starts_raw = (
        value.get("char_start_times_ms")
        or value.get("charStartTimesMs")
        or value.get("character_start_times_ms")
    )

    durations_raw = (
        value.get("char_durations_ms")
        or value.get("charsDurationsMs")
        or value.get("charDurationsMs")
        or value.get("character_durations_ms")
        or []
    )

    if callable(chars_raw) or callable(starts_raw) or callable(durations_raw):
        return None

    if not isinstance(chars_raw, (list, tuple)):
        return None

    if not isinstance(starts_raw, (list, tuple)):
        return None

    if not isinstance(durations_raw, (list, tuple)):
        durations_raw = []

    chars, starts, durations = [], [], []
    count = min(len(chars_raw), len(starts_raw))

    for i in range(count):
        char = chars_raw[i]
        start = starts_raw[i]
        duration = durations_raw[i] if i < len(durations_raw) else 0

        if not isinstance(char, str):
            continue

        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue

        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            duration = 0

        chars.append(char)
        starts.append(float(start))
        durations.append(float(duration))

    if not chars:
        return None

    return {
        "chars": chars,
        "char_start_times_ms": starts,
        "char_durations_ms": durations,
    }


async def _send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({
            "type": "error",
            "message": message,
        })
    except Exception:
        pass


async def _post_turn_updates(
    session: Session,
    user_text: str,
    bot_text: str,
    message_count: int,
) -> None:
    results = await asyncio.gather(
        summarize_exchange_in_background(
            session_id=session.session_id,
            user_text=user_text,
            bot_text=bot_text,
            message_count=message_count,
        ),
        memory_manager.update_last_messages(
            session_id=session.session_id,
            user_message=user_text,
            assistant_message=bot_text,
        ),
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            logger.warning(
                f"[{session.session_id}] "
                f"Background memory update failed: {result}"
            )


async def _process_turn(
    session: Session,
    websocket: WebSocket,
    user_text: str,
    daily_usage_start: int,
) -> None:
    user_text = " ".join(user_text.split()).strip()

    if not _valid_transcript(user_text):
        logger.info(
            f"[{session.session_id}] "
            f"Transcript ignored: '{user_text}'"
        )
        return
    
    if session.state_machine.state != CallState.LISTENING:
        logger.info(
            f"[{session.session_id}] "
            "Ignoring transcript while agent is busy"
        )
        return

    used_before_turn = (
        daily_usage_start
        + session.message_count
    )

    if used_before_turn >= MAX_QUESTIONS_PER_DAY:
        logger.info(
            f"[{session.session_id}] "
            f"Daily browser question limit reached "
            f"({MAX_QUESTIONS_PER_DAY})"
        )

        await websocket.send_json({
            "type": "play_limit_reached",
            "used": MAX_QUESTIONS_PER_DAY,
            "limit": MAX_QUESTIONS_PER_DAY,
        })

        return

    session.interrupted = False
    session.state_machine.transition(
        CallState.THINKING
    )

    turn_started = time.perf_counter()
    tts_connect_task = None
    tts_ws = None
    audio_started = False
    first_audio_sent = False

    try:
        # Connect ElevenLabs while memory + RAG run.
        tts_connect_task = asyncio.create_task(
            voice_manager.connect_stream()
        )

        message_count_task = asyncio.create_task(
            memory_manager.increment_message_count(
                session.session_id
            )
        )

        rag_task = asyncio.create_task(
            asyncio.to_thread(
                retriever.retrieve,
                user_text,
            )
        )

        message_count, retrieved_context = await asyncio.gather(
            message_count_task,
            rag_task,
        )

        logger.info(
            f"[{session.session_id}] "
            f"Memory + RAG ready in "
            f"{time.perf_counter() - turn_started:.2f}s"
        )

        contents = await build_gemini_context(
            session_id=session.session_id,
            new_user_text=user_text,
            retrieved_context=retrieved_context,
            message_count=message_count,
        )

        try:
            tts_ws = await tts_connect_task

        except AllElevenLabsKeysExhausted:
            logger.error(
                f"[{session.session_id}] "
                "All ElevenLabs accounts exhausted"
            )

            await _send_error(
                websocket,
                "Voice generation is temporarily unavailable.",
            )
            return

        bot_parts = []

        async def llm_text_stream():
            async for chunk in (
                llm_service
                .generate_reply_stream_from_contents(
                    contents
                )
            ):
                if chunk:
                    bot_parts.append(chunk)
                    yield chunk

        session.state_machine.transition(CallState.SPEAKING)

        try:
            await websocket.send_json({
                "type": "audio_start",
            })

            audio_started = True

            async for packet in voice_manager.stream(
                llm_text_stream(),
                ws=tts_ws,
            ):
                if session.interrupted:
                    logger.info(
                        f"[{session.session_id}] Reply interrupted"
                    )
                    break

                audio = b""
                alignment = None

                if isinstance(packet, (bytes, bytearray, memoryview)):
                    audio = bytes(packet)

                elif isinstance(packet, dict):
                    raw_audio = packet.get("audio", b"")

                    if isinstance(
                        raw_audio,
                        (bytes, bytearray, memoryview),
                    ):
                        audio = bytes(raw_audio)

                    raw_alignment = (
                        packet.get("alignment")
                        or packet.get("normalized_alignment")
                        or packet.get("normalizedAlignment")
                    )

                    alignment = _plain_alignment(
                        raw_alignment
                    )

                else:
                    logger.warning(
                        f"[{session.session_id}] "
                        f"Unsupported ElevenLabs packet: "
                        f"{type(packet).__name__}"
                    )

                if alignment:
                    await websocket.send_json({
                        "type": "audio_alignment",
                        "alignment": alignment,
                    })

                if audio:
                    if not first_audio_sent:
                        first_audio_sent = True

                        logger.info(
                            f"[{session.session_id}] "
                            f"FIRST AUDIO in "
                            f"{time.perf_counter() - turn_started:.2f}s ✅"
                        )

                    await websocket.send_bytes(audio)

        except AllLLMProvidersFailed as exc:
            logger.error(
                f"[{session.session_id}] "
                f"All LLM providers failed: {exc}"
            )

            await _send_error(
                websocket,
                "AI service is temporarily unavailable.",
            )
            return

        except LLMApplicationError as exc:
            logger.error(
                f"[{session.session_id}] "
                f"LLM application error: {exc}"
            )

            await _send_error(
                websocket,
                "AI response generation failed.",
            )
            return

        except AllElevenLabsKeysExhausted:
            logger.error(
                f"[{session.session_id}] "
                "All ElevenLabs accounts exhausted"
            )

            await _send_error(
                websocket,
                "Voice generation is temporarily unavailable.",
            )
            return

        except ApiError as exc:
            logger.error(
                f"[{session.session_id}] "
                f"ElevenLabs error: {exc}"
            )

            await _send_error(
                websocket,
                "Voice generation failed.",
            )
            return

        except Exception as exc:
            logger.exception(
                f"[{session.session_id}] "
                f"Streaming response failed: {exc}"
            )

            await _send_error(
                websocket,
                "Voice response failed.",
            )
            return

        finally:
            if audio_started:
                try:
                    await websocket.send_json({
                        "type": "audio_end",
                    })
                except Exception:
                    pass

        if session.interrupted:
            return

        bot_text = "".join(bot_parts).strip()

        if not bot_text:
            logger.warning(
                f"[{session.session_id}] "
                "LLM returned empty response"
            )

            await _send_error(
                websocket,
                "The agent could not generate a response.",
            )
            return

        session.conversation_history.extend([
            {
                "role": "user",
                "text": user_text,
            },
            {
                "role": "model",
                "text": bot_text,
            },
        ])

        session.message_count = message_count + 1

        daily_used = min(
            MAX_QUESTIONS_PER_DAY,
            daily_usage_start
            + session.message_count,
        )

        await websocket.send_json({
            "type": "usage_update",
            "used": daily_used,
            "limit": MAX_QUESTIONS_PER_DAY,
        })

        session.last_messages = {
            "user": user_text,
            "assistant": bot_text,
        }

        asyncio.create_task(
            _post_turn_updates(
                session=session,
                user_text=user_text,
                bot_text=bot_text,
                message_count=message_count,
            )
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        logger.exception(
            f"[{session.session_id}] "
            f"Turn processing failed: {exc}"
        )

        await _send_error(
            websocket,
            "Something went wrong. Please try again.",
        )

    finally:
        if tts_connect_task:
            if not tts_connect_task.done():
                tts_connect_task.cancel()

            await asyncio.gather(
                tts_connect_task,
                return_exceptions=True,
            )

        if tts_ws and not getattr(tts_ws, "closed", True):
            try:
                await tts_ws.close()
            except Exception:
                pass

        if session.state_machine.state != CallState.LISTENING:
            session.state_machine.interrupt()


async def _consume_deepgram_events(
    session: Session,
    websocket: WebSocket,
    deepgram_session,
) -> None:
    """
    Streams Deepgram interim transcripts directly to the browser.

    Example:
        You: Hello
        You: Hello I want
        You: Hello I want to know
        You: Hello I want to know more about Vishal
    """

    try:
        while True:
            event = await deepgram_session.next_event()

            if not event:
                continue

            if event.type == "speech_started":
                await websocket.send_json({
                    "type": "user_speech_start",
                })
                continue

            if event.type == "transcript":
                text = " ".join(
                    event.text.split()
                ).strip()

                if not text:
                    continue

                await websocket.send_json({
                    "type": "user_transcript",
                    "text": text,
                    "is_final": bool(event.is_final),
                    "speech_final": bool(event.speech_final),
                })

                continue

            if event.type == "utterance":
                text = " ".join(
                    event.text.split()
                ).strip()

                if text:
                    await websocket.send_json({
                        "type": "user_transcript_end",
                        "text": text,
                    })

                continue

            if event.type == "connection_closed":
                logger.warning(
                    f"[{session.session_id}] "
                    "Deepgram event stream closed"
                )

            elif event.type == "error":
                logger.warning(
                    f"[{session.session_id}] "
                    f"Deepgram event error: {event.text}"
                )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        logger.error(
            f"[{session.session_id}] "
            f"Deepgram live transcript consumer error: {exc}"
        )


async def _consume_deepgram_utterances(
    session: Session,
    websocket: WebSocket,
    deepgram_session,
    daily_usage_start: int,
) -> None:
    try:
        while True:
            user_text = await deepgram_session.next_utterance()

            if user_text:
                await _process_turn(
                    session=session,
                    websocket=websocket,
                    user_text=user_text,
                    daily_usage_start=daily_usage_start,
                )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        logger.error(
            f"[{session.session_id}] "
            f"Deepgram consumer error: {exc}"
        )


@router.websocket("/ws/call")
async def call_websocket(websocket: WebSocket) -> None:
    try:
        daily_usage_start = int(
            websocket.query_params.get(
                "used",
                "0",
            )
        )
    except (TypeError, ValueError):
        daily_usage_start = 0

    daily_usage_start = max(
        0,
        min(
            MAX_QUESTIONS_PER_DAY,
            daily_usage_start,
        ),
    )

    await websocket.accept()

    session = session_manager.create_session()

    deepgram_session = None
    utterance_task = None
    event_task = None
    memory_task = None
    deepgram_task = None

    try:
        await websocket.send_json({
            "type": "session_started",
            "session_id": session.session_id,
        })

        logger.info(
            f"[{session.session_id}] "
            f"Browser daily usage at call start: "
            f"{daily_usage_start}/{MAX_QUESTIONS_PER_DAY}"
        )

        if daily_usage_start >= MAX_QUESTIONS_PER_DAY:
            logger.info(
                f"[{session.session_id}] "
                "Daily browser question limit already reached"
            )

            await websocket.send_json({
                "type": "play_limit_reached",
                "used": MAX_QUESTIONS_PER_DAY,
                "limit": MAX_QUESTIONS_PER_DAY,
            })

            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(
                        code=message.get("code", 1000)
                    )

        memory_task = asyncio.create_task(
            memory_manager.create_session(
                session.session_id
            )
        )

        deepgram_task = asyncio.create_task(
            deepgram_stt.create_session(
                session.session_id
            )
        )

        await websocket.send_json({
            "type": "play_greeting",
        })

        memory_result, deepgram_result = await asyncio.gather(
            memory_task,
            deepgram_task,
            return_exceptions=True,
        )

        if isinstance(memory_result, Exception):
            raise memory_result

        if isinstance(deepgram_result, Exception):
            raise deepgram_result

        deepgram_session = deepgram_result

        logger.info(
            f"[{session.session_id}] "
            f"Voice pipeline ready ✅ "
            f"(Deepgram key "
            f"{deepgram_session.active_key_number})"
        )

        # Final utterances trigger the AI response.
        utterance_task = asyncio.create_task(
            _consume_deepgram_utterances(
                session=session,
                websocket=websocket,
                deepgram_session=deepgram_session,
                daily_usage_start=daily_usage_start,
            ),
            name=f"deepgram-consumer-{session.session_id}",
        )

        # Interim transcripts are streamed live to the browser.
        event_task = asyncio.create_task(
            _consume_deepgram_events(
                session=session,
                websocket=websocket,
                deepgram_session=deepgram_session,
            ),
            name=f"deepgram-events-{session.session_id}",
        )

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(
                    code=message.get("code", 1000)
                )

            pcm_bytes = message.get("bytes")

            if pcm_bytes is not None:
                if session.state_machine.state == CallState.LISTENING:
                    await deepgram_session.send_audio(
                        pcm_bytes
                    )

                continue

            control = message.get("text")

            if control is not None:
                logger.debug(
                    f"[{session.session_id}] "
                    f"Control: {control}"
                )

    except WebSocketDisconnect:
        logger.info(
            f"[{session.session_id}] "
            "Client disconnected"
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        logger.exception(
            f"[{session.session_id}] "
            f"WebSocket error: {exc}"
        )

        await _send_error(
            websocket,
            "Voice agent initialization failed.",
        )

    finally:
        consumer_tasks = [
            task
            for task in (
                utterance_task,
                event_task,
            )
            if task
        ]

        for task in consumer_tasks:
            if not task.done():
                task.cancel()

        if consumer_tasks:
            await asyncio.gather(
                *consumer_tasks,
                return_exceptions=True,
            )

        startup_tasks = [
            task
            for task in (
                memory_task,
                deepgram_task,
            )
            if task
        ]

        for task in startup_tasks:
            if not task.done():
                task.cancel()

        if startup_tasks:
            await asyncio.gather(
                *startup_tasks,
                return_exceptions=True,
            )

        if deepgram_session:
            try:
                await deepgram_session.close()

            except Exception as exc:
                logger.warning(
                    f"[{session.session_id}] "
                    f"Deepgram cleanup failed: {exc}"
                )

        session_manager.end_session(
            session.session_id
        )

        logger.info(
            f"[{session.session_id}] "
            "Session cleaned up"
        )