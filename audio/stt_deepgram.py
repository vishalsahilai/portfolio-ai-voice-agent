import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlencode

from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import WebSocketClientProtocol, connect as websocket_connect

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ENDPOINTING_MS = 350
DEFAULT_UTTERANCE_END_MS = 1000

KEEPALIVE_INTERVAL = 4.0
CONNECT_TIMEOUT = 10.0
MAX_EVENT_QUEUE = 100
MAX_UTTERANCE_QUEUE = 10


class DeepgramSTTError(Exception):
    pass


class DeepgramNotConfiguredError(DeepgramSTTError):
    pass


class AllDeepgramKeysExhausted(DeepgramSTTError):
    pass


@dataclass(slots=True)
class DeepgramEvent:
    type: str
    text: str = ""
    is_final: bool = False
    speech_final: bool = False
    confidence: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class DeepgramSTTSession:
    def __init__(
        self,
        session_id: str,
        api_keys: List[str],
        start_key_index: int = 0,
    ):
        if not api_keys:
            raise DeepgramNotConfiguredError(
                "No Deepgram API keys configured."
            )

        self.session_id = session_id
        self.api_keys = api_keys

        self.model = getattr(
            settings,
            "DEEPGRAM_MODEL",
            DEFAULT_MODEL,
        )

        self.language = getattr(
            settings,
            "DEEPGRAM_LANGUAGE",
            DEFAULT_LANGUAGE,
        )

        self.sample_rate = getattr(
            settings,
            "AUDIO_SAMPLE_RATE",
            DEFAULT_SAMPLE_RATE,
        )

        self.endpointing_ms = getattr(
            settings,
            "DEEPGRAM_ENDPOINTING_MS",
            DEFAULT_ENDPOINTING_MS,
        )

        self.utterance_end_ms = max(
            1000,
            getattr(
                settings,
                "DEEPGRAM_UTTERANCE_END_MS",
                DEFAULT_UTTERANCE_END_MS,
            ),
        )

        self._key_index = start_key_index % len(api_keys)
        self._active_key_index: Optional[int] = None

        self._ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._connected = asyncio.Event()

        self._closing = False
        self._last_audio_at = time.monotonic()

        self._final_parts: List[str] = []
        self._last_utterance = ""
        self._last_live_transcript = ""

        self._event_queue: asyncio.Queue[DeepgramEvent] = asyncio.Queue(
            maxsize=MAX_EVENT_QUEUE
        )

        self._utterance_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=MAX_UTTERANCE_QUEUE
        )

    @property
    def connected(self) -> bool:
        return (
            self._connected.is_set()
            and self._ws is not None
            and not self._ws.closed
        )

    @property
    def active_key_number(self) -> Optional[int]:
        return (
            self._active_key_index + 1
            if self._active_key_index is not None
            else None
        )

    def _build_url(self) -> str:
        params = {
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "punctuate": "true",
            "vad_events": "true",
            "endpointing": self.endpointing_ms,
            "utterance_end_ms": self.utterance_end_ms,
        }

        return f"{DEEPGRAM_URL}?{urlencode(params)}"

    async def connect(self) -> None:
        if self._closing:
            raise DeepgramSTTError(
                "Deepgram session is closed."
            )

        if self.connected:
            return

        async with self._connect_lock:
            if self.connected:
                return

            await self._cleanup_transport()

            last_error = None
            total = len(self.api_keys)
            start = self._key_index

            for offset in range(total):
                index = (start + offset) % total
                api_key = self.api_keys[index]

                try:
                    logger.info(
                        f"[{self.session_id}] "
                        f"Connecting Deepgram with key "
                        f"{index + 1}/{total}"
                    )

                    await self._open_connection(
                        api_key,
                        index,
                    )

                    self._key_index = index

                    logger.info(
                        f"[{self.session_id}] "
                        f"Deepgram connected ✅ "
                        f"(key {index + 1})"
                    )

                    return

                except asyncio.CancelledError:
                    raise

                except Exception as exc:
                    last_error = exc

                    logger.warning(
                        f"[{self.session_id}] "
                        f"Deepgram key {index + 1} failed: "
                        f"{self._safe_error(exc)}"
                    )

                    await self._cleanup_transport()

            raise AllDeepgramKeysExhausted(
                "All Deepgram API keys failed."
            ) from last_error

    async def _open_connection(
        self,
        api_key: str,
        key_index: int,
    ) -> None:
        try:
            ws = await asyncio.wait_for(
                websocket_connect(
                    self._build_url(),
                    extra_headers={
                        "Authorization": f"Token {api_key}"
                    },
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=4 * 1024 * 1024,
                    compression=None,
                ),
                timeout=CONNECT_TIMEOUT,
            )

        except asyncio.TimeoutError as exc:
            raise DeepgramSTTError(
                "Deepgram connection timed out."
            ) from exc

        self._ws = ws
        self._active_key_index = key_index
        self._connected.set()
        self._last_audio_at = time.monotonic()

        self._receive_task = asyncio.create_task(
            self._receive_loop(ws),
            name=f"deepgram-receive-{self.session_id}",
        )

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(ws),
            name=f"deepgram-keepalive-{self.session_id}",
        )

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or self._closing:
            return

        async with self._send_lock:
            if not self.connected:
                await self.connect()

            try:
                await self._ws.send(pcm_bytes)
                self._last_audio_at = time.monotonic()
                return

            except asyncio.CancelledError:
                raise

            except (ConnectionClosed, OSError, RuntimeError) as exc:
                logger.warning(
                    f"[{self.session_id}] "
                    f"Deepgram send failed: "
                    f"{self._safe_error(exc)}"
                )

            self._advance_key()
            await self._cleanup_transport()

            logger.info(
                f"[{self.session_id}] "
                "Switching Deepgram API key..."
            )

            await self.connect()

            try:
                await self._ws.send(pcm_bytes)
                self._last_audio_at = time.monotonic()

            except Exception as exc:
                raise DeepgramSTTError(
                    "Deepgram send failed after reconnect."
                ) from exc

    async def finalize(self) -> None:
        if not self.connected:
            return

        try:
            await self._ws.send(
                json.dumps({
                    "type": "Finalize"
                })
            )

        except Exception as exc:
            logger.debug(
                f"[{self.session_id}] "
                f"Finalize failed: {self._safe_error(exc)}"
            )

    async def next_utterance(
        self,
        timeout: Optional[float] = None,
    ) -> str:
        if timeout is None:
            return await self._utterance_queue.get()

        try:
            return await asyncio.wait_for(
                self._utterance_queue.get(),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            return ""

    async def next_event(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[DeepgramEvent]:
        if timeout is None:
            return await self._event_queue.get()

        try:
            return await asyncio.wait_for(
                self._event_queue.get(),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            return None

    async def events(self) -> AsyncIterator[DeepgramEvent]:
        while not self._closing:
            event = await self.next_event()

            if event:
                yield event

    async def _receive_loop(
        self,
        ws: WebSocketClientProtocol,
    ) -> None:
        try:
            async for raw in ws:
                if self._closing:
                    break

                if isinstance(raw, bytes):
                    continue

                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                await self._handle_message(
                    message
                )

        except asyncio.CancelledError:
            raise

        except ConnectionClosed as exc:
            if not self._closing:
                logger.warning(
                    f"[{self.session_id}] "
                    f"Deepgram disconnected "
                    f"(code={exc.code}, reason={exc.reason})"
                )

                await self._put_event(
                    DeepgramEvent(
                        type="connection_closed",
                        text=str(exc.reason or ""),
                    )
                )

                self._advance_key()

        except Exception as exc:
            if not self._closing:
                logger.error(
                    f"[{self.session_id}] "
                    f"Deepgram receive error: "
                    f"{self._safe_error(exc)}"
                )

                await self._put_event(
                    DeepgramEvent(
                        type="error",
                        text=self._safe_error(exc),
                    )
                )

                self._advance_key()

        finally:
            if self._ws is ws:
                self._connected.clear()

    async def _handle_message(
        self,
        message: Dict[str, Any],
    ) -> None:
        message_type = message.get(
            "type",
            "",
        )

        if message_type == "Results":
            await self._handle_results(
                message
            )
            return

        if message_type == "SpeechStarted":
            await self._put_event(
                DeepgramEvent(
                    type="speech_started",
                    raw=message,
                )
            )
            return

        if message_type == "UtteranceEnd":
            await self._flush_utterance()

            await self._put_event(
                DeepgramEvent(
                    type="utterance_end",
                    raw=message,
                )
            )
            return

        if message_type == "Metadata":
            await self._put_event(
                DeepgramEvent(
                    type="metadata",
                    raw=message,
                )
            )
            return

        if message_type == "Error":
            description = (
                message.get("description")
                or message.get("message")
                or message.get("error")
                or "Deepgram API error"
            )

            logger.error(
                f"[{self.session_id}] "
                f"Deepgram error: {description}"
            )

            await self._put_event(
                DeepgramEvent(
                    type="error",
                    text=str(description),
                    raw=message,
                )
            )

            self._advance_key()
            self._connected.clear()

    async def _handle_results(
        self,
        message: Dict[str, Any],
    ) -> None:
        alternatives = (
            message.get("channel", {})
            .get("alternatives", [])
        )

        if not alternatives:
            return

        best = alternatives[0]

        segment = " ".join(
            (
                best.get("transcript")
                or ""
            ).split()
        ).strip()

        is_final = bool(
            message.get("is_final")
        )

        speech_final = bool(
            message.get("speech_final")
        )

        confidence = float(
            best.get("confidence")
            or 0
        )

        if segment:
            if (
                is_final
                and (
                    not self._final_parts
                    or self._final_parts[-1] != segment
                )
            ):
                self._final_parts.append(
                    segment
                )

            if is_final:
                live_text = self._join_parts(
                    self._final_parts
                )
            else:
                live_text = self._join_parts(
                    [
                        *self._final_parts,
                        segment,
                    ]
                )

            if (
                live_text
                and live_text
                != self._last_live_transcript
            ):
                self._last_live_transcript = (
                    live_text
                )

                await self._put_event(
                    DeepgramEvent(
                        type="transcript",
                        text=live_text,
                        is_final=is_final,
                        speech_final=speech_final,
                        confidence=confidence,
                        raw=message,
                    )
                )

        if speech_final:
            await self._flush_utterance()

    async def _flush_utterance(self) -> None:
        if not self._final_parts:
            return

        text = self._join_parts(
            self._final_parts
        )

        self._final_parts.clear()
        self._last_live_transcript = ""

        if (
            not text
            or text == self._last_utterance
        ):
            return

        self._last_utterance = text

        logger.info(
            f"[{self.session_id}] "
            f"Deepgram: '{text}'"
        )

        await self._put_utterance(
            text
        )

        await self._put_event(
            DeepgramEvent(
                type="utterance",
                text=text,
                is_final=True,
                speech_final=True,
            )
        )

    async def _keepalive_loop(
        self,
        ws: WebSocketClientProtocol,
    ) -> None:
        try:
            while (
                not self._closing
                and self._ws is ws
                and not ws.closed
            ):
                await asyncio.sleep(1)

                if (
                    time.monotonic()
                    - self._last_audio_at
                    < KEEPALIVE_INTERVAL
                ):
                    continue

                await ws.send(
                    json.dumps({
                        "type": "KeepAlive"
                    })
                )

                self._last_audio_at = (
                    time.monotonic()
                )

        except asyncio.CancelledError:
            raise

        except (
            ConnectionClosed,
            OSError,
            RuntimeError,
        ):
            pass

        except Exception as exc:
            if not self._closing:
                logger.debug(
                    f"[{self.session_id}] "
                    f"KeepAlive stopped: "
                    f"{self._safe_error(exc)}"
                )

    async def _put_event(
        self,
        event: DeepgramEvent,
    ) -> None:
        self._queue_put(
            self._event_queue,
            event,
        )

    async def _put_utterance(
        self,
        text: str,
    ) -> None:
        self._queue_put(
            self._utterance_queue,
            text,
        )

    @staticmethod
    def _queue_put(
        queue: asyncio.Queue,
        item: Any,
    ) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _join_parts(
        parts: List[str],
    ) -> str:
        return " ".join(
            " ".join(parts).split()
        ).strip()

    def _advance_key(self) -> None:
        if not self.api_keys:
            return

        current = (
            self._active_key_index
            if self._active_key_index
            is not None
            else self._key_index
        )

        self._key_index = (
            current + 1
        ) % len(self.api_keys)

        self._active_key_index = None

    async def _cleanup_transport(
        self,
    ) -> None:
        current_task = (
            asyncio.current_task()
        )

        tasks = [
            task
            for task in (
                self._receive_task,
                self._keepalive_task,
            )
            if (
                task
                and task is not current_task
                and not task.done()
            )
        ]

        self._receive_task = None
        self._keepalive_task = None

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        ws = self._ws

        self._ws = None
        self._connected.clear()

        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

    async def close(self) -> None:
        if self._closing:
            return

        self._closing = True

        if self.connected:
            try:
                await self._ws.send(
                    json.dumps({
                        "type": "CloseStream"
                    })
                )
            except Exception:
                pass

        await self._cleanup_transport()

        self._final_parts.clear()
        self._last_live_transcript = ""

        logger.info(
            f"[{self.session_id}] "
            "Deepgram session closed"
        )

    @staticmethod
    def _safe_error(
        exc: Exception,
    ) -> str:
        status = getattr(
            exc,
            "status_code",
            None,
        )

        if status:
            return (
                f"{exc.__class__.__name__} "
                f"(HTTP {status})"
            )

        return (
            f"{exc.__class__.__name__}: "
            f"{exc}"
        )


class DeepgramSTT:
    def __init__(self):
        self._next_key_index = 0
        self._rotation_lock = (
            asyncio.Lock()
        )

    def _get_api_keys(self) -> List[str]:
        return [
            key.strip()
            for key
            in settings.DEEPGRAM_API_KEYS
            if key and key.strip()
        ]

    @property
    def configured(self) -> bool:
        return bool(
            self._get_api_keys()
        )

    async def create_session(
        self,
        session_id: str,
    ) -> DeepgramSTTSession:
        api_keys = (
            self._get_api_keys()
        )

        if not api_keys:
            raise DeepgramNotConfiguredError(
                "Configure DEEPGRAM_API_KEY1 "
                "through DEEPGRAM_API_KEY4."
            )

        async with self._rotation_lock:
            start_index = (
                self._next_key_index
                % len(api_keys)
            )

            self._next_key_index = (
                self._next_key_index + 1
            ) % len(api_keys)

        session = DeepgramSTTSession(
            session_id=session_id,
            api_keys=api_keys,
            start_key_index=start_index,
        )

        await session.connect()

        return session


deepgram_stt = DeepgramSTT()