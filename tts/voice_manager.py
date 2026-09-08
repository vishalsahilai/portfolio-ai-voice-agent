import asyncio
import base64
from ctypes import alignment
import json
from threading import Lock
from typing import AsyncIterator, List, Optional
from urllib.parse import urlencode

from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError
from websockets import ConnectionClosed
from websockets.legacy.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)

ELEVENLABS_WS = "wss://api.elevenlabs.io/v1/text-to-speech"
OUTPUT_FORMAT = "mp3_44100_128"
CONNECT_TIMEOUT = 10
ROTATE_CODES = {401, 403, 429}


class AllElevenLabsKeysExhausted(Exception):
    pass


class ElevenLabsStreamError(Exception):
    pass


class ElevenLabsVoiceManager:
    def __init__(
        self,
        account_pool: Optional[List[dict]] = None,
        model_id: Optional[str] = None,
    ):
        self.account_pool = (
            account_pool
            if account_pool is not None
            else settings.ELEVENLABS_ACCOUNT_POOL
        )
        self.model_id = model_id or settings.ELEVENLABS_MODEL_ID
        self._current_index = 0
        self._lock = Lock()

        if not self.account_pool:
            logger.warning("No ElevenLabs accounts configured")

    def _get_index(self) -> int:
        with self._lock:
            return self._current_index

    def _set_index(self, index: int) -> None:
        with self._lock:
            self._current_index = index

    @staticmethod
    def _should_rotate(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) in ROTATE_CODES

    def synthesize(self, text: str) -> bytes:
        if not self.account_pool:
            raise RuntimeError("No ElevenLabs accounts configured")

        total = len(self.account_pool)
        start = self._get_index()

        for attempt in range(total):
            index = (start + attempt) % total
            account = self.account_pool[index]

            try:
                client = ElevenLabs(
                    api_key=account["api_key"]
                )

                audio = client.text_to_speech.convert(
                    voice_id=account["voice_id"],
                    model_id=self.model_id,
                    text=text,
                )

                self._set_index(index)
                return b"".join(audio)

            except ApiError as exc:
                if not self._should_rotate(exc):
                    raise

                logger.warning(
                    f"ElevenLabs account {index + 1}/{total} "
                    f"failed ({getattr(exc, 'status_code', 'unknown')})"
                )

        raise AllElevenLabsKeysExhausted(
            f"All {total} ElevenLabs accounts exhausted"
        )

    def _build_ws_url(self, voice_id: str) -> str:
        query = urlencode({
            "model_id": self.model_id,
            "output_format": OUTPUT_FORMAT,
            "sync_alignment": "true",
            "inactivity_timeout": 60,
        })

        return (
            f"{ELEVENLABS_WS}/"
            f"{voice_id}/stream-input?{query}"
        )

    async def _connect_stream(self):
        if not self.account_pool:
            raise RuntimeError("No ElevenLabs accounts configured")

        total = len(self.account_pool)
        start = self._get_index()
        last_error = None

        for attempt in range(total):
            index = (start + attempt) % total
            account = self.account_pool[index]

            try:
                ws = await asyncio.wait_for(
                    websocket_connect(
                        self._build_ws_url(
                            account["voice_id"]
                        ),
                        extra_headers={
                            "xi-api-key": account["api_key"]
                        },
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=5,
                        max_size=8 * 1024 * 1024,
                        compression=None,
                    ),
                    timeout=CONNECT_TIMEOUT,
                )

                await ws.send(
                    json.dumps({
                        "text": " ",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.8,
                            "speed": 1.0,
                            "use_speaker_boost": False,
                        },
                        "generation_config": {
                            "chunk_length_schedule": [
                                50,
                                90,
                                140,
                                200,
                            ]
                        },
                    })
                )

                self._set_index(index)

                logger.info(
                    f"ElevenLabs WebSocket connected ✅ "
                    f"(account {index + 1})"
                )

                return ws

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                last_error = exc

                logger.warning(
                    f"ElevenLabs account "
                    f"{index + 1}/{total} connection failed: "
                    f"{type(exc).__name__}"
                )

        raise AllElevenLabsKeysExhausted(
            f"All {total} ElevenLabs accounts failed"
        ) from last_error

    async def _send_text_stream(
        self,
        ws,
        text_stream: AsyncIterator[str],
    ) -> None:
        buffer = ""

        async for chunk in text_stream:
            if not chunk:
                continue

            buffer += chunk

            cut = max(
                buffer.rfind(" "),
                buffer.rfind("\n"),
            )

            if cut < 0:
                continue

            text = buffer[:cut + 1]
            buffer = buffer[cut + 1:]

            if text.strip():
                await ws.send(
                    json.dumps({"text": text})
                )

        if buffer.strip():
            await ws.send(
                json.dumps({
                    "text": buffer,
                    "flush": True,
                })
            )

        await ws.send(
            json.dumps({"text": ""})
        )

        async def stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[dict]:
            ws = await self._connect_stream()

        sender = asyncio.create_task(
            self._send_text_stream(
                ws,
                text_stream,
            )
        )

        receiver = None
        audio_started = False

        try:
            receiver = asyncio.create_task(
                ws.recv()
            )

            while True:
                active = {receiver}

                if not sender.done():
                    active.add(sender)

                done, _ = await asyncio.wait(
                    active,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if sender in done:
                    error = sender.exception()

                    if error:
                        raise error

                if receiver in done:
                    raw = receiver.result()

                    if isinstance(raw, bytes):
                        audio_started = True

                        yield {
                            "audio": raw,
                            "alignment": None,
                        }

                    else:
                        message = json.loads(raw)

                        audio_b64 = message.get(
                            "audio"
                        )

                        alignment = (
                            message.get("alignment")
                            or message.get(
                                "normalized_alignment"
                            )
                            or message.get(
                                "normalizedAlignment"
                            )
                        )

                        audio_bytes = b""

                        if (
                            isinstance(
                                audio_b64,
                                str,
                            )
                            and audio_b64
                        ):
                            audio_bytes = (
                                base64.b64decode(
                                    audio_b64
                                )
                            )

                        if audio_bytes:
                            audio_started = True

                        if (
                            audio_bytes
                            or isinstance(
                                alignment,
                                dict,
                            )
                        ):
                            yield {
                                "audio": audio_bytes,
                                "alignment": (
                                    alignment
                                    if isinstance(
                                        alignment,
                                        dict,
                                    )
                                    else None
                                ),
                            }

                        if (
                            message.get(
                                "is_final"
                            )
                            or message.get(
                                "isFinal"
                            )
                        ):
                            break

                    receiver = (
                        asyncio.create_task(
                            ws.recv()
                        )
                    )

        except asyncio.CancelledError:
            raise

        except ConnectionClosedOK:
            logger.info(
                "ElevenLabs WebSocket finished normally ✅"
            )
            return

        except ConnectionClosed as exc:
            if (
                getattr(
                    exc,
                    "code",
                    None,
                )
                == 1000
            ):
                logger.info(
                    "ElevenLabs WebSocket finished normally ✅"
                )
                return

            raise ElevenLabsStreamError(
                f"ElevenLabs stream stopped: {exc}"
            ) from exc

        except Exception as exc:
            raise ElevenLabsStreamError(
                f"ElevenLabs stream failed: {exc}"
            ) from exc

        finally:
            for task in (
                sender,
                receiver,
            ):
                if (
                    task
                    and not task.done()
                ):
                    task.cancel()

            await asyncio.gather(
                *[
                    task
                    for task in (
                        sender,
                        receiver,
                    )
                    if task
                ],
                return_exceptions=True,
            )

            try:
                await ws.close()

            except Exception:
                pass

voice_manager = ElevenLabsVoiceManager()