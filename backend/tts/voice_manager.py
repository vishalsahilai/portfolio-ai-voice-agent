import asyncio
import base64
import json
from threading import Lock
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets.legacy.client import connect as websocket_connect

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)

ELEVENLABS_WS = "wss://api.elevenlabs.io/v1/text-to-speech"
OUTPUT_FORMAT = "mp3_44100_128"
CONNECT_TIMEOUT = 10

ROTATE_CODES = {
    401,
    403,
    429,
}


class AllElevenLabsKeysExhausted(Exception):
    pass


class ElevenLabsStreamError(Exception):
    pass


class _QuotaRotationRequired(Exception):
    def __init__(
        self,
        message: str,
        audio_started: bool = False,
    ):
        super().__init__(message)
        self.audio_started = audio_started


class _ReplayableTextBuffer:
    """
    Reads the LLM stream once and keeps every generated text chunk.

    If ElevenLabs account 1 fails before audio playback starts,
    account 2 can replay the exact same LLM response without
    asking the LLM to generate it again.
    """

    def __init__(
        self,
        source: AsyncIterator[str],
    ):
        self._source = source
        self._chunks: list[str] = []
        self._done = False
        self._error: Optional[BaseException] = None
        self._condition = asyncio.Condition()

        self._producer_task = asyncio.create_task(
            self._produce()
        )

    async def _produce(self) -> None:
        try:
            async for chunk in self._source:
                if not chunk:
                    continue

                async with self._condition:
                    self._chunks.append(chunk)
                    self._condition.notify_all()

        except asyncio.CancelledError:
            raise

        except BaseException as exc:
            async with self._condition:
                self._error = exc
                self._condition.notify_all()

        finally:
            async with self._condition:
                self._done = True
                self._condition.notify_all()

    async def replay(self) -> AsyncIterator[str]:
        index = 0

        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: (
                        index < len(self._chunks)
                        or self._done
                    )
                )

                if index < len(self._chunks):
                    chunk = self._chunks[index]
                    index += 1

                elif self._error is not None:
                    raise self._error

                else:
                    return

            yield chunk

    async def close(self) -> None:
        if not self._producer_task.done():
            self._producer_task.cancel()

        await asyncio.gather(
            self._producer_task,
            return_exceptions=True,
        )


class ElevenLabsVoiceManager:
    def __init__(
        self,
        account_pool: Optional[list[dict]] = None,
        model_id: Optional[str] = None,
    ):
        raw_pool = (
            account_pool
            if account_pool is not None
            else settings.ELEVENLABS_ACCOUNT_POOL
        )

        # Automatically skip empty ElevenLabs keys.
        #
        # Example:
        # key7 exists
        # key8 empty
        #
        # Rotation becomes:
        # 1 -> 2 -> ... -> 7 -> 1
        self.account_pool = []

        for account in raw_pool or []:
            api_key = (
                account.get("api_key") or ""
            ).strip()

            voice_id = (
                account.get("voice_id") or ""
            ).strip()

            if not api_key:
                continue

            if not voice_id:
                logger.warning(
                    "Skipping ElevenLabs account because "
                    "voice_id is empty"
                )
                continue

            clean_account = dict(account)
            clean_account["api_key"] = api_key
            clean_account["voice_id"] = voice_id

            self.account_pool.append(
                clean_account
            )

        self.model_id = (
            model_id
            or settings.ELEVENLABS_MODEL_ID
        )

        self._current_index = 0
        self._lock = Lock()

        # Remember which account owns a preconnected
        # WebSocket.
        self._ws_account_indexes: dict[int, int] = {}

        if not self.account_pool:
            logger.warning(
                "No valid ElevenLabs accounts configured"
            )
        else:
            logger.info(
                f"Loaded {len(self.account_pool)} "
                f"valid ElevenLabs account(s)"
            )

    def _get_index(self) -> int:
        with self._lock:
            return self._current_index

    def _set_index(
        self,
        index: int,
    ) -> None:
        if not self.account_pool:
            return

        with self._lock:
            self._current_index = (
                index % len(self.account_pool)
            )

    def _next_index(
        self,
        index: int,
    ) -> int:
        if not self.account_pool:
            return 0

        return (
            index + 1
        ) % len(self.account_pool)

    def _register_ws(
        self,
        ws,
        account_index: int,
    ) -> None:
        with self._lock:
            self._ws_account_indexes[
                id(ws)
            ] = account_index

    def _get_ws_index(
        self,
        ws,
    ) -> int:
        with self._lock:
            return self._ws_account_indexes.get(
                id(ws),
                self._current_index,
            )

    def _unregister_ws(
        self,
        ws,
    ) -> None:
        with self._lock:
            self._ws_account_indexes.pop(
                id(ws),
                None,
            )

    @staticmethod
    def _is_quota_error(
        exc: Exception,
    ) -> bool:
        message = str(exc).lower()

        reason = str(
            getattr(
                exc,
                "reason",
                "",
            )
        ).lower()

        combined = (
            message + " " + reason
        )

        return (
            "exceeds your quota" in combined
            or "credits remaining" in combined
            or "quota exceeded" in combined
            or "quota_exceeded" in combined
            or (
                "1008" in combined
                and "quota" in combined
            )
        )

    @classmethod
    def _should_rotate(
        cls,
        exc: Exception,
    ) -> bool:
        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code in ROTATE_CODES:
            return True

        return cls._is_quota_error(
            exc
        )

    def synthesize(
        self,
        text: str,
    ) -> bytes:
        if not self.account_pool:
            raise RuntimeError(
                "No ElevenLabs accounts configured"
            )

        total = len(
            self.account_pool
        )

        start = self._get_index()

        last_error = None

        for attempt in range(total):
            index = (
                start + attempt
            ) % total

            account = self.account_pool[
                index
            ]

            try:
                client = ElevenLabs(
                    api_key=account[
                        "api_key"
                    ]
                )

                audio = (
                    client.text_to_speech.convert(
                        voice_id=account[
                            "voice_id"
                        ],
                        model_id=self.model_id,
                        text=text,
                    )
                )

                self._set_index(
                    index
                )

                return b"".join(
                    audio
                )

            except ApiError as exc:
                last_error = exc

                if not self._should_rotate(
                    exc
                ):
                    raise

                logger.warning(
                    f"ElevenLabs account "
                    f"{index + 1}/{total} "
                    f"unavailable — trying next account"
                )

                self._set_index(
                    self._next_index(
                        index
                    )
                )

        raise AllElevenLabsKeysExhausted(
            f"All {total} ElevenLabs "
            f"accounts are currently unavailable"
        ) from last_error

    def _build_ws_url(
        self,
        voice_id: str,
    ) -> str:
        query = urlencode(
            {
                "model_id": self.model_id,
                "output_format": OUTPUT_FORMAT,
                "sync_alignment": "true",
                "inactivity_timeout": 60,
            }
        )

        return (
            f"{ELEVENLABS_WS}/"
            f"{voice_id}/stream-input?"
            f"{query}"
        )

    async def _close_ws(
        self,
        ws,
    ) -> None:
        if ws is None:
            return

        self._unregister_ws(
            ws
        )

        try:
            await ws.close()
        except Exception:
            pass

    async def _connect_stream(
        self,
        excluded_indexes: Optional[
            set[int]
        ] = None,
    ):
        if not self.account_pool:
            raise RuntimeError(
                "No ElevenLabs accounts configured"
            )

        excluded_indexes = (
            excluded_indexes
            or set()
        )

        total = len(
            self.account_pool
        )

        start = self._get_index()

        last_error = None

        checked = 0

        for attempt in range(total):
            index = (
                start + attempt
            ) % total

            if index in excluded_indexes:
                continue

            checked += 1

            account = self.account_pool[
                index
            ]

            try:
                ws = await asyncio.wait_for(
                    websocket_connect(
                        self._build_ws_url(
                            account[
                                "voice_id"
                            ]
                        ),
                        extra_headers={
                            "xi-api-key": account[
                                "api_key"
                            ]
                        },
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=5,
                        max_size=(
                            8
                            * 1024
                            * 1024
                        ),
                        compression=None,
                    ),
                    timeout=CONNECT_TIMEOUT,
                )

                await ws.send(
                    json.dumps(
                        {
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
                        }
                    )
                )

                self._set_index(
                    index
                )

                self._register_ws(
                    ws,
                    index,
                )

                logger.info(
                    f"ElevenLabs WebSocket "
                    f"connected ✅ "
                    f"(account {index + 1})"
                )

                return ws

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                last_error = exc

                logger.warning(
                    f"ElevenLabs account "
                    f"{index + 1}/{total} "
                    f"connection failed: "
                    f"{type(exc).__name__}"
                )

                self._set_index(
                    self._next_index(
                        index
                    )
                )

        if checked == 0:
            raise AllElevenLabsKeysExhausted(
                "All ElevenLabs accounts "
                "were already attempted"
            )

        raise AllElevenLabsKeysExhausted(
            f"All {total} ElevenLabs "
            f"accounts failed"
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

            text = buffer[
                :cut + 1
            ]

            buffer = buffer[
                cut + 1:
            ]

            if text.strip():
                await ws.send(
                    json.dumps(
                        {
                            "text": text
                        }
                    )
                )

        if buffer.strip():
            await ws.send(
                json.dumps(
                    {
                        "text": (
                            buffer.rstrip()
                            + " "
                        ),
                        "flush": True,
                    }
                )
            )

        else:
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "flush": True,
                    }
                )
            )

        await ws.send(
            json.dumps(
                {
                    "text": ""
                }
            )
        )

    async def connect_stream(self):
        """
        Preconnect ElevenLabs while RAG/memory
        work is happening.

        Existing websocket_routes.py can continue
        using this exactly as before.
        """

        return await self._connect_stream()

    async def _stream_once(
        self,
        text_stream: AsyncIterator[str],
        ws,
    ) -> AsyncIterator[dict]:
        sender = asyncio.create_task(
            self._send_text_stream(
                ws,
                text_stream,
            )
        )

        receiver = None

        # We delay the very first audio packets
        # until the complete text has been accepted
        # by the active ElevenLabs connection.
        #
        # This makes quota failover safe without
        # playing half of the answer and then
        # repeating it from another account.
        pending_packets: list[dict] = []

        playback_started = False
        sender_verified = False

        try:
            receiver = asyncio.create_task(
                ws.recv()
            )

            while True:
                active = {
                    receiver
                }

                if not sender.done():
                    active.add(
                        sender
                    )

                done, _ = (
                    await asyncio.wait(
                        active,
                        return_when=(
                            asyncio.FIRST_COMPLETED
                        ),
                    )
                )

                receiver_processed = False
                final_message = False

                if receiver in done:
                    try:
                        raw = receiver.result()

                    except ConnectionClosedOK:
                        if sender.done():
                            error = (
                                sender.exception()
                            )

                            if error:
                                if self._is_quota_error(
                                    error
                                ):
                                    raise _QuotaRotationRequired(
                                        str(error),
                                        playback_started,
                                    )

                                raise error

                        if not playback_started:
                            for packet in pending_packets:
                                yield packet

                            pending_packets.clear()

                        logger.info(
                            "ElevenLabs WebSocket "
                            "finished normally ✅"
                        )

                        return

                    except ConnectionClosed as exc:
                        if getattr(
                            exc,
                            "code",
                            None,
                        ) == 1000:
                            if not playback_started:
                                for packet in pending_packets:
                                    yield packet

                                pending_packets.clear()

                            logger.info(
                                "ElevenLabs WebSocket "
                                "finished normally ✅"
                            )

                            return

                        if self._is_quota_error(
                            exc
                        ):
                            raise _QuotaRotationRequired(
                                str(exc),
                                playback_started,
                            ) from exc

                        raise

                    receiver_processed = True

                    if isinstance(
                        raw,
                        bytes,
                    ):
                        packet = {
                            "audio": raw,
                            "alignment": None,
                        }

                        if sender_verified:
                            playback_started = True
                            yield packet
                        else:
                            pending_packets.append(
                                packet
                            )

                    else:
                        message = json.loads(
                            raw
                        )

                        audio_b64 = (
                            message.get(
                                "audio"
                            )
                        )

                        alignment = (
                            message.get(
                                "alignment"
                            )
                            or message.get(
                                "normalizedAlignment"
                            )
                            or message.get(
                                "normalized_alignment"
                            )
                        )

                        audio_bytes = (
                            base64.b64decode(
                                audio_b64
                            )
                            if (
                                isinstance(
                                    audio_b64,
                                    str,
                                )
                                and audio_b64
                            )
                            else b""
                        )

                        if (
                            audio_bytes
                            or isinstance(
                                alignment,
                                dict,
                            )
                        ):
                            packet = {
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

                            if sender_verified:
                                if audio_bytes:
                                    playback_started = True

                                yield packet

                            else:
                                pending_packets.append(
                                    packet
                                )

                        if (
                            message.get(
                                "is_final"
                            )
                            or message.get(
                                "isFinal"
                            )
                        ):
                            final_message = True

                if sender.done():
                    try:
                        sender_error = (
                            sender.exception()
                        )

                    except asyncio.CancelledError:
                        raise

                    if sender_error:
                        if self._is_quota_error(
                            sender_error
                        ):
                            raise _QuotaRotationRequired(
                                str(sender_error),
                                playback_started,
                            ) from sender_error

                        raise sender_error

                    # Wait for at least one successful
                    # server response after text sending
                    # has completed before releasing
                    # buffered startup audio.
                    if (
                        receiver_processed
                        and not sender_verified
                    ):
                        sender_verified = True

                        for packet in pending_packets:
                            if packet.get(
                                "audio"
                            ):
                                playback_started = True

                            yield packet

                        pending_packets.clear()

                if final_message:
                    if not sender_verified:
                        if sender.done():
                            sender_error = (
                                sender.exception()
                            )

                            if sender_error:
                                if self._is_quota_error(
                                    sender_error
                                ):
                                    raise _QuotaRotationRequired(
                                        str(
                                            sender_error
                                        ),
                                        playback_started,
                                    )

                                raise sender_error

                        sender_verified = True

                        for packet in pending_packets:
                            if packet.get(
                                "audio"
                            ):
                                playback_started = True

                            yield packet

                        pending_packets.clear()

                    logger.info(
                        "ElevenLabs stream "
                        "finished ✅"
                    )

                    break

                if receiver in done:
                    receiver = (
                        asyncio.create_task(
                            ws.recv()
                        )
                    )

        except asyncio.CancelledError:
            raise

        except _QuotaRotationRequired:
            raise

        except ConnectionClosed as exc:
            if self._is_quota_error(
                exc
            ):
                raise _QuotaRotationRequired(
                    str(exc),
                    playback_started,
                ) from exc

            raise ElevenLabsStreamError(
                f"ElevenLabs stream "
                f"stopped: {exc}"
            ) from exc

        except Exception as exc:
            if self._is_quota_error(
                exc
            ):
                raise _QuotaRotationRequired(
                    str(exc),
                    playback_started,
                ) from exc

            raise ElevenLabsStreamError(
                f"ElevenLabs stream "
                f"failed: {exc}"
            ) from exc

        finally:
            tasks = [
                task
                for task in (
                    sender,
                    receiver,
                )
                if task
            ]

            for task in tasks:
                if not task.done():
                    task.cancel()

            if tasks:
                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

    async def stream(
        self,
        text_stream: AsyncIterator[str],
        ws=None,
    ) -> AsyncIterator[dict]:
        """
        Stream ElevenLabs audio with automatic
        circular account rotation.

        Example with 7 valid accounts:

        1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 1

        Empty configured keys are ignored.

        During one response, each valid account is
        attempted at most once. This prevents an
        infinite loop when every account is out
        of quota.
        """

        if not self.account_pool:
            raise RuntimeError(
                "No ElevenLabs accounts configured"
            )

        total = len(
            self.account_pool
        )

        replay_buffer = (
            _ReplayableTextBuffer(
                text_stream
            )
        )

        attempted_indexes: set[int] = set()

        current_ws = ws

        try:
            while (
                len(attempted_indexes)
                < total
            ):
                if current_ws is None:
                    current_ws = (
                        await self._connect_stream(
                            excluded_indexes=(
                                attempted_indexes
                            )
                        )
                    )

                current_index = (
                    self._get_ws_index(
                        current_ws
                    )
                )

                if (
                    current_index
                    in attempted_indexes
                ):
                    await self._close_ws(
                        current_ws
                    )

                    current_ws = None

                    self._set_index(
                        self._next_index(
                            current_index
                        )
                    )

                    continue

                attempted_indexes.add(
                    current_index
                )

                try:
                    async for packet in (
                        self._stream_once(
                            replay_buffer.replay(),
                            current_ws,
                        )
                    ):
                        yield packet

                    # Successful response.
                    # Keep using this account until it
                    # later becomes unavailable.
                    self._set_index(
                        current_index
                    )

                    return

                except _QuotaRotationRequired as exc:
                    logger.warning(
                        f"ElevenLabs account "
                        f"{current_index + 1}/{total} "
                        f"quota exhausted — "
                        f"switching to next account"
                    )

                    await self._close_ws(
                        current_ws
                    )

                    current_ws = None

                    self._set_index(
                        self._next_index(
                            current_index
                        )
                    )

                    if exc.audio_started:
                        # Normally quota rejection occurs
                        # before playback because startup
                        # packets are held until text is
                        # accepted.
                        #
                        # If ElevenLabs somehow rejects
                        # after audio has already been
                        # delivered, replaying the entire
                        # response would cause duplicated
                        # speech, so we stop this turn.
                        raise ElevenLabsStreamError(
                            "ElevenLabs quota became "
                            "unavailable after audio "
                            "playback had already started."
                        ) from exc

                    # No audio reached the visitor yet.
                    # Replay the same cached LLM text
                    # through the next ElevenLabs account.
                    continue

                except asyncio.CancelledError:
                    raise

                except Exception:
                    await self._close_ws(
                        current_ws
                    )

                    current_ws = None

                    raise

            raise AllElevenLabsKeysExhausted(
                f"All {total} configured "
                f"ElevenLabs accounts are "
                f"currently unavailable"
            )

        finally:
            if current_ws is not None:
                await self._close_ws(
                    current_ws
                )

            await replay_buffer.close()


voice_manager = ElevenLabsVoiceManager()