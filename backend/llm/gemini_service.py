from threading import Lock
from typing import AsyncIterator, List, Optional

import httpx

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import settings
from llm.prompt_builder import SYSTEM_INSTRUCTION
from utils.logger import get_logger


logger = get_logger(__name__)

GEMINI_TIMEOUT_MS = 15000  # 15 seconds
ROTATE_CODES = {401, 403, 429, 500, 502, 503, 504}


class AllGeminiKeysExhausted(Exception):
    pass


class GeminiService:
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: Optional[str] = None,
    ):
        self.api_keys = api_keys or settings.GEMINI_API_KEYS
        self.model = model or settings.GEMINI_MODEL

        self._lock = Lock()
        self._current_index = 0

        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        )

        self._clients = [
            genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=GEMINI_TIMEOUT_MS
                ),
            )
            for key in self.api_keys
        ]

        if not self._clients:
            logger.warning("No Gemini API keys configured")

    def _get_current_index(self) -> int:
        with self._lock:
            return self._current_index

    def _set_current_index(self, index: int) -> None:
        with self._lock:
            self._current_index = index

    @staticmethod
    def _get_error_code(exc: Exception):
        return (
            getattr(exc, "code", None)
            or getattr(exc, "status_code", None)
        )

    @classmethod
    def _should_rotate(cls, exc: Exception) -> bool:
        return cls._get_error_code(exc) in ROTATE_CODES

    @staticmethod
    def _extract_text(response) -> str:
        text = getattr(response, "text", None)

        if text:
            return text

        content = getattr(response, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        return ""

    def generate_reply_from_contents(
        self,
        contents: list,
    ) -> str:
        if not self._clients:
            raise RuntimeError(
                "Gemini API keys are not configured"
            )

        total = len(self._clients)
        start = self._get_current_index()

        for attempt in range(total):
            index = (start + attempt) % total
            client = self._clients[index]

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._config,
                )

                text = self._extract_text(
                    response
                ).strip()

                self._set_current_index(index)

                if text:
                    logger.info(
                        f"Gemini key {index + 1}: "
                        f"'{text[:120]}"
                        f"{'...' if len(text) > 120 else ''}'"
                    )

                return text

            except APIError as exc:
                code = self._get_error_code(exc)

                if not self._should_rotate(exc):
                    raise

                logger.warning(
                    f"Gemini key {index + 1}/{total} "
                    f"failed ({code}) — rotating"
                )

            except httpx.TimeoutException:
                logger.warning(
                    f"Gemini key {index + 1}/{total} "
                    f"timed out — rotating"
                )

            except Exception as exc:
                logger.error(
                    f"Gemini failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return ""

        raise AllGeminiKeysExhausted(
            f"All {total} Gemini API keys exhausted"
        )

    async def generate_reply_stream_from_contents(
        self,
        contents: list,
    ) -> AsyncIterator[str]:
        if not self._clients:
            raise RuntimeError(
                "Gemini API keys are not configured"
            )

        total = len(self._clients)
        start = self._get_current_index()

        for attempt in range(total):
            index = (start + attempt) % total
            client = self._clients[index]

            emitted = False
            full_response = []

            try:
                stream = await (
                    client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=self._config,
                    )
                )

                async for chunk in stream:
                    text = self._extract_text(chunk)

                    if not text:
                        continue

                    emitted = True
                    full_response.append(text)

                    yield text

                self._set_current_index(index)

                final_text = "".join(
                    full_response
                ).strip()

                if final_text:
                    logger.info(
                        f"Gemini stream key {index + 1}: "
                        f"'{final_text[:120]}"
                        f"{'...' if len(final_text) > 120 else ''}'"
                    )

                return

            except APIError as exc:
                code = self._get_error_code(exc)

                if emitted:
                    logger.error(
                        f"Gemini stream stopped after "
                        f"partial response: {exc}"
                    )
                    return

                if not self._should_rotate(exc):
                    raise

                logger.warning(
                    f"Gemini key {index + 1}/{total} "
                    f"failed ({code}) — rotating"
                )

            except httpx.TimeoutException:
                if emitted:
                    logger.error(
                        f"Gemini stream key {index + 1} "
                        f"timed out after partial response"
                    )
                    return

                logger.warning(
                    f"Gemini key {index + 1}/{total} "
                    f"timed out — rotating"
                )

            except Exception as exc:
                logger.error(
                    f"Gemini stream failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

        raise AllGeminiKeysExhausted(
            f"All {total} Gemini API keys exhausted"
        )

    async def generate_reply_from_contents_async(
        self,
        contents: list,
    ) -> str:
        chunks = []

        async for chunk in (
            self.generate_reply_stream_from_contents(
                contents
            )
        ):
            chunks.append(chunk)

        return "".join(chunks).strip()

    def generate_reply(
        self,
        conversation_history: List[dict],
        user_text: str,
    ) -> str:
        contents = [
            types.Content(
                role=turn["role"],
                parts=[
                    types.Part(
                        text=turn["text"]
                    )
                ],
            )
            for turn in conversation_history
        ]

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=user_text
                    )
                ],
            )
        )

        return self.generate_reply_from_contents(
            contents
        )


gemini_service = GeminiService()