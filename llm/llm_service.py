from typing import AsyncIterator

from google.genai.errors import APIError as GeminiApiError

from llm.alibaba_service import (
    AlibabaApplicationError,
    AlibabaNotConfiguredError,
    AlibabaPartialStreamError,
    AlibabaProviderUnavailableError,
    AllAlibabaModelsFailed,
    alibaba_service,
)
from llm.gemini_service import (
    AllGeminiKeysExhausted,
    gemini_service,
)
from utils.logger import get_logger


logger = get_logger(__name__)


class AllLLMProvidersFailed(Exception):
    pass


class LLMApplicationError(Exception):
    pass


class LLMPartialStreamError(Exception):
    pass


class LLMService:
    async def generate_reply_stream_from_contents(
        self,
        contents: list,
    ) -> AsyncIterator[str]:
        try:
            async for chunk in (
                alibaba_service
                .generate_reply_stream_from_contents(
                    contents
                )
            ):
                yield chunk

            return

        except AlibabaApplicationError as exc:
            raise LLMApplicationError(
                str(exc)
            ) from exc

        except AlibabaPartialStreamError as exc:
            raise LLMPartialStreamError(
                str(exc)
            ) from exc

        except AlibabaNotConfiguredError:
            logger.warning(
                "Alibaba is not configured "
                "— switching to Gemini"
            )

        except AlibabaProviderUnavailableError as exc:
            logger.warning(
                f"Alibaba provider unavailable: "
                f"{exc} — switching to Gemini"
            )

        except AllAlibabaModelsFailed as exc:
            logger.warning(
                f"{exc} — switching to Gemini"
            )

        logger.info(
            "LLM fallback: Alibaba → Gemini"
        )

        gemini_emitted = False

        try:
            async for chunk in (
                gemini_service
                .generate_reply_stream_from_contents(
                    contents
                )
            ):
                if not chunk:
                    continue

                gemini_emitted = True
                yield chunk

            if not gemini_emitted:
                raise AllLLMProvidersFailed(
                    "Gemini returned an empty response"
                )

        except AllGeminiKeysExhausted as exc:
            raise AllLLMProvidersFailed(
                "Alibaba models and all "
                "Gemini keys failed"
            ) from exc

        except GeminiApiError as exc:
            code = getattr(
                exc,
                "code",
                None,
            )

            if code in {
                400,
                422,
            }:
                raise LLMApplicationError(
                    f"Gemini rejected the "
                    f"request ({code})"
                ) from exc

            raise AllLLMProvidersFailed(
                f"Gemini provider failed "
                f"({code or 'unknown'})"
            ) from exc

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


llm_service = LLMService()