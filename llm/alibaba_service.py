import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from config.settings import settings
from llm.prompt_builder import SYSTEM_INSTRUCTION
from utils.logger import get_logger


logger = get_logger(__name__)

FREE_QUOTA_CODE = "AllocationQuota.FreeTierOnly"


class AlibabaServiceError(Exception):
    pass


class AlibabaNotConfiguredError(AlibabaServiceError):
    pass


class AlibabaProviderUnavailableError(AlibabaServiceError):
    pass


class AlibabaApplicationError(AlibabaServiceError):
    pass


class AlibabaPartialStreamError(AlibabaServiceError):
    pass


class AllAlibabaModelsFailed(AlibabaServiceError):
    pass


@dataclass
class ModelState:
    status: str = "available"
    cooldown_until: float = 0.0
    reason: str = ""


class AlibabaService:
    def __init__(self):
        self.api_key = settings.ALIBABA_API_KEY.strip()
        self.base_url = settings.ALIBABA_BASE_URL.rstrip("/")
        self.models = list(settings.ALIBABA_MODEL_LIST)

        self.timeout_seconds = float(
            settings.ALIBABA_TIMEOUT_SECONDS
        )

        self.transient_cooldown = float(
            settings.ALIBABA_TRANSIENT_COOLDOWN_SECONDS
        )

        self.rate_limit_cooldown = float(
            settings.ALIBABA_RATE_LIMIT_COOLDOWN_SECONDS
        )

        self.access_cooldown = float(
            settings.ALIBABA_ACCESS_COOLDOWN_SECONDS
        )

        self._states: Dict[str, ModelState] = {
            model: ModelState()
            for model in self.models
        }

        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning(
                "Alibaba API key is not configured"
            )

        if not self.models:
            logger.warning(
                "No Alibaba models configured"
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=min(
                    3.0,
                    self.timeout_seconds,
                ),
            )

            self._client = httpx.AsyncClient(
                timeout=timeout,
            )

        return self._client

    @property
    def endpoint(self) -> str:
        return (
            f"{self.base_url}/chat/completions"
        )

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_part_text(part) -> str:
        if isinstance(part, str):
            return part

        if isinstance(part, dict):
            value = part.get("text")
            return (
                value
                if isinstance(value, str)
                else ""
            )

        value = getattr(
            part,
            "text",
            None,
        )

        return (
            value
            if isinstance(value, str)
            else ""
        )

    def _contents_to_messages(
        self,
        contents: list,
    ) -> List[dict]:
        messages: List[dict] = []

        if SYSTEM_INSTRUCTION:
            messages.append({
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            })

        for content in contents:
            if isinstance(content, dict):
                role = content.get(
                    "role",
                    "user",
                )

                parts = content.get(
                    "parts",
                    [],
                )

                direct_text = content.get(
                    "text"
                )

            else:
                role = getattr(
                    content,
                    "role",
                    "user",
                )

                parts = getattr(
                    content,
                    "parts",
                    [],
                )

                direct_text = getattr(
                    content,
                    "text",
                    None,
                )

            if role == "model":
                role = "assistant"

            if role not in {
                "system",
                "user",
                "assistant",
            }:
                role = "user"

            text_parts = []

            if isinstance(
                direct_text,
                str,
            ):
                text_parts.append(
                    direct_text
                )

            if parts:
                for part in parts:
                    text = (
                        self._extract_part_text(
                            part
                        )
                    )

                    if text:
                        text_parts.append(
                            text
                        )

            text = "".join(
                text_parts
            ).strip()

            if not text:
                continue

            messages.append({
                "role": role,
                "content": text,
            })

        return messages

    @staticmethod
    def _extract_delta_text(
        value,
    ) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts = []

            for item in value:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                text = item.get("text")

                if isinstance(
                    text,
                    str,
                ):
                    parts.append(text)

            return "".join(parts)

        return ""

    @staticmethod
    def _error_details(
        body: str,
    ) -> Tuple[str, str]:
        try:
            payload = json.loads(body)
        except Exception:
            return "", body[:200]

        error = payload.get(
            "error",
            payload,
        )

        if isinstance(error, dict):
            code = str(
                error.get("code")
                or error.get("type")
                or ""
            )

            message = str(
                error.get("message")
                or ""
            )

            return code, message

        return "", str(error)[:200]

    @staticmethod
    def _is_quota_exhausted(
        code: str,
        message: str,
    ) -> bool:
        combined = (
            f"{code} {message}"
        ).lower()

        return (
            FREE_QUOTA_CODE.lower()
            in combined
            or "free tier only"
            in combined
            or "free-tier only"
            in combined
            or "free quota"
            in combined
            and "exhaust"
            in combined
        )

    @staticmethod
    def _looks_model_unavailable(
        code: str,
        message: str,
    ) -> bool:
        combined = (
            f"{code} {message}"
        ).lower()

        phrases = (
            "model not found",
            "model_not_found",
            "model unavailable",
            "model is unavailable",
            "model does not exist",
            "model not available",
            "model access",
            "not activated",
        )

        return any(
            phrase in combined
            for phrase in phrases
        )

    def _mark_quota_exhausted(
        self,
        model: str,
    ) -> None:
        state = self._states[model]

        state.status = (
            "quota_exhausted"
        )

        state.cooldown_until = 0.0
        state.reason = (
            FREE_QUOTA_CODE
        )

        logger.warning(
            f"Alibaba model {model}: "
            "free quota exhausted — "
            "disabled for this process"
        )

    def _mark_cooldown(
        self,
        model: str,
        status: str,
        seconds: float,
        reason: str,
    ) -> None:
        state = self._states[model]

        state.status = status
        state.cooldown_until = (
            time.monotonic()
            + seconds
        )
        state.reason = reason

        logger.warning(
            f"Alibaba model {model}: "
            f"{status} — cooldown "
            f"{seconds:.0f}s"
        )

    def _should_skip(
        self,
        model: str,
    ) -> Optional[str]:
        state = self._states[model]

        if (
            state.status
            == "quota_exhausted"
        ):
            return (
                "quota exhausted"
            )

        if (
            state.cooldown_until
            > time.monotonic()
        ):
            remaining = (
                state.cooldown_until
                - time.monotonic()
            )

            return (
                f"{state.status}, "
                f"{remaining:.0f}s cooldown"
            )

        if state.status != "available":
            state.status = "available"
            state.cooldown_until = 0.0
            state.reason = ""

        return None

    def _handle_http_error(
        self,
        model: str,
        status_code: int,
        body: str,
    ) -> None:
        code, message = (
            self._error_details(body)
        )

        if self._is_quota_exhausted(
            code,
            message,
        ):
            self._mark_quota_exhausted(
                model
            )
            return

        if status_code == 401:
            raise (
                AlibabaProviderUnavailableError(
                    "Alibaba authentication "
                    "or region configuration failed"
                )
            )

        if status_code in {
            400,
            422,
        }:
            if self._looks_model_unavailable(
                code,
                message,
            ):
                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.access_cooldown
                    ),
                    reason=(
                        f"{status_code}:{code}"
                    ),
                )
                return

            raise AlibabaApplicationError(
                f"Alibaba rejected the "
                f"request ({status_code}, "
                f"{code or 'invalid_request'})"
            )

        if status_code in {
            403,
            404,
        }:
            self._mark_cooldown(
                model=model,
                status=(
                    "temporarily_unavailable"
                ),
                seconds=(
                    self.access_cooldown
                ),
                reason=(
                    f"{status_code}:{code}"
                ),
            )
            return

        if status_code == 429:
            self._mark_cooldown(
                model=model,
                status="rate_limited",
                seconds=(
                    self.rate_limit_cooldown
                ),
                reason=(
                    f"429:{code}"
                ),
            )
            return

        if (
            status_code == 408
            or status_code >= 500
        ):
            self._mark_cooldown(
                model=model,
                status=(
                    "temporarily_unavailable"
                ),
                seconds=(
                    self.transient_cooldown
                ),
                reason=(
                    f"{status_code}:{code}"
                ),
            )
            return

        raise AlibabaApplicationError(
            f"Unexpected Alibaba response "
            f"status {status_code}"
        )

    async def generate_reply_stream_from_contents(
        self,
        contents: list,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise AlibabaNotConfiguredError(
                "Alibaba API key is not configured"
            )

        if not self.models:
            raise AlibabaNotConfiguredError(
                "No Alibaba models configured"
            )

        messages = (
            self._contents_to_messages(
                contents
            )
        )

        if not messages:
            raise AlibabaApplicationError(
                "LLM message list is empty"
            )

        client = self._get_client()
        last_error = None

        for model in self.models:
            skip_reason = (
                self._should_skip(model)
            )

            if skip_reason:
                logger.info(
                    f"Alibaba model {model} "
                    f"skipped: {skip_reason}"
                )
                continue

            logger.info(
                f"LLM provider=Alibaba "
                f"model={model} attempting"
            )

            emitted = False
            full_response = []

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
            }

            try:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if (
                        response.status_code
                        >= 400
                    ):
                        body = (
                            await response.aread()
                        ).decode(
                            "utf-8",
                            errors="replace",
                        )

                        self._handle_http_error(
                            model=model,
                            status_code=(
                                response.status_code
                            ),
                            body=body,
                        )

                        last_error = (
                            f"HTTP "
                            f"{response.status_code}"
                        )

                        continue

                    async for line in (
                        response.aiter_lines()
                    ):
                        line = line.strip()

                        if not line:
                            continue

                        if not line.startswith(
                            "data:"
                        ):
                            continue

                        data = (
                            line[5:].strip()
                        )

                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(
                                data
                            )
                        except json.JSONDecodeError:
                            continue

                        choices = event.get(
                            "choices",
                            [],
                        )

                        if not choices:
                            continue

                        delta = (
                            choices[0].get(
                                "delta",
                                {},
                            )
                        )

                        text = (
                            self._extract_delta_text(
                                delta.get(
                                    "content"
                                )
                            )
                        )

                        if not text:
                            continue

                        emitted = True
                        full_response.append(
                            text
                        )

                        yield text

                if emitted:
                    self._states[
                        model
                    ] = ModelState()

                    final_text = "".join(
                        full_response
                    ).strip()

                    logger.info(
                        f"LLM provider=Alibaba "
                        f"model={model} success "
                        f"chars={len(final_text)}"
                    )

                    return

                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason="empty_response",
                )

                last_error = (
                    "empty response"
                )

            except (
                AlibabaApplicationError,
                AlibabaProviderUnavailableError,
            ):
                raise

            except httpx.TimeoutException as exc:
                if emitted:
                    raise (
                        AlibabaPartialStreamError(
                            f"Alibaba model "
                            f"{model} timed out "
                            "after partial output"
                        )
                    ) from exc

                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason="timeout",
                )

                last_error = "timeout"

            except httpx.RequestError as exc:
                if emitted:
                    raise (
                        AlibabaPartialStreamError(
                            f"Alibaba model "
                            f"{model} connection "
                            "failed after partial "
                            "output"
                        )
                    ) from exc

                self._mark_cooldown(
                    model=model,
                    status=(
                        "temporarily_unavailable"
                    ),
                    seconds=(
                        self.transient_cooldown
                    ),
                    reason=(
                        type(exc).__name__
                    ),
                )

                last_error = (
                    type(exc).__name__
                )

        raise AllAlibabaModelsFailed(
            "All configured Alibaba "
            f"models failed"
            + (
                f": {last_error}"
                if last_error
                else ""
            )
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


alibaba_service = AlibabaService()