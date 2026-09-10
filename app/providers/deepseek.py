from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import os
from typing import Any

from openai import OpenAI

from ..agents.agent import AgentMessage
from ..token_usage import ModelResult, ModelTokenUsage


class LlmConfigurationError(RuntimeError):
    """The provider cannot be called because local configuration is missing."""


class LlmRequestError(RuntimeError):
    """The provider rejected or failed to complete a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_message = provider_message

    @property
    def is_context_overflow(self) -> bool:
        details = f"{self.provider_code or ''} {self.provider_message or ''}".casefold()
        describes_context_limit = (
            "context length" in details
            or "context window" in details
            or "too many tokens" in details
            or all(
                marker in details
                for marker in ("maximum", "tokens", "requested")
            )
        )
        return self.status_code in {400, 413, 422} and describes_context_limit


class LlmEmptyStreamError(LlmRequestError):
    """The provider finished a stream without a visible answer."""

    def __init__(self, finish_reason: str | None = None) -> None:
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        super().__init__(f"DeepSeek вернул пустой потоковый ответ{reason}")
        self.finish_reason = finish_reason


DEFAULT_MAX_TOKENS = 2_000
DEFAULT_REASONING_EFFORT = "high"


@dataclass(frozen=True, slots=True)
class LlmStreamChunk:
    reasoning: str = ""
    content: str = ""
    finish_reason: str | None = None
    status: str = ""
    usage: ModelTokenUsage | None = None
    model: str = ""


class DeepSeekProvider:
    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        thinking_enabled: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._thinking_enabled = thinking_enabled

    def _model_name(self) -> str:
        return self._model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise LlmConfigurationError("DEEPSEEK_API_KEY не задан")

        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def _build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
    ) -> dict[str, Any]:
        return self._build_chat_request(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
            stream=stream,
        )

    def _build_chat_request(
        self,
        *,
        messages: Sequence[AgentMessage],
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model_name(),
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": stream,
            "extra_body": {"thinking": {"type": thinking_type}},
        }
        if stream:
            request["stream_options"] = {"include_usage": True}
        if thinking_type == "enabled":
            request["reasoning_effort"] = DEFAULT_REASONING_EFFORT
        if response_format is not None:
            request["response_format"] = response_format
        return request

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ModelResult:
        thinking_type = "enabled" if self._thinking_enabled else "disabled"
        request = self._build_chat_request(
            messages=messages,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
        )
        first_result = self._extract_result(self._request_completion(request))
        if first_result.content:
            return first_result

        if thinking_type == "enabled" and first_result.finish_reason != "content_filter":
            fallback = self._build_chat_request(
                messages=messages,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )
            fallback_result = self._extract_result(self._request_completion(fallback))
            if fallback_result.content:
                return ModelResult(
                    content=fallback_result.content,
                    usage=first_result.usage + fallback_result.usage,
                    finish_reason=fallback_result.finish_reason,
                    model=fallback_result.model,
                )
            first_result = fallback_result

        finish_reason = first_result.finish_reason
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LlmRequestError(f"DeepSeek вернул пустой ответ{reason}")

    def _request_completion(self, request: dict[str, Any]) -> Any:
        try:
            return self._get_client().chat.completions.create(**request)
        except LlmConfigurationError:
            raise
        except Exception as error:
            raise self._request_error(
                error,
                "Запрос к DeepSeek завершился ошибкой",
            ) from error

    @classmethod
    def _request_error(cls, error: Exception, fallback: str) -> LlmRequestError:
        body = getattr(error, "body", None)
        details = body.get("error", body) if isinstance(body, dict) else {}
        if not isinstance(details, dict):
            details = {}
        provider_message = details.get("message")
        provider_code = details.get("code") or details.get("type")
        return LlmRequestError(
            fallback,
            status_code=getattr(error, "status_code", None),
            provider_code=str(provider_code) if provider_code is not None else None,
            provider_message=(
                str(provider_message) if provider_message is not None else None
            ),
        )

    @staticmethod
    def _extract_content(response: Any) -> tuple[str, str | None]:
        if not response.choices:
            raise LlmRequestError("DeepSeek не вернул вариантов ответа")

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        return content, getattr(choice, "finish_reason", None)

    def _extract_result(self, response: Any) -> ModelResult:
        content, finish_reason = self._extract_content(response)
        usage = self._extract_usage(self._read(response, "usage"))
        return ModelResult(
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            model=str(self._read(response, "model", self._model_name())),
        )

    @classmethod
    def _extract_usage(cls, usage: Any) -> ModelTokenUsage:
        details = cls._read(usage, "completion_tokens_details")
        return ModelTokenUsage(
            prompt_tokens=int(cls._read(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(cls._read(usage, "completion_tokens", 0) or 0),
            cache_hit_tokens=int(
                cls._read(usage, "prompt_cache_hit_tokens", 0) or 0
            ),
            cache_miss_tokens=int(
                cls._read(usage, "prompt_cache_miss_tokens", 0) or 0
            ),
            reasoning_tokens=int(cls._read(details, "reasoning_tokens", 0) or 0),
        )

    @staticmethod
    def _read(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _extract_stream_chunk(cls, chunk: Any) -> LlmStreamChunk:
        raw_usage = cls._read(chunk, "usage")
        usage = cls._extract_usage(raw_usage) if raw_usage is not None else None
        model = str(cls._read(chunk, "model", "") or "")
        choices = cls._read(chunk, "choices", []) or []
        if not choices:
            return LlmStreamChunk(usage=usage, model=model)

        choice = choices[0]
        delta = cls._read(choice, "delta")
        if delta is None:
            return LlmStreamChunk(
                finish_reason=cls._read(choice, "finish_reason"),
                usage=usage,
                model=model,
            )

        reasoning = cls._read(delta, "reasoning_content", "")
        if not reasoning:
            reasoning = cls._read(delta, "reasoning", "")
        content = cls._read(delta, "content", "")
        return LlmStreamChunk(
            reasoning=reasoning if isinstance(reasoning, str) else str(reasoning or ""),
            content=content if isinstance(content, str) else str(content or ""),
            finish_reason=cls._read(choice, "finish_reason"),
            usage=usage,
            model=model,
        )

    def _stream_chat_once(
        self,
        *,
        messages: Sequence[AgentMessage],
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        request = self._build_chat_request(
            messages=messages,
            response_format=response_format,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            response = self._get_client().chat.completions.create(**request)
            saw_content = False
            finish_reason: str | None = None
            for chunk in response:
                parsed = self._extract_stream_chunk(chunk)
                saw_content = saw_content or bool(parsed.content)
                finish_reason = parsed.finish_reason or finish_reason
                if parsed.reasoning or parsed.content or parsed.finish_reason:
                    yield parsed
        except LlmConfigurationError:
            raise
        except LlmEmptyStreamError:
            raise
        except Exception as error:
            raise self._request_error(
                error,
                "Потоковый запрос к DeepSeek завершился ошибкой",
            ) from error

        if not saw_content:
            raise LlmEmptyStreamError(finish_reason)

    def _stream_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        yield from self._stream_chat_once(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
        )

    def generate_stream(
        self,
        *,
        messages: Sequence[AgentMessage],
        response_format: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        thinking_type = "enabled" if self._thinking_enabled else "disabled"
        try:
            yield from self._stream_chat_once(
                messages=messages,
                response_format=response_format,
                thinking_type=thinking_type,
                max_tokens=max_tokens,
            )
        except LlmEmptyStreamError as error:
            if thinking_type == "disabled" or error.finish_reason == "content_filter":
                raise
            yield LlmStreamChunk(
                status="Финальный ответ не пришёл в thinking-режиме; повторяем без thinking…",
            )
            yield from self._stream_chat_once(
                messages=messages,
                response_format=response_format,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )

    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        messages: list[AgentMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        yield from self.generate_stream(
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
        )

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        request = self._build_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            thinking_type="enabled",
            max_tokens=max_tokens,
        )
        response = self._request_completion(request)
        content, finish_reason = self._extract_content(response)
        if content:
            return content

        if finish_reason != "content_filter":
            fallback_request = self._build_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )
            fallback_response = self._request_completion(fallback_request)
            fallback_content, fallback_finish_reason = self._extract_content(
                fallback_response,
            )
            if fallback_content:
                return fallback_content
            finish_reason = fallback_finish_reason or finish_reason

        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LlmRequestError(f"DeepSeek вернул пустой ответ{reason}")
