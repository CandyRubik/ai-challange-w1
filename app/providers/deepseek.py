from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Any

from openai import OpenAI

from ..agents.agent import AgentMessage
from ..token_usage import ModelResult, ModelTokenUsage


class LlmConfigurationError(RuntimeError):
    pass


class LlmRequestError(RuntimeError):
    pass


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

    def _model_name(self) -> str:
        return self._model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    @staticmethod
    def _read(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int,
    ) -> ModelResult:
        request: dict[str, Any] = {
            "model": self._model_name(),
            "messages": list(messages),
            "max_tokens": max_tokens,
            "extra_body": {
                "thinking": {
                    "type": "enabled" if self._thinking_enabled else "disabled"
                }
            },
        }
        if self._thinking_enabled:
            request["reasoning_effort"] = "high"
        try:
            response = self._get_client().chat.completions.create(**request)
        except LlmConfigurationError:
            raise
        except Exception as error:
            raise LlmRequestError("Запрос к DeepSeek завершился ошибкой") from error

        if not response.choices:
            raise LlmRequestError("DeepSeek не вернул вариантов ответа")
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            raise LlmRequestError("DeepSeek вернул пустой ответ")

        usage = self._read(response, "usage")
        details = self._read(usage, "completion_tokens_details")
        return ModelResult(
            content=content,
            usage=ModelTokenUsage(
                prompt_tokens=int(self._read(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(self._read(usage, "completion_tokens", 0) or 0),
                cache_hit_tokens=int(
                    self._read(usage, "prompt_cache_hit_tokens", 0) or 0
                ),
                cache_miss_tokens=int(
                    self._read(usage, "prompt_cache_miss_tokens", 0) or 0
                ),
                reasoning_tokens=int(self._read(details, "reasoning_tokens", 0) or 0),
            ),
            finish_reason=self._read(choice, "finish_reason"),
            model=str(self._read(response, "model", self._model_name())),
        )

