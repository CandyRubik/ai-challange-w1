from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from ..token_usage import (
    AgentTokenMetrics,
    ModelResult,
    TokenCounter,
    estimate_cost_usd,
)


class AgentMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


AgentContext = Sequence[AgentMessage]


class LanguageModel(Protocol):
    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int,
    ) -> ModelResult: ...


class AgentInputError(ValueError):
    pass


class AgentOutputError(RuntimeError):
    pass


class AgentContextOverflow(RuntimeError):
    def __init__(
        self,
        *,
        prompt_tokens: int,
        reserved_output_tokens: int,
        context_limit_tokens: int,
        current_message_tokens: int,
        history_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.context_limit_tokens = context_limit_tokens
        self.current_message_tokens = current_message_tokens
        self.history_tokens = history_tokens
        self.overflow_tokens = max(
            0,
            prompt_tokens + reserved_output_tokens - context_limit_tokens,
        )
        super().__init__(
            "Контекст переполнен: "
            f"{prompt_tokens} входных + {reserved_output_tokens} выходных токенов "
            f"> лимита {context_limit_tokens}"
        )


@dataclass(frozen=True, slots=True)
class AgentResult:
    content: str
    metrics: AgentTokenMetrics


class Agent:
    def __init__(
        self,
        model: LanguageModel,
        token_counter: TokenCounter,
        *,
        system_prompt: str,
        max_tokens: int = 128,
        context_enabled: bool = True,
        context_limit_tokens: int = 1_000_000,
        overflow_strategy: Literal["reject", "trim"] = "reject",
    ) -> None:
        self._model = model
        self._counter = token_counter
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._context_enabled = context_enabled
        self._context_limit_tokens = context_limit_tokens
        self._overflow_strategy = overflow_strategy

    def _history_tokens(self, messages: Sequence[AgentMessage]) -> int:
        return sum(self._counter.count_text(item["content"]) + 4 for item in messages)

    def respond(self, context: AgentContext, current_message: str) -> AgentResult:
        content = current_message.strip()
        if not content:
            raise AgentInputError("Сообщение не должно быть пустым")
        original_history = [dict(item) for item in context]
        if any(item["role"] not in {"user", "assistant"} for item in original_history):
            raise AgentInputError("История содержит недопустимую роль")

        sent_history = original_history.copy() if self._context_enabled else []
        current: AgentMessage = {"role": "user", "content": content}

        def model_messages() -> list[AgentMessage]:
            return [
                {"role": "system", "content": self._system_prompt},
                *sent_history,
                current,
            ]

        current_tokens = self._counter.count_text(content)
        history_tokens = self._history_tokens(original_history)
        estimated_prompt = self._counter.count_messages(model_messages())
        dropped_messages = 0
        while (
            estimated_prompt + self._max_tokens > self._context_limit_tokens
            and self._overflow_strategy == "trim"
            and sent_history
        ):
            remove_count = 2 if len(sent_history) >= 2 else 1
            del sent_history[:remove_count]
            dropped_messages += remove_count
            estimated_prompt = self._counter.count_messages(model_messages())

        if estimated_prompt + self._max_tokens > self._context_limit_tokens:
            raise AgentContextOverflow(
                prompt_tokens=estimated_prompt,
                reserved_output_tokens=self._max_tokens,
                context_limit_tokens=self._context_limit_tokens,
                current_message_tokens=current_tokens,
                history_tokens=history_tokens,
            )

        model_result = self._model.generate(
            messages=model_messages(),
            max_tokens=self._max_tokens,
        )
        answer = model_result.content.strip()
        if not answer:
            raise AgentOutputError("Модель вернула пустой ответ")
        usage = model_result.usage
        return AgentResult(
            content=answer,
            metrics=AgentTokenMetrics(
                current_message_tokens=current_tokens,
                history_tokens=history_tokens,
                sent_history_tokens=self._history_tokens(sent_history),
                system_prompt_tokens=self._counter.count_text(self._system_prompt),
                estimated_prompt_tokens=estimated_prompt,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cache_hit_tokens=usage.cache_hit_tokens,
                cache_miss_tokens=usage.cache_miss_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                total_tokens=usage.total_tokens,
                context_limit_tokens=self._context_limit_tokens,
                dropped_messages=dropped_messages,
                finish_reason=model_result.finish_reason,
                model=model_result.model,
                estimated_cost_usd=estimate_cost_usd(usage, model_result.model),
            ),
        )

