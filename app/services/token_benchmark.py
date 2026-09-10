from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from dataclasses import asdict
import hashlib
from typing import Any, Protocol

from ..agents.agent import Agent, AgentMessage, LanguageModel
from ..providers.deepseek import LlmRequestError, LlmStreamChunk
from ..schemas import (
    ChatTurnTokenUsage,
    TokenBenchmarkPlan,
    TokenBenchmarkReport,
    TokenBenchmarkScenarioPlan,
    TokenBenchmarkScenarioResult,
    TokenBenchmarkTurn,
    TokenOverflow,
)
from ..token_usage import ModelResult, ModelTokenUsage, TokenCounter


MODEL_CONTEXT_LIMIT_TOKENS = 1_048_576
OVERFLOW_MARGIN_TOKENS = 50_000
OVERFLOW_MAX_TOKENS = 64
OVERFLOW_PREFIX = (
    "Это проверка реального переполнения контекстного окна. "
    "Ответь одним словом: готово. Данные:"
)
OVERFLOW_FILLER = " x"


class StreamingLanguageModel(LanguageModel, Protocol):
    def generate_stream(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int,
    ) -> Iterator[LlmStreamChunk]: ...


SHORT_REQUESTS = [
    "Коротко объясни, что такое токен в LLM. Ответь одним предложением.",
]

LONG_REQUESTS = [
    "Коротко объясни, что такое токен в LLM. Ответь одним предложением.",
    (
        "Продолжи объяснение: перечисли три причины, почему длинная история "
        "диалога увеличивает стоимость запроса."
    ),
    (
        "Используя весь предыдущий контекст, сравни короткий и длинный диалог "
        "по расходу входных и выходных токенов. Дай компактную таблицу."
    ),
]

OVERFLOW_REQUESTS = [
    (
        "Сгенерированный payload: инструкция + повторитель « x» до объёма "
        "> 1 048 576 токенов. Перед отправкой страница покажет точный размер "
        "и SHA-256 фактического текста."
    ),
]


def benchmark_plan() -> TokenBenchmarkPlan:
    return TokenBenchmarkPlan(
        api_calls=len(SHORT_REQUESTS) + len(LONG_REQUESTS) + len(OVERFLOW_REQUESTS),
        scenarios=[
            TokenBenchmarkScenarioPlan(
                id="short",
                title="Короткий диалог",
                description="Один реальный запрос к DeepSeek.",
                requests=SHORT_REQUESTS,
            ),
            TokenBenchmarkScenarioPlan(
                id="long",
                title="Длинный диалог",
                description="Три реальных запроса с передачей всей истории.",
                requests=LONG_REQUESTS,
            ),
            TokenBenchmarkScenarioPlan(
                id="overflow",
                title="Переполнение",
                description=(
                    "Реальный API-вызов с payload больше контекста DeepSeek V4."
                ),
                requests=OVERFLOW_REQUESTS,
            ),
        ],
    )


class TokenBenchmarkService:
    def __init__(
        self,
        model: LanguageModel,
        counter: TokenCounter,
        *,
        system_prompt: str,
        model_context_limit_tokens: int = MODEL_CONTEXT_LIMIT_TOKENS,
        overflow_margin_tokens: int = OVERFLOW_MARGIN_TOKENS,
    ) -> None:
        self._model = model
        self._counter = counter
        self._system_prompt = system_prompt
        self._model_context_limit_tokens = model_context_limit_tokens
        self._overflow_margin_tokens = overflow_margin_tokens

    @staticmethod
    def _usage(metrics, turn: int) -> ChatTurnTokenUsage:
        return ChatTurnTokenUsage(turn=turn, **asdict(metrics))

    def _build_overflow_request(self) -> tuple[str, int]:
        target = self._model_context_limit_tokens + self._overflow_margin_tokens
        repetitions = target
        request = OVERFLOW_PREFIX + OVERFLOW_FILLER * repetitions
        messages: list[AgentMessage] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": request},
        ]
        prompt_tokens = self._counter.count_messages(messages)
        while prompt_tokens <= self._model_context_limit_tokens:
            repetitions += max(target - prompt_tokens, 1_024)
            request = OVERFLOW_PREFIX + OVERFLOW_FILLER * repetitions
            messages[-1] = {"role": "user", "content": request}
            prompt_tokens = self._counter.count_messages(messages)
        return request, prompt_tokens

    def _stream_scenario(
        self,
        plan: TokenBenchmarkScenarioPlan,
        *,
        context_limit_tokens: int,
        max_tokens: int,
    ) -> Generator[dict[str, Any], None, TokenBenchmarkScenarioResult]:
        context: list[AgentMessage] = []
        turns: list[TokenBenchmarkTurn] = []
        overflow: TokenOverflow | None = None
        generate_stream = getattr(self._model, "generate_stream", None)
        if not callable(generate_stream):
            raise TypeError("Benchmark model must support generate_stream")

        yield {"type": "scenario_started", "scenario_id": plan.id}
        for turn_number, display_request in enumerate(plan.requests, start=1):
            request_text = display_request
            generated_prompt_tokens: int | None = None
            if plan.id == "overflow":
                request_text, generated_prompt_tokens = self._build_overflow_request()

            request_sha256 = hashlib.sha256(request_text.encode()).hexdigest()
            yield {
                "type": "turn_started",
                "scenario_id": plan.id,
                "turn": turn_number,
                "request": display_request,
                "request_chars": len(request_text),
                "request_sha256": request_sha256,
                "estimated_prompt_tokens": generated_prompt_tokens,
            }

            agent_context_limit = context_limit_tokens
            if generated_prompt_tokens is not None:
                agent_context_limit = max(
                    agent_context_limit,
                    generated_prompt_tokens + max_tokens + 1,
                )
            agent = Agent(
                self._model,
                self._counter,
                system_prompt=self._system_prompt,
                max_tokens=max_tokens,
                context_limit_tokens=agent_context_limit,
                overflow_strategy="reject",
            )
            prepared = agent.prepare(context, request_text)
            content_parts: list[str] = []
            usage: ModelTokenUsage | None = None
            finish_reason: str | None = None
            model = ""
            try:
                for chunk in generate_stream(
                    messages=prepared.messages,
                    max_tokens=max_tokens,
                ):
                    if chunk.status:
                        yield {
                            "type": "status",
                            "scenario_id": plan.id,
                            "turn": turn_number,
                            "message": chunk.status,
                        }
                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield {
                            "type": "response_delta",
                            "scenario_id": plan.id,
                            "turn": turn_number,
                            "delta": chunk.content,
                        }
                    usage = chunk.usage or usage
                    finish_reason = chunk.finish_reason or finish_reason
                    model = chunk.model or model
            except LlmRequestError as error:
                if plan.id != "overflow" or not error.is_context_overflow:
                    raise
                overflow = TokenOverflow(
                    prompt_tokens=prepared.estimated_prompt_tokens,
                    reserved_output_tokens=max_tokens,
                    context_limit_tokens=self._model_context_limit_tokens,
                    overflow_tokens=max(
                        0,
                        prepared.estimated_prompt_tokens
                        + max_tokens
                        - self._model_context_limit_tokens,
                    ),
                    request_sent_to_api=True,
                    request_chars=len(request_text),
                    request_sha256=request_sha256,
                    provider_status_code=error.status_code,
                    provider_error_code=error.provider_code,
                    provider_error_message=error.provider_message,
                )
                turns.append(
                    TokenBenchmarkTurn(turn=turn_number, request=display_request)
                )
                yield {
                    "type": "overflow",
                    "scenario_id": plan.id,
                    "turn": turn_number,
                    "overflow": overflow.model_dump(),
                }
                break

            if usage is None:
                raise LlmRequestError("DeepSeek stream completed without token usage")
            result = agent.complete(
                prepared,
                ModelResult(
                    content="".join(content_parts),
                    usage=usage,
                    finish_reason=finish_reason,
                    model=model,
                ),
            )
            turn_usage = self._usage(result.metrics, turn_number)
            turns.append(
                TokenBenchmarkTurn(
                    turn=turn_number,
                    request=display_request,
                    response=result.content,
                    token_usage=turn_usage,
                )
            )
            context.extend(
                [
                    {"role": "user", "content": request_text},
                    {"role": "assistant", "content": result.content},
                ]
            )
            yield {
                "type": "turn_completed",
                "scenario_id": plan.id,
                "turn": turn_number,
                "token_usage": turn_usage.model_dump(),
            }

        completed = [turn.token_usage for turn in turns if turn.token_usage]
        scenario = TokenBenchmarkScenarioResult(
            **plan.model_dump(),
            status="overflow" if overflow else "completed",
            turns=turns,
            prompt_tokens=sum(item.prompt_tokens for item in completed),
            completion_tokens=sum(item.completion_tokens for item in completed),
            total_tokens=sum(item.total_tokens for item in completed),
            estimated_cost_usd=sum(
                item.estimated_cost_usd or 0.0 for item in completed
            ),
            overflow=overflow,
        )
        yield {
            "type": "scenario_completed",
            "scenario_id": plan.id,
            "scenario": scenario.model_dump(mode="json"),
        }
        return scenario

    def stream_events(self) -> Iterator[dict[str, Any]]:
        plan = benchmark_plan()
        yield {"type": "benchmark_started", "api_calls": plan.api_calls}
        scenarios: list[TokenBenchmarkScenarioResult] = []
        limits = [
            (self._model_context_limit_tokens, 192),
            (self._model_context_limit_tokens, 192),
            (self._model_context_limit_tokens * 2, OVERFLOW_MAX_TOKENS),
        ]
        for scenario_plan, (agent_limit, max_tokens) in zip(
            plan.scenarios,
            limits,
            strict=True,
        ):
            scenario = yield from self._stream_scenario(
                scenario_plan,
                context_limit_tokens=agent_limit,
                max_tokens=max_tokens,
            )
            scenarios.append(scenario)

        succeeded = sum(
            1
            for scenario in scenarios
            for turn in scenario.turns
            if turn.token_usage is not None
        )
        attempted = succeeded + sum(
            1
            for scenario in scenarios
            if scenario.overflow and scenario.overflow.request_sent_to_api
        )
        report = TokenBenchmarkReport(
            source="DeepSeek streaming API usage + официальный локальный tokenizer",
            api_calls_attempted=attempted,
            api_calls_succeeded=succeeded,
            scenarios=scenarios,
        )
        yield {
            "type": "benchmark_completed",
            "report": report.model_dump(mode="json"),
        }
