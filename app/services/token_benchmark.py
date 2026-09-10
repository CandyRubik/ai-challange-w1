from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import asdict
from typing import Any, Protocol

from ..agents.agent import Agent, AgentContextOverflow, AgentMessage, LanguageModel
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


class StreamingLanguageModel(LanguageModel, Protocol):
    def generate_stream(
        self,
        *,
        messages: list[AgentMessage],
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
    "Проверь этот текст при маленьком контекстном окне: "
    + "история диалога увеличивает вход модели и стоимость запроса; " * 80,
]


def benchmark_plan() -> TokenBenchmarkPlan:
    return TokenBenchmarkPlan(
        api_calls=len(SHORT_REQUESTS) + len(LONG_REQUESTS),
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
                description="Локальная проверка блокирует запрос до DeepSeek API.",
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
    ) -> None:
        self._model = model
        self._counter = counter
        self._system_prompt = system_prompt

    @staticmethod
    def _usage(metrics, turn: int) -> ChatTurnTokenUsage:
        return ChatTurnTokenUsage(turn=turn, **asdict(metrics))

    def _run(
        self,
        plan: TokenBenchmarkScenarioPlan,
        *,
        context_limit_tokens: int,
        max_tokens: int,
    ) -> TokenBenchmarkScenarioResult:
        agent = Agent(
            self._model,
            self._counter,
            system_prompt=self._system_prompt,
            max_tokens=max_tokens,
            context_limit_tokens=context_limit_tokens,
            overflow_strategy="reject",
        )
        context: list[AgentMessage] = []
        turns: list[TokenBenchmarkTurn] = []
        overflow: TokenOverflow | None = None

        for turn_number, request in enumerate(plan.requests, start=1):
            try:
                result = agent.respond(context, request)
            except AgentContextOverflow as error:
                overflow = TokenOverflow(
                    prompt_tokens=error.prompt_tokens,
                    reserved_output_tokens=error.reserved_output_tokens,
                    context_limit_tokens=error.context_limit_tokens,
                    overflow_tokens=error.overflow_tokens,
                )
                turns.append(TokenBenchmarkTurn(turn=turn_number, request=request))
                break

            usage = self._usage(result.metrics, turn_number)
            turns.append(
                TokenBenchmarkTurn(
                    turn=turn_number,
                    request=request,
                    response=result.content,
                    token_usage=usage,
                )
            )
            context.extend(
                [
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": result.content},
                ]
            )

        completed = [turn.token_usage for turn in turns if turn.token_usage]
        return TokenBenchmarkScenarioResult(
            **plan.model_dump(),
            status="overflow" if overflow else "completed",
            turns=turns,
            prompt_tokens=sum(usage.prompt_tokens for usage in completed),
            completion_tokens=sum(usage.completion_tokens for usage in completed),
            total_tokens=sum(usage.total_tokens for usage in completed),
            estimated_cost_usd=sum(
                usage.estimated_cost_usd or 0.0 for usage in completed
            ),
            overflow=overflow,
        )

    def run(self) -> TokenBenchmarkReport:
        plan = benchmark_plan()
        short, long, overflow = plan.scenarios
        return TokenBenchmarkReport(
            source="DeepSeek API usage + официальный локальный tokenizer",
            scenarios=[
                self._run(short, context_limit_tokens=1_000_000, max_tokens=192),
                self._run(long, context_limit_tokens=1_000_000, max_tokens=192),
                self._run(overflow, context_limit_tokens=256, max_tokens=64),
            ],
        )

    def _stream_scenario(
        self,
        plan: TokenBenchmarkScenarioPlan,
        *,
        context_limit_tokens: int,
        max_tokens: int,
    ) -> Generator[dict[str, Any], None, TokenBenchmarkScenarioResult]:
        agent = Agent(
            self._model,
            self._counter,
            system_prompt=self._system_prompt,
            max_tokens=max_tokens,
            context_limit_tokens=context_limit_tokens,
            overflow_strategy="reject",
        )
        context: list[AgentMessage] = []
        turns: list[TokenBenchmarkTurn] = []
        overflow: TokenOverflow | None = None
        generate_stream = getattr(self._model, "generate_stream", None)
        if not callable(generate_stream):
            raise TypeError("Benchmark model must support generate_stream")

        yield {
            "type": "scenario_started",
            "scenario_id": plan.id,
        }
        for turn_number, request_text in enumerate(plan.requests, start=1):
            yield {
                "type": "turn_started",
                "scenario_id": plan.id,
                "turn": turn_number,
                "request": request_text,
            }
            try:
                prepared = agent.prepare(context, request_text)
            except AgentContextOverflow as error:
                overflow = TokenOverflow(
                    prompt_tokens=error.prompt_tokens,
                    reserved_output_tokens=error.reserved_output_tokens,
                    context_limit_tokens=error.context_limit_tokens,
                    overflow_tokens=error.overflow_tokens,
                )
                turns.append(TokenBenchmarkTurn(turn=turn_number, request=request_text))
                yield {
                    "type": "overflow",
                    "scenario_id": plan.id,
                    "turn": turn_number,
                    "overflow": overflow.model_dump(),
                }
                break

            content_parts: list[str] = []
            usage: ModelTokenUsage | None = None
            finish_reason: str | None = None
            model = ""
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
            turn = TokenBenchmarkTurn(
                turn=turn_number,
                request=request_text,
                response=result.content,
                token_usage=turn_usage,
            )
            turns.append(turn)
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
        limits = [(1_000_000, 192), (1_000_000, 192), (256, 64)]
        for scenario_plan, (context_limit, max_tokens) in zip(
            plan.scenarios,
            limits,
            strict=True,
        ):
            scenario = yield from self._stream_scenario(
                scenario_plan,
                context_limit_tokens=context_limit,
                max_tokens=max_tokens,
            )
            scenarios.append(scenario)

        report = TokenBenchmarkReport(
            source="DeepSeek streaming API usage + официальный локальный tokenizer",
            scenarios=scenarios,
        )
        yield {
            "type": "benchmark_completed",
            "report": report.model_dump(mode="json"),
        }
