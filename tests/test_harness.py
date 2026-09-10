from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import re

import pytest

from app.agents.agent import Agent, AgentContextOverflow, AgentMessage
from app.services.token_experiments import HaystackGenerator, TokenExperimentService
from app.token_usage import ModelResult, ModelTokenUsage, estimate_cost_usd


class WordCounter:
    def count_text(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(self.count_text(item["content"]) + 4 for item in messages) + 2


class NeedleModel:
    def __init__(self) -> None:
        self.calls: list[list[AgentMessage]] = []

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int,
    ) -> ModelResult:
        del max_tokens
        self.calls.append(list(messages))
        match = re.search(r"ORBIT-[0-9A-F]{8}", messages[-1]["content"])
        assert match
        return ModelResult(
            content=match.group(0),
            usage=ModelTokenUsage(
                prompt_tokens=1_050,
                completion_tokens=8,
                cache_miss_tokens=1_050,
            ),
            finish_reason="stop",
            model="deepseek-v4-flash",
        )


def make_agent(model: NeedleModel, *, limit: int = 10_000) -> Agent:
    return Agent(
        model,
        WordCounter(),
        system_prompt="Answer briefly.",
        max_tokens=16,
        context_limit_tokens=limit,
    )


def test_token_sized_haystack_is_exact_and_places_fact() -> None:
    generator = HaystackGenerator(WordCounter())
    document, secret = generator.generate_tokens(10_000, 0.5)

    assert WordCounter().count_text(document) == 10_000
    assert secret in document
    assert 0.45 <= document.index(secret) / len(document) <= 0.55


def test_byte_sized_haystack_is_exactly_five_mib() -> None:
    generator = HaystackGenerator(WordCounter())
    document, secret = generator.generate_bytes(5 * 1024 * 1024, 0.5)

    assert len(document.encode("utf-8")) == 5 * 1024 * 1024
    assert secret in document


def test_needle_experiment_returns_real_provider_usage() -> None:
    model = NeedleModel()
    counter = WordCounter()
    service = TokenExperimentService(
        make_agent(model),
        HaystackGenerator(counter),
        counter,
    )

    response = service.run(
        target_tokens=1_000,
        target_bytes=None,
        needle_position=0.5,
    )

    assert response.status == "completed"
    assert response.document_tokens == 1_000
    assert response.found is True
    assert response.answer == response.secret
    assert response.token_usage is not None
    assert response.token_usage.prompt_tokens == 1_050


def test_overflow_stops_before_provider_call() -> None:
    model = NeedleModel()
    counter = WordCounter()
    service = TokenExperimentService(
        make_agent(model, limit=900),
        HaystackGenerator(counter),
        counter,
    )

    response = service.run(
        target_tokens=1_000,
        target_bytes=None,
        needle_position=0.5,
    )

    assert response.status == "overflow"
    assert response.overflow is not None
    assert response.overflow.overflow_tokens > 0
    assert model.calls == []


def test_agent_counts_current_message_and_history() -> None:
    model = NeedleModel()
    agent = make_agent(model)
    context: list[AgentMessage] = [
        {"role": "user", "content": "ORBIT-12345678 first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    result = agent.respond(context, "find ORBIT-12345678")

    assert result.metrics.current_message_tokens == 2
    assert result.metrics.history_tokens == 13
    assert result.metrics.prompt_tokens == 1_050
    assert result.metrics.completion_tokens == 8


def test_agent_rejects_context_overflow() -> None:
    model = NeedleModel()
    agent = make_agent(model, limit=20)

    with pytest.raises(AgentContextOverflow):
        agent.respond([], "ORBIT-12345678 " + "word " * 20)

    assert model.calls == []


def test_cost_uses_cache_breakdown() -> None:
    usage = ModelTokenUsage(
        prompt_tokens=1_000,
        completion_tokens=100,
        cache_hit_tokens=800,
        cache_miss_tokens=200,
    )
    off_peak = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)

    assert estimate_cost_usd(usage, "deepseek-v4-flash", at=off_peak) == pytest.approx(
        0.0000924
    )

