from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 2


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int: ...


@dataclass(frozen=True, slots=True)
class ModelTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: ModelTokenUsage) -> ModelTokenUsage:
        return ModelTokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelResult:
    content: str
    usage: ModelTokenUsage = ModelTokenUsage()
    finish_reason: str | None = None
    model: str = ""


@dataclass(frozen=True, slots=True)
class AgentTokenMetrics:
    current_message_tokens: int
    history_tokens: int
    sent_history_tokens: int
    system_prompt_tokens: int
    estimated_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    reasoning_tokens: int
    total_tokens: int
    context_limit_tokens: int
    reserved_output_tokens: int
    dropped_messages: int
    finish_reason: str | None
    model: str
    estimated_cost_usd: float | None
    summary_prompt_tokens: int = 0
    summary_completion_tokens: int = 0
    summary_total_tokens: int = 0
    summary_tokens: int = 0
    summary_estimated_cost_usd: float | None = None
    compressed_messages: int = 0
    retained_messages: int = 0


@dataclass(frozen=True, slots=True)
class TokenPrices:
    cache_hit: float
    cache_miss: float
    output: float


# USD per 1M tokens. Source checked 2026-09-10:
# https://api-docs.deepseek.com/quick_start/pricing/
FLASH_OFF_PEAK = TokenPrices(cache_hit=0.007, cache_miss=0.22, output=0.66)
FLASH_PEAK = TokenPrices(cache_hit=0.014, cache_miss=0.44, output=1.32)
PRO_OFF_PEAK = TokenPrices(cache_hit=0.022, cache_miss=0.66, output=1.98)
PRO_PEAK = TokenPrices(cache_hit=0.044, cache_miss=1.32, output=3.96)


def is_deepseek_peak(at: datetime | None = None) -> bool:
    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    utc = moment.astimezone(timezone.utc)
    return utc.weekday() < 5 and utc.hour in {1, 2, 3, 6, 7, 8, 9}


def prices_for_model(model: str, *, at: datetime | None = None) -> TokenPrices | None:
    normalized = model.casefold()
    peak = is_deepseek_peak(at)
    if normalized in {
        "deepseek-flash",
        "deepseek-chat",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
    }:
        return FLASH_PEAK if peak else FLASH_OFF_PEAK
    if normalized in {"deepseek-reasoner", "deepseek-v4-pro"}:
        return PRO_PEAK if peak else PRO_OFF_PEAK
    return None


def estimate_cost_usd(
    usage: ModelTokenUsage,
    model: str,
    *,
    at: datetime | None = None,
) -> float | None:
    prices = prices_for_model(model, at=at)
    if prices is None:
        return None
    cache_hit = usage.cache_hit_tokens
    cache_miss = usage.cache_miss_tokens
    if cache_hit + cache_miss == 0:
        cache_miss = usage.prompt_tokens
    return (
        cache_hit * prices.cache_hit
        + cache_miss * prices.cache_miss
        + usage.completion_tokens * prices.output
    ) / 1_000_000
