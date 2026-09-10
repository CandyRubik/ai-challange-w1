from __future__ import annotations

from types import SimpleNamespace

from app.providers.deepseek import DeepSeekProvider


class Completions:
    def create(self, **request):
        assert request["max_tokens"] == 128
        return SimpleNamespace(
            model="deepseek-v4-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Готово"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=8,
                prompt_cache_hit_tokens=100,
                prompt_cache_miss_tokens=20,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
            ),
        )


def test_provider_exposes_api_usage() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = DeepSeekProvider(client=client, model="deepseek-v4-flash")

    result = provider.generate(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=128,
    )

    assert result.content == "Готово"
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 8
    assert result.usage.cache_hit_tokens == 100
    assert result.usage.reasoning_tokens == 4
    assert result.finish_reason == "stop"

