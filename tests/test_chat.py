from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from fastapi.testclient import TestClient
import pytest

import app.main as main_module
from app.agents.agent import (
    Agent,
    AgentContextOverflow,
    AgentInputError,
    AgentMessage,
    AgentOutputError,
)
from app.main import app, get_chat_session_service
from app.providers.deepseek import LlmStreamChunk
from app.schemas import ChatExperimentSettings
from app.services.chat_sessions import ChatSessionService, SQLiteChatSessionRepository
from app.services.token_benchmark import TokenBenchmarkService, benchmark_plan
from app.token_usage import ModelResult, ModelTokenUsage


class WordCounter:
    def count_text(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(self.count_text(item["content"]) + 4 for item in messages) + 2


class FakeLanguageModel:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or ["Ответ агента"]
        self.calls: list[tuple[list[AgentMessage], int]] = []

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = 2_000,
    ) -> ModelResult:
        self.calls.append((list(messages), max_tokens))
        answer = self.answers.pop(0)
        prompt_tokens = WordCounter().count_messages(messages)
        return ModelResult(
            content=answer,
            usage=ModelTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=WordCounter().count_text(answer),
                cache_miss_tokens=prompt_tokens,
            ),
            finish_reason="stop",
            model="deepseek-v4-flash",
        )


class FakeStreamingLanguageModel(FakeLanguageModel):
    def generate_stream(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = 2_000,
    ):
        self.calls.append((list(messages), max_tokens))
        answer = self.answers.pop(0)
        midpoint = max(1, len(answer) // 2)
        yield LlmStreamChunk(
            content=answer[:midpoint],
            model="deepseek-v4-flash",
        )
        yield LlmStreamChunk(
            content=answer[midpoint:],
            finish_reason="stop",
            usage=ModelTokenUsage(
                prompt_tokens=WordCounter().count_messages(messages),
                completion_tokens=WordCounter().count_text(answer),
            ),
            model="deepseek-v4-flash",
        )


def make_agent(model: FakeLanguageModel, **kwargs: object) -> Agent:
    return Agent(model, WordCounter(), **kwargs)


def repository(tmp_path: Path) -> SQLiteChatSessionRepository:
    return SQLiteChatSessionRepository(tmp_path / "chat.sqlite3")


def test_frontend_uses_same_origin_api_by_default() -> None:
    javascript = (Path(__file__).parents[1] / "static" / "app.js").read_text()

    assert 'window.API_BASE_URL || ""' in javascript
    assert "http://localhost:8000" not in javascript


def test_agent_counts_request_history_and_response() -> None:
    model = FakeLanguageModel(["  Готово  "])
    agent = make_agent(model)

    result = agent.respond(
        [{"role": "user", "content": "Раньше"}, {"role": "assistant", "content": "Да"}],
        "  Продолжим?  ",
    )

    assert result.content == "Готово"
    assert result.metrics.current_message_tokens == 1
    assert result.metrics.history_tokens == 10
    assert result.metrics.completion_tokens == 1
    assert model.calls[0][0][0]["role"] == "system"
    assert model.calls[0][0][-1] == {"role": "user", "content": "Продолжим?"}


def test_agent_trims_old_context_by_token_budget() -> None:
    model = FakeLanguageModel()
    agent = make_agent(
        model,
        system_prompt="short system",
        context_limit_tokens=30,
        max_tokens=5,
        overflow_strategy="trim",
    )
    context: list[AgentMessage] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]

    result = agent.respond(context, "new question")

    sent_messages = model.calls[0][0]
    assert [message["content"] for message in sent_messages[1:]] == ["new question"]
    assert result.metrics.dropped_messages == 2
    assert result.metrics.history_tokens == 12


def test_agent_rejects_context_overflow_before_model_call() -> None:
    model = FakeLanguageModel()
    agent = make_agent(
        model,
        system_prompt="system",
        context_limit_tokens=20,
        max_tokens=5,
    )

    with pytest.raises(AgentContextOverflow) as raised:
        agent.respond([], "word " * 20)

    assert raised.value.overflow_tokens > 0
    assert model.calls == []


def test_agent_rejects_empty_or_oversized_values() -> None:
    model = FakeLanguageModel(["x " * 800_001, "Ответ"])
    agent = make_agent(model)

    with pytest.raises(AgentInputError):
        agent.respond([], "   ")
    with pytest.raises(AgentInputError):
        agent.respond([], "x" * 8_000_001)
    with pytest.raises(AgentOutputError):
        agent.respond([], "Вопрос")

    assert len(model.calls) == 1


def test_sessions_keep_contexts_isolated(tmp_path: Path) -> None:
    model = FakeLanguageModel(["Ответ A", "Ответ B", "Ответ A2"])
    service = ChatSessionService(repository(tmp_path), make_agent(model))
    first = service.create()
    second = service.create()

    service.send(first.id, "Вопрос A")
    service.send(second.id, "Вопрос B")
    service.send(first.id, "Ещё A")

    assert [message.content for message in service.get(first.id).messages] == [
        "Вопрос A", "Ответ A", "Ещё A", "Ответ A2",
    ]
    assert [message.content for message in service.get(second.id).messages] == [
        "Вопрос B", "Ответ B",
    ]
    assert "Вопрос B" not in [message["content"] for message in model.calls[2][0]]


def test_context_and_token_usage_survive_backend_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "persistent-chat.sqlite3"
    first_service = ChatSessionService(
        SQLiteChatSessionRepository(database_path),
        make_agent(FakeLanguageModel(["Тебя зовут Лена"])),
    )
    session = first_service.create()
    first_service.send(session.id, "Запомни: меня зовут Лена")

    restarted_model = FakeLanguageModel(["Тебя зовут Лена"])
    restarted_service = ChatSessionService(
        SQLiteChatSessionRepository(database_path),
        make_agent(restarted_model),
    )
    restarted_service.send(session.id, "Как меня зовут?")

    loaded = restarted_service.get(session.id)
    assert len(loaded.token_usage.turns) == 2
    assert loaded.token_usage.total_tokens > 0
    assert [message.content for message in loaded.messages] == [
        "Запомни: меня зовут Лена",
        "Тебя зовут Лена",
        "Как меня зовут?",
        "Тебя зовут Лена",
    ]


def test_repository_adds_usage_table_to_existing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "old-chat.sqlite3"
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (session_id, position)
            );
            """
        )
        connection.execute(
            "INSERT INTO chat_sessions VALUES (?, ?, ?, ?)",
            ("legacy", "Старый чат", now, now),
        )

    migrated = SQLiteChatSessionRepository(database_path)

    assert migrated.get("legacy").token_metrics == ()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'chat_turn_usage'"
        ).fetchone() is not None


def test_agent_applies_runtime_experiment_options() -> None:
    model = FakeLanguageModel()
    agent = make_agent(
        model,
        system_prompt="Экспериментальный prompt",
        max_tokens=777,
        context_enabled=False,
    )

    agent.respond([{"role": "user", "content": "Скрытый контекст"}], "Новый вопрос")

    messages, max_tokens = model.calls[0]
    assert messages == [
        {"role": "system", "content": "Экспериментальный prompt"},
        {"role": "user", "content": "Новый вопрос"},
    ]
    assert max_tokens == 777


def test_chat_session_http_flow_returns_token_usage(tmp_path: Path) -> None:
    service = ChatSessionService(
        repository(tmp_path),
        make_agent(FakeLanguageModel(["Привет! Чем помочь?"])),
    )
    app.dependency_overrides[get_chat_session_service] = lambda: service
    client = TestClient(app)
    try:
        created = client.post("/api/chat/sessions")
        session_id = created.json()["id"]
        sent = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"content": "Привет"},
        )
        loaded = client.get(f"/api/chat/sessions/{session_id}")
        sessions = client.get("/api/chat/sessions")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert sent.status_code == 200
    assert sent.json()["assistant_message"]["content"] == "Привет! Чем помочь?"
    assert sent.json()["token_usage"]["current_message_tokens"] == 1
    assert loaded.json()["token_usage"]["turns"][0]["completion_tokens"] == 3
    assert sessions.json()[0]["title"] == "Привет"


def test_chat_http_reports_structured_context_overflow(tmp_path: Path) -> None:
    model = FakeLanguageModel()
    service = ChatSessionService(
        repository(tmp_path),
        make_agent(
            model,
            system_prompt="system",
            context_limit_tokens=20,
            max_tokens=5,
        ),
    )
    session = service.create()
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        response = TestClient(app).post(
            f"/api/chat/sessions/{session.id}/messages",
            json={"content": "word " * 20},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "context_overflow"
    assert response.json()["detail"]["overflow_tokens"] > 0
    assert model.calls == []


def test_benchmark_exposes_real_requests_growth_and_overflow() -> None:
    model = FakeLanguageModel(["Короткий ответ", "Ответ 1", "Ответ 2", "Ответ 3"])

    report = TokenBenchmarkService(model, WordCounter(), system_prompt="system").run()

    short, long, overflow = report.scenarios
    assert long.total_tokens > short.total_tokens
    assert long.turns[0].token_usage is not None
    assert long.turns[-1].token_usage is not None
    assert (
        long.turns[-1].token_usage.history_tokens
        > long.turns[0].token_usage.history_tokens
    )
    assert overflow.status == "overflow"
    assert overflow.overflow is not None
    assert overflow.overflow.overflow_tokens > 0
    assert overflow.turns[0].response is None
    assert len(model.calls) == benchmark_plan().api_calls == 4


def test_benchmark_streams_requests_deltas_usage_and_overflow() -> None:
    model = FakeStreamingLanguageModel(
        ["Короткий ответ", "Ответ 1", "Ответ 2", "Ответ 3"]
    )

    events = list(
        TokenBenchmarkService(model, WordCounter(), system_prompt="system")
        .stream_events()
    )

    event_types = [event["type"] for event in events]
    assert event_types[0] == "benchmark_started"
    assert event_types.count("turn_started") == 5
    assert event_types.count("response_delta") == 8
    assert event_types.count("turn_completed") == 4
    assert event_types.count("overflow") == 1
    assert event_types[-1] == "benchmark_completed"
    report = events[-1]["report"]
    assert report["scenarios"][1]["turns"][2]["response"] == "Ответ 3"
    assert report["scenarios"][2]["status"] == "overflow"
    assert len(model.calls) == 4


def test_benchmark_plan_api_shows_prompts_before_paid_run() -> None:
    response = TestClient(app).get("/api/benchmark/plan")

    assert response.status_code == 200
    assert response.json()["api_calls"] == 4
    assert response.json()["scenarios"][0]["requests"][0].startswith(
        "Коротко объясни"
    )


def test_benchmark_http_stream_finishes_and_saves_latest(monkeypatch) -> None:
    model = FakeStreamingLanguageModel(
        ["Короткий ответ", "Ответ 1", "Ответ 2", "Ответ 3"]
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "DeepSeekProvider", lambda **_: model)
    monkeypatch.setattr(main_module, "_token_counter", WordCounter())
    monkeypatch.setattr(main_module, "_latest_benchmark_report", None)

    client = TestClient(app)
    response = client.post("/api/benchmark/run")
    events = [json.loads(line) for line in response.text.splitlines()]
    latest = client.get("/api/benchmark/latest")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events[-1]["type"] == "benchmark_completed"
    assert latest.status_code == 200
    assert latest.json()["scenarios"][1]["turns"][2]["response"] == "Ответ 3"


def test_clear_chat_database_removes_all_sessions(tmp_path: Path) -> None:
    service = ChatSessionService(
        repository(tmp_path),
        make_agent(FakeLanguageModel(["Ответ"])),
    )
    session = service.create()
    service.send(session.id, "Сообщение")
    service.create()
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        response = TestClient(app).delete("/api/chat/sessions")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204
    assert service.list() == []


def test_unknown_chat_session_returns_404(tmp_path: Path) -> None:
    service = ChatSessionService(repository(tmp_path), make_agent(FakeLanguageModel()))
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        response = TestClient(app).get("/api/chat/sessions/missing")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404


def test_debug_settings_can_change_context_policy() -> None:
    changed = ChatExperimentSettings(
        model="deepseek-v4-pro",
        thinking_enabled=False,
        history_enabled=False,
        max_tokens=128,
        context_limit_tokens=512,
        overflow_strategy="trim",
        system_prompt="Тестовый prompt",
    )
    client = TestClient(app)
    try:
        response = client.put("/api/debug/settings", json=changed.model_dump())
        loaded = client.get("/api/debug/settings")
    finally:
        client.put("/api/debug/settings", json=ChatExperimentSettings().model_dump())

    assert response.status_code == 200
    assert loaded.json()["context_limit_tokens"] == 512
    assert loaded.json()["overflow_strategy"] == "trim"
