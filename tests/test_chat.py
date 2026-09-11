from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

from fastapi.testclient import TestClient
import pytest

from app.agents.agent import (
    Agent,
    AgentContextOverflow,
    AgentInputError,
    AgentMessage,
    AgentOutputError,
)
from app.main import app, get_chat_session_service
from app.services.chat_sessions import ChatSessionService, SQLiteChatSessionRepository
from app.services.context_compression import ConversationSummarizer
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


def make_agent(model: FakeLanguageModel, **kwargs: object) -> Agent:
    return Agent(model, WordCounter(), **kwargs)


def repository(tmp_path: Path) -> SQLiteChatSessionRepository:
    return SQLiteChatSessionRepository(tmp_path / "chat.sqlite3")


def test_frontend_uses_same_origin_api_by_default() -> None:
    javascript = (Path(__file__).parents[1] / "static" / "app.js").read_text()

    assert 'window.API_BASE_URL || ""' in javascript
    assert "http://localhost:8000" not in javascript


def test_frontend_keeps_messages_scrollable_and_submits_on_enter() -> None:
    static_dir = Path(__file__).parents[1] / "static"
    javascript = (static_dir / "app.js").read_text()
    styles = (static_dir / "styles.css").read_text()

    assert 'messageInput.addEventListener("keydown"' in javascript
    assert 'event.key !== "Enter" || event.shiftKey || event.isComposing' in javascript
    assert "messageForm.requestSubmit()" in javascript
    assert "appendPendingExchange(content)" in javascript
    assert 'pending.className = "message assistant pending"' in javascript
    assert "@keyframes typing-pulse" in styles
    assert "min-height: 0; overflow: hidden;" in styles
    assert "min-height: 0; overflow-y: auto;" in styles


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


def test_agent_places_summary_before_unsummarized_tail() -> None:
    model = FakeLanguageModel(["Готово"])
    agent = make_agent(model)

    result = agent.respond(
        [{"role": "assistant", "content": "Свежий ответ"}],
        "Продолжим?",
        context_summary="Пользователя зовут Лена.",
    )

    assert model.calls[0][0] == [
        {"role": "system", "content": Agent.default_system_prompt},
        {
            "role": "system",
            "content": "Conversation summary (older messages):\nПользователя зовут Лена.",
        },
        {"role": "assistant", "content": "Свежий ответ"},
        {"role": "user", "content": "Продолжим?"},
    ]
    assert result.metrics.summary_tokens > 0
    assert result.metrics.retained_messages == 1


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


def test_ten_messages_are_compacted_without_hiding_the_dialogue(
    tmp_path: Path,
) -> None:
    answers = [f"Ответ {index}" for index in range(5)] + [
        "Сжатая память: секретный код 42",
        "Код по-прежнему 42",
    ]
    model = FakeLanguageModel(answers)
    counter = WordCounter()
    service = ChatSessionService(
        repository(tmp_path),
        make_agent(model),
        summarizer=ConversationSummarizer(model, max_tokens=100),
        token_counter=counter,
        summary_batch_messages=10,
    )
    session = service.create()
    for index in range(5):
        service.send(session.id, f"Сообщение {index}")

    loaded = service.get(session.id)

    assert len(loaded.messages) == 10
    assert [message.position for message in loaded.messages] == list(range(10))
    assert loaded.messages[0].content == "Сообщение 0"
    assert loaded.messages[-1].content == "Ответ 4"
    assert len(loaded.compactions) == 1
    assert loaded.compactions[0].summary == "Сжатая память: секретный код 42"
    assert loaded.compactions[0].source_start_position == 0
    assert loaded.compactions[0].source_end_position == 9
    assert loaded.compactions[0].summarized_message_count == 10
    assert loaded.context_summary.content == "Сжатая память: секретный код 42"
    assert loaded.context_summary.summarized_message_count == 10
    assert loaded.token_usage.summary_total_tokens > 0
    assert loaded.token_usage.turns[-1].compressed_messages == 10
    answer_call, summary_call = model.calls[-2:]
    assert summary_call[0][0]["content"].startswith("You maintain durable memory")
    assert "Сообщение 0" in summary_call[0][1]["content"]
    assert answer_call[0][-1]["content"] == "Сообщение 4"

    service.send(session.id, "Какой код?")
    next_prompt = model.calls[-1][0]
    assert next_prompt[1]["content"].startswith("Conversation summary")
    assert next_prompt[-1]["content"] == "Какой код?"


def test_context_summary_survives_repository_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "summary.sqlite3"
    model = FakeLanguageModel(["Ответ"] * 5 + ["Summary"])
    counter = WordCounter()
    service = ChatSessionService(
        SQLiteChatSessionRepository(database_path),
        make_agent(model),
        summarizer=ConversationSummarizer(model),
        token_counter=counter,
        summary_batch_messages=10,
    )
    session = service.create()
    for index in range(5):
        service.send(session.id, f"Вопрос {index}")

    restarted = SQLiteChatSessionRepository(database_path).get(session.id)

    assert restarted.context_summary.content == "Summary"
    assert restarted.context_summary.summarized_message_count == 10
    assert len(restarted.messages) == 10
    assert len(restarted.compactions) == 1
    assert restarted.compactions[0].source_start_position == 0
    assert restarted.compactions[0].source_end_position == 9


def test_legacy_summary_is_migrated_to_a_positioned_compaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-summary.sqlite3"
    repository_before_upgrade = SQLiteChatSessionRepository(database_path)
    session = repository_before_upgrade.create()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO chat_context_summaries
                (session_id, content, summarized_message_count, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (session.id, "Legacy summary", 10, now),
        )
        connection.execute("DROP TABLE chat_context_compactions")

    migrated = SQLiteChatSessionRepository(database_path).get(session.id)

    assert len(migrated.compactions) == 1
    assert migrated.compactions[0].summary == "Legacy summary"
    assert migrated.compactions[0].source_start_position == 0
    assert migrated.compactions[0].source_end_position == 9


def test_next_ten_messages_are_merged_into_the_existing_summary(
    tmp_path: Path,
) -> None:
    answers = [
        "Ответ 0", "Ответ 1", "Ответ 2", "Ответ 3", "Ответ 4", "Summary 1",
        "Ответ 5", "Ответ 6", "Ответ 7", "Ответ 8", "Ответ 9", "Summary 2",
    ]
    model = FakeLanguageModel(answers)
    counter = WordCounter()
    service = ChatSessionService(
        repository(tmp_path),
        make_agent(model),
        summarizer=ConversationSummarizer(model),
        token_counter=counter,
        summary_batch_messages=10,
    )
    session = service.create()

    for index in range(10):
        service.send(session.id, f"Вопрос {index}")

    loaded = service.get(session.id)
    second_summary_prompt = model.calls[-1][0][1]["content"]
    assert len(loaded.messages) == 20
    assert [message.position for message in loaded.messages] == list(range(20))
    assert len(loaded.compactions) == 2
    assert [
        (item.source_start_position, item.source_end_position)
        for item in loaded.compactions
    ] == [(0, 9), (10, 19)]
    assert loaded.compactions[-1].summary == "Summary 2"
    assert loaded.context_summary.summarized_message_count == 20
    assert "Summary 1" in second_summary_prompt
    assert "Вопрос 5" in second_summary_prompt


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
