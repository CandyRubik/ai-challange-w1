from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Protocol, cast
from uuid import uuid4

from ..agents.agent import Agent, AgentMessage
from ..schemas import (
    ChatCheckpoint,
    ChatMessage,
    ChatSendResponse,
    ChatSession,
    ChatSessionSummary,
    ChatSessionTokenUsage,
    ChatTurnTokenUsage,
)
from ..token_usage import AgentTokenMetrics, MESSAGE_OVERHEAD_TOKENS, TokenCounter
from .context_strategies import ContextStrategyName, FactsExtractor, strategy_for


DEFAULT_CHAT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"
DEFAULT_DB_PATH = DEFAULT_CHAT_DB_PATH


class ChatSessionNotFound(LookupError):
    pass


class ChatCheckpointNotFound(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    position: int
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredCheckpoint:
    id: str
    name: str
    message_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    strategy: ContextStrategyName = "sliding_window"
    window_size: int = 6
    parent_session_id: str | None = None
    checkpoint_id: str | None = None
    branch_name: str = "main"
    messages: tuple[StoredMessage, ...] = ()
    token_metrics: tuple[AgentTokenMetrics, ...] = ()
    facts: dict[str, str] | None = None
    checkpoints: tuple[StoredCheckpoint, ...] = ()


class ChatSessionRepository(Protocol):
    def create(
        self,
        *,
        strategy: ContextStrategyName = "sliding_window",
        window_size: int = 6,
    ) -> StoredSession: ...
    def list(self) -> list[StoredSession]: ...
    def get(self, session_id: str) -> StoredSession: ...
    def clear(self) -> None: ...
    def update_strategy(
        self,
        session_id: str,
        strategy: ContextStrategyName,
        window_size: int,
    ) -> StoredSession: ...
    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        token_metrics: AgentTokenMetrics,
        facts: dict[str, str] | None = None,
    ) -> StoredSession: ...
    def create_checkpoint(self, session_id: str, name: str) -> StoredCheckpoint: ...
    def fork(self, session_id: str, checkpoint_id: str, name: str) -> StoredSession: ...


class SQLiteChatSessionRepository:
    """Durable history, structured facts and copy-on-checkpoint chat branches."""

    def __init__(self, database_path: str | Path = DEFAULT_CHAT_DB_PATH) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, position);
                CREATE TABLE IF NOT EXISTS chat_turn_usage (
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    turn INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn)
                );
                CREATE TABLE IF NOT EXISTS chat_facts (
                    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = self._columns(connection, "chat_sessions")
            additions = {
                "strategy": "TEXT NOT NULL DEFAULT 'sliding_window'",
                "window_size": "INTEGER NOT NULL DEFAULT 6",
                "parent_session_id": "TEXT",
                "checkpoint_id": "TEXT",
                "branch_name": "TEXT NOT NULL DEFAULT 'main'",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE chat_sessions ADD COLUMN {name} {declaration}"
                    )

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _load_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> StoredSession:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at, strategy, window_size,
                   parent_session_id, checkpoint_id, branch_name
            FROM chat_sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ChatSessionNotFound(session_id)
        message_rows = connection.execute(
            """SELECT id, position, role, content, created_at
               FROM chat_messages WHERE session_id = ? ORDER BY position""",
            (session_id,),
        ).fetchall()
        usage_rows = connection.execute(
            "SELECT payload FROM chat_turn_usage WHERE session_id = ? ORDER BY turn",
            (session_id,),
        ).fetchall()
        facts_row = connection.execute(
            "SELECT payload FROM chat_facts WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checkpoint_rows = connection.execute(
            """SELECT id, name, message_count, created_at FROM chat_checkpoints
               WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        ).fetchall()
        return StoredSession(
            id=row["id"],
            title=row["title"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            strategy=cast(ContextStrategyName, row["strategy"]),
            window_size=row["window_size"],
            parent_session_id=row["parent_session_id"],
            checkpoint_id=row["checkpoint_id"],
            branch_name=row["branch_name"],
            messages=tuple(
                StoredMessage(
                    id=item["id"],
                    position=item["position"],
                    role=item["role"],
                    content=item["content"],
                    created_at=self._datetime(item["created_at"]),
                )
                for item in message_rows
            ),
            token_metrics=tuple(
                AgentTokenMetrics(
                    **{
                        key: value
                        for key, value in json.loads(item["payload"]).items()
                        if key in {field.name for field in fields(AgentTokenMetrics)}
                    }
                )
                for item in usage_rows
            ),
            facts=json.loads(facts_row["payload"]) if facts_row else {},
            checkpoints=tuple(
                StoredCheckpoint(
                    id=item["id"],
                    name=item["name"],
                    message_count=item["message_count"],
                    created_at=self._datetime(item["created_at"]),
                )
                for item in checkpoint_rows
            ),
        )

    def create(
        self,
        *,
        strategy: ContextStrategyName = "sliding_window",
        window_size: int = 6,
    ) -> StoredSession:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO chat_sessions (
                    id, title, created_at, updated_at, strategy, window_size,
                    parent_session_id, checkpoint_id, branch_name
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'main')""",
                (
                    session_id,
                    "Новый чат",
                    self._timestamp(now),
                    self._timestamp(now),
                    strategy,
                    window_size,
                ),
            )
        return self.get(session_id)

    def list(self) -> list[StoredSession]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM chat_sessions ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
            return [self._load_session(connection, row["id"]) for row in rows]

    def get(self, session_id: str) -> StoredSession:
        with self._connection() as connection:
            return self._load_session(connection, session_id)

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chat_sessions")

    def update_strategy(
        self,
        session_id: str,
        strategy: ContextStrategyName,
        window_size: int,
    ) -> StoredSession:
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE chat_sessions SET strategy = ?, window_size = ? WHERE id = ?",
                (strategy, window_size, session_id),
            )
            if updated.rowcount == 0:
                raise ChatSessionNotFound(session_id)
        return self.get(session_id)

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        token_metrics: AgentTokenMetrics,
        facts: dict[str, str] | None = None,
    ) -> StoredSession:
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            position = len(session.messages)
            title = session.title
            if not session.messages:
                title = user_content.replace("\n", " ").strip()[:60] or "Новый чат"
            connection.executemany(
                """INSERT INTO chat_messages
                   (id, session_id, position, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (str(uuid4()), session_id, position, "user", user_content, timestamp),
                    (
                        str(uuid4()), session_id, position + 1, "assistant",
                        assistant_content, timestamp,
                    ),
                ],
            )
            connection.execute(
                "INSERT INTO chat_turn_usage (session_id, turn, payload) VALUES (?, ?, ?)",
                (
                    session_id,
                    len(session.token_metrics) + 1,
                    json.dumps(asdict(token_metrics), ensure_ascii=False),
                ),
            )
            if facts is not None:
                connection.execute(
                    """INSERT INTO chat_facts (session_id, payload, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         payload = excluded.payload, updated_at = excluded.updated_at""",
                    (session_id, json.dumps(facts, ensure_ascii=False), timestamp),
                )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, session_id),
            )
        return self.get(session_id)

    def create_checkpoint(self, session_id: str, name: str) -> StoredCheckpoint:
        checkpoint_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            session = self._load_session(connection, session_id)
            if not session.messages:
                raise ValueError("Нельзя создать checkpoint в пустом диалоге")
            connection.execute(
                """INSERT INTO chat_checkpoints
                   (id, session_id, name, message_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    session_id,
                    name.strip(),
                    len(session.messages),
                    self._timestamp(now),
                ),
            )
        return StoredCheckpoint(checkpoint_id, name.strip(), len(session.messages), now)

    def fork(self, session_id: str, checkpoint_id: str, name: str) -> StoredSession:
        branch_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            source = self._load_session(connection, session_id)
            checkpoint = connection.execute(
                """SELECT id, message_count FROM chat_checkpoints
                   WHERE id = ? AND session_id = ?""",
                (checkpoint_id, session_id),
            ).fetchone()
            if checkpoint is None:
                raise ChatCheckpointNotFound(checkpoint_id)
            message_count = int(checkpoint["message_count"])
            connection.execute(
                """INSERT INTO chat_sessions (
                    id, title, created_at, updated_at, strategy, window_size,
                    parent_session_id, checkpoint_id, branch_name
                ) VALUES (?, ?, ?, ?, 'branching', ?, ?, ?, ?)""",
                (
                    branch_id,
                    f"{source.title} · {name.strip()}",
                    timestamp,
                    timestamp,
                    source.window_size,
                    session_id,
                    checkpoint_id,
                    name.strip(),
                ),
            )
            for message in source.messages[:message_count]:
                connection.execute(
                    """INSERT INTO chat_messages
                       (id, session_id, position, role, content, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), branch_id, message.position, message.role,
                        message.content, self._timestamp(message.created_at),
                    ),
                )
            turn_count = message_count // 2
            for turn, metrics in enumerate(source.token_metrics[:turn_count], start=1):
                connection.execute(
                    "INSERT INTO chat_turn_usage (session_id, turn, payload) VALUES (?, ?, ?)",
                    (branch_id, turn, json.dumps(asdict(metrics), ensure_ascii=False)),
                )
            if source.facts:
                connection.execute(
                    "INSERT INTO chat_facts (session_id, payload, updated_at) VALUES (?, ?, ?)",
                    (branch_id, json.dumps(source.facts, ensure_ascii=False), timestamp),
                )
        return self.get(branch_id)


class ChatSessionService:
    def __init__(
        self,
        repository: ChatSessionRepository,
        agent: Agent,
        *,
        facts_extractor: FactsExtractor | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._repository = repository
        self._agent = agent
        self._facts_extractor = facts_extractor
        self._token_counter = token_counter

    @staticmethod
    def _summary(session: StoredSession) -> ChatSessionSummary:
        return ChatSessionSummary(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            strategy=session.strategy,
            window_size=session.window_size,
            parent_session_id=session.parent_session_id,
            checkpoint_id=session.checkpoint_id,
            branch_name=session.branch_name,
        )

    @staticmethod
    def _message(message: StoredMessage) -> ChatMessage:
        return ChatMessage(
            id=message.id,
            position=message.position,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
        )

    @staticmethod
    def _checkpoint(item: StoredCheckpoint) -> ChatCheckpoint:
        return ChatCheckpoint(
            id=item.id,
            name=item.name,
            message_count=item.message_count,
            created_at=item.created_at,
        )

    @staticmethod
    def _turn_usage(metrics: AgentTokenMetrics, turn: int) -> ChatTurnTokenUsage:
        return ChatTurnTokenUsage(turn=turn, **asdict(metrics))

    @classmethod
    def _token_usage(cls, session: StoredSession) -> ChatSessionTokenUsage:
        turns = [
            cls._turn_usage(metrics, turn)
            for turn, metrics in enumerate(session.token_metrics, start=1)
        ]
        return ChatSessionTokenUsage(
            turns=turns,
            prompt_tokens=sum(item.prompt_tokens for item in turns),
            completion_tokens=sum(item.completion_tokens for item in turns),
            total_tokens=sum(item.total_tokens + item.memory_total_tokens for item in turns),
            estimated_cost_usd=sum(
                (item.estimated_cost_usd or 0.0)
                + (item.memory_estimated_cost_usd or 0.0)
                for item in turns
            ),
            memory_prompt_tokens=sum(item.memory_prompt_tokens for item in turns),
            memory_completion_tokens=sum(item.memory_completion_tokens for item in turns),
            memory_total_tokens=sum(item.memory_total_tokens for item in turns),
            memory_estimated_cost_usd=sum(
                item.memory_estimated_cost_usd or 0.0 for item in turns
            ),
        )

    def _session(self, session: StoredSession) -> ChatSession:
        return ChatSession(
            **self._summary(session).model_dump(),
            messages=[self._message(item) for item in session.messages],
            facts=session.facts or {},
            checkpoints=[self._checkpoint(item) for item in session.checkpoints],
            token_usage=self._token_usage(session),
        )

    def create(
        self,
        *,
        strategy: ContextStrategyName = "sliding_window",
        window_size: int = 6,
    ) -> ChatSession:
        return self._session(
            self._repository.create(strategy=strategy, window_size=window_size)
        )

    def list(self) -> list[ChatSessionSummary]:
        return [self._summary(session) for session in self._repository.list()]

    def get(self, session_id: str) -> ChatSession:
        return self._session(self._repository.get(session_id))

    def clear(self) -> None:
        self._repository.clear()

    def update_strategy(
        self,
        session_id: str,
        strategy: ContextStrategyName,
        window_size: int,
    ) -> ChatSession:
        return self._session(
            self._repository.update_strategy(session_id, strategy, window_size)
        )

    def create_checkpoint(self, session_id: str, name: str) -> ChatCheckpoint:
        return self._checkpoint(self._repository.create_checkpoint(session_id, name))

    def fork(self, session_id: str, checkpoint_id: str, name: str) -> ChatSession:
        return self._session(self._repository.fork(session_id, checkpoint_id, name))

    def send(self, session_id: str, content: str) -> ChatSendResponse:
        session = self._repository.get(session_id)
        history: list[AgentMessage] = [
            {"role": message.role, "content": message.content}  # type: ignore[typeddict-item]
            for message in session.messages
        ]
        facts = dict(session.facts or {})
        facts_update = None
        if session.strategy == "sticky_facts":
            if self._facts_extractor is None:
                raise RuntimeError("Извлечение facts не настроено")
            facts_update = self._facts_extractor.update(facts, content.strip())
            facts = facts_update.facts

        prepared = strategy_for(session.strategy, session.window_size).prepare(
            history,
            facts,
        )
        history_tokens = (
            sum(
                self._token_counter.count_text(item["content"])
                + MESSAGE_OVERHEAD_TOKENS
                for item in history
            )
            if self._token_counter
            else None
        )
        result = self._agent.respond(
            prepared.messages,
            content,
            memory_block=prepared.memory_block,
            total_history_tokens=history_tokens,
            strategy_dropped_messages=prepared.dropped_messages,
        )
        metrics = result.metrics
        if facts_update is not None:
            metrics = replace(
                metrics,
                memory_prompt_tokens=facts_update.usage.prompt_tokens,
                memory_completion_tokens=facts_update.usage.completion_tokens,
                memory_total_tokens=facts_update.usage.total_tokens,
                memory_estimated_cost_usd=facts_update.estimated_cost_usd,
            )
        updated = self._repository.append_exchange(
            session_id,
            content.strip(),
            result.content,
            metrics,
            facts if session.strategy == "sticky_facts" else None,
        )
        return ChatSendResponse(
            session=self._summary(updated),
            user_message=self._message(updated.messages[-2]),
            assistant_message=self._message(updated.messages[-1]),
            token_usage=self._turn_usage(metrics, len(updated.token_metrics)),
        )
