from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Protocol
from uuid import uuid4

from ..agents.agent import Agent, AgentMessage
from ..schemas import (
    ChatContextSummary,
    ChatContextCompaction,
    ChatMessage,
    ChatSendResponse,
    ChatSession,
    ChatSessionSummary,
    ChatSessionTokenUsage,
    ChatTurnTokenUsage,
)
from ..token_usage import AgentTokenMetrics, TokenCounter
from .context_compression import ConversationSummarizer, SummaryResult


DEFAULT_CHAT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"
DEFAULT_DB_PATH = DEFAULT_CHAT_DB_PATH


class ChatSessionNotFound(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    position: int
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredContextSummary:
    content: str = ""
    summarized_message_count: int = 0
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StoredContextCompaction:
    id: str
    summary: str
    source_start_position: int
    source_end_position: int
    summarized_message_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[StoredMessage, ...] = ()
    token_metrics: tuple[AgentTokenMetrics, ...] = ()
    compactions: tuple[StoredContextCompaction, ...] = ()
    context_summary: StoredContextSummary = StoredContextSummary()


class ChatSessionRepository(Protocol):
    def create(self) -> StoredSession: ...

    def list(self) -> list[StoredSession]: ...

    def get(self, session_id: str) -> StoredSession: ...

    def clear(self) -> None: ...

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        token_metrics: AgentTokenMetrics,
        context_summary: StoredContextSummary | None = None,
        compactions: Sequence[StoredContextCompaction] = (),
    ) -> StoredSession: ...


class SQLiteChatSessionRepository:
    """Durable chat history isolated behind a repository boundary."""

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

                CREATE TABLE IF NOT EXISTS chat_context_summaries (
                    session_id TEXT PRIMARY KEY
                        REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    summarized_message_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_context_compactions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL
                        REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    source_start_position INTEGER NOT NULL,
                    source_end_position INTEGER NOT NULL,
                    summarized_message_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, summarized_message_count)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_context_compactions_session
                ON chat_context_compactions(session_id, source_end_position);

                INSERT OR IGNORE INTO chat_context_compactions (
                    id,
                    session_id,
                    summary,
                    source_start_position,
                    source_end_position,
                    summarized_message_count,
                    created_at
                )
                SELECT
                    'legacy:' || session_id || ':' || summarized_message_count,
                    session_id,
                    content,
                    0,
                    summarized_message_count - 1,
                    summarized_message_count,
                    updated_at
                FROM chat_context_summaries
                WHERE summarized_message_count > 0;
                """,
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
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ChatSessionNotFound(session_id)

        message_rows = connection.execute(
            """
            SELECT id, position, role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        messages = tuple(
            StoredMessage(
                id=message["id"],
                position=message["position"],
                role=message["role"],
                content=message["content"],
                created_at=self._datetime(message["created_at"]),
            )
            for message in message_rows
        )
        usage_rows = connection.execute(
            """
            SELECT payload
            FROM chat_turn_usage
            WHERE session_id = ?
            ORDER BY turn
            """,
            (session_id,),
        ).fetchall()
        token_metrics = tuple(
            AgentTokenMetrics(**json.loads(usage["payload"])) for usage in usage_rows
        )
        summary_row = connection.execute(
            """
            SELECT content, summarized_message_count, updated_at
            FROM chat_context_summaries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        context_summary = (
            StoredContextSummary(
                content=summary_row["content"],
                summarized_message_count=summary_row["summarized_message_count"],
                updated_at=self._datetime(summary_row["updated_at"]),
            )
            if summary_row is not None
            else StoredContextSummary()
        )
        compaction_rows = connection.execute(
            """
            SELECT
                id,
                summary,
                source_start_position,
                source_end_position,
                summarized_message_count,
                created_at
            FROM chat_context_compactions
            WHERE session_id = ?
            ORDER BY source_end_position
            """,
            (session_id,),
        ).fetchall()
        compactions = tuple(
            StoredContextCompaction(
                id=compaction["id"],
                summary=compaction["summary"],
                source_start_position=compaction["source_start_position"],
                source_end_position=compaction["source_end_position"],
                summarized_message_count=compaction["summarized_message_count"],
                created_at=self._datetime(compaction["created_at"]),
            )
            for compaction in compaction_rows
        )
        return StoredSession(
            id=row["id"],
            title=row["title"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            messages=messages,
            token_metrics=token_metrics,
            compactions=compactions,
            context_summary=context_summary,
        )

    def create(self) -> StoredSession:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, "Новый чат", self._timestamp(now), self._timestamp(now)),
            )
        return StoredSession(session_id, "Новый чат", now, now)

    def list(self) -> list[StoredSession]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT 100
                """,
            ).fetchall()
            return [self._load_session(connection, row["id"]) for row in rows]

    def get(self, session_id: str) -> StoredSession:
        with self._connection() as connection:
            return self._load_session(connection, session_id)

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chat_sessions")

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        token_metrics: AgentTokenMetrics,
        context_summary: StoredContextSummary | None = None,
        compactions: Sequence[StoredContextCompaction] = (),
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
                """
                INSERT INTO chat_messages
                    (id, session_id, position, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (str(uuid4()), session_id, position, "user", user_content, timestamp),
                    (
                        str(uuid4()), session_id, position + 1, "assistant",
                        assistant_content, timestamp,
                    ),
                ],
            )
            connection.execute(
                """
                INSERT INTO chat_turn_usage (session_id, turn, payload)
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    len(session.token_metrics) + 1,
                    json.dumps(asdict(token_metrics), ensure_ascii=False),
                ),
            )
            if context_summary is not None:
                summary_updated_at = context_summary.updated_at or now
                connection.execute(
                    """
                    INSERT INTO chat_context_summaries
                        (session_id, content, summarized_message_count, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        content = excluded.content,
                        summarized_message_count = excluded.summarized_message_count,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        context_summary.content,
                        context_summary.summarized_message_count,
                        self._timestamp(summary_updated_at),
                    ),
                )
            if compactions:
                connection.executemany(
                    """
                    INSERT INTO chat_context_compactions (
                        id,
                        session_id,
                        summary,
                        source_start_position,
                        source_end_position,
                        summarized_message_count,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            compaction.id,
                            session_id,
                            compaction.summary,
                            compaction.source_start_position,
                            compaction.source_end_position,
                            compaction.summarized_message_count,
                            self._timestamp(compaction.created_at),
                        )
                        for compaction in compactions
                    ],
                )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, session_id),
            )

        return self.get(session_id)


class ChatSessionService:
    """Adapt persistent sessions to the storage-agnostic Agent."""

    def __init__(
        self,
        repository: ChatSessionRepository,
        agent: Agent,
        *,
        summarizer: ConversationSummarizer | None = None,
        token_counter: TokenCounter | None = None,
        summary_batch_messages: int = 10,
    ) -> None:
        self._repository = repository
        self._agent = agent
        self._summarizer = summarizer
        self._token_counter = token_counter
        self._summary_batch_messages = summary_batch_messages

    @staticmethod
    def _summary(session: StoredSession) -> ChatSessionSummary:
        return ChatSessionSummary(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
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
    def _compaction(compaction: StoredContextCompaction) -> ChatContextCompaction:
        return ChatContextCompaction(
            id=compaction.id,
            summary=compaction.summary,
            source_start_position=compaction.source_start_position,
            source_end_position=compaction.source_end_position,
            summarized_message_count=compaction.summarized_message_count,
            created_at=compaction.created_at,
        )

    def _context_summary(self, session: StoredSession) -> ChatContextSummary:
        summary = session.context_summary
        return ChatContextSummary(
            content=summary.content,
            summarized_message_count=summary.summarized_message_count,
            summary_tokens=(
                self._token_counter.count_text(summary.content)
                if self._token_counter and summary.content
                else 0
            ),
            updated_at=summary.updated_at,
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
            prompt_tokens=sum(turn.prompt_tokens for turn in turns),
            completion_tokens=sum(turn.completion_tokens for turn in turns),
            total_tokens=sum(
                turn.total_tokens + turn.summary_total_tokens for turn in turns
            ),
            estimated_cost_usd=sum(
                (turn.estimated_cost_usd or 0.0)
                + (turn.summary_estimated_cost_usd or 0.0)
                for turn in turns
            ),
            summary_prompt_tokens=sum(turn.summary_prompt_tokens for turn in turns),
            summary_completion_tokens=sum(
                turn.summary_completion_tokens for turn in turns
            ),
            summary_total_tokens=sum(turn.summary_total_tokens for turn in turns),
            summary_estimated_cost_usd=sum(
                turn.summary_estimated_cost_usd or 0.0 for turn in turns
            ),
        )

    def create(self) -> ChatSession:
        session = self._repository.create()
        return ChatSession(
            **self._summary(session).model_dump(),
            messages=[],
            compactions=[],
            context_summary=ChatContextSummary(),
            token_usage=ChatSessionTokenUsage(),
        )

    def list(self) -> list[ChatSessionSummary]:
        return [self._summary(session) for session in self._repository.list()]

    def get(self, session_id: str) -> ChatSession:
        session = self._repository.get(session_id)
        return ChatSession(
            **self._summary(session).model_dump(),
            messages=[self._message(message) for message in session.messages],
            compactions=[
                self._compaction(compaction) for compaction in session.compactions
            ],
            context_summary=self._context_summary(session),
            token_usage=self._token_usage(session),
        )

    def clear(self) -> None:
        self._repository.clear()

    def send(self, session_id: str, content: str) -> ChatSendResponse:
        session = self._repository.get(session_id)
        context_summary = session.context_summary
        summarized_count = min(
            context_summary.summarized_message_count,
            len(session.messages),
        )
        context: list[AgentMessage] = [
            {"role": message.role, "content": message.content}
            for message in session.messages[summarized_count:]
        ]
        result = self._agent.respond(
            context,
            content,
            context_summary=context_summary.content,
        )
        metrics = result.metrics
        summary_usage = None
        summary_cost = 0.0
        compacted = 0
        compactions: list[StoredContextCompaction] = []
        pending: list[AgentMessage] = [
            *context,
            {"role": "user", "content": content.strip()},
            {"role": "assistant", "content": result.content},
        ]
        next_summary = context_summary.content
        while len(pending) >= self._summary_batch_messages:
            if self._summarizer is None or self._token_counter is None:
                raise RuntimeError("Компрессия контекста не настроена")
            summary_result: SummaryResult = self._summarizer.summarize(
                next_summary,
                pending[:self._summary_batch_messages],
            )
            next_summary = summary_result.content
            summary_usage = (
                summary_result.usage
                if summary_usage is None
                else summary_usage + summary_result.usage
            )
            summary_cost += summary_result.estimated_cost_usd or 0.0
            del pending[:self._summary_batch_messages]
            source_start_position = summarized_count
            summarized_count += self._summary_batch_messages
            compacted += self._summary_batch_messages
            compactions.append(
                StoredContextCompaction(
                    id=str(uuid4()),
                    summary=next_summary,
                    source_start_position=source_start_position,
                    source_end_position=summarized_count - 1,
                    summarized_message_count=summarized_count,
                    created_at=datetime.now(timezone.utc),
                )
            )
        if compacted:
            context_summary = StoredContextSummary(
                content=next_summary,
                summarized_message_count=summarized_count,
                updated_at=datetime.now(timezone.utc),
            )

        if summary_usage is not None:
            metrics = replace(
                metrics,
                summary_prompt_tokens=summary_usage.prompt_tokens,
                summary_completion_tokens=summary_usage.completion_tokens,
                summary_total_tokens=summary_usage.total_tokens,
                summary_tokens=self._token_counter.count_text(context_summary.content),
                summary_estimated_cost_usd=summary_cost,
                compressed_messages=summarized_count,
            )
        else:
            metrics = replace(
                metrics,
                summary_tokens=(
                    self._token_counter.count_text(context_summary.content)
                    if context_summary.content
                    else 0
                ),
                compressed_messages=summarized_count,
            )
        updated = self._repository.append_exchange(
            session_id,
            content.strip(),
            result.content,
            metrics,
            context_summary,
            compactions,
        )
        return ChatSendResponse(
            session=self._summary(updated),
            user_message=self._message(updated.messages[-2]),
            assistant_message=self._message(updated.messages[-1]),
            token_usage=self._turn_usage(metrics, len(updated.token_metrics)),
        )
