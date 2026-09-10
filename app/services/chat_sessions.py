from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from ..agents.agent import Agent, AgentMessage
from ..schemas import (
    ChatMessage,
    ChatSendResponse,
    ChatSession,
    ChatSessionSummary,
    ChatSessionTokenUsage,
    ChatTurnTokenUsage,
)
from ..token_usage import AgentTokenMetrics


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"


class ChatSessionNotFound(LookupError):
    pass


class SQLiteChatSessionRepository:
    def __init__(self, database_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, position)
                );
                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    turn INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, turn)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create(self) -> str:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (session_id, "Новый чат", now, now),
            )
        return session_id

    def list_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions")

    def load(self, session_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise ChatSessionNotFound(session_id)
            messages = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
            turns = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn",
                (session_id,),
            ).fetchall()
        return session, messages, turns

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        metrics: AgentTokenMetrics,
    ) -> None:
        session, messages, _ = self.load(session_id)
        now = datetime.now(timezone.utc).isoformat()
        position = len(messages)
        title = session["title"] if messages else user_content.replace("\n", " ")[:60]
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (str(uuid4()), session_id, position, "user", user_content, now),
                    (
                        str(uuid4()), session_id, position + 1, "assistant",
                        assistant_content, now,
                    ),
                ],
            )
            connection.execute(
                "INSERT INTO turns VALUES (?, ?, ?)",
                (session_id, position // 2 + 1, json.dumps(asdict(metrics))),
            )
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title or "Новый чат", now, session_id),
            )


class ChatSessionService:
    def __init__(self, repository: SQLiteChatSessionRepository, agent: Agent) -> None:
        self._repository = repository
        self._agent = agent

    @staticmethod
    def _summary(row: sqlite3.Row) -> ChatSessionSummary:
        return ChatSessionSummary(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _session(self, session_id: str) -> ChatSession:
        row, message_rows, turn_rows = self._repository.load(session_id)
        messages = [
            ChatMessage(
                id=item["id"],
                role=item["role"],
                content=item["content"],
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            for item in message_rows
        ]
        turns = [
            ChatTurnTokenUsage(turn=item["turn"], **json.loads(item["metrics_json"]))
            for item in turn_rows
        ]
        return ChatSession(
            **self._summary(row).model_dump(),
            messages=messages,
            token_usage=ChatSessionTokenUsage(
                turns=turns,
                prompt_tokens=sum(item.prompt_tokens for item in turns),
                completion_tokens=sum(item.completion_tokens for item in turns),
                total_tokens=sum(item.total_tokens for item in turns),
                estimated_cost_usd=sum(item.estimated_cost_usd or 0 for item in turns),
            ),
        )

    def create(self) -> ChatSession:
        return self._session(self._repository.create())

    def list(self) -> list[ChatSessionSummary]:
        return [
            ChatSessionSummary(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            for item in self._repository.list_ids()
            for session in [self._session(item)]
        ]

    def get(self, session_id: str) -> ChatSession:
        return self._session(session_id)

    def clear(self) -> None:
        self._repository.clear()

    def send(self, session_id: str, content: str) -> ChatSendResponse:
        before = self._session(session_id)
        context: list[AgentMessage] = [
            {"role": item.role, "content": item.content} for item in before.messages
        ]
        result = self._agent.respond(context, content)
        self._repository.append_exchange(session_id, content, result.content, result.metrics)
        after = self._session(session_id)
        return ChatSendResponse(
            session=ChatSessionSummary(
                id=after.id,
                title=after.title,
                created_at=after.created_at,
                updated_at=after.updated_at,
            ),
            user_message=after.messages[-2],
            assistant_message=after.messages[-1],
            token_usage=after.token_usage.turns[-1],
        )
