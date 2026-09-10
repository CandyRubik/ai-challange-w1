from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from .agents.agent import Agent, AgentContextOverflow, AgentInputError, AgentOutputError
from .providers.deepseek import DeepSeekProvider, LlmConfigurationError, LlmRequestError
from .schemas import (
    ChatSendRequest,
    ChatSendResponse,
    ChatSession,
    ChatSessionSummary,
    ChatSettings,
    TokenExperimentRequest,
    TokenExperimentResponse,
)
from .services.chat_sessions import (
    ChatSessionNotFound,
    ChatSessionService,
    DEFAULT_DB_PATH,
    SQLiteChatSessionRepository,
)
from .services.settings import SettingsStore
from .services.token_experiments import HaystackGenerator, TokenExperimentService
from .tokenizer import DeepSeekTokenCounter, TokenizerSetupError


logger = logging.getLogger(__name__)
app = FastAPI(title="Rubik Study Harness")
settings_store = SettingsStore()
token_counter = DeepSeekTokenCounter()


@lru_cache(maxsize=1)
def repository() -> SQLiteChatSessionRepository:
    return SQLiteChatSessionRepository(os.getenv("CHAT_DB_PATH", str(DEFAULT_DB_PATH)))


def build_agent() -> Agent:
    settings = settings_store.get()
    return Agent(
        DeepSeekProvider(
            model=settings.model,
            thinking_enabled=settings.thinking_enabled,
        ),
        token_counter,
        system_prompt=settings.system_prompt,
        max_tokens=settings.max_tokens,
        context_enabled=settings.history_enabled,
        context_limit_tokens=settings.context_limit_tokens,
        overflow_strategy=settings.overflow_strategy,
    )


def chat_service() -> ChatSessionService:
    return ChatSessionService(repository(), build_agent())


def experiment_service() -> TokenExperimentService:
    return TokenExperimentService(
        build_agent(),
        HaystackGenerator(token_counter),
        token_counter,
    )


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.get("/api/settings", response_model=ChatSettings)
def get_settings() -> ChatSettings:
    return settings_store.get()


@app.put("/api/settings", response_model=ChatSettings)
def update_settings(settings: ChatSettings) -> ChatSettings:
    return settings_store.replace(settings)


@app.post("/api/experiments/needle", response_model=TokenExperimentResponse)
def run_experiment(
    request: TokenExperimentRequest,
    service: TokenExperimentService = Depends(experiment_service),
) -> TokenExperimentResponse:
    try:
        return service.run(
            target_tokens=request.target_tokens,
            target_bytes=request.target_bytes,
            needle_position=request.needle_position,
        )
    except TokenizerSetupError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except LlmConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except LlmRequestError:
        logger.exception("DeepSeek experiment failed")
        raise HTTPException(status_code=502, detail="Запрос к модели завершился ошибкой") from None


@app.post("/api/chat/sessions", response_model=ChatSession, status_code=201)
def create_session(service: ChatSessionService = Depends(chat_service)) -> ChatSession:
    return service.create()


@app.get("/api/chat/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    service: ChatSessionService = Depends(chat_service),
) -> list[ChatSessionSummary]:
    return service.list()


@app.delete("/api/chat/sessions", status_code=204)
def clear_sessions(service: ChatSessionService = Depends(chat_service)) -> Response:
    service.clear()
    return Response(status_code=204)


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
def get_session(
    session_id: str,
    service: ChatSessionService = Depends(chat_service),
) -> ChatSession:
    try:
        return service.get(session_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None


@app.post(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatSendResponse,
)
def send_message(
    session_id: str,
    request: ChatSendRequest,
    service: ChatSessionService = Depends(chat_service),
) -> ChatSendResponse:
    try:
        return service.send(session_id, request.content)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except AgentContextOverflow as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(error),
                "code": "context_overflow",
                "prompt_tokens": error.prompt_tokens,
                "reserved_output_tokens": error.reserved_output_tokens,
                "context_limit_tokens": error.context_limit_tokens,
                "overflow_tokens": error.overflow_tokens,
            },
        ) from None
    except (AgentInputError, TokenizerSetupError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except LlmConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except (AgentOutputError, LlmRequestError):
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail="Запрос к модели завершился ошибкой") from None


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")

