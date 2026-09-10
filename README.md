# RubikStudyHarness

Небольшой standalone-бейзлайн изолированного чата с агентом. Проект переносит
общую чат-инфраструктуру FlowScout без его продуктовой функциональности:
ролевого анализа, multi-agent judge, workflow-интеграций и n8n-пайплайна здесь
нет.

## Что внутри

- отдельные чат-сессии с независимым контекстом;
- SQLite-хранилище истории, переживающее перезапуск backend;
- границы input/output policy агента: ограничение размера истории, сообщения и
  ответа модели;
- runtime-настройки модели, thinking, передачи истории и system prompt;
- изолированный DeepSeek provider с повтором запроса без thinking, если первый
  ответ оказался пустым;
- компактный web-интерфейс и CI-проверки backend/frontend.

Токен-эксперименты не входят в этот чистый baseline и оформляются отдельно.

## Запуск

Требуется Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Заполните `DEEPSEEK_API_KEY` в `.env`, затем экспортируйте переменные и
запустите сервер:

```bash
set -a
source .env
set +a
uvicorn app.main:app --reload --port 8000
```

Откройте <http://127.0.0.1:8000>. По умолчанию история хранится в
`data/chat.sqlite3`; путь можно изменить через `CHAT_DB_PATH`.

## API

- `GET /api/health` — состояние backend и наличие ключа DeepSeek;
- `GET/PUT /api/debug/settings` — process-local настройки следующих сообщений;
- `POST /api/chat/sessions` — создать чат;
- `GET /api/chat/sessions` — список чатов;
- `DELETE /api/chat/sessions` — удалить все чаты и сообщения;
- `GET /api/chat/sessions/{id}` — получить историю чата;
- `POST /api/chat/sessions/{id}/messages` — отправить сообщение агенту.

Архитектура разделяет HTTP API, `ChatSessionService`, SQLite repository,
storage-agnostic `Agent` и vendor-specific `DeepSeekProvider`:

```text
HTTP API → ChatSessionService → ChatSessionRepository → SQLite
                         └──→ Agent → LanguageModel → DeepSeek API
```

Настройки применяются только к новым запросам и не записываются в историю.
API-ключ хранится только на backend.

## Проверки

```bash
python -m pytest -q
python -m compileall -q app tests
node --check static/app.js
```
