# RubikStudyHarness

Standalone-лаборатория изолированного чата с агентом. Проект переносит общую
чат-инфраструктуру FlowScout без его продуктовой функциональности:
ролевого анализа, multi-agent judge, workflow-интеграций и n8n-пайплайна здесь
нет.

## День 8: работа с токенами

Каждый вызов агента показывает:

- токены текущего сообщения и всей сохранённой истории — локальная оценка
  официальным DeepSeek tokenizer;
- оценку полного prompt до запроса и фактические `prompt_tokens` после него;
- токены ответа, cache hit/miss, reasoning tokens и `finish_reason` из API;
- накопительный расход и примерную стоимость всей сессии;
- заполнение контекстного окна и количество удалённых сообщений при стратегии
  `trim`.

Отдельная страница <http://127.0.0.1:8000/benchmark.html> заранее показывает
точные prompt и только после подтверждения запускает benchmark: один короткий и
три связанных запроса длинного диалога действительно отправляются в DeepSeek.
Сценарий переполнения показывает полный отклонённый prompt: агент блокирует его
**до вызова API**, поэтому ответ и новое списание не возникают. Для обычных
чатов доступны стратегии `reject` и `trim`.

Локальный подсчёт немного отличается от серверного chat template, поэтому для
биллинга источником истины остаётся `usage` ответа API. Контекст DeepSeek V4 —
1M токенов, максимальный output — 384K. Цены в коде сверены 10 сентября 2026
с [официальной таблицей DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/).

Короткое видео с интерфейсом и результатами сценариев:
[day-8-token-demo.mp4](artifacts/day-8-token-demo.mp4).

## Что внутри

- отдельные чат-сессии с независимым контекстом;
- SQLite-хранилище истории, переживающее перезапуск backend;
- token-aware input policy с проверкой контекстного лимита до вызова модели;
- сохранение метрик каждого хода рядом с историей в SQLite;
- runtime-настройки модели, thinking, передачи истории и system prompt;
- изолированный DeepSeek provider с повтором запроса без thinking, если первый
  ответ оказался пустым;
- dashboard роста токенов и отдельная benchmark-страница с реальными запросами;
- компактный web-интерфейс и CI-проверки backend/frontend.

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

При первом подсчёте приложение скачает опубликованный DeepSeek
`tokenizer.json` в `data/`. Этот файл и база данных не попадают в Git.

## API

- `GET /api/health` — состояние backend и наличие ключа DeepSeek;
- `GET/PUT /api/debug/settings` — process-local настройки следующих сообщений;
- `GET /api/benchmark/plan` — увидеть все prompt до платного запуска;
- `POST /api/benchmark/run` — выполнить short/long и проверить overflow;
- `GET /api/benchmark/latest` — загрузить последний результат процесса;
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

Настройки применяются только к новым запросам. API-ключ хранится только на
backend. Метрики завершённых ходов сохраняются; переполненный запрос в историю
не добавляется.

## Проверки

```bash
python -m pytest -q
python -m compileall -q app tests
node --check static/app.js
```
