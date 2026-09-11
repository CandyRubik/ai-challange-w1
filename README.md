# Rubik Context Lab

Лаборатория трёх стратегий управления контекстом чат-агента — без summary.
Интерфейс разделён на три независимые вкладки, каждая со своими диалогами и
инструментами:

- **Sliding Window** — отправляет последние `N` сообщений и отбрасывает остальные;
- **Sticky Facts** — после каждого сообщения пользователя обновляет отдельную
  key-value память и отправляет `facts + последние N сообщений`;
- **Branching** — сохраняет checkpoint, создаёт от него независимые ветки и
  позволяет продолжать каждую как отдельный диалог.

Полная история всегда остаётся в SQLite и видна пользователю. Стратегии меняют
только контекст, отправляемый модели.

Новый диалог создаётся сразу внутри активной вкладки. Существующий диалог нельзя
случайно перевести в другую стратегию: при переключении вкладки открывается
последний диалог этой стратегии либо создаётся новый.

## Branching

1. Откройте вкладку `Branching`.
2. Соберите общую часть диалога.
3. Сохраните checkpoint.
4. Дважды выберите этот checkpoint и создайте, например, ветки `MVP` и `Pro`.
5. Переключайтесь между ними через список слева.

Каждая ветка получает копию сообщений только до checkpoint. Последующие
сообщения и метрики изолированы.

## Запуск

Требуется Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Добавьте `DEEPSEEK_API_KEY` в `.env`, затем:

```bash
set -a
source .env
set +a
uvicorn app.main:app --reload --port 8000
```

Откройте <http://127.0.0.1:8000>. База по умолчанию находится в
`data/chat.sqlite3`; путь меняется через `CHAT_DB_PATH`.

## API

- `POST /api/chat/sessions` — создать диалог;
- `GET /api/chat/sessions` — список диалогов и веток;
- `GET /api/chat/sessions/{id}` — история, facts, checkpoint и метрики;
- `PUT /api/chat/sessions/{id}/strategy` — переключить стратегию и `N`;
- `POST /api/chat/sessions/{id}/messages` — отправить сообщение;
- `POST /api/chat/sessions/{id}/checkpoints` — сохранить checkpoint;
- `POST /api/chat/sessions/{id}/branches` — создать ветку;
- `DELETE /api/chat/sessions` — очистить историю.

## Проверки

```bash
python -m pytest -q
python -m compileall -q app tests
node --check static/app.js
```
