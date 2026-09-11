# Rubik Chat

Минимальный чат с агентом, SQLite-историей, inline-метриками токенов и
автоматической компактизацией контекста.

## Как работает компактизация

После каждых 10 несжатых сообщений — пяти пар `user/assistant` — агент создаёт
новое накопительное summary. В чате исходный диалог остаётся видимым, а после
свёрнутой группы появляется служебная отметка:

```text
✦ Контекст скомпактизирован · сообщения 1–10 заменены summary для LLM
```

В SQLite отдельно хранятся полный диалог и события компактизации с диапазоном
позиций исходных сообщений. Когда новый хвост снова достигает 10 сообщений, он
объединяется с предыдущим summary. В следующий запрос модели уходят накопительное
summary и только несжатый хвост; полная история доступна интерфейсу и API.

У каждого ответа в чате показываются фактические `input`, `output`, суммарные
токены и стоимость. Если на этом ходу выполнялась компактизация, её input/output
и стоимость также включаются в метрики.

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

Откройте <http://127.0.0.1:8000>.

По умолчанию база находится в `data/chat.sqlite3`. Путь можно изменить через
`CHAT_DB_PATH`.

## API

- `GET /api/health` — состояние backend;
- `POST /api/chat/sessions` — создать чат;
- `GET /api/chat/sessions` — список чатов;
- `DELETE /api/chat/sessions` — удалить все чаты;
- `GET /api/chat/sessions/{id}` — полный диалог, события компактизации и метрики;
- `POST /api/chat/sessions/{id}/messages` — отправить сообщение.

## Проверки

```bash
python -m pytest -q
python -m compileall -q app tests
node --check static/app.js
```
