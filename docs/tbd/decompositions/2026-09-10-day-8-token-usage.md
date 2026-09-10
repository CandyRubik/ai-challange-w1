# День 8: работа с токенами — PR Decomposition Map

- **Created:** 2026-09-10
- **Epic reference:** запрос пользователя в Codex
- **Trunk:** `main`
- **Size budgets:** target ≤600 reviewable lines, cap 1000

## Slices

| # | PR title | Purpose (one sentence) | Strategy | Size budget | Depends on | Status |
|---|----------|------------------------|----------|-------------|------------|--------|
| 1 | Token-aware agent core | Считать request/history/response usage, сохранять метрики и блокировать переполнение до API. | naturally safe | ~830 lines | — | in-progress |
| 2 | Token Lab UI and benchmark | Показать рост токенов в чате и вынести реальные short/long/overflow запросы на отдельную benchmark-страницу. | naturally safe | ~839 lines | 1 | in-progress |
| 3 | Live benchmark stream | Показывать отправку каждого запроса, фрагменты ответа и финальный API usage непосредственно во время DeepSeek-вызова. | naturally safe | ~785 lines | 2 | in-progress |
| 4 | Provider-side context overflow | Реально отправить oversized payload и показать исходную ошибку лимита, возвращённую DeepSeek. | naturally safe | ~475 lines | 3 | in-progress |

## Slice details

### Slice 1 — Token-aware agent core

- **In scope:** официальный локальный tokenizer; фактический DeepSeek API usage;
  расчёт стоимости; token-aware `Agent`; `reject`/`trim`; SQLite-миграция и
  сохранение метрик; расширенный chat API; backend-тесты.
- **Out of scope:** dashboard, benchmark-страница и видео.
- **Ships safely because:** API меняется аддитивно, существующий frontend
  игнорирует новые поля; настройки по умолчанию используют официальный лимит
  DeepSeek V4 в 1M токенов; CI остаётся зелёным.
- **Cleanup owed:** none.
- **Budget justification:** контракт usage проходит через provider, agent,
  storage и API одной атомарной вертикалью; разрыв между ними оставил бы
  несохраняемые или недостоверные метрики.

### Slice 2 — Token Lab UI and benchmark

- **In scope:** dashboard текущего хода и сессии; график роста; отдельная
  benchmark-страница, заранее показывающая точные prompt; реальные DeepSeek
  short/long вызовы; локально заблокированный overflow; README; MP4-демо.
- **Out of scope:** изменение токенизации или правил агента из slice 1.
- **Ships safely because:** после slice 1 все используемые API-поля доступны;
  benchmark запускается только явной кнопкой с подтверждением стоимости, а
  отдельная страница и её endpoints аддитивны.
- **Cleanup owed:** none.
- **Budget justification:** UI, benchmark endpoints и видео образуют одну
  проверяемую демонстрацию задания; разделение оставило бы скрытый endpoint или
  интерфейс без воспроизводимого сравнения.

### Slice 3 — Live benchmark stream

- **In scope:** streaming Chat Completions с финальным API usage; подготовка и
  завершение agent-вызова вокруг потока; NDJSON-события backend → browser;
  пошаговое обновление request/response; тесты потока; новая запись реального
  браузерного запуска.
- **Out of scope:** изменение сценариев и их prompt, обычный chat UI.
- **Ships safely because:** заменяет только транспорт benchmark endpoint;
  обычные чаты не затрагиваются, а последний завершённый отчёт остаётся
  доступен через прежний read endpoint.
- **Cleanup owed:** none.
- **Budget justification:** provider, agent, endpoint и browser образуют одну
  сквозную потоковую вертикаль; каждый промежуточный вариант либо теряет
  фактический usage, либо по-прежнему показывает результат только в конце.

### Slice 4 — Provider-side context overflow

- **In scope:** генерация payload выше фактического окна DeepSeek; пятый
  API-вызов с обходом локального guard только внутри benchmark; безопасное
  извлечение provider status/code/message; отображение размера и SHA-256;
  тесты и новая запись живого прогона.
- **Out of scope:** изменение безопасного preflight guard обычного чата,
  хранение многомегабайтного payload или повтор provider-side ошибки.
- **Ships safely because:** oversized-вызов доступен только по явному запуску
  отдельной benchmark-страницы; ожидаемой считается только подтверждённая
  ошибка контекстного лимита, остальные ошибки не маскируются.
- **Cleanup owed:** none.
- **Size:** ~475 reviewable lines; документация и MP4 исключены из метрики.

## Decision log

- 2026-09-10: рабочий прототип достиг ~1.4K reviewable lines; до коммитов
  разделён на backend core и демонстрационный UI, оба ниже жёсткого лимита 1000.
- 2026-09-10: по обратной связи сравнение вынесено из чата на отдельную страницу;
  локальные ответы заменены реальными DeepSeek-вызовами с видимыми prompt.
- 2026-09-10: пользователь отклонил монтаж готовых состояний; real-time
  transport и запись настоящего запуска выделены из разросшегося slice 2 в
  отдельный slice 3.
- 2026-09-10: локальная имитация overflow не удовлетворяла условию; реальный
  provider-side отказ выделен в slice 4, чтобы live-срез не превысил жёсткий
  лимит 1000 reviewable lines.
