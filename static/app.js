const numberFormat = new Intl.NumberFormat("ru-RU");
const $ = (selector) => document.querySelector(selector);

const statusNode = $("#status");
const titleNode = $("#title");
const sessionsNode = $("#sessions");
const messagesNode = $("#messages");
const settingsForm = $("#settings-form");
const messageForm = $("#message-form");
const messageInput = $("#message-input");
const experimentForm = $("#experiment-form");
const experimentResult = $("#experiment-result");
const contextFill = $("#context-fill");
const contextCaption = $("#context-caption");
const turnChart = $("#turn-chart");

let sessions = [];
let currentSessionId = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    const message = typeof detail === "object" ? detail.message : detail;
    const error = new Error(message || "Backend не выполнил запрос");
    error.details = detail;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

function setBusy(value, text = "Агент отвечает…") {
  busy = value;
  document.querySelectorAll("button, textarea").forEach((node) => {
    node.disabled = value;
  });
  if (value) statusNode.textContent = text;
}

function metric(id, value) {
  $(id).textContent = numberFormat.format(value || 0);
}

function renderTurnChart(turns) {
  if (!turns.length) {
    turnChart.innerHTML = '<span class="caption">Рост по ходам появится после ответа.</span>';
    return;
  }
  const maximum = Math.max(...turns.map((turn) => (
    (turn.prompt_tokens || turn.estimated_prompt_tokens) + turn.completion_tokens
  )), 1);
  turnChart.replaceChildren(...turns.map((turn) => {
    const prompt = turn.prompt_tokens || turn.estimated_prompt_tokens;
    const row = document.createElement("div");
    row.className = "turn-row";
    const label = document.createElement("span");
    label.textContent = `Ход ${turn.turn}`;
    const bars = document.createElement("div");
    bars.className = "turn-bars";
    const inputBar = document.createElement("i");
    inputBar.className = "input-bar";
    inputBar.style.width = `${Math.max(1, prompt / maximum * 100)}%`;
    const outputBar = document.createElement("i");
    outputBar.className = "output-bar";
    outputBar.style.width = `${Math.max(1, turn.completion_tokens / maximum * 100)}%`;
    bars.append(inputBar, outputBar);
    const total = document.createElement("strong");
    total.textContent = numberFormat.format(prompt + turn.completion_tokens);
    row.append(label, bars, total);
    return row;
  }));
}

function renderUsage(usage = { turns: [] }) {
  const turns = usage.turns || [];
  const latest = turns.at(-1);
  metric("#metric-current", latest?.current_message_tokens);
  metric("#metric-history", latest?.history_tokens);
  metric("#metric-prompt", latest && (latest.prompt_tokens || latest.estimated_prompt_tokens));
  metric("#metric-completion", latest?.completion_tokens);
  metric("#metric-total", usage.total_tokens);
  $("#metric-cost").textContent = `$${(usage.estimated_cost_usd || 0).toFixed(6)}`;
  if (!latest) {
    contextFill.style.width = "0";
    contextCaption.textContent = "Контекст ещё не использован";
    renderTurnChart([]);
    return;
  }
  const prompt = latest.prompt_tokens || latest.estimated_prompt_tokens;
  const percent = Math.min(100, prompt / latest.context_limit_tokens * 100);
  contextFill.style.width = `${percent}%`;
  contextFill.className = percent >= 100 ? "overflow" : percent >= 80 ? "warning" : "";
  const notes = [
    `${numberFormat.format(prompt)} / ${numberFormat.format(latest.context_limit_tokens)} (${percent.toFixed(1)}%)`,
  ];
  if (latest.dropped_messages) notes.push(`удалено сообщений: ${latest.dropped_messages}`);
  if (latest.finish_reason === "length") notes.push("ответ обрезан: finish_reason=length");
  contextCaption.textContent = notes.join(" · ");
  renderTurnChart(turns);
}

function renderSessions() {
  sessionsNode.replaceChildren(...sessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === currentSessionId ? "session active" : "session";
    button.textContent = session.title;
    button.addEventListener("click", () => openSession(session.id));
    return button;
  }));
}

function renderMessages(items) {
  if (!items.length) {
    messagesNode.innerHTML = '<div class="empty"><strong>Начните диалог</strong><span>Токены каждого хода появятся сверху.</span></div>';
    return;
  }
  messagesNode.replaceChildren(...items.map((message) => {
    const node = document.createElement("article");
    node.className = `message ${message.role}`;
    const label = document.createElement("span");
    label.textContent = message.role === "user" ? "Вы" : "Агент";
    const text = document.createElement("p");
    text.textContent = message.content;
    node.append(label, text);
    return node;
  }));
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

async function openSession(sessionId) {
  if (busy) return;
  try {
    const session = await api(`/api/chat/sessions/${sessionId}`);
    currentSessionId = session.id;
    titleNode.textContent = session.title;
    renderSessions();
    renderMessages(session.messages);
    renderUsage(session.token_usage);
    statusNode.textContent = "Готов";
  } catch (error) {
    statusNode.textContent = error.message;
  }
}

async function createSession() {
  const session = await api("/api/chat/sessions", { method: "POST" });
  sessions.unshift(session);
  await openSession(session.id);
}

async function loadSessions() {
  sessions = await api("/api/chat/sessions");
  if (sessions.length) await openSession(sessions[0].id);
  else await createSession();
}

function renderSettings(settings) {
  Object.entries(settings).forEach(([key, value]) => {
    const field = settingsForm.elements[key];
    if (!field) return;
    if (field.type === "checkbox") field.checked = value;
    else field.value = value;
  });
}

function readSettings() {
  return {
    model: settingsForm.elements.model.value.trim(),
    thinking_enabled: settingsForm.elements.thinking_enabled.checked,
    history_enabled: settingsForm.elements.history_enabled.checked,
    max_tokens: Number(settingsForm.elements.max_tokens.value),
    context_limit_tokens: Number(settingsForm.elements.context_limit_tokens.value),
    overflow_strategy: settingsForm.elements.overflow_strategy.value,
    system_prompt: settingsForm.elements.system_prompt.value.trim(),
  };
}

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = messageInput.value.trim();
  if (!content || !currentSessionId || busy) return;
  setBusy(true);
  try {
    const result = await api(`/api/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    messageInput.value = "";
    const session = await api(`/api/chat/sessions/${currentSessionId}`);
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    renderSessions();
    renderMessages(session.messages);
    renderUsage(session.token_usage);
    titleNode.textContent = session.title;
    statusNode.textContent = "Готов";
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    renderSettings(await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify(readSettings()),
    }));
    statusNode.textContent = "Настройки применены к следующим запросам";
  } catch (error) {
    statusNode.textContent = error.message;
  }
});

experimentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const [kind, rawValue] = experimentForm.elements.size.value.split(":");
  const value = Number(rawValue);
  const large = kind === "tokens" ? value > 100000 : value > 1024 * 1024;
  if (large && !window.confirm("Большой тест может стоить денег. Продолжить?")) return;

  const payload = {
    target_tokens: kind === "tokens" ? value : null,
    target_bytes: kind === "bytes" ? value : null,
    needle_position: Number(experimentForm.elements.needle_position.value),
  };
  setBusy(true, "Генерируем документ…");
  experimentResult.textContent = "Загружаем tokenizer и строим документ точного размера…";
  try {
    const result = await api("/api/experiments/needle", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const sizeLines = [
      `Документ: ${numberFormat.format(result.document_tokens)} токенов`,
      `Размер: ${numberFormat.format(result.document_bytes)} байт`,
    ];
    if (result.status === "overflow") {
      experimentResult.textContent = [
        "CONTEXT OVERFLOW",
        ...sizeLines,
        `Вход + резерв: ${numberFormat.format(result.overflow.prompt_tokens)} + ${numberFormat.format(result.overflow.reserved_output_tokens)}`,
        `Лимит: ${numberFormat.format(result.overflow.context_limit_tokens)}`,
        `Превышение: ${numberFormat.format(result.overflow.overflow_tokens)}`,
        "Запрос к модели не отправлен.",
      ].join("\n");
      statusNode.textContent = "Контекст переполнен";
      return;
    }
    const turn = result.token_usage;
    renderUsage({
      turns: [turn],
      total_tokens: turn.total_tokens,
      estimated_cost_usd: turn.estimated_cost_usd || 0,
    });
    experimentResult.textContent = [
      result.found ? "✅ Факт найден" : "❌ Факт не найден",
      ...sizeLines,
      `Фактический вход API: ${numberFormat.format(turn.prompt_tokens)} токенов`,
      `Позиция факта: ${(result.needle_position * 100).toFixed(0)}%`,
      `Ожидали: ${result.secret}`,
      `Ответ: ${result.answer}`,
    ].join("\n");
    statusNode.textContent = "Эксперимент завершён";
  } catch (error) {
    experimentResult.textContent = error.message;
    statusNode.textContent = "Ошибка эксперимента";
  } finally {
    setBusy(false);
  }
});

$("#new-session").addEventListener("click", () => createSession().catch((error) => {
  statusNode.textContent = error.message;
}));

$("#clear-sessions").addEventListener("click", async () => {
  if (!window.confirm("Удалить все локальные чаты?")) return;
  await api("/api/chat/sessions", { method: "DELETE" });
  sessions = [];
  currentSessionId = null;
  renderSessions();
  renderMessages([]);
  renderUsage();
  await createSession();
});

Promise.all([api("/api/settings"), loadSessions()])
  .then(([settings]) => renderSettings(settings))
  .catch((error) => { statusNode.textContent = error.message; });

