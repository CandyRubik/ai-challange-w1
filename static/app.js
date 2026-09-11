const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const numberFormat = new Intl.NumberFormat("ru-RU");
const $ = (selector) => document.querySelector(selector);

const sessionList = $("#sessions");
const messagesNode = $("#messages");
const titleNode = $("#chat-title");
const statusNode = $("#status");
const messageForm = $("#message-form");
const messageInput = $("#message-input");
const newButton = $("#new-session");
const clearButton = $("#clear-database");
const sessionTokens = $("#session-tokens");
const sessionCost = $("#session-cost");

let sessions = [];
let currentSessionId = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const message = typeof detail === "string"
      ? detail
      : detail?.message || (Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : "");
    throw new Error(message || "Backend не смог выполнить запрос");
  }
  return response.status === 204 ? null : response.json();
}

function setBusy(value, text = "Агент отвечает…") {
  busy = value;
  messageInput.disabled = value;
  messageForm.querySelector("button").disabled = value;
  newButton.disabled = value;
  clearButton.disabled = value;
  if (value) statusNode.textContent = text;
}

function renderSessions() {
  sessionList.replaceChildren(...sessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === currentSessionId ? "session active" : "session";
    button.textContent = session.title;
    button.addEventListener("click", () => openSession(session.id));
    return button;
  }));
}

function renderUsage(usage) {
  sessionTokens.textContent = `${numberFormat.format(usage?.total_tokens || 0)} токенов`;
  sessionCost.textContent = `$${(usage?.estimated_cost_usd || 0).toFixed(6)}`;
}

function assistantMeta(turn) {
  if (!turn) return "Агент";
  const prompt = turn.prompt_tokens || turn.estimated_prompt_tokens;
  const total = turn.total_tokens + turn.summary_total_tokens;
  const cost = (turn.estimated_cost_usd || 0) + (turn.summary_estimated_cost_usd || 0);
  const summaryPart = turn.summary_total_tokens
    ? ` · компактизация ${numberFormat.format(turn.summary_total_tokens)} ток.`
    : "";
  return `Агент · ${numberFormat.format(prompt)} input · ${numberFormat.format(turn.completion_tokens)} output · ${numberFormat.format(total)} всего · $${cost.toFixed(6)}${summaryPart}`;
}

function messageElement(message, labelText) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  const label = document.createElement("span");
  const content = document.createElement("p");
  label.textContent = labelText;
  content.textContent = message.content;
  article.append(label, content);
  return article;
}

function compactionElement(compaction, turns) {
  const turn = turns.find((item) => (
    item.summary_total_tokens > 0
    && item.compressed_messages === compaction.summarized_message_count
  ));
  const cost = turn
    ? ` · ${numberFormat.format(turn.summary_total_tokens)} ток. · $${(turn.summary_estimated_cost_usd || 0).toFixed(6)}`
    : "";
  const article = messageElement(
    { role: "summary", content: compaction.summary },
    `✦ Контекст скомпактизирован · сообщения ${numberFormat.format(compaction.source_start_position + 1)}–${numberFormat.format(compaction.source_end_position + 1)} заменены summary для LLM${cost}`,
  );
  article.dataset.compactionId = compaction.id;
  return article;
}

function renderMessages(session) {
  const items = session.messages || [];
  if (!items.length) {
    messagesNode.innerHTML = '<div class="empty"><span>✦</span><h2>Начните диалог</h2><p>После каждых 10 сообщений контекст автоматически сжимается.</p></div>';
    return;
  }

  const turns = session.token_usage?.turns || [];
  const compactionsByEnd = new Map(
    (session.compactions || []).map((compaction) => [
      compaction.source_end_position,
      compaction,
    ]),
  );
  let assistantIndex = 0;
  const nodes = [];
  items.forEach((message) => {
    let label = "Вы";
    if (message.role === "assistant") {
      label = assistantMeta(turns[assistantIndex]);
      assistantIndex += 1;
    }
    nodes.push(messageElement(message, label));
    const compaction = compactionsByEnd.get(message.position);
    if (compaction) nodes.push(compactionElement(compaction, turns));
  });
  messagesNode.replaceChildren(...nodes);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function appendPendingExchange(content) {
  messagesNode.querySelector(".empty")?.remove();
  const userMessage = messageElement({ role: "user", content }, "Вы");
  userMessage.classList.add("optimistic");

  const pending = document.createElement("article");
  pending.className = "message assistant pending";
  const label = document.createElement("span");
  label.textContent = "Агент формирует ответ";
  const indicator = document.createElement("p");
  indicator.className = "typing-indicator";
  indicator.setAttribute("aria-label", "Ожидание ответа агента");
  for (let index = 0; index < 3; index += 1) {
    indicator.append(document.createElement("i"));
  }
  pending.append(label, indicator);
  messagesNode.append(userMessage, pending);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function renderSession(session) {
  currentSessionId = session.id;
  titleNode.textContent = session.title;
  renderSessions();
  renderMessages(session);
  renderUsage(session.token_usage);
}

async function openSession(sessionId) {
  if (busy) return;
  try {
    renderSession(await api(`/api/chat/sessions/${sessionId}`));
    statusNode.textContent = "Готов";
    messageInput.focus();
  } catch (error) {
    statusNode.textContent = error.message;
  }
}

async function createSession() {
  if (busy) return;
  try {
    const session = await api("/api/chat/sessions", { method: "POST" });
    sessions.unshift(session);
    renderSession(session);
    statusNode.textContent = "Готов";
  } catch (error) {
    statusNode.textContent = error.message;
  }
}

async function loadSessions() {
  sessions = await api("/api/chat/sessions");
  if (sessions.length) await openSession(sessions[0].id);
  else await createSession();
}

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = messageInput.value.trim();
  if (!content || !currentSessionId || busy) return;
  messageInput.value = "";
  appendPendingExchange(content);
  setBusy(true);
  try {
    const result = await api(`/api/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    const session = await api(`/api/chat/sessions/${currentSessionId}`);
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    renderSession(session);
    statusNode.textContent = "Готов";
  } catch (error) {
    try {
      renderSession(await api(`/api/chat/sessions/${currentSessionId}`));
    } catch (_) {
      messagesNode.querySelectorAll(".optimistic, .pending").forEach((node) => node.remove());
    }
    messageInput.value = content;
    statusNode.textContent = error.message;
  } finally {
    setBusy(false);
    messageInput.focus();
  }
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (!busy) messageForm.requestSubmit();
});

newButton.addEventListener("click", createSession);
clearButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Удалить все чат-сессии и сообщения без возможности восстановления?")) return;
  setBusy(true, "Очищаем историю…");
  try {
    await api("/api/chat/sessions", { method: "DELETE" });
    sessions = [];
    currentSessionId = null;
    renderSessions();
    renderUsage();
  } catch (error) {
    statusNode.textContent = error.message;
    return;
  } finally {
    setBusy(false);
  }
  await createSession();
});

loadSessions().catch((error) => { statusNode.textContent = error.message; });
