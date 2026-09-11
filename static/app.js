const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const numberFormat = new Intl.NumberFormat("ru-RU");
const $ = (selector) => document.querySelector(selector);

const strategyLabels = {
  sliding_window: "Sliding Window",
  sticky_facts: "Sticky Facts",
  branching: "Branching",
};
const strategyHints = {
  sliding_window: "В модель уходят только последние N сообщений; ранние детали отбрасываются.",
  sticky_facts: "Facts обновляются после каждого сообщения пользователя и отправляются вместе с последними N сообщениями.",
  branching: "В модель уходит полная история активной ветки; ветки от checkpoint развиваются независимо.",
};

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
const strategyTabs = [...document.querySelectorAll(".strategy-tab")];
const strategyName = $("#strategy-name");
const windowSize = $("#window-size");
const windowControls = $("#window-controls");
const applyWindowButton = $("#apply-window");
const strategyHint = $("#strategy-hint");
const branchTools = $("#branch-tools");
const checkpointName = $("#checkpoint-name");
const checkpointSelect = $("#checkpoint-select");
const branchName = $("#branch-name");
const saveCheckpointButton = $("#save-checkpoint");
const createBranchButton = $("#create-branch");
const factsPanel = $("#facts-panel");
const factsList = $("#facts-list");
const factsCount = $("#facts-count");

let sessions = [];
let currentSessionId = null;
let currentSession = null;
let busy = false;
let activeStrategy = "sliding_window";

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
  [messageInput, newButton, clearButton, applyWindowButton, saveCheckpointButton,
    createBranchButton].forEach((node) => { node.disabled = value; });
  strategyTabs.forEach((node) => { node.disabled = value; });
  messageForm.querySelector("button").disabled = value;
  if (value) statusNode.textContent = text;
}

function renderSessions() {
  const visibleSessions = sessions.filter((session) => session.strategy === activeStrategy);
  if (!visibleSessions.length) {
    const empty = document.createElement("p");
    empty.className = "empty-sessions";
    empty.textContent = "В этой стратегии пока нет диалогов";
    sessionList.replaceChildren(empty);
    return;
  }
  sessionList.replaceChildren(...visibleSessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === currentSessionId ? "session active" : "session";
    const badge = document.createElement("small");
    badge.textContent = session.parent_session_id ? `⑂ ветка ${session.branch_name}` : "основной диалог";
    const label = document.createElement("span");
    label.textContent = session.title;
    button.append(label, badge);
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
  const memory = turn.memory_total_tokens ? ` · facts ${numberFormat.format(turn.memory_total_tokens)}` : "";
  const dropped = turn.dropped_messages ? ` · отброшено ${turn.dropped_messages}` : "";
  return `Агент · ${numberFormat.format(prompt)} input · ${numberFormat.format(turn.completion_tokens)} output${memory}${dropped}`;
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

function renderMessages(session) {
  const items = session.messages || [];
  if (!items.length) {
    messagesNode.innerHTML = '<div class="empty"><span>✦</span><h2>Начните диалог</h2><p>В метриках ответа будет видно сохранённый и отброшенный контекст.</p></div>';
    return;
  }
  const turns = session.token_usage?.turns || [];
  let assistantIndex = 0;
  const nodes = items.map((message) => {
    if (message.role === "user") return messageElement(message, "Вы");
    const node = messageElement(message, assistantMeta(turns[assistantIndex]));
    assistantIndex += 1;
    return node;
  });
  messagesNode.replaceChildren(...nodes);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function renderFacts(facts = {}) {
  const entries = Object.entries(facts);
  factsCount.textContent = `${numberFormat.format(entries.length)} фактов`;
  if (!entries.length) {
    const empty = document.createElement("dd");
    empty.textContent = "Память заполнится после сообщения пользователя.";
    factsList.replaceChildren(empty);
    return;
  }
  const nodes = [];
  entries.forEach(([key, value]) => {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = key;
    description.textContent = value;
    nodes.push(term, description);
  });
  factsList.replaceChildren(...nodes);
}

function renderStrategy(session) {
  activeStrategy = session.strategy;
  strategyTabs.forEach((tab) => {
    const selected = tab.dataset.strategy === activeStrategy;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
  strategyName.textContent = strategyLabels[session.strategy];
  windowSize.value = session.window_size;
  windowControls.hidden = session.strategy === "branching";
  strategyHint.textContent = strategyHints[session.strategy];
  branchTools.hidden = session.strategy !== "branching";
  factsPanel.hidden = session.strategy !== "sticky_facts";
  renderFacts(session.facts);
  checkpointSelect.replaceChildren(...(session.checkpoints || []).map((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint.id;
    option.textContent = `${checkpoint.name} · ${checkpoint.message_count} сообщ.`;
    return option;
  }));
  createBranchButton.disabled = busy || !session.checkpoints?.length;
}

function renderSession(session) {
  currentSession = session;
  currentSessionId = session.id;
  titleNode.textContent = session.title;
  renderSessions();
  renderMessages(session);
  renderUsage(session.token_usage);
  renderStrategy(session);
}

function appendPendingExchange(content) {
  messagesNode.querySelector(".empty")?.remove();
  const userMessage = messageElement({ role: "user", content }, "Вы");
  userMessage.classList.add("optimistic");
  const pending = document.createElement("article");
  pending.className = "message assistant pending";
  const label = document.createElement("span");
  label.textContent = currentSession?.strategy === "sticky_facts"
    ? "Агент обновляет facts и формирует ответ"
    : "Агент формирует ответ";
  const indicator = document.createElement("p");
  indicator.className = "typing-indicator";
  for (let index = 0; index < 3; index += 1) indicator.append(document.createElement("i"));
  pending.append(label, indicator);
  messagesNode.append(userMessage, pending);
  messagesNode.scrollTop = messagesNode.scrollHeight;
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
    const session = await api("/api/chat/sessions", {
      method: "POST",
      body: JSON.stringify({
        strategy: activeStrategy,
        window_size: activeStrategy === "branching" ? 6 : Number(windowSize.value || 6),
      }),
    });
    sessions.unshift(session);
    renderSession(session);
    statusNode.textContent = "Готов";
  } catch (error) {
    statusNode.textContent = error.message;
  }
}

async function loadSessions() {
  sessions = await api("/api/chat/sessions");
  const initial = sessions.find((session) => session.strategy === activeStrategy);
  if (initial) await openSession(initial.id);
  else await createSession();
}

async function openStrategy(strategy) {
  if (busy || strategy === activeStrategy) return;
  activeStrategy = strategy;
  renderSessions();
  const latest = sessions.find((session) => session.strategy === strategy);
  if (latest) await openSession(latest.id);
  else await createSession();
}

async function refreshCurrent() {
  renderSession(await api(`/api/chat/sessions/${currentSessionId}`));
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
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    await refreshCurrent();
    statusNode.textContent = "Готов";
  } catch (error) {
    await refreshCurrent().catch(() => {});
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

strategyTabs.forEach((tab) => {
  tab.addEventListener("click", () => openStrategy(tab.dataset.strategy));
});

applyWindowButton.addEventListener("click", async () => {
  if (!currentSessionId || busy) return;
  setBusy(true, "Сохраняем размер окна…");
  try {
    const session = await api(`/api/chat/sessions/${currentSessionId}/strategy`, {
      method: "PUT",
      body: JSON.stringify({
        strategy: activeStrategy,
        window_size: Number(windowSize.value),
      }),
    });
    const index = sessions.findIndex((item) => item.id === session.id);
    if (index >= 0) sessions[index] = session;
    renderSession(session);
    statusNode.textContent = "Размер окна сохранён для этого диалога";
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

saveCheckpointButton.addEventListener("click", async () => {
  if (!currentSessionId || busy) return;
  setBusy(true, "Сохраняем checkpoint…");
  try {
    await api(`/api/chat/sessions/${currentSessionId}/checkpoints`, {
      method: "POST",
      body: JSON.stringify({ name: checkpointName.value.trim() || "Checkpoint" }),
    });
    await refreshCurrent();
    statusNode.textContent = "Checkpoint сохранён — теперь создайте две ветки";
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

createBranchButton.addEventListener("click", async () => {
  const name = branchName.value.trim();
  if (!currentSessionId || !checkpointSelect.value || !name || busy) {
    if (!name) statusNode.textContent = "Введите название ветки";
    return;
  }
  setBusy(true, "Создаём независимую ветку…");
  try {
    const branch = await api(`/api/chat/sessions/${currentSessionId}/branches`, {
      method: "POST",
      body: JSON.stringify({ checkpoint_id: checkpointSelect.value, name }),
    });
    sessions.unshift(branch);
    branchName.value = "";
    renderSession(branch);
    statusNode.textContent = `Ветка «${name}» создана и открыта`;
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

newButton.addEventListener("click", createSession);
clearButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Удалить все диалоги, facts, checkpoint и ветки?")) return;
  setBusy(true, "Очищаем историю…");
  try {
    await api("/api/chat/sessions", { method: "DELETE" });
    sessions = [];
    currentSessionId = null;
    renderSessions();
  } catch (error) {
    statusNode.textContent = error.message;
    return;
  } finally {
    setBusy(false);
  }
  await createSession();
});

loadSessions().catch((error) => { statusNode.textContent = error.message; });
