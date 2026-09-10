const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const numberFormat = new Intl.NumberFormat("ru-RU");
const $ = (selector) => document.querySelector(selector);

const runButton = $("#run-benchmark");
const callCount = $("#api-call-count");
const statusNode = $("#benchmark-status");
const resultsNode = $("#benchmark-results");
const summaryNode = $("#benchmark-summary");
let plan = null;
let runStartedAt = 0;
let elapsedTimer = null;

async function api(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : "Benchmark завершился ошибкой");
  }
  return response.json();
}

function requestBlock(scenarioId, request, index, overflow = false) {
  const block = document.createElement("article");
  block.id = `request-${scenarioId}-${index + 1}`;
  block.className = "benchmark-turn request";
  const label = overflow ? "ГОТОВ К РЕАЛЬНОМУ OVERFLOW-ВЫЗОВУ" : "ГОТОВ К ОТПРАВКЕ";
  block.innerHTML = `<header><span>REQUEST ${index + 1}</span><small>${label}</small></header>`;
  const text = document.createElement("pre");
  text.textContent = request;
  block.append(text);
  return block;
}

function scenarioCard(scenario) {
  const card = document.createElement("article");
  card.id = `scenario-${scenario.id}`;
  card.className = `benchmark-card ${scenario.id}`;
  card.innerHTML = `
    <header>
      <div><span class="scenario-mark">${scenario.id === "overflow" ? "✕" : "●"}</span><h2>${scenario.title}</h2></div>
      <p>${scenario.description}</p>
      <dl hidden>
        <div><dt>Input</dt><dd data-metric="prompt">0</dd></div>
        <div><dt>Output</dt><dd data-metric="completion">0</dd></div>
        <div><dt>Total</dt><dd data-metric="total">0</dd></div>
        <div><dt>Цена</dt><dd data-metric="cost">$0.000000</dd></div>
      </dl>
    </header>
  `;
  const turns = document.createElement("div");
  turns.className = "benchmark-turns";
  turns.append(...scenario.requests.map((request, index) => (
    requestBlock(scenario.id, request, index, scenario.id === "overflow")
  )));
  card.append(turns);
  return card;
}

function visibleScenarios(scenarios) {
  const focus = new URLSearchParams(window.location.search).get("focus");
  return focus ? scenarios.filter((item) => item.id === focus) : scenarios;
}

function renderPlan(value) {
  summaryNode.hidden = true;
  resultsNode.replaceChildren(...visibleScenarios(value.scenarios).map(scenarioCard));
}

function usageLine(usage) {
  const parts = [
    `current ${numberFormat.format(usage.current_message_tokens)}`,
    `history ${numberFormat.format(usage.history_tokens)}`,
    `input ${numberFormat.format(usage.prompt_tokens)}`,
    `output ${numberFormat.format(usage.completion_tokens)}`,
    `$${(usage.estimated_cost_usd || 0).toFixed(6)}`,
  ];
  if (usage.finish_reason === "length") parts.push("ответ обрезан: length");
  return parts.join(" · ");
}

function responseBlock(scenarioId, turn) {
  let block = $(`#response-${scenarioId}-${turn}`);
  if (block) return block;
  block = document.createElement("article");
  block.id = `response-${scenarioId}-${turn}`;
  block.className = "benchmark-turn response receiving";
  block.innerHTML = `<header><span>RESPONSE ${turn}</span><small>ожидаем первый токен…</small></header><pre></pre>`;
  $(`#request-${scenarioId}-${turn}`).after(block);
  return block;
}

function showScenarioResult(scenario) {
  const card = $(`#scenario-${scenario.id}`);
  if (!card) return;
  card.classList.remove("running");
  card.classList.add(scenario.status);
  card.querySelector(".scenario-mark").textContent = scenario.status === "overflow" ? "✕" : "✓";
  const metrics = card.querySelector("dl");
  metrics.hidden = false;
  if (scenario.overflow) {
    const items = metrics.querySelectorAll("div");
    items[0].querySelector("dt").textContent = "Attempted input";
    items[0].querySelector("dd").textContent = numberFormat.format(scenario.overflow.prompt_tokens);
    items[1].querySelector("dt").textContent = "HTTP";
    items[1].querySelector("dd").textContent = scenario.overflow.provider_status_code || "error";
    items[2].querySelector("dt").textContent = "Billed tokens";
    items[2].querySelector("dd").textContent = "0";
    items[3].querySelector("dd").textContent = "$0.000000";
    return;
  }
  metrics.querySelector('[data-metric="prompt"]').textContent = numberFormat.format(scenario.prompt_tokens);
  metrics.querySelector('[data-metric="completion"]').textContent = numberFormat.format(scenario.completion_tokens);
  metrics.querySelector('[data-metric="total"]').textContent = numberFormat.format(scenario.total_tokens);
  metrics.querySelector('[data-metric="cost"]').textContent = `$${scenario.estimated_cost_usd.toFixed(6)}`;
}

function showOverflow(event) {
  const card = $(`#scenario-${event.scenario_id}`);
  if (!card) return;
  const request = $(`#request-${event.scenario_id}-${event.turn}`);
  request.querySelector("small").textContent = "ОТПРАВЛЕН · DEEPSEEK ВЕРНУЛ ОШИБКУ";
  const value = event.overflow;
  const response = responseBlock(event.scenario_id, event.turn);
  response.classList.remove("receiving");
  response.classList.add("provider-error");
  response.querySelector("header span").textContent = "DEEPSEEK API ERROR";
  response.querySelector("header small").textContent = `HTTP ${value.provider_status_code || "error"} · ${value.provider_error_code || "context overflow"}`;
  response.querySelector("pre").textContent = value.provider_error_message || "DeepSeek отклонил запрос из-за переполнения контекста.";
  const message = document.createElement("p");
  message.className = "benchmark-overflow";
  message.textContent = `API-вызов выполнен. Локальная оценка: ${numberFormat.format(value.prompt_tokens)} input + ${numberFormat.format(value.reserved_output_tokens)} output > ${numberFormat.format(value.context_limit_tokens)} limit. Payload: ${numberFormat.format(value.request_chars)} символов · SHA-256 ${value.request_sha256}. Usage отсутствует, списано $0.`;
  card.querySelector(".benchmark-turns").append(message);
}

function renderSummary(report) {
  $("#summary-calls").textContent = `${numberFormat.format(report.api_calls_attempted)} (${numberFormat.format(report.api_calls_succeeded)} ok)`;
  $("#summary-prompt").textContent = numberFormat.format(report.scenarios.reduce((sum, item) => sum + item.prompt_tokens, 0));
  $("#summary-completion").textContent = numberFormat.format(report.scenarios.reduce((sum, item) => sum + item.completion_tokens, 0));
  $("#summary-cost").textContent = `$${report.scenarios.reduce((sum, item) => sum + item.estimated_cost_usd, 0).toFixed(6)}`;
  summaryNode.hidden = false;
}

function handleEvent(event) {
  if (event.type === "scenario_started") {
    $(`#scenario-${event.scenario_id}`)?.classList.add("running");
  } else if (event.type === "turn_started") {
    const request = $(`#request-${event.scenario_id}-${event.turn}`);
    request.querySelector("small").textContent = event.scenario_id === "overflow"
      ? `ОТПРАВЛЕН В DEEPSEEK · ~${numberFormat.format(event.estimated_prompt_tokens)} TOKENS`
      : "ОТПРАВЛЕН В DEEPSEEK · ЖДЁМ ПОТОК";
    responseBlock(event.scenario_id, event.turn);
  } else if (event.type === "response_delta") {
    const block = responseBlock(event.scenario_id, event.turn);
    const text = block.querySelector("pre");
    text.textContent += event.delta;
    block.querySelector("small").textContent = `получаем ответ · ${numberFormat.format(text.textContent.length)} символов`;
    text.scrollTop = text.scrollHeight;
  } else if (event.type === "turn_completed") {
    const block = responseBlock(event.scenario_id, event.turn);
    block.classList.remove("receiving");
    block.querySelector("small").textContent = usageLine(event.token_usage);
    $(`#request-${event.scenario_id}-${event.turn} small`).textContent = "ОТПРАВЛЕН В DEEPSEEK · ЗАВЕРШЕНО";
  } else if (event.type === "overflow") {
    showOverflow(event);
  } else if (event.type === "scenario_completed") {
    showScenarioResult(event.scenario);
  } else if (event.type === "status") {
    statusNode.textContent = event.message;
  } else if (event.type === "benchmark_completed") {
    renderSummary(event.report);
  } else if (event.type === "error") {
    throw new Error(event.message);
  }
}

function stopElapsedTimer() {
  if (elapsedTimer) window.clearInterval(elapsedTimer);
  elapsedTimer = null;
}

function startElapsedTimer() {
  runStartedAt = performance.now();
  stopElapsedTimer();
  const update = () => {
    const elapsed = ((performance.now() - runStartedAt) / 1000).toFixed(1);
    statusNode.textContent = `Live: DeepSeek отвечает потоком · ${elapsed} с`;
  };
  update();
  elapsedTimer = window.setInterval(update, 100);
}

async function readEventStream(response) {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : "Benchmark не запустился");
  }
  if (!response.body) throw new Error("Браузер не поддерживает потоковый ответ");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) handleEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) handleEvent(JSON.parse(buffer));
}

function renderStoredReport(report) {
  renderPlan({ scenarios: report.scenarios });
  visibleScenarios(report.scenarios).forEach((scenario) => {
    scenario.turns.forEach((turn) => {
      if (!turn.response) return;
      const block = responseBlock(scenario.id, turn.turn);
      block.querySelector("pre").textContent = turn.response;
      block.classList.remove("receiving");
      block.querySelector("small").textContent = usageLine(turn.token_usage);
    });
    if (scenario.overflow) showOverflow({
      scenario_id: scenario.id,
      turn: scenario.turns.at(-1).turn,
      overflow: scenario.overflow,
    });
    showScenarioResult(scenario);
  });
  renderSummary(report);
}

async function loadPlan() {
  plan = await api("/api/benchmark/plan");
  renderPlan(plan);
  callCount.textContent = `${plan.api_calls} реальных API-вызовов · последний должен вернуть 400`;
  statusNode.textContent = "Проверьте тексты ниже и запустите live benchmark.";
  runButton.disabled = false;
}

async function runBenchmark() {
  if (!plan) return;
  renderPlan(plan);
  runButton.disabled = true;
  startElapsedTimer();
  try {
    const response = await fetch(`${API_BASE_URL}/api/benchmark/run`, { method: "POST" });
    await readEventStream(response);
    stopElapsedTimer();
    const elapsed = ((performance.now() - runStartedAt) / 1000).toFixed(1);
    statusNode.textContent = `Готово за ${elapsed} с: это был живой поток DeepSeek API.`;
  } catch (error) {
    stopElapsedTimer();
    statusNode.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runBenchmark);

const params = new URLSearchParams(window.location.search);
if (params.get("latest") === "1") {
  Promise.all([api("/api/benchmark/plan"), api("/api/benchmark/latest")])
    .then(([loadedPlan, report]) => {
      plan = loadedPlan;
      callCount.textContent = `${plan.api_calls} реальных API-вызовов · последний должен вернуть 400`;
      renderStoredReport(report);
      statusNode.textContent = "Последний завершённый live benchmark.";
      runButton.disabled = false;
    })
    .catch((error) => { statusNode.textContent = error.message; });
} else {
  loadPlan()
    .then(() => {
      if (params.get("autorun") === "1") runBenchmark();
    })
    .catch((error) => { statusNode.textContent = error.message; });
}
