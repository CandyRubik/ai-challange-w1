const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const numberFormat = new Intl.NumberFormat("ru-RU");
const $ = (selector) => document.querySelector(selector);

const runButton = $("#run-benchmark");
const callCount = $("#api-call-count");
const statusNode = $("#benchmark-status");
const resultsNode = $("#benchmark-results");
const summaryNode = $("#benchmark-summary");
let plan = null;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || "Benchmark завершился ошибкой");
  }
  return response.json();
}

function requestBlock(request, index, overflow = false) {
  const block = document.createElement("article");
  block.className = "benchmark-turn request";
  const label = overflow ? "БУДЕТ ОТКЛОНЁН ДО API" : "БУДЕТ ОТПРАВЛЕН В DEEPSEEK";
  block.innerHTML = `<header><span>REQUEST ${index + 1}</span><small>${label}</small></header>`;
  const text = document.createElement("pre");
  text.textContent = request;
  block.append(text);
  return block;
}

function renderPlan(value) {
  const focus = new URLSearchParams(window.location.search).get("focus");
  const scenarios = focus ? value.scenarios.filter((item) => item.id === focus) : value.scenarios;
  resultsNode.replaceChildren(...scenarios.map((scenario) => {
    const card = document.createElement("article");
    card.className = `benchmark-card ${scenario.id}`;
    card.innerHTML = `<header><div><span>${scenario.id === "overflow" ? "✕" : "●"}</span><h2>${scenario.title}</h2></div><p>${scenario.description}</p></header>`;
    const turns = document.createElement("div");
    turns.className = "benchmark-turns";
    turns.append(...scenario.requests.map((request, index) => requestBlock(request, index, scenario.id === "overflow")));
    card.append(turns);
    return card;
  }));
}

function usageLine(usage) {
  if (!usage) return "";
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

function renderReport(report) {
  const focus = new URLSearchParams(window.location.search).get("focus");
  const scenarios = focus ? report.scenarios.filter((item) => item.id === focus) : report.scenarios;
  resultsNode.replaceChildren(...scenarios.map((scenario) => {
    const card = document.createElement("article");
    card.className = `benchmark-card ${scenario.id} ${scenario.status}`;
    card.innerHTML = `
      <header>
        <div><span>${scenario.status === "overflow" ? "✕" : "✓"}</span><h2>${scenario.title}</h2></div>
        <p>${scenario.description}</p>
        <dl>
          <div><dt>Input</dt><dd>${numberFormat.format(scenario.prompt_tokens)}</dd></div>
          <div><dt>Output</dt><dd>${numberFormat.format(scenario.completion_tokens)}</dd></div>
          <div><dt>Total</dt><dd>${numberFormat.format(scenario.total_tokens)}</dd></div>
          <div><dt>Цена</dt><dd>$${scenario.estimated_cost_usd.toFixed(6)}</dd></div>
        </dl>
      </header>
    `;
    const turns = document.createElement("div");
    turns.className = "benchmark-turns";
    scenario.turns.forEach((turn) => {
      turns.append(requestBlock(turn.request, turn.turn - 1, scenario.id === "overflow"));
      if (turn.response) {
        const response = document.createElement("article");
        response.className = "benchmark-turn response";
        response.innerHTML = `<header><span>RESPONSE ${turn.turn}</span><small>${usageLine(turn.token_usage)}</small></header>`;
        const text = document.createElement("pre");
        text.textContent = turn.response;
        response.append(text);
        turns.append(response);
      }
    });
    if (scenario.overflow) {
      const overflow = document.createElement("p");
      overflow.className = "benchmark-overflow";
      overflow.textContent = `${numberFormat.format(scenario.overflow.prompt_tokens)} input + ${numberFormat.format(scenario.overflow.reserved_output_tokens)} reserved output > ${numberFormat.format(scenario.overflow.context_limit_tokens)} limit. Превышение: ${numberFormat.format(scenario.overflow.overflow_tokens)}. Вызова API не было.`;
      turns.append(overflow);
    }
    card.append(turns);
    return card;
  }));

  const completedTurns = report.scenarios.flatMap((scenario) => scenario.turns).filter((turn) => turn.token_usage);
  $("#summary-calls").textContent = numberFormat.format(completedTurns.length);
  $("#summary-prompt").textContent = numberFormat.format(report.scenarios.reduce((sum, item) => sum + item.prompt_tokens, 0));
  $("#summary-completion").textContent = numberFormat.format(report.scenarios.reduce((sum, item) => sum + item.completion_tokens, 0));
  $("#summary-cost").textContent = `$${report.scenarios.reduce((sum, item) => sum + item.estimated_cost_usd, 0).toFixed(6)}`;
  summaryNode.hidden = false;
}

async function loadPlan() {
  plan = await api("/api/benchmark/plan");
  renderPlan(plan);
  callCount.textContent = `${plan.api_calls} платных API-вызова · overflow без вызова`;
  statusNode.textContent = "Проверьте тексты ниже и запустите benchmark.";
  runButton.disabled = false;
}

async function runBenchmark() {
  if (!plan || !window.confirm(`Отправить ${plan.api_calls} показанных запроса в DeepSeek API?`)) return;
  runButton.disabled = true;
  statusNode.textContent = "DeepSeek обрабатывает реальные запросы…";
  try {
    const report = await api("/api/benchmark/run", { method: "POST" });
    renderReport(report);
    statusNode.textContent = "Готово: request, response и фактический usage показаны ниже.";
  } catch (error) {
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
      callCount.textContent = `${plan.api_calls} платных API-вызова · overflow без вызова`;
      renderReport(report);
      statusNode.textContent = "Последний реальный benchmark.";
      runButton.disabled = false;
    })
    .catch((error) => { statusNode.textContent = error.message; });
} else {
  loadPlan().catch((error) => { statusNode.textContent = error.message; });
}
