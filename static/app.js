const searchInput = document.getElementById("searchInput");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultsList = document.getElementById("resultsList");
const compareSection = document.getElementById("compareSection");
const scoreStrip = document.getElementById("scoreStrip");
const detailSection = document.getElementById("detailSection");
const tabs = document.querySelectorAll(".tab");

let mode = "name";
let debounceTimer = null;
let activeQuery = "";

const placeholders = {
  name: "Search medicine name (e.g. augmentin, azee)",
  symptom: "Search symptom or condition (e.g. cough, bacterial infection, hypertension)",
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    tabs.forEach((t) => {
      const active = t === tab;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });
    searchInput.placeholder = placeholders[mode];
    runSearch();
  });
});

searchInput.addEventListener("input", () => {
  clearBtn.hidden = !searchInput.value;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runSearch, 220);
});

clearBtn.addEventListener("click", () => {
  searchInput.value = "";
  clearBtn.hidden = true;
  resultsList.hidden = true;
  compareSection.hidden = true;
  detailSection.hidden = true;
  statusEl.textContent = "Type to search. Symptom mode compares BM25, TF-IDF, and BoW cosine.";
});

async function runSearch() {
  const query = searchInput.value.trim();
  activeQuery = query;

  if (!query) {
    resultsList.hidden = true;
    compareSection.hidden = true;
    detailSection.hidden = true;
    statusEl.textContent = mode === "name"
      ? "Type a medicine name."
      : "Type symptom(s). Use commas for separate concerns (e.g. belly pain, sugar bp).";
    return;
  }

  statusEl.textContent = "Searching…";

  try {
    if (mode === "name") {
      compareSection.hidden = true;
      const res = await fetch(`/api/search/name?q=${encodeURIComponent(query)}&limit=12`);
      if (!res.ok) throw new Error(`name search HTTP ${res.status}`);
      const data = await res.json();
      if (query !== activeQuery) return;
      renderResults(data.results || [], query, false);
      statusEl.textContent = data.results?.length
        ? `${data.results.length} result(s) — click to view details`
        : "No matches. Try another query.";
      return;
    }

    const compareRes = await fetch(
      `/api/search/symptom/compare?q=${encodeURIComponent(query)}&per_algo=8&combined=12`,
    );

    if (compareRes.status === 404) {
      await runSymptomFallback(query);
      return;
    }
    if (!compareRes.ok) throw new Error(`symptom compare HTTP ${compareRes.status}`);

    const data = await compareRes.json();
    if (query !== activeQuery) return;

    renderCompare(data);
    renderResults(data.combined || [], query, true);
    const stats = data.stats || {};
    const bm25Strong = stats.bm25 ?? 0;
    const tfidfStrong = stats.tfidf ?? 0;
    const cosineStrong = stats.cosine ?? 0;
    const bm25Shown = stats.bm25_shown ?? 0;
    const tfidfShown = stats.tfidf_shown ?? 0;
    const cosineShown = stats.cosine_shown ?? 0;
    const bm25Rel = stats.bm25_relevant ?? 0;
    const tfidfRel = stats.tfidf_relevant ?? 0;
    const cosineRel = stats.cosine_relevant ?? 0;
    const clauseNote = formatClauseStats(stats);
    const expansionNote = formatExpansion(data);
    const anyMatch = stats.any_clause_matches ?? 0;

    if (!anyMatch && !bm25Shown && !tfidfShown && !cosineShown) {
      statusEl.textContent = clauseNote
        ? `No matches for combined filter. ${clauseNote}`
        : "No matches. Try another symptom.";
      return;
    }

    statusEl.textContent = [
      clauseNote,
      `any clause: ${anyMatch}`,
      `BM25 ${bm25Strong} top / ${bm25Rel} related (${bm25Shown} shown)`,
      `TF-IDF ${tfidfStrong} top / ${tfidfRel} related (${tfidfShown} shown)`,
      `Cosine ${cosineStrong} top / ${cosineRel} related (${cosineShown} shown)`,
      expansionNote ? `expanded: ${expansionNote}` : "",
      "click any result",
    ].filter(Boolean).join(" · ");
  } catch (err) {
    setStatusError("Search failed. Restart server: python app.py → http://127.0.0.1:5001");
    console.error(err);
  }
}

async function runSymptomFallback(query) {
  const res = await fetch(`/api/search/symptom?q=${encodeURIComponent(query)}&limit=12`);
  if (!res.ok) throw new Error(`symptom fallback HTTP ${res.status}`);
  const data = await res.json();
  if (query !== activeQuery) return;

  compareSection.hidden = true;
  renderResults(data.results || [], query, true);
  statusEl.textContent = data.results?.length
    ? `${data.results.length} BM25 result(s) — restart server for TF-IDF compare panel`
    : "No matches. Try another symptom.";
}

const COMPARE_ALGOS = ["bm25", "tfidf", "cosine"];

function renderCompare(data) {
  const stats = data.stats || {};
  const maxHits = stats.max_hits || 1;
  const algoLabels = {
    bm25: "BM25",
    tfidf: "TF-IDF",
    cosine: "Cosine",
  };
  const algoTotals = {
    bm25: stats.bm25_relevant ?? stats.bm25 ?? 0,
    tfidf: stats.tfidf_relevant ?? stats.tfidf ?? 0,
    cosine: stats.cosine_relevant ?? stats.cosine ?? 0,
  };

  compareSection.hidden = false;
  scoreStrip.innerHTML = COMPARE_ALGOS.map((algo) => `
    <div class="score-row">
      <span class="score-label">${algoLabels[algo]}</span>
      <div class="score-bar"><div class="score-fill ${algo}" style="width:${(algoTotals[algo] / maxHits) * 100}%"></div></div>
      <span class="score-count">${algoTotals[algo]}</span>
    </div>
  `).join("");

  COMPARE_ALGOS.forEach((algo) => {
    renderAlgoPanel(algo, data.algorithms?.[algo] || []);
  });
}

function renderAlgoPanel(algo, items) {
  const list = document.getElementById(`list-${algo}`);
  list.innerHTML = "";

  if (!items.length) {
    list.innerHTML = '<li class="panel-empty">No hits</li>';
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.dataset.algo = algo;
    li.innerHTML = `
      <span class="name">${escapeHtml(item.name)}</span>
      <span class="meta">${escapeHtml(truncate(item.uses, 55))}</span>
      <span class="score">score ${item.score}</span>
    `;
    li.addEventListener("click", () => {
      document.querySelectorAll(".panel-list li").forEach((el) => el.classList.remove("active"));
      li.classList.add("active");
      loadMedicine(item.name, null, algo);
    });
    list.appendChild(li);
  });
}

function renderResults(results, query, showSource) {
  resultsList.innerHTML = "";
  resultsList.hidden = results.length === 0;

  results.forEach((item) => {
    const li = document.createElement("li");
    li.className = "result-item";
    const sourceBadge = showSource && item.source
      ? `<span class="result-source ${item.source}">${item.source}</span>`
      : "";
    li.innerHTML = `
      <div>
        <strong>${escapeHtml(item.name)}${sourceBadge}</strong>
        <small>${escapeHtml(item.manufacturer || "")} · ${escapeHtml(truncate(item.uses, 70))}</small>
      </div>
      ${item.score != null ? `<span class="result-score">score ${item.score}</span>` : ""}
    `;
    li.addEventListener("click", () => loadMedicine(item.name, li, item.source));
    resultsList.appendChild(li);
  });

  if (results.length === 1 && mode === "name" && results[0].name.toLowerCase() === query.toLowerCase()) {
    loadMedicine(results[0].name, resultsList.firstChild);
  }
}

async function loadMedicine(name, rowEl, sourceAlgo) {
  document.querySelectorAll(".result-item").forEach((el) => el.classList.remove("active"));
  if (rowEl) rowEl.classList.add("active");

  if (sourceAlgo) {
    document.querySelectorAll(`.panel-list li[data-algo="${sourceAlgo}"]`).forEach((el) => {
      el.classList.toggle("active", el.querySelector(".name")?.textContent === name);
    });
  }

  statusEl.textContent = `Loading ${name}…`;

  try {
    const [detailRes, altRes] = await Promise.all([
      fetch(`/api/medicine?name=${encodeURIComponent(name)}`),
      fetch(`/api/alternatives?name=${encodeURIComponent(name)}&limit=20`),
    ]);

    if (!detailRes.ok) {
      statusEl.textContent = "Medicine not found.";
      return;
    }

    const detailData = await detailRes.json();
    const altData = await altRes.json();
    renderDetail(detailData.medicine);
    renderAlternatives(altData);
    detailSection.hidden = false;
    statusEl.textContent = `Showing ${name} and alternatives.`;
    detailSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    statusEl.textContent = "Failed to load medicine details.";
    console.error(err);
  }
}

function renderDetail(med) {
  document.getElementById("detailName").textContent = med.name;
  document.getElementById("detailManufacturer").textContent = med.manufacturer || "—";
  document.getElementById("detailComposition").textContent = med.composition || "—";
  document.getElementById("detailUses").textContent = med.uses || med.uses_raw || "—";

  setBar("Excellent", med.excellent_pct);
  setBar("Average", med.average_pct);
  setBar("Poor", med.poor_pct);

  const img = document.getElementById("detailImage");
  if (med.image_url) {
    img.src = med.image_url;
    img.alt = med.name;
    img.hidden = false;
  } else {
    img.hidden = true;
  }

  const list = document.getElementById("sideEffectsList");
  list.innerHTML = "";
  (med.side_effects_list || []).forEach((effect) => {
    const li = document.createElement("li");
    li.textContent = effect;
    list.appendChild(li);
  });
}

function setBar(kind, value) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  document.getElementById(`bar${kind}`).style.width = `${pct}%`;
  document.getElementById(`pct${kind}`).textContent = `${pct}%`;
}

function renderAlternatives(data) {
  const body = document.getElementById("altBody");
  const empty = document.getElementById("altEmpty");
  body.innerHTML = "";

  const alts = data.alternatives || [];
  empty.hidden = alts.length > 0;
  document.getElementById("altNote").textContent = alts.length
    ? `${data.total_alternatives} brand(s) share this composition — showing top by review score`
    : "Sorted by review score (Excellent ×2 + Average − Poor)";

  alts.forEach((alt) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(alt.name)}</strong></td>
      <td>${escapeHtml(alt.manufacturer || "—")}</td>
      <td>${alt.excellent_pct}% / ${alt.average_pct}% / ${alt.poor_pct}%</td>
      <td>${alt.side_effects_count}</td>
    `;
    tr.addEventListener("click", () => loadMedicine(alt.name, null));
    body.appendChild(tr);
  });
}

function formatExpansion(data) {
  const raw = (data.tokens || []).join(", ");
  const expanded = (data.expanded_tokens || []).join(", ");
  if (!expanded || expanded === raw) return "";
  return expanded;
}

function formatClauseStats(stats) {
  const clauses = stats.clause_matches || [];
  if (!clauses.length) return "";
  return clauses.map((c) => `${c.label} (${c.matches})`).join(", ");
}

function truncate(text, max) {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setStatusError(message) {
  statusEl.textContent = message;
  statusEl.classList.add("status-error");
}

async function checkServerHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(`health HTTP ${res.status}`);
    const data = await res.json();
    if (!data.ok) {
      setStatusError("Index not loaded. Restart: python app.py");
      return;
    }
    statusEl.classList.remove("status-error");
  } catch {
    setStatusError("Server offline. Run: python app.py then open http://127.0.0.1:5001");
  }
}

checkServerHealth();
