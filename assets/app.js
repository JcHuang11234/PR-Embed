const state = {
  modelTags: [],
  modelCache: new Map(),
  papers: [],
  paperVectors: null,
  paperVectorSize: 0,
  topics: [],
  topicVectors: null,
  topicVectorSize: 0,
  selectedPaper: null,
};

const $ = (id) => document.getElementById(id);

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.json();
}

async function fetchFloat32(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return new Float32Array(await res.arrayBuffer());
}

function setStatus(text) {
  $("loadStatus").textContent = text;
}

function ensureTooltip() {
  let tooltip = $("topicTooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "topicTooltip";
    tooltip.className = "topic-tooltip";
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  return Number(value).toFixed(digits);
}

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function vectorSlice(vectors, index, size) {
  return vectors.subarray(index * size, (index + 1) * size);
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
  return total;
}

function normalize(vec) {
  let sum = 0;
  for (const x of vec) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  return Float32Array.from(vec, (x) => x / norm);
}

function mixColor(a, b, t) {
  const ah = a.replace("#", "");
  const bh = b.replace("#", "");
  const ar = parseInt(ah.slice(0, 2), 16);
  const ag = parseInt(ah.slice(2, 4), 16);
  const ab = parseInt(ah.slice(4, 6), 16);
  const br = parseInt(bh.slice(0, 2), 16);
  const bg = parseInt(bh.slice(2, 4), 16);
  const bb = parseInt(bh.slice(4, 6), 16);
  const r = Math.round(ar + (br - ar) * t).toString(16).padStart(2, "0");
  const g = Math.round(ag + (bg - ag) * t).toString(16).padStart(2, "0");
  const bl = Math.round(ab + (bb - ab) * t).toString(16).padStart(2, "0");
  return `#${r}${g}${bl}`;
}

async function loadModel(tag) {
  if (state.modelCache.has(tag)) return state.modelCache.get(tag);
  setStatus(`Loading ${tag} model...`);
  const meta = await fetchJSON(`data/models/${tag}.json`);
  const vectors = await fetchFloat32(`data/models/${tag}.bin`);
  const wordToIndex = new Map(meta.words.map((word, index) => [word, index]));
  const model = { tag, words: meta.words, wordToIndex, vectorSize: meta.vectorSize, vectors };
  state.modelCache.set(tag, model);
  setStatus("Ready");
  return model;
}

function getWordVector(model, word) {
  const key = String(word || "").trim().toLowerCase();
  if (!model.wordToIndex.has(key)) return null;
  return vectorSlice(model.vectors, model.wordToIndex.get(key), model.vectorSize);
}

function topSimilar(model, queryVec, topN, exclude = new Set()) {
  const rows = [];
  for (let i = 0; i < model.words.length; i += 1) {
    const word = model.words[i];
    if (exclude.has(word)) continue;
    const sim = dot(queryVec, vectorSlice(model.vectors, i, model.vectorSize));
    rows.push({ word, similarity: sim });
  }
  rows.sort((a, b) => b.similarity - a.similarity);
  return rows.slice(0, topN);
}

function renderTable(target, rows, columns) {
  if (!rows.length) {
    target.innerHTML = `<div class="empty">No results.</div>`;
    return;
  }
  const head = columns.map((c) => `<th class="${c.num ? "num" : ""}">${escapeHTML(c.label)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((c) => {
          const raw = c.render ? c.render(row) : row[c.key];
          return `<td class="${c.num ? "num" : ""}">${raw}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  target.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderCompactTable(target, rows, columns) {
  renderTable(target, rows, columns);
  const table = target.querySelector("table");
  if (table) table.classList.add("compact-table");
}

function renderBars(target, rows, labelKey, valueKey, options = {}) {
  if (!rows.length) {
    target.innerHTML = `<div class="empty">No results.</div>`;
    return;
  }
  const values = rows.map((r) => Number(r[valueKey]));
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 0.001);
  const scoreLabel = options.scoreLabel || (valueKey === "similarity" ? "Similarity" : "Score");
  target.classList.remove("empty");
  target.innerHTML = `<div class="bar-row bar-head">
      <div></div>
      <div class="bar-axis-name">${escapeHTML(scoreLabel)}</div>
      <div></div>
    </div>` + rows
    .map((row) => {
      const value = Number(row[valueKey]);
      if (options.diverging) {
        const width = Math.abs(value / maxAbs) * 50;
        const cls = value < 0 ? "negative" : "positive";
        return `<div class="bar-row">
          <div class="bar-label" title="${escapeHTML(row[labelKey])}">${escapeHTML(row[labelKey])}</div>
          <div class="bar-track"><div class="bar-fill ${cls}" style="--w:${width}%"></div></div>
          <div class="bar-value">${fmt(value)}</div>
        </div>`;
      }
      const width = Math.max(1, (value / maxAbs) * 100);
      return `<div class="bar-row">
        <div class="bar-label" title="${escapeHTML(row[labelKey])}">${escapeHTML(row[labelKey])}</div>
        <div class="bar-track"><div class="bar-fill" style="--w:${width}%"></div></div>
        <div class="bar-value">${fmt(value)}</div>
      </div>`;
    })
    .join("");
}

function findPaper(query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return null;
  const paperFields = (p) => [p.title, p.doi, p.url, p.alternative_id, ...(p.authors || [])];
  return (
    state.papers.find((p) => paperDisplay(p).toLowerCase() === q) ||
    state.papers.find((p) => paperFields(p).some((v) => String(v || "").toLowerCase() === q)) ||
    state.papers.find((p) => paperFields(p).some((v) => String(v || "").toLowerCase().includes(q)))
  );
}

function paperDisplay(paper) {
  const authors = (paper.authors || []).join(", ") || "Unknown author";
  return `${paper.title}, ${authors} (${paper.year}, ${paper.journal})`;
}

function paperVector(paper) {
  return vectorSlice(state.paperVectors, paper.id, state.paperVectorSize);
}

function topPapersByVector(vec, topN, excludeId = null) {
  const rows = state.papers
    .filter((paper) => paper.id !== excludeId)
    .map((paper) => ({ ...paper, similarity: dot(vec, paperVector(paper)) }));
  rows.sort((a, b) => b.similarity - a.similarity);
  return rows.slice(0, topN);
}

function paperColumns() {
  return [
    { label: "Title", render: (r) => escapeHTML(r.title) },
    { label: "Journal", render: (r) => escapeHTML(r.journal) },
    { label: "Year", key: "year", num: true },
    { label: "Similarity", num: true, render: (r) => fmt(r.similarity) },
    {
      label: "DOI",
      render: (r) => (r.doi ? `<a href="https://doi.org/${escapeHTML(r.doi)}" target="_blank">${escapeHTML(r.doi)}</a>` : ""),
    },
  ];
}

async function showNeighbors() {
  const model = await loadModel($("wordModel").value);
  const word = $("wordInput").value.trim().toLowerCase();
  const vec = getWordVector(model, word);
  if (!vec) {
    $("neighborChart").innerHTML = `<div class="empty">${escapeHTML(word)} is not in this model.</div>`;
    return;
  }
  const topN = Number($("wordTopN").value);
  const rows = topSimilar(model, vec, topN, new Set([word]));
  renderBars($("neighborChart"), rows, "word", "similarity");
}

async function showWordPapers() {
  const model = await loadModel($("wordModel").value);
  const word = $("wordInput").value.trim().toLowerCase();
  const vec = getWordVector(model, word);
  if (!vec) {
    $("wordPaperTable").innerHTML = `<div class="empty">${escapeHTML(word)} is not in this model.</div>`;
    return;
  }
  const rows = topPapersByVector(vec, Number($("wordTopN").value));
  $("wordPaperTable").classList.remove("empty");
  renderTable($("wordPaperTable"), rows, paperColumns());
}

async function showDrift() {
  const a = $("driftWordA").value.trim().toLowerCase();
  const b = $("driftWordB").value.trim().toLowerCase();
  const rows = [];
  for (const tag of state.modelTags.filter((t) => t !== "full")) {
    const model = await loadModel(tag);
    const va = getWordVector(model, a);
    const vb = getWordVector(model, b);
    rows.push({ model: tag, similarity: va && vb ? dot(va, vb) : null });
  }
  const trimmedRows = trimEmptyEdges(rows);
  renderDriftChart($("driftChart"), trimmedRows, a, b);
  renderCompactTable($("driftTable"), trimmedRows, [
    { label: "Model", render: (r) => escapeHTML(r.model) },
    { label: "Similarity", num: true, render: (r) => (r.similarity === null ? "" : fmt(r.similarity)) },
  ]);
}

function trimEmptyEdges(rows) {
  const first = rows.findIndex((row) => row.similarity !== null);
  const last = rows.findLastIndex((row) => row.similarity !== null);
  if (first === -1 || last === -1) return [];
  return rows.slice(first, last + 1);
}

function parseEquation(text) {
  const tokens = String(text || "").toLowerCase().match(/[+-]?\s*[\w-]+/g) || [];
  const positive = [];
  const negative = [];
  for (const token of tokens) {
    const t = token.trim();
    if (t.startsWith("-")) negative.push(t.slice(1).trim());
    else positive.push(t.replace(/^\+/, "").trim());
  }
  return { positive: positive.filter(Boolean), negative: negative.filter(Boolean) };
}

async function runEquation() {
  const model = await loadModel($("wordModel").value);
  const { positive, negative } = parseEquation($("equationInput").value);
  const missing = [...positive, ...negative].filter((word) => !model.wordToIndex.has(word));
  if (!positive.length || missing.length) {
    $("equationChart").innerHTML = `<div class="empty">Missing terms: ${escapeHTML(missing.join(", ") || "no positive terms")}</div>`;
    return;
  }
  const raw = new Float32Array(model.vectorSize);
  for (const word of positive) {
    const vec = getWordVector(model, word);
    for (let i = 0; i < raw.length; i += 1) raw[i] += vec[i];
  }
  for (const word of negative) {
    const vec = getWordVector(model, word);
    for (let i = 0; i < raw.length; i += 1) raw[i] -= vec[i];
  }
  const vec = normalize(raw);
  const rows = topSimilar(model, vec, Number($("equationTopN").value), new Set([...positive, ...negative]));
  renderBars($("equationChart"), rows, "word", "similarity");
}

function renderDriftChart(target, rows, wordA, wordB) {
  if (!rows.length) {
    target.innerHTML = `<div class="empty">No overlapping yearly vectors for this pair.</div>`;
    return;
  }
  const width = 560;
  const height = 190;
  const pad = { l: 54, r: 16, t: 24, b: 38 };
  const values = rows.filter((r) => r.similarity !== null).map((r) => r.similarity);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const spread = Math.max(rawMax - rawMin, 0.08);
  const min = Math.max(-1, rawMin - spread * 0.25);
  const max = Math.min(1, rawMax + spread * 0.25);
  const x = (i) => pad.l + (i / Math.max(rows.length - 1, 1)) * (width - pad.l - pad.r);
  const y = (v) => height - pad.b - ((v - min) / (max - min || 1)) * (height - pad.t - pad.b);
  const grid = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);
  let started = false;
  const path = rows
    .map((row, i) => {
      if (row.similarity === null) {
        started = false;
        return "";
      }
      const command = started ? "L" : "M";
      started = true;
      return `${command} ${x(i).toFixed(1)} ${y(row.similarity).toFixed(1)}`;
    })
    .filter(Boolean)
    .join(" ");

  target.classList.remove("empty");
  target.innerHTML = `
    <svg class="drift-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Semantic drift from ${escapeHTML(wordA)} to ${escapeHTML(wordB)}">
      <text class="axis-title" x="${width / 2}" y="17">${escapeHTML(wordA)} ↔ ${escapeHTML(wordB)}</text>
      ${grid
        .map((v) => {
          const yy = y(v);
          return `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${width - pad.r}" y2="${yy}"></line>
            <text class="tick-label ytick" x="${pad.l - 12}" y="${yy + 4}">${fmt(v)}</text>`;
        })
        .join("")}
      ${path ? `<path class="line-path" d="${path}"></path>` : ""}
      ${rows
        .map((row, i) => {
          const xx = x(i);
          const tick = `<text class="tick-label xtick" x="${xx}" y="${height - 16}" transform="rotate(-35 ${xx} ${height - 16})">${escapeHTML(row.model)}</text>`;
          if (row.similarity === null) return tick;
          const yy = y(row.similarity);
          return `<circle class="line-dot" cx="${xx}" cy="${yy}" r="5"></circle>${tick}`;
        })
        .join("")}
      <text class="axis-name" x="14" y="${height / 2}" transform="rotate(-90 14 ${height / 2})">Similarity</text>
    </svg>`;
}

function openPaper() {
  const paper = findPaper($("paperSearch").value);
  state.selectedPaper = paper;
  if (!paper) {
    $("paperInfo").innerHTML = `<div class="empty">No matching paper found.</div>`;
    $("paperTreemap").innerHTML = `<div class="empty">Topic proportions will appear here.</div>`;
    $("similarPapers").innerHTML = `<div class="empty">Open a paper to see similar papers.</div>`;
    return;
  }
  $("paperInfo").classList.remove("empty");
  $("paperInfo").innerHTML = `
    <h3>${escapeHTML(paper.title)}</h3>
    <p>${escapeHTML(paper.authors.join(", ") || "Unknown authors")}</p>
    <p><strong>${escapeHTML(paper.journal)}</strong> (${paper.year})</p>
    <p>${paper.doi ? `<a href="https://doi.org/${escapeHTML(paper.doi)}" target="_blank">${escapeHTML(paper.doi)}</a>` : `<a href="${escapeHTML(paper.url)}" target="_blank">${escapeHTML(paper.url)}</a>`}</p>
  `;
  renderTreemap(paper.topics);
  showSimilarPapers();
}

function binaryTreemap(items, rect, out) {
  if (!items.length) return;
  if (items.length === 1) {
    out.push({ topic: items[0].topic, ...rect });
    return;
  }
  const total = items.reduce((sum, item) => sum + item.weight, 0);
  let split = 1;
  let running = items[0].weight;
  let best = Math.abs(total / 2 - running);
  for (let i = 1; i < items.length - 1; i += 1) {
    running += items[i].weight;
    const diff = Math.abs(total / 2 - running);
    if (diff < best) {
      best = diff;
      split = i + 1;
    }
  }
  const first = items.slice(0, split);
  const second = items.slice(split);
  const firstWeight = first.reduce((sum, item) => sum + item.weight, 0);
  const ratio = firstWeight / total;
  if (rect.w >= rect.h) {
    const w1 = rect.w * ratio;
    binaryTreemap(first, { x: rect.x, y: rect.y, w: w1, h: rect.h }, out);
    binaryTreemap(second, { x: rect.x + w1, y: rect.y, w: rect.w - w1, h: rect.h }, out);
  } else {
    const h1 = rect.h * ratio;
    binaryTreemap(first, { x: rect.x, y: rect.y, w: rect.w, h: h1 }, out);
    binaryTreemap(second, { x: rect.x, y: rect.y + h1, w: rect.w, h: rect.h - h1 }, out);
  }
}

function renderTreemap(topics) {
  const allTopics = topics.slice().sort((a, b) => b.value - a.value);
  const max = Math.max(...allTopics.map((t) => t.value), 0.001);
  const treemap = $("paperTreemap");
  const width = Math.max(520, Math.floor(treemap.clientWidth || 720));
  const height = Math.max(320, Math.round(width * 0.52));
  const items = allTopics.map((topic) => ({
    topic,
    weight: Math.max(topic.value, 0.00004),
  }));
  const tiles = [];
  binaryTreemap(items, { x: 0, y: 0, w: width, h: height }, tiles);
  $("paperTreemap").classList.remove("empty");
  $("topicClickInfo").textContent = "Click a topic block for details.";
  treemap.style.width = `${width}px`;
  treemap.style.height = `${height}px`;
  treemap.innerHTML = tiles
    .map((tile) => {
      const { topic } = tile;
      const gap = 2;
      const x = tile.x + gap;
      const y = tile.y + gap;
      const w = Math.max(1, tile.w - gap * 2);
      const h = Math.max(1, tile.h - gap * 2);
      const area = w * h;
      const intensity = Math.min(1, Math.sqrt(topic.value / max));
      const tileColor = mixColor("#CBDCEB", "#6D94C5", intensity * 0.85);
      const showLabel = area >= 4200 && w > 70 && h > 42;
      const showScore = area >= 2600 && w > 52 && h > 34;
      return `<button class="tile" data-topic-label="${escapeHTML(topic.name)}" data-topic-value="${fmt(topic.value * 100, 3)}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;background:${tileColor};">
        ${showLabel ? `<strong>${escapeHTML(topic.label)}</strong>` : ""}${showScore ? `<span>${fmt(topic.value * 100, 2)}%</span>` : ""}
      </button>`;
    })
    .join("");
  const tooltip = ensureTooltip();
  $("paperTreemap").querySelectorAll(".tile").forEach((tile) => {
    const label = `${tile.dataset.topicLabel}: ${tile.dataset.topicValue}%`;
    tile.addEventListener("click", () => {
      $("topicClickInfo").textContent = label;
    });
    tile.addEventListener("mouseenter", () => {
      $("topicClickInfo").textContent = label;
      tooltip.textContent = label;
      tooltip.classList.add("visible");
    });
    tile.addEventListener("mousemove", (event) => {
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
    });
    tile.addEventListener("mouseleave", () => {
      tooltip.classList.remove("visible");
    });
  });
}

function showSimilarPapers() {
  if (!state.selectedPaper) openPaper();
  if (!state.selectedPaper) return;
  const rows = topPapersByVector(paperVector(state.selectedPaper), 10, state.selectedPaper.id);
  $("similarPapers").classList.remove("empty");
  renderTable($("similarPapers"), rows, paperColumns());
}

function renderTopics() {
  $("topicList").innerHTML = state.topics
    .map(
      (topic) => `<details>
        <summary>Topic ${topic.id}: ${escapeHTML(topic.label)}</summary>
        <div class="topic-body">
          <h3>Prob Words</h3>
          <div class="word-pills">${topic.prob.map((w) => `<span class="pill">${escapeHTML(w)}</span>`).join("")}</div>
          <h3>FREX Words</h3>
          <div class="word-pills">${topic.frex.map((w) => `<span class="pill green">${escapeHTML(w)}</span>`).join("")}</div>
          <button data-topic="${topic.id}">Show Top Papers</button>
          <div id="topicPapers${topic.id}"></div>
        </div>
      </details>`,
    )
    .join("");
  $("topicList").querySelectorAll("button[data-topic]").forEach((button) => {
    button.addEventListener("click", () => showTopicPapers(Number(button.dataset.topic)));
  });
}

function showTopicPapers(topicId) {
  const topN = Number($("topicTopN").value);
  const prefix = `Topic ${topicId}:`;
  const rows = state.papers
    .map((paper) => {
      const topic = paper.topics.find((t) => t.name.startsWith(prefix));
      return { ...paper, proportion: topic ? topic.value : 0 };
    })
    .sort((a, b) => b.proportion - a.proportion)
    .slice(0, topN);
  renderTable($(`topicPapers${topicId}`), rows, [
    { label: "Title", render: (r) => escapeHTML(r.title) },
    { label: "Journal", render: (r) => escapeHTML(r.journal) },
    { label: "Year", key: "year", num: true },
    { label: "Topic Proportion", num: true, render: (r) => fmt(r.proportion, 4) },
    { label: "URL", render: (r) => (r.url ? `<a href="${escapeHTML(r.url)}" target="_blank">Open</a>` : "") },
  ]);
}

async function projectTopics() {
  const model = await loadModel($("projectionModel").value);
  const pos = $("positivePole").value.trim().toLowerCase();
  const neg = $("negativePole").value.trim().toLowerCase();
  const vPos = getWordVector(model, pos);
  const vNeg = getWordVector(model, neg);
  if (!vPos || !vNeg) {
    $("projectionChart").innerHTML = `<div class="empty">One or both pole words are missing from this model.</div>`;
    return;
  }
  $("projectionAxis").textContent = `${neg.toUpperCase()}  ←          →  ${pos.toUpperCase()}`;
  const raw = new Float32Array(model.vectorSize);
  for (let i = 0; i < raw.length; i += 1) raw[i] = vPos[i] - vNeg[i];
  const dim = normalize(raw);
  const rows = state.topics
    .map((topic, i) => ({
      topic: `Topic ${topic.id}: ${topic.label}`,
      projection: dot(dim, vectorSlice(state.topicVectors, i, state.topicVectorSize)),
    }))
    .sort((a, b) => a.projection - b.projection);
  renderBars($("projectionChart"), rows, "topic", "projection", { diverging: true, scoreLabel: "Projection Score" });
}

function setupTabs() {
  const activateTab = (tabName) => {
    const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
    const panel = $(tabName);
    if (!tab || !panel) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    panel.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activateTab(tab.dataset.tab);
      history.replaceState(null, "", `#${tab.dataset.tab}`);
    });
  });

  const initialTab = window.location.hash.replace("#", "");
  if (initialTab) activateTab(initialTab);
}

function fillSelect(select, values) {
  select.innerHTML = values.map((v) => `<option value="${escapeHTML(v)}">${escapeHTML(v)}</option>`).join("");
}

function fillDatalist(list, values, limit = values.length) {
  list.innerHTML = values.slice(0, limit).map((v) => `<option value="${escapeHTML(v)}"></option>`).join("");
}

async function init() {
  try {
    setupTabs();
    state.modelTags = await fetchJSON("data/model_tags.json");
    const paperData = await fetchJSON("data/papers.json");
    const topicData = await fetchJSON("data/topics.json");
    state.papers = paperData.papers;
    state.paperVectorSize = paperData.vectorSize;
    state.paperVectors = await fetchFloat32("data/paper_embeddings.bin");
    state.topics = topicData.topics;
    state.topicVectorSize = topicData.vectorSize;
    state.topicVectors = await fetchFloat32("data/topic_embeddings.bin");

    fillSelect($("wordModel"), state.modelTags);
    fillSelect($("projectionModel"), state.modelTags);
    const fullModel = await loadModel("full");
    fillDatalist($("vocabList"), fullModel.words);
    fillDatalist($("projectionVocabList"), fullModel.words);
    fillDatalist($("paperList"), state.papers.map(paperDisplay));
    renderTopics();

    $("wordModel").addEventListener("change", async () => {
      const model = await loadModel($("wordModel").value);
      fillDatalist($("vocabList"), model.words);
    });
    $("projectionModel").addEventListener("change", async () => {
      const model = await loadModel($("projectionModel").value);
      fillDatalist($("projectionVocabList"), model.words);
    });

    $("showNeighbors").addEventListener("click", showNeighbors);
    $("showWordPapers").addEventListener("click", showWordPapers);
    $("showDrift").addEventListener("click", showDrift);
    $("runEquation").addEventListener("click", runEquation);
    $("selectPaper").addEventListener("click", openPaper);
    $("paperSearch").addEventListener("change", openPaper);
    $("projectTopics").addEventListener("click", projectTopics);

    setStatus("Ready");
  } catch (error) {
    console.error(error);
    setStatus("Load failed");
    document.querySelector("main").insertAdjacentHTML(
      "afterbegin",
      `<div class="surface"><strong>Could not load the static data.</strong><p class="note">${escapeHTML(error.message)}. Run this site from a local web server rather than opening the HTML file directly.</p></div>`,
    );
  }
}

init();
