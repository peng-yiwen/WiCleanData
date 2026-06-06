function normalizeConstraints(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (Array.isArray(payload.constraints)) {
    return payload.constraints;
  }

  if (payload.relation && (payload.subject_types || payload.object_types)) {
    return [payload];
  }

  throw new Error("Unsupported JSON format.");
}

let currentConstraints = [];

function asArray(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    return [value];
  }
  return [];
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function parseTypeEntry(typeEntry) {
  if (typeof typeEntry !== "string") {
    return { qid: null, label: String(typeEntry || ""), href: null };
  }

  const trimmed = typeEntry.trim();
  const wdMatch = trimmed.match(/^wd:(Q\d+)\b(?:\s+(.*))?$/i);
  if (wdMatch) {
    const qid = wdMatch[1].toUpperCase();
    const label = wdMatch[2] ? wdMatch[2].trim() : "";
    return {
      qid,
      label,
      href: `https://www.wikidata.org/wiki/${qid}`
    };
  }

  return { qid: null, label: trimmed, href: null };
}

function relationToLink(relation, relationLabel) {
  if (typeof relation !== "string") {
    return { text: String(relation || ""), href: null };
  }

  const trimmed = relation.trim();
  const wikidataProperty = trimmed.match(/^(wdt:|wd:)?(P\d+)$/i);
  if (wikidataProperty) {
    const pid = wikidataProperty[2].toUpperCase();
    const lab = relationLabel && String(relationLabel).trim();
    const text = lab ? `${pid} (${lab})` : pid;
    return {
      text,
      href: `https://www.wikidata.org/wiki/Property:${pid}`
    };
  }

  const prefixed = trimmed.match(/^([a-z][a-z0-9_-]*):([^\s]+)$/i);
  if (prefixed) {
    const prefix = prefixed[1].toLowerCase();
    const localName = prefixed[2];
    const prefixMap = {
      schema: "https://schema.org/",
      yago: "http://yago-knowledge.org/resource/",
      wdt: "http://www.wikidata.org/prop/direct/",
      wd: "http://www.wikidata.org/entity/"
    };
    if (prefixMap[prefix]) {
      return {
        text: trimmed,
        href: `${prefixMap[prefix]}${localName}`
      };
    }
  }

  const httpMatch = trimmed.match(/^https?:\/\/\S+$/i);
  if (httpMatch) {
    return { text: trimmed, href: trimmed };
  }

  return { text: trimmed, href: null };
}

function renderCellList(types) {
  if (types.length === 0) {
    return "<span class=\"placeholder\">(empty)</span>";
  }

  const listItems = types
    .map(parseTypeEntry)
    .map((entry) => {
      if (entry.qid && entry.href) {
        const labelPart = entry.label ? ` <span class="label-text">(${escapeHtml(entry.label)})</span>` : "";
        return `<li class="cell-item"><a href="${escapeHtml(entry.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(entry.qid)}</a>${labelPart}</li>`;
      }
      return `<li class="cell-item">${escapeHtml(entry.label)}</li>`;
    })
    .join("");

  return `<ul class="cell-list">${listItems}</ul>`;
}

function renderConstraints(constraints, containerId = "constraints-container", options = {}) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }
  container.innerHTML = "";

  if (constraints.length === 0) {
    container.innerHTML = "<p class=\"placeholder\">No constraints found in file.</p>";
    return;
  }

  const domainHeader = options.domainHeader || "Domain";
  const showRelationCol = !options.hideRelationColumn;

  const table = document.createElement("table");
  table.className = "constraints-table";
  const headerCells = showRelationCol
    ? `<th>${escapeHtml(domainHeader)}</th><th>Relation</th><th>Range</th>`
    : `<th class="header-relation">${escapeHtml(domainHeader)}</th><th>Range</th>`;
  table.innerHTML = `<thead><tr>${headerCells}</tr></thead>`;

  const tbody = document.createElement("tbody");
  constraints.forEach((constraint, idx) => {
    const subjectTypes = asArray(constraint.subject_types);
    const objectTypes = asArray(constraint.object_types);
    const relationEntry = relationToLink(
      constraint.relation || `relation_${idx + 1}`,
      constraint.relation_label
    );

    const tr = document.createElement("tr");
    if (showRelationCol) {
      tr.innerHTML = [
        `<td>${renderCellList(subjectTypes)}</td>`,
        "<td>",
        relationEntry.href
          ? `<a class="relation-link" href="${escapeHtml(relationEntry.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(relationEntry.text)}</a>`
          : `<span class="relation-text">${escapeHtml(relationEntry.text)}</span>`,
        "</td>",
        `<td>${renderCellList(objectTypes)}</td>`
      ].join("");
    } else {
      tr.innerHTML = [
        `<td>${renderCellList(subjectTypes)}</td>`,
        `<td>${renderCellList(objectTypes)}</td>`
      ].join("");
    }
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.appendChild(table);
}

function setStatus(message, kind) {
  const node = document.getElementById("status-message");
  if (!node) {
    return;
  }
  node.textContent = message;
  node.className = "sample-status";
  if (kind) {
    node.classList.add(kind);
  }
}

function collectQidsFromConstraints(constraints) {
  const qids = new Set();
  for (const c of constraints) {
    for (const t of [...asArray(c.subject_types), ...asArray(c.object_types)]) {
      const p = parseTypeEntry(typeof t === "string" ? t : String(t));
      if (p.qid) {
        qids.add(p.qid);
      }
    }
  }
  return [...qids];
}

async function fetchLabelsForQids(qids) {
  if (qids.length === 0) {
    return {};
  }
  try {
    const res = await fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ qids })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || res.statusText);
    }
    return data.labels || {};
  } catch (e) {
    console.warn("Labels API:", e);
    return {};
  }
}

function enrichTypesList(types, labelMap) {
  return asArray(types).map((t) => {
    if (typeof t !== "string") {
      return t;
    }
    const p = parseTypeEntry(t);
    if (!p.qid) {
      return t;
    }
    const w = labelMap[p.qid];
    if (p.label && p.label.trim()) {
      return t;
    }
    if (!w || w === p.qid) {
      return t;
    }
    return `wd:${p.qid} ${w}`;
  });
}

function enrichConstraints(constraints, labelMap) {
  return constraints.map((c) => ({
    ...c,
    subject_types: enrichTypesList(c.subject_types, labelMap),
    object_types: enrichTypesList(c.object_types, labelMap)
  }));
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    setStatus("No file selected.", "error");
    return;
  }

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const raw = String(reader.result || "");
      const parsed = JSON.parse(raw);
      let constraints = normalizeConstraints(parsed);
      setStatus("Resolving labels…", "");
      const qids = collectQidsFromConstraints(constraints);
      const labelMap = await fetchLabelsForQids(qids);
      constraints = enrichConstraints(constraints, labelMap);
      currentConstraints = constraints;
      renderConstraints(currentConstraints);
      setStatus(`Loaded ${constraints.length} constraint(s) from ${file.name}.`, "success");
    } catch (error) {
      setStatus(`Failed to parse JSON: ${error.message}`, "error");
      currentConstraints = [];
      renderConstraints(currentConstraints);
    }
  };

  reader.onerror = () => {
    setStatus("Failed to read file.", "error");
  };

  reader.readAsText(file, "utf-8");
}

const fileInput = document.getElementById("file-input");
if (fileInput) {
  fileInput.addEventListener("change", handleFileUpload);
}

// ── API: model list + relation summary ─────────────────────────────────────

function setApiStatus(message, kind) {
  const node = document.getElementById("api-status");
  if (!node) {
    return;
  }
  node.textContent = message || "";
  node.className = "sample-status api-status-line";
  if (kind) {
    node.classList.add(kind);
  }
}

async function loadModels() {
  const select = document.getElementById("model-select");
  if (!select) {
    return;
  }
  select.innerHTML = "";
  try {
    const res = await fetch("/api/models");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const models = Array.isArray(data.models) ? data.models : [];
    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No models found";
      select.appendChild(opt);
      setApiStatus("No *_wikc.txt + *_mapping.txt pairs found beside the server or in taxonomy_viewer/data.", "error");
      return;
    }
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select model…";
    select.appendChild(placeholder);
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    }
    setApiStatus(`${models.length} model(s) available.`, "success");
  } catch (e) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Failed to load";
    select.appendChild(opt);
    setApiStatus(`Could not load /api/models (is server.py running?). ${e.message}`, "error");
  }
}

function showApiResults(data) {
  const block = document.getElementById("api-results");
  const pre = document.getElementById("summary-pre");
  if (block) {
    block.classList.remove("hidden");
  }
  if (pre && data.summary != null) {
    pre.textContent = data.summary;
  }
  const rel = data.relation || "";
  const relLabel = data.relation_label || "";
  const domain = Array.isArray(data.domain_types) ? data.domain_types : [];
  const range = Array.isArray(data.range_types) ? data.range_types : [];

  const relationEntry = relationToLink(rel, relLabel);
  const headerText = relationEntry.text || rel;

  renderConstraints(
    [
      {
        relation: rel,
        relation_label: relLabel,
        subject_types: domain,
        object_types: range
      }
    ],
    "final-constraints-container",
    { domainHeader: headerText, hideRelationColumn: true }
  );
}

async function runSummary() {
  const model = (document.getElementById("model-select") || {}).value || "";
  let relation = (document.getElementById("relation-input") || {}).value || "";
  relation = relation.trim().toUpperCase();
  if (!model) {
    setApiStatus("Select a model.", "error");
    return;
  }
  if (!relation || !/^P\d+$/.test(relation)) {
    setApiStatus("Enter a Wikidata property id (e.g. P800).", "error");
    return;
  }
  setApiStatus("Loading…", "");
  try {
    const q = new URLSearchParams({ model, relation });
    const res = await fetch(`/api/summary?${q.toString()}`);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || res.statusText);
    }
    showApiResults(data);
    setApiStatus(`OK — ${data.model} / ${data.relation}`, "success");
  } catch (e) {
    setApiStatus(e.message || String(e), "error");
  }
}

const runBtn = document.getElementById("run-summary");
if (runBtn) {
  runBtn.addEventListener("click", runSummary);
}
const relInput = document.getElementById("relation-input");
if (relInput) {
  relInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      runSummary();
    }
  });
}

// ── Tab panels: Constraint summary | Upload JSON ───────────────────────────

function activateMainTab(tabId) {
  const tabSummary = document.getElementById("tab-summary");
  const tabUpload = document.getElementById("tab-upload");
  const panelSummary = document.getElementById("panel-summary");
  const panelUpload = document.getElementById("panel-upload");
  if (!tabSummary || !tabUpload || !panelSummary || !panelUpload) {
    return;
  }

  const isUpload = tabId === "upload";

  tabSummary.classList.toggle("is-active", !isUpload);
  tabSummary.setAttribute("aria-selected", String(!isUpload));
  tabSummary.tabIndex = isUpload ? -1 : 0;

  tabUpload.classList.toggle("is-active", isUpload);
  tabUpload.setAttribute("aria-selected", String(isUpload));
  tabUpload.tabIndex = isUpload ? 0 : -1;

  panelSummary.classList.toggle("hidden", isUpload);
  panelSummary.hidden = isUpload;

  panelUpload.classList.toggle("hidden", !isUpload);
  panelUpload.hidden = !isUpload;

  if (history.replaceState) {
    const path = location.pathname + location.search;
    history.replaceState(null, "", isUpload ? `${path}#upload` : path);
  }
}

function initMainTabs() {
  const tabSummary = document.getElementById("tab-summary");
  const tabUpload = document.getElementById("tab-upload");
  if (!tabSummary || !tabUpload) {
    return;
  }

  tabSummary.addEventListener("click", () => activateMainTab("summary"));
  tabUpload.addEventListener("click", () => activateMainTab("upload"));

  function applyHash() {
    if (location.hash === "#upload") {
      activateMainTab("upload");
    } else {
      activateMainTab("summary");
    }
  }

  applyHash();
  window.addEventListener("hashchange", applyHash);
}

initMainTabs();

if (document.getElementById("model-select")) {
  loadModels();
}

// ── Export: SVG / PDF for the final constraints table ───────────────────────

function getConstraintsData() {
  const container = document.getElementById("final-constraints-container");
  if (!container) return null;
  const table = container.querySelector("table.constraints-table");
  if (!table) return null;

  const thead = table.querySelector("thead");
  const headerTh = thead ? thead.querySelector("th") : null;
  const relation = headerTh ? headerTh.textContent.trim() : "";

  const rows = [];
  const tbody = table.querySelector("tbody");
  if (!tbody) return null;

  for (const tr of tbody.querySelectorAll("tr")) {
    const cells = tr.querySelectorAll("td");
    if (cells.length < 2) continue;
    const domain = [];
    const range = [];
    cells[0].querySelectorAll(".cell-item").forEach(li => {
      domain.push(li.textContent.trim());
    });
    cells[1].querySelectorAll(".cell-item").forEach(li => {
      range.push(li.textContent.trim());
    });
    rows.push({ domain, relation, range });
  }
  return rows.length > 0 ? rows : null;
}

function escapeXml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildConstraintSVG(rows) {
  const fontSize = 13;
  const headerFontSize = 14;
  const lineHeight = fontSize * 1.6;
  const headerHeight = headerFontSize * 2.2;
  const colPadding = 24;
  const rowPadding = 12;
  const topMargin = 10;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = `${fontSize}px "Open Sans", sans-serif`;

  function textWidth(text) {
    return ctx.measureText(text).width;
  }

  const relationText = rows[0].relation;
  const headers = [relationText, "Range"];

  ctx.font = `bold ${headerFontSize}px "Open Sans", sans-serif`;
  let colWidths = headers.map(h => textWidth(h) + colPadding * 2);
  ctx.font = `${fontSize}px "Open Sans", sans-serif`;

  const rowHeights = [];
  for (const row of rows) {
    const domainLines = row.domain.length || 1;
    const rangeLines = row.range.length || 1;
    const maxLines = Math.max(domainLines, rangeLines, 1);
    rowHeights.push(maxLines * lineHeight + rowPadding * 2);

    const domW = row.domain.reduce((mx, t) => Math.max(mx, textWidth(t)), 0) + colPadding * 2;
    const ranW = row.range.reduce((mx, t) => Math.max(mx, textWidth(t)), 0) + colPadding * 2;
    colWidths[0] = Math.max(colWidths[0], domW);
    colWidths[1] = Math.max(colWidths[1], ranW);
  }

  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const totalHeight = topMargin + headerHeight + rowHeights.reduce((a, b) => a + b, 0) + 2;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="${totalHeight}" font-family="'Open Sans', sans-serif">\n`;
  svg += `<rect width="${totalWidth}" height="${totalHeight}" fill="white"/>\n`;

  let y = topMargin;
  svg += `<line x1="0" y1="${y + headerHeight}" x2="${totalWidth}" y2="${y + headerHeight}" stroke="#bdbdbd" stroke-width="1.5"/>\n`;
  let x = 0;
  for (let i = 0; i < 2; i++) {
    const weight = (i === 0) ? "700" : "600";
    const fill = (i === 0) ? "#1565c0" : "#333";
    svg += `<text x="${x + colPadding}" y="${y + headerHeight * 0.65}" font-size="${headerFontSize}" font-weight="${weight}" fill="${fill}">${escapeXml(headers[i])}</text>\n`;
    x += colWidths[i];
  }
  y += headerHeight;

  for (let ri = 0; ri < rows.length; ri++) {
    const row = rows[ri];
    const rh = rowHeights[ri];
    svg += `<line x1="0" y1="${y + rh}" x2="${totalWidth}" y2="${y + rh}" stroke="#e0e0e0" stroke-width="0.75"/>\n`;

    x = 0;
    const domItems = row.domain.length > 0 ? row.domain : ["(empty)"];
    for (let li = 0; li < domItems.length; li++) {
      const ty = y + rowPadding + (li + 0.75) * lineHeight;
      const fillColor = domItems[li] === "(empty)" ? "#999" : "#333";
      svg += `<text x="${x + colPadding}" y="${ty}" font-size="${fontSize}" fill="${fillColor}">${escapeXml(domItems[li])}</text>\n`;
    }
    x += colWidths[0];

    const ranItems = row.range.length > 0 ? row.range : ["(empty)"];
    for (let li = 0; li < ranItems.length; li++) {
      const ty = y + rowPadding + (li + 0.75) * lineHeight;
      const fillColor = ranItems[li] === "(empty)" ? "#999" : "#333";
      svg += `<text x="${x + colPadding}" y="${ty}" font-size="${fontSize}" fill="${fillColor}">${escapeXml(ranItems[li])}</text>\n`;
    }

    y += rh;
  }

  x = colWidths[0];
  svg += `<line x1="${x}" y1="${topMargin}" x2="${x}" y2="${totalHeight}" stroke="#e0e0e0" stroke-width="0.5"/>\n`;

  svg += `</svg>`;
  return svg;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

function exportSVG() {
  const rows = getConstraintsData();
  if (!rows) {
    alert("No constraint data to export. Run a summary first.");
    return;
  }
  const svgStr = buildConstraintSVG(rows);
  const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  downloadBlob(blob, "constraints.svg");
}

function exportPDF() {
  const rows = getConstraintsData();
  if (!rows) {
    alert("No constraint data to export. Run a summary first.");
    return;
  }
  const svgStr = buildConstraintSVG(rows);

  const parser = new DOMParser();
  const svgDoc = parser.parseFromString(svgStr, "image/svg+xml");
  const svgEl = svgDoc.documentElement;
  const w = parseFloat(svgEl.getAttribute("width"));
  const h = parseFloat(svgEl.getAttribute("height"));

  const svgBlob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  const printWin = window.open("", "_blank", `width=${Math.ceil(w + 40)},height=${Math.ceil(h + 40)}`);
  if (!printWin) {
    alert("Pop-up blocked. Please allow pop-ups, or use the SVG export and convert with Inkscape.");
    URL.revokeObjectURL(svgUrl);
    return;
  }
  printWin.document.write(`<!DOCTYPE html><html><head><title>Constraints PDF</title>
<style>@page{size:${w + 20}px ${h + 20}px;margin:10px}body{margin:0;padding:10px;}</style>
</head><body>
<img src="${svgUrl}" width="${w}" height="${h}">
<script>window.onload=function(){setTimeout(function(){window.print();window.close();},300)};<\/script></body></html>`);
  printWin.document.close();
}

const exportSvgBtn = document.getElementById("export-svg");
const exportPdfBtn = document.getElementById("export-pdf");
if (exportSvgBtn) exportSvgBtn.addEventListener("click", exportSVG);
if (exportPdfBtn) exportPdfBtn.addEventListener("click", exportPDF);
