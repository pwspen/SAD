// Entry point for the SAD trainer web page.
// Hugging-Face-ish UI: target image in the top bar, advanced options
// (mask, init-sites, candidates/lr…) in the collapsible left panel, and a
// side-by-side target vs reconstruction view, both aspect-fit.

import { loadShaders, tileSlotsForLimit } from "../trainer/shader_loader.js?v=shared-wgsl-20261017";
import { requestTrainingDevice } from "../trainer/device.js?v=shared-wgsl-20261017";
import { Trainer, buildArgs, GRAD_GROUPS } from "../trainer/trainer.js?v=shared-wgsl-20261017";
import { loadImageToFloat32, loadMaskRgba32Float, whiteMaskRgba32Float } from "../trainer/textures.js?v=shared-wgsl-20261017";
import { loadSitesFromFile, serializeSitesTxt, triggerDownload, floatImageToRgba8, computeBpp, BITS_PER_SITE } from "../trainer/io.js?v=shared-wgsl-20261017";
import { SITE_FLOATS } from "../trainer/params.js?v=shared-wgsl-20261017";

const CONFIG_URL = "../training_config.json";
const DEMO_MAX_IMAGE_SIDE = 2048;
// Kodak set lives outside this repo; the default resolves under the
// kodak-bench server (which serves the parent dir). Override with ?kodak=<dir>.
const KODAK_DIR = new URLSearchParams(location.search).get("kodak") || "../../comp/data/kodak/";
const KODAK_COUNT = 24;

// Fields that live in the advanced panel. The target image is in the top bar;
// init-sites / mask are pickers inside the panel (not listed here).
const UI_FIELDS = [
  ["warmupIters", "p-warmupIters"],
  ["decayIters", "p-decayIters"],
  ["sites", "p-sites"],
  ["logFreq", "p-logFreq"],
  ["viewerFreq", "p-viewerFreq"],
  ["candFreq", "p-candFreq"],
  ["candUpdatePasses", "p-candUpdatePasses"],
  ["candDownscale", "p-candDownscale"],
  ["candRadiusScale", "p-candRadiusScale"],
  ["candRadiusProbes", "p-candRadiusProbes"],
  ["candInjectCount", "p-candInjectCount"],
  ["candHilbertWindow", "p-candHilbertWindow"],
  ["candHilbertProbes", "p-candHilbertProbes"],
  ["lrMult", "p-lrMult"],
  ["lrPosBase", "p-lrPosBase"],
  ["lrTauBase", "p-lrTauBase"],
  ["lrRadiusBase", "p-lrRadiusBase"],
  ["lrColorBase", "p-lrColorBase"],
  ["lrDirBase", "p-lrDirBase"],
  ["lrAnisoBase", "p-lrAnisoBase"],
  ["flowStrength", "p-flowStrength"],
  ["flowStartIter", "p-flowStartIter"],
  ["flowEndIter", "p-flowEndIter"],
  ["flowMaxStep", "p-flowMaxStep"],
  ["flowEmaAlpha", "p-flowEmaAlpha"],
  ["flowGeoMix", "p-flowGeoMix"],
  ["flowPyrStrength", "p-flowPyrStrength"],
  ["flowPyrCell", "p-flowPyrCell"],
  ["flowPyrLevels", "p-flowPyrLevels"],
  ["flowPyrGamma", "p-flowPyrGamma"],
];

const state = {
  device: null,
  shaders: null,
  config: null,
  targetImage: null,     // { data, width, height }
  targetToken: 0,        // bumped per target load so stale loads are dropped
  maskData: null,        // { mask, rgba, maskSum } or null
  initSites: null,       // { data, width, height, count } or null
  running: false,
  abortCtrl: null,
  trainer: null,
  lastRender: null,      // { rgba, width, height }
  lastHashed: null,
  lastHeatmap: null,
  lastSites: null,
  lastSiteCount: 0,
  lastPsnr: null,        // persists between log ticks so preview updates don't blank the PSNR
  lastLrScale: null,     // schedule × LR mult at the last log tick
  psnrHistory: [],       // [{ it, psnr }] from log ticks, plotted in the PSNR card
  psnrIters: 0,
  psnrHover: null,       // index into psnrHistory under the cursor
  bppRuns: [],           // [{ sites, bpp, points: [{ it, psnr }] }] per run on this target; last is current
  bppHover: null,        // { run, point } under the cursor
  gradHistory: [],       // [{ it, stats }] per-param |grad| mean/max from log ticks
  gradHover: null,       // index into gradHistory under the cursor
};

const ui = {
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  imageLabel: document.getElementById("imageLabel"),
  kodakSelect: document.getElementById("kodakSelect"),
  kodakPrev: document.getElementById("kodakPrev"),
  kodakNext: document.getElementById("kodakNext"),
  maskInput: document.getElementById("maskInput"),
  maskLabel: document.getElementById("maskLabel"),
  initSitesInput: document.getElementById("initSitesInput"),
  initLabel: document.getElementById("initLabel"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  toggleParamsBtn: document.getElementById("toggleParamsBtn"),
  saveBtn: document.getElementById("saveBtn"),
  savePopover: document.getElementById("savePopover"),
  savePng: document.getElementById("savePng"),
  saveTxt: document.getElementById("saveTxt"),
  paramsPanel: document.getElementById("paramsPanel"),
  layoutSelect: document.getElementById("layoutSelect"),
  canvases: document.getElementById("canvases"),
  metricRow: document.getElementById("metricRow"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  log: document.getElementById("logPanel"),
};

function setStatus(msg) { ui.status.textContent = msg; }
function appendLog(line) {
  ui.log.textContent += line + "\n";
  ui.log.scrollTop = ui.log.scrollHeight;
}

function applyConfigToForm(config) {
  const defaults = buildArgs(config, {});
  for (const [key, elId] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    const v = defaults[key];
    if (el.type === "checkbox") el.checked = Boolean(v);
    else if (v === undefined) el.value = "";
    else el.value = v;
  }
}

function readFormArgs() {
  const overrides = {};
  for (const [key, elId] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    if (el.type === "checkbox") overrides[key] = el.checked;
    else if (el.value === "") continue;
    else overrides[key] = Number(el.value);
  }
  return buildArgs(state.config, overrides);
}

async function initDevice() {
  const { device } = await requestTrainingDevice(navigator.gpu);
  device.lost.then((info) => appendLog(`Device lost: ${info.reason} ${info.message}`));
  device.addEventListener?.("uncapturederror", (e) => {
    appendLog(`GPU error: ${e.error?.message || e.error}`);
    console.error(e.error);
  });
  return device;
}

async function loadConfig() {
  const url = new URL(CONFIG_URL, import.meta.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load training_config.json: ${res.status}`);
  return await res.json();
}

function drawToCanvas(canvas, rgbaF, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  fitCanvasToCard(canvas, width, height);
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(floatImageToRgba8(rgbaF, width, height), width, height), 0, 0);
}

function fitCanvasToCard(canvas, width = canvas.width, height = canvas.height) {
  if (!canvas || !width || !height) return;
  const card = canvas.parentElement;
  if (!card) return;
  const rect = card.getBoundingClientRect();
  const cardWidth = card.clientWidth || rect.width;
  const cardHeight = card.clientHeight || rect.height;
  if (cardWidth <= 0 || cardHeight <= 0) return;
  const scale = Math.min(cardWidth / width, cardHeight / height);
  canvas.style.width = `${Math.max(1, width * scale)}px`;
  canvas.style.height = `${Math.max(1, height * scale)}px`;
}

// ---- Views & layouts ----
// Every card in the viewer grid is a view. "image" views aspect-fit a canvas
// at the target's resolution; "plot" views fill their card and redraw at
// device resolution. `blank` is the fill shown before the first render
// (views without one are drawn straight from the target). Only views named in
// the active layout get a card; the others keep canvas = null and their draw
// functions return early, so hidden views cost nothing.
const VIEWS = {
  target:     { label: "Target", kind: "image" },
  preview:    { label: "Reconstruction + centroids", kind: "image", overlay: true, blank: "#0a0a0f" },
  residual:   { label: "Residual |recon − target|", kind: "image", blank: "#000004" },
  importance: { label: "Importance (Laplacian bands)", kind: "image" },
  siteIds:    { label: "Site IDs", kind: "image", blank: "#0a0a0f" },
  tau:        { label: "Tau heatmap", kind: "image", blank: "#0a0a0f" },
  psnr:       { label: "PSNR vs iteration", kind: "plot", draw: drawPsnrPlot, hover: handlePsnrHover,
                leave: () => { state.psnrHover = null; drawPsnrPlot(); } },
  bpp:        { label: "PSNR vs bpp", kind: "plot", draw: drawBppPlot, hover: handleBppHover,
                leave: () => { state.bppHover = null; drawBppPlot(); } },
  grad:       { label: "|gradient| vs iteration (log)", kind: "plot", draw: drawGradPlot, hover: handleGradHover,
                leave: () => { state.gradHover = null; drawGradPlot(); } },
};

// Layouts are CSS grid-template-areas: one quoted string per row, one view id
// per cell, "." for an empty cell. Repeat an id to span cells (each view must
// cover a rectangle). `rows` optionally sizes the rows; the default splits the
// height evenly. Columns are always equal width.
const LAYOUTS = {
  overview: {
    label: "Overview",
    areas: `"target preview residual importance psnr bpp"
            "grad   grad    grad     grad       grad grad"`,
    rows: "minmax(0, 1fr) 320px",
  },
  images: {
    label: "Images 2×2",
    areas: `"target   preview"
            "residual importance"`,
  },
  sites: {
    label: "Sites & tau",
    areas: `"preview  siteIds    tau"
            "residual importance psnr"`,
  },
  plots: {
    label: "Plots",
    areas: `"psnr bpp"
            "grad grad"`,
  },
};
const DEFAULT_LAYOUT = "overview";
const LAYOUT_STORAGE_KEY = "sad.train.layout";

// Parses and validates a layout's areas string into a grid of view ids.
function parseLayoutAreas(areas) {
  const grid = [...areas.matchAll(/"([^"]*)"/g)].map((m) => m[1].trim().split(/\s+/));
  if (!grid.length) throw new Error("no rows");
  const cols = grid[0].length;
  if (grid.some((row) => row.length !== cols)) throw new Error("rows have different cell counts");
  const boxes = new Map();
  grid.forEach((row, r) => row.forEach((id, c) => {
    if (id === ".") return;
    if (!VIEWS[id]) throw new Error(`unknown view "${id}"`);
    const b = boxes.get(id) || { r0: r, r1: r, c0: c, c1: c, n: 0 };
    b.r0 = Math.min(b.r0, r); b.r1 = Math.max(b.r1, r);
    b.c0 = Math.min(b.c0, c); b.c1 = Math.max(b.c1, c);
    b.n += 1;
    boxes.set(id, b);
  }));
  for (const [id, b] of boxes) {
    if (b.n !== (b.r1 - b.r0 + 1) * (b.c1 - b.c0 + 1)) throw new Error(`"${id}" is not a rectangle`);
  }
  return { grid, ids: [...boxes.keys()] };
}

function buildCards(name) {
  const layout = LAYOUTS[name];
  const { grid, ids } = parseLayoutAreas(layout.areas);
  for (const view of Object.values(VIEWS)) {
    view.card = null;
    view.canvas = null;
    view.overlayCanvas = null;
  }
  const root = ui.canvases;
  root.replaceChildren();
  root.style.setProperty("--layout-areas", grid.map((row) => `"${row.join(" ")}"`).join(" "));
  root.style.setProperty("--layout-cols", `repeat(${grid[0].length}, minmax(0, 1fr))`);
  root.style.setProperty("--layout-rows", layout.rows || `repeat(${grid.length}, minmax(0, 1fr))`);
  for (const id of ids) {
    const view = VIEWS[id];
    const card = document.createElement("div");
    card.className = `canvas-card ${view.kind}`;
    card.style.setProperty("--view", id);
    const label = document.createElement("span");
    label.className = "card-label";
    label.textContent = view.label;
    const canvas = document.createElement("canvas");
    card.append(label, canvas);
    if (view.overlay) {
      view.overlayCanvas = document.createElement("canvas");
      view.overlayCanvas.className = "overlay-canvas";
      card.append(view.overlayCanvas);
    }
    if (view.hover) canvas.addEventListener("mousemove", view.hover);
    if (view.leave) canvas.addEventListener("mouseleave", view.leave);
    root.append(card);
    view.card = card;
    view.canvas = canvas;
  }
}

async function setLayout(name) {
  if (!LAYOUTS[name]) name = DEFAULT_LAYOUT;
  try {
    buildCards(name);
  } catch (err) {
    appendLog(`Layout "${name}" is invalid (${err.message}); using ${DEFAULT_LAYOUT}.`);
    name = DEFAULT_LAYOUT;
    buildCards(name);
  }
  ui.layoutSelect.value = name;
  try { localStorage.setItem(LAYOUT_STORAGE_KEY, name); } catch {}
  // Fresh canvases are empty: repaint everything we have.
  if (state.targetImage) {
    const { width, height } = state.targetImage;
    blankRenderViews(width, height);
    drawTarget();
    drawImportance();
  } else {
    updateCanvasLayout();
  }
  await drawView();
}

function savedLayout() {
  try { return localStorage.getItem(LAYOUT_STORAGE_KEY) || DEFAULT_LAYOUT; } catch { return DEFAULT_LAYOUT; }
}

// Fills render-driven image views with their placeholder at the target size so
// the layout matches before the first render arrives.
function blankRenderViews(width, height) {
  for (const view of Object.values(VIEWS)) {
    if (!view.canvas || !view.blank) continue;
    view.canvas.width = width;
    view.canvas.height = height;
    const ctx = view.canvas.getContext("2d");
    ctx.fillStyle = view.blank;
    ctx.fillRect(0, 0, width, height);
  }
  const overlay = VIEWS.preview.overlayCanvas;
  if (overlay) {
    overlay.width = width;
    overlay.height = height;
  }
}

function updateCanvasLayout(width = state.targetImage?.width, height = state.targetImage?.height) {
  for (const view of Object.values(VIEWS)) {
    if (!view.canvas) continue;
    if (view.kind === "image") fitCanvasToCard(view.canvas, width || view.canvas.width, height || view.canvas.height);
    else view.draw();
  }
  positionOverlay();
}

// ---- PSNR vs iteration plot ----

const PSNR_PAD = { left: 46, right: 16, top: 42, bottom: 34 };
const PSNR_LINE_COLOR = "#e0611a"; // darker accent: >= 3:1 against --canvas-bg

function niceTicks(min, max, count = 5) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

function psnrPlotGeometry() {
  const canvas = VIEWS.psnr.canvas;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const data = state.psnrHistory;
  const xMax = Math.max(1, state.psnrIters, data.length ? data[data.length - 1].it : 0);
  let yMin = Infinity, yMax = -Infinity;
  for (const p of data) {
    yMin = Math.min(yMin, p.psnr);
    yMax = Math.max(yMax, p.psnr);
  }
  if (!Number.isFinite(yMin)) { yMin = 0; yMax = 40; }
  const yPad = Math.max(0.5, (yMax - yMin) * 0.08);
  yMin = Math.floor(yMin - yPad);
  yMax = Math.ceil(yMax + yPad);
  const plotW = Math.max(1, w - PSNR_PAD.left - PSNR_PAD.right);
  const plotH = Math.max(1, h - PSNR_PAD.top - PSNR_PAD.bottom);
  return {
    w, h, xMax, yMin, yMax,
    x: (it) => PSNR_PAD.left + (it / xMax) * plotW,
    y: (v) => PSNR_PAD.top + (1 - (v - yMin) / (yMax - yMin)) * plotH,
  };
}

function drawPsnrPlot() {
  const canvas = VIEWS.psnr.canvas;
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const lineCol = css.getPropertyValue("--line").trim() || "#e6e6df";
  const dimCol = css.getPropertyValue("--text-dim").trim() || "#66666e";
  const softCol = css.getPropertyValue("--text-soft").trim() || "#9a9aa2";
  const textCol = css.getPropertyValue("--text").trim() || "#1d1d1f";
  const surface = css.getPropertyValue("--canvas-bg").trim() || "#f1f1ec";

  const data = state.psnrHistory;
  const g = psnrPlotGeometry();
  const left = PSNR_PAD.left, right = w - PSNR_PAD.right;
  const top = PSNR_PAD.top, bottom = h - PSNR_PAD.bottom;

  ctx.font = "10.5px Inter, system-ui, -apple-system, sans-serif";
  ctx.lineWidth = 1;

  // Horizontal grid + y labels
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of niceTicks(g.yMin, g.yMax, Math.max(2, Math.floor((bottom - top) / 48)))) {
    const y = Math.round(g.y(v)) + 0.5;
    ctx.strokeStyle = lineCol;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = softCol;
    ctx.fillText(String(Math.round(v * 10) / 10), left - 6, y);
  }

  // Baseline + x labels
  ctx.strokeStyle = softCol;
  ctx.beginPath(); ctx.moveTo(left, bottom + 0.5); ctx.lineTo(right, bottom + 0.5); ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = softCol;
  for (const v of niceTicks(0, g.xMax, Math.max(2, Math.floor((right - left) / 70)))) {
    ctx.fillText(v >= 10000 ? `${v / 1000}k` : String(v), g.x(v), bottom + 6);
  }

  // Axis titles
  ctx.fillStyle = dimCol;
  ctx.fillText("iteration", (left + right) / 2, bottom + 20);
  ctx.save();
  ctx.translate(12, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText("PSNR (dB)", 0, 0);
  ctx.restore();

  if (!data.length) {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = softCol;
    ctx.fillText(state.running ? "Waiting for first log tick…" : "Train to see PSNR", (left + right) / 2, (top + bottom) / 2);
    return;
  }

  // Series line
  ctx.strokeStyle = PSNR_LINE_COLOR;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  data.forEach((p, i) => {
    const x = g.x(p.it), y = g.y(p.psnr);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const drawMarker = (p) => {
    ctx.beginPath();
    ctx.arc(g.x(p.it), g.y(p.psnr), 4, 0, Math.PI * 2);
    ctx.fillStyle = PSNR_LINE_COLOR;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = surface;
    ctx.stroke();
  };

  const hover = state.psnrHover != null ? data[state.psnrHover] : null;
  if (!hover) {
    drawMarker(data[data.length - 1]);
    return;
  }

  // Crosshair + tooltip
  const hx = Math.round(g.x(hover.it)) + 0.5;
  ctx.strokeStyle = softCol;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(hx, top); ctx.lineTo(hx, bottom); ctx.stroke();
  ctx.setLineDash([]);
  drawMarker(hover);

  const label = `iter ${hover.it} · ${hover.psnr.toFixed(2)} dB`;
  ctx.font = "11px Inter, system-ui, -apple-system, sans-serif";
  const tw = ctx.measureText(label).width + 14;
  const th = 22;
  let tx = hx + 10;
  if (tx + tw > right) tx = hx - 10 - tw;
  const ty = Math.max(top, Math.min(bottom - th, g.y(hover.psnr) - th - 8));
  ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
  ctx.strokeStyle = lineCol;
  ctx.beginPath();
  ctx.roundRect(tx, ty, tw, th, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = textCol;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, tx + 7, ty + th / 2);
}

function handlePsnrHover(ev) {
  const data = state.psnrHistory;
  if (!data.length) return;
  const rect = VIEWS.psnr.canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const g = psnrPlotGeometry();
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < data.length; i++) {
    const d = Math.abs(g.x(data[i].it) - mx);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  if (state.psnrHover !== best) {
    state.psnrHover = best;
    drawPsnrPlot();
  }
}

// ---- PSNR vs bpp plot (log x) ----
// Raw rate: active sites × BITS_PER_SITE / pixels. Runs on the current target
// accumulate: earlier runs show as their final point (joined in bpp order, an
// RD curve); the current run shows its log-tick trail at its fixed bpp.

const BPP_PAST_COLOR = "#66666e";

function logTicks(min, max) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9)) ticks.push(v);
    }
  }
  return ticks;
}

function fmtBpp(v) {
  return v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : String(Number(v.toPrecision(2)));
}

// Plotted points: { x: bpp, y: psnr, run, point, current }.
function bppPlotPoints() {
  const runs = state.bppRuns;
  const pts = [];
  runs.forEach((r, ri) => {
    const current = ri === runs.length - 1;
    if (current) {
      r.points.forEach((p, pi) => pts.push({ x: r.bpp, y: p.psnr, run: ri, point: pi, current }));
    } else if (r.points.length) {
      const pi = r.points.length - 1;
      pts.push({ x: r.bpp, y: r.points[pi].psnr, run: ri, point: pi, current });
    }
  });
  return pts;
}

function bppPlotGeometry(pts) {
  const canvas = VIEWS.bpp.canvas;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const r of state.bppRuns) {
    xMin = Math.min(xMin, r.bpp);
    xMax = Math.max(xMax, r.bpp);
  }
  for (const p of pts) {
    yMin = Math.min(yMin, p.y);
    yMax = Math.max(yMax, p.y);
  }
  if (!(xMin > 0)) { xMin = 0.1; xMax = 10; }
  // At least a decade of x range, centred on the data.
  const span = Math.max(1, Math.log10(xMax / xMin) * 1.2);
  const mid = (Math.log10(xMin) + Math.log10(xMax)) / 2;
  const lx0 = mid - span / 2, lx1 = mid + span / 2;
  if (!Number.isFinite(yMin)) { yMin = 0; yMax = 40; }
  const yPad = Math.max(0.5, (yMax - yMin) * 0.08);
  yMin = Math.floor(yMin - yPad);
  yMax = Math.ceil(yMax + yPad);
  const plotW = Math.max(1, w - PSNR_PAD.left - PSNR_PAD.right);
  const plotH = Math.max(1, h - PSNR_PAD.top - PSNR_PAD.bottom);
  return {
    w, h, xMin: 10 ** lx0, xMax: 10 ** lx1, yMin, yMax,
    x: (v) => PSNR_PAD.left + ((Math.log10(v) - lx0) / (lx1 - lx0)) * plotW,
    y: (v) => PSNR_PAD.top + (1 - (v - yMin) / (yMax - yMin)) * plotH,
  };
}

function drawBppPlot() {
  const canvas = VIEWS.bpp.canvas;
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const lineCol = css.getPropertyValue("--line").trim() || "#e6e6df";
  const dimCol = css.getPropertyValue("--text-dim").trim() || "#66666e";
  const softCol = css.getPropertyValue("--text-soft").trim() || "#9a9aa2";
  const textCol = css.getPropertyValue("--text").trim() || "#1d1d1f";
  const surface = css.getPropertyValue("--canvas-bg").trim() || "#f1f1ec";

  const pts = bppPlotPoints();
  const g = bppPlotGeometry(pts);
  const left = PSNR_PAD.left, right = w - PSNR_PAD.right;
  const top = PSNR_PAD.top, bottom = h - PSNR_PAD.bottom;

  ctx.font = "10.5px Inter, system-ui, -apple-system, sans-serif";
  ctx.lineWidth = 1;

  // Horizontal grid + y labels
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of niceTicks(g.yMin, g.yMax, Math.max(2, Math.floor((bottom - top) / 48)))) {
    const y = Math.round(g.y(v)) + 0.5;
    ctx.strokeStyle = lineCol;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = softCol;
    ctx.fillText(String(Math.round(v * 10) / 10), left - 6, y);
  }

  // Baseline + log x labels (thin out 2/5 ticks when crowded)
  ctx.strokeStyle = softCol;
  ctx.beginPath(); ctx.moveTo(left, bottom + 0.5); ctx.lineTo(right, bottom + 0.5); ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = softCol;
  let xt = logTicks(g.xMin, g.xMax);
  if (xt.length > Math.max(2, Math.floor((right - left) / 40))) {
    xt = xt.filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
  }
  for (const v of xt) ctx.fillText(fmtBpp(v), g.x(v), bottom + 6);

  // Axis titles
  ctx.fillStyle = dimCol;
  ctx.fillText(`bpp (${BITS_PER_SITE} bits/site)`, (left + right) / 2, bottom + 20);
  ctx.save();
  ctx.translate(12, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText("PSNR (dB)", 0, 0);
  ctx.restore();

  if (!pts.length) {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = softCol;
    ctx.fillText(state.running ? "Waiting for first log tick…" : "Train to see PSNR vs bpp", (left + right) / 2, (top + bottom) / 2);
    return;
  }

  const drawMarker = (p, color, r = 4) => {
    ctx.beginPath();
    ctx.arc(g.x(p.x), g.y(p.y), r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = surface;
    ctx.stroke();
  };

  // Earlier runs: final points joined in bpp order
  const past = pts.filter((p) => !p.current).sort((a, b) => a.x - b.x);
  if (past.length > 1) {
    ctx.strokeStyle = BPP_PAST_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    past.forEach((p, i) => (i === 0 ? ctx.moveTo(g.x(p.x), g.y(p.y)) : ctx.lineTo(g.x(p.x), g.y(p.y))));
    ctx.stroke();
  }
  for (const p of past) drawMarker(p, BPP_PAST_COLOR, 3.5);

  // Current run: vertical trail of log ticks, latest marked
  const cur = pts.filter((p) => p.current);
  if (cur.length) {
    ctx.strokeStyle = PSNR_LINE_COLOR;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    const cx = g.x(cur[0].x);
    const ys = cur.map((p) => g.y(p.y));
    ctx.beginPath(); ctx.moveTo(cx, Math.min(...ys)); ctx.lineTo(cx, Math.max(...ys)); ctx.stroke();
    drawMarker(cur[cur.length - 1], PSNR_LINE_COLOR);
  }

  const hover = state.bppHover &&
    pts.find((p) => p.run === state.bppHover.run && p.point === state.bppHover.point);
  if (!hover) return;

  drawMarker(hover, hover.current ? PSNR_LINE_COLOR : BPP_PAST_COLOR, 5);
  const run = state.bppRuns[hover.run];
  const label = `${run.sites} sites · ${fmtBpp(hover.x)} bpp · ${hover.y.toFixed(2)} dB`;
  ctx.font = "11px Inter, system-ui, -apple-system, sans-serif";
  const tw = ctx.measureText(label).width + 14;
  const th = 22;
  const hx = g.x(hover.x);
  let tx = hx + 10;
  if (tx + tw > right) tx = hx - 10 - tw;
  tx = Math.max(4, Math.min(w - tw - 4, tx));
  const ty = Math.max(top, Math.min(bottom - th, g.y(hover.y) - th - 8));
  ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
  ctx.strokeStyle = lineCol;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tx, ty, tw, th, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = textCol;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, tx + 7, ty + th / 2);
}

function handleBppHover(ev) {
  const pts = bppPlotPoints();
  if (!pts.length) return;
  const rect = VIEWS.bpp.canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const g = bppPlotGeometry(pts);
  let best = null, bestDist = Infinity;
  for (const p of pts) {
    const d = Math.hypot(g.x(p.x) - mx, g.y(p.y) - my);
    if (d < bestDist) { bestDist = d; best = p; }
  }
  const next = bestDist < 40 ? { run: best.run, point: best.point } : null;
  if (next?.run !== state.bppHover?.run || next?.point !== state.bppHover?.point) {
    state.bppHover = next;
    drawBppPlot();
  }
}

// ---- |gradient| vs iteration plot (log y) ----
// One hue per param group (fixed order, validated against --canvas-bg);
// solid = max over sites, dashed = mean over sites with any nonzero grad.

const GRAD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
const GRAD_PAD = { left: 56, right: 16, bottom: 34 };
const GRAD_MEAN_DASH = [6, 4];

function fmtExp(v) {
  return v > 0 ? v.toExponential(2) : "0";
}

function gradLegendLayout(ctx, w) {
  // Legend wraps under the card label; returns item positions + bottom edge.
  const items = GRAD_GROUPS.map((g, i) => ({ label: g.label, color: GRAD_COLORS[i] }));
  items.push({ label: "max", line: [] }, { label: "mean", line: GRAD_MEAN_DASH });
  const x0 = 12, rowH = 16, swatch = 18;
  let x = x0, y = 34;
  for (const it of items) {
    const iw = swatch + 6 + ctx.measureText(it.label).width + 14;
    if (x + iw > w - 12 && x > x0) { x = x0; y += rowH; }
    it.x = x; it.y = y;
    x += iw;
  }
  return { items, bottom: y + 14, swatch };
}

function gradPlotGeometry(top) {
  const canvas = VIEWS.grad.canvas;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const data = state.gradHistory;
  const xMax = Math.max(1, state.psnrIters, data.length ? data[data.length - 1].it : 0);
  let lo = Infinity, hi = -Infinity;
  for (const p of data) {
    for (const g of GRAD_GROUPS) {
      const s = p.stats.groups[g.key];
      for (const v of [s.mean, s.max]) {
        if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
  }
  if (!Number.isFinite(lo)) { lo = 1e-8; hi = 1e-2; }
  const yMin = Math.floor(Math.log10(lo));
  const yMax = Math.max(yMin + 1, Math.ceil(Math.log10(hi)));
  const plotW = Math.max(1, w - GRAD_PAD.left - GRAD_PAD.right);
  const plotH = Math.max(1, h - top - GRAD_PAD.bottom);
  return {
    w, h, xMax, yMin, yMax, top,
    x: (it) => GRAD_PAD.left + (it / xMax) * plotW,
    y: (v) => top + (1 - (Math.log10(v) - yMin) / (yMax - yMin)) * plotH,
  };
}

function drawGradPlot() {
  const canvas = VIEWS.grad.canvas;
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const lineCol = css.getPropertyValue("--line").trim() || "#e6e6df";
  const dimCol = css.getPropertyValue("--text-dim").trim() || "#66666e";
  const softCol = css.getPropertyValue("--text-soft").trim() || "#9a9aa2";
  const textCol = css.getPropertyValue("--text").trim() || "#1d1d1f";
  const surface = css.getPropertyValue("--canvas-bg").trim() || "#f1f1ec";

  ctx.font = "10.5px Inter, system-ui, -apple-system, sans-serif";
  ctx.lineWidth = 1;

  // Legend
  const legend = gradLegendLayout(ctx, w);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const it of legend.items) {
    ctx.save();
    ctx.strokeStyle = it.color || dimCol;
    ctx.lineWidth = 2;
    if (it.line) ctx.setLineDash(it.line);
    ctx.beginPath(); ctx.moveTo(it.x, it.y); ctx.lineTo(it.x + legend.swatch, it.y); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = dimCol;
    ctx.fillText(it.label, it.x + legend.swatch + 6, it.y);
  }

  const top = legend.bottom + 10;
  const data = state.gradHistory;
  const g = gradPlotGeometry(top);
  const left = GRAD_PAD.left, right = w - GRAD_PAD.right;
  const bottom = h - GRAD_PAD.bottom;

  // Decade grid + labels
  ctx.textAlign = "right";
  for (let e = g.yMin; e <= g.yMax; e += 1) {
    const y = Math.round(g.y(10 ** e)) + 0.5;
    ctx.strokeStyle = lineCol;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = softCol;
    ctx.fillText(`1e${e}`, left - 6, y);
  }

  // Baseline + x labels
  ctx.strokeStyle = softCol;
  ctx.beginPath(); ctx.moveTo(left, bottom + 0.5); ctx.lineTo(right, bottom + 0.5); ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = softCol;
  for (const v of niceTicks(0, g.xMax, Math.max(2, Math.floor((right - left) / 70)))) {
    ctx.fillText(v >= 10000 ? `${v / 1000}k` : String(v), g.x(v), bottom + 6);
  }
  ctx.fillStyle = dimCol;
  ctx.fillText("iteration", (left + right) / 2, bottom + 20);
  ctx.save();
  ctx.translate(12, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText("|grad|", 0, 0);
  ctx.restore();

  if (!data.length) {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = softCol;
    ctx.fillText(state.running ? "Waiting for first log tick…" : "Train to see gradients", (left + right) / 2, (top + bottom) / 2);
    return;
  }

  // Series: gaps where a value is 0 (can't be drawn on a log axis)
  const drawSeries = (key, field, color, dash) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (const p of data) {
      const v = p.stats.groups[key][field];
      if (!(v > 0)) { pen = false; continue; }
      const x = g.x(p.it), y = g.y(v);
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
    ctx.restore();
  };
  GRAD_GROUPS.forEach((grp, i) => {
    drawSeries(grp.key, "mean", GRAD_COLORS[i], GRAD_MEAN_DASH);
    drawSeries(grp.key, "max", GRAD_COLORS[i], []);
  });

  const hover = state.gradHover != null ? data[state.gradHover] : null;
  if (!hover) return;

  // Crosshair + markers + tooltip table
  const hx = Math.round(g.x(hover.it)) + 0.5;
  ctx.strokeStyle = softCol;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(hx, top); ctx.lineTo(hx, bottom); ctx.stroke();
  ctx.setLineDash([]);
  GRAD_GROUPS.forEach((grp, i) => {
    for (const field of ["mean", "max"]) {
      const v = hover.stats.groups[grp.key][field];
      if (!(v > 0)) continue;
      ctx.beginPath();
      ctx.arc(hx, g.y(v), 4, 0, Math.PI * 2);
      ctx.fillStyle = GRAD_COLORS[i];
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = surface;
      ctx.stroke();
    }
  });

  ctx.font = "11px Inter, system-ui, -apple-system, sans-serif";
  const rows = GRAD_GROUPS.map((grp, i) => ({
    color: GRAD_COLORS[i],
    cells: [grp.label, fmtExp(hover.stats.groups[grp.key].mean), fmtExp(hover.stats.groups[grp.key].max)],
  }));
  const header = [`iter ${hover.it}`, "mean", "max"];
  const footer = `${hover.stats.sites} sites with nonzero grad`;
  const colW = [0, 0, 0];
  for (const r of [header, ...rows.map((r) => r.cells)]) {
    r.forEach((c, j) => { colW[j] = Math.max(colW[j], ctx.measureText(c).width); });
  }
  const rowH = 17, padX = 8, sw = 12, gap = 14;
  const tw = Math.max(padX * 2 + sw + 6 + colW[0] + gap + colW[1] + gap + colW[2],
                      padX * 2 + ctx.measureText(footer).width);
  const th = rowH * (rows.length + 2) + 8;
  let tx = hx + 12;
  if (tx + tw > right) tx = hx - 12 - tw;
  const ty = Math.max(4, Math.min(h - th - 4, top));
  ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
  ctx.strokeStyle = lineCol;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tx, ty, tw, th, 5);
  ctx.fill();
  ctx.stroke();

  ctx.textBaseline = "middle";
  const c0 = tx + padX + sw + 6;
  const c1 = c0 + colW[0] + gap + colW[1];
  const c2 = c1 + gap + colW[2];
  const drawRow = (cells, y, col) => {
    ctx.fillStyle = col;
    ctx.textAlign = "left";
    ctx.fillText(cells[0], c0, y);
    ctx.textAlign = "right";
    ctx.fillText(cells[1], c1, y);
    ctx.fillText(cells[2], c2, y);
  };
  let y = ty + 4 + rowH / 2;
  drawRow(header, y, dimCol);
  for (const r of rows) {
    y += rowH;
    ctx.fillStyle = r.color;
    ctx.fillRect(tx + padX, y - 1, sw, 2.5);
    drawRow(r.cells, y, textCol);
  }
  y += rowH;
  ctx.fillStyle = dimCol;
  ctx.textAlign = "left";
  ctx.fillText(footer, tx + padX, y);
}

function handleGradHover(ev) {
  const data = state.gradHistory;
  if (!data.length) return;
  const rect = VIEWS.grad.canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const g = gradPlotGeometry(0); // x mapping doesn't depend on top
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < data.length; i++) {
    const d = Math.abs(g.x(data[i].it) - mx);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  if (state.gradHover !== best) {
    state.gradHover = best;
    drawGradPlot();
  }
}

// ---- Residual heatmap ----

// Mean |recon - target| over RGB, mapped through a magma-like ramp on a fixed
// [0, RESIDUAL_MAX] scale so frames stay comparable as training progresses.
const RESIDUAL_MAX = 0.25;
const RESIDUAL_STOPS = [
  [0.00, [0, 0, 4]],
  [0.25, [81, 18, 124]],
  [0.50, [183, 55, 121]],
  [0.75, [252, 137, 97]],
  [1.00, [252, 253, 191]],
];
const RESIDUAL_LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let k = 1;
    while (k < RESIDUAL_STOPS.length - 1 && RESIDUAL_STOPS[k][0] < t) k += 1;
    const [t0, c0] = RESIDUAL_STOPS[k - 1];
    const [t1, c1] = RESIDUAL_STOPS[k];
    const f = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c += 1) lut[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
  }
  return lut;
})();

function drawResidual() {
  const canvas = VIEWS.residual.canvas;
  const target = state.targetImage;
  const render = state.lastRender;
  if (!canvas || !target || !render || render.width !== target.width || render.height !== target.height) return;
  const { width, height } = render;
  const a = render.rgba;
  const b = target.data;
  const out = new Uint8ClampedArray(width * height * 4);
  const scale = 255 / (3 * RESIDUAL_MAX);
  for (let p = 0, i = 0; p < width * height; p += 1, i += 4) {
    const err = Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
    const idx = Math.min(255, Math.round(err * scale)) * 3;
    out[i] = RESIDUAL_LUT[idx];
    out[i + 1] = RESIDUAL_LUT[idx + 1];
    out[i + 2] = RESIDUAL_LUT[idx + 2];
    out[i + 3] = 255;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  fitCanvasToCard(canvas, width, height);
  canvas.getContext("2d").putImageData(new ImageData(out, width, height), 0, 0);
}

function drawTarget() {
  if (!state.targetImage) return;
  const { data, width, height } = state.targetImage;
  updateCanvasLayout(width, height);
  if (VIEWS.target.canvas) drawToCanvas(VIEWS.target.canvas, data, width, height);
}

// Multi-scale importance map (visualization only — not used by the optimizer).
// Luma Laplacian pyramid L_k = G_k − up(G_{k+1}); each band's energy L_k² is
// upsampled to full res and combined as
//   w = mask · (ε + sqrt(Σ_k β_k · up(L_k²)))^γ
// so smooth large-scale structure (coarse bands) counts as well as fine edges.
const IMPORTANCE_LEVELS = 6;
const IMPORTANCE_BAND_WEIGHTS = [0.5, 1, 1, 1, 1, 1]; // β_k, finest band first
const IMPORTANCE_GAMMA = 1.0;
const IMPORTANCE_EPS = 1e-3;
const IMPORTANCE_DISPLAY_PCT = 0.995; // map this percentile to the top of the ramp
const IMPORTANCE_STOPS = [ // viridis-like, to keep it distinct from the residual card
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
];
const IMPORTANCE_LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let k = 1;
    while (k < IMPORTANCE_STOPS.length - 1 && IMPORTANCE_STOPS[k][0] < t) k += 1;
    const [t0, c0] = IMPORTANCE_STOPS[k - 1];
    const [t1, c1] = IMPORTANCE_STOPS[k];
    const f = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c += 1) lut[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
  }
  return lut;
})();

// Separable [1 4 6 4 1]/16 blur followed by 2× decimation (clamped edges).
function pyrDown(src, w, h) {
  const k = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  const dw = Math.ceil(w / 2);
  const dh = Math.ceil(h / 2);
  const tmp = new Float32Array(dw * h);
  for (let y = 0; y < h; y += 1) {
    for (let dx = 0; dx < dw; dx += 1) {
      const x = dx * 2;
      let s = 0;
      for (let t = -2; t <= 2; t += 1) {
        const xx = Math.min(w - 1, Math.max(0, x + t));
        s += k[t + 2] * src[y * w + xx];
      }
      tmp[y * dw + dx] = s;
    }
  }
  const out = new Float32Array(dw * dh);
  for (let dy = 0; dy < dh; dy += 1) {
    const y = dy * 2;
    for (let dx = 0; dx < dw; dx += 1) {
      let s = 0;
      for (let t = -2; t <= 2; t += 1) {
        const yy = Math.min(h - 1, Math.max(0, y + t));
        s += k[t + 2] * tmp[yy * dw + dx];
      }
      out[dy * dw + dx] = s;
    }
  }
  return { data: out, width: dw, height: dh };
}

// Bilinear resample of a (sw×sh) grid onto (w×h), pixel-center aligned.
function upsampleBilinear(src, sw, sh, w, h) {
  const out = new Float32Array(w * h);
  const sx = sw / w;
  const sy = sh / h;
  for (let y = 0; y < h; y += 1) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x += 1) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = fx - x0;
      const a = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
      const b = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
      out[y * w + x] = a + (b - a) * ty;
    }
  }
  return out;
}

function computeImportanceMap(rgbaF, width, height, maskRgba) {
  const n = width * height;
  let level = { data: new Float32Array(n), width, height };
  for (let p = 0, i = 0; p < n; p += 1, i += 4) {
    level.data[p] = 0.299 * rgbaF[i] + 0.587 * rgbaF[i + 1] + 0.114 * rgbaF[i + 2];
  }
  const energy = new Float32Array(n);
  for (let k = 0; k < IMPORTANCE_LEVELS; k += 1) {
    if (level.width < 2 || level.height < 2) break;
    const next = pyrDown(level.data, level.width, level.height);
    const up = upsampleBilinear(next.data, next.width, next.height, level.width, level.height);
    const band = new Float32Array(level.width * level.height);
    for (let p = 0; p < band.length; p += 1) {
      const l = level.data[p] - up[p];
      band[p] = l * l;
    }
    const full = k === 0 ? band : upsampleBilinear(band, level.width, level.height, width, height);
    const beta = IMPORTANCE_BAND_WEIGHTS[k] ?? 1;
    for (let p = 0; p < n; p += 1) energy[p] += beta * full[p];
    level = next;
  }
  const out = new Float32Array(n);
  for (let p = 0; p < n; p += 1) {
    const m = maskRgba ? maskRgba[p * 4] : 1;
    out[p] = m * Math.pow(IMPORTANCE_EPS + Math.sqrt(energy[p]), IMPORTANCE_GAMMA);
  }
  return out;
}

function drawImportance() {
  const canvas = VIEWS.importance.canvas;
  const target = state.targetImage;
  if (!canvas || !target) return;
  const { data, width, height } = target;
  const n = width * height;
  const maskRgba = state.maskData?.rgba?.length === n * 4 ? state.maskData.rgba : null;
  const w = computeImportanceMap(data, width, height, maskRgba);
  const sorted = Float32Array.from(w).sort();
  const top = sorted[Math.min(n - 1, Math.floor(IMPORTANCE_DISPLAY_PCT * n))] || 1;
  const out = new Uint8ClampedArray(n * 4);
  for (let p = 0, i = 0; p < n; p += 1, i += 4) {
    const idx = Math.min(255, Math.round((w[p] / top) * 255)) * 3;
    out[i] = IMPORTANCE_LUT[idx];
    out[i + 1] = IMPORTANCE_LUT[idx + 1];
    out[i + 2] = IMPORTANCE_LUT[idx + 2];
    out[i + 3] = 255;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  fitCanvasToCard(canvas, width, height);
  canvas.getContext("2d").putImageData(new ImageData(out, width, height), 0, 0);
}

async function handleImageInput(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  ui.kodakSelect.value = "";
  await setTarget(file, file.name);
}

function kodakName(i) {
  return `kodim${String(i).padStart(2, "0")}.png`;
}

async function loadKodak(i) {
  if (state.running) return;
  const name = kodakName(i);
  // Fast flicks overlap loads; only the latest one may land.
  const token = ++state.targetToken;
  ui.kodakSelect.value = String(i);
  // Clear so re-picking the same local file after a Kodak image still fires change.
  ui.imageInput.value = "";
  setStatus(`Loading ${name}…`);
  ui.imageLabel.textContent = name;
  try {
    const res = await fetch(new URL(name, new URL(KODAK_DIR, location.href)));
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    const blob = await res.blob();
    if (token !== state.targetToken) return;
    await setTarget(blob, name, token);
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
  }
}

function stepKodak(delta) {
  const cur = Number(ui.kodakSelect.value) || (delta > 0 ? 0 : 1);
  const next = ((cur - 1 + delta + KODAK_COUNT) % KODAK_COUNT) + 1;
  loadKodak(next);
}

function setKodakEnabled(enabled) {
  ui.kodakSelect.disabled = !enabled;
  ui.kodakPrev.disabled = !enabled;
  ui.kodakNext.disabled = !enabled;
}

async function setTarget(file, name, token = ++state.targetToken) {
  const maxDim = DEMO_MAX_IMAGE_SIDE;
  setStatus(`Loading ${name}…`);
  ui.imageLabel.textContent = name;
  try {
    const img = await loadImageToFloat32(file, maxDim);
    if (token !== state.targetToken) return;
    const scaled = img.sourceWidth && img.sourceHeight &&
      (img.sourceWidth !== img.width || img.sourceHeight !== img.height);
    state.targetImage = img;
    state.bppRuns = [];
    state.bppHover = null;
    drawBppPlot();
    drawTarget();
    drawImportance();
    blankRenderViews(img.width, img.height);
    updateCanvasLayout(img.width, img.height);
    setStatus(scaled
      ? `Target ${img.width}×${img.height} (downscaled from ${img.sourceWidth}×${img.sourceHeight})`
      : `Target ${img.width}×${img.height}`);
    maybeEnableStart();
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
  }
}

async function handleMaskInput(ev) {
  const file = ev.target.files[0];
  if (!file || !state.targetImage) return;
  try {
    state.maskData = await loadMaskRgba32Float(file, state.targetImage.width, state.targetImage.height);
    ui.maskLabel.textContent = file.name;
    appendLog(`Mask loaded (sum=${state.maskData.maskSum}).`);
    drawImportance();
  } catch (err) {
    appendLog(`Mask load failed: ${err.message}`);
  }
}

async function handleInitSitesInput(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  try {
    state.initSites = await loadSitesFromFile(file);
    ui.initLabel.textContent = `${file.name} (${state.initSites.count})`;
    appendLog(`Init sites: ${state.initSites.count}` +
      (state.initSites.width ? ` (${state.initSites.width}×${state.initSites.height})` : ""));
  } catch (err) {
    appendLog(`Init sites load failed: ${err.message}`);
  }
}

function maybeEnableStart() {
  ui.startBtn.disabled = !(state.device && state.targetImage);
}

function scaleInitSitesIfNeeded(init, width, height) {
  if (!init) return null;
  if (!init.width || !init.height || (init.width === width && init.height === height)) return init;
  const sx = width / init.width;
  const sy = height / init.height;
  const data = new Float32Array(init.data);
  for (let i = 0; i < init.count; i += 1) {
    data[i * SITE_FLOATS + 0] *= sx;
    data[i * SITE_FLOATS + 1] *= sy;
  }
  return { ...init, data, width, height };
}

function overrideInitLogTau(init, initLogTau) {
  if (!init) return null;
  const data = new Float32Array(init.data);
  for (let i = 0; i < init.count; i += 1) {
    data[i * SITE_FLOATS + 2] = initLogTau;
  }
  return { ...init, data };
}

async function startTraining() {
  if (state.running) return;
  if (!state.device || !state.targetImage) return;

  ui.saveBtn.disabled = true;
  ui.savePopover.classList.add("hidden");
  ui.startBtn.disabled = true;
  ui.stopBtn.disabled = false;
  setKodakEnabled(false);
  state.running = true;
  state.abortCtrl = new AbortController();
  state.lastHashed = null;
  state.lastHeatmap = null;

  const args = readFormArgs();
  const { width, height } = state.targetImage;

  let maskRgba = null;
  let maskSum = width * height;
  if (state.maskData) {
    maskRgba = state.maskData.rgba;
    maskSum = state.maskData.maskSum;
  } else {
    maskRgba = whiteMaskRgba32Float(width, height).rgba;
  }

  let init = scaleInitSitesIfNeeded(state.initSites, width, height);
  if (init) init = overrideInitLogTau(init, args.initLogTau);
  const initSiteCount = init ? init.count : args.sites;

  state.psnrHistory = [];
  state.lastLrScale = null;
  state.psnrIters = args.iters;
  state.psnrHover = null;
  state.gradHistory = [];
  state.gradHover = null;
  state.bppRuns.push({ sites: initSiteCount, bpp: computeBpp(initSiteCount, width, height), points: [] });
  state.bppHover = null;
  drawPsnrPlot();
  drawBppPlot();
  drawGradPlot();

  appendLog(`Target: ${width}×${height} | sites: ${initSiteCount} | iters: ${args.warmupIters} warmup + ${args.decayIters} decay | lr ×${args.lrMult} | log: ${args.logFreq} | view: ${args.viewerFreq}`);
  updateProgress(0, args.iters, "Preparing…");

  const trainer = new Trainer(state.device, state.shaders, args, {
    signal: state.abortCtrl.signal,
    onLog: async (info) => {
      if (info.type === "msg") {
        appendLog(info.message);
      } else if (info.type === "iter") {
        const line = `Iter ${String(info.it).padStart(4)} | PSNR: ${info.psnr.toFixed(2)} dB | ` +
                     `Active: ${info.activeSites}/${info.totalSites} | ${info.itsPerSec.toFixed(1)} it/s | ${info.elapsedSec.toFixed(1)}s`;
        appendLog(line);
        state.lastRender = { rgba: info.renderRgba, width: info.width, height: info.height };
        state.lastSites = info.sitesData;
        state.lastSiteCount = info.totalSites;
        state.lastPsnr = info.psnr;
        state.lastLrScale = info.lrScale;
        if (Number.isFinite(info.psnr)) {
          state.psnrHistory.push({ it: info.it, psnr: info.psnr });
          drawPsnrPlot();
          const run = state.bppRuns[state.bppRuns.length - 1];
          run.sites = info.activeSites;
          run.bpp = computeBpp(info.activeSites, info.width, info.height);
          run.points.push({ it: info.it, psnr: info.psnr });
          drawBppPlot();
        }
        if (info.gradStats) {
          state.gradHistory.push({ it: info.it, stats: info.gradStats });
          drawGradPlot();
        }
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `log iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await drawView();
      } else if (info.type === "preview") {
        // Full image preview. Site overlays arrive through a separate lighter
        // event so they are not blocked by full-image readback.
        state.lastRender = { rgba: info.renderRgba, width: info.width, height: info.height };
        if (info.sitesData) {
          state.lastSites = info.sitesData;
          state.lastSiteCount = info.totalSites;
        }
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `image iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await drawView();
      } else if (info.type === "sites") {
        state.lastSites = info.sitesData;
        state.lastSiteCount = info.totalSites;
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `sites iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        drawCentroidOverlay();
      }
    },
  });
  state.trainer = trainer;

  try {
    await trainer.prepare({
      width, height,
      targetRgba32f: state.targetImage.data,
      maskRgba32f: maskRgba,
      maskSum,
      initSites: init ? init.data : null,
      initSiteCount,
    });
    await trainer.run();
    const final = await trainer.finalize();
    state.lastRender = { rgba: final.renderRgba, width: final.width, height: final.height };
    state.lastSites = final.sites;
    state.lastSiteCount = final.count;
    state.lastHashed = null;
    state.lastHeatmap = null;
    await drawView();
    appendLog(`Done. Best PSNR: ${final.bestPsnr.toFixed(2)} dB | time: ${final.trainSeconds.toFixed(2)} s | active: ${final.count}`);
    updateProgress(args.iters, args.iters,
      `done · best ${final.bestPsnr.toFixed(2)} dB · ${final.trainSeconds.toFixed(1)}s`);
    ui.saveBtn.disabled = false;
  } catch (err) {
    appendLog(`Error: ${err.message}`);
    console.error(err);
  } finally {
    state.running = false;
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled = true;
    setKodakEnabled(true);
  }
}

// Unified metric renderer — PSNR sticks between log ticks so preview updates
// don't momentarily blank it.
function renderMetrics(info) {
  const psnr = Number.isFinite(info.psnr) ? info.psnr : state.lastPsnr;
  const psnrText = psnr != null ? `${psnr.toFixed(2)} dB` : "—";
  ui.metricRow.innerHTML =
    `<span>iter <strong>${info.it + 1}/${info.iters}</strong></span>` +
    `<span>PSNR <strong>${psnrText}</strong></span>` +
    (state.lastLrScale != null ? `<span>lr <strong>×${state.lastLrScale.toPrecision(3)}</strong></span>` : "") +
    `<span>active <strong>${info.activeSites}</strong>/${info.totalSites}</span>` +
    `<span><strong>${info.itsPerSec.toFixed(1)}</strong> it/s</span>`;
}

function updateProgress(cur, total, text) {
  const pct = total > 0 ? (cur / total) * 100 : 0;
  ui.progressFill.style.width = `${Math.min(100, pct).toFixed(1)}%`;
  ui.progressText.textContent = text;
}

function clearOverlay() {
  const overlay = VIEWS.preview.overlayCanvas;
  if (!overlay) return;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

// The overlay canvas sits inside the same card as the preview but is absolutely
// positioned, so we need to match its drawing buffer and CSS box to the
// preview's actual displayed rect (which is aspect-fit inside the card).
function positionOverlay() {
  const { canvas, card, overlayCanvas: overlay } = VIEWS.preview;
  if (!canvas || !overlay) return;
  const prect = canvas.getBoundingClientRect();
  const crect = card.getBoundingClientRect();
  // Absolute offsets are relative to the parent's padding box, while
  // getBoundingClientRect is its border box, so subtract the card border.
  overlay.style.left = `${prect.left - crect.left - card.clientLeft}px`;
  overlay.style.top = `${prect.top - crect.top - card.clientTop}px`;
  overlay.style.width = `${prect.width}px`;
  overlay.style.height = `${prect.height}px`;
}

function prepareOverlay(width, height) {
  const overlay = VIEWS.preview.overlayCanvas;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  positionOverlay();
  clearOverlay();
  return overlay.getContext("2d");
}

function drawCentroidOverlay() {
  if (!VIEWS.preview.overlayCanvas) return;
  if (!state.lastRender || !state.lastSites) return;
  const ctx = prepareOverlay(state.lastRender.width, state.lastRender.height);
  drawCentroids(ctx);
}

// Redraws every view that depends on the latest render. Site IDs and tau need
// an extra GPU pass each, so they only run when their view is in the layout.
async function drawView() {
  if (!state.lastRender) return;
  drawResidual();
  const { rgba, width, height } = state.lastRender;
  if (VIEWS.preview.canvas) {
    drawToCanvas(VIEWS.preview.canvas, rgba, width, height);
    drawCentroidOverlay();
  }
  for (const [id, mode] of [["siteIds", "hashed"], ["tau", "heatmap"]]) {
    if (!VIEWS[id].canvas) continue;
    const img = await ensureRender(mode);
    // The layout may have changed while the render was in flight.
    if (img && VIEWS[id].canvas) drawToCanvas(VIEWS[id].canvas, img.rgba, img.width, img.height);
  }
}

async function ensureRender(mode) {
  if (!state.trainer) return null;
  if (mode === "hashed" && state.lastHashed) return state.lastHashed;
  if (mode === "heatmap" && state.lastHeatmap) return state.lastHeatmap;
  try {
    if (state.lastSites) state.trainer.setLastSites(state.lastSites);
    const rgba = await state.trainer.renderView(mode);
    const out = { rgba, width: state.lastRender.width, height: state.lastRender.height };
    if (mode === "hashed") state.lastHashed = out;
    else if (mode === "heatmap") state.lastHeatmap = out;
    return out;
  } catch (err) {
    appendLog(`View render failed: ${err.message}`);
    return null;
  }
}

function drawCentroids(ctx) {
  if (!state.lastSites || !state.lastRender) return;
  const { rgba, width, height } = state.lastRender;
  ctx.save();
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const base = i * SITE_FLOATS;
    const x = state.lastSites[base + 0];
    const y = state.lastSites[base + 1];
    if (x < 0) continue;
    // Fill with the inverse of the rendered pixel under the dot so it stands out.
    const px = Math.min(width - 1, Math.max(0, Math.floor(x)));
    const py = Math.min(height - 1, Math.max(0, Math.floor(y)));
    const p = (py * width + px) * 4;
    const inv = (v) => 255 - Math.round(Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)) * 255);
    ctx.fillStyle = `rgb(${inv(rgba[p])}, ${inv(rgba[p + 1])}, ${inv(rgba[p + 2])})`;
    ctx.beginPath();
    ctx.arc(x, y, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function stopTraining() {
  if (state.abortCtrl) state.abortCtrl.abort();
  appendLog("Stop requested.");
}

function exportPng() {
  if (!state.lastRender) return;
  const { rgba, width, height } = state.lastRender;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(floatImageToRgba8(rgba, width, height), width, height), 0, 0);
  canvas.toBlob((blob) => {
    if (blob) triggerDownload(blob, `sad_${Date.now()}.png`);
  }, "image/png");
  ui.savePopover.classList.add("hidden");
}

function exportSites() {
  if (!state.lastSites) return;
  const text = serializeSitesTxt(state.lastSites, state.lastSiteCount,
    state.lastRender.width, state.lastRender.height);
  triggerDownload(new Blob([text], { type: "text/plain" }), `sad_${Date.now()}_sites.txt`);
  ui.savePopover.classList.add("hidden");
}

function toggleSaveMenu(ev) {
  ev.stopPropagation();
  if (ui.saveBtn.disabled) return;
  ui.savePopover.classList.toggle("hidden");
}

function toggleParams() {
  // The viewer's ResizeObserver refits the cards.
  ui.paramsPanel.classList.toggle("hidden");
}

async function main() {
  for (const [name, layout] of Object.entries(LAYOUTS)) ui.layoutSelect.add(new Option(layout.label, name));
  setLayout(savedLayout());
  try {
    state.config = await loadConfig();
    applyConfigToForm(state.config);
    state.device = await initDevice();
    state.shaders = await loadShaders({
      tileSlots: tileSlotsForLimit(state.device.limits.maxComputeWorkgroupStorageSize),
    });
    setStatus("Ready");
    maybeEnableStart();
  } catch (err) {
    setStatus(`Setup failed: ${err.message}`);
    console.error(err);
  }

  ui.imageInput.addEventListener("change", handleImageInput);
  for (let i = 1; i <= KODAK_COUNT; i += 1) {
    ui.kodakSelect.add(new Option(kodakName(i).replace(".png", ""), String(i)));
  }
  ui.kodakSelect.addEventListener("change", () => {
    if (ui.kodakSelect.value) loadKodak(Number(ui.kodakSelect.value));
  });
  ui.kodakPrev.addEventListener("click", () => stepKodak(-1));
  ui.kodakNext.addEventListener("click", () => stepKodak(1));
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = ev.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    ev.preventDefault();
    stepKodak(ev.key === "ArrowRight" ? 1 : -1);
  });
  ui.maskInput.addEventListener("change", handleMaskInput);
  ui.initSitesInput.addEventListener("change", handleInitSitesInput);
  ui.startBtn.addEventListener("click", startTraining);
  ui.stopBtn.addEventListener("click", stopTraining);
  ui.toggleParamsBtn.addEventListener("click", toggleParams);
  ui.saveBtn.addEventListener("click", toggleSaveMenu);
  ui.savePng.addEventListener("click", exportPng);
  ui.saveTxt.addEventListener("click", exportSites);
  document.addEventListener("click", (ev) => {
    if (!ui.savePopover.contains(ev.target) && ev.target !== ui.saveBtn) {
      ui.savePopover.classList.add("hidden");
    }
  });
  ui.layoutSelect.addEventListener("change", () => setLayout(ui.layoutSelect.value));
  // Card sizes follow the viewer area (window resize, advanced panel toggle).
  new ResizeObserver(() => updateCanvasLayout()).observe(ui.canvases);
}

main();
