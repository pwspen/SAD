// Entry point for the SAD trainer web page.
// Hugging-Face-ish UI: target image in the top bar, advanced options
// (mask, init-sites, candidates/lr…) in the collapsible left panel, and a
// side-by-side target vs reconstruction view, both aspect-fit.

import { loadShaders, tileSlotsForLimit } from "./src/shader_loader.js?v=adam-bounds-20261016b";
import { requestTrainingDevice } from "./src/device.js?v=adam-bounds-20261016b";
import { Trainer, buildArgs, GRAD_GROUPS } from "./src/trainer.js?v=adam-bounds-20261016b";
import { loadImageToFloat32, loadMaskRgba32Float, whiteMaskRgba32Float } from "./src/textures.js?v=adam-bounds-20261016b";
import { loadSitesFromFile, serializeSitesTxt, triggerDownload, floatImageToRgba8 } from "./src/io.js?v=adam-bounds-20261016b";
import { SITE_FLOATS } from "./src/params.js?v=adam-bounds-20261016b";

const CONFIG_URL = "../../training_config.json";
const DEMO_MAX_IMAGE_SIDE = 2048;

// Fields that live in the advanced panel. The target image is in the top bar;
// init-sites / mask are pickers inside the panel (not listed here).
const UI_FIELDS = [
  ["iters", "p-iters"],
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
  ["lrPosBase", "p-lrPosBase"],
  ["lrTauBase", "p-lrTauBase"],
  ["lrRadiusBase", "p-lrRadiusBase"],
  ["lrColorBase", "p-lrColorBase"],
  ["lrDirBase", "p-lrDirBase"],
  ["lrAnisoBase", "p-lrAnisoBase"],
];

const state = {
  device: null,
  shaders: null,
  config: null,
  targetImage: null,     // { data, width, height }
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
  psnrHistory: [],       // [{ it, psnr }] from log ticks, plotted in the PSNR card
  psnrIters: 0,
  psnrHover: null,       // index into psnrHistory under the cursor
  gradHistory: [],       // [{ it, stats }] per-param |grad| mean/max from log ticks
  gradHover: null,       // index into gradHistory under the cursor
};

const ui = {
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  imageLabel: document.getElementById("imageLabel"),
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
  viewMode: document.getElementById("viewMode"),
  previewLabel: document.getElementById("previewLabel"),
  previewCanvas: document.getElementById("previewCanvas"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  targetCanvas: document.getElementById("targetCanvas"),
  psnrCanvas: document.getElementById("psnrCanvas"),
  gradCanvas: document.getElementById("gradCanvas"),
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

function updateCanvasLayout(width = state.targetImage?.width, height = state.targetImage?.height) {
  const cards = [ui.targetCanvas.parentElement, ui.previewCanvas.parentElement];
  for (const card of cards) {
    card.style.removeProperty("aspect-ratio");
    card.style.removeProperty("width");
    card.style.removeProperty("height");
  }
  fitCanvasToCard(ui.targetCanvas, width || ui.targetCanvas.width, height || ui.targetCanvas.height);
  fitCanvasToCard(ui.previewCanvas, width || ui.previewCanvas.width, height || ui.previewCanvas.height);
  positionOverlay();
  drawPsnrPlot();
  drawGradPlot();
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
  const canvas = ui.psnrCanvas;
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
  const canvas = ui.psnrCanvas;
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
  const rect = ui.psnrCanvas.getBoundingClientRect();
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
  const canvas = ui.gradCanvas;
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
  const canvas = ui.gradCanvas;
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
  const rect = ui.gradCanvas.getBoundingClientRect();
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

function drawTarget() {
  if (!state.targetImage) return;
  const { data, width, height } = state.targetImage;
  updateCanvasLayout(width, height);
  drawToCanvas(ui.targetCanvas, data, width, height);
}

async function handleImageInput(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const maxDim = DEMO_MAX_IMAGE_SIDE;
  setStatus(`Loading ${file.name}…`);
  ui.imageLabel.textContent = file.name;
  try {
    const img = await loadImageToFloat32(file, maxDim);
    const scaled = img.sourceWidth && img.sourceHeight &&
      (img.sourceWidth !== img.width || img.sourceHeight !== img.height);
    state.targetImage = img;
    drawTarget();
    // Blank the preview canvas to same size so layout matches immediately.
    ui.previewCanvas.width = img.width;
    ui.previewCanvas.height = img.height;
    const ctx = ui.previewCanvas.getContext("2d");
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, img.width, img.height);
    ui.overlayCanvas.width = img.width;
    ui.overlayCanvas.height = img.height;
    fitCanvasToCard(ui.previewCanvas, img.width, img.height);
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
  state.psnrIters = args.iters;
  state.psnrHover = null;
  state.gradHistory = [];
  state.gradHover = null;
  drawPsnrPlot();
  drawGradPlot();

  appendLog(`Target: ${width}×${height} | sites: ${initSiteCount} | iters: ${args.iters} | log: ${args.logFreq} | view: ${args.viewerFreq}`);
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
        if (Number.isFinite(info.psnr)) {
          state.psnrHistory.push({ it: info.it, psnr: info.psnr });
          drawPsnrPlot();
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
    `<span>active <strong>${info.activeSites}</strong>/${info.totalSites}</span>` +
    `<span><strong>${info.itsPerSec.toFixed(1)}</strong> it/s</span>`;
}

function updateProgress(cur, total, text) {
  const pct = total > 0 ? (cur / total) * 100 : 0;
  ui.progressFill.style.width = `${Math.min(100, pct).toFixed(1)}%`;
  ui.progressText.textContent = text;
}

function clearOverlay() {
  const ctx = ui.overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, ui.overlayCanvas.width, ui.overlayCanvas.height);
}

// The overlay canvas sits inside the same card as the preview but is absolutely
// positioned, so we need to match its drawing buffer and CSS box to the
// preview's actual displayed rect (which is aspect-fit inside the card).
function positionOverlay() {
  const parent = ui.previewCanvas.parentElement;
  const prect = ui.previewCanvas.getBoundingClientRect();
  const crect = parent.getBoundingClientRect();
  ui.overlayCanvas.style.left = `${prect.left - crect.left}px`;
  ui.overlayCanvas.style.top = `${prect.top - crect.top}px`;
  ui.overlayCanvas.style.width = `${prect.width}px`;
  ui.overlayCanvas.style.height = `${prect.height}px`;
}

function prepareOverlay(width, height) {
  if (ui.overlayCanvas.width !== width || ui.overlayCanvas.height !== height) {
    ui.overlayCanvas.width = width;
    ui.overlayCanvas.height = height;
  }
  positionOverlay();
  clearOverlay();
  return ui.overlayCanvas.getContext("2d");
}

function drawCentroidOverlay() {
  if (ui.viewMode.value !== "centroids") return;
  if (!state.lastRender || !state.lastSites) return;
  const ctx = prepareOverlay(state.lastRender.width, state.lastRender.height);
  drawCentroids(ctx);
}

async function drawView() {
  if (!state.lastRender) return;
  const mode = ui.viewMode.value;
  const { rgba, width, height } = state.lastRender;
  ui.previewLabel.textContent = {
    render: "Reconstruction",
    centroids: "Reconstruction + centroids",
    diagram: "Site IDs",
    heatmap: "Tau heatmap",
  }[mode] || "Reconstruction";

  if (mode === "render") {
    drawToCanvas(ui.previewCanvas, rgba, width, height);
    positionOverlay();
    clearOverlay();
    return;
  }
  if (mode === "centroids") {
    drawToCanvas(ui.previewCanvas, rgba, width, height);
    drawCentroidOverlay();
    return;
  }
  if (mode === "diagram") {
    const img = await ensureRender("hashed");
    if (img) {
      drawToCanvas(ui.previewCanvas, img.rgba, img.width, img.height);
      positionOverlay();
      clearOverlay();
    }
    return;
  }
  if (mode === "heatmap") {
    const img = await ensureRender("heatmap");
    if (img) {
      drawToCanvas(ui.previewCanvas, img.rgba, img.width, img.height);
      positionOverlay();
      clearOverlay();
    }
    return;
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
  if (!state.lastSites) return;
  ctx.save();
  ctx.fillStyle = "rgba(255, 196, 64, 0.82)";
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const base = i * SITE_FLOATS;
    const x = state.lastSites[base + 0];
    const y = state.lastSites[base + 1];
    if (x < 0) continue;
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
  ui.previewCanvas.toBlob((blob) => {
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
  ui.paramsPanel.classList.toggle("hidden");
  // Positioning depends on the preview's bounding rect, which changes when the
  // panel collapses/expands.
  requestAnimationFrame(() => {
    if (state.targetImage) updateCanvasLayout(state.targetImage.width, state.targetImage.height);
    else positionOverlay();
  });
}

async function main() {
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
  ui.viewMode.addEventListener("change", drawView);
  ui.psnrCanvas.addEventListener("mousemove", handlePsnrHover);
  ui.psnrCanvas.addEventListener("mouseleave", () => {
    state.psnrHover = null;
    drawPsnrPlot();
  });
  ui.gradCanvas.addEventListener("mousemove", handleGradHover);
  ui.gradCanvas.addEventListener("mouseleave", () => {
    state.gradHover = null;
    drawGradPlot();
  });
  window.addEventListener("resize", () => {
    requestAnimationFrame(() => {
      if (state.targetImage) updateCanvasLayout(state.targetImage.width, state.targetImage.height);
      else {
        positionOverlay();
        drawPsnrPlot();
        drawGradPlot();
      }
    });
  });
  drawPsnrPlot();
  drawGradPlot();
}

main();
