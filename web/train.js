// Entry point for the SAD trainer web page.
// Hugging-Face-ish UI: target image in the top bar, advanced options
// (mask, init-sites, candidates/lr…) in the collapsible left panel, and a
// side-by-side target vs reconstruction view, both aspect-fit.

import { loadShaders, tileSlotsForLimit } from "../trainer/shader_loader.js?v=shared-wgsl-20261017";
import { requestTrainingDevice } from "../trainer/device.js?v=shared-wgsl-20261017";
import { Trainer, buildArgs, GRAD_GROUPS, lrSchedule, LR_SCHEDULES, growthCounts } from "../trainer/trainer.js?v=batch-20261017";
import { loadImageToFloat32, loadMaskRgba32Float, whiteMaskRgba32Float } from "../trainer/textures.js?v=shared-wgsl-20261017";
import { loadSitesFromFile, serializeSitesTxt, triggerDownload, floatImageToRgba8, computeBpp, BITS_PER_SITE } from "../trainer/io.js?v=shared-wgsl-20261017";
import { SITE_FLOATS } from "../trainer/params.js?v=shared-wgsl-20261017";
import { COLOR_SPACES } from "../trainer/color_spaces.js?v=color-space-20261017";
import {
  luma, maskWeights, laplacianBands, maskLevels, bandEnergies, IMPORTANCE, importanceMap,
  spectralPlan, windowedSpectrum, spectrumDiff, radialStats, frcCrossing, logPowerImage,
} from "./analysis.js?v=spectral-20261017";

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
  ["logFreq", "p-logFreq"], // also sets viewerFreq (one "refresh every" field)
  ["candFreq", "p-candFreq"],
  ["candUpdatePasses", "p-candUpdatePasses"],
  ["candDownscale", "p-candDownscale"],
  ["candRadiusScale", "p-candRadiusScale"],
  ["candRadiusProbes", "p-candRadiusProbes"],
  ["candInjectCount", "p-candInjectCount"],
  ["candHilbertWindow", "p-candHilbertWindow"],
  ["candHilbertProbes", "p-candHilbertProbes"],
  ["lrMult", "p-lrMult"],
  ["lrSchedule", "p-lrSchedule"],
  ["lrPosBase", "p-lrPosBase"],
  ["lrTauBase", "p-lrTauBase"],
  ["lrRadiusBase", "p-lrRadiusBase"],
  ["lrColorBase", "p-lrColorBase"],
  ["lrDirBase", "p-lrDirBase"],
  ["lrAnisoBase", "p-lrAnisoBase"],
  ["colorSpace", "p-colorSpace"],
  ["siteMode", "p-siteMode"],
  ["growLambda", "p-growLambda"],
  ["growInitSites", "p-growInitSites"],
  ["growMaxSites", "p-growMaxSites"],
  ["growFactor", "p-growFactor"],
  ["growEvery", "p-growEvery"],
  ["growBlur", "p-growBlur"],
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
  lastActiveSites: 0,    // active sites at the last log tick (site-spacing reference lines)
  psnrHistory: [],       // [{ it, psnr }] from log ticks, plotted in the PSNR card
  psnrIters: 0,
  bppRuns: [],           // [{ sites, bpp, colorSpace, points: [{ it, psnr }] }] per run on this target; last is current
  gradHistory: [],       // [{ it, stats }] per-param |grad| mean/max from log ticks
  lrHistory: [],         // [{ it, lr }] LR multiplier (schedule × LR mult) from log ticks
  lrPlan: null,          // { warmupIters, decayIters, lrMult, kind, growDoneIter } of the current run
  bandHistory: [],       // [{ it, snr: [dB per band] }] from every render
  targetAnalysis: null,  // viewer-side target luma / pyramid / spectrum (see targetAnalysis())
  spectra: null,         // spectra of the latest render (see currentSpectra())
  importanceImage: null, // RGBA8 importance map for the current target + mask
  layoutName: null,      // active key in LAYOUTS
  editing: false,        // layout edit mode (drag to move, add / remove, rows & cols)
  dragId: null,          // view being dragged in edit mode
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
  initSitesClear: document.getElementById("initSitesClear"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  loopToggle: document.getElementById("loopToggle"),
  toggleParamsBtn: document.getElementById("toggleParamsBtn"),
  saveBtn: document.getElementById("saveBtn"),
  savePopover: document.getElementById("savePopover"),
  savePng: document.getElementById("savePng"),
  saveTxt: document.getElementById("saveTxt"),
  paramsPanel: document.getElementById("paramsPanel"),
  layoutSelect: document.getElementById("layoutSelect"),
  editLayoutBtn: document.getElementById("editLayoutBtn"),
  layoutEditTools: document.getElementById("layoutEditTools"),
  addRowBtn: document.getElementById("addRowBtn"),
  removeRowBtn: document.getElementById("removeRowBtn"),
  addColBtn: document.getElementById("addColBtn"),
  removeColBtn: document.getElementById("removeColBtn"),
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
  const spaceSelect = document.getElementById("p-colorSpace");
  spaceSelect.replaceChildren(...Object.entries(COLOR_SPACES).map(([key, space]) => new Option(space.label, key)));
  document.getElementById("p-lrSchedule")
    .replaceChildren(...Object.entries(LR_SCHEDULES).map(([key, label]) => new Option(label, key)));
  const defaults = buildArgs(config, {});
  for (const [key, elId, opts] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    const v = opts?.blankDefault ? undefined : defaults[key];
    if (el.type === "checkbox") el.checked = Boolean(v);
    else if (v === undefined) el.value = "";
    else el.value = v;
  }
  syncSiteFields();
}

function readFormArgs() {
  const overrides = {};
  for (const [key, elId] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    if (el.disabled) continue; // locked to a loaded sites file's count (syncSiteFields)
    if (el.type === "checkbox") overrides[key] = el.checked;
    else if (el.tagName === "SELECT") overrides[key] = el.value;
    else if (el.value === "") continue;
    else overrides[key] = Number(el.value);
  }
  if (overrides.logFreq !== undefined) overrides.viewerFreq = overrides.logFreq;
  return buildArgs(state.config, overrides);
}

// Show only the chosen site mode's fields, and lock its starting-count field
// (sites when fixed, start sites when growing) to a loaded sites file's count,
// restoring the typed value once the file is cleared.
function syncSiteFields() {
  const mode = document.getElementById("p-siteMode").value;
  ui.paramsPanel.dataset.siteMode = mode;
  const locked = mode === "grow" ? "p-growInitSites" : "p-sites";
  for (const id of ["p-sites", "p-growInitSites"]) {
    const el = document.getElementById(id);
    const lock = id === locked && state.initSites != null;
    if (lock && !el.disabled) el.dataset.typed = el.value;
    if (!lock && el.disabled) el.value = el.dataset.typed ?? el.value;
    el.disabled = lock;
    if (lock) el.value = state.initSites.count;
    el.title = lock ? "Set by the loaded sites file (Clear it to edit)" : "";
  }
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
// device resolution. `blank` is the fill shown before the first render.
// Only views named in the active layout get a card; the others keep
// canvas = null and their draw functions return early, so hidden views cost
// nothing (band SNR history is the exception: it's recorded on every render).
//
// `source` says where a view's data comes from and is shown as a badge on its
// card; `note` (shown on hover) says what exactly is drawn. `on` lists the
// events that redraw the view:
//   target – a new target or mask      render – a new reconstruction
//   log    – a log tick's metrics      run    – a run starting
// Plot views also redraw on resize and hover; a plot with `hit` is hoverable.
const SOURCES = {
  input:   { badge: "Input",    desc: "Input: what you loaded." },
  model:   { badge: "Model",    desc: "Model: rendered from the model's current state by the trainer (GPU)." },
  train:   { badge: "Training", desc: "Training: reported by the trainer while it optimizes." },
  derived: { badge: "Viz only", desc: "Viz only: computed in the viewer (CPU) for display. The optimizer never sees it." },
};
const VIEW_EVENTS = ["target", "render", "log", "run"];

const VIEWS = {
  target:     { label: "Target", kind: "image", source: "input", on: ["target"], draw: drawTarget,
                note: `The target image, downscaled to at most ${DEMO_MAX_IMAGE_SIDE} px per side.` },
  preview:    { label: "Reconstruction + centroids", kind: "image", source: "model", overlay: true, blank: "#0a0a0f",
                on: ["render"], draw: drawPreview,
                note: "The trainer's latest render, with site centroids blinking on top." },
  residual:   { label: "Residual |recon − target|", kind: "image", source: "derived", blank: "#000004",
                on: ["render"], draw: drawResidual,
                note: "Mean |recon − target| over RGB on a fixed 0–0.25 scale." },
  importance: { label: "Importance (Laplacian bands)", kind: "image", source: "derived", on: ["target"], draw: drawImportance,
                note: `Luma Laplacian band energy, mask · (ε + √Σ β_k L_k²)^γ with β = [${IMPORTANCE.bandWeights.join(", ")}]. ` +
                      "Not used by the optimizer." },
  siteIds:    { label: "Site IDs", kind: "image", source: "model", blank: "#0a0a0f", on: ["render"],
                draw: () => drawGpuView("siteIds", "hashed"),
                note: "Each pixel's winning site, hashed to a color (extra GPU render pass)." },
  tau:        { label: "Tau heatmap", kind: "image", source: "model", blank: "#0a0a0f", on: ["render"],
                draw: () => drawGpuView("tau", "heatmap"),
                note: "Per-site tau as a heatmap (extra GPU render pass)." },
  psnr:       { label: "PSNR vs iteration", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawPsnrPlot, hit: hitPsnr,
                note: "Masked PSNR the trainer computes at each log tick." },
  bpp:        { label: "PSNR vs bpp", kind: "plot", source: "derived", on: ["log", "run", "target"],
                draw: drawBppPlot, hit: hitBpp,
                note: `Estimated rate: active sites × ${BITS_PER_SITE} bits / pixels. The bits per site are an ` +
                      "assumption, not a measured code length. PSNR comes from the trainer." },
  grad:       { label: "|gradient| vs iteration (log)", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawGradPlot, hit: hitGrad,
                note: "Per-parameter-group |gradient| over sites, reported by the trainer at each log tick." },
  lr:         { label: "LR multiplier vs iteration", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawLrPlot, hit: hitLr,
                note: "LR multiplier (warmup / cosine-decay schedule × LR mult) the trainer applied to every base " +
                      "LR at each log tick. Dashed: the run's full planned schedule." },
  bandSnr:    { label: "Band SNR vs iteration", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawBandSnrPlot, hit: hitBandSnr,
                note: "Per luma Laplacian band: 10·log₁₀(Σ L_k(target)² / Σ L_k(recon − target)²), mask-weighted. " +
                      "Recorded on every render, even while this card is hidden." },
  spectrum:   { label: "Radial power spectrum (luma)", kind: "plot", source: "derived", on: ["render", "target"],
                draw: drawSpectrumPlot, hit: hitSpectrum,
                note: "Luma with the mean removed, times a 2D Hann window and the mask, zero-padded FFT; |F|² " +
                      "averaged over log-spaced rings. Dashed line: 1/(2·site spacing), spacing = √(mask area / sites)." },
  agreement:  { label: "Spectral SNR & FRC", kind: "plot", source: "derived", on: ["render", "target"],
                draw: drawAgreementPlot, hit: hitSpectrum,
                note: "Per frequency ring: spectral SNR = 10·log₁₀(P_target / P_residual), and Fourier ring " +
                      "correlation Σ Re(F_r·F_t*) / √(Σ|F_r|² Σ|F_t|²): 1 = matches, 0 = unrelated." },
  spectrum2d: { label: "2D power spectrum", kind: "plot", source: "derived", on: ["render", "target"],
                draw: drawSpectrum2d,
                note: "log₁₀ |F|² with DC at the centre, on one shared scale set by the target; " +
                      "the target tile's axes are fx and fy in cycles/px. " +
                      "Dashed ring: the site-spacing frequency." },
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
    areas: `"psnr bpp  lr"
            "grad grad grad"`,
  },
  spectral: {
    label: "Spectral",
    areas: `"spectrum  spectrum  spectrum2d spectrum2d"
            "agreement agreement bandSnr    bandSnr"`,
  },
};
const DEFAULT_LAYOUT = "overview";
const LAYOUT_STORAGE_KEY = "sad.train.layout";
// Edits in edit mode go to one "Custom" layout (editing a preset copies it
// there first), saved in the same areas format as the presets.
const CUSTOM_LAYOUT = "custom";
const CUSTOM_STORAGE_KEY = "sad.train.customLayout";

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
    view.geom = null;
    view.hoverKey = null;
  }
  const root = ui.canvases;
  root.replaceChildren();
  root.style.setProperty("--layout-areas", grid.map((row) => `"${row.join(" ")}"`).join(" "));
  root.style.setProperty("--layout-cols", `repeat(${grid[0].length}, minmax(0, 1fr))`);
  root.style.setProperty("--layout-rows", layout.rows || `repeat(${grid.length}, minmax(0, 1fr))`);
  for (const id of ids) {
    const view = VIEWS[id];
    const source = SOURCES[view.source];
    const card = document.createElement("div");
    card.className = `canvas-card ${view.kind}`;
    card.dataset.source = view.source;
    card.style.setProperty("--view", id);
    const label = document.createElement("span");
    label.className = "card-label";
    label.title = view.note ? `${source.desc}\n\n${view.note}` : source.desc;
    const badge = document.createElement("span");
    badge.className = "source-badge";
    badge.textContent = source.badge;
    label.append(badge, document.createTextNode(view.label));
    const canvas = document.createElement("canvas");
    card.append(label, canvas);
    if (view.overlay) {
      view.overlayCanvas = document.createElement("canvas");
      view.overlayCanvas.className = "overlay-canvas";
      card.append(view.overlayCanvas);
    }
    if (view.hit) attachPlotHover(view, card, canvas);
    if (state.editing) {
      const r = viewRect(grid, id);
      attachEditControls(view, id, card, label, r.r0, r.c0);
    }
    root.append(card);
    view.card = card;
    view.canvas = canvas;
  }
  root.classList.toggle("editing", state.editing);
  if (state.editing) {
    grid.forEach((row, r) => row.forEach((id, c) => { if (id === ".") root.append(emptyCell(ids, r, c)); }));
  }
  updateEditTools(grid);
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
  state.layoutName = name;
  ui.layoutSelect.value = name;
  try { localStorage.setItem(LAYOUT_STORAGE_KEY, name); } catch {}
  // Fresh canvases are empty: repaint everything we have.
  if (state.targetImage) blankRenderViews(state.targetImage.width, state.targetImage.height);
  await refreshViews(...VIEW_EVENTS);
  updateCanvasLayout();
}

function savedLayout() {
  try { return localStorage.getItem(LAYOUT_STORAGE_KEY) || DEFAULT_LAYOUT; } catch { return DEFAULT_LAYOUT; }
}

// ---- Layout editing ----
// Cards keep the size their layout gives them; the grid only grows or shrinks
// by whole rows / columns at the bottom / right. Dragging a card by its label
// onto another card swaps the two if they're the same size; onto an empty
// cell moves it there (its top-left lands on that cell) if it fits.

function loadCustomLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem(CUSTOM_STORAGE_KEY));
    if (!saved?.areas) return;
    parseLayoutAreas(saved.areas);
    LAYOUTS[CUSTOM_LAYOUT] = { label: "Custom", areas: saved.areas, rows: saved.rows };
  } catch {}
}

function gridToAreas(grid) {
  return grid.map((row) => `"${row.join(" ")}"`).join("\n");
}

// Bounding cells of a view in the grid (views are rectangles).
function viewRect(grid, id) {
  let r0 = Infinity, r1 = -1, c0 = Infinity, c1 = -1;
  grid.forEach((row, r) => row.forEach((v, c) => {
    if (v !== id) return;
    r0 = Math.min(r0, r); r1 = Math.max(r1, r);
    c0 = Math.min(c0, c); c1 = Math.max(c1, c);
  }));
  return { r0, r1, c0, c1 };
}

function fillRect(grid, { r0, r1, c0, c1 }, id) {
  for (let r = r0; r <= r1; r += 1) for (let c = c0; c <= c1; c += 1) grid[r][c] = id;
}

// Moves view `id` so it covers cell (r, c); returns false if not allowed.
function moveView(grid, id, r, c) {
  const s = viewRect(grid, id);
  const h = s.r1 - s.r0, w = s.c1 - s.c0;
  const occupant = grid[r][c];
  if (occupant !== "." && occupant !== id) {
    const t = viewRect(grid, occupant);
    if (t.r1 - t.r0 !== h || t.c1 - t.c0 !== w) return false;
    fillRect(grid, s, occupant);
    fillRect(grid, t, id);
    return true;
  }
  const dest = { r0: r, r1: r + h, c0: c, c1: c + w };
  if (dest.r1 >= grid.length || dest.c1 >= grid[0].length) return false;
  for (let y = dest.r0; y <= dest.r1; y += 1) {
    for (let x = dest.c0; x <= dest.c1; x += 1) if (grid[y][x] !== "." && grid[y][x] !== id) return false;
  }
  fillRect(grid, s, ".");
  fillRect(grid, dest, id);
  return true;
}

function activeGrid() {
  return parseLayoutAreas(LAYOUTS[state.layoutName].areas).grid.map((row) => [...row]);
}

// Applies `mutate` to a copy of the active grid and switches to the result as
// the Custom layout. `mutate` returns false to refuse; an invalid result is
// refused too. Explicit row sizes survive only while the row count does.
async function editLayout(mutate) {
  const base = LAYOUTS[state.layoutName];
  const grid = activeGrid();
  const rowCount = grid.length;
  if (mutate(grid) === false) return;
  const areas = gridToAreas(grid);
  try {
    parseLayoutAreas(areas);
  } catch (err) {
    appendLog(`Layout edit refused: ${err.message}`);
    return;
  }
  LAYOUTS[CUSTOM_LAYOUT] = { label: "Custom", areas, rows: grid.length === rowCount ? base.rows : undefined };
  try {
    localStorage.setItem(CUSTOM_STORAGE_KEY, JSON.stringify({ areas, rows: LAYOUTS[CUSTOM_LAYOUT].rows }));
  } catch {}
  if (![...ui.layoutSelect.options].some((o) => o.value === CUSTOM_LAYOUT)) {
    ui.layoutSelect.add(new Option("Custom", CUSTOM_LAYOUT));
  }
  await setLayout(CUSTOM_LAYOUT);
}

function canMove(id, r, c) {
  return Boolean(id) && moveView(activeGrid(), id, r, c);
}

// Highlights a drop target while a card is dragged over it, and moves the
// card on drop. (r, c) is the cell the dragged card lands on.
function attachDropTarget(el, r, c) {
  const clear = () => el.classList.remove("drop-ok", "drop-bad");
  el.addEventListener("dragover", (ev) => {
    if (!state.dragId) return;
    const ok = canMove(state.dragId, r, c);
    el.classList.toggle("drop-ok", ok);
    el.classList.toggle("drop-bad", !ok);
    if (ok) {
      ev.preventDefault();
      ev.dataTransfer.dropEffect = "move";
    }
  });
  el.addEventListener("dragleave", clear);
  el.addEventListener("drop", (ev) => {
    ev.preventDefault();
    clear();
    const id = state.dragId;
    state.dragId = null;
    editLayout((grid) => moveView(grid, id, r, c));
  });
}

function attachEditControls(view, id, card, label, r, c) {
  label.draggable = true;
  label.addEventListener("dragstart", (ev) => {
    state.dragId = id;
    ev.dataTransfer.setData("text/plain", id);
    ev.dataTransfer.effectAllowed = "move";
    card.classList.add("dragging");
    ui.canvases.classList.add("drag-active");
  });
  label.addEventListener("dragend", () => {
    state.dragId = null;
    card.classList.remove("dragging");
    ui.canvases.classList.remove("drag-active");
  });
  attachDropTarget(card, r, c);
  const remove = document.createElement("button");
  remove.className = "card-remove";
  remove.textContent = "×";
  remove.title = `Remove ${view.label}`;
  remove.setAttribute("aria-label", `Remove ${view.label}`);
  remove.addEventListener("click", () => editLayout((grid) => fillRect(grid, viewRect(grid, id), ".")));
  card.append(remove);
}

// An empty cell in edit mode: a drop target with a picker of unplaced views,
// grouped by source.
function emptyCell(placed, r, c) {
  const cell = document.createElement("div");
  cell.className = "empty-cell";
  cell.style.gridRow = String(r + 1);
  cell.style.gridColumn = String(c + 1);
  attachDropTarget(cell, r, c);
  const free = Object.keys(VIEWS).filter((id) => !placed.includes(id));
  if (!free.length) {
    cell.textContent = "All views placed";
    return cell;
  }
  const select = document.createElement("select");
  select.setAttribute("aria-label", `Add a view at row ${r + 1}, column ${c + 1}`);
  select.add(new Option("+ Add view…", ""));
  for (const [key, source] of Object.entries(SOURCES)) {
    const ids = free.filter((id) => VIEWS[id].source === key);
    if (!ids.length) continue;
    const group = document.createElement("optgroup");
    group.label = source.badge;
    for (const id of ids) group.append(new Option(VIEWS[id].label, id));
    select.append(group);
  }
  select.addEventListener("change", () => {
    const id = select.value;
    if (id) editLayout((grid) => { grid[r][c] = id; });
  });
  cell.append(select);
  return cell;
}

const rowEmpty = (grid, r) => grid[r].every((v) => v === ".");
const colEmpty = (grid, c) => grid.every((row) => row[c] === ".");

function updateEditTools(grid) {
  ui.editLayoutBtn.textContent = state.editing ? "Done" : "Edit layout";
  ui.editLayoutBtn.classList.toggle("btn-primary", state.editing);
  ui.layoutEditTools.classList.toggle("hidden", !state.editing);
  // Only an empty last row / column can go, and the grid keeps at least one.
  ui.removeRowBtn.disabled = grid.length < 2 || !rowEmpty(grid, grid.length - 1);
  ui.removeColBtn.disabled = grid[0].length < 2 || !colEmpty(grid, grid[0].length - 1);
}

function toggleEditing() {
  state.editing = !state.editing;
  setLayout(state.layoutName);
}

// Redraws the views in the layout that listen for any of `events`, in VIEWS
// order. GPU-backed views await their render pass.
async function refreshViews(...events) {
  for (const view of Object.values(VIEWS)) {
    if (!view.canvas || !view.on.some((e) => events.includes(e))) continue;
    await view.draw(view);
  }
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
    else view.draw(view);
  }
  positionOverlay();
}

// Builds a 256-entry RGB lookup from [t, [r, g, b]] stops.
function rampLut(stops) {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let k = 1;
    while (k < stops.length - 1 && stops[k][0] < t) k += 1;
    const [t0, c0] = stops[k - 1];
    const [t1, c1] = stops[k];
    const f = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c += 1) lut[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
  }
  return lut;
}

// ---- Plot kit ----
// Shared drawing for every plot view: a DPR-sized canvas, linear or log axes
// over a box, a recessive grid, series lines, markers, a legend and a table
// tooltip. A plot's draw() stores its axes in view.geom; the hover handler
// passes them to view.hit(geom, mx, my), which returns a primitive key (or
// null) that draw() reads back from view.hoverKey.

const PLOT_FONT = "10.5px Inter, system-ui, -apple-system, sans-serif";
const TIP_FONT = "11px Inter, system-ui, -apple-system, sans-serif";
const PLOT_PAD = { left: 46, right: 16, top: 42, bottom: 34 };
const LEGEND_PAD = { left: 56, right: 16, bottom: 34 }; // plots whose top follows a legend
// Categorical hues in fixed order, validated against --canvas-bg. The lighter
// ones fall under 3:1, so every plot using them has a legend and a tooltip.
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
const PSNR_LINE_COLOR = "#e0611a"; // darker accent: >= 3:1 against --canvas-bg
const SIGNALS = [
  { key: "target", label: "target", color: SERIES_COLORS[0] },
  { key: "recon", label: "recon", color: SERIES_COLORS[1] },
  { key: "residual", label: "residual", color: SERIES_COLORS[2] },
];

function beginPlot(view) {
  const canvas = view.canvas;
  if (!canvas) return null;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return null;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = PLOT_FONT;
  ctx.lineWidth = 1;
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const col = {
    line: v("--line", "#e6e6df"),
    dim: v("--text-dim", "#66666e"),
    soft: v("--text-soft", "#9a9aa2"),
    text: v("--text", "#1d1d1f"),
    surface: v("--canvas-bg", "#f1f1ec"),
  };
  return { ctx, w, h, col };
}

function plotBox(p, top = PLOT_PAD.top, pad = PLOT_PAD) {
  return { left: pad.left, right: p.w - pad.right, top, bottom: p.h - pad.bottom };
}

// Axes over a pixel box; xr / yr are { min, max, log }. Log axes map log10.
function plotAxes(box, xr, yr) {
  const f = (r) => (r.log ? Math.log10 : (v) => v);
  const fx = f(xr), fy = f(yr);
  const x0 = fx(xr.min), x1 = fx(xr.max);
  const y0 = fy(yr.min), y1 = fy(yr.max);
  return {
    ...box, xr, yr,
    x: (v) => box.left + ((fx(v) - x0) / (x1 - x0 || 1)) * (box.right - box.left),
    y: (v) => box.bottom - ((fy(v) - y0) / (y1 - y0 || 1)) * (box.bottom - box.top),
    ok: (x, y) => Number.isFinite(x) && Number.isFinite(y) && (!xr.log || x > 0) && (!yr.log || y > 0),
  };
}

// Linear range padded around [lo, hi] and rounded out to whole units.
function linRange(lo, hi, fallback) {
  if (!Number.isFinite(lo)) [lo, hi] = fallback;
  const pad = Math.max(0.5, (hi - lo) * 0.08);
  return { min: Math.floor(lo - pad), max: Math.ceil(hi + pad) };
}

// Log range rounded out to whole decades (at least one).
function decadeRange(lo, hi, fallback) {
  if (!(lo > 0) || !Number.isFinite(hi)) [lo, hi] = fallback;
  const e0 = Math.floor(Math.log10(lo));
  const e1 = Math.max(e0 + 1, Math.ceil(Math.log10(hi)));
  return { min: 10 ** e0, max: 10 ** e1, log: true };
}

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

const isDecade = (v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9;

// Ticks for a range spanning `px` pixels, about `spacing` px apart. Log axes
// use 1-2-5 steps, thinned to decades when crowded.
function axisTicks(r, px, spacing) {
  const max = Math.max(2, Math.floor(px / spacing));
  if (!r.log) return niceTicks(r.min, r.max, max);
  const t = logTicks(r.min, r.max);
  return t.length > max ? t.filter(isDecade) : t;
}

const fmtNum = (v) => String(Math.round(v * 10) / 10 || 0);
const fmtIter = (v) => (v >= 10000 ? `${v / 1000}k` : String(Number(v.toPrecision(6))));
const fmtSig = (v) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : String(Number(v.toPrecision(2))));
const fmtExp = (v) => (v > 0 ? v.toExponential(2) : "0");
const fmtLogTick = (v) => (isDecade(v) ? `1e${Math.round(Math.log10(v))}` : v.toExponential(0));

// Grid + y labels, baseline + x labels, axis titles.
function drawAxes(p, g, { xTitle, yTitle, xFmt = fmtNum, yFmt = fmtNum, xSpacing = 70, ySpacing = 48, xLabels = true } = {}) {
  const { ctx, col } = p;
  ctx.font = PLOT_FONT;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of axisTicks(g.yr, g.bottom - g.top, ySpacing)) {
    const y = Math.round(g.y(v)) + 0.5;
    ctx.strokeStyle = col.line;
    ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(g.right, y); ctx.stroke();
    ctx.fillStyle = col.soft;
    ctx.fillText(yFmt(v), g.left - 6, y);
  }
  ctx.strokeStyle = col.soft;
  ctx.beginPath(); ctx.moveTo(g.left, g.bottom + 0.5); ctx.lineTo(g.right, g.bottom + 0.5); ctx.stroke();
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  if (xLabels) {
    ctx.fillStyle = col.soft;
    for (const v of axisTicks(g.xr, g.right - g.left, xSpacing)) ctx.fillText(xFmt(v), g.x(v), g.bottom + 6);
  }
  ctx.fillStyle = col.dim;
  if (xTitle) ctx.fillText(xTitle, (g.left + g.right) / 2, g.bottom + 20);
  if (yTitle) {
    ctx.save();
    ctx.translate(12, (g.top + g.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(yTitle, 0, 0);
    ctx.restore();
  }
}

function drawEmpty(p, g, msg) {
  const { ctx, col } = p;
  ctx.font = PLOT_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = col.soft;
  ctx.fillText(msg, (g.left + g.right) / 2, (g.top + g.bottom) / 2);
}

// Polyline through [x, y] points; points the axes can't show break the line.
function drawLine(p, g, pts, color, { dash = [], width = 2 } = {}) {
  const { ctx } = p;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  let pen = false;
  for (const [x, y] of pts) {
    if (!g.ok(x, y)) { pen = false; continue; }
    if (pen) ctx.lineTo(g.x(x), g.y(y)); else ctx.moveTo(g.x(x), g.y(y));
    pen = true;
  }
  ctx.stroke();
  ctx.restore();
}

function drawMarker(p, g, x, y, color, r = 4) {
  if (!g.ok(x, y)) return;
  const { ctx, col } = p;
  ctx.beginPath();
  ctx.arc(g.x(x), g.y(y), r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = col.surface;
  ctx.stroke();
}

function drawCrosshair(p, g, x) {
  const { ctx, col } = p;
  const hx = Math.round(g.x(x)) + 0.5;
  ctx.save();
  ctx.strokeStyle = col.soft;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(hx, g.top); ctx.lineTo(hx, g.bottom); ctx.stroke();
  ctx.restore();
}

// Dashed reference line at x = v (or y = v) with an optional label.
function drawRefLine(p, g, { x, y, label }) {
  const { ctx, col } = p;
  ctx.save();
  ctx.strokeStyle = col.dim;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.font = PLOT_FONT;
  ctx.fillStyle = col.dim;
  ctx.beginPath();
  if (x != null) {
    if (!(x >= g.xr.min && x <= g.xr.max)) { ctx.restore(); return; }
    const px = Math.round(g.x(x)) + 0.5;
    ctx.moveTo(px, g.top); ctx.lineTo(px, g.bottom); ctx.stroke();
    if (label) {
      const right = px + 4 + ctx.measureText(label).width > g.right;
      ctx.textAlign = right ? "right" : "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, right ? px - 4 : px + 4, g.top + 2);
    }
  } else {
    const py = Math.round(g.y(y)) + 0.5;
    ctx.moveTo(g.left, py); ctx.lineTo(g.right, py); ctx.stroke();
    if (label) {
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, g.right - 2, py - 2);
    }
  }
  ctx.restore();
}

// Legend wrapping under the card label; items are { label, color?, dash? }.
// Returns the y just below it.
function drawLegend(p, items, y0 = 46) {
  const { ctx, col, w } = p;
  ctx.font = PLOT_FONT;
  const x0 = 12, rowH = 16, swatch = 18;
  let x = x0, y = y0;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const it of items) {
    const iw = swatch + 6 + ctx.measureText(it.label).width + 14;
    if (x + iw > w - 12 && x > x0) { x = x0; y += rowH; }
    ctx.save();
    ctx.strokeStyle = it.color || col.dim;
    ctx.lineWidth = 2;
    ctx.setLineDash(it.dash || []);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + swatch, y); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = col.dim;
    ctx.fillText(it.label, x + swatch + 6, y);
    x += iw;
  }
  return y + 14;
}

// Table tooltip next to (ax, ay); ay = null pins it to the top of the plot.
// tip = { header?: [cells], rows: [{ color?, cells }], footer?: string }. The
// first column is left-aligned, the rest right-aligned.
function drawTooltip(p, g, ax, ay, tip) {
  const { ctx, col, w, h } = p;
  ctx.font = TIP_FONT;
  const lines = [...(tip.header ? [tip.header] : []), ...tip.rows.map((r) => r.cells)];
  const ncol = Math.max(...lines.map((c) => c.length));
  const colW = new Array(ncol).fill(0);
  for (const cells of lines) cells.forEach((c, j) => { colW[j] = Math.max(colW[j], ctx.measureText(c).width); });
  const rowH = 17, padX = 8, gap = 14;
  const sw = tip.rows.some((r) => r.color) ? 18 : 0;
  const tableW = colW.reduce((a, b) => a + b, 0) + gap * (ncol - 1);
  const tw = Math.max(padX * 2 + sw + tableW, tip.footer ? padX * 2 + ctx.measureText(tip.footer).width : 0);
  const th = rowH * (lines.length + (tip.footer ? 1 : 0)) + 8;
  let tx = ax + 12;
  if (tx + tw > g.right) tx = ax - 12 - tw;
  tx = Math.max(4, Math.min(w - tw - 4, tx));
  const want = ay == null ? g.top : ay - th - 8;
  const ty = Math.max(4, Math.min(h - th - 4, Math.max(g.top, Math.min(g.bottom - th, want))));
  ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
  ctx.strokeStyle = col.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tx, ty, tw, th, 5);
  ctx.fill();
  ctx.stroke();

  const xs = [];
  let cx = tx + padX + sw;
  colW.forEach((cw, j) => {
    if (j === 0) { xs.push(cx); cx += cw; } else { cx += gap + cw; xs.push(cx); }
  });
  ctx.textBaseline = "middle";
  const drawRow = (cells, y, color) => {
    ctx.fillStyle = color;
    cells.forEach((c, j) => {
      ctx.textAlign = j === 0 ? "left" : "right";
      ctx.fillText(c, xs[j], y);
    });
  };
  let y = ty + 4 + rowH / 2;
  if (tip.header) { drawRow(tip.header, y, col.dim); y += rowH; }
  for (const r of tip.rows) {
    if (r.color) {
      ctx.fillStyle = r.color;
      ctx.fillRect(tx + padX, y - 1, 12, 2.5);
    }
    drawRow(r.cells, y, col.text);
    y += rowH;
  }
  if (tip.footer) {
    ctx.fillStyle = col.dim;
    ctx.textAlign = "left";
    ctx.fillText(tip.footer, tx + padX, y);
  }
}

// Index of the item whose x is nearest the cursor, or null when there are none.
function nearestX(g, items, getX, mx) {
  let best = null, bestDist = Infinity;
  items.forEach((it, i) => {
    const d = Math.abs(g.x(getX(it)) - mx);
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

function attachPlotHover(view, card, canvas) {
  card.classList.add("hoverable");
  canvas.addEventListener("mousemove", (ev) => {
    if (!view.geom) return;
    const rect = canvas.getBoundingClientRect();
    const key = view.hit(view.geom, ev.clientX - rect.left, ev.clientY - rect.top);
    if (key !== view.hoverKey) {
      view.hoverKey = key;
      view.draw(view);
    }
  });
  canvas.addEventListener("mouseleave", () => {
    if (view.hoverKey == null) return;
    view.hoverKey = null;
    view.draw(view);
  });
}

// ---- PSNR vs iteration ----

function iterRange() {
  const last = Math.max(state.psnrHistory.at(-1)?.it ?? 0, state.gradHistory.at(-1)?.it ?? 0,
                        state.bandHistory.at(-1)?.it ?? 0, state.lrHistory.at(-1)?.it ?? 0);
  return { min: 0, max: Math.max(1, state.psnrIters, last) };
}

function drawPsnrPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const data = state.psnrHistory;
  let lo = Infinity, hi = -Infinity;
  for (const d of data) { lo = Math.min(lo, d.psnr); hi = Math.max(hi, d.psnr); }
  const g = view.geom = plotAxes(plotBox(p), iterRange(), linRange(lo, hi, [0, 40]));
  drawAxes(p, g, { xTitle: "iteration", yTitle: "PSNR (dB)", xFmt: fmtIter });
  if (!data.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see PSNR");
    return;
  }
  drawLine(p, g, data.map((d) => [d.it, d.psnr]), PSNR_LINE_COLOR);
  const hover = view.hoverKey != null ? data[view.hoverKey] : null;
  if (hover) drawCrosshair(p, g, hover.it);
  const mark = hover || data[data.length - 1];
  drawMarker(p, g, mark.it, mark.psnr, PSNR_LINE_COLOR);
  if (hover) {
    drawTooltip(p, g, g.x(hover.it), g.y(hover.psnr),
      { rows: [{ cells: [`iter ${hover.it} · ${hover.psnr.toFixed(2)} dB`] }] });
  }
}

function hitPsnr(g, mx) {
  return nearestX(g, state.psnrHistory, (d) => d.it, mx);
}

// ---- PSNR vs bpp (log x) ----
// Raw rate: active sites × BITS_PER_SITE / pixels. Runs on the current target
// accumulate: earlier runs show as their final point (joined in bpp order, an
// RD curve per site color space); the current run shows its log-tick trail at
// its fixed bpp.

const BPP_PAST_COLOR = "#66666e";

// RGB keeps the neutral past-run color; other spaces take series hues.
function colorSpaceColor(key) {
  const i = Object.keys(COLOR_SPACES).indexOf(key);
  return i <= 0 ? BPP_PAST_COLOR : SERIES_COLORS[(i - 1) % SERIES_COLORS.length];
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

function drawBppPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const pts = bppPlotPoints();
  let xMin = Infinity, xMax = -Infinity, lo = Infinity, hi = -Infinity;
  for (const r of state.bppRuns) { xMin = Math.min(xMin, r.bpp); xMax = Math.max(xMax, r.bpp); }
  for (const pt of pts) { lo = Math.min(lo, pt.y); hi = Math.max(hi, pt.y); }
  if (!(xMin > 0)) { xMin = 0.1; xMax = 10; }
  // At least a decade of x range, centred on the data.
  const span = Math.max(1, Math.log10(xMax / xMin) * 1.2);
  const mid = (Math.log10(xMin) + Math.log10(xMax)) / 2;
  const xr = { min: 10 ** (mid - span / 2), max: 10 ** (mid + span / 2), log: true };
  // Color-space legend only once runs in more than one space are on the plot.
  const spaces = [...new Set(state.bppRuns.map((r) => r.colorSpace))];
  const top = spaces.length > 1
    ? drawLegend(p, spaces.map((key) => ({ label: COLOR_SPACES[key].label, color: colorSpaceColor(key) }))) + 10
    : undefined;
  const g = view.geom = plotAxes(plotBox(p, top), xr, linRange(lo, hi, [0, 40]));
  drawAxes(p, g, { xTitle: `bpp (${BITS_PER_SITE} bits/site)`, yTitle: "PSNR (dB)", xFmt: fmtSig, xSpacing: 40 });
  if (!pts.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see PSNR vs bpp");
    return;
  }

  // Earlier runs: final points joined in bpp order, one curve per color space
  for (const space of spaces) {
    const color = colorSpaceColor(space);
    const past = pts.filter((pt) => !pt.current && state.bppRuns[pt.run].colorSpace === space)
      .sort((a, b) => a.x - b.x);
    if (past.length > 1) drawLine(p, g, past.map((pt) => [pt.x, pt.y]), color, { width: 1.5 });
    for (const pt of past) drawMarker(p, g, pt.x, pt.y, color, 3.5);
  }

  // Current run: vertical trail of log ticks, latest marked
  const cur = pts.filter((pt) => pt.current);
  if (cur.length) {
    const ys = cur.map((pt) => pt.y);
    drawLine(p, g, [[cur[0].x, Math.min(...ys)], [cur[0].x, Math.max(...ys)]], PSNR_LINE_COLOR);
    const last = cur[cur.length - 1];
    drawMarker(p, g, last.x, last.y, PSNR_LINE_COLOR);
  }

  const hover = view.hoverKey && pts.find((pt) => `${pt.run}:${pt.point}` === view.hoverKey);
  if (!hover) return;
  const run = state.bppRuns[hover.run];
  drawMarker(p, g, hover.x, hover.y, hover.current ? PSNR_LINE_COLOR : colorSpaceColor(run.colorSpace), 5);
  drawTooltip(p, g, g.x(hover.x), g.y(hover.y), {
    rows: [{ cells: [`${run.sites} sites · ${fmtSig(hover.x)} bpp · ${hover.y.toFixed(2)} dB · ${COLOR_SPACES[run.colorSpace].label}`] }],
  });
}

function hitBpp(g, mx, my) {
  let best = null, bestDist = Infinity;
  for (const pt of bppPlotPoints()) {
    const d = Math.hypot(g.x(pt.x) - mx, g.y(pt.y) - my);
    if (d < bestDist) { bestDist = d; best = pt; }
  }
  return best && bestDist < 40 ? `${best.run}:${best.point}` : null;
}

// ---- |gradient| vs iteration (log y) ----
// One hue per param group (fixed order); solid = max over sites, dashed =
// mean over sites with any nonzero grad.

const GRAD_MEAN_DASH = [6, 4];

function drawGradPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const items = GRAD_GROUPS.map((grp, i) => ({ label: grp.label, color: SERIES_COLORS[i] }));
  items.push({ label: "max" }, { label: "mean", dash: GRAD_MEAN_DASH });
  const top = drawLegend(p, items) + 10;
  const data = state.gradHistory;
  let lo = Infinity, hi = -Infinity;
  for (const d of data) {
    for (const grp of GRAD_GROUPS) {
      const s = d.stats.groups[grp.key];
      for (const v of [s.mean, s.max]) {
        if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
  }
  const g = view.geom = plotAxes(plotBox(p, top, LEGEND_PAD), iterRange(), decadeRange(lo, hi, [1e-8, 1e-2]));
  drawAxes(p, g, { xTitle: "iteration", yTitle: "|grad|", xFmt: fmtIter, yFmt: fmtLogTick, ySpacing: 22 });
  if (!data.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see gradients");
    return;
  }
  GRAD_GROUPS.forEach((grp, i) => {
    drawLine(p, g, data.map((d) => [d.it, d.stats.groups[grp.key].mean]), SERIES_COLORS[i], { dash: GRAD_MEAN_DASH });
    drawLine(p, g, data.map((d) => [d.it, d.stats.groups[grp.key].max]), SERIES_COLORS[i]);
  });

  const hover = view.hoverKey != null ? data[view.hoverKey] : null;
  if (!hover) return;
  drawCrosshair(p, g, hover.it);
  GRAD_GROUPS.forEach((grp, i) => {
    for (const field of ["mean", "max"]) drawMarker(p, g, hover.it, hover.stats.groups[grp.key][field], SERIES_COLORS[i]);
  });
  drawTooltip(p, g, g.x(hover.it), null, {
    header: [`iter ${hover.it}`, "mean", "max"],
    rows: GRAD_GROUPS.map((grp, i) => ({
      color: SERIES_COLORS[i],
      cells: [grp.label, fmtExp(hover.stats.groups[grp.key].mean), fmtExp(hover.stats.groups[grp.key].max)],
    })),
    footer: `${hover.stats.sites} sites with nonzero grad`,
  });
}

function hitGrad(g, mx) {
  return nearestX(g, state.gradHistory, (d) => d.it, mx);
}

// ---- LR multiplier vs iteration ----

function drawLrPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const data = state.lrHistory;
  const plan = state.lrPlan;
  const xr = iterRange();
  // The planned schedule, sampled ~1 point per 2 px of plot width.
  const planned = [];
  if (plan) {
    const n = Math.max(2, Math.floor((p.w - PLOT_PAD.left - PLOT_PAD.right) / 2));
    const total = plan.warmupIters + plan.decayIters;
    for (let i = 0; i <= n; i += 1) {
      const it = Math.round((i / n) * Math.max(0, total - 1));
      planned.push([it, plan.lrMult * lrSchedule(it, plan.warmupIters, plan.decayIters, plan.kind, plan.growDoneIter)]);
    }
  }
  let hi = 0;
  for (const [, v] of planned) hi = Math.max(hi, v);
  for (const d of data) hi = Math.max(hi, d.lr);
  const yr = { min: 0, max: hi > 0 ? niceTicks(0, hi * 1.08, 4).at(-1) || hi * 1.08 : 1 };
  if (yr.max < hi) yr.max = hi * 1.08;
  const g = view.geom = plotAxes(plotBox(p), xr, yr);
  drawAxes(p, g, { xTitle: "iteration", yTitle: "LR multiplier", xFmt: fmtIter, yFmt: (v) => String(Number(v.toPrecision(3))) });
  if (planned.length) drawLine(p, g, planned, PSNR_LINE_COLOR, { dash: [5, 4], width: 1.5 });
  if (!data.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see the LR schedule");
    return;
  }
  drawLine(p, g, data.map((d) => [d.it, d.lr]), PSNR_LINE_COLOR);
  const hover = view.hoverKey != null ? data[view.hoverKey] : null;
  if (hover) drawCrosshair(p, g, hover.it);
  const mark = hover || data[data.length - 1];
  drawMarker(p, g, mark.it, mark.lr, PSNR_LINE_COLOR);
  if (hover) {
    drawTooltip(p, g, g.x(hover.it), g.y(hover.lr),
      { rows: [{ cells: [`iter ${hover.it} · LR ×${hover.lr.toPrecision(3)}`] }] });
  }
}

function hitLr(g, mx) {
  return nearestX(g, state.lrHistory, (d) => d.it, mx);
}

// ---- Band SNR vs iteration ----
// Laplacian band k of luma holds periods of roughly 2^(k+1)–2^(k+2) px.

const BAND_LEVELS = IMPORTANCE.levels;
const bandLabel = (k) => `L${k} · ${2 ** (k + 1)}–${2 ** (k + 2)} px`;

function drawBandSnrPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const data = state.bandHistory;
  const nb = data.length ? data[0].snr.length : BAND_LEVELS;
  const items = Array.from({ length: nb }, (_, k) => ({ label: bandLabel(k), color: SERIES_COLORS[k] }));
  const top = drawLegend(p, items) + 10;
  let lo = Infinity, hi = -Infinity;
  for (const d of data) for (const v of d.snr) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const g = view.geom = plotAxes(plotBox(p, top, LEGEND_PAD), iterRange(), linRange(lo, hi, [0, 30]));
  drawAxes(p, g, { xTitle: "iteration", yTitle: "band SNR (dB)", xFmt: fmtIter });
  if (!data.length) {
    drawEmpty(p, g, state.running ? "Waiting for first render…" : "Train to see band SNR");
    return;
  }
  for (let k = 0; k < nb; k += 1) drawLine(p, g, data.map((d) => [d.it, d.snr[k]]), SERIES_COLORS[k]);
  const hover = view.hoverKey != null ? data[view.hoverKey] : null;
  if (!hover) return;
  drawCrosshair(p, g, hover.it);
  hover.snr.forEach((v, k) => drawMarker(p, g, hover.it, v, SERIES_COLORS[k]));
  drawTooltip(p, g, g.x(hover.it), null, {
    header: [`iter ${hover.it}`, "SNR"],
    rows: hover.snr.map((v, k) => ({ color: SERIES_COLORS[k], cells: [bandLabel(k), `${v.toFixed(1)} dB`] })),
  });
}

function hitBandSnr(g, mx) {
  return nearestX(g, state.bandHistory, (d) => d.it, mx);
}

// ---- Target / render analysis caches ----
// Target-side work (luma, pyramid, spectrum) is cached per target + mask;
// render-side spectra are cached per render. Both are dropped on a new target
// or mask.

function targetAnalysis() {
  const t = state.targetImage;
  if (!t) return null;
  if (state.targetAnalysis) return state.targetAnalysis;
  const { data, width, height } = t;
  const mask = maskWeights(state.maskData?.rgba, width, height);
  let maskArea = width * height;
  if (mask) { maskArea = 0; for (const m of mask) maskArea += m; }
  const y = luma(data, width, height);
  const masks = maskLevels(mask, width, height, BAND_LEVELS);
  state.targetAnalysis = {
    width, height, mask, maskArea, luma: y, masks,
    bandEnergy: bandEnergies(laplacianBands(y, width, height, BAND_LEVELS), masks),
    plan: null, F: null, bins: null, tile: null, // spectrum, filled on first use
  };
  return state.targetAnalysis;
}

function invalidateAnalysis() {
  state.targetAnalysis = null;
  state.spectra = null;
  state.importanceImage = null;
}

function renderMatchesTarget(r = state.lastRender) {
  const t = state.targetImage;
  return Boolean(r && t && r.width === t.width && r.height === t.height);
}

// Records band SNR for the latest render. Runs on every render, whether or
// not the band SNR card is in the layout, so its history has no gaps.
function recordBandSnr(it) {
  const ta = targetAnalysis();
  const r = state.lastRender;
  if (!ta || !renderMatchesTarget(r)) return;
  const res = luma(r.rgba, r.width, r.height);
  for (let p = 0; p < res.length; p += 1) res[p] -= ta.luma[p];
  const e = bandEnergies(laplacianBands(res, r.width, r.height, BAND_LEVELS), ta.masks);
  const snr = e.map((v, k) => 10 * Math.log10(ta.bandEnergy[k] / Math.max(v, 1e-30)));
  const hist = state.bandHistory;
  if (hist.length && hist[hist.length - 1].it === it) hist[hist.length - 1].snr = snr;
  else hist.push({ it, snr });
}

// Spectra for the current target and (if it matches) the latest render:
// { ta, bins, render } where bins carry target / recon / residual power,
// spectral SNR and FRC, or only target power before the first render.
function currentSpectra() {
  const ta = targetAnalysis();
  if (!ta) return null;
  if (!ta.plan) {
    ta.plan = spectralPlan(ta.width, ta.height, ta.mask);
    ta.F = windowedSpectrum(ta.plan, ta.luma);
    ta.bins = radialStats(ta.plan, ta.F, ta.F, ta.F);
  }
  const r = state.lastRender;
  if (!renderMatchesTarget(r)) return { ta, bins: ta.bins, render: null };
  if (state.spectra?.render === r && state.spectra.ta === ta) return state.spectra;
  const Fr = windowedSpectrum(ta.plan, luma(r.rgba, r.width, r.height));
  const Fe = spectrumDiff(Fr, ta.F);
  const bins = radialStats(ta.plan, ta.F, Fr, Fe);
  state.spectra = { ta, render: r, Fr, Fe, bins, frcHalf: frcCrossing(bins, 0.5), tiles: null };
  return state.spectra;
}

// Frequency (cycles/px) at which the mean site spacing √(area / N) is two
// samples: roughly the finest detail the current site count can place.
function siteFrequency(ta) {
  const n = state.lastActiveSites;
  return n > 0 && ta.maskArea > 0 ? 0.5 * Math.sqrt(n / ta.maskArea) : null;
}

const fmtFreq = (f) => `${fmtSig(f)} c/px · ${fmtSig(1 / f)} px`;

// ---- Radial power spectrum (log-log) ----

function drawSpectrumPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const s = currentSpectra();
  const signals = s?.render ? SIGNALS : SIGNALS.slice(0, 1);
  const top = drawLegend(p, SIGNALS.map((sig) => ({ label: sig.label, color: sig.color }))) + 10;
  const bins = s?.bins ?? [];
  let lo = Infinity, hi = -Infinity;
  for (const b of bins) for (const sig of signals) {
    const v = b[sig.key];
    if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  const xr = { min: bins.length ? bins[0].f : 0.004, max: 0.5, log: true };
  const g = view.geom = plotAxes(plotBox(p, top, LEGEND_PAD), xr, decadeRange(lo, hi, [1e-8, 1e-2]));
  drawAxes(p, g, { xTitle: "frequency (cycles/px)", yTitle: "power", xFmt: fmtSig, yFmt: fmtLogTick,
                   xSpacing: 44, ySpacing: 22 });
  if (!s) {
    drawEmpty(p, g, "Load a target to see its spectrum");
    return;
  }
  const fSite = siteFrequency(s.ta);
  if (fSite) drawRefLine(p, g, { x: fSite, label: "site spacing" });
  // Target last, so the reference stays visible where recon overlaps it.
  for (const sig of [...signals].reverse()) drawLine(p, g, bins.map((b) => [b.f, b[sig.key]]), sig.color);
  const hover = view.hoverKey != null ? bins[view.hoverKey] : null;
  if (!hover) return;
  drawCrosshair(p, g, hover.f);
  for (const sig of signals) drawMarker(p, g, hover.f, hover[sig.key], sig.color);
  drawTooltip(p, g, g.x(hover.f), null, {
    header: [fmtFreq(hover.f), "power"],
    rows: signals.map((sig) => ({ color: sig.color, cells: [sig.label, fmtExp(hover[sig.key])] })),
  });
}

// Shared by the spectrum and agreement plots: nearest frequency bin.
function hitSpectrum(g, mx) {
  const bins = currentSpectra()?.bins ?? [];
  return nearestX(g, bins, (b) => b.f, mx);
}

// ---- Spectral SNR & FRC (two panels, one frequency axis) ----

function drawAgreementPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const s = currentSpectra();
  const bins = s?.render ? s.bins : [];
  const box = plotBox(p);
  const mid = (box.top + box.bottom) / 2;
  const xr = { min: s?.bins.length ? s.bins[0].f : 0.004, max: 0.5, log: true };
  let lo = Infinity, hi = -Infinity, frcLo = 0;
  for (const b of bins) {
    if (Number.isFinite(b.ssnr)) { lo = Math.min(lo, b.ssnr); hi = Math.max(hi, b.ssnr); }
    frcLo = Math.min(frcLo, b.frc);
  }
  const gS = plotAxes({ ...box, bottom: mid - 10 }, xr, linRange(lo, hi, [0, 30]));
  const gF = plotAxes({ ...box, top: mid + 14 }, xr, { min: Math.floor(frcLo * 10) / 10, max: 1 });
  view.geom = gF;
  drawAxes(p, gS, { yTitle: "SSNR (dB)", ySpacing: 30, xLabels: false });
  drawAxes(p, gF, { xTitle: "frequency (cycles/px)", yTitle: "FRC", xFmt: fmtSig, xSpacing: 44, ySpacing: 30 });
  if (!bins.length) {
    drawEmpty(p, { ...box }, s ? "Train to compare with the target" : "Load a target and train");
    return;
  }
  const fSite = siteFrequency(s.ta);
  if (fSite) {
    drawRefLine(p, gS, { x: fSite, label: "site spacing" });
    drawRefLine(p, gF, { x: fSite });
  }
  drawRefLine(p, gS, { y: 0, label: "0 dB" });
  drawRefLine(p, gF, { y: 0.5, label: s.frcHalf ? `FRC ½ at ${fmtFreq(s.frcHalf)}` : "0.5" });
  drawLine(p, gS, bins.map((b) => [b.f, b.ssnr]), PSNR_LINE_COLOR);
  drawLine(p, gF, bins.map((b) => [b.f, b.frc]), PSNR_LINE_COLOR);
  const hover = view.hoverKey != null ? bins[view.hoverKey] : null;
  if (!hover) return;
  drawCrosshair(p, gS, hover.f);
  drawCrosshair(p, gF, hover.f);
  drawMarker(p, gS, hover.f, hover.ssnr, PSNR_LINE_COLOR);
  drawMarker(p, gF, hover.f, hover.frc, PSNR_LINE_COLOR);
  drawTooltip(p, { ...box }, gF.x(hover.f), null, {
    header: [fmtFreq(hover.f)],
    rows: [
      { cells: ["spectral SNR", `${hover.ssnr.toFixed(1)} dB`] },
      { cells: ["FRC", hover.frc.toFixed(3)] },
    ],
  });
}

// ---- 2D power spectra (target / recon / residual tiles) ----
// Sequential single-hue ramp: low power recedes into the card, high is dark.

const SPECTRUM_TILE = 256;
const SPECTRUM_LUT = rampLut([
  [0.00, [241, 241, 236]],
  [0.20, [205, 226, 251]],
  [0.40, [134, 182, 239]],
  [0.60, [42, 120, 214]],
  [0.80, [28, 92, 171]],
  [1.00, [13, 54, 107]],
]);

function spectrumTileCanvas(logs, size, lo, hi) {
  const out = new Uint8ClampedArray(size * size * 4);
  const scale = 255 / Math.max(1e-6, hi - lo);
  for (let p = 0, i = 0; p < logs.length; p += 1, i += 4) {
    const idx = Math.max(0, Math.min(255, Math.round((logs[p] - lo) * scale))) * 3;
    out[i] = SPECTRUM_LUT[idx];
    out[i + 1] = SPECTRUM_LUT[idx + 1];
    out[i + 2] = SPECTRUM_LUT[idx + 2];
    out[i + 3] = 255;
  }
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  c.getContext("2d").putImageData(new ImageData(out, size, size), 0, 0);
  return c;
}

// The shared scale runs from the target's 1st to 99.9th percentile log power.
function spectrumTiles(s) {
  const ta = s.ta;
  const size = Math.min(SPECTRUM_TILE, ta.plan.nx, ta.plan.ny);
  if (!ta.tile) {
    const logs = logPowerImage(ta.plan, ta.F, size);
    const sorted = Float32Array.from(logs).sort();
    const lo = sorted[Math.floor(0.01 * (sorted.length - 1))];
    const hi = sorted[Math.floor(0.999 * (sorted.length - 1))];
    ta.tile = { size, lo, hi, canvas: spectrumTileCanvas(logs, size, lo, hi) };
  }
  const tiles = { target: ta.tile.canvas, lo: ta.tile.lo, hi: ta.tile.hi };
  if (s.render) {
    if (!s.tiles) {
      s.tiles = {
        recon: spectrumTileCanvas(logPowerImage(ta.plan, s.Fr, size), size, ta.tile.lo, ta.tile.hi),
        residual: spectrumTileCanvas(logPowerImage(ta.plan, s.Fe, size), size, ta.tile.lo, ta.tile.hi),
      };
    }
    Object.assign(tiles, s.tiles);
  }
  return tiles;
}

function drawSpectrum2d(view) {
  const p = beginPlot(view);
  if (!p) return;
  const { ctx, col, w, h } = p;
  const s = currentSpectra();
  // The target tile carries the frequency axes: tick labels below and to its left.
  const axisLeft = 44, axisBottom = 16;
  const box = { left: 12 + axisLeft, right: w - 12, top: PLOT_PAD.top, bottom: h - 40 - axisBottom };
  if (!s) {
    drawEmpty(p, box, "Load a target to see its spectrum");
    return;
  }
  const tiles = spectrumTiles(s);
  const gap = 12;
  const side = Math.max(1, Math.min((box.right - box.left - 2 * gap) / 3, box.bottom - box.top));
  const x0 = axisLeft + (w - axisLeft - (3 * side + 2 * gap)) / 2;
  const fSite = siteFrequency(s.ta);
  ctx.font = PLOT_FONT;
  SIGNALS.forEach((sig, i) => {
    const x = Math.round(x0 + i * (side + gap));
    const y = Math.round(box.top);
    const tile = tiles[sig.key];
    if (tile) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tile, x, y, side, side);
    } else {
      ctx.fillStyle = col.soft;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("after first render", x + side / 2, y + side / 2);
    }
    ctx.strokeStyle = col.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, side - 1, side - 1);
    // Tiles span ±0.5 cycles/px, so frequency f sits f·side from the centre.
    if (tile && fSite && fSite < 0.5) {
      ctx.save();
      ctx.strokeStyle = PSNR_LINE_COLOR;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(x + side / 2, y + side / 2, fSite * side, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    if (i === 0) drawSpectrumTileAxes(p, x, y, side);
    ctx.fillStyle = col.dim;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(sig.label, x + side / 2, y + side + 6 + axisBottom);
  });
  ctx.fillStyle = col.soft;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(`log₁₀ power, shared scale ${tiles.lo.toFixed(1)} … ${tiles.hi.toFixed(1)}` +
               (fSite ? " · dashed ring: site spacing" : ""), w / 2, h - 8);
}

// Ticks in cycles/px along one tile: fx left→right, fy top→bottom (image rows),
// matching logPowerImage's layout with DC at the centre.
const SPECTRUM_TICKS = [-0.5, -0.25, 0, 0.25, 0.5];

function drawSpectrumTileAxes(p, x, y, side) {
  const { ctx, col } = p;
  ctx.save();
  ctx.font = PLOT_FONT;
  ctx.strokeStyle = col.soft;
  ctx.fillStyle = col.soft;
  ctx.lineWidth = 1;
  for (const f of SPECTRUM_TICKS) {
    const t = Math.round((f + 0.5) * side);
    const tx = Math.min(x + side - 0.5, x + t + 0.5);
    const ty = Math.min(y + side - 0.5, y + t + 0.5);
    ctx.beginPath();
    ctx.moveTo(tx, y + side); ctx.lineTo(tx, y + side + 4);
    ctx.moveTo(x, ty); ctx.lineTo(x - 4, ty);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(fmtSig(f), x + t, y + side + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(fmtSig(f), x - 6, y + t);
  }
  ctx.fillStyle = col.dim;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText("fx", x - 14, y + side + 5);
  ctx.translate(x - 36, y + side / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("fy (cycles/px)", 0, 0);
  ctx.restore();
}

// ---- Image views ----

function drawTarget() {
  const canvas = VIEWS.target.canvas;
  if (!canvas || !state.targetImage) return;
  const { data, width, height } = state.targetImage;
  drawToCanvas(canvas, data, width, height);
}

function drawPreview() {
  if (!VIEWS.preview.canvas || !state.lastRender) return;
  const { rgba, width, height } = state.lastRender;
  drawToCanvas(VIEWS.preview.canvas, rgba, width, height);
  drawCentroidOverlay();
}

// Site IDs and tau need an extra GPU pass each (cached per render).
async function drawGpuView(id, mode) {
  if (!VIEWS[id].canvas || !state.lastRender) return;
  const img = await ensureRender(mode);
  // The layout may have changed while the render was in flight.
  if (img && VIEWS[id].canvas) drawToCanvas(VIEWS[id].canvas, img.rgba, img.width, img.height);
}

function putImage(canvas, out, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  fitCanvasToCard(canvas, width, height);
  canvas.getContext("2d").putImageData(new ImageData(out, width, height), 0, 0);
}

// Mean |recon - target| over RGB, mapped through a magma-like ramp on a fixed
// [0, RESIDUAL_MAX] scale so frames stay comparable as training progresses.
const RESIDUAL_MAX = 0.25;
const RESIDUAL_LUT = rampLut([
  [0.00, [0, 0, 4]],
  [0.25, [81, 18, 124]],
  [0.50, [183, 55, 121]],
  [0.75, [252, 137, 97]],
  [1.00, [252, 253, 191]],
]);

function drawResidual() {
  const canvas = VIEWS.residual.canvas;
  const render = state.lastRender;
  if (!canvas || !renderMatchesTarget(render)) return;
  const { width, height } = render;
  const a = render.rgba;
  const b = state.targetImage.data;
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
  putImage(canvas, out, width, height);
}

// Multi-scale importance map (see IMPORTANCE in analysis.js), cached per
// target + mask.
const IMPORTANCE_DISPLAY_PCT = 0.995; // map this percentile to the top of the ramp
const IMPORTANCE_LUT = rampLut([ // viridis-like, to keep it distinct from the residual card
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
]);

function drawImportance() {
  const canvas = VIEWS.importance.canvas;
  const target = state.targetImage;
  if (!canvas || !target) return;
  const { data, width, height } = target;
  if (!state.importanceImage) {
    const n = width * height;
    const w = importanceMap(data, width, height, maskWeights(state.maskData?.rgba, width, height));
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
    state.importanceImage = out;
  }
  putImage(canvas, state.importanceImage, width, height);
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
    // A render of the previous target says nothing about this one.
    state.lastRender = null;
    state.lastSites = null;
    state.lastHashed = null;
    state.lastHeatmap = null;
    ui.saveBtn.disabled = true;
    invalidateAnalysis();
    blankRenderViews(img.width, img.height);
    await refreshViews("target", "render");
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
    invalidateAnalysis();
    await refreshViews("target", "render");
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
    ui.initSitesClear.hidden = false;
    appendLog(`Init sites: ${state.initSites.count}` +
      (state.initSites.width ? ` (${state.initSites.width}×${state.initSites.height})` : ""));
  } catch (err) {
    appendLog(`Init sites load failed: ${err.message}`);
  }
  syncSiteFields();
}

function clearInitSites() {
  state.initSites = null;
  ui.initSitesInput.value = "";
  ui.initLabel.textContent = "gradient init";
  ui.initSitesClear.hidden = true;
  syncSiteFields();
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
  let args;
  try {
    args = readFormArgs();
  } catch (err) {
    appendLog(`Settings error: ${err.message}`);
    return;
  }

  ui.saveBtn.disabled = true;
  ui.savePopover.classList.add("hidden");
  ui.startBtn.disabled = true;
  ui.stopBtn.disabled = false;
  setKodakEnabled(false);
  state.running = true;
  state.abortCtrl = new AbortController();
  const signal = state.abortCtrl.signal;
  state.lastHashed = null;
  state.lastHeatmap = null;
  // The previous run's trainer only served site-ID / tau renders of its last
  // state; free its GPU buffers so repeated (or looped) runs don't pile up.
  state.trainer?.destroy();
  state.trainer = null;

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
  state.gradHistory = [];
  state.bandHistory = [];
  state.lrHistory = [];
  // Growth multiplies the count by growFactor every growEvery iterations, so
  // its end (where "decay after growth" starts decaying) is predictable unless
  // λ stops it early; the plan is corrected once the trainer reports the real end.
  const startCount = init ? init.count : args.growInitSites;
  const plannedGrowEnd = args.siteMode !== "grow" ? 0
    : Math.min(args.growEndIter, growthCounts(startCount, args.growFactor, args.growMaxSites).length * args.growEvery);
  state.lrPlan = { warmupIters: args.warmupIters, decayIters: args.decayIters, lrMult: args.lrMult,
                   kind: args.lrSchedule, growDoneIter: plannedGrowEnd };
  state.lastActiveSites = initSiteCount;
  state.bppRuns.push({
    sites: initSiteCount, bpp: computeBpp(initSiteCount, width, height), colorSpace: args.colorSpace, points: [],
  });
  for (const view of Object.values(VIEWS)) view.hoverKey = null;
  await refreshViews("run");

  const sitesDesc = args.siteMode === "grow"
    ? `grow ${startCount} → ${Math.max(args.growMaxSites, startCount)}, ×${args.growFactor} every ${args.growEvery} iters ` +
      `(λ stop ${args.growLambda})`
    : `${initSiteCount} fixed`;
  appendLog(`Target: ${width}×${height} | sites: ${sitesDesc} | iters: ${args.warmupIters} warmup + ${args.decayIters} decay | lr ×${args.lrMult} ${LR_SCHEDULES[args.lrSchedule]} | color: ${COLOR_SPACES[args.colorSpace].label} | refresh: ${args.logFreq}`);
  updateProgress(0, args.iters, "Preparing…");

  const trainer = new Trainer(state.device, state.shaders, args, {
    signal,
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
        state.lastActiveSites = info.activeSites;
        if (info.growDoneIter != null && state.lrPlan) state.lrPlan.growDoneIter = info.growDoneIter;
        if (Number.isFinite(info.psnr)) {
          state.psnrHistory.push({ it: info.it, psnr: info.psnr });
          const run = state.bppRuns[state.bppRuns.length - 1];
          run.sites = info.activeSites;
          run.bpp = computeBpp(info.activeSites, info.width, info.height);
          run.points.push({ it: info.it, psnr: info.psnr });
        }
        if (info.gradStats) state.gradHistory.push({ it: info.it, stats: info.gradStats });
        if (Number.isFinite(info.lrScale)) state.lrHistory.push({ it: info.it, lr: info.lrScale });
        recordBandSnr(info.it);
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `log iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await refreshViews("log", "render");
      } else if (info.type === "preview") {
        // Full image preview. Site overlays arrive through a separate lighter
        // event so they are not blocked by full-image readback.
        state.lastRender = { rgba: info.renderRgba, width: info.width, height: info.height };
        if (info.sitesData) {
          state.lastSites = info.sitesData;
          state.lastSiteCount = info.totalSites;
        }
        recordBandSnr(info.it);
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `image iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await refreshViews("render");
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

  let completed = false;
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
    recordBandSnr(args.iters);
    await refreshViews("render");
    appendLog(`Done. Best PSNR: ${final.bestPsnr.toFixed(2)} dB | time: ${final.trainSeconds.toFixed(2)} s | active: ${final.count}`);
    updateProgress(args.iters, args.iters,
      `done · best ${final.bestPsnr.toFixed(2)} dB · ${final.trainSeconds.toFixed(1)}s`);
    ui.saveBtn.disabled = false;
    completed = !signal.aborted;
  } catch (err) {
    appendLog(`Error: ${err.message}`);
    console.error(err);
  } finally {
    state.running = false;
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled = true;
    setKodakEnabled(true);
  }
  // Loop mode: a run that finished on its own starts over from iteration 0
  // (with the form's current settings). Stop or an error ends the loop.
  if (completed && ui.loopToggle.checked) {
    appendLog("Loop: starting again from iteration 0.");
    setTimeout(startTraining, 0);
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
  loadCustomLayout();
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
  ui.initSitesClear.addEventListener("click", clearInitSites);
  document.getElementById("p-siteMode").addEventListener("change", syncSiteFields);
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
  ui.editLayoutBtn.addEventListener("click", toggleEditing);
  ui.addRowBtn.addEventListener("click", () => editLayout((grid) => { grid.push(grid[0].map(() => ".")); }));
  ui.removeRowBtn.addEventListener("click", () => editLayout((grid) => {
    if (grid.length < 2 || !rowEmpty(grid, grid.length - 1)) return false;
    grid.pop();
  }));
  ui.addColBtn.addEventListener("click", () => editLayout((grid) => { for (const row of grid) row.push("."); }));
  ui.removeColBtn.addEventListener("click", () => editLayout((grid) => {
    if (grid[0].length < 2 || !colEmpty(grid, grid[0].length - 1)) return false;
    for (const row of grid) row.pop();
  }));
  // Card sizes follow the viewer area (window resize, advanced panel toggle).
  new ResizeObserver(() => updateCanvasLayout()).observe(ui.canvases);
}

main();
