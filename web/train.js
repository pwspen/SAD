// Entry point for the SAD trainer web page.
// Hugging-Face-ish UI: target image in the top bar, advanced options
// (mask, init-sites, candidates/lr…) in the collapsible left panel, and a
// side-by-side target vs reconstruction view, both aspect-fit.

import { loadShaders, tileSlotsForLimit } from "../trainer/shader_loader.js?v=20261019";
import { requestTrainingDevice } from "../trainer/device.js?v=20261019";
import { Trainer, buildArgs, GRAD_GROUPS, GRAD_CHANNELS, lrSchedule, LR_SCHEDULES, LOSSES, COUNT_SCHEDULES, LAYER_MODES, countPlan, countVaries, countHorizon } from "../trainer/trainer.js?v=klt-20261020";
import { loadImageToFloat32, loadMaskRgba32Float, whiteMaskRgba32Float } from "../trainer/textures.js?v=20261019";
import { loadSitesFromFile, serializeSitesTxt, triggerDownload, floatImageToRgba8, computeBpp, computePsnr, BITS_PER_SITE } from "../trainer/io.js?v=20261019";
import { SITE_FLOATS } from "../trainer/params.js?v=20261019";
import { BatchTrainer, planBatches, tileSites, tileActive, tileSize, atlasCapacity } from "../trainer/batch.js?v=klt-20261020";
import { COLOR_SPACES } from "../trainer/color_spaces.js?v=klt-20261020";
import { encodeSites, codecSteps, CODEC_PLANES } from "../trainer/codec.js?v=shared-20261020";
import { parquetMetadata, parquetReadObjects } from "./vendor/hyparquet/src/index.js";
import {
  luma, maskWeights, laplacianBands, maskLevels, bandEnergies, IMPORTANCE, importanceMap,
  spectralPlan, windowedSpectrum, spectrumDiff, radialStats, frcCrossing, logPowerImage,
  mutualInfoMatrix, MI_BINS, neighborFeatures, NEIGHBOR_K, kMeans,
} from "./analysis.js?v=20261019";

const CONFIG_URL = "../training_config.json";
// Saved settings (presets/<key>.json: { _note, args }, keys as in
// node/sweep.mjs --set). "" = training_config.json and buildArgs defaults.
const PRESET_DIR = "../presets/";
const PRESETS = { "": "defaults", kodak: "Kodak", mnist: "MNIST", cifar10: "CIFAR-10" };
const DEFAULTS_NOTE = "training_config.json and the trainer's built-in defaults.";
// "Save as defaults": the form's args (and the config patch) as a preset that
// replaces "defaults", in this browser only.
const DEFAULTS_STORAGE_KEY = "sad.train.defaults";
const SAVED_DEFAULTS_NOTE = "Your saved defaults (Save as defaults, this browser only).";
const DEMO_MAX_IMAGE_SIDE = 2048;
const VALUE_BINS = 24; // bins per panel of the site parameter values view
const MI_ROUND_ANISO = 0.05; // |log_aniso| below this, a site has no orientation (site MI view)
const SNAP_MAP_K = 4; // snap-analysis fit drawn on the image (one of SNAP_KS)
// Entropy coding: PSNR budgets (dB, predicted) to spend on quantization, and
// the one a run's coded point on the PSNR vs bpp plot uses.
const CODEC_LOSSES = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2];
const CODEC_REPORT_LOSS = 0.1;
const CODED_COLOR = "#1baf7a";
// Datasets live outside this repo, as Hugging Face parquet files (see
// DATASETS): ../comp/data/<key>/<file>, which resolves under the kodak-bench
// server (it serves the parent dir). Override a folder with ?<key>=<dir>.
const DATA_DIR = "../../comp/data/";
const MNIST_DIGITS = Array.from({ length: 10 }, (_, d) => String(d));

// Fields that live in the advanced panel. The target image is in the top bar;
// init-sites / mask are pickers inside the panel (not listed here).
const UI_FIELDS = [
  ["iters", "p-iters"],
  ["sites", "p-sites"],
  ["startSites", "p-startSites"],
  ["endSites", "p-endSites"],
  ["countSchedule", "p-countSchedule"],
  ["countEnd", "p-countEnd"],
  ["sawTeeth", "p-sawTeeth"],
  ["sawAmp", "p-sawAmp"],
  ["sawRatio", "p-sawRatio"],
  ["logFreq", "p-logFreq"], // also sets viewerFreq (one "refresh every" field)
  ["candFreq", "p-candFreq"],
  ["candUpdatePasses", "p-candUpdatePasses"],
  ["candDownscale", "p-candDownscale"],
  ["candRadiusScale", "p-candRadiusScale"],
  ["candRadiusProbes", "p-candRadiusProbes"],
  ["candInjectCount", "p-candInjectCount"],
  ["lrMult", "p-lrMult"],
  ["lrSchedule", "p-lrSchedule"],
  ["lrPosBase", "p-lrPosBase"],
  ["lrTauBase", "p-lrTauBase"],
  ["lrRadiusBase", "p-lrRadiusBase"],
  ["lrColorBase", "p-lrColorBase"],
  ["lrDirBase", "p-lrDirBase"],
  ["lrAnisoBase", "p-lrAnisoBase"],
  ["colorSpace", "p-colorSpace"],
  ["loss", "p-loss"],
  ["errWeight", "p-errWeight"],
  ["errBeta", "p-errBeta"],
  ["errCap", "p-errCap"],
  ["errGrowth", "p-errGrowth"],
  ["growLambda", "p-growLambda"],
  ["growEvery", "p-growEvery"],
  ["layer", "p-layer"],
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
  lastOwner: null,       // { rgba, width, height } of renderView("owner"): r = largest-weight site id
  lastSites: null,
  lastSiteCount: 0,
  lastPsnr: null,        // persists between log ticks so preview updates don't blank the PSNR
  lastLrScale: null,     // schedule × LR mult at the last log tick
  lastActiveSites: 0,    // active sites at the last log tick (site-spacing reference lines)
  psnrHistory: [],       // [{ it, psnr, sites }] from log ticks, plotted in the PSNR card
  psnrIters: 0,
  bppRuns: [],           // [{ sites, bpp, colorSpace, points: [{ it, psnr, sites, bpp }], coded? }] per run on this target; last is current
  gradHistory: [],       // [{ it, stats }] per-param |grad| mean/max from log ticks
  gradSnapshot: null,    // { grads, siteCount, numPixels } raw per-site grads from the last log tick
  gradHeatParam: "mag",  // gradient heatmap: "mag" or a channel index into GRAD_CHANNELS
  lrHistory: [],         // [{ it, lr }] LR multiplier (schedule × LR mult) from log ticks
  growthGain: [],        // [{ step, addedAt, measuredAt, points: [{ site, predicted, realized }] }] per growth batch
  mergeEdges: null,      // { it, groups, scales, edgeWeights, mergedWeights } from the last "neighbor" merge step
  tuneTrace: null,       // { name, start, best, points: [{ v, median, round }] } of the setting auto-tune is on
  lrPlan: null,          // { iters, lrMult, kind, growDoneIter } of the current run
  lastGrowth: null,      // { mode, doneIter, stop, sites, it, iters } at the latest log tick (Schedule summary)
  runSiteMode: null,     // siteMode of the current/last run
  bandHistory: [],       // [{ it, snr: [dB per band] }] from every render
  targetAnalysis: null,  // viewer-side target luma / pyramid / spectrum (see targetAnalysis())
  spectra: null,         // spectra of the latest render (see currentSpectra())
  siteMI: null,          // { sites, mi } parameter mutual information for the latest readback
  snapAnalysis: null,    // { sites, status, result, error, promise } for the latest sites (see snapAnalysis())
  sensitivity: null,     // { sites, promise } of Trainer.siteSensitivity() for the latest sites (sensitivityOf())
  codec: null,           // { sites, status, result, error, promise } entropy coding of the latest sites (codecJob())
  sensitivityHist: null, // { sites, status, result, error } for the sensitivity histogram (sensitivityJob())
  siteValues: null,      // { sites, panels } parameter value histograms for the latest readback
  importanceImage: null, // RGBA8 importance map for the current target + mask
  layoutName: null,      // active key in LAYOUTS
  editing: false,        // layout edit mode (drag to move, add / remove, rows & cols)
  dragId: null,          // view being dragged in edit mode
  baseConfig: null,      // training_config.json as loaded; state.config adds the preset's UPPER_CASE keys
  showCentroids: true,   // centroid dots over the preview (its card's checkbox)
  runMode: "batch",      // "batch" (the default on load) or "single"; the Train button follows it
  batch: null,           // the last batch of runs (runBatch): { dataset, args, runs, running, selected, … }
};

const ui = {
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  imageLabel: document.getElementById("imageLabel"),
  datasetSelect: document.getElementById("datasetSelect"),
  presetSelect: document.getElementById("presetSelect"),
  presetNote: document.getElementById("presetNote"),
  saveDefaultsBtn: document.getElementById("saveDefaultsBtn"),
  resetDefaultsBtn: document.getElementById("resetDefaultsBtn"),
  itemSelect: document.getElementById("itemSelect"),
  itemPrev: document.getElementById("itemPrev"),
  itemNext: document.getElementById("itemNext"),
  maskInput: document.getElementById("maskInput"),
  maskLabel: document.getElementById("maskLabel"),
  initSitesInput: document.getElementById("initSitesInput"),
  initLabel: document.getElementById("initLabel"),
  initSitesClear: document.getElementById("initSitesClear"),
  startBtn: document.getElementById("startBtn"),
  singleModeBtn: document.getElementById("singleModeBtn"),
  batchModeBtn: document.getElementById("batchModeBtn"),
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
  batchBtn: document.getElementById("batchBtn"),
  batchPopover: document.getElementById("batchPopover"),
  batchFrom: document.getElementById("batchFrom"),
  batchCount: document.getElementById("batchCount"),
  batchRandom: document.getElementById("batchRandom"),
  batchFill: document.getElementById("batchFill"),
  batchJobs: document.getElementById("batchJobs"),
  batchLogFreq: document.getElementById("batchLogFreq"),
  batchSnaps: document.getElementById("batchSnaps"),
  batchNote: document.getElementById("batchNote"),
  batchRunBtn: document.getElementById("batchRunBtn"),
  batchTiled: document.getElementById("batchTiled"),
  batchBar: document.getElementById("batchBar"),
  batchRunSelect: document.getElementById("batchRunSelect"),
  batchPrev: document.getElementById("batchPrev"),
  batchNext: document.getElementById("batchNext"),
  batchSnap: document.getElementById("batchSnap"),
  batchSnapLabel: document.getElementById("batchSnapLabel"),
  tuneProgress: document.getElementById("tuneProgress"),
  tuneProgressFill: document.getElementById("tuneProgressFill"),
  batchRerun: document.getElementById("batchRerun"),
  batchExport: document.getElementById("batchExport"),
  batchSummary: document.getElementById("batchSummary"),
};

function setStatus(msg) { ui.status.textContent = msg; }
// One text node per line, oldest dropped past LOG_MAX_LINES: `textContent +=`
// re-copied and re-laid-out the whole log on every tick, so a long session
// (or loop mode) slowed training from ~2700 to ~300 it/s on MNIST.
const LOG_MAX_LINES = 1000;
function appendLog(line) {
  ui.log.append(line + "\n");
  while (ui.log.childNodes.length > LOG_MAX_LINES) ui.log.firstChild.remove();
  ui.log.scrollTop = ui.log.scrollHeight;
}

function applyConfigToForm(config, overrides = {}) {
  const spaceSelect = document.getElementById("p-colorSpace");
  spaceSelect.replaceChildren(...Object.entries(COLOR_SPACES).map(([key, space]) => new Option(space.label, key)));
  document.getElementById("p-lrSchedule")
    .replaceChildren(...Object.entries(LR_SCHEDULES).map(([key, label]) => new Option(label, key)));
  document.getElementById("p-loss")
    .replaceChildren(...Object.entries(LOSSES).map(([key, loss]) => new Option(loss.label, key)));
  document.getElementById("p-countSchedule")
    .replaceChildren(...Object.entries(COUNT_SCHEDULES).map(([key, label]) => new Option(label, key)));
  document.getElementById("p-layer")
    .replaceChildren(...Object.entries(LAYER_MODES).map(([key, label]) => new Option(label, key)));
  const defaults = buildArgs(config, overrides);
  for (const [key, elId, opts] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    const v = opts?.blankDefault ? undefined : defaults[key];
    if (el.type === "checkbox") el.checked = Boolean(v);
    else if (v === undefined) el.value = "";
    else el.value = v;
  }
  syncSiteFields();
  syncBppSites();
}

// The advanced panel's fields as buildArgs overrides.
function readFormOverrides() {
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
  return overrides;
}

function readFormArgs() {
  return buildArgs(state.config, readFormOverrides());
}

// The Schedule section's summary: what the current fields will do (run
// length, LR schedule, growth steps and when growth ends) and, after a run,
// when and why growth actually ended, naming the field to change.
function updatePlanSummary() {
  const el = document.getElementById("planSummary");
  if (!el || !state.config) return;
  let args;
  try {
    args = readFormArgs();
  } catch (err) {
    el.innerHTML = `<span class="warn">${err.message}</span>`;
    return;
  }
  const lines = [];
  const n = args.iters;
  lines.push(`Run: <b>${n}</b> iterations.`);
  let growEnd = 0;
  const start = state.initSites ? state.initSites.count : args.startSites;
  if (!countVaries(args, start)) {
    lines.push(`Sites: fixed at <b>${start}</b>.`);
  } else {
    const plan = countPlan(args, start);
    growEnd = countHorizon(args);
    const seq = [start, ...plan.map((p) => p.count)];
    const seqText = seq.length > 8 ? `${seq.slice(0, 4).join(", ")}, …, ${seq.slice(-3).join(", ")}` : seq.join(", ");
    const peak = Math.max(...seq);
    const dir = args.endSites > start ? "grow" : args.endSites < start ? "prune" : "hold";
    lines.push(`Sites: ${seqText} — ${args.countSchedule}` +
               (args.countSchedule === "sawtooth" ? ` (${args.sawTeeth} teeth, +${Math.round(args.sawAmp * 100)}%, ` +
                 `${args.sawRatio}:1 grow:merge, peak ${peak})` : "") +
               `, ${dir} ${start} → ${args.endSites} by iter <b>${growEnd}</b>, a step every ${args.growEvery}` +
               (dir === "grow" ? " (λ (stop) can end growth early)." : "."));
    if (growEnd >= n) lines.push(`<span class="warn">No iterations left to refine after the count settles: lower reach end by.</span>`);
    if (args.countSchedule === "sawtooth" && !plan.some((p, i) => p.count < (i ? plan[i - 1].count : start))) {
      lines.push(`<span class="warn">The teeth are too short for a step every ${args.growEvery}: the count never goes ` +
                 `down. Use fewer teeth, a later reach end by, or a smaller step every.</span>`);
    }
  }
  if (args.lrSchedule === "fixed") {
    lines.push(`LR: full throughout.`);
  } else if (args.lrSchedule === "decay-after-growth" && growEnd > 0) {
    lines.push(`LR: full until the count settles (iter ${growEnd}), then down to 0 by ${n}.`);
  } else {
    lines.push(`LR: full at iter 0, down to 0 by ${n}.`);
  }
  const g = state.lastGrowth;
  if (g && g.mode === "grow") {
    lines.push(g.doneIter != null
      ? `Last run: growth ended at iter <b>${g.doneIter}</b> with ${g.sites} sites — ${g.stop}.`
      : `Last run: still growing at iter ${g.it + 1} of ${g.iters} with ${g.sites} sites (the run ended first).`);
  }
  el.innerHTML = lines.join("<br>");
}

// Show the count schedule's fields (growth / merge fields only if the count
// changes, sawtooth's only for sawtooth, sites instead of start / end sites
// for constant), and lock start sites and sites to a loaded
// sites file's count, restoring the typed value once the file is cleared.
function syncSiteFields() {
  const lock = state.initSites != null;
  for (const id of ["p-startSites", "p-sites"]) {
    const el = document.getElementById(id);
    if (lock && !el.disabled) el.dataset.typed = el.value;
    if (!lock && el.disabled) el.value = el.dataset.typed ?? el.value;
    el.disabled = lock;
    if (lock) el.value = state.initSites.count;
    el.title = lock ? "Set by the loaded sites file (Clear it to edit)" : "";
  }
  let varies = true;
  try { const args = readFormArgs(); varies = countVaries(args, state.initSites?.count ?? args.startSites); } catch {}
  ui.paramsPanel.dataset.siteMode = varies ? "grow" : "fixed";
  ui.paramsPanel.dataset.countSchedule = document.getElementById("p-countSchedule").value;
  updatePlanSummary();
}

// "from bpp": sites and end sites follow a target bpp on the loaded image
// (BITS_PER_SITE bits per site), read-only while it's on. Called on the
// checkbox, the bpp field and every new target.
function syncBppSites() {
  const on = document.getElementById("p-fromBpp").checked;
  const bppEl = document.getElementById("p-bpp");
  const note = document.getElementById("bppSitesNote");
  const img = state.targetImage;
  const pixels = img ? img.width * img.height : 0;
  const countEls = ["p-sites", "p-endSites"].map((id) => document.getElementById(id));
  bppEl.disabled = !on;
  for (const el of countEls) {
    el.readOnly = on;
    el.title = on ? "Set by from bpp (untick it to edit)" : el.title.startsWith("Set by from bpp") ? "" : el.title;
  }
  if (!on) { note.textContent = "set the count from a target bpp"; return; }
  if (bppEl.value === "" && pixels) {
    const sites = Number(document.getElementById(ui.paramsPanel.dataset.countSchedule === "constant" ? "p-sites" : "p-endSites").value);
    if (sites > 0) bppEl.value = fmtSig(computeBpp(sites, img.width, img.height));
  }
  const bpp = Number(bppEl.value);
  if (!pixels || bppEl.value === "" || !(bpp >= 0)) {
    note.textContent = pixels ? "enter a bpp" : "load an image to set the count";
    return;
  }
  const sites = Math.max(1, Math.round((bpp * pixels) / BITS_PER_SITE));
  for (const el of countEls) if (!el.disabled) el.value = sites;
  note.textContent = `${sites} sites on ${img.width}×${img.height} (${fmtSig(computeBpp(sites, img.width, img.height))} bpp)`;
  syncSiteFields();
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

// Fills the advanced panel from a preset: camelCase keys are trainer args,
// UPPER_CASE keys patch training_config.json (they have no field).
let presetToken = 0;
async function applyPreset(key) {
  if (!state.baseConfig) return;
  ui.presetSelect.value = key;
  const token = ++presetToken;
  let preset = savedDefaults() || { _note: DEFAULTS_NOTE, args: {} };
  if (key) {
    try {
      const res = await fetch(new URL(`${PRESET_DIR}${key}.json`, import.meta.url), { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      preset = await res.json();
    } catch (err) {
      setStatus(`Preset ${PRESETS[key]} failed: ${err.message}`);
      console.error(err);
      return;
    }
  }
  if (token !== presetToken) return;
  const configPatch = {}, overrides = {};
  for (const [k, v] of Object.entries(preset.args || {})) {
    if (/^[A-Z0-9_]+$/.test(k)) configPatch[k] = v;
    else overrides[k] = v;
  }
  state.config = { ...state.baseConfig, ...configPatch };
  applyConfigToForm(state.config, overrides);
  ui.presetNote.textContent = `Fills every field below; picking a dataset switches to its preset. ${preset._note || ""}`;
  updatePlanSummary();
}

function savedDefaults() {
  try {
    const args = JSON.parse(localStorage.getItem(DEFAULTS_STORAGE_KEY));
    return args ? { _note: SAVED_DEFAULTS_NOTE, args } : null;
  } catch { return null; }
}

// Saves the current fields, plus the config keys the preset patched (they
// have no field), as the "defaults" preset.
function saveDefaults() {
  const args = {};
  for (const [k, v] of Object.entries(state.config || {})) {
    if (/^[A-Z0-9_]+$/.test(k) && JSON.stringify(v) !== JSON.stringify(state.baseConfig[k])) args[k] = v;
  }
  const overrides = readFormOverrides();
  delete overrides.viewerFreq; // follows logFreq
  Object.assign(args, overrides);
  try {
    localStorage.setItem(DEFAULTS_STORAGE_KEY, JSON.stringify(args));
  } catch (err) {
    setStatus(`Save as defaults failed: ${err.message}`);
    return;
  }
  ui.presetSelect.value = "";
  ui.presetNote.textContent = `Fills every field below; picking a dataset switches to its preset. ${SAVED_DEFAULTS_NOTE}`;
  syncDefaultsButtons();
  setStatus("Saved as defaults");
}

function resetDefaults() {
  try { localStorage.removeItem(DEFAULTS_STORAGE_KEY); } catch {}
  syncDefaultsButtons();
  if (ui.presetSelect.value === "") applyPreset("");
  setStatus("Defaults reset");
}

function syncDefaultsButtons() {
  ui.resetDefaultsBtn.hidden = !savedDefaults();
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
//   batch  – a batch's progress        tune   – an auto-tune batch finishing
// Plot views also redraw on resize and hover; a plot with `hit` is hoverable.
const SOURCES = {
  input:   { badge: "Input",    desc: "Input: what you loaded." },
  model:   { badge: "Model",    desc: "Model: rendered from the model's current state by the trainer (GPU)." },
  train:   { badge: "Training", desc: "Training: reported by the trainer while it optimizes." },
  derived: { badge: "Viz only", desc: "Viz only: computed in the viewer (CPU) for display. The optimizer never sees it." },
};
const VIEW_EVENTS = ["target", "render", "log", "run", "batch", "tune"];

const VIEWS = {
  target:     { label: "Target", kind: "image", source: "input", on: ["target"], draw: drawTarget,
                note: `The target image, downscaled to at most ${DEMO_MAX_IMAGE_SIDE} px per side.` },
  preview:    { label: "Reconstruction + centroids", kind: "image", source: "model", overlay: true, blank: "#0a0a0f",
                on: ["render"], draw: drawPreview,
                note: "The trainer's latest render, with site centroids on top (the dots checkbox hides them)." },
  editor:     { label: "Site editor", kind: "image", source: "model", overlay: true, editor: true, blank: "#0a0a0f",
                on: ["render", "run"], draw: drawEditor,
                note: "The latest render with site centroids; once a run has stopped, drag a site to move it or " +
                      "select one and edit its parameters below. Every change re-renders on the GPU with the " +
                      "candidate lists updated for the edited site; the other views follow when you let go." },
  residual:   { label: "Residual |recon − target|", kind: "image", source: "derived", blank: "#000004",
                on: ["render"], draw: drawResidual,
                note: "Mean |recon − target| over RGB on a fixed 0–0.25 scale." },
  errEma:     { label: "Persistent error |EWMA(recon − target)|", kind: "image", source: "train", blank: "#000004",
                on: ["render"], draw: drawErrEma,
                note: "The trainer's per-pixel running average of the signed error (decay = persist decay), as mean " +
                      "|ē| over RGB on the residual card's 0–0.25 scale. Error that keeps its sign stays bright; " +
                      "error that flips averages to dark. With persist weight > 0 this is what weights the loss. " +
                      "Read back from the GPU on each render while the card is shown." },
  importance: { label: "Importance (Laplacian bands)", kind: "image", source: "derived", on: ["target"], draw: drawImportance,
                note: `Luma Laplacian band energy, mask · (ε + √Σ β_k L_k²)^γ with β = [${IMPORTANCE.bandWeights.join(", ")}]. ` +
                      "Not used by the optimizer." },
  siteIds:    { label: "Site IDs", kind: "image", source: "model", blank: "#0a0a0f", on: ["render"],
                draw: () => drawGpuView("siteIds", "hashed"),
                note: "Each pixel's winning site, hashed to a color (extra GPU render pass)." },
  tau:        { label: "Tau heatmap", kind: "image", source: "model", blank: "#0a0a0f", on: ["render"],
                draw: () => drawGpuView("tau", "heatmap"),
                note: "Per-site tau as a heatmap (extra GPU render pass)." },
  gradHeat:   { label: "Gradient heatmap", kind: "image", source: "train", blank: "#00204d",
                on: ["log", "render"], draw: drawGradHeat, controls: buildGradHeatControls,
                note: "Each pixel shows the loss gradient of the site with the largest weight there, from the last " +
                      "log tick, in the units Adam sees (summed grad / pixels). Magnitude is √Σ g² over all ten " +
                      "parameters (mixed units, so position tends to dominate); or pick one parameter for its |g|. " +
                      "Color params are the run's color space. Log scale through cividis from the 1st to 99.5th " +
                      "percentile over sites; black = no owner or masked out. Ownership is an extra GPU render pass." },
  psnr:       { label: "PSNR vs iteration", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawPsnrPlot, hit: hitPsnr,
                note: "Masked PSNR the trainer computes at each log tick, with the active site count and each site's LR multiplier (schedule × growth LR decay) on the right axes." },
  bpp:        { label: "PSNR vs bpp", kind: "plot", source: "derived", on: ["log", "run", "target"],
                draw: drawBppPlot, hit: hitBpp,
                note: `Grey and orange: the raw rate, active sites × ${BITS_PER_SITE} bits / pixels (an assumption), ` +
                      "at the trainer's PSNR. Green, once a run has stopped: the entropy-coded rate (the Entropy " +
                      "coding card's bitstream) at the PSNR of the decoded sites; the current run shows the codec's " +
                      `whole curve, earlier runs their point at the ${CODEC_REPORT_LOSS} dB budget.` },
  codec:      { label: "Entropy coding", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawCodec,
                note: "After a run: the sites coded to a real bitstream (trainer/codec.js), decoded, and " +
                      "re-rendered with candidate lists rebuilt from the decoded sites, as a decoder would. " +
                      "Stages: quantize (uniform steps per parameter, from the Gauss–Newton sensitivity so each " +
                      "parameter adds the same predicted error; aniso_dir as an angle mod π, its step divided by " +
                      "|2 sinh log_aniso| per site, so none for round sites (the θ row shows the undivided step); colors as lossless YCoCg-R), sort by position and code positions as raster gaps, " +
                      "predict each value from its 3 nearest earlier sites or a constant (whichever is shorter), " +
                      "then adaptive arithmetic coding. Left: one row per PSNR budget spent on quantization " +
                      "(predicted), with the measured loss. Right: where the bits go at the " +
                      `${CODEC_REPORT_LOSS} dB budget (nbr = neighbour prediction, const = constant).` },
  grad:       { label: "|gradient| vs iteration (log)", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawGradPlot, hit: hitGrad,
                note: "Per-parameter-group |gradient| over sites, reported by the trainer at each log tick." },
  lr:         { label: "LR multiplier vs iteration", kind: "plot", source: "train", on: ["log", "run"],
                draw: drawLrPlot, hit: hitLr,
                note: "LR multiplier (schedule × LR mult) the trainer applied to every base " +
                      "LR at each log tick. Dashed: the run's full planned schedule." },
  growGain:   { label: "Site gain: predicted vs real", kind: "plot", source: "train", on: ["growth", "run"],
                draw: drawGrowGainPlot, hit: hitGrowGain,
                note: "One dot per site added by growth. x: the error it was predicted to remove when placed; y: " +
                      "the error that would come back if it were removed one growth step later, others unchanged " +
                      "(8-bit levels², summed over RGB and pixels). Diagnostic only; growth never uses it. The " +
                      "realized side is biased high: by then the site and its neighbours have adapted to each other. " +
                      "Sites at the bottom edge are worth nothing or harm the fit. Dashed: realized = predicted." },
  mergeEdges: { label: "Merge edge weights", kind: "plot", source: "train", on: ["merge", "run"],
                draw: drawMergeEdgesPlot, hit: hitMergeEdges,
                note: "Edge weights of the last merge step (mergeCriterion \"neighbor\"). An edge joins two sites " +
                      "that are owner and runner-up in some candidate list. Its weight is the RMS over color, log τ, " +
                      "radius and shape (log anisotropy tensor) of the pair's difference, each divided by that " +
                      "group's median over the step's edges, so 1 (dashed) is a typical neighbour; position is left " +
                      "out. Bars: every edge before merging. Orange: the edges merged along, weighed when merged " +
                      "(chained merges reweigh the kept site's edges, so these can sit past the initial bars)." },
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
  siteMI:     { label: "Site parameter mutual information", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawSiteMI, hit: hitSiteMI,
                note: `Normalized mutual information I(X;Y) / min(H(X), H(Y)) between site parameters over active ` +
                      `sites: ${MI_BINS} equal-width bins per parameter (1st–99th percentile; θ over [0, π)), ` +
                      `entropies with the Miller–Madow correction. θ is the major axis, skipped for sites with ` +
                      `|log_aniso| < ${MI_ROUND_ANISO}. Colors are decoded RGB. Neighbour columns use each site's ` +
                      `${NEIGHBOR_K} nearest sites: log distance to the farthest of them, and the main direction φ ` +
                      `and eccentricity (λ1 − λ2) / (λ1 + λ2) of the structure tensor of the unit offsets to them. ` +
                      `Hover a cell for bits.` },
  siteValues: { label: "Site parameter values", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawSiteValues,
                note: `Per parameter, how many active sites (y) have each value (x): ${VALUE_BINS} equal-width ` +
                      "bins over the 1st–99th percentile, with values outside counted in the end bins (\"clipped\"). " +
                      `Colors are decoded RGB. θ is the major axis mod π, skipped for near-round sites ` +
                      `(|log_aniso| < ${MI_ROUND_ANISO}, counted as "round"). The histogram layer is left out.` },
  siteClusters: { label: "Site snap analysis", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawSnapAnalysis,
                note: "After a run: every site parameter except position (aniso_dir sign-canonicalized) is " +
                      "clustered with k-means, measuring distance by each site's Gauss–Newton diagonal (the " +
                      "trainer's extra GPU pass), so the k-means cost is the predicted rise in squared error from " +
                      "snapping every site to its centroid. Left: PSNR loss predicted that way, and the actual loss " +
                      "from re-rendering with snapped sites (same candidate lists). bits = log₂ k per site. Right: " +
                      "the same if every site shared one value of that parameter (its weighted mean), others " +
                      "unchanged; near 0 means the parameter is slack. Predictions are second order, so they " +
                      "overshoot for large snaps." },
  sensitivity: { label: "Site sensitivity (Gauss–Newton)", kind: "plot", source: "derived", on: ["render", "run"],
                draw: drawSensitivity,
                note: "After a run: per parameter group, how many sites (y) have each Gauss–Newton sensitivity " +
                      "(x, log bins): F = Σ over pixels of (∂render/∂θ)², summed over RGB and the group's " +
                      "parameters (colors decoded to RGB, 0..1), so nudging θ by δ raises the summed squared " +
                      "error by about F·δ². Units differ per group (pixels for position, log units for τ and " +
                      "anisotropy), so compare sites within a panel, not across panels. Sites with F = 0 (no " +
                      "pixels, or not reachable) are counted in the title." },
  batchPsnr:  { label: "Batch: PSNR vs iteration", kind: "plot", source: "train", on: ["batch"],
                draw: drawBatchViews, hit: hitBatchPsnr, click: clickBatchRun,
                note: "Every run of the last batch, from its PSNR log ticks (log every in the Batch menu). Grey: " +
                      "finished; blue: still running; orange: the selected run; green: the mean over runs at each " +
                      "tick, labeled at its start, middle and end. Hover a curve to find its run; once the batch " +
                      "has finished, click it to load the run into the other views." },
  batchHist:  { label: "Batch: final PSNR", kind: "plot", source: "train", on: ["batch"],
                draw: drawBatchViews, hit: hitBatchHist, click: clickBatchRun,
                note: "How many runs ended at each PSNR. Each run is one cell, stacked in its bin by PSNR, so " +
                      "hovering picks out a single run; click it to load it once the batch has finished. Final " +
                      "PSNR is measured on the last render, so it can sit below the run's best log tick." },
  batchDigits: { label: "Batch: PSNR by class", kind: "plot", source: "train", on: ["batch"],
                draw: drawBatchViews, hit: hitBatchDigits, click: clickBatchRun,
                note: "Final PSNR of each finished run, grouped by its label (the digit for MNIST, the class for CIFAR-10). Each " +
                      "circle is one run; the box spans the quartiles, the thick line the median and the whiskers " +
                      "1.5× the interquartile range. Drawn once the batch has finished; hover a run to find it " +
                      "in the other batch plots, click to load it." },
  tune:       { label: "Auto-tune: PSNR vs setting", kind: "plot", source: "train", on: ["tune"],
                draw: drawTunePlot, hit: hitTune,
                note: "The setting the last auto-tune climbed (a tune button or optimize all, one setting at a " +
                      "time): one dot per value tried, at the median final PSNR of its batch on the same images. " +
                      "Orange: the best so far; the dashed line marks where it started. x is log-scaled, since " +
                      "each round multiplies or divides by the tune ratio." },
  clusterMap: { label: `Snap clusters on the image (k = ${SNAP_MAP_K})`, kind: "image", source: "derived",
                blank: "#f1f1ec", on: ["render"], draw: drawClusterMap,
                note: "The latest render, washed out, with each site's centroid in its cluster's color from the " +
                      `k = ${SNAP_MAP_K} fit of the Site snap analysis.` },
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
    areas: `"target   preview importance"
            "residual errEma  gradHeat"`,
  },
  sites: {
    label: "Sites & tau",
    areas: `"preview  siteIds    tau"
            "residual importance psnr"`,
  },
  rate: {
    label: "Rate",
    areas: `"bpp  bpp     codec codec"
            "psnr preview codec codec"`,
  },
  plots: {
    label: "Plots",
    areas: `"psnr bpp      lr"
            "grad growGain mergeEdges"`,
  },
  siteStats: {
    label: "Site stats",
    areas: `"siteValues siteMI       sensitivity"
            "clusterMap siteClusters sensitivity"`,
  },
  editor: {
    label: "Site editor",
    areas: `"editor target   residual"
            "editor siteIds  tau"`,
  },
  batch: {
    label: "Batch",
    areas: `"batchHist   batchPsnr tune"
            "batchDigits batchDigits batchDigits"
            "preview     target    psnr"`,
  },
  spectral: {
    label: "Spectral",
    areas: `"spectrum  spectrum  spectrum2d spectrum2d"
            "agreement agreement bandSnr    bandSnr"`,
  },
};
const DEFAULT_LAYOUT = "batch";
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
    card.className = `canvas-card ${view.kind}${view.editor ? " editor-card" : ""}`;
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
    if (view.editor) {
      // The image fits a stage above the parameter fields; the overlay takes
      // the pointer for picking and dragging sites.
      const stage = document.createElement("div");
      stage.className = "editor-stage";
      const overlay = document.createElement("canvas");
      overlay.className = "overlay-canvas";
      stage.append(canvas, overlay);
      const fields = document.createElement("div");
      fields.className = "editor-fields";
      card.append(label, stage, fields);
      view.overlayCanvas = overlay;
      buildEditorFields(fields);
      attachEditorPointer(view, overlay);
    } else {
      card.append(label, canvas);
    }
    if (view.overlay && !view.editor) {
      const overlay = document.createElement("canvas");
      overlay.className = "overlay-canvas";
      overlay.hidden = !state.showCentroids;
      card.append(overlay);
      view.overlayCanvas = overlay;
      const toggle = document.createElement("label");
      toggle.className = "overlay-toggle";
      toggle.title = "Show site centroids";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = state.showCentroids;
      box.addEventListener("change", () => {
        state.showCentroids = box.checked;
        overlay.hidden = !box.checked;
      });
      toggle.append(box, "dots");
      card.append(toggle);
    }
    if (view.controls) view.controls(card);
    if (view.hit) attachPlotHover(view, card, canvas);
    if (state.editing) {
      attachEditControls(view, id, card, label, grid);
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

// Grows (d = +1) or shrinks (d = -1) view `id` by one cell along `axis`
// ("w" or "h"), at its right / bottom edge. Growing past the grid's edge adds
// an empty column / row; it needs the cells it takes to be empty. Returns
// false if not allowed.
function resizeView(grid, id, axis, d) {
  const s = viewRect(grid, id);
  if (axis === "w") {
    if (d < 0) {
      if (s.c1 === s.c0) return false;
      for (let r = s.r0; r <= s.r1; r += 1) grid[r][s.c1] = ".";
      return true;
    }
    const c = s.c1 + 1;
    if (c === grid[0].length) for (const row of grid) row.push(".");
    for (let r = s.r0; r <= s.r1; r += 1) if (grid[r][c] !== ".") return false;
    for (let r = s.r0; r <= s.r1; r += 1) grid[r][c] = id;
    return true;
  }
  if (d < 0) {
    if (s.r1 === s.r0) return false;
    for (let c = s.c0; c <= s.c1; c += 1) grid[s.r1][c] = ".";
    return true;
  }
  const r = s.r1 + 1;
  if (r === grid.length) grid.push(grid[0].map(() => "."));
  for (let c = s.c0; c <= s.c1; c += 1) if (grid[r][c] !== ".") return false;
  for (let c = s.c0; c <= s.c1; c += 1) grid[r][c] = id;
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

function attachEditControls(view, id, card, label, grid) {
  const rect = viewRect(grid, id);
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
  attachDropTarget(card, rect.r0, rect.c0);
  card.append(sizeControls(view, id, grid, rect));
  const remove = document.createElement("button");
  remove.className = "card-remove";
  remove.textContent = "×";
  remove.title = `Remove ${view.label}`;
  remove.setAttribute("aria-label", `Remove ${view.label}`);
  remove.addEventListener("click", () => editLayout((grid) => fillRect(grid, viewRect(grid, id), ".")));
  card.append(remove);
}

// Width and height of a card in grid cells, each with − / + buttons
// (resizeView); a button is disabled when its resize would be refused.
function sizeControls(view, id, grid, rect) {
  const bar = document.createElement("div");
  bar.className = "card-size";
  for (const [axis, name, n] of [["w", "width", rect.c1 - rect.c0 + 1], ["h", "height", rect.r1 - rect.r0 + 1]]) {
    const group = document.createElement("span");
    group.className = "card-size-group";
    const button = (d, text) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      const what = `${d > 0 ? "Grow" : "Shrink"} ${view.label} to ${name} ${n + d}`;
      b.title = what;
      b.setAttribute("aria-label", what);
      b.disabled = resizeView(grid.map((row) => [...row]), id, axis, d) === false;
      b.addEventListener("click", () => editLayout((g) => resizeView(g, id, axis, d)));
      return b;
    };
    const value = document.createElement("span");
    value.className = "card-size-value";
    value.textContent = `${axis.toUpperCase()} ${n}`;
    group.append(button(-1, "−"), value, button(1, "+"));
    bar.append(group);
  }
  return bar;
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
  clearOverlay();
  clearOverlay(VIEWS.editor);
}

function updateCanvasLayout(width = state.targetImage?.width, height = state.targetImage?.height) {
  for (const view of Object.values(VIEWS)) {
    if (!view.canvas) continue;
    if (view.kind === "image") fitCanvasToCard(view.canvas, width || view.canvas.width, height || view.canvas.height);
    else view.draw(view);
  }
  drawCentroidOverlay();
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
  // At most one hit test and redraw per frame, for the latest position.
  let pending = null;
  canvas.addEventListener("mousemove", (ev) => {
    const first = !pending;
    pending = ev;
    if (!first) return;
    requestAnimationFrame(() => {
      const e = pending;
      pending = null;
      if (!e || !view.geom) return;
      const rect = canvas.getBoundingClientRect();
      const key = view.hit(view.geom, e.clientX - rect.left, e.clientY - rect.top);
      if (key !== view.hoverKey) {
        view.hoverKey = key;
        view.draw(view);
      }
    });
  });
  canvas.addEventListener("mouseleave", () => {
    pending = null;
    if (view.hoverKey == null) return;
    view.hoverKey = null;
    view.draw(view);
  });
  if (view.click) {
    canvas.addEventListener("click", (ev) => {
      if (!view.geom) return;
      const rect = canvas.getBoundingClientRect();
      const key = view.hit(view.geom, ev.clientX - rect.left, ev.clientY - rect.top);
      if (key != null) view.click(key);
    });
  }
}

// ---- PSNR vs iteration ----

function iterRange() {
  const last = Math.max(state.psnrHistory.at(-1)?.it ?? 0, state.gradHistory.at(-1)?.it ?? 0,
                        state.bandHistory.at(-1)?.it ?? 0, state.lrHistory.at(-1)?.it ?? 0);
  return { min: 0, max: Math.max(1, state.psnrIters, last) };
}

// Right-hand y axis for another series on the same box (labels and title
// only; the grid follows the left axis), `offset` px right of the plot.
// Linear axes label integer ticks only unless `integers` is false.
function drawRightAxis(p, g, { title, offset = 0, fmt = String, spacing = 48, integers = true }) {
  const { ctx, col } = p;
  ctx.font = PLOT_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = col.soft;
  for (const v of axisTicks(g.yr, g.bottom - g.top, spacing)) {
    if (g.yr.log || !integers || Number.isInteger(v)) ctx.fillText(fmt(v), g.right + offset + 6, Math.round(g.y(v)) + 0.5);
  }
  ctx.save();
  ctx.fillStyle = col.dim;
  ctx.textAlign = "center";
  ctx.translate(g.right + offset + 38, (g.top + g.bottom) / 2);
  ctx.rotate(Math.PI / 2);
  ctx.fillText(title, 0, 0);
  ctx.restore();
}

const SITES_LINE_COLOR = SERIES_COLORS[0];
const LR_LINE_COLOR = SERIES_COLORS[2];
const PSNR_PAD = { ...LEGEND_PAD, right: 100 }; // room for the sites and LR axes

function drawPsnrPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const data = state.psnrHistory;
  let lo = Infinity, hi = -Infinity, maxSites = 1;
  for (const d of data) {
    lo = Math.min(lo, d.psnr); hi = Math.max(hi, d.psnr); maxSites = Math.max(maxSites, d.sites);
  }
  const top = drawLegend(p, [
    { label: "PSNR", color: PSNR_LINE_COLOR },
    { label: "sites", color: SITES_LINE_COLOR },
    { label: "LR mult", color: LR_LINE_COLOR },
  ]) + 10;
  const lrData = state.lrHistory;
  let maxLr = 0;
  for (const d of lrData) maxLr = Math.max(maxLr, d.lr);
  const box = plotBox(p, top, PSNR_PAD);
  const g = view.geom = plotAxes(box, iterRange(), linRange(lo, hi, [0, 40]));
  const gs = plotAxes(box, g.xr, { min: 0, max: Math.ceil(maxSites * 1.08) });
  const gl = plotAxes(box, g.xr, { min: 0, max: maxLr > 0 ? maxLr * 1.08 : 1 });
  drawAxes(p, g, { xTitle: "iteration", yTitle: "PSNR (dB)", xFmt: fmtIter });
  drawRightAxis(p, gs, { title: "sites" });
  drawRightAxis(p, gl, { title: "LR multiplier", offset: 50, integers: false, fmt: (v) => String(Number(v.toPrecision(3))) });
  if (!data.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see PSNR");
    return;
  }
  // Sites change in steps at growth; draw them as a step line.
  drawLine(p, gs, data.flatMap((d, i) => (i && d.sites !== data[i - 1].sites
    ? [[d.it, data[i - 1].sites], [d.it, d.sites]] : [[d.it, d.sites]])), SITES_LINE_COLOR, { width: 1.5 });
  if (lrData.length) drawLine(p, gl, lrData.map((d) => [d.it, d.lr]), LR_LINE_COLOR, { width: 1.5 });
  drawLine(p, g, data.map((d) => [d.it, d.psnr]), PSNR_LINE_COLOR);
  const hover = view.hoverKey != null ? data[view.hoverKey] : null;
  if (hover) drawCrosshair(p, g, hover.it);
  const mark = hover || data[data.length - 1];
  // LR samples come from the same log ticks; match by iteration.
  const markLr = lrData.find((d) => d.it === mark.it);
  drawMarker(p, gs, mark.it, mark.sites, SITES_LINE_COLOR);
  if (markLr) drawMarker(p, gl, markLr.it, markLr.lr, LR_LINE_COLOR);
  drawMarker(p, g, mark.it, mark.psnr, PSNR_LINE_COLOR);
  if (hover) {
    const lrText = markLr ? ` · LR ×${markLr.lr.toPrecision(3)}` : "";
    drawTooltip(p, g, g.x(hover.it), g.y(hover.psnr), {
      rows: [{ cells: [`iter ${hover.it} · ${hover.psnr.toFixed(2)} dB · ${hover.sites} sites${lrText}`] }],
    });
  }
}

function hitPsnr(g, mx) {
  return nearestX(g, state.psnrHistory, (d) => d.it, mx);
}

// ---- PSNR vs bpp (log x) ----
// Raw rate: active sites × BITS_PER_SITE / pixels. Runs on the current target
// accumulate: earlier runs show as their final point (joined in bpp order, an
// RD curve per site color space); the current run shows its log-tick trail,
// each tick at the bpp of the sites active then (grow mode moves it right).
// Entropy-coded points (codecJob) go next to them in CODED_COLOR: the current
// run's whole codec curve, earlier runs' CODEC_REPORT_LOSS points.

const BPP_PAST_COLOR = "#66666e";

// RGB keeps the neutral past-run color; other spaces take series hues.
function colorSpaceColor(key) {
  const i = Object.keys(COLOR_SPACES).indexOf(key);
  return i <= 0 ? BPP_PAST_COLOR : SERIES_COLORS[(i - 1) % SERIES_COLORS.length];
}

// Plotted points: { x: bpp, y: psnr, sites, run, point, current, coded? }
// (coded: { bitsPerSite, lossDb } of an entropy-coded point).
function bppPlotPoints() {
  const runs = state.bppRuns;
  const pts = [];
  const job = state.codec?.status === "done" && state.codec.sites === state.lastSites ? state.codec : null;
  runs.forEach((r, ri) => {
    const current = ri === runs.length - 1;
    if (current) {
      r.points.forEach((p, pi) => pts.push({ x: p.bpp, y: p.psnr, sites: p.sites, run: ri, point: pi, current }));
      if (job && r.coded) {
        job.result.rows.forEach((row, ci) => pts.push({
          x: row.bpp, y: row.psnr, sites: row.n, run: ri, point: `c${ci}`, current,
          coded: { bitsPerSite: row.bitsPerSite, lossDb: row.lossDb },
        }));
      }
    } else if (r.points.length) {
      const pi = r.points.length - 1;
      const p = r.points[pi];
      pts.push({ x: p.bpp, y: p.psnr, sites: p.sites, run: ri, point: pi, current });
    }
    if (r.coded && !(current && job)) {
      pts.push({ x: r.coded.bpp, y: r.coded.psnr, sites: r.coded.n, run: ri, point: "c", current,
                 coded: { bitsPerSite: r.coded.bitsPerSite, lossDb: CODEC_REPORT_LOSS } });
    }
  });
  return pts;
}

function drawBppPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  codecJob(); // starts coding the latest sites once training has stopped
  const pts = bppPlotPoints();
  let xMin = Infinity, xMax = -Infinity, lo = Infinity, hi = -Infinity;
  const cur0 = state.bppRuns[state.bppRuns.length - 1];
  if (cur0) { xMin = xMax = cur0.bpp; }
  for (const pt of pts) {
    xMin = Math.min(xMin, pt.x); xMax = Math.max(xMax, pt.x);
    lo = Math.min(lo, pt.y); hi = Math.max(hi, pt.y);
  }
  if (!(xMin > 0)) { xMin = 0.1; xMax = 10; }
  // At least a decade of x range, centred on the data.
  const span = Math.max(1, Math.log10(xMax / xMin) * 1.2);
  const mid = (Math.log10(xMin) + Math.log10(xMax)) / 2;
  const xr = { min: 10 ** (mid - span / 2), max: 10 ** (mid + span / 2), log: true };
  // Color-space legend only once runs in more than one space are on the plot.
  const spaces = [...new Set(state.bppRuns.map((r) => r.colorSpace))];
  const coded = pts.filter((pt) => pt.coded);
  const legend = spaces.length > 1
    ? spaces.map((key) => ({ label: `${COLOR_SPACES[key].label} (${BITS_PER_SITE} bits/site)`, color: colorSpaceColor(key) }))
    : coded.length ? [{ label: `${BITS_PER_SITE} bits/site`, color: BPP_PAST_COLOR }] : [];
  if (coded.length) legend.push({ label: "entropy coded", color: CODED_COLOR });
  const top = legend.length ? drawLegend(p, legend) + 10 : undefined;
  const g = view.geom = plotAxes(plotBox(p, top), xr, linRange(lo, hi, [0, 40]));
  drawAxes(p, g, { xTitle: coded.length ? "bpp" : `bpp (${BITS_PER_SITE} bits/site)`, yTitle: "PSNR (dB)",
                   xFmt: fmtSig, xSpacing: 40 });
  if (!pts.length) {
    drawEmpty(p, g, state.running ? "Waiting for first log tick…" : "Train to see PSNR vs bpp");
    return;
  }

  // Earlier runs: final points joined in bpp order, one curve per color space
  // (raw rate solid, coded dashed)
  for (const space of spaces) {
    const color = colorSpaceColor(space);
    for (const isCoded of [false, true]) {
      const past = pts.filter((pt) => !pt.current && !pt.coded === !isCoded && state.bppRuns[pt.run].colorSpace === space)
        .sort((a, b) => a.x - b.x);
      const c = isCoded ? CODED_COLOR : color;
      if (past.length > 1) drawLine(p, g, past.map((pt) => [pt.x, pt.y]), c, { width: 1.5, dash: isCoded ? [5, 4] : [] });
      for (const pt of past) drawMarker(p, g, pt.x, pt.y, c, 3.5);
    }
  }

  // Current run's codec curve, the report budget marked larger
  const curCoded = coded.filter((pt) => pt.current).sort((a, b) => a.x - b.x);
  if (curCoded.length > 1) drawLine(p, g, curCoded.map((pt) => [pt.x, pt.y]), CODED_COLOR);
  for (const pt of curCoded) drawMarker(p, g, pt.x, pt.y, CODED_COLOR, pt.coded.lossDb === CODEC_REPORT_LOSS ? 4.5 : 3);

  // Current run: trail of log ticks in iteration order, latest marked
  const cur = pts.filter((pt) => pt.current && !pt.coded);
  if (cur.length) {
    if (cur.length > 1) drawLine(p, g, cur.map((pt) => [pt.x, pt.y]), PSNR_LINE_COLOR);
    const last = cur[cur.length - 1];
    drawMarker(p, g, last.x, last.y, PSNR_LINE_COLOR);
  }

  const hover = view.hoverKey && pts.find((pt) => `${pt.run}:${pt.point}` === view.hoverKey);
  if (!hover) return;
  const run = state.bppRuns[hover.run];
  drawMarker(p, g, hover.x, hover.y,
             hover.coded ? CODED_COLOR : hover.current ? PSNR_LINE_COLOR : colorSpaceColor(run.colorSpace), 5);
  const rate = hover.coded
    ? `${hover.coded.bitsPerSite.toFixed(1)} bits/site coded (${hover.coded.lossDb} dB budget)`
    : `${BITS_PER_SITE} bits/site`;
  drawTooltip(p, g, g.x(hover.x), g.y(hover.y), {
    rows: [{ cells: [`${hover.sites} sites · ${fmtSig(hover.x)} bpp · ${hover.y.toFixed(2)} dB · ${COLOR_SPACES[run.colorSpace].label}`] },
           { cells: [rate] }],
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
    for (let i = 0; i <= n; i += 1) {
      const it = Math.round((i / n) * Math.max(0, plan.iters - 1));
      planned.push([it, plan.lrMult * lrSchedule(it, plan.iters, plan.kind, plan.growDoneIter)]);
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

// ---- Auto-tune: median PSNR vs the setting's value ----

const TUNE_DOT_COLOR = SERIES_COLORS[0];

function tunePoints() {
  return [...(state.tuneTrace?.points ?? [])].sort((a, b) => a.v - b.v);
}

function drawTunePlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const trace = state.tuneTrace;
  const pts = tunePoints();
  let xr = { min: 0, max: 1 }, yr = { min: 0, max: 1 };
  if (pts.length) {
    const vs = pts.map((pt) => pt.v), ms = pts.map((pt) => pt.median);
    const lo = Math.min(...vs), hi = Math.max(...vs);
    xr = lo > 0
      ? { min: lo / 1.25, max: hi * 1.25, log: true }
      : { min: lo - Math.max(1e-6, (hi - lo) * 0.15) , max: hi + Math.max(1e-6, (hi - lo) * 0.15) };
    const mlo = Math.min(...ms), mhi = Math.max(...ms);
    const pad = Math.max(0.05, (mhi - mlo) * 0.15);
    yr = { min: mlo - pad, max: mhi + pad };
  }
  const g = view.geom = plotAxes(plotBox(p), xr, yr);
  drawAxes(p, g, { xTitle: trace?.name ?? "setting", yTitle: "median PSNR (dB)", xLabels: false,
                   yFmt: (v) => v.toFixed(yr.max - yr.min < 1 ? 2 : 1) });
  if (!pts.length) {
    drawEmpty(p, g, "Click a tune button (or optimize all) to see its search");
    return;
  }
  // x labels at the values tried, skipping any that would overlap the last one.
  const { ctx, col } = p;
  ctx.font = PLOT_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = col.soft;
  let lastRight = -Infinity;
  for (const pt of pts) {
    const label = String(pt.v);
    const half = ctx.measureText(label).width / 2;
    if (g.x(pt.v) - half < lastRight + 6) continue;
    ctx.fillText(label, g.x(pt.v), g.bottom + 6);
    lastRight = g.x(pt.v) + half;
  }
  drawRefLine(p, g, { x: trace.start, label: "start" });
  drawLine(p, g, pts.map((pt) => [pt.v, pt.median]), col.soft, { width: 1 });
  for (const pt of pts) drawMarker(p, g, pt.v, pt.median, pt.v === trace.best ? PSNR_LINE_COLOR : TUNE_DOT_COLOR);
  const hover = view.hoverKey != null ? pts[view.hoverKey] : null;
  if (hover) {
    drawMarker(p, g, hover.v, hover.median, hover.v === trace.best ? PSNR_LINE_COLOR : TUNE_DOT_COLOR, 6);
    drawTooltip(p, g, g.x(hover.v), g.y(hover.median), {
      rows: [{ cells: [`${trace.name} ${hover.v}`, `${hover.median.toFixed(3)} dB`] }],
      footer: (hover.round ? `round ${hover.round}` : "starting value") + (hover.v === trace.best ? " · best" : ""),
    });
  }
}

function hitTune(g, mx) {
  return nearestX(g, tunePoints(), (pt) => pt.v, mx);
}

// ---- Predicted vs realized gain of grown sites (diagnostic) ----

function growGainPoints() {
  const n = state.growthGain.length;
  return state.growthGain.flatMap((b, bi) => b.points.map((pt) => ({ ...pt, step: b.step, addedAt: b.addedAt, age: n - 1 - bi })));
}

function drawGrowGainPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const pts = growGainPoints();
  let lo = Infinity, hi = -Infinity;
  for (const pt of pts) for (const v of [pt.predicted, pt.realized]) if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const r = decadeRange(lo, hi, [1e3, 1e8]);
  const g = view.geom = plotAxes(plotBox(p), r, { ...r });
  drawAxes(p, g, { xTitle: "predicted gain (levels²)", yTitle: "realized (levels²)", xFmt: fmtLogTick, yFmt: fmtLogTick,
                   xSpacing: 44, ySpacing: 26 });
  if (!pts.length) {
    drawEmpty(p, g, state.running ? "Waiting for the first growth step…" : "Train in grow mode to see this");
    return;
  }
  drawLine(p, g, [[r.min, r.min], [r.max, r.max]], p.col.soft, { dash: [4, 4], width: 1 });
  // Newest batch strongest; sites worth ≤ 0 sit on the bottom edge, hollow.
  const yOf = (pt) => (pt.realized > 0 ? pt.realized : r.min);
  for (const pt of [...pts].sort((a, b) => b.age - a.age)) {
    const alpha = Math.max(0.2, 1 - pt.age * 0.15);
    const x = g.x(pt.predicted), y = g.y(yOf(pt));
    if (!g.ok(pt.predicted, yOf(pt))) continue;
    p.ctx.beginPath();
    p.ctx.arc(x, y, 3, 0, Math.PI * 2);
    p.ctx.globalAlpha = alpha;
    if (pt.realized > 0) { p.ctx.fillStyle = PSNR_LINE_COLOR; p.ctx.fill(); }
    else { p.ctx.strokeStyle = PSNR_LINE_COLOR; p.ctx.lineWidth = 1.5; p.ctx.stroke(); }
    p.ctx.globalAlpha = 1;
  }
  const last = state.growthGain.at(-1);
  const sum = (b, k) => b.points.reduce((a, q) => a + q[k], 0);
  const worthless = last.points.filter((q) => q.realized <= 0).length;
  p.ctx.font = PLOT_FONT;
  p.ctx.fillStyle = p.col.dim;
  p.ctx.textAlign = "right";
  p.ctx.textBaseline = "top";
  p.ctx.fillText(`step ${last.step}: realized/predicted ${(sum(last, "realized") / sum(last, "predicted")).toFixed(2)} · ` +
                 `${worthless}/${last.points.length} worth ≤ 0`, g.right, g.top + 2);
  const hover = view.hoverKey != null ? pts[view.hoverKey] : null;
  if (!hover) return;
  drawMarker(p, g, hover.predicted, yOf(hover), PSNR_LINE_COLOR, 5);
  drawTooltip(p, g, g.x(hover.predicted), g.y(yOf(hover)), {
    header: [`site ${hover.site} · step ${hover.step} (iter ${hover.addedAt})`, ""],
    rows: [
      { cells: ["predicted", fmtExp(hover.predicted)] },
      { cells: ["realized", hover.realized > 0 ? fmtExp(hover.realized) : `${hover.realized.toExponential(1)} (≤ 0)`] },
      { cells: ["ratio", (hover.realized / hover.predicted).toFixed(2)] },
    ],
  });
}

function hitGrowGain(g, mx, my) {
  const pts = growGainPoints();
  const lo = g.yr.min;
  let best = null, bestD = 144; // within 12 px
  pts.forEach((pt, i) => {
    const d = (g.x(pt.predicted) - mx) ** 2 + (g.y(pt.realized > 0 ? pt.realized : lo) - my) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function hitLr(g, mx) {
  return nearestX(g, state.lrHistory, (d) => d.it, mx);
}

// ---- Merge edge weights (histogram, log x) ----

const MERGE_BINS = 40;

// Log-spaced bins over the step's weights: { r (x range), edges, all, merged
// (counts), zeros } with weights at or below `lo` in the first bin.
function mergeEdgeBins(d) {
  const pos = d.edgeWeights.filter((w) => w > 0).sort();
  const q = (f) => pos[Math.min(pos.length - 1, Math.floor(f * pos.length))];
  let max = pos.length ? pos.at(-1) : 1;
  for (const w of d.mergedWeights) if (w > max) max = w;
  const r = decadeRange(pos.length ? Math.max(q(0.005), max * 1e-4) : 1e-2, max, [1e-2, 10]);
  const l0 = Math.log10(r.min), step = (Math.log10(r.max) - l0) / MERGE_BINS;
  const edges = Array.from({ length: MERGE_BINS + 1 }, (_, i) => 10 ** (l0 + i * step));
  const bin = (w) => Math.min(MERGE_BINS - 1, Math.max(0, Math.floor((Math.log10(Math.max(w, r.min)) - l0) / step)));
  const all = new Array(MERGE_BINS).fill(0), merged = new Array(MERGE_BINS).fill(0);
  for (const w of d.edgeWeights) all[bin(w)] += 1;
  for (const w of d.mergedWeights) merged[bin(w)] += 1;
  return { r, edges, all, merged, zeros: d.edgeWeights.length - pos.length };
}

function drawMergeEdgesPlot(view) {
  const p = beginPlot(view);
  if (!p) return;
  const d = state.mergeEdges;
  const items = [{ label: "all edges", color: SERIES_COLORS[0] }, { label: "merged", color: PSNR_LINE_COLOR }];
  const top = drawLegend(p, items) + 10;
  const bins = d && d.edgeWeights.length ? mergeEdgeBins(d) : null;
  const hi = bins ? Math.max(...bins.all, ...bins.merged) : 1;
  // Headroom for the two caption lines.
  const yr = { min: 0, max: niceTicks(0, hi * 1.4, 4).at(-1) || hi };
  const g = view.geom = plotAxes(plotBox(p, top, LEGEND_PAD), bins ? bins.r : { min: 1e-2, max: 10, log: true }, yr);
  g.bins = bins;
  drawAxes(p, g, { xTitle: "edge weight (1 = typical neighbour)", yTitle: "edges", xFmt: fmtLogTick, xSpacing: 44 });
  if (!bins) {
    drawEmpty(p, g, state.running ? "Waiting for a merge step…" : "Train with a count schedule that merges");
    return;
  }
  const { ctx } = p;
  const bar = (counts, color, alpha) => {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    counts.forEach((c, i) => {
      if (!c) return;
      const x0 = g.x(bins.edges[i]), x1 = g.x(bins.edges[i + 1]);
      ctx.fillRect(x0 + 0.5, g.y(c), Math.max(1, x1 - x0 - 1), g.bottom - g.y(c));
    });
    ctx.globalAlpha = 1;
  };
  bar(bins.all, SERIES_COLORS[0], 0.55);
  bar(bins.merged, PSNR_LINE_COLOR, 0.9);
  if (1 > g.xr.min && 1 < g.xr.max) drawLine(p, g, [[1, 0], [1, yr.max]], p.col.soft, { dash: [4, 4], width: 1 });
  ctx.font = PLOT_FONT;
  ctx.fillStyle = p.col.dim;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  const fmtScale = (v) => (Number.isFinite(v) ? Number(v.toPrecision(2)) : "–");
  ctx.fillText(`iter ${d.it} · ${d.edgeWeights.length} edges · ${d.mergedWeights.length} merged`, g.right, g.top + 2);
  ctx.fillText(`scales: ${d.groups.map((n, k) => `${n} ${fmtScale(d.scales[k])}`).join(" · ")}`, g.right, g.top + 16);
  if (view.hoverKey == null) return;
  const i = view.hoverKey;
  const xm = Math.sqrt(bins.edges[i] * bins.edges[i + 1]);
  drawTooltip(p, g, g.x(xm), g.y(Math.max(bins.all[i], bins.merged[i])), {
    header: [`weight ${bins.edges[i].toPrecision(2)} – ${bins.edges[i + 1].toPrecision(2)}`, ""],
    rows: [
      { cells: ["edges", String(bins.all[i]) + (i === 0 && bins.zeros ? ` (${bins.zeros} at 0)` : "")] },
      { cells: ["merged", String(bins.merged[i])] },
    ],
  });
}

function hitMergeEdges(g, mx) {
  if (!g.bins || mx < g.left || mx > g.right) return null;
  const i = g.bins.edges.findIndex((e, k) => k < MERGE_BINS && mx >= g.x(e) && mx < g.x(g.bins.edges[k + 1]));
  return i < 0 ? null : i;
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

// ---- Site parameter mutual information ----
// One column per parameter in the form that matters for coding. Orientation
// is the major axis mod π: voronoi_dmix2 weights distance along aniso_dir by
// exp(log_aniso), so for log_aniso > 0 the cell is long across that direction.
// It's left undefined (NaN) for near-round sites, whose θ is noise. The last
// four columns describe each site's k nearest neighbours (neighborFeatures),
// which a decoder knows once positions are decoded.

const modPi = (t) => ((t % Math.PI) + Math.PI) % Math.PI;
function majorAxis(s, b) {
  if (!(Math.abs(s[b + 9]) >= MI_ROUND_ANISO)) return NaN;
  return modPi(Math.atan2(s[b + 8], s[b + 7]) + (s[b + 9] > 0 ? Math.PI / 2 : 0));
}

// get(sites, base, neighbours, active index) → value, NaN = undefined.
const MI_COLUMNS = [
  { short: "τ", label: "log_tau", get: (s, b) => s[b + 2] },
  { short: "rad", label: "radius (distance offset)", get: (s, b) => s[b + 3] },
  { short: "elong", label: "elongation |log_aniso|", get: (s, b) => Math.abs(s[b + 9]) },
  { short: "θ", label: "major axis θ mod π", range: [0, Math.PI], get: majorAxis },
  { short: "R", label: "red (decoded)", get: (s, b) => s[b + 4] },
  { short: "G", label: "green (decoded)", get: (s, b) => s[b + 5] },
  { short: "B", label: "blue (decoded)", get: (s, b) => s[b + 6] },
  { short: "space", label: `log spacing (${NEIGHBOR_K}th-neighbour distance)`,
    get: (s, b, nb, a) => Math.log(nb.spacing[a]) },
  { short: "φ", label: "neighbour direction φ mod π", range: [0, Math.PI], get: (s, b, nb, a) => nb.phi[a] },
  { short: "θ−φ", label: "major axis vs neighbours, θ − φ mod π", range: [0, Math.PI],
    get: (s, b, nb, a) => modPi(majorAxis(s, b) - nb.phi[a]) },
  { short: "ecc", label: "neighbour eccentricity", get: (s, b, nb, a) => nb.ecc[a] },
];

// MI over the active sites of the latest readback, cached per readback.
function currentSiteMI() {
  const sites = state.lastSites;
  if (!sites) return null;
  if (state.siteMI?.sites === sites) return state.siteMI.mi;
  const active = [];
  for (let i = 0; i < state.lastSiteCount; i += 1) if (sites[i * SITE_FLOATS] >= 0) active.push(i * SITE_FLOATS);
  const nb = neighborFeatures(Float32Array.from(active, (b) => sites[b]),
                             Float32Array.from(active, (b) => sites[b + 1]));
  const columns = MI_COLUMNS.map((c) => ({
    range: c.range,
    values: Float32Array.from(active, (b, a) => c.get(sites, b, nb, a)),
  }));
  const mi = active.length > 1 ? mutualInfoMatrix(columns) : null;
  state.siteMI = { sites, mi };
  return mi;
}

function drawSiteMI(view) {
  const p = beginPlot(view);
  if (!p) return;
  const { ctx, col, w, h } = p;
  const mi = currentSiteMI();
  const k = MI_COLUMNS.length;
  // Column labels sit above the cells, or turn vertical when the cells are
  // too narrow for them.
  ctx.font = PLOT_FONT;
  const maxLabel = Math.max(...MI_COLUMNS.map((c) => ctx.measureText(c.short).width));
  const labelW = Math.ceil(maxLabel) + 10;
  const fit = (labelH) => {
    const box = { left: 12 + labelW, right: w - 12, top: PLOT_PAD.top + labelH, bottom: h - 28 };
    return { box, cell: Math.max(1, Math.floor(Math.min((box.right - box.left) / k, (box.bottom - box.top) / k))) };
  };
  let { box, cell } = fit(18);
  const vertical = maxLabel + 4 > cell;
  if (vertical) ({ box, cell } = fit(Math.ceil(maxLabel) + 8));
  if (!mi) {
    view.geom = null;
    drawEmpty(p, { ...box, left: 12, top: PLOT_PAD.top }, "Sites appear after the first render");
    return;
  }
  const x0 = Math.round(box.left + (box.right - box.left - cell * k) / 2);
  const y0 = box.top;
  view.geom = { ...box, x0, y0, cell, k };

  ctx.font = PLOT_FONT;
  for (let i = 0; i < k; i += 1) {
    for (let j = 0; j < k; j += 1) {
      const v = mi.norm[i][j];
      const idx = Math.max(0, Math.min(255, Math.round(v * 255))) * 3;
      ctx.fillStyle = `rgb(${SPECTRUM_LUT[idx]}, ${SPECTRUM_LUT[idx + 1]}, ${SPECTRUM_LUT[idx + 2]})`;
      const x = x0 + j * cell, y = y0 + i * cell;
      ctx.fillRect(x, y, cell - 1, cell - 1);
      if (cell >= 28 && i !== j) {
        ctx.fillStyle = v > 0.5 ? "#ffffff" : col.text;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(v.toFixed(2), x + cell / 2, y + cell / 2);
      }
    }
  }
  if (view.hoverKey != null) {
    const [i, j] = view.hoverKey.split(",").map(Number);
    ctx.strokeStyle = col.text;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 + j * cell + 0.75, y0 + i * cell + 0.75, cell - 2.5, cell - 2.5);
  }
  ctx.fillStyle = col.dim;
  MI_COLUMNS.forEach((c, i) => {
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(c.short, x0 - 6, y0 + i * cell + cell / 2);
    if (vertical) {
      ctx.save();
      ctx.translate(x0 + i * cell + cell / 2, y0 - 4);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "left";
      ctx.fillText(c.short, 0, 0);
      ctx.restore();
    } else {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(c.short, x0 + i * cell + cell / 2, y0 - 4);
    }
  });
  ctx.fillStyle = col.soft;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(`I / min(H) · ${mi.n[0][0]} sites · ${MI_BINS} bins · Miller–Madow`, w / 2, h - 8);

  if (view.hoverKey != null) {
    const [i, j] = view.hoverKey.split(",").map(Number);
    const a = MI_COLUMNS[i], b = MI_COLUMNS[j];
    const ax = x0 + j * cell + cell / 2, ay = y0 + i * cell;
    const tip = i === j
      ? { rows: [{ cells: [a.label, ""] }, { cells: ["H", `${mi.h[i].toFixed(2)} bits`] },
                 { cells: ["sites", String(mi.n[i][i])] }] }
      : { rows: [{ cells: [`${a.short} vs ${b.short}`, ""] },
                 { cells: ["I / min(H)", mi.norm[i][j].toFixed(3)] },
                 { cells: ["I", `${mi.bits[i][j].toFixed(3)} bits`] },
                 { cells: [`H(${a.short})`, `${mi.h[i].toFixed(2)} bits`] },
                 { cells: [`H(${b.short})`, `${mi.h[j].toFixed(2)} bits`] },
                 { cells: ["sites", String(mi.n[i][j])] }],
          footer: `${a.label} · ${b.label}` };
    drawTooltip(p, { ...view.geom, top: 4, bottom: h - 4, right: w - 4 }, ax, ay, tip);
  }
}

function hitSiteMI(g, mx, my) {
  const j = Math.floor((mx - g.x0) / g.cell), i = Math.floor((my - g.y0) / g.cell);
  return i >= 0 && j >= 0 && i < g.k && j < g.k ? `${i},${j}` : null;
}

// ---- Site snap analysis ----
// How many distinct values do the sites need? Every parameter except position
// is clustered with k-means under each site's own Gauss–Newton metric
// (Trainer.siteSensitivity), so the k-means cost is the predicted rise in
// squared error from snapping every site to its cluster's centroid. Nothing
// here depends on a parameter's units or meaning, with two exceptions:
// position stays out (it's the site's identity and gets coded separately) and
// aniso_dir is sign-canonicalized, since ±dir render the same. Each k is also
// checked by re-rendering with snapped sites. Runs between training runs.

const SNAP_KS = [4, 8, 16, 32, 64];
const SNAP_PARAMS = [ // [site-float index, label]
  [2, "log_tau"], [3, "radius"], [4, "red"], [5, "green"], [6, "blue"],
  [7, "dir_x"], [8, "dir_y"], [9, "log_aniso"],
];
const SNAP_COLORS = SERIES_COLORS.slice(0, SNAP_MAP_K);

// The analysis of the latest sites: started here on first use once training
// has stopped, then redrawn when it finishes. null while there's nothing to do.
function snapAnalysis() {
  const sites = state.lastSites;
  if (state.snapAnalysis?.sites === sites) return state.snapAnalysis;
  if (!sites || !state.trainer || state.running) return null;
  const job = { sites, status: "running", result: null, error: null };
  const trainer = state.trainer;
  job.promise = withGpu(() => runSnapAnalysis(job, trainer))
    .then((result) => { job.result = result; job.status = "done"; })
    .catch((err) => { job.error = err.message; job.status = "error"; console.error(err); })
    .finally(() => { if (state.snapAnalysis === job) drawSnapViews(); });
  state.snapAnalysis = job;
  return job;
}

// Analyses that swap other sites into the trainer to re-render (snap
// analysis, entropy coding) take turns: each runs once the last has finished.
// Training and snapshot loads wait for gpuIdle() before touching the trainer.
let gpuChain = Promise.resolve();
function withGpu(fn) {
  const job = gpuChain.then(fn);
  gpuChain = job.catch(() => {});
  return job;
}
const gpuIdle = () => gpuChain;

function drawSnapViews() {
  drawCodecViews();
  if (VIEWS.siteClusters.canvas) drawSnapAnalysis(VIEWS.siteClusters);
  drawClusterMap();
  if (VIEWS.sensitivity.canvas) drawSensitivity(VIEWS.sensitivity);
}

// Gauss–Newton diagonal of `sites` (Trainer.siteSensitivity), computed once per
// sites and shared by the snap analysis and the sensitivity histogram, so the
// GPU pass runs before the snap analysis starts swapping sites in to re-render.
function sensitivityOf(trainer, sites) {
  if (state.sensitivity?.sites !== sites) state.sensitivity = { sites, promise: trainer.siteSensitivity() };
  return state.sensitivity.promise;
}

async function runSnapAnalysis(job, trainer) {
  const sites = job.sites;
  const { F, samples } = await sensitivityOf(trainer, sites);
  const d = SNAP_PARAMS.length;
  const bases = [];
  for (let i = 0; i < state.lastSiteCount; i += 1) if (sites[i * SITE_FLOATS] >= 0) bases.push(i * SITE_FLOATS);
  const n = bases.length;
  const X = new Float64Array(n * d), W = new Float64Array(n * d);
  bases.forEach((b, a) => {
    const flip = sites[b + 7] < 0 || (sites[b + 7] === 0 && sites[b + 8] < 0) ? -1 : 1;
    const f = (b / SITE_FLOATS) * 10;
    SNAP_PARAMS.forEach(([idx], t) => {
      X[a * d + t] = (idx === 7 || idx === 8 ? flip : 1) * sites[b + idx];
      const w = F[f + idx];
      W[a * d + t] = Number.isFinite(w) && w > 0 ? w : 0;
    });
  });
  const yieldUi = () => new Promise((r) => setTimeout(r, 0));

  // One shared value per parameter (its weighted mean): what each costs alone.
  const tDir = [7, 8].map((idx) => SNAP_PARAMS.findIndex(([j]) => j === idx));
  const single = [];
  for (const [t, [idx, label]] of SNAP_PARAMS.entries()) {
    let sw = 0, swx = 0;
    for (let a = 0; a < n; a += 1) { sw += W[a * d + t]; swx += W[a * d + t] * X[a * d + t]; }
    const mean = sw > 0 ? swx / sw : 0;
    let cost = 0;
    const snapped = Float32Array.from(sites);
    bases.forEach((b, a) => {
      cost += W[a * d + t] * (X[a * d + t] - mean) ** 2;
      // Sign-canonicalized dir: write the whole canonical pair, then the mean.
      if (idx === 7 || idx === 8) { snapped[b + 7] = X[a * d + tDir[0]]; snapped[b + 8] = X[a * d + tDir[1]]; }
      snapped[b + idx] = mean;
    });
    const { psnr } = await trainer.snapPsnr(snapped);
    single.push({ label, cost, psnr });
  }

  const fits = [];
  for (const k of SNAP_KS) {
    await yieldUi();
    const fit = kMeans(X, n, d, k, { weights: W });
    const snapped = Float32Array.from(sites);
    bases.forEach((b, a) => {
      const c = fit.labels[a];
      SNAP_PARAMS.forEach(([idx], t) => { snapped[b + idx] = fit.centroids[c * d + t]; });
    });
    const { psnr, basePsnr } = await trainer.snapPsnr(snapped);
    fits.push({ k, cost: fit.cost, labels: fit.labels, psnr, basePsnr });
  }
  // Predicted ΔPSNR from a predicted rise in summed squared error.
  const basePsnr = fits[0].basePsnr;
  const sse0 = samples * 10 ** (-basePsnr / 10);
  const predDb = (cost) => 10 * Math.log10(1 + cost / sse0);
  return {
    n, basePsnr,
    rows: fits.map((f) => ({ k: f.k, bits: Math.log2(f.k), pred: predDb(f.cost), real: basePsnr - f.psnr })),
    single: single.map((p) => ({ label: p.label, pred: predDb(p.cost), real: basePsnr - p.psnr })),
    map: { bases, labels: fits.find((f) => f.k === SNAP_MAP_K).labels },
  };
}

// A text table between x0 and x1 from box.top down: the first column left
// aligned, the rest right aligned. A row may be { cells, color }.
function drawTable(p, box, x0, x1, header, rows) {
  const { ctx, col } = p;
  const rowH = Math.min(18, (box.bottom - box.top) / (rows.length + 1));
  const n = header.length;
  const xs = header.map((_, j) => (j === 0 ? x0 : x0 + ((x1 - x0) * (j + 1)) / (n + 0.5)));
  ctx.font = PLOT_FONT;
  ctx.textBaseline = "middle";
  const drawRow = (cells, y, color) => {
    ctx.fillStyle = color;
    cells.forEach((c, j) => { ctx.textAlign = j === 0 ? "left" : "right"; ctx.fillText(c, xs[j], y); });
  };
  let y = box.top + rowH / 2;
  drawRow(header, y, col.dim);
  ctx.strokeStyle = col.line;
  ctx.beginPath(); ctx.moveTo(x0, y + rowH / 2); ctx.lineTo(x1, y + rowH / 2); ctx.stroke();
  for (const row of rows) { y += rowH; drawRow(row.cells || row, y, row.color || col.text); }
}

// Two tables side by side: per k, predicted vs re-rendered PSNR loss of the
// joint snap; per parameter, the predicted loss if every site shared one value.
function drawSnapAnalysis(view) {
  const p = beginPlot(view);
  if (!p) return;
  const { ctx, col, w, h } = p;
  const box = { left: 12, right: w - 12, top: PLOT_PAD.top, bottom: h - 28 };
  const job = snapAnalysis();
  const msg = state.running ? "Runs when training stops"
    : !job ? "Sites appear after the first render"
    : job.status === "running" ? "Analyzing…"
    : job.status === "error" ? `Analysis failed: ${job.error}` : null;
  if (msg) {
    drawEmpty(p, box, msg);
    return;
  }
  const r = job.result;
  const fmtDb = (v) => (v < 0.005 ? "<0.01" : v.toFixed(2));
  const table = (x0, x1, header, rows) => drawTable(p, box, x0, x1, header, rows);
  const mid = Math.round(w * 0.52);
  table(box.left, mid - 12, ["k clusters", "bits", "pred.", "actual"],
        r.rows.map((row) => [String(row.k), row.bits.toFixed(0), fmtDb(row.pred), fmtDb(row.real)]));
  table(mid + 4, box.right, ["one value for all", "pred.", "actual"],
        r.single.map((row) => [row.label, fmtDb(row.pred), fmtDb(row.real)]));
  ctx.fillStyle = col.soft;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(`PSNR loss (dB) from snapping: predicted, re-rendered · ${r.n} sites · base ${r.basePsnr.toFixed(2)} dB`,
               w / 2, h - 8);
}

// ---- Entropy coding ----
// The latest sites through trainer/codec.js at each CODEC_LOSSES budget: the
// bitstream's length is the rate, and the decoded sites re-rendered with
// rebuilt candidate lists give the PSNR. Runs between training runs; the
// CODEC_REPORT_LOSS row also goes on the run's PSNR vs bpp entry.

function codecJob() {
  const sites = state.lastSites;
  if (state.codec?.sites === sites) return state.codec;
  if (!sites || !state.trainer || state.running) return null;
  const job = { sites, status: "running", result: null, error: null };
  const trainer = state.trainer, count = state.lastSiteCount, run = state.bppRuns.at(-1);
  job.promise = withGpu(() => runCodec(sites, count, trainer))
    .then((result) => {
      job.result = result;
      job.status = "done";
      if (run && state.lastSites === sites) run.coded = result.report;
    })
    .catch((err) => { job.error = err.message; job.status = "error"; console.error(err); })
    .finally(() => { if (state.codec === job) drawCodecViews(); });
  state.codec = job;
  return job;
}

function drawCodecViews() {
  if (VIEWS.codec.canvas) drawCodec(VIEWS.codec);
  if (VIEWS.bpp.canvas) drawBppPlot(VIEWS.bpp);
}

async function runCodec(sites, count, trainer) {
  const { F, samples } = await sensitivityOf(trainer, sites);
  const { width, height } = trainer;
  const pixels = width * height;
  const globalSite = trainer.layerIndex ?? -1;
  // klt runs code colors along the run's own basis (the codec's colorBasis),
  // others with YCoCg-R.
  const colorBasis = trainer.colorSpace?.needsBasis ? trainer.colorSpace.basis : null;
  const { basePsnr } = await trainer.decodedPsnr(sites);
  const sse0 = samples * 10 ** (-basePsnr / 10);
  const rows = [];
  for (const lossDb of CODEC_LOSSES) {
    await new Promise((r) => setTimeout(r, 0));
    const steps = codecSteps(sites, count, F, { lossDb, sse0 });
    const enc = encodeSites(sites, count, width, height, steps, { globalSite, colorBasis });
    // Decoded sites back in their slots, so the layer keeps its index.
    const placed = Float32Array.from(sites);
    enc.slots.forEach((slot, i) => placed.set(enc.decoded.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS), slot * SITE_FLOATS));
    const { psnr, basePsnr: base } = await trainer.decodedPsnr(placed);
    rows.push({ lossDb, steps, enc, n: enc.n, bits: enc.bits, bitsPerSite: enc.bits / Math.max(1, enc.n),
                bpp: enc.bits / pixels, psnr, loss: base - psnr });
  }
  const report = rows.find((r) => r.lossDb === CODEC_REPORT_LOSS);
  return {
    basePsnr, rows, pixels, klt: !!colorBasis,
    report: { bpp: report.bpp, psnr: report.psnr, bitsPerSite: report.bitsPerSite, n: report.n },
  };
}

// Left: per budget, the rate and the measured loss. Right: the report
// budget's bits per site by stage.
function drawCodec(view) {
  const p = beginPlot(view);
  if (!p) return;
  const { ctx, col, w, h } = p;
  const box = { left: 12, right: w - 12, top: PLOT_PAD.top, bottom: h - 28 };
  const job = codecJob();
  const msg = state.running ? "Runs when training stops"
    : !job ? "Sites appear after the first render"
    : job.status === "running" ? "Coding…"
    : job.status === "error" ? `Coding failed: ${job.error}` : null;
  if (msg) {
    drawEmpty(p, box, msg);
    return;
  }
  const r = job.result;
  const rep = r.rows.find((row) => row.lossDb === CODEC_REPORT_LOSS);
  const mid = Math.round(w * 0.55);
  drawTable(p, box, box.left, mid - 12, ["budget", "bits/site", "bpp", "PSNR", "loss"],
    r.rows.map((row) => ({
      color: row === rep ? CODED_COLOR : undefined,
      cells: [String(row.lossDb), row.bitsPerSite.toFixed(1), fmtSig(row.bpp), row.psnr.toFixed(2), row.loss.toFixed(2)],
    })));
  const { enc, steps } = rep;
  const per = (bits) => (bits / Math.max(1, enc.n)).toFixed(2);
  const stepOf = { log_aniso: steps.logAniso, "θ": steps.theta, log_tau: steps.logTau, radius: steps.radius,
                   Y: steps.color, Co: steps.color, Cg: steps.color };
  // klt runs: the color planes are the run's KLT axes, not YCoCg-R
  const planeLabel = (name, i) => (r.klt && i >= 4 ? `K${i - 4}` : name);
  drawTable(p, box, mid + 4, box.right, ["stage", "bits/site", "step", "pred."], [
    ["header", per(enc.headerBits), "", ""],
    ["position", per(enc.positionBits), fmtSig(steps.pos) + " px", "gaps"],
    ...CODEC_PLANES.map((name, i) => [planeLabel(name, i), per(enc.planeBits[i]),
                                      Number.isFinite(stepOf[name]) ? fmtSig(stepOf[name]) : "∞", enc.modes[i] ? "nbr" : "const"]),
    { color: CODED_COLOR, cells: ["total", per(enc.bits), "", ""] },
  ]);
  ctx.fillStyle = col.soft;
  ctx.font = PLOT_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${enc.n} sites · decoded at ${r.basePsnr.toFixed(2)} dB unquantized · ` +
               `${BITS_PER_SITE} bits/site would be ${fmtSig((BITS_PER_SITE * enc.n) / r.pixels)} bpp`,
               w / 2, h - 8);
}

// ---- Site sensitivity histogram ----
// Per GRAD_GROUPS parameter group, a histogram over active sites of the
// summed Gauss–Newton diagonal F (log bins from the 2nd percentile up; lower
// values count in the first bin); y = number of sites per bin.

const SENS_BINS = 16;

function sensitivityJob() {
  const sites = state.lastSites;
  if (state.sensitivityHist?.sites === sites) return state.sensitivityHist;
  if (!sites || !state.trainer || state.running) return null;
  const job = { sites, status: "running", result: null, error: null };
  const count = state.lastSiteCount;
  sensitivityOf(state.trainer, sites)
    .then(({ F }) => { job.result = sensitivityHistograms(F, sites, count); job.status = "done"; })
    .catch((err) => { job.error = err.message; job.status = "error"; console.error(err); })
    .finally(() => { if (state.sensitivityHist === job && VIEWS.sensitivity.canvas) drawSensitivity(VIEWS.sensitivity); });
  state.sensitivityHist = job;
  return job;
}

// [{ label, zero, lo, hi, counts }] per group: log10-spaced bins between the
// smallest and largest positive F over active sites.
function sensitivityHistograms(F, sites, count) {
  return GRAD_GROUPS.map(({ label, channels }) => {
    const vals = [];
    let zero = 0;
    for (let i = 0; i < count; i += 1) {
      if (!(sites[i * SITE_FLOATS] >= 0)) continue;
      let f = 0;
      for (const c of channels) f += F[i * 10 + c];
      if (f > 0 && Number.isFinite(f)) vals.push(Math.log10(f)); else zero += 1;
    }
    // Bins span the 2nd percentile to the max; lower values go in the first.
    vals.sort((a, b) => a - b);
    let lo = vals.length ? vals[Math.floor(vals.length * 0.02)] : 0, hi = vals.length ? vals.at(-1) : 1;
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const counts = new Array(SENS_BINS).fill(0);
    for (const v of vals) counts[Math.max(0, Math.min(SENS_BINS - 1, Math.floor(((v - lo) / (hi - lo)) * SENS_BINS)))] += 1;
    return { label, zero, lo, hi, counts };
  });
}

function drawSensitivity(view) {
  const p = beginPlot(view);
  if (!p) return;
  const { ctx, col, w, h } = p;
  const job = sensitivityJob();
  const msg = state.running ? "Runs when training stops"
    : !job ? "Sites appear after the first render"
    : job.status === "running" ? "Computing…"
    : job.status === "error" ? `Failed: ${job.error}` : null;
  if (msg) {
    drawEmpty(p, plotBox(p), msg);
    return;
  }
  drawHistPanels(p, job.result.map((g) => ({
    title: g.label + (g.zero ? ` · ${g.zero} at 0` : ""),
    counts: g.counts,
    loText: `≤${fmtExp(10 ** g.lo)}`,
    hiText: fmtExp(10 ** g.hi),
  })));
}

// A grid of bar histograms, one per panel { title, counts, loText, hiText }:
// y = count per bin, x labelled only at the range ends.
function drawHistPanels(p, panels) {
  const { ctx, col, w, h } = p;
  const cols = w > h * 1.2 ? 3 : 2;
  const rows = Math.ceil(panels.length / cols);
  const head = PLOT_PAD.top; // below the card's label
  const cw = w / cols, ch = (h - head) / rows;
  panels.forEach((g, k) => {
    const x0 = (k % cols) * cw, y0 = head + Math.floor(k / cols) * ch;
    const box = { left: x0 + 34, right: x0 + cw - 8, top: y0 + 22, bottom: y0 + ch - 22 };
    const yMax = Math.max(1, ...g.counts);
    const yTicks = niceTicks(0, yMax, Math.max(2, Math.floor((box.bottom - box.top) / 22)));
    const top = Math.max(yMax, yTicks.at(-1));
    const yOf = (v) => box.bottom - (v / top) * (box.bottom - box.top);
    ctx.font = PLOT_FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of yTicks) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.strokeStyle = col.line;
      ctx.beginPath(); ctx.moveTo(box.left, y); ctx.lineTo(box.right, y); ctx.stroke();
      ctx.fillStyle = col.soft;
      ctx.fillText(String(v), box.left - 4, y);
    }
    const bw = (box.right - box.left) / g.counts.length;
    ctx.fillStyle = SERIES_COLORS[0];
    g.counts.forEach((c, b) => {
      if (!c) return;
      const y = yOf(c);
      ctx.fillRect(box.left + b * bw + 0.5, y, Math.max(1, bw - 1), box.bottom - y);
    });
    ctx.strokeStyle = col.soft;
    ctx.beginPath(); ctx.moveTo(box.left, box.bottom + 0.5); ctx.lineTo(box.right, box.bottom + 0.5); ctx.stroke();
    ctx.fillStyle = col.soft;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(g.loText, box.left, box.bottom + 4);
    ctx.textAlign = "right";
    ctx.fillText(g.hiText, box.right, box.bottom + 4);
    ctx.fillStyle = col.dim;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(g.title, box.left, box.top - 6);
  });
}

// ---- Site parameter values ----
// Per parameter, a histogram over active sites of its value: VALUE_BINS
// equal-width bins over the 1st–99th percentile (θ over [0, π)), with values
// outside counted in the end bins and noted in the panel title. The histogram
// layer is left out.

const VALUE_COLUMNS = [
  { label: "log τ", get: (s, b) => s[b + 2] },
  { label: "radius", get: (s, b) => s[b + 3] },
  { label: "log aniso", get: (s, b) => s[b + 9] },
  { label: "θ (major axis)", range: [0, Math.PI], get: majorAxis },
  { label: "red", get: (s, b) => s[b + 4] },
  { label: "green", get: (s, b) => s[b + 5] },
  { label: "blue", get: (s, b) => s[b + 6] },
];

// Panels for the latest readback, cached per readback.
function siteValueHistograms() {
  const sites = state.lastSites;
  if (!sites) return null;
  if (state.siteValues?.sites === sites) return state.siteValues.panels;
  const layer = state.trainer?.layerIndex;
  const active = [];
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    if (sites[i * SITE_FLOATS] >= 0 && i !== layer) active.push(i * SITE_FLOATS);
  }
  const fmt = (v) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2) ? fmtExp(v) : String(+v.toPrecision(3)));
  const panels = active.length ? VALUE_COLUMNS.map((c) => {
    const vals = active.map((b) => c.get(sites, b)).filter(Number.isFinite).sort((a, b) => a - b);
    const undef = active.length - vals.length;
    let [lo, hi] = c.range || (vals.length
      ? [vals[Math.floor((vals.length - 1) * 0.01)], vals[Math.ceil((vals.length - 1) * 0.99)]] : [0, 1]);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const counts = new Array(VALUE_BINS).fill(0);
    let clipped = 0;
    for (const v of vals) {
      if (v < lo || v > hi) clipped += 1;
      counts[Math.max(0, Math.min(VALUE_BINS - 1, Math.floor(((v - lo) / (hi - lo)) * VALUE_BINS)))] += 1;
    }
    const notes = [clipped && `${clipped} clipped`, undef && `${undef} round`].filter(Boolean);
    return {
      title: c.label + (notes.length ? ` · ${notes.join(", ")}` : ""),
      counts,
      loText: c.range ? "0" : fmt(lo),
      hiText: c.range ? "π" : fmt(hi),
    };
  }) : null;
  state.siteValues = { sites, panels };
  return panels;
}

function drawSiteValues(view) {
  const p = beginPlot(view);
  if (!p) return;
  const panels = siteValueHistograms();
  if (!panels) {
    drawEmpty(p, plotBox(p), "Sites appear after the first render");
    return;
  }
  drawHistPanels(p, panels);
}

// The reconstruction, washed out, with each site's centroid in the color of
// its cluster from the k = SNAP_MAP_K snap fit.
function drawClusterMap() {
  const canvas = VIEWS.clusterMap.canvas;
  const render = state.lastRender;
  if (!canvas || !render) return;
  const { rgba, width, height } = render;
  drawToCanvas(canvas, rgba, width, height);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
  ctx.fillRect(0, 0, width, height);
  const job = snapAnalysis();
  if (job?.status !== "done" || job.sites !== state.lastSites) return;
  const { bases, labels } = job.result.map;
  const r = Math.max(1.5, Math.max(width, height) / 350);
  bases.forEach((b, i) => {
    ctx.fillStyle = SNAP_COLORS[labels[i]];
    ctx.beginPath();
    ctx.arc(state.lastSites[b] + 0.5, state.lastSites[b + 1] + 0.5, r, 0, Math.PI * 2);
    ctx.fill();
  });
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

// The trainer's error EWMA (Trainer.readErrEma), drawn like the residual.
async function drawErrEma() {
  const render = state.lastRender;
  if (!VIEWS.errEma.canvas || !state.trainer || !renderMatchesTarget(render)) return;
  let ema;
  try {
    ema = await state.trainer.readErrEma();
  } catch (err) {
    return; // the trainer was torn down mid-readback
  }
  const canvas = VIEWS.errEma.canvas;
  const { width, height } = render;
  if (!canvas || !ema || ema.length !== width * height * 4) return;
  const out = new Uint8ClampedArray(width * height * 4);
  const scale = 255 / (3 * RESIDUAL_MAX);
  for (let i = 0; i < out.length; i += 4) {
    const err = Math.abs(ema[i]) + Math.abs(ema[i + 1]) + Math.abs(ema[i + 2]);
    const idx = Math.min(255, Math.round(err * scale)) * 3;
    out[i] = RESIDUAL_LUT[idx];
    out[i + 1] = RESIDUAL_LUT[idx + 1];
    out[i + 2] = RESIDUAL_LUT[idx + 2];
    out[i + 3] = 255;
  }
  putImage(canvas, out, width, height);
}

// Per-site loss gradients (state.gradSnapshot) painted over the pixels each
// site owns (renderView("owner")), log scale through cividis.
const CIVIDIS_LUT = rampLut([
  "#00204d", "#00336f", "#39486b", "#575c6d", "#707173",
  "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#ffea46",
].map((hex, i, all) => [i / (all.length - 1), [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16))]));
const GRAD_HEAT_LO_PCT = 0.01;
const GRAD_HEAT_HI_PCT = 0.995;

function buildGradHeatControls(card) {
  const box = document.createElement("label");
  box.className = "overlay-toggle";
  box.title = "Which gradient to show";
  const select = document.createElement("select");
  select.add(new Option("magnitude", "mag"));
  GRAD_CHANNELS.forEach((name, c) => select.add(new Option(name, String(c))));
  select.value = state.gradHeatParam;
  select.addEventListener("change", () => {
    state.gradHeatParam = select.value;
    drawGradHeat();
  });
  const range = document.createElement("span");
  box.append(select, range);
  card.append(box);
  VIEWS.gradHeat.rangeEl = range;
}

async function drawGradHeat() {
  const view = VIEWS.gradHeat;
  const snap = state.gradSnapshot;
  if (!view.canvas || !snap || !state.lastRender) return;
  const img = await ensureRender("owner");
  if (!img || !view.canvas || state.gradSnapshot !== snap) return;
  const { grads, siteCount, numPixels } = snap;
  const param = state.gradHeatParam;
  const values = new Float32Array(siteCount);
  for (let i = 0; i < siteCount; i += 1) {
    const base = i * 10;
    if (param === "mag") {
      let sq = 0;
      for (let c = 0; c < 10; c += 1) sq += grads[base + c] * grads[base + c];
      values[i] = Math.sqrt(sq) / numPixels;
    } else {
      values[i] = Math.abs(grads[base + Number(param)]) / numPixels;
    }
  }
  const sorted = values.filter((v) => v > 0 && Number.isFinite(v)).sort();
  const pct = (q) => Math.log10(sorted[Math.floor(q * (sorted.length - 1))]);
  const lo = sorted.length ? pct(GRAD_HEAT_LO_PCT) : 0;
  const span = sorted.length ? Math.max(pct(GRAD_HEAT_HI_PCT) - lo, 1e-6) : 1;
  const colors = new Uint8Array(siteCount * 3);
  for (let i = 0; i < siteCount; i += 1) {
    const v = values[i];
    const t = v > 0 ? Math.min(1, Math.max(0, (Math.log10(v) - lo) / span)) : 0;
    const idx = Math.round(t * 255) * 3;
    colors.set(CIVIDIS_LUT.subarray(idx, idx + 3), i * 3);
  }
  const { rgba, width, height } = img;
  const mask = state.maskData?.rgba;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p += 1) {
    const o = p * 4;
    const id = rgba[o];
    out[o + 3] = 255;
    if (id < 0 || id >= siteCount || (mask && mask[o] <= 0)) continue;
    out[o] = colors[id * 3];
    out[o + 1] = colors[id * 3 + 1];
    out[o + 2] = colors[id * 3 + 2];
  }
  putImage(view.canvas, out, width, height);
  if (view.rangeEl) {
    view.rangeEl.textContent = sorted.length ? `${fmtExp(10 ** lo)} – ${fmtExp(10 ** (lo + span))}` : "no gradient";
  }
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
  ui.itemSelect.value = "";
  await setTarget(file, file.name);
}

// One Hugging Face parquet file per dataset: a row per image, with its
// encoded PNG / WebP bytes in an Image column, and for labelled sets a
// ClassLabel column, whose class names come from the file's "huggingface"
// metadata. The whole file is read once, on the first prepare().
function parquetDataset({ key, label, preset, file, source, imageColumn = "image", labelColumn = null,
                          nameColumn = null, prefix = key, classTitle = null }) {
  let rows = null; // promise of { images, labels, names, classNames }
  const read = () => {
    rows ??= (async () => {
      const dir = new URLSearchParams(location.search).get(key) || `${DATA_DIR}${key}/`;
      const res = await fetch(new URL(file, new URL(dir, location.href)));
      if (!res.ok) throw new Error(`${file}: HTTP ${res.status} (get it from ${source})`);
      const buf = await res.arrayBuffer();
      const metadata = parquetMetadata(buf);
      const hf = metadata.key_value_metadata?.find((kv) => kv.key === "huggingface");
      const features = hf ? JSON.parse(hf.value).info?.features ?? {} : {};
      const columns = [imageColumn, labelColumn, nameColumn].filter(Boolean);
      // utf8: false keeps the image bytes raw (string columns are still decoded).
      const objects = await parquetReadObjects({ file: buf, metadata, columns, utf8: false });
      return {
        images: objects.map((o) => o[imageColumn].bytes),
        labels: labelColumn ? Int32Array.from(objects, (o) => Number(o[labelColumn])) : null,
        names: nameColumn ? objects.map((o) => String(o[nameColumn]).split("/").pop().replace(/\.[^.]*$/, "")) : null,
        classNames: labelColumn ? features[labelColumn]?.names ?? null : null,
      };
    })();
    rows.catch(() => { rows = null; }); // let a failed load retry
    return rows;
  };
  return {
    label, preset, classTitle,
    count: 0,
    labels: null,
    names: null,
    classNames: null,
    async prepare() {
      const r = await read();
      ({ labels: this.labels, names: this.names, classNames: this.classNames } = r);
      this.count = r.images.length;
    },
    itemName(i) {
      if (this.names) return this.names[i];
      const tag = this.labels ? `_${this.classNames?.[this.labels[i]] ?? this.labels[i]}` : "";
      return `${prefix}${String(i).padStart(4, "0")}${tag}`;
    },
    labelOf(i) { return this.labels?.[i] ?? null; },
    async load(i) { return new Blob([(await read()).images[i]]); },
  };
}

// A folder of PNGs inside this repo (../data/<folder>/), listed through the
// dev server's directory index (python's http.server: a link per file), one
// item per PNG in name order. `name` shortens a file's stem for the picker.
// Override the folder with ?<key>=<dir>.
function pngFolderDataset({ key, label, preset, folder, source, name = (stem) => stem }) {
  let files = null; // promise of [{ url, stem }]
  const list = () => {
    files ??= (async () => {
      const dir = new URL(new URLSearchParams(location.search).get(key) || `../data/${folder}/`, location.href);
      const res = await fetch(dir);
      if (!res.ok) throw new Error(`${folder}: HTTP ${res.status} (get it from ${source})`);
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      const found = [...doc.querySelectorAll("a[href]")]
        .map((a) => a.getAttribute("href"))
        .filter((href) => /\.png$/i.test(href))
        .map((href) => ({ url: new URL(href, dir), stem: decodeURIComponent(href.split("/").pop()).replace(/\.png$/i, "") }))
        .sort((x, y) => (x.stem < y.stem ? -1 : x.stem > y.stem ? 1 : 0));
      if (!found.length) throw new Error(`${folder}: no PNGs listed (get them from ${source})`);
      return found;
    })();
    files.catch(() => { files = null; }); // let a failed load retry
    return files;
  };
  return {
    label, preset, classTitle: null,
    count: 0,
    names: null,
    async prepare() {
      const f = await list();
      this.names = f.map((x) => name(x.stem));
      this.count = f.length;
    },
    itemName(i) { return this.names[i]; },
    labelOf() { return null; },
    async load(i) {
      const { url } = (await list())[i];
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${url.pathname}: HTTP ${res.status}`);
      return res.blob();
    },
  };
}

// Each dataset: item names for the picker (after prepare()), load(i)
// returning something createImageBitmap takes, and the preset (PRESETS key)
// selecting it applies. A labelled one also has labelOf(i) and classNames
// (the batch's PSNR-by-class plot). Every image is checked against the
// original release: Kodak's lossless WebPs match the PNGs, MNIST's PNGs the
// IDX files.
const DATASETS = {
  kodak: parquetDataset({
    key: "kodak", label: "Kodak", preset: "kodak",
    file: "validation.parquet",
    source: "https://huggingface.co/datasets/danjacobellis/kodak/resolve/main/data/validation-00000-of-00001.parquet",
    nameColumn: "path",
  }),
  mnist: parquetDataset({
    key: "mnist", label: "MNIST", preset: "mnist",
    file: "test.parquet",
    source: "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet",
    labelColumn: "label", classTitle: "MNIST digit",
  }),
  cifar10: parquetDataset({
    key: "cifar10", label: "CIFAR-10",
    // MNIST's fixed-site preset (one atlas per batch) with LR bases tuned on CIFAR.
    preset: "cifar10",
    file: "test.parquet",
    source: "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    imageColumn: "img", labelColumn: "label", prefix: "cifar", classTitle: "CIFAR-10 class",
  }),
  clic2025: pngFolderDataset({
    key: "clic2025", label: "CLIC 2025 test",
    // Large photos like Kodak's (up to 2048 px), so its preset; not tuned on CLIC.
    preset: "kodak",
    folder: "clic2025_image_test",
    source: "https://clic2027.compression.cc/tasks/#image",
    // Stems are 64-hex-digit hashes; the first 8 tell the 30 apart.
    name: (stem) => `clic_${stem.slice(0, 8)}`,
  }),
};

function currentDataset() {
  return DATASETS[ui.datasetSelect.value];
}

async function selectDataset(key) {
  const dataset = DATASETS[key];
  ui.datasetSelect.value = key;
  const presetApplied = applyPreset(dataset.preset ?? "");
  ui.itemSelect.length = 0;
  ui.itemSelect.add(new Option(`${dataset.label}…`, ""));
  const prevStatus = ui.status.textContent;
  try {
    setStatus(`Loading ${dataset.label}…`);
    await dataset.prepare();
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
    return false;
  }
  await presetApplied;
  if (ui.datasetSelect.value !== key) return false;
  for (let i = 0; i < dataset.count; i += 1) {
    ui.itemSelect.add(new Option(dataset.itemName(i), String(i)));
  }
  setStatus(prevStatus);
  maybeEnableStart();
  return true;
}

async function loadItem(i) {
  if (state.running) return;
  const dataset = currentDataset();
  const name = dataset.itemName(i);
  // Fast flicks overlap loads; only the latest one may land.
  const token = ++state.targetToken;
  ui.itemSelect.value = String(i);
  // Clear so re-picking the same local file after a dataset image still fires change.
  ui.imageInput.value = "";
  setStatus(`Loading ${name}…`);
  ui.imageLabel.textContent = name;
  try {
    const source = await dataset.load(i);
    if (token !== state.targetToken) return;
    await setTarget(source, name, token);
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
  }
}

function stepItem(delta) {
  const count = currentDataset().count;
  if (ui.itemSelect.length <= 1) return; // still preparing
  const cur = ui.itemSelect.value === "" ? (delta > 0 ? -1 : 0) : Number(ui.itemSelect.value);
  loadItem((cur + delta + count) % count);
}

function setDatasetEnabled(enabled) {
  ui.datasetSelect.disabled = !enabled;
  ui.itemSelect.disabled = !enabled;
  ui.itemPrev.disabled = !enabled;
  ui.itemNext.disabled = !enabled;
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
    state.lastOwner = null;
    ui.saveBtn.disabled = true;
    invalidateAnalysis();
    syncBppSites();
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

// Single mode needs a loaded target image; batch mode only needs a dataset.
function maybeEnableStart() {
  const batch = state.runMode === "batch";
  const datasetReady = (currentDataset()?.count ?? 0) > 0;
  ui.startBtn.disabled = state.running || !!state.tune || !(state.device && (batch ? datasetReady : state.targetImage));
  ui.batchBtn.disabled = !state.device || state.running;
  ui.singleModeBtn.disabled = state.running;
  ui.batchModeBtn.disabled = state.running;
  syncTuneButtons();
}

// Batch is the default on every page load: Train runs the batch directly, and
// only then does the batch-settings button appear.
function setRunMode(mode) {
  if (state.running) return;
  state.runMode = mode === "single" ? "single" : "batch";
  const batch = state.runMode === "batch";
  ui.singleModeBtn.setAttribute("aria-pressed", String(!batch));
  ui.batchModeBtn.setAttribute("aria-pressed", String(batch));
  ui.startBtn.textContent = batch ? "Run batch" : "Train";
  ui.batchBtn.hidden = !batch;
  ui.loopToggle.disabled = batch;
  if (!batch) ui.batchPopover.classList.add("hidden");
  maybeEnableStart();
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
  ui.batchBtn.disabled = true;
  syncTuneButtons();
  ui.singleModeBtn.disabled = true;
  ui.batchModeBtn.disabled = true;
  ui.stopBtn.disabled = false;
  setDatasetEnabled(false);
  state.running = true;
  state.abortCtrl = new AbortController();
  const signal = state.abortCtrl.signal;
  state.lastHashed = null;
  state.lastHeatmap = null;
  state.lastOwner = null;
  // The previous run's trainer only served site-ID / tau renders of its last
  // state; free its GPU buffers so repeated (or looped) runs don't pile up.
  await gpuIdle(); // the analyses render with the old trainer
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
  const initSiteCount = init ? init.count : args.startSites;

  state.psnrHistory = [];
  state.lastLrScale = null;
  state.psnrIters = args.iters;
  state.gradHistory = [];
  state.gradSnapshot = null;
  state.bandHistory = [];
  state.growthGain = [];
  state.mergeEdges = null;
  state.lrHistory = [];
  state.snapAnalysis = null;
  // The count schedule ends at countHorizon (where "decay after growth"
  // starts decaying) unless λ stops it early; the plan is corrected once the
  // trainer reports the real end.
  const startCount = initSiteCount;
  const varies = countVaries(args, startCount);
  const plannedGrowEnd = varies ? countHorizon(args) : 0;
  state.runSiteMode = varies ? "grow" : "fixed";
  state.lastGrowth = null;
  state.lrPlan = { iters: args.iters, lrMult: args.lrMult,
                   kind: args.lrSchedule, growDoneIter: plannedGrowEnd };
  state.lastActiveSites = initSiteCount;
  state.bppRuns.push({
    sites: initSiteCount, bpp: computeBpp(initSiteCount, width, height), colorSpace: args.colorSpace, points: [],
  });
  for (const view of Object.values(VIEWS)) view.hoverKey = null;
  await refreshViews("run");

  const sitesDesc = varies
    ? `${startCount} → ${args.endSites} ${args.countSchedule}` +
      (args.countSchedule === "sawtooth" ? ` (${args.sawTeeth} teeth, +${Math.round(args.sawAmp * 100)}%, ${args.sawRatio}:1)` : "") +
      ` by iter ${countHorizon(args)}, step every ${args.growEvery} iters (λ stop ${args.growLambda})`
    : `${initSiteCount} fixed`;
  appendLog(`Target: ${width}×${height} | sites: ${sitesDesc} | iters: ${args.iters} | lr ×${args.lrMult} ${LR_SCHEDULES[args.lrSchedule]} | color: ${COLOR_SPACES[args.colorSpace].label} | loss: ${LOSSES[args.loss].label} | refresh: ${args.logFreq}`);
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
        state.lastGrowth = { mode: state.runSiteMode, doneIter: info.growDoneIter, stop: info.growStop,
                             sites: info.totalSites, it: info.it, iters: info.iters };
        updatePlanSummary();
        if (Number.isFinite(info.psnr)) {
          state.psnrHistory.push({ it: info.it, psnr: info.psnr, sites: info.activeSites });
          const run = state.bppRuns[state.bppRuns.length - 1];
          run.sites = info.activeSites;
          run.bpp = computeBpp(info.activeSites, info.width, info.height);
          run.points.push({ it: info.it, psnr: info.psnr, sites: run.sites, bpp: run.bpp });
        }
        if (info.gradStats) state.gradHistory.push({ it: info.it, stats: info.gradStats });
        if (info.grads) {
          state.gradSnapshot = { grads: info.grads, siteCount: info.totalSites, numPixels: info.width * info.height };
        }
        if (Number.isFinite(info.lrScale)) state.lrHistory.push({ it: info.it, lr: info.lrScale });
        recordBandSnr(info.it);
        state.lastHashed = null;
        state.lastHeatmap = null;
        state.lastOwner = null;
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
        state.lastOwner = null;
        updateProgress(info.it + 1, info.iters,
          `image iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await refreshViews("render");
      } else if (info.type === "mergeEdges") {
        state.mergeEdges = info;
        await refreshViews("merge");
      } else if (info.type === "growthGain") {
        state.growthGain.push({ step: info.step, addedAt: info.addedAt, measuredAt: info.measuredAt, points: info.points });
        await refreshViews("growth");
      } else if (info.type === "sites") {
        state.lastSites = info.sitesData;
        state.lastSiteCount = info.totalSites;
        state.lastHashed = null;
        state.lastHeatmap = null;
        state.lastOwner = null;
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
    state.lastOwner = null;
    recordBandSnr(args.iters);
    await refreshViews("render");
    appendLog(`Done. Best PSNR: ${final.bestPsnr.toFixed(2)} dB | time: ${final.trainSeconds.toFixed(2)} s | active: ${final.activeSites ?? final.count}`);
    updateProgress(args.iters, args.iters,
      `done · best ${final.bestPsnr.toFixed(2)} dB · ${final.trainSeconds.toFixed(1)}s`);
    ui.saveBtn.disabled = false;
    completed = !signal.aborted;
  } catch (err) {
    appendLog(`Error: ${err.message}`);
    console.error(err);
  } finally {
    state.running = false;
    maybeEnableStart();
    ui.stopBtn.disabled = true;
    setDatasetEnabled(true);
    updateBatchBar();
    drawEditor(); // editing opens once the run has stopped
  }
  // The snap analysis waits for training to stop; looping runs skip it.
  if (!(completed && ui.loopToggle.checked)) drawSnapViews();
  // Loop mode: a run that finished on its own starts over from iteration 0
  // (with the form's current settings). Stop or an error ends the loop.
  if (completed && ui.loopToggle.checked) {
    appendLog("Loop: starting again from iteration 0.");
    setTimeout(startTraining, 0);
  }
}

// ---- Batch: many dataset images with one set of settings ----
// A pool of `jobs` trainers shares the GPU. Each run keeps only its log-tick
// scalars (PSNR, sites, LR, gradient stats) and up to `snaps` site snapshots
// plus its final sites; no images. Selecting a finished run reloads its target
// and puts its snapshots on a viewer-only trainer (Trainer.showSites), so
// every view drawn from the render or the sites works on them. Views that need
// training state (gradient heatmap, persistent error, merge edges) need
// "Re-run with full logging", which trains the image again: GPU float atomics
// make runs differ slightly, so it is not a replay.

// `n` distinct images drawn from [from, total), in random order. Partial
// Fisher–Yates: one O(dataset) allocation per batch, negligible next to training.
function randomBatchItems(from, total, n) {
  const pool = Array.from({ length: total - from }, (_, i) => from + i);
  for (let i = 0; i < n; i += 1) {
    const j = i + Math.floor(Math.random() * (pool.length - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, n);
}

// Trains one batch with the current settings. `items` replays a given image
// list (auto-tune compares settings on the same images); `quiet` skips opening
// the best run afterwards. Resolves to { median, items }, or null when the
// batch was stopped or not every run finished.
async function runBatch({ items: fixedItems = null, quiet = false } = {}) {
  if (state.running || !state.device) return null;
  ui.batchPopover.classList.add("hidden");
  let args;
  try {
    args = readFormArgs();
  } catch (err) {
    appendLog(`Settings error: ${err.message}`);
    return null;
  }
  const datasetKey = ui.datasetSelect.value;
  const dataset = DATASETS[datasetKey];
  if (!dataset?.count) return null;
  const int = (el, min, max) => Math.min(max, Math.max(min, Math.floor(Number(el.value)) || min));
  const from = int(ui.batchFrom, 0, dataset.count - 1);
  if (ui.batchFill.checked && ui.batchTiled.checked && canTileBatch(args)) await syncBatchFill();
  const random = ui.batchRandom.checked;
  const count = int(ui.batchCount, 1, dataset.count - from);
  const items = fixedItems ?? (random ? randomBatchItems(from, dataset.count, count)
                                      : Array.from({ length: count }, (_, i) => from + i));
  const n = items.length;
  const jobs = int(ui.batchJobs, 1, 256);
  const logFreq = int(ui.batchLogFreq, 1, args.iters);
  const snapCount = int(ui.batchSnaps, 1, 1000);
  // No previews: a batch only reads back at its log ticks.
  const runArgs = { ...args, logFreq, viewerFreq: args.iters + 1 };
  const tiled = ui.batchTiled.checked && canTileBatch(runArgs);
  batchVersion += 1;
  const batch = state.batch = {
    dataset: datasetKey, args: runArgs, from, random, jobs, snapCount, tiled, running: true,
    selected: null, snapIndex: 0, startMs: performance.now(), wallSec: 0,
    runs: items.map((item, i) => ({
      id: i, item, name: dataset.itemName(item), label: dataset.labelOf?.(item) ?? null, status: "queued",
      it: -1, curve: [], snaps: [], final: null, width: 0, height: 0,
    })),
  };

  state.running = true;
  state.abortCtrl = new AbortController();
  const signal = state.abortCtrl.signal;
  ui.startBtn.disabled = true;
  ui.batchBtn.disabled = true;
  syncTuneButtons();
  ui.singleModeBtn.disabled = true;
  ui.batchModeBtn.disabled = true;
  ui.stopBtn.disabled = false;
  ui.saveBtn.disabled = true;
  setDatasetEnabled(false);
  await gpuIdle(); // the analyses render with the old trainer
  state.trainer?.destroy();
  state.trainer = null;
  updateBatchBar();
  appendLog(`Batch: ${n} ${random ? "random " : ""}${dataset.label} image(s) ` +
            (random ? `from ${dataset.itemName(from)}–${dataset.itemName(dataset.count - 1)}`
                    : `${dataset.itemName(from)}–${dataset.itemName(from + n - 1)}`) + ", " +
            (tiled ? "one atlas per image size" : `${jobs} at once`) +
            `, log every ${logFreq}, ${snapCount} snapshot(s) per run.` +
            (ui.batchTiled.checked && !tiled ? " (Separate trainers: one atlas needs a fixed site count.)" : "") +
            (state.maskData || state.initSites ? " The mask and init sites are ignored." : ""));
  drawBatchViews();

  let next = 0;
  const worker = async () => {
    while (next < n && !signal.aborted) {
      const run = batch.runs[next++];
      await trainBatchRun(batch, run, dataset, signal);
      scheduleBatchDraw();
    }
  };
  try {
    if (tiled) await trainTiledBatch(batch, dataset, signal);
    else await Promise.all(Array.from({ length: Math.min(jobs, n) }, worker));
  } finally {
    batch.running = false;
    batch.wallSec = (performance.now() - batch.startMs) / 1000;
    for (const run of batch.runs) if (run.status === "queued") run.status = "skipped";
    batchVersion += 1;
    state.running = false;
    maybeEnableStart();
    ui.stopBtn.disabled = true;
    setDatasetEnabled(true);
  }
  const done = batchRunsByPsnr();
  const iters = done.length * runArgs.iters;
  const mean = done.reduce((s, r) => s + r.final.psnr, 0) / Math.max(1, done.length);
  appendLog(`Batch done: ${done.length}/${n} ok in ${batch.wallSec.toFixed(1)}s ` +
            `(${(iters / Math.max(batch.wallSec, 1e-6)).toFixed(0)} it/s over all runs)` +
            (done.length ? ` | final PSNR mean ${mean.toFixed(2)} dB, ` +
              `${done.at(-1).final.psnr.toFixed(2)}–${done[0].final.psnr.toFixed(2)} dB` : ""));
  if (state.tune?.progress) state.tune.progress.batches += 1;
  updateBatchProgress();
  updateBatchBar();
  drawBatchViews();
  if (done.length && !quiet) await selectBatchRun(done[0].id);
  if (signal.aborted || done.length < n) return null;
  return { median: medianOf(done.map((r) => r.final.psnr)), items };
}

function medianOf(values) {
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// Auto-tune one float setting on the current batch settings: from the current
// value, try × and ÷ the tune ratio (each a full batch on the same images),
// move to whichever raises the median final PSNR most, and repeat from there
// (values already measured aren't re-run). Halts when neither beats the
// current value, after tune max rounds, or on stop.
function tuneSettings() {
  const ratio = Number(document.getElementById("p-tuneRatio").value);
  const rounds = Math.floor(Number(document.getElementById("p-tuneRounds").value));
  if (!(ratio > 1)) throw new Error("tune ratio must be above 1");
  if (!(rounds >= 1)) throw new Error("tune max rounds must be at least 1");
  return { up: ratio, down: 1 / ratio, rounds };
}

// Float fields, plus integer ones marked data-tune="int" (tuned in whole
// steps). Optimize all leaves out the integer ones (withInts false): sites is
// the only one, and more sites almost always raise the PSNR, so it would just
// spend the bit budget.
function tunableInputs(withInts = true) {
  return [...ui.paramsPanel.querySelectorAll('input[type="number"]:not([data-no-tune])')]
    .filter((el) => (el.getAttribute("step") ?? "").includes(".") || el.hasAttribute("data-float") ||
                    (withInts && el.dataset.tune === "int"));
}

function setupAutoTune() {
  for (const input of tunableInputs()) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tune-btn";
    btn.textContent = "tune";
    btn.title = "Auto-tune: run a batch at this value × and ÷ the tune ratio (under Tuning) with the current batch " +
                "settings, keep whichever raises the median PSNR, repeat until neither helps. Click again to stop.";
    btn.addEventListener("click", () => {
      if (state.tune) {
        if (state.tune.input === input) state.tune.stop();
        return;
      }
      autoTune(input, btn);
    });
    input.after(btn);
  }
  const all = document.getElementById("optimizeAllBtn");
  all.addEventListener("click", () => {
    if (state.tune) {
      if (state.tune.btn === all) state.tune.stop();
      return;
    }
    optimizeAll(all);
  });
}

function syncTuneButtons() {
  const all = document.getElementById("optimizeAllBtn");
  for (const btn of [...ui.paramsPanel.querySelectorAll(".tune-btn"), all]) {
    const active = state.tune?.btn === btn;
    btn.textContent = active ? "stop" : btn === all ? "optimize all" : "tune";
    btn.classList.toggle("active", active);
    btn.disabled = state.tune ? !active : state.running || !state.device;
  }
}

function tuneName(input) {
  return input.closest(".row")?.querySelector("label")?.textContent ?? input.id;
}

const fmtDb = (db) => `${db >= 0 ? "+" : ""}${db.toFixed(3)} dB`;

// Hill-climbs one setting from `t.best` (median `t.bestMedian`, measured on
// `items`), updating `t` as it goes so the caller can keep the best value even
// when stopped. Resolves to false when stopped or a batch didn't finish.
async function climbSetting(input, t, tune, items, isStopped, prefix = "") {
  const name = tuneName(input);
  const seen = new Map([[t.best, t.bestMedian]]); // same images, so no need to re-run a value
  const trace = state.tuneTrace = { name, start: t.best, best: t.best, points: [{ v: t.best, median: t.bestMedian, round: 0 }] };
  refreshViews("tune");
  for (let round = 1; round <= tune.rounds; round += 1) {
    const center = t.best;
    const centerMedian = t.bestMedian;
    const tried = [];
    for (const f of [tune.up, tune.down]) {
      const v = tuneValue(input, center * f);
      if (v === center || tried.some((x) => x.v === v)) continue;
      let median = seen.get(v);
      if (median === undefined) {
        setTuneValue(input, v);
        setStatus(`${prefix}Auto-tune ${name}: round ${round}, trying ${v} (best ${center} @ ${centerMedian.toFixed(3)} dB)`);
        const r = await runBatch({ items, quiet: true });
        if (!r || isStopped()) return false;
        median = r.median;
        seen.set(v, median);
        trace.points.push({ v, median, round });
      }
      tried.push({ v, median });
      if (median > t.bestMedian) { t.best = v; t.bestMedian = median; }
      trace.best = t.best;
      refreshViews("tune");
    }
    appendLog(`Auto-tune ${name} round ${round}: ` +
              tried.map((x) => `${x.v} → ${x.median.toFixed(3)}`).join(", ") +
              ` vs ${center} → ${centerMedian.toFixed(3)} dB; ` +
              (t.best === center ? "neither helps, done." : `moving to ${t.best}.`));
    if (t.best === center) return true;
    if (round === tune.rounds) appendLog(`Auto-tune ${name}: stopped after ${tune.rounds} rounds.`);
  }
  return true;
}

// Auto-tunes every setting with a tune button, top to bottom, for `optimize
// cycles` passes, all on the images of one baseline batch. Each setting starts
// from the median the previous one ended on. Logs the median gain per setting
// and overall.
async function optimizeAll(btn) {
  if (state.running || state.tune || !state.device) return;
  let stopped = false;
  const isStopped = () => stopped;
  state.tune = { input: null, btn, stop: () => { stopped = true; state.abortCtrl?.abort(); } };
  syncTuneButtons();
  maybeEnableStart();
  const gains = []; // { cycle, name, from, to, gain }
  let baseline = null;
  let current = null;
  try {
    let tune;
    let cycles;
    try {
      tune = tuneSettings();
      cycles = Math.floor(Number(document.getElementById("p-tuneCycles").value));
      if (!(cycles >= 1)) throw new Error("optimize cycles must be at least 1");
    } catch (err) {
      appendLog(`Optimize all: ${err.message}.`);
      return;
    }
    const inputs = tunableInputs(false);
    state.tune.progress = { units: 1 + cycles * inputs.length, done: -1 };
    nextTuneUnit(true);
    setStatus("Optimize all: baseline");
    const base = await runBatch({ quiet: true });
    if (!base || stopped) return;
    const items = base.items;
    baseline = current = base.median;
    appendLog(`Optimize all: ${inputs.length} settings × ${cycles} cycle(s), baseline median ` +
              `${baseline.toFixed(3)} dB (${items.length} images)`);
    for (let cycle = 1; cycle <= cycles; cycle += 1) {
      for (const [i, input] of inputs.entries()) {
        nextTuneUnit();
        const name = tuneName(input);
        const start = Number(input.value);
        if (!Number.isFinite(start) || start === 0) {
          appendLog(`Optimize all: skipping ${name} (needs a non-zero value to scale).`);
          continue;
        }
        const t = { best: start, bestMedian: current };
        const prefix = `Optimize all ${cycle}/${cycles}, ${i + 1}/${inputs.length}: `;
        let ok;
        try {
          ok = await climbSetting(input, t, tune, items, isStopped, prefix);
        } finally {
          setTuneValue(input, t.best);
          gains.push({ cycle, name, from: start, to: t.best, gain: t.bestMedian - current });
          current = t.bestMedian;
        }
        appendLog(`Optimize all: ${name} ${start} → ${t.best}, ${fmtDb(gains.at(-1).gain)}`);
        if (!ok) return;
      }
    }
  } finally {
    if (baseline !== null) {
      const perSetting = new Map();
      for (const g of gains) perSetting.set(g.name, (perSetting.get(g.name) ?? 0) + g.gain);
      const changed = [...perSetting].filter(([, g]) => g > 0).sort((a, b) => b[1] - a[1]);
      appendLog(`Optimize all${stopped ? " stopped" : " done"}: median ${baseline.toFixed(3)} → ` +
                `${current.toFixed(3)} dB, ${fmtDb(current - baseline)} overall. ` +
                (changed.length ? "Gains: " + changed.map(([n, g]) => `${n} ${fmtDb(g)}`).join(", ") + "."
                                : "No setting helped."));
      console.table(gains.map((g) => ({ ...g, gain: Number(g.gain.toFixed(4)) })));
      console.log(`Optimize all: ${fmtDb(current - baseline)} overall (${baseline.toFixed(3)} → ${current.toFixed(3)} dB)`);
      setStatus(`Optimize all: ${fmtDb(current - baseline)}`);
    }
    state.tune = null;
    updateTuneProgress();
    syncTuneButtons();
    maybeEnableStart();
  }
}

function tuneValue(input, v) {
  const min = input.min === "" ? -Infinity : Number(input.min);
  const max = input.max === "" ? Infinity : Number(input.max);
  const c = Math.min(max, Math.max(min, v));
  return input.dataset.tune === "int" ? Math.max(min, Math.round(c)) : Number(c.toPrecision(4));
}

function setTuneValue(input, v) {
  input.value = String(v);
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

async function autoTune(input, btn) {
  if (state.running || state.tune || !state.device) return;
  const name = tuneName(input);
  let stopped = false;
  state.tune = { input, btn, stop: () => { stopped = true; state.abortCtrl?.abort(); } };
  syncTuneButtons();
  maybeEnableStart();
  let best = Number(input.value);
  try {
    let tune;
    try {
      tune = tuneSettings();
    } catch (err) {
      appendLog(`Auto-tune ${name}: ${err.message}.`);
      return;
    }
    if (!Number.isFinite(best) || best === 0) {
      appendLog(`Auto-tune ${name}: needs a non-zero value to scale.`);
      return;
    }
    state.tune.progress = { units: 2, done: -1 };
    nextTuneUnit(true);
    setStatus(`Auto-tune ${name}: baseline ${best}`);
    const base = await runBatch({ quiet: true });
    if (!base || stopped) return;
    appendLog(`Auto-tune ${name}: ${best} → median ${base.median.toFixed(3)} dB (${base.items.length} images)`);
    const t = { best, bestMedian: base.median };
    nextTuneUnit();
    try {
      await climbSetting(input, t, tune, base.items, () => stopped);
    } finally {
      best = t.best;
    }
  } finally {
    if (Number.isFinite(best)) setTuneValue(input, best);
    appendLog(`Auto-tune ${name}${stopped ? " stopped" : ""}: kept ${best}.`);
    setStatus(`Auto-tune ${name}: ${best}`);
    state.tune = null;
    updateTuneProgress();
    syncTuneButtons();
    maybeEnableStart();
  }
}

async function trainBatchRun(batch, run, dataset, signal) {
  const args = batch.args;
  run.status = "running";
  let trainer = null;
  try {
    const img = await loadImageToFloat32(await dataset.load(run.item), DEMO_MAX_IMAGE_SIDE);
    const { width, height } = img;
    run.width = width;
    run.height = height;
    // Snapshots at the first log tick past each of snapCount even steps; the
    // final sites are added after the run.
    const snapEvery = args.iters / batch.snapCount;
    let nextSnap = 0;
    trainer = new Trainer(state.device, state.shaders, args, {
      signal,
      onLog: (info) => {
        if (info.type !== "iter") return;
        run.it = info.it;
        if (Number.isFinite(info.psnr)) {
          run.curve.push({ it: info.it, psnr: info.psnr, sites: info.activeSites, lr: info.lrScale, grads: info.gradStats });
        }
        if (info.it >= nextSnap && info.it < args.iters - 1) {
          run.snaps.push({ it: info.it, sites: info.sitesData, count: info.totalSites, layer: info.layerIndex ?? null });
          nextSnap = (Math.floor(info.it / snapEvery) + 1) * snapEvery;
        }
        scheduleBatchDraw();
      },
    });
    await trainer.prepare({
      width, height,
      targetRgba32f: img.data,
      maskRgba32f: whiteMaskRgba32Float(width, height).rgba,
      maskSum: width * height,
      initSites: null,
      initSiteCount: args.startSites,
    });
    await trainer.run();
    if (signal.aborted) {
      run.status = "stopped";
      return;
    }
    const final = await trainer.finalize();
    run.snaps.push({ it: args.iters, sites: final.sites, count: final.count,
                     layer: final.globalSite >= 0 ? final.globalSite : null });
    run.final = {
      psnr: computePsnr(final.renderRgba, img.data, width, height),
      bestPsnr: final.bestPsnr,
      activeSites: final.activeSites ?? final.count,
      seconds: final.trainSeconds,
      growDoneIter: final.growDoneIter,
      growStop: final.growStop,
      growthGain: final.growthGain,
    };
    run.status = "done";
  } catch (err) {
    run.status = "error";
    run.error = err.message;
    appendLog(`Batch ${run.name}: ${err.message}`);
    console.error(err);
  } finally {
    trainer?.destroy();
  }
}

// Whether a batch with these args can share atlases (BatchTrainer): a fixed
// site count.
function canTileBatch(args) {
  return !countVaries(args, args.startSites) && !(args.turnover > 0);
}

// The batch as atlases of one image size each (batch.js), one after another.
// Each image keeps the same record a separate trainer gives it: log-tick
// curve, snapshots in its own coordinates, final sites.
async function trainTiledBatch(batch, dataset, signal) {
  const args = batch.args;
  const images = [];
  for (const run of batch.runs) {
    if (signal.aborted) return;
    const img = await loadImageToFloat32(await dataset.load(run.item), DEMO_MAX_IMAGE_SIDE);
    run.width = img.width;
    run.height = img.height;
    images.push(img);
  }
  const snapEvery = args.iters / batch.snapCount;
  for (const idx of planBatches(images, args.candDownscale)) {
    if (signal.aborted) break;
    const runs = idx.map((i) => batch.runs[i]);
    for (const run of runs) run.status = "running";
    let nextSnap = 0;
    const bt = new BatchTrainer(state.device, state.shaders, args, {
      signal,
      onLog: (info) => {
        if (info.type !== "batchIter") return;
        const snap = info.it >= nextSnap && info.it < args.iters - 1;
        if (snap) nextSnap = (Math.floor(info.it / snapEvery) + 1) * snapEvery;
        runs.forEach((run, b) => {
          run.it = info.it;
          run.curve.push({ it: info.it, psnr: info.psnrs[b], sites: tileActive(info.sitesData, bt.layout, b),
                           lr: info.lrScale });
          if (snap) {
            run.snaps.push({ it: info.it, sites: tileSites(info.sitesData, bt.layout, b),
                             count: bt.layout.sitesPerTile, layer: info.layer });
          }
        });
        scheduleBatchDraw();
      },
    });
    try {
      await bt.prepare(idx.map((i) => images[i]));
      const l = bt.layout;
      appendLog(`Batch atlas: ${idx.length} image(s) of ${l.tileW}×${l.tileH} as ${l.tilesX}×${l.tilesY} tiles, ` +
                `${l.width}×${l.height} px.`);
      await bt.run();
      if (signal.aborted) {
        for (const run of runs) run.status = "stopped";
        break;
      }
      const finals = await bt.finalize();
      finals.forEach((f, b) => {
        const run = runs[b];
        run.snaps.push({ it: args.iters, sites: f.sites, count: f.count, layer: f.layer });
        run.final = {
          psnr: f.psnr, bestPsnr: Math.max(f.psnr, ...run.curve.map((d) => d.psnr)), activeSites: f.activeSites,
          seconds: f.trainSeconds, growDoneIter: 0, growStop: null, growthGain: [],
        };
        run.status = "done";
      });
    } catch (err) {
      for (const run of runs) { run.status = "error"; run.error = err.message; }
      appendLog(`Batch atlas: ${err.message}`);
      console.error(err);
    } finally {
      bt.destroy();
    }
    scheduleBatchDraw();
  }
}

// Finished runs, best final PSNR first.
function batchRunsByPsnr() {
  return (state.batch?.runs ?? []).filter((r) => r.status === "done").sort((a, b) => b.final.psnr - a.final.psnr);
}

// Live drawing is throttled: every log tick of every run asks for a redraw.
let batchDrawTimer = null;
function scheduleBatchDraw() {
  batchVersion += 1;
  if (batchDrawTimer) return;
  batchDrawTimer = setTimeout(() => {
    batchDrawTimer = null;
    drawBatchViews();
    updateBatchProgress();
  }, 250);
}

// Iterations per second summed over every run of the batch so far.
function batchItersDone(b) {
  return b.runs.reduce((s, r) => s + (r.status === "done" ? b.args.iters : Math.max(0, r.it + 1)), 0);
}

function batchItersPerSec(b) {
  const elapsed = b.running ? (performance.now() - b.startMs) / 1000 : b.wallSec;
  return { elapsed, rate: batchItersDone(b) / Math.max(elapsed, 1e-6) };
}

// The green bar across the top while auto-tune or optimize all runs. Progress
// is in units (the baseline batch, then each setting tuned); a setting's batch
// count isn't known ahead, so after k batches it shows k / (k + 2) of its unit.
function updateTuneProgress() {
  const p = state.tune?.progress;
  ui.tuneProgress.classList.toggle("hidden", !p);
  if (!p) return;
  const b = state.batch;
  const k = p.batches + (b?.running ? Math.min(1, batchItersDone(b) / Math.max(1, b.runs.length * b.args.iters)) : 0);
  const unit = p.baseline ? Math.min(1, k) : k / (k + 2);
  ui.tuneProgressFill.style.width = `${(100 * Math.min(1, (p.done + unit) / p.units)).toFixed(2)}%`;
}

// Moves the tune bar on to the next unit (`baseline`: it is the one baseline batch).
function nextTuneUnit(baseline = false) {
  const p = state.tune.progress;
  p.done += 1;
  p.batches = 0;
  p.baseline = baseline;
  updateTuneProgress();
}

function updateBatchProgress() {
  const b = state.batch;
  if (!b) return;
  const total = b.runs.length;
  const finished = b.runs.filter((r) => r.status !== "queued" && r.status !== "running").length;
  const { elapsed, rate } = batchItersPerSec(b);
  updateProgress(finished, total, `batch ${finished}/${total} runs · ` +
    `${rate.toFixed(0)} it/s over all runs · ${elapsed.toFixed(1)}s`);
  updateBatchSummary();
  updateTuneProgress();
}

// "123k iter/s · done/total done · mean · median" in the batch bar.
function updateBatchSummary() {
  const b = state.batch;
  if (!b) return;
  const { rate } = batchItersPerSec(b);
  const speed = rate >= 1000 ? `${(rate / 1000).toFixed(0)}k` : rate.toFixed(0);
  const psnrs = batchRunsByPsnr().map((r) => r.final.psnr);
  ui.batchSummary.textContent = `${speed} iter/s · ${psnrs.length}/${b.runs.length} done` + (psnrs.length
    ? ` · mean ${(psnrs.reduce((s, v) => s + v, 0) / psnrs.length).toFixed(2)} dB · median ` +
      `${psnrs[Math.floor(psnrs.length / 2)].toFixed(2)} dB` : "");
}

function updateBatchBar() {
  const b = state.batch;
  ui.batchBar.classList.toggle("hidden", !b);
  if (!b) return;
  const done = batchRunsByPsnr();
  const select = ui.batchRunSelect;
  // Thousands of options: rebuilt only when the batch's runs changed.
  const listKey = `${batchVersion}|${b.running}`;
  if (select.dataset.key !== listKey) {
    select.dataset.key = listKey;
    select.replaceChildren(new Option(b.running ? "running…" : done.length ? "pick a run…" : "no finished runs", ""));
    for (const r of done) select.add(new Option(`${r.name} · ${r.final.psnr.toFixed(2)} dB`, String(r.id)));
  }
  select.value = b.selected != null ? String(b.selected) : "";
  const idle = !b.running && !state.running;
  select.disabled = !idle || !done.length;
  ui.batchPrev.disabled = ui.batchNext.disabled = select.disabled;
  const run = b.selected != null ? b.runs[b.selected] : null;
  ui.batchSnap.disabled = !idle || !run;
  ui.batchSnap.max = String(Math.max(0, (run?.snaps.length ?? 1) - 1));
  ui.batchSnap.value = String(b.snapIndex);
  ui.batchSnapLabel.textContent = run ? `iter ${run.snaps[b.snapIndex].it}` : "—";
  ui.batchRerun.disabled = !idle || !run;
  ui.batchExport.disabled = b.running || !b.runs.some((r) => r.curve.length);
  updateBatchSummary();
}

function clickBatchRun(id) {
  const b = state.batch;
  if (!b) return;
  if (b.running) {
    setStatus("Runs can be opened once the batch has finished");
    return;
  }
  selectBatchRun(id);
}

// Loads a finished run: its target, its log-tick histories, and a viewer-only
// trainer holding its snapshots (showBatchSnapshot), at the final one.
async function selectBatchRun(id) {
  const b = state.batch;
  const run = b?.runs[id];
  if (!run || run.status !== "done" || b.running || state.running) return;
  const token = ++state.targetToken;
  b.selected = id;
  b.snapIndex = run.snaps.length - 1;
  updateBatchBar();
  drawBatchViews();
  setStatus(`Loading ${run.name}…`);
  const img = await loadImageToFloat32(await DATASETS[b.dataset].load(run.item), DEMO_MAX_IMAGE_SIDE);
  if (token !== state.targetToken) return;
  const { width, height } = img;
  if (ui.datasetSelect.value === b.dataset) ui.itemSelect.value = String(run.item);
  ui.imageInput.value = "";
  ui.imageLabel.textContent = run.name;
  state.targetImage = img;
  state.maskData = null;
  ui.maskLabel.textContent = "none";
  invalidateAnalysis();
  syncBppSites();

  const args = b.args;
  const varies = countVaries(args, args.startSites);
  state.psnrHistory = run.curve.map((d) => ({ it: d.it, psnr: d.psnr, sites: d.sites }));
  state.psnrIters = args.iters;
  state.lrHistory = run.curve.filter((d) => Number.isFinite(d.lr)).map((d) => ({ it: d.it, lr: d.lr }));
  state.gradHistory = run.curve.filter((d) => d.grads).map((d) => ({ it: d.it, stats: d.grads }));
  state.gradSnapshot = null;
  state.bandHistory = [];
  state.growthGain = run.final.growthGain ?? [];
  state.mergeEdges = null;
  state.snapAnalysis = null;
  state.runSiteMode = varies ? "grow" : "fixed";
  state.lastGrowth = null;
  state.lrPlan = { iters: args.iters, lrMult: args.lrMult, kind: args.lrSchedule,
                   growDoneIter: run.final.growDoneIter ?? (varies ? countHorizon(args) : 0) };
  state.bppRuns = [{
    sites: run.final.activeSites, bpp: computeBpp(run.final.activeSites, width, height), colorSpace: args.colorSpace,
    points: run.curve.map((d) => ({ it: d.it, psnr: d.psnr, sites: d.sites, bpp: computeBpp(d.sites, width, height) })),
  }];
  for (const view of Object.values(VIEWS)) if (view.on.includes("run")) view.hoverKey = null;

  // A fixed-count trainer with room for the largest snapshot; showSites puts
  // each snapshot (and its layer slot) on it.
  await state.snapAnalysis?.promise;
  state.trainer?.destroy();
  state.trainer = null;
  const slots = Math.max(...run.snaps.map((s) => s.count));
  const trainer = new Trainer(state.device, state.shaders, {
    ...args, layer: "off", startSites: slots, endSites: slots, countSchedule: "linear", turnover: 0,
  });
  await trainer.prepare({
    width, height,
    targetRgba32f: img.data,
    maskRgba32f: whiteMaskRgba32Float(width, height).rgba,
    maskSum: width * height,
    initSites: null,
    initSiteCount: slots,
  });
  if (token !== state.targetToken) {
    trainer.destroy();
    return;
  }
  state.trainer = trainer;
  blankRenderViews(width, height);
  await refreshViews("target", "log", "run", "growth", "merge");
  updateCanvasLayout(width, height);
  await showBatchSnapshot(b.snapIndex);
  ui.saveBtn.disabled = false;
  ui.startBtn.disabled = false;
  setStatus(`${run.name}: final ${run.final.psnr.toFixed(2)} dB, ${run.final.activeSites} sites, ` +
            `${run.final.seconds.toFixed(1)}s ${b.tiled ? "for its atlas" : "in the batch"}`);
}

// Shows snapshot k of the selected run on the viewer trainer. Scrubbing
// faster than renders finish keeps only the latest request.
let batchSnapBusy = null;
async function showBatchSnapshot(k) {
  const b = state.batch;
  const run = b?.runs[b.selected];
  const trainer = state.trainer;
  if (!run || !trainer || state.running) return;
  b.snapIndex = k;
  ui.batchSnapLabel.textContent = `iter ${run.snaps[k].it}`;
  if (batchSnapBusy) return;
  batchSnapBusy = true;
  try {
    let shown;
    do {
      shown = b.snapIndex;
      const snap = run.snaps[shown];
      await gpuIdle(); // an analysis may still be rendering the last snapshot
      if (state.trainer !== trainer) return;
      trainer.showSites(snap.sites, snap.count, snap.layer);
      const rgba = await trainer.renderView("render");
      if (state.trainer !== trainer) return;
      state.lastRender = { rgba, width: run.width, height: run.height };
      state.lastSites = snap.sites;
      state.lastSiteCount = snap.count;
      state.lastHashed = null;
      state.lastHeatmap = null;
      state.lastOwner = null;
      let active = 0;
      for (let i = 0; i < snap.count; i += 1) if (snap.sites[i * SITE_FLOATS] >= 0) active += 1;
      state.lastActiveSites = active;
      state.lastPsnr = computePsnr(rgba, state.targetImage.data, run.width, run.height);
      ui.metricRow.innerHTML =
        `<span>snapshot <strong>iter ${snap.it}/${b.args.iters}</strong></span>` +
        `<span>PSNR <strong>${state.lastPsnr.toFixed(2)} dB</strong></span>` +
        `<span>active <strong>${active}</strong>/${snap.count}</span>`;
      await refreshViews("render");
      drawCentroidOverlay();
    } while (shown !== b.snapIndex);
    drawEditor();
    drawSnapViews();
  } finally {
    batchSnapBusy = null;
  }
}

function stepBatchRun(delta) {
  const done = batchRunsByPsnr();
  if (!done.length) return;
  const i = done.findIndex((r) => r.id === state.batch.selected);
  const j = i < 0 ? 0 : (i + delta + done.length) % done.length;
  selectBatchRun(done[j].id);
}

// Re-trains the selected run's image with the form's settings and every view
// live (the batch's settings unless the form changed since).
function rerunBatchRun() {
  const b = state.batch;
  const run = b?.runs[b.selected];
  if (!run || state.running) return;
  let args;
  try { args = readFormArgs(); } catch { args = null; }
  const changed = args && Object.keys(b.args).filter((k) => k !== "logFreq" && k !== "viewerFreq" &&
    JSON.stringify(args[k]) !== JSON.stringify(b.args[k]));
  if (changed?.length) appendLog(`Re-run uses the form's settings, which differ from the batch's in: ${changed.join(", ")}.`);
  startTraining();
}

function exportBatch() {
  const b = state.batch;
  if (!b) return;
  const lines = b.runs.filter((r) => r.curve.length || r.final).map((r) => JSON.stringify({
    name: r.name, dataset: b.dataset, item: r.item, status: r.status, width: r.width, height: r.height,
    finalPsnr: r.final?.psnr ?? null, bestPsnr: r.final?.bestPsnr ?? null, sites: r.final?.activeSites ?? null,
    bpp: r.final ? computeBpp(r.final.activeSites, r.width, r.height) : null, bitsPerSite: BITS_PER_SITE,
    trainSeconds: r.final?.seconds ?? null, growStop: r.final?.growStop ?? null, args: b.args,
    curve: r.curve.map(({ it, psnr, sites }) => ({ it, psnr, sites })),
  }));
  const tag = b.random ? `random${b.runs.length}` : `${b.from}-${b.from + b.runs.length - 1}`;
  triggerDownload(new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" }),
    `batch_${b.dataset}_${tag}.jsonl`);
}

// Images one atlas holds (atlasCapacity) for the size of the batch's first
// image, from there to the end of the dataset; null if it can't be loaded.
async function batchFillCount(dataset, from, candDownscale) {
  try {
    const bitmap = await createImageBitmap(await dataset.load(from));
    let { width, height } = bitmap;
    bitmap.close();
    const scale = Math.min(1, DEMO_MAX_IMAGE_SIDE / Math.max(width, height));
    width = Math.max(1, Math.floor(width * scale));
    height = Math.max(1, Math.floor(height * scale));
    const { tileW, tileH } = tileSize([{ width, height }], candDownscale);
    return Math.min(dataset.count - from, atlasCapacity(tileW, tileH));
  } catch (err) {
    console.error(err);
    return null;
  }
}

// "fill GPU": the images field shows (and is locked to) one atlas's worth.
// Only with one atlas (its checkbox, and a fixed site count).
async function syncBatchFill() {
  const dataset = currentDataset();
  let args = null;
  try { args = readFormArgs(); } catch {}
  const can = Boolean(args && dataset?.count && ui.batchTiled.checked && canTileBatch(args));
  ui.batchFill.disabled = !can;
  const fill = can && ui.batchFill.checked;
  ui.batchCount.disabled = fill;
  if (!fill) return;
  const from = Math.min(dataset.count - 1, Math.max(0, Math.floor(Number(ui.batchFrom.value)) || 0));
  const n = await batchFillCount(dataset, from, args.candDownscale);
  if (n && ui.batchFill.checked) ui.batchCount.value = String(n);
}

function updateBatchNote() {
  const dataset = currentDataset();
  if (!dataset?.count) return;
  ui.batchFrom.max = String(dataset.count - 1);
  ui.batchCount.max = String(dataset.count);
  let tile = false;
  try { tile = canTileBatch(readFormArgs()); } catch {}
  ui.batchNote.textContent = `${dataset.label}: images 0–${dataset.count - 1}. ` +
    (ui.batchRandom.checked
      ? `Random: each batch draws its images at random from first image to ${dataset.count - 1}. `
      : "") +
    "Every run uses the settings in Advanced, without the mask or init sites. " + (tile
      ? "The site count is fixed, so one atlas trains them all at once (at once then doesn't apply)."
      : "The site count changes, so each image gets its own trainer.");
}

function toggleBatchMenu() {
  syncBatchFill();
  updateBatchNote();
  ui.batchPopover.classList.toggle("hidden");
}

// ---- Batch plots ----

const BATCH_RUNNING_COLOR = SERIES_COLORS[0];
const BATCH_AVG_COLOR = SERIES_COLORS[5];
const batchHoverRun = () => VIEWS.batchPsnr.hoverKey ?? VIEWS.batchHist.hoverKey ??
  VIEWS.batchDigits.hoverKey ?? null;

// All batch cards share hover and selection, so each draws all three.
function drawBatchViews() {
  if (VIEWS.batchPsnr.canvas) drawBatchPsnr(VIEWS.batchPsnr);
  if (VIEWS.batchHist.canvas) drawBatchHist(VIEWS.batchHist);
  if (VIEWS.batchDigits.canvas) drawBatchDigits(VIEWS.batchDigits);
}

// Bumped whenever a batch's runs change (scheduleBatchDraw, runBatch), so the
// plots' cached layers (batchLayer) know to redraw.
let batchVersion = 0;

// The batch plots' expensive part (axes and thousands of curves or cells) is
// drawn by `draw` only when `key` or the canvas size changes, and kept as an
// offscreen copy; hovering copies it back and draws just the highlight on top.
// Redrawing thousands of strokes per mouse move made hovering lag (Firefox
// especially). Returns what `draw` returned (the plot's geometry).
function batchLayer(view, p, key, draw) {
  const c = view.canvas;
  const cache = view.batchLayer;
  if (cache && cache.key === key && cache.canvas.width === c.width && cache.canvas.height === c.height) {
    p.ctx.save();
    p.ctx.setTransform(1, 0, 0, 1, 0, 0);
    p.ctx.drawImage(cache.canvas, 0, 0);
    p.ctx.restore();
    return cache.info;
  }
  const info = draw();
  const copy = cache?.canvas ?? document.createElement("canvas");
  copy.width = c.width;
  copy.height = c.height;
  copy.getContext("2d").drawImage(c, 0, 0);
  view.batchLayer = { key, canvas: copy, info };
  return info;
}

function batchRunTip(run) {
  const cells = [run.name];
  if (run.final) cells.push(`final ${run.final.psnr.toFixed(2)} dB`, `${run.final.activeSites} sites`);
  else if (run.curve.length) cells.push(`${run.curve.at(-1).psnr.toFixed(2)} dB @ ${run.curve.at(-1).it}`, run.status);
  else cells.push(run.status);
  return { rows: [{ cells }], footer: state.batch.running ? undefined : "click to open" };
}

// Curves per stroke in the faint layer: one stroke each is thousands of
// strokes; one for all would paint overlaps once and lose the band's density.
const BATCH_CURVES_PER_STROKE = 16;

// Mean PSNR over the runs at each log tick, in iteration order. A run that has
// not logged a tick yet (or stopped before it) does not count there.
function batchAverageCurve(runs) {
  const byIt = new Map();
  for (const r of runs) {
    for (const d of r.curve) {
      if (!Number.isFinite(d.psnr)) continue;
      let s = byIt.get(d.it);
      if (!s) byIt.set(d.it, s = { sum: 0, n: 0 });
      s.sum += d.psnr;
      s.n += 1;
    }
  }
  return [...byIt].sort((a, b) => a[0] - b[0]).map(([it, s]) => ({ it, psnr: s.sum / s.n }));
}

// A dB value beside an average-curve point, in the line's color and kept
// inside the plot box. The canvas-colored halo keeps it legible over the band.
function drawBatchValueTag(p, g, d, align) {
  const { ctx, col } = p;
  const text = `${d.psnr.toFixed(2)} dB`;
  ctx.font = PLOT_FONT;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  const width = ctx.measureText(text).width;
  const lo = align === "left" ? g.left + 3 : align === "right" ? g.left + width + 3 : g.left + width / 2 + 3;
  const hi = align === "left" ? g.right - width - 3 : align === "right" ? g.right - 3 : g.right - width / 2 - 3;
  const px = g.x(d.it), py = g.y(d.psnr);
  const tx = Math.max(lo, Math.min(hi, px + (align === "left" ? 8 : align === "right" ? -8 : 0)));
  const ty = Math.max(g.top + 8, Math.min(g.bottom - 8, py - 12 < g.top + 8 ? py + 13 : py - 12));
  ctx.save();
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.strokeStyle = col.surface;
  ctx.strokeText(text, tx, ty);
  ctx.fillStyle = BATCH_AVG_COLOR;
  ctx.fillText(text, tx, ty);
  ctx.restore();
}

function drawBatchPsnr(view) {
  const p = beginPlot(view);
  if (!p) return;
  const b = state.batch;
  const runs = b?.runs ?? [];
  const g = view.geom = batchLayer(view, p, String(batchVersion), () => {
    const avg = batchAverageCurve(runs);
    let lo = Infinity, hi = -Infinity;
    for (const r of runs) for (const d of r.curve) { lo = Math.min(lo, d.psnr); hi = Math.max(hi, d.psnr); }
    const g = plotAxes(plotBox(p), { min: 0, max: Math.max(1, b?.args.iters ?? 1) }, linRange(lo, hi, [0, 40]));
    drawAxes(p, g, { xTitle: "iteration", yTitle: "PSNR (dB)", xFmt: fmtIter });
    if (!Number.isFinite(lo)) {
      drawEmpty(p, g, b?.running ? "Waiting for the first log ticks…" : "Run a batch (Batch… in the top bar)");
      return { ...g, empty: true };
    }
    // Faint lines so thousands of overlapping runs still read as a band.
    const { ctx } = p;
    ctx.save();
    ctx.globalAlpha = Math.max(0.12, Math.min(0.7, 4 / Math.sqrt(runs.length)));
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";
    const groups = [[p.col.dim, runs.filter((r) => r.status !== "running")],
                    [BATCH_RUNNING_COLOR, runs.filter((r) => r.status === "running")]];
    for (const [color, list] of groups) {
      ctx.strokeStyle = color;
      for (let s = 0; s < list.length; s += BATCH_CURVES_PER_STROKE) {
        ctx.beginPath();
        for (const r of list.slice(s, s + BATCH_CURVES_PER_STROKE)) {
          r.curve.forEach((d, i) => (i ? ctx.lineTo(g.x(d.it), g.y(d.psnr)) : ctx.moveTo(g.x(d.it), g.y(d.psnr))));
        }
        ctx.stroke();
      }
    }
    ctx.restore();
    // The batch mean over the run band, with its first, middle and last values.
    if (avg.length) {
      drawLine(p, g, avg.map((d) => [d.it, d.psnr]), BATCH_AVG_COLOR, { width: 2.5 });
      const picks = [...new Set([0, Math.floor((avg.length - 1) / 2), avg.length - 1])];
      for (const i of picks) {
        const d = avg[i];
        drawMarker(p, g, d.it, d.psnr, BATCH_AVG_COLOR, 3.5);
        const align = avg.length === 1 ? "center" : i === 0 ? "left" : i === avg.length - 1 ? "right" : "center";
        drawBatchValueTag(p, g, d, align);
      }
    }
    return g;
  });
  if (g.empty) return;
  const pts = (r) => r.curve.map((d) => [d.it, d.psnr]);
  const hover = batchHoverRun();
  const sel = b.selected != null ? runs[b.selected] : null;
  if (sel) drawLine(p, g, pts(sel), PSNR_LINE_COLOR, { width: 2 });
  const hov = hover != null ? runs[hover] : null;
  if (hov && hov !== sel) drawLine(p, g, pts(hov), p.col.text, { width: 2 });
  if (hov?.curve.length) {
    const last = hov.curve.at(-1);
    drawMarker(p, g, last.it, last.psnr, hov === sel ? PSNR_LINE_COLOR : p.col.text);
    drawTooltip(p, g, g.x(last.it), g.y(last.psnr), batchRunTip(hov));
  }
}

// The run whose curve passes nearest the cursor (within 10 px). Only each
// curve's segments around the cursor's iteration can be that close.
function hitBatchPsnr(g, mx, my) {
  if (g.empty) return null;
  const it = g.xr.min + ((mx - g.left) / (g.right - g.left)) * (g.xr.max - g.xr.min);
  let best = null, bestD = 100;
  for (const r of state.batch?.runs ?? []) {
    const c = r.curve;
    if (!c.length) continue;
    let lo = 0, hi = c.length - 1; // first point at or past `it`
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (c[m].it < it) lo = m + 1; else hi = m;
    }
    for (let k = Math.max(0, lo - 2); k <= Math.min(c.length - 1, lo + 1); k += 1) {
      const x = g.x(c[k].it), y = g.y(c[k].psnr);
      let dd = (x - mx) ** 2 + (y - my) ** 2;
      if (k > 0) {
        const px = g.x(c[k - 1].it), py = g.y(c[k - 1].psnr);
        const vx = x - px, vy = y - py;
        const t = Math.max(0, Math.min(1, ((mx - px) * vx + (my - py) * vy) / (vx * vx + vy * vy || 1)));
        dd = (px + t * vx - mx) ** 2 + (py + t * vy - my) ** 2;
      }
      if (dd < bestD) { bestD = dd; best = r.id; }
    }
  }
  return best;
}

// Bin width giving about 20 bins over [lo, hi], on a 1-2-2.5-5 step.
function batchBinStep(lo, hi) {
  const raw = Math.max(hi - lo, 0.5) / 20;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
}

function drawBatchHist(view) {
  const p = beginPlot(view);
  if (!p) return;
  const b = state.batch;
  const g = view.geom = batchLayer(view, p, String(batchVersion), () => {
    const done = batchRunsByPsnr().reverse(); // ascending, so each bin stacks low to high
    if (!done.length) {
      const g = plotAxes(plotBox(p), { min: 0, max: 40 }, { min: 0, max: 1 });
      drawAxes(p, g, { xTitle: "final PSNR (dB)", yTitle: "runs" });
      drawEmpty(p, g, b?.running ? "Waiting for the first finished run…" : "Run a batch (Batch… in the top bar)");
      return { ...g, empty: true };
    }
    const lo = done[0].final.psnr, hi = done.at(-1).final.psnr;
    const step = batchBinStep(lo, hi);
    const b0 = Math.floor(lo / step) * step;
    const nb = Math.max(1, Math.floor((hi - b0) / step) + 1);
    const bins = Array.from({ length: nb }, () => []);
    for (const r of done) bins[Math.min(nb - 1, Math.floor((r.final.psnr - b0) / step))].push(r);
    const maxStack = Math.max(...bins.map((bin) => bin.length));
    const g = plotAxes(plotBox(p), { min: b0, max: b0 + nb * step }, { min: 0, max: maxStack });
    drawAxes(p, g, { xTitle: "final PSNR (dB)", yTitle: "runs", yFmt: (v) => (Number.isInteger(v) ? String(v) : ""),
                     xFmt: (v) => String(Number(v.toFixed(2))) });
    const geom = { ...g, b0, step, bins, cellH: (g.bottom - g.top) / maxStack, binW: g.x(b0 + step) - g.x(b0) };
    p.ctx.fillStyle = SERIES_COLORS[0];
    bins.forEach((bin, k) => bin.forEach((r, j) => batchCell(p, geom, k, j)));
    return geom;
  });
  if (g.empty) return;
  // The selected and hovered runs on top of the cached cells.
  const at = (run) => {
    if (!run?.final) return null;
    const k = Math.min(g.bins.length - 1, Math.floor((run.final.psnr - g.b0) / g.step));
    const j = g.bins[k].indexOf(run);
    return j >= 0 ? [k, j] : null;
  };
  const sel = b.selected != null ? b.runs[b.selected] : null;
  const selAt = at(sel);
  if (selAt) {
    p.ctx.fillStyle = PSNR_LINE_COLOR;
    batchCell(p, g, ...selAt);
  }
  const hov = batchHoverRun() != null ? b.runs[batchHoverRun()] : null;
  const hovAt = at(hov);
  if (hovAt) {
    p.ctx.fillStyle = hov === sel ? PSNR_LINE_COLOR : p.col.text;
    batchCell(p, g, ...hovAt);
    drawTooltip(p, g, g.x(g.b0 + (hovAt[0] + 0.5) * g.step), g.bottom - (hovAt[1] + 1) * g.cellH, batchRunTip(hov));
  }
}

// Fills run j (from the bottom) of bin k of the histogram.
function batchCell(p, g, k, j) {
  p.ctx.fillRect(g.x(g.b0 + k * g.step) + 1, g.bottom - (j + 1) * g.cellH,
                 Math.max(1, g.binW - 2), Math.max(1, g.cellH - (g.cellH >= 4 ? 1 : 0)));
}

function hitBatchHist(g, mx, my) {
  if (!g.bins) return null;
  const v = g.xr.min + ((mx - g.left) / (g.right - g.left)) * (g.xr.max - g.xr.min);
  const k = Math.floor((v - g.b0) / g.step);
  const j = Math.floor((g.bottom - my) / g.cellH);
  return g.bins[k]?.[j]?.id ?? null;
}

// Final PSNR per class (MNIST digit, CIFAR-10 class): one dot per finished run, with a box for the
// quartiles and whiskers at 1.5× the interquartile range. Nothing is drawn
// while the batch runs, so the distributions never shift under the reader.
function batchDigitGroups(runs, classCount) {
  const groups = Array.from({ length: classCount }, () => []);
  for (const r of runs) {
    if (r.status !== "done" || r.final == null || r.label == null) continue;
    if (r.label >= 0 && r.label < groups.length) groups[r.label].push(r);
  }
  return groups;
}

// Quantile of an ascending array, interpolating between the nearest ranks.
function quantile(sorted, p) {
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h), hi = Math.ceil(h);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
}

// Stable horizontal jitter in [-0.5, 0.5]: a run keeps its dot while the
// batch grows, and the hover marker lands on the same dot.
function batchDigitJitter(run) {
  const x = Math.sin(run.id * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x) - 0.5;
}

function drawBatchDigits(view) {
  const p = beginPlot(view);
  if (!p) return;
  const b = state.batch;
  const g = view.geom = batchLayer(view, p, String(batchVersion), () => {
    const runs = b?.runs ?? [];
    let lo = Infinity, hi = -Infinity;
    for (const r of runs) {
      if (r.status !== "done" || r.final == null || r.label == null) continue;
      lo = Math.min(lo, r.final.psnr);
      hi = Math.max(hi, r.final.psnr);
    }
    // Classes of the batch's dataset (MNIST's digits before any batch).
    const dataset = DATASETS[b?.dataset] ?? DATASETS.mnist;
    const classes = dataset.classNames ?? MNIST_DIGITS;
    const g = plotAxes(plotBox(p), { min: -0.6, max: classes.length - 0.4 }, linRange(lo, hi, [0, 40]));
    drawAxes(p, g, { xTitle: dataset.classTitle ?? "class", yTitle: "final PSNR (dB)", xLabels: false });
    // Classes are categorical: label every one, whatever the tick spacing.
    const { ctx } = p;
    ctx.font = PLOT_FONT;
    ctx.fillStyle = p.col.soft;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let d = 0; d < classes.length; d += 1) ctx.fillText(classes[d], g.x(d), g.bottom + 6);
    if (b?.running) {
      drawEmpty(p, g, "Available when the batch finishes");
      return { ...g, empty: true };
    }
    if (b && !dataset.classNames) {
      drawEmpty(p, g, "Only labelled datasets (MNIST, CIFAR-10) have classes");
      return { ...g, empty: true };
    }
    const groups = batchDigitGroups(runs, classes.length);
    if (!groups.some((group) => group.length)) {
      drawEmpty(p, g, b ? "No finished runs" : "Run a batch (Batch… in the top bar)");
      return { ...g, empty: true };
    }
    const unit = g.x(1) - g.x(0);
    const half = Math.max(3, Math.min(unit * 0.3, 26)); // box half-width, px
    const points = [];
    for (let d = 0; d < groups.length; d += 1) {
      const group = groups[d];
      if (!group.length) continue;
      const sorted = group.map((r) => r.final.psnr).sort((a, z) => a - z);
      const q1 = quantile(sorted, 0.25);
      const med = quantile(sorted, 0.5);
      const q3 = quantile(sorted, 0.75);
      const iqr = q3 - q1;
      const wLo = sorted.find((v) => v >= q1 - 1.5 * iqr) ?? sorted[0];
      const wHi = [...sorted].reverse().find((v) => v <= q3 + 1.5 * iqr) ?? sorted.at(-1);
      const cx = g.x(d);
      const top = g.y(q3), bot = g.y(q1);
      ctx.save();
      // Whiskers behind the box, then the box and its median.
      ctx.strokeStyle = p.col.dim;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(cx, g.y(wHi));
      ctx.lineTo(cx, g.y(wLo));
      for (const y of [g.y(wHi), g.y(wLo)]) {
        ctx.moveTo(cx - half * 0.45, y);
        ctx.lineTo(cx + half * 0.45, y);
      }
      ctx.stroke();
      ctx.fillStyle = SERIES_COLORS[0];
      ctx.globalAlpha = 0.16;
      ctx.fillRect(cx - half, top, half * 2, Math.max(1, bot - top));
      ctx.globalAlpha = 1;
      ctx.strokeStyle = SERIES_COLORS[0];
      ctx.strokeRect(cx - half, top, half * 2, Math.max(1, bot - top));
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - half, g.y(med));
      ctx.lineTo(cx + half, g.y(med));
      ctx.stroke();
      // One dot per run, jittered inside the box width.
      ctx.fillStyle = p.col.soft;
      ctx.globalAlpha = 0.75;
      for (const r of group) {
        const x = d + (batchDigitJitter(r) * half * 1.2) / unit;
        points.push({ run: r.id, x, psnr: r.final.psnr });
        ctx.beginPath();
        ctx.arc(g.x(x), g.y(r.final.psnr), 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
    return { ...g, points };
  });
  if (g.empty || !b) return;
  const find = (run) => (run ? g.points.find((pt) => pt.run === run.id) : null);
  const sel = b.selected != null ? b.runs[b.selected] : null;
  const selPt = find(sel);
  if (selPt) drawMarker(p, g, selPt.x, selPt.psnr, PSNR_LINE_COLOR, 4);
  const hov = batchHoverRun() != null ? b.runs[batchHoverRun()] : null;
  const hovPt = find(hov);
  if (hovPt) {
    if (hovPt !== selPt) drawMarker(p, g, hovPt.x, hovPt.psnr, p.col.text, 4);
    drawTooltip(p, g, g.x(hovPt.x), g.y(hovPt.psnr), batchRunTip(hov));
  }
}

// The run dot nearest the cursor, within 10 px.
function hitBatchDigits(g, mx, my) {
  if (!g.points) return null;
  let best = null, bestD = 100;
  for (const pt of g.points) {
    const dd = (g.x(pt.x) - mx) ** 2 + (g.y(pt.psnr) - my) ** 2;
    if (dd < bestD) { bestD = dd; best = pt.run; }
  }
  return best;
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

function clearOverlay(view = VIEWS.preview) {
  const overlay = view.overlayCanvas;
  if (!overlay) return;
  const ctx = overlay.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

// The overlay canvas sits next to its view's image but is absolutely
// positioned, so we need to match its drawing buffer and CSS box to the
// image's actual displayed rect (which is aspect-fit inside its parent).
function positionOverlay(view = VIEWS.preview) {
  const { canvas, overlayCanvas: overlay } = view;
  if (!canvas || !overlay) return;
  const host = canvas.parentElement;
  const prect = canvas.getBoundingClientRect();
  const hrect = host.getBoundingClientRect();
  // Absolute offsets are relative to the parent's padding box, while
  // getBoundingClientRect is its border box, so subtract the parent's border.
  overlay.style.left = `${prect.left - hrect.left - host.clientLeft}px`;
  overlay.style.top = `${prect.top - hrect.top - host.clientTop}px`;
  overlay.style.width = `${prect.width}px`;
  overlay.style.height = `${prect.height}px`;
}

// The overlay's drawing buffer matches its displayed size in device pixels (not
// the image's), so marks stay sharp and a few screen pixels wide however far a
// small image is scaled up. Returns a context in image pixel units and the
// size of one screen (CSS) pixel in those units.
function prepareOverlay(width, height, view = VIEWS.preview) {
  const overlay = view.overlayCanvas;
  positionOverlay(view);
  const rect = view.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.max(1, Math.round(rect.width * dpr));
  const bh = Math.max(1, Math.round(rect.height * dpr));
  if (overlay.width !== bw || overlay.height !== bh) {
    overlay.width = bw;
    overlay.height = bh;
  }
  clearOverlay(view);
  const ctx = overlay.getContext("2d");
  ctx.setTransform(bw / width, 0, 0, bh / height, 0, 0);
  return { ctx, cssPx: width / Math.max(1, rect.width) };
}

function drawCentroidOverlay() {
  drawEditorOverlay();
  if (!VIEWS.preview.overlayCanvas) return;
  if (!state.lastRender || !state.lastSites) return;
  const { ctx, cssPx } = prepareOverlay(state.lastRender.width, state.lastRender.height);
  drawCentroids(ctx, cssPx);
}

// ---- Site editor ----
// A copy of the preview whose sites can be dragged and edited once a run has
// stopped (never during one). Edits go to `editor.sites`, a working copy of
// the run's final sites (decoded colors), and to the stopped run's trainer,
// which updates the candidate lists for each edited site and re-renders.
// While dragging only this card redraws; when the edits settle they become
// state.lastSites / lastRender (a new array, so analyses keyed on the sites
// rerun) and the other views and exports follow.

const editor = {
  trainer: null,       // the trainer the working copy belongs to
  sites: null,         // working copy of the sites (Float32Array, decoded colors)
  original: null,      // the run's final sites, for Reset
  selected: -1,        // selected site index, -1 for none
  drag: null,          // { i, dx, dy } while dragging a site
  pending: new Set(),  // edited sites not yet written to the trainer
  busy: false,         // a write + render is in flight
  commit: null,        // promise of the in-flight commit to the other views
  commitTimer: 0,
  basePsnr: null,      // PSNR of the unedited sites
  psnr: null,          // PSNR of the latest edited render
  fields: null,        // { root, header, message, inputs: { key: input }, color, index, ... }
};
const EDIT_COMMIT_MS = 250;
const EDIT_PICK_PX = 8;  // pick radius around a centroid, CSS pixels

// The editable parameters: `idx` into a site's SITE_FLOATS, or get / set for
// derived ones (θ is the angle of the unit aniso_dir).
const EDIT_FIELDS = [
  { key: "x", label: "x", unit: "px", idx: 0, step: 0.1, digits: 2 },
  { key: "y", label: "y", unit: "px", idx: 1, step: 0.1, digits: 2 },
  { key: "logTau", label: "log τ", unit: "", idx: 2, step: 0.05, digits: 3 },
  { key: "radius", label: "radius", unit: "", idx: 3, step: 0.1, digits: 3 },
  { key: "r", label: "R", unit: "", idx: 4, step: 0.01, digits: 3 },
  { key: "g", label: "G", unit: "", idx: 5, step: 0.01, digits: 3 },
  { key: "b", label: "B", unit: "", idx: 6, step: 0.01, digits: 3 },
  { key: "theta", label: "θ (aniso dir)", unit: "°", step: 1, digits: 1,
    get: (s, b) => (Math.atan2(s[b + 8], s[b + 7]) * 180) / Math.PI,
    set: (s, b, v) => { const t = (v * Math.PI) / 180; s[b + 7] = Math.cos(t); s[b + 8] = Math.sin(t); } },
  { key: "logAniso", label: "log aniso", unit: "", idx: 9, step: 0.05, digits: 3 },
];

function editorAvailable() {
  return !state.running && !!state.trainer && !!state.lastSites && !!state.lastRender;
}

// Starts a fresh working copy when the run (or its sites) changed under it,
// and drops it while editing isn't available.
function syncEditorSites() {
  if (!editorAvailable()) {
    if (editor.sites) Object.assign(editor, { trainer: null, sites: null, original: null, selected: -1, drag: null,
                                              basePsnr: null, psnr: null });
    editor.pending.clear();
    return;
  }
  if (editor.trainer === state.trainer && editor.sites) return;
  editor.trainer = state.trainer;
  editor.sites = Float32Array.from(state.lastSites);
  editor.original = Float32Array.from(state.lastSites);
  editor.selected = -1;
  editor.drag = null;
  editor.basePsnr = null;
  editor.psnr = null;
}

function editorSites() { return editor.sites || state.lastSites; }

function drawEditor() {
  syncEditorSites();
  const view = VIEWS.editor;
  if (view.canvas && state.lastRender) {
    const { rgba, width, height } = state.lastRender;
    drawToCanvas(view.canvas, rgba, width, height);
  }
  drawEditorOverlay();
  updateEditorFields();
}

function drawEditorOverlay() {
  const view = VIEWS.editor;
  if (!view.overlayCanvas) return;
  view.overlayCanvas.classList.toggle("editable", editorAvailable());
  if (!state.lastRender || !editorSites()) { clearOverlay(view); return; }
  const { width, height } = state.lastRender;
  const { ctx, cssPx } = prepareOverlay(width, height, view);
  drawCentroids(ctx, cssPx, editorSites());
  const i = editor.selected;
  if (i < 0 || !editor.sites) return;
  const s = editor.sites;
  const b = i * SITE_FLOATS;
  const x = s[b] + 0.5, y = s[b + 1] + 0.5;
  ctx.save();
  ctx.lineWidth = 2 * cssPx;
  ctx.strokeStyle = "#ff7d33";
  ctx.beginPath();
  ctx.arc(x, y, 6 * cssPx, 0, Math.PI * 2);
  ctx.stroke();
  // The anisotropy axis, when the site has one.
  if (Math.abs(s[b + 9]) > 0.05) {
    const len = 14 * cssPx;
    const n = Math.hypot(s[b + 7], s[b + 8]) || 1;
    const ux = s[b + 7] / n, uy = s[b + 8] / n;
    ctx.beginPath();
    ctx.moveTo(x - ux * len, y - uy * len);
    ctx.lineTo(x + ux * len, y + uy * len);
    ctx.stroke();
  }
  ctx.restore();
}

// Image-pixel coordinates (site units) of a pointer event over the overlay.
function editorPointer(ev, overlay) {
  const rect = overlay.getBoundingClientRect();
  const { width, height } = state.lastRender;
  return {
    x: ((ev.clientX - rect.left) / Math.max(1, rect.width)) * width - 0.5,
    y: ((ev.clientY - rect.top) / Math.max(1, rect.height)) * height - 0.5,
    px: width / Math.max(1, rect.width), // image pixels per CSS pixel
  };
}

// The active site nearest to `p` within EDIT_PICK_PX screen pixels, or -1.
function pickSite(p) {
  const s = editor.sites;
  let best = -1;
  let bestD = (EDIT_PICK_PX * p.px) ** 2;
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const b = i * SITE_FLOATS;
    if (s[b] < 0) continue;
    const d = (s[b] - p.x) ** 2 + (s[b + 1] - p.y) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function attachEditorPointer(view, overlay) {
  overlay.addEventListener("pointerdown", (ev) => {
    if (!editorAvailable() || ev.button !== 0) return;
    syncEditorSites();
    const p = editorPointer(ev, overlay);
    const i = pickSite(p);
    editor.selected = i;
    if (i >= 0) {
      const b = i * SITE_FLOATS;
      editor.drag = { i, dx: editor.sites[b] - p.x, dy: editor.sites[b + 1] - p.y };
      overlay.setPointerCapture(ev.pointerId);
      overlay.style.cursor = "grabbing";
    }
    ev.preventDefault();
    drawEditorOverlay();
    updateEditorFields();
  });
  overlay.addEventListener("pointermove", (ev) => {
    if (!editorAvailable() || !editor.sites) return;
    const p = editorPointer(ev, overlay);
    if (!editor.drag) {
      overlay.style.cursor = pickSite(p) >= 0 ? "grab" : "default";
      return;
    }
    const { i, dx, dy } = editor.drag;
    const { width, height } = state.lastRender;
    const b = i * SITE_FLOATS;
    editor.sites[b] = Math.min(width - 1, Math.max(0, p.x + dx));
    editor.sites[b + 1] = Math.min(height - 1, Math.max(0, p.y + dy));
    markEdited(i);
  });
  const endDrag = (ev) => {
    if (!editor.drag) return;
    editor.drag = null;
    if (overlay.hasPointerCapture(ev.pointerId)) overlay.releasePointerCapture(ev.pointerId);
    overlay.style.cursor = "grab";
    scheduleCommit();
  };
  overlay.addEventListener("pointerup", endDrag);
  overlay.addEventListener("pointercancel", endDrag);
}

function buildEditorFields(root) {
  const header = document.createElement("div");
  header.className = "editor-header";
  const indexLabel = document.createElement("label");
  indexLabel.textContent = "Site ";
  const index = document.createElement("input");
  index.type = "number";
  index.min = "0";
  index.step = "1";
  index.placeholder = "—";
  index.setAttribute("aria-label", "Selected site index");
  index.addEventListener("change", () => {
    const i = Math.round(Number(index.value));
    editor.selected = Number.isFinite(i) && i >= 0 && i < state.lastSiteCount ? i : -1;
    drawEditorOverlay();
    updateEditorFields();
  });
  const count = document.createElement("span");
  count.className = "editor-count";
  indexLabel.append(index, count);
  const spacer = document.createElement("span");
  spacer.className = "editor-spacer";
  const resetSite = document.createElement("button");
  resetSite.className = "btn";
  resetSite.textContent = "Reset site";
  resetSite.addEventListener("click", () => {
    if (editor.selected >= 0) resetSites([editor.selected]);
  });
  const resetAll = document.createElement("button");
  resetAll.className = "btn";
  resetAll.textContent = "Reset all";
  resetAll.addEventListener("click", () => resetSites(editedSites()));
  header.append(indexLabel, spacer, resetSite, resetAll);

  // One line under the header: what to do, or the edited render's PSNR.
  const status = document.createElement("p");
  status.className = "editor-status";

  const grid = document.createElement("div");
  grid.className = "editor-grid";
  const inputs = {};
  for (const f of EDIT_FIELDS) {
    const label = document.createElement("label");
    label.className = "editor-field";
    const name = document.createElement("span");
    name.textContent = f.unit ? `${f.label} (${f.unit})` : f.label;
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(f.step);
    input.addEventListener("input", () => {
      const v = Number(input.value);
      const i = editor.selected;
      if (input.value === "" || !Number.isFinite(v) || i < 0 || !editor.sites) return;
      const b = i * SITE_FLOATS;
      if (f.set) f.set(editor.sites, b, v);
      else editor.sites[b + f.idx] = v;
      markEdited(i);
      scheduleCommit();
    });
    label.append(name, input);
    grid.append(label);
    inputs[f.key] = input;
  }
  const colorLabel = document.createElement("label");
  colorLabel.className = "editor-field";
  const colorName = document.createElement("span");
  colorName.textContent = "color";
  const color = document.createElement("input");
  color.type = "color";
  color.addEventListener("input", () => {
    const i = editor.selected;
    if (i < 0 || !editor.sites) return;
    const b = i * SITE_FLOATS;
    for (let c = 0; c < 3; c += 1) editor.sites[b + 4 + c] = parseInt(color.value.slice(1 + 2 * c, 3 + 2 * c), 16) / 255;
    markEdited(i);
    scheduleCommit();
  });
  colorLabel.append(colorName, color);
  grid.append(colorLabel);

  root.append(header, status, grid);
  editor.fields = { root, index, count, status, resetSite, resetAll, inputs, color };
}

function editedSites() {
  if (!editor.sites) return [];
  const out = [];
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const b = i * SITE_FLOATS;
    for (let t = 0; t < SITE_FLOATS; t += 1) {
      if (editor.sites[b + t] !== editor.original[b + t]) { out.push(i); break; }
    }
  }
  return out;
}

function resetSites(ids) {
  if (!editor.sites || !ids.length) return;
  for (const i of ids) {
    const b = i * SITE_FLOATS;
    editor.sites.set(editor.original.subarray(b, b + SITE_FLOATS), b);
    markEdited(i);
  }
  scheduleCommit();
}

function updateEditorFields() {
  const f = editor.fields;
  if (!f || !VIEWS.editor.card) return;
  const available = editorAvailable() && !!editor.sites;
  const i = available ? editor.selected : -1;
  f.index.disabled = !available;
  f.index.max = String(Math.max(0, state.lastSiteCount - 1));
  if (document.activeElement !== f.index) f.index.value = i >= 0 ? String(i) : "";
  f.count.textContent = available ? ` of ${state.lastSiteCount}` : "";
  const edited = available ? editedSites().length : 0;
  f.resetSite.disabled = !(i >= 0 && edited);
  f.resetAll.disabled = !edited;
  const delta = editor.psnr - editor.basePsnr;
  f.status.textContent = state.running ? "Editing is available once the run has stopped."
    : !available ? "Run training to get sites to edit."
    : editor.psnr != null ? `PSNR ${editor.psnr.toFixed(2)} dB (${delta >= 0 ? "+" : ""}${delta.toFixed(2)} from the run's ` +
                            `${editor.basePsnr.toFixed(2)}) · ${edited} site${edited === 1 ? "" : "s"} edited`
    : i < 0 ? "Click a site to select it, or drag it to move it."
    : "Drag the site or edit its parameters.";
  const s = editor.sites;
  const b = i * SITE_FLOATS;
  for (const field of EDIT_FIELDS) {
    const input = f.inputs[field.key];
    input.disabled = i < 0;
    if (document.activeElement === input) continue;
    input.value = i < 0 ? "" : (field.get ? field.get(s, b) : s[b + field.idx]).toFixed(field.digits);
  }
  f.color.disabled = i < 0;
  if (i >= 0 && document.activeElement !== f.color) {
    const hex = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");
    f.color.value = `#${hex(s[b + 4])}${hex(s[b + 5])}${hex(s[b + 6])}`;
  }
}

function markEdited(i) {
  editor.pending.add(i);
  drawEditorOverlay();
  updateEditorFields();
  pumpEdits();
}

// Writes the pending edits to the trainer and re-renders, as long as edits
// keep coming; one write + render is in flight at a time, so a fast drag
// skips frames instead of queueing them. The snap analysis swaps the
// trainer's sites while it runs, so edits wait for it.
async function pumpEdits() {
  if (editor.busy) return;
  editor.busy = true;
  try {
    while (editor.pending.size) {
      await editor.commit;
      await state.snapAnalysis?.promise;
      const trainer = editor.trainer;
      if (!editorAvailable() || trainer !== state.trainer) { editor.pending.clear(); break; }
      if (editor.basePsnr == null) editor.basePsnr = (await trainer.renderEdited()).psnr;
      for (const i of editor.pending) trainer.editSite(i, editor.sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS));
      editor.pending.clear();
      const { rgba, psnr } = await trainer.renderEdited();
      if (trainer !== state.trainer || state.running) break;
      state.lastRender = { rgba, width: state.lastRender.width, height: state.lastRender.height };
      editor.psnr = psnr;
      const view = VIEWS.editor;
      if (view.canvas) drawToCanvas(view.canvas, rgba, state.lastRender.width, state.lastRender.height);
      drawEditorOverlay();
      updateEditorFields();
    }
  } catch (err) {
    appendLog(`Edit failed: ${err.message}`);
    console.error(err);
  } finally {
    editor.busy = false;
  }
}

function scheduleCommit() {
  clearTimeout(editor.commitTimer);
  editor.commitTimer = setTimeout(commitEdits, EDIT_COMMIT_MS);
}

// Hands the settled edits to the rest of the page: the other views, the
// exports and the site analyses.
async function commitEdits() {
  if (!editor.sites || !editorAvailable()) return;
  if (editor.drag || editor.busy || editor.pending.size) { scheduleCommit(); return; }
  editor.commit = (async () => {
    state.lastSites = Float32Array.from(editor.sites);
    state.trainer.setLastSites(state.lastSites);
    state.lastHashed = null;
    state.lastHeatmap = null;
    state.lastOwner = null;
    if (editor.psnr != null) state.lastPsnr = editor.psnr;
    await refreshViews("render");
    drawCentroidOverlay();
  })();
  try { await editor.commit; } finally { editor.commit = null; }
}

async function ensureRender(mode) {
  if (!state.trainer) return null;
  if (mode === "hashed" && state.lastHashed) return state.lastHashed;
  if (mode === "heatmap" && state.lastHeatmap) return state.lastHeatmap;
  if (mode === "owner" && state.lastOwner) return state.lastOwner;
  try {
    if (state.lastSites) state.trainer.setLastSites(state.lastSites);
    const rgba = await state.trainer.renderView(mode);
    const out = { rgba, width: state.lastRender.width, height: state.lastRender.height };
    if (mode === "hashed") state.lastHashed = out;
    else if (mode === "heatmap") state.lastHeatmap = out;
    else if (mode === "owner") state.lastOwner = out;
    return out;
  } catch (err) {
    appendLog(`View render failed: ${err.message}`);
    return null;
  }
}

// Each site as a dot a few screen pixels wide with a dark rim. Site positions
// are in the renderer's pixel coordinates, where pixel (i, j) is sampled at
// (i, j); on a canvas that pixel's centre is (i + 0.5, j + 0.5).
function drawCentroids(ctx, cssPx = 1, sites = state.lastSites) {
  if (!sites || !state.lastRender) return;
  const { rgba, width, height } = state.lastRender;
  ctx.save();
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const base = i * SITE_FLOATS;
    const x = sites[base + 0];
    const y = sites[base + 1];
    if (x < 0) continue;
    // Fill with the inverse of the rendered pixel under the dot so it stands out.
    const px = Math.min(width - 1, Math.max(0, Math.floor(x + 0.5)));
    const py = Math.min(height - 1, Math.max(0, Math.floor(y + 0.5)));
    const p = (py * width + px) * 4;
    const inv = (v) => 255 - Math.round(Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)) * 255);
    ctx.fillStyle = `rgb(${inv(rgba[p])}, ${inv(rgba[p + 1])}, ${inv(rgba[p + 2])})`;
    ctx.beginPath();
    ctx.arc(x + 0.5, y + 0.5, Math.min(1.4, 2.5 * cssPx), 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 0.75 * cssPx;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
    ctx.stroke();
  }
  ctx.restore();
}

function stopTraining() {
  if (state.tune) state.tune.stop();
  else if (state.abortCtrl) state.abortCtrl.abort();
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
    state.lastRender.width, state.lastRender.height, { globalSite: state.trainer?.layerIndex ?? -1 });
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
    state.baseConfig = await loadConfig();
    state.config = state.baseConfig;
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
  for (const [key, label] of Object.entries(PRESETS)) ui.presetSelect.add(new Option(label, key));
  ui.presetSelect.addEventListener("change", () => applyPreset(ui.presetSelect.value));
  ui.saveDefaultsBtn.addEventListener("click", saveDefaults);
  ui.resetDefaultsBtn.addEventListener("click", resetDefaults);
  syncDefaultsButtons();
  for (const [key, dataset] of Object.entries(DATASETS)) ui.datasetSelect.add(new Option(dataset.label, key));
  const datasetReady = selectDataset("mnist");
  ui.datasetSelect.addEventListener("change", () => selectDataset(ui.datasetSelect.value));
  ui.itemSelect.addEventListener("change", () => {
    if (ui.itemSelect.value) loadItem(Number(ui.itemSelect.value));
  });
  ui.itemPrev.addEventListener("click", () => stepItem(-1));
  ui.itemNext.addEventListener("click", () => stepItem(1));
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = ev.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    ev.preventDefault();
    stepItem(ev.key === "ArrowRight" ? 1 : -1);
  });
  ui.maskInput.addEventListener("change", handleMaskInput);
  ui.initSitesInput.addEventListener("change", handleInitSitesInput);
  ui.initSitesClear.addEventListener("click", clearInitSites);
  for (const id of ["p-fromBpp", "p-bpp"]) document.getElementById(id).addEventListener("input", syncBppSites);
  for (const id of ["p-sites", "p-startSites", "p-endSites", "p-countSchedule", "p-sawAmp", "p-sawTeeth"]) {
    document.getElementById(id).addEventListener("input", syncSiteFields);
    document.getElementById(id).addEventListener("change", syncSiteFields);
  }
  ui.paramsPanel.addEventListener("input", updatePlanSummary);
  ui.startBtn.addEventListener("click", () => {
    if (state.runMode === "batch") runBatch();
    else startTraining();
  });
  ui.singleModeBtn.addEventListener("click", () => setRunMode("single"));
  ui.batchModeBtn.addEventListener("click", () => setRunMode("batch"));
  ui.stopBtn.addEventListener("click", stopTraining);
  ui.toggleParamsBtn.addEventListener("click", toggleParams);
  ui.saveBtn.addEventListener("click", toggleSaveMenu);
  ui.savePng.addEventListener("click", exportPng);
  ui.saveTxt.addEventListener("click", exportSites);
  ui.batchBtn.addEventListener("click", toggleBatchMenu);
  ui.batchRunBtn.addEventListener("click", () => runBatch());
  setupAutoTune();
  ui.batchFill.addEventListener("change", syncBatchFill);
  ui.batchTiled.addEventListener("change", syncBatchFill);
  ui.batchFrom.addEventListener("change", syncBatchFill);
  ui.batchRandom.addEventListener("change", updateBatchNote);
  ui.batchRunSelect.addEventListener("change", () => {
    if (ui.batchRunSelect.value) selectBatchRun(Number(ui.batchRunSelect.value));
  });
  ui.batchPrev.addEventListener("click", () => stepBatchRun(1));
  ui.batchNext.addEventListener("click", () => stepBatchRun(-1));
  ui.batchSnap.addEventListener("input", () => showBatchSnapshot(Number(ui.batchSnap.value)));
  ui.batchRerun.addEventListener("click", rerunBatchRun);
  ui.batchExport.addEventListener("click", exportBatch);
  setRunMode("batch");
  document.addEventListener("click", (ev) => {
    if (!ui.savePopover.contains(ev.target) && ev.target !== ui.saveBtn) {
      ui.savePopover.classList.add("hidden");
    }
    if (!ui.batchPopover.contains(ev.target) && ev.target !== ui.batchBtn) {
      ui.batchPopover.classList.add("hidden");
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

  // Load straight into a batch run with the default settings.
  if (!(await datasetReady)) return;
  runBatch();
}

main();
