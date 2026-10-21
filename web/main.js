// Viewer for saved *_sites.txt files. It renders with the trainer's own
// kernels (candidate lists from trainer/candidates.js, the blend from
// render_compute.wgsl), so a file looks exactly as it did while training.
// Views: Color, IDs (each site's hashColor), and Tau / Anisotropy / Radius
// (the same blend with every site colored by that parameter, normalized).

import { loadShaders, tileSlotsForLimit } from "../trainer/shader_loader.js?v=rate-20261020";
import { requestTrainingDevice } from "../trainer/device.js?v=20261019";
import { buildArgs } from "../trainer/trainer.js?v=rate-20261020";
import { CandidateLists } from "../trainer/candidates.js?v=20261019";
import { RenderEncoder } from "../trainer/encoders.js?v=rate-20261020";
import { SITE_FLOATS, createParamsBuffer } from "../trainer/params.js?v=20261019";
import { createRgba32FloatTexture, readRgba32FloatTexture } from "../trainer/textures.js?v=20261019";
import { parseSitesTxt, floatImageToRgba8, triggerDownload } from "../trainer/io.js?v=20261019";

const fileInput = document.getElementById("fileInput");
const modeSelect = document.getElementById("renderModeSelect");
const dotsToggle = document.getElementById("dotsToggle");
const exportBtn = document.getElementById("exportBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("viewCanvas");
const fallback = document.getElementById("fallback");

// Regular candidate updates after the jump floods of a rebuild.
const SETTLE_PASSES = 16;
// Site field (in the SITE_FLOATS layout) each parameter view shows.
const PARAM_FIELD = { 2: 2, 3: 9, 4: 3 }; // tau → log_tau, anisotropy → log_aniso, radius

let gpu = null;     // { device, shaders, renderEnc, idsEnc, config }
let view = null;    // the loaded file: { data, count, width, height, sitesBuf, cands, ... }

function setStatus(message) { statusEl.textContent = message; }

async function init() {
  try {
    const { device } = await requestTrainingDevice(navigator.gpu);
    const shaders = await loadShaders({ tileSlots: tileSlotsForLimit(device.limits.maxComputeWorkgroupStorageSize) });
    const config = await (await fetch("../training_config.json")).json();
    gpu = {
      device, shaders, args: buildArgs(config),
      renderEnc: new RenderEncoder(device, shaders),
      idsEnc: new RenderEncoder(device, shaders, { hashIds: true }),
    };
    setStatus("Ready. Choose a sites.txt file.");
  } catch (err) {
    fallback.classList.remove("hidden");
    setStatus(`WebGPU unavailable: ${err.message || err}`);
  }
}

function releaseView() {
  if (!view) return;
  view.cands.destroy();
  for (const b of [view.sitesBuf, view.vizBuf, view.params, view.out]) b.destroy();
  view = null;
}

// Uploads the sites and builds their candidate lists.
function loadView({ data, count, width, height, globalSite }) {
  releaseView();
  const device = gpu.device;
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const sitesBuf = device.createBuffer({ size: data.byteLength, usage: storage });
  device.queue.writeBuffer(sitesBuf, 0, data);
  const cands = new CandidateLists(device, gpu.shaders, { ...gpu.args, width, height, capacity: count });
  cands.globalCand = globalSite >= 0 ? globalSite : 0xffffffff;
  cands.rebuild(sitesBuf, count);
  const encoder = device.createCommandEncoder();
  for (let p = 0; p < SETTLE_PASSES; p += 1) cands.encodeUpdate(encoder, sitesBuf, count);
  device.queue.submit([encoder.finish()]);
  view = {
    data, count, width, height, sitesBuf, cands,
    vizBuf: device.createBuffer({ size: data.byteLength, usage: storage }),
    params: createParamsBuffer(device, cands.paramsArgs(count)),
    out: createRgba32FloatTexture(device, width, height, { storage: true, copySrc: true }),
  };
}

// A copy of the sites colored gray by one parameter, normalized to 0..1 over
// the active sites.
function paramSites(field) {
  const { data, count } = view;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < count; i += 1) {
    if (data[i * SITE_FLOATS] < 0) continue;
    lo = Math.min(lo, data[i * SITE_FLOATS + field]);
    hi = Math.max(hi, data[i * SITE_FLOATS + field]);
  }
  const span = hi > lo ? hi - lo : 1;
  const out = data.slice();
  for (let i = 0; i < count; i += 1) {
    const g = (data[i * SITE_FLOATS + field] - lo) / span;
    out.fill(g, i * SITE_FLOATS + 4, i * SITE_FLOATS + 7);
  }
  return out;
}

// The current view as RGBA8 at image resolution.
async function renderRgba8(mode) {
  const device = gpu.device;
  let sites = view.sitesBuf;
  if (PARAM_FIELD[mode] !== undefined) {
    device.queue.writeBuffer(view.vizBuf, 0, paramSites(PARAM_FIELD[mode]));
    sites = view.vizBuf;
  }
  const enc = mode === 1 ? gpu.idsEnc : gpu.renderEnc;
  const encoder = device.createCommandEncoder();
  enc.encode(encoder, sites, view.params, view.cands.cand0, view.cands.cand1, view.out, view.width, view.height);
  device.queue.submit([encoder.finish()]);
  const rgba = await readRgba32FloatTexture(device, view.out, view.width, view.height);
  return floatImageToRgba8(rgba, view.width, view.height);
}

async function draw() {
  if (!view) return;
  const { width, height, data, count } = view;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(await renderRgba8(Number(modeSelect.value)), width, height), 0, 0);
  if (dotsToggle.checked) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    const r = Math.max(1, Math.max(width, height) / 400);
    for (let i = 0; i < count; i += 1) {
      const b = i * SITE_FLOATS;
      if (data[b] < 0) continue;
      ctx.beginPath();
      ctx.arc(data[b] + 0.5, data[b + 1] + 0.5, r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

async function onFile() {
  const file = fileInput.files[0];
  if (!file || !gpu) return;
  const label = fileInput.closest(".file")?.querySelector("span");
  if (label) label.textContent = file.name;
  try {
    const parsed = parseSitesTxt(await file.text());
    if (!(parsed.width > 0 && parsed.height > 0)) throw new Error("missing image size (expected '# Image size: W H')");
    loadView(parsed);
    await draw();
    exportBtn.disabled = false;
    setStatus(`Loaded ${parsed.count} sites at ${parsed.width}×${parsed.height}` +
      (parsed.globalSite >= 0 ? `, layer at site ${parsed.globalSite}.` : "."));
  } catch (err) {
    setStatus(`Load failed: ${err.message || err}`);
  }
}

// The color view at image resolution, without dots.
async function exportPNG() {
  if (!view) return;
  const out = document.createElement("canvas");
  out.width = view.width;
  out.height = view.height;
  out.getContext("2d").putImageData(new ImageData(await renderRgba8(0), view.width, view.height), 0, 0);
  out.toBlob((blob) => blob && triggerDownload(blob, `sad_render_${view.width}x${view.height}.png`), "image/png");
}

fileInput.addEventListener("change", onFile);
modeSelect.addEventListener("change", draw);
dotsToggle.addEventListener("change", draw);
exportBtn.addEventListener("click", exportPNG);
init();
