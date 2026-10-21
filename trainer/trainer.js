// SAD training loop. Originally ported from the wgpu-py trainer (train_wgpu.py
// at the `multi-backend` tag). The site count follows a schedule from
// startSites to endSites (countTarget), growing by splitting cells and
// shrinking by merging neighbours (see "Site count" below); optionally with a
// histogram layer behind the sites (LAYER_MODES).

import {
  SITE_FLOATS,
  ADAM_FLOATS,
  createParamsBuffer,
  updateParamsBuffer,
  createInitParamsBuffer,
  createGradParamsBuffer,
  createAdamParamsBuffer,
  updateAdamParamsBuffer,
  createClearParamsBuffer,
  updateGradParamsBuffer,
  invScaleSqFor,
  createUniform32,
} from "./params.js?v=cmoment-20261020";

import {
  createSitesBuffer,
  createAdamBuffer,
  createSeedCounter,
  createGradsBuffer,
  readSites,
} from "./buffers.js?v=20261019";

import {
  createRgba32FloatTexture,
  writeFloat32ToRgba32FloatTexture,
  readRgba32FloatTexture,
  whiteMaskRgba32Float,
} from "./textures.js?v=20261019";

import {
  InitGradientEncoder,
  RenderEncoder,
  RenderTauHeatmapEncoder,
  GradientsEncoder,
  AdamEncoder,
  ClearBufferEncoder,
  SiteStatsEncoder,
} from "./encoders.js?v=20261019";

import { BITS_PER_SITE } from "./io.js?v=20261019";
import {
  cellMoments, pickMerges, mergedSite, mergePrecision, sitePrecision, candidateEdges, neighborMerge, EDGE_GROUPS,
} from "./merge.js?v=20261019";
import { CandidateLists } from "./candidates.js?v=20261019";
import {
  colorSpaceFor,
  pcaBasis,
  specializeShaders,
  encodeSiteColors,
  decodeSiteColors,
} from "./color_spaces.js?v=klt-20261020";

const MAX_PREVIEW_IMAGE_READBACKS = 2;
const MAX_PREVIEW_SITE_READBACKS = 8;
// Direction and anisotropy only start learning at this iteration. Removing the
// delay cost 0.82 ± 0.24 dB on 40 held-out MNIST digits at 8 sites (5 repeats;
// -0.08 at 32, within noise on Kodak).
const ANI_START_ITER = 50;

// Training losses (args.loss): label, and the LOSS id gradients_tiled.wgsl
// switches on. Growth, λ and PSNR measure squared error either way.
export const LOSSES = {
  l1: { label: "L1 (absolute error)", id: 1 },
  l2: { label: "L2 (squared error)", id: 0 },
};

// Adam second moments for the three color params (args.colorMoment): the id
// adam.wgsl switches on. "channel" keeps one per param; "shared" keeps one for
// the 3-vector (EMA of |g|², as position), so a param's step follows its share
// of the color gradient: on greyscale targets chroma params with only float
// noise for gradient stay put instead of taking lr-sized steps. Shared is the
// default: with each space's six LR bases retuned (node/colorspaces.mjs, 2048
// held-out images × 3), median PSNR vs "channel" on MNIST rgb −0.02 dB,
// logchroma +0.25, ycocg +0.24, klt +0.17; CIFAR-10 −0.01, +0.06, +0.07, −0.03;
// Kodak (8 images, 1000 sites, untuned) within ±0.1.
export const COLOR_MOMENTS = {
  channel: { label: "per param", id: 0 },
  shared: { label: "shared", id: 1 },
};

// LR schedules (args.lrSchedule), over a run of `iters` iterations. No warmup:
// on MNIST and Kodak it was within ±0.2 dB of a 200-iteration warmup (sites
// added by growth never had one; Adam's bias correction restarts per site).
export const LR_SCHEDULES = {
  cosine: "cosine",                            // cosine from 1 to 0 over the run
  fixed: "fixed",                              // 1 throughout
  "decay-after-growth": "decay after growth",  // 1 until growth ends, then cosine to 0 by the end
};

// Multiplier applied to every base LR at step `it`. `growDoneIter` is the
// iteration growth ended (Infinity while still growing); only
// "decay-after-growth" uses it.
export function lrSchedule(it, iters, kind = "cosine", growDoneIter = 0) {
  if (kind === "fixed") return 1;
  const start = kind === "decay-after-growth" ? Math.max(0, growDoneIter) : 0;
  if (it < start) return 1;
  const t = Math.min(1, (it - start) / Math.max(1, iters - start));
  return 0.5 * (1 + Math.cos(Math.PI * t));
}

// How the site count moves from startSites to endSites (args.countSchedule).
// The count reaches endSites at countEnd × iters and holds there; start >
// end prunes (merges) instead of growing, start = end is a fixed count.
// "constant" holds one count throughout: buildArgs sets startSites = endSites
// = sites, and a loaded sites file's count holds as it is.
export const COUNT_SCHEDULES = {
  constant: "constant",
  linear: "linear",
  exponential: "exponential",
  sawtooth: "sawtooth",
};

// Iteration the schedule reaches endSites by.
export function countHorizon(args) {
  return Math.max(1, Math.round(args.countEnd * args.iters));
}

// Site count the schedule wants at iteration `it` (real-valued), from `start`
// (startSites, or a loaded sites file's count). "sawtooth" rides on the
// exponential curve: sawTeeth teeth up to the horizon, each rising to
// (1 + sawAmp) × the curve over sawRatio / (1 + sawRatio) of the tooth, then
// falling back to it.
export function countTarget(args, it, start = args.startSites) {
  const end = args.endSites;
  const t = Math.min(1, Math.max(0, it / countHorizon(args)));
  const base = args.countSchedule === "linear" ? start + (end - start) * t : start * (end / start) ** t;
  if (args.countSchedule !== "sawtooth" || t >= 1) return base;
  const phase = (t * args.sawTeeth) % 1;
  const rise = args.sawRatio / (1 + args.sawRatio);
  const tooth = phase < rise ? phase / rise : (1 - phase) / (1 - rise);
  return base * (1 + args.sawAmp * tooth);
}

// Whether the schedule ever changes the count.
export function countVaries(args, start = args.startSites) {
  if (args.countSchedule === "constant") return false;
  return start !== args.endSites || (args.countSchedule === "sawtooth" && args.sawAmp > 0 && args.sawTeeth > 0);
}

// The planned count after each step: [{ it, count }] for every step that
// changes it, up to the horizon.
export function countPlan(args, start = args.startSites) {
  const plan = [];
  if (!countVaries(args, start)) return plan;
  const horizon = countHorizon(args);
  let n = start;
  for (let it = args.growEvery; it < horizon + args.growEvery; it += args.growEvery) {
    const target = Math.max(1, Math.round(countTarget(args, Math.min(it, horizon), start)));
    if (target !== n) plan.push({ it, count: (n = target) });
  }
  return plan;
}

// Most sites the schedule ever wants (sizes the GPU buffers).
export function countCapacity(args, start = args.startSites) {
  return countPlan(args, start).reduce((m, p) => Math.max(m, p.count), Math.max(start, args.endSites));
}

// Histogram layer (args.layer). "learn": a flat site behind all the others,
// colored by the peak of the target's smoothed color histogram (colorPeak).
// Its logit is a constant β (τ = e^LEARN_LOG_TAU, so flat to 0.03 logits
// across the image): a site wins a pixel where its own logit -τ(d - r) beats
// β, and the layer's color shows wherever none does. β is learned like any
// other parameter (adam.wgsl, params.layer_site), starting from log(p / (1 - p)) of
// the peak's mass p: a common color starts in front of the sites' edges and
// takes the background, a rare one starts behind and the gradient decides.
// Every pixel blends the layer besides its 8 listed candidates (globalCand;
// it never takes a list slot); it never splits or merges, counts against
// startSites and endSites, and is an ordinary site in the sites file ("# Global site").
// A per-site bias from p(color) on every site would not do this: a constant
// added to -τ(d - r) is exactly a radius change, which the optimizer undoes.
// MNIST: +0.78 / +3.06 dB at 8 / 32 sites; Kodak at 256 px: -0.13 dB, the
// layer staying behind (β ≈ -4.7, ~4% of pixels).
export const LAYER_MODES = {
  off: "off",
  learn: "learned depth",
};
// τ of the layer and its depth range; β is stored as radius β / (τ · inv_scale).
const LEARN_LOG_TAU = -4;
const LAYER_BETA_MAX = 8; // also in adam.wgsl
// colorPeak of each image tile of an atlas (batch.js) of width `width`.
function tilePeaks(target, mask, width, tiling) {
  const { tileW, tileH, tilesX, count } = tiling;
  const n = tileW * tileH;
  const sub = new Float32Array(n * 4);
  const subMask = mask ? new Float32Array(n) : null;
  const peaks = [];
  for (let b = 0; b < count; b += 1) {
    const ox = (b % tilesX) * tileW, oy = Math.floor(b / tilesX) * tileH;
    for (let y = 0; y < tileH; y += 1) {
      const row = (oy + y) * width + ox;
      sub.set(target.subarray(row * 4, (row + tileW) * 4), y * tileW * 4);
      if (subMask) subMask.set(mask.subarray(row, row + tileW), y * tileW);
    }
    peaks.push(colorPeak(sub, n, subMask));
  }
  return peaks;
}

// Color bins (8-bit levels per bin and channel) of colorPeak's histogram.
const PEAK_BIN = 16;

// The peak of the target's color histogram smoothed over ±1 bin of PEAK_BIN
// levels (a box kernel ~±16 levels wide): { color (the most common exact color
// within the peak's kernel, 0..1 RGB), mass (fraction of pixels in the kernel) }.
export function colorPeak(target, n, mask = null) {
  const B = 256 / PEAK_BIN;
  const hist = new Float64Array(B * B * B);
  const exact = new Map();
  let total = 0;
  const bin = (v) => Math.min(B - 1, Math.floor(Math.round(v * 255) / PEAK_BIN));
  for (let p = 0; p < n; p += 1) {
    if (mask && !(mask[p] > 0)) continue;
    const r = target[p * 4], g = target[p * 4 + 1], b = target[p * 4 + 2];
    hist[(bin(r) * B + bin(g)) * B + bin(b)] += 1;
    total += 1;
  }
  let best = 0, bestMass = -1;
  for (let i = 0; i < B; i += 1) for (let j = 0; j < B; j += 1) for (let k = 0; k < B; k += 1) {
    let m = 0;
    for (let a = Math.max(0, i - 1); a <= Math.min(B - 1, i + 1); a += 1)
      for (let c = Math.max(0, j - 1); c <= Math.min(B - 1, j + 1); c += 1)
        for (let d = Math.max(0, k - 1); d <= Math.min(B - 1, k + 1); d += 1) m += hist[(a * B + c) * B + d];
    if (m > bestMass) { bestMass = m; best = (i * B + j) * B + k; }
  }
  for (let p = 0; p < n; p += 1) {
    if (mask && !(mask[p] > 0)) continue;
    const r = target[p * 4], g = target[p * 4 + 1], b = target[p * 4 + 2];
    const bi = Math.floor(best / (B * B)), bj = Math.floor(best / B) % B, bk = best % B;
    if (Math.abs(bin(r) - bi) > 1 || Math.abs(bin(g) - bj) > 1 || Math.abs(bin(b) - bk) > 1) continue;
    const key = (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
    exact.set(key, (exact.get(key) || 0) + 1);
  }
  let key = 0, cnt = -1;
  for (const [k, c] of exact) if (c > cnt) { key = k; cnt = c; }
  return { color: [(key >> 16) / 255, ((key >> 8) & 255) / 255, (key & 255) / 255], mass: total ? bestMass / total : 0 };
}

// Collect default args from a training_config.json-style object + user
// overrides. Overrides use the argument name (e.g. "iters") and take precedence
// over the matching config-style key (e.g. "DEFAULT_ITERS").
export function buildArgs(config, overrides = {}) {
  // `argName` is what the rest of the code uses; `configKey` is the
  // training_config.json-style key; `fallback` applies if neither source has a
  // value. Overrides always check `argName` first.
  const pick = (argName, configKey, fallback) => {
    if (overrides[argName] !== undefined) return overrides[argName];
    if (configKey && config[configKey] !== undefined) return config[configKey];
    return fallback;
  };
  const args = {
    // Run length. The schedule, LR and growth defaults (and LR_POS_BASE / LR_TAU_BASE in
    // training_config.json) were tuned for growing 1 -> 100 sites over 1000
    // iterations, by mean final PSNR over 8 Kodak images, then checked on the
    // other 16. The fallback is now 500 iterations.
    iters: Math.max(1, Math.floor(Number(pick("iters", null, 500)))),
    candFreq: Number(pick("candFreq", "CAND_UPDATE_FREQ", 1)),
    candDownscale: Math.max(1, Number(pick("candDownscale", "CAND_DOWNSCALE", 1))),
    candUpdatePasses: Math.max(1, Number(pick("candUpdatePasses", "CAND_UPDATE_PASSES", 1))),
    candRadiusScale: Number(pick("candRadiusScale", "CAND_RADIUS_SCALE", 64.0)),
    candRadiusProbes: Number(pick("candRadiusProbes", "CAND_RADIUS_PROBES", 0)),
    candInjectCount: Number(pick("candInjectCount", "CAND_INJECT_COUNT", 16)),
    logFreq: Math.max(1, Number(pick("logFreq", null, 50))),
    viewerFreq: Math.max(1, Number(pick("viewerFreq", null, 50))),
    lrMult: Number(pick("lrMult", "LR_MULT", 8)),
    lrSchedule: String(pick("lrSchedule", "LR_SCHEDULE", "fixed")),
    // Site color parameterization (a key of COLOR_SPACES in color_spaces.js).
    // Sites stay RGB outside the trainer; only the optimized params change.
    colorSpace: String(pick("colorSpace", "COLOR_SPACE", "rgb")),
    // How Adam scales the color steps (COLOR_MOMENTS).
    colorMoment: String(pick("colorMoment", "COLOR_MOMENT", "shared")),
    loss: String(pick("loss", "LOSS", "l1")),
    // Persistent-error weighting (gradients_tiled.wgsl): each pixel's loss is
    // scaled by (1 + errWeight · min(b / mean b, errCap)) / (1 + errWeight),
    // b the squared (L1: absolute) EWMA of its signed error with decay
    // errBeta per iteration, so error that keeps its sign pulls harder than
    // error that flips. errWeight 0 = off. errGrowth: growth's split moments
    // use the same weights, so splits go where the error persists.
    errWeight: Math.max(0, Number(pick("errWeight", "ERR_WEIGHT", 0))),
    errBeta: Math.min(0.9999, Math.max(0, Number(pick("errBeta", "ERR_BETA", 0.95)))),
    errCap: Math.max(1, Number(pick("errCap", "ERR_CAP", 8))),
    errGrowth: Boolean(pick("errGrowth", "ERR_GROWTH", false)),
    lrPosBase: Number(pick("lrPosBase", "LR_POS_BASE", 0.05)),
    lrTauBase: Number(pick("lrTauBase", "LR_TAU_BASE", 0.01)),
    lrRadiusBase: Number(pick("lrRadiusBase", "LR_RADIUS_BASE", 0.02)),
    lrColorBase: Number(pick("lrColorBase", "LR_COLOR_BASE", 0.02)),
    lrDirBase: Number(pick("lrDirBase", "LR_DIR_BASE", 0.02)),
    lrAnisoBase: Number(pick("lrAnisoBase", "LR_ANISO_BASE", 0.02)),
    initLogTau: Number(config.INIT_LOG_TAU),
    initRadius: Number(config.INIT_RADIUS),
    beta1: Number(pick("beta1", "BETA1", 0.9)),
    beta2: Number(pick("beta2", "BETA2", 0.999)),
    eps: Number(pick("eps", "EPS", 1e-8)),
    maxDim: Number(pick("maxDim", "MAX_DIM", 2048)),
    // Site count: from startSites at iteration 0 to endSites by countEnd ×
    // iters on the countSchedule curve (COUNT_SCHEDULES), stepping every
    // growEvery iterations: up by splitting cells (siteSplit), down by
    // merging the cheapest neighbouring pairs (merge.js). sawTeeth, sawAmp and
    // sawRatio shape "sawtooth" (countTarget). These are the only settings
    // that set the count; the buffers hold countCapacity sites. "constant"
    // (the default) ignores them and holds `sites` (default endSites).
    startSites: Math.max(1, Math.floor(Number(pick("startSites", "START_SITES", 1)))),
    endSites: Math.max(1, Math.floor(Number(pick("endSites", "END_SITES", 100)))),
    countSchedule: String(pick("countSchedule", "COUNT_SCHEDULE", "constant")),
    countEnd: Math.min(1, Math.max(0, Number(pick("countEnd", "COUNT_END", 0.5)))),
    sawTeeth: Math.max(0, Math.floor(Number(pick("sawTeeth", "SAW_TEETH", 6)))),
    sawAmp: Math.max(0, Number(pick("sawAmp", "SAW_AMP", 2))),
    sawRatio: Math.max(0.01, Number(pick("sawRatio", "SAW_RATIO", 3))),
    growEvery: Math.max(1, Math.floor(Number(pick("growEvery", "GROW_EVERY", 5)))),
    // growLambda stops growth (and with it the count schedule) early: the slope
    // -dD/dR, i.e. the squared error (8-bit levels², summed over RGB) a new
    // site must remove per bit it costs (BITS_PER_SITE bits). 0 = no early
    // stop (any positive gain still qualifies).
    growLambda: Math.max(0, Number(pick("growLambda", "GROW_LAMBDA", 400))),
    // How a merge averages the pair's parameters (merge.js): "precision" weighs
    // each parameter by the site's Adam second moment (how much the loss cares
    // about it), "area" by the cells' pixel counts.
    mergeWeights: String(pick("mergeWeights", "MERGE_WEIGHTS", "area")),
    // Which sites a merge step joins (merge.js): "neighbor" by parameter
    // similarity to the site's mergeTopK nearest neighbours (position left
    // out), "ward" by the rise in error with each cell at its mean color.
    mergeCriterion: String(pick("mergeCriterion", "MERGE_CRITERION", "neighbor")),
    mergeTopK: Math.max(1, Math.round(Number(pick("mergeTopK", "MERGE_TOP_K", 3)))),
    // Turnover (_turnover): every turnoverEvery iterations while the schedule
    // holds the count (startSites = endSites with a linear or exponential
    // schedule, or after the schedule ends), merge this fraction of the sites
    // (at least one) and split as many cells back, until turnoverEnd × iters.
    // 0 = off. Each step costs a GPU sync and a candidate rebuild (~0.1 s at
    // 256 px), so fewer, larger steps are cheaper.
    turnover: Math.min(0.5, Math.max(0, Number(pick("turnover", "TURNOVER", 0)))),
    turnoverEvery: Math.max(1, Math.floor(Number(pick("turnoverEvery", "TURNOVER_EVERY", 20)))),
    turnoverEnd: Math.min(1, Math.max(0, Number(pick("turnoverEnd", "TURNOVER_END", 0.8)))),
    // Histogram layer (LAYER_MODES): a flat site behind all others, colored by
    // the peak of the target's color histogram, with a learned depth.
    layer: String(pick("layer", "LAYER", "off")),
  };
  if (!COUNT_SCHEDULES[args.countSchedule]) {
    throw new Error(`countSchedule must be one of ${Object.keys(COUNT_SCHEDULES).join(", ")} (got "${args.countSchedule}")`);
  }
  args.sites = Math.max(1, Math.floor(Number(pick("sites", "SITES", args.endSites))));
  if (args.countSchedule === "constant") args.startSites = args.endSites = args.sites;
  if (!(args.lrSchedule in LR_SCHEDULES)) {
    throw new Error(`lrSchedule must be one of ${Object.keys(LR_SCHEDULES).join(", ")} (got "${args.lrSchedule}")`);
  }
  colorSpaceFor(args.colorSpace); // fail early on an unknown name
  if (!COLOR_MOMENTS[args.colorMoment]) {
    throw new Error(`colorMoment must be one of ${Object.keys(COLOR_MOMENTS).join(", ")} (got "${args.colorMoment}")`);
  }
  if (!LOSSES[args.loss]) {
    throw new Error(`loss must be one of ${Object.keys(LOSSES).join(", ")} (got "${args.loss}")`);
  }
  if (!LAYER_MODES[args.layer]) {
    throw new Error(`layer must be one of ${Object.keys(LAYER_MODES).join(", ")} (got "${args.layer}")`);
  }
  if (!Number.isFinite(args.initLogTau) || !Number.isFinite(args.initRadius)) {
    throw new Error("training_config.json missing INIT_LOG_TAU / INIT_RADIUS (required).");
  }
  return args;
}

// Cooperative-yield helper — allows the browser to render and the caller to
// cancel long runs.
async function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class Trainer {
  constructor(device, shaders, args, { onLog = null, signal = null } = {}) {
    this.device = device;
    this.args = args;
    this.onLog = onLog;
    this.signal = signal;
    this.baseShaders = shaders;

    this._previewImageInFlight = 0;
    this._previewSiteInFlight = 0;

    // klt without args.colorBasis takes its basis from the target, so its
    // shaders are compiled in prepare() rather than twice.
    const space = colorSpaceFor(args.colorSpace, args.colorBasis ?? null);
    if (space.needsBasis && !args.colorBasis) this.colorSpace = space;
    else this._buildEncoders(space);
  }

  // Shaders and encoders for `colorSpace` (klt's basis is compiled in).
  _buildEncoders(colorSpace) {
    const device = this.device, args = this.args;
    this.colorSpace = colorSpace;
    const shaders = this.shaders = specializeShaders(this.baseShaders, colorSpace);
    this.gnEnc = null;
    this.initEnc = new InitGradientEncoder(device, shaders);
    this.renderEnc = new RenderEncoder(device, shaders);
    this.renderIdsEnc = new RenderEncoder(device, shaders, { hashIds: true });
    this.renderOwnerEnc = new RenderEncoder(device, shaders, { ownerIds: true });
    this.renderTauEnc = new RenderTauHeatmapEncoder(device, shaders);
    this.gradsEnc = new GradientsEncoder(device, shaders, { loss: LOSSES[args.loss].id });
    this.adamEnc = new AdamEncoder(device, shaders);
    this.clearI32Enc = new ClearBufferEncoder(device, shaders, { useI32: true });
    this.statsEnc = new SiteStatsEncoder(device, shaders);
  }

  // `tiling` (batch.js) trains an atlas of images in one run: a fixed count
  // of args.startSites sites per tile (the layer included), no init sites.
  async prepare({ width, height, targetRgba32f, maskRgba32f, maskSum, initSites, initSiteCount, tiling = null }) {
    const args = this.args;
    const device = this.device;

    this.width = width;
    this.height = height;
    this.tiling = tiling;
    this.target = targetRgba32f; // keep a CPU copy for PSNR
    this.maskSum = maskSum;
    this.maskCpu = null;
    if (maskRgba32f) {
      this.maskCpu = new Float32Array(width * height);
      for (let i = 0; i < width * height; i += 1) this.maskCpu[i] = maskRgba32f[i * 4];
    }
    // klt without args.colorBasis: the KLT of the masked target pixels (an
    // atlas's images share it).
    if (this.colorSpace.needsBasis && !args.colorBasis) {
      this._buildEncoders(colorSpaceFor(args.colorSpace, pcaBasis(targetRgba32f, width * height, this.maskCpu)));
    }

    // Textures
    this.targetTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    writeFloat32ToRgba32FloatTexture(device, this.targetTex, targetRgba32f, width, height);
    this.maskTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    const maskData = maskRgba32f || whiteMaskRgba32Float(width, height).rgba;
    writeFloat32ToRgba32FloatTexture(device, this.maskTex, maskData, width, height);
    // The mask again as f32 per pixel (0 or 1), for gradients and site stats.
    const maskF = new Float32Array(width * height);
    for (let i = 0; i < width * height; i += 1) maskF[i] = this.maskCpu ? (this.maskCpu[i] > 0 ? 1 : 0) : 1;
    this.maskBuf = device.createBuffer({ size: maskF.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.maskBuf, 0, maskF);
    // Persistent-error weighting (args.errWeight): per pixel the EWMA of the
    // signed error and the weight (w starts at 1), and the masked mean's sums.
    let maskCount = 0;
    for (let i = 0; i < maskF.length; i += 1) maskCount += maskF[i];
    this.errInvCount = 1 / Math.max(1, maskCount);
    const ema0 = new Float32Array(width * height * 4);
    for (let i = 3; i < ema0.length; i += 4) ema0[i] = 1;
    this.errEmaBuf = device.createBuffer({ size: ema0.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(this.errEmaBuf, 0, ema0);
    this.errEmaSumBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.errSlot = 0;
    this.errKeep = 0;

    // The run starts from startSites (or the loaded sites), plus the layer
    // (LAYER_MODES) after them; the buffers hold the most the count schedule
    // ever wants. Unused slots are inactive (position.x < 0), which every
    // kernel skips. The schedule runs over the ordinary sites, and the layer
    // counts against startSites and endSites alike (countArgs), so start =
    // end holds the count.
    const useLayer = args.layer === "learn";
    let baseSiteCount;
    if (tiling) {
      // Every tile's sitesPerTile slots are placed at once; the layer is the
      // last slot of each tile (kernels map the slot to the pixel's tile).
      if (initSites) throw new Error("Batched training starts from the gradient init, not init sites");
      if (countVaries(args, args.startSites) || args.turnover > 0) {
        throw new Error("Batched training needs a fixed site count (start sites = end sites, no sawtooth, no turnover)");
      }
      const n = tiling.sitesPerTile;
      if (useLayer) this.histPeaks = tilePeaks(targetRgba32f, this.maskCpu, width, tiling);
      this.layerIndex = useLayer ? n - 1 : null;
      this.countArgs = args;
      this.usesGrow = false;
      this.bufferCapacity = initSiteCount = baseSiteCount = tiling.tiles * n;
    } else {
      if (!initSites) initSiteCount = useLayer ? Math.max(1, args.startSites - 1) : args.startSites;
      if (useLayer) this.histPeak = colorPeak(targetRgba32f, width * height, this.maskCpu);
      baseSiteCount = initSiteCount;
      this.layerIndex = useLayer ? baseSiteCount : null;
      if (useLayer) initSiteCount += 1;
      this.countArgs = useLayer ? { ...args, endSites: Math.max(1, args.endSites - 1) } : args;
      this.usesGrow = countVaries(this.countArgs, baseSiteCount) || args.turnover > 0;
      this.bufferCapacity = (this.usesGrow ? countCapacity(this.countArgs, baseSiteCount) : baseSiteCount) + (useLayer ? 1 : 0);
      // Turnover's merges leave holes and its splits append, so leave room for
      // 8 steps between compactions (_growTo).
      if (args.turnover > 0) this.bufferCapacity += 8 * Math.max(1, Math.ceil(args.turnover * this.bufferCapacity));
    }
    this.siteCount = initSiteCount;

    // Buffers
    this.sitesBuf = createSitesBuffer(device, this.bufferCapacity);
    this.adamBuf = createAdamBuffer(device, this.bufferCapacity);
    this.seedCounter = createSeedCounter(device, tiling ? tiling.tiles : 1);
    this.grads = createGradsBuffer(device, this.bufferCapacity);

    // Candidate lists (candidates.js)
    this.cands = new CandidateLists(device, this.shaders, { ...args, width, height, capacity: this.bufferCapacity, tiling });
    this.candWidth = this.cands.candWidth;
    this.candHeight = this.cands.candHeight;

    // Render texture and param buffers
    this.renderTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false, copySrc: true });
    this.renderParams = createParamsBuffer(device, this._paramsArgs(initSiteCount));
    this.invScaleSq = invScaleSqFor(width, height, tiling);
    this.gradParams = createGradParamsBuffer(device, initSiteCount, this.invScaleSq, this._globalCand(), this._gradEma(), tiling);
    this.adamParams = createAdamParamsBuffer(device, {
      ...this._adamSize(), layerSite: this._globalCand(), colorMoment: COLOR_MOMENTS[args.colorMoment].id,
    });
    this.gradCountParams = createClearParamsBuffer(device, this.bufferCapacity * 10);

    if (this.bufferCapacity > initSiteCount) {
      const inactive = new Float32Array((this.bufferCapacity - initSiteCount) * SITE_FLOATS);
      for (let i = 0; i < inactive.length; i += SITE_FLOATS) { inactive[i] = -1; inactive[i + 1] = -1; }
      device.queue.writeBuffer(this.sitesBuf, initSiteCount * SITE_FLOATS * 4, inactive);
    }

    // Site growth: per-site statistics, and the predicted vs realized gain of
    // each batch (diagnostic only; never feeds back into growth).
    if (this.usesGrow) {
      this.siteStatsParams = createUniform32(device);
      this.siteStatsBuf = device.createBuffer({
        size: this.bufferCapacity * SITE_STAT_CH * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.growPending = null;
      this.growGainHistory = [];
    }

    // Initialize sites
    if (initSites) {
      const maxBytes = this.bufferCapacity * SITE_FLOATS * 4;
      const src = initSites.byteLength <= maxBytes
        ? initSites
        : new Float32Array(initSites.buffer, initSites.byteOffset, maxBytes / 4);
      device.queue.writeBuffer(this.sitesBuf, 0,
        encodeSiteColors(src.slice(0, baseSiteCount * SITE_FLOATS), baseSiteCount, this.colorSpace));
      if (useLayer) this._writeLayerSite();
      this.actualSites = initSiteCount;
      this.activeEstimate = useLayer ? 1 : 0;
      for (let i = 0; i < baseSiteCount; i += 1) {
        if (initSites[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Loaded init sites: ${baseSiteCount} (active ${this.activeEstimate})`);
    } else {
      const gradThreshold = 0.01;
      const initParamsBuf = createInitParamsBuffer(device, baseSiteCount, gradThreshold, 256,
                                                    args.initLogTau, args.initRadius, tiling,
                                                    tiling ? this._globalCand() : undefined);
      const encoder = device.createCommandEncoder();
      this.initEnc.encode(encoder, this.sitesBuf, this.seedCounter, initParamsBuf,
                           this.targetTex, this.maskTex, baseSiteCount);
      device.queue.submit([encoder.finish()]);
      if (useLayer) this._writeLayerSite();
      const sitesData = await readSites(device, this.sitesBuf, initSiteCount);
      initParamsBuf.destroy();
      this.actualSites = initSiteCount;
      this.activeEstimate = 0;
      for (let i = 0; i < initSiteCount; i += 1) {
        if (sitesData[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Initialized sites: ${initSiteCount} (active ${this.activeEstimate})`);
    }
    if (useLayer && tiling) {
      this._log(`Histogram layer: slot ${this.layerIndex} of each of ${tiling.count} tiles, from each image's own peak`);
    } else if (useLayer) {
      const c = this.histPeak.color.map((v) => Math.round(v * 255)).join(", ");
      this._log(`Histogram layer: rgb(${c}) covers ${(100 * this.histPeak.mass).toFixed(1)}% of pixels → ` +
                `site ${this.layerIndex}, depth β starts at ${this._layerBeta0().toFixed(2)}`);
    }

    // Candidate lists for the initial sites.
    this._rebuildCandidates(initSiteCount);

    // Zero the gradient accumulators once; Adam re-zeroes each site after use.
    {
      const encoder = device.createCommandEncoder();
      this.clearI32Enc.encode(encoder, this.grads, this.gradCountParams, initSiteCount * 10);
      device.queue.submit([encoder.finish()]);
    }

    // Growth state: the iteration it ended (null while growing) and why.
    this.growDoneIter = this.usesGrow ? null : 0;
    this.growStop = null;
    this.growStartCount = baseSiteCount; // where the count schedule starts (ordinary sites)
    this.heldCount = null; // active sites turnover keeps once the schedule ends

    this.trainStartMs = performance.now();
    this.bestPsnr = 0;
  }

  // Candidate lists from scratch for sites [0, n) (CandidateLists.rebuild).
  _rebuildCandidates(n) {
    this.cands.globalCand = this._globalCand();
    this.cands.rebuild(this.sitesBuf, n);
  }

  // Params (sad_shared.wgsl) for `siteCount` sites, for render passes.
  _paramsArgs(siteCount) {
    return this.cands.paramsArgs(siteCount);
  }

  // The site every pixel blends besides its candidate list (cand_ids in
  // sad_shared.wgsl): the histogram layer, or none.
  _globalCand() { return this.layerIndex ?? 0xffffffff; }

  // Site count, gradient and Adam params after the site count or the layer's
  // index changed.
  _updateCountParams(n) {
    const queue = this.device.queue;
    updateGradParamsBuffer(queue, this.gradParams, n, this.invScaleSq, this._globalCand(), this._gradEma(), this.tiling);
    this.cands.globalCand = this._globalCand();
  }

  // GradParams' persistent-error fields for the next gradient pass.
  _gradEma() {
    const a = this.args;
    return { slot: this.errSlot, keep: this.errKeep, alpha: a.errWeight, cap: a.errCap, invCount: this.errInvCount };
  }

  // AdamParams' size fields: the image's, or a tile's with the tiling.
  _adamSize() {
    const t = this.tiling;
    return t ? { width: t.tileW, height: t.tileH, tiling: t } : { width: this.width, height: this.height };
  }

  _log(msg) { if (this.onLog) this.onLog({ type: "msg", message: msg }); }

  // The histogram layer as a site at layerIndex: the peak color, logit
  // -e^LEARN_LOG_TAU · (d - r) ≈ β everywhere (isotropic, image center).
  // Tiled: one per image tile, at its slot of the tile, from the tile's peak.
  _writeLayerSite() {
    const t = this.tiling;
    const tiles = t ? t.count : 1;
    const data = new Float32Array(SITE_FLOATS);
    for (let b = 0; b < tiles; b += 1) {
      const peak = t ? this.histPeaks[b] : this.histPeak;
      const ox = t ? (b % t.tilesX) * t.tileW : 0, oy = t ? Math.floor(b / t.tilesX) * t.tileH : 0;
      const w = t ? t.tileW : this.width, h = t ? t.tileH : this.height;
      const site = data;
      site[0] = ox + (w - 1) / 2;
      site[1] = oy + (h - 1) / 2;
      site[2] = LEARN_LOG_TAU;
      // Logit -τ(d - r·inv_scale) at d = 0 is the depth β.
      site[3] = this._layerBeta0(peak) * Math.max(w, h) / Math.exp(LEARN_LOG_TAU);
      site.set(peak.color, 4);
      site[7] = 1;
      site[8] = 0;
      site[9] = 0;
      const idx = t ? b * t.sitesPerTile + this.layerIndex : this.layerIndex;
      this.device.queue.writeBuffer(this.sitesBuf, idx * SITE_FLOATS * 4, encodeSiteColors(site, 1, this.colorSpace));
    }
  }

  // The layer's starting depth: the log-odds of its color's peak mass. The
  // start carries the prior: on MNIST, starting at β = -6 instead of from p
  // cost 1.4 dB at 32 sites (β only climbed back to -4, vs +3.4).
  _layerBeta0(peak = this.histPeak) {
    const p = Math.min(1 - 1e-3, Math.max(1e-3, peak.mass));
    return Math.max(-LAYER_BETA_MAX, Math.min(LAYER_BETA_MAX, Math.log(p / (1 - p))));
  }

  async run() {
    const args = this.args;
    const iters = args.iters;
    const logFreq = args.logFreq;
    const viewerFreq = Math.max(1, args.viewerFreq || 10);
    const candUpdateFreq = Math.max(1, args.candFreq);
    const device = this.device;
    const queue = device.queue;

    for (let it = 0; it < iters; it += 1) {
      if (this.signal && this.signal.aborted) {
        this._log("Training aborted.");
        break;
      }

      if (this.usesGrow) await this._countStep(it);
      const siteCount = this.siteCount;
      const activeSites = this.activeEstimate;

      const lrScale = args.lrMult * lrSchedule(it, args.iters, args.lrSchedule, this.growDoneIter ?? Infinity);
      updateAdamParamsBuffer(queue, this.adamParams, {
        lrPos: args.lrPosBase * lrScale, lrTau: args.lrTauBase * lrScale, lrRadius: args.lrRadiusBase * lrScale,
        lrColor: args.lrColorBase * lrScale,
        lrDir: it >= ANI_START_ITER ? args.lrDirBase * lrScale : 0,
        lrAniso: it >= ANI_START_ITER ? args.lrAnisoBase * lrScale : 0,
        beta1: args.beta1, beta2: args.beta2, eps: args.eps, t: it + 1,
        ...this._adamSize(), layerSite: this._globalCand(), colorMoment: COLOR_MOMENTS[args.colorMoment].id,
      });

      const shouldLog = (it % logFreq === 0) || (it === iters - 1);
      const shouldPreview = shouldLog || (it % viewerFreq === 0);
      if (shouldLog || shouldPreview) {
        updateParamsBuffer(queue, this.renderParams, this._paramsArgs(siteCount));
      }

      // The error EWMA's decay, ramped in so step t averages its t + 1
      // residuals until that reaches errBeta.
      this.errKeep = Math.min(args.errBeta, it / (it + 1));
      updateGradParamsBuffer(queue, this.gradParams, siteCount, this.invScaleSq, this._globalCand(), this._gradEma(), this.tiling);

      const encoder = device.createCommandEncoder();
      if (it % candUpdateFreq === 0) this.cands.encodeUpdate(encoder, this.sitesBuf, siteCount);
      encoder.clearBuffer(this.errEmaSumBuf, (1 - this.errSlot) * 4, 4);
      this.gradsEnc.encode(encoder, this.sitesBuf, this.gradParams,
        this.cands.cand0, this.cands.cand1, this.targetTex, this.maskBuf,
        this.grads, this.errEmaBuf, this.errEmaSumBuf, this.width, this.height);
      this.errSlot = 1 - this.errSlot;

      // Snapshot the accumulated gradients on log ticks — Adam zeroes them.
      let gradStaging = null;
      if (shouldLog) {
        const gradBytes = Math.max(1, siteCount) * 10 * 4;
        gradStaging = device.createBuffer({
          size: gradBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        encoder.copyBufferToBuffer(this.grads, 0, gradStaging, 0, gradBytes);
      }

      this.adamEnc.encode(encoder, this.sitesBuf, this.adamBuf, this.grads, this.adamParams, siteCount);

      // Preview/log renders. The log path does a blocking readback for PSNR.
      // Viewer ticks wait only for the small site readback so centroid overlays
      // stay on the requested cadence; full image readback stays asynchronous.
      let previewReadback = null;
      if (shouldPreview) {
        this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
          this.cands.cand0, this.cands.cand1, this.renderTex, this.width, this.height);
        if (!shouldLog) {
          previewReadback = this._encodePreviewReadback(encoder, it, iters, activeSites, siteCount);
        }
      }
      queue.submit([encoder.finish()]);

      if (shouldLog) {
        const rgba = await readRgba32FloatTexture(device, this.renderTex, this.width, this.height);
        const sitesData = await this._readSitesRgb(siteCount);
        this._lastSitesCpu = sitesData;
        await gradStaging.mapAsync(GPUMapMode.READ);
        const grads = new Float32Array(gradStaging.getMappedRange()).slice();
        const gradStats = computeGradStats(grads, siteCount, this.width * this.height);
        gradStaging.unmap();
        gradStaging.destroy();
        const psnr = computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum);
        this.bestPsnr = Math.max(this.bestPsnr, psnr);
        const elapsed = (performance.now() - this.trainStartMs) / 1000;
        const itsPerSec = (it + 1) / Math.max(elapsed, 1e-6);
        if (this.onLog) {
          this.onLog({
            type: "iter",
            it, iters,
            lrScale,
            psnr,
            gradStats,
            grads,
            activeSites,
            totalSites: siteCount,
            growDoneIter: this.growDoneIter,
            growStop: this.growStop,
            itsPerSec,
            elapsedSec: elapsed,
            renderRgba: rgba,
            sitesData,
            layerIndex: this.layerIndex,
            width: this.width,
            height: this.height,
          });
        }
        await yieldToBrowser();
      } else if (shouldPreview) {
        const previewDone = previewReadback ? this._finishPreviewReadback(previewReadback) : null;
        if (previewDone?.site) await previewDone.site;
        await yieldToBrowser();
      } else if (((it + 1) & 0x1f) === 0) {
        await yieldToBrowser();
      }
    }

    updateParamsBuffer(queue, this.renderParams, this._paramsArgs(this.actualSites));
    const encoder = device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cands.cand0, this.cands.cand1, this.renderTex, this.width, this.height);
    queue.submit([encoder.finish()]);
  }

  // ---- Site count: growth and merging ----
  // Every growEvery iterations the count moves to the schedule's value
  // (countTarget, _countStep): down by merging (_mergeTo), up in a growth
  // batch (_growTo). Growth scores each site by the error a two-way split of
  // its cell would remove (siteSplit, from site_stats.wgsl's moments) and
  // splits the best cells, one new site each. Only splits whose gain beats
  // λ·BITS_PER_SITE (squared error; λ = 0 only requires a positive gain)
  // qualify; when none does, the count schedule ends there.

  // Per-site stats of the current state (site_stats.wgsl): Float32Array of
  // SITE_STAT_CH per site, for sites [0, siteCount).
  async _siteStats() {
    const device = this.device;
    const n = this.siteCount;
    const u = new Uint32Array(8);
    u[0] = n;
    u[1] = this._globalCand();
    u[2] = this.args.errGrowth && this.args.errWeight > 0 ? 1 : 0;
    new Float32Array(u.buffer)[4] = this.invScaleSq;
    device.queue.writeBuffer(this.siteStatsParams, 0, u);
    const bytes = n * SITE_STAT_CH * 4;
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    this.statsEnc.encode(encoder, this.sitesBuf, this.siteStatsParams, this.cands.cand0, this.cands.cand1,
      this.targetTex, this.maskBuf, this.siteStatsBuf, this.errEmaBuf, this.width, this.height);
    encoder.copyBufferToBuffer(this.siteStatsBuf, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return out;
  }

  // Diagnostic: realized gain of the last batch's sites = the error that would
  // come back if each were removed now, others unchanged (in 8-bit levels²,
  // like the predicted gain). Reported, never used for decisions.
  _reportGrowthGain(it, stats) {
    const batch = this.growPending;
    if (!batch) return;
    this.growPending = null;
    const points = batch.sites.map(({ id, predicted }) => ({
      site: id, predicted, realized: stats[id * SITE_STAT_CH + 17] * 255 * 255,
    }));
    const entry = { step: this.growGainHistory.length + 1, addedAt: batch.it, measuredAt: it, points };
    this.growGainHistory.push(entry);
    if (this.onLog) this.onLog({ type: "growthGain", ...entry });
  }

  _endGrowth(it, reason) {
    this.growDoneIter = it;
    this.heldCount = this.activeEstimate;
    this.growStop = reason;
    this._log(`Growth done at iter ${it}: ${this.activeEstimate} sites (${reason}).`);
  }

  // Split growth: the `count` cells whose two-way split (siteSplit) gains most,
  // above `threshold`. Rewrites the split parents in place and returns the new
  // sites (RGB) as { data, k, predicted, best }, or null.
  async _splitBatch(it, stats, n0, count, threshold) {
    const queue = this.device.queue;
    const sitesRaw = await readSites(this.device, this.sitesBuf, n0);
    const splits = [];
    for (let i = 0; i < n0; i += 1) {
      if (sitesRaw[i * SITE_FLOATS] < 0 || i === this.layerIndex) continue;
      const sp = siteSplit(stats, i);
      if (sp && sp.gain > threshold) splits.push(sp);
    }
    splits.sort((x, y) => y.gain - x.gain);
    splits.length = Math.min(splits.length, count);
    if (!splits.length) return null;

    // Each split: the existing site moves to one half's centroid with that
    // half's ramp color, the new site takes the other half with an exact copy
    // of the site's shape (τ, radius, anisotropy), so the boundary between the
    // two lands on the split line. Both get fresh Adam moments.
    const k = splits.length;
    const data = new Float32Array(k * SITE_FLOATS);
    const adam = new Float32Array(ADAM_FLOATS);
    adam[20] = it;
    const clampPos = (v, max) => Math.min(max - 1, Math.max(0, v));
    splits.forEach((sp, j) => {
      const b = sp.site * SITE_FLOATS;
      const px = sitesRaw[b], py = sitesRaw[b + 1];
      const old = sitesRaw.slice(b, b + SITE_FLOATS);
      old[0] = clampPos(px + sp.oldOffset[0], this.width);
      old[1] = clampPos(py + sp.oldOffset[1], this.height);
      old.set(sp.oldColor, 4);
      const o = j * SITE_FLOATS;
      data.set(sitesRaw.subarray(b, b + SITE_FLOATS), o);
      data[o] = clampPos(px + sp.newOffset[0], this.width);
      data[o + 1] = clampPos(py + sp.newOffset[1], this.height);
      data.set(sp.newColor, o + 4);
      // Rewrite the existing site whole: its stored colors are in the run's
      // color space, so re-encode from RGB like the new site.
      queue.writeBuffer(this.sitesBuf, sp.site * SITE_FLOATS * 4, encodeSiteColors(old, 1, this.colorSpace));
      queue.writeBuffer(this.adamBuf, sp.site * ADAM_FLOATS * 4, adam);
    });
    return { data, k, predicted: splits.map((sp) => sp.gain), best: splits[0].gain };
  }

  // The candidate lists as 8 u32 site ids per candidate pixel.
  _readCandidates() { return this.cands.read(); }

  // Count step down: merge neighbouring sites (merge.js, args.mergeCriterion)
  // until `target` sites are left. "neighbor" merges by parameter similarity
  // along the candidate lists' edges, chaining merges, so one step can reach
  // any target; "ward" takes the cheapest pairs by Ward's criterion, each site
  // in at most one merge, so a step removes at most half of them. Each kept
  // site becomes the weighted average of what merged into it (mergedSite;
  // per-parameter precision or pixel counts, args.mergeWeights), each dropped
  // site is deactivated, both get fresh Adam moments, and the candidate lists
  // are rebuilt. Returns the number merged.
  async _mergeTo(it, target) {
    const n = this.siteCount;
    const args = this.args;
    const sites = await this._readSitesRgb(n);
    const cand = await this._readCandidates();
    const pinned = new Set(this.layerIndex === null ? [] : [this.layerIndex]);
    const budget = this.activeEstimate - target;
    const adamState = args.mergeWeights === "precision" ? await this._readAdam(n) : null;
    // [site, RGB params or null to deactivate]
    const writes = [];
    let merged = 0;
    if (args.mergeCriterion === "ward") {
      const moments = cellMoments({
        width: this.width, height: this.height, sites, n, cand,
        candWidth: this.candWidth, candHeight: this.candHeight, target: this.target, mask: this.maskCpu,
      });
      const merges = pickMerges(moments, budget, pinned);
      for (const { keep, drop } of merges) {
        const [wk, wd] = adamState
          ? mergePrecision(adamState, keep, drop, it, args.beta2, moments.count[keep], moments.count[drop])
          : [moments.count[keep], moments.count[drop]];
        writes.push([keep, mergedSite(sites, keep, drop, wk, wd)], [drop, null]);
      }
      merged = merges.length;
    } else {
      const graph = candidateEdges({
        sites, n, cand, candWidth: this.candWidth, candHeight: this.candHeight,
        width: this.width, height: this.height, mask: this.maskCpu, pinned,
      });
      let weights = null;
      if (adamState) {
        weights = new Float64Array(n * SITE_FLOATS);
        for (let i = 0; i < n; i += 1) weights.set(sitePrecision(adamState, i, it, args.beta2), i * SITE_FLOATS);
      }
      const res = neighborMerge({ sites, n, graph, budget, topK: args.mergeTopK, weights });
      for (const i of res.changed) writes.push([i, res.sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS)]);
      for (const i of res.dropped) writes.push([i, null]);
      merged = res.merges.length;
      if (this.onLog) {
        this.onLog({
          type: "mergeEdges", it, groups: EDGE_GROUPS, scales: res.scales, edgeWeights: res.edgeWeights,
          mergedWeights: Float32Array.from(res.merges, (m) => m.weight),
        });
      }
    }
    if (!merged) return 0;
    const queue = this.device.queue;
    const adam = new Float32Array(ADAM_FLOATS);
    adam[20] = it;
    const inactive = new Float32Array(SITE_FLOATS);
    inactive[0] = -1;
    inactive[1] = -1;
    for (const [i, params] of writes) {
      queue.writeBuffer(this.sitesBuf, i * SITE_FLOATS * 4, params ? encodeSiteColors(params, 1, this.colorSpace) : inactive);
      queue.writeBuffer(this.adamBuf, i * ADAM_FLOATS * 4, adam);
    }
    this.activeEstimate -= merged;
    this._rebuildCandidates(this.siteCount);
    this._log(`Merge @${it}: ${merged} site(s) merged (${args.mergeCriterion}) → ${this.activeEstimate} sites (schedule wants ${target})`);
    return merged;
  }

  // Adam state of sites [0, count): ADAM_FLOATS floats per site (adam.wgsl).
  async _readAdam(count) {
    const device = this.device;
    const bytes = count * ADAM_FLOATS * 4;
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.adamBuf, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const adam = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return adam;
  }

  // Moves the active sites (and their Adam moments) to the front, in order,
  // marks the rest inactive, and rebuilds the candidate lists.
  async _compactSites() {
    const device = this.device;
    const n = this.siteCount;
    const sites = await readSites(device, this.sitesBuf, n);
    const adam = await this._readAdam(n);
    const outSites = new Float32Array(n * SITE_FLOATS);
    const outAdam = new Float32Array(n * ADAM_FLOATS);
    let m = 0;
    let layer = null;
    for (let i = 0; i < n; i += 1) {
      if (!(sites[i * SITE_FLOATS] >= 0)) continue;
      outSites.set(sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS), m * SITE_FLOATS);
      outAdam.set(adam.subarray(i * ADAM_FLOATS, (i + 1) * ADAM_FLOATS), m * ADAM_FLOATS);
      if (i === this.layerIndex) layer = m;
      m += 1;
    }
    this.layerIndex = layer;
    for (let i = m; i < n; i += 1) { outSites[i * SITE_FLOATS] = -1; outSites[i * SITE_FLOATS + 1] = -1; }
    device.queue.writeBuffer(this.sitesBuf, 0, outSites);
    device.queue.writeBuffer(this.adamBuf, 0, outAdam);
    this.siteCount = this.actualSites = this.activeEstimate = m;
    this._updateCountParams(m);
    this.growPending = null; // site ids changed
    this._rebuildCandidates(m);
  }

  // One count step (every growEvery iterations, until the schedule's horizon):
  // grow or merge to the schedule's count (countTarget), then end the
  // schedule once the horizon is reached. Merging goes on past the horizon
  // until the count is down to endSites: a sawtooth with a big amplitude and
  // a short merge phase can't get back down in time (a "ward" step merges at most
  // half the sites), and stopping there kept up to 60% more sites than asked.
  // Turnover steps (args.turnoverEvery) where the schedule holds the count,
  // and all of them after it ends, turn sites over (_turnover).
  async _countStep(it) {
    const args = this.args;
    if (this.growDoneIter !== null) {
      if (this._turnsOver(it)) {
        await this._turnover(it, this.heldCount);
        return;
      }
      // One more measurement for the final batch, a growth step after it.
      if (this.growPending && it >= this.growPending.it + args.growEvery) {
        this._reportGrowthGain(it, await this._siteStats());
      }
      return;
    }
    const step = it > 0 && it % args.growEvery === 0;
    if (!step && !this._turnsOver(it)) return;
    // In active sites, the layer included.
    const layers = this.layerIndex === null ? 0 : 1;
    const target = layers + Math.max(1, Math.round(countTarget(this.countArgs, it, this.growStartCount)));
    let merged = 0;
    if (step && target < this.activeEstimate) merged = await this._mergeTo(it, target);
    else if (step && target > this.activeEstimate) await this._growTo(it, target);
    else if (target === this.activeEstimate && this._turnsOver(it)) await this._turnover(it, target);
    const catchingUp = this.activeEstimate > target && merged > 0;
    if (step && it >= countHorizon(args) && !catchingUp) {
      this._endGrowth(it, `count schedule done (${args.countSchedule}, ${layers + this.growStartCount} → ${args.endSites})`);
    }
  }

  // Whether iteration `it` is a turnover step.
  _turnsOver(it) {
    const args = this.args;
    return args.turnover > 0 && it > 0 && it % args.turnoverEvery === 0 && it < args.turnoverEnd * args.iters;
  }

  // Turnover at a held count: merge args.turnover of the ordinary sites (at
  // least one; _mergeTo, args.mergeCriterion), then split cells (_growTo,
  // any positive gain, λ ignored) back up to `target` active sites, so sites
  // leave where they're redundant and reappear where the error is. A step
  // that splits fewer than it merged leaves the rest to the next one.
  async _turnover(it, target) {
    const args = this.args;
    const layers = this.layerIndex === null ? 0 : 1;
    const k = Math.max(1, Math.round(args.turnover * (this.activeEstimate - layers)));
    if (this.activeEstimate - k > layers) await this._mergeTo(it, this.activeEstimate - k);
    if (target > this.activeEstimate) await this._growTo(it, target, { turnover: true });
  }

  // Count step up: split cells until `target` sites are active, unless λ
  // ends the schedule first. With `turnover`, any positive gain qualifies and
  // finding none doesn't end the schedule.
  async _growTo(it, target, { turnover = false } = {}) {
    const args = this.args;
    // Merged-away sites leave inactive slots; make room at the end first.
    if (this.siteCount + (target - this.activeEstimate) > this.bufferCapacity) await this._compactSites();
    const n0 = this.siteCount;
    const want = Math.min(target - this.activeEstimate, this.bufferCapacity - n0);
    if (want <= 0) return;
    const stats = await this._siteStats();
    this._reportGrowthGain(it, stats);

    const device = this.device;
    const queue = device.queue;
    const threshold = turnover ? 0 : args.growLambda * BITS_PER_SITE;
    const batch = await this._splitBatch(it, stats, n0, want, threshold);
    if (!batch && turnover) return;
    if (!batch) {
      this._endGrowth(it, `λ (stop): no split would remove more than λ × ${BITS_PER_SITE} = ` +
                          `${fmtLambda(threshold)} levels²`);
      return;
    }
    const { data, k } = batch;
    queue.writeBuffer(this.sitesBuf, n0 * SITE_FLOATS * 4, encodeSiteColors(data, k, this.colorSpace));
    const adamNew = new Float32Array(k * ADAM_FLOATS);
    for (let j = 0; j < k; j += 1) adamNew[j * ADAM_FLOATS + 20] = it;
    queue.writeBuffer(this.adamBuf, n0 * ADAM_FLOATS * 4, adamNew);
    this.growPending = { it, sites: batch.predicted.map((predicted, j) => ({ id: n0 + j, predicted })) };

    const n1 = n0 + k;
    this.siteCount = n1;
    this.actualSites = n1;
    this.activeEstimate += k;
    this._updateCountParams(n1);

    // Seed the new sites and run a jump flood so they claim their regions now;
    // left to the regular updates (1 px steps by then) they would barely spread
    // before the next step, which would then see the same error in the same places.
    const seedEnc = device.createCommandEncoder();
    this.cands.encodeAdded(seedEnc, this.sitesBuf, n0, n1);
    queue.submit([seedEnc.finish()]);

    this._log(`${turnover ? "Turnover" : "Grow"} @${it}: +${k} → ${this.activeEstimate} sites` +
              (k < want ? ` (${want} wanted; the rest under λ` +
                (want > this.activeEstimate - k ? " or no cell left to split)" : ")") : "") +
              ` | best split ${fmtLambda(batch.best / BITS_PER_SITE)} levels²/bit`);
  }

  // Encode preview readbacks into the same submission as the preview render.
  // Site readbacks are separated from image readbacks so centroid overlays can
  // be awaited cheaply while full-image readback continues in the background.
  _encodePreviewReadback(encoder, it, iters, postActive, postSites) {
    const device = this.device;
    const width = this.width;
    const height = this.height;
    const bytesPerPixel = 16;
    const bytesPerRow = width * bytesPerPixel;
    const aligned = Math.floor((bytesPerRow + 255) / 256) * 256;
    const imageBytes = aligned * height;
    const siteBytes = postSites * SITE_FLOATS * 4;
    const meta = {
      it,
      iters,
      postActive,
      postSites,
      width,
      height,
      bytesPerRow,
      aligned,
      startMs: this.trainStartMs,
    };

    let imageReq = null;
    let siteReq = null;

    if (this._previewSiteInFlight < MAX_PREVIEW_SITE_READBACKS) {
      this._previewSiteInFlight += 1;
      const siteStaging = device.createBuffer({
        size: siteBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(this.sitesBuf, 0, siteStaging, 0, siteBytes);
      siteReq = { ...meta, siteStaging };
    }

    if (this._previewImageInFlight < MAX_PREVIEW_IMAGE_READBACKS) {
      this._previewImageInFlight += 1;
      const imageStaging = device.createBuffer({
        size: imageBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyTextureToBuffer(
        { texture: this.renderTex },
        { buffer: imageStaging, bytesPerRow: aligned, rowsPerImage: height },
        [width, height, 1],
      );
      imageReq = { ...meta, imageStaging };
    }

    return imageReq || siteReq ? { imageReq, siteReq } : null;
  }

  _finishPreviewReadback(req) {
    const site = req.siteReq ? this._finishPreviewSiteReadback(req.siteReq) : null;
    const image = req.imageReq ? this._finishPreviewImageReadback(req.imageReq) : null;
    return { site, image };
  }

  _finishPreviewSiteReadback(req) {
    return req.siteStaging.mapAsync(GPUMapMode.READ).then(async () => {
      const sitesData = decodeSiteColors(new Float32Array(req.siteStaging.getMappedRange()).slice(),
        req.postSites, this.colorSpace);
      req.siteStaging.unmap();
      req.siteStaging.destroy();
      this._lastSitesCpu = sitesData;
      this._lastHashedRgba = null;
      this._lastHeatmapRgba = null;
      if (this.onLog) {
        const elapsed = (performance.now() - req.startMs) / 1000;
        const itsPerSec = (req.it + 1) / Math.max(elapsed, 1e-6);
        await this.onLog({
          type: "sites",
          it: req.it,
          iters: req.iters,
          activeSites: req.postActive,
          totalSites: req.postSites,
          itsPerSec,
          elapsedSec: elapsed,
          sitesData,
          width: req.width,
          height: req.height,
        });
      }
      this._previewSiteInFlight -= 1;
    }).catch((err) => {
      req.siteStaging.destroy();
      this._previewSiteInFlight -= 1;
      console.warn("Preview site readback failed", err);
    });
  }

  _finishPreviewImageReadback(req) {
    req.imageStaging.mapAsync(GPUMapMode.READ).then(() => {
      const mapped = new Uint8Array(req.imageStaging.getMappedRange());
      const rgba = new Float32Array(req.width * req.height * 4);
      for (let y = 0; y < req.height; y += 1) {
        const row = mapped.subarray(y * req.aligned, y * req.aligned + req.bytesPerRow);
        rgba.set(new Float32Array(row.buffer, row.byteOffset, req.width * 4), y * req.width * 4);
      }
      req.imageStaging.unmap();
      req.imageStaging.destroy();
      if (this.onLog) {
        const elapsed = (performance.now() - req.startMs) / 1000;
        const itsPerSec = (req.it + 1) / Math.max(elapsed, 1e-6);
        this.onLog({
          type: "preview",
          it: req.it,
          iters: req.iters,
          activeSites: req.postActive,
          totalSites: req.postSites,
          itsPerSec,
          elapsedSec: elapsed,
          renderRgba: rgba,
          width: req.width,
          height: req.height,
        });
      }
      this._previewImageInFlight -= 1;
    }).catch((err) => {
      req.imageStaging.destroy();
      this._previewImageInFlight -= 1;
      console.warn("Preview image readback failed", err);
    });
  }

  // On-demand render using the current site + candidate state. `mode` is one
  // of "render" (color), "hashed" (cell-id hashColor blend), "heatmap"
  // (tau centroid heatmap) or "owner" (largest-weight site id in r, -1 for
  // none).
  async renderView(mode) {
    const device = this.device;
    const queue = device.queue;
    updateParamsBuffer(queue, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));

    const encoder = device.createCommandEncoder();
    if (mode === "hashed" || mode === "owner") {
      (mode === "owner" ? this.renderOwnerEnc : this.renderIdsEnc).encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cands.cand0, this.cands.cand1, this.renderTex, this.width, this.height);
    } else if (mode === "heatmap") {
      const heat = this._taskHeatmapParamsBuffer();
      this.renderTauEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cands.cand0, this.cands.cand1, this.renderTex,
        heat, this.width, this.height);
    } else {
      this.renderEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cands.cand0, this.cands.cand1, this.renderTex, this.width, this.height);
    }
    queue.submit([encoder.finish()]);
    return await readRgba32FloatTexture(device, this.renderTex, this.width, this.height);
  }

  // The persistent-error state (gradients_tiled.wgsl) as of the last
  // gradient pass: Float32Array of W·H·4, per pixel the EWMA ē of the signed
  // error (pred − target, RGB) and the loss weight w. Null once destroyed.
  async readErrEma() {
    if (this._destroyed) return null;
    const bytes = this.width * this.height * 16;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.errEmaBuf, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return out;
  }

  _taskHeatmapParamsBuffer() {
    // Compute tau min / mean / max from the last-read sites before every
    // heatmap render, as the former Metal backend did.
    if (!this._tauHeatBuf) {
      this._tauHeatBuf = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    let minT = Infinity;
    let maxT = -Infinity;
    let sumT = 0;
    let count = 0;
    const data = this._lastSitesCpu;
    if (data) {
      for (let i = 0; i < this.actualSites; i += 1) {
        const base = i * SITE_FLOATS;
        if (data[base] < 0) continue;
        const t = Math.exp(data[base + 2]);
        if (!Number.isFinite(t)) continue;
        if (t < minT) minT = t;
        if (t > maxT) maxT = t;
        sumT += t;
        count += 1;
      }
    }
    if (!Number.isFinite(minT)) { minT = 0; maxT = 1; sumT = 0.5; count = 1; }
    const meanT = sumT / Math.max(count, 1);
    const bytes = new ArrayBuffer(16);
    const f = new Float32Array(bytes);
    f[0] = minT;
    f[1] = meanT;
    f[2] = maxT;
    f[3] = 2.5; // dot radius in pixels; matches Metal's centroid_radius
    this.device.queue.writeBuffer(this._tauHeatBuf, 0, bytes);
    return this._tauHeatBuf;
  }

  setLastSites(data) { this._lastSitesCpu = data; }

  // ---- Site editing (viewer only; call between runs, never during run()) ----

  // Overwrites site `i` with `rgbSite` (SITE_FLOATS values, decoded colors)
  // and updates the candidate lists for it in place: the site is seeded at its
  // home pixel, then one jump-flood sequence (long jumps down to 1 px) spreads
  // it over the lists as they are, so the rest of the image keeps its lists.
  editSite(i, rgbSite) {
    const device = this.device;
    const queue = device.queue;
    const n = this.actualSites;
    if (!(i >= 0 && i < n)) throw new Error(`site ${i} out of range (0–${n - 1})`);
    queue.writeBuffer(this.sitesBuf, i * SITE_FLOATS * 4, encodeSiteColors(Float32Array.from(rgbSite), 1, this.colorSpace));
    const encoder = device.createCommandEncoder();
    this.cands.encodeAdded(encoder, this.sitesBuf, i, i + 1, n);
    queue.submit([encoder.finish()]);
  }

  // Viewer only: replaces every site with `rgbSites` (count · SITE_FLOATS,
  // decoded colors; inactive slots have x < 0) and rebuilds the candidate
  // lists, e.g. to show a saved snapshot of another run on this target.
  // `layerIndex` is the histogram layer's slot in it, or null. The buffers
  // must hold `count` sites (prepare with startSites = endSites ≥ count).
  showSites(rgbSites, count, layerIndex = null) {
    if (count > this.bufferCapacity) throw new Error(`${count} sites exceed the buffers (${this.bufferCapacity})`);
    this.device.queue.writeBuffer(this.sitesBuf, 0,
      encodeSiteColors(Float32Array.from(rgbSites.subarray(0, count * SITE_FLOATS)), count, this.colorSpace));
    this.siteCount = this.actualSites = count;
    this.activeEstimate = 0;
    for (let i = 0; i < count; i += 1) if (rgbSites[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
    this.layerIndex = layerIndex;
    if (layerIndex !== null) this.histPeak ??= colorPeak(this.target, this.width * this.height, this.maskCpu);
    this._updateCountParams(count);
    this._rebuildCandidates(count);
  }

  // Renders the current (edited) sites. Returns { rgba, psnr }, PSNR masked as
  // in training.
  async renderEdited() {
    const rgba = await this.renderView("render");
    return { rgba, psnr: computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum) };
  }

  // ---- Snap analysis (viewer only; call between runs, never during run()) ----

  // Per-site Gauss–Newton diagonal of the summed squared error over the masked
  // pixels: F = Float32Array(actualSites · 10) in site-parameter order, colors
  // per decoded RGB channel. Δ(Σ squared error) ≈ Σ_t F[t] · Δθ_t² for small
  // changes Δθ of one site's parameters, cross terms ignored. `samples` is the
  // number of squared errors PSNR averages (masked pixels × 3).
  async siteSensitivity() {
    const device = this.device;
    this.gnEnc ||= new GradientsEncoder(device, this.shaders, { gaussNewton: true });
    this._gnBuf ||= createGradsBuffer(device, this.bufferCapacity);
    const bytes = Math.max(1, this.actualSites) * 10 * 4;
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(this._gnBuf);
    this.gnEnc.encode(encoder, this.sitesBuf, this.gradParams,
      this.cands.cand0, this.cands.cand1, this.targetTex, this.maskBuf,
      this._gnBuf, this.errEmaBuf, this.errEmaSumBuf, this.width, this.height);
    encoder.copyBufferToBuffer(this._gnBuf, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const F = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    let pixels = this.width * this.height;
    if (this.maskCpu) { pixels = 0; for (const m of this.maskCpu) if (m > 0) pixels += 1; }
    return { F, samples: 3 * pixels };
  }

  // Renders with `rgbSites` (actualSites · SITE_FLOATS, decoded colors) in
  // place of the current sites, then puts the originals back. Candidate lists
  // stay those of the current sites. Returns { psnr, basePsnr }.
  async snapPsnr(rgbSites) {
    const queue = this.device.queue;
    const n = this.actualSites;
    const original = await readSites(this.device, this.sitesBuf, n);
    queue.writeBuffer(this.sitesBuf, 0, encodeSiteColors(rgbSites.slice(0, n * SITE_FLOATS), n, this.colorSpace));
    const snapped = await this.renderView("render");
    queue.writeBuffer(this.sitesBuf, 0, original);
    const base = await this.renderView("render");
    const psnr = (rgba) => computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum);
    return { psnr: psnr(snapped), basePsnr: psnr(base) };
  }

  // Renders with `rgbSites` (actualSites · SITE_FLOATS, decoded colors) as a
  // decoder would: candidate lists rebuilt from them, not kept from the
  // current sites. Then puts the originals back, rebuilds their lists the same
  // way and renders them for the baseline. Returns { psnr, basePsnr }.
  async decodedPsnr(rgbSites) {
    const queue = this.device.queue;
    const n = this.actualSites;
    const original = await readSites(this.device, this.sitesBuf, n);
    const psnr = (rgba) => computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum);
    queue.writeBuffer(this.sitesBuf, 0, encodeSiteColors(rgbSites.slice(0, n * SITE_FLOATS), n, this.colorSpace));
    this._rebuildCandidates(n);
    const decoded = psnr(await this.renderView("render"));
    queue.writeBuffer(this.sitesBuf, 0, original);
    this._rebuildCandidates(n);
    return { psnr: decoded, basePsnr: psnr(await this.renderView("render")) };
  }

  // Site readback with colors decoded to RGB, so nothing outside the trainer
  // sees color-space params.
  async _readSitesRgb(count) {
    const data = await readSites(this.device, this.sitesBuf, count);
    return decodeSiteColors(data, count, this.colorSpace);
  }

  async finalize() {
    const rgba = await readRgba32FloatTexture(this.device, this.renderTex, this.width, this.height);
    const sitesArray = await this._readSitesRgb(this.actualSites);
    return {
      renderRgba: rgba,
      sites: sitesArray,
      count: this.actualSites,
      globalSite: this.layerIndex ?? -1, // for serializeSitesTxt
      width: this.width,
      height: this.height,
      bestPsnr: this.bestPsnr,
      growDoneIter: this.growDoneIter,
      growStop: this.growStop,
      growthGain: this.growGainHistory || [],
      activeSites: this.activeEstimate,
      trainSeconds: (performance.now() - this.trainStartMs) / 1000,
      layer: await this._layerStats(sitesArray),
    };
  }

  // The histogram layer at the end of a run: { peakMass, beta (its depth),
  // area (fraction of masked pixels it owns) }, or null without one.
  async _layerStats(sitesRgb) {
    if (this.layerIndex === null || this.tiling) return null;
    const b = this.layerIndex * SITE_FLOATS;
    const moments = cellMoments({
      width: this.width, height: this.height, sites: sitesRgb, n: this.actualSites, cand: await this._readCandidates(),
      candWidth: this.candWidth, candHeight: this.candHeight, target: this.target, mask: this.maskCpu,
      global: this.layerIndex,
    });
    let pixels = 0;
    for (let i = 0; i < moments.count.length; i += 1) pixels += moments.count[i];
    return {
      peakMass: this.histPeak.mass,
      beta: Math.exp(sitesRgb[b + 2]) * sitesRgb[b + 3] / Math.max(this.width, this.height),
      area: moments.count[this.layerIndex] / Math.max(1, pixels),
    };
  }

  // Free every GPU buffer/texture this trainer allocated, so sweeps can run
  // many trainers on one device. Call after run()/finalize() have resolved;
  // the trainer is unusable afterwards. Safe to call more than once.
  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    const owned = [
      this.targetTex, this.maskTex, this.renderTex,
      this.sitesBuf, this.adamBuf, this.seedCounter, this.grads,
      this.renderParams, this.gradParams, this.adamParams, this.gradCountParams,
      this._tauHeatBuf, this._gnBuf, this.siteStatsParams, this.siteStatsBuf, this.maskBuf,
      this.errEmaBuf, this.errEmaSumBuf,
    ];
    for (const res of owned) res?.destroy();
    this.cands?.destroy();
    this.target = null;
    this.maskCpu = null;
    this._lastSitesCpu = null;
  }
}

const fmtLambda = (v) => (v >= 100 ? v.toFixed(0) : v.toPrecision(2));

// The best two-way split of site `i`'s cell, from site_stats.wgsl's owner
// moments (offsets d from the site, fit target t, over the pixels it owns).
// Adding a site to a partition model means splitting a cell, so the gain is
// the color variance a split can explain: here the variance explained by a
// linear color ramp across the cell, n·tr(B C⁻¹ Bᵀ) with C the spatial
// covariance and B the color-position covariance, in 8-bit levels² summed
// over RGB. (A two-level step along a ramp captures 3/4 of it, while a step
// fitted to an edge captures more than a ramp does; the ramp is the middle.)
// Error from soft edges or texture finer than any site isn't explained by a
// split, so it doesn't count. The split runs across u, the direction colors
// change fastest; the halves' centroids sit ±0.866σ along u (uniform cell).
// Returns { site, gain, oldOffset, newOffset, oldColor, newColor } or null.
export function siteSplit(stats, i) {
  const b = i * SITE_STAT_CH;
  const n = stats[b];
  if (!(n >= 4)) return null;
  const mx = stats[b + 1] / n, my = stats[b + 2] / n;
  // Spatial covariance, plus 1/12 px² for pixel quantization.
  const cxx = stats[b + 3] / n - mx * mx + 1 / 12;
  const cxy = stats[b + 4] / n - mx * my;
  const cyy = stats[b + 5] / n - my * my + 1 / 12;
  const det = cxx * cyy - cxy * cxy;
  if (!(det > 1e-9)) return null;
  const ixx = cyy / det, ixy = -cxy / det, iyy = cxx / det;
  const mean = [0, 1, 2].map((c) => stats[b + 6 + c] / n);
  let explained = 0;
  const A = [];   // per channel: color gradient (per px) = B_c C⁻¹
  for (let c = 0; c < 3; c += 1) {
    const bx = stats[b + 9 + c] / n - mean[c] * mx;
    const by = stats[b + 12 + c] / n - mean[c] * my;
    const ax = bx * ixx + by * ixy, ay = bx * ixy + by * iyy;
    A.push([ax, ay]);
    explained += ax * bx + ay * by;
  }
  const total = Math.max(0, stats[b + 15] / n - (mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2));
  const gain = n * Math.min(Math.max(0, explained), total) * 255 * 255;
  if (!(gain > 0) || !Number.isFinite(gain)) return null;
  // u = top eigenvector of AᵀA (2×2, symmetric).
  let mxx = 0, mxy = 0, myy = 0;
  for (const [ax, ay] of A) { mxx += ax * ax; mxy += ax * ay; myy += ay * ay; }
  const tr = mxx + myy, gap = Math.sqrt(Math.max(0, (mxx - myy) ** 2 / 4 + mxy * mxy));
  const lam = tr / 2 + gap;
  let ux = mxy, uy = lam - mxx;
  if (Math.hypot(ux, uy) < 1e-12) { ux = lam - myy; uy = mxy; }
  const ul = Math.hypot(ux, uy);
  if (ul < 1e-12) { ux = 1; uy = 0; } else { ux /= ul; uy /= ul; }
  const sigma = Math.sqrt(Math.max(1e-6, ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy));
  const s = 0.866 * sigma;
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const color = (sign) => A.map(([ax, ay], c) => clamp01(mean[c] + sign * s * (ax * ux + ay * uy)));
  return {
    site: i, gain,
    newOffset: [mx + s * ux, my + s * uy], oldOffset: [mx - s * ux, my - s * uy],
    newColor: color(1), oldColor: color(-1),
  };
}

const SITE_STAT_CH = 18; // channels per site in site_stats.wgsl's output


// Names of the ten grads[idx*10+ch] channels (colors are the run's color-space
// params, not necessarily RGB).
export const GRAD_CHANNELS = ["x", "y", "log τ", "radius", "color 0", "color 1", "color 2",
  "direction x", "direction y", "log aniso"];

// Gradient groups over the consolidated grads[idx*10+ch] channels. Multi-channel
// groups use the vector magnitude (e.g. |(dx, dy)| for position).
export const GRAD_GROUPS = [
  { key: "pos", label: "position", channels: [0, 1] },
  { key: "tau", label: "log τ", channels: [2] },
  { key: "radius", label: "radius", channels: [3] },
  { key: "color", label: "color", channels: [4, 5, 6] },
  { key: "dir", label: "direction", channels: [7, 8] },
  { key: "aniso", label: "log aniso", channels: [9] },
];

// Per-group mean/max of |grad| in the units Adam sees (summed grad / (W*H)).
// Sites with every channel zero (inactive, or covering no pixels) are skipped.
function computeGradStats(raw, siteCount, numPixels) {
  const scale = 1 / numPixels;
  const sums = new Float64Array(GRAD_GROUPS.length);
  const maxs = new Float64Array(GRAD_GROUPS.length);
  let counted = 0;
  for (let i = 0; i < siteCount; i += 1) {
    const base = i * 10;
    let any = false;
    for (let c = 0; c < 10; c += 1) {
      if (raw[base + c] !== 0) { any = true; break; }
    }
    if (!any) continue;
    counted += 1;
    for (let g = 0; g < GRAD_GROUPS.length; g += 1) {
      let sq = 0;
      for (const c of GRAD_GROUPS[g].channels) sq += raw[base + c] * raw[base + c];
      const mag = Math.sqrt(sq) * scale;
      sums[g] += mag;
      if (mag > maxs[g]) maxs[g] = mag;
    }
  }
  const groups = {};
  GRAD_GROUPS.forEach((grp, g) => {
    groups[grp.key] = { mean: counted ? sums[g] / counted : 0, max: maxs[g] };
  });
  return { groups, sites: counted };
}


// Mirrors main()'s PSNR computation: only pixels inside the mask count, values
// are sanitized (NaN → 0, +Inf → 1).
function computePsnrSafe(render, target, width, height, maskCpu, maskSum) {
  const n = width * height;
  let acc = 0;
  let counted = 0;
  for (let i = 0; i < n; i += 1) {
    if (maskCpu && maskCpu[i] <= 0) continue;
    for (let c = 0; c < 3; c += 1) {
      let r = render[i * 4 + c];
      if (!Number.isFinite(r)) r = r === Number.POSITIVE_INFINITY ? 1 : 0;
      const d = r - target[i * 4 + c];
      acc += d * d;
    }
    counted += 3;
  }
  if (maskSum && maskSum > 0 && counted === 0) return 0;
  if (counted === 0) counted = 1;
  const mse = acc / counted;
  if (mse <= 0) return 100.0;
  return 20.0 * Math.log10(1.0 / Math.sqrt(mse));
}
