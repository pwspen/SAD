// SAD training loop. Originally ported from the wgpu-py trainer (train_wgpu.py
// at the `multi-backend` tag), minus densify/prune. By default it optimizes a
// fixed site count; with growLambda > 0 it adds sites by rate-distortion (see
// "Site growth" below). Sites are never removed.

import {
  SITE_FLOATS,
  ADAM_FLOATS,
  PACKED_CAND_BYTES,
  TAU_DIFFUSE_PASSES,
  TAU_DIFFUSE_LAMBDA,
  PARAMS_SIZE,
  createParamsBuffer,
  updateParamsBuffer,
  createInitParamsBuffer,
  createGradParamsBuffer,
  createAdamParamsBuffer,
  updateAdamParamsBuffer,
  createClearParamsBuffer,
  createTauParamsBuffer,
  updateTauParamsBuffer,
  createHilbertParamsBuffer,
  updateHilbertParamsBuffer,
  updateGradParamsBuffer,
  updateClearParamsBuffer,
  createUniform32,
  updateGrowBlurParamsBuffer,
  updateInfoParamsBuffer,
  packJumpStep,
  hilbertBitsForSize,
} from "./params.js?v=info-20261019";

import {
  createSitesBuffer,
  createAdamBuffer,
  createPackedCandidatesBuffer,
  createSeedCounter,
  createGradsBuffer,
  createAtomicU32Buffer,
  createHilbertBuffers,
  createDummyStorageBuffer,
  readSites,
} from "./buffers.js?v=shared-wgsl-20261017";

import {
  createRgba32FloatTexture,
  createCandidateTexture,
  writeFloat32ToRgba32FloatTexture,
  readRgba32FloatTexture,
  whiteMaskRgba32Float,
} from "./textures.js?v=shared-wgsl-20261017";

import {
  InitGradientEncoder,
  CandidatesEncoder,
  SeedCandidatesEncoder,
  CandidatePackEncoder,
  RenderEncoder,
  RenderHashedEncoder,
  RenderTauHeatmapEncoder,
  GradientsEncoder,
  AdamEncoder,
  TauEncoder,
  ClearBufferEncoder,
  HilbertEncoder,
  GrowEncoder,
  SiteStatsEncoder,
  InfoEncoder,
} from "./encoders.js?v=info-20261019";

import { RadixSortUInt2 } from "./radix_sort.js?v=shared-wgsl-20261017";
import { BITS_PER_SITE } from "./io.js";
import { insertionField, pickInsertions } from "./insert.js?v=insert-20261018";
import { cellMoments, pickMerges, mergedSite } from "./merge.js?v=info-20261019";
import {
  colorSpaceFor,
  specializeShaders,
  encodeSiteColors,
  decodeSiteColors,
} from "./color_spaces.js?v=color-space-20261017";

const MAX_PREVIEW_IMAGE_READBACKS = 2;
// Direction/anisotropy start learning here. Previously derived from the prune
// schedule as max(10, PRUNE_START / 2) = 50 with the shared config.
const ANI_START_ITER = 50;
const MAX_PREVIEW_SITE_READBACKS = 8;

// Training losses (args.loss): label, and the LOSS id gradients_tiled.wgsl
// switches on. Growth, λ and PSNR measure squared error either way.
export const LOSSES = {
  l1: { label: "L1 (absolute error)", id: 1 },
  l2: { label: "L2 (squared error)", id: 0 },
};

// Keeping the best state (args.keepBest): every log tick is scored and the run
// ends on the best state seen, restored exactly (sites and Adam moments).
// "loss": the training loss alone (default); it only ever trades the final
// state for one with lower loss, at the same or a lower site count.
// "mdl": total bits of a two-part code, sites × BITS_PER_SITE plus the
// residual coded at 8-bit precision under the training loss's likelihood
// (L1 → Laplace, L2 → Gaussian; selectionScore). Fewer sites at the same loss
// always win, and a site is kept only if it pays for its own bits. Compared on
// converged fits it picks 8-16 sites on MNIST digits and 400-800 on Kodak at
// 256 px, but mid-growth states are far from converged (2-10× the converged
// loss with the MNIST preset), so within one growing run it tends to return
// an unconverged early state; use it to compare finished runs, or with
// growPatience only once growth checkpoints have settled. "off": the last state.
export const KEEP_BEST = {
  loss: "lowest training loss",
  mdl: "description length (bits)",
  off: "off (final state)",
};

// Score of a state for args.keepBest (lower is better). `err` is the mean
// training loss per sample (mean |r| for L1, mean r² for L2, colors in 0..1),
// `samples` the number of residuals (masked pixels × 3). For "mdl" it's bits:
// the ML Laplace (L1) / Gaussian (L2) fit to the residual has entropy
// log2(2e·b) / ½·log2(2πe·σ²) per sample, at a quantization step of 1/255,
// floored at 0 bits per sample.
const QUANT = 1 / 255;
export function selectionScore(kind, { err, sites, samples, loss }) {
  if (!Number.isFinite(err)) return Infinity;
  if (kind !== "mdl") return err;
  const perSample = loss === "l2"
    ? 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(err, 1e-30) / (QUANT * QUANT))
    : Math.log2(2 * Math.E * Math.max(err, 1e-30) / QUANT);
  return sites * BITS_PER_SITE + samples * Math.max(0, perSample);
}

// When growth stops early (args.growCriterion).
// "lambda": a new site's predicted gain must beat λ × BITS_PER_SITE
// (growLambda), in the gain's units.
// "mdl": growth goes on while each batch pays for its bits. At every growth
// step the current cells are scored by description length,
// sites × BITS_PER_SITE + (S/2)·log2(2πe·σ²) with σ² the squared error per
// sample of every cell at its mean color (S samples; cellMdlBits), and growth
// ends once a batch, after growEvery iterations to settle, hasn't lowered it.
// Scoring the cells at their best colors leaves out the error the fit simply
// hasn't removed yet. Predicted gains can't be used here: at small counts they
// miss most of what a new site does once optimized (a ramp across a centred
// digit explains almost nothing), and an MDL bar on them stops MNIST at 1 site.
export const GROW_CRITERIA = {
  lambda: "λ (predicted gain per bit)",
  mdl: "description length (realized)",
};

// How growth adds sites (args.growMode).
export const GROW_MODES = {
  split: "split cells",       // split the cells a linear color ramp explains best (siteSplit)
  insert: "insert (exact loss)", // insert new sites where they cut the training loss most (insert.js)
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
export const COUNT_SCHEDULES = {
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

// Histogram layer (args.layer). "auto": if the target's most common exact
// 8-bit color (histogramMode) covers at least layerModeMin of the pixels, that
// color gets a flat layer: a pinned site whose logit is ~0 everywhere
// (τ = e^LAYER_LOG_TAU, radius 0), so a site wins a pixel where -τ(d - r) > 0,
// i.e. inside its radius, and the layer's color shows wherever no site
// reaches. The layer is offered to every pixel's candidate list
// (Params.globalCand), never moves (sitePin), never splits, and can absorb
// sites in a merge. It's an ordinary site in the sites file.
export const LAYER_MODES = {
  off: "off",
  auto: "auto (dominant color)",
};
const LAYER_LOG_TAU = -9;

// Most common exact 8-bit RGB color over the masked pixels: { color (0..1
// RGB), mass (fraction of pixels) }.
export function histogramMode(target, n, mask = null) {
  const counts = new Map();
  let total = 0;
  for (let p = 0; p < n; p += 1) {
    if (mask && !(mask[p] > 0)) continue;
    const key = (Math.round(target[p * 4] * 255) << 16) | (Math.round(target[p * 4 + 1] * 255) << 8) |
      Math.round(target[p * 4 + 2] * 255);
    counts.set(key, (counts.get(key) || 0) + 1);
    total += 1;
  }
  let best = 0, bestCount = 0;
  for (const [k, c] of counts) if (c > bestCount) { best = k; bestCount = c; }
  return { color: [(best >> 16) / 255, ((best >> 8) & 255) / 255, (best & 255) / 255], mass: total ? bestCount / total : 0 };
}

// Remaining-information pass (args.info, info_update.wgsl), every infoEvery
// iterations. Pixels are measured in quantization bins of the target (1/255):
// within half a bin the decoded pixel is exact and has nothing left to explain.
// "deadzone": loss weight = 0 below 1/4 bin of error, ramping to 1 at 1/2
// bin (where rounding would go wrong), per pixel, each pass.
// "commit": full weight until a pixel has been exact for commitAfter passes
// in a row, then 0 (committed); with commitRecheck it's still checked and comes
// back if it drifts, without it it's never looked at again. With commitFreeze,
// sites with no weight above INFO_LIVE_EPS on an uncommitted pixel don't move.
// The weights replace the mask in the gradients and the growth statistics;
// PSNR, keepBest and pruning still score every masked pixel.
export const INFO_MODES = {
  off: "off",
  deadzone: "dead zone (exact pixels)",
  commit: "commit settled pixels",
};
const INFO_QUANT = 1 / 255;
const INFO_LIVE_EPS = 1e-3;

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
    // WebGPU-specific defaults; training_config.json's DEFAULT_ITERS/SITES are
    // tuned for the other backends' densify/prune schedule, which JS doesn't run.
    // Run length. The schedule, LR and growth defaults (and LR_POS_BASE / LR_TAU_BASE in
    // training_config.json) were tuned for growing 1 -> 100 sites over 1000
    // iterations, by mean final PSNR over 8 Kodak images, then checked on the
    // other 16.
    iters: Math.max(1, Math.floor(Number(pick("iters", null, 1000)))),
    candFreq: Number(pick("candFreq", "CAND_UPDATE_FREQ", 1)),
    candHilbertWindow: Number(pick("candHilbertWindow", "CAND_HILBERT_WINDOW", 0)),
    candHilbertProbes: Number(pick("candHilbertProbes", "CAND_HILBERT_PROBES", 0)),
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
    loss: String(pick("loss", "LOSS", "l1")),
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
    // growEvery iterations: up by splitting / inserting (growMode), down by
    // merging the cheapest neighbouring pairs (merge.js). sawTeeth, sawAmp and
    // sawRatio shape "sawtooth" (countTarget). These are the only settings
    // that set the count; the buffers hold countCapacity sites.
    startSites: Math.max(1, Math.floor(Number(pick("startSites", "START_SITES", 1)))),
    endSites: Math.max(1, Math.floor(Number(pick("endSites", "END_SITES", 100)))),
    // Defaults: the Kodak preset's sawtooth (presets/kodak.json).
    countSchedule: String(pick("countSchedule", "COUNT_SCHEDULE", "sawtooth")),
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
    // While growing, the fit target is blurred with sigma = growBlur × site
    // spacing √(mask area / sites); 0 = always fit the original.
    growBlur: Math.max(0, Number(pick("growBlur", "GROW_BLUR", 0))),
    // Each growth step multiplies the learning rates of every site that
    // existed before it (split parents count as new) by this; 1 = off.
    growLrDecay: Math.max(0, Number(pick("growLrDecay", "GROW_LR_DECAY", 1))),
    growMode: String(pick("growMode", "GROW_MODE", "split")),
    // What a new site's gain must beat (GROW_CRITERIA); growLambda applies
    // only to "lambda".
    growCriterion: String(pick("growCriterion", "GROW_CRITERION", "lambda")),
    // Keep the best log tick by this score (KEEP_BEST) and end on it.
    keepBest: String(pick("keepBest", "KEEP_BEST", "loss")),
    // While growing: end growth and go back to the best state once that many
    // growth steps in a row haven't beaten it (keepBest score; 0 = never).
    growPatience: Math.max(0, Math.floor(Number(pick("growPatience", "GROW_PATIENCE", 0)))),
    // Merge neighbouring sites that don't pay for their bits (merge.js) every
    // this many iterations once the count schedule is over (0 = never); see
    // _pruneStep.
    pruneEvery: Math.max(0, Math.floor(Number(pick("pruneEvery", "PRUNE_EVERY", 0)))),
    // Histogram layer (LAYER_MODES): when one exact 8-bit color holds at least
    // layerModeMin of the (masked) pixels, add a pinned flat site of that
    // color offered to every pixel. It counts as one of the sites.
    layer: String(pick("layer", "LAYER", "off")),
    layerModeMin: Number(pick("layerModeMin", "LAYER_MODE_MIN", 0.5)),
    // Remaining-information pass (INFO_MODES), every infoEvery iterations
    // (0 = candFreq): per-pixel loss weights from the error in 8-bit bins.
    info: String(pick("info", "INFO", "off")),
    infoEvery: Math.max(0, Math.floor(Number(pick("infoEvery", "INFO_EVERY", 0)))),
    // Whether the info weights also weight the training gradients (else only
    // growth's site statistics use them).
    infoGrad: Boolean(pick("infoGrad", "INFO_GRAD", true)),
    // "commit": passes in a row a pixel must be exact to leave the loss;
    // whether committed pixels are still checked (and come back if they
    // drift); whether sites serving no uncommitted pixel freeze.
    commitAfter: Math.max(1, Math.floor(Number(pick("commitAfter", "COMMIT_AFTER", 4)))),
    commitRecheck: Boolean(pick("commitRecheck", "COMMIT_RECHECK", true)),
    commitFreeze: Boolean(pick("commitFreeze", "COMMIT_FREEZE", true)),
  };
  if (!COUNT_SCHEDULES[args.countSchedule]) {
    throw new Error(`countSchedule must be one of ${Object.keys(COUNT_SCHEDULES).join(", ")} (got "${args.countSchedule}")`);
  }
  if (!(args.lrSchedule in LR_SCHEDULES)) {
    throw new Error(`lrSchedule must be one of ${Object.keys(LR_SCHEDULES).join(", ")} (got "${args.lrSchedule}")`);
  }
  colorSpaceFor(args.colorSpace); // fail early on an unknown name
  if (!GROW_MODES[args.growMode]) {
    throw new Error(`growMode must be one of ${Object.keys(GROW_MODES).join(", ")} (got "${args.growMode}")`);
  }
  if (!GROW_CRITERIA[args.growCriterion]) {
    throw new Error(`growCriterion must be one of ${Object.keys(GROW_CRITERIA).join(", ")} (got "${args.growCriterion}")`);
  }
  if (!KEEP_BEST[args.keepBest]) {
    throw new Error(`keepBest must be one of ${Object.keys(KEEP_BEST).join(", ")} (got "${args.keepBest}")`);
  }
  if (!LOSSES[args.loss]) {
    throw new Error(`loss must be one of ${Object.keys(LOSSES).join(", ")} (got "${args.loss}")`);
  }
  if (!LAYER_MODES[args.layer]) {
    throw new Error(`layer must be one of ${Object.keys(LAYER_MODES).join(", ")} (got "${args.layer}")`);
  }
  if (!INFO_MODES[args.info]) {
    throw new Error(`info must be one of ${Object.keys(INFO_MODES).join(", ")} (got "${args.info}")`);
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
    this.colorSpace = colorSpaceFor(args.colorSpace);
    shaders = specializeShaders(shaders, this.colorSpace);
    this.shaders = shaders;
    this.args = args;
    this.onLog = onLog;
    this.signal = signal;

    this.usesHilbert = args.candHilbertWindow > 0 && args.candHilbertProbes > 0;
    this._previewImageInFlight = 0;
    this._previewSiteInFlight = 0;

    // Encoders
    this.initEnc = new InitGradientEncoder(device, shaders);
    this.candsEnc = new CandidatesEncoder(device, shaders);
    this.seedEnc = new SeedCandidatesEncoder(device, shaders);
    this.packEnc = new CandidatePackEncoder(device, shaders);
    this.renderEnc = new RenderEncoder(device, shaders);
    this.renderHashedEnc = new RenderHashedEncoder(device, shaders);
    this.renderTauEnc = new RenderTauHeatmapEncoder(device, shaders);
    this.gradsEnc = new GradientsEncoder(device, shaders, { loss: LOSSES[args.loss].id });
    this.adamEnc = new AdamEncoder(device, shaders);
    this.tauEnc = new TauEncoder(device, shaders);
    this.clearI32Enc = new ClearBufferEncoder(device, shaders, { useI32: true });
    this.hilbertEnc = this.usesHilbert ? new HilbertEncoder(device, shaders) : null;
    // Whether the count changes is settled in prepare() (a loaded sites file
    // sets the start), so the growth kernels are always built.
    this.growEnc = new GrowEncoder(device, shaders);
    this.statsEnc = new SiteStatsEncoder(device, shaders);
    this.infoEnc = new InfoEncoder(device, shaders);
  }

  async prepare({ width, height, targetRgba32f, maskRgba32f, maskSum, initSites, initSiteCount }) {
    const args = this.args;
    const device = this.device;
    const shaders = this.shaders;

    this.width = width;
    this.height = height;
    this.candDownscale = Math.max(1, args.candDownscale);
    this.candWidth = Math.max(1, Math.floor((width + this.candDownscale - 1) / this.candDownscale));
    this.candHeight = Math.max(1, Math.floor((height + this.candDownscale - 1) / this.candDownscale));
    this.target = targetRgba32f; // keep a CPU copy for PSNR
    this.maskSum = maskSum;
    this.maskCpu = null;
    if (maskRgba32f) {
      this.maskCpu = new Float32Array(width * height);
      for (let i = 0; i < width * height; i += 1) this.maskCpu[i] = maskRgba32f[i * 4];
    }

    // Textures
    this.targetTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    writeFloat32ToRgba32FloatTexture(device, this.targetTex, targetRgba32f, width, height);
    this.maskTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    const maskData = maskRgba32f || whiteMaskRgba32Float(width, height).rgba;
    writeFloat32ToRgba32FloatTexture(device, this.maskTex, maskData, width, height);

    // The run starts from startSites (or the loaded sites); the buffers hold
    // the most the count schedule ever wants. Unused slots are inactive
    // (position.x < 0), which every kernel skips.
    if (!initSites) initSiteCount = args.startSites;
    // Histogram layer: one more site after the initial ones (LAYER_MODES).
    this.histMode = histogramMode(targetRgba32f, width * height, this.maskCpu);
    const useLayer = args.layer === "auto" && this.histMode.mass >= args.layerModeMin;
    const baseSiteCount = initSiteCount;
    this.layerIndex = useLayer ? baseSiteCount : null;
    if (useLayer) initSiteCount += 1;
    this.usesGrow = countVaries(args, initSiteCount);
    this.bufferCapacity = this.usesGrow ? countCapacity(args, initSiteCount) : initSiteCount;
    this.siteCount = initSiteCount;

    // Buffers
    this.sitesBuf = createSitesBuffer(device, this.bufferCapacity);
    this.packedSitesBuf = createPackedCandidatesBuffer(device, this.bufferCapacity);
    this.adamBuf = createAdamBuffer(device, this.bufferCapacity);
    this.seedCounter = createSeedCounter(device);
    this.grads = createGradsBuffer(device, this.bufferCapacity);
    this.tauGradRaw = createAtomicU32Buffer(device, this.bufferCapacity);
    this.tauGradTmp = createAtomicU32Buffer(device, this.bufferCapacity);
    this.tauGradTmp2 = createAtomicU32Buffer(device, this.bufferCapacity);
    // Per-site flags: sitePin (CPU-set, never optimized) and siteLive (set by
    // the info pass; all live until one runs). See adam.wgsl.
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.sitePinned = new Uint32Array(this.bufferCapacity);
    if (useLayer) this.sitePinned[this.layerIndex] = 1;
    this.sitePinBuf = device.createBuffer({ size: this.bufferCapacity * 4, usage: storage });
    device.queue.writeBuffer(this.sitePinBuf, 0, this.sitePinned);
    this.siteLiveBuf = device.createBuffer({ size: this.bufferCapacity * 4, usage: storage });
    this._markAllLive();
    // Per-pixel loss weights (the mask unless an info pass rewrites them),
    // the mask itself, and the info pass's commit counts.
    const maskF = new Float32Array(width * height);
    for (let i = 0; i < width * height; i += 1) maskF[i] = this.maskCpu ? (this.maskCpu[i] > 0 ? 1 : 0) : 1;
    this.maskBuf = device.createBuffer({ size: maskF.byteLength, usage: storage });
    device.queue.writeBuffer(this.maskBuf, 0, maskF);
    this.lossWeightBuf = device.createBuffer({ size: maskF.byteLength, usage: storage });
    device.queue.writeBuffer(this.lossWeightBuf, 0, maskF);
    this.settledBuf = device.createBuffer({ size: width * height * 4, usage: storage });
    this.infoParams = createUniform32(device);

    // Candidate textures
    this.cand0A = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand0B = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand1A = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand1B = createCandidateTexture(device, this.candWidth, this.candHeight);

    // Render texture
    this.renderTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false, copySrc: true });

    // Param buffers
    this.candUpdatePasses = Math.max(1, args.candUpdatePasses);
    this.candParams = [];
    for (let i = 0; i < this.candUpdatePasses; i += 1) {
      this.candParams.push(createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0)));
    }
    this.renderParams = createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0));

    const invScaleSq = 1.0 / Math.pow(Math.max(width, height), 2);
    this.invScaleSq = invScaleSq;
    this.gradParams = createGradParamsBuffer(device, initSiteCount, invScaleSq);
    this.adamParams = createAdamParamsBuffer(device,
      args.lrPosBase, args.lrTauBase, args.lrRadiusBase, args.lrColorBase,
      args.lrDirBase, args.lrAnisoBase, args.beta1, args.beta2, args.eps,
      1, width, height);
    this.countParams = createClearParamsBuffer(device, initSiteCount);
    this.gradCountParams = createClearParamsBuffer(device, this.bufferCapacity * 10);
    this.tauParams = createTauParamsBuffer(device, initSiteCount, TAU_DIFFUSE_LAMBDA, this.candDownscale);

    // Hilbert resources
    this.dummyHilbertBuf = createDummyStorageBuffer(device);
    if (this.usesHilbert) {
      this.hilbertRes = createHilbertBuffers(device, this.bufferCapacity);
      this.hilbertReady = false;
      this.hilbertSiteCount = 0;
      this.hilbertParams = createHilbertParamsBuffer(
        device, 1, this.hilbertRes.paddedCount, width, height, hilbertBitsForSize(width, height),
      );
      this.hilbertSort = new RadixSortUInt2(device, shaders, this.hilbertRes.paddedCount);
    } else {
      this.hilbertRes = null;
      this.hilbertReady = true;
      this.hilbertSiteCount = 0;
      this.hilbertParams = null;
      this.hilbertSort = null;
    }

    if (this.bufferCapacity > initSiteCount) {
      const inactive = new Float32Array((this.bufferCapacity - initSiteCount) * SITE_FLOATS);
      for (let i = 0; i < inactive.length; i += SITE_FLOATS) { inactive[i] = -1; inactive[i + 1] = -1; }
      device.queue.writeBuffer(this.sitesBuf, initSiteCount * SITE_FLOATS * 4, inactive);
    }

    // Site growth: the fit target (blurred while growing) and gain resources.
    this.fitTex = this.targetTex;
    if (this.usesGrow) {
      this.maskArea = maskSum > 0 ? maskSum : width * height;
      this.growBlurParams = createUniform32(device);
      this.growSeedParams = createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0));
      if (args.growBlur > 0) {
        this.blurTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false });
        this.blurTmpTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false });
      }
      this.siteStatsParams = createUniform32(device);
      this.siteStatsBuf = device.createBuffer({
        size: this.bufferCapacity * SITE_STAT_CH * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      // Predicted vs realized gain (diagnostic only; never feeds back into
      // growth): the last batch's predicted gains, scored at the next step.
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
      this.activeEstimate = 0;
      for (let i = 0; i < baseSiteCount; i += 1) {
        if (initSites[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      if (useLayer) this.activeEstimate += 1;
      this._log(`Loaded init sites: ${baseSiteCount} (active ${this.activeEstimate})`);
    } else {
      const gradThreshold = 0.01;
      const initParamsBuf = createInitParamsBuffer(device, baseSiteCount, gradThreshold, 256,
                                                    args.initLogTau, args.initRadius);
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
    if (useLayer) {
      const c = this.histMode.color.map((v) => Math.round(v * 255)).join(", ");
      this._log(`Histogram layer: rgb(${c}) covers ${(100 * this.histMode.mass).toFixed(1)}% of pixels → pinned site ${this.layerIndex}`);
    }

    // Prime candidate state (init + seed) — mirrors the one-time setup in Python.
    {
      updateParamsBuffer(device.queue, this.candParams[0], this._paramsArgs(initSiteCount, 0, 0));
      const encoder = device.createCommandEncoder();
      this.candsEnc.encodeInit(encoder, this.candParams[0], this.cand0A, this.cand1A,
                                this.candWidth, this.candHeight);
      encoder.copyTextureToTexture(
        { texture: this.cand0A },
        { texture: this.cand0B },
        [this.candWidth, this.candHeight, 1],
      );
      this.seedEnc.encode(encoder, this.sitesBuf, this.candParams[0], this.cand0A, this.cand0B,
                           initSiteCount);
      device.queue.submit([encoder.finish()]);
      // After seed, the freshly-written texture for cand0 is cand0B; swap so
      // cand0A is current. cand1 was written by init into cand1A, so leave it.
      [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
    }

    // Zero the gradient accumulators once; Adam re-zeroes each site after use.
    {
      const encoder = device.createCommandEncoder();
      this.clearI32Enc.encode(encoder, this.grads, this.gradCountParams, initSiteCount * 10);
      device.queue.submit([encoder.finish()]);
    }

    // Growth state: the iteration it ended (null while growing) and why.
    this.growDoneIter = this.usesGrow ? null : 0;
    this.growGen = 0; // growth steps that added sites (see growLrDecay)
    this.growStartCount = initSiteCount; // where the count schedule starts
    // CPU copy of each site's birth growth step (adam[21]), for per-site LRs.
    this.siteBirthGen = new Uint32Array(this.bufferCapacity);
    this.growSteps = 0; // growth steps taken (growPatience)
    this.growMdlBits = undefined; // cellMdlBits at the last growth step ("mdl" criterion)
    // Pruning: the merge batch on trial ({ it, bits, k }) and why pruning ended.
    this.prunePending = null;
    this.pruneStop = null;
    this.pruneMerged = 0;
    this.growStop = null;
    if (this.usesGrow) this._updateFitTarget();

    this.jumpPassIndex = 0;
    this.aniStartIter = ANI_START_ITER;
    this.trainStartMs = performance.now();
    this.bestPsnr = 0;

    // Best state by args.keepBest: GPU copies of the sites and Adam buffers
    // (so a restore resumes exactly, growth LR ages included) and its stats.
    this.best = null;
    this.restoredBest = null;
    if (args.keepBest !== "off") {
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      this.bestSitesBuf = device.createBuffer({ size: this.bufferCapacity * SITE_FLOATS * 4, usage });
      this.bestAdamBuf = device.createBuffer({ size: this.bufferCapacity * ADAM_FLOATS * 4, usage });
    }
    this.samples = 3 * (this.maskCpu ? this.maskCpu.reduce((a, m) => a + (m > 0 ? 1 : 0), 0) : width * height);
  }

  // Score the state just rendered (log tick) and snapshot it if it's the best.
  _trackBest(it, rgba, psnr) {
    if (this.args.keepBest === "off") return;
    const err = computeLoss(rgba, this.target, this.width, this.height, this.maskCpu, this.args.loss);
    const sites = this.activeEstimate;
    const score = selectionScore(this.args.keepBest, { err, sites, samples: this.samples, loss: this.args.loss });
    this.lastScore = { it, score, err, sites, psnr };
    if (this.best && !(score < this.best.score)) return;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.sitesBuf, 0, this.bestSitesBuf, 0, this.bufferCapacity * SITE_FLOATS * 4);
    encoder.copyBufferToBuffer(this.adamBuf, 0, this.bestAdamBuf, 0, this.bufferCapacity * ADAM_FLOATS * 4);
    this.device.queue.submit([encoder.finish()]);
    this.best = {
      it, score, err, psnr, sites, siteCount: this.siteCount, growGen: this.growGen,
      growStep: this.growSteps, siteBirthGen: this.siteBirthGen.slice(),
      sitePinned: this.sitePinned.slice(), layerIndex: this.layerIndex,
    };
  }

  // Put the best state back: sites, Adam moments, counts, and fresh candidate
  // lists (a full jump flood from the restored sites), then render it.
  _restoreBest(reason) {
    const b = this.best;
    const device = this.device;
    const queue = device.queue;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.bestSitesBuf, 0, this.sitesBuf, 0, this.bufferCapacity * SITE_FLOATS * 4);
    encoder.copyBufferToBuffer(this.bestAdamBuf, 0, this.adamBuf, 0, this.bufferCapacity * ADAM_FLOATS * 4);
    queue.submit([encoder.finish()]);
    this.siteCount = this.actualSites = b.siteCount;
    this.activeEstimate = b.sites;
    this.growGen = b.growGen;
    this.siteBirthGen.set(b.siteBirthGen);
    this.layerIndex = b.layerIndex;
    this._setPinned(b.sitePinned.slice());
    this._markAllLive();
    updateClearParamsBuffer(queue, this.countParams, b.siteCount);
    updateGradParamsBuffer(queue, this.gradParams, b.siteCount, this.invScaleSq);
    this._rebuildCandidates(b.siteCount);
    this.restoredBest = { ...b, reason };
    this._log(`Restored best state from iter ${b.it} (${reason}): ${b.sites} sites, ` +
              `${this.args.keepBest === "mdl" ? `${(b.score / 8192).toFixed(2)} KiB` : `loss ${b.err.toPrecision(4)}`}.`);
  }

  // Candidate lists from scratch for sites [0, n): init, seed every site at
  // its home pixel, then jump-flood passes from long jumps down to 1 px, twice.
  _rebuildCandidates(n) {
    const device = this.device;
    const queue = device.queue;
    const stages = Math.max(1, Math.ceil(Math.log2(Math.max(this.candWidth, this.candHeight, 2))));
    this.growJfaParams ??= [];
    for (let st = 0; st < stages; st += 1) {
      this.growJfaParams[st] ??= createParamsBuffer(device, this._paramsArgs(n, 0, 0));
      updateParamsBuffer(queue, this.growJfaParams[st],
        this._paramsArgs(n, packJumpStep(st, this.candWidth, this.candHeight), 0));
    }
    this.rebuildParams ??= createParamsBuffer(device, this._paramsArgs(n, 0, 0));
    updateParamsBuffer(queue, this.rebuildParams, this._paramsArgs(n, 0, 0));
    const encoder = device.createCommandEncoder();
    this.candsEnc.encodeInit(encoder, this.rebuildParams, this.cand0A, this.cand1A, this.candWidth, this.candHeight);
    encoder.copyTextureToTexture({ texture: this.cand0A }, { texture: this.cand0B },
      [this.candWidth, this.candHeight, 1]);
    this.seedEnc.encode(encoder, this.sitesBuf, this.rebuildParams, this.cand0A, this.cand0B, n);
    [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
    this.hilbertReady = false;
    this._maybeRefreshHilbert(encoder, n);
    this.packEnc.encode(encoder, this.sitesBuf, this.packedSitesBuf, this.countParams, n);
    for (let pass = 0; pass < 2; pass += 1) {
      for (let st = 0; st < stages; st += 1) {
        this.candsEnc.encodeUpdate(encoder, this.packedSitesBuf, this.growJfaParams[st],
          this.cand0A, this.cand1A, this.cand0B, this.cand1B,
          this._hilbertOrderBuf(), this._hilbertPosBuf(), this.candWidth, this.candHeight);
        [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
        [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
      }
    }
    queue.submit([encoder.finish()]);
  }

  _paramsArgs(siteCount, step, seed) {
    return {
      width: this.width,
      height: this.height,
      siteCount,
      step,
      seed,
      radiusScale: this.args.candRadiusScale,
      radiusProbes: this.args.candRadiusProbes,
      injectCount: this.args.candInjectCount,
      hilbertProbes: this.usesHilbert ? this.args.candHilbertProbes : 0,
      hilbertWindow: this.usesHilbert ? this.args.candHilbertWindow : 0,
      candDownscale: this.candDownscale,
      candWidth: this.candWidth,
      candHeight: this.candHeight,
      globalCand: this.layerIndex ?? 0xffffffff,
    };
  }

  _log(msg) { if (this.onLog) this.onLog({ type: "msg", message: msg }); }

  // The histogram layer as a site at layerIndex: the mode color, logit
  // -e^LAYER_LOG_TAU · d ≈ 0 everywhere (radius 0, isotropic, image center).
  _writeLayerSite() {
    const site = new Float32Array(SITE_FLOATS);
    site[0] = (this.width - 1) / 2;
    site[1] = (this.height - 1) / 2;
    site[2] = LAYER_LOG_TAU;
    site[3] = 0;
    site.set(this.histMode.color, 4);
    site[7] = 1;
    site[8] = 0;
    site[9] = 0;
    this.device.queue.writeBuffer(this.sitesBuf, this.layerIndex * SITE_FLOATS * 4,
      encodeSiteColors(site, 1, this.colorSpace));
  }

  // Every site counts as serving a pixel in the loss until the next info pass
  // (after anything that adds, removes or moves sites wholesale).
  _markAllLive() {
    this.device.queue.writeBuffer(this.siteLiveBuf, 0, new Uint32Array(this.bufferCapacity).fill(1));
  }

  _hilbertOrderBuf() {
    return this.usesHilbert && this.hilbertRes ? this.hilbertRes.order : this.dummyHilbertBuf;
  }
  _hilbertPosBuf() {
    return this.usesHilbert && this.hilbertRes ? this.hilbertRes.pos : this.dummyHilbertBuf;
  }

  _maybeRefreshHilbert(encoder, preSites) {
    if (!this.usesHilbert) return;
    if (this.hilbertReady && this.hilbertSiteCount === preSites) return;
    const bits = hilbertBitsForSize(this.width, this.height);
    updateHilbertParamsBuffer(this.device.queue, this.hilbertParams,
      preSites, this.hilbertRes.paddedCount, this.width, this.height, bits);
    this.hilbertEnc.encodePairs(encoder, this.sitesBuf, this.hilbertRes.pairs, this.hilbertParams,
      this.hilbertRes.paddedCount);
    const maxKeyExclusive = bits >= 16 ? 0xffffffff : (1 << (bits * 2));
    this.hilbertSort.encode(encoder, this.hilbertRes.pairs, maxKeyExclusive);
    this.hilbertEnc.encodeOrder(encoder, this.hilbertRes.pairs, this.hilbertRes.order,
      this.hilbertRes.pos, this.hilbertParams, preSites);
    this.hilbertReady = true;
    this.hilbertSiteCount = preSites;
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
      if (args.pruneEvery > 0) await this._pruneStep(it);
      const siteCount = this.siteCount;
      const activeSites = this.activeEstimate;
      const updateCandidates = (it % candUpdateFreq === 0);
      const candPasses = this.candUpdatePasses;

      // Update candidate params for each pass
      if (updateCandidates) {
        for (let p = 0; p < candPasses; p += 1) {
          const stepIndex = this.jumpPassIndex + p;
          const step = packJumpStep(stepIndex, this.candWidth, this.candHeight);
          const seed = (stepIndex >>> 16) & 0xffff;
          updateParamsBuffer(queue, this.candParams[p], this._paramsArgs(siteCount, step, seed));
        }
      }

      const lrScale = args.lrMult * lrSchedule(it, args.iters, args.lrSchedule, this.growDoneIter ?? Infinity);
      const lrDir = it >= this.aniStartIter ? args.lrDirBase * lrScale : 0;
      const lrAniso = it >= this.aniStartIter ? args.lrAnisoBase * lrScale : 0;
      updateAdamParamsBuffer(queue, this.adamParams,
        args.lrPosBase * lrScale, args.lrTauBase * lrScale, args.lrRadiusBase * lrScale,
        args.lrColorBase * lrScale, lrDir, lrAniso, args.beta1, args.beta2, args.eps, it + 1, this.width, this.height,
        this.growGen, args.growLrDecay, args.info === "commit" && args.commitFreeze);
      const infoEvery = args.infoEvery || candUpdateFreq;
      const runInfo = args.info !== "off" && it % infoEvery === 0;
      if (runInfo) {
        updateInfoParamsBuffer(queue, this.infoParams, {
          siteCount, mode: args.info === "commit" ? 2 : 1, commitAfter: args.commitAfter,
          recheck: args.commitRecheck, invScaleSq: this.invScaleSq, quant: INFO_QUANT, liveEps: INFO_LIVE_EPS,
        });
      }

      const blend = it / Math.max(1, iters);
      const lambda = TAU_DIFFUSE_LAMBDA * (0.1 + 0.9 * blend);
      updateTauParamsBuffer(queue, this.tauParams, siteCount, lambda, this.candDownscale);

      const shouldLog = (it % logFreq === 0) || (it === iters - 1);
      const shouldPreview = shouldLog || (it % viewerFreq === 0);
      if (shouldLog || shouldPreview) {
        updateParamsBuffer(queue, this.renderParams, this._paramsArgs(siteCount, 0, 0));
      }

      const encoder = device.createCommandEncoder();

      // Candidate update
      if (updateCandidates) {
        this._maybeRefreshHilbert(encoder, siteCount);
        this.packEnc.encode(encoder, this.sitesBuf, this.packedSitesBuf, this.countParams, siteCount);
        for (let p = 0; p < candPasses; p += 1) {
          this.candsEnc.encodeUpdate(encoder,
            this.packedSitesBuf, this.candParams[p],
            this.cand0A, this.cand1A, this.cand0B, this.cand1B,
            this._hilbertOrderBuf(), this._hilbertPosBuf(),
            this.candWidth, this.candHeight);
          [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
          [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
          this.jumpPassIndex += 1;
        }
      }

      // Remaining information: per-pixel loss weights from the current lists.
      if (runInfo) this._encodeInfo(encoder);

      // Gradients
      this.gradsEnc.encode(encoder, this.sitesBuf, this.gradParams,
        this.cand0A, this.cand1A, this.fitTex, args.infoGrad ? this.lossWeightBuf : this.maskBuf,
        this.grads, this.width, this.height);

      // Tau diffusion
      if (TAU_DIFFUSE_PASSES > 0 && TAU_DIFFUSE_LAMBDA > 0) {
        this.tauEnc.encodeExtract(encoder, this.grads, this.tauGradRaw,
          this.countParams, siteCount);
        let curIn = this.tauGradRaw;
        let curOut = this.tauGradTmp;
        for (let p = 0; p < TAU_DIFFUSE_PASSES; p += 1) {
          this.tauEnc.encodeDiffuse(encoder, this.cand0A, this.cand1A,
            this.sitesBuf, this.tauGradRaw, curIn, curOut, this.tauParams, siteCount);
          if (curOut === this.tauGradTmp) {
            [curIn, curOut] = [curOut, this.tauGradTmp2];
          } else {
            [curIn, curOut] = [curOut, this.tauGradTmp];
          }
        }
        this.tauEnc.encodeWriteback(encoder, curIn, this.grads,
          this.countParams, siteCount);
      }

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

      // Adam
      this.adamEnc.encode(encoder, this.sitesBuf, this.adamBuf, this.grads,
        this.adamParams, siteCount, this.sitePinBuf, this.siteLiveBuf);

      // Preview/log renders. The log path does a blocking readback for PSNR.
      // Viewer ticks wait only for the small site readback so centroid overlays
      // stay on the requested cadence; full image readback stays asynchronous.
      let previewReadback = null;
      if (shouldPreview) {
        this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
          this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
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
        const gradStats = computeGradStats(new Float32Array(gradStaging.getMappedRange()),
          siteCount, this.width * this.height);
        gradStaging.unmap();
        gradStaging.destroy();
        const psnr = computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum);
        this.bestPsnr = Math.max(this.bestPsnr, psnr);
        this._trackBest(it, rgba, psnr);
        const elapsed = (performance.now() - this.trainStartMs) / 1000;
        const itsPerSec = (it + 1) / Math.max(elapsed, 1e-6);
        if (this.onLog) {
          this.onLog({
            type: "iter",
            it, iters,
            lrScale,
            siteLr: this._siteLrs(lrScale, sitesData, siteCount),
            psnr,
            gradStats,
            activeSites,
            totalSites: siteCount,
            fitSigma: this.fitSigma || 0,
            growDoneIter: this.growDoneIter,
            growStop: this.growStop,
            score: this.lastScore,
            best: this.best,
            itsPerSec,
            elapsedSec: elapsed,
            renderRgba: rgba,
            sitesData,
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

    if (this.best && this.lastScore && this.best.it !== this.lastScore.it) this._restoreBest("end of run");
    const queue2 = device.queue;
    updateParamsBuffer(queue2, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));
    const encoder = device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    device.queue.submit([encoder.finish()]);
  }

  // ---- Site count: growth and merging ----
  // Every growEvery iterations the count moves to the schedule's value
  // (countTarget, _countStep): down by merging (_mergeTo), up in a growth
  // batch (_growTo). growMode "split": each existing
  // site is scored by the error a two-way split of its cell would remove
  // (siteSplit, from site_stats.wgsl's moments) and the best cells are split,
  // one new site each. growMode "insert": new sites are picked one at a time
  // by their exact gain in the training loss (insert.js), leaving existing
  // sites alone. Only batches whose gain beats λ·BITS_PER_SITE (the cost of a
  // site's bits, in squared error for splits and in the training loss for
  // insertions; λ = 0 only requires a positive gain) qualify; when nothing
  // does, the count schedule ends there. Optionally (growBlur > 0) the fit target is the
  // image blurred to the current spacing until growth ends; in tests that cost
  // 0.5-2 dB at a matched site count, since sites place edges far finer than
  // their spacing.

  _siteSpacing() { return Math.sqrt(this.maskArea / Math.max(1, this.siteCount)); }

  // Point the gradients at the blurred (while growing) or original target.
  _updateFitTarget() {
    const sigma = this.blurTex && this.growDoneIter === null ? this.args.growBlur * this._siteSpacing() : 0;
    this.fitSigma = sigma >= 0.25 ? sigma : 0;
    if (!this.fitSigma) {
      this.fitTex = this.targetTex;
      return;
    }
    updateGrowBlurParamsBuffer(this.device.queue, this.growBlurParams,
      { width: this.width, height: this.height, sigma: this.fitSigma });
    const encoder = this.device.createCommandEncoder();
    this.growEnc.encodeBlur(encoder, this.growBlurParams, this.targetTex, this.maskTex,
      this.blurTmpTex, this.blurTex, this.width, this.height);
    this.device.queue.submit([encoder.finish()]);
    this.fitTex = this.blurTex;
  }

  // Per-site stats of the current state (site_stats.wgsl): Float32Array of
  // SITE_STAT_CH per site, for sites [0, siteCount).
  async _siteStats() {
    const device = this.device;
    const n = this.siteCount;
    const u = new Uint32Array(8);
    u[0] = n;
    new Float32Array(u.buffer)[4] = this.invScaleSq;
    device.queue.writeBuffer(this.siteStatsParams, 0, u);
    const bytes = n * SITE_STAT_CH * 4;
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    this.statsEnc.encode(encoder, this.sitesBuf, this.siteStatsParams, this.cand0A, this.cand1A,
      this.fitTex, this.lossWeightBuf, this.siteStatsBuf, this.width, this.height, this.sitePinBuf);
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

  // LR multiplier of each site this step: lrScale × growLrDecay^age, as in
  // adam.wgsl (age = growth steps since the site's birth). NaN for removed
  // sites. `sites` is the RGB readback (SITE_FLOATS each).
  _siteLrs(lrScale, sites, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      const age = this.growGen - this.siteBirthGen[i];
      out[i] = sites[i * SITE_FLOATS] >= 0
        ? lrScale * (age > 0 ? this.args.growLrDecay ** age : 1)
        : NaN;
    }
    return out;
  }

  _endGrowth(it, reason) {
    this.growDoneIter = it;
    this.growStop = reason;
    this._updateFitTarget();
    this._log(`Growth done at iter ${it}: ${this.activeEstimate} sites (${reason}).`);
  }

  // Split growth: the `count` cells whose two-way split (siteSplit) gains most,
  // above `threshold`. Rewrites the split parents in place and returns the new
  // sites (RGB) as { data, k, predicted, best }, or null.
  async _splitBatch(it, stats, n0, count, threshold) {
    const device = this.device;
    const queue = device.queue;
    const sitesRaw = await readSites(device, this.sitesBuf, n0);
    const splits = [];
    for (let i = 0; i < n0; i += 1) {
      if (sitesRaw[i * SITE_FLOATS] < 0 || this.sitePinned[i]) continue;
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
    this.growGen += 1;
    const adam = new Float32Array(ADAM_FLOATS);
    adam[20] = it;
    adam[21] = this.growGen;
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
      const enc = encodeSiteColors(old, 1, this.colorSpace);
      queue.writeBuffer(this.sitesBuf, sp.site * SITE_FLOATS * 4, enc);
      queue.writeBuffer(this.adamBuf, sp.site * ADAM_FLOATS * 4, adam);
      this.siteBirthGen[sp.site] = this.growGen;
    });
    return { data, k, predicted: splits.map((sp) => sp.gain), best: splits[0].gain };
  }

  // The candidate lists as 8 u32 site ids per candidate pixel. The textures
  // are rgba32uint; the float readback copies their bytes unchanged
  // (same-type typed-array copies are bitwise).
  async _readCandidates() {
    const [c0, c1] = await Promise.all([
      readRgba32FloatTexture(this.device, this.cand0A, this.candWidth, this.candHeight),
      readRgba32FloatTexture(this.device, this.cand1A, this.candWidth, this.candHeight),
    ]);
    const u0 = new Uint32Array(c0.buffer), u1 = new Uint32Array(c1.buffer);
    const cand = new Uint32Array(this.candWidth * this.candHeight * 8);
    for (let i = 0; i < this.candWidth * this.candHeight; i += 1) {
      cand.set(u0.subarray(i * 4, i * 4 + 4), i * 8);
      cand.set(u1.subarray(i * 4, i * 4 + 4), i * 8 + 4);
    }
    return cand;
  }

  // ---- Pruning (args.pruneEvery) ----
  // Every pruneEvery iterations after growth (any time in fixed mode), until
  // pruneEvery before the end: the last merge batch, refined since, is kept
  // only if the render's description length (selectionScore "mdl": sites ×
  // BITS_PER_SITE + the training loss's residual bits) went down, else it's
  // undone and pruning ends. Ward costs on cells at their mean colors
  // underestimate what a merge costs the soft render, so they only pick the
  // candidates: judged on the cell model instead, Kodak at 256 px went from
  // 800 to 300 sites for a worse render score than keeping all 800. Then the
  // merges that pay for their bits are applied (pickMerges): the larger cell's site becomes
  // the pixel-weighted average of the pair, the other is deactivated, both
  // get fresh Adam moments, and the candidate lists are rebuilt. Pruning also
  // ends when no merge pays. An accepted batch restarts keepBest, which
  // would otherwise go back to a state with more sites.

  async _pruneStep(it) {
    const args = this.args;
    const every = args.pruneEvery;
    if (this.pruneStop || it === 0 || it % every !== 0) return;
    if (this.usesGrow && (this.growDoneIter === null || it - this.growDoneIter < every)) return;
    const n = this.siteCount;
    const sites = await this._readSitesRgb(n);
    const moments = cellMoments({
      width: this.width, height: this.height, sites, n, cand: await this._readCandidates(),
      candWidth: this.candWidth, candHeight: this.candHeight, target: this.target, mask: this.maskCpu,
    });
    const bits = selectionScore("mdl", {
      err: await this._renderLoss(), sites: this.activeEstimate, samples: this.samples, loss: args.loss,
    });
    const pending = this.prunePending;
    this.prunePending = null;
    if (pending) {
      if (!(bits < pending.bits)) {
        this._restorePruneSnapshot();
        await this._endPrune(it, `the last ${pending.k} merge(s) didn't pay for their bits ` +
          `(${(pending.bits / 8192).toFixed(2)} → ${(bits / 8192).toFixed(2)} KiB), undone`);
        return;
      }
      this.pruneMerged += pending.k;
      this.best = null; // keepBest restarts from the pruned state
    }
    if (it > args.iters - every) { await this._endPrune(it, "too close to the end of the run"); return; }
    const merges = pickMerges(moments, this.samples, BITS_PER_SITE, { pinned: this._pinnedColors(sites, n) });
    if (!merges.length) { await this._endPrune(it, "no merge would pay for its bits"); return; }

    const device = this.device;
    const queue = device.queue;
    const snap = device.createCommandEncoder();
    this.pruneSitesBuf ??= device.createBuffer({ size: this.bufferCapacity * SITE_FLOATS * 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.pruneAdamBuf ??= device.createBuffer({ size: this.bufferCapacity * ADAM_FLOATS * 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    snap.copyBufferToBuffer(this.sitesBuf, 0, this.pruneSitesBuf, 0, this.bufferCapacity * SITE_FLOATS * 4);
    snap.copyBufferToBuffer(this.adamBuf, 0, this.pruneAdamBuf, 0, this.bufferCapacity * ADAM_FLOATS * 4);
    queue.submit([snap.finish()]);
    this.pruneSnapshot = { active: this.activeEstimate, growGen: this.growGen };
    this._applyMerges(it, sites, moments, merges);
    this.prunePending = { it, bits, k: merges.length };
    this._log(`Prune @${it}: merged ${merges.length} pair(s) → ${this.activeEstimate} sites ` +
              `| cheapest ${fmtLambda(merges[0].cost * 255 * 255)} levels², on trial until iter ${it + every}`);
  }

  // Writes `merges` (pickMerges) out: each `keep` site becomes the
  // pixel-weighted average of the pair (mergedSite), each `drop` site is
  // deactivated, both get fresh Adam moments, and the candidate lists are
  // rebuilt. `sites` is the RGB readback the merges were picked from.
  _applyMerges(it, sites, moments, merges) {
    const queue = this.device.queue;
    // Merged sites restart at age 0 (adam[21] = growGen), but a merge step
    // isn't a growth step: it doesn't age everyone else (growLrDecay), which
    // over a long prune schedule froze every site.
    const adam = new Float32Array(ADAM_FLOATS);
    adam[20] = it;
    adam[21] = this.growGen;
    const inactive = new Float32Array(SITE_FLOATS);
    inactive[0] = -1;
    inactive[1] = -1;
    for (const { keep, drop } of merges) {
      // A pinned site absorbs its partner unchanged.
      if (!this.sitePinned[keep]) {
        const merged = mergedSite(sites, keep, drop, moments.count[keep], moments.count[drop]);
        queue.writeBuffer(this.sitesBuf, keep * SITE_FLOATS * 4, encodeSiteColors(merged, 1, this.colorSpace));
        queue.writeBuffer(this.adamBuf, keep * ADAM_FLOATS * 4, adam);
        this.siteBirthGen[keep] = this.growGen;
      }
      queue.writeBuffer(this.sitesBuf, drop * SITE_FLOATS * 4, inactive);
      queue.writeBuffer(this.adamBuf, drop * ADAM_FLOATS * 4, adam);
    }
    this.activeEstimate -= merges.length;
    this._markAllLive();
    this._rebuildCandidates(this.siteCount);
  }

  // Pinned sites among [0, n) as Map(id → RGB color), for pickMerges.
  _pinnedColors(sites, n) {
    const out = new Map();
    for (let i = 0; i < n; i += 1) {
      if (this.sitePinned[i]) out.set(i, Array.from(sites.subarray(i * SITE_FLOATS + 4, i * SITE_FLOATS + 7)));
    }
    return out;
  }

  // The remaining-information pass (info_update.wgsl) into `encoder`.
  _encodeInfo(encoder) {
    this.infoEnc.encode(encoder, {
      sites: this.sitesBuf, params: this.infoParams, cand0: this.cand0A, cand1: this.cand1A,
      targetTex: this.targetTex, mask: this.maskBuf, weight: this.lossWeightBuf,
      settled: this.settledBuf, siteLive: this.siteLiveBuf, width: this.width, height: this.height,
    });
  }

  // Count step down: merge the cheapest neighbouring pairs (Ward's criterion,
  // merge.js) until `target` sites are left, each site in at most one merge,
  // so a step removes at most half of them. Returns the number merged.
  async _mergeTo(it, target) {
    const n = this.siteCount;
    const sites = await this._readSitesRgb(n);
    const moments = cellMoments({
      width: this.width, height: this.height, sites, n, cand: await this._readCandidates(),
      candWidth: this.candWidth, candHeight: this.candHeight, target: this.target, mask: this.maskCpu,
    });
    const merges = pickMerges(moments, this.samples, BITS_PER_SITE,
      { count: this.activeEstimate - target, pinned: this._pinnedColors(sites, n) });
    if (!merges.length) return 0;
    this._applyMerges(it, sites, moments, merges);
    this.best = null; // keepBest restarts, so it can't go back to more sites
    this._log(`Merge @${it}: ${merges.length} pair(s) → ${this.activeEstimate} sites (schedule wants ${target})`);
    return merges.length;
  }

  // Mean training loss per sample of the current state (renders and reads back).
  async _renderLoss() {
    updateParamsBuffer(this.device.queue, this.renderParams, this._paramsArgs(this.siteCount, 0, 0));
    const encoder = this.device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    this.device.queue.submit([encoder.finish()]);
    const rgba = await readRgba32FloatTexture(this.device, this.renderTex, this.width, this.height);
    return computeLoss(rgba, this.target, this.width, this.height, this.maskCpu, this.args.loss);
  }

  _restorePruneSnapshot() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.pruneSitesBuf, 0, this.sitesBuf, 0, this.bufferCapacity * SITE_FLOATS * 4);
    encoder.copyBufferToBuffer(this.pruneAdamBuf, 0, this.adamBuf, 0, this.bufferCapacity * ADAM_FLOATS * 4);
    this.device.queue.submit([encoder.finish()]);
    this.activeEstimate = this.pruneSnapshot.active;
    this.growGen = this.pruneSnapshot.growGen;
    this._rebuildCandidates(this.siteCount);
  }

  async _endPrune(it, reason) {
    // A batch still on trial at the end of the run stays (keepBest decides).
    this.pruneStop = reason;
    this._log(`Pruning done at iter ${it}: ${this.activeEstimate} sites, ${this.pruneMerged} merged (${reason}).`);
  }

  // Moves the active sites (and their Adam moments) to the front, in order,
  // marks the rest inactive, and rebuilds the candidate lists.
  async _compactSites() {
    const device = this.device;
    const n = this.siteCount;
    const sites = await readSites(device, this.sitesBuf, n);
    const adamBytes = n * ADAM_FLOATS * 4;
    const staging = device.createBuffer({ size: adamBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.adamBuf, 0, staging, 0, adamBytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const adam = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    const outSites = new Float32Array(n * SITE_FLOATS);
    const outAdam = new Float32Array(n * ADAM_FLOATS);
    let m = 0;
    const pinned = new Uint32Array(this.bufferCapacity);
    for (let i = 0; i < n; i += 1) {
      if (!(sites[i * SITE_FLOATS] >= 0)) continue;
      outSites.set(sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS), m * SITE_FLOATS);
      outAdam.set(adam.subarray(i * ADAM_FLOATS, (i + 1) * ADAM_FLOATS), m * ADAM_FLOATS);
      this.siteBirthGen[m] = this.siteBirthGen[i];
      pinned[m] = this.sitePinned[i];
      if (i === this.layerIndex) this.layerIndex = m;
      m += 1;
    }
    this._setPinned(pinned);
    this._markAllLive();
    for (let i = m; i < n; i += 1) { outSites[i * SITE_FLOATS] = -1; outSites[i * SITE_FLOATS + 1] = -1; }
    device.queue.writeBuffer(this.sitesBuf, 0, outSites);
    device.queue.writeBuffer(this.adamBuf, 0, outAdam);
    this.siteCount = this.actualSites = this.activeEstimate = m;
    updateClearParamsBuffer(device.queue, this.countParams, m);
    updateGradParamsBuffer(device.queue, this.gradParams, m, this.invScaleSq);
    this.growPending = null; // site ids changed
    this._rebuildCandidates(m);
  }

  _setPinned(pinned) {
    this.sitePinned = pinned;
    this.device.queue.writeBuffer(this.sitePinBuf, 0, pinned);
  }

  // Insertion growth (insert.js): up to `count` new sites picked greedily by
  // their exact gain in the training loss; existing sites are untouched.
  async _insertBatch(n0, count, threshold) {
    const sitesRgb = await this._readSitesRgb(n0);
    const cand = await this._readCandidates();
    const field = insertionField({
      width: this.width, height: this.height, sites: sitesRgb, siteCount: n0,
      cand, candWidth: this.candWidth, candHeight: this.candHeight,
    });
    const picks = pickInsertions({
      width: this.width, height: this.height, target: this.target, mask: this.maskCpu,
      sites: sitesRgb, field, count, threshold, loss: this.args.loss,
      spacing: Math.sqrt(this.maskArea / (n0 + count)),
    });
    if (!picks.length) return null;
    this.growGen += 1;
    const k = picks.length;
    const data = new Float32Array(k * SITE_FLOATS);
    picks.forEach((pk, j) => data.set(pk.site, j * SITE_FLOATS));
    // predicted: squared-error gains, to match the realized-gain diagnostic.
    return { data, k, predicted: picks.map((pk) => pk.gainL2), best: picks[0].gain };
  }

  // Units of growth gains and λ: split gains are squared error; insertion
  // gains are in the training loss.
  _gainUnit() {
    return this.args.growMode === "insert" && this.args.loss === "l1" ? "levels" : "levels²";
  }

  // One count step (every growEvery iterations, until the schedule's horizon):
  // grow or merge to the schedule's count (countTarget), then end the
  // schedule once the horizon is reached. Merging goes on past the horizon
  // until the count is down to endSites: a sawtooth with a big amplitude and
  // a short merge phase can't get back down in time (a step merges at most
  // half the sites), and stopping there kept up to 60% more sites than asked.
  async _countStep(it) {
    const args = this.args;
    if (this.growDoneIter !== null) {
      // One more measurement for the final batch, a growth step after it.
      if (this.growPending && it >= this.growPending.it + args.growEvery) {
        this._reportGrowthGain(it, await this._siteStats());
      }
      return;
    }
    if (it === 0 || it % args.growEvery !== 0) return;
    const target = Math.max(1, Math.round(countTarget(args, it, this.growStartCount)));
    let merged = 0;
    if (target < this.activeEstimate) merged = await this._mergeTo(it, target);
    else if (target > this.activeEstimate) await this._growTo(it, target);
    const catchingUp = this.activeEstimate > target && merged > 0;
    if (this.growDoneIter === null && it >= countHorizon(args) && !catchingUp) {
      this._endGrowth(it, `count schedule done (${args.countSchedule}, ${this.growStartCount} → ${args.endSites})`);
    }
  }

  // Count step up: add sites (growMode) until `target` are active, unless λ,
  // the MDL criterion or patience ends the schedule first.
  async _growTo(it, target) {
    const args = this.args;
    // Merged-away sites leave inactive slots; make room at the end first.
    if (this.siteCount + (target - this.activeEstimate) > this.bufferCapacity) await this._compactSites();
    const n0 = this.siteCount;
    const want = Math.min(target - this.activeEstimate, this.bufferCapacity - n0);
    if (want <= 0) return;
    if (args.growPatience > 0 && this.best && this.growSteps - this.best.growStep >= args.growPatience) {
      this._restoreBest(`no better ${args.keepBest} score in ${args.growPatience} growth step(s)`);
      this._endGrowth(it, `patience: ${args.growPatience} growth step(s) without a better ${args.keepBest} score`);
      return;
    }
    this.growSteps += 1;
    const stats = await this._siteStats();
    this._reportGrowthGain(it, stats);

    const device = this.device;
    const queue = device.queue;
    const mdl = args.growCriterion === "mdl";
    if (mdl) {
      const bits = cellMdlBits(stats, n0, this.activeEstimate, this.samples);
      const prev = this.growMdlBits;
      this.growMdlBits = bits;
      if (prev !== undefined && !(bits < prev)) {
        this._endGrowth(it, `MDL (stop): the last batch didn't pay for its bits ` +
          `(${(prev / 8192).toFixed(2)} → ${(bits / 8192).toFixed(2)} KiB)`);
        return;
      }
    }
    const threshold = mdl ? 0 : args.growLambda * BITS_PER_SITE;
    const batch = args.growMode === "insert"
      ? await this._insertBatch(n0, want, threshold)
      : await this._splitBatch(it, stats, n0, want, threshold);
    if (!batch) {
      const what = args.growMode === "insert" ? "insertion" : "split";
      this._endGrowth(it, `λ (stop): no ${what} would remove more than λ × ${BITS_PER_SITE} = ` +
                          `${fmtLambda(threshold)} ${this._gainUnit()}`);
      return;
    }
    const { data, k } = batch;
    queue.writeBuffer(this.sitesBuf, n0 * SITE_FLOATS * 4, encodeSiteColors(data, k, this.colorSpace));
    const adamNew = new Float32Array(k * ADAM_FLOATS);
    for (let j = 0; j < k; j += 1) {
      adamNew[j * ADAM_FLOATS + 20] = it;
      adamNew[j * ADAM_FLOATS + 21] = this.growGen;
    }
    queue.writeBuffer(this.adamBuf, n0 * ADAM_FLOATS * 4, adamNew);
    this.siteBirthGen.fill(this.growGen, n0, n0 + k);
    this._markAllLive();
    this.growPending = { it, sites: batch.predicted.map((predicted, j) => ({ id: n0 + j, predicted })) };

    const n1 = n0 + k;
    this.siteCount = n1;
    this.actualSites = n1;
    this.activeEstimate += k;
    updateClearParamsBuffer(queue, this.countParams, n1);
    updateGradParamsBuffer(queue, this.gradParams, n1, this.invScaleSq);

    // Seed the new sites into their home candidate pixels, then run a full
    // jump-flood sequence of candidate passes (long jumps down to 1 px) so they
    // claim their regions now. Left to the regular updates (every candFreq
    // iterations, 1 px steps by then) they would barely spread before the next
    // step, which would then see the same error in the same places.
    updateParamsBuffer(queue, this.growSeedParams, this._paramsArgs(n1, n0, 0));
    const stages = Math.max(1, Math.ceil(Math.log2(Math.max(this.candWidth, this.candHeight, 2))));
    this.growJfaParams ??= [];
    for (let st = 0; st < stages; st += 1) {
      this.growJfaParams[st] ??= createParamsBuffer(device, this._paramsArgs(n1, 0, 0));
      updateParamsBuffer(queue, this.growJfaParams[st],
        this._paramsArgs(n1, packJumpStep(st, this.candWidth, this.candHeight), 0));
    }
    const seedEnc = device.createCommandEncoder();
    seedEnc.copyTextureToTexture({ texture: this.cand0A }, { texture: this.cand0B },
      [this.candWidth, this.candHeight, 1]);
    this.seedEnc.encode(seedEnc, this.sitesBuf, this.growSeedParams, this.cand0A, this.cand0B, k);
    [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
    this._maybeRefreshHilbert(seedEnc, n1);
    this.packEnc.encode(seedEnc, this.sitesBuf, this.packedSitesBuf, this.countParams, n1);
    for (let st = 0; st < stages; st += 1) {
      this.candsEnc.encodeUpdate(seedEnc, this.packedSitesBuf, this.growJfaParams[st],
        this.cand0A, this.cand1A, this.cand0B, this.cand1B,
        this._hilbertOrderBuf(), this._hilbertPosBuf(), this.candWidth, this.candHeight);
      [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
      [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
    }
    queue.submit([seedEnc.finish()]);

    this._updateFitTarget();
    this._log(`Grow @${it}: +${k} → ${this.activeEstimate} sites` +
              (k < want ? ` (${want} wanted; the rest under λ` +
                (args.growMode === "split" && want > this.activeEstimate - k ? " or no cell left to split)" : ")") : "") +
              ` | best ${args.growMode === "insert" ? "insertion" : "split"} ` +
              `${fmtLambda(batch.best / BITS_PER_SITE)} ${this._gainUnit()}/bit` +
              (this.fitSigma ? ` | blur σ ${this.fitSigma.toFixed(1)}px` : ""));
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
  // of "render" (color), "hashed" (cell-id hashColor blend) or "heatmap"
  // (tau centroid heatmap).
  async renderView(mode) {
    const device = this.device;
    const queue = device.queue;
    updateParamsBuffer(queue, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));

    const encoder = device.createCommandEncoder();
    if (mode === "hashed") {
      this.renderHashedEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    } else if (mode === "heatmap") {
      const heat = this._taskHeatmapParamsBuffer();
      this.renderTauEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex,
        heat, this.width, this.height);
    } else {
      this.renderEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    }
    queue.submit([encoder.finish()]);
    return await readRgba32FloatTexture(device, this.renderTex, this.width, this.height);
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
    this.editSeedParams ??= createParamsBuffer(device, this._paramsArgs(n, 0, 0));
    updateParamsBuffer(queue, this.editSeedParams, this._paramsArgs(i + 1, i, 0));
    const stages = Math.max(1, Math.ceil(Math.log2(Math.max(this.candWidth, this.candHeight, 2))));
    this.growJfaParams ??= [];
    for (let st = 0; st < stages; st += 1) {
      this.growJfaParams[st] ??= createParamsBuffer(device, this._paramsArgs(n, 0, 0));
      updateParamsBuffer(queue, this.growJfaParams[st],
        this._paramsArgs(n, packJumpStep(st, this.candWidth, this.candHeight), 0));
    }
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture({ texture: this.cand0A }, { texture: this.cand0B },
      [this.candWidth, this.candHeight, 1]);
    this.seedEnc.encode(encoder, this.sitesBuf, this.editSeedParams, this.cand0A, this.cand0B, 1);
    [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
    this.hilbertReady = false; // the moved site's place in the Hilbert order
    this._maybeRefreshHilbert(encoder, n);
    this.packEnc.encode(encoder, this.sitesBuf, this.packedSitesBuf, this.countParams, n);
    for (let st = 0; st < stages; st += 1) {
      this.candsEnc.encodeUpdate(encoder, this.packedSitesBuf, this.growJfaParams[st],
        this.cand0A, this.cand1A, this.cand0B, this.cand1B,
        this._hilbertOrderBuf(), this._hilbertPosBuf(), this.candWidth, this.candHeight);
      [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
      [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
    }
    queue.submit([encoder.finish()]);
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
      this.cand0A, this.cand1A, this.fitTex, this.maskBuf,
      this._gnBuf, this.width, this.height);
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
      width: this.width,
      height: this.height,
      bestPsnr: this.bestPsnr,
      growDoneIter: this.growDoneIter,
      growStop: this.growStop,
      growthGain: this.growGainHistory || [],
      best: this.best,
      restoredBest: this.restoredBest,
      pruneStop: this.pruneStop,
      pruneMerged: this.pruneMerged,
      activeSites: this.activeEstimate,
      trainSeconds: (performance.now() - this.trainStartMs) / 1000,
      diag: await this._infoDiag(sitesArray),
    };
  }

  // Layer / info-pass state at the end of a run: the histogram mode, whether
  // a layer was added, the mean loss weight over masked pixels, the committed
  // fraction ("commit") and how many active unpinned sites were frozen.
  async _infoDiag(sitesRgb) {
    const n = this.width * this.height;
    const [weight, settled, live] = await Promise.all([
      readBufferF32(this.device, this.lossWeightBuf, n),
      readBufferU32(this.device, this.settledBuf, n),
      readBufferU32(this.device, this.siteLiveBuf, this.actualSites),
    ]);
    let wsum = 0, counted = 0, committed = 0;
    for (let p = 0; p < n; p += 1) {
      if (this.maskCpu && !(this.maskCpu[p] > 0)) continue;
      counted += 1;
      wsum += weight[p];
      if (this.args.info === "commit" && settled[p] >= this.args.commitAfter) committed += 1;
    }
    let frozen = 0;
    for (let i = 0; i < this.actualSites; i += 1) {
      if (sitesRgb[i * SITE_FLOATS] >= 0 && !this.sitePinned[i] && !live[i]) frozen += 1;
    }
    return {
      modeMass: this.histMode.mass, layer: this.layerIndex === null ? 0 : 1,
      meanWeight: counted ? wsum / counted : 0, committedFrac: counted ? committed / counted : 0,
      frozenSites: this.args.info === "commit" && this.args.commitFreeze ? frozen : 0,
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
      this.cand0A, this.cand0B, this.cand1A, this.cand1B,
      this.sitesBuf, this.packedSitesBuf, this.adamBuf, this.seedCounter, this.grads,
      this.tauGradRaw, this.tauGradTmp, this.tauGradTmp2,
      ...(this.candParams || []),
      this.renderParams, this.gradParams, this.adamParams,
      this.countParams, this.gradCountParams, this.tauParams,
      this.dummyHilbertBuf, this.hilbertParams, this._tauHeatBuf, this._gnBuf,
      this.blurTex, this.blurTmpTex, this.growBlurParams, this.growSeedParams,
      ...(this.growJfaParams || []), this.siteStatsParams, this.siteStatsBuf,
      this.bestSitesBuf, this.bestAdamBuf, this.rebuildParams, this.editSeedParams,
      this.pruneSitesBuf, this.pruneAdamBuf,
      this.sitePinBuf, this.siteLiveBuf, this.maskBuf, this.lossWeightBuf, this.settledBuf, this.infoParams,
    ];
    if (this.hilbertRes) {
      owned.push(this.hilbertRes.pairs, this.hilbertRes.order, this.hilbertRes.pos);
    }
    if (this.hilbertSort) {
      const s = this.hilbertSort;
      owned.push(s.histBuffer, s.blockSums, s.scratch, ...s.paramsBuffers);
    }
    for (const res of owned) res?.destroy();
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


// Description length in bits of sites [0, n)'s cells, each at its mean color
// (site_stats.wgsl's owner moments): `sites` × BITS_PER_SITE plus `samples`
// residuals coded as a Gaussian at 8-bit precision, floored at 0 bits each.
function cellMdlBits(stats, n, sites, samples) {
  let sse = 0;
  for (let i = 0; i < n; i += 1) {
    const b = i * SITE_STAT_CH;
    const cnt = stats[b];
    if (!(cnt > 0)) continue;
    let ss = stats[b + 15];
    for (let c = 0; c < 3; c += 1) ss -= stats[b + 6 + c] ** 2 / cnt;
    sse += Math.max(0, ss);
  }
  const mse = (sse * 255 * 255) / Math.max(1, samples);
  return sites * BITS_PER_SITE + samples * Math.max(0, 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(mse, 1e-30)));
}

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

async function readBufferBytes(device, buffer, bytes) {
  const staging = device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, Math.max(4, bytes));
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}
const readBufferF32 = async (device, buffer, count) => new Float32Array(await readBufferBytes(device, buffer, count * 4));
const readBufferU32 = async (device, buffer, count) => new Uint32Array(await readBufferBytes(device, buffer, count * 4));

// Mean training loss per sample over the masked pixels: |r| for "l1", r² for
// "l2" (NaN → 0, +Inf → 1, as in computePsnrSafe).
function computeLoss(render, target, width, height, maskCpu, loss) {
  let acc = 0;
  let counted = 0;
  for (let i = 0; i < width * height; i += 1) {
    if (maskCpu && maskCpu[i] <= 0) continue;
    for (let c = 0; c < 3; c += 1) {
      let r = render[i * 4 + c];
      if (!Number.isFinite(r)) r = r === Number.POSITIVE_INFINITY ? 1 : 0;
      const d = r - target[i * 4 + c];
      acc += loss === "l2" ? d * d : Math.abs(d);
    }
    counted += 3;
  }
  return acc / Math.max(1, counted);
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
