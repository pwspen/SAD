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
  updateGrowGainParamsBuffer,
  packJumpStep,
  hilbertBitsForSize,
} from "./params.js?v=noflow-20261017";

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
  JFAClearEncoder,
  JFAFloodEncoder,
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
} from "./encoders.js?v=noflow-20261017";

import { RadixSortUInt2 } from "./radix_sort.js?v=shared-wgsl-20261017";
import { BITS_PER_SITE } from "./io.js";
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

// LR schedules (args.lrSchedule). All start with a linear warmup to 1 over
// `warmupIters`; the run lasts warmupIters + decayIters.
export const LR_SCHEDULES = {
  "warmup-cosine": "warmup + cosine",          // cosine decay to 0 over decayIters
  fixed: "fixed",                              // stays at 1 after warmup
  "decay-after-growth": "decay after growth",  // 1 until growth ends, then cosine to 0 by the end
};

// Multiplier applied to every base LR at step `it`. `growDoneIter` is the
// iteration growth ended (Infinity while still growing); only
// "decay-after-growth" uses it.
export function lrSchedule(it, warmupIters, decayIters, kind = "warmup-cosine", growDoneIter = 0) {
  if (it < warmupIters) return (it + 1) / warmupIters;
  if (kind === "fixed") return 1;
  const total = warmupIters + decayIters;
  const start = kind === "decay-after-growth" ? Math.max(warmupIters, growDoneIter) : warmupIters;
  if (it < start) return 1;
  const t = Math.min(1, (it - start) / Math.max(1, total - start));
  return 0.5 * (1 + Math.cos(Math.PI * t));
}

// Site count after each growth step: start × factor^k (at least +1 per step),
// capped at max. Returns the counts after steps 1, 2, ... (empty if start ≥ max).
export function growthCounts(start, factor, max) {
  const counts = [];
  for (let n = start; n < max;) {
    n = Math.min(max, Math.max(n + 1, Math.ceil(n * factor)));
    counts.push(n);
  }
  return counts;
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
    // WebGPU-specific defaults; training_config.json's DEFAULT_ITERS/SITES are
    // tuned for the other backends' densify/prune schedule, which JS doesn't run.
    // Total iters = warmupIters + decayIters; an explicit "iters" override
    // (e.g. from sweeps) sets the total and shortens the decay phase to fit.
    warmupIters: Math.max(0, Math.floor(Number(pick("warmupIters", "WARMUP_ITERS", 100)))),
    // Site count in "fixed" siteMode (ignored when growing; see below).
    sites: Number(pick("sites", null, 1000)),
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
    lrMult: Number(pick("lrMult", "LR_MULT", 3)),
    lrSchedule: String(pick("lrSchedule", "LR_SCHEDULE", "warmup-cosine")),
    // Site color parameterization (a key of COLOR_SPACES in color_spaces.js).
    // Sites stay RGB outside the trainer; only the optimized params change.
    colorSpace: String(pick("colorSpace", "COLOR_SPACE", "rgb")),
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
    // Site growth, used when siteMode is "grow": start from growInitSites and
    // every growEvery iterations multiply the count by growFactor (at least +1)
    // up to growMaxSites (which also sizes the GPU buffers). growLambda stops it
    // early: the slope -dD/dR, i.e. the squared error (8-bit levels², summed
    // over RGB) a new site must remove per bit it costs (BITS_PER_SITE bits).
    // 0 = no early stop (any positive gain still qualifies).
    growLambda: Math.max(0, Number(pick("growLambda", "GROW_LAMBDA", 400))),
    growMaxSites: Math.max(1, Math.floor(Number(pick("growMaxSites", "GROW_MAX_SITES", 64)))),
    growInitSites: Math.max(1, Math.floor(Number(pick("growInitSites", "GROW_INIT_SITES", 1)))),
    growEvery: Math.max(1, Math.floor(Number(pick("growEvery", "GROW_EVERY", 25)))),
    growFactor: Number(pick("growFactor", "GROW_FACTOR", 2)),
    // While growing, the fit target is blurred with sigma = growBlur × site
    // spacing √(mask area / sites); 0 = always fit the original.
    growBlur: Math.max(0, Number(pick("growBlur", "GROW_BLUR", 0))),
  };
  args.decayIters = Math.max(0, Math.floor(Number(overrides.iters !== undefined && overrides.decayIters === undefined
    ? overrides.iters - args.warmupIters
    : pick("decayIters", "DECAY_ITERS", 500))));
  args.iters = args.warmupIters + args.decayIters;
  // "fixed": exactly `sites` for the whole run. "grow": rate-distortion growth.
  const siteMode = pick("siteMode", "SITE_MODE", "grow");
  args.siteMode = String(siteMode);
  if (args.siteMode !== "fixed" && args.siteMode !== "grow") {
    throw new Error(`siteMode must be "fixed" or "grow" (got "${args.siteMode}")`);
  }
  if (!(args.growFactor > 1)) throw new Error(`growFactor must be > 1 (got ${args.growFactor})`);
  if (!(args.lrSchedule in LR_SCHEDULES)) {
    throw new Error(`lrSchedule must be one of ${Object.keys(LR_SCHEDULES).join(", ")} (got "${args.lrSchedule}")`);
  }
  // Hidden safety stop for growth (not in the UI): growth normally ends at
  // growMaxSites or by λ. Default: never.
  const growEnd = pick("growEndIter", "GROW_END_ITER", null);
  args.growEndIter = growEnd == null || growEnd === "" ? args.iters : Math.max(0, Math.floor(Number(growEnd)));
  colorSpaceFor(args.colorSpace); // fail early on an unknown name
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
    this.jfaClearEnc = new JFAClearEncoder(device, shaders);
    this.jfaFloodEnc = new JFAFloodEncoder(device, shaders);
    this.packEnc = new CandidatePackEncoder(device, shaders);
    this.renderEnc = new RenderEncoder(device, shaders);
    this.renderHashedEnc = new RenderHashedEncoder(device, shaders);
    this.renderTauEnc = new RenderTauHeatmapEncoder(device, shaders);
    this.gradsEnc = new GradientsEncoder(device, shaders);
    this.adamEnc = new AdamEncoder(device, shaders);
    this.tauEnc = new TauEncoder(device, shaders);
    this.clearI32Enc = new ClearBufferEncoder(device, shaders, { useI32: true });
    this.hilbertEnc = this.usesHilbert ? new HilbertEncoder(device, shaders) : null;
    this.usesGrow = args.siteMode === "grow";
    this.growEnc = this.usesGrow ? new GrowEncoder(device, shaders) : null;
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

    // Fixed mode: exactly initSiteCount sites. Grow mode: buffers hold up to
    // growMaxSites (or the loaded sites, if more) and the run starts from
    // growInitSites (or the loaded sites). Unused slots are inactive
    // (position.x < 0), which every kernel skips.
    if (this.usesGrow) {
      this.bufferCapacity = Math.max(initSites ? initSiteCount : 0, args.growMaxSites);
      if (!initSites) initSiteCount = Math.min(args.growInitSites, this.bufferCapacity);
    } else {
      this.bufferCapacity = initSiteCount;
    }
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
      this.growGainParams = createUniform32(device);
      this.growSeedParams = createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0));
      if (args.growBlur > 0) {
        this.blurTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false });
        this.blurTmpTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false });
      }
      this.growCellsBuf = null;
      this.growCellsCap = 0;
    }

    // Initialize sites
    if (initSites) {
      const maxBytes = this.bufferCapacity * SITE_FLOATS * 4;
      const src = initSites.byteLength <= maxBytes
        ? initSites
        : new Float32Array(initSites.buffer, initSites.byteOffset, maxBytes / 4);
      device.queue.writeBuffer(this.sitesBuf, 0,
        encodeSiteColors(src.slice(), initSiteCount, this.colorSpace));
      this.actualSites = initSiteCount;
      this.activeEstimate = 0;
      for (let i = 0; i < initSiteCount; i += 1) {
        if (initSites[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Loaded init sites: ${initSiteCount} (active ${this.activeEstimate})`);
    } else {
      const gradThreshold = 0.01;
      const initParamsBuf = createInitParamsBuffer(device, initSiteCount, gradThreshold, 256,
                                                    args.initLogTau, args.initRadius);
      const encoder = device.createCommandEncoder();
      this.initEnc.encode(encoder, this.sitesBuf, this.seedCounter, initParamsBuf,
                           this.targetTex, this.maskTex, initSiteCount);
      device.queue.submit([encoder.finish()]);
      const sitesData = await readSites(device, this.sitesBuf, initSiteCount);
      initParamsBuf.destroy();
      this.actualSites = initSiteCount;
      this.activeEstimate = 0;
      for (let i = 0; i < initSiteCount; i += 1) {
        if (sitesData[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Initialized sites: ${initSiteCount} (active ${this.activeEstimate})`);
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
    this.growStop = null;
    if (this.usesGrow) this._updateFitTarget();

    this.jumpPassIndex = 0;
    this.aniStartIter = ANI_START_ITER;
    this.trainStartMs = performance.now();
    this.bestPsnr = 0;
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
    };
  }

  _log(msg) { if (this.onLog) this.onLog({ type: "msg", message: msg }); }

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

      if (this.usesGrow) await this._growStep(it);
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

      const lrScale = args.lrMult * lrSchedule(it, args.warmupIters, args.decayIters, args.lrSchedule,
        this.growDoneIter ?? Infinity);
      const lrDir = it >= this.aniStartIter ? args.lrDirBase * lrScale : 0;
      const lrAniso = it >= this.aniStartIter ? args.lrAnisoBase * lrScale : 0;
      updateAdamParamsBuffer(queue, this.adamParams,
        args.lrPosBase * lrScale, args.lrTauBase * lrScale, args.lrRadiusBase * lrScale,
        args.lrColorBase * lrScale, lrDir, lrAniso, args.beta1, args.beta2, args.eps, it + 1, this.width, this.height);

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

      // Gradients
      this.gradsEnc.encode(encoder, this.sitesBuf, this.gradParams,
        this.cand0A, this.cand1A, this.fitTex, this.maskTex,
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
        this.adamParams, siteCount);

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
        const elapsed = (performance.now() - this.trainStartMs) / 1000;
        const itsPerSec = (it + 1) / Math.max(elapsed, 1e-6);
        if (this.onLog) {
          this.onLog({
            type: "iter",
            it, iters,
            lrScale,
            psnr,
            gradStats,
            activeSites,
            totalSites: siteCount,
            fitSigma: this.fitSigma || 0,
            growDoneIter: this.growDoneIter,
            growStop: this.growStop,
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

    const queue2 = device.queue;
    updateParamsBuffer(queue2, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));
    const encoder = device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    device.queue.submit([encoder.finish()]);
  }

  // ---- Site growth ----
  // Rate-distortion growth in batches. Every growEvery iterations the count is
  // multiplied by growFactor (growthCounts). Every dyadic block of a grid finer
  // than the site spacing is scored by the error one new site taking it over
  // would remove (growthBlocks), and the batch goes to the best blocks, skipping
  // any closer than the new spacing to one already picked, so a batch never
  // stacks sites on one spot. Only blocks whose gain beats λ·BITS_PER_SITE (the
  // cost of a site's bits; λ = 0 only requires a positive gain) qualify; growth
  // ends at growMaxSites, when no block qualifies, or at the hidden growEndIter
  // safety stop. Optionally (growBlur > 0) the fit target is the
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

  _endGrowth(it, reason) {
    this.growDoneIter = it;
    this.growStop = reason;
    this._updateFitTarget();
    this._log(`Growth done at iter ${it}: ${this.siteCount} sites (${reason}).`);
  }

  async _growStep(it) {
    const args = this.args;
    if (this.growDoneIter !== null) return;
    if (this.siteCount >= this.bufferCapacity) { this._endGrowth(it, "reached max sites"); return; }
    if (it >= args.growEndIter) { this._endGrowth(it, "growEndIter safety stop"); return; }
    if (it === 0 || it % args.growEvery !== 0) return;

    const device = this.device;
    const queue = device.queue;
    const n0 = this.siteCount;
    // Stats on cells of a quarter site spacing, but at most GROW_MAX_CELL px so
    // even the first few sites can land on a specific feature.
    const spacing = this._siteSpacing();
    const cell = Math.max(1, Math.min(GROW_MAX_CELL, Math.round(spacing / 4)));
    const cellsX = Math.ceil(this.width / cell);
    const cellsY = Math.ceil(this.height / cell);
    const cellCount = cellsX * cellsY;
    const cellBytes = cellCount * 12 * 4;
    if (cellBytes > this.growCellsCap) {
      this.growCellsBuf?.destroy();
      this.growCellsBuf = device.createBuffer({ size: cellBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.growCellsCap = cellBytes;
    }
    updateGrowGainParamsBuffer(queue, this.growGainParams,
      { width: this.width, height: this.height, cellSize: cell, cellsX, cellsY, siteCount: n0 });
    updateParamsBuffer(queue, this.renderParams, this._paramsArgs(n0, 0, 0));
    const staging = device.createBuffer({ size: cellBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    this.growEnc.encodeGain(encoder, this.growGainParams, this.sitesBuf, this.renderTex, this.fitTex,
      this.maskTex, this.cand0A, this.growCellsBuf, cellCount);
    encoder.copyBufferToBuffer(this.growCellsBuf, 0, staging, 0, cellBytes);
    queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const cells = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();

    const target = growthCounts(n0, args.growFactor, this.bufferCapacity)[0];
    const minDist = Math.sqrt(this.maskArea / target);
    const threshold = args.growLambda * BITS_PER_SITE;
    const picked = growthBlocks(cells, cellsX, cellsY, cell, Math.max(cell, spacing), threshold,
      target - n0, minDist);
    if (!picked.length) {
      this._endGrowth(it, `λ (stop): no new site would remove more than λ × ${BITS_PER_SITE} = ` +
                          `${fmtLambda(threshold)} levels²`);
      return;
    }

    // New sites: error centroid of each block, its mean fit color, its owners'
    // tau and radius (so it competes as an equal), isotropic. A small jitter
    // keeps them off existing positions.
    const k = picked.length;
    const data = new Float32Array(k * SITE_FLOATS);
    picked.forEach((blk, i) => {
      const o = i * SITE_FLOATS;
      data[o] = Math.min(this.width - 1, Math.max(0, blk.x + (Math.random() - 0.5) * 0.5));
      data[o + 1] = Math.min(this.height - 1, Math.max(0, blk.y + (Math.random() - 0.5) * 0.5));
      data[o + 2] = blk.ownerW > 0 ? blk.tau / blk.ownerW : args.initLogTau;
      data[o + 3] = blk.ownerW > 0 ? blk.radius / blk.ownerW : args.initRadius;
      data.set(blk.color, o + 4);
      data[o + 7] = 1;
    });
    queue.writeBuffer(this.sitesBuf, n0 * SITE_FLOATS * 4, encodeSiteColors(data, k, this.colorSpace));
    // Fresh Adam moments, with the birth step for per-site bias correction.
    const adam = new Float32Array(k * ADAM_FLOATS);
    for (let i = 0; i < k; i += 1) adam[i * ADAM_FLOATS + 20] = it;
    queue.writeBuffer(this.adamBuf, n0 * ADAM_FLOATS * 4, adam);

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
    this._log(`Grow @${it}: +${k} → ${n1} sites` + (k < target - n0 ? ` (${target - n0} wanted; no other spot both beats λ and is a spacing away)` : "") +
              ` | best gain ${fmtLambda(picked[0].gain / BITS_PER_SITE)} levels²/bit | spacing ${minDist.toFixed(0)}px` +
              (this.fitSigma ? ` | blur σ ${this.fitSigma.toFixed(1)}px` : ""));
    if (n1 >= this.bufferCapacity) this._endGrowth(it, "reached max sites");
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
      trainSeconds: (performance.now() - this.trainStartMs) / 1000,
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
      this.dummyHilbertBuf, this.hilbertParams, this._tauHeatBuf,
      this.blurTex, this.blurTmpTex, this.growBlurParams, this.growGainParams, this.growSeedParams,
      this.growCellsBuf, ...(this.growJfaParams || []),
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

// Places for up to `count` new sites, from grow_gain.wgsl's per-cell stats
// (`cells`, 12 floats per cell of `cell` px). Cells are merged 2×2 into a
// pyramid of aligned dyadic blocks, up to `maxBlock` px (about one site
// spacing), and each block is scored by the error a site taking it over with
// the block's mean color would remove: sum|fit - pred|² - sum|fit - mean|²,
// in 8-bit levels² summed over RGB. Scoring sub-spacing blocks matters
// because the remaining error concentrates at edges finer than the spacing.
// Blocks whose gain beats `threshold` are taken best first, skipping any whose
// site would land within `minDist` px of one already taken (non-maximum
// suppression across all scales). Returns [{ gain, size, x, y, color, tau,
// radius, ownerW }], best first.
const GROW_MAX_CELL = 32;

export function growthBlocks(cells, cellsX, cellsY, cell, maxBlock, threshold, count, minDist) {
  const L2 = 255 * 255;
  // Per block: weight, color sum, variance sum (about the block's own mean),
  // error energy, energy-weighted absolute position, owner tau/radius sums.
  let w = cellsX, h = cellsY;
  let lvl = new Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const b = (y * w + x) * 12;
      const n = cells[b];
      if (!(n > 0)) { lvl[y * w + x] = null; continue; }
      const e = cells[b + 5];
      lvl[y * w + x] = {
        n, c: [n * cells[b + 1], n * cells[b + 2], n * cells[b + 3]], v: cells[b + 4], e,
        ex: cells[b + 6] + e * x * cell, ey: cells[b + 7] + e * y * cell,
        tau: cells[b + 8], rad: cells[b + 9], ow: cells[b + 10],
      };
    }
  }
  const cands = [];
  for (let size = cell; ; size *= 2) {
    for (const blk of lvl) {
      if (!blk || !(blk.e > 0)) continue;
      const gain = (blk.e - blk.v) * L2;
      if (gain > threshold) cands.push({ blk, gain, size });
    }
    if (size * 2 > maxBlock || (w === 1 && h === 1)) break;
    // Merge 2×2; pooled variance = within-child spread + spread of child means.
    const w2 = Math.ceil(w / 2), h2 = Math.ceil(h / 2);
    const next = new Array(w2 * h2);
    for (let y = 0; y < h2; y += 1) {
      for (let x = 0; x < w2; x += 1) {
        const kids = [];
        for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
          const cx = 2 * x + dx, cy = 2 * y + dy;
          if (cx < w && cy < h && lvl[cy * w + cx]) kids.push(lvl[cy * w + cx]);
        }
        if (!kids.length) { next[y * w2 + x] = null; continue; }
        const m = { n: 0, c: [0, 0, 0], v: 0, e: 0, ex: 0, ey: 0, tau: 0, rad: 0, ow: 0 };
        for (const k of kids) {
          m.n += k.n; m.v += k.v; m.e += k.e; m.ex += k.ex; m.ey += k.ey;
          m.tau += k.tau; m.rad += k.rad; m.ow += k.ow;
          for (let c = 0; c < 3; c += 1) m.c[c] += k.c[c];
        }
        for (const k of kids) {
          let d2 = 0;
          for (let c = 0; c < 3; c += 1) d2 += (k.c[c] / k.n - m.c[c] / m.n) ** 2;
          m.v += k.n * d2;
        }
        next[y * w2 + x] = m;
      }
    }
    lvl = next; w = w2; h = h2;
  }
  cands.sort((a, b) => b.gain - a.gain);
  // Greedy suppression, with picks bucketed in a minDist grid so each test
  // only looks at the 3×3 neighbouring buckets.
  const out = [];
  const grid = new Map();
  const r2 = minDist * minDist;
  for (const { blk, gain, size } of cands) {
    if (out.length >= count) break;
    const x = blk.ex / blk.e, y = blk.ey / blk.e;
    const gx = Math.floor(x / minDist), gy = Math.floor(y / minDist);
    let near = false;
    for (let dy = -1; dy <= 1 && !near; dy += 1) {
      for (let dx = -1; dx <= 1 && !near; dx += 1) {
        for (const p of grid.get(`${gx + dx},${gy + dy}`) || []) {
          if ((p.x - x) ** 2 + (p.y - y) ** 2 < r2) { near = true; break; }
        }
      }
    }
    if (near) continue;
    const pick = { gain, size, x, y, color: blk.c.map((v) => v / blk.n), tau: blk.tau, radius: blk.rad, ownerW: blk.ow };
    out.push(pick);
    const key = `${gx},${gy}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(pick);
  }
  return out;
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
