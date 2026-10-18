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
  createFlowParamsBuffer,
  updateFlowParamsBuffer,
  createPyrParamsBuffer,
  updatePyrParamsBuffer,
  updateGradParamsBuffer,
  updateClearParamsBuffer,
  createUniform32,
  updateGrowBlurParamsBuffer,
  updateGrowGainParamsBuffer,
  pyramidLevels,
  packJumpStep,
  hilbertBitsForSize,
} from "./params.js?v=grow-20261017";

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
  FlowEncoder,
  PyramidEncoder,
  GrowEncoder,
} from "./encoders.js?v=grow-20261017";

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

// Linear warmup to 1 over `warmupIters`, then cosine decay to 0 over
// `decayIters`. Returns the multiplier applied to every base LR at step `it`.
export function lrSchedule(it, warmupIters, decayIters) {
  if (it < warmupIters) return (it + 1) / warmupIters;
  const t = Math.min(1, (it - warmupIters) / Math.max(1, decayIters));
  return 0.5 * (1 + Math.cos(Math.PI * t));
}

// Residual-flow strength multiplier at step `it`: 0 before `startIter`, then a
// linear ramp from 1 down to 0 at `endIter`, so the end of training is pure
// gradient descent on the loss.
export function flowSchedule(it, startIter, endIter) {
  if (it < startIter || it >= endIter) return 0;
  return 1 - (it - startIter) / Math.max(1, endIter - startIter);
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
    // Residual-driven site flow (error-weighted Lloyd). flowStrength is the
    // fraction of the way to the cell's error centroid moved per step; 0 = off.
    flowStrength: Number(pick("flowStrength", "FLOW_STRENGTH", 0)),
    flowStartIter: Number(pick("flowStartIter", "FLOW_START_ITER", 200)),
    flowEndIter: Number(pick("flowEndIter", "FLOW_END_ITER", 2000)),
    flowMaxStep: Number(pick("flowMaxStep", "FLOW_MAX_STEP", 1.0)),
    flowEmaAlpha: Number(pick("flowEmaAlpha", "FLOW_EMA_ALPHA", 0.1)),
    // 0 = weighted Lloyd (error centroid); 1 = pure residual pull (error
    // centroid minus geometric centroid, so uniform error doesn't move sites).
    flowGeoMix: Number(pick("flowGeoMix", "FLOW_GEO_MIX", 0)),
    // Multi-scale pull toward regions with more error per site than average
    // (see flow_step.wgsl); 0 = off. flowPyrCell is the finest cell size in px.
    flowPyrStrength: Number(pick("flowPyrStrength", "FLOW_PYR_STRENGTH", 0)),
    flowPyrCell: Math.max(1, Math.floor(Number(pick("flowPyrCell", "FLOW_PYR_CELL", 8)))),
    flowPyrLevels: Math.max(0, Math.floor(Number(pick("flowPyrLevels", "FLOW_PYR_LEVELS", 0)))),
    // Target density exponent: the pyramid equalizes err^gamma per site.
    flowPyrGamma: Number(pick("flowPyrGamma", "FLOW_PYR_GAMMA", 1)),
    // Site growth (rate-distortion). growLambda is the final slope -dD/dR: the
    // squared error (8-bit levels², summed over RGB) a site must remove per bit
    // it costs (BITS_PER_SITE bits). 0 = off: the site count stays fixed. With
    // growth on, `sites` is the cap and runs start from growInitSites.
    growLambda: Number(pick("growLambda", "GROW_LAMBDA", 0)),
    growInitSites: Math.max(1, Math.floor(Number(pick("growInitSites", "GROW_INIT_SITES", 256)))),
    growEvery: Math.max(1, Math.floor(Number(pick("growEvery", "GROW_EVERY", 10)))),
    // While growing, the fit target is blurred with sigma = growBlur × site
    // spacing √(mask area / sites); 0 = always fit the original.
    growBlur: Math.max(0, Number(pick("growBlur", "GROW_BLUR", 0))),
  };
  args.decayIters = Math.max(0, Math.floor(Number(overrides.iters !== undefined && overrides.decayIters === undefined
    ? overrides.iters - args.warmupIters
    : pick("decayIters", "DECAY_ITERS", 500))));
  args.iters = args.warmupIters + args.decayIters;
  // Growth (and the blurred target) stops here; the rest fits the original.
  const growEnd = pick("growEndIter", "GROW_END_ITER", null);
  args.growEndIter = Math.max(0, Math.floor(growEnd == null || growEnd === "" ? 0.6 * args.iters : Number(growEnd)));
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
    this.usesFlow = (args.flowStrength > 0 || args.flowPyrStrength > 0) &&
      args.flowEndIter > args.flowStartIter;
    this.usesPyr = this.usesFlow && args.flowPyrStrength > 0;
    this.flowEnc = this.usesFlow ? new FlowEncoder(device, shaders) : null;
    this.pyrEnc = this.usesPyr ? new PyramidEncoder(device, shaders) : null;
    this.usesGrow = args.growLambda > 0;
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

    // Without growth the site count is fixed. With growth, buffers hold up to
    // `sites` and the run starts from growInitSites (or the loaded sites).
    // Unused slots are inactive (position.x < 0), which every kernel skips.
    this.bufferCapacity = this.usesGrow ? Math.max(initSiteCount, args.sites) : initSiteCount;
    if (this.usesGrow && !initSites) initSiteCount = Math.min(args.growInitSites, this.bufferCapacity);
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

    // Residual flow: per-pixel error EMA + per-site centroid moments (6 f32).
    // WebGPU zero-initializes both; flow_step re-zeroes the moments after use.
    if (this.usesFlow) {
      this.errEmaBuf = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.STORAGE });
      this.flowBuf = createAtomicU32Buffer(device, this.bufferCapacity * 6);
      this.flowParams = createFlowParamsBuffer(device, this._flowParamsArgs(initSiteCount, 0));
      // Pyramid (only allocated at full size when the pyramid term is on;
      // flow_step always binds it).
      const pyr = this.usesPyr
        ? pyramidLevels(width, height, args.flowPyrCell)
        : { levels: [{ w: 1, h: 1, offset: 0, cell: Math.max(width, height) }], totalCells: 1 };
      this.pyrLevels = pyr.levels;
      this.pyrCnt0 = device.createBuffer({
        size: pyr.levels[0].w * pyr.levels[0].h * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.pyrCells = device.createBuffer({ size: pyr.totalCells * 8, usage: GPUBufferUsage.STORAGE });
      this.pyrParams = pyr.levels.map((_, level) => createPyrParamsBuffer(device, {
        levels: pyr.levels, level, siteCount: initSiteCount, width, height, gamma: args.flowPyrGamma,
      }));
    }

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
      if (args.growBlur > 0 && args.growEndIter > 0) {
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

  _flowParamsArgs(siteCount, scale) {
    return {
      siteCount,
      width: this.width,
      height: this.height,
      geoMix: this.args.flowGeoMix,
      invScaleSq: this.invScaleSq,
      emaAlpha: this.args.flowEmaAlpha,
      strength: this.args.flowStrength * scale,
      maxStep: this.args.flowMaxStep,
      pyrStrength: this.args.flowPyrStrength * scale,
      pyrLevels: this.args.flowPyrLevels,
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

      const lrScale = args.lrMult * lrSchedule(it, args.warmupIters, args.decayIters);
      const lrDir = it >= this.aniStartIter ? args.lrDirBase * lrScale : 0;
      const lrAniso = it >= this.aniStartIter ? args.lrAnisoBase * lrScale : 0;
      updateAdamParamsBuffer(queue, this.adamParams,
        args.lrPosBase * lrScale, args.lrTauBase * lrScale, args.lrRadiusBase * lrScale,
        args.lrColorBase * lrScale, lrDir, lrAniso, args.beta1, args.beta2, args.eps, it + 1, this.width, this.height);

      const blend = it / Math.max(1, iters);
      const lambda = TAU_DIFFUSE_LAMBDA * (0.1 + 0.9 * blend);
      updateTauParamsBuffer(queue, this.tauParams, siteCount, lambda, this.candDownscale);

      const flowScale = this.usesFlow ? flowSchedule(it, args.flowStartIter, args.flowEndIter) : 0;
      if (flowScale > 0) {
        updateFlowParamsBuffer(queue, this.flowParams, this._flowParamsArgs(siteCount, flowScale));
      }

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

      // Residual flow moments, from the same pre-Adam sites as the gradients.
      if (flowScale > 0) {
        this.flowEnc.encodeAccum(encoder, this.sitesBuf, this.flowParams,
          this.cand0A, this.cand1A, this.fitTex, this.maskTex,
          this.errEmaBuf, this.flowBuf, this.width, this.height);
      }

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

      // Residual flow step, added on top of the Adam position update. The
      // pyramid bins the post-Adam positions it is about to move.
      if (flowScale > 0) {
        if (this.usesPyr) {
          this.pyrEnc.encode(encoder, this.sitesBuf, this.errEmaBuf, this.pyrCnt0, this.pyrCells,
            this.pyrParams, this.pyrLevels, siteCount);
        }
        this.flowEnc.encodeStep(encoder, this.sitesBuf, this.flowBuf, this.flowParams,
          this.pyrCells, this.pyrParams[0], siteCount);
      }

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
            flowScale,
            psnr,
            gradStats,
            activeSites,
            totalSites: siteCount,
            fitSigma: this.fitSigma || 0,
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
  // Rate-distortion: minimize D + λ·R, with R = BITS_PER_SITE bits per site.
  // Every growEvery iterations (before growEndIter), each cell of a grid at the
  // current site spacing h = √(mask area / sites) is scored by the most error
  // one new site could remove in it (pickGrowthBlocks); cells whose gain beats
  // λ·BITS_PER_SITE get a site. One site per cell caps a step at roughly
  // doubling the count, so growth runs coarse to fine and stops by itself once
  // no cell pays for a site. Optionally (growBlur > 0) the fit target is the
  // image blurred to the current spacing; in tests that cost 0.5-2 dB at a
  // matched site count, since sites place edges far finer than their spacing.

  _siteSpacing() { return Math.sqrt(this.maskArea / Math.max(1, this.siteCount)); }

  // Point the gradients at the blurred (while growing) or original target.
  _updateFitTarget() {
    const sigma = this.blurTex && this._growing(this._iter ?? 0)
      ? this.args.growBlur * this._siteSpacing() : 0;
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

  _growing(it) { return it < this.args.growEndIter; }

  async _growStep(it) {
    this._iter = it;
    const args = this.args;
    if (!this._growing(it)) {
      if (this.fitTex !== this.targetTex) {
        this._updateFitTarget();
        this._log(`Growth done at iter ${it}: ${this.siteCount} sites; fitting the original target.`);
      }
      return;
    }
    if (it === 0 || it % args.growEvery !== 0 || this.siteCount >= this.bufferCapacity) return;

    const device = this.device;
    const queue = device.queue;
    const n0 = this.siteCount;
    // Stats on a grid of h/4 cells; each spacing-sized parent of 4×4 of them
    // can receive one site (so a step at most about doubles the count).
    const cell = Math.max(1, Math.round(this._siteSpacing() / GROW_SUBDIV));
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

    const lambda = args.growLambda;
    const picked = pickGrowthBlocks(cells, cellsX, cellsY, cell, lambda * BITS_PER_SITE);
    picked.sort((a, b) => b.gain - a.gain);
    picked.length = Math.min(picked.length, this.bufferCapacity - n0);
    if (!picked.length) return;

    // New sites: error centroid of the block, its mean fit color, its owners'
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
      data[o + 4] = blk.color[0];
      data[o + 5] = blk.color[1];
      data[o + 6] = blk.color[2];
      data[o + 7] = 1;
      data[o + 8] = 0;
      data[o + 9] = 0;
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
    if (this.usesPyr) {
      for (let l = 0; l < this.pyrLevels.length; l += 1) {
        updatePyrParamsBuffer(queue, this.pyrParams[l], {
          levels: this.pyrLevels, level: l, siteCount: n1, width: this.width, height: this.height,
          gamma: args.flowPyrGamma,
        });
      }
    }

    // Seed only the new sites into their home candidate pixels; the regular
    // candidate updates spread them from there.
    updateParamsBuffer(queue, this.growSeedParams, this._paramsArgs(n1, n0, 0));
    const seedEnc = device.createCommandEncoder();
    seedEnc.copyTextureToTexture({ texture: this.cand0A }, { texture: this.cand0B },
      [this.candWidth, this.candHeight, 1]);
    this.seedEnc.encode(seedEnc, this.sitesBuf, this.growSeedParams, this.cand0A, this.cand0B, k);
    queue.submit([seedEnc.finish()]);
    [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];

    this._updateFitTarget();
    this._log(`Grow @${it}: +${k} → ${n1} sites | best gain ${fmtLambda(picked[0].gain / BITS_PER_SITE)}/bit | ` +
              `cell ${cell * GROW_SUBDIV}px` + (this.fitSigma ? ` | blur σ ${this.fitSigma.toFixed(1)}px` : ""));
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
      this.errEmaBuf, this.flowBuf, this.flowParams, this.pyrCnt0, this.pyrCells,
      ...(this.pyrParams || []),
      this.blurTex, this.blurTmpTex, this.growBlurParams, this.growGainParams, this.growSeedParams,
      this.growCellsBuf,
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

// Growth candidates from grow_gain.wgsl's per-cell stats (`cells`, 12 floats
// per cell of `cell` px). Each parent of GROW_SUBDIV × GROW_SUBDIV cells (about
// one site spacing) is scored by its best dyadic block (the whole parent, its
// quarters, ... down to single cells): the error a site taking that block over
// with the block's mean color would remove, sum|fit - pred|² - sum|fit - mean|²,
// in 8-bit levels² summed over RGB. Remaining error concentrates at edges finer
// than the spacing, which a whole-parent score would miss. Returns the parents
// whose best gain beats `threshold`, as { gain, x, y, color, tau, radius, ownerW }.
const GROW_SUBDIV = 4;

export function pickGrowthBlocks(cells, cellsX, cellsY, cell, threshold) {
  const out = [];
  const L2 = 255 * 255;
  const block = (bx, by, size) => {
    let n = 0, e = 0, ex = 0, ey = 0, tau = 0, rad = 0, ow = 0, var0 = 0;
    const mean = [0, 0, 0];
    const parts = [];
    for (let y = by; y < Math.min(by + size, cellsY); y += 1) {
      for (let x = bx; x < Math.min(bx + size, cellsX); x += 1) {
        const b = (y * cellsX + x) * 12;
        const ni = cells[b];
        if (!(ni > 0)) continue;
        n += ni;
        for (let c = 0; c < 3; c += 1) mean[c] += ni * cells[b + 1 + c];
        var0 += cells[b + 4];
        const ei = cells[b + 5];
        e += ei;
        ex += cells[b + 6] + ei * x * cell;
        ey += cells[b + 7] + ei * y * cell;
        tau += cells[b + 8];
        rad += cells[b + 9];
        ow += cells[b + 10];
        parts.push(b);
      }
    }
    if (n < 1) return null;
    for (let c = 0; c < 3; c += 1) mean[c] /= n;
    // Pooled variance: within-cell spread plus spread of the cell means.
    let v = var0;
    for (const b of parts) {
      let d2 = 0;
      for (let c = 0; c < 3; c += 1) d2 += (cells[b + 1 + c] - mean[c]) ** 2;
      v += cells[b] * d2;
    }
    const gain = (e - v) * L2;
    if (!Number.isFinite(gain)) return null;
    return { gain, x: e > 0 ? ex / e : (bx + size / 2) * cell, y: e > 0 ? ey / e : (by + size / 2) * cell,
             color: mean, tau, radius: rad, ownerW: ow };
  };
  for (let py = 0; py < cellsY; py += GROW_SUBDIV) {
    for (let px = 0; px < cellsX; px += GROW_SUBDIV) {
      let best = null;
      for (let size = GROW_SUBDIV; size >= 1; size >>= 1) {
        for (let by = py; by < py + GROW_SUBDIV; by += size) {
          for (let bx = px; bx < px + GROW_SUBDIV; bx += size) {
            const blk = block(bx, by, size);
            if (blk && (!best || blk.gain > best.gain)) best = blk;
          }
        }
      }
      if (best && best.gain > threshold) out.push(best);
    }
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
