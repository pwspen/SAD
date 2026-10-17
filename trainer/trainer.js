// SAD training loop. Originally ported from the wgpu-py trainer (train_wgpu.py
// at the `multi-backend` tag), minus densify/prune: it optimizes a fixed site
// count for the whole run.

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
  packJumpStep,
  hilbertBitsForSize,
} from "./params.js?v=shared-wgsl-20261017";

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
} from "./encoders.js?v=shared-wgsl-20261017";

import { RadixSortUInt2 } from "./radix_sort.js?v=shared-wgsl-20261017";

const MAX_PREVIEW_IMAGE_READBACKS = 2;
// Direction/anisotropy start learning here. Previously derived from the prune
// schedule as max(10, PRUNE_START / 2) = 50 with the shared config.
const ANI_START_ITER = 50;
const MAX_PREVIEW_SITE_READBACKS = 8;

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
    iters: Number(pick("iters", null, 10000)),
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
  };
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

    // Site count is fixed for the whole run (no densify/prune).
    this.siteCount = initSiteCount;
    this.bufferCapacity = initSiteCount;

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
    this.gradCountParams = createClearParamsBuffer(device, initSiteCount * 10);
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

    // Initialize sites
    if (initSites) {
      const maxBytes = this.bufferCapacity * SITE_FLOATS * 4;
      const src = initSites.byteLength <= maxBytes
        ? initSites
        : new Float32Array(initSites.buffer, initSites.byteOffset, maxBytes / 4);
      device.queue.writeBuffer(this.sitesBuf, 0, src);
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

      const lrDir = it >= this.aniStartIter ? args.lrDirBase : 0;
      const lrAniso = it >= this.aniStartIter ? args.lrAnisoBase : 0;
      updateAdamParamsBuffer(queue, this.adamParams,
        args.lrPosBase, args.lrTauBase, args.lrRadiusBase, args.lrColorBase,
        lrDir, lrAniso, args.beta1, args.beta2, args.eps, it + 1, this.width, this.height);

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
        this.cand0A, this.cand1A, this.targetTex, this.maskTex,
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
        const sitesData = await readSites(device, this.sitesBuf, siteCount);
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
            psnr,
            gradStats,
            activeSites,
            totalSites: siteCount,
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
      const sitesData = new Float32Array(req.siteStaging.getMappedRange()).slice();
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

  async finalize() {
    const rgba = await readRgba32FloatTexture(this.device, this.renderTex, this.width, this.height);
    const sitesArray = await readSites(this.device, this.sitesBuf, this.actualSites);
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
