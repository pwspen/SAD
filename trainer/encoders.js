// Pipeline + bind-group wrappers (originally ported from wgpu-py's encoders.py).
// One class per WGSL kernel entry point; bindings match the WGSL @group(0)
// declarations one-for-one.

// Pipelines per device, keyed by source, entry point and override constants:
// every Trainer builds the same ~20, and compiling them again for each run of
// a sweep cost twice the training time on MNIST.
const pipelineCache = new WeakMap();

function makePipeline(device, code, entryPoint = "main", constants = undefined) {
  let cache = pipelineCache.get(device);
  if (!cache) pipelineCache.set(device, (cache = new Map()));
  const key = `${entryPoint}\n${JSON.stringify(constants ?? null)}\n${code}`;
  const cached = cache.get(key);
  if (cached) return cached;
  const module = device.createShaderModule({ code });
  if (module.getCompilationInfo) {
    module.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type === "error") {
          console.error(`WGSL ${entryPoint} ${m.lineNum}:${m.linePos}: ${m.message}`);
        }
      }
    });
  }
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint, ...(constants ? { constants } : {}) },
  });
  cache.set(key, pipeline);
  return pipeline;
}

function getTextureView(viewCache, texture) {
  let view = viewCache.get(texture);
  if (!view) {
    view = texture.createView();
    viewCache.set(texture, view);
  }
  return view;
}

// Fresh bind groups per dispatch (cheap; caching them keyed on resources
// isn't worth the bookkeeping).
function getBindGroup(device, _cache, layout, entries) {
  return device.createBindGroup({ layout, entries });
}

function dispatch1D(encoder, pipeline, bg, workgroups) {
  const p = encoder.beginComputePass();
  p.setPipeline(pipeline);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(workgroups, 1, 1);
  p.end();
}

function dispatch2D(encoder, pipeline, bg, gx, gy) {
  const p = encoder.beginComputePass();
  p.setPipeline(pipeline);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(gx, gy, 1);
  p.end();
}

// ---------------------------------------------------------------------------

export class InitGradientEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.INIT_GRADIENT);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, seedCounter, initParams, targetTex, maskTex, numSites) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: seedCounter } },
      { binding: 2, resource: { buffer: initParams } },
      { binding: 3, resource: getTextureView(this._views, targetTex) },
      { binding: 4, resource: getTextureView(this._views, maskTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(numSites / 64));
  }
}

export class CandidatesEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.initPipeline = makePipeline(device, shaders.INIT_CAND);
    this.updatePipeline = makePipeline(device, shaders.VPT);
    this._views = new Map();
    this._bgs = new Map();
  }
  encodeInit(encoder, paramsBuf, cand0, cand1, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: getTextureView(this._views, cand0) },
      { binding: 2, resource: getTextureView(this._views, cand1) },
    ];
    const layout = this.initPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.initPipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
  encodeUpdate(encoder, packedSitesBuffer, paramsBuf, cand0In, cand1In, cand0Out, cand1Out,
                hilbertOrder, hilbertPos, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: packedSitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0In) },
      { binding: 3, resource: getTextureView(this._views, cand1In) },
      { binding: 4, resource: getTextureView(this._views, cand0Out) },
      { binding: 5, resource: getTextureView(this._views, cand1Out) },
      { binding: 6, resource: { buffer: hilbertOrder } },
      { binding: 7, resource: { buffer: hilbertPos } },
    ];
    const layout = this.updatePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.updatePipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
}

export class SeedCandidatesEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.SEED_CAND);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0In, cand0Out, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0In) },
      { binding: 3, resource: getTextureView(this._views, cand0Out) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 64));
  }
}

export class CandidatePackEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.CANDIDATE_PACK);
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, packedBuffer, paramsBuf, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: packedBuffer } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 256));
  }
}

// hashIds: color each site by hashColor(id) instead of its color (the
// cell-id view).
export class RenderEncoder {
  constructor(device, shaders, { hashIds = false } = {}) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.RENDER, "main", { HASH_IDS: hashIds ? 1 : 0 });
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0, cand1, outputTex, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, outputTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class GradientsEncoder {
  // gaussNewton: build the GN_MODE variant (per-site Gauss–Newton diagonal,
  // see gradients_tiled.wgsl) instead of the loss gradient. loss: the LOSS
  // id (0 = L2, 1 = L1).
  constructor(device, shaders, { gaussNewton = false, loss = 0 } = {}) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.GRADIENTS_TILED, "main",
      { GN_MODE: gaussNewton ? 1 : 0, LOSS: loss });
    this._views = new Map();
    this._bgs = new Map();
  }
  // gradsBuffer is the consolidated interleaved atomic<i32> buffer of size
  // capacity * 10; index with `idx * 10 + channel`. weightBuffer: per-pixel
  // loss weight, f32 W*H (the mask, or info_update's weights).
  encode(encoder, sitesBuffer, gradParams, cand0, cand1, targetTex, weightBuffer,
         gradsBuffer, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: gradParams } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, targetTex) },
      { binding: 5, resource: { buffer: gradsBuffer } },
      { binding: 6, resource: { buffer: weightBuffer } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class AdamEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.ADAM);
    this._bgs = new Map();
  }
  // sitePin / siteLive: u32 per site (see adam.wgsl).
  encode(encoder, sitesBuffer, adamBuffer, gradsBuffer, adamParams, siteCount, sitePin, siteLive) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: adamBuffer } },
      { binding: 2, resource: { buffer: gradsBuffer } },
      { binding: 3, resource: { buffer: adamParams } },
      { binding: 4, resource: { buffer: sitePin } },
      { binding: 5, resource: { buffer: siteLive } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 256));
  }
}

export class TauEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.extractPipeline = makePipeline(device, shaders.TAU_EXTRACT);
    this.diffusePipeline = makePipeline(device, shaders.TAU_DIFFUSE);
    this.writebackPipeline = makePipeline(device, shaders.TAU_WRITEBACK);
    this._views = new Map();
    this._bgs = new Map();
  }
  // gradsBuffer is the consolidated gradient buffer; extract reads channel 2
  // (log_tau) into a plain f32 buffer.
  encodeExtract(encoder, gradsBuffer, gradOut, clearParams, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: gradsBuffer } },
      { binding: 1, resource: { buffer: gradOut } },
      { binding: 2, resource: { buffer: clearParams } },
    ];
    const layout = this.extractPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.extractPipeline, bg, Math.ceil(siteCount / 256));
  }
  encodeDiffuse(encoder, cand0, cand1, sitesBuffer, gradRaw, gradIn, gradOut, tauParams, siteCount) {
    const entries = [
      { binding: 0, resource: getTextureView(this._views, cand0) },
      { binding: 1, resource: getTextureView(this._views, cand1) },
      { binding: 2, resource: { buffer: sitesBuffer } },
      { binding: 3, resource: { buffer: gradRaw } },
      { binding: 4, resource: { buffer: gradIn } },
      { binding: 5, resource: { buffer: gradOut } },
      { binding: 6, resource: { buffer: tauParams } },
    ];
    const layout = this.diffusePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.diffusePipeline, bg, Math.ceil(siteCount / 256));
  }
  // gradsBuffer is the consolidated gradient buffer; writeback writes diffused
  // tau values back to channel 2.
  encodeWriteback(encoder, gradIn, gradsBuffer, clearParams, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: gradIn } },
      { binding: 1, resource: { buffer: gradsBuffer } },
      { binding: 2, resource: { buffer: clearParams } },
    ];
    const layout = this.writebackPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.writebackPipeline, bg, Math.ceil(siteCount / 256));
  }
}

export class ClearBufferEncoder {
  constructor(device, shaders, { useI32 = false } = {}) {
    this.device = device;
    const code = useI32 ? shaders.CLEAR_BUFFER_I32 : shaders.CLEAR_BUFFER_U32;
    this.pipeline = makePipeline(device, code);
    this._bgs = new Map();
  }
  encode(encoder, buffer, clearParams, count) {
    const entries = [
      { binding: 0, resource: { buffer } },
      { binding: 1, resource: { buffer: clearParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(count / 256));
  }
}

// Softmax-blended hashColor rendering — same bindings as the normal render
// kernel (sites, params, cand0, cand1, outTex).
// Tau heatmap overlay — adds a TauHeatmapParams uniform (bind 5) on top of the
// normal render bindings.
export class RenderTauHeatmapEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.RENDER_TAU_HEATMAP);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0, cand1, outputTex, heatParams, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, outputTex) },
      { binding: 5, resource: { buffer: heatParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class HilbertEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.buildPipeline = makePipeline(device, shaders.HILBERT, "buildHilbertPairs");
    this.writePipeline = makePipeline(device, shaders.HILBERT, "writeHilbertOrder");
    this._bgs = new Map();
  }
  encodePairs(encoder, sitesBuffer, pairsBuffer, paramsBuf, paddedCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: pairsBuffer } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ];
    const layout = this.buildPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.buildPipeline, bg, Math.ceil(paddedCount / 256));
  }
  encodeOrder(encoder, pairsBuffer, orderBuffer, posBuffer, paramsBuf, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: pairsBuffer } },
      { binding: 1, resource: { buffer: orderBuffer } },
      { binding: 2, resource: { buffer: posBuffer } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ];
    const layout = this.writePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.writePipeline, bg, Math.ceil(siteCount / 256));
  }
}

// Site growth: mask-normalized separable Gaussian blur of the target (into
// `outTex`, via `tmpTex`).
export class GrowEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.blurHPipeline = makePipeline(device, shaders.GROW_BLUR, "blur_h");
    this.blurVPipeline = makePipeline(device, shaders.GROW_BLUR, "blur_v");
    this._views = new Map();
  }
  encodeBlur(encoder, blurParams, srcTex, maskTex, tmpTex, outTex, width, height) {
    const dev = this.device;
    const view = (t) => getTextureView(this._views, t);
    let bg = dev.createBindGroup({ layout: this.blurHPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: blurParams } },
      { binding: 1, resource: view(srcTex) },
      { binding: 2, resource: view(maskTex) },
      { binding: 3, resource: view(tmpTex) },
    ] });
    dispatch2D(encoder, this.blurHPipeline, bg, Math.ceil(width / 16), Math.ceil(height / 16));
    bg = dev.createBindGroup({ layout: this.blurVPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: blurParams } },
      { binding: 1, resource: view(tmpTex) },
      { binding: 3, resource: view(outTex) },
    ] });
    dispatch2D(encoder, this.blurVPipeline, bg, Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

// Site growth: per-site owner moments and removal deltas (site_stats.wgsl).
export class SiteStatsEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.SITE_STATS);
    this._views = new Map();
  }
  encode(encoder, sitesBuffer, statsParams, cand0, cand1, fitTex, weightBuffer, statsBuffer, width, height, sitePin) {
    const view = (t) => getTextureView(this._views, t);
    const bg = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: statsParams } },
      { binding: 2, resource: view(cand0) },
      { binding: 3, resource: view(cand1) },
      { binding: 4, resource: view(fitTex) },
      { binding: 5, resource: { buffer: weightBuffer } },
      { binding: 6, resource: { buffer: statsBuffer } },
      { binding: 7, resource: { buffer: sitePin } },
    ] });
    encoder.clearBuffer(statsBuffer);
    dispatch2D(encoder, this.pipeline, bg, Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

// Remaining-information pass (info_update.wgsl): per-pixel loss weights and
// commit counts, plus which sites still serve a pixel in the loss.
export class InfoEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.INFO_UPDATE);
    this._views = new Map();
  }
  encode(encoder, { sites, params, cand0, cand1, targetTex, mask, weight, settled, siteLive, width, height }) {
    const view = (t) => getTextureView(this._views, t);
    const bg = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: sites } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: view(cand0) },
      { binding: 3, resource: view(cand1) },
      { binding: 4, resource: view(targetTex) },
      { binding: 5, resource: { buffer: mask } },
      { binding: 6, resource: { buffer: weight } },
      { binding: 7, resource: { buffer: settled } },
      { binding: 8, resource: { buffer: siteLive } },
    ] });
    encoder.clearBuffer(siteLive);
    dispatch2D(encoder, this.pipeline, bg, Math.ceil(width / 16), Math.ceil(height / 16));
  }
}
