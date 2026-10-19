// Uniform/param buffer creators for every pipeline (originally ported from
// wgpu-py's sad_shared.py and train_wgpu.py).
// Every struct layout here matches the WGSL declarations in sad_shared.wgsl.

export const SITE_FLOATS = 10;
export const ADAM_FLOATS = 24;
export const PACKED_CAND_BYTES = 16;

function u32View(buffer, offsetBytes, nU32) {
  return new Uint32Array(buffer, offsetBytes, nU32);
}
function f32View(buffer, offsetBytes, nF32) {
  return new Float32Array(buffer, offsetBytes, nF32);
}

// ---------------------------------------------------------------------------
// Params (used by seed_cand, vpt, candidate_pack (as ClearParams),
// render_compute, etc.)
// Layout (matches sad_shared.py create_params_buffer, struct "<IIIIfIfIIIIIIIII"):
//   u32 width, u32 height, u32 siteCount, u32 step,
//   f32 invScaleSq, u32 seed, f32 radiusScale, u32 radiusProbes,
//   u32 injectCount, u32 candDownscale, u32 candWidth, u32 candHeight,
//   u32 globalCand, u32 _pad × 3   (total 16*4 = 64 bytes)
// globalCand: a site id vpt offers to every pixel each pass (default none).
// ---------------------------------------------------------------------------

export const PARAMS_SIZE = 64;

export function writeParams(
  view,
  { width, height, siteCount, step = 0, seed = 0,
    radiusScale, radiusProbes, injectCount,
    candDownscale,
    candWidth, candHeight, globalCand = 0xffffffff },
) {
  const scale = Math.max(width, height);
  const invScaleSq = 1.0 / (scale * scale);
  const cd = Math.max(1, candDownscale | 0);
  const u = new Uint32Array(view.buffer, view.byteOffset, 16);
  const f = new Float32Array(view.buffer, view.byteOffset, 16);
  u[0] = width >>> 0;
  u[1] = height >>> 0;
  u[2] = siteCount >>> 0;
  u[3] = step >>> 0;
  f[4] = invScaleSq;
  u[5] = seed >>> 0;
  f[6] = radiusScale;
  u[7] = radiusProbes >>> 0;
  u[8] = injectCount >>> 0;
  u[9] = cd >>> 0;
  u[10] = candWidth >>> 0;
  u[11] = candHeight >>> 0;
  u[12] = globalCand >>> 0;
}

export function createParamsBuffer(device, opts) {
  const buf = device.createBuffer({
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateParamsBuffer(device.queue, buf, opts);
  return buf;
}

export function updateParamsBuffer(queue, buffer, opts) {
  const bytes = new ArrayBuffer(PARAMS_SIZE);
  writeParams(new DataView(bytes), opts);
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// InitParams (init_gradient.wgsl):
//   u32 numSites, f32 gradThreshold, u32 maxAttempts, u32 _pad,
//   f32 initLogTau, f32 initRadius, u32 _pad1, u32 _pad2  (32 bytes)
// Python equivalent: struct "<IfIIffII" in train_wgpu.main()
// ---------------------------------------------------------------------------

export function createInitParamsBuffer(device, numSites, gradThreshold, maxAttempts,
                                       initLogTau, initRadius) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = numSites >>> 0;
  f[1] = gradThreshold;
  u[2] = maxAttempts >>> 0;
  u[3] = 0;
  f[4] = initLogTau;
  f[5] = initRadius;
  u[6] = 0;
  u[7] = 0;
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, bytes);
  return buf;
}

// ---------------------------------------------------------------------------
// GradParams (gradients_tiled.wgsl):
//   u32 siteCount, u32 globalCand, u32 _pad0, u32 _pad1,
//   f32 invScaleSq, f32 _pad2, f32 _pad3, f32 _pad4     (32 bytes)
// globalCand: the site every pixel blends besides its list (none by default).
// ---------------------------------------------------------------------------

export function createGradParamsBuffer(device, siteCount, invScaleSq, globalCand = 0xffffffff) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateGradParamsBuffer(device.queue, buf, siteCount, invScaleSq, globalCand);
  return buf;
}

export function updateGradParamsBuffer(queue, buffer, siteCount, invScaleSq, globalCand = 0xffffffff) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = globalCand >>> 0;
  u[2] = 0;
  u[3] = 0;
  f[4] = invScaleSq;
  f[5] = 0;
  f[6] = 0;
  f[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// AdamParams (adam.wgsl):
//   f32 lrPos, lrTau, lrRadius, lrColor, lrDir, lrAniso, beta1, beta2, eps,
//   u32 t, u32 width, u32 height, u32 layerSite, u32 _pad × 3 (64 bytes).
// ---------------------------------------------------------------------------

export function createAdamParamsBuffer(device, opts) {
  const buf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  updateAdamParamsBuffer(device.queue, buf, opts);
  return buf;
}

export function updateAdamParamsBuffer(queue, buffer, {
  lrPos = 0, lrTau = 0, lrRadius = 0, lrColor = 0, lrDir = 0, lrAniso = 0,
  beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1, width, height, layerSite = 0xffffffff,
}) {
  const bytes = new ArrayBuffer(64);
  const f = new Float32Array(bytes);
  const u = new Uint32Array(bytes);
  f.set([lrPos, lrTau, lrRadius, lrColor, lrDir, lrAniso, beta1, beta2, eps], 0);
  u[9] = t >>> 0;
  u[10] = width >>> 0;
  u[11] = height >>> 0;
  u[12] = layerSite >>> 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// ClearParams: single u32 count + pad
// (used by clear_buffer_*.wgsl, candidate_pack.wgsl, tau_extract.wgsl, tau_writeback.wgsl).
// Slot 1 is f32 invScale (1 / max(width, height)), read only by candidate_pack.
// ---------------------------------------------------------------------------

export function createCountParamsBuffer(device, count, invScale = 0) {
  const buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateCountParamsBuffer(device.queue, buf, count, invScale);
  return buf;
}

export function updateCountParamsBuffer(queue, buffer, count, invScale = 0) {
  const bytes = new ArrayBuffer(16);
  const u = new Uint32Array(bytes);
  u[0] = count >>> 0;
  new Float32Array(bytes)[1] = invScale;
  u[2] = 0;
  u[3] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// Aliases to match Python naming
export const createClearParamsBuffer = createCountParamsBuffer;
export const updateClearParamsBuffer = updateCountParamsBuffer;

// ---------------------------------------------------------------------------
// SiteStatsParams (site_stats.wgsl): u32 siteCount, u32 globalCand, _pad x2,
// f32 invScaleSq, _pad x3 (32 bytes).
// ---------------------------------------------------------------------------

export function createUniform32(device) {
  return device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

// ---------------------------------------------------------------------------
// Jump-flood step packing (candidates.js, vpt.wgsl): the jump in the high 16
// bits, the pass index in the low 16. Jumps halve from half the grid down to
// 1 px over the first passes, then stay at 1 px.
// ---------------------------------------------------------------------------

export function packJumpStep(stepIndex, width, height) {
  const maxDim = Math.max(width, height);
  let pow2 = 1;
  while (pow2 < maxDim) pow2 <<= 1;
  if (pow2 <= 1) return 1;
  const stages = 32 - Math.clz32(pow2) - 1;
  let stage;
  if (stages <= 0) stage = 0;
  else stage = Math.min(stepIndex, stages - 1);
  let step = pow2 >> (stage + 1);
  step = Math.max(step, 1);
  step = Math.min(step, 0xffff);
  return ((step << 16) | (stepIndex & 0xffff)) >>> 0;
}
