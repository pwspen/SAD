// Gradients live in one interleaved buffer `grads[idx*10+ch]` (WebGPU allows
// only 10 storage buffers per shader stage):
//   0 pos_x, 1 pos_y, 2 log_tau, 3 radius_sq,
//   4 color_r, 5 color_g, 6 color_b,
//   7 dir_x, 8 dir_y, 9 log_aniso
// Each slot holds f32 bits (atomic<u32>, summed with a CAS loop). It used to be
// fixed-point i32 (x1e6), which wraps once a site's per-pixel sum exceeds ~2147
// — routine at low site counts, where each site covers thousands of pixels.
// Per-tile sums stay fixed-point (fast integer atomics) with a carry word, so
// they're exact 64-bit; only the once-per-site-per-tile flush pays for the CAS.
//
// Workgroup-tiled atomic accumulation (open-addressed hash per workgroup) that
// flushes to `grads` once at the end. This is the speed-critical path — direct
// global atomics per pixel was ~10x slower. To satisfy Dawn's strict uniformity
// analysis (barriers must be reachable from uniform control flow), NO thread
// takes an early `return` before either barrier: every skip is expressed as an
// `if (guard) { work }` with no return inside.
//
// GN_MODE (a pipeline-overridable constant) turns the same pass into the
// Gauss–Newton diagonal of squared error (whatever LOSS is): per site and parameter,
// Σ_pixels Σ_rgb (∂pred/∂θ)², with color taken in decoded RGB. That is the
// predicted rise in summed squared error per unit² change of θ, used by the
// viewer's site-clustering analysis; training never runs it. GN sums skip the
// fixed-point tile path (squared terms can exceed its per-pixel range) and
// add straight to `grads` as f32.

override GN_MODE : bool = false;

// Training loss on the per-channel error e = pred − tgt (args.loss, see LOSSES
// in trainer.js): 0 = L2, Σ e²; 1 = L1, Σ |e|. Under L1 each site's color
// settles at the weighted median of its pixels rather than the mean, so a cell
// that is mostly one color takes that color instead of a blend.
override LOSS : u32 = 0u;

// Persistent-error weighting. Each pixel keeps an EWMA of its signed error
// ē ← keep·ē + (1 − keep)·e in errEma.xyz (keep = emaKeep, the trainer's
// β ramped in so the first steps average instead of decaying from 0). Its
// bias b = |ē|² (|ē| under L1, so b is in the loss's units) is high where the
// error stays the same sign step after step; error that flips sign averages
// out. The pixel's loss is scaled by the stop-gradient weight
//   w = (1 + α · min(b / mean b, cap)) / (1 + α)
// with mean b over the masked pixels from the previous step (errEmaSum, two
// slots ping-ponged by emaSlot; each workgroup adds its tile's Σb once). The
// division by 1 + α keeps the mean weight near 1. α = 0 gives w = 1. errEma.w
// holds w for site_stats.wgsl. GN_MODE neither reads nor updates any of it.
@group(0) @binding(7) var<storage, read_write> errEma : array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> errEmaSum : array<atomic<u32>, 2>;

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : GradParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> grads : array<atomic<u32>>;
// The mask, f32 per pixel (row-major W*H): 0 skips the pixel.
@group(0) @binding(6) var<storage, read> mask : array<f32>;

fn atomicAddGrad(i : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&grads[i]);
  loop {
    let res = atomicCompareExchangeWeak(&grads[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

// Tile hash slot count. shader_loader.js rewrites this line with the value from
// tileSlotsForLimit(): 256 needs ~21 KB of workgroup storage, 128 (~11 KB) fits
// the 16 KB default. Tiles touching more sites than the table holds fall back
// to global adds.
const TILE_SLOTS : u32 = 128u;
const K_TILE_HASH_SIZE : u32 = TILE_SLOTS;
const K_TILE_HASH_MASK : u32 = K_TILE_HASH_SIZE - 1u;
const K_TILE_GRAD_SLOTS : u32 = K_TILE_HASH_SIZE * 10u;
const K_TILE_MAX_PROBES : u32 = 8u;
const K_TILE_EMPTY_KEY : u32 = 0xffffffffu;

// Per-tile Σb in fixed point: b ≤ 4 (clamped), so 256 · 4e6 fits a u32.
const K_EMA_QUANT : f32 = 1.0e6;
var<workgroup> wg_bias : atomic<u32>;

// The pixel's weight w for error e (see errEma), updating its EWMA.
fn errWeight(p : u32, e : vec3<f32>) -> f32 {
  var eb = mix(e, errEma[p].xyz, params.emaKeep);
  if (is_bad3(eb)) { eb = vec3<f32>(0.0); }
  var bias = dot(eb, eb);
  if (LOSS == 1u) { bias = sqrt(bias); }
  bias = min(bias, 4.0);
  let mean = bitcast<f32>(atomicLoad(&errEmaSum[params.emaSlot])) * params.emaInvCount;
  var w = 1.0;
  if (params.emaAlpha > 0.0 && mean > 1.0e-12) {
    w = (1.0 + params.emaAlpha * min(bias / mean, params.emaCap)) / (1.0 + params.emaAlpha);
  }
  errEma[p] = vec4<f32>(eb, w);
  atomicAdd(&wg_bias, u32(bias * K_EMA_QUANT));
  return w;
}

fn tile_hash(key : u32) -> u32 { return key * 2654435761u; }

var<workgroup> tg_keys : array<atomic<u32>, K_TILE_HASH_SIZE>;
// Ten grad channels laid out interleaved: tg_grads[slot * 10 + channel].
// True sum = tg_grads_hi * 2^32 + tg_grads (signed), in units of 1/K_GRAD_QUANT_SCALE.
var<workgroup> tg_grads : array<atomic<i32>, K_TILE_GRAD_SLOTS>;
var<workgroup> tg_grads_hi : array<atomic<i32>, K_TILE_GRAD_SLOTS>;

fn wgAddGrad(slot : u32, ch : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  let i = slot * 10u + ch;
  // f32 -> i32 saturates, so a single pixel is capped at ~2147; sums can't wrap.
  let q = i32(val * K_GRAD_QUANT_SCALE);
  let old = atomicAdd(&tg_grads[i], q);
  if (q > 0 && old > 2147483647 - q) {
    atomicAdd(&tg_grads_hi[i], 1);
  } else if (q < 0 && old < -2147483647 - 1 - q) {
    atomicAdd(&tg_grads_hi[i], -1);
  }
}

fn wgLoadGrad(i : u32) -> f32 {
  let lo = f32(atomicLoad(&tg_grads[i]));
  let hi = f32(atomicLoad(&tg_grads_hi[i]));
  return (hi * 4294967296.0 + lo) * K_GRAD_QUANT_SCALE_INV;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) tid : vec3<u32>) {
  let local_idx = tid.y * 16u + tid.x;
  let local_threads = 256u;

  // Init workgroup memory — uniform loop over local_idx.
  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    atomicStore(&tg_keys[i], K_TILE_EMPTY_KEY);
  }
  for (var i = local_idx; i < K_TILE_GRAD_SLOTS; i = i + local_threads) {
    atomicStore(&tg_grads[i], 0);
    atomicStore(&tg_grads_hi[i], 0);
  }
  if (local_idx == 0u) { atomicStore(&wg_bias, 0u); }
  workgroupBarrier();

  // Per-pixel gradient contribution — every thread takes the same path to the
  // next barrier. All guards are plain if-blocks; no early returns inside.
  let dims = textureDimensions(targetTex);
  let in_bounds = gid.x < dims.x && gid.y < dims.y;
  var mask_val : f32 = 0.0;
  if (in_bounds) {
    mask_val = mask[gid.y * dims.x + gid.x];
  }
  let should_work = in_bounds && mask_val > 0.0;

  if (should_work) {
    let uv = vec2<f32>(f32(gid.x), f32(gid.y));
    let inv_scale = sqrt(params.invScaleSq);
    let candSize = textureDimensions(inCand0);
    let candCoord = candidateCoord(vec2<u32>(gid.xy), dims, candSize);
    let c0 = textureLoad(inCand0, vec2<i32>(candCoord), 0);
    let c1 = textureLoad(inCand1, vec2<i32>(candCoord), 0);
    let candIds = cand_ids(c0, c1, params.globalCand);

    var logits : array<f32, NCAND>;
    for (var i = 0u; i < NCAND; i = i + 1u) {
      let idx = candIds[i];
      logits[i] = NEG_INF;
      if (idx < params.siteCount) { logits[i] = site_logit(sites[idx], uv, params.invScaleSq, inv_scale); }
    }
    let weights = softmax_c(logits);
    var pred = vec3<f32>(0.0, 0.0, 0.0);
    var sum_w = 0.0;
    for (var i = 0u; i < NCAND; i = i + 1u) {
      sum_w = sum_w + weights[i];
      if (weights[i] > 0.0 && !is_bad3(site_color(sites[candIds[i]]))) {
        pred = pred + weights[i] * site_color(sites[candIds[i]]);
      }
    }
    if (sum_w > 0.0) {
      let tgt = textureLoad(targetTex, vec2<i32>(gid.xy), 0).rgb;
      var dL_dpred = 2.0 * (pred - tgt);
      if (LOSS == 1u) { dL_dpred = sign(pred - tgt); } // sign(0) = 0
      if (!GN_MODE) { dL_dpred = dL_dpred * errWeight(gid.y * dims.x + gid.x, pred - tgt); }
      let dL_ok = !is_bad3(dL_dpred);
      if (dL_ok) {
        for (var i = 0u; i < NCAND; i = i + 1u) {
          let idx = candIds[i];
          if (idx < params.siteCount) {
            let site = sites[idx];
            if (site.position.x >= 0.0) {
              let w = weights[i];
              let dL_dcolor = w * dL_dpred;
              let site_col = site_color(site);
              let dL_dlogit = dot(dL_dpred, w * (site_col - pred));

              let tau = exp(site.log_tau);
              if (!is_bad(tau)) {
                let gn_k = dot(w * (site_col - pred), w * (site_col - pred));
                let diff = uv - site.position;
                let diff2 = dot(diff, diff);
                let dir = site_aniso_dir(site);
                let proj = dot(dir, diff);
                let proj2 = proj * proj;
                let perp2 = max(diff2 - proj2, 0.0);
                let l1 = exp(site.log_aniso);
                let l2 = 1.0 / l1;

                let d2_norm = (l1 * proj2 + l2 * perp2) * params.invScaleSq;
                let d2_safe = max(d2_norm, 1.0e-8);
                let inv_sqrt_d2 = inverseSqrt(d2_safe);
                let r_norm = site.radius_sq * inv_scale;
                let dmix2 = (sqrt(d2_safe) - r_norm);
                if (!is_bad(dmix2)) {
                  // ∂logit/∂θ for the geometric parameters.
                  let g_diff = l2 * diff + (l1 - l2) * dir * proj;
                  let dz_dpos = (tau * params.invScaleSq * inv_sqrt_d2) * g_diff;
                  let dz_dlog_tau = (-dmix2) * tau;
                  let dz_dradius_sq = tau * inv_scale;
                  let d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * params.invScaleSq;
                  let dz_dlog_aniso = (-tau) * (0.5 * inv_sqrt_d2) * d2_dlog_aniso;
                  let d2_ddir = (2.0 * (l1 - l2) * proj * params.invScaleSq) * diff;
                  let dz_ddir = (-tau) * (0.5 * inv_sqrt_d2) * d2_ddir;

                  // Gradient: dL/dlogit · ∂logit/∂θ. GN: |∂pred/∂logit|² · (∂logit/∂θ)²,
                  // with ∂pred/∂logit = w (c − pred); per color channel, (∂pred/∂c)² = w².
                  var dL_dpos = dL_dlogit * dz_dpos;
                  var dL_dlog_tau = dL_dlogit * dz_dlog_tau;
                  var dL_dradius_sq = dL_dlogit * dz_dradius_sq;
                  var dL_dlog_aniso = dL_dlogit * dz_dlog_aniso;
                  var dL_ddir = dL_dlogit * dz_ddir;
                  var dL_dcolor_out = dL_dcolor;
                  if (GN_MODE) {
                    dL_dpos = gn_k * dz_dpos * dz_dpos;
                    dL_dlog_tau = gn_k * dz_dlog_tau * dz_dlog_tau;
                    dL_dradius_sq = gn_k * dz_dradius_sq * dz_dradius_sq;
                    dL_dlog_aniso = gn_k * dz_dlog_aniso * dz_dlog_aniso;
                    dL_ddir = gn_k * dz_ddir * dz_ddir;
                    dL_dcolor_out = vec3<f32>(w * w);
                  }

                  var inserted = false;
                  let base = tile_hash(idx) & K_TILE_HASH_MASK;
                  for (var probe = 0u; probe < K_TILE_MAX_PROBES && !GN_MODE; probe = probe + 1u) {
                    let slot = (base + probe) & K_TILE_HASH_MASK;
                    var expected = K_TILE_EMPTY_KEY;
                    let res = atomicCompareExchangeWeak(&tg_keys[slot], expected, idx);
                    if (res.exchanged || res.old_value == idx) {
                      wgAddGrad(slot, 0u, dL_dpos.x);
                      wgAddGrad(slot, 1u, dL_dpos.y);
                      wgAddGrad(slot, 2u, dL_dlog_tau);
                      wgAddGrad(slot, 3u, dL_dradius_sq);
                      wgAddGrad(slot, 4u, dL_dcolor_out.x);
                      wgAddGrad(slot, 5u, dL_dcolor_out.y);
                      wgAddGrad(slot, 6u, dL_dcolor_out.z);
                      wgAddGrad(slot, 7u, dL_ddir.x);
                      wgAddGrad(slot, 8u, dL_ddir.y);
                      wgAddGrad(slot, 9u, dL_dlog_aniso);
                      inserted = true; break;
                    }
                  }
                  if (!inserted) {
                    atomicAddGrad(idx * 10u + 0u, dL_dpos.x);
                    atomicAddGrad(idx * 10u + 1u, dL_dpos.y);
                    atomicAddGrad(idx * 10u + 2u, dL_dlog_tau);
                    atomicAddGrad(idx * 10u + 3u, dL_dradius_sq);
                    atomicAddGrad(idx * 10u + 4u, dL_dcolor_out.x);
                    atomicAddGrad(idx * 10u + 5u, dL_dcolor_out.y);
                    atomicAddGrad(idx * 10u + 6u, dL_dcolor_out.z);
                    atomicAddGrad(idx * 10u + 7u, dL_ddir.x);
                    atomicAddGrad(idx * 10u + 8u, dL_ddir.y);
                    atomicAddGrad(idx * 10u + 9u, dL_dlog_aniso);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  workgroupBarrier();

  // Flush workgroup-local accumulators to global. Uniform loop over slot.
  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    let siteID = atomicLoad(&tg_keys[i]);
    if (siteID != K_TILE_EMPTY_KEY) {
      for (var ch = 0u; ch < 10u; ch = ch + 1u) {
        atomicAddGrad(siteID * 10u + ch, wgLoadGrad(i * 10u + ch));
      }
    }
  }

  // The tile's Σb into this step's slot (the one emaSlot doesn't read).
  if (local_idx == 0u && !GN_MODE) {
    let val = f32(atomicLoad(&wg_bias)) / K_EMA_QUANT;
    if (val > 0.0) {
      let slot = 1u - params.emaSlot;
      var old = atomicLoad(&errEmaSum[slot]);
      loop {
        let res = atomicCompareExchangeWeak(&errEmaSum[slot], old, bitcast<u32>(bitcast<f32>(old) + val));
        if (res.exchanged) { break; }
        old = res.old_value;
      }
    }
  }
}
