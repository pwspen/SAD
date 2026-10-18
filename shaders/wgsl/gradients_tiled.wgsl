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
// Gauss–Newton diagonal of the squared-error loss: per site and parameter,
// Σ_pixels Σ_rgb (∂pred/∂θ)², with color taken in decoded RGB. That is the
// predicted rise in summed squared error per unit² change of θ, used by the
// viewer's site-clustering analysis; training never runs it. GN sums skip the
// fixed-point tile path (squared terms can exceed its per-pixel range) and
// add straight to `grads` as f32.

override GN_MODE : bool = false;

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : GradParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> grads : array<atomic<u32>>;
@group(0) @binding(6) var maskTex : texture_2d<f32>;

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
  workgroupBarrier();

  // Per-pixel gradient contribution — every thread takes the same path to the
  // next barrier. All guards are plain if-blocks; no early returns inside.
  let dims = textureDimensions(targetTex);
  let in_bounds = gid.x < dims.x && gid.y < dims.y;
  var mask_val : f32 = 0.0;
  if (in_bounds) {
    mask_val = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r;
  }
  let should_work = in_bounds && mask_val > 0.0;

  if (should_work) {
    let uv = vec2<f32>(f32(gid.x), f32(gid.y));
    let inv_scale = sqrt(params.invScaleSq);
    let candSize = textureDimensions(inCand0);
    let candCoord = candidateCoord(vec2<u32>(gid.xy), dims, candSize);
    let c0 = textureLoad(inCand0, vec2<i32>(candCoord), 0);
    let c1 = textureLoad(inCand1, vec2<i32>(candCoord), 0);
    let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

    let neg_inf = -1e30;
    var logits : array<f32, 8>;
    var max_logit : f32 = neg_inf;
    for (var i = 0u; i < 8u; i = i + 1u) {
      let idx = candIds[i];
      var lv : f32 = neg_inf;
      if (idx < params.siteCount) {
        let site = sites[idx];
        if (site.position.x >= 0.0 && !is_bad(site.position.x) && !is_bad(site.position.y) &&
            !is_bad(site.log_tau) && !is_bad(site.log_aniso)) {
          let tau = exp(site.log_tau);
          if (!is_bad(tau)) {
            let dmix2 = voronoi_dmix2_fast(site, uv, params.invScaleSq, inv_scale);
            if (!is_bad(dmix2)) {
              let l = -tau * dmix2;
              if (!is_bad(l)) { lv = l; }
            }
          }
        }
      }
      logits[i] = lv;
      max_logit = max(max_logit, lv);
    }

    let have_max = (max_logit != neg_inf) && !is_bad(max_logit);
    if (have_max) {
      var weights : array<f32, 8>;
      var sum_w : f32 = 0.0;
      for (var i = 0u; i < 8u; i = i + 1u) {
        var w = exp(logits[i] - max_logit);
        if (is_bad(w)) { w = 0.0; }
        weights[i] = w;
        sum_w = sum_w + w;
      }
      let sum_ok = !is_bad(sum_w) && sum_w > 0.0;
      if (sum_ok) {
        let inv_sum = 1.0 / max(sum_w, 1e-8);
        var pred = vec3<f32>(0.0, 0.0, 0.0);
        for (var i = 0u; i < 8u; i = i + 1u) {
          weights[i] = weights[i] * inv_sum;
          let idx = candIds[i];
          if (idx < params.siteCount && sites[idx].position.x >= 0.0 && !is_bad3(site_color(sites[idx]))) {
            pred = pred + weights[i] * site_color(sites[idx]);
          }
        }
        let tgt = textureLoad(targetTex, vec2<i32>(gid.xy), 0).rgb;
        let dL_dpred = 2.0 * (pred - tgt);
        let dL_ok = !is_bad3(dL_dpred);
        if (dL_ok) {
          for (var i = 0u; i < 8u; i = i + 1u) {
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
}
