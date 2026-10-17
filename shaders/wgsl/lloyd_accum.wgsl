// Error-weighted Lloyd moments for residual-driven site flow.
//
// Per pixel: recompute the softmax prediction (same as gradients_tiled), update
// a per-pixel EMA of the squared residual, and credit that error to the pixel's
// hard (argmax) Voronoi owner. Each site accumulates six f32 sums in
// `flow[idx*6+ch]`:
//   0,1 sum e * (uv - pos), 2 sum e, 3,4 sum (uv - pos), 5 pixel count
// so flow_step.wgsl can move the site toward its cell's error-weighted centroid
// (optionally relative to the plain geometric centroid).
// Offsets are relative to the site so sums stay small and precise.
//
// Accumulation reuses gradients_tiled's workgroup hash (fixed-point with a
// carry word, flushed once per site per tile). Same uniformity rule: no early
// return before either barrier.

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : FlowParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var maskTex : texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> errEma : array<f32>;
@group(0) @binding(7) var<storage, read_write> flow : array<atomic<u32>>;

fn atomicAddFlow(i : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&flow[i]);
  loop {
    let res = atomicCompareExchangeWeak(&flow[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

// Argmax ownership touches far fewer sites per tile than the 8-way softmax, so
// a fixed 128-slot table (~6.5 KB) is plenty.
const K_FLOW_HASH_SIZE : u32 = 128u;
const K_FLOW_HASH_MASK : u32 = K_FLOW_HASH_SIZE - 1u;
const K_FLOW_SLOTS : u32 = K_FLOW_HASH_SIZE * 6u;
const K_FLOW_MAX_PROBES : u32 = 8u;
const K_FLOW_EMPTY_KEY : u32 = 0xffffffffu;

var<workgroup> tg_keys : array<atomic<u32>, K_FLOW_HASH_SIZE>;
var<workgroup> tg_sums : array<atomic<i32>, K_FLOW_SLOTS>;
var<workgroup> tg_sums_hi : array<atomic<i32>, K_FLOW_SLOTS>;

fn wgAddFlow(slot : u32, ch : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  let i = slot * 6u + ch;
  let q = i32(val * K_GRAD_QUANT_SCALE);
  let old = atomicAdd(&tg_sums[i], q);
  if (q > 0 && old > 2147483647 - q) {
    atomicAdd(&tg_sums_hi[i], 1);
  } else if (q < 0 && old < -2147483647 - 1 - q) {
    atomicAdd(&tg_sums_hi[i], -1);
  }
}

fn wgLoadFlow(i : u32) -> f32 {
  let lo = f32(atomicLoad(&tg_sums[i]));
  let hi = f32(atomicLoad(&tg_sums_hi[i]));
  return (hi * 4294967296.0 + lo) * K_GRAD_QUANT_SCALE_INV;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) tid : vec3<u32>) {
  let local_idx = tid.y * 16u + tid.x;
  let local_threads = 256u;

  for (var i = local_idx; i < K_FLOW_HASH_SIZE; i = i + local_threads) {
    atomicStore(&tg_keys[i], K_FLOW_EMPTY_KEY);
  }
  for (var i = local_idx; i < K_FLOW_SLOTS; i = i + local_threads) {
    atomicStore(&tg_sums[i], 0);
    atomicStore(&tg_sums_hi[i], 0);
  }
  workgroupBarrier();

  let dims = textureDimensions(targetTex);
  let in_bounds = gid.x < dims.x && gid.y < dims.y;
  var mask_val : f32 = 0.0;
  if (in_bounds) {
    mask_val = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r;
  }

  if (in_bounds && mask_val > 0.0) {
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
    var best_id : u32 = K_FLOW_EMPTY_KEY;
    for (var i = 0u; i < 8u; i = i + 1u) {
      let idx = candIds[i];
      var lv : f32 = neg_inf;
      if (idx < params.siteCount) {
        let site = sites[idx];
        if (site.position.x >= 0.0 && !is_bad(site.position.x) && !is_bad(site.position.y) &&
            !is_bad(site.log_tau) && !is_bad(site.log_aniso) && !is_bad3(site_color(site))) {
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
      if (lv > max_logit) {
        max_logit = lv;
        best_id = idx;
      }
    }

    if (best_id != K_FLOW_EMPTY_KEY) {
      var sum_w : f32 = 0.0;
      var pred = vec3<f32>(0.0, 0.0, 0.0);
      for (var i = 0u; i < 8u; i = i + 1u) {
        if (logits[i] > neg_inf) {
          let w = exp(logits[i] - max_logit);
          if (!is_bad(w)) {
            sum_w = sum_w + w;
            pred = pred + w * site_color(sites[candIds[i]]);
          }
        }
      }
      if (sum_w > 0.0 && !is_bad(sum_w)) {
        pred = pred / sum_w;
        let tgt = textureLoad(targetTex, vec2<i32>(gid.xy), 0).rgb;
        let r = pred - tgt;
        let e_now = dot(r, r);
        let pix = gid.y * dims.x + gid.x;
        var e = mix(errEma[pix], e_now, params.emaAlpha);
        if (is_bad(e)) { e = e_now; }
        errEma[pix] = e;

        let d = uv - sites[best_id].position;
        let ex = e * d.x;
        let ey = e * d.y;
        var inserted = false;
        let base = (best_id * 2654435761u) & K_FLOW_HASH_MASK;
        for (var probe = 0u; probe < K_FLOW_MAX_PROBES; probe = probe + 1u) {
          let slot = (base + probe) & K_FLOW_HASH_MASK;
          let res = atomicCompareExchangeWeak(&tg_keys[slot], K_FLOW_EMPTY_KEY, best_id);
          if (res.exchanged || res.old_value == best_id) {
            wgAddFlow(slot, 0u, ex);
            wgAddFlow(slot, 1u, ey);
            wgAddFlow(slot, 2u, e);
            wgAddFlow(slot, 3u, d.x);
            wgAddFlow(slot, 4u, d.y);
            wgAddFlow(slot, 5u, 1.0);
            inserted = true; break;
          }
        }
        if (!inserted) {
          atomicAddFlow(best_id * 6u + 0u, ex);
          atomicAddFlow(best_id * 6u + 1u, ey);
          atomicAddFlow(best_id * 6u + 2u, e);
          atomicAddFlow(best_id * 6u + 3u, d.x);
          atomicAddFlow(best_id * 6u + 4u, d.y);
          atomicAddFlow(best_id * 6u + 5u, 1.0);
        }
      }
    }
  }

  workgroupBarrier();

  for (var i = local_idx; i < K_FLOW_HASH_SIZE; i = i + local_threads) {
    let siteID = atomicLoad(&tg_keys[i]);
    if (siteID != K_FLOW_EMPTY_KEY) {
      for (var ch = 0u; ch < 6u; ch = ch + 1u) {
        atomicAddFlow(siteID * 6u + ch, wgLoadFlow(i * 6u + ch));
      }
    }
  }
}
