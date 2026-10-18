// Per-site statistics for site growth (trainer.js), run at growth steps only.
// Per pixel (masked): softmax-blend the 8 candidates exactly as the render
// does, then
//   - credit the pixel to its owner (largest weight) with moments of the fit
//     target over the owner's pixels, in offsets d = pixel - owner position
//     (so sums stay small): channels 0..16;
//   - credit every candidate with the error increase if it were removed and
//     the others renormalized (the "removal delta"): channel 17.
// Output `stats[site * 18 + ch]` (f32 bits, summed with a CAS loop):
//   0 n   1,2 Σd   3,4,5 Σdx², Σdx·dy, Σdy²   6..8 Σt   9..11 Σt·dx
//   12..14 Σt·dy   15 Σ|t|²   16 Σ|t - pred|²   17 Σ removal delta
// Sums go through a per-workgroup hash table first (as in gradients_tiled),
// so one global add per site per tile. Same uniformity rule: no early return
// before either barrier. The buffer must be zeroed before each run.

struct SiteStatsParams {
  siteCount : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  invScaleSq : f32,
  _pad3 : f32,
  _pad4 : f32,
  _pad5 : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : SiteStatsParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var fitTex : texture_2d<f32>;
@group(0) @binding(5) var maskTex : texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> stats : array<atomic<u32>>;

const STATS_CH : u32 = 18u;
const ST_SLOTS : u32 = 128u;
const ST_MASK : u32 = ST_SLOTS - 1u;
const ST_PROBES : u32 = 8u;
const ST_EMPTY : u32 = 0xffffffffu;

var<workgroup> st_keys : array<atomic<u32>, ST_SLOTS>;
var<workgroup> st_vals : array<atomic<u32>, 2304>; // ST_SLOTS * STATS_CH

fn globalAdd(i : u32, v : f32) {
  if (v == 0.0 || is_bad(v)) { return; }
  var old = atomicLoad(&stats[i]);
  loop {
    let res = atomicCompareExchangeWeak(&stats[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn localAdd(i : u32, v : f32) {
  if (v == 0.0 || is_bad(v)) { return; }
  var old = atomicLoad(&st_vals[i]);
  loop {
    let res = atomicCompareExchangeWeak(&st_vals[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

// Slot for `site` in this tile's table, or ST_EMPTY if the table is full.
fn slotFor(site : u32) -> u32 {
  let base = (site * 2654435761u) & ST_MASK;
  for (var probe = 0u; probe < ST_PROBES; probe = probe + 1u) {
    let slot = (base + probe) & ST_MASK;
    let res = atomicCompareExchangeWeak(&st_keys[slot], ST_EMPTY, site);
    if (res.exchanged || res.old_value == site) { return slot; }
  }
  return ST_EMPTY;
}

fn addStat(slot : u32, site : u32, ch : u32, v : f32) {
  if (slot != ST_EMPTY) { localAdd(slot * STATS_CH + ch, v); }
  else { globalAdd(site * STATS_CH + ch, v); }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) tid : vec3<u32>) {
  let local_idx = tid.y * 16u + tid.x;
  for (var i = local_idx; i < ST_SLOTS; i = i + 256u) { atomicStore(&st_keys[i], ST_EMPTY); }
  for (var i = local_idx; i < ST_SLOTS * STATS_CH; i = i + 256u) { atomicStore(&st_vals[i], 0u); }
  workgroupBarrier();

  let dims = textureDimensions(fitTex);
  let in_bounds = gid.x < dims.x && gid.y < dims.y;
  var m = 0.0;
  if (in_bounds) { m = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r; }

  if (in_bounds && m > 0.0) {
    let uv = vec2<f32>(f32(gid.x), f32(gid.y));
    let inv_scale = sqrt(params.invScaleSq);
    let cc = candidateCoord(vec2<u32>(gid.xy), dims, textureDimensions(inCand0));
    let c0 = textureLoad(inCand0, vec2<i32>(cc), 0);
    let c1 = textureLoad(inCand1, vec2<i32>(cc), 0);
    let ids = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

    var logits : array<f32, 8>;
    var cols : array<vec3<f32>, 8>;
    var maxl = -1e30;
    for (var i = 0u; i < 8u; i = i + 1u) {
      var lv = -1e30;
      cols[i] = vec3<f32>(0.0);
      let idx = ids[i];
      if (idx < params.siteCount) {
        let s = sites[idx];
        if (s.position.x >= 0.0 && !is_bad(s.log_tau)) {
          let l = -exp(s.log_tau) * voronoi_dmix2_fast(s, uv, params.invScaleSq, inv_scale);
          if (!is_bad(l)) { lv = l; cols[i] = site_color(s); }
        }
      }
      logits[i] = lv;
      maxl = max(maxl, lv);
    }
    if (maxl > -1e29) {
      var w : array<f32, 8>;
      var sw = 0.0;
      for (var i = 0u; i < 8u; i = i + 1u) {
        w[i] = select(0.0, exp(logits[i] - maxl), logits[i] > -1e29);
        sw = sw + w[i];
      }
      var pred = vec3<f32>(0.0);
      var owner = 0u;
      for (var i = 0u; i < 8u; i = i + 1u) {
        w[i] = w[i] / sw;
        pred = pred + w[i] * cols[i];
        if (w[i] > w[owner]) { owner = i; }
      }
      let t = textureLoad(fitTex, vec2<i32>(gid.xy), 0).rgb;
      let et = t - pred;
      let err = dot(et, et);

      // Owner moments.
      let oid = ids[owner];
      let d = uv - sites[oid].position;
      let os = slotFor(oid);
      addStat(os, oid, 0u, m);
      addStat(os, oid, 1u, m * d.x);
      addStat(os, oid, 2u, m * d.y);
      addStat(os, oid, 3u, m * d.x * d.x);
      addStat(os, oid, 4u, m * d.x * d.y);
      addStat(os, oid, 5u, m * d.y * d.y);
      for (var c = 0u; c < 3u; c = c + 1u) {
        addStat(os, oid, 6u + c, m * t[c]);
        addStat(os, oid, 9u + c, m * t[c] * d.x);
        addStat(os, oid, 12u + c, m * t[c] * d.y);
      }
      addStat(os, oid, 15u, m * dot(t, t));
      addStat(os, oid, 16u, m * err);

      // Removal deltas: blend of the others, renormalized.
      for (var k = 0u; k < 8u; k = k + 1u) {
        if (w[k] > 1.0e-6) {
          var rest = vec3<f32>(0.0);
          var rw = 0.0;
          var next = 8u; // best other candidate, for when the others' weights underflow
          for (var j = 0u; j < 8u; j = j + 1u) {
            if (j != k) {
              rest = rest + w[j] * cols[j];
              rw = rw + w[j];
              if (logits[j] > -1e29 && (next == 8u || logits[j] > logits[next])) { next = j; }
            }
          }
          var delta = 0.0;
          if (rw > 1.0e-12) {
            let e2 = t - rest / rw;
            delta = dot(e2, e2) - err;
          } else if (next < 8u) {
            let e2 = t - cols[next];
            delta = dot(e2, e2) - err;
          }
          addStat(slotFor(ids[k]), ids[k], 17u, m * delta);
        }
      }
    }
  }

  workgroupBarrier();
  for (var i = local_idx; i < ST_SLOTS; i = i + 256u) {
    let site = atomicLoad(&st_keys[i]);
    if (site != ST_EMPTY && site < params.siteCount) {
      for (var c = 0u; c < STATS_CH; c = c + 1u) {
        globalAdd(site * STATS_CH + c, bitcast<f32>(atomicLoad(&st_vals[i * STATS_CH + c])));
      }
    }
  }
}
