// Per-site statistics for site growth (trainer.js), run at growth steps only.
// Per pixel (masked): softmax-blend the 8 candidates exactly as the render
// does, then
//   - credit the pixel to its owner (largest weight) with moments of the fit
//     target over the owner's pixels, in offsets d = pixel - owner position
//     (so sums stay small): channels 0..16. A pinned owner (sitePin, e.g. the
//     histogram layer) can't split, so its pixels go to the best unpinned
//     candidate instead, weighted by how far the render is from exact there
//     (info_update's u: 0 within 1/4 bin, 1 from 1/2), so only what the layer
//     doesn't explain counts;
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
  globalCand : u32,
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
@group(0) @binding(5) var<storage, read> lossWeight : array<f32>; // mask or info weights (W*H)
@group(0) @binding(6) var<storage, read_write> stats : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> sitePin : array<u32>;

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
  if (in_bounds) { m = lossWeight[gid.y * dims.x + gid.x]; }

  if (in_bounds && m > 0.0) {
    let uv = vec2<f32>(f32(gid.x), f32(gid.y));
    let inv_scale = sqrt(params.invScaleSq);
    let cc = candidateCoord(vec2<u32>(gid.xy), dims, textureDimensions(inCand0));
    let c0 = textureLoad(inCand0, vec2<i32>(cc), 0);
    let c1 = textureLoad(inCand1, vec2<i32>(cc), 0);
    let ids = cand_ids(c0, c1, params.globalCand);

    var logits : array<f32, NCAND>;
    var cols : array<vec3<f32>, NCAND>;
    for (var i = 0u; i < NCAND; i = i + 1u) {
      logits[i] = NEG_INF;
      cols[i] = vec3<f32>(0.0);
      if (ids[i] < params.siteCount) {
        logits[i] = site_logit(sites[ids[i]], uv, params.invScaleSq, inv_scale);
        if (logits[i] > -1e29) { cols[i] = site_color(sites[ids[i]]); }
      }
    }
    let w = softmax_c(logits);
    if (w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7] + w[8] > 0.0) {
      var pred = vec3<f32>(0.0);
      var owner = 0u;
      for (var i = 0u; i < NCAND; i = i + 1u) {
        pred = pred + w[i] * cols[i];
        if (w[i] > w[owner]) { owner = i; }
      }
      let t = textureLoad(fitTex, vec2<i32>(gid.xy), 0).rgb;
      let et = t - pred;
      let err = dot(et, et);

      // Owner moments (a pinned owner hands the pixel to the best unpinned
      // candidate, weighted by the render's error in bins beyond half a bin).
      var oid = ids[owner];
      if (sitePin[oid] != 0u) {
        var alt = NCAND;
        for (var i = 0u; i < NCAND; i = i + 1u) {
          if (w[i] > 0.0 && sitePin[ids[i]] == 0u && (alt == NCAND || w[i] > w[alt])) { alt = i; }
        }
        let ed = abs(clamp(pred, vec3<f32>(0.0), vec3<f32>(1.0)) - t) * 255.0;
        m = m * clamp((max(ed.x, max(ed.y, ed.z)) - 0.25) * 4.0, 0.0, 1.0);
        if (alt == NCAND) { m = 0.0; } else { oid = ids[alt]; }
      }
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
      for (var k = 0u; k < NCAND; k = k + 1u) {
        if (w[k] > 1.0e-6) {
          var rest = vec3<f32>(0.0);
          var rw = 0.0;
          var next = NCAND; // best other candidate, for when the others' weights underflow
          for (var j = 0u; j < NCAND; j = j + 1u) {
            if (j != k) {
              rest = rest + w[j] * cols[j];
              rw = rw + w[j];
              if (logits[j] > -1e29 && (next == NCAND || logits[j] > logits[next])) { next = j; }
            }
          }
          var delta = 0.0;
          if (rw > 1.0e-12) {
            let e2 = t - rest / rw;
            delta = dot(e2, e2) - err;
          } else if (next < NCAND) {
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
