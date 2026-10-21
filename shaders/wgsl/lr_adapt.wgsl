// Per-group LR adaptation statistics (trainer.js, args.lrAdapt): run after the
// gradient (and rate) passes, before adam, it accumulates for each LR group
// the agreement between this step's gradient g and the direction of the last
// Adam step, d = m / (√v + eps) from the stored moments: Σ g·d, Σ g², Σ d²
// over the active sites (the histogram layer left out). The cosine
// Σ g·d / √(Σ g² Σ d²) is the normalized hypergradient of the loss with
// respect to the group's LR: positive when the gradient still points along
// the last steps (steps too short), negative when it reverses (too long).
// Colors are compared in the color space's own params (adam's g_color);
// directions along their tangent, as adam steps them.
//
// acc: 6 groups × 3 sums as f32 bits, accumulated over iterations until the
// trainer reads and clears it. params: siteCount, layer slot, sitesPerTile.

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read> adam : array<f32>;
@group(0) @binding(2) var<storage, read> grads : array<u32>;
@group(0) @binding(3) var<storage, read_write> acc : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> lrParams : array<u32>;

const GROUPS : u32 = 6u;
const WG : u32 = 64u;
var<workgroup> partial : array<f32, 1152>; // WG × GROUPS × 3

fn g_at(i : u32) -> f32 { return bitcast<f32>(grads[i]); }

fn atomicAddF(i : u32, v : f32) {
  var old = atomicLoad(&acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let idx = gid.x;
  let base = lid.x * GROUPS * 3u;
  for (var k = 0u; k < GROUPS * 3u; k = k + 1u) { partial[base + k] = 0.0; }
  let count = lrParams[0];
  let perTile = lrParams[2];
  var ok = idx < count && idx < arrayLength(&sites);
  if (ok) {
    let slot = select(idx, idx % perTile, perTile > 0u);
    ok = slot != lrParams[1] && sites[idx].position.x >= 0.0;
  }
  if (ok) {
    let s = sites[idx];
    let gb = idx * 10u;
    let ab = idx * 24u;
    let eps = 1.0e-12;
    var gs : array<f32, 6>;
    var ds : array<f32, 6>;
    var gg : array<f32, 6>;
    var dd : array<f32, 6>;
    // position: shared second moment in slot 2
    let gp = vec2<f32>(g_at(gb), g_at(gb + 1u));
    let dp = vec2<f32>(adam[ab], adam[ab + 1u]) / (sqrt(adam[ab + 2u]) + eps);
    gs[0] = dot(gp, dp); gg[0] = dot(gp, gp); dd[0] = dot(dp, dp);
    let gt = g_at(gb + 2u);
    let dt = adam[ab + 4u] / (sqrt(adam[ab + 5u]) + eps);
    gs[1] = gt * dt; gg[1] = gt * gt; dd[1] = dt * dt;
    let gr = g_at(gb + 3u);
    let dr = adam[ab + 6u] / (sqrt(adam[ab + 7u]) + eps);
    gs[2] = gr * dr; gg[2] = gr * gr; dd[2] = dr * dr;
    let gc = color_vjp(site_color_params(s), vec3<f32>(g_at(gb + 4u), g_at(gb + 5u), g_at(gb + 6u)));
    let dc = vec3<f32>(adam[ab + 8u], adam[ab + 9u], adam[ab + 10u]) /
             (sqrt(vec3<f32>(adam[ab + 11u], adam[ab + 12u], adam[ab + 13u])) + eps);
    gs[3] = dot(gc, dc); gg[3] = dot(gc, gc); dd[3] = dot(dc, dc);
    let dir = safe_dir(site_aniso_dir(s));
    var gdir = vec2<f32>(g_at(gb + 7u), g_at(gb + 8u));
    gdir = gdir - dir * dot(gdir, dir);
    let ddir = vec2<f32>(adam[ab + 14u], adam[ab + 15u]) / (sqrt(adam[ab + 16u]) + eps);
    gs[4] = dot(gdir, ddir); gg[4] = dot(gdir, gdir); dd[4] = dot(ddir, ddir);
    let ga = g_at(gb + 9u);
    let da = adam[ab + 18u] / (sqrt(adam[ab + 19u]) + eps);
    gs[5] = ga * da; gg[5] = ga * ga; dd[5] = da * da;
    for (var k = 0u; k < GROUPS; k = k + 1u) {
      if (is_bad(gs[k]) || is_bad(gg[k]) || is_bad(dd[k])) { continue; }
      partial[base + 3u * k] = gs[k];
      partial[base + 3u * k + 1u] = gg[k];
      partial[base + 3u * k + 2u] = dd[k];
    }
  }
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) {
      let other = (lid.x + stride) * GROUPS * 3u;
      for (var k = 0u; k < GROUPS * 3u; k = k + 1u) { partial[base + k] = partial[base + k] + partial[other + k]; }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    for (var k = 0u; k < GROUPS * 3u; k = k + 1u) {
      if (partial[k] != 0.0) { atomicAddF(k, partial[k]); }
    }
  }
}
