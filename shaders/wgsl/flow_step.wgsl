// Residual-driven site flow, applied after Adam. Two terms, summed and capped at
// `maxStep` pixels:
//
// Lloyd (`strength`): move a fraction of the way to the cell's error-weighted
// centroid (moments from lloyd_accum.wgsl). `geoMix` subtracts that much of the
// geometric centroid, so at 1 a cell with uniform error doesn't move its site.
//
// Pyramid (`pyrStrength`): a multi-scale Poisson-like pull toward regions with
// more error than sites. Per cell, the excess q = err * N / E - count (in sites)
// sums to zero over the image, so the flow settles when error per site is equal
// everywhere. Each level contributes a Gaussian-gradient kernel with sigma = its
// cell size over a 5x5 window; summed over levels this approximates the 2D 1/r
// force. Scaled by h^2 (mean area per site), a one-site surplus at one spacing
// pulls ~pyrStrength * h px. The site's own count is removed from its cell.
// `pyrLevels` > 0 limits the force to that many finest levels (local moves only).
//
// Clears the Lloyd moments after use.

@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> flow : array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params : FlowParams;
@group(0) @binding(3) var<storage, read> cells : array<vec2<f32>>;
@group(0) @binding(4) var<uniform> pyr : PyrParams;

fn loadFlow(i : u32) -> f32 { return bitcast<f32>(atomicLoad(&flow[i])); }

fn pyrForce(p : vec2<f32>) -> vec2<f32> {
  let tot = cells[pyr.levels[pyr.numLevels - 1u].z];
  let E = tot.x;
  let N = tot.y;
  if (!(E > 1.0e-20) || N < 1.0 || is_bad(E)) { return vec2<f32>(0.0); }
  let k = N / E;
  let h2 = f32(pyr.width * pyr.height) / N;
  var F = vec2<f32>(0.0);
  var nl = pyr.numLevels;
  if (params.pyrLevels > 0.0) { nl = min(nl, u32(params.pyrLevels)); }
  for (var l = 0u; l < nl; l = l + 1u) {
    let lv = pyr.levels[l];
    let cs = f32(lv.w);
    let inv2 = 1.0 / (cs * cs);
    let ci = vec2<i32>(min(vec2<u32>(p / cs), vec2<u32>(lv.x - 1u, lv.y - 1u)));
    for (var dy = -2; dy <= 2; dy = dy + 1) {
      for (var dx = -2; dx <= 2; dx = dx + 1) {
        let c = ci + vec2<i32>(dx, dy);
        if (c.x < 0 || c.y < 0 || c.x >= i32(lv.x) || c.y >= i32(lv.y)) { continue; }
        let v = cells[lv.z + u32(c.y) * lv.x + u32(c.x)];
        var q = v.x * k - v.y;
        if (dx == 0 && dy == 0) { q = q + 1.0; }
        let d = (vec2<f32>(c) + vec2<f32>(0.5)) * cs - p;
        F = F + q * d * inv2 * exp(-0.5 * dot(d, d) * inv2);
      }
    }
  }
  return F * h2;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.siteCount) { return; }
  let base = idx * 6u;
  let m = vec2<f32>(loadFlow(base + 0u), loadFlow(base + 1u));
  let mass = loadFlow(base + 2u);
  let g = vec2<f32>(loadFlow(base + 3u), loadFlow(base + 4u));
  let count = loadFlow(base + 5u);
  for (var c = 0u; c < 6u; c = c + 1u) { atomicStore(&flow[base + c], 0u); }

  let pos = sites[idx].position;
  if (pos.x < 0.0 || is_bad2(pos)) { return; }

  var step = vec2<f32>(0.0);
  if (params.strength > 0.0 && !is_bad2(m) && !is_bad(mass) && mass > 1.0e-12 &&
      !is_bad2(g) && count > 0.0) {
    step = step + params.strength * (m / mass - params.geoMix * (g / count));
  }
  if (params.pyrStrength > 0.0) {
    let f = pyrForce(pos);
    if (!is_bad2(f)) { step = step + params.pyrStrength * f; }
  }
  let len = length(step);
  if (!(len > 0.0) || is_bad(len)) { return; }
  if (len > params.maxStep) { step = step * (params.maxStep / len); }
  sites[idx].position = clamp(pos + step,
    vec2<f32>(0.0), vec2<f32>(f32(params.width - 1u), f32(params.height - 1u)));
}
