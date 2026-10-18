// Per-cell statistics for site growth (trainer.js). One thread per cell of a
// cellSize-pixel grid; the CPU turns them into the expected error reduction of
// a new site that takes over the cell with the cell's mean target color:
//   gain = sum |fit - pred|^2  -  sum |fit - mean(fit)|^2
// and places it at the cell's error centroid with its current owners' tau and
// radius (so it competes as an equal).
//
// Output `cells[c * 12 + k]`, masked sums over the cell's pixels:
//   0     mask weight
//   1..3  mean fit color
//   4     |fit - mean fit|^2 (summed relative to the centre pixel's color,
//         which keeps it well conditioned in f32)
//   5     |fit - pred|^2   (error energy E)
//   6,7   E * (pixel - cell origin)
//   8,9   E * owner log_tau, E * owner radius
//   10    E where the owner is known (weight for 8, 9); 11 unused
// where pred is the current render and the owner is the pixel's top candidate.

struct GrowGainParams {
  width : u32,
  height : u32,
  cellSize : u32,
  cellsX : u32,
  cellsY : u32,
  siteCount : u32,
  _pad0 : u32,
  _pad1 : u32,
}

@group(0) @binding(0) var<uniform> params : GrowGainParams;
@group(0) @binding(1) var<storage, read> sites : array<Site>;
@group(0) @binding(2) var predTex : texture_2d<f32>;
@group(0) @binding(3) var fitTex : texture_2d<f32>;
@group(0) @binding(4) var maskTex : texture_2d<f32>;
@group(0) @binding(5) var inCand0 : texture_2d<u32>;
@group(0) @binding(6) var<storage, read_write> cells : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = gid.x;
  if (c >= params.cellsX * params.cellsY) { return; }
  let cs = params.cellSize;
  let ox = (c % params.cellsX) * cs;
  let oy = (c / params.cellsX) * cs;
  let x1 = min(ox + cs, params.width);
  let y1 = min(oy + cs, params.height);
  let dims = vec2<u32>(params.width, params.height);
  let candSize = textureDimensions(inCand0);
  let centre = vec2<i32>(i32((ox + x1) / 2u), i32((oy + y1) / 2u));
  let refc = textureLoad(fitTex, centre, 0).rgb;

  var n = 0.0;
  var st = vec3<f32>(0.0);
  var st2 = 0.0;
  var se = 0.0;
  var sex = vec2<f32>(0.0);
  var stau = 0.0;
  var srad = 0.0;
  var sow = 0.0;
  for (var y = oy; y < y1; y = y + 1u) {
    for (var x = ox; x < x1; x = x + 1u) {
      let p = vec2<i32>(i32(x), i32(y));
      let m = textureLoad(maskTex, p, 0).r;
      if (m <= 0.0) { continue; }
      let t = textureLoad(fitTex, p, 0).rgb;
      let e = t - textureLoad(predTex, p, 0).rgb;
      let dt = t - refc;
      let ee = dot(e, e);
      if (is_bad(ee)) { continue; }
      n = n + m;
      st = st + m * dt;
      st2 = st2 + m * dot(dt, dt);
      se = se + m * ee;
      sex = sex + m * ee * vec2<f32>(f32(x - ox) + 0.5, f32(y - oy) + 0.5);
      let cc = candidateCoord(vec2<u32>(x, y), dims, candSize);
      let owner = textureLoad(inCand0, vec2<i32>(cc), 0).x;
      if (owner < params.siteCount) {
        let s = sites[owner];
        stau = stau + m * ee * s.log_tau;
        srad = srad + m * ee * s.radius_sq;
        sow = sow + m * ee;
      }
    }
  }
  let b = c * 12u;
  cells[b + 0u] = n;
  let mean = st / max(n, 1.0e-6);
  cells[b + 1u] = refc.x + mean.x;
  cells[b + 2u] = refc.y + mean.y;
  cells[b + 3u] = refc.z + mean.z;
  cells[b + 4u] = max(0.0, st2 - n * dot(mean, mean));
  cells[b + 5u] = se;
  cells[b + 6u] = sex.x;
  cells[b + 7u] = sex.y;
  cells[b + 8u] = stau;
  cells[b + 9u] = srad;
  cells[b + 10u] = sow;
  cells[b + 11u] = 0.0;
}
