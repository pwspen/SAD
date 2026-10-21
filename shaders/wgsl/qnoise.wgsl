// Quantization noise (args.quantNoise): perturbs each coded site by uniform
// noise U(−Δ/2, Δ/2) per coded plane, the additive-noise stand-in for the
// codec's quantizer (codec.js), so the gradient pass sees sites as a decoder
// would. The trainer copies the clean sites aside before this kernel and back
// after the gradient pass, so Adam updates the clean values.
//
// Planes and steps as the codec quantizes them: position on its grid step
// (both axes; clamped to the site's tile like adam.wgsl), log τ, radius,
// log aniso on theirs, the color along the codec's basis axes (RGB, then back
// to the color space's params), and θ by the site's own level step π / m,
// m = round(π / Δθ · |2 sinh a|) clamped to [1, 4096] (codec.js
// thetaLevelsOf, at the perturbed a): a rotation of the direction.
//
// qn[] layout (f32; u32 fields bitcast):
//   0 siteCount, 1 layerSlot (0xffffffff = none), 2 sitesPerTile (0 = untiled),
//   3 seed, 4 tileW (image width untiled), 5 tileH, 6 tilesX, 7 θ noise scale,
//   8 Δpos, 9 Δlog_tau, 10 Δradius, 11 Δcolor, 12 Δlog_aniso (all scaled),
//   13 Δθ (unscaled; 0 = θ off; the noise is scale × π / m wide),
//   14..16 basis mean (RGB), 17..25 basis axes (3 × RGB).

@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read> qn : array<f32>;

const QN_MAX_THETA_LEVELS : f32 = 4096.0;

fn qn_hash(x : u32) -> u32 {
  // PCG output permutation
  var s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// Uniform in [−0.5, 0.5) for draw k of site idx.
fn qn_uniform(idx : u32, k : u32) -> f32 {
  let h = qn_hash(qn_hash(idx * 16u + k) ^ bitcast<u32>(qn[3]));
  return f32(h >> 8u) / 16777216.0 - 0.5;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let count = bitcast<u32>(qn[0]);
  if (idx >= count || idx >= arrayLength(&sites)) { return; }
  var site = sites[idx];
  if (site.position.x < 0.0) { return; }
  let perTile = bitcast<u32>(qn[2]);
  let slot = select(idx, idx % perTile, perTile > 0u);
  if (slot == bitcast<u32>(qn[1])) { return; } // the histogram layer isn't coded like a site

  var origin = vec2<f32>(0.0);
  if (perTile > 0u) {
    let b = idx / perTile;
    let tilesX = bitcast<u32>(qn[6]);
    origin = vec2<f32>(f32((b % tilesX) * bitcast<u32>(qn[4])), f32((b / tilesX) * bitcast<u32>(qn[5])));
  }
  let size = vec2<f32>(f32(bitcast<u32>(qn[4]) - 1u), f32(bitcast<u32>(qn[5]) - 1u));
  let dp = qn[8] * vec2<f32>(qn_uniform(idx, 0u), qn_uniform(idx, 1u));
  site.position = clamp(site.position + dp, origin, origin + size);
  site.log_tau = site.log_tau + qn[9] * qn_uniform(idx, 2u);
  site.radius_sq = site.radius_sq + qn[10] * qn_uniform(idx, 3u);

  let rgb = site_color(site);
  if (!is_bad3(rgb)) {
    var d = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k = k + 1u) {
      let axis = vec3<f32>(qn[17u + 3u * k], qn[18u + 3u * k], qn[19u + 3u * k]);
      d = d + qn[11] * qn_uniform(idx, 4u + k) * axis;
    }
    let p = encode_color(rgb + d);
    site.color_r = p.x;
    site.color_g = p.y;
    site.color_b = p.z;
  }

  site.log_aniso = site.log_aniso + qn[12] * qn_uniform(idx, 7u);
  let thetaStep = qn[13];
  if (thetaStep > 0.0) {
    let m = clamp(round(3.14159265 / thetaStep * abs(2.0 * sinh(site.log_aniso))), 1.0, QN_MAX_THETA_LEVELS);
    let phi = qn[7] * (3.14159265 / m) * qn_uniform(idx, 8u);
    let c = cos(phi);
    let s = sin(phi);
    let dir = site_aniso_dir(site);
    site.aniso_dir_x = c * dir.x - s * dir.y;
    site.aniso_dir_y = s * dir.x + c * dir.y;
  }
  sites[idx] = site;
}
