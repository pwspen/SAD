// Rate term: adds λ · ∂bits/∂param to each site's accumulated gradients (run
// after gradients_tiled, before adam), so training minimizes squared error +
// λ · coded bits. The bits are trainer/rate.js's model: per coded plane, the
// probability of a value's quantizer bin under a Gaussian mixture refit on
// the CPU, P(v) ≈ Σ_k exp(norm_k − (v − μ_k)² / (2 s_k²)) (norm_k folds in the
// weight, the 1/√(2π s²) and the bin width; s² includes the bin's Δ²/12),
// bits = −log₂ min(1, P). Colors are coded along the KLT axes in the header,
// so their gradient goes back to RGB (the grads buffer's color channels are
// RGB; adam maps them to the color space). θ costs log₂ of its level count,
// π · |2 sinh a| / Δθ for log_aniso a, when that is above 1.
//
// rate[] layout (f32; u32 fields bitcast): see RATE_* in trainer/rate.js.

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> grads : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> rate : array<f32>;

const RATE_HEADER : u32 = 20u;
const RATE_KMAX : u32 = 8u;
const RATE_PLANE : u32 = 2u + 3u * RATE_KMAX; // Δ, K, then K × (norm, μ, s²)
const INV_LN2 : f32 = 1.44269504;

fn addGrad(i : u32, g : f32) {
  if (is_bad(g) || g == 0.0) { return; }
  atomicStore(&grads[i], bitcast<u32>(bitcast<f32>(atomicLoad(&grads[i])) + g));
}

// d bits / dv for plane p at value v (0 where the bin is certain, P ≥ 1).
fn planeGrad(p : u32, v : f32) -> f32 {
  let b = RATE_HEADER + p * RATE_PLANE;
  let k = u32(rate[b + 1u]);
  if (k == 0u) { return 0.0; }
  var top = -1.0e30;
  for (var j = 0u; j < k; j = j + 1u) {
    let c = b + 2u + 3u * j;
    let d = v - rate[c + 1u];
    top = max(top, rate[c] - d * d / (2.0 * rate[c + 2u]));
  }
  var sum = 0.0;
  var num = 0.0;
  for (var j = 0u; j < k; j = j + 1u) {
    let c = b + 2u + 3u * j;
    let d = v - rate[c + 1u];
    let e = exp(rate[c] - d * d / (2.0 * rate[c + 2u]) - top);
    sum = sum + e;
    num = num + e * d / rate[c + 2u];
  }
  if (top + log(sum) >= 0.0) { return 0.0; }
  return INV_LN2 * num / sum;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let count = bitcast<u32>(rate[1]);
  if (idx >= count || idx >= arrayLength(&sites)) { return; }
  let site = sites[idx];
  if (site.position.x < 0.0) { return; }
  let perTile = bitcast<u32>(rate[3]);
  let slot = select(idx, idx % perTile, perTile > 0u);
  if (slot == bitcast<u32>(rate[2])) { return; } // the histogram layer isn't coded like a site
  let lambda = rate[0];
  let g = idx * 10u;

  addGrad(g + 2u, lambda * planeGrad(0u, site.log_tau));
  addGrad(g + 3u, lambda * planeGrad(1u, site.radius_sq));

  let rgb = site_color(site);
  if (!is_bad3(rgb)) {
    let d = rgb - vec3<f32>(rate[6], rate[7], rate[8]);
    var g_rgb = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k = k + 1u) {
      let axis = vec3<f32>(rate[9u + 3u * k], rate[10u + 3u * k], rate[11u + 3u * k]);
      g_rgb = g_rgb + planeGrad(2u + k, dot(axis, d)) * axis;
    }
    addGrad(g + 4u, lambda * g_rgb.x);
    addGrad(g + 5u, lambda * g_rgb.y);
    addGrad(g + 6u, lambda * g_rgb.z);
  }

  var g_aniso = planeGrad(5u, site.log_aniso);
  let thetaStep = rate[4];
  let a = site.log_aniso;
  if (thetaStep > 0.0 && 3.14159265 * abs(2.0 * sinh(a)) > thetaStep && abs(a) > 1.0e-4) {
    g_aniso = g_aniso + INV_LN2 / tanh(a); // d/da log₂ |2 sinh a|
  }
  addGrad(g + 9u, lambda * g_aniso);
}
