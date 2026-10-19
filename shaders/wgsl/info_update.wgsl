// Remaining-information pass (args.info, trainer.js). Per pixel: blend the 8
// candidates as the render does, clamp to the output range, and measure the
// error in quantization bins of the target (params.quant, 1/255 for 8-bit):
// e < 1/2 bin means the decoded pixel is already exact, so it carries no
// information the sites still have to explain. The weight u ramps from 0 at
// 1/4 bin to 1 at 1/2 bin: zero only well inside the exact zone, so the
// equilibrium of a pixel left alone stays exact (a zone reaching 1/2 bin let
// pixels settle right at the rounding edge and round wrong).
//   mode 1 ("deadzone"): lossWeight = u, every pass.
//   mode 2 ("commit"): settled[p] counts consecutive exact passes; at
//     commitAfter the pixel is committed (lossWeight 0), otherwise it's 1.
//     Without params.recheck a committed pixel is never evaluated again.
// Every candidate with weight > liveEps on a pixel still in the loss is
// marked in siteLive (cleared before each pass); Adam can freeze the rest.
// lossWeight feeds gradients_tiled and site_stats in place of the mask.

struct InfoParams {
  siteCount : u32,
  mode : u32,
  commitAfter : u32,
  recheck : u32,
  invScaleSq : f32,
  quant : f32,
  liveEps : f32,
  _pad0 : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : InfoParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read> mask : array<f32>;
@group(0) @binding(6) var<storage, read_write> lossWeight : array<f32>;
@group(0) @binding(7) var<storage, read_write> settled : array<u32>;
@group(0) @binding(8) var<storage, read_write> siteLive : array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(targetTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let p = gid.y * dims.x + gid.x;
  if (!(mask[p] > 0.0)) { lossWeight[p] = 0.0; return; }
  let committed = params.mode == 2u && settled[p] >= params.commitAfter;
  if (committed && params.recheck == 0u) { lossWeight[p] = 0.0; return; }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y));
  let inv_scale = sqrt(params.invScaleSq);
  let cc = vec2<i32>(candidateCoord(vec2<u32>(gid.xy), dims, textureDimensions(inCand0)));
  let ids = cand_ids(textureLoad(inCand0, cc, 0), textureLoad(inCand1, cc, 0));
  var logits : array<f32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    logits[i] = NEG_INF;
    if (ids[i] < params.siteCount) { logits[i] = site_logit(sites[ids[i]], uv, params.invScaleSq, inv_scale); }
  }
  let w = softmax8(logits);
  var pred = vec3<f32>(0.0);
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (w[i] > 0.0) {
      let c = site_color(sites[ids[i]]);
      if (!is_bad3(c)) { pred = pred + w[i] * c; }
    }
  }
  let t = textureLoad(targetTex, vec2<i32>(gid.xy), 0).rgb;
  let d = abs(clamp(pred, vec3<f32>(0.0), vec3<f32>(1.0)) - t) / params.quant;
  var e = max(d.x, max(d.y, d.z));
  if (is_bad(e)) { e = 1e6; }
  let u = clamp((e - 0.25) * 4.0, 0.0, 1.0);

  var weight = u;
  if (params.mode == 2u) {
    var s = settled[p];
    if (u == 0.0) { s = min(s + 1u, params.commitAfter); } else { s = 0u; }
    settled[p] = s;
    weight = select(1.0, 0.0, s >= params.commitAfter);
  }
  lossWeight[p] = weight;
  if (weight > 0.0) {
    for (var i = 0u; i < 8u; i = i + 1u) {
      if (w[i] > params.liveEps) { atomicOr(&siteLive[ids[i]], 1u); }
    }
  }
}
