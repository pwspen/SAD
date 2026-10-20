// Blends the candidates' colors, or with HASH_IDS each site's hashColor(id)
// (the cell-id view). With OWNER_IDS, writes the id of the candidate with the
// largest weight as f32 in r (-1 for none) instead of a color.
override HASH_IDS : bool = false;
override OWNER_IDS : bool = false;

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = params.width;
  let h = params.height;
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y));
  let inv_scale = sqrt(params.invScaleSq);
  let candCoord = candidateCoord(vec2<u32>(gid.xy), vec2<u32>(w, h), vec2<u32>(params.candWidth, params.candHeight));
  let candXY = vec2<i32>(i32(candCoord.x), i32(candCoord.y));
  let candIds = cand_ids(textureLoad(inCand0, candXY, 0), textureLoad(inCand1, candXY, 0), params.globalCand);

  var logits : array<f32, NCAND>;
  for (var i = 0u; i < NCAND; i = i + 1u) {
    let idx = candIds[i];
    logits[i] = NEG_INF;
    if (idx < params.siteCount) { logits[i] = site_logit(sites[idx], uv, params.invScaleSq, inv_scale); }
  }
  let weights = softmax_c(logits);

  if (OWNER_IDS) {
    var owner : f32 = -1.0;
    var best : f32 = 0.0;
    for (var i = 0u; i < NCAND; i = i + 1u) {
      if (weights[i] > best) { best = weights[i]; owner = f32(candIds[i]); }
    }
    textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(owner, 0.0, 0.0, 1.0));
    return;
  }

  var color = vec3<f32>(0.0);
  for (var i = 0u; i < NCAND; i = i + 1u) {
    if (weights[i] > 0.0) {
      let c = select(site_color(sites[candIds[i]]), hashColor(candIds[i]), HASH_IDS);
      if (!is_bad3(c)) { color = color + weights[i] * c; }
    }
  }
  if (is_bad3(color)) { color = vec3<f32>(0.0); }
  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
