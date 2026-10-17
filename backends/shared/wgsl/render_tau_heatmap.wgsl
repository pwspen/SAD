// Tau heatmap: centroid dots blended over a dark background, colored on a
// blue->white->red scale from min->mean->max tau. Mirrors Metal's
// renderCentroidsTauHeatmap. The candidate textures give us the 8 nearest
// sites per pixel, which is all the overlap we need for dot rendering.
// Layout for heat (TauHeatmapParams, 16 bytes):
//   f32 minTau, f32 meanTau, f32 maxTau, f32 dotRadius.

struct TauHeatmapParams {
  minTau : f32,
  meanTau : f32,
  maxTau : f32,
  dotRadius : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> heat : TauHeatmapParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = params.width;
  let h = params.height;
  if (gid.x >= w || gid.y >= h) { return; }
  let uv = vec2<f32>(f32(gid.x), f32(gid.y));

  let candCoord = candidateCoord(vec2<u32>(gid.xy), vec2<u32>(w, h),
    vec2<u32>(params.candWidth, params.candHeight));
  let candXY = vec2<i32>(i32(candCoord.x), i32(candCoord.y));
  let c0 = textureLoad(inCand0, candXY, 0);
  let c1 = textureLoad(inCand1, candXY, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  let radius = heat.dotRadius;
  var accColor = vec3<f32>(0.0);
  var accAlpha : f32 = 0.0;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let id = candIds[i];
    if (id >= params.siteCount) { continue; }
    let site = sites[id];
    if (site.position.x < 0.0) { continue; }

    let dist = length(uv - site.position);
    let alpha = 1.0 - smoothstep(radius - 1.0, radius + 1.0, dist);
    if (alpha <= 0.0) { continue; }

    let tau = exp(site.log_tau);
    var color : vec3<f32>;
    if (tau <= heat.meanTau) {
      var t : f32 = 0.5;
      if (heat.meanTau > heat.minTau) { t = (tau - heat.minTau) / (heat.meanTau - heat.minTau); }
      color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), pow(clamp(t, 0.0, 1.0), 0.2));
    } else {
      var t : f32 = 0.5;
      if (heat.maxTau > heat.meanTau) { t = (tau - heat.meanTau) / (heat.maxTau - heat.meanTau); }
      color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), pow(clamp(t, 0.0, 1.0), 0.2));
    }

    let blend = alpha * (1.0 - accAlpha);
    accColor = accColor + color * blend;
    accAlpha = accAlpha + blend;
  }

  let bg = vec3<f32>(0.05, 0.05, 0.05);
  let finalColor = accColor + bg * (1.0 - accAlpha);
  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(finalColor, 1.0));
}
