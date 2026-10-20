@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
// One counter per tile (one untiled): each image walks the sequence below
// exactly as it would alone.
@group(0) @binding(1) var<storage, read_write> seedCounter : array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params : InitParams;
@group(0) @binding(3) var targetTex : texture_2d<f32>;
@group(0) @binding(4) var maskTex : texture_2d<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.numSites) { return; }
  // The layer's slot (written by the trainer) draws nothing from its tile's
  // counter, as a lone run's init never places its layer.
  if (tiled(params.tiling) && gid.x % params.tiling.sitesPerTile == params.skipSlot) { return; }

  let dims = textureDimensions(targetTex);
  let width = i32(dims.x);
  let height = i32(dims.y);
  // Tiled: each site samples inside its own tile, whose edge bounds the
  // gradient's neighbourhood.
  var span = vec2<f32>(f32(width - 1), f32(height - 1));
  var origin = vec2<f32>(0.0);
  if (tiled(params.tiling)) {
    span = vec2<f32>(f32(params.tiling.tileW - 1u), f32(params.tiling.tileH - 1u));
    origin = site_tile_origin(params.tiling, gid.x);
  }
  let lo = vec2<i32>(origin);
  let hi = lo + vec2<i32>(span);

  var acceptedPos = vec2<f32>(-1.0, -1.0);
  var acceptedColor = vec3<f32>(0.5, 0.5, 0.5);
  var acceptedGradient = 0.0;

  for (var attempt = 0u; attempt < params.maxAttempts; attempt = attempt + 1u) {
    var tile = 0u;
    if (tiled(params.tiling)) { tile = gid.x / params.tiling.sitesPerTile; }
    let seed = atomicAdd(&seedCounter[tile], 1u);
    let g = 1.324717957244746;
    let a1 = 1.0 / g;
    let a2 = 1.0 / (g * g);
    let uv = fract(vec2<f32>(0.5, 0.5) + vec2<f32>(f32(seed) * a1, f32(seed) * a2));
    let pos = origin + uv * span;
    let ip = vec2<i32>(pos);

    let mask_val = textureLoad(maskTex, ip, 0).r;
    if (mask_val <= 0.0) {
      continue;
    }

    let pTL = clamp(ip + vec2<i32>(-1, -1), lo, hi);
    let pT  = clamp(ip + vec2<i32>( 0, -1), lo, hi);
    let pTR = clamp(ip + vec2<i32>( 1, -1), lo, hi);
    let pL  = clamp(ip + vec2<i32>(-1,  0), lo, hi);
    let pR  = clamp(ip + vec2<i32>( 1,  0), lo, hi);
    let pBL = clamp(ip + vec2<i32>(-1,  1), lo, hi);
    let pB  = clamp(ip + vec2<i32>( 0,  1), lo, hi);
    let pBR = clamp(ip + vec2<i32>( 1,  1), lo, hi);

    let TL = dot(textureLoad(targetTex, pTL, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let T  = dot(textureLoad(targetTex, pT, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let TR = dot(textureLoad(targetTex, pTR, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let L  = dot(textureLoad(targetTex, pL, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let R  = dot(textureLoad(targetTex, pR, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let BL = dot(textureLoad(targetTex, pBL, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let B  = dot(textureLoad(targetTex, pB, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let BR = dot(textureLoad(targetTex, pBR, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));

    let gx = (-TL + TR - 2.0 * L + 2.0 * R - BL + BR) / 4.0;
    let gy = (-TL - 2.0 * T - TR + BL + 2.0 * B + BR) / 4.0;
    let grad = sqrt(gx * gx + gy * gy) * mask_val;

    if (grad > params.gradThreshold) {
      acceptedPos = pos;
      acceptedColor = textureLoad(targetTex, ip, 0).rgb;
      acceptedGradient = grad;
      break;
    }
  }

  var s = sites[gid.x];
  s.position = acceptedPos;
  s.log_tau = params.initLogTau + pow(acceptedGradient, 0.25) * 2.0;
  s.radius_sq = params.initRadius;
  let colorParams = encode_color(acceptedColor);
  s.color_r = colorParams.x;
  s.color_g = colorParams.y;
  s.color_b = colorParams.z;
  s.aniso_dir_x = 1.0;
  s.aniso_dir_y = 0.0;
  s.log_aniso = 0.0;
  sites[gid.x] = s;
}
