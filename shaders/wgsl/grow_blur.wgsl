// Mask-normalized separable Gaussian blur of the target, for site growth
// (trainer.js): while sites are added, the trainer fits this blurred target
// instead of the original, with sigma tied to the current site spacing.
//
// blur_h: (target * mask, mask) blurred along x into `outTex`.
// blur_v: that blurred along y, then rgb / mask weight, into `outTex`
//         (normalized convolution, so masked-out pixels don't darken edges).

struct GrowBlurParams {
  width : u32,
  height : u32,
  radius : u32,
  _pad0 : u32,
  invTwoSigmaSq : f32,
  _pad1 : f32,
  _pad2 : f32,
  _pad3 : f32,
}

@group(0) @binding(0) var<uniform> params : GrowBlurParams;
@group(0) @binding(1) var srcTex : texture_2d<f32>;
@group(0) @binding(2) var maskTex : texture_2d<f32>;
@group(0) @binding(3) var outTex : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let r = i32(params.radius);
  let x0 = i32(gid.x);
  let y = i32(gid.y);
  var acc = vec4<f32>(0.0);
  for (var dx = -r; dx <= r; dx = dx + 1) {
    let x = x0 + dx;
    if (x < 0 || x >= i32(params.width)) { continue; }
    let w = exp(-f32(dx * dx) * params.invTwoSigmaSq);
    let m = textureLoad(maskTex, vec2<i32>(x, y), 0).r;
    let c = textureLoad(srcTex, vec2<i32>(x, y), 0).rgb;
    acc = acc + w * vec4<f32>(c * m, m);
  }
  textureStore(outTex, vec2<i32>(gid.xy), acc);
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let r = i32(params.radius);
  let x = i32(gid.x);
  let y0 = i32(gid.y);
  var acc = vec4<f32>(0.0);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    let y = y0 + dy;
    if (y < 0 || y >= i32(params.height)) { continue; }
    let w = exp(-f32(dy * dy) * params.invTwoSigmaSq);
    acc = acc + w * textureLoad(srcTex, vec2<i32>(x, y), 0);
  }
  var rgb = vec3<f32>(0.0);
  if (acc.a > 1.0e-6) { rgb = acc.rgb / acc.a; }
  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}
