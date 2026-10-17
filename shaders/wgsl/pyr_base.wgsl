// Residual-flow pyramid, step 2: level-0 cells hold (sum of per-pixel error
// EMA ^ gamma, site count) over their cs x cs pixel block.

@group(0) @binding(0) var<storage, read> errEma : array<f32>;
@group(0) @binding(1) var<storage, read> cnt0 : array<u32>;
@group(0) @binding(2) var<storage, read_write> cells : array<vec2<f32>>;
@group(0) @binding(3) var<uniform> pyr : PyrParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let lv = pyr.levels[0];
  if (gid.x >= lv.x || gid.y >= lv.y) { return; }
  let x0 = gid.x * lv.w;
  let y0 = gid.y * lv.w;
  let x1 = min(x0 + lv.w, pyr.width);
  let y1 = min(y0 + lv.w, pyr.height);
  var e = 0.0;
  for (var y = y0; y < y1; y = y + 1u) {
    for (var x = x0; x < x1; x = x + 1u) {
      let v = errEma[y * pyr.width + x];
      if (!is_bad(v) && v > 0.0) { e = e + pow(v, pyr.gamma); }
    }
  }
  let c = gid.y * lv.x + gid.x;
  cells[lv.z + c] = vec2<f32>(e, f32(cnt0[c]));
}
