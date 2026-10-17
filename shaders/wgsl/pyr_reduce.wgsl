// Residual-flow pyramid, step 3: level `pyr.level` cells sum their 2x2 children
// from the level below. Run once per level, coarsening to 1x1.

@group(0) @binding(0) var<storage, read_write> cells : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> pyr : PyrParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let lv = pyr.levels[pyr.level];
  if (gid.x >= lv.x || gid.y >= lv.y) { return; }
  let fine = pyr.levels[pyr.level - 1u];
  var s = vec2<f32>(0.0);
  for (var dy = 0u; dy < 2u; dy = dy + 1u) {
    for (var dx = 0u; dx < 2u; dx = dx + 1u) {
      let fx = gid.x * 2u + dx;
      let fy = gid.y * 2u + dy;
      if (fx < fine.x && fy < fine.y) { s = s + cells[fine.z + fy * fine.x + fx]; }
    }
  }
  cells[lv.z + gid.y * lv.x + gid.x] = s;
}
