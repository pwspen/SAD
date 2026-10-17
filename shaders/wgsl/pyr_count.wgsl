// Residual-flow pyramid, step 1: bin each active site into its level-0 cell
// (cell size levels[0].w px). `cnt0` is cleared by the encoder beforehand.

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> cnt0 : array<atomic<u32>>;
@group(0) @binding(2) var<uniform> pyr : PyrParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= pyr.siteCount) { return; }
  let p = sites[idx].position;
  if (p.x < 0.0 || is_bad2(p)) { return; }
  let lv = pyr.levels[0];
  let cs = f32(lv.w);
  let cx = min(u32(max(p.x, 0.0) / cs), lv.x - 1u);
  let cy = min(u32(max(p.y, 0.0) / cs), lv.y - 1u);
  atomicAdd(&cnt0[cy * lv.x + cx], 1u);
}
