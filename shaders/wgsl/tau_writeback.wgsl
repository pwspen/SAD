// Writes diffused log_tau gradients back into channel 2 of `grads`.

@group(0) @binding(0) var<storage, read> grad_in : array<f32>;
@group(0) @binding(1) var<storage, read_write> grads : array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  let val = grad_in[idx];
  if (is_bad(val)) {
    atomicStore(&grads[idx * 10u + 2u], 0u);
  } else {
    atomicStore(&grads[idx * 10u + 2u], bitcast<u32>(val));
  }
}
