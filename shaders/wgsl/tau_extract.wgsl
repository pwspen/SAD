// Copies the log_tau channel (2) of the interleaved `grads` buffer into a
// plain f32 array for tau_diffuse.

@group(0) @binding(0) var<storage, read_write> grads : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> grad_out : array<f32>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  grad_out[idx] = bitcast<f32>(atomicLoad(&grads[idx * 10u + 2u]));
}
