// Adapter/device setup shared by train.js, bench.html and the Node runner.
// `gpu` is navigator.gpu in the browser, or dawn.node's create([]) in Node.

export async function requestTrainingDevice(gpu) {
  if (!gpu) throw new Error("WebGPU not available.");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  const want = Math.max(12, adapter.limits.maxStorageBuffersPerShaderStage || 8);
  const requiredLimits = {
    maxStorageBuffersPerShaderStage: Math.min(want, adapter.limits.maxStorageBuffersPerShaderStage),
    maxBufferSize: Math.min(1 << 30, adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize: Math.min(1 << 30, adapter.limits.maxStorageBufferBindingSize),
    // Room for the 256-slot gradient tile table when the adapter allows it.
    maxComputeWorkgroupStorageSize: Math.min(32768, adapter.limits.maxComputeWorkgroupStorageSize),
  };
  const device = await adapter.requestDevice({ requiredLimits });
  return { adapter, device };
}

export function adapterLabel(adapter) {
  const info = adapter.info || {};
  return info.device || info.description || info.vendor || "unknown";
}
