// Site / Adam / gradient / stats / pairs / indices buffer helpers.

import { SITE_FLOATS, ADAM_FLOATS, PACKED_CAND_BYTES } from "./params.js?v=20261019";

export function createSitesBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * SITE_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

export function createAdamBuffer(device, capacity) {
  const buf = device.createBuffer({
    size: capacity * ADAM_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const zeros = new Float32Array(capacity * ADAM_FLOATS);
  device.queue.writeBuffer(buf, 0, zeros);
  return buf;
}

export function createPackedCandidatesBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * PACKED_CAND_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

// init_gradient's sample counters: one per tile (batch.js), one untiled.
export function createSeedCounter(device, tiles = 1) {
  const buf = device.createBuffer({
    size: tiles * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, new Uint32Array(tiles));
  return buf;
}

// Consolidated gradient buffer: `grads[idx * 10 + channel]`, f32 bits in atomic<u32>.
// This replaces the 10 separate per-channel buffers the Python/Metal backends
// use, so we stay within WebGPU's 10-storage-buffer-per-stage limit.
export function createGradsBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * 10 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

// Read `count` sites from `sitesBuffer` as a Float32Array of shape (count, SITE_FLOATS).
// Mirrors read_sites() in train_wgpu.py.
export async function readSites(device, sitesBuffer, count) {
  const size = count * SITE_FLOATS * 4;
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sitesBuffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return data;
}
