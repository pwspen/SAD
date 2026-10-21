// Checks every color space in trainer/color_spaces.js:
//   1. decode(encode(rgb)) == rgb                       (JS)
//   2. vjp matches finite differences of decode         (JS)
//   3. WGSL encode/decode/color_vjp match the JS ones   (GPU, via Dawn)
// Usage: node color_spaces_test.mjs [--no-gpu]

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { COLOR_SPACES, specializeShaders } from "../trainer/color_spaces.js?v=ycocg-20261020";

const SHARED_URL = new URL("../shaders/sad_shared.wgsl", import.meta.url);

// Deterministic samples: corners and edges of the cube plus pseudo-random
// interior points; gradients in [-1, 1].
function samples(n = 256) {
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const rgb = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0.02, 0.9, 0.3]];
  while (rgb.length < n) rgb.push([rand(), rand(), rand()]);
  const grad = rgb.map(() => [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);
  return { rgb, grad };
}

let failures = 0;
function check(name, ok, detail) {
  if (!ok) {
    failures += 1;
    console.error(`FAIL ${name}: ${detail}`);
  }
}
const maxAbs = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const scaledErr = (a, b) => maxAbs(a, b) / Math.max(1, ...b.map(Math.abs));

function testJs(name, space, { rgb, grad }) {
  for (let i = 0; i < rgb.length; i += 1) {
    const p = space.encode(...rgb[i]);
    const back = space.decode(...p);
    check(`${name} round trip`, maxAbs(back, rgb[i]) < 1e-9, `rgb ${rgb[i]} -> ${back}`);

    // dL/dp_k = sum_c g_c dRGB_c/dp_k, by central differences.
    const h = 1e-6;
    const fd = [0, 1, 2].map((k) => {
      const hi = p.slice(); hi[k] += h;
      const lo = p.slice(); lo[k] -= h;
      const dHi = space.decode(...hi);
      const dLo = space.decode(...lo);
      return [0, 1, 2].reduce((acc, c) => acc + grad[i][c] * (dHi[c] - dLo[c]) / (2 * h), 0);
    });
    const vjp = space.vjp(p, grad[i]);
    check(`${name} vjp`, scaledErr(vjp, fd) < 1e-6, `p ${p}: vjp ${vjp} vs fd ${fd}`);
  }
}

const KERNEL = `
struct Sample { rgb : vec4<f32>, grad : vec4<f32> }
struct Result { encoded : vec4<f32>, decoded : vec4<f32>, vjp : vec4<f32> }
@group(1) @binding(0) var<storage, read> inputs : array<Sample>;
@group(1) @binding(1) var<storage, read_write> results : array<Result>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inputs)) { return; }
  let p = encode_color(inputs[i].rgb.xyz);
  results[i].encoded = vec4<f32>(p, 0.0);
  results[i].decoded = vec4<f32>(decode_color(p), 0.0);
  results[i].vjp = vec4<f32>(color_vjp(p, inputs[i].grad.xyz), 0.0);
}
`;

async function testGpu(common, { rgb, grad }) {
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const n = rgb.length;
  const input = new Float32Array(n * 8);
  rgb.forEach((c, i) => { input.set(c, i * 8); input.set(grad[i], i * 8 + 4); });
  const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input);
  const outBytes = n * 12 * 4;

  for (const [name, space] of Object.entries(COLOR_SPACES)) {
    const code = specializeShaders({ COMMON: common + KERNEL }, space).COMMON;
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    check(`${name} WGSL compiles`, errors.length === 0,
      errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
    if (errors.length) continue;

    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, staging, 0, outBytes);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();

    // f32 on the GPU vs f64 here; log/exp near EPS loses a few digits.
    const TOL = 2e-4;
    for (let i = 0; i < n; i += 1) {
      const row = (k) => Array.from(out.subarray(i * 12 + k * 4, i * 12 + k * 4 + 3));
      const p = space.encode(...rgb[i]);
      check(`${name} WGSL encode`, scaledErr(row(0), p) < TOL, `rgb ${rgb[i]}: ${row(0)} vs ${p}`);
      check(`${name} WGSL decode`, scaledErr(row(1), rgb[i]) < TOL, `rgb ${rgb[i]}: ${row(1)}`);
      const vjp = space.vjp(p, grad[i]);
      check(`${name} WGSL vjp`, scaledErr(row(2), vjp) < TOL, `rgb ${rgb[i]}: ${row(2)} vs ${vjp}`);
    }
    outBuf.destroy();
    staging.destroy();
  }
  device.destroy();
  return gpu;
}

const data = samples();
for (const [name, space] of Object.entries(COLOR_SPACES)) testJs(name, space, data);

if (!process.argv.includes("--no-gpu")) {
  const shared = await readFile(fileURLToPath(SHARED_URL), "utf8");
  const common = shared;
  globalThis.__keepGpu = await testGpu(common, data); // dawn.node segfaults if the GPU object is collected
}

const spaces = Object.keys(COLOR_SPACES).join(", ");
if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log(`ok: ${spaces}`);
