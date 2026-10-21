// Checks every color space in trainer/color_spaces.js:
//   1. decode(encode(rgb)) == rgb                       (JS)
//   2. vjp matches finite differences of decode         (JS)
//   3. WGSL encode/decode/color_vjp match the JS ones   (GPU, via Dawn)
// klt with a k-axis basis (colorDims k < 3) instead of 1. checks that encode
// is the projection onto the first k axes (dropped coordinates exactly 0),
// that decode(encode(·)) is idempotent and ignores the dropped coordinates,
// and that the VJP is 0 for them; 2. and 3. as for the others.
// Usage: node color_spaces_test.mjs [--no-gpu]

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { COLOR_SPACES, specializeShaders, colorSpaceFor, pcaBasis, withDims } from "../trainer/color_spaces.js?v=cdims-20261021";

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

// rgb projected onto the affine span of the basis's first `dims` axes.
function project(basis, rgb) {
  const { mean: m, axes: a } = basis;
  const out = m.slice();
  for (let k = 0; k < (basis.dims ?? 3); k += 1) {
    const t = a[k][0] * (rgb[0] - m[0]) + a[k][1] * (rgb[1] - m[1]) + a[k][2] * (rgb[2] - m[2]);
    for (let c = 0; c < 3; c += 1) out[c] += t * a[k][c];
  }
  return out;
}

function testJs(name, space, { rgb, grad }) {
  const dims = space.basis?.dims ?? 3;
  for (let i = 0; i < rgb.length; i += 1) {
    const p = space.encode(...rgb[i]);
    const back = space.decode(...p);
    if (dims === 3) {
      check(`${name} round trip`, maxAbs(back, rgb[i]) < 1e-9, `rgb ${rgb[i]} -> ${back}`);
    } else {
      const proj = project(space.basis, rgb[i]);
      check(`${name} projects`, maxAbs(back, proj) < 1e-9, `rgb ${rgb[i]} -> ${back} vs ${proj}`);
      check(`${name} dropped coords 0`, p.slice(dims).every((v) => v === 0), `rgb ${rgb[i]} -> p ${p}`);
      check(`${name} idempotent`, maxAbs(space.encode(...back), p) < 1e-9, `p ${p} -> ${space.encode(...back)}`);
      const noisy = p.map((v, k) => (k < dims ? v : v + 0.37 * (k + 1)));
      check(`${name} decode ignores dropped`, maxAbs(space.decode(...noisy), back) < 1e-12, `p ${noisy}`);
      check(`${name} vjp dropped 0`, space.vjp(p, grad[i]).slice(dims).every((v) => v === 0), `vjp ${space.vjp(p, grad[i])}`);
    }

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
struct Result { encoded : vec4<f32>, decoded : vec4<f32>, vjp : vec4<f32>, decodedRaw : vec4<f32> }
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
  // The sample as params directly (every coordinate nonzero, the dropped
  // ones of a k-axis basis included).
  results[i].decodedRaw = vec4<f32>(decode_color(inputs[i].rgb.xyz), 0.0);
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
  const outBytes = n * 16 * 4;

  for (const [name, space] of Object.entries(SPACES)) {
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
      const row = (k) => Array.from(out.subarray(i * 16 + k * 4, i * 16 + k * 4 + 3));
      const p = space.encode(...rgb[i]);
      check(`${name} WGSL encode`, scaledErr(row(0), p) < TOL, `rgb ${rgb[i]}: ${row(0)} vs ${p}`);
      check(`${name} WGSL decode`, scaledErr(row(1), space.decode(...p)) < TOL, `rgb ${rgb[i]}: ${row(1)}`);
      const vjp = space.vjp(p, grad[i]);
      check(`${name} WGSL vjp`, scaledErr(row(2), vjp) < TOL, `rgb ${rgb[i]}: ${row(2)} vs ${vjp}`);
      const raw = space.decode(...rgb[i]);
      check(`${name} WGSL decode (raw params)`, scaledErr(row(3), raw) < TOL, `p ${rgb[i]}: ${row(3)} vs ${raw}`);
    }
    outBuf.destroy();
    staging.destroy();
  }
  device.destroy();
  return gpu;
}

const data = samples();
// klt as listed has the identity basis; also check one from correlated colors.
const corr = new Float32Array(data.rgb.length * 4);
data.rgb.forEach(([r, g, b], i) => corr.set([0.7 * r + 0.2 * g, 0.5 * r + 0.4 * g + 0.1 * b, 0.3 * b + 0.6 * g, 1], i * 4));
const corrBasis = pcaBasis(corr, data.rgb.length);
const SPACES = {
  ...COLOR_SPACES,
  "klt (data basis)": colorSpaceFor("klt", corrBasis),
  "klt (data basis, 2 dims)": colorSpaceFor("klt", withDims(corrBasis, 2)),
  "klt (data basis, 1 dim)": colorSpaceFor("klt", withDims(corrBasis, 1)),
};
const { axes } = SPACES["klt (data basis)"].basis;
for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) {
  const d = axes[i][0] * axes[j][0] + axes[i][1] * axes[j][1] + axes[i][2] * axes[j][2];
  check("klt axes orthonormal", Math.abs(d - (i === j ? 1 : 0)) < 1e-9, `axes ${i}·${j} = ${d}`);
}
for (const [name, space] of Object.entries(SPACES)) testJs(name, space, data);

if (!process.argv.includes("--no-gpu")) {
  const shared = await readFile(fileURLToPath(SHARED_URL), "utf8");
  const common = shared;
  globalThis.__keepGpu = await testGpu(common, data); // dawn.node segfaults if the GPU object is collected
}

const spaces = Object.keys(SPACES).join(", ");
if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log(`ok: ${spaces}`);
