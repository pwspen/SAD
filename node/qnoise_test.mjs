// Checks the quantization noise (shaders/wgsl/qnoise.wgsl, args.quantNoise):
//   1. the kernel: each coded plane moves by at most Δ/2 (colors along the
//      basis axes, θ by π / 2m), uniformly (mean ≈ 0, variance ≈ Δ²/12);
//      positions stay in their tile; the layer and inactive sites don't move
//   2. the trainer removes it exactly: with every LR at 0, a batched run with
//      quantNoise leaves the sites bit-identical, like one without
// Usage: node qnoise_test.mjs

import { SITE_FLOATS } from "../trainer/params.js?v=m3-20261020";
import { colorSpaceFor, specializeShaders, encodeSiteColors, decodeSiteColors } from "../trainer/color_spaces.js?v=klt-20261020";
import { QuantNoiseEncoder } from "../trainer/encoders.js?v=qat-20261021";
import { readSites } from "../trainer/buffers.js?v=20261019";
import { quantNoiseBuffer, codecStep, QNOISE_FLOATS } from "../trainer/rate.js?v=qat-20261021";
import { BatchTrainer } from "../trainer/batch.js?v=qat-20261021";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { argsFor } from "./common.mjs";

let failures = 0;
function check(name, ok, detail) {
  if (!ok) {
    failures += 1;
    if (failures <= 20) console.error(`FAIL ${name}: ${detail}`);
  }
}

let seed = 777;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);

async function testKernel(device, baseShaders, spaceName) {
  // A rotated orthonormal basis, so color noise along it differs from RGB noise.
  const c = Math.cos(0.6), s = Math.sin(0.6);
  const basis = { mean: [0.4, 0.5, 0.45], axes: [[c, s, 0], [-s * 0.6, c * 0.6, 0.8], [s * 0.8, -c * 0.8, 0.6]] };
  const space = colorSpaceFor(spaceName, basis);
  const shaders = specializeShaders(baseShaders, space);
  const tiling = { tileW: 32, tileH: 32, sitesPerTile: 8, tilesX: 4 };
  const layerSlot = 0, n = 8 * 16 * 64;
  const sites = new Float32Array(n * SITE_FLOATS);
  for (let i = 0; i < n; i += 1) {
    const b = i * SITE_FLOATS, t = Math.floor(i / tiling.sitesPerTile);
    const ox = (t % tiling.tilesX) * 32, oy = Math.floor(t / tiling.tilesX) * 32;
    const th = rand() * Math.PI;
    sites.set([ox + rand() * 31, oy + rand() * 31, rand() * 4 - 2, rand() * 10, 0.2 + 0.6 * rand(), 0.2 + 0.6 * rand(), 0.2 + 0.6 * rand(),
               Math.cos(th), Math.sin(th), rand() * 4 - 2], b);
    if (i % 97 === 5) sites[b] = -1; // inactive
  }
  const steps = { pos: 1.3, logTau: 0.21, radius: 0.7, color: 0.05, logAniso: 0.17, theta: 0.3 };
  const buf = device.createBuffer({ size: sites.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(buf, 0, encodeSiteColors(sites.slice(), n, space));
  const qn = device.createBuffer({ size: QNOISE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(qn, 0, quantNoiseBuffer(steps, basis, { scale: 1, siteCount: n, layerSlot, seed: 99, width: 128, height: 512, tiling }));
  const enc = device.createCommandEncoder();
  new QuantNoiseEncoder(device, shaders).encode(enc, buf, qn, n);
  device.queue.submit([enc.finish()]);
  const out = decodeSiteColors(await readSites(device, buf, n), n, space);
  buf.destroy();
  qn.destroy();

  const D = { pos: codecStep(steps.pos), logTau: codecStep(steps.logTau), radius: codecStep(steps.radius),
              color: codecStep(steps.color), logAniso: codecStep(steps.logAniso), theta: codecStep(steps.theta) };
  const planes = { x: [], logTau: [], radius: [], k0: [], k1: [], k2: [], logAniso: [] };
  const tag = `${spaceName}`;
  for (let i = 0; i < n; i += 1) {
    const b = i * SITE_FLOATS;
    const a = sites.subarray(b, b + SITE_FLOATS), o = out.subarray(b, b + SITE_FLOATS);
    if (i % tiling.sitesPerTile === layerSlot || !(a[0] >= 0)) {
      const same = a.every((v, k) => Math.abs(v - o[k]) <= 1e-6 * Math.max(1, Math.abs(v)));
      check(`${tag} untouched`, same, `site ${i}: ${Array.from(a)} -> ${Array.from(o)}`);
      continue;
    }
    const t = Math.floor(i / tiling.sitesPerTile);
    const ox = (t % tiling.tilesX) * 32, oy = Math.floor(t / tiling.tilesX) * 32;
    check(`${tag} in tile`, o[0] >= ox && o[0] <= ox + 31 && o[1] >= oy && o[1] <= oy + 31, `site ${i}: ${o[0]}, ${o[1]}`);
    const dx = o[0] - a[0], dy = o[1] - a[1];
    check(`${tag} position`, Math.abs(dx) <= D.pos / 2 + 1e-4 && Math.abs(dy) <= D.pos / 2 + 1e-4, `site ${i}: ${dx}, ${dy}`);
    if (a[0] - ox > D.pos && ox + 31 - a[0] > D.pos) planes.x.push(dx);
    planes.logTau.push(o[2] - a[2]);
    planes.radius.push(o[3] - a[3]);
    const d = [o[4] - a[4], o[5] - a[5], o[6] - a[6]];
    basis.axes.forEach((ax, k) => planes[`k${k}`].push(ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2]));
    planes.logAniso.push(o[9] - a[9]);
    const m = Math.max(1, Math.min(4096, Math.round((Math.PI / D.theta) * Math.abs(2 * Math.sinh(o[9])))));
    const phi = Math.atan2(a[7] * o[8] - a[8] * o[7], a[7] * o[7] + a[8] * o[8]);
    check(`${tag} θ`, Math.abs(phi) <= Math.PI / (2 * m) + 1e-4, `site ${i}: φ ${phi}, m ${m}`);
    check(`${tag} |dir|`, Math.abs(Math.hypot(o[7], o[8]) - 1) < 1e-5, `site ${i}`);
  }
  const stepOf = { x: D.pos, logTau: D.logTau, radius: D.radius, k0: D.color, k1: D.color, k2: D.color, logAniso: D.logAniso };
  for (const [k, xs] of Object.entries(planes)) {
    const step = stepOf[k];
    const tol = k.startsWith("k") ? 2e-4 : 1e-4;
    const worst = Math.max(...xs.map(Math.abs));
    check(`${tag} ${k} bound`, worst <= step / 2 + tol, `max |δ| ${worst} > Δ/2 ${step / 2}`);
    const mean = xs.reduce((s, v) => s + v, 0) / xs.length;
    const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
    check(`${tag} ${k} mean`, Math.abs(mean) < 0.05 * step, `mean ${mean}, Δ ${step}`);
    check(`${tag} ${k} variance`, Math.abs(v / (step * step / 12) - 1) < 0.08, `var ${v} vs Δ²/12 ${step * step / 12}`);
  }
}

async function testRemoved(device, shaders) {
  const images = Array.from({ length: 6 }, (_, k) => {
    const data = new Float32Array(16 * 16 * 4);
    for (let p = 0; p < 256; p += 1) data.set([((p * (k + 3)) % 17) / 16, ((p >> 4) % 5) / 4, (p % 16) / 15, 1], p * 4);
    return { data, width: 16, height: 16 };
  });
  const run = async (quantNoise) => {
    const { args } = await argsFor({ iters: 12, sites: 6, countSchedule: "constant", layer: "learn", colorSpace: "klt",
                                     lrMult: 0, rateEvery: 3, quantNoise, rateLossDb: 2 });
    args.logFreq = args.iters;
    const bt = new BatchTrainer(device, shaders, args);
    await bt.prepare(images);
    const t = bt.trainer;
    const before = await readSites(device, t.sitesBuf, t.siteCount);
    await bt.run();
    const after = await readSites(device, t.sitesBuf, t.siteCount);
    const used = !!t._qnClean;
    bt.destroy();
    const u0 = new Uint32Array(before.buffer), u1 = new Uint32Array(after.buffer);
    let diff = 0;
    for (let i = 0; i < u0.length; i += 1) if (u0[i] !== u1[i]) diff += 1;
    return { diff, used };
  };
  const off = await run(0);
  check("lr 0 control", off.diff === 0, `${off.diff} site words changed without noise`);
  const on = await run(1);
  check("noise was applied", on.used, "quantNoise never ran");
  check("noise removed exactly", on.diff === 0, `${on.diff} site words changed with noise`);
}

const { device } = await createHeadlessDevice();
const shaders = await loadShadersFromDisk(device);
for (const name of ["rgb", "ycocg", "klt"]) await testKernel(device, shaders, name);
await testRemoved(device, shaders);
device.destroy();
if (failures) {
  console.error(`qnoise_test: ${failures} failure(s)`);
  process.exit(1);
}
console.log("qnoise_test: ok");
process.exit(0);
