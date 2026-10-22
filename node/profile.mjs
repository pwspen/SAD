#!/usr/bin/env node
// Where a training run's time goes: GPU time per kernel (timestamp queries on
// every compute pass, grouped by the pipeline's shader) against the run's
// wall clock, and the wall clock again without log/preview ticks.
//
//   node profile.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 4096 \
//     --preset ../presets/mnist.json
//   node profile.mjs --images ../../comp/data/kodak --count 1 --preset ../presets/kodak.json
//
// Three runs of the same images: as configured (wall clock), with log and
// preview ticks off (wall clock), and as configured with every pass timed.
// Timing adds a query resolve and readback per submit, so the third run's wall
// clock is not reported; its per-kernel GPU times are.

import { parseArgs } from "node:util";
import { create, globals } from "webgpu";

import { requestTrainingDevice, adapterLabel } from "../trainer/device.js?v=20261019";
import { Trainer } from "../trainer/trainer.js?v=w2b-20261021";
import { planBatches } from "../trainer/batch.js?v=rui-20261021";
import { whiteMaskRgba32Float } from "../trainer/textures.js?v=20261019";
import { loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched } from "./common.mjs";

Object.assign(globalThis, globals);

const HELP = `Usage: node profile.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir of PNGs>) [options]

  --from <n> / --count <n>  Images (default 0 / 64).
  --preset <json>    presets/<name>.json ({ args }); --set overrides it.
  --set key=value    Trainer arg or training_config.json key. Repeatable.
  --alone            Train each image alone (Trainer) instead of in atlases.
  --iters <n>        Override the iteration count.
`;

const QUERIES = 2048;
const keepAlive = [];

// A Dawn device with timestamp queries (unquantized), set up as
// requestTrainingDevice does otherwise.
async function createProfilingDevice() {
  const gpu = create(["disable-dawn-features=timestamp_quantization"]);
  keepAlive.push(gpu);
  const wrapped = {
    requestAdapter: async (opts) => {
      const adapter = await gpu.requestAdapter(opts);
      keepAlive.push(adapter);
      const request = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = (desc) => request({ ...desc, requiredFeatures: ["timestamp-query"] });
      return adapter;
    },
  };
  const { adapter, device } = await requestTrainingDevice(wrapped);
  device.addEventListener?.("uncapturederror", (e) => console.error(`GPU error: ${e.error?.message || e.error}`));
  return { device, label: adapterLabel(adapter) };
}

// Wraps `device` so every compute pass writes begin/end timestamps, keyed by
// its pipeline's shader name (from `shaders`). Must run before any pipeline
// is made. Returns { enable(on), drain(): Promise<Map name → { ns, passes }> }.
function instrument(device, shaders) {
  // Kernels are sad_shared.wgsl + their own snippet; the trainer may rewrite
  // the shared part (color spaces), so match on the start of the snippet.
  const codes = Object.entries(shaders).filter(([, v]) => typeof v === "string");
  let common = codes[0][1].length;
  for (const [, v] of codes) while (v.slice(0, common) !== codes[0][1].slice(0, common)) common -= 1;
  const heads = codes.map(([k, v]) => [k, v.slice(common, common + 400)]);
  const nameOf = (code) => heads.find(([, h]) => code.includes(h))?.[0] ?? "other";
  const moduleName = new WeakMap();
  const pipelineName = new WeakMap();
  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    const m = createShaderModule(desc);
    moduleName.set(m, nameOf(desc.code));
    return m;
  };
  const createComputePipeline = device.createComputePipeline.bind(device);
  device.createComputePipeline = (desc) => {
    const p = createComputePipeline(desc);
    const entry = desc.compute.entryPoint && desc.compute.entryPoint !== "main" ? `:${desc.compute.entryPoint}` : "";
    pipelineName.set(p, (moduleName.get(desc.compute.module) ?? "other") + entry);
    return p;
  };

  let on = false;
  const pool = [];
  const pending = [];
  const totals = new Map();
  const byBuffer = new WeakMap();
  const take = () => pool.pop() ?? {
    qs: device.createQuerySet({ type: "timestamp", count: QUERIES }),
    resolve: device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    staging: device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  };

  const createCommandEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (desc) => {
    const enc = createCommandEncoder(desc);
    if (!on) return enc;
    const slot = take();
    const names = [];
    const begin = enc.beginComputePass.bind(enc);
    enc.beginComputePass = (d = {}) => {
      if (names.length * 2 >= QUERIES) return begin(d);
      const q = names.length * 2;
      names.push("other");
      const pass = begin({ ...d, timestampWrites: { querySet: slot.qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
      const setPipeline = pass.setPipeline.bind(pass);
      pass.setPipeline = (pl) => { names[q / 2] = pipelineName.get(pl) ?? "other"; setPipeline(pl); };
      return pass;
    };
    const finish = enc.finish.bind(enc);
    enc.finish = (d) => {
      if (!names.length) { pool.push(slot); return finish(d); }
      enc.resolveQuerySet(slot.qs, 0, names.length * 2, slot.resolve, 0);
      enc.copyBufferToBuffer(slot.resolve, 0, slot.staging, 0, names.length * 16);
      const cb = finish(d);
      byBuffer.set(cb, { slot, names });
      return cb;
    };
    return enc;
  };

  const queue = device.queue;
  const submit = queue.submit.bind(queue);
  queue.submit = (cbs) => {
    submit(cbs);
    for (const cb of cbs) {
      const rec = byBuffer.get(cb);
      if (!rec) continue;
      const { slot, names } = rec;
      pending.push(slot.staging.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(slot.staging.getMappedRange(0, names.length * 16));
        names.forEach((name, i) => {
          const ns = Number(t[2 * i + 1] - t[2 * i]);
          const e = totals.get(name) ?? { ns: 0, passes: 0 };
          if (ns > 0) e.ns += ns;
          e.passes += 1;
          totals.set(name, e);
        });
        slot.staging.unmap();
        pool.push(slot);
      }));
    }
  };

  return {
    enable(v) { on = v; },
    async drain() { await Promise.all(pending.splice(0)); return totals; },
  };
}

// Wall time in Trainer.prepare / run / finalize (BatchTrainer delegates to
// them), summed over every trainer while `on`.
function phaseTimer() {
  const totals = { prepare: 0, run: 0, finalize: 0 };
  let on = false;
  for (const k of Object.keys(totals)) {
    const fn = Trainer.prototype[k];
    Trainer.prototype[k] = async function (...a) {
      const t0 = performance.now();
      try { return await fn.apply(this, a); } finally { if (on) totals[k] += (performance.now() - t0) / 1000; }
    };
  }
  return { totals, enable(v) { on = v; } };
}

async function trainAlone(ctx, images) {
  const t0 = performance.now();
  for (const img of images) {
    const trainer = new Trainer(ctx.device, ctx.shaders, ctx.args);
    try {
      await trainer.prepare({
        width: img.width, height: img.height, targetRgba32f: img.data,
        maskRgba32f: whiteMaskRgba32Float(img.width, img.height).rgba, maskSum: img.width * img.height,
        initSites: null, initSiteCount: ctx.args.startSites,
      });
      await trainer.run();
      await trainer.finalize();
    } finally {
      trainer.destroy();
    }
  }
  return { seconds: (performance.now() - t0) / 1000 };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      alone: { type: "boolean", default: false }, iters: { type: "string" },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  if (cli.iters) args.iters = Number(cli.iters);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));

  const { device, label } = await createProfilingDevice();
  const shaders = await loadShadersFromDisk(device);
  const prof = instrument(device, shaders);
  const phases = phaseTimer();
  const run = (a) => {
    const ctx = { device, shaders, args: a, quiet: true };
    return cli.alone ? trainAlone(ctx, images) : trainBatched(ctx, images);
  };

  const atlases = cli.alone ? images.length : planBatches(images, args.candDownscale).length;
  const px = images.reduce((s, im) => s + im.width * im.height, 0);
  console.log(`Device  | ${label}`);
  console.log(`Run     | ${images.length} image(s) of ${images[0].width}×${images[0].height} ` +
              `${cli.alone ? "alone" : `in ${atlases} atlas(es)`}, ${args.startSites}→${args.endSites} sites, ` +
              `${args.iters} iterations, logFreq ${args.logFreq}, viewerFreq ${args.viewerFreq}, candFreq ${args.candFreq}`);

  await run({ ...args, iters: Math.min(args.iters, 20) }); // warm up pipelines
  phases.enable(true);
  const wall = (await run(args)).seconds;
  phases.enable(false);
  const quiet = (await run({ ...args, logFreq: 1e9, viewerFreq: 1e9 })).seconds;
  prof.enable(true);
  await run(args);
  prof.enable(false);
  const totals = await prof.drain();

  const its = args.iters * atlases;
  const rate = (sec) => `${(sec * 1000 / its).toFixed(2)} ms/it, ${(px * args.iters / sec / 1e6).toFixed(0)} Mpx-it/s`;
  console.log(`Wall    | ${wall.toFixed(2)} s (${rate(wall)})`);
  console.log(`No logs | ${quiet.toFixed(2)} s (${rate(quiet)}): log/preview ticks cost ${((wall - quiet) / wall * 100).toFixed(0)}%`);
  const p = phases.totals;
  console.log(`Phases  | prepare ${p.prepare.toFixed(2)} s, run ${p.run.toFixed(2)} s (${(p.run * 1000 / its).toFixed(2)} ms/it), finalize ${p.finalize.toFixed(2)} s, other ${(wall - p.prepare - p.run - p.finalize).toFixed(2)} s`);
  const gpuNs = [...totals.values()].reduce((s, e) => s + e.ns, 0);
  const gpu = gpuNs / 1e9;
  console.log(`GPU     | ${gpu.toFixed(2)} s in compute passes = ${(gpu / wall * 100).toFixed(0)}% of the wall clock ` +
              `(the rest: CPU encode/submit, readback waits, copies, idle)`);
  console.log("");
  console.log("(ms/it and it: per atlas-iteration, or per image-iteration with --alone)");
  console.log("kernel                 passes   passes/it   GPU s   ms/it   % GPU   % wall");
  for (const [name, e] of [...totals].sort((a, b) => b[1].ns - a[1].ns)) {
    const s = e.ns / 1e9;
    console.log(`${name.padEnd(22)} ${String(e.passes).padStart(6)} ${(e.passes / its).toFixed(2).padStart(11)} ` +
                `${s.toFixed(3).padStart(7)} ${(s * 1000 / its).toFixed(3).padStart(7)} ` +
                `${(s / gpu * 100).toFixed(1).padStart(7)} ${(s / wall * 100).toFixed(1).padStart(8)}`);
  }
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
