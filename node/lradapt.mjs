#!/usr/bin/env node
// Per-group LR adaptation (args.lrAdapt, lr_adapt.wgsl): can it replace
// tuning the six LR bases? Batch mode, held-out images:
//
//   node lradapt.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --other ../presets/cifar10.json --rates 0,0.1,0.3,1
//
// Starting points: the preset's tuned bases; all six scaled ×1/4 and ×4; and
// --other's bases (a preset tuned on another dataset). Each at every
// adaptation rate in --rates (0 = off) steering to cosine --target, --repeats times; reports the median
// PSNR and the final per-group multipliers.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { LR_GROUP_KEYS } from "../trainer/trainer.js?v=lra-20261021";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";
import { LR_KEYS } from "./tune.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

async function run(ctx, params, images) {
  const { args } = await argsFor(params);
  args.logFreq = args.iters;
  let log = null;
  const { results } = await trainBatched({ ...ctx, args, quiet: true }, images, {
    after: async (bt) => { log = bt.trainer.lrAdaptLog; },
  });
  return { psnr: median(results.map((r) => r.psnr)), log };
}

async function main() {
  const { values: cli } = parseArgs({
    options: { ...DATASET_OPTIONS, other: { type: "string" }, rates: { type: "string", default: "0,0.1,0.3,1" }, target: { type: "string", default: "0" },
               repeats: { type: "string", default: "3" }, out: { type: "string" } },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  const scaled = (f) => Object.fromEntries(LR_KEYS.map((k) => [k, Number((base[k] * f).toPrecision(4))]));
  const starts = [
    { name: "tuned", lrs: {} },
    { name: "×1/4", lrs: scaled(0.25) },
    { name: "×4", lrs: scaled(4) },
  ];
  if (cli.other) {
    const o = await loadParams(cli.other);
    starts.push({ name: "other's", lrs: Object.fromEntries(LR_KEYS.map((k) => [k, o[k]])) });
  }
  const rates = cli.rates.split(",").map(Number);
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  const rows = [];
  console.error(`start     ${rates.map((r) => `rate ${r}`.padEnd(10)).join("")}  final multipliers at rate ${rates.at(-1)} (${LR_GROUP_KEYS.join(", ")})`);
  for (const s of starts) {
    const cells = [];
    let lastLog = null;
    for (const rate of rates) {
      const reps = [];
      for (let k = 0; k < repeats; k += 1) reps.push(await run(ctx, { ...base, ...s.lrs, lrAdapt: rate, lrAdaptTarget: Number(cli.target) }, images));
      const m = median(reps.map((r) => r.psnr));
      cells.push(m);
      lastLog = reps[0].log;
      rows.push({ start: s.name, lrs: s.lrs, rate, psnr: m, runs: reps.map((r) => r.psnr), mults: reps[0].log.at(-1)?.mults ?? null,
                  trace: reps[0].log });
    }
    const mults = lastLog?.at(-1)?.mults?.map((x) => x.toFixed(2)).join(", ") ?? "—";
    console.error(`${s.name.padEnd(9)} ${cells.map((c) => c.toFixed(3).padEnd(10)).join("")}  ${mults}`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/lradapt_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, rows }, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
