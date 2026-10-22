#!/usr/bin/env node
// Coordinate search vs CMA-ES (tune.mjs) at equal cost: held-out gain per
// tuning batch.
//
//   node tunecmp.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 --holdout 2048 \
//     --preset ../presets/mnist.json --budget 240 --repeats 3 --out ../results/tunecmp/cifar_lr.json
//
// Each repeat runs both searches from the preset on the same --count tuning
// images, capped at --budget batches (coord with enough cycles that the budget
// binds; cma with seed = repeat). Every --every batches, each search's
// incumbent at that point (coord: its current point; cma: the last scored
// mean, and the best sample so far) is scored on the --holdout images after
// the tuning ones. Each distinct setting is scored there once; the start is
// scored --start-runs times (runs aren't bit-reproducible) and gains are
// against the median of those. Prints mean (min…max) held-out gain over the
// repeats per checkpoint and method.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";
import { makeEvaluator, tune, cmaes, LR_KEYS } from "./tune.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      keys: { type: "string" }, budget: { type: "string", default: "240" }, repeats: { type: "string", default: "3" },
      every: { type: "string", default: "20" }, holdout: { type: "string", default: "2048" },
      "start-runs": { type: "string", default: "5" }, sigma: { type: "string", default: "0.25" },
      lambda: { type: "string", default: "0" }, out: { type: "string" },
    },
  });
  const budget = Number(cli.budget) | 0, repeats = Number(cli.repeats) | 0, every = Number(cli.every) | 0;
  const keys = cli.keys ? cli.keys.split(",") : LR_KEYS;
  const preset = await loadParams(cli.preset, cli.set);
  const { args: args0, config } = await argsFor(preset);
  for (const k of keys) preset[k] ??= args0[k];
  const maxDim = Number(config.MAX_DIM || 0);
  const images = await loadImages(cli, maxDim);
  const held = await loadImages({ ...cli, from: String((Number(cli.from) | 0) + images.length), count: cli.holdout }, maxDim);

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  console.error(`Compare| ${keys.join(", ")}: ${images.length} tuning images, ${held.length} held out, ` +
                `${budget} batches × ${repeats} repeats`);
  const coordOpts = { ratio: 2, decay: 0.5, minRatio: 1.1, rounds: 8, cycles: 1000, budget };
  const cmaOpts = { sigma: Number(cli.sigma), lambda: Number(cli.lambda) | 0, minRatio: 1.1, budget };

  const runs = [];
  for (let rep = 1; rep <= repeats; rep += 1) {
    for (const method of ["coord", "cma"]) {
      console.error(`\n-- ${method}, repeat ${rep}`);
      const params = { ...preset };
      const ev = await makeEvaluator({ ...ctx, trace: [] }, params, images);
      const quiet = () => {};
      const t0 = performance.now();
      const r = method === "coord"
        ? await tune(ev, params, keys, coordOpts, quiet)
        : await cmaes(ev, params, keys, { ...cmaOpts, seed: rep }, quiet);
      const seconds = (performance.now() - t0) / 1000;
      console.error(`${method} | ${ev.runs} batches in ${seconds.toFixed(0)}s: tuning median ` +
                    `${r.history[0].median.toFixed(3)} → ${r.best.toFixed(3)} dB` +
                    (r.bestSample ? ` (best sample ${r.bestSample.median.toFixed(3)})` : ""));
      runs.push({ method, rep, seconds, history: r.history });
    }
  }

  const heldCache = new Map();
  const scoreHeld = async (p) => {
    const key = JSON.stringify(Object.entries(p).sort());
    if (!heldCache.has(key)) {
      const { args } = await argsFor(p);
      args.logFreq = args.iters;
      const { results } = await trainBatched({ ...ctx, args, quiet: true }, held);
      heldCache.set(key, median(results.map((x) => x.psnr)));
    }
    return heldCache.get(key);
  };
  const startRuns = [];
  for (let i = 0; i < (Number(cli["start-runs"]) | 0); i += 1) {
    const { args } = await argsFor(preset);
    args.logFreq = args.iters;
    const { results } = await trainBatched({ ...ctx, args, quiet: true }, held);
    startRuns.push(median(results.map((x) => x.psnr)));
  }
  const base = median(startRuns);
  console.error(`\nStart  | held out ${base.toFixed(3)} dB (median of ${startRuns.length}: ` +
                `${Math.min(...startRuns).toFixed(3)}…${Math.max(...startRuns).toFixed(3)})`);

  const checkpoints = [];
  for (let c = 0; c <= budget; c += every) checkpoints.push(c);
  if (checkpoints.at(-1) !== budget) checkpoints.push(budget);
  const series = { coord: [], cma: [], "cma best": [] }; // per checkpoint, per repeat: { tune, held }
  for (const run of runs) {
    const at = (c) => run.history.filter((h) => h.runs <= c).at(-1) ?? run.history[0];
    const kinds = run.method === "coord" ? [["coord", "params", "median"]]
                                         : [["cma", "params", "median"], ["cma best", "bestParams", "bestMedian"]];
    for (const [name, pk, mk] of kinds) {
      for (const [i, c] of checkpoints.entries()) {
        const h = at(c);
        (series[name][i] ??= []).push({ tune: h[mk] - run.history[0].median, held: (await scoreHeld(h[pk])) - base });
      }
    }
  }

  const fmt = (xs) => {
    const m = Number((xs.reduce((s, x) => s + x, 0) / xs.length).toFixed(3)) || 0; // no "-0.000"
    return `${m >= 0 ? "+" : ""}${m.toFixed(3)} (${Math.min(...xs).toFixed(2)}…${Math.max(...xs).toFixed(2)})`.padEnd(22);
  };
  console.error(`\nHeld-out gain vs start, mean (min…max) over ${repeats} repeats; [tuning-image gain]`);
  console.error(`batches  ${Object.keys(series).map((n) => n.padEnd(32)).join("")}`);
  for (const [i, c] of checkpoints.entries()) {
    console.error(`${String(c).padStart(7)}  ` + Object.values(series).map((s) =>
      fmt(s[i].map((x) => x.held)) + `[${fmt(s[i].map((x) => x.tune)).trim().split(" ")[0]}]`.padEnd(10)).join(""));
  }

  const outPath = cli.out || resolve(HERE, `../results/tunecmp_${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({
    dataset: cli.mnist || cli.parquet || cli.images, count: images.length, holdout: held.length, keys, preset,
    budget, repeats, coordOpts, cmaOpts, startRuns, base, checkpoints, series,
    runs: runs.map((r) => ({ ...r, final: r.history.at(-1).params })),
  }, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
