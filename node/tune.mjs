#!/usr/bin/env node
// Headless auto-tune in batch mode: train.html's tune buttons without the
// browser, with a step that decays instead of stopping.
//
//   node tune.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 \
//     --preset ../presets/mnist.json --out ../results/tune_mnist.json
//   node tune.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
//     --preset ../presets/mnist.json --keys lrMult
//
// Every evaluation is one batched run over the same images, scored by their
// median final PSNR, so a round's runs are a small population around the
// current point. Each setting is searched in log space, one at a time, for
// --cycles passes over --keys:
//   - try center × r and center ÷ r (values already run aren't re-run);
//   - fit a parabola through the three medians and, when it opens down with
//     its peak inside (−2, 2) steps and not on a point already run, also try
//     that peak (a Newton step from the population);
//   - move to the best point measured; if that is still the center, decay the
//     step, r ← r^--decay (√r by default), rather than stopping;
//   - the setting is done when r falls below --min-ratio or after --rounds.
// Each setting keeps its step between cycles, so later cycles search finer.
// Afterwards the start and tuned settings both run on held-out images.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const LR_KEYS = ["lrPosBase", "lrTauBase", "lrRadiusBase", "lrColorBase", "lrDirBase", "lrAnisoBase"];

const HELP = `Usage: node tune.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [options]

  --from <n> --count <n>  Tuning images (default 0, 64).
  --preset <json> --set key=value   Start settings, as batch.mjs.
  --keys a,b,…        Settings to tune (default the six LR bases: ${LR_KEYS.join(",")}).
  --ratio <r>         Starting step (default 2).
  --decay <p>         Step decay when no neighbour helps: r ← r^p (default 0.5).
  --min-ratio <r>     A setting is done below this step (default 1.1).
  --rounds <n>        At most this many rounds per setting per cycle (default 8).
  --cycles <n>        Passes over --keys (default 2).
  --holdout <n>       Held-out images after the tuning ones for the final check (default --count; 0 = none).
  --out <json>        Result (default results/tune_<time>.json): start and tuned settings, every evaluation.
`;

// Settings are stored to 4 significant digits, as train.html's tune does.
const round4 = (v) => Number(v.toPrecision(4));

export async function makeEvaluator(ctx, params, images) {
  const cache = new Map();
  let runs = 0;
  return {
    get runs() { return runs; },
    async score(p) {
      const key = JSON.stringify(Object.entries(p).sort());
      if (cache.has(key)) return cache.get(key);
      const { args } = await argsFor(p);
      args.logFreq = args.iters; // log only the last iteration: readbacks cost more than these steps
      const { results, seconds } = await trainBatched({ ...ctx, args, quiet: true }, images);
      runs += 1;
      const m = median(results.map((r) => r.psnr));
      cache.set(key, m);
      ctx.trace.push({ params: { ...p }, median: m, seconds });
      return m;
    },
  };
}

// Tunes `keys` of `params` in place (see the header). Returns per-key logs.
export async function tune(evaluator, params, keys, opts, log = console.error) {
  const ratio = new Map(keys.map((k) => [k, opts.ratio]));
  let best = await evaluator.score(params);
  log(`Start  | median ${best.toFixed(3)} dB`);
  const changes = [];
  for (let cycle = 1; cycle <= opts.cycles; cycle += 1) {
    for (const key of keys) {
      const start = params[key], startMedian = best;
      if (!(start > 0)) { log(`Skip   | ${key} needs a positive value`); continue; }
      for (let round = 1; round <= opts.rounds && ratio.get(key) >= opts.minRatio; round += 1) {
        const r = ratio.get(key), center = params[key], centerMedian = best;
        const at = (x) => round4(center * r ** x);
        const score = (v) => evaluator.score({ ...params, [key]: v });
        const pts = [{ x: 0, v: center, m: centerMedian }];
        for (const x of [1, -1]) {
          const v = at(x);
          if (v !== center) pts.push({ x, v, m: await score(v) });
        }
        // Parabola through (−1, m−), (0, m0), (1, m+): peak at −b / 2a when a < 0.
        const up = pts.find((p) => p.x === 1), down = pts.find((p) => p.x === -1);
        if (up && down) {
          const a = (up.m + down.m) / 2 - centerMedian, b = (up.m - down.m) / 2;
          const xs = a < 0 ? -b / (2 * a) : NaN;
          if (Math.abs(xs) < 2 && pts.every((p) => Math.abs(p.x - xs) > 0.25)) {
            const v = at(xs);
            if (pts.every((p) => p.v !== v)) pts.push({ x: xs, v, m: await score(v), newton: true });
          }
        }
        const top = pts.reduce((p, q) => (q.m > p.m ? q : p));
        const moved = top.v !== center;
        if (moved) { params[key] = top.v; best = top.m; } else ratio.set(key, r ** opts.decay);
        log(`${key.padEnd(13)} c${cycle} r${round} ×${r.toFixed(3)} | ` +
            pts.map((p) => `${p.v}${p.newton ? "*" : ""} ${p.m.toFixed(3)}`).join(", ") +
            ` | ${moved ? `→ ${top.v}` : `stay, step → ×${ratio.get(key).toFixed(3)}`}`);
      }
      changes.push({ cycle, key, from: start, to: params[key], gain: best - startMedian, step: ratio.get(key) });
    }
  }
  return { best, changes };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      keys: { type: "string" }, ratio: { type: "string", default: "2" }, decay: { type: "string", default: "0.5" },
      "min-ratio": { type: "string", default: "1.1" }, rounds: { type: "string", default: "8" },
      cycles: { type: "string", default: "2" }, holdout: { type: "string" }, out: { type: "string" },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const opts = {
    ratio: Number(cli.ratio), decay: Number(cli.decay), minRatio: Number(cli["min-ratio"]),
    rounds: Number(cli.rounds) | 0, cycles: Number(cli.cycles) | 0,
  };
  if (!(opts.ratio > 1) || !(opts.decay > 0 && opts.decay < 1) || !(opts.minRatio > 1)) {
    throw new Error("need --ratio > 1, 0 < --decay < 1, --min-ratio > 1");
  }
  const keys = cli.keys ? cli.keys.split(",") : LR_KEYS;
  const params = await loadParams(cli.preset, cli.set);
  const { args: args0, config } = await argsFor(params);
  for (const k of keys) params[k] ??= args0[k]; // tune from buildArgs' default when the preset doesn't set it
  const startParams = { ...params };
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders, trace: [] };
  console.error(`Device | ${label}`);
  console.error(`Tune   | ${keys.join(", ")} on ${images.length} images, ${args0.sites ?? args0.startSites} sites, ` +
                `${args0.iters} iterations; step ×${opts.ratio}, decay ^${opts.decay}, until ×${opts.minRatio}`);
  const t0 = performance.now();
  const ev = await makeEvaluator(ctx, params, images);
  const { best, changes } = await tune(ev, params, keys, opts);
  const seconds = (performance.now() - t0) / 1000;
  console.error(`Tuned  | ${ev.runs} batches in ${seconds.toFixed(0)}s, median ${ctx.trace[0].median.toFixed(3)} → ${best.toFixed(3)} dB`);
  for (const k of keys) console.error(`         ${k.padEnd(13)} ${startParams[k]} → ${params[k]}`);

  let holdout = null;
  const nHold = Number(cli.holdout ?? cli.count) | 0;
  if (nHold > 0) {
    const held = await loadImages({ ...cli, from: String((Number(cli.from) | 0) + images.length), count: String(nHold) },
                                  Number(config.MAX_DIM || 0));
    const hev = await makeEvaluator({ ...ctx, trace: [] }, params, held);
    holdout = { images: held.length, start: await hev.score(startParams), tuned: await hev.score(params) };
    console.error(`Holdout| ${held.length} images: start ${holdout.start.toFixed(3)} → tuned ${holdout.tuned.toFixed(3)} dB ` +
                  `(${holdout.tuned >= holdout.start ? "+" : ""}${(holdout.tuned - holdout.start).toFixed(3)})`);
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/tune_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({
    dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from) | 0, count: images.length,
    keys, opts, start: startParams, tuned: params, startMedian: ctx.trace[0].median, tunedMedian: best,
    holdout, changes, seconds, trace: ctx.trace,
  }, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
  });
}
