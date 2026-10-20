#!/usr/bin/env node
// E1: can one constant times a measured unit replace the six tuned LR bases?
// MNIST and CIFAR-10 in batch mode (probe.mjs has the units):
//
//   node e1.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz \
//     --parquet ../../comp/data/cifar10/test.parquet --preset ../presets/mnist.json
//
// Per dataset, on --count tuning images (from 0) and --holdout images after them:
//   1. probe the preset: the units Δ, 1/√F, σ per LR group;
//   2. tune the six LR bases from the preset (tune.mjs), then probe there too;
//   3. for each unit (and "none", one LR for every group): set every base to
//      K · unit, K the geometric mean of tuned LR / unit, then tune K alone
//      (lrMult) — one free number instead of six;
//   4. cross: each dataset's settings with the other's K and its own units,
//      untuned.
// Every setting is then scored on the held-out images, --repeats times
// (runs aren't bit-reproducible), and the median of those medians reported.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { loadParams, argsFor, loadMnist, loadParquet, trainBatched, median, mean } from "./common.mjs";
import { makeEvaluator, tune, LR_KEYS } from "./tune.mjs";
import { probe, printReport, LR_GROUPS, collapse } from "./probe.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const UNITS = ["none", "delta", "invSqrtF", "sigma"];

async function main() {
  const { values: cli } = parseArgs({
    options: {
      mnist: { type: "string" }, parquet: { type: "string" }, preset: { type: "string" },
      set: { type: "string", multiple: true, default: [] },
      count: { type: "string", default: "1024" }, holdout: { type: "string", default: "2048" },
      repeats: { type: "string", default: "3" }, cycles: { type: "string", default: "2" },
      out: { type: "string" },
    },
  });
  const count = Number(cli.count) | 0, nHold = Number(cli.holdout) | 0, repeats = Number(cli.repeats) | 0;
  const datasets = [];
  if (cli.mnist) datasets.push({ name: "mnist", load: (from, n) => loadMnist(cli.mnist, from, n) });
  if (cli.parquet) datasets.push({ name: "cifar10", load: (from, n) => loadParquet(cli.parquet, from, n) });
  const preset = await loadParams(cli.preset, cli.set);
  const { args: args0 } = await argsFor(preset);
  for (const k of [...LR_KEYS, "lrMult"]) preset[k] ??= args0[k];

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  const tuneOpts = { ratio: 2, decay: 0.5, minRatio: 1.1, rounds: 8, cycles: Number(cli.cycles) | 0 };
  const out = { preset, tuneOpts, count, holdout: nHold, repeats, datasets: {} };

  for (const ds of datasets) {
    console.error(`\n==== ${ds.name} ====`);
    const images = await ds.load(0, count);
    const r = out.datasets[ds.name] = { settings: {} };
    const { args: presetArgs } = await argsFor(preset);
    r.presetProbe = await probe(ctx, presetArgs, images);
    console.error("-- preset");
    printReport(r.presetProbe);

    console.error("-- tuning the six LR bases");
    const tuned = { ...preset };
    const trace = [];
    const tr = await tune(await makeEvaluator({ ...ctx, trace }, tuned, images), tuned, LR_KEYS, tuneOpts);
    r.tune = { changes: tr.changes, trace, best: tr.best };
    r.settings.preset = preset;
    r.settings.tuned = tuned;
    const { args: tunedArgs } = await argsFor(tuned);
    r.tunedProbe = await probe(ctx, tunedArgs, images, { e2: false, e3: false });
    console.error("-- tuned");
    printReport(r.tunedProbe);

    // One constant per unit, from the preset's units (what a single run gives before any tuning).
    r.K = {};
    for (const unit of UNITS) {
      const u = (g) => (unit === "none" ? 1 : r.presetProbe.e1.units[g.key][unit]);
      const K = Math.exp(mean(LR_GROUPS.map((g) => Math.log(tuned[g.lr] / u(g)))));
      const p = { ...tuned };
      for (const g of LR_GROUPS) p[g.lr] = Number((K * u(g) / tuned.lrMult).toPrecision(4));
      // lrMult scales every base, so K lives in the bases and lrMult starts at the tuned value.
      console.error(`-- unit ${unit}: K = ${K.toPrecision(4)}, tuning lrMult alone`);
      const t = await tune(await makeEvaluator({ ...ctx, trace: [] }, p, images), p, ["lrMult"], { ...tuneOpts, cycles: 1 });
      r.K[unit] = K * p.lrMult / tuned.lrMult;
      r.settings[`K·${unit}`] = p;
      r.settings[`K·${unit} untuned`] = { ...p, lrMult: tuned.lrMult };
      console.error(`         tuning-set median ${t.best.toFixed(3)} dB vs ${tr.best.toFixed(3)} for six tuned bases`);
    }
  }

  // Cross: the other dataset's K with this dataset's own preset units.
  if (datasets.length === 2) {
    for (const [a, b] of [[0, 1], [1, 0]]) {
      const ra = out.datasets[datasets[a].name], rb = out.datasets[datasets[b].name];
      for (const unit of UNITS) {
        const p = { ...ra.settings.tuned, lrMult: 1 };
        for (const g of LR_GROUPS) p[g.lr] = Number((rb.K[unit] * (unit === "none" ? 1 : ra.presetProbe.e1.units[g.key][unit])).toPrecision(4));
        ra.settings[`K·${unit} from ${datasets[b].name}`] = p;
      }
      ra.settings[`six tuned on ${datasets[b].name}`] = rb.settings.tuned;
    }
  }

  // Held-out scoring.
  for (const ds of datasets) {
    const r = out.datasets[ds.name];
    const held = await ds.load(count, nHold);
    r.holdout = {};
    console.error(`\n==== ${ds.name}: ${held.length} held-out images, ${repeats} repeats ====`);
    for (const [name, p] of Object.entries(r.settings)) {
      const { args } = await argsFor(p);
      args.logFreq = args.iters;
      const meds = [];
      for (let k = 0; k < repeats; k += 1) {
        const { results } = await trainBatched({ ...ctx, args, quiet: true }, held);
        meds.push(median(results.map((x) => x.psnr)));
      }
      r.holdout[name] = { median: median(meds), runs: meds };
      console.error(`         ${name.padEnd(28)} ${median(meds).toFixed(3)} dB  (runs ${meds.map((m) => m.toFixed(3)).join(", ")})`);
    }
    const lr = (p) => LR_GROUPS.map((g) => `${g.key} ${p[g.lr] * p.lrMult}`.padEnd(18)).join("");
    console.error(`         effective LRs (base × lrMult):`);
    for (const name of ["preset", "tuned"]) console.error(`           ${name.padEnd(8)} ${lr(r.settings[name])}`);
    const c = (units) => Object.fromEntries(UNITS.map((u) => [u, collapse(units, u === "none" ? null : u).decades.toFixed(2)]));
    console.error(`         spread of log10(tuned lr / preset unit), decades: ${JSON.stringify(c(Object.fromEntries(
      LR_GROUPS.map((g) => [g.key, { ...r.presetProbe.e1.units[g.key], lr: r.settings.tuned[g.lr] }]))))}`);
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/e1_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(out, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
