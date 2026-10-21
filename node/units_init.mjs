#!/usr/bin/env node
// LR bases for a new dataset from one probe run: base_g = K · Δ_g / lrMult, so
// every LR group's effective rate (base × lrMult) is one constant K times its
// unit Δ_g, the codec quantizer step for that group measured on this dataset
// (probe.mjs e1Units). E1 (e1.mjs) found K · Δ transfers across datasets far
// better than per-group tuned LRs: MNIST's six tuned bases on CIFAR-10 lost
// 0.32 dB against CIFAR's own, CIFAR's on MNIST 2.9 dB; K · Δ with the other
// dataset's K was within 0.05 / 0.12 dB of K · Δ with its own.
//
//   # apply: probe CIFAR-10 at the MNIST preset, set the bases, tune lrMult alone, check on held-out images
//   node units_init.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
//     --preset ../presets/mnist.json --tune-mult --holdout 2048 --compare ../presets/cifar10.json
//   # fit: K from LRs already tuned on this dataset (--tuned, default --preset), Δ probed at --preset:
//   # K₀ = geometric mean of lr / Δ over the groups, then lrMult tuned on K₀ · Δ
//   node units_init.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
//     --tuned ../presets/cifar10.json --preset ../presets/mnist.json
//
// Δ depends on where it is measured (the final error and the Gauss–Newton
// diagonal of the probe run): K only carries over between datasets when Δ is
// probed at the same settings on both, and settings that fit the new data
// badly give poor units (CIFAR's preset on MNIST: −2.8 dB, and Δ_dir 6× too
// small). --passes > 1 re-probes at the new bases (a fixed-point iteration; it
// did not help on MNIST).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";
import { makeEvaluator, tune, LR_KEYS } from "./tune.mjs";
import { probe, LR_GROUPS } from "./probe.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// K fitted on each dataset's tuned preset with Δ probed at presets/mnist.json (test images
// 0–1023, lrMult tuned on K₀ · Δ): `--fit --preset ../presets/mnist.json` on MNIST, `--tuned
// ../presets/cifar10.json --preset ../presets/mnist.json` on CIFAR-10. The default is their
// geometric mean. K depends on where Δ is probed, so probe at mnist.json-like settings. See node/README.md.
export const K_FIT = { mnist: 0.02046, cifar10: 0.01126 };
export const K_DEFAULT = Number(Math.sqrt(K_FIT.mnist * K_FIT.cifar10).toPrecision(4));

const HELP = `Usage: node units_init.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [options]

  --from <n> --count <n>  Probe / tuning images (default 0, 64).
  --preset <json> --set key=value   Settings to probe at (and to keep, apart from the six LR bases).
  --k <K>             Effective LR per unit Δ (default ${K_DEFAULT}, the geometric mean of ${JSON.stringify(K_FIT)}).
  --passes <n>        Probe, set the bases, repeat n times (default 1).
  --tune-mult         Then tune lrMult alone (tune.mjs's search: step ×2 decaying to ×1.1).
  --fit               Instead: take --preset as tuned for this dataset and print its K (see the header).
  --tuned <json>      As --fit, with the tuned LRs from this preset and Δ probed at --preset.
  --holdout <n>       Score on the n images after the probe ones (default 0 = none), --repeats times (default 3):
                      the preset, K · Δ, K · Δ with lrMult tuned, and each --compare preset.
  --compare a.json,b.json   Presets to score on the held-out images too (e.g. one fully tuned on this dataset).
  --out <json>        Result (default results/units_init_<time>.json), with the settings as a preset ({ args }).
`;

const round4 = (v) => Number(v.toPrecision(4));

// Δ per LR group at `params`, from one probe batch.
async function measureUnits(ctx, params, images) {
  const { args } = await argsFor(params);
  const rep = await probe(ctx, args, images, { e2: false, e3: false, audit: false });
  return { delta: Object.fromEntries(LR_GROUPS.map((g) => [g.key, rep.e1.units[g.key].delta])), median: rep.medianPsnr };
}

// `params` with every base set to K · Δ_g / lrMult (lrMult kept).
export function unitBases(params, delta, K) {
  const p = { ...params };
  for (const g of LR_GROUPS) p[g.lr] = round4((K * delta[g.key]) / p.lrMult);
  return p;
}

const effective = (p) => LR_GROUPS.map((g) => `${g.key} ${round4(p[g.lr] * p.lrMult)}`).join(", ");

async function holdoutScore(ctx, params, held, repeats) {
  const { args } = await argsFor(params);
  args.logFreq = args.iters;
  const runs = [];
  for (let k = 0; k < repeats; k += 1) {
    const { results } = await trainBatched({ ...ctx, args, quiet: true }, held);
    runs.push(median(results.map((x) => x.psnr)));
  }
  return { median: median(runs), runs };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      k: { type: "string" }, passes: { type: "string", default: "1" }, "tune-mult": { type: "boolean", default: false },
      fit: { type: "boolean", default: false }, tuned: { type: "string" }, holdout: { type: "string", default: "0" },
      repeats: { type: "string", default: "3" }, compare: { type: "string" }, out: { type: "string" },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const tuneOpts = { ratio: 2, decay: 0.5, minRatio: 1.1, rounds: 8, cycles: 1 };
  const preset = await loadParams(cli.preset, cli.set);
  const { args: args0, config } = await argsFor(preset);
  for (const k of [...LR_KEYS, "lrMult"]) preset[k] ??= args0[k];
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  const out = { dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from) | 0, count: images.length,
                preset: cli.preset ?? null, start: preset };
  const t0 = performance.now();

  if (cli.fit || cli.tuned) {
    // K₀: geometric mean over the groups of the tuned effective LR / Δ measured at --preset.
    const tuned = cli.tuned ? await loadParams(cli.tuned, cli.set) : preset;
    for (const k of [...LR_KEYS, "lrMult"]) tuned[k] ??= args0[k];
    const u = await measureUnits(ctx, preset, images);
    const K0 = Math.exp(mean(LR_GROUPS.map((g) => Math.log((tuned[g.lr] * tuned.lrMult) / u.delta[g.key]))));
    console.error(`Probe  | ${images.length} images, median ${u.median.toFixed(3)} dB; Δ ${LR_GROUPS.map((g) => `${g.key} ${u.delta[g.key].toPrecision(4)}`).join(", ")}`);
    console.error(`         lr / Δ: ${LR_GROUPS.map((g) => `${g.key} ${((tuned[g.lr] * tuned.lrMult) / u.delta[g.key]).toPrecision(3)}`).join(", ")}; K₀ ${K0.toPrecision(4)}`);
    const p = unitBases(tuned, u.delta, K0);
    const t = await tune(await makeEvaluator({ ...ctx, trace: [] }, p, images), p, ["lrMult"], tuneOpts);
    const K = K0 * p.lrMult / tuned.lrMult;
    console.error(`Fit    | K = ${K.toPrecision(4)} (K₀ ${K0.toPrecision(4)} × lrMult ${tuned.lrMult} → ${p.lrMult}); ` +
                  `K · Δ tuning-set median ${t.best.toFixed(3)} dB`);
    Object.assign(out, { mode: "fit", tuned: cli.tuned ?? cli.preset ?? null, delta: u.delta, K0, K, kMedian: t.best, settings: p });
  } else {
    const K = cli.k !== undefined ? Number(cli.k) : K_DEFAULT;
    if (!(K > 0)) throw new Error("--k must be positive");
    let p = { ...preset };
    const passes = [];
    for (let pass = 1; pass <= Math.max(1, Number(cli.passes) | 0); pass += 1) {
      const u = await measureUnits(ctx, p, images);
      p = unitBases(p, u.delta, K);
      passes.push({ delta: u.delta, probeMedian: u.median, bases: Object.fromEntries(LR_KEYS.map((k) => [k, p[k]])) });
      console.error(`Pass ${pass} | probe median ${u.median.toFixed(3)} dB; Δ ${LR_GROUPS.map((g) => `${g.key} ${u.delta[g.key].toPrecision(4)}`).join(", ")}`);
      console.error(`         K ${K} → effective LRs ${effective(p)}`);
    }
    const settings = { "K·Δ": p };
    if (cli["tune-mult"]) {
      const q = { ...p };
      const t = await tune(await makeEvaluator({ ...ctx, trace: [] }, q, images), q, ["lrMult"], tuneOpts);
      settings["K·Δ, lrMult tuned"] = q;
      console.error(`Tuned  | lrMult ${p.lrMult} → ${q.lrMult} (K ${K} → ${round4(K * q.lrMult / p.lrMult)}), tuning-set median ${t.best.toFixed(3)} dB`);
    }
    Object.assign(out, { mode: "apply", K, passes, settings });

    const nHold = Number(cli.holdout) | 0;
    if (nHold > 0) {
      const held = await loadImages({ ...cli, from: String(out.from + images.length), count: String(nHold) }, Number(config.MAX_DIM || 0));
      const repeats = Math.max(1, Number(cli.repeats) | 0);
      const all = { preset, ...settings };
      for (const path of cli.compare ? cli.compare.split(",") : []) all[path] = await loadParams(path, cli.set);
      out.holdout = { images: held.length, repeats, scores: {} };
      console.error(`Holdout| ${held.length} images from ${out.from + images.length}, ${repeats} repeats (median of medians)`);
      for (const [name, s] of Object.entries(all)) {
        const r = out.holdout.scores[name] = await holdoutScore(ctx, s, held, repeats);
        const d = r.median - out.holdout.scores.preset.median;
        console.error(`         ${name.padEnd(28)} ${r.median.toFixed(3)} dB (${d >= 0 ? "+" : ""}${d.toFixed(3)} vs preset; ` +
                      `runs ${r.runs.map((x) => x.toFixed(3)).join(", ")})`);
      }
    }
  }
  out.seconds = (performance.now() - t0) / 1000;
  const final = out.settings["K·Δ, lrMult tuned"] ?? out.settings["K·Δ"] ?? out.settings;
  out.presetJson = { _note: `LR bases from node/units_init.mjs (${out.mode}, K ${round4(out.K)}) on ${out.dataset}.`, args: final };
  for (const k of LR_KEYS) console.error(`         ${k.padEnd(13)} ${preset[k]} → ${final[k]}`);
  console.error(`         lrMult        ${preset.lrMult} → ${final.lrMult}`);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/units_init_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(out, null, 1) + "\n");
  console.error(`Wrote  | ${outPath} (${out.seconds.toFixed(0)}s)`);
  device.destroy();
  process.exit(0);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
  });
}
