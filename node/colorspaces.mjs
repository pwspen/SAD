#!/usr/bin/env node
// Training color spaces compared fairly: each gets its own six tuned LR bases
// (the color LR's units differ per space), then is scored on held-out images
// for PSNR and for the codec's measured rate (probe.mjs's audit).
//
//   node colorspaces.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz \
//     --parquet ../../comp/data/cifar10/test.parquet
//
// Per dataset, from presets/<dataset>.json with colorSpace swapped: tune the
// six bases on --count images (from 0), then on --holdout images after them
// run --repeats batches for the median PSNR and probe once for the bits.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { COLOR_SPACES } from "../trainer/color_spaces.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { loadParams, argsFor, loadMnist, loadParquet, trainBatched, median } from "./common.mjs";
import { makeEvaluator, tune, LR_KEYS } from "./tune.mjs";
import { probe } from "./probe.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

async function main() {
  const { values: cli } = parseArgs({
    options: {
      mnist: { type: "string" }, parquet: { type: "string" }, spaces: { type: "string" },
      count: { type: "string", default: "1024" }, holdout: { type: "string", default: "2048" },
      repeats: { type: "string", default: "3" }, out: { type: "string" },
    },
  });
  const count = Number(cli.count) | 0, nHold = Number(cli.holdout) | 0, repeats = Number(cli.repeats) | 0;
  const spaces = cli.spaces ? cli.spaces.split(",") : Object.keys(COLOR_SPACES);
  const datasets = [];
  if (cli.mnist) datasets.push({ name: "mnist", load: (from, n) => loadMnist(cli.mnist, from, n) });
  if (cli.parquet) datasets.push({ name: "cifar10", load: (from, n) => loadParquet(cli.parquet, from, n) });

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  const tuneOpts = { ratio: 2, decay: 0.5, minRatio: 1.1, rounds: 8, cycles: 1 };
  const out = { tuneOpts, count, holdout: nHold, repeats, datasets: {} };

  for (const ds of datasets) {
    const images = await ds.load(0, count), held = await ds.load(count, nHold);
    const preset = await loadParams(resolve(HERE, `../presets/${ds.name}.json`));
    const rows = out.datasets[ds.name] = {};
    for (const space of spaces) {
      console.error(`\n==== ${ds.name}, ${space} ====`);
      const p = { ...preset, colorSpace: space };
      const t = await tune(await makeEvaluator({ ...ctx, trace: [] }, p, images), p, LR_KEYS, tuneOpts);
      const { args } = await argsFor(p);
      args.logFreq = args.iters;
      const meds = [];
      for (let k = 0; k < repeats; k += 1) {
        const { results } = await trainBatched({ ...ctx, args, quiet: true }, held);
        meds.push(median(results.map((x) => x.psnr)));
      }
      const { args: probeArgs } = await argsFor(p);
      const rep = await probe(ctx, probeArgs, held, { e2: false, e3: false });
      rows[space] = {
        params: p, tuningMedian: t.best, holdout: median(meds), runs: meds,
        pooledBits: rep.audit.pooled.perSite, perImageBits: rep.audit.perImage.perSite, staticColorBits: rep.audit.staticBits,
        pooledKltBits: rep.audit.pooledKlt.perSite, perImageKltBits: rep.audit.perImageKlt.perSite,
      };
      console.error(`         held out ${median(meds).toFixed(3)} dB; codec pooled ${rep.audit.pooled.perSite.total.toFixed(2)} ` +
                    `bits/site (color ${rep.audit.pooled.perSite.color.toFixed(2)}), per image ${rep.audit.perImage.perSite.total.toFixed(2)}`);
    }
    console.error(`\n==== ${ds.name}: ${held.length} held-out images ====`);
    console.error(`         space      PSNR (median of ${repeats})  bits/site pooled (color): YCoCg-R codec  KLT codec      per image: YCoCg-R  KLT    color LR`);
    for (const [space, r] of Object.entries(rows)) {
      const b = (x) => `${x.total.toFixed(2)} (${x.color.toFixed(2)})`.padEnd(15);
      console.error(`         ${space.padEnd(10)} ${r.holdout.toFixed(3).padEnd(47)} ${b(r.pooledBits)}${b(r.pooledKltBits)}` +
                    `${r.perImageBits.total.toFixed(2).padEnd(9)}${r.perImageKltBits.total.toFixed(2).padEnd(7)}${r.params.lrColorBase}`);
    }
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/colorspaces_${stamp}.json`);
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
