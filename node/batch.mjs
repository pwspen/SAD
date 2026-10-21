#!/usr/bin/env node
// Batched training on Dawn (trainer/batch.js): many images with one fixed site
// count in a single run, one tile per image. Writes one JSON line per image
// with its final PSNR and sites (its own pixel coordinates), e.g. as training
// data for a model that predicts sites from images.
//
//   node batch.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 0 --count 1000 \
//     --preset ../presets/mnist.json --set startSites=8 --set endSites=8
//   node batch.mjs --images ../../comp/data/kodak --count 8 --preset ../presets/kodak.json \
//     --set startSites=1000 --set endSites=1000 --compare 2
//
// --compare <jobs> also trains every image alone (that many at once, as
// sweep.mjs --jobs) and reports both runs' speed and PSNR.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { Trainer } from "../trainer/trainer.js?v=ycocg-20261020";
import { planBatches } from "../trainer/batch.js?v=ycocg-20261020";
import { computePsnr } from "../trainer/io.js?v=20261019";
import { whiteMaskRgba32Float } from "../trainer/textures.js?v=20261019";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, mean } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

const HELP = `Usage: node batch.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir of PNGs>) [options]

  --parquet <file>   Hugging Face parquet with PNG bytes (CIFAR-10's test.parquet).

  --from <n>         First image (default 0).
  --count <n>        Images (default 64).
  --preset <json>    presets/<name>.json ({ args }); --set overrides it.
  --set key=value    Trainer arg (camelCase) or training_config.json key. Repeatable.
                     The site count must be fixed: startSites = endSites, no turnover.
  --chunk <n>        At most this many images per atlas (default: as many as fit;
                     images of different sizes go to different atlases).
  --compare <jobs>   Also train each image alone, <jobs> at once, and compare.
  --out <jsonl>      Per-image results (default results/batch_<time>.jsonl).
`;

async function trainAlone(ctx, images, jobs) {
  const results = new Array(images.length);
  let next = 0;
  const t0 = performance.now();
  const worker = async () => {
    while (next < images.length) {
      const i = next++;
      const img = images[i];
      const trainer = new Trainer(ctx.device, ctx.shaders, ctx.args);
      try {
        await trainer.prepare({
          width: img.width, height: img.height, targetRgba32f: img.data,
          maskRgba32f: whiteMaskRgba32Float(img.width, img.height).rgba, maskSum: img.width * img.height,
          initSites: null, initSiteCount: ctx.args.startSites,
        });
        await trainer.run();
        const final = await trainer.finalize();
        results[i] = { name: img.name, psnr: computePsnr(final.renderRgba, img.data, img.width, img.height) };
      } finally {
        trainer.destroy();
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(jobs, images.length) }, worker));
  return { results, seconds: (performance.now() - t0) / 1000 };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      chunk: { type: "string" }, compare: { type: "string" }, out: { type: "string" },
      verbose: { type: "boolean", default: false }, help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  let batches = planBatches(images, args.candDownscale);
  const chunk = Number(cli.chunk);
  if (chunk > 0) batches = batches.flatMap((b) => Array.from({ length: Math.ceil(b.length / chunk) }, (_, k) => b.slice(k * chunk, (k + 1) * chunk)));

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders, args, verbose: cli.verbose };
  console.error(`Device | ${label}`);
  console.error(`Images | ${images.length} in ${batches.length} atlas(es), ${args.startSites} sites, ${args.iters} iterations`);

  const batched = await trainBatched(ctx, images, { batches });
  const its = (n, sec) => (n * args.iters / Math.max(sec, 1e-6)).toFixed(0);
  console.error(`Batched | ${batched.seconds.toFixed(1)}s, ${its(images.length, batched.seconds)} it/s over all ` +
                `images | mean PSNR ${mean(batched.results.map((r) => r.psnr)).toFixed(2)} dB`);

  if (cli.compare) {
    const jobs = Math.max(1, Number(cli.compare) | 0);
    const alone = await trainAlone(ctx, images, jobs);
    const diffs = alone.results.map((r, i) => batched.results[i].psnr - r.psnr);
    console.error(`Alone   | ${alone.seconds.toFixed(1)}s, ${its(images.length, alone.seconds)} it/s over all ` +
                  `images (${jobs} at once) | mean PSNR ${mean(alone.results.map((r) => r.psnr)).toFixed(2)} dB`);
    console.error(`Compare | batched ${(alone.seconds / batched.seconds).toFixed(1)}× faster | PSNR batched − alone: ` +
                  `mean ${mean(diffs).toFixed(2)} dB, range ${Math.min(...diffs).toFixed(2)} to ${Math.max(...diffs).toFixed(2)}`);
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/batch_${stamp}.jsonl`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, batched.results.map((r) => JSON.stringify({
    name: r.name, width: r.width, height: r.height, psnr: r.psnr, activeSites: r.activeSites,
    layer: r.layer, sites: Array.from(r.sites),
  })).join("\n") + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
