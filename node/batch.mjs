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

import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { gunzipSync } from "node:zlib";

import { Trainer, buildArgs } from "../trainer/trainer.js?v=20261019";
import { BatchTrainer, planBatches } from "../trainer/batch.js?v=20261019";
import { computePsnr } from "../trainer/io.js?v=20261019";
import { whiteMaskRgba32Float } from "../trainer/textures.js?v=20261019";
import { createHeadlessDevice, loadShadersFromDisk, loadJson, loadPngToFloat32 } from "./headless.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

const HELP = `Usage: node batch.mjs (--mnist <idx3.gz> | --images <dir of PNGs>) [options]

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

function parseValue(raw) {
  const s = String(raw).trim();
  if (s === "true") return true;
  if (s === "false") return false;
  if (/^[A-Za-z][A-Za-z0-9_-]*$/.test(s)) return s;
  const n = Number(s);
  if (s === "" || Number.isNaN(n)) throw new Error(`Not a number: "${raw}"`);
  return n;
}

// Test-split MNIST digits [from, from + count) as rgba32f images.
async function loadMnist(path, from, count) {
  const bytes = gunzipSync(await readFile(path));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const total = view.getUint32(4), side = view.getUint32(8);
  const n = side * side;
  const images = [];
  for (let i = from; i < Math.min(total, from + count); i += 1) {
    const data = new Float32Array(n * 4);
    for (let p = 0; p < n; p += 1) {
      const v = bytes[16 + i * n + p] / 255;
      data.set([v, v, v, 1], p * 4);
    }
    images.push({ name: `mnist${String(i).padStart(4, "0")}`, data, width: side, height: side });
  }
  return images;
}

async function loadPngDir(dir, from, count, maxDim) {
  const names = (await readdir(dir)).filter((f) => f.toLowerCase().endsWith(".png")).sort().slice(from, from + count);
  return Promise.all(names.map(async (f) => ({ name: basename(f, ".png"), ...(await loadPngToFloat32(join(dir, f), maxDim)) })));
}

async function trainBatched(ctx, images, batches) {
  const results = new Array(images.length);
  const t0 = performance.now();
  for (const idx of batches) {
    const part = idx.map((i) => images[i]);
    const bt = new BatchTrainer(ctx.device, ctx.shaders, ctx.args, {
      onLog: (info) => { if (info.type === "msg" && ctx.verbose) console.error(`  ${info.message}`); },
    });
    try {
      await bt.prepare(part);
      const l = bt.layout;
      console.error(`Batch  | ${part.length} image(s) from ${part[0].name}: atlas ${l.width}×${l.height}, ` +
                    `${l.tilesX}×${l.tilesY} tiles of ${l.tileW}×${l.tileH}, ${l.sitesPerTile} sites each`);
      await bt.run();
      const final = await bt.finalize();
      final.forEach((r, i) => { results[idx[i]] = { name: part[i].name, ...r }; });
    } finally {
      bt.destroy();
    }
  }
  return { results, seconds: (performance.now() - t0) / 1000 };
}

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

const mean = (xs) => xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length);

async function main() {
  const { values: cli } = parseArgs({
    options: {
      mnist: { type: "string" }, images: { type: "string" },
      from: { type: "string", default: "0" }, count: { type: "string", default: "64" },
      preset: { type: "string" }, set: { type: "string", multiple: true, default: [] },
      chunk: { type: "string" }, compare: { type: "string" }, out: { type: "string" },
      verbose: { type: "boolean", default: false }, help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const params = cli.preset ? { ...(await loadJson(cli.preset)).args } : {};
  for (const spec of cli.set) {
    const eq = spec.indexOf("=");
    if (eq <= 0) throw new Error(`--set expects key=value (got "${spec}")`);
    params[spec.slice(0, eq)] = parseValue(spec.slice(eq + 1));
  }
  const configPatch = {}, overrides = {};
  for (const [k, v] of Object.entries(params)) (/^[A-Z0-9_]+$/.test(k) ? configPatch : overrides)[k] = v;
  const config = { ...(await loadJson(resolve(HERE, "../training_config.json"))), ...configPatch };
  const args = buildArgs(config, overrides);
  args.viewerFreq = args.iters + 1; // no previews

  const from = Number(cli.from) | 0, count = Number(cli.count) | 0;
  const images = cli.mnist
    ? await loadMnist(cli.mnist, from, count)
    : await loadPngDir(cli.images, from, count, Number(config.MAX_DIM || 0));
  let batches = planBatches(images, args.candDownscale);
  const chunk = Number(cli.chunk);
  if (chunk > 0) batches = batches.flatMap((b) => Array.from({ length: Math.ceil(b.length / chunk) }, (_, k) => b.slice(k * chunk, (k + 1) * chunk)));

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders, args, verbose: cli.verbose };
  console.error(`Device | ${label}`);
  console.error(`Images | ${images.length} in ${batches.length} atlas(es), ${args.startSites} sites, ${args.iters} iterations`);

  const batched = await trainBatched(ctx, images, batches);
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
