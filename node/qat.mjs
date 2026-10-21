#!/usr/bin/env node
// Quantization-aware training (args.quantNoise, shaders/wgsl/qnoise.wgsl)
// against post-hoc quantization and the rate term, by rate–distortion after
// the real codec, in batch mode.
//
//   node qat.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --db 0.25,0.5,1,2 --lambdas 0.01,0.03 --repeats 3
//
// Arms: (a) "post-hoc": trained plainly, quantized at every --db; (b)
// "noise": quantNoise at each --db's steps (rateLossDb = db), quantized at
// that db; (c) "rate λ": the rate term at rateLossDb = db; (d) "rate λ +
// noise": both. Each trained batch's sites (the histogram layer left out, as
// rate.mjs codes them) go through the codec as one stream at codecSteps' steps
// for the db (from the final sites' Gauss–Newton diagonal and squared error),
// colors along the rate basis. The decoded sites (positions clamped to their
// tile, the layer kept as trained) are rendered as a decoder would (candidate
// lists rebuilt) and scored per image. Reported: median decoded PSNR over
// images and measured codec bits per pixel (all bits / all pixels), medians
// over repeats, plus the unquantized PSNR (lists rebuilt the same way).
// --arms picks a subset (a,b,c,d).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=w2b-20261021";
import { codecSteps, encodeSites } from "../trainer/codec.js?v=w2b-20261021";
import { encodeSiteColors } from "../trainer/color_spaces.js?v=w2b-20261021";
import { readSites } from "../trainer/buffers.js?v=20261019";
import { tilePsnrs, tileOrigin } from "../trainer/batch.js?v=w2b-20261021";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// Per image decoded PSNR and the batch's codec bits at each db.
async function evaluate(bt, dbs) {
  const t = bt.trainer, n = t.siteCount, layout = bt.layout;
  const raw = await readSites(t.device, t.sitesBuf, n);
  const sites = await t._readSitesRgb(n);
  const { F } = await t.siteSensitivity();
  const render = async () => {
    t._rebuildCandidates(n);
    return t.renderView("render");
  };
  const clean = await render();
  let sse = 0;
  for (let i = 0; i < t.width * t.height; i += 1) {
    if (t.maskCpu && !(t.maskCpu[i] > 0)) continue;
    for (let c = 0; c < 3; c += 1) sse += (clean[i * 4 + c] - t.target[i * 4 + c]) ** 2;
  }
  const out = { clean: Array.from(tilePsnrs(clean, bt.images, layout)), dbs: {} };
  const coded = Float32Array.from(sites);
  for (let i = 0; i < n; i += 1) if (t._isLayer(i)) coded[i * SITE_FLOATS] = -1;
  for (const db of dbs) {
    const steps = codecSteps(coded, n, F, { lossDb: db, sse0: sse });
    const enc = encodeSites(coded, n, t.width, t.height, steps, { colorBasis: t.rateBasis });
    const dec = Float32Array.from(sites);
    enc.slots.forEach((slot, k) => {
      const src = enc.decoded.subarray(k * SITE_FLOATS, (k + 1) * SITE_FLOATS);
      const b = slot * SITE_FLOATS;
      dec.set(src, b);
      const [ox, oy] = tileOrigin(layout, Math.floor(slot / layout.sitesPerTile));
      dec[b] = Math.min(ox + layout.tileW - 1, Math.max(ox, dec[b]));
      dec[b + 1] = Math.min(oy + layout.tileH - 1, Math.max(oy, dec[b + 1]));
    });
    t.device.queue.writeBuffer(t.sitesBuf, 0, encodeSiteColors(dec, n, t.colorSpace));
    const psnrs = tilePsnrs(await render(), bt.images, layout);
    out.dbs[db] = { psnrs: Array.from(psnrs), bits: enc.bits, sites: enc.n, steps };
  }
  t.device.queue.writeBuffer(t.sitesBuf, 0, raw);
  await render();
  return out;
}

async function run(ctx, params, images, dbs) {
  const { args } = await argsFor(params);
  args.logFreq = args.iters;
  const parts = [];
  const { results } = await trainBatched({ ...ctx, args, quiet: true }, images, {
    after: async (bt, idx) => parts.push({ idx, ev: await evaluate(bt, dbs) }),
  });
  const pixels = images.reduce((s, im) => s + im.width * im.height, 0);
  const clean = new Array(images.length);
  const per = Object.fromEntries(dbs.map((db) => [db, { psnrs: new Array(images.length), bits: 0, sites: 0 }]));
  for (const { idx, ev } of parts) {
    idx.forEach((im, k) => { clean[im] = ev.clean[k]; });
    for (const db of dbs) {
      idx.forEach((im, k) => { per[db].psnrs[im] = ev.dbs[db].psnrs[k]; });
      per[db].bits += ev.dbs[db].bits;
      per[db].sites += ev.dbs[db].sites;
    }
  }
  return {
    train: median(results.map((r) => r.psnr)), clean: median(clean),
    dbs: Object.fromEntries(dbs.map((db) => [db, { psnr: median(per[db].psnrs), bpp: per[db].bits / pixels,
                                                   bitsPerSite: per[db].bits / per[db].sites, psnrs: per[db].psnrs }])),
  };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, db: { type: "string", default: "0.25,0.5,1,2" }, lambdas: { type: "string", default: "0.01,0.03" },
      arms: { type: "string", default: "a,b,c,d" }, noise: { type: "string", default: "1" },
      repeats: { type: "string", default: "1" }, out: { type: "string" },
    },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label} | ${images.length} images from ${images[0].name}`);
  const dbs = cli.db.split(",").map(Number);
  const lambdas = cli.lambdas.split(",").filter(Boolean).map(Number);
  const arms = new Set(cli.arms.split(","));
  const noise = Number(cli.noise);
  const configs = [];
  if (arms.has("a")) configs.push({ arm: "a", name: "post-hoc", params: { ...base, rateLambda: 0, quantNoise: 0 }, dbs });
  for (const db of dbs) {
    if (arms.has("b")) configs.push({ arm: "b", name: "noise", params: { ...base, rateLambda: 0, quantNoise: noise, rateLossDb: db }, dbs: [db] });
    for (const l of lambdas) {
      if (arms.has("c")) configs.push({ arm: "c", name: `rate ${l}`, params: { ...base, rateLambda: l, quantNoise: 0, rateLossDb: db }, dbs: [db] });
      if (arms.has("d")) configs.push({ arm: "d", name: `rate ${l} + noise`, params: { ...base, rateLambda: l, quantNoise: noise, rateLossDb: db }, dbs: [db] });
    }
  }
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  const rows = [];
  for (const c of configs) {
    const reps = [];
    for (let k = 0; k < repeats; k += 1) reps.push(await run(ctx, c.params, images, c.dbs));
    for (const db of c.dbs) {
      const m = (f) => median(reps.map(f));
      const r = { arm: c.arm, name: c.name, db, psnr: m((x) => x.dbs[db].psnr), bpp: m((x) => x.dbs[db].bpp),
                  bitsPerSite: m((x) => x.dbs[db].bitsPerSite), clean: m((x) => x.clean), train: m((x) => x.train),
                  repPsnr: reps.map((x) => x.dbs[db].psnr), repBpp: reps.map((x) => x.dbs[db].bpp), params: c.params };
      rows.push(r);
      console.error(`${c.name.padEnd(18)} | ${db} dB | decoded ${r.psnr.toFixed(3)} dB at ${r.bpp.toFixed(4)} bpp ` +
                    `(${r.bitsPerSite.toFixed(2)} bits/site) | clean ${r.clean.toFixed(3)} | reps ${r.repPsnr.map((v) => v.toFixed(3)).join(" ")}`);
    }
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/qat_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: cli.from, count: images.length, rows }, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
