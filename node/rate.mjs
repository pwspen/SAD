#!/usr/bin/env node
// Rate term (trainer/rate.js): calibration against the real codec, and the
// rate–distortion trade it buys, in batch mode.
//
//   node rate.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --lambdas 0,0.01,0.03,0.1,0.3 --sites 6,8,10
//
// Each run trains the images, then refits the rate model to the final sites
// and codes them with the codec as one stream (every image's sites in the
// atlas, the histogram layer left out; colors along the same KLT basis, at
// the same steps). Calibration: the model's estimated bits per plane vs the
// codec's measured ones. Rate–distortion: median PSNR vs measured bits per
// pixel, for the λ sweep at the preset's site count and for λ = 0 at other
// site counts (the trade the rate term has to beat). --repeats repeats every
// run; the reported PSNR and bits are medians over repeats.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=20261019";
import { encodeSites } from "../trainer/codec.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PLANE_NAMES = ["log_aniso", "θ", "log_tau", "radius", "K0", "K1", "K2"];

async function run(ctx, params, images) {
  const { args } = await argsFor(params);
  args.logFreq = args.iters;
  const out = { estimate: null, codec: null, pixels: 0, sites: 0 };
  const after = async (bt) => {
    const t = bt.trainer, n = t.siteCount;
    await t._rateRefit(n); // fits the model and the steps to the final sites (λ is irrelevant here)
    const est = t.rateLast;
    const sites = await t._readSitesRgb(n);
    for (let i = 0; i < n; i += 1) if (t._isLayer(i)) sites[i * SITE_FLOATS] = -1;
    const enc = encodeSites(sites, n, t.width, t.height, est.steps, { colorBasis: t.rateBasis });
    const add = (o, k, v) => { o[k] = (o[k] ?? 0) + v; };
    out.estimate ??= {};
    out.codec ??= {};
    for (const [k, v] of Object.entries(est.planes)) add(out.estimate, k === "radius" ? "radius" : k, v);
    PLANE_NAMES.forEach((k, p) => add(out.codec, k, enc.planeBits[p]));
    add(out.codec, "position", enc.positionBits);
    add(out.codec, "header", enc.headerBits);
    out.sites += enc.n;
  };
  const { results } = await trainBatched({ ...ctx, args, quiet: true }, images, { after });
  out.pixels = images.reduce((s, im) => s + im.width * im.height, 0);
  out.psnr = median(results.map((r) => r.psnr));
  out.codecTotal = Object.values(out.codec).reduce((s, v) => s + v, 0);
  out.estimateTotal = Object.values(out.estimate).reduce((s, v) => s + v, 0);
  out.bpp = out.codecTotal / out.pixels;
  return out;
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, lambdas: { type: "string", default: "0,0.01,0.03,0.1,0.3" },
      sites: { type: "string", default: "" }, repeats: { type: "string", default: "1" }, out: { type: "string" },
    },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config, args: args0 } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label}`);
  const configs = [
    ...cli.lambdas.split(",").map(Number).map((l) => ({ name: `λ ${l}`, params: { ...base, rateLambda: l } })),
    ...cli.sites.split(",").filter(Boolean).map(Number).filter((n) => n !== args0.sites)
      .map((n) => ({ name: `${n} sites`, params: { ...base, sites: n, rateLambda: 0 } })),
  ];
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  const rows = [];
  for (const c of configs) {
    const reps = [];
    for (let k = 0; k < repeats; k += 1) reps.push(await run(ctx, c.params, images));
    const m = (f) => median(reps.map(f));
    const r = { name: c.name, params: c.params, psnr: m((x) => x.psnr), bpp: m((x) => x.bpp),
                bitsPerSite: m((x) => x.codecTotal / x.sites), estPerSite: m((x) => x.estimateTotal / x.sites), reps };
    rows.push(r);
    const e = reps[0], per = (o, k) => ((o[k] ?? 0) / e.sites).toFixed(2);
    console.error(`${c.name.padEnd(10)} | PSNR ${r.psnr.toFixed(3)} dB, ${r.bpp.toFixed(4)} bpp, ${r.bitsPerSite.toFixed(2)} bits/site ` +
                  `(model ${r.estPerSite.toFixed(2)} for the planes it covers)`);
    console.error(`           | per site, model vs codec: ` +
                  ["log_tau", "radius", "K0", "K1", "K2", "log_aniso", "θ"].map((k) => `${k} ${per(e.estimate, k)}/${per(e.codec, k)}`).join(", ") +
                  ` | codec position ${per(e.codec, "position")}`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/rate_${stamp}.json`);
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
