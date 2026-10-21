#!/usr/bin/env node
// k-dimensional color basis (colorDims, klt only): rate–distortion curves,
// median PSNR vs measured codec bits per pixel, sweeping the site count per
// color configuration, in batch mode.
//
//   node colordims.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --configs klt:1,klt:2,klt:3 --sites 4,6,8,10,12
//
// --configs: comma-separated space:dims[:lrColorBase] (dims only for klt; the
// LR overrides the preset's for that configuration, e.g. tuned per k with
// tune.mjs --set colorDims=k --keys lrColorBase). Each run trains the images,
// refits the codec steps to the final sites (codecSteps at rateLossDb, as
// rate.mjs) and codes every image's sites as one stream (histogram layer left
// out; colors along the run's KLT basis: the training basis for klt, the
// target's KLT otherwise, so a k-axis basis codes its dropped planes as all
// zeros). PSNR is the trained (unquantized) sites', median over images; with
// --repeats, PSNR and bits are medians over repeats. The report lists every
// point, the Pareto frontier over all of them, and for each point the PSNR
// difference to the reference configuration's curve (--ref, default the last
// config) at equal bits per pixel (linear interpolation; blank outside it).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=m3-20261020";
import { encodeSites } from "../trainer/codec.js?v=shared-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PLANE_NAMES = ["log_aniso", "θ", "log_tau", "radius", "K0", "K1", "K2"];

async function run(ctx, params, images) {
  const { args } = await argsFor(params);
  args.logFreq = args.iters;
  const codec = {};
  let sites = 0;
  const after = async (bt) => {
    const t = bt.trainer, n = t.siteCount;
    await t._rateRefit(n); // the codec steps for the final sites
    const s = await t._readSitesRgb(n);
    for (let i = 0; i < n; i += 1) if (t._isLayer(i)) s[i * SITE_FLOATS] = -1;
    const enc = encodeSites(s, n, t.width, t.height, t.rateLast.steps, { colorBasis: t.rateBasis });
    const add = (k, v) => { codec[k] = (codec[k] ?? 0) + v; };
    PLANE_NAMES.forEach((k, p) => add(k, enc.planeBits[p]));
    add("position", enc.positionBits);
    add("header", enc.headerBits);
    sites += enc.n;
  };
  const { results } = await trainBatched({ ...ctx, args, quiet: true }, images, { after });
  const pixels = images.reduce((s, im) => s + im.width * im.height, 0);
  const total = Object.values(codec).reduce((s, v) => s + v, 0);
  return { psnr: median(results.map((r) => r.psnr)), bpp: total / pixels, bitsPerSite: total / sites,
           colorPerSite: ["K0", "K1", "K2"].map((k) => codec[k] / sites), codec, sites };
}

// PSNR of the curve `pts` (sorted by bpp) at `bpp`, or null outside it.
function interp(pts, bpp) {
  for (let i = 1; i < pts.length; i += 1) {
    const a = pts[i - 1], b = pts[i];
    if (bpp >= a.bpp && bpp <= b.bpp) return a.psnr + ((b.psnr - a.psnr) * (bpp - a.bpp)) / Math.max(1e-12, b.bpp - a.bpp);
  }
  return null;
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, configs: { type: "string", default: "klt:1,klt:2,klt:3" },
      sites: { type: "string", default: "4,6,8,10,12" }, repeats: { type: "string", default: "1" },
      ref: { type: "string" }, out: { type: "string" },
    },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label} | ${images.length} images from ${cli.from}`);
  const configs = cli.configs.split(",").map((spec) => {
    const [space, dims = "3", lr] = spec.split(":");
    const params = { ...base, colorSpace: space, colorDims: Number(dims) };
    if (lr) params.lrColorBase = Number(lr);
    return { name: `${space}:${dims}`, params };
  });
  const siteList = cli.sites.split(",").map(Number);
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  const rows = [];
  for (const c of configs) {
    for (const n of siteList) {
      const reps = [];
      for (let k = 0; k < repeats; k += 1) reps.push(await run(ctx, { ...c.params, sites: n }, images));
      const m = (f) => median(reps.map(f));
      const r = { config: c.name, sites: n, params: { ...c.params, sites: n }, psnr: m((x) => x.psnr), bpp: m((x) => x.bpp),
                  bitsPerSite: m((x) => x.bitsPerSite), colorPerSite: [0, 1, 2].map((j) => m((x) => x.colorPerSite[j])),
                  psnrReps: reps.map((x) => x.psnr), reps };
      rows.push(r);
      console.error(`${c.name.padEnd(8)} ${String(n).padStart(3)} sites | PSNR ${r.psnr.toFixed(3)} dB` +
                    (repeats > 1 ? ` (${r.psnrReps.map((v) => v.toFixed(3)).join(", ")})` : "") +
                    ` | ${r.bpp.toFixed(4)} bpp, ${r.bitsPerSite.toFixed(2)} bits/site, color ${r.colorPerSite.map((v) => v.toFixed(2)).join("/")}`);
    }
  }
  // Pareto frontier: no other point has ≥ PSNR at ≤ bpp.
  for (const r of rows) r.frontier = !rows.some((o) => o !== r && o.psnr >= r.psnr && o.bpp <= r.bpp && (o.psnr > r.psnr || o.bpp < r.bpp));
  const ref = cli.ref ?? configs[configs.length - 1].name;
  const refPts = rows.filter((r) => r.config === ref).sort((a, b) => a.bpp - b.bpp);
  for (const r of rows) {
    const p = interp(refPts, r.bpp);
    r.dPsnrVsRef = p === null ? null : r.psnr - p;
  }
  console.error(`\n| config | sites | PSNR (dB) | bpp | bits/site | color K0/K1/K2 bits/site | Δ vs ${ref} at equal bpp | frontier |`);
  console.error("|---|---|---|---|---|---|---|---|");
  for (const r of [...rows].sort((a, b) => a.bpp - b.bpp)) {
    console.error(`| ${r.config} | ${r.sites} | ${r.psnr.toFixed(3)} | ${r.bpp.toFixed(4)} | ${r.bitsPerSite.toFixed(2)} | ` +
                  `${r.colorPerSite.map((v) => v.toFixed(2)).join(" / ")} | ` +
                  `${r.dPsnrVsRef === null ? "" : (r.dPsnrVsRef >= 0 ? "+" : "") + r.dPsnrVsRef.toFixed(3)} | ${r.frontier ? "yes" : ""} |`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/colordims_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: cli.from, count: images.length, ref, rows }, null, 1) + "\n");
  console.error(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
