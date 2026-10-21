#!/usr/bin/env node
// 3c: MNIST binarization. MNIST digits are near-binary; with the rate term on,
// do the sites' colors collapse onto two levels (black and white) as λ grows,
// and where on the rate–distortion curve is that free?
//
//   node binarize.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --lambdas 0,0.03,0.1,0.3,1,3 --components 2,4 --snaps none,2means,bw
//
// Distribution. After training, the coded sites' (histogram layer left out)
//   luma coordinate K0 (the first KLT axis of rate.js / the codec; on grey
//   digits rgb = mean + K0 · axis₀, so K0 maps to a grey level g = mean(m) +
//   K0 · mean(axis₀)): a histogram of g in 0.1 bins (with under/overflow), the
//   share of sites within 0.1 of black or white, and the rate model's fitted
//   K0 mixture (components ≥ 3% weight, as grey mean ± sd). rateComponents
//   (--components) is the mixture size the rate term trains against.
// Forcing (--snaps). Projected training: from --snap-start × iters on, after
//   every step (the log hook, as lossmap.mjs' projected refit) every site's K0
//   is moved to the nearer of two levels, K1 / K2 left as they are. 2means:
//   the levels are refit to the current K0 values each step (two-means,
//   started from the 5th / 95th percentile); bw: fixed at grey 0 and 1.
// Measurement (rate.mjs' codecMeasure): median PSNR over images, the real
//   codec's bits per pixel and per site, its K0 plane bits per site, and each
//   snapped run's ΔPSNR at matched bpp against the free curve at the same
//   rateComponents (warp.mjs' psnrAt). H: the empirical entropy of the K0
//   symbols at the rate model's color step, what a static model ideally pays
//   (the codec's Exp-Golomb residuals send the low mantissa bits raw, so a
//   two-level plane with levels many steps apart costs it more). --repeats: medians over repeats.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=w2b-20261021";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, median, mean } from "./common.mjs";
import { writeRgbSites } from "./lossmap.mjs";
import { runOnce, psnrAt, bppAt, summarizeReps } from "./warp.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const BINS = [-Infinity, ...Array.from({ length: 11 }, (_, k) => k / 10), Infinity];

const k0Of = (s, b, m, a) => a[0] * (s[b + 4] - m[0]) + a[1] * (s[b + 5] - m[1]) + a[2] * (s[b + 6] - m[2]);

// Coded sites' rows of an atlas (layer and inactive slots skipped).
function codedRows(bt, sites) {
  const per = bt.layout.sitesPerTile, layer = bt.layerSlot, rows = [];
  for (let i = 0; i < bt.trainer.siteCount; i += 1) if (sites[i * SITE_FLOATS] >= 0 && !(layer !== null && i % per === layer)) rows.push(i);
  return rows;
}

// Two-means on xs from the 5th / 95th percentile.
function twoMeans(xs) {
  const s = Float64Array.from(xs).sort();
  let c = [s[Math.floor(0.05 * (s.length - 1))], s[Math.floor(0.95 * (s.length - 1))]];
  for (let it = 0; it < 20; it += 1) {
    const sum = [0, 0], n = [0, 0], cut = (c[0] + c[1]) / 2;
    for (const x of xs) { const j = x < cut ? 0 : 1; sum[j] += x; n[j] += 1; }
    c = c.map((v, j) => (n[j] ? sum[j] / n[j] : v));
  }
  return c;
}

// Projection hook: K0 → nearer of two levels (mode "2means" | "bw").
function snapHook(mode, start) {
  return (info, bt) => {
    if (info.it < Math.floor(start * info.iters)) return;
    const t = bt.trainer, { mean: m, axes } = t.rateBasis, a = axes[0], s = info.sitesData;
    const rows = codedRows(bt, s);
    const k0 = rows.map((i) => k0Of(s, i * SITE_FLOATS, m, a));
    const ga = mean(a), gm = mean(m);
    const levels = mode === "bw" ? [(0 - gm) / ga, (1 - gm) / ga] : twoMeans(k0);
    rows.forEach((i, r) => {
      const to = Math.abs(k0[r] - levels[0]) <= Math.abs(k0[r] - levels[1]) ? levels[0] : levels[1];
      for (let c = 0; c < 3; c += 1) s[i * SITE_FLOATS + 4 + c] += (to - k0[r]) * a[c];
    });
    bt._snapLevels = levels.map((l) => gm + l * ga);
    writeRgbSites(t, s);
  };
}

// Distribution of the final K0 (as grey levels) and the fitted K0 mixture.
async function distribution(bt) {
  const t = bt.trainer;
  await t._rateRefit(t.siteCount);
  const s = await t._readSitesRgb(t.siteCount), { mean: m, axes } = t.rateBasis, a = axes[0];
  const ga = mean(a), gm = mean(m);
  const k0 = codedRows(bt, s).map((i) => k0Of(s, i * SITE_FLOATS, m, a));
  const greys = k0.map((v) => gm + ga * v);
  const counts = new Map(), step = t.rateLast.steps.color;
  for (const v of k0) { const q = Math.round(v / step); counts.set(q, (counts.get(q) ?? 0) + 1); }
  const entropy = [...counts.values()].reduce((h, c) => h - (c / k0.length) * Math.log2(c / k0.length), 0);
  const hist = BINS.slice(0, -1).map((lo, k) => greys.filter((g) => g >= lo && g < BINS[k + 1]).length / greys.length);
  const mix = t.rateModel.mixtures[2];
  const modes = mix.means.map((mu, j) => ({ w: mix.weights[j], grey: gm + ga * mu, sd: Math.abs(ga) * Math.sqrt(mix.vars[j]) }))
    .filter((c) => c.w >= 0.03).sort((x, y) => x.grey - y.grey);
  return { sites: greys.length, hist, nearBW: greys.filter((g) => Math.abs(g) < 0.1 || Math.abs(g - 1) < 0.1).length / greys.length,
           greyMedian: median(greys), k0Entropy: entropy, k0Symbols: counts.size, colorStepGrey: Math.abs(ga) * t.rateLast.steps.color, modes, snapLevels: bt._snapLevels ?? null };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, lambdas: { type: "string", default: "0,0.03,0.1,0.3,1,3" },
      components: { type: "string", default: "2,4" }, snaps: { type: "string", default: "none,2means,bw" },
      "snap-start": { type: "string", default: "0.5" }, repeats: { type: "string", default: "1" }, out: { type: "string" },
    },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label} | ${images.length} images from ${cli.from}`);
  const repeats = Math.max(1, Number(cli.repeats) | 0), start = Number(cli["snap-start"]);
  const rows = [];
  for (const comps of cli.components.split(",").map(Number)) {
    for (const snap of cli.snaps.split(",")) {
      for (const lambda of cli.lambdas.split(",").map(Number)) {
        const reps = [];
        for (let k = 0; k < repeats; k += 1) {
          reps.push(await runOnce(ctx, { ...base, rateLambda: lambda, rateComponents: comps }, images,
            snap === "none" ? { onAtlas: distribution } : { hook: snapHook(snap, start), logFreq: 1, onAtlas: distribution }));
        }
        const d = reps[0].atlas[0];
        const r = { comps, snap, lambda, ...summarizeReps(reps), dist: reps.map((x) => x.atlas), reps: reps.map(({ psnr, bpp, bitsPerSite, codec, sites }) => ({ psnr, bpp, bitsPerSite, codec, sites })) };
        rows.push(r);
        console.error(`K ${comps} ${snap.padEnd(6)} λ ${String(lambda).padEnd(5)} | PSNR ${r.psnr.toFixed(3)}, ${r.bpp.toFixed(4)} bpp, ` +
                      `${r.bitsPerSite.toFixed(2)} bits/site, K0 ${r.planesPerSite.K0.toFixed(2)} (H ${d.k0Entropy.toFixed(2)}, ${d.k0Symbols} bins) | near b/w ${(100 * d.nearBW).toFixed(1)}%, ` +
                      `step ${d.colorStepGrey.toFixed(3)}, modes ${d.modes.map((c) => `${c.w.toFixed(2)}@${c.grey.toFixed(2)}±${c.sd.toFixed(2)}`).join(" ")}` +
                      (d.snapLevels ? ` | levels ${d.snapLevels.map((l) => l.toFixed(2)).join("/")}` : ""));
        console.error(`                       | grey hist % (<0, 0–.1, …, .9–1, ≥1.0): ${d.hist.map((h) => (100 * h).toFixed(0)).join(" ")}`);
      }
    }
  }
  console.error(`Result | snapped vs free (same rateComponents): ΔPSNR at matched bpp, Δbpp at matched PSNR, and at the same λ`);
  for (const r of rows.filter((x) => x.snap !== "none")) {
    const free = rows.filter((x) => x.snap === "none" && x.comps === r.comps);
    const same = free.find((x) => x.lambda === r.lambda);
    Object.assign(r, { dPsnrAtBpp: r.psnr - psnrAt(free, r.bpp), dBppAtPsnr: r.bpp - bppAt(free, r.psnr),
                       dPsnrSameLambda: r.psnr - same.psnr, dBppSameLambda: r.bpp - same.bpp, dK0SameLambda: r.planesPerSite.K0 - same.planesPerSite.K0 });
    const f = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "—");
    console.error(`         K ${r.comps} ${r.snap.padEnd(6)} λ ${String(r.lambda).padEnd(5)} | ΔPSNR@bpp ${f(r.dPsnrAtBpp)}, Δbpp@PSNR ${f(r.dBppAtPsnr, 4)} | ` +
                  `same λ: ΔPSNR ${f(r.dPsnrSameLambda)}, Δbpp ${f(r.dBppSameLambda, 4)}, ΔK0 bits/site ${f(r.dK0SameLambda, 2)}`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/binarize_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from), count: images.length,
                                            preset: cli.preset, set: cli.set, repeats, snapStart: start, bins: BINS.slice(1, -1), rows }, null, 1) + "\n");
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
