#!/usr/bin/env node
// E7: does a loss map over parameter space predict what restricting the space
// costs? One batched run, then counterfactual snaps of regions of it:
//
//   node lossmap.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 \
//     --preset ../presets/mnist.json
//
// Loss map. Each masked pixel's squared error goes to the site that owns it
//   (largest blend weight: the trainer's "owner" render), so the per-site
//   losses partition the atlas SSE exactly; the histogram layer keeps its
//   share but is in no region. (A soft share Σ w·e² would need the weights per
//   pixel, which no render exports; with 8 candidates and trained τ the owner
//   carries most of the weight.) The sites are binned on 2D slices, equal-width
//   bins over each axis' 1st–99th percentile (outliers in the edge bins), so
//   bins differ in site count and density shows up: chroma |rgb − grey| ×
//   luma (mean rgb), log τ × radius, |log aniso| × radius. Per bin: sites,
//   owned pixels, SSE (the naive map), SSE per site, SSE per pixel.
// Counterfactual. Each bin with ≥ --min-sites sites is snapped into a
//   restricted space on its slice: grey (colors onto the grey axis, chroma ×
//   luma), tau (log τ to the median over all sites), tauIqr (log τ clamped to
//   its interquartile range over all sites, a milder restriction; bins inside
//   it don't move and are skipped) and radius (to the median; these three on
//   log τ × radius), iso (log aniso to 0; |log aniso| × radius). The
//   ΔPSNR is measured three ways: render only with the trained candidate
//   lists (Trainer.snapPsnr); render with the lists rebuilt from the snapped
//   sites (decodedPsnr, as a decoder would); and after a refit: --refit-iters
//   more iterations from the trained state (fresh Adam moments, LR
//   --refit-lr × the preset's, cosine to 0, no persistent-error weighting, so
//   plain L2), with the bin's snapped values projected back after every step
//   (projected gradient descent through the trainer's log hook), against the
//   same refit with nothing snapped (two controls give the noise). Direction
//   and anisotropy only learn from iteration 50 (ANI_START_ITER), so a refit
//   shorter than that leaves them as trained, snapped or not.
// Prediction. ΔSSE ≈ Σ F·δ² over the bin's sites and parameters, F the
//   Gauss–Newton diagonal (Trainer.siteSensitivity), δ the snap's change per
//   parameter; also with the first-order term Σ g·δ, g the exact SSE gradient
//   at the trained state (one zero-LR iteration with weighting off). ΔSSE →
//   ΔPSNR = −10 log10(1 + ΔSSE / SSE₀), SSE₀ the atlas SSE. Reports Spearman
//   and the slope (through 0) of measured on predicted ΔPSNR per snap and
//   pooled, and whether the naive loss map is confounded by site count /
//   area: Spearman of the measured cost against SSE, sites and pixels per
//   bin, and of the cost per site against SSE per site and per pixel.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=20261019";
import { readSites } from "../trainer/buffers.js?v=20261019";
import { encodeSiteColors, decodeSiteColors } from "../trainer/color_spaces.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// ---- Trainer state helpers (no trainer changes: buffers and args from outside) ----

// The trainer's raw sites (color-space params), to put back with restoreSites.
export const saveSites = (tr) => readSites(tr.device, tr.sitesBuf, tr.actualSites);

export function restoreSites(tr, raw) {
  tr.device.queue.writeBuffer(tr.sitesBuf, 0, raw);
  tr._rebuildCandidates(tr.actualSites);
}

// RGB sites → the trainer's buffer (a copy is encoded).
export function writeRgbSites(tr, rgb) {
  tr.device.queue.writeBuffer(tr.sitesBuf, 0, encodeSiteColors(Float32Array.from(rgb), tr.actualSites, tr.colorSpace));
}

export const rgbSitesOf = (tr, raw) => decodeSiteColors(Float32Array.from(raw), tr.actualSites, tr.colorSpace);

// Runs the trainer on from its current sites: fresh Adam moments, `patch` on
// its args for the run (iters, lrMult, errWeight, logFreq…; restored after),
// onIter(info) at each log tick ("iter" events: sitesData in RGB, grads).
export async function continueRun(tr, patch, onIter = null) {
  const saved = Object.fromEntries(Object.keys(patch).map((k) => [k, tr.args[k]]));
  const savedLog = tr.onLog;
  const enc = tr.device.createCommandEncoder();
  enc.clearBuffer(tr.adamBuf);
  tr.device.queue.submit([enc.finish()]);
  Object.assign(tr.args, patch);
  tr.onLog = (info) => { if (info.type === "iter" && onIter) onIter(info); };
  try { await tr.run(); } finally { Object.assign(tr.args, saved); tr.onLog = savedLog; }
}

// Exact gradient of the summed squared error at the current sites (10 per
// site, colors per RGB channel): one iteration at zero LR, weighting off.
export async function sseGradient(tr) {
  let grads = null;
  await continueRun(tr, { iters: 1, lrMult: 0, errWeight: 0, logFreq: 1 }, (info) => { grads = info.grads; });
  return grads;
}

const sseOf = (psnr, samples) => samples * 10 ** (-psnr / 10);
export const dbOf = (dSse, sse0) => -10 * Math.log10(Math.max(1e-12, 1 + dSse / sse0));

// ---- Stats ----

function ranks(xs) {
  const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  const r = new Float64Array(xs.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && xs[idx[j + 1]] === xs[idx[i]]) j += 1;
    for (let k = i; k <= j; k += 1) r[idx[k]] = (i + j) / 2;
    i = j + 1;
  }
  return Array.from(r);
}

export function pearson(xs, ys) {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0, syy = 0;
  xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
}

export function spearman(xs, ys) {
  const ok = xs.map((x, i) => Number.isFinite(x) && Number.isFinite(ys[i]));
  const a = xs.filter((_, i) => ok[i]), b = ys.filter((_, i) => ok[i]);
  return a.length >= 3 ? pearson(ranks(a), ranks(b)) : NaN;
}

// Least-squares slope of y on x through the origin.
export function slope0(xs, ys) {
  let sxy = 0, sxx = 0;
  xs.forEach((x, i) => { if (Number.isFinite(x) && Number.isFinite(ys[i])) { sxy += x * ys[i]; sxx += x * x; } });
  return sxx > 0 ? sxy / sxx : NaN;
}

function quantile(xs, q) {
  const s = Float64Array.from(xs).sort();
  return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : NaN;
}

// ---- Slices and snaps ----

const greyOf = (s) => (s[4] + s[5] + s[6]) / 3;
const AXES = {
  chroma: (s) => { const m = greyOf(s); return Math.hypot(s[4] - m, s[5] - m, s[6] - m); },
  luma: greyOf,
  logTau: (s) => s[2],
  radius: (s) => s[3],
  absAniso: (s) => Math.abs(s[9]),
};
export const SLICES = [
  { key: "chroma × luma", x: "chroma", y: "luma", snaps: ["grey"] },
  { key: "log τ × radius", x: "logTau", y: "radius", snaps: ["tau", "tauIqr", "radius"] },
  { key: "|log aniso| × radius", x: "absAniso", y: "radius", snaps: ["iso"] },
];
// In place on an RGB site row; med: medians over every active site.
const SNAPS = {
  grey: (s) => { const m = greyOf(s); s[4] = s[5] = s[6] = m; },
  tau: (s, med) => { s[2] = med.logTau; },
  tauIqr: (s, med) => { s[2] = Math.min(med.tauHi, Math.max(med.tauLo, s[2])); },
  radius: (s, med) => { s[3] = med.radius; },
  iso: (s) => { s[9] = 0; },
};

// Equal-width bins over [p1, p99] of each axis; returns per site its bin
// (row-major y · bins + x) and the edges.
function binSites(rows, slice, bins) {
  const xs = rows.map((r) => AXES[slice.x](r.s)), ys = rows.map((r) => AXES[slice.y](r.s));
  const range = (v) => { const lo = quantile(v, 0.01), hi = quantile(v, 0.99); return [lo, hi > lo ? hi : lo + 1e-6]; };
  const [x0, x1] = range(xs), [y0, y1] = range(ys);
  const at = (v, lo, hi) => Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
  return { bin: rows.map((_, i) => at(ys[i], y0, y1) * bins + at(xs[i], x0, x1)), x: [x0, x1], y: [y0, y1] };
}

// ---- One run ----

// Trains `images` once and runs the loss map and every snap on the trained
// atlas (one atlas expected; more are analysed one by one and pooled).
export async function lossmap(ctx, args, images, opts) {
  const { bins = 5, minSites = 8, refitIters = 40, refitLr = 0.25 } = opts;
  const regions = [];
  const maps = [];
  const atlases = [];
  const after = async (bt) => {
    const tr = bt.trainer;
    const n = tr.actualSites, per = bt.layout.sitesPerTile, layer = bt.layerSlot;
    const g = await sseGradient(tr);
    const { F, samples } = await tr.siteSensitivity();
    const raw0 = await saveSites(tr);
    const rgb0 = rgbSitesOf(tr, raw0);
    // Loss per site from the owner render.
    const owner = await tr.renderView("owner");
    const { rgba, psnr: psnr0 } = await tr.renderEdited();
    const loss = new Float64Array(n), area = new Float64Array(n);
    let sse0 = 0;
    for (let p = 0; p < tr.width * tr.height; p += 1) {
      if (tr.maskCpu && !(tr.maskCpu[p] > 0)) continue;
      let e = 0;
      for (let c = 0; c < 3; c += 1) e += (rgba[p * 4 + c] - tr.target[p * 4 + c]) ** 2;
      sse0 += e;
      const id = owner[p * 4];
      if (id >= 0) { loss[id] += e; area[id] += 1; }
    }
    const rows = [];
    for (let i = 0; i < n; i += 1) {
      if (layer !== null && i % per === layer) continue;
      if (!(rgb0[i * SITE_FLOATS] >= 0)) continue;
      rows.push({ i, s: rgb0.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS) });
    }
    const taus = rows.map((r) => r.s[2]);
    const med = { logTau: median(taus), tauLo: quantile(taus, 0.25), tauHi: quantile(taus, 0.75), radius: median(rows.map((r) => r.s[3])) };
    const layerLoss = layer === null ? 0 : Array.from({ length: n / per }, (_, b) => loss[b * per + layer]).reduce((a, b) => a + b, 0);
    const refitPatch = { iters: refitIters, lrMult: args.lrMult * refitLr, errWeight: 0 };
    // Control refits: the trained state, nothing snapped.
    const controls = [];
    if (refitIters > 0) {
      for (let k = 0; k < 2; k += 1) {
        restoreSites(tr, raw0);
        await continueRun(tr, { ...refitPatch, logFreq: refitIters + 1 });
        controls.push((await tr.renderEdited()).psnr);
      }
    }
    const ctl = mean(controls);
    const atlas = { images: images.length, sites: rows.length, samples, psnr0, sse0, layerLossShare: layerLoss / sse0, med,
                    controls, controlPsnr: ctl };
    atlases.push(atlas);
    console.error(`Atlas  | ${rows.length} sites, PSNR ${psnr0.toFixed(3)} dB, layer owns ${(100 * layerLoss / sse0).toFixed(1)}% of SSE` +
                  (controls.length ? `, refit control ${controls.map((c) => (c - psnr0).toFixed(3)).join(" / ")} dB` : ""));
    for (const slice of SLICES) {
      const { bin, x, y } = binSites(rows, slice, bins);
      const cells = Array.from({ length: bins * bins }, () => ({ members: [], sites: 0, pixels: 0, sse: 0 }));
      rows.forEach((r, k) => { const c = cells[bin[k]]; c.members.push(r.i); c.sites += 1; c.pixels += area[r.i]; c.sse += loss[r.i]; });
      maps.push({ slice: slice.key, x: [slice.x, ...x], y: [slice.y, ...y], bins,
                  cells: cells.map((c) => ({ sites: c.sites, pixels: c.pixels, sseShare: c.sse / sse0 })) });
      for (const snap of slice.snaps) {
        for (let cell = 0; cell < cells.length; cell += 1) {
          const c = cells[cell];
          if (c.sites < minSites) continue;
          // Snapped RGB sites, round-tripped through the color space so δ is what the buffer holds.
          const snapped = Float32Array.from(rgb0);
          for (const i of c.members) SNAPS[snap](snapped.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS), med);
          const held = rgbSitesOf(tr, encodeSiteColors(Float32Array.from(snapped), n, tr.colorSpace));
          let gn = 0, lin = 0, moved = 0;
          for (const i of c.members) {
            let any = false;
            for (let t = 0; t < 10; t += 1) {
              const d = held[i * SITE_FLOATS + t] - rgb0[i * SITE_FLOATS + t];
              if (d === 0) continue;
              any = true;
              gn += F[i * 10 + t] * d * d;
              lin += g[i * 10 + t] * d;
            }
            if (any) moved += 1;
          }
          if (!moved) continue; // the snap leaves this bin as it is (tauIqr inside the band)
          const snapR = await tr.snapPsnr(held);
          const decR = await tr.decodedPsnr(held);
          let refit = NaN;
          if (refitIters > 0) {
            restoreSites(tr, raw0);
            writeRgbSites(tr, held);
            const project = (info) => {
              const cur = info.sitesData;
              for (const i of c.members) SNAPS[snap](cur.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS), med);
              writeRgbSites(tr, cur);
            };
            await continueRun(tr, { ...refitPatch, logFreq: 1 }, project);
            refit = (await tr.renderEdited()).psnr - ctl;
            restoreSites(tr, raw0);
          }
          const region = {
            slice: slice.key, snap, cell, bx: cell % bins, by: Math.floor(cell / bins),
            sites: c.sites, moved, pixels: c.pixels, sse: c.sse, sseShare: c.sse / sse0,
            ssePerSite: c.sse / c.sites, ssePerPixel: c.pixels > 0 ? c.sse / c.pixels : NaN,
            predGnSse: gn, predLinSse: lin,
            predGn: dbOf(gn, sse0), predGnLin: dbOf(gn + lin, sse0),
            snap_: snapR.psnr - snapR.basePsnr, decoded: decR.psnr - decR.basePsnr, refit,
          };
          region.snapSse = sseOf(snapR.psnr, samples) - sseOf(snapR.basePsnr, samples);
          regions.push(region);
        }
        const mine = regions.filter((r) => r.snap === snap);
        console.error(`         ${snap.padEnd(6)} ${mine.length} regions done`);
      }
    }
    restoreSites(tr, raw0);
  };
  const { results, seconds } = await trainBatched({ ...ctx, args, quiet: true }, images, { after });
  return { images: images.length, seconds, medianPsnr: median(results.map((r) => r.psnr)), atlases, maps, regions,
           summary: summarize(regions) };
}

// Correlations per snap and pooled (measured ΔPSNR is negative for a cost).
export function summarize(regions) {
  const groups = { ...Object.fromEntries([...new Set(regions.map((r) => r.snap))].map((s) => [s, regions.filter((r) => r.snap === s)])), all: regions };
  const out = {};
  for (const [name, rs] of Object.entries(groups)) {
    const col = (k) => rs.map((r) => r[k]);
    const perSite = (k) => rs.map((r) => r[k] / r.sites);
    const cost = col("snapSse");
    out[name] = {
      regions: rs.length,
      predict: Object.fromEntries(["snap_", "decoded", "refit"].map((m) => [m, {
        spearmanGn: spearman(col("predGn"), col(m)), slopeGn: slope0(col("predGn"), col(m)),
        spearmanGnLin: spearman(col("predGnLin"), col(m)), slopeGnLin: slope0(col("predGnLin"), col(m)),
        pearsonGn: pearson(col("predGn"), col(m)),
      }])),
      snapVsRefit: spearman(col("snap_"), col("refit")),
      // Naive loss map: does the cost follow SSE, or just how many sites / pixels a bin holds?
      lossMap: {
        costVsSse: spearman(cost, col("sse")), costVsSites: spearman(cost, col("sites")), costVsPixels: spearman(cost, col("pixels")),
        sseVsSites: spearman(col("sse"), col("sites")),
        perSiteCostVsSsePerSite: spearman(perSite("snapSse"), col("ssePerSite")),
        perSiteCostVsSsePerPixel: spearman(perSite("snapSse"), col("ssePerPixel")),
        perSiteCostVsGnPerSite: spearman(perSite("snapSse"), perSite("predGnSse")),
        refitVsSse: spearman(col("refit"), col("sse").map((x) => -x)),
        refitPerSiteVsSsePerPixel: spearman(perSite("refit"), col("ssePerPixel").map((x) => -x)),
      },
    };
  }
  return out;
}

// ---- Printing ----

const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : "  —");
const f3 = (x) => (Number.isFinite(x) ? (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(1) : x.toFixed(3)) : "—");

export function printReport(rep, log = console.error) {
  log(`Run    | ${rep.images} images, median PSNR ${rep.medianPsnr.toFixed(3)} dB, ${rep.seconds.toFixed(1)}s`);
  for (const m of rep.maps) {
    log(`Map    | ${m.slice}: SSE share % (rows ${m.y[0]} ${f2(m.y[1])}…${f2(m.y[2])} up, cols ${m.x[0]} ${f2(m.x[1])}…${f2(m.x[2])}) | sites`);
    for (let by = m.bins - 1; by >= 0; by -= 1) {
      const row = m.cells.slice(by * m.bins, (by + 1) * m.bins);
      log(`         ${row.map((c) => (c.sites ? (100 * c.sseShare).toFixed(1) : "·").padStart(6)).join("")}   |` +
          `${row.map((c) => String(c.sites || "·").padStart(6)).join("")}`);
    }
  }
  log(`Predict| snap   regions  measured  Spearman(GN)  slope(GN)  Spearman(GN+g)  slope(GN+g)`);
  for (const [name, s] of Object.entries(rep.summary)) {
    for (const [m, p] of Object.entries(s.predict)) {
      log(`         ${name.padEnd(6)} ${String(s.regions).padEnd(8)} ${m.replace("snap_", "render").padEnd(9)} ` +
          `${f2(p.spearmanGn).padEnd(13)} ${f2(p.slopeGn).padEnd(10)} ${f2(p.spearmanGnLin).padEnd(15)} ${f2(p.slopeGnLin)}`);
    }
  }
  log(`LossMap| snap   cost~SSE  cost~sites  cost~pixels  SSE~sites | per site: cost~SSE/site  cost~SSE/px  cost~GN | refit~SSE  refit/site~SSE/px`);
  for (const [name, s] of Object.entries(rep.summary)) {
    const l = s.lossMap;
    log(`         ${name.padEnd(6)} ${f2(l.costVsSse).padEnd(9)} ${f2(l.costVsSites).padEnd(11)} ${f2(l.costVsPixels).padEnd(12)} ` +
        `${f2(l.sseVsSites).padEnd(10)}   ${f2(l.perSiteCostVsSsePerSite).padEnd(15)} ${f2(l.perSiteCostVsSsePerPixel).padEnd(12)} ` +
        `${f2(l.perSiteCostVsGnPerSite).padEnd(9)} ${f2(l.refitVsSse).padEnd(10)} ${f2(l.refitPerSiteVsSsePerPixel)}`);
  }
  log(`Regions| largest measured render costs (ΔPSNR dB): snap slice-bin sites | pred GN  pred GN+g  render  decoded  refit`);
  for (const r of [...rep.regions].sort((a, b) => a.snap_ - b.snap_).slice(0, 12)) {
    log(`         ${r.snap.padEnd(6)} (${r.bx},${r.by}) ${String(r.sites).padEnd(5)} | ${f3(r.predGn).padEnd(8)} ${f3(r.predGnLin).padEnd(10)} ` +
        `${f3(r.snap_).padEnd(7)} ${f3(r.decoded).padEnd(8)} ${f3(r.refit)}`);
  }
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, count: { type: "string", default: "1024" },
      bins: { type: "string", default: "5" }, "min-sites": { type: "string", default: "8" },
      "refit-iters": { type: "string", default: "40" }, "refit-lr": { type: "string", default: "0.25" },
      out: { type: "string" }, help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write("Usage: node lossmap.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [--from n] [--count n]\n" +
                         "                        [--preset json] [--set key=value …] [--bins 5] [--min-sites 8]\n" +
                         "                        [--refit-iters 40] [--refit-lr 0.25] [--out json]\n");
    process.exit(cli.help ? 0 : 1);
  }
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label}`);
  const opts = { bins: Number(cli.bins), minSites: Number(cli["min-sites"]), refitIters: Number(cli["refit-iters"]),
                 refitLr: Number(cli["refit-lr"]) };
  const rep = await lossmap({ device, shaders }, args, images, opts);
  printReport(rep);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/lossmap_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from), opts, args, ...rep }, null, 1) + "\n");
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
