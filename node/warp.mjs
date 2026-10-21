#!/usr/bin/env node
// 3a: a shared, learned monotone warp per coded parameter under D + λR. Does
// a collective move of all sites with similar values improve the rate–
// distortion trade over the rate term alone?
//
//   node warp.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --lambdas 0,0.01,0.03,0.1,0.3,1 --lrs 0.1 --repeats 3
//
// Warp. Per coded plane p (rate.js's RATE_PLANES: log τ, radius, the three
//   KLT color coordinates, log aniso) a 1-D map φ_p(v) = v + Σ_k a_k b_k(v),
//   b_k(v) = exp(−(v − μ_k)² / 2h²), K centres μ_k evenly over the 1st–99th
//   percentile of the plane's current values, h their spacing. φ_p moves every
//   site of the atlas (all images: a dataset-level transform). At a = 0,
//   ∂L/∂a_k = Σ_i g_i b_k(v_i), a kernel-smoothed histogram of the per-site
//   gradients g_i of squared error + λ · bits: the warp is the collective move
//   of sites with similar values. It has no gradient where per-site
//   gradients vanish (a distortion-only optimum, λ = 0); it can only matter
//   through the rate term (or as an optimizer, before convergence).
// Learning. Every --every iterations (the trainer's log tick; no trainer
//   changes) the hook takes the log event's raw gradients (squared error
//   only: the trainer snapshots them before rate.wgsl adds its term) and adds
//   λ · ∂bits/∂v on the CPU from the trainer's current rate model (rate.wgsl's
//   planeGrad, and θ's log₂|2 sinh a| on log aniso). Colors: g along KLT axis
//   k is axis_k · g_rgb. The step on a is Adam over the warp steps (state per
//   plane and centre), a_k = −lr · h · m̂ / (√v̂ + ε), scale-free; a is shrunk
//   until min φ' ≥ 0.5 on a grid (monotone). The sites are moved through φ
//   (colors: rgb += Σ_k Δ_k axis_k) and written back (lossmap.mjs'
//   writeRgbSites); per-site Adam state is kept. The accumulated warp is the
//   composition of these steps. From --start × iters on.
// Measurement (rate.mjs' codecMeasure): the final sites coded as one stream by
//   the real codec, median PSNR over images vs codec bits per pixel. Each λ is
//   run with the rate term alone (the baseline curve) and with the warp; the
//   warp's gain is ΔPSNR at matched bpp, against the baseline curve
//   interpolated linearly in bpp (and Δbpp at matched PSNR). --lrs > 1 value:
//   every lr is run (for tuning on images 0–1023; score held out 1024–3071).
//   --repeats: medians over repeats. --grads dr,d also runs the warp on the
//   squared-error gradient alone (rate term still on per site): what the
//   λ · bits part of the collective move adds, apart from its optimizer effect
//   (the warp moves sites a lot even at λ = 0: 100 iterations are far from a
//   distortion-only optimum).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=m3-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";
import { writeRgbSites } from "./lossmap.mjs";
import { codecMeasure, addMeasure } from "./rate.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const LOG_2PI = Math.log(2 * Math.PI);
export const PLANES = ["log_tau", "radius", "K0", "K1", "K2", "log_aniso"];

// d bits / dv for one plane of a fitted rate model (rate.wgsl planeGrad).
export function planeGradOf(model, p) {
  const mix = model.mixtures[p], step = model.stepOf[p];
  if (!mix) return () => 0;
  const comp = mix.means.map((mu, j) => {
    const s2 = mix.vars[j] + (step * step) / 12;
    return { norm: Math.log(step) + Math.log(mix.weights[j]) - 0.5 * (LOG_2PI + Math.log(s2)), mu, s2 };
  });
  return (v) => {
    let top = -Infinity;
    for (const c of comp) top = Math.max(top, c.norm - ((v - c.mu) ** 2) / (2 * c.s2));
    let sum = 0, num = 0;
    for (const c of comp) { const e = Math.exp(c.norm - ((v - c.mu) ** 2) / (2 * c.s2) - top); sum += e; num += (e * (v - c.mu)) / c.s2; }
    return top + Math.log(sum) >= 0 ? 0 : num / sum / Math.LN2;
  };
}

const thetaGrad = (a, steps) => (steps.theta > 0 && Math.PI * Math.abs(2 * Math.sinh(a)) > steps.theta && Math.abs(a) > 1e-4
  ? 1 / (Math.LN2 * Math.tanh(a)) : 0);

// Quantile of a sorted array.
const qs = (s, q) => s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];

// The warp's per-plane state and its step from one log event.
export function makeWarp({ kernels = 16, lr = 0.1, planes = PLANES, rate = true, beta1 = 0.9, beta2 = 0.999 } = {}) {
  const state = planes.map(() => ({ m: new Float64Array(kernels), v: new Float64Array(kernels), t: 0 }));
  const stats = { steps: 0, moved: planes.map(() => 0) };
  // sites: RGB atlas rows (modified in place); grads: raw squared-error
  // gradients; model: the trainer's rate model (null = λ ignored); skip(i).
  function step(sites, grads, n, basis, model, lambda, skip) {
    const rows = [];
    for (let i = 0; i < n; i += 1) if (sites[i * SITE_FLOATS] >= 0 && !skip(i)) rows.push(i);
    if (!rows.length) return;
    const { mean: cm, axes } = basis;
    const valueOf = [
      (b) => sites[b + 2], (b) => sites[b + 3],
      ...[0, 1, 2].map((k) => (b) => axes[k][0] * (sites[b + 4] - cm[0]) + axes[k][1] * (sites[b + 5] - cm[1]) + axes[k][2] * (sites[b + 6] - cm[2])),
      (b) => sites[b + 9],
    ];
    const gradOf = [
      (b) => grads[b + 2], (b) => grads[b + 3],
      ...[0, 1, 2].map((k) => (b) => axes[k][0] * grads[b + 4] + axes[k][1] * grads[b + 5] + axes[k][2] * grads[b + 6]),
      (b) => grads[b + 9],
    ];
    const delta = PLANES.map(() => null);
    planes.forEach((name, j) => {
      const p = PLANES.indexOf(name);
      const vs = new Float64Array(rows.length), gs = new Float64Array(rows.length);
      const rg = rate && model && lambda > 0 ? planeGradOf(model, p) : null;
      rows.forEach((i, r) => {
        const b = i * SITE_FLOATS;
        vs[r] = valueOf[p](b);
        let g = gradOf[p](b);
        if (rg) g += lambda * (rg(vs[r]) + (p === 5 ? thetaGrad(vs[r], model.steps) : 0));
        gs[r] = g;
      });
      const sorted = Float64Array.from(vs).sort();
      const lo = qs(sorted, 0.01), hi = qs(sorted, 0.99);
      if (!(hi > lo)) return;
      const h = (hi - lo) / (kernels - 1);
      const mu = Array.from({ length: kernels }, (_, k) => lo + k * h);
      const G = new Float64Array(kernels);
      for (let r = 0; r < rows.length; r += 1) for (let k = 0; k < kernels; k += 1) G[k] += gs[r] * Math.exp(-((vs[r] - mu[k]) ** 2) / (2 * h * h));
      const st = state[j];
      st.t += 1;
      const a = new Float64Array(kernels);
      for (let k = 0; k < kernels; k += 1) {
        st.m[k] = beta1 * st.m[k] + (1 - beta1) * G[k];
        st.v[k] = beta2 * st.v[k] + (1 - beta2) * G[k] * G[k];
        const mh = st.m[k] / (1 - beta1 ** st.t), vh = st.v[k] / (1 - beta2 ** st.t);
        a[k] = (-lr * h * mh) / (Math.sqrt(vh) + 1e-30);
      }
      // Monotone: φ' = 1 + Σ a_k b_k'(v) ≥ 0.5 on a grid over [lo − 3h, hi + 3h].
      let minD = Infinity;
      for (let x = lo - 3 * h; x <= hi + 3 * h; x += h / 8) {
        let d = 1;
        for (let k = 0; k < kernels; k += 1) d += (a[k] * -(x - mu[k]) / (h * h)) * Math.exp(-((x - mu[k]) ** 2) / (2 * h * h));
        minD = Math.min(minD, d);
      }
      if (minD < 0.5) { const s = 0.5 / (1 - minD); for (let k = 0; k < kernels; k += 1) a[k] *= s; }
      const dv = new Float64Array(rows.length);
      let moved = 0;
      for (let r = 0; r < rows.length; r += 1) {
        let d = 0;
        for (let k = 0; k < kernels; k += 1) d += a[k] * Math.exp(-((vs[r] - mu[k]) ** 2) / (2 * h * h));
        dv[r] = d;
        moved += Math.abs(d) / h;
      }
      stats.moved[j] += moved / rows.length; // mean |move| in kernel spacings
      delta[p] = dv;
    });
    // Apply all planes at once (the values above are pre-move).
    rows.forEach((i, r) => {
      const b = i * SITE_FLOATS;
      if (delta[0]) sites[b + 2] += delta[0][r];
      if (delta[1]) sites[b + 3] += delta[1][r];
      for (let k = 0; k < 3; k += 1) {
        const d = delta[2 + k];
        if (d) for (let c = 0; c < 3; c += 1) sites[b + 4 + c] += d[r] * axes[k][c];
      }
      if (delta[5]) sites[b + 9] += delta[5][r];
    });
    stats.steps += 1;
  }
  return { step, stats };
}

// One batched run (rate term at params.rateLambda, warp if `warp` options),
// measured by the codec. onAtlas(bt) (optional) runs before the measurement.
export async function runOnce(ctx, params, images, { warp = null, every = 1, start = 0, hook = null, logFreq = null, onAtlas = null } = {}) {
  const { args } = await argsFor(params);
  args.logFreq = warp ? every : (logFreq ?? args.iters);
  const out = { estimate: null, codec: null, sites: 0, warp: [], atlas: [] };
  const warps = new Map();
  const onLog = (info, idx, bt) => {
    if (info.type !== "batchIter") return;
    const t = bt.trainer, n = t.siteCount;
    if (hook) hook(info, bt);
    if (!warp || info.it < Math.floor(start * info.iters) || info.it === info.iters - 1) return;
    if (!warps.has(bt)) warps.set(bt, makeWarp(warp));
    const per = bt.layout.sitesPerTile, layer = bt.layerSlot;
    const skip = (i) => layer !== null && layer !== undefined && i % per === layer;
    const sites = info.sitesData;
    warps.get(bt).step(sites, info.grads, n, t.rateBasis, t.rateModel, args.rateLambda, skip);
    writeRgbSites(t, sites);
  };
  const after = async (bt) => {
    if (onAtlas) out.atlas.push(await onAtlas(bt));
    addMeasure(out, await codecMeasure(bt.trainer));
    if (warps.has(bt)) out.warp.push(warps.get(bt).stats);
  };
  const { results } = await trainBatched({ ...ctx, args, quiet: true }, images, { onLog, after });
  out.pixels = images.reduce((s, im) => s + im.width * im.height, 0);
  out.psnr = median(results.map((r) => r.psnr));
  out.codecTotal = Object.values(out.codec).reduce((s, v) => s + v, 0);
  out.bpp = out.codecTotal / out.pixels;
  out.bitsPerSite = out.codecTotal / out.sites;
  return out;
}

// Baseline PSNR at `bpp` from a curve [{ bpp, psnr }] (linear in bpp; NaN outside).
export function psnrAt(curve, bpp) {
  const c = [...curve].sort((a, b) => a.bpp - b.bpp);
  for (let i = 0; i + 1 < c.length; i += 1) {
    if (bpp >= c[i].bpp && bpp <= c[i + 1].bpp) {
      const t = (bpp - c[i].bpp) / Math.max(1e-12, c[i + 1].bpp - c[i].bpp);
      return c[i].psnr + t * (c[i + 1].psnr - c[i].psnr);
    }
  }
  return NaN;
}

// Baseline bpp at `psnr` (the curve sorted by PSNR; NaN outside).
export function bppAt(curve, psnr) {
  const c = [...curve].sort((a, b) => a.psnr - b.psnr);
  for (let i = 0; i + 1 < c.length; i += 1) {
    if (psnr >= c[i].psnr && psnr <= c[i + 1].psnr) {
      const t = (psnr - c[i].psnr) / Math.max(1e-12, c[i + 1].psnr - c[i].psnr);
      return c[i].bpp + t * (c[i + 1].bpp - c[i].bpp);
    }
  }
  return NaN;
}

// Medians over repeats of a list of runOnce results.
export function summarizeReps(reps) {
  const m = (f) => median(reps.map(f));
  const per = (k) => m((x) => (x.codec[k] ?? 0) / x.sites);
  return {
    psnr: m((x) => x.psnr), bpp: m((x) => x.bpp), bitsPerSite: m((x) => x.bitsPerSite),
    psnrSpread: Math.max(...reps.map((x) => x.psnr)) - Math.min(...reps.map((x) => x.psnr)),
    planesPerSite: Object.fromEntries(["position", "log_tau", "radius", "K0", "K1", "K2", "log_aniso", "θ"].map((k) => [k, per(k)])),
  };
}

const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : "—");
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : "—");

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, lambdas: { type: "string", default: "0,0.01,0.03,0.1,0.3,1" },
      lrs: { type: "string", default: "0.1" }, grads: { type: "string", default: "dr" }, kernels: { type: "string", default: "16" },
      every: { type: "string", default: "1" }, start: { type: "string", default: "0" },
      planes: { type: "string", default: PLANES.join(",") },
      repeats: { type: "string", default: "1" }, out: { type: "string" },
    },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders };
  console.error(`Device | ${label} | ${images.length} images from ${cli.from}`);
  const lambdas = cli.lambdas.split(",").map(Number), lrs = cli.lrs.split(",").map(Number);
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  const grads = cli.grads.split(",");
  const wopt = (lr, g) => ({ warp: { kernels: Number(cli.kernels), lr, planes: cli.planes.split(","), rate: g === "dr" }, every: Number(cli.every), start: Number(cli.start) });
  const rows = [];
  const runRow = async (name, lambda, lr, g = null) => {
    const reps = [];
    for (let k = 0; k < repeats; k += 1) reps.push(await runOnce(ctx, { ...base, rateLambda: lambda }, images, lr === null ? {} : wopt(lr, g)));
    const r = { name, lambda, lr, grad: g, ...summarizeReps(reps), reps: reps.map(({ psnr, bpp, bitsPerSite, codec, sites, warp }) => ({ psnr, bpp, bitsPerSite, codec, sites, warp })) };
    rows.push(r);
    const mv = reps[0].warp.length ? ` | mean |move| per step (spacings) ${reps[0].warp[0].moved.map((x) => (x / Math.max(1, reps[0].warp[0].steps)).toFixed(3)).join("/")}` : "";
    console.error(`${name.padEnd(18)} | PSNR ${f3(r.psnr)} dB (spread ${f3(r.psnrSpread)}), ${f4(r.bpp)} bpp, ${r.bitsPerSite.toFixed(2)} bits/site${mv}`);
    return r;
  };
  const baseline = [];
  for (const l of lambdas) baseline.push(await runRow(`rate λ ${l}`, l, null));
  const warped = [];
  for (const g of grads) for (const lr of lrs) for (const l of lambdas) {
    if (g === "d" && l === 0 && grads.includes("dr")) continue; // the same run at λ 0
    warped.push(await runRow(`warp ${g} lr ${lr} λ ${l}`, l, lr, g));
  }
  console.error(`Result | ΔPSNR at matched bpp and Δbpp at matched PSNR vs the rate-only curve (— outside its range)`);
  console.error(`         grad lr      λ       PSNR     bpp      | same-λ ΔPSNR  Δbpp    | ΔPSNR@bpp  Δbpp@PSNR`);
  for (const w of warped) {
    const b = baseline.find((x) => x.lambda === w.lambda);
    w.dPsnrSameLambda = w.psnr - b.psnr;
    w.dBppSameLambda = w.bpp - b.bpp;
    w.dPsnrAtBpp = w.psnr - psnrAt(baseline, w.bpp);
    w.dBppAtPsnr = w.bpp - bppAt(baseline, w.psnr);
    console.error(`         ${w.grad.padEnd(4)} ${String(w.lr).padEnd(7)} ${String(w.lambda).padEnd(7)} ${f3(w.psnr).padEnd(8)} ${f4(w.bpp).padEnd(8)} | ` +
                  `${f3(w.dPsnrSameLambda).padEnd(13)} ${f4(w.dBppSameLambda).padEnd(7)} | ${f3(w.dPsnrAtBpp).padEnd(10)} ${f4(w.dBppAtPsnr)}`);
  }
  for (const g of grads) for (const lr of lrs) {
    const ws = warped.filter((w) => w.grad === g && w.lr === lr && Number.isFinite(w.dPsnrAtBpp));
    console.error(`Mean   | ${g} lr ${lr}: ΔPSNR@bpp ${f3(mean(ws.map((w) => w.dPsnrAtBpp)))} dB over ${ws.length} λ inside the curve`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/warp_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from), count: images.length,
                                            preset: cli.preset, set: cli.set, repeats, kernels: Number(cli.kernels), every: Number(cli.every),
                                            start: Number(cli.start), planes: cli.planes, baseline, warped }, null, 1) + "\n");
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
