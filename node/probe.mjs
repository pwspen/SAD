#!/usr/bin/env node
// Diagnostics for normalizing the per-group learning rates (E1–E3), on one
// batched run:
//
//   node probe.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 \
//     --preset ../presets/mnist.json
//
// E1, natural units. For each LR group, three candidate units in the
//   coordinates Adam steps in (an Adam step is about lr in those units,
//   whatever the gradient's scale): Δ, the codecSteps quantizer step (same
//   formula, pooled Gauss–Newton sensitivity); 1/√F, the median per-site
//   sensitivity scale; σ, the robust spread of the values across sites (the
//   site-values histogram's width). Reports lr / unit per group; if a unit
//   is right, those ratios are about equal.
// E2, per-bin gradient. The site parameters' values are binned (equal-count
//   bins pooled over every image), and at each log tick the gradient is
//   reduced per bin, which is the gradient of shifting every site in the bin
//   together (a shared warp of that parameter). Reports, early and late in
//   the run: R², the share of Σg² a per-bin mean explains (null ≈ bins / sites);
//   the median |t| of the bin means (null ≈ 0.8); how often a bin's mean
//   keeps its sign between ticks (null 0.5).
// E3, joint vs per-parameter coding. On the final sites: each parameter's
//   entropy at the codec's quantizer steps, summed (about what coding them
//   one by one costs), against the total correlation TC = Σ h(Xᵢ) − h(X), the
//   most a joint model could save at high rate. TC from a k-nearest-neighbour
//   estimate of the copula entropy, minus the same on columns shuffled
//   independently (the estimator's bias), and from a Gaussian copula.
// Codec audit. E3's per-parameter sum is a static model of RGB shared by
//   every image; the real codec differs in three ways: it codes YCoCg-R
//   integers, predicts each value from neighbouring sites (or a per-image
//   constant), and its adaptive models start from scratch in every image.
//   At E3's steps, bits per site for the color: static entropies of RGB, of
//   the codec's YCoCg-R integers, and of the joint RGB triple; then the codec
//   itself, once per image (as it runs) and once over every image's sites in
//   one stream (no per-image header or model start-up), with every plane's
//   share; and the same with colors coded along a KLT basis instead of
//   YCoCg-R: the PCA of all the sites' colors, and (klt runs) the training
//   basis. The histogram layer is left out throughout.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=m3-20261020";
import { codecSteps, encodeSites, CODEC_PLANES } from "../trainer/codec.js?v=shared-20261020";
import { COLOR_SPACES, pcaBasis } from "../trainer/color_spaces.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const LR_GROUPS = [
  { key: "pos", lr: "lrPosBase" }, { key: "tau", lr: "lrTauBase" }, { key: "radius", lr: "lrRadiusBase" },
  { key: "color", lr: "lrColorBase" }, { key: "dir", lr: "lrDirBase" }, { key: "aniso", lr: "lrAnisoBase" },
];
const ROUND_ANISO = 0.05; // |log_aniso| below this, a site has no orientation (as train.js's MI view)
const VALUE_BINS = 24;
const LOSS_DB = 0.5; // quantization loss the Δ unit and E3's steps are set for

// ---- Shared ----

// Robust spread: interquartile range / 1.349 (σ for a Gaussian).
function spread(xs) {
  const s = Float64Array.from(xs).sort();
  if (s.length < 4) return NaN;
  return (s[Math.floor(0.75 * (s.length - 1))] - s[Math.floor(0.25 * (s.length - 1))]) / 1.349;
}

const theta = (dx, dy) => { const t = Math.atan2(dy, dx); return t < 0 ? t + Math.PI : t >= Math.PI ? t - Math.PI : t; };

// Spread of an axis angle (mod π): circular σ of 2θ, halved.
function axisSpread(ts) {
  if (ts.length < 4) return NaN;
  let c = 0, s = 0;
  for (const t of ts) { c += Math.cos(2 * t); s += Math.sin(2 * t); }
  const R = Math.min(1 - 1e-12, Math.hypot(c, s) / ts.length);
  return Math.sqrt(-2 * Math.log(R)) / 2;
}

// ---- E1 ----

// Per active site: its sensitivity in Adam's coordinates for each LR group
// (per scalar, so groups of several scalars average them), and its value(s).
// F: Gauss–Newton diagonal (10 per site, colors per RGB channel). sites: RGB.
function adamSensitivities(sites, F, slots, colorSpace) {
  const cs = COLOR_SPACES[colorSpace];
  const out = Object.fromEntries(LR_GROUPS.map((g) => [g.key, []]));
  for (const i of slots) {
    const b = i * SITE_FLOATS, f = i * 10;
    const r = sites[b + 4], g = sites[b + 5], bl = sites[b + 6];
    out.pos.push((F[f] + F[f + 1]) / 2);
    out.tau.push(F[f + 2]);
    out.radius.push(F[f + 3]);
    // Color params p = encode(rgb); F_k = Σ_c F_c (∂rgb_c / ∂p_k)², rows of J from the VJP.
    const p = cs.encode(r, g, bl);
    const rows = [0, 1, 2].map((c) => cs.vjp(p, [0, 1, 2].map((k) => (k === c ? 1 : 0))));
    let fc = 0;
    for (let k = 0; k < 3; k += 1) for (let c = 0; c < 3; c += 1) fc += F[f + 4 + c] * rows[c][k] ** 2;
    out.color.push(fc / 3);
    // Adam steps dir along its tangent: F_θ = F_x · dir_y² + F_y · dir_x².
    if (Math.abs(sites[b + 9]) >= ROUND_ANISO) out.dir.push(F[f + 7] * sites[b + 8] ** 2 + F[f + 8] * sites[b + 7] ** 2);
    out.aniso.push(F[f + 9]);
  }
  return out;
}

// Values in Adam's coordinates per group, for σ.
function adamValues(sites, slots, colorSpace) {
  const cs = COLOR_SPACES[colorSpace];
  const v = { pos: [], tau: [], radius: [], color: [[], [], []], dir: [], aniso: [] };
  for (const i of slots) {
    const b = i * SITE_FLOATS;
    v.pos.push(sites[b], sites[b + 1]);
    v.tau.push(sites[b + 2]);
    v.radius.push(sites[b + 3]);
    cs.encode(sites[b + 4], sites[b + 5], sites[b + 6]).forEach((x, k) => v.color[k].push(x));
    if (Math.abs(sites[b + 9]) >= ROUND_ANISO) v.dir.push(theta(sites[b + 7], sites[b + 8]));
    v.aniso.push(sites[b + 9]);
  }
  return v;
}

export function e1Units({ sites, F, slots, sse0, args }) {
  const sens = adamSensitivities(sites, F, slots, args.colorSpace);
  const vals = adamValues(sites, slots, args.colorSpace);
  const d = (sse0 * (10 ** (LOSS_DB / 10) - 1)) / 9; // codecSteps' added error per scalar type
  const groups = {};
  for (const { key, lr } of LR_GROUPS) {
    const f = sens[key].filter((x) => Number.isFinite(x) && x > 0);
    const A = f.reduce((s, x) => s + x, 0);
    const sigma = key === "color" ? mean(vals.color.map(spread)) : key === "dir" ? axisSpread(vals.dir) : spread(vals[key]);
    groups[key] = {
      lr: args[lr], sites: f.length,
      delta: A > 0 ? Math.sqrt((12 * d * f.length) / A) : NaN,
      invSqrtF: median(f.map((x) => 1 / Math.sqrt(x))),
      sigma,
    };
  }
  return groups;
}

// Spread of log10(lr / unit) across groups, in decades: 0 means the unit
// explains every group's LR up to one shared constant.
export function collapse(groups, unit, keys = LR_GROUPS.map((g) => g.key)) {
  const logs = keys.map((k) => Math.log10(groups[k].lr / (unit ? groups[k][unit] : 1))).filter(Number.isFinite);
  const m = mean(logs);
  return { decades: Math.sqrt(mean(logs.map((x) => (x - m) ** 2))), logMean: m };
}

// ---- E2 ----

const E2_PARAMS = [
  { key: "log τ", ch: 2 }, { key: "radius", ch: 3 }, { key: "R", ch: 4 }, { key: "G", ch: 5 }, { key: "B", ch: 6 },
  { key: "log aniso", ch: 9 },
];

// ticks: [{ it, v: Float32Array per E2 param, g: Float32Array per E2 param }].
export function e2Report(ticks, iters, phases = { early: [0, 0.3], late: [0.7, 1.01] }) {
  const out = {};
  E2_PARAMS.forEach((p, pi) => {
    const all = Float64Array.from(ticks.flatMap((t) => Array.from(t.v[pi]))).sort();
    if (all.length < VALUE_BINS * 4 || all[0] === all[all.length - 1]) { out[p.key] = null; return; }
    const edges = Array.from({ length: VALUE_BINS - 1 }, (_, k) => all[Math.floor(((k + 1) * all.length) / VALUE_BINS)]);
    const binOf = (x) => { let lo = 0, hi = edges.length; while (lo < hi) { const m = (lo + hi) >> 1; if (x < edges[m]) hi = m; else lo = m + 1; } return lo; };
    const perTick = ticks.map((t) => {
      const n = new Float64Array(VALUE_BINS), s = new Float64Array(VALUE_BINS), s2 = new Float64Array(VALUE_BINS);
      const v = t.v[pi], g = t.g[pi];
      for (let i = 0; i < v.length; i += 1) { const b = binOf(v[i]); n[b] += 1; s[b] += g[i]; s2[b] += g[i] * g[i]; }
      return { it: t.it, n, s, s2 };
    });
    out[p.key] = {};
    for (const [phase, [a, b]] of Object.entries(phases)) {
      const sel = perTick.filter((t) => t.it >= a * iters && t.it < b * iters);
      if (!sel.length) continue;
      const r2 = [], r2null = [], tabs = [], persist = [];
      for (const t of sel) {
        let between = 0, total = 0, sites = 0, occupied = 0;
        for (let k = 0; k < VALUE_BINS; k += 1) {
          if (!t.n[k]) continue;
          occupied += 1; sites += t.n[k];
          between += (t.s[k] * t.s[k]) / t.n[k];
          total += t.s2[k];
          if (t.n[k] >= 3) {
            const m = t.s[k] / t.n[k], varr = (t.s2[k] - t.n[k] * m * m) / (t.n[k] - 1);
            if (varr > 0) tabs.push(Math.abs(m) / Math.sqrt(varr / t.n[k]));
          }
        }
        if (total > 0) { r2.push(between / total); r2null.push(occupied / sites); }
      }
      for (let j = 1; j < sel.length; j += 1) {
        for (let k = 0; k < VALUE_BINS; k += 1) {
          if (sel[j].s[k] !== 0 && sel[j - 1].s[k] !== 0) persist.push(Math.sign(sel[j].s[k]) === Math.sign(sel[j - 1].s[k]) ? 1 : 0);
        }
      }
      out[p.key][phase] = { ticks: sel.length, r2: mean(r2), r2Null: mean(r2null), medianT: median(tabs), signPersist: mean(persist) };
    }
  });
  return out;
}

// ---- E3 ----

function digamma(x) {
  let r = 0;
  while (x < 6) { r -= 1 / x; x += 1; }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))));
}

// Copula columns: ranks scaled to (0, 1), ties broken at random.
function copula(cols, rand) {
  const n = cols[0].length;
  return cols.map((c) => {
    const idx = Array.from({ length: n }, (_, i) => i);
    const jit = Float64Array.from({ length: n }, rand);
    idx.sort((a, b) => c[a] - c[b] || jit[a] - jit[b]);
    const u = new Float64Array(n);
    idx.forEach((i, r) => { u[i] = (r + 0.5) / n; });
    return u;
  });
}

// Kozachenko–Leonenko entropy (nats) of the rows of `cols`, max norm, k-th
// neighbour, distances floored at half a rank step.
function klEntropy(cols, k = 4) {
  const d = cols.length, n = cols[0].length;
  const floor = 0.5 / n;
  const kth = new Float64Array(k);
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    kth.fill(Infinity);
    for (let j = 0; j < n; j += 1) {
      if (j === i) continue;
      let dist = 0;
      for (let c = 0; c < d && dist < kth[k - 1]; c += 1) { const x = Math.abs(cols[c][i] - cols[c][j]); if (x > dist) dist = x; }
      if (dist < kth[k - 1]) {
        let p = k - 1;
        while (p > 0 && kth[p - 1] > dist) { kth[p] = kth[p - 1]; p -= 1; }
        kth[p] = dist;
      }
    }
    acc += Math.log(2 * Math.max(kth[k - 1], floor));
  }
  return digamma(n) - digamma(k) + (d * acc) / n;
}

// Inverse normal CDF (Acklam).
function probit(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const dd = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const q = Math.min(p, 1 - p);
  let x;
  if (q < 0.02425) {
    const r = Math.sqrt(-2 * Math.log(q));
    x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((dd[0] * r + dd[1]) * r + dd[2]) * r + dd[3]) * r + 1);
    return p < 0.5 ? x : -x;
  }
  const r = (p - 0.5) ** 2;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * (p - 0.5)) /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// Gaussian-copula total correlation (bits): −½ log det of the normal scores'
// correlation matrix, by Cholesky. Infinity when singular.
function gaussianTc(u) {
  const d = u.length, n = u[0].length;
  const z = u.map((c) => Float64Array.from(c, probit));
  const C = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => {
    let s = 0, si = 0, sj = 0;
    for (let t = 0; t < n; t += 1) { s += z[i][t] * z[j][t]; si += z[i][t] ** 2; sj += z[j][t] ** 2; }
    return s / Math.sqrt(si * sj);
  }));
  let logdet = 0;
  const L = C.map((r) => r.slice());
  for (let j = 0; j < d; j += 1) {
    let s = L[j][j];
    for (let k = 0; k < j; k += 1) s -= L[j][k] ** 2;
    if (!(s > 1e-12)) return Infinity;
    L[j][j] = Math.sqrt(s);
    logdet += Math.log(s);
    for (let i = j + 1; i < d; i += 1) {
      let t = L[i][j];
      for (let k = 0; k < j; k += 1) t -= L[i][k] * L[j][k];
      L[i][j] = t / L[j][j];
    }
  }
  return -0.5 * logdet / Math.LN2;
}

function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Miller–Madow entropy (bits) of the integers round(x / step).
function quantizedEntropy(xs, step) {
  const counts = new Map();
  for (const x of xs) { const q = Math.round(x / step); counts.set(q, (counts.get(q) ?? 0) + 1); }
  let h = 0;
  for (const c of counts.values()) h -= (c / xs.length) * Math.log2(c / xs.length);
  return h + (counts.size - 1) / (2 * xs.length * Math.LN2);
}

const E3_VARS = ["log τ", "radius", "R", "G", "B", "log aniso", "x", "y"];
const E3_SUBSETS = {
  color: ["R", "G", "B"],
  shape: ["log τ", "radius", "log aniso"],
  "all but position": ["log τ", "radius", "R", "G", "B", "log aniso"],
  all: E3_VARS,
};

// rows: per active site [log τ, radius, R, G, B, log aniso, x / tileW, y / tileH].
// steps: codecSteps. maxRows caps the k-NN work (O(n²)).
export function e3Report(rows, steps, { maxRows = 6000, seed = 1 } = {}) {
  const rand = rng(seed);
  let sample = rows;
  if (rows.length > maxRows) {
    sample = rows.map((r) => [rand(), r]).sort((a, b) => a[0] - b[0]).slice(0, maxRows).map((x) => x[1]);
  }
  const col = (name) => sample.map((r) => r[E3_VARS.indexOf(name)]);
  const stepOf = { "log τ": steps.logTau, radius: steps.radius, R: steps.color, G: steps.color, B: steps.color,
                   "log aniso": steps.logAniso, x: null, y: null };
  const marginalBits = {};
  for (const v of E3_VARS) if (stepOf[v]) marginalBits[v] = quantizedEntropy(rows.map((r) => r[E3_VARS.indexOf(v)]), stepOf[v]);
  const subsets = {};
  for (const [name, vars] of Object.entries(E3_SUBSETS)) {
    const u = copula(vars.map(col), rand);
    const shuffled = u.map((c) => { const s = Float64Array.from(c); for (let i = s.length - 1; i > 0; i -= 1) { const j = Math.floor(rand() * (i + 1)); [s[i], s[j]] = [s[j], s[i]]; } return s; });
    const tc = -klEntropy(u) / Math.LN2, bias = -klEntropy(shuffled) / Math.LN2;
    const coded = vars.filter((v) => marginalBits[v] !== undefined);
    subsets[name] = {
      vars, tcKnn: tc - bias, tcBias: bias, tcGaussian: gaussianTc(u),
      marginalBits: coded.reduce((s, v) => s + marginalBits[v], 0), codedVars: coded.length,
    };
  }
  // Color PCA (RGB): share of variance on the first axis, and that axis.
  const rgb = ["R", "G", "B"].map((v) => rows.map((r) => r[E3_VARS.indexOf(v)]));
  const mu = rgb.map(mean);
  const cov = [0, 1, 2].map((i) => [0, 1, 2].map((j) => mean(rgb[i].map((x, t) => (x - mu[i]) * (rgb[j][t] - mu[j])))));
  let vec = [1, 1, 1];
  for (let it = 0; it < 100; it += 1) {
    const w = cov.map((r) => r[0] * vec[0] + r[1] * vec[1] + r[2] * vec[2]);
    const nrm = Math.hypot(...w) || 1;
    vec = w.map((x) => x / nrm);
  }
  const lam1 = vec.reduce((s, x, i) => s + x * cov[i].reduce((t, c, j) => t + c * vec[j], 0), 0);
  const trace = cov[0][0] + cov[1][1] + cov[2][2];
  return {
    sites: rows.length, knnSample: sample.length, marginalBits, subsets,
    colorPca: { firstAxisShare: trace > 0 ? lam1 / trace : NaN, axis: vec.map((x) => Number(x.toFixed(3))) },
  };
}

// ---- Codec audit ----

function entropyOf(keys) {
  const counts = new Map();
  for (const k of keys) counts.set(k, (counts.get(k) ?? 0) + 1);
  let h = 0;
  for (const c of counts.values()) h -= (c / keys.length) * Math.log2(c / keys.length);
  return h + (counts.size - 1) / (2 * keys.length * Math.LN2);
}

// Per-plane codec bits per site: { header, position, <plane>…, total }.
// `shared`: code against that shared model (its steps and basis) instead.
export function codecBits(list, steps, colorBasis = null, shared = null) {
  if (shared) colorBasis = shared.colorBasis;
  const planes = colorBasis ? [...CODEC_PLANES.slice(0, 4), "K0", "K1", "K2"] : CODEC_PLANES;
  const acc = { header: 0, position: 0, ...Object.fromEntries(planes.map((p) => [p, 0])) };
  let n = 0, modes = null;
  for (const im of list) {
    const enc = encodeSites(im.sites, im.n, im.width, im.height, steps, { colorBasis, shared });
    acc.header += enc.headerBits;
    acc.position += enc.positionBits;
    planes.forEach((p, k) => { acc[p] += enc.planeBits[k]; });
    n += enc.n;
    modes = enc.modes;
  }
  const out = Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, v / Math.max(1, n)]));
  out.total = Object.values(out).reduce((s, v) => s + v, 0);
  out.color = planes.slice(4).reduce((t, p) => t + out[p], 0);
  return { perSite: out, sites: n, modes: list.length === 1 ? Object.fromEntries(planes.map((p, k) => [p, modes[k] ? "neighbours" : "constant"])) : null };
}

// Every image's sites as one image: tiles in a near-square grid.
export function poolImages(perImage) {
  const W = Math.max(...perImage.map((im) => im.width)), H = Math.max(...perImage.map((im) => im.height));
  const T = Math.ceil(Math.sqrt(perImage.length));
  const total = perImage.reduce((s, im) => s + im.n, 0);
  const pooled = new Float32Array(total * SITE_FLOATS);
  let at = 0;
  perImage.forEach((im, b) => {
    const ox = (b % T) * W, oy = Math.floor(b / T) * H;
    for (let i = 0; i < im.n; i += 1, at += 1) {
      const src = im.sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS);
      pooled.set(src, at * SITE_FLOATS);
      if (src[0] >= 0) { pooled[at * SITE_FLOATS] += ox; pooled[at * SITE_FLOATS + 1] += oy; }
    }
  });
  return [{ sites: pooled, n: total, width: T * W, height: Math.ceil(perImage.length / T) * H }];
}

// perImage: each image's sites in its own coordinates. rows: E3's rows.
export function codecAudit(perImage, rows, steps, trainBasis = null) {
  const c = steps.color;
  const rgba = new Float32Array(rows.length * 4);
  rows.forEach((r, i) => rgba.set([r[2], r[3], r[4], 1], i * 4));
  const siteBasis = pcaBasis(rgba, rows.length);
  const kltInts = (basis) => rows.map((r) => basis.axes.map((a) => Math.round(
    (a[0] * (r[2] - basis.mean[0]) + a[1] * (r[3] - basis.mean[1]) + a[2] * (r[4] - basis.mean[2])) / c)));
  const q = rows.map((r) => [Math.round(r[2] / c), Math.round(r[3] / c), Math.round(r[4] / c)]);
  const ycocg = q.map(([r, g, b]) => { const co = r - b, t = b + (co >> 1), cg = g - t; return [t + (cg >> 1), co, cg]; });
  const col = (arr, k) => arr.map((v) => v[k]);
  const staticBits = {
    "R+G+B": [0, 1, 2].reduce((s, k) => s + entropyOf(col(q, k)), 0),
    "Y+Co+Cg": [0, 1, 2].reduce((s, k) => s + entropyOf(col(ycocg, k)), 0),
    "K0+K1+K2": [0, 1, 2].reduce((s, k) => s + entropyOf(col(kltInts(siteBasis), k)), 0),
    "joint RGB": entropyOf(q.map((v) => v.join(","))),
  };
  const perPlaneStatic = { Y: entropyOf(col(ycocg, 0)), Co: entropyOf(col(ycocg, 1)), Cg: entropyOf(col(ycocg, 2)) };
  const pooledList = poolImages(perImage);
  const out = {
    sites: rows.length, staticBits, perPlaneStatic, siteBasis, trainBasis,
    perImage: codecBits(perImage, steps),
    pooled: codecBits(pooledList, steps),
    perImageKlt: codecBits(perImage, steps, siteBasis),
    pooledKlt: codecBits(pooledList, steps, siteBasis),
  };
  if (trainBasis) out.pooledTrainKlt = codecBits(pooledList, steps, trainBasis);
  return out;
}

// ---- One probed run ----

// Trains `images` once with `args`, recording E2's per-tick gradients and,
// after the run, the Gauss–Newton diagonal for E1 and the final sites for E3.
// `keep`: an object that gets the per-image sites (perImage, as codecAudit
// takes them), E3's rows and the training basis, for further audits
// (shared_codec.mjs).
export async function probe(ctx, args, images, { e2 = true, e3 = true, audit = true, keep = null } = {}) {
  const ticks = [];
  const e1Parts = { sites: [], F: [], sse0: 0 };
  const rows = [];
  const perImage = []; // { sites (layer marked inactive), n, width, height }
  let trainBasis = null; // klt runs: the training basis (the last atlas's)
  const record = (info, idx, bt) => {
    if (!e2 || info.type !== "batchIter") return;
    const n = bt.layout.sitesPerTile, layer = info.layer;
    const v = E2_PARAMS.map(() => []), g = E2_PARAMS.map(() => []);
    for (let i = 0; i < idx.length * n; i += 1) {
      if (i % n === layer || !(info.sitesData[i * SITE_FLOATS] >= 0)) continue;
      let any = false;
      for (let c = 0; c < 10; c += 1) if (info.grads[i * 10 + c] !== 0) { any = true; break; }
      if (!any) continue; // covers no pixel
      E2_PARAMS.forEach((p, pi) => { v[pi].push(info.sitesData[i * SITE_FLOATS + p.ch]); g[pi].push(info.grads[i * 10 + p.ch]); });
    }
    ticks.push({ it: info.it, v: v.map((x) => Float32Array.from(x)), g: g.map((x) => Float32Array.from(x)) });
  };
  const after = async (bt, idx) => {
    const { F } = await bt.trainer.siteSensitivity();
    if (bt.trainer.colorSpace.needsBasis) trainBasis = bt.trainer.colorSpace.basis;
    const final = await bt.finalize();
    const n = bt.layout.sitesPerTile, layer = bt.layerSlot;
    final.forEach((r, b) => {
      const img = images[idx[b]];
      const own = Float32Array.from(r.sites);
      if (layer !== null && layer !== undefined) own[layer * SITE_FLOATS] = -1;
      perImage.push({ sites: own, n, width: img.width, height: img.height });
      e1Parts.sse0 += 3 * img.width * img.height * 10 ** (-r.psnr / 10);
      for (let i = 0; i < n; i += 1) {
        if (i === layer || !(r.sites[i * SITE_FLOATS] >= 0)) continue;
        const s = r.sites.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS);
        e1Parts.sites.push(s);
        e1Parts.F.push(F.subarray((b * n + i) * 10, (b * n + i + 1) * 10));
        rows.push([s[2], s[3], s[4], s[5], s[6], s[9], s[0] / img.width, s[1] / img.height]);
      }
    });
  };
  const { results, seconds } = await trainBatched({ ...ctx, args, quiet: true }, images, { onLog: record, after });
  const count = e1Parts.sites.length;
  const sites = new Float32Array(count * SITE_FLOATS), F = new Float32Array(count * 10);
  e1Parts.sites.forEach((s, i) => sites.set(s, i * SITE_FLOATS));
  e1Parts.F.forEach((f, i) => F.set(f, i * 10));
  const slots = Array.from({ length: count }, (_, i) => i);
  const units = e1Units({ sites, F, slots, sse0: e1Parts.sse0, args });
  const steps = codecSteps(sites, count, F, { lossDb: LOSS_DB, sse0: e1Parts.sse0 });
  if (keep) Object.assign(keep, { perImage, rows, trainBasis });
  return {
    images: images.length, sites: count, seconds, medianPsnr: median(results.map((r) => r.psnr)),
    e1: { units, collapse: Object.fromEntries(["raw", "delta", "invSqrtF", "sigma"].map((u) => [u, collapse(units, u === "raw" ? null : u)])) },
    steps,
    e2: e2 ? e2Report(ticks, args.iters) : null,
    e3: e3 ? e3Report(rows, steps) : null,
    audit: audit ? codecAudit(perImage, rows, steps, trainBasis) : null,
  };
}

// ---- Printing ----

const f3 = (x) => (Number.isFinite(x) ? (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toPrecision(3)) : String(x));

export function printReport(rep, log = console.error) {
  log(`Run    | ${rep.images} images, ${rep.sites} sites, median PSNR ${rep.medianPsnr.toFixed(3)} dB, ${rep.seconds.toFixed(1)}s`);
  log(`E1     | group   lr          Δ          1/√F       σ          lr/Δ       lr/(1/√F)  lr/σ`);
  for (const [k, g] of Object.entries(rep.e1.units)) {
    log(`         ${k.padEnd(7)} ${[g.lr, g.delta, g.invSqrtF, g.sigma, g.lr / g.delta, g.lr / g.invSqrtF, g.lr / g.sigma].map((x) => f3(x).padEnd(10)).join(" ")}`);
  }
  log(`         spread of log10(lr / unit) across groups: ` +
      Object.entries(rep.e1.collapse).map(([u, c]) => `${u} ${c.decades.toFixed(2)}`).join(", ") + " decades");
  if (rep.e2) {
    log(`E2     | param      phase  R²      R² null  median|t| (null 0.8)  sign kept (null 0.5)`);
    for (const [k, ph] of Object.entries(rep.e2)) {
      if (!ph) { log(`         ${k.padEnd(10)} (constant)`); continue; }
      for (const [phase, s] of Object.entries(ph)) {
        log(`         ${k.padEnd(10)} ${phase.padEnd(6)} ${s.r2.toFixed(4)}  ${s.r2Null.toFixed(4)}   ${f3(s.medianT).padEnd(21)} ${s.signPersist.toFixed(2)}`);
      }
    }
  }
  if (rep.e3) {
    const e = rep.e3;
    log(`E3     | ${e.sites} sites (k-NN on ${e.knnSample}); per-parameter entropy at the codec steps (bits/site): ` +
        Object.entries(e.marginalBits).map(([k, b]) => `${k} ${b.toFixed(2)}`).join(", "));
    log(`         subset            Σ marginal  TC k-NN  (bias)   TC Gaussian`);
    for (const [name, s] of Object.entries(e.subsets)) {
      log(`         ${name.padEnd(17)} ${s.marginalBits.toFixed(2).padEnd(11)} ${s.tcKnn.toFixed(2).padEnd(8)} (${s.tcBias.toFixed(2)})   ${f3(s.tcGaussian)}`);
    }
    log(`         color PCA: first axis ${(100 * e.colorPca.firstAxisShare).toFixed(1)}% of variance, along RGB ${e.colorPca.axis.join(", ")}`);
  }
  if (rep.audit) {
    const a = rep.audit, fmt = (o) => Object.entries(o).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ");
    log(`Audit  | color bits/site at E3's steps. Static: ${fmt(a.staticBits)} (Y ${a.perPlaneStatic.Y.toFixed(2)}, ` +
        `Co ${a.perPlaneStatic.Co.toFixed(2)}, Cg ${a.perPlaneStatic.Cg.toFixed(2)})`);
    const names = { perImage: "per image", pooled: "pooled", perImageKlt: "per image, KLT", pooledKlt: "pooled, KLT",
                    pooledTrainKlt: "pooled, train KLT" };
    for (const [name, label] of Object.entries(names)) {
      const b = a[name];
      if (!b) continue;
      log(`         codec ${label.padEnd(17)}: color ${b.perSite.color.toFixed(2)} | ${fmt(b.perSite)}` +
          (b.modes ? ` | modes ${Object.entries(b.modes).map(([k, m]) => `${k}:${m[0]}`).join(" ")}` : ""));
    }
  }
}

async function main() {
  const { values: cli } = parseArgs({
    options: { ...DATASET_OPTIONS, out: { type: "string" }, help: { type: "boolean", short: "h", default: false } },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write("Usage: node probe.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [--from n] [--count n]\n" +
                         "                      [--preset json] [--set key=value …] [--out json]\n");
    process.exit(cli.help ? 0 : 1);
  }
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label}`);
  const rep = await probe({ device, shaders }, args, images);
  printReport(rep);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/probe_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, args, ...rep }, null, 1) + "\n");
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
