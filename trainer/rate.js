// Rate model for training with squared error + λ · bits (shaders/wgsl/rate.wgsl
// applies its gradient). No DOM, no GPU.
//
// The coded planes are the codec's (codec.js): log_tau, radius, the three
// color coordinates along a KLT basis (pcaBasis; the codec takes the same one
// as colorBasis), log_aniso, and θ. Each of the first six gets a Gaussian
// mixture fit to the current sites' values; a value's bits are −log₂ of its
// quantizer bin's probability, approximated by Δ times the mixture density
// with every variance widened by the bin's Δ²/12 (capped at 1). That is what
// an adaptive model of the plane's values pays at high rate, i.e. the codec
// with its "constant" predictor coding a pooled stream (probe.mjs's audit),
// not the per-image header and model start-up. θ is coded uniformly over its
// levels, so it costs log₂ levels = log₂(π · |2 sinh a| / Δθ) when that is
// above 0 (codec.js thetaLevelsOf).
//
// Steps Δ come from codecSteps at a fixed quantization loss (args.rateLossDb):
// the rate is that of coding the current sites at that fidelity.

import { SITE_FLOATS } from "./params.js?v=20261019";

export const RATE_PLANES = ["log_tau", "radius", "K0", "K1", "K2", "log_aniso"];
const RATE_HEADER = 20;
const RATE_KMAX = 8;
const RATE_PLANE = 2 + 3 * RATE_KMAX;
export const RATE_FLOATS = RATE_HEADER + RATE_PLANES.length * RATE_PLANE;

const LOG_2PI = Math.log(2 * Math.PI);

// Each active site's plane values (RATE_PLANES order) in a basis; skips
// inactive sites and those with `skip(i)`.
export function planeValues(sites, count, basis, skip = () => false) {
  const out = RATE_PLANES.map(() => []);
  const { mean: m, axes: a } = basis;
  for (let i = 0; i < count; i += 1) {
    const b = i * SITE_FLOATS;
    if (!(sites[b] >= 0) || skip(i)) continue;
    const d = [sites[b + 4] - m[0], sites[b + 5] - m[1], sites[b + 6] - m[2]];
    out[0].push(sites[b + 2]);
    out[1].push(sites[b + 3]);
    for (let k = 0; k < 3; k += 1) out[2 + k].push(a[k][0] * d[0] + a[k][1] * d[1] + a[k][2] * d[2]);
    out[5].push(sites[b + 9]);
  }
  return out.map((v) => Float64Array.from(v));
}

// 1-D Gaussian mixture by EM: { weights, means, vars }, at most k components,
// variances at least minVar. `prev` (same shape) warm-starts it.
export function fitMixture(xs, k, minVar, prev = null, iters = 30) {
  const n = xs.length;
  if (!n) return { weights: [1], means: [0], vars: [1] };
  let w, mu, v;
  if (prev && prev.means.length === k) {
    ({ weights: w, means: mu, vars: v } = { weights: prev.weights.slice(), means: prev.means.slice(), vars: prev.vars.map((x) => Math.max(x, minVar)) });
  } else {
    const s = Float64Array.from(xs).sort();
    mu = Array.from({ length: k }, (_, j) => s[Math.min(n - 1, Math.floor(((j + 0.5) * n) / k))]);
    const spread = (s[n - 1] - s[0]) / (2 * k);
    v = mu.map(() => Math.max(minVar, spread * spread));
    w = mu.map(() => 1 / k);
  }
  const r = new Float64Array(k);
  for (let it = 0; it < iters; it += 1) {
    const sw = new Float64Array(k), sx = new Float64Array(k), sxx = new Float64Array(k);
    for (let i = 0; i < n; i += 1) {
      const x = xs[i];
      let top = -Infinity;
      for (let j = 0; j < k; j += 1) {
        r[j] = Math.log(w[j]) - 0.5 * Math.log(v[j]) - ((x - mu[j]) ** 2) / (2 * v[j]);
        if (r[j] > top) top = r[j];
      }
      let z = 0;
      for (let j = 0; j < k; j += 1) { r[j] = Math.exp(r[j] - top); z += r[j]; }
      for (let j = 0; j < k; j += 1) { const q = r[j] / z; sw[j] += q; sx[j] += q * x; sxx[j] += q * x * x; }
    }
    for (let j = 0; j < k; j += 1) {
      if (sw[j] < 1e-9) { w[j] = 1e-9; continue; }
      w[j] = sw[j] / n;
      mu[j] = sx[j] / sw[j];
      v[j] = Math.max(minVar, sxx[j] / sw[j] - mu[j] * mu[j]);
    }
  }
  return { weights: w, means: mu, vars: v };
}

// −log₂ min(1, Δ · Σ wⱼ N(x; μⱼ, vⱼ + Δ²/12)) for one plane's mixture.
function bitsOf(x, mix, step) {
  const q = (step * step) / 12;
  let top = -Infinity;
  const t = mix.means.map((m, j) => {
    const s2 = mix.vars[j] + q;
    const e = Math.log(mix.weights[j]) - 0.5 * (LOG_2PI + Math.log(s2)) - ((x - m) ** 2) / (2 * s2);
    if (e > top) top = e;
    return e;
  });
  const logP = Math.log(step) + top + Math.log(t.reduce((s, e) => s + Math.exp(e - top), 0));
  return logP >= 0 ? 0 : -logP / Math.LN2;
}

const thetaBits = (a, steps) => {
  const m = (Math.PI * Math.abs(2 * Math.sinh(a))) / steps.theta;
  return Number.isFinite(m) && m > 1 ? Math.log2(m) : 0;
};

// Fits the model to the active sites (RGB colors) at quantizer `steps`
// (codecSteps). skip(i): sites left out (the histogram layer).
export function fitRateModel(sites, count, basis, steps, { components = 4, prev = null, skip } = {}) {
  const vals = planeValues(sites, count, basis, skip);
  const stepOf = [steps.logTau, steps.radius, steps.color, steps.color, steps.color, steps.logAniso];
  const mixtures = vals.map((xs, p) => {
    const step = stepOf[p];
    if (!(step > 0) || !Number.isFinite(step)) return null;
    return fitMixture(xs, components, (step / 4) ** 2, prev?.mixtures?.[p] ?? null);
  });
  return { basis, steps, stepOf, mixtures };
}

// Estimated bits per plane (summed over the active sites) and the site count:
// { planes: { log_tau, …, θ }, total, sites }. Values are rounded to their
// quantizer bins first, as the codec codes them.
export function rateBits(model, sites, count, skip) {
  const vals = planeValues(sites, count, model.basis, skip);
  const planes = {};
  let total = 0;
  RATE_PLANES.forEach((name, p) => {
    const mix = model.mixtures[p], step = model.stepOf[p];
    let s = 0;
    if (mix) for (const x of vals[p]) s += bitsOf(Math.round(x / step) * step, mix, step);
    planes[name] = s;
    total += s;
  });
  let th = 0;
  for (const a of vals[5]) th += thetaBits(Math.round(a / model.steps.logAniso) * model.steps.logAniso, model.steps);
  planes["θ"] = th;
  total += th;
  return { planes, total, sites: vals[0].length };
}

// The rate.wgsl buffer for `model` (null = no model: the kernel adds nothing).
// u32 fields: siteCount, layerSlot (0xffffffff = none), sitesPerTile (0 = untiled).
export function rateBuffer(model, { lambda, siteCount, layerSlot = null, sitesPerTile = 0 }) {
  const f = new Float32Array(RATE_FLOATS);
  const u = new Uint32Array(f.buffer);
  f[0] = model ? lambda : 0;
  u[1] = siteCount >>> 0;
  u[2] = layerSlot === null || layerSlot === undefined ? 0xffffffff : layerSlot >>> 0;
  u[3] = sitesPerTile >>> 0;
  if (!model) return f;
  f[4] = Number.isFinite(model.steps.theta) ? model.steps.theta : 0;
  f[5] = model.steps.logAniso;
  f.set(model.basis.mean, 6);
  model.basis.axes.forEach((ax, k) => f.set(ax, 9 + 3 * k));
  model.mixtures.forEach((mix, p) => {
    const b = RATE_HEADER + p * RATE_PLANE, step = model.stepOf[p];
    if (!mix) return;
    const k = Math.min(RATE_KMAX, mix.means.length);
    f[b] = step;
    f[b + 1] = k;
    for (let j = 0; j < k; j += 1) {
      const s2 = mix.vars[j] + (step * step) / 12;
      f.set([Math.log(step) + Math.log(mix.weights[j]) - 0.5 * (LOG_2PI + Math.log(s2)), mix.means[j], s2], b + 2 + 3 * j);
    }
  });
  return f;
}
