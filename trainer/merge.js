// Merging neighbouring sites, for a count schedule going down (trainer.js
// _mergeTo).
//
// Each pixel goes to its owner (the candidate with the largest logit, as in
// site_stats.wgsl) and its runner-up; a pair of sites is adjacent where one
// owns pixels the other is runner-up on. With each cell at its mean target
// color, merging cells i and j (nᵢ, nⱼ pixels, means μᵢ, μⱼ) raises the
// squared error by Ward's criterion nᵢnⱼ/(nᵢ + nⱼ)·‖μᵢ − μⱼ‖². Merges are
// taken cheapest first, each site in at most one per step.
//
// The merged site is a weighted average of the pair's parameters (position,
// log τ, radius, RGB color, log anisotropy), and of their anisotropy axes as
// orientations: dir and −dir render the same, so the axes are averaged as
// doubled angles. The weights are either the cells' pixel counts or, per
// parameter, each site's precision (mergePrecision): pixel counts assume every
// pixel carries the same information, precision weighs a parameter by how
// much the loss cares about it.

import { SITE_FLOATS, ADAM_FLOATS } from "./params.js?v=20261019";

// Logit of the site at sites[b..b+SITE_FLOATS) at pixel (x, y): the CPU copy
// of site_logit in sad_shared.wgsl.
export function siteLogit(sites, b, x, y, invScale) {
  const dx = x - sites[b], dy = y - sites[b + 1];
  const dirX = sites[b + 7], dirY = sites[b + 8];
  const proj = dirX * dx + dirY * dy;
  const perp2 = Math.max(dx * dx + dy * dy - proj * proj, 0);
  const l1 = Math.exp(sites[b + 9]);
  const d2 = (l1 * proj * proj + perp2 / l1) * invScale * invScale;
  return -Math.exp(sites[b + 2]) * (Math.sqrt(Math.max(d2, 1e-8)) - sites[b + 3] * invScale);
}

// Per-cell moments of the target and the adjacency of sites [0, n): for every
// masked pixel, its owner and runner-up among the pixel's 8 candidates and
// `global` (the site every pixel blends, as cand_ids in sad_shared.wgsl; -1 =
// none). Merging leaves `global` out: a layer behind everything would be the
// runner-up almost everywhere and hide which sites are neighbours. `sites` holds RGB colors; `cand` is 8 u32 site ids per candidate pixel.
// Returns { count, sum (RGB), pairs: Map of
// "i,j" (i < j) → pixels where one owns and the other is runner-up }.
export function cellMoments({ width, height, sites, n, cand, candWidth, candHeight, target, mask, global = -1 }) {
  const count = new Float64Array(n), sum = new Float64Array(n * 3);
  const pairs = new Map();
  const invScale = 1 / Math.max(width, height);
  for (let y = 0; y < height; y += 1) {
    const cy = Math.min(Math.floor((y * candHeight) / height), candHeight - 1);
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      if (mask && mask[p] <= 0) continue;
      const cx = Math.min(Math.floor((x * candWidth) / width), candWidth - 1);
      const cb = (cy * candWidth + cx) * 8;
      let best = -1, second = -1, l0 = -Infinity, l1 = -Infinity;
      for (let k = 0; k < 9; k += 1) {
        const id = k < 8 ? cand[cb + k] : global;
        if (id < 0 || id >= n || id === best || id === second) continue;
        const b = id * SITE_FLOATS;
        if (!(sites[b] >= 0)) continue;
        const l = siteLogit(sites, b, x, y, invScale);
        if (!Number.isFinite(l)) continue;
        if (l > l0) { second = best; l1 = l0; best = id; l0 = l; } else if (l > l1) { second = id; l1 = l; }
      }
      if (best < 0) continue;
      count[best] += 1;
      for (let c = 0; c < 3; c += 1) {
        const t = target[p * 4 + c];
        sum[best * 3 + c] += t;
      }
      if (second >= 0) {
        const key = best < second ? `${best},${second}` : `${second},${best}`;
        pairs.set(key, (pairs.get(key) || 0) + 1);
      }
    }
  }
  return { count, sum, pairs };
}

// Ward's criterion for merging cells i and j (0..1 colors, summed over RGB).
export function mergeCost({ count, sum }, i, j) {
  const ni = count[i], nj = count[j];
  if (!(ni > 0) || !(nj > 0)) return 0;
  let d2 = 0;
  for (let c = 0; c < 3; c += 1) d2 += (sum[i * 3 + c] / ni - sum[j * 3 + c] / nj) ** 2;
  return (ni * nj) / (ni + nj) * d2;
}

// The `count` cheapest merges, each site at most once: [{ keep, drop, cost }]
// with `keep` the larger cell. Sites in `pinned` (a Set, e.g. the histogram
// layer) never merge: what a layer would take over depends on its depth,
// which Ward's criterion knows nothing about.
export function pickMerges(moments, count, pinned = new Set()) {
  const options = [];
  for (const key of moments.pairs.keys()) {
    const [i, j] = key.split(",").map(Number);
    if (pinned.has(i) || pinned.has(j)) continue;
    options.push({ i, j, cost: mergeCost(moments, i, j) });
  }
  options.sort((a, b) => a.cost - b.cost);
  const used = new Set();
  const merges = [];
  for (const { i, j, cost } of options) {
    if (merges.length >= count) break;
    if (used.has(i) || used.has(j)) continue;
    used.add(i);
    used.add(j);
    const keep = moments.count[i] >= moments.count[j] ? i : j;
    merges.push({ keep, drop: keep === i ? j : i, cost });
  }
  return merges;
}

// Adam second-moment slot (adam.wgsl) holding each site float's v; position
// and the axis share one v per pair, and the color slots are the run's
// color-space params, so every RGB channel takes their mean.
const V_SLOT = [2, 2, 5, 7, null, null, null, 16, 16, 19];
const V_COLOR = [11, 12, 13];

// Bias-corrected Adam second moment v̂ of each of site i's floats at step t:
// the mean squared gradient, a diagonal estimate of the Fisher information.
// adam[20] is the step the site's moments were reset at.
export function sitePrecision(adam, i, t, beta2) {
  const b = i * ADAM_FLOATS;
  const tt = Math.max(1, t - adam[b + 20]);
  const corr = 1 / (1 - beta2 ** tt);
  const color = V_COLOR.reduce((acc, s) => acc + adam[b + s], 0) / 3;
  return V_SLOT.map((s) => (s === null ? color : adam[b + s]) * corr);
}

// Per-float merge weights for sites a and b: their precisions, or their
// pixel counts na, nb where neither has any (fresh moments, or a NaN).
export function mergePrecision(adam, a, b, t, beta2, na, nb) {
  const pa = sitePrecision(adam, a, t, beta2), pb = sitePrecision(adam, b, t, beta2);
  for (let k = 0; k < SITE_FLOATS; k += 1) {
    if (!(pa[k] + pb[k] > 0) || !Number.isFinite(pa[k] + pb[k])) { pa[k] = na; pb[k] = nb; }
  }
  return [pa, pb];
}

// Weighted average of sites a and b (RGB colors), as a new SITE_FLOATS array.
// wa, wb are numbers (pixel counts) or SITE_FLOATS per-float weights
// (mergePrecision). Anisotropy axes average as doubled angles, keeping the
// pair's weighted mean axis length.
export function mergedSite(sites, a, b, wa, wb) {
  const A = a * SITE_FLOATS, B = b * SITE_FLOATS;
  const frac = (k) => {
    const x = typeof wa === "number" ? wa : wa[k], y = typeof wb === "number" ? wb : wb[k];
    return x + y > 0 ? x / (x + y) : 0.5;
  };
  const out = new Float32Array(SITE_FLOATS);
  for (let k = 0; k < SITE_FLOATS; k += 1) {
    const w = frac(k);
    out[k] = w * sites[A + k] + (1 - w) * sites[B + k];
  }
  const w = frac(7);
  const len = (o) => Math.hypot(sites[o + 7], sites[o + 8]);
  const ang = (o) => Math.atan2(sites[o + 8], sites[o + 7]);
  const cx = w * Math.cos(2 * ang(A)) + (1 - w) * Math.cos(2 * ang(B));
  const sy = w * Math.sin(2 * ang(A)) + (1 - w) * Math.sin(2 * ang(B));
  const theta = Math.hypot(cx, sy) > 1e-6 ? Math.atan2(sy, cx) / 2 : ang(w >= 0.5 ? A : B);
  const l = w * len(A) + (1 - w) * len(B);
  out[7] = l * Math.cos(theta);
  out[8] = l * Math.sin(theta);
  return out;
}
