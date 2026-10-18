// Merging neighbouring sites (args.pruneEvery), scored by description length.
//
// Each pixel goes to its owner (the candidate with the largest logit, as in
// site_stats.wgsl) and its runner-up; a pair of sites is adjacent where one
// owns pixels the other is runner-up on. With each cell at its mean target
// color, merging cells i and j (nᵢ, nⱼ pixels, means μᵢ, μⱼ) raises the
// squared error by Ward's criterion nᵢnⱼ/(nᵢ + nⱼ)·‖μᵢ − μⱼ‖². A merge
// saves BITS_PER_SITE bits and costs (S/2)·log2(1 + ΔE/E) bits of residual
// (Gaussian at 8-bit precision, E the summed squared error of all cells, S
// samples; the model of trainer.js's cellMdlBits), so it pays if
// ΔE < E·(2^(2B/S) − 1). Merges are taken cheapest first, each site in at most
// one per step, with E growing by each accepted ΔE.
//
// The merged site is the pixel-count-weighted average of the pair's
// parameters (position, log τ, radius, RGB color, log anisotropy), and of
// their anisotropy axes as orientations: dir and −dir render the same, so
// the axes are averaged as doubled angles.

import { SITE_FLOATS } from "./params.js?v=lr-decay-20261018";

// Logit of site `b` (sites[b..b+SITE_FLOATS)) at pixel (x, y), as rendered.
function siteLogit(sites, b, x, y, invScale) {
  const dx = x - sites[b], dy = y - sites[b + 1];
  const dirX = sites[b + 7], dirY = sites[b + 8];
  const proj = dirX * dx + dirY * dy;
  const perp2 = Math.max(dx * dx + dy * dy - proj * proj, 0);
  const l1 = Math.exp(sites[b + 9]);
  const d2 = (l1 * proj * proj + perp2 / l1) * invScale * invScale;
  return -Math.exp(sites[b + 2]) * (Math.sqrt(Math.max(d2, 1e-8)) - sites[b + 3] * invScale);
}

// Per-cell moments of the target and the adjacency of sites [0, n): for every
// masked pixel, its owner and runner-up among the pixel's 8 candidates.
// `sites` holds RGB colors; `cand` is 8 u32 site ids per candidate pixel.
// Returns { count, sum (RGB), sumSq (|t|², summed over RGB), pairs: Map of
// "i,j" (i < j) → pixels where one owns and the other is runner-up }.
export function cellMoments({ width, height, sites, n, cand, candWidth, candHeight, target, mask }) {
  const count = new Float64Array(n), sum = new Float64Array(n * 3), sumSq = new Float64Array(n);
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
      for (let k = 0; k < 8; k += 1) {
        const id = cand[cb + k];
        if (id >= n || id === best || id === second) continue;
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
        sumSq[best] += t * t;
      }
      if (second >= 0) {
        const key = best < second ? `${best},${second}` : `${second},${best}`;
        pairs.set(key, (pairs.get(key) || 0) + 1);
      }
    }
  }
  return { count, sum, sumSq, pairs };
}

// Summed squared error (0..1 colors, summed over RGB) of every cell at its
// mean color.
export function cellSse({ count, sum, sumSq }) {
  let e = 0;
  for (let i = 0; i < count.length; i += 1) {
    if (!(count[i] > 0)) continue;
    let s = sumSq[i];
    for (let c = 0; c < 3; c += 1) s -= sum[i * 3 + c] ** 2 / count[i];
    e += Math.max(0, s);
  }
  return e;
}

// Ward's criterion for merging cells i and j (0..1 colors, summed over RGB).
export function mergeCost({ count, sum }, i, j) {
  const ni = count[i], nj = count[j];
  if (!(ni > 0) || !(nj > 0)) return 0;
  let d2 = 0;
  for (let c = 0; c < 3; c += 1) d2 += (sum[i * 3 + c] / ni - sum[j * 3 + c] / nj) ** 2;
  return (ni * nj) / (ni + nj) * d2;
}

// The merges that pay for themselves, cheapest first, each site at most once:
// [{ keep, drop, cost }] with `keep` the larger cell. `samples` = masked
// pixels × 3. With `count`, the `count` cheapest merges instead, paying or
// not (a count schedule going down).
export function pickMerges(moments, samples, bitsPerSite, { count = null } = {}) {
  const frac = Math.pow(2, (2 * bitsPerSite) / Math.max(1, samples)) - 1;
  let err = cellSse(moments);
  const options = [];
  for (const key of moments.pairs.keys()) {
    const [i, j] = key.split(",").map(Number);
    options.push({ i, j, cost: mergeCost(moments, i, j) });
  }
  options.sort((a, b) => a.cost - b.cost);
  const used = new Set();
  const merges = [];
  for (const { i, j, cost } of options) {
    if (count != null ? merges.length >= count : !(cost < err * frac)) break;
    if (used.has(i) || used.has(j)) continue;
    used.add(i);
    used.add(j);
    err += cost;
    const keep = moments.count[i] >= moments.count[j] ? i : j;
    merges.push({ keep, drop: keep === i ? j : i, cost });
  }
  return merges;
}

// Pixel-count-weighted average of sites a and b (RGB colors), as a new
// SITE_FLOATS array. Anisotropy axes average as doubled angles, keeping the
// pair's weighted mean axis length.
export function mergedSite(sites, a, b, wa, wb) {
  const A = a * SITE_FLOATS, B = b * SITE_FLOATS;
  const w = wa + wb > 0 ? wa / (wa + wb) : 0.5;
  const out = new Float32Array(SITE_FLOATS);
  for (let k = 0; k < SITE_FLOATS; k += 1) out[k] = w * sites[A + k] + (1 - w) * sites[B + k];
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
