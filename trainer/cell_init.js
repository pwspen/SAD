// Cell init (args.init "cells"): every starting parameter from one fit.
//
// As τ → ∞ the model is a hard partition into flat-colored cells, and the
// starting sites fit that limit's squared error:
//   - Where sites go: a flat cell of area A costs about its pixels' variance
//     at that scale, V(x)·A per unit area, with V the local variance at the
//     mean site spacing S. Minimizing ∫ V/λ over the site density λ, with
//     ∫ λ = the site count, gives λ ∝ √V. Seeds are R2 points accepted with
//     probability λ / max λ.
//   - Each site's cell is its Voronoi cell, and each site takes the
//     squared-error fit to its cell: color the cell's mean, and the shape
//     from the cell's spatial covariance Σ (eigenvalues λ₁ ≥ λ₂, major
//     axis u):
//       direction      u
//       log anisotropy ½ log(λ₂ / λ₁): the metric whose level sets are Σ's
//                      ellipses
//       radius         2 (λ₁λ₂)^¼: the cell's edge in that metric (a uniform
//                      ellipse with semi-axes a, b has λ = a² / 4, b² / 4), so
//                      neighbouring cells of different sizes meet at their
//                      edges, not halfway between their sites.
//     args.initIters > 0 first runs that many Lloyd passes (each site to
//     its cell's λ-weighted centroid, a centroidal Voronoi diagram for λ).
//     They assign by plain distance: feeding the fitted metric and radius
//     back into the assignment runs away (a stretched metric stretches its
//     cell further each pass, a larger radius grows its cell). On Kodak they
//     didn't help (the centroidal step moves geometry, not color error).
//   - log τ, the one parameter the limit leaves open (it is ∞ there): one
//     value per image, the minimizer of the soft model's own squared error
//     over a grid of log τ, each sampled pixel blending its 8 sites of least
//     d − r as the candidate lists would.
// Tiled runs (batch.js) fit each tile on its own, as a lone run would.

import { SITE_FLOATS } from "./params.js";

const TAU_GRID = { lo: 0, hi: 10, step: 0.25 }; // log τ
const TAU_SAMPLES = 60000; // pixels sampled per region for the τ search
const NEAREST = 8;

// Sites for every region, in RGB, positions in atlas pixels. `count` sites per
// region; tiled runs leave slot `skipSlot` of each tile for the layer.
// Returns { sites, info: [{ logTau, active }] per region }.
export function cellInit({ target, mask, width, height, count, tiling = null, skipSlot = -1, iters = 10 }) {
  const regions = [];
  if (tiling) {
    for (let t = 0; t < tiling.tiles; t += 1) {
      regions.push({ x0: (t % tiling.tilesX) * tiling.tileW, y0: Math.floor(t / tiling.tilesX) * tiling.tileH,
                     w: tiling.tileW, h: tiling.tileH, base: t * tiling.sitesPerTile });
    }
  } else {
    regions.push({ x0: 0, y0: 0, w: width, h: height, base: 0 });
  }
  const scale = tiling ? Math.max(tiling.tileW, tiling.tileH) : Math.max(width, height);
  const slots = tiling ? regions.length * tiling.sitesPerTile : count;
  const sites = new Float32Array(slots * SITE_FLOATS);
  const info = [];
  for (const reg of regions) {
    const fit = fitRegion(target, mask, width, reg, count, iters);
    const logTau = searchLogTau(target, mask, width, reg, fit, scale);
    let slot = reg.base;
    fit.comps.forEach((c) => {
      if (slot - reg.base === skipSlot) slot += 1;
      writeSite(sites, slot, c, logTau);
      slot += 1;
    });
    info.push({ logTau, active: fit.comps.filter((c) => c.active).length });
  }
  return { sites, info };
}

function fitRegion(target, mask, width, reg, count, iters) {
  const { x0, y0, w, h } = reg;
  const n = w * h;
  const pix = (i) => (y0 + Math.floor(i / w)) * width + x0 + (i % w);
  const weight = new Float32Array(n);
  let area = 0;
  for (let i = 0; i < n; i += 1) {
    const m = mask ? mask[pix(i)] : 1;
    weight[i] = m > 0 ? m : 0;
    area += weight[i] > 0 ? 1 : 0;
  }
  const comps = [];
  if (!area || !count) return { comps: Array.from({ length: count }, () => ({ active: false })) };

  // Density λ ∝ √(local variance at the site spacing), as sites per pixel.
  const spacing = Math.sqrt(area / count);
  const density = localStd(target, weight, width, reg, Math.max(1, Math.round(spacing / 2)));
  let sum = 0, peak = 0;
  for (let i = 0; i < n; i += 1) if (weight[i] > 0) sum += density[i];
  const floor = 1e-3 * sum / area; // keeps flat regions from an exactly zero density
  sum = 0;
  for (let i = 0; i < n; i += 1) {
    density[i] = weight[i] > 0 ? density[i] + floor : 0;
    sum += density[i];
    peak = Math.max(peak, density[i]);
  }
  for (let i = 0; i < n; i += 1) density[i] *= count / sum;
  peak *= count / sum;

  // Seeds: R2 points (the plastic-number sequence in 3D: x, y and the
  // acceptance draw), accepted with probability λ / max λ.
  const g3 = 1.2207440846057596, a1 = 1 / g3, a2 = 1 / (g3 * g3), a3 = 1 / (g3 * g3 * g3);
  for (let seed = 0; comps.length < count && seed < 1e7; seed += 1) {
    const px = ((0.5 + seed * a1) % 1) * (w - 1), py = ((0.5 + seed * a2) % 1) * (h - 1);
    const i = Math.floor(py) * w + Math.floor(px);
    if (((0.5 + seed * a3) % 1) * peak >= density[i]) continue;
    const v = 1 / (4 * Math.PI * density[i]); // equal-area disk of 1 / λ
    comps.push({ active: true, mx: px, my: py, sxx: v, sxy: 0, syy: v, c: [0, 0, 0] });
  }
  while (comps.length < count) comps.push({ active: false });

  const label = new Int32Array(n);
  const best = new Float64Array(n);
  for (let it = 0; it <= iters; it += 1) {
    // Assign: each pixel to its nearest site (Voronoi).
    label.fill(-1);
    best.fill(Infinity);
    comps.forEach((c, k) => {
      if (!c.active) return;
      // Out to twice the cell's semi-major axis, 2√λ₁.
      const ext = 4 * Math.sqrt(shapeOf(c).major) + 2;
      const xa = Math.max(0, Math.floor(c.mx - ext)), xb = Math.min(w - 1, Math.ceil(c.mx + ext));
      const ya = Math.max(0, Math.floor(c.my - ext)), yb = Math.min(h - 1, Math.ceil(c.my + ext));
      for (let y = ya; y <= yb; y += 1) {
        const dy = y - c.my;
        for (let x = xa; x <= xb; x += 1) {
          const i = y * w + x;
          if (!(weight[i] > 0)) continue;
          const dx = x - c.mx;
          const e = dx * dx + dy * dy;
          if (e < best[i]) { best[i] = e; label[i] = k; }
        }
      }
    });

    // Refit. Moments per cell: Σλ, Σλx, Σλy (centroid); n, Σx, Σy, Σx²,
    // Σxy, Σy² (shape); Σt (color).
    const K = comps.length, C = 12;
    const M = new Float64Array(K * C);
    for (let i = 0; i < n; i += 1) {
      const k = label[i];
      if (k < 0) continue;
      const m = weight[i], d = density[i] * m, x = i % w, y = Math.floor(i / w), p = pix(i) * 4, b = k * C;
      M[b] += d; M[b + 1] += d * x; M[b + 2] += d * y;
      M[b + 3] += m; M[b + 4] += m * x; M[b + 5] += m * y;
      M[b + 6] += m * x * x; M[b + 7] += m * x * y; M[b + 8] += m * y * y;
      M[b + 9] += m * target[p]; M[b + 10] += m * target[p + 1]; M[b + 11] += m * target[p + 2];
    }
    comps.forEach((c, k) => {
      if (!c.active) return;
      const b = k * C, nk = M[b + 3];
      if (!(nk > 0)) return; // no pixels this pass: keep its place and shape
      const ux = M[b + 4] / nk, uy = M[b + 5] / nk;
      // Plus 1/12 px² for pixel quantization (as siteSplit).
      c.sxx = M[b + 6] / nk - ux * ux + 1 / 12;
      c.sxy = M[b + 7] / nk - ux * uy;
      c.syy = M[b + 8] / nk - uy * uy + 1 / 12;
      c.c = [M[b + 9] / nk, M[b + 10] / nk, M[b + 11] / nk];
      if (it < iters && M[b] > 0) { c.mx = M[b + 1] / M[b]; c.my = M[b + 2] / M[b]; }
    });
  }
  for (const c of comps) if (c.active) { c.mx += x0; c.my += y0; }
  return { comps };
}

// Per pixel of the region: √ of the local color variance (summed over RGB)
// in a (2r + 1)² box, over masked pixels.
function localStd(target, weight, width, reg, r) {
  const { x0, y0, w, h } = reg;
  const n = w * h;
  // Box sums by summed-area tables of w, w·t and w·|t|².
  const W = new Float64Array((w + 1) * (h + 1)), T = [0, 1, 2].map(() => new Float64Array((w + 1) * (h + 1)));
  const Q = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x, m = weight[i], p = ((y0 + y) * width + x0 + x) * 4;
      const o = (y + 1) * (w + 1) + x + 1, up = o - (w + 1);
      W[o] = m + W[o - 1] + W[up] - W[up - 1];
      let q = 0;
      for (let c = 0; c < 3; c += 1) {
        const t = target[p + c];
        T[c][o] = m * t + T[c][o - 1] + T[c][up] - T[c][up - 1];
        q += t * t;
      }
      Q[o] = m * q + Q[o - 1] + Q[up] - Q[up - 1];
    }
  }
  const box = (A, xa, ya, xb, yb) => A[(yb + 1) * (w + 1) + xb + 1] - A[ya * (w + 1) + xb + 1] - A[(yb + 1) * (w + 1) + xa] + A[ya * (w + 1) + xa];
  const out = new Float32Array(n);
  for (let y = 0; y < h; y += 1) {
    const ya = Math.max(0, y - r), yb = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x += 1) {
      const xa = Math.max(0, x - r), xb = Math.min(w - 1, x + r);
      const sw = box(W, xa, ya, xb, yb);
      if (!(sw > 0)) continue;
      let v = box(Q, xa, ya, xb, yb) / sw;
      for (let c = 0; c < 3; c += 1) v -= (box(T[c], xa, ya, xb, yb) / sw) ** 2;
      out[y * w + x] = Math.sqrt(Math.max(0, v));
    }
  }
  return out;
}

// Major axis, log anisotropy and radius from a component's covariance.
function shapeOf(c) {
  const tr = c.sxx + c.syy, gap = Math.sqrt(Math.max(0, ((c.sxx - c.syy) / 2) ** 2 + c.sxy * c.sxy));
  const l1 = tr / 2 + gap, l2 = Math.max(1e-6, tr / 2 - gap);
  let ux = c.sxy, uy = l1 - c.sxx;
  if (Math.hypot(ux, uy) < 1e-12) { ux = l1 - c.syy; uy = c.sxy; }
  const ul = Math.hypot(ux, uy);
  if (ul < 1e-12) { ux = 1; uy = 0; } else { ux /= ul; uy /= ul; }
  return { ux, uy, major: l1, logAniso: 0.5 * Math.log(l2 / l1), radius: 2 * Math.pow(l1 * l2, 0.25) };
}

function writeSite(sites, slot, c, logTau) {
  const o = slot * SITE_FLOATS;
  if (!c.active) { sites[o] = -1; sites[o + 1] = -1; return; }
  const s = shapeOf(c);
  sites[o] = c.mx; sites[o + 1] = c.my;
  sites[o + 2] = logTau;
  sites[o + 3] = s.radius;
  sites[o + 4] = c.c[0]; sites[o + 5] = c.c[1]; sites[o + 6] = c.c[2];
  sites[o + 7] = s.ux; sites[o + 8] = s.uy;
  sites[o + 9] = s.logAniso;
}

// The log τ (one per region) minimizing the soft model's squared error with
// the fitted sites: each sampled pixel blends its 8 sites of least d − r
// (site_logit's −τ(d − r) / scale), as the candidate lists would offer.
function searchLogTau(target, mask, width, reg, fit, scale) {
  const sites = fit.comps.filter((c) => c.active).map((c) => {
    const s = shapeOf(c);
    return { c, s: { ...s, l: Math.exp(s.logAniso) } };
  });
  if (!sites.length) return 0;
  const { x0, y0, w, h } = reg;
  const stride = Math.max(1, Math.round(Math.sqrt((w * h) / TAU_SAMPLES)));
  // Sites bucketed on a grid; a pixel looks through buckets out to the
  // largest radius plus a few spacings.
  const spacing = Math.sqrt((w * h) / sites.length);
  const bucket = Math.max(1, spacing);
  const bw = Math.ceil(w / bucket), bh = Math.ceil(h / bucket);
  const buckets = Array.from({ length: bw * bh }, () => []);
  let maxR = 0;
  sites.forEach((e, k) => {
    const bx = Math.min(bw - 1, Math.floor((e.c.mx - x0) / bucket));
    const by = Math.min(bh - 1, Math.floor((e.c.my - y0) / bucket));
    buckets[by * bw + bx].push(k);
    maxR = Math.max(maxR, e.s.radius * Math.exp(Math.abs(e.s.logAniso) / 2));
  });
  const reach = Math.ceil((maxR + 3 * spacing) / bucket);
  const samples = []; // per pixel: target rgb, mask, then NEAREST (g, site)
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const p = (y0 + y) * width + x0 + x;
      const m = mask ? mask[p] : 1;
      if (!(m > 0)) continue;
      const bx = Math.min(bw - 1, Math.floor(x / bucket)), by = Math.min(bh - 1, Math.floor(y / bucket));
      // The NEAREST least d − r, kept sorted by insertion.
      const near = [];
      for (let j = Math.max(0, by - reach); j <= Math.min(bh - 1, by + reach); j += 1) {
        for (let i = Math.max(0, bx - reach); i <= Math.min(bw - 1, bx + reach); i += 1) {
          for (const k of buckets[j * bw + i]) {
            const { c, s } = sites[k];
            const dx = x0 + x - c.mx, dy = y0 + y - c.my;
            const proj = s.ux * dx + s.uy * dy, perp2 = Math.max(0, dx * dx + dy * dy - proj * proj);
            const g = Math.sqrt(s.l * proj * proj + perp2 / s.l) - s.radius;
            if (near.length === NEAREST && g >= near[NEAREST - 1][0]) continue;
            let q = Math.min(near.length, NEAREST - 1);
            near[q] = [g, k];
            while (q > 0 && near[q - 1][0] > g) { [near[q - 1], near[q]] = [near[q], near[q - 1]]; q -= 1; }
          }
        }
      }
      samples.push({ t: [target[p * 4], target[p * 4 + 1], target[p * 4 + 2]], m, near });
    }
  }
  const loss = (logTau) => {
    const k = Math.exp(logTau) / scale;
    let total = 0;
    for (const { t, m, near } of samples) {
      const g0 = near[0][0];
      let sw = 0, r = 0, gg = 0, b = 0;
      for (const [gk, idx] of near) {
        const wk = Math.exp(-k * (gk - g0));
        const col = sites[idx].c.c;
        sw += wk; r += wk * col[0]; gg += wk * col[1]; b += wk * col[2];
      }
      total += m * ((t[0] - r / sw) ** 2 + (t[1] - gg / sw) ** 2 + (t[2] - b / sw) ** 2);
    }
    return total;
  };
  let best = TAU_GRID.lo, bestLoss = Infinity;
  const losses = [];
  for (let lt = TAU_GRID.lo; lt <= TAU_GRID.hi + 1e-9; lt += TAU_GRID.step) {
    const l = loss(lt);
    losses.push([lt, l]);
    if (l < bestLoss) { bestLoss = l; best = lt; }
  }
  // Parabola through the best grid point and its neighbours.
  const j = losses.findIndex(([lt]) => lt === best);
  if (j > 0 && j < losses.length - 1) {
    const [[, fa], [, fb], [, fc]] = losses.slice(j - 1, j + 2);
    const den = fa - 2 * fb + fc;
    if (den > 0) best += 0.5 * TAU_GRID.step * (fa - fc) / den;
  }
  return best;
}
