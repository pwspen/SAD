// Greedy site insertion scored on the training loss (growMode "insert").
//
// The render is a softmax over each pixel's 8 candidate sites, with logit
// -τ·(‖x − p‖_A / L − r / L) (L = long side; see voronoi_dmix2_fast). Adding
// one site with logit z and color c to a pixel whose current logits have max m
// and Σ exp(logit − m) = s changes its prediction in closed form:
//   w = e^(z−m) / (s + e^(z−m)),   pred′ = (1 − w)·pred + w·c,
// so the change in loss from a candidate site can be computed exactly, pixel
// by pixel, without rendering. (The render keeps only 8 candidates per pixel;
// a 9th is ignored here.)
//
// Candidates: positions at pixel centres (every pixel on small images, else
// the pixels with the largest residual), each with a few shapes: the shape of
// the site that owns the pixel (straight boundaries against it), and
// optionally sharper discs (DISC_RADII), which can close off a region inside
// another site's cell. Color: the per-channel median
// of the target over the pixels the new site would win (w > ½), the L1
// optimum for that region. Picks are taken one at a time, best first; after
// each, the per-pixel state is updated as if the site were there, and the
// candidates it affects are rescored.

import { SITE_FLOATS } from "./params.js?v=lr-decay-20261018";

const SMALL_IMAGE = 16384;    // pixels; at or below, every pixel is a candidate and windows are the whole image
const MAX_POSITIONS = 1024;   // candidate positions on larger images
const MAX_WINDOW_SAMPLES = 4096;
const MIN_RESIDUAL = 0.5 / 255; // summed over RGB; positions below this can't gain
const DISC_LOG_TAU = 1;
// Sharper disc candidates (radius × new site spacing). Off: on MNIST (20
// digits, 8 and 32 sites) they cost 3.7 dB of final PSNR. A disc wins on
// insertion but its edges (log τ + 1) are too sharp for the fit to move it
// afterwards. Set e.g. [0.25, 0.5] to try them again.
const DISC_RADII = [];

// Per-pixel softmax state of the current sites: m (max logit), s (Σ exp(logit
// − m)), pred (RGB), owner (top candidate). `sites` holds RGB colors.
export function insertionField({ width, height, sites, siteCount, cand, candWidth, candHeight }) {
  const n = width * height;
  const m = new Float32Array(n), s = new Float32Array(n), pred = new Float32Array(n * 3);
  const owner = new Int32Array(n).fill(-1);
  const invScale = 1 / Math.max(width, height);
  const logits = new Float64Array(8), ids = new Int32Array(8);
  for (let y = 0; y < height; y += 1) {
    const cy = Math.min(Math.floor((y * candHeight) / height), candHeight - 1);
    for (let x = 0; x < width; x += 1) {
      const cx = Math.min(Math.floor((x * candWidth) / width), candWidth - 1);
      const cb = (cy * candWidth + cx) * 8;
      let count = 0, mx = -Infinity, best = -1;
      for (let k = 0; k < 8; k += 1) {
        const id = cand[cb + k];
        if (id >= siteCount) continue;
        const b = id * SITE_FLOATS;
        if (!(sites[b] >= 0)) continue;
        const l = siteLogit(sites, b, x, y, invScale);
        if (!Number.isFinite(l)) continue;
        logits[count] = l; ids[count] = id; count += 1;
        if (l > mx) { mx = l; best = id; }
      }
      const p = y * width + x;
      if (!count) { m[p] = -1e30; continue; }
      let sum = 0, r = 0, g = 0, bl = 0;
      for (let k = 0; k < count; k += 1) {
        const e = Math.exp(logits[k] - mx);
        const b = ids[k] * SITE_FLOATS;
        sum += e; r += e * sites[b + 4]; g += e * sites[b + 5]; bl += e * sites[b + 6];
      }
      m[p] = mx; s[p] = sum; owner[p] = best;
      pred[p * 3] = r / sum; pred[p * 3 + 1] = g / sum; pred[p * 3 + 2] = bl / sum;
    }
  }
  return { m, s, pred, owner };
}

function siteLogit(sites, b, x, y, invScale) {
  const dx = x - sites[b], dy = y - sites[b + 1];
  const dirX = sites[b + 7], dirY = sites[b + 8];
  const proj = dirX * dx + dirY * dy;
  const diff2 = dx * dx + dy * dy;
  const perp2 = Math.max(diff2 - proj * proj, 0);
  const l1 = Math.exp(sites[b + 9]);
  const d2 = (l1 * proj * proj + perp2 / l1) * invScale * invScale;
  return -Math.exp(sites[b + 2]) * (Math.sqrt(Math.max(d2, 1e-8)) - sites[b + 3] * invScale);
}

// Isotropic candidate logit at pixel (x, y).
function candLogit(c, x, y, invScale) {
  const dx = x - c.x, dy = y - c.y;
  const d2 = (dx * dx + dy * dy) * invScale * invScale;
  return -c.tau * (Math.sqrt(Math.max(d2, 1e-8)) - c.radius * invScale);
}

// Up to `count` new sites (RGB, SITE_FLOATS each) whose exact gain in the
// training loss beats `threshold`. loss: "l1" (8-bit levels) or "l2" (levels²),
// summed over RGB and masked pixels. Returns [{ site, gain, gainL2 }], in pick
// order; gainL2 is the same pick's squared-error gain (for diagnostics).
export function pickInsertions({
  width, height, target, mask, sites, field, count, threshold, loss, spacing,
}) {
  const n = width * height;
  const invScale = 1 / Math.max(width, height);
  const small = n <= SMALL_IMAGE;
  const { m, s, pred, owner } = field;
  const L1 = loss === "l1";
  const pixelLoss = L1 ? (e) => Math.abs(e) * 255 : (e) => e * e * 65025;
  const maskAt = (p) => (mask ? mask[p] : 1);

  const residual = (p) => {
    let r = 0;
    for (let c = 0; c < 3; c += 1) r += Math.abs(target[p * 4 + c] - pred[p * 3 + c]);
    return r;
  };
  let positions = [];
  for (let p = 0; p < n; p += 1) if (maskAt(p) > 0 && owner[p] >= 0 && residual(p) > MIN_RESIDUAL) positions.push(p);
  if (!small && positions.length > MAX_POSITIONS) {
    positions = positions.map((p) => [residual(p), p]).sort((a, b) => b[0] - a[0]).slice(0, MAX_POSITIONS).map((e) => e[1]);
  }
  if (!positions.length) return [];

  // Window around a candidate (whole image when small), with a stride so it
  // holds at most MAX_WINDOW_SAMPLES pixels; gains are scaled by stride².
  const half = Math.max(2, Math.ceil(2.5 * spacing));
  function windowOf(x, y) {
    if (small) return { x0: 0, y0: 0, x1: width, y1: height, step: 1 };
    const x0 = Math.max(0, x - half), x1 = Math.min(width, x + half + 1);
    const y0 = Math.max(0, y - half), y1 = Math.min(height, y + half + 1);
    const step = Math.max(1, Math.ceil(Math.sqrt(((x1 - x0) * (y1 - y0)) / MAX_WINDOW_SAMPLES)));
    return { x0, y0, x1, y1, step };
  }

  const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  const median = (h, total) => {
    let acc = 0;
    for (let v = 0; v < 256; v += 1) { acc += h[v]; if (acc * 2 >= total) return v / 255; }
    return 1;
  };

  // Exact gain of candidate `c` (x, y, tau, radius); fills c.color.
  function score(c) {
    const win = windowOf(c.x, c.y);
    for (const h of hist) h.fill(0);
    let owned = 0;
    for (let y = win.y0; y < win.y1; y += win.step) {
      for (let x = win.x0; x < win.x1; x += win.step) {
        const p = y * width + x;
        if (!(maskAt(p) > 0)) continue;
        const u = Math.exp(candLogit(c, x, y, invScale) - m[p]);
        if (u > s[p]) { // w > ½
          owned += 1;
          for (let ch = 0; ch < 3; ch += 1) hist[ch][Math.round(Math.min(1, Math.max(0, target[p * 4 + ch])) * 255)] += 1;
        }
      }
    }
    const p0 = c.y * width + c.x;
    c.color = owned
      ? hist.map((h) => median(h, owned))
      : [0, 1, 2].map((ch) => target[p0 * 4 + ch]);
    let gain = 0, gainL2 = 0;
    for (let y = win.y0; y < win.y1; y += win.step) {
      for (let x = win.x0; x < win.x1; x += win.step) {
        const p = y * width + x;
        const mk = maskAt(p);
        if (!(mk > 0)) continue;
        const u = Math.exp(candLogit(c, x, y, invScale) - m[p]);
        const w = u / (s[p] + u);
        if (!(w > 1e-6)) continue;
        for (let ch = 0; ch < 3; ch += 1) {
          const t = target[p * 4 + ch], old = pred[p * 3 + ch];
          const nw = old + w * (c.color[ch] - old);
          gain += mk * (pixelLoss(t - old) - pixelLoss(t - nw));
          gainL2 += mk * ((t - old) ** 2 - (t - nw) ** 2) * 65025;
        }
      }
    }
    const k = win.step * win.step;
    c.gain = gain * k;
    c.gainL2 = gainL2 * k;
    return c.gain;
  }

  // Candidate list: every position × shape.
  const cands = [];
  for (const p of positions) {
    const x = p % width, y = (p - x) / width;
    const ob = owner[p] * SITE_FLOATS;
    const logTau = sites[ob + 2], radius = sites[ob + 3];
    cands.push({ x, y, logTau, radius });
    for (const f of DISC_RADII) cands.push({ x, y, logTau: logTau + DISC_LOG_TAU, radius: f * spacing });
  }
  for (const c of cands) { c.tau = Math.exp(c.logTau); score(c); }

  // Adds candidate `c` to the per-pixel state.
  function apply(c) {
    const win = windowOf(c.x, c.y);
    // Update every pixel in the box (not just the strided samples).
    for (let y = win.y0; y < win.y1; y += 1) {
      for (let x = win.x0; x < win.x1; x += 1) {
        const p = y * width + x;
        const z = candLogit(c, x, y, invScale);
        const u = Math.exp(z - m[p]);
        const w = u / (s[p] + u);
        for (let ch = 0; ch < 3; ch += 1) pred[p * 3 + ch] += w * (c.color[ch] - pred[p * 3 + ch]);
        if (z > m[p]) { s[p] = s[p] * Math.exp(m[p] - z) + 1; m[p] = z; }
        else s[p] += u;
      }
    }
  }

  const picks = [];
  while (picks.length < count) {
    let best = null;
    for (const c of cands) if (!c.taken && (!best || c.gain > best.gain)) best = c;
    if (!best || !(best.gain > threshold)) break;
    best.taken = true;
    apply(best);
    const site = new Float32Array(SITE_FLOATS);
    site.set([best.x, best.y, best.logTau, best.radius, ...best.color, 1, 0, 0]);
    picks.push({ site, gain: best.gain, gainL2: best.gainL2 });
    // Rescore candidates whose window overlaps the pick's.
    for (const c of cands) {
      if (c.taken) continue;
      if (small || (Math.abs(c.x - best.x) <= 2 * half && Math.abs(c.y - best.y) <= 2 * half)) score(c);
    }
  }
  return picks;
}
