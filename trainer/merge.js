// Merging neighbouring sites, for a count schedule going down (trainer.js
// _mergeTo). Two criteria (args.mergeCriterion):
//
// "neighbor" (candidateEdges, neighborMerge): sites are adjacent where one is
// the other's runner-up in a candidate list; each edge is weighted by how
// different the pair's parameters are, position left out. The site most like
// its top-k neighbours merges with the nearest one, repeatedly, updating only
// the merged site's edges, until the step's budget is spent.
//
// "ward" (cellMoments, pickMerges): each pixel goes to its owner (the candidate with the largest logit, as in
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

import { SITE_FLOATS, ADAM_FLOATS } from "./params.js?v=m3-20261020";

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
  return pairWeights(sitePrecision(adam, a, t, beta2), sitePrecision(adam, b, t, beta2), na, nb);
}

// Per-float weights pa, pb (modified in place), falling back to the pixel
// counts na, nb on the floats where neither has a usable weight.
function pairWeights(pa, pb, na, nb) {
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

// ---- Neighbour-similarity merging ----

// Adjacency straight from the candidate lists, no logits: each list is sorted
// by logit (vpt.wgsl), so its first two usable sites are the owner and the
// runner-up at that candidate pixel. Sites in `pinned` (the histogram layer)
// and `global` are skipped, as in cellMoments: a layer behind everything
// would be the runner-up almost everywhere. Returns { count (candidate pixels
// owned, scaled to image pixels), edges: Map a·n + b (a < b) → candidate
// pixels where the pair are owner and runner-up }.
export function candidateEdges({ sites, n, cand, candWidth, candHeight, width, height, mask, global = -1, pinned = new Set() }) {
  const count = new Float64Array(n);
  const edges = new Map();
  const area = (width * height) / (candWidth * candHeight);
  for (let cy = 0; cy < candHeight; cy += 1) {
    const y = Math.min(height - 1, Math.floor(((cy + 0.5) * height) / candHeight));
    for (let cx = 0; cx < candWidth; cx += 1) {
      if (mask) {
        const x = Math.min(width - 1, Math.floor(((cx + 0.5) * width) / candWidth));
        if (mask[y * width + x] <= 0) continue;
      }
      const cb = (cy * candWidth + cx) * 8;
      let a = -1, b = -1;
      for (let k = 0; k < 8; k += 1) {
        const id = cand[cb + k];
        if (id >= n || id === global || id === a || pinned.has(id) || !(sites[id * SITE_FLOATS] >= 0)) continue;
        if (a < 0) a = id;
        else { b = id; break; }
      }
      if (a < 0) continue;
      count[a] += area;
      if (b < 0) continue;
      const key = a < b ? a * n + b : b * n + a;
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  return { count, edges };
}

// What an edge compares, position left out: RGB color, log τ, radius, and
// the shape as its log metric tensor, log M = a·(2ddᵀ − I) for a = log
// anisotropy and unit axis d, i.e. the 2-vector a·(cos 2θ, sin 2θ). The axis
// then counts in proportion to how anisotropic the site is, and d ≡ −d.
export const EDGE_GROUPS = ["color", "log τ", "radius", "shape"];
const FEATS = 7;
const GROUP_SLICES = [[0, 3], [3, 4], [4, 5], [5, 7]];

function siteFeatures(sites, i, out) {
  const b = i * SITE_FLOATS, o = i * FEATS;
  out[o] = sites[b + 4];
  out[o + 1] = sites[b + 5];
  out[o + 2] = sites[b + 6];
  out[o + 3] = sites[b + 2];
  out[o + 4] = sites[b + 3];
  const a = sites[b + 9], theta = 2 * Math.atan2(sites[b + 8], sites[b + 7]);
  out[o + 5] = a * Math.cos(theta);
  out[o + 6] = a * Math.sin(theta);
}

// Each group's Euclidean distance between sites i and j, into out[0..4).
function groupDistances(F, i, j, out) {
  GROUP_SLICES.forEach(([s, e], g) => {
    let d2 = 0;
    for (let k = s; k < e; k += 1) d2 += (F[i * FEATS + k] - F[j * FEATS + k]) ** 2;
    out[g] = Math.sqrt(d2);
  });
}

// A group's scale: its median distance over the step's edges, so 1 is a
// typical neighbour difference in every group and no per-parameter rates are
// needed. Falls back to the mean where most edges don't differ beyond
// rounding (1e-6 of the largest difference), and leaves out a group no edge
// differs in.
function groupScale(values) {
  const sorted = Float64Array.from(values).sort();
  if (!sorted.length) return Infinity;
  const tol = 1e-6 * sorted[sorted.length - 1];
  const median = sorted[sorted.length >> 1];
  if (median > tol) return median;
  const mean = sorted.reduce((s, v) => s + v, 0) / sorted.length;
  return mean > tol && mean > 0 ? mean : Infinity;
}

// Edge weight: RMS over the groups of distance / scale (dist: 4 group
// distances). 0 = identical apart from position, ~1 = a typical neighbour.
function edgeWeight(dist, scales) {
  let s = 0;
  for (let g = 0; g < GROUP_SLICES.length; g += 1) s += (dist[g] / scales[g]) ** 2;
  return Math.sqrt(s / GROUP_SLICES.length);
}

// Mean of the k smallest values of a Map's weights (Infinity if none).
function topKMean(adj, k) {
  if (!adj.size) return Infinity;
  const ws = Array.from(adj.values()).sort((a, b) => a - b);
  const m = Math.min(k, ws.length);
  let s = 0;
  for (let i = 0; i < m; i += 1) s += ws[i];
  return s / m;
}

// Binary min-heap of [score, site, version].
class MinHeap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(x) {
    const a = this.a;
    a.push(x);
    for (let i = a.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

// Up to `budget` merges by neighbour similarity. A site's score is the mean
// weight of its `topK` lightest edges; the lowest-scoring site merges with its
// nearest neighbour (the larger cell is kept), the pair's edges move to the
// kept site and are reweighed, and the kept site's and its neighbours' scores
// are updated, so merges chain. Group scales come from the step's initial
// edges.
//
// sites: n·SITE_FLOATS (RGB colors), not modified. graph: candidateEdges.
// weights: n·SITE_FLOATS per-float merge weights (precision, or zeros where
// unknown; see mergePrecision) or null for pixel counts. A merged site's
// weight is the sum of the pair's, or 0 on floats that fell back to counts.
//
// Returns { sites (copy with the merges applied), changed (kept sites still
// alive), dropped, merges: [{ keep, drop, weight }], edgeWeights (initial,
// one per edge), scales }.
export function neighborMerge({ sites, n, graph, budget, topK = 3, weights = null }) {
  const cur = Float32Array.from(sites.subarray(0, n * SITE_FLOATS));
  const count = Float64Array.from(graph.count);
  const W = weights ? Float64Array.from(weights) : null;
  const F = new Float64Array(n * FEATS);
  for (let i = 0; i < n; i += 1) siteFeatures(cur, i, F);

  const E = graph.edges.size;
  const ea = new Int32Array(E), eb = new Int32Array(E);
  const groupD = GROUP_SLICES.map(() => new Float64Array(E));
  const dist = new Float64Array(GROUP_SLICES.length);
  let e = 0;
  for (const key of graph.edges.keys()) {
    ea[e] = Math.floor(key / n);
    eb[e] = key - ea[e] * n;
    groupDistances(F, ea[e], eb[e], dist);
    for (let g = 0; g < dist.length; g += 1) groupD[g][e] = dist[g];
    e += 1;
  }
  const scales = groupD.map(groupScale);

  const adj = Array.from({ length: n }, () => new Map());
  const edgeWeights = new Float32Array(E);
  for (let k = 0; k < E; k += 1) {
    for (let g = 0; g < dist.length; g += 1) dist[g] = groupD[g][k];
    const w = edgeWeight(dist, scales);
    edgeWeights[k] = w;
    adj[ea[k]].set(eb[k], w);
    adj[eb[k]].set(ea[k], w);
  }

  const version = new Uint32Array(n);
  const alive = new Uint8Array(n).fill(1);
  const heap = new MinHeap();
  const rescore = (i) => {
    version[i] += 1;
    const s = topKMean(adj[i], topK);
    if (Number.isFinite(s)) heap.push([s, i, version[i]]);
  };
  for (let i = 0; i < n; i += 1) if (adj[i].size) rescore(i);

  const merges = [];
  const changed = new Set(), dropped = [];
  while (merges.length < budget && heap.size) {
    const [, i, ver] = heap.pop();
    if (!alive[i] || ver !== version[i]) continue;
    let j = -1, wij = Infinity;
    for (const [nb, w] of adj[i]) if (w < wij) { wij = w; j = nb; }
    if (j < 0) continue;
    const keep = count[i] >= count[j] ? i : j, drop = keep === i ? j : i;

    let wk = count[keep], wd = count[drop];
    if (W) {
      const pk = Array.from(W.subarray(keep * SITE_FLOATS, (keep + 1) * SITE_FLOATS));
      const pd = Array.from(W.subarray(drop * SITE_FLOATS, (drop + 1) * SITE_FLOATS));
      const fellBack = pk.map((v, k) => !(v + pd[k] > 0) || !Number.isFinite(v + pd[k]));
      [wk, wd] = pairWeights(pk, pd, count[keep], count[drop]);
      for (let k = 0; k < SITE_FLOATS; k += 1) W[keep * SITE_FLOATS + k] = fellBack[k] ? 0 : wk[k] + wd[k];
    }
    cur.set(mergedSite(cur, keep, drop, wk, wd), keep * SITE_FLOATS);
    siteFeatures(cur, keep, F);
    count[keep] += count[drop];
    alive[drop] = 0;
    changed.add(keep);
    changed.delete(drop);
    dropped.push(drop);
    merges.push({ keep, drop, weight: wij });

    // The dropped site's neighbours become the kept site's; reweigh them all.
    adj[keep].delete(drop);
    for (const nb of adj[drop].keys()) {
      adj[nb].delete(drop);
      if (nb !== keep) adj[keep].set(nb, 0);
    }
    adj[drop].clear();
    version[drop] += 1;
    for (const nb of adj[keep].keys()) {
      groupDistances(F, keep, nb, dist);
      const w = edgeWeight(dist, scales);
      adj[keep].set(nb, w);
      adj[nb].set(keep, w);
    }
    rescore(keep);
    for (const nb of adj[keep].keys()) rescore(nb);
  }
  return { sites: cur, changed: [...changed], dropped, merges, edgeWeights, scales };
}
