// Site codec: a real bitstream for a set of sites, so the rate is a measured
// code length rather than an assumed BITS_PER_SITE. No DOM, no GPU.
//
// Stages, each undone exactly by decodeSites:
//   1. Quantize. Uniform scalar steps per parameter group (codecSteps picks
//      them from the Gauss–Newton diagonal). Position snaps to a grid of
//      cells; aniso_dir becomes an angle θ mod π (±dir render the same, and
//      the trainer keeps |dir| = 1) with a step per site that shrinks as the
//      decoded log_aniso grows, none at all for round sites; colors share one
//      step, so the integer RGB can go through the lossless YCoCg-R transform.
//      With a `colorBasis` (an orthonormal KLT, pcaBasis in color_spaces.js,
//      shared by encoder and decoder like a codebook) the colors are instead
//      quantized along its axes with that step: the same error as in RGB, and
//      the three color planes are those coordinates.
//   2. Order. Sites are sorted by grid cell in raster order (their order
//      carries no information), and positions are coded as gaps between
//      successive cell indices.
//   3. Predict. Every other value is predicted from its 3 nearest earlier
//      sites (median; nearest for the circular θ) or from one constant per
//      plane, whichever codes shorter (one flag per plane). The residual's
//      context is the neighbours' spread.
//   4. Entropy code. Residuals are binarized Exp-Golomb style (adaptive
//      bucket symbol + mantissa bits, the top two adaptive, the rest raw)
//      and arithmetic coded with adaptive frequency models. The header (image
//      size, count, steps, plane flags) is raw Exp-Golomb.
// With a shared model (buildSharedModel: steps, color basis, plane constants
// and primed starting frequencies fit on a training set, held by encoder and
// decoder like a codebook) the stream drops the steps, codes its short header
// with the arithmetic coder too, and its models start primed (Shared model).
//
// The site layout is io.js's: [x, y, log_tau, radius, r, g, b, dir_x, dir_y,
// log_aniso], colors decoded RGB.

import { SITE_FLOATS } from "./params.js?v=m3-20261020";

// ---- Bit I/O ----

class BitWriter {
  constructor() { this.bytes = []; this.cur = 0; this.n = 0; this.bits = 0; }
  bit(b) {
    this.cur = (this.cur << 1) | b;
    this.bits += 1;
    if (++this.n === 8) { this.bytes.push(this.cur); this.cur = 0; this.n = 0; }
  }
  // Unsigned Exp-Golomb (order 0) of v ≥ 0.
  eg(v) {
    let k = 0;
    while (2 ** (k + 1) <= v + 1) k += 1;
    for (let j = 0; j < k; j += 1) this.bit(0);
    for (let j = k; j >= 0; j -= 1) this.bit(Math.floor((v + 1) / 2 ** j) % 2);
  }
  seg(v) { this.eg(v > 0 ? 2 * v - 1 : -2 * v); }
  finish() {
    const out = this.bytes.slice();
    if (this.n) out.push(this.cur << (8 - this.n));
    return Uint8Array.from(out);
  }
}

class BitReader {
  constructor(bytes) { this.bytes = bytes; this.pos = 0; }
  bit() { // zeros past the end, as the arithmetic decoder expects
    const byte = this.bytes[this.pos >> 3] ?? 0;
    return (byte >> (7 - (this.pos++ & 7))) & 1;
  }
  eg() {
    let k = 0;
    while (this.bit() === 0) { k += 1; if (k > 52) throw new Error("bad Exp-Golomb code"); }
    let v = 1;
    for (let j = 0; j < k; j += 1) v = v * 2 + this.bit();
    return v - 1;
  }
  seg() { const u = this.eg(); return u % 2 ? (u + 1) / 2 : -u / 2; }
}

// ---- Arithmetic coding (Witten–Neal–Cleary, 32-bit) ----
// Products stay under 2^48 (range < 2^32, totals ≤ 2^16), exact in doubles.

const HALF = 2 ** 31, QTR = 2 ** 30;
const MAX_TOTAL = 1 << 16;
const INC = 24;

class Model {
  // `init`: starting frequencies (a shared model's primed table), else
  // uniform; `inc`: the increment per coded symbol (smaller = a stronger prior).
  constructor(size, frozen = false, init = null, inc = INC) {
    this.f = init ? init.slice() : new Array(size).fill(1);
    this.total = init ? init.reduce((a, b) => a + b, 0) : size;
    this.frozen = frozen;
    this.inc = inc;
  }
  cumLo(s) { let c = 0; for (let i = 0; i < s; i += 1) c += this.f[i]; return c; }
  bits(s) { return -Math.log2(this.f[s] / this.total); }
  update(s) {
    if (this.frozen) return;
    this.f[s] += this.inc;
    this.total += this.inc;
    if (this.total > MAX_TOTAL) {
      this.total = 0;
      for (let i = 0; i < this.f.length; i += 1) { this.f[i] = (this.f[i] + 1) >> 1; this.total += this.f[i]; }
    }
  }
}
const BYPASS = new Model(2, true);

// Every adaptive model of one stream, made on first use under a string key
// (the same key in encoder and decoder). With a shared model's `tables` and
// `incs` (buildSharedModel) a model starts from its primed frequencies instead
// of uniform, then adapts as before (by its inc if listed). Keys: "h…"
// header, "x…" positions, "p<plane>/…".
class ModelSet {
  constructor(tables = null, incs = null) { this.tables = tables; this.incs = incs; this.models = new Map(); }
  get(key, size) {
    let m = this.models.get(key);
    if (!m) {
      const t = this.tables?.[key];
      m = t && t.length === size ? new Model(size, false, t, this.incs?.[key] ?? INC) : new Model(size);
      m.key = key;
      this.models.set(key, m);
    }
    return m;
  }
}

// The three ways of running the same coding code: count ideal bits only,
// encode, decode. sym(model, s) returns the symbol (s when coding it).
class CostIO {
  constructor() { this.bits = 0; }
  sym(m, s) { this.bits += m.bits(s); m.update(s); return s; }
}

// CostIO that also counts each adaptive model's symbols (by key) and the
// bits per model family (familyOf), for fitting a shared model.
class CountIO extends CostIO {
  constructor(counts = new Map(), famBits = {}) { super(); this.counts = counts; this.famBits = famBits; }
  sym(m, s) {
    if (!m.frozen) {
      if (!this.counts.has(m.key)) this.counts.set(m.key, new Array(m.f.length).fill(0));
      this.counts.get(m.key)[s] += 1;
    }
    const fam = m.frozen ? "raw" : familyOf(m.key);
    this.famBits[fam] = (this.famBits[fam] ?? 0) + m.bits(s);
    return super.sym(m, s);
  }
}
// "h" header, "x" positions, "p0"… a plane: each gets its own prior strength.
const familyOf = (key) => (key[0] === "p" ? key.slice(0, key.indexOf("/")) : key[0]);

class EncodeIO {
  constructor(w) { this.w = w; this.low = 0; this.high = 2 ** 32 - 1; this.pending = 0; this.bits = 0; }
  sym(m, s) {
    const lo = m.cumLo(s);
    const range = this.high - this.low + 1;
    this.high = this.low + Math.floor((range * (lo + m.f[s])) / m.total) - 1;
    this.low += Math.floor((range * lo) / m.total);
    for (;;) {
      if (this.high < HALF) this._emit(0);
      else if (this.low >= HALF) { this._emit(1); this.low -= HALF; this.high -= HALF; }
      else if (this.low >= QTR && this.high < 3 * QTR) { this.pending += 1; this.low -= QTR; this.high -= QTR; }
      else break;
      this.low *= 2;
      this.high = this.high * 2 + 1;
    }
    this.bits += m.bits(s);
    m.update(s);
    return s;
  }
  _emit(b) {
    this.w.bit(b);
    for (; this.pending > 0; this.pending -= 1) this.w.bit(1 - b);
  }
  finish() { this.pending += 1; this._emit(this.low < QTR ? 0 : 1); }
}

class DecodeIO {
  constructor(r) {
    this.r = r; this.low = 0; this.high = 2 ** 32 - 1; this.value = 0;
    for (let i = 0; i < 32; i += 1) this.value = this.value * 2 + r.bit();
  }
  sym(m) {
    const range = this.high - this.low + 1;
    const target = Math.floor(((this.value - this.low + 1) * m.total - 1) / range);
    let s = 0, lo = 0;
    while (lo + m.f[s] <= target) { lo += m.f[s]; s += 1; }
    this.high = this.low + Math.floor((range * (lo + m.f[s])) / m.total) - 1;
    this.low += Math.floor((range * lo) / m.total);
    for (;;) {
      if (this.high < HALF) { /* nothing */ }
      else if (this.low >= HALF) { this.low -= HALF; this.high -= HALF; this.value -= HALF; }
      else if (this.low >= QTR && this.high < 3 * QTR) { this.low -= QTR; this.high -= QTR; this.value -= QTR; }
      else break;
      this.low *= 2;
      this.high = this.high * 2 + 1;
      this.value = this.value * 2 + this.r.bit();
    }
    m.update(s);
    return s;
  }
}

// ---- Binarization ----

const MAX_K = 48;
const ADAPTIVE_MANTISSA_BITS = 2;

// Models for one stream of integers: the bucket symbol and the top mantissa
// bits per bucket. `signed`: 0 | ±[2^k, 2^(k+1)) → 1 + 2k + sign; unsigned:
// v + 1 ∈ [2^k, 2^(k+1)) → k.
function intModels(signed, ms = new ModelSet(), key = "") {
  return { signed, ms, key, bucket: ms.get(`${key}b`, signed ? 1 + 2 * MAX_K : MAX_K), mant: [] };
}

function floorLog2(v) { let k = 0; while (2 ** (k + 1) <= v) k += 1; return k; }

function codeMantissa(io, m, k, value) {
  m.mant[k] ||= Array.from({ length: ADAPTIVE_MANTISSA_BITS }, (_, t) => m.ms.get(`${m.key}m${k}.${t}`, 2));
  let acc = 0;
  for (let j = k - 1; j >= 0; j -= 1) {
    const top = k - 1 - j;
    const bit = value === undefined ? undefined : Math.floor(value / 2 ** j) % 2;
    acc = acc * 2 + io.sym(top < ADAPTIVE_MANTISSA_BITS ? m.mant[k][top] : BYPASS, bit);
  }
  return acc;
}

// Codes `v` (undefined when decoding) and returns it.
function codeInt(io, m, v) {
  const enc = v !== undefined;
  if (m.signed) {
    let s;
    if (enc) s = v === 0 ? 0 : 1 + 2 * floorLog2(Math.abs(v)) + (v < 0 ? 1 : 0);
    s = io.sym(m.bucket, s);
    if (s === 0) return 0;
    const k = (s - 1) >> 1;
    const mag = 2 ** k + codeMantissa(io, m, k, enc ? Math.abs(v) - 2 ** k : undefined);
    return (s - 1) & 1 ? -mag : mag;
  }
  const k = io.sym(m.bucket, enc ? floorLog2(v + 1) : undefined);
  return 2 ** k + codeMantissa(io, m, k, enc ? v + 1 - 2 ** k : undefined) - 1;
}

// ---- Neighbours ----

const NEIGHBORS = 3;

// For each site i (in coded order), up to NEIGHBORS nearest sites j < i by
// grid position, nearest first (ties by index). Grid buckets keep it near
// linear; encoder and decoder run the same code on the same integers.
function priorNeighbors(xs, ys) {
  const n = xs.length;
  let maxX = 0, maxY = 0;
  for (let i = 0; i < n; i += 1) { maxX = Math.max(maxX, xs[i]); maxY = Math.max(maxY, ys[i]); }
  const cell = Math.max(1, Math.ceil(Math.sqrt(((maxX + 1) * (maxY + 1)) / Math.max(1, n)) * 2));
  const gw = Math.floor(maxX / cell) + 1, gh = Math.floor(maxY / cell) + 1;
  const buckets = new Map();
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const cx = Math.floor(xs[i] / cell), cy = Math.floor(ys[i] / cell);
    const best = []; // [d2, j], nearest first
    const consider = (j) => {
      const d2 = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2;
      if (best.length === NEIGHBORS && d2 >= best[NEIGHBORS - 1][0]) return;
      let at = best.length;
      while (at > 0 && best[at - 1][0] > d2) at -= 1;
      best.splice(at, 0, [d2, j]);
      if (best.length > NEIGHBORS) best.pop();
    };
    for (let r = 0; r <= gw + gh; r += 1) {
      // Unvisited sites are at least (r − 1) cells away.
      if (best.length === NEIGHBORS && r >= 1 && best[NEIGHBORS - 1][0] <= ((r - 1) * cell) ** 2) break;
      for (let dy = -r; dy <= r; dy += 1) {
        const y = cy + dy;
        if (y < 0 || y >= gh) continue;
        const step = Math.abs(dy) === r ? 1 : 2 * r;
        for (let dx = -r; dx <= r; dx += step || 1) {
          const x = cx + dx;
          if (x < 0 || x >= gw) continue;
          const b = buckets.get(y * gw + x);
          if (b) for (const j of b) consider(j);
        }
      }
    }
    out[i] = best.map(([, j]) => j);
    const key = cy * gw + cx;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(i);
  }
  return out;
}

// ---- Planes ----
// A plane is one integer per site (or per site with `has`), coded in site
// order with residuals against a prediction.

const SPREAD_CTX = 5;
function spreadCtx(vals) {
  if (!vals.length) return 4;
  const s = Math.max(...vals) - Math.min(...vals);
  return s === 0 ? 0 : s <= 2 ? 1 : s <= 8 ? 2 : 3;
}
function median(vals) {
  const v = vals.slice().sort((a, b) => a - b);
  return v.length % 2 ? v[(v.length - 1) >> 1] : Math.floor((v[v.length / 2 - 1] + v[v.length / 2]) / 2);
}

// Codes plane `vals` (filled in when decoding) with mode 0 (predict `constant`)
// or 1 (predict from neighbours; `constant` for a site without any). `levels`:
// per site, M for a value mod M (an angle in M levels over π; the nearest
// neighbour's angle is the prediction, rescaled to M), or null for integers.
// Models come from `ms` under `key` (fresh uniform ones by default).
function codePlane(io, { vals, has, levels }, nbrs, mode, constant, enc, ms = new ModelSet(), key = "") {
  const models = Array.from({ length: SPREAD_CTX }, (_, c) => intModels(true, ms, `${key}${c}/`));
  for (let i = 0; i < vals.length; i += 1) {
    if (has && !has[i]) continue;
    const near = nbrs[i].filter((j) => !has || has[j]);
    const nv = near.map((j) => vals[j]);
    const m = levels ? levels[i] : 0;
    let pred = m ? constant % m : constant;
    if (mode === 1 && near.length) pred = m ? Math.round((vals[near[0]] * m) / levels[near[0]]) % m : median(nv);
    const ctx = m ? 0 : spreadCtx(nv);
    let r;
    if (enc) {
      r = vals[i] - pred;
      if (m) r = ((r % m) + m + Math.floor(m / 2)) % m - Math.floor(m / 2);
    }
    r = codeInt(io, models[ctx], r);
    if (!enc) vals[i] = m ? (((pred + r) % m) + m) % m : pred + r;
  }
}

// ---- Quantization ----

// Steps are 2^(e / STEP_RES) so the header carries a small integer each.
const STEP_RES = 8;
const STEP_EXP_MAX = 20 * STEP_RES;
const STEP_GROUPS = ["pos", "logTau", "radius", "color", "logAniso", "theta"];
const MAX_THETA_LEVELS = 4096;

function quantizeStep(step) {
  const e = Number.isFinite(step) && step > 0 ? Math.round(STEP_RES * Math.log2(step)) : STEP_EXP_MAX;
  return Math.max(-STEP_EXP_MAX, Math.min(STEP_EXP_MAX, e));
}
const stepOf = (e) => 2 ** (e / STEP_RES);

// A site's θ levels over π. Its step is the θ step over |e^a − e^−a|, a =
// the decoded log_aniso: rotating by δθ moves the distance metric by that
// difference of axis lengths times δθ, so elongated sites need finer angles
// and round ones (a = 0) get a single level, i.e. no θ coded.
function thetaLevelsOf(q, aniso) {
  const s = Math.abs(2 * Math.sinh(aniso * stepOf(q.logAniso)));
  return Math.max(1, Math.min(MAX_THETA_LEVELS, Math.round((Math.PI / stepOf(q.theta)) * s)));
}

// Every stage's integers for the active sites of `sites`.
function quantize(sites, count, width, height, q, basis = null) {
  const ps = stepOf(q.pos);
  const gx = Math.max(1, Math.ceil(width / ps)), gy = Math.max(1, Math.ceil(height / ps));
  const rows = [];
  for (let i = 0; i < count; i += 1) {
    const b = i * SITE_FLOATS;
    if (!(sites[b] >= 0)) continue;
    const cell = (v, g) => Math.max(0, Math.min(g - 1, Math.floor(v / ps)));
    const x = cell(sites[b], gx), y = cell(sites[b + 1], gy);
    const lin = (v, e) => Math.round(v / stepOf(e));
    const aniso = lin(sites[b + 9], q.logAniso);
    const m = thetaLevelsOf(q, aniso);
    let theta = Math.atan2(sites[b + 8], sites[b + 7]);
    theta = ((theta % Math.PI) + Math.PI) % Math.PI;
    let yy, co, cg;
    if (basis) {
      const d = [0, 1, 2].map((c) => sites[b + 4 + c] - basis.mean[c]);
      [yy, co, cg] = basis.axes.map((a) => Math.round((a[0] * d[0] + a[1] * d[1] + a[2] * d[2]) / stepOf(q.color)));
    } else {
      // YCoCg-R (lossless on integers)
      const r = lin(sites[b + 4], q.color), g = lin(sites[b + 5], q.color), bl = lin(sites[b + 6], q.color);
      co = r - bl;
      const t = bl + (co >> 1);
      cg = g - t;
      yy = t + (cg >> 1);
    }
    rows.push({
      slot: i, x, y, idx: y * gx + x, m,
      planes: [aniso, m > 1 ? Math.round((theta * m) / Math.PI) % m : 0,
               lin(sites[b + 2], q.logTau), lin(sites[b + 3], q.radius), yy, co, cg],
    });
  }
  rows.sort((a, b) => a.idx - b.idx || a.slot - b.slot);
  return { rows, gx, gy };
}

// Sites (coded order, decoded RGB) from the integers; `levels` per site.
function dequantize(xs, ys, planes, levels, q, basis = null) {
  const n = xs.length;
  const ps = stepOf(q.pos);
  const out = new Float32Array(n * SITE_FLOATS);
  for (let i = 0; i < n; i += 1) {
    const b = i * SITE_FLOATS;
    const [aniso, thetaQ, tau, radius, yy, co, cg] = planes.map((p) => p[i]);
    let rgb;
    if (basis) {
      const k = [yy, co, cg].map((v) => v * stepOf(q.color));
      rgb = [0, 1, 2].map((c) => basis.mean[c] + basis.axes[0][c] * k[0] + basis.axes[1][c] * k[1] + basis.axes[2][c] * k[2]);
    } else {
      const t = yy - (cg >> 1), g = cg + t, bl = t - (co >> 1), r = bl + co;
      rgb = [r, g, bl].map((v) => v * stepOf(q.color));
    }
    const theta = levels[i] > 1 ? (thetaQ * Math.PI) / levels[i] : 0;
    out[b] = (xs[i] + 0.5) * ps;
    out[b + 1] = (ys[i] + 0.5) * ps;
    out[b + 2] = tau * stepOf(q.logTau);
    out[b + 3] = radius * stepOf(q.radius);
    out[b + 4] = rgb[0];
    out[b + 5] = rgb[1];
    out[b + 6] = rgb[2];
    out[b + 7] = Math.cos(theta);
    out[b + 8] = Math.sin(theta);
    out[b + 9] = aniso * stepOf(q.logAniso);
  }
  return out;
}

// The last three are the YCoCg-R planes, or the color basis's axes K0–K2.
export const CODEC_PLANES = ["log_aniso", "θ", "log_tau", "radius", "Y", "Co", "Cg"];
const THETA_PLANE = 1;

// Positions: raster gaps between cell indices, context = the previous gap's bucket.
function codePositions(io, idx, n, enc, ms = new ModelSet()) {
  const models = Array.from({ length: MAX_K + 1 }, (_, c) => intModels(false, ms, `x${c}/`));
  let prev = 0, ctx = MAX_K;
  for (let i = 0; i < n; i += 1) {
    const gap = codeInt(io, models[ctx], enc ? idx[i] - prev : undefined);
    if (!enc) idx[i] = prev + gap;
    prev = idx[i];
    ctx = floorLog2(gap + 1);
  }
}

// Everything the coder needs about one image's active sites at steps `q`.
function analyze(sites, count, width, height, q, basis) {
  const { rows } = quantize(sites, count, width, height, q, basis);
  const xsPos = rows.map((r) => r.x), ysPos = rows.map((r) => r.y), idx = rows.map((r) => r.idx);
  const levels = rows.map((r) => r.m);
  const has = levels.map((m) => m > 1);
  const planes = CODEC_PLANES.map((_, p) => rows.map((r) => r.planes[p]));
  const specs = planes.map((vals, p) => (p === THETA_PLANE ? { vals, has, levels } : { vals, has: null, levels: null }));
  return { rows, n: rows.length, xsPos, ysPos, idx, nbrs: priorNeighbors(xsPos, ysPos), levels, planes, specs };
}

// A plane's median (θ: over the sites that have one), its per-image constant.
function ownConstant(s) {
  const v = s.vals.filter((_, i) => !s.has || s.has[i]);
  return v.length ? median(v) : 0;
}

// ---- Shared model ----
// Side information that encoder and decoder hold like a codebook (as a
// colorBasis), fit on a training set of images by buildSharedModel, so a
// per-image stream carries only what is specific to its image:
//   q          quantizer step exponents (no steps in the stream)
//   colorBasis the color basis, or null for YCoCg-R
//   width, height  the usual image size (one flag in the stream when it matches)
//   constants  per plane, the dataset's median value
//   tables     primed starting frequencies of the adaptive models, by key
//   incs       their increments per symbol where smaller than usual
//   strengths  per model family, the prior weight the tables were made with
// A shared stream is arithmetic coded from its first bit, header included:
// a size flag (then the size if it differs), the count, the layer's index + 1,
// and per plane a mode: 0 = predict the shared constant, 1 = neighbours (the
// shared constant for a site without any), 2 = predict an own constant (coded
// against the shared one). These use primed models too. Then positions and
// planes as in a per-image stream, their models starting primed and adapting
// per image. The stream does not identify its shared model: decode with the
// one it was coded with.

const SHARED_VERSION = 1;
const SHARED_MODES = 3;
const PRIOR_STRENGTHS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const PRIOR_MASS = INC * 2048; // most a primed table holds (plus 1 per symbol), under MAX_TOTAL

// Codes one image's shared stream (st filled in when decoding): header,
// positions, planes. Returns the ideal bits by stage.
function sharedStream(io, S, st, enc, ms = new ModelSet(S.tables, S.incs)) {
  const same = io.sym(ms.get("hs", 2), enc ? (st.width === S.width && st.height === S.height ? 1 : 0) : undefined);
  if (same) { st.width = S.width; st.height = S.height; } else {
    st.width = codeInt(io, intModels(false, ms, "hw/"), enc ? st.width : undefined);
    st.height = codeInt(io, intModels(false, ms, "hh/"), enc ? st.height : undefined);
  }
  st.n = codeInt(io, intModels(false, ms, "hn/"), enc ? st.n : undefined);
  st.globalIndex = codeInt(io, intModels(false, ms, "hg/"), enc ? st.globalIndex + 1 : undefined) - 1;
  if (!enc) { st.modes = []; st.constants = []; }
  for (let p = 0; p < CODEC_PLANES.length; p += 1) sharedPlaneHeader(io, S, st, p, enc, ms);
  const headerBits = io.bits;
  if (!enc) st.idx = new Array(st.n);
  codePositions(io, st.idx, st.n, enc, ms);
  const positionBits = io.bits - headerBits;
  if (!enc) {
    const gx = Math.max(1, Math.ceil(st.width / stepOf(S.q.pos)));
    st.xs = st.idx.map((v) => v % gx); st.ys = st.idx.map((v) => Math.floor(v / gx));
    st.nbrs = priorNeighbors(st.xs, st.ys);
    st.planes = CODEC_PLANES.map(() => new Array(st.n).fill(0));
  }
  const planeBits = CODEC_PLANES.map((_, p) => {
    const before = io.bits;
    sharedPlane(io, S, st, p, enc, ms);
    return io.bits - before;
  });
  return { headerBits, positionBits, planeBits };
}

function sharedPlaneHeader(io, S, st, p, enc, ms) {
  const mode = io.sym(ms.get(`hm${p}`, SHARED_MODES), enc ? st.modes[p] : undefined);
  let c = S.constants[p];
  if (mode === 2) c += codeInt(io, intModels(true, ms, `hc${p}/`), enc ? st.constants[p] - S.constants[p] : undefined);
  if (!enc) { st.modes.push(mode); st.constants.push(c); }
}

function sharedPlane(io, S, st, p, enc, ms) {
  if (p === THETA_PLANE) st.levels = st.planes[0].map((a) => thetaLevelsOf(S.q, a)); // log_aniso is coded by now
  const spec = p === THETA_PLANE ? { vals: st.planes[p], has: st.levels.map((m) => m > 1), levels: st.levels }
                                 : { vals: st.planes[p], has: null, levels: null };
  const mode = st.modes[p];
  codePlane(io, spec, st.nbrs, mode === 1 ? 1 : 0, mode === 2 ? st.constants[p] : S.constants[p], enc, ms, `p${p}/`);
}

// An analyzed image (analyze) as a shared stream's state, each plane's mode
// whichever codes shortest with S's primed models (its mode symbol and
// constant included; planes use disjoint models, so each is costed alone).
function sharedState(a, width, height, globalIndex, S) {
  const st = { width, height, n: a.n, globalIndex, idx: a.idx, nbrs: a.nbrs, planes: a.planes, modes: [], constants: [] };
  a.specs.forEach((spec, p) => {
    const own = ownConstant(spec);
    let best = Infinity;
    for (let mode = 0; mode < SHARED_MODES; mode += 1) {
      if (mode === 2 && own === S.constants[p]) continue;
      const io = new CostIO(), ms = new ModelSet(S.tables, S.incs);
      const trial = { modes: [], constants: [] };
      trial.modes[p] = mode; trial.constants[p] = own;
      sharedPlaneHeader(io, S, trial, p, true, ms);
      codePlane(io, spec, a.nbrs, mode === 1 ? 1 : 0, mode === 2 ? own : S.constants[p], true, ms, `p${p}/`);
      if (io.bits < best) { best = io.bits; st.modes[p] = mode; st.constants[p] = mode === 2 ? own : S.constants[p]; }
    }
  });
  return st;
}

function encodeShared(sites, count, width, height, globalSite, S) {
  const basis = S.colorBasis ?? null;
  const a = analyze(sites, count, width, height, S.q, basis);
  const globalIndex = a.rows.findIndex((r) => r.slot === globalSite);
  const st = sharedState(a, width, height, globalIndex, S);
  const w = new BitWriter();
  const io = new EncodeIO(w);
  const { headerBits, positionBits, planeBits } = sharedStream(io, S, st, true);
  io.finish();
  const bytes = w.finish();
  return {
    bytes, bits: w.bits, n: a.n, headerBits, positionBits, planeBits, modes: st.modes,
    decoded: dequantize(a.xsPos, a.ysPos, a.planes, a.levels, S.q, basis),
    slots: a.rows.map((r) => r.slot), globalIndex,
  };
}

function decodeShared(bytes, S) {
  const st = {};
  sharedStream(new DecodeIO(new BitReader(bytes)), S, st, false);
  return { sites: dequantize(st.xs, st.ys, st.planes, st.levels, S.q, S.colorBasis ?? null),
           count: st.n, width: st.width, height: st.height, globalSite: st.globalIndex };
}

// Primed tables from symbol counts: a model seen N times starts at
// 1 + INC·K'·count/N per symbol, i.e. as if it had already adapted to
// K' = K·N/(N + K) observations (K its family's strength: few counts, less
// trust). Past PRIOR_MASS the table is scaled down to it and the model's
// increment instead (incs), so the ratio of prior to update holds.
function primedTables(counts, strengths) {
  const tables = {}, incs = {};
  for (const [key, c] of counts) {
    const N = c.reduce((a, b) => a + b, 0), K = strengths[familyOf(key)] ?? 0;
    if (!N || !K) continue;
    const mass = (INC * K * N) / (N + K), scale = Math.min(1, PRIOR_MASS / mass);
    tables[key] = c.map((x) => 1 + Math.round((mass * scale * x) / N));
    if (scale < 1) incs[key] = Math.max(1, Math.round(INC * scale));
  }
  return { tables, incs };
}

// Builds a shared model (see above) from training images `list` ([{ sites,
// n (slots), width, height, globalSite? }], each in its own coordinates) at
// `steps` (codecSteps; any positive numbers work). `colorBasis`: code colors
// along it. Constants are the pooled per-plane medians; the tables are the
// pooled symbol counts of every image coded with its best modes. Each model
// family's strength (header, positions, each plane) is picked from
// PRIOR_STRENGTHS by two-fold cross-validation over the images (counts from
// one half, bits on the other), so it is not fit to the images it is scored
// on. `iterations` alternate modes and tables. Returns a plain JSON object.
export function buildSharedModel(list, steps, { colorBasis = null, iterations = 2 } = {}) {
  const q = {};
  for (const k of STEP_GROUPS) q[k] = quantizeStep(steps[k]);
  const items = list.map((im) => {
    const a = analyze(im.sites, im.n, im.width, im.height, q, colorBasis);
    const g = im.globalSite ?? -1;
    return { a, width: im.width, height: im.height, globalIndex: a.rows.findIndex((r) => r.slot === g) };
  });
  const sizes = new Map();
  for (const it of items) { const k = `${it.width}x${it.height}`; sizes.set(k, (sizes.get(k) ?? 0) + 1); }
  const [width, height] = ([...sizes].sort((x, y) => y[1] - x[1])[0]?.[0] ?? "0x0").split("x").map(Number);
  const constants = CODEC_PLANES.map((_, p) => {
    const v = [];
    for (const { a } of items) a.specs[p].vals.forEach((x, i) => { if (!a.specs[p].has || a.specs[p].has[i]) v.push(x); });
    return v.length ? median(v) : 0;
  });
  const S = { version: SHARED_VERSION, q, colorBasis, width, height, constants, tables: {}, incs: {}, strengths: {} };
  for (let iter = 0; iter < iterations; iter += 1) {
    const states = items.map((it) => sharedState(it.a, it.width, it.height, it.globalIndex, S));
    const count = (sel) => { const io = new CountIO(); sel.forEach((st) => sharedStream(io, S, st, true, new ModelSet())); return io.counts; };
    const halves = [states.filter((_, i) => i % 2 === 0), states.filter((_, i) => i % 2 === 1)];
    const halfCounts = halves.map(count);
    const fams = new Set([...halfCounts[0].keys(), ...halfCounts[1].keys()].map(familyOf));
    const best = {};
    for (const K of PRIOR_STRENGTHS) {
      const strengths = Object.fromEntries([...fams].map((f) => [f, K]));
      const famBits = {};
      for (let h = 0; h < 2; h += 1) {
        const T = { ...S, ...primedTables(halfCounts[1 - h], strengths) };
        const io = new CountIO(new Map(), famBits);
        halves[h].forEach((st) => sharedStream(io, T, st, true));
      }
      for (const f of fams) if (!(best[f]?.bits <= (famBits[f] ?? 0))) best[f] = { K, bits: famBits[f] ?? 0 };
    }
    S.strengths = Object.fromEntries(Object.entries(best).map(([f, b]) => [f, b.K]));
    Object.assign(S, primedTables(count(states), S.strengths));
  }
  return S;
}

// ---- Public API ----

// Quantizer steps for `lossDb`, the PSNR loss to spend on quantization, from
// the Gauss–Newton diagonal F (Trainer.siteSensitivity: F[i·10 + t], colors
// per RGB channel) and the summed squared error sse0 of the unquantized fit.
// Uniform noise of step Δ on a parameter with sensitivity F adds F·Δ²/12 to
// the squared error. At high rate, the steps minimizing that plus λ × bits
// give every coded scalar the same share of it, so a group of m scalars with
// summed sensitivity A gets Δ = √(12·m·d / A), d = the added error / 9
// scalars (x, y, log τ, radius, R, G, B, θ, log aniso).
export function codecSteps(sites, count, F, { lossDb, sse0 }) {
  const A = { pos: 0, logTau: 0, radius: 0, color: 0, logAniso: 0, theta: 0 };
  const pos = (v) => (Number.isFinite(v) && v > 0 ? v : 0);
  for (let i = 0; i < count; i += 1) {
    const b = i * SITE_FLOATS, f = i * 10;
    if (!(sites[b] >= 0)) continue;
    A.pos += pos(F[f]) + pos(F[f + 1]);
    A.logTau += pos(F[f + 2]);
    A.radius += pos(F[f + 3]);
    A.color += pos(F[f + 4]) + pos(F[f + 5]) + pos(F[f + 6]);
    // dir = (cos θ, sin θ): d dir / dθ = (−dir_y, dir_x). The site's own θ
    // step is steps.theta / |2 sinh a| (thetaLevelsOf), so its share of the
    // error is that sensitivity / (2 sinh a)².
    const s2 = (2 * Math.sinh(sites[b + 9])) ** 2;
    if (s2 > 1e-6) A.theta += (pos(F[f + 7]) * sites[b + 8] ** 2 + pos(F[f + 8]) * sites[b + 7] ** 2) / s2;
    A.logAniso += pos(F[f + 9]);
  }
  const d = (sse0 * (10 ** (lossDb / 10) - 1)) / 9;
  const scalars = { pos: 2, logTau: 1, radius: 1, color: 3, logAniso: 1, theta: 1 };
  const step = (k) => (A[k] > 0 ? Math.sqrt((12 * scalars[k] * d) / A[k]) : Infinity);
  return {
    pos: step("pos"), logTau: step("logTau"), radius: step("radius"), color: step("color"),
    logAniso: step("logAniso"), theta: step("theta"),
  };
}

// Encodes the active sites of `sites` (count · SITE_FLOATS, decoded RGB).
// `steps` from codecSteps (any positive numbers work). `globalSite`: the slot
// of the histogram layer, or −1. `colorBasis`: code colors along its axes
// (decodeSites needs the same one); null = YCoCg-R. `shared`: a shared model
// (buildSharedModel) to code against instead; its steps and color basis then
// replace `steps` and `colorBasis`, and modes are 0/1/2 (see Shared model).
// Returns
//   { bytes, bits, n, headerBits, positionBits, planeBits: [per CODEC_PLANES],
//     modes, decoded: the sites as decodeSites returns them (coded order),
//     slots: each decoded site's slot in `sites`, globalIndex: the layer's
//     index in decoded, or −1 }.
export function encodeSites(sites, count, width, height, steps, { globalSite = -1, colorBasis = null, shared = null } = {}) {
  if (shared) return encodeShared(sites, count, width, height, globalSite, shared);
  const q = {};
  for (const k of STEP_GROUPS) q[k] = quantizeStep(steps[k]);
  const { rows, n, xsPos, ysPos, idx, nbrs, levels, planes, specs } = analyze(sites, count, width, height, q, colorBasis);

  // Per plane: whichever predictor codes shorter.
  const constants = specs.map(ownConstant);
  const modes = specs.map((s, p) => {
    const cost = [0, 1].map((mode) => { const io = new CostIO(); codePlane(io, s, nbrs, mode, constants[p], true); return io.bits; });
    return cost[1] < cost[0] ? 1 : 0;
  });

  const globalIndex = rows.findIndex((r) => r.slot === globalSite);
  const w = new BitWriter();
  w.eg(width); w.eg(height); w.eg(n); w.eg(globalIndex + 1);
  w.bit(colorBasis ? 1 : 0);
  for (const k of STEP_GROUPS) w.seg(q[k]);
  specs.forEach((_, p) => { w.bit(modes[p]); w.seg(constants[p]); });
  const headerBits = w.bits;

  const io = new EncodeIO(w);
  codePositions(io, idx, n, true);
  const positionBits = io.bits;
  const planeBits = specs.map((s, p) => {
    const before = io.bits;
    codePlane(io, s, nbrs, modes[p], constants[p], true);
    return io.bits - before;
  });
  io.finish();
  const bytes = w.finish();
  return {
    bytes, bits: w.bits, n, headerBits, positionBits, planeBits, modes,
    decoded: dequantize(xsPos, ysPos, planes, levels, q, colorBasis),
    slots: rows.map((r) => r.slot), globalIndex,
  };
}

// Inverse of encodeSites: { sites (n · SITE_FLOATS, decoded RGB), count,
// width, height, globalSite (index in sites, or −1) }. `colorBasis`, `shared`:
// the ones the stream was coded with, if any.
export function decodeSites(bytes, { colorBasis = null, shared = null } = {}) {
  if (shared) return decodeShared(bytes, shared);
  const r = new BitReader(bytes);
  const width = r.eg(), height = r.eg(), n = r.eg(), globalSite = r.eg() - 1;
  if (r.bit() && !colorBasis) throw new Error("decodeSites: stream was coded along a color basis; pass it");
  const q = {};
  for (const k of STEP_GROUPS) q[k] = r.seg();
  const modes = [], constants = [];
  for (let p = 0; p < CODEC_PLANES.length; p += 1) {
    modes.push(r.bit());
    constants.push(r.seg());
  }
  const io = new DecodeIO(r);
  const idx = new Array(n);
  codePositions(io, idx, n, false);
  const gx = Math.max(1, Math.ceil(width / stepOf(q.pos)));
  const xs = idx.map((v) => v % gx), ys = idx.map((v) => Math.floor(v / gx));
  const nbrs = priorNeighbors(xs, ys);
  const planes = CODEC_PLANES.map(() => new Array(n).fill(0));
  let levels = null;
  CODEC_PLANES.forEach((_, p) => {
    if (p === THETA_PLANE) levels = planes[0].map((a) => thetaLevelsOf(q, a)); // log_aniso is decoded by now
    const spec = p === THETA_PLANE ? { vals: planes[p], has: levels.map((m) => m > 1), levels }
                                   : { vals: planes[p], has: null, levels: null };
    codePlane(io, spec, nbrs, modes[p], constants[p], false);
  });
  return { sites: dequantize(xs, ys, planes, levels, q, colorBasis), count: n, width, height, globalSite };
}
