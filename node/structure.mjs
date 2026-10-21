#!/usr/bin/env node
// E9: greedy search over discrete structure decisions, each priced in
// measured bits. Batched runs; calibration on images 0–1023, the search on
// held-out ones:
//
//   node structure.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --lambdas 0.03,0.1,0.3 --sites 4,6,8,12
//
// Objective. J = D + λ · R over the atlas: D the summed squared error of the
//   decoded sites (codec-quantized, candidate lists rebuilt from them, as a
//   decoder renders; the histogram layer unquantized and uncoded, as in
//   rate.mjs), R the real codec's bits for every image's sites coded as one
//   stream (encodeSites, colors along the target's KLT). λ is in squared error
//   per bit, the trainer's rateLambda units. Steps: codecSteps at rateLossDb
//   for the start state, then fixed except where an action moves them.
// Actions (ACTIONS, pluggable: each builds a preview state from the current
//   one and says how to apply it):
//   drop          per image, merge the pair of sites whose merge is predicted
//                 cheapest (merge.js mergedSite, pixel-count weights from the
//                 owner render; the larger site survives at the merged
//                 parameters, the other is deactivated), in every image where
//                 cal · ΔD − λ · (bits per coded site) < 0. ΔD predicted as
//                 Gauss–Newton with the first-order term for both sites moving
//                 to the merged one (or Ward's nᵢnⱼ/(nᵢ+nⱼ)·‖μᵢ−μⱼ‖² with
//                 --drop-pred ward); one site ≈ 30–40 bits.
//   coarsen:<g>   one quantizer step group ×2 (pos, logTau, radius, color,
//   refine:<g>    logAniso, theta), or ÷2 (down to ¼ of the start step).
//   restrict:<r>  a parameter restricted for every site: iso (log aniso → 0,
//                 which also drops θ), grey (colors onto the grey axis), tau
//                 (log τ → the median), radius (→ the median). Kept by
//                 projection: after every refit step the restricted values
//                 are written back through the trainer's log hook (projected
//                 gradient descent, as lossmap.mjs; no trainer changes).
//   lift:<r>      lifts a restriction again (not iso: direction and anisotropy
//                 only learn from iteration 50, so a refit shorter than that
//                 can't regrow them). Predicted ΔD: the Newton gain −Σ g²/4F on
//                 the freed parameter; ΔR from the Newton-stepped preview.
// Prediction. ΔD ≈ Σ F·δ² + g·δ (F the Gauss–Newton diagonal, g the exact SSE
//   gradient, both at the current state) for δ = decoded preview − current
//   continuous sites, minus the same for the current decoded sites (so
//   quantization changes are priced like any other move; direction is left
//   out where the decoded site is round, since it does nothing there).
//   Calibrated: × the ratio of pooled measured refit ΔD (coarsen: render) to
//   pooled prediction per action, on the calibration images.
//   ΔR: the codec on the preview state (for drops the remaining sites).
// Greedy loop. Each step ranks the actions by cal · ΔD + λ · ΔR, refits a
//   control (the current state, nothing changed: --refit-iters iterations,
//   fresh Adam, LR --refit-lr × the preset's, no error weighting) and the best
//   action the same way, and accepts it if its J beats the control's; else the
//   next one (up to --tries); stops when no action predicts a gain, none is
//   accepted, or after --max-steps. Every attempt logs predicted ΔD (GN+g,
//   calibrated), the render-only ΔD, the measured ΔD after the refit and
//   predicted vs measured ΔR (both against the control).
// Calibration (images --calib-from, --calib-count). On the trained preset,
//   every action type once (drop in every image; each coarsen; each restrict),
//   per image: predicted ΔD vs measured by render and after a projected refit
//   (against the mean of two control refits; for coarsen the control state
//   re-quantized), predicted ΔR (per image streams) vs measured after the
//   refit. Spearman and slope per type.
// Comparison (held-out images, per λ): the greedy result vs the start state
//   refit as often as the greedy accepted actions (control arm: refits alone),
//   the preset trained with the rate term at the same λ, and the site-count
//   sweep at λ = 0 (--sites), all measured the same way: median per-image PSNR
//   of the decoded sites, measured bpp, and J per image.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=w2b-20261021";
import { encodeSites, codecSteps } from "../trainer/codec.js?v=w2b-20261021";
import { mergedSite } from "../trainer/merge.js?v=20261019";
import { tileOrigin } from "../trainer/batch.js?v=w2b-20261021";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";
import { saveSites, restoreSites, writeRgbSites, rgbSitesOf, continueRun, sseGradient, spearman, slope0 } from "./lossmap.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const GROUPS = ["pos", "logTau", "radius", "color", "logAniso", "theta"];
const PLANES = ["log_aniso", "θ", "log_tau", "radius", "K0", "K1", "K2"];

// ---- Atlas context: trainer, layout, measurement ----

function atlasCtx(bt, images, args) {
  const tr = bt.trainer, layout = bt.layout;
  const n = tr.actualSites, per = layout.sitesPerTile, layer = bt.layerSlot;
  const isLayer = (i) => layer !== null && layer !== undefined && i % per === layer;
  const pixels = images.reduce((s, im) => s + im.width * im.height, 0);
  return { tr, layout, images, args, n, per, isLayer, pixels, basis: tr.rateBasis, lossDb: args.rateLossDb };
}

const row = (s, i) => s.subarray(i * SITE_FLOATS, (i + 1) * SITE_FLOATS);
const active = (s, i) => s[i * SITE_FLOATS] >= 0;

// Sites with the histogram layer deactivated (it is not coded).
function coded(ctx, sites) {
  const c = Float32Array.from(sites);
  for (let i = 0; i < ctx.n; i += 1) if (ctx.isLayer(i)) c[i * SITE_FLOATS] = -1;
  return c;
}

// Pooled stream: bits, per-plane bits, and the decoded sites back in slot order.
function encode(ctx, sites, steps) {
  const { tr, n } = ctx;
  const enc = encodeSites(coded(ctx, sites), n, tr.width, tr.height, steps, { colorBasis: ctx.basis });
  const dec = Float32Array.from(sites);
  enc.slots.forEach((slot, k) => dec.set(enc.decoded.subarray(k * SITE_FLOATS, (k + 1) * SITE_FLOATS), slot * SITE_FLOATS));
  return { bits: enc.bits, planeBits: enc.planeBits, positionBits: enc.positionBits, coded: enc.n, dec };
}

// Bits per image, each tile coded alone in its own coordinates.
function tileBits(ctx, sites, steps) {
  const { layout, images, per } = ctx;
  return Float64Array.from(images, (img, b) => {
    const [ox, oy] = tileOrigin(layout, b);
    const t = Float32Array.from(sites.subarray(b * per * SITE_FLOATS, (b + 1) * per * SITE_FLOATS));
    for (let k = 0; k < per; k += 1) {
      if (ctx.isLayer(b * per + k)) t[k * SITE_FLOATS] = -1;
      if (!(t[k * SITE_FLOATS] >= 0)) continue;
      t[k * SITE_FLOATS] -= ox;
      t[k * SITE_FLOATS + 1] -= oy;
    }
    return encodeSites(t, per, img.width, img.height, steps, { colorBasis: ctx.basis }).bits;
  });
}

// Summed squared error per image (and over the atlas) of a render.
function tileSse(ctx, rgba) {
  const { tr, layout, images } = ctx;
  const out = new Float64Array(images.length);
  images.forEach((img, b) => {
    const [ox, oy] = tileOrigin(layout, b);
    let acc = 0;
    for (let y = 0; y < img.height; y += 1) {
      for (let x = 0; x < img.width; x += 1) {
        if (img.mask && !(img.mask[y * img.width + x] > 0)) continue;
        const p = ((oy + y) * layout.width + ox + x) * 4;
        for (let c = 0; c < 3; c += 1) acc += (rgba[p + c] - tr.target[p + c]) ** 2;
      }
    }
    out[b] = acc;
  });
  return out;
}

// Renders `sites` (RGB) with candidate lists rebuilt from them; the GPU then
// holds them (callers restore their state).
async function renderSse(ctx, sites) {
  writeRgbSites(ctx.tr, sites);
  ctx.tr._rebuildCandidates(ctx.n);
  const tile = tileSse(ctx, await ctx.tr.renderView("render"));
  return { tile, total: tile.reduce((a, b) => a + b, 0) };
}

// Decoded distortion and measured rate of `sites` at `steps`; restores `raw`.
async function measure(ctx, sites, steps, raw, { perTile = false } = {}) {
  const e = encode(ctx, sites, steps);
  const r = await renderSse(ctx, e.dec);
  restoreSites(ctx.tr, raw);
  const out = { bits: e.bits, planeBits: e.planeBits, positionBits: e.positionBits, coded: e.coded, sse: r.total, tileSse: r.tile, dec: e.dec };
  if (perTile) out.tileBits = tileBits(ctx, sites, steps);
  return out;
}

// Median per-image PSNR, atlas PSNR, bpp, J per image.
function scores(ctx, m, lambda = 0) {
  const psnrs = Array.from(m.tileSse, (s, b) => 10 * Math.log10((3 * ctx.images[b].width * ctx.images[b].height) / Math.max(s, 1e-12)));
  return { psnr: median(psnrs), atlasPsnr: 10 * Math.log10((3 * ctx.pixels) / m.sse), bpp: m.bits / ctx.pixels,
           bitsPerSite: m.bits / Math.max(1, m.coded), sites: m.coded / ctx.images.length,
           jPerImage: (m.sse + lambda * m.bits) / ctx.images.length };
}

// Codec steps for the current state at rateLossDb (GPU must hold `sites`).
async function stepsFor(ctx, sites, F) {
  const cont = await renderSse(ctx, sites);
  return codecSteps(coded(ctx, sites), ctx.n, F, { lossDb: ctx.lossDb, sse0: cont.total });
}

// Everything the actions need about the current state.
async function snapshot(ctx, sites, raw, steps) {
  const g = await sseGradient(ctx.tr);
  const { F } = await ctx.tr.siteSensitivity();
  restoreSites(ctx.tr, raw);
  const owner = await ctx.tr.renderView("owner");
  const area = new Float64Array(ctx.n), tsum = new Float64Array(ctx.n * 3);
  for (let p = 0; p < ctx.tr.width * ctx.tr.height; p += 1) {
    if (ctx.tr.maskCpu && !(ctx.tr.maskCpu[p] > 0)) continue;
    const id = owner[p * 4];
    if (!(id >= 0)) continue;
    area[id] += 1;
    for (let c = 0; c < 3; c += 1) tsum[id * 3 + c] += ctx.tr.target[p * 4 + c];
  }
  return { g, F, area, tsum, steps, sites, raw };
}

// ---- Prediction ----

// Σ F·δ² and Σ g·δ per image for δ = to − from over active, non-layer sites
// active in both; direction left out where `to` is round.
function gnTerms(ctx, snap, from, to) {
  const { F, g } = snap;
  const gn = new Float64Array(ctx.images.length), lin = new Float64Array(ctx.images.length);
  for (let i = 0; i < ctx.n; i += 1) {
    if (ctx.isLayer(i) || !active(from, i) || !active(to, i)) continue;
    const o = i * SITE_FLOATS, b = Math.floor(i / ctx.per), round = to[o + 9] === 0;
    // dir and −dir render the same (the codec codes θ mod π): compare like with like.
    const flip = to[o + 7] * from[o + 7] + to[o + 8] * from[o + 8] < 0 ? -1 : 1;
    for (let t = 0; t < 10; t += 1) {
      if (round && (t === 7 || t === 8)) continue;
      const d = (t === 7 || t === 8 ? flip * to[o + t] : to[o + t]) - from[o + t];
      if (!d || !Number.isFinite(F[i * 10 + t])) continue;
      gn[b] += F[i * 10 + t] * d * d;
      lin[b] += g[i * 10 + t] * d;
    }
  }
  return { gn, lin };
}

const sum = (a) => a.reduce((s, v) => s + v, 0);

// ---- Restrictions ----

const RESTRICTS = {
  iso: (s) => { s[9] = 0; },
  grey: (s) => { const m = (s[4] + s[5] + s[6]) / 3; s[4] = s[5] = s[6] = m; },
  tau: (s, med) => { s[2] = med.logTau; },
  radius: (s, med) => { s[3] = med.radius; },
};
// Freed parameters per restriction (for lift's Newton step) and their clip.
const FREES = { grey: { ts: [4, 5, 6], clip: 0.25 }, tau: { ts: [2], clip: 1 }, radius: { ts: [3], clip: 2 } };

function project(ctx, sites, restrict, med) {
  if (!restrict.size) return sites;
  for (let i = 0; i < ctx.n; i += 1) {
    if (ctx.isLayer(i) || !active(sites, i)) continue;
    for (const r of restrict) RESTRICTS[r](row(sites, i), med);
  }
  return sites;
}

function medians(ctx, sites) {
  const t = [], r = [];
  for (let i = 0; i < ctx.n; i += 1) if (!ctx.isLayer(i) && active(sites, i)) { t.push(sites[i * SITE_FLOATS + 2]); r.push(sites[i * SITE_FLOATS + 3]); }
  return { logTau: median(t), radius: median(r) };
}

// ---- Actions ----
// state: { sites (RGB, continuous), raw, stepMul {group: ×}, restrict Set, med, baseSteps }.
// Each action: preview(ctx, state, snap, env) → null or { state: next state,
// pred: per-image predicted ΔD (raw GN+g; drops: chosen predictor), predGn,
// predLin, (drops) ward, images }.

const stepsOf = (state) => Object.fromEntries(GROUPS.map((k) => [k, state.baseSteps[k] * state.stepMul[k]]));
const withState = (state, patch) => ({ ...state, stepMul: { ...state.stepMul }, restrict: new Set(state.restrict), ...patch });

// Quantization-aware GN+g of a preview against the current decoded sites.
function quantPred(ctx, snap, cur, curDec, nextSites, nextSteps) {
  const nextDec = encode(ctx, nextSites, nextSteps).dec;
  const a = gnTerms(ctx, snap, cur, nextDec), b = gnTerms(ctx, snap, cur, curDec);
  const gn = a.gn.map((v, i) => v - b.gn[i]), lin = a.lin.map((v, i) => v - b.lin[i]);
  return { predGn: gn, predLin: lin, pred: gn.map((v, i) => v + lin[i]) };
}

// Per image, the cheapest merge: { b, keep, drop, site, gn, lin, ward }.
function bestMerges(ctx, snap, sites) {
  const { area, tsum, F, g } = snap;
  const out = [];
  for (let b = 0; b < ctx.images.length; b += 1) {
    const ids = [];
    for (let k = 0; k < ctx.per; k += 1) { const i = b * ctx.per + k; if (!ctx.isLayer(i) && active(sites, i)) ids.push(i); }
    if (ids.length < 2) continue;
    let best = null;
    for (let x = 0; x < ids.length; x += 1) {
      for (let y = x + 1; y < ids.length; y += 1) {
        const i = ids[x], j = ids[y];
        const m = mergedSite(sites, i, j, Math.max(area[i], 1e-3), Math.max(area[j], 1e-3));
        let gn = 0, lin = 0;
        for (const s of [i, j]) {
          for (let t = 0; t < 10; t += 1) {
            const d = m[t] - sites[s * SITE_FLOATS + t];
            if (!Number.isFinite(F[s * 10 + t])) continue;
            gn += F[s * 10 + t] * d * d;
            lin += g[s * 10 + t] * d;
          }
        }
        const na = area[i], nb = area[j];
        let dm = 0;
        for (let c = 0; c < 3; c += 1) dm += (tsum[i * 3 + c] / Math.max(na, 1) - tsum[j * 3 + c] / Math.max(nb, 1)) ** 2;
        const ward = na + nb > 0 ? ((na * nb) / (na + nb)) * dm : 0;
        const keep = na >= nb ? i : j;
        const c = { b, keep, drop: keep === i ? j : i, site: m, gn, lin, ward };
        const key = (z) => (ctx.dropPred === "ward" ? z.ward : z.gn + z.lin);
        if (!best || key(c) < key(best)) best = c;
      }
    }
    out.push(best);
  }
  return out;
}

function applyMerges(sites, merges) {
  const s = Float32Array.from(sites);
  for (const m of merges) { s.set(m.site, m.keep * SITE_FLOATS); s[m.drop * SITE_FLOATS] = -1; s[m.drop * SITE_FLOATS + 1] = -1; }
  return s;
}

function mergePred(ctx, merges) {
  const z = () => new Float64Array(ctx.images.length);
  const predGn = z(), predLin = z(), ward = z();
  for (const m of merges) { predGn[m.b] = m.gn; predLin[m.b] = m.lin; ward[m.b] = m.ward; }
  const pred = ctx.dropPred === "ward" ? Float64Array.from(ward) : predGn.map((v, i) => v + predLin[i]);
  return { predGn, predLin, ward, pred, images: merges.map((m) => m.b) };
}

const ACTIONS = [
  {
    key: "drop", type: "drop",
    preview(ctx, state, snap, env) {
      let merges = bestMerges(ctx, snap, state.sites);
      if (env.lambda !== undefined) {
        const c = env.cal.drop ?? 1;
        merges = merges.filter((m) => c * (ctx.dropPred === "ward" ? m.ward : m.gn + m.lin) - env.lambda * env.bitsPerSite < 0);
      }
      if (!merges.length) return null;
      return { state: withState(state, { sites: applyMerges(state.sites, merges) }), ...mergePred(ctx, merges) };
    },
  },
  ...GROUPS.flatMap((grp) => [
    {
      key: `coarsen:${grp}`, type: "coarsen", group: grp,
      preview(ctx, state, snap, env) {
        if (!Number.isFinite(state.baseSteps[grp])) return null;
        const next = withState(state, {});
        next.stepMul[grp] *= 2;
        return { state: next, ...quantPred(ctx, snap, state.sites, env.curDec, next.sites, stepsOf(next)) };
      },
    },
    {
      key: `refine:${grp}`, type: "refine", group: grp, calKey: `coarsen:${grp}`,
      preview(ctx, state, snap, env) {
        if (!Number.isFinite(state.baseSteps[grp]) || state.stepMul[grp] <= 0.25) return null;
        const next = withState(state, {});
        next.stepMul[grp] /= 2;
        return { state: next, ...quantPred(ctx, snap, state.sites, env.curDec, next.sites, stepsOf(next)) };
      },
    },
  ]),
  ...Object.keys(RESTRICTS).map((r) => ({
    key: `restrict:${r}`, type: "restrict", restrict: r,
    preview(ctx, state, snap, env) {
      if (state.restrict.has(r)) return null;
      const next = withState(state, {});
      next.restrict.add(r);
      next.med = state.med ?? medians(ctx, state.sites);
      next.sites = project(ctx, Float32Array.from(state.sites), next.restrict, next.med);
      return { state: next, ...quantPred(ctx, snap, state.sites, env.curDec, next.sites, stepsOf(next)) };
    },
  })),
  ...Object.keys(FREES).map((r) => ({
    key: `lift:${r}`, type: "lift", restrict: r,
    preview(ctx, state, snap) {
      if (!state.restrict.has(r)) return null;
      const next = withState(state, {});
      next.restrict.delete(r);
      // Preview for ΔR: one clipped Newton step on the freed parameter.
      const s = Float32Array.from(state.sites), gain = new Float64Array(ctx.images.length);
      const { ts, clip } = FREES[r];
      for (let i = 0; i < ctx.n; i += 1) {
        if (ctx.isLayer(i) || !active(s, i)) continue;
        for (const t of ts) {
          const f = snap.F[i * 10 + t], gg = snap.g[i * 10 + t];
          if (!(f > 1e-9)) continue;
          const d = Math.max(-clip, Math.min(clip, -gg / (2 * f)));
          s[i * SITE_FLOATS + t] += d;
          gain[Math.floor(i / ctx.per)] += f * d * d + gg * d;
        }
      }
      return { state: next, bitsSites: s, pred: gain, predGn: gain, predLin: gain.map(() => 0) };
    },
  })),
];

// ---- Refits ----

// Refits the GPU's current sites with the state's projection; returns the new
// { sites, raw }.
async function refit(ctx, state, opts) {
  const { tr, args } = ctx;
  restoreSites(tr, state.raw);
  const patch = { iters: opts.refitIters, lrMult: args.lrMult * opts.refitLr, errWeight: 0 };
  if (state.restrict.size) {
    await continueRun(tr, { ...patch, logFreq: 1 }, (info) => writeRgbSites(tr, project(ctx, info.sitesData, state.restrict, state.med)));
  } else {
    await continueRun(tr, { ...patch, logFreq: opts.refitIters + 1 });
  }
  const raw = await saveSites(tr);
  tr._rebuildCandidates(ctx.n);
  return { sites: rgbSitesOf(tr, raw), raw };
}

// Writes a state's sites to the GPU and returns its raw buffer copy.
async function load(ctx, sites) {
  writeRgbSites(ctx.tr, sites);
  ctx.tr._rebuildCandidates(ctx.n);
  return saveSites(ctx.tr);
}

async function startState(ctx) {
  const raw = await saveSites(ctx.tr);
  const sites = rgbSitesOf(ctx.tr, raw);
  const { F } = await ctx.tr.siteSensitivity();
  const baseSteps = await stepsFor(ctx, sites, F);
  restoreSites(ctx.tr, raw);
  return { sites, raw, baseSteps, stepMul: Object.fromEntries(GROUPS.map((k) => [k, 1])), restrict: new Set(), med: null };
}

// ---- Calibration ----

async function calibrate(ctx, opts) {
  const start = await startState(ctx);
  const steps = stepsOf(start);
  const base = await measure(ctx, start.sites, steps, start.raw, { perTile: true });
  const snap = await snapshot(ctx, start.sites, start.raw, steps);
  const env = { curDec: base.dec };
  const controls = [];
  for (let k = 0; k < 2; k += 1) {
    const c = await refit(ctx, start, opts);
    controls.push({ ...c, m: await measure(ctx, c.sites, steps, c.raw, { perTile: true }) });
  }
  const ctlSse = controls[0].m.tileSse.map((v, b) => (v + controls[1].m.tileSse[b]) / 2);
  const ctlBits = controls[0].m.tileBits.map((v, b) => (v + controls[1].m.tileBits[b]) / 2);
  const ctlPooled = { sse: mean(controls.map((c) => c.m.sse)), bits: mean(controls.map((c) => c.m.bits)) };
  console.error(`Calib  | start ${scores(ctx, base).psnr.toFixed(3)} dB, ${base.bits} bits; control refits ` +
                controls.map((c) => scores(ctx, c.m).psnr.toFixed(3)).join(" / ") + " dB");
  const noise = spearman(Array.from(controls[0].m.tileSse.map((v, b) => v - base.tileSse[b])),
                         Array.from(controls[1].m.tileSse.map((v, b) => v - base.tileSse[b])));
  const rows = [];
  for (const act of ACTIONS.filter((a) => ["drop", "coarsen", "restrict"].includes(a.type))) {
    const pv = act.preview(ctx, start, snap, env);
    if (!pv) continue;
    const next = pv.state, nsteps = stepsOf(next);
    const nraw = await load(ctx, next.sites);
    const ren = await measure(ctx, next.sites, nsteps, start.raw, { perTile: true });
    let ref;
    if (act.type === "coarsen") {
      ref = await measure(ctx, controls[0].sites, nsteps, controls[0].raw, { perTile: true });
      ref.tileSse = ref.tileSse.map((v, b) => v - controls[0].m.tileSse[b]);
      ref.tileBits = ref.tileBits.map((v, b) => v - controls[0].m.tileBits[b]);
      ref.dSse = ref.sse - controls[0].m.sse; ref.dBits = ref.bits - controls[0].m.bits;
    } else {
      const r = await refit(ctx, { ...next, raw: nraw }, opts);
      ref = await measure(ctx, r.sites, nsteps, r.raw, { perTile: true });
      ref.tileSse = ref.tileSse.map((v, b) => v - ctlSse[b]);
      ref.tileBits = ref.tileBits.map((v, b) => v - ctlBits[b]);
      ref.dSse = ref.sse - ctlPooled.sse; ref.dBits = ref.bits - ctlPooled.bits;
    }
    restoreSites(ctx.tr, start.raw);
    const imgs = pv.images ?? ctx.images.map((_, b) => b);
    const pick = (a) => imgs.map((b) => a[b]);
    const r = {
      key: act.key, type: act.type, images: imgs.length,
      pred: pick(pv.pred), predGn: pick(pv.predGn), ward: pv.ward ? pick(pv.ward) : null,
      render: pick(ren.tileSse.map((v, b) => v - base.tileSse[b])), refit: pick(ref.tileSse),
      predBits: pick(ren.tileBits.map((v, b) => v - base.tileBits[b])), measBits: pick(ref.tileBits),
      pooled: { pred: sum(pv.pred), predGn: sum(pv.predGn), render: ren.sse - base.sse, refit: ref.dSse,
                predBits: ren.bits - base.bits, measBits: ref.dBits },
    };
    const st = (x, y) => ({ spearman: spearman(x, y), slope: slope0(x, y) });
    r.stats = {
      gnLinRender: st(r.pred, r.render), gnLinRefit: st(r.pred, r.refit), gnRender: st(r.predGn, r.render), gnRefit: st(r.predGn, r.refit),
      wardRefit: r.ward ? st(r.ward, r.refit) : null, bits: st(r.predBits, r.measBits),
    };
    rows.push(r);
    const s = r.stats, p = r.pooled;
    console.error(`         ${act.key.padEnd(16)} ${String(imgs.length).padStart(5)} img | ΔD pooled pred ${p.pred.toFixed(1)} render ${p.render.toFixed(1)} ` +
                  `refit ${p.refit.toFixed(1)} | ρ/slope render ${f2(s.gnLinRender.spearman)}/${f3(s.gnLinRender.slope)} refit ` +
                  `${f2(s.gnLinRefit.spearman)}/${f3(s.gnLinRefit.slope)} | ΔR pred ${p.predBits} meas ${p.measBits.toFixed(0)} ρ ${f2(s.bits.spearman)} slope ${f3(s.bits.slope)}`);
  }
  // Calibration factor per action key: pooled refit-measured ΔD over pooled
  // prediction (coarsen: render), as the greedy prices pooled sums (the
  // per-image slope, weighted to the largest images, comes out lower), clamped.
  const cal = {};
  for (const r of rows) {
    const p = r.pooled, pred = ctx.dropPred === "ward" && r.ward ? sum(r.ward) : p.pred;
    const s = (r.type === "coarsen" ? p.render : p.refit) / pred;
    cal[r.key] = Number.isFinite(s) && pred > 0 ? Math.min(1.5, Math.max(0.01, s)) : 1;
  }
  return { start: scores(ctx, base), controls: controls.map((c) => scores(ctx, c.m)), controlNoiseSpearman: noise, rows, cal };
}

// ---- Greedy ----

async function greedy(ctx, start, lambda, cal, opts) {
  let state = start;
  let cur = await measure(ctx, state.sites, stepsOf(state), state.raw);
  const J = (m) => m.sse + lambda * m.bits;
  const log = [];
  let accepted = 0;
  for (let step = 0; step < opts.maxSteps; step += 1) {
    const snap = await snapshot(ctx, state.sites, state.raw, stepsOf(state));
    const env = { lambda, cal: { drop: cal.drop ?? 1 }, bitsPerSite: cur.bits / Math.max(1, cur.coded), curDec: cur.dec };
    const cands = [];
    for (const act of ACTIONS) {
      const pv = act.preview(ctx, state, snap, env);
      if (!pv) continue;
      const c = cal[act.calKey ?? act.key] ?? 1;
      const predD = sum(pv.pred), nsteps = stepsOf(pv.state);
      const e = encode(ctx, pv.bitsSites ?? pv.state.sites, nsteps);
      const ren = await renderSse(ctx, encode(ctx, pv.state.sites, nsteps).dec);
      restoreSites(ctx.tr, state.raw);
      cands.push({ act, pv, cal: c, predD, predDcal: c * predD, predR: e.bits - cur.bits, renderD: ren.total - cur.sse,
                   predJ: c * predD + lambda * (e.bits - cur.bits), images: pv.images?.length });
    }
    cands.sort((a, b) => a.predJ - b.predJ);
    const good = cands.filter((c) => c.predJ < 0);
    console.error(`  λ ${lambda} step ${step}: J ${J(cur).toFixed(1)}, ${cur.bits} bits | best ` +
                  cands.slice(0, 4).map((c) => `${c.act.key} ${c.predJ.toFixed(1)}`).join(", "));
    if (!good.length) break;
    const ctl = await refit(ctx, state, opts);
    const ctlM = await measure(ctx, ctl.sites, stepsOf(state), ctl.raw);
    let took = false;
    for (const c of good.slice(0, opts.tries)) {
      const next = c.pv.state;
      next.raw = await load(ctx, next.sites);
      const r = await refit(ctx, next, opts);
      const m = await measure(ctx, r.sites, stepsOf(next), r.raw);
      const entry = {
        step, action: c.act.key, type: c.act.type, images: c.images, cal: c.cal,
        predD: c.predD, predDcal: c.predDcal, renderD: c.renderD, measD: m.sse - ctlM.sse,
        predR: c.predR, measR: m.bits - ctlM.bits, predJ: c.predJ, measJ: J(m) - J(ctlM),
        controlGainD: ctlM.sse - cur.sse, controlGainR: ctlM.bits - cur.bits,
      };
      entry.accepted = entry.measJ < 0;
      log.push(entry);
      console.error(`    ${c.act.key.padEnd(16)} ΔD pred ${c.predD.toFixed(1)} cal ${c.predDcal.toFixed(1)} render ${c.renderD.toFixed(1)} ` +
                    `refit ${entry.measD.toFixed(1)} | ΔR pred ${c.predR} meas ${entry.measR} | ΔJ pred ${c.predJ.toFixed(1)} ` +
                    `meas ${entry.measJ.toFixed(1)} ${entry.accepted ? "accepted" : "rejected"}`);
      if (entry.accepted) {
        state = { ...next, sites: r.sites, raw: r.raw };
        cur = m;
        accepted += 1;
        took = true;
        break;
      }
    }
    restoreSites(ctx.tr, state.raw);
    if (!took) break;
  }
  return { state, final: cur, accepted, log };
}

// The start state refit k times with nothing changed, for k = 0…maxK.
async function controlArm(ctx, start, maxK, opts) {
  const out = [await measure(ctx, start.sites, stepsOf(start), start.raw)];
  let s = start;
  for (let k = 1; k <= maxK; k += 1) {
    const r = await refit(ctx, s, opts);
    s = { ...s, ...r };
    out.push(await measure(ctx, s.sites, stepsOf(start), s.raw));
  }
  restoreSites(ctx.tr, start.raw);
  return out;
}

// Evaluates a trained baseline (codec steps of its own state at rateLossDb),
// then after 1…maxK refits as the greedy's (the trainer's own args, so the
// rate term stays on for a rate baseline), at the same steps.
async function evalTrained(ctx, maxK, opts, lambda = 0) {
  const st = await startState(ctx);
  return (await controlArm(ctx, st, maxK, opts)).map((m) => ({ ...scores(ctx, m, lambda), bits: m.bits, sse: m.sse }));
}

// ---- Printing ----

const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : "—");
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : "—");

// Frontier PSNR at a given bpp: linear between neighbouring sweep points.
function frontierAt(points, bpp) {
  const p = [...points].sort((a, b) => a.bpp - b.bpp);
  for (let k = 0; k + 1 < p.length; k += 1) {
    if (bpp >= p[k].bpp && bpp <= p[k + 1].bpp) return p[k].psnr + ((bpp - p[k].bpp) / (p[k + 1].bpp - p[k].bpp)) * (p[k + 1].psnr - p[k].psnr);
  }
  return NaN;
}

function summarizeGreedy(logs) {
  const out = {};
  const types = [...new Set(logs.map((e) => e.type))];
  for (const t of [...types, "all"]) {
    const es = logs.filter((e) => t === "all" || e.type === t);
    const col = (k) => es.map((e) => e[k]);
    out[t] = { n: es.length, accepted: es.filter((e) => e.accepted).length,
               dSpearman: spearman(col("predDcal"), col("measD")), dSlopeRaw: slope0(col("predD"), col("measD")),
               dSlopeCal: slope0(col("predDcal"), col("measD")), renderSlope: slope0(col("predD"), col("renderD")),
               rSpearman: spearman(col("predR"), col("measR")), rSlope: slope0(col("predR"), col("measR")) };
  }
  return out;
}

// ---- Main ----

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, from: { type: "string", default: "1024" }, count: { type: "string", default: "2048" },
      "calib-from": { type: "string", default: "0" }, "calib-count": { type: "string", default: "1024" }, calib: { type: "string" },
      lambdas: { type: "string", default: "0.03,0.1,0.3" }, sites: { type: "string", default: "4,6,8,12" },
      "refit-iters": { type: "string", default: "40" }, "refit-lr": { type: "string", default: "0.25" },
      "max-steps": { type: "string", default: "20" }, tries: { type: "string", default: "3" },
      repeats: { type: "string", default: "1" }, "drop-pred": { type: "string", default: "gn" },
      "no-baselines": { type: "boolean", default: false }, out: { type: "string" }, help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write("Usage: node structure.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [--from 1024] [--count 2048]\n" +
                         "         [--preset json] [--set key=value …] [--calib-from 0] [--calib-count 1024 | --calib json]\n" +
                         "         [--lambdas 0.03,0.1,0.3] [--sites 4,6,8,12] [--refit-iters 40] [--refit-lr 0.25]\n" +
                         "         [--max-steps 20] [--tries 3] [--repeats 1] [--drop-pred gn|ward] [--no-baselines] [--out json]\n");
    process.exit(cli.help ? 0 : 1);
  }
  const base = await loadParams(cli.preset, cli.set);
  const { args, config } = await argsFor({ ...base, rateLambda: 0 });
  args.logFreq = args.iters;
  const maxDim = Number(config.MAX_DIM || 0);
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label}`);
  const opts = { refitIters: Number(cli["refit-iters"]), refitLr: Number(cli["refit-lr"]), maxSteps: Number(cli["max-steps"]),
                 tries: Number(cli.tries) };
  const lambdas = cli.lambdas.split(",").filter(Boolean).map(Number);
  const dropPred = cli["drop-pred"];
  const mk = (bt, part) => ({ ...atlasCtx(bt, part, bt.args ?? args), dropPred });
  const result = { dataset: cli.mnist || cli.parquet || cli.images, preset: cli.preset, from: Number(cli.from), count: Number(cli.count), opts, lambdas, dropPred };

  // Calibration
  let calib = null;
  if (cli.calib) calib = JSON.parse(await readFile(cli.calib, "utf8")).calibration;
  else if (Number(cli["calib-count"]) > 0) {
    const imgs = await loadImages({ ...cli, from: cli["calib-from"], count: cli["calib-count"] }, maxDim);
    console.error(`Calib  | images ${cli["calib-from"]}–${Number(cli["calib-from"]) + imgs.length - 1}`);
    await trainBatched({ device, shaders, args, quiet: true }, imgs, { after: async (bt, idx) => {
      calib = await calibrate(mk(bt, idx.map((i) => imgs[i])), opts);
    } });
  }
  const cal = calib?.cal ?? {};
  result.calibration = calib;
  console.error(`Cal    | ${Object.entries(cal).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ")}`);

  // Held out
  const images = await loadImages(cli, maxDim);
  const repeats = Math.max(1, Number(cli.repeats) | 0);
  result.repeats = [];
  for (let rep = 0; rep < repeats; rep += 1) {
    const R = { greedy: {}, controlArm: [], start: null, rate: {}, sweep: {} };
    let maxK = 0;
    await trainBatched({ device, shaders, args, quiet: true }, images, { after: async (bt, idx) => {
      const ctx = mk(bt, idx.map((i) => images[i]));
      const start = await startState(ctx);
      const sm = await measure(ctx, start.sites, stepsOf(start), start.raw);
      R.start = { ...scores(ctx, sm), bits: sm.bits, sse: sm.sse, steps: start.baseSteps, planeBits: sm.planeBits, positionBits: sm.positionBits };
      console.error(`Start  | rep ${rep}: ${R.start.psnr.toFixed(3)} dB median, ${R.start.bpp.toFixed(4)} bpp, ${R.start.bitsPerSite.toFixed(1)} bits/site`);
      for (const lambda of lambdas) {
        const g = await greedy(ctx, start, lambda, cal, opts);
        restoreSites(ctx.tr, start.raw);
        maxK = Math.max(maxK, g.accepted);
        R.greedy[lambda] = { ...scores(ctx, g.final, lambda), accepted: g.accepted, log: g.log, actions: g.log.filter((e) => e.accepted).map((e) => e.action),
                             stepMul: g.state.stepMul, restrict: [...g.state.restrict], planeBits: g.final.planeBits, positionBits: g.final.positionBits };
        console.error(`Greedy | λ ${lambda}: ${R.greedy[lambda].psnr.toFixed(3)} dB, ${R.greedy[lambda].bpp.toFixed(4)} bpp after ${g.accepted} actions: ` +
                      R.greedy[lambda].actions.join(", "));
      }
      const arm = await controlArm(ctx, start, maxK, opts);
      R.controlArm = arm.map((m) => ({ ...scores(ctx, m), bits: m.bits, sse: m.sse }));
      for (const lambda of lambdas) R.greedy[lambda].control = { ...scores(ctx, arm[R.greedy[lambda].accepted], lambda) };
    } });
    if (!cli["no-baselines"]) {
      for (const lambda of lambdas) {
        const { args: a } = await argsFor({ ...base, rateLambda: lambda });
        a.logFreq = a.iters;
        const k = R.greedy[lambda].accepted;
        await trainBatched({ device, shaders, args: a, quiet: true }, images, { after: async (bt, idx) => {
          const arm = await evalTrained(mk(bt, idx.map((i) => images[i])), k, opts, lambda);
          R.rate[lambda] = { ...arm[0], refit: arm[k] };
        } });
        console.error(`Rate   | λ ${lambda}: ${R.rate[lambda].psnr.toFixed(3)} dB, ${R.rate[lambda].bpp.toFixed(4)} bpp; ` +
                      `after ${k} refits ${R.rate[lambda].refit.psnr.toFixed(3)} dB, ${R.rate[lambda].refit.bpp.toFixed(4)} bpp`);
      }
      for (const s of cli.sites.split(",").filter(Boolean).map(Number)) {
        const { args: a } = await argsFor({ ...base, sites: s, rateLambda: 0 });
        a.logFreq = a.iters;
        await trainBatched({ device, shaders, args: a, quiet: true }, images, { after: async (bt, idx) => {
          const arm = await evalTrained(mk(bt, idx.map((i) => images[i])), maxK, opts);
          R.sweep[s] = { ...arm[0], arm };
        } });
        console.error(`Sweep  | ${s} sites: ${R.sweep[s].psnr.toFixed(3)} dB, ${R.sweep[s].bpp.toFixed(4)} bpp`);
      }
    }
    result.repeats.push(R);
  }

  // Summary: medians over repeats.
  const med = (f) => median(result.repeats.map(f).filter(Number.isFinite));
  const n = images.length;
  const jOf = (r, lambda) => (r.sse + lambda * r.bits) / n;
  // The sweep frontier (the start state is its preset-count point), each point after k refits.
  const sweepPts = (R, k = 0) => [...Object.values(R.sweep).map((p) => p.arm[k]), R.controlArm[k]];
  result.summary = lambdas.map((lambda) => ({
    lambda,
    greedy: { psnr: med((R) => R.greedy[lambda].psnr), bpp: med((R) => R.greedy[lambda].bpp), j: med((R) => R.greedy[lambda].jPerImage) },
    control: { psnr: med((R) => R.greedy[lambda].control.psnr), bpp: med((R) => R.greedy[lambda].control.bpp), j: med((R) => R.greedy[lambda].control.jPerImage) },
    rate: { psnr: med((R) => R.rate[lambda]?.psnr), bpp: med((R) => R.rate[lambda]?.bpp), j: med((R) => R.rate[lambda]?.jPerImage) },
    rateRefit: { psnr: med((R) => R.rate[lambda]?.refit.psnr), bpp: med((R) => R.rate[lambda]?.refit.bpp), j: med((R) => R.rate[lambda]?.refit.jPerImage) },
    sweepBestJ: med((R) => Math.min(...sweepPts(R).map((p) => jOf(p, lambda)))),
    sweepBestJRefit: med((R) => Math.min(...sweepPts(R, R.greedy[lambda].accepted).map((p) => jOf(p, lambda)))),
    frontierAtGreedy: med((R) => frontierAt(sweepPts(R, R.greedy[lambda].accepted), R.greedy[lambda].bpp)),
    frontierAtRate: med((R) => (R.rate[lambda] ? frontierAt(sweepPts(R), R.rate[lambda].bpp) : NaN)),
  }));
  result.greedyCalibration = summarizeGreedy(result.repeats.flatMap((R) => Object.values(R.greedy).flatMap((g) => g.log)));

  const log = console.error;
  if (calib) {
    log(`Calibration (images ${cli["calib-from"]}+, per image; ΔD predicted GN+g vs measured; ΔR predicted pre-refit vs after refit)`);
    log(`  action           imgs  ρ render  slope render  ρ refit  slope refit  ρ GN-only refit  ρ ward refit  ρ ΔR  slope ΔR | pooled ΔD pred/render/refit  ΔR pred/meas`);
    for (const r of calib.rows) {
      const s = r.stats, p = r.pooled;
      log(`  ${r.key.padEnd(16)} ${String(r.images).padStart(5)}  ${f2(s.gnLinRender.spearman).padStart(8)}  ${f3(s.gnLinRender.slope).padStart(12)}  ` +
          `${f2(s.gnLinRefit.spearman).padStart(7)}  ${f3(s.gnLinRefit.slope).padStart(11)}  ${f2(s.gnRefit.spearman).padStart(15)}  ` +
          `${f2(s.wardRefit?.spearman).padStart(12)}  ${f2(s.bits.spearman).padStart(4)}  ${f3(s.bits.slope).padStart(8)} | ` +
          `${p.pred.toFixed(0)}/${p.render.toFixed(0)}/${p.refit.toFixed(0)}  ${p.predBits}/${p.measBits.toFixed(0)}`);
    }
  }
  log(`Greedy steps, predicted vs measured (against the control refit), pooled over λ and repeats:`);
  for (const [t, s] of Object.entries(result.greedyCalibration)) {
    log(`  ${t.padEnd(9)} n ${String(s.n).padStart(3)} acc ${String(s.accepted).padStart(3)} | ΔD ρ ${f2(s.dSpearman)} slope raw ${f3(s.dSlopeRaw)} ` +
        `cal ${f3(s.dSlopeCal)} render/raw ${f3(s.renderSlope)} | ΔR ρ ${f2(s.rSpearman)} slope ${f3(s.rSlope)}`);
  }
  log(`Held out (${n} images from ${cli.from}, ${repeats} repeat(s); median per-image PSNR of decoded sites, measured bpp, J/image = (SSE + λ·bits)/images):`);
  log(`  start (preset, λ 0): ${med((R) => R.start.psnr).toFixed(3)} dB, ${med((R) => R.start.bpp).toFixed(4)} bpp`);
  for (const s of Object.keys(result.repeats[0].sweep)) log(`  sweep ${s} sites: ${med((R) => R.sweep[s].psnr).toFixed(3)} dB, ${med((R) => R.sweep[s].bpp).toFixed(4)} bpp`);
  log(`  (k = the greedy's accepted actions; "+k" = after k refits as the greedy's, nothing changed; frontier = the sweep's, linear in bpp)`);
  log(`  λ      greedy dB / bpp / J        control +k dB / bpp / J    rate term dB / bpp / J     rate +k dB / bpp / J       sweep best J / +k   frontier dB +k @greedy bpp  frontier @rate bpp`);
  for (const s of result.summary) {
    const t = (o) => `${f3(o.psnr)} / ${o.bpp?.toFixed(4)} / ${f3(o.j)}`;
    log(`  ${String(s.lambda).padEnd(6)} ${t(s.greedy).padEnd(26)} ${t(s.control).padEnd(26)} ${t(s.rate).padEnd(26)} ${t(s.rateRefit).padEnd(26)} ` +
        `${(f3(s.sweepBestJ) + " / " + f3(s.sweepBestJRefit)).padEnd(19)} ${f3(s.frontierAtGreedy).padEnd(27)} ${f3(s.frontierAtRate)}`);
  }
  for (const lambda of lambdas) log(`  λ ${lambda} actions: ${result.repeats.map((R) => R.greedy[lambda].actions.join(" ") || "(none)").join(" | ")}`);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/structure_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(result, (k, v) => (v instanceof Float64Array || v instanceof Float32Array ? undefined : v), 1) + "\n");
  log(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
  });
}
