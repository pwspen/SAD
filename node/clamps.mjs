#!/usr/bin/env node
// E8: is "share of sites at a bound × their outward gradient" a signal for
// extending a parameter's range? Batched runs on held-out images:
//
//   node clamps.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 \
//     --preset ../presets/mnist.json --repeats 3
//
// The bounds the trainer applies to site parameters (shaders and trainer.js):
//   aniso      log aniso clamped to ±5 after every Adam step (adam.wgsl).
//   position   clamped to the site's own tile, [origin, origin + W − 1] per
//              axis (adam.wgsl).
//   beta       the histogram layer's depth β = τ · radius / max(W, H) clamped
//              to ±8 (adam.wgsl; trainer.js LAYER_BETA_MAX clamps its start).
//   colorFloor logchroma decodes rgb = e^(…) − 1/255, so every channel stays
//              above −1/255, approached only asymptotically (no hard clamp).
//              LOG_CHROMA_EPS is shared by the WGSL and color_spaces.js (a JS
//              const: the layer's color is encoded there and never learned),
//              so it can't be moved by patching shader text: signal only.
//   log τ, radius: no bound anywhere (vpt.wgsl floors τ at 1e-4 for candidate
//              ranking only; the share below it is reported as tauFloor).
// Signal, at the end of each baseline run: per bound, the share of active
//   sites at it (within 1e-3; layers per tile for beta), their mean outward
//   gradient (−∂SSE/∂θ along the outward normal, exact SSE gradient from one
//   zero-LR iteration with weighting off; positive = wants out), the share of
//   those pointing out, share × mean outward gradient (the raw signal, in each
//   parameter's own units), and a unit-free one: the SSE a Newton step past
//   the bound would remove, Σ max(0, out)² / (4 F) with F the Gauss–Newton
//   diagonal, as dB on the atlas.
// Extension: each bound is moved by injecting a patched copy of adam.wgsl's
//   text into this run's shaders (shaders/ is untouched): log aniso ±8 and
//   ±12, positions --pos-margin px past the tile, β ±16. Every variant
//   retrains the held-out images (--from/--count, one atlas), --repeats
//   times, interleaved; ΔPSNR is the mean over images of the per-image PSNR
//   minus the same repeat's baseline (paired), mean ± sd over repeats. The
//   signal is also read at the extended bound, to see whether sites pile up
//   there again. --base tightens bounds first (e.g. aniso=0.5,posMargin=-2,
//   beta=1, a negative margin keeping sites inside the tile): the baseline
//   runs there and each variant restores one bound to stock, so the signal
//   is tested where it fires.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=20261019";
import { tileOrigin } from "../trainer/batch.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median, mean } from "./common.mjs";
import { sseGradient, saveSites, rgbSitesOf, dbOf, spearman } from "./lossmap.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const AT = 1e-3; // within this of a bound counts as at it
const LOG_CHROMA_FLOOR = -1 / 255;
const TAU_RANK_FLOOR = Math.log(1e-4);

const BASE = { aniso: 5, posMargin: 0, beta: 8 };
const ANISO_RE = /clamp\(sites\[idx\]\.log_aniso, -5\.0, 5\.0\)/;
const POS_RE = /origin, origin \+ vec2<f32>\(f32\(params\.width - 1u\), f32\(params\.height - 1u\)\)\);/;
const BETA_RE = /, -8\.0, 8\.0\);/;
const wf = (x) => (Number.isInteger(x) ? `${x}.0` : String(x));

// adam.wgsl with `bounds` in place of the stock ones (a copy of the shader set).
export function patchShaders(shaders, b) {
  let src = shaders.ADAM;
  for (const re of [ANISO_RE, POS_RE, BETA_RE]) if (!re.test(src)) throw new Error(`adam.wgsl: no match for ${re}`);
  src = src.replace(ANISO_RE, `clamp(sites[idx].log_aniso, -${wf(b.aniso)}, ${wf(b.aniso)})`)
    .replace(POS_RE, `origin - vec2<f32>(${wf(b.posMargin)}), origin + vec2<f32>(f32(params.width - 1u) + ${wf(b.posMargin)}, ` +
                     `f32(params.height - 1u) + ${wf(b.posMargin)}));`)
    .replace(BETA_RE, `, -${wf(b.beta)}, ${wf(b.beta)});`);
  return { ...shaders, ADAM: src };
}

const CLAMPS = ["aniso", "position", "beta", "colorFloor", "tauFloor"];
const newAcc = () => Object.fromEntries(CLAMPS.map((k) => [k, { n: 0, at: 0, outSum: 0, outPos: 0, gainSse: 0 }]));

// Adds one at-bound coordinate: out = outward descent component, F its GN diagonal.
function addOut(a, out, F) {
  a.outSum += out;
  if (out > 0) { a.outPos += 1; if (F > 0) a.gainSse += (out * out) / (4 * F); }
}

// Bound statistics of a trained atlas, added into acc (see header).
async function clampStats(bt, b, acc) {
  const tr = bt.trainer, L = bt.layout, n = tr.actualSites, per = L.sitesPerTile, layer = bt.layerSlot;
  const g = await sseGradient(tr);
  const { F, samples } = await tr.siteSensitivity();
  const s = rgbSitesOf(tr, await saveSites(tr));
  const { psnr } = await tr.renderEdited();
  acc.sse0 = (acc.sse0 ?? 0) + samples * 10 ** (-psnr / 10);
  const maxDim = Math.max(L.tileW, L.tileH);
  for (let i = 0; i < n; i += 1) {
    const o = i * SITE_FLOATS, f = i * 10;
    if (!(s[o] >= 0)) continue;
    if (layer !== null && i % per === layer) {
      const k = Math.exp(s[o + 2]) / maxDim, beta = s[o + 3] * k, a = acc.beta;
      a.n += 1;
      if (Math.abs(beta) >= b.beta - AT) { a.at += 1; addOut(a, -Math.sign(beta) * g[f + 3] / k, F[f + 3] / (k * k)); }
      continue;
    }
    // log aniso
    {
      const a = acc.aniso, la = s[o + 9];
      a.n += 1;
      if (Math.abs(la) >= b.aniso - AT) { a.at += 1; addOut(a, -Math.sign(la) * g[f + 9], F[f + 9]); }
    }
    // position, per axis and side
    {
      const a = acc.position, org = tileOrigin(L, Math.floor(i / per)), size = [L.tileW, L.tileH];
      a.n += 1;
      let at = false;
      for (let c = 0; c < 2; c += 1) {
        const lo = org[c] - b.posMargin, hi = org[c] + size[c] - 1 + b.posMargin;
        if (s[o + c] <= lo + AT) { at = true; addOut(a, g[f + c], F[f + c]); }
        else if (s[o + c] >= hi - AT) { at = true; addOut(a, -g[f + c], F[f + c]); }
      }
      if (at) a.at += 1;
    }
    // logchroma floor, per channel
    {
      const a = acc.colorFloor;
      a.n += 1;
      let at = false;
      for (let c = 0; c < 3; c += 1) {
        if (s[o + 4 + c] <= LOG_CHROMA_FLOOR + AT) { at = true; addOut(a, g[f + 4 + c], F[f + 4 + c]); }
      }
      if (at) a.at += 1;
    }
    // τ below the candidate-ranking floor
    {
      const a = acc.tauFloor;
      a.n += 1;
      if (s[o + 2] <= TAU_RANK_FLOOR) { a.at += 1; addOut(a, g[f + 2], F[f + 2]); }
    }
  }
}

function finishStats(acc) {
  const out = {};
  for (const k of CLAMPS) {
    const a = acc[k];
    out[k] = {
      n: a.n, atShare: a.n ? a.at / a.n : NaN, meanOut: a.at ? a.outSum / a.at : 0, outShare: a.at ? a.outPos / a.at : 0,
      shareTimesOut: a.n ? a.outSum / a.n : NaN, predGainDb: dbOf(-a.gainSse, acc.sse0),
    };
  }
  return out;
}

// Stock bounds: each extended past the stock value. With `base` (tightened
// bounds, e.g. { aniso: 0.5, posMargin: -2, beta: 1 }): the baseline runs
// at those, and each variant puts one of them back to stock, a check that the
// signal ranks gains where bounds do bind.
export const VARIANTS = (margin, base = null) => base ? [
  { name: "baseline", bounds: { ...BASE, ...base }, clamp: null },
  ...Object.keys(base).map((k) => ({ name: `${k} ${base[k]} → ${BASE[k]}`, bounds: { ...BASE, ...base, [k]: BASE[k] },
                                     clamp: k === "posMargin" ? "position" : k })),
] : [
  { name: "baseline", bounds: BASE, clamp: null },
  { name: "aniso ±8", bounds: { ...BASE, aniso: 8 }, clamp: "aniso" },
  { name: "aniso ±12", bounds: { ...BASE, aniso: 12 }, clamp: "aniso" },
  { name: `position +${margin}px`, bounds: { ...BASE, posMargin: margin }, clamp: "position" },
  { name: "beta ±16", bounds: { ...BASE, beta: 16 }, clamp: "beta" },
];

export async function clampsExperiment(ctx, args, images, { repeats = 3, margin = 4, only = null, base = null } = {}) {
  const variants = VARIANTS(margin, base).filter((v) => !only || v.clamp === null || only.includes(v.clamp));
  const runs = variants.map((v) => ({ ...v, reps: [] }));
  for (let r = 0; r < repeats; r += 1) {
    for (const v of runs) {
      const acc = newAcc();
      const shaders = patchShaders(ctx.shaders, v.bounds);
      const { results, seconds } = await trainBatched({ ...ctx, shaders, args, quiet: true }, images,
                                                      { after: (bt) => clampStats(bt, v.bounds, acc) });
      const psnrs = results.map((x) => x.psnr);
      v.reps.push({ psnrs, meanPsnr: mean(psnrs), medianPsnr: median(psnrs), seconds, stats: finishStats(acc) });
      console.error(`Run    | repeat ${r + 1}/${repeats} ${v.name.padEnd(16)} mean ${mean(psnrs).toFixed(3)} dB, median ${median(psnrs).toFixed(3)} dB, ${seconds.toFixed(1)}s`);
    }
  }
  const ref = runs[0];
  for (const v of runs) {
    v.delta = v.reps.map((rep, r) => mean(rep.psnrs.map((p, i) => p - ref.reps[r].psnrs[i])));
    v.deltaMedian = v.reps.map((rep, r) => rep.medianPsnr - ref.reps[r].medianPsnr);
    v.deltaMean = mean(v.delta);
    v.deltaSd = Math.sqrt(mean(v.delta.map((d) => (d - v.deltaMean) ** 2)) * repeats / Math.max(1, repeats - 1));
  }
  // Signal from the baseline runs, averaged over repeats.
  const signal = Object.fromEntries(CLAMPS.map((k) => [k, Object.fromEntries(
    Object.keys(ref.reps[0].stats[k]).map((f) => [f, mean(ref.reps.map((rep) => rep.stats[k][f]))]))]));
  const ext = runs.filter((v) => v.clamp);
  const ranking = {
    shareTimesOut: spearman(ext.map((v) => signal[v.clamp].shareTimesOut), ext.map((v) => v.deltaMean)),
    predGainDb: spearman(ext.map((v) => signal[v.clamp].predGainDb), ext.map((v) => v.deltaMean)),
    atShare: spearman(ext.map((v) => signal[v.clamp].atShare), ext.map((v) => v.deltaMean)),
  };
  const baseSpread = ref.reps.map((rep) => rep.meanPsnr);
  return {
    images: images.length, repeats, margin, signal, ranking,
    baselineRepeatSd: Math.sqrt(mean(baseSpread.map((x) => (x - mean(baseSpread)) ** 2)) * repeats / Math.max(1, repeats - 1)),
    variants: runs.map((v) => ({
      name: v.name, clamp: v.clamp, bounds: v.bounds, delta: v.delta, deltaMedian: v.deltaMedian, deltaMean: v.deltaMean, deltaSd: v.deltaSd,
      reps: v.reps.map(({ psnrs, ...rest }) => rest),
    })),
  };
}

const g3 = (x) => (Number.isFinite(x) ? (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4) ? x.toExponential(2) : x.toPrecision(3)) : String(x));

export function printReport(rep, log = console.error) {
  log(`Signal | baseline, mean of ${rep.repeats} runs on ${rep.images} images (outward gradient in each parameter's units: ` +
      `log aniso, px, β logit, rgb, log τ)`);
  log(`         bound       at bound   mean out   out share  share×out   Newton gain dB`);
  for (const [k, s] of Object.entries(rep.signal)) {
    log(`         ${k.padEnd(11)} ${(100 * s.atShare).toFixed(2).padStart(7)}%  ${g3(s.meanOut).padEnd(10)} ${(100 * s.outShare).toFixed(0).padStart(4)}%      ` +
        `${g3(s.shareTimesOut).padEnd(11)} ${g3(s.predGainDb)}`);
  }
  log(`         bounds ${JSON.stringify(rep.variants[0].bounds)}`);
  log(`Extend | variant           ΔPSNR mean ± sd (dB)   Δ median   at new bound   (baseline repeat sd ${rep.baselineRepeatSd.toFixed(3)} dB)`);
  for (const v of rep.variants) {
    if (!v.clamp) continue;
    const at = mean(v.reps.map((r) => r.stats[v.clamp].atShare));
    log(`         ${v.name.padEnd(17)} ${v.deltaMean >= 0 ? "+" : ""}${v.deltaMean.toFixed(3)} ± ${v.deltaSd.toFixed(3)}        ` +
        `${mean(v.deltaMedian) >= 0 ? "+" : ""}${mean(v.deltaMedian).toFixed(3)}     ${(100 * at).toFixed(2)}%`);
  }
  log(`Rank   | Spearman of signal vs ΔPSNR over the extensions: share×out ${g3(rep.ranking.shareTimesOut)}, ` +
      `Newton gain ${g3(rep.ranking.predGainDb)}, at-bound share ${g3(rep.ranking.atShare)}`);
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, from: { type: "string", default: "1024" }, count: { type: "string", default: "2048" },
      repeats: { type: "string", default: "3" }, "pos-margin": { type: "string", default: "4" },
      only: { type: "string" }, base: { type: "string" }, out: { type: "string" }, help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write("Usage: node clamps.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [--from 1024] [--count 2048]\n" +
                         "                       [--preset json] [--set key=value …] [--repeats 3] [--pos-margin 4]\n" +
                         "                       [--only aniso,position,beta] [--base aniso=0.5,posMargin=-2,beta=1] [--out json]\n");
    process.exit(cli.help ? 0 : 1);
  }
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label}`);
  const rep = await clampsExperiment({ device, shaders }, args, images, {
    repeats: Number(cli.repeats), margin: Number(cli["pos-margin"]), only: cli.only ? cli.only.split(",") : null,
    base: cli.base ? Object.fromEntries(cli.base.split(",").map((kv) => { const [k, v] = kv.split("="); if (!(k in BASE)) throw new Error(`--base: unknown bound ${k}`); return [k, Number(v)]; })) : null,
  });
  printReport(rep);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/clamps_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from), args, ...rep }, null, 1) + "\n");
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
