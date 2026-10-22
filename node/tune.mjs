#!/usr/bin/env node
// Headless auto-tune in batch mode: train.html's tune buttons without the
// browser, with a step that decays instead of stopping.
//
//   node tune.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 \
//     --preset ../presets/mnist.json --out ../results/tune_mnist.json
//   node tune.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
//     --preset ../presets/mnist.json --keys lrMult
//
// Every evaluation is one batched run over the same images, scored by their
// median final PSNR, so a round's runs are a small population around the
// current point. Each setting is searched in log space, one at a time, for
// --cycles passes over --keys:
//   - try center × r and center ÷ r (values already run aren't re-run);
//   - fit a parabola through the three medians and, when it opens down with
//     its peak inside (−2, 2) steps and not on a point already run, also try
//     that peak (a Newton step from the population);
//   - move to the best point measured; if that is still the center, decay the
//     step, r ← r^--decay (√r by default), rather than stopping;
//   - the setting is done when r falls below --min-ratio or after --rounds.
// Each setting keeps its step between cycles, so later cycles search finer.
// --method cma searches all --keys at once instead, with CMA-ES (cmaes below):
// a population of --lambda points per generation from a Gaussian in log space
// that learns which joint directions pay (e.g. lrMult against the bases).
// --budget caps either search at that many batches.
// Afterwards the start and tuned settings both run on held-out images.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, median } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const LR_KEYS = ["lrPosBase", "lrTauBase", "lrRadiusBase", "lrColorBase", "lrDirBase", "lrAnisoBase"];

const HELP = `Usage: node tune.mjs (--mnist <idx3.gz> | --parquet <file> | --images <dir>) [options]

  --from <n> --count <n>  Tuning images (default 0, 64).
  --preset <json> --set key=value   Start settings, as batch.mjs.
  --keys a,b,…        Settings to tune (default the six LR bases: ${LR_KEYS.join(",")}).
  --ratio <r>         Starting step (default 2).
  --decay <p>         Step decay when no neighbour helps: r ← r^p (default 0.5).
  --min-ratio <r>     A setting is done below this step (default 1.1).
  --rounds <n>        At most this many rounds per setting per cycle (default 8).
  --cycles <n>        Passes over --keys (default 2).
  --method <m>        coord (above, default) or cma: CMA-ES over all --keys at once.
  --sigma <s>         cma: starting step in natural-log units (default 0.25, ±1σ ≈ ×1.28).
  --lambda <n>        cma: points per generation (default 4 + ⌊3 ln n⌋ for n keys).
  --seed <n>          cma: sampling seed (default 1).
  --budget <n>        Stop after this many batches (default 0 = no cap; cma needs one or --min-ratio).
  --holdout <n>       Held-out images after the tuning ones for the final check (default --count; 0 = none).
  --out <json>        Result (default results/tune_<time>.json): start and tuned settings, every evaluation.
`;

// Settings are stored to 4 significant digits, as train.html's tune does.
const round4 = (v) => Number(v.toPrecision(4));

export async function makeEvaluator(ctx, params, images) {
  const cache = new Map();
  let runs = 0;
  return {
    get runs() { return runs; },
    async score(p) {
      const key = JSON.stringify(Object.entries(p).sort());
      if (cache.has(key)) return cache.get(key);
      const { args } = await argsFor(p);
      args.logFreq = args.iters; // log only the last iteration: readbacks cost more than these steps
      const { results, seconds } = await trainBatched({ ...ctx, args, quiet: true }, images);
      runs += 1;
      const m = median(results.map((r) => r.psnr));
      cache.set(key, m);
      ctx.trace.push({ params: { ...p }, median: m, seconds });
      return m;
    },
  };
}

// Tunes `keys` of `params` in place (see the header). opts.budget (optional)
// stops once the evaluator has run that many batches. Returns per-key logs and
// `history`, the incumbent each time it changes ({ runs, params, median }).
export async function tune(evaluator, params, keys, opts, log = console.error) {
  const ratio = new Map(keys.map((k) => [k, opts.ratio]));
  const spent = () => opts.budget > 0 && evaluator.runs >= opts.budget;
  let best = await evaluator.score(params);
  log(`Start  | median ${best.toFixed(3)} dB`);
  const changes = [];
  const history = [{ runs: evaluator.runs, params: { ...params }, median: best }];
  for (let cycle = 1; cycle <= opts.cycles && !spent(); cycle += 1) {
    for (const key of keys) {
      if (spent()) break;
      const start = params[key], startMedian = best;
      if (!(start > 0)) { log(`Skip   | ${key} needs a positive value`); continue; }
      for (let round = 1; round <= opts.rounds && ratio.get(key) >= opts.minRatio && !spent(); round += 1) {
        const r = ratio.get(key), center = params[key], centerMedian = best;
        const at = (x) => round4(center * r ** x);
        const score = (v) => evaluator.score({ ...params, [key]: v });
        const pts = [{ x: 0, v: center, m: centerMedian }];
        for (const x of [1, -1]) {
          const v = at(x);
          if (v !== center && !spent()) pts.push({ x, v, m: await score(v) });
        }
        // Parabola through (−1, m−), (0, m0), (1, m+): peak at −b / 2a when a < 0.
        const up = pts.find((p) => p.x === 1), down = pts.find((p) => p.x === -1);
        if (up && down) {
          const a = (up.m + down.m) / 2 - centerMedian, b = (up.m - down.m) / 2;
          const xs = a < 0 ? -b / (2 * a) : NaN;
          if (Math.abs(xs) < 2 && pts.every((p) => Math.abs(p.x - xs) > 0.25)) {
            const v = at(xs);
            if (pts.every((p) => p.v !== v) && !spent()) pts.push({ x: xs, v, m: await score(v), newton: true });
          }
        }
        const top = pts.reduce((p, q) => (q.m > p.m ? q : p));
        const moved = top.v !== center;
        if (moved) {
          params[key] = top.v;
          best = top.m;
          history.push({ runs: evaluator.runs, params: { ...params }, median: best });
        } else if (!spent()) ratio.set(key, r ** opts.decay); // not when the budget cut the round short
        log(`${key.padEnd(13)} c${cycle} r${round} ×${r.toFixed(3)} | ` +
            pts.map((p) => `${p.v}${p.newton ? "*" : ""} ${p.m.toFixed(3)}`).join(", ") +
            ` | ${moved ? `→ ${top.v}` : `stay, step → ×${ratio.get(key).toFixed(3)}`}`);
      }
      changes.push({ cycle, key, from: start, to: params[key], gain: best - startMedian, step: ratio.get(key) });
    }
  }
  return { best, changes, history };
}

// Seeded uniform [0, 1) (mulberry32) and standard normals from it (Box–Muller).
function normals(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
}

// Eigen-decomposition of the symmetric n×n `a` (rows) by cyclic Jacobi
// rotations: { values, vectors } with vectors[i][k] the i-th entry of the k-th
// eigenvector. n here is the number of tuned keys, so O(n³) sweeps are free.
function eigSym(a) {
  const n = a.length;
  const m = a.map((row) => [...row]);
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 64; sweep += 1) {
    let off = 0;
    for (let p = 0; p < n; p += 1) for (let q = p + 1; q < n; q += 1) off += m[p][q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        if (Math.abs(m[p][q]) < 1e-300) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k += 1) {
          const mkp = m[k][p], mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k += 1) {
          const mpk = m[p][k], mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: m.map((row, i) => Math.max(row[i], 1e-20)), vectors: v };
}

// CMA-ES over `keys` of `params` jointly, in natural-log space around their
// start values, updating `params` in place to the search mean. Each
// generation samples opts.lambda points (default 4 + ⌊3 ln n⌋) from
// N(mean, σ² C), each a full batch scored by its median PSNR; the mean moves
// to the weighted best half, C learns the directions that paid (rank-one and
// rank-μ updates), and σ follows the evolution path's length (Hansen's
// standard settings). The mean itself is scored once per generation, counted
// in the budget, so progress is on a point the search would actually return.
// Stops after opts.budget batches, or when σ·√(largest eigenvalue of C) is
// below ln(opts.minRatio), the scale the coordinate search stops at.
// Returns { best (the mean's median), bestSample, changes, history }; history
// has the mean after every generation ({ runs, params, median, bestParams,
// bestMedian }, the last two the best sample so far).
export async function cmaes(evaluator, params, keys, opts, log = console.error) {
  const n = keys.length;
  const start = { ...params };
  for (const k of keys) if (!(start[k] > 0)) throw new Error(`cma: ${k} needs a positive value`);
  const toParams = (x) => ({ ...params, ...Object.fromEntries(keys.map((k, i) => [k, round4(start[k] * Math.exp(x[i]))])) });
  const randn = normals(opts.seed ?? 1);
  const lambda = opts.lambda > 0 ? opts.lambda : 4 + Math.floor(3 * Math.log(n));
  const mu = lambda >> 1;
  const w0 = Array.from({ length: mu }, (_, i) => Math.log((lambda + 1) / 2) - Math.log(i + 1));
  const wsum = w0.reduce((s, w) => s + w, 0);
  const w = w0.map((x) => x / wsum);
  const mueff = 1 / w.reduce((s, x) => s + x * x, 0);
  const cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
  const cs = (mueff + 2) / (n + mueff + 5);
  const c1 = 2 / ((n + 1.3) ** 2 + mueff);
  const cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff));
  const ds = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  const spent = () => opts.budget > 0 && evaluator.runs >= opts.budget;

  let mean = new Array(n).fill(0);
  let sigma = opts.sigma;
  let C = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  let pc = new Array(n).fill(0), ps = new Array(n).fill(0);
  let { values: D2, vectors: B } = eigSym(C);
  const score = async (x) => {
    const m = await evaluator.score(toParams(x));
    return Number.isFinite(m) ? m : -Infinity; // a diverged run ranks last
  };
  let best = await score(mean);
  let bestSample = { x: mean, m: best };
  log(`Start  | median ${best.toFixed(3)} dB; CMA-ES over ${keys.join(", ")}: λ ${lambda}, μ ${mu}, σ ${sigma}`);
  const history = [{ runs: evaluator.runs, params: toParams(mean), median: best, bestParams: toParams(mean), bestMedian: best }];
  for (let gen = 1; !spent(); gen += 1) {
    if (sigma * Math.sqrt(Math.max(...D2)) < Math.log(opts.minRatio)) {
      log(`cma    g${gen} | σ·√λmax ${(sigma * Math.sqrt(Math.max(...D2))).toFixed(4)} < ln ${opts.minRatio}: done`);
      break;
    }
    const D = D2.map(Math.sqrt);
    const pop = [];
    for (let k = 0; k < lambda && !spent(); k += 1) {
      const z = Array.from({ length: n }, randn);
      const y = Array.from({ length: n }, (_, i) => B[i].reduce((s, b, j) => s + b * D[j] * z[j], 0));
      const x = mean.map((m, i) => m + sigma * y[i]);
      pop.push({ y, x, m: await score(x) });
    }
    if (pop.length < lambda) break; // the budget ran out mid-generation: keep the last mean
    pop.sort((a, b) => b.m - a.m);
    if (pop[0].m > bestSample.m) bestSample = pop[0];
    const yw = new Array(n).fill(0);
    for (let i = 0; i < mu; i += 1) for (let j = 0; j < n; j += 1) yw[j] += w[i] * pop[i].y[j];
    mean = mean.map((m, j) => m + sigma * yw[j]);
    // C^(-1/2) yw = B D⁻¹ Bᵀ yw
    const bty = B[0].map((_, k) => B.reduce((s, row, i) => s + row[k] * yw[i], 0) / Math.sqrt(D2[k]));
    const cinvY = B.map((row) => row.reduce((s, b, k) => s + b * bty[k], 0));
    ps = ps.map((p, j) => (1 - cs) * p + Math.sqrt(cs * (2 - cs) * mueff) * cinvY[j]);
    const psNorm = Math.hypot(...ps);
    const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    pc = pc.map((p, j) => (1 - cc) * p + hsig * Math.sqrt(cc * (2 - cc) * mueff) * yw[j]);
    C = C.map((row, i) => row.map((c, j) => {
      let rankMu = 0;
      for (let k = 0; k < mu; k += 1) rankMu += w[k] * pop[k].y[i] * pop[k].y[j];
      return (1 - c1 - cmu) * c + c1 * (pc[i] * pc[j] + (1 - hsig) * cc * (2 - cc) * c) + cmu * rankMu;
    }));
    sigma *= Math.exp((cs / ds) * (psNorm / chiN - 1));
    ({ values: D2, vectors: B } = eigSym(C));
    if (spent()) break; // no batch left to score the new mean: report the last scored one
    best = await score(mean);
    history.push({ runs: evaluator.runs, params: toParams(mean), median: best,
                   bestParams: toParams(bestSample.x), bestMedian: bestSample.m });
    log(`cma    g${gen} ${String(evaluator.runs).padStart(4)} runs | pop ${pop[0].m.toFixed(3)}…${pop.at(-1).m.toFixed(3)} | ` +
        `mean ${best.toFixed(3)} | best ${bestSample.m.toFixed(3)} | σ ${sigma.toFixed(3)}, axes ×` +
        `${Math.exp(sigma * Math.sqrt(Math.min(...D2))).toFixed(3)}–×${Math.exp(sigma * Math.sqrt(Math.max(...D2))).toFixed(3)}`);
  }
  const last = history.at(-1);
  Object.assign(params, last.params);
  const changes = keys.map((key) => ({ key, from: start[key], to: params[key] }));
  return { best: last.median, bestSample: { params: last.bestParams, median: last.bestMedian }, changes, history };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS,
      keys: { type: "string" }, ratio: { type: "string", default: "2" }, decay: { type: "string", default: "0.5" },
      "min-ratio": { type: "string", default: "1.1" }, rounds: { type: "string", default: "8" },
      cycles: { type: "string", default: "2" }, method: { type: "string", default: "coord" },
      sigma: { type: "string", default: "0.25" }, lambda: { type: "string", default: "0" },
      seed: { type: "string", default: "1" }, budget: { type: "string", default: "0" }, holdout: { type: "string" }, out: { type: "string" },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (cli.help || (!cli.mnist && !cli.parquet && !cli.images)) {
    process.stdout.write(HELP);
    process.exit(cli.help ? 0 : 1);
  }
  const opts = {
    ratio: Number(cli.ratio), decay: Number(cli.decay), minRatio: Number(cli["min-ratio"]),
    rounds: Number(cli.rounds) | 0, cycles: Number(cli.cycles) | 0,
    sigma: Number(cli.sigma), lambda: Number(cli.lambda) | 0, seed: Number(cli.seed) | 0, budget: Number(cli.budget) | 0,
  };
  if (!["coord", "cma"].includes(cli.method)) throw new Error(`--method is coord or cma (got "${cli.method}")`);
  if (cli.method === "cma" && !(opts.sigma > 0)) throw new Error("need --sigma > 0");
  if (!(opts.ratio > 1) || !(opts.decay > 0 && opts.decay < 1) || !(opts.minRatio > 1)) {
    throw new Error("need --ratio > 1, 0 < --decay < 1, --min-ratio > 1");
  }
  const keys = cli.keys ? cli.keys.split(",") : LR_KEYS;
  const params = await loadParams(cli.preset, cli.set);
  const { args: args0, config } = await argsFor(params);
  for (const k of keys) params[k] ??= args0[k]; // tune from buildArgs' default when the preset doesn't set it
  const startParams = { ...params };
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));

  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const ctx = { device, shaders, trace: [] };
  console.error(`Device | ${label}`);
  console.error(`Tune   | ${keys.join(", ")} on ${images.length} images, ${args0.sites ?? args0.startSites} sites, ` +
                `${args0.iters} iterations; ` +
                (cli.method === "cma" ? `CMA-ES σ ${opts.sigma}, seed ${opts.seed}` : `step ×${opts.ratio}, decay ^${opts.decay}`) +
                `, until ×${opts.minRatio}` + (opts.budget ? ` or ${opts.budget} batches` : ""));
  const t0 = performance.now();
  const ev = await makeEvaluator(ctx, params, images);
  const { best, changes, history, bestSample } = await (cli.method === "cma" ? cmaes : tune)(ev, params, keys, opts);
  const seconds = (performance.now() - t0) / 1000;
  console.error(`Tuned  | ${ev.runs} batches in ${seconds.toFixed(0)}s, median ${ctx.trace[0].median.toFixed(3)} → ${best.toFixed(3)} dB`);
  for (const k of keys) console.error(`         ${k.padEnd(13)} ${startParams[k]} → ${params[k]}`);

  let holdout = null;
  const nHold = Number(cli.holdout ?? cli.count) | 0;
  if (nHold > 0) {
    const held = await loadImages({ ...cli, from: String((Number(cli.from) | 0) + images.length), count: String(nHold) },
                                  Number(config.MAX_DIM || 0));
    const hev = await makeEvaluator({ ...ctx, trace: [] }, params, held);
    holdout = { images: held.length, start: await hev.score(startParams), tuned: await hev.score(params) };
    if (bestSample) holdout.bestSample = await hev.score(bestSample.params);
    console.error(`Holdout| ${held.length} images: start ${holdout.start.toFixed(3)} → tuned ${holdout.tuned.toFixed(3)} dB ` +
                  `(${holdout.tuned >= holdout.start ? "+" : ""}${(holdout.tuned - holdout.start).toFixed(3)})` +
                  (bestSample ? `; best sample ${holdout.bestSample.toFixed(3)} dB` : ""));
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/tune_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({
    dataset: cli.mnist || cli.parquet || cli.images, from: Number(cli.from) | 0, count: images.length,
    method: cli.method, keys, opts, start: startParams, tuned: params, startMedian: ctx.trace[0].median, tunedMedian: best,
    holdout, changes, bestSample, history, seconds, trace: ctx.trace,
  }, null, 1) + "\n");
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
