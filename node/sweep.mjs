#!/usr/bin/env node
// Headless parameter sweeps for the WebGPU JS trainer (src/trainer.js) on Dawn.
//
//   node sweep.mjs --image kodim01.png --set sites=500,1000,2000 --set lrPosBase=0.025,0.05
//   node sweep.mjs --image kodim01.png --grid grid.json --jobs 2 --out results/sweep.jsonl
//
// Run `node sweep.mjs --help` for all options.

import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { dirname, join, resolve, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { Trainer, buildArgs } from "../trainer/trainer.js?v=rate-20261020";
import { parseSitesTxt, serializeSitesTxt, computePsnr, computeBpp, BITS_PER_SITE } from "../trainer/io.js?v=20261019";
import {
  createHeadlessDevice,
  loadShadersFromDisk,
  loadJson,
  loadPngToFloat32,
  loadPngMask,
  writeFloatPng,
} from "./headless.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_CONFIG = resolve(HERE, "../training_config.json");

const HELP = `Usage: node sweep.mjs --image <png> [options]

Grid (cartesian product of all --set / --grid axes):
  --set key=v1,v2,...   Sweep a trainer arg (camelCase, e.g. sites, iters,
                        lrPosBase, candFreq, colorSpace) or a
                        training_config.json key (UPPER_CASE, e.g.
                        INIT_LOG_TAU). Repeatable.
  --grid <json>         {"key": [values], ...} for a cartesian grid, or
                        [{"key": value, ...}, ...] for an explicit run list.
  --repeat <n>          Run each grid point n times (default 1).

Inputs:
  --image <png>         Target image (PNG only in Node). Required.
  --mask <png>          Optional mask (pixels with gray > 0 count).
  --init-sites <txt>    Start every run from this sites file.
  --config <json>       training_config.json (default: repo root).

Execution:
  --jobs <n>            Trainers running at once on the GPU (default 1).
  --log-freq <n>        PSNR checkpoint interval (default: trainer's logFreq).
  --dawn <flag>         Extra Dawn flag, e.g. backend=metal. Repeatable.
  --verbose             Print every PSNR checkpoint.

Output:
  --out <jsonl>         One JSON line per run (default results/sweep_<time>.jsonl).
  --save-dir <dir>      Also write <run>.png and <run>_sites.txt per run.
`;

function parseCli() {
  const { values } = parseArgs({
    options: {
      image: { type: "string" },
      mask: { type: "string" },
      "init-sites": { type: "string" },
      config: { type: "string", default: DEFAULT_CONFIG },
      set: { type: "string", multiple: true, default: [] },
      grid: { type: "string" },
      repeat: { type: "string", default: "1" },
      jobs: { type: "string", default: "1" },
      "log-freq": { type: "string" },
      dawn: { type: "string", multiple: true, default: [] },
      out: { type: "string" },
      "save-dir": { type: "string" },
      verbose: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (values.help) {
    process.stdout.write(HELP);
    process.exit(0);
  }
  if (!values.image) {
    process.stderr.write(HELP);
    throw new Error("--image is required");
  }
  return values;
}

function parseValue(raw) {
  const s = String(raw).trim();
  if (s === "true") return true;
  if (s === "false") return false;
  // Bare names are string args (e.g. colorSpace=logchroma).
  if (/^[A-Za-z][A-Za-z0-9_-]*$/.test(s)) return s;
  const n = Number(s);
  if (s === "" || Number.isNaN(n)) throw new Error(`Not a number: "${raw}"`);
  return n;
}

function cartesian(axes) {
  let points = [{}];
  for (const [key, vals] of axes) {
    const next = [];
    for (const p of points) for (const v of vals) next.push({ ...p, [key]: v });
    points = next;
  }
  return points;
}

async function buildRunList(cli) {
  const axes = new Map();
  let explicit = null;
  if (cli.grid) {
    const grid = await loadJson(cli.grid);
    if (Array.isArray(grid)) {
      explicit = grid;
    } else {
      for (const [k, v] of Object.entries(grid)) axes.set(k, Array.isArray(v) ? v : [v]);
    }
  }
  for (const spec of cli.set) {
    const eq = spec.indexOf("=");
    if (eq <= 0) throw new Error(`--set expects key=v1,v2,... (got "${spec}")`);
    axes.set(spec.slice(0, eq), spec.slice(eq + 1).split(",").map(parseValue));
  }
  const base = explicit || [{}];
  const points = [];
  for (const e of base) for (const p of cartesian([...axes])) points.push({ ...e, ...p });
  const repeat = Math.max(1, Number(cli.repeat) | 0);
  const runs = [];
  points.forEach((params, point) => {
    for (let r = 0; r < repeat; r += 1) runs.push({ index: runs.length, point, repeat: r, params });
  });
  return runs;
}

// UPPER_CASE keys patch training_config.json; camelCase keys are trainer args.
function splitParams(params, validArgs) {
  const configPatch = {};
  const overrides = {};
  for (const [k, v] of Object.entries(params)) {
    if (/^[A-Z0-9_]+$/.test(k)) {
      configPatch[k] = v;
    } else if (validArgs.has(k)) {
      overrides[k] = v;
    } else {
      throw new Error(`Unknown sweep key "${k}". Trainer args: ${[...validArgs].join(", ")}`);
    }
  }
  return { configPatch, overrides };
}

function runName(run) {
  const parts = Object.entries(run.params).map(([k, v]) => `${k}=${v}`);
  if (run.repeat > 0) parts.push(`r${run.repeat}`);
  return parts.length ? parts.join("_") : `run${run.index}`;
}

async function runOne(ctx, run) {
  const { device, shaders, config, target, mask, initSites, cli, validArgs } = ctx;
  const { configPatch, overrides } = splitParams(run.params, validArgs);
  if (cli["log-freq"]) overrides.logFreq = Number(cli["log-freq"]);
  const args = buildArgs({ ...config, ...configPatch }, overrides);
  // Previews only matter for the UI; keep readbacks to PSNR checkpoints.
  args.viewerFreq = args.iters + 1;

  const name = runName(run);
  const curve = [];
  const trainer = new Trainer(device, shaders, args, {
    onLog: (info) => {
      if (info.type === "iter") {
        curve.push({ it: info.it, psnr: info.psnr, bpp: computeBpp(info.activeSites, info.width, info.height),
          itsPerSec: info.itsPerSec, grads: info.gradStats?.groups });
        if (cli.verbose) {
          console.error(`  [${name}] iter ${info.it} | ${info.psnr.toFixed(2)} dB | ${info.itsPerSec.toFixed(1)} it/s`);
        }
      } else if (info.type === "msg" && cli.verbose) {
        console.error(`  [${name}] ${info.message}`);
      }
    },
  });

  const startedAt = new Date().toISOString();
  try {
    const initSiteCount = initSites ? initSites.count : args.startSites;
    await trainer.prepare({
      width: target.width,
      height: target.height,
      targetRgba32f: target.data,
      maskRgba32f: mask ? mask.rgba : null,
      maskSum: mask ? mask.maskSum : target.width * target.height,
      initSites: initSites ? initSites.data : null,
      initSiteCount,
    });
    const t0 = performance.now();
    await trainer.run();
    const trainSeconds = (performance.now() - t0) / 1000;
    const final = await trainer.finalize();
    const finalPsnr = computePsnr(final.renderRgba, target.data, target.width, target.height, mask ? mask.mask : null);

    if (cli["save-dir"]) {
      await writeFloatPng(join(cli["save-dir"], `${name}.png`), final.renderRgba, final.width, final.height);
      await writeFile(join(cli["save-dir"], `${name}_sites.txt`),
        serializeSitesTxt(final.sites, final.count, final.width, final.height, { globalSite: final.globalSite }));
    }
    return {
      name, ...run, startedAt, status: "ok",
      finalPsnr, bestPsnr: final.bestPsnr, sites: final.activeSites ?? final.count, growStop: final.growStop, growDoneIter: final.growDoneIter, growthGain: final.growthGain,
      bpp: computeBpp(final.activeSites ?? final.count, final.width, final.height), bitsPerSite: BITS_PER_SITE,
      trainSeconds, itPerSecond: args.iters / Math.max(trainSeconds, 1e-6),
      args, curve,
    };
  } catch (err) {
    return { name, ...run, startedAt, status: "error", error: err.stack || String(err), args, curve };
  } finally {
    trainer.destroy();
  }
}

async function main() {
  const cli = parseCli();
  const config = await loadJson(cli.config);
  const validArgs = new Set(Object.keys(buildArgs(config, {})));
  const runs = await buildRunList(cli);
  // Fail on bad keys before touching the GPU.
  for (const run of runs) splitParams(run.params, validArgs);

  const { device, label } = await createHeadlessDevice(cli.dawn);
  let lost = null;
  device.lost.then((info) => { lost = info; });
  const shaders = await loadShadersFromDisk(device);

  const maxDim = Number(config.MAX_DIM || 0);
  const target = await loadPngToFloat32(cli.image, maxDim);
  const mask = cli.mask ? await loadPngMask(cli.mask, target.width, target.height) : null;
  const initSites = cli["init-sites"] ? parseSitesTxt(await readFile(cli["init-sites"], "utf8")) : null;

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/sweep_${stamp}.jsonl`);
  await mkdir(dirname(outPath), { recursive: true });
  if (cli["save-dir"]) await mkdir(cli["save-dir"], { recursive: true });

  const jobs = Math.max(1, Number(cli.jobs) | 0);
  console.error(`Device | ${label}`);
  console.error(`Image  | ${basename(cli.image)} ${target.width}x${target.height}`);
  console.error(`Sweep  | ${runs.length} runs, ${jobs} at a time -> ${outPath}`);

  const ctx = { device, shaders, config, target, mask, initSites, cli, validArgs };
  const results = [];
  let next = 0;
  const sweepStart = performance.now();
  async function worker() {
    while (next < runs.length && !lost) {
      const run = runs[next++];
      const res = await runOne(ctx, run);
      results.push(res);
      const tag = `[${results.length}/${runs.length}] ${res.name}`;
      await appendFile(outPath, JSON.stringify(res) + "\n");
      if (res.status === "ok") {
        console.error(`${tag} | final ${res.finalPsnr.toFixed(2)} dB @ ${res.bpp.toFixed(3)} bpp | best ${res.bestPsnr.toFixed(2)} dB | ` +
          `${res.itPerSecond.toFixed(1)} it/s | ${res.trainSeconds.toFixed(1)}s` +
          (res.growStop ? ` | grew to ${res.sites} by iter ${res.growDoneIter} (${res.growStop})` : ""));
      } else {
        console.error(`${tag} | FAILED: ${res.error.split("\n")[0]}`);
      }
    }
  }
  await Promise.all(Array.from({ length: jobs }, worker));
  const wallSeconds = (performance.now() - sweepStart) / 1000;

  if (lost) console.error(`Device lost: ${lost.reason} ${lost.message}`);
  const ok = results.filter((r) => r.status === "ok").sort((a, b) => b.finalPsnr - a.finalPsnr);
  const totalIters = ok.reduce((s, r) => s + r.args.iters, 0);
  console.error(`\nDone | ${ok.length}/${runs.length} ok | ${wallSeconds.toFixed(1)}s wall | ` +
    `${(totalIters / Math.max(wallSeconds, 1e-6)).toFixed(1)} it/s aggregate`);
  for (const r of ok.slice(0, 10)) console.error(`  ${r.finalPsnr.toFixed(2)} dB  ${r.bpp.toFixed(3)} bpp  ${r.name}`);

  device.destroy();
  process.exit(ok.length === runs.length ? 0 : 1);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
