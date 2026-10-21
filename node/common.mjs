// Shared by the batch-mode Node tools (batch.mjs, tune.mjs, probe.mjs): CLI
// value parsing, trainer args from a preset plus --set overrides, dataset
// loading (MNIST IDX, Hugging Face parquet such as CIFAR-10, a folder of
// PNGs), and one batched run over a list of images.

import { readFile, readdir } from "node:fs/promises";
import { dirname, join, resolve, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { PNG } from "pngjs";

import { buildArgs } from "../trainer/trainer.js?v=cmoment-20261020";
import { BatchTrainer, planBatches } from "../trainer/batch.js?v=klt-20261020";
import { parquetMetadata, parquetReadObjects } from "../web/vendor/hyparquet/src/index.js";
import { loadJson, loadPngToFloat32 } from "./headless.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export function parseValue(raw) {
  const s = String(raw).trim();
  if (s === "true") return true;
  if (s === "false") return false;
  if (/^[A-Za-z][A-Za-z0-9_-]*$/.test(s)) return s;
  const n = Number(s);
  if (s === "" || Number.isNaN(n)) throw new Error(`Not a number: "${raw}"`);
  return n;
}

// Preset args ({ args } JSON) with `key=value` overrides, split into trainer
// args (camelCase) and training_config.json keys (UPPER_CASE).
export async function loadParams(presetPath, sets = []) {
  const params = presetPath ? { ...(await loadJson(presetPath)).args } : {};
  for (const spec of sets) {
    const eq = spec.indexOf("=");
    if (eq <= 0) throw new Error(`--set expects key=value (got "${spec}")`);
    params[spec.slice(0, eq)] = parseValue(spec.slice(eq + 1));
  }
  return params;
}

// Trainer args for `params` (as loadParams returns), previews off.
export async function argsFor(params) {
  const configPatch = {}, overrides = {};
  for (const [k, v] of Object.entries(params)) (/^[A-Z0-9_]+$/.test(k) ? configPatch : overrides)[k] = v;
  const config = { ...(await loadJson(resolve(HERE, "../training_config.json"))), ...configPatch };
  const args = buildArgs(config, overrides);
  args.viewerFreq = args.iters + 1;
  return { args, config };
}

// Test-split MNIST digits [from, from + count) as rgba32f images.
export async function loadMnist(path, from, count) {
  const bytes = gunzipSync(await readFile(path));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const total = view.getUint32(4), side = view.getUint32(8);
  const n = side * side;
  const images = [];
  for (let i = from; i < Math.min(total, from + count); i += 1) {
    const data = new Float32Array(n * 4);
    for (let p = 0; p < n; p += 1) {
      const v = bytes[16 + i * n + p] / 255;
      data.set([v, v, v, 1], p * 4);
    }
    images.push({ name: `mnist${String(i).padStart(4, "0")}`, data, width: side, height: side });
  }
  return images;
}

export async function loadPngDir(dir, from, count, maxDim) {
  const names = (await readdir(dir)).filter((f) => f.toLowerCase().endsWith(".png")).sort().slice(from, from + count);
  return Promise.all(names.map(async (f) => ({ name: basename(f, ".png"), ...(await loadPngToFloat32(join(dir, f), maxDim)) })));
}

// Rows [from, from + count) of a Hugging Face parquet file whose image column
// holds PNG bytes (CIFAR-10's "img", MNIST's "image"), as rgba32f images.
export async function loadParquet(path, from, count, column = null) {
  const b = await readFile(path);
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const metadata = parquetMetadata(buf);
  const names = metadata.schema.map((s) => s.name);
  const col = column ?? ["img", "image"].find((c) => names.includes(c));
  if (!col) throw new Error(`${path}: no img/image column (pass --column)`);
  const rows = await parquetReadObjects({ file: buf, metadata, columns: [col], utf8: false, rowStart: from, rowEnd: from + count });
  const prefix = basename(dirname(resolve(path)));
  return rows.map((r, k) => {
    const png = PNG.sync.read(Buffer.from(r[col].bytes));
    const data = new Float32Array(png.width * png.height * 4);
    for (let i = 0; i < data.length; i += 1) data[i] = png.data[i] / 255;
    return { name: `${prefix}_${String(from + k).padStart(4, "0")}`, data, width: png.width, height: png.height };
  });
}

// --mnist / --parquet / --images, as the tools' CLIs take them.
export function loadImages(cli, maxDim = 0) {
  const from = Number(cli.from) | 0, count = Number(cli.count) | 0;
  if (cli.mnist) return loadMnist(cli.mnist, from, count);
  if (cli.parquet) return loadParquet(cli.parquet, from, count, cli.column ?? null);
  return loadPngDir(cli.images, from, count, maxDim);
}

export const DATASET_OPTIONS = {
  mnist: { type: "string" }, parquet: { type: "string" }, column: { type: "string" }, images: { type: "string" },
  from: { type: "string", default: "0" }, count: { type: "string", default: "64" },
  preset: { type: "string" }, set: { type: "string", multiple: true, default: [] },
};

// Trains `images` in atlases (hooks.batches, default planBatches) with
// ctx.args, logging each atlas unless ctx.quiet. hooks.onLog gets the
// BatchTrainer's events with the batch's `part` (image indices); hooks.after(bt,
// part) runs on each trained batch before it is finalized (e.g. sensitivity
// readback). Returns { results: per image { name, psnr, sites, … }, seconds }.
export async function trainBatched(ctx, images, hooks = {}) {
  const batches = hooks.batches ?? planBatches(images, ctx.args.candDownscale);
  const results = new Array(images.length);
  const t0 = performance.now();
  for (const idx of batches) {
    const part = idx.map((i) => images[i]);
    const bt = new BatchTrainer(ctx.device, ctx.shaders, ctx.args, {
      onLog: (info) => {
        if (info.type === "msg" && ctx.verbose) console.error(`  ${info.message}`);
        hooks.onLog?.(info, idx, bt);
      },
    });
    try {
      await bt.prepare(part);
      if (!ctx.quiet) {
        const l = bt.layout;
        console.error(`Batch  | ${part.length} image(s) from ${part[0].name}: atlas ${l.width}×${l.height}, ` +
                      `${l.tilesX}×${l.tilesY} tiles of ${l.tileW}×${l.tileH}, ${l.sitesPerTile} sites each`);
      }
      await bt.run();
      if (hooks.after) await hooks.after(bt, idx);
      const final = await bt.finalize();
      final.forEach((r, i) => { results[idx[i]] = { name: part[i].name, ...r }; });
    } finally {
      bt.destroy();
    }
  }
  return { results, seconds: (performance.now() - t0) / 1000 };
}

export const mean = (xs) => xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length);

export function median(xs) {
  const s = Float64Array.from(xs).sort();
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
