#!/usr/bin/env node
// The codec with a dataset-level shared model (codec.js buildSharedModel),
// fit on one set of images and measured on a disjoint held-out set:
//
//   node shared_codec.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz \
//     --parquet ../../comp/data/cifar10/test.parquet
//
// Per dataset, with presets/<dataset>.json: one batched run (probe.mjs, no
// E2/E3) on --fit images from 0 and one on --holdout images after them. The
// quantizer steps come from the fit run (codecSteps on its pooled
// sensitivity, as probe.mjs's audit), the shared models from its sites: one
// with YCoCg-R colors and one along the KLT (PCA) of the fit sites' colors.
// On the held-out sites, bits per site by stage (header, positions, each
// plane) for: the per-image codec (each image alone, its own header and
// models from uniform); with the shared model; and pooled (every held-out
// image's sites as one stream: no per-image header or model start-up). The
// shared model on its own fit images is printed too, as a check on
// overfitting. The histogram layer is left out throughout (as the audit).

import { mkdir, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { buildSharedModel } from "../trainer/codec.js?v=shared-20261020";
import { pcaBasis } from "../trainer/color_spaces.js?v=klt-20261020";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { loadParams, argsFor, loadMnist, loadParquet } from "./common.mjs";
import { probe, codecBits, poolImages } from "./probe.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// Trains `images`, returning { perImage, rows, steps, medianPsnr }.
async function sitesOf(ctx, args, images) {
  const keep = {};
  const rep = await probe(ctx, args, images, { e2: false, e3: false, audit: false, keep });
  return { ...keep, steps: rep.steps, medianPsnr: rep.medianPsnr };
}

export async function sharedAudit(fit, held) {
  const steps = fit.steps;
  const rgba = new Float32Array(fit.rows.length * 4);
  fit.rows.forEach((r, i) => rgba.set([r[2], r[3], r[4], 1], i * 4));
  const basis = pcaBasis(rgba, fit.rows.length);
  const t0 = performance.now();
  const shared = buildSharedModel(fit.perImage, steps);
  const sharedKlt = buildSharedModel(fit.perImage, steps, { colorBasis: basis });
  const fitSeconds = (performance.now() - t0) / 1000;
  const pooled = poolImages(held.perImage);
  const size = (S) => gzipSync(JSON.stringify(S)).length;
  return {
    steps, fitSeconds, sharedBytes: size(shared), sharedKltBytes: size(sharedKlt), strengths: shared.strengths,
    rows: {
      "per image": codecBits(held.perImage, steps),
      shared: codecBits(held.perImage, steps, null, shared),
      pooled: codecBits(pooled, steps),
      "per image, KLT": codecBits(held.perImage, steps, basis),
      "shared, KLT": codecBits(held.perImage, steps, null, sharedKlt),
      "pooled, KLT": codecBits(pooled, steps, basis),
      "shared, on fit images": codecBits(fit.perImage, steps, null, shared),
    },
  };
}

export function printShared(name, rep, log = console.error) {
  log(`${name}: steps ${Object.entries(rep.steps).map(([k, v]) => `${k} ${v.toPrecision(3)}`).join(", ")}; ` +
      `shared model ${rep.sharedBytes} bytes gzipped JSON (KLT ${rep.sharedKltBytes}), fit in ${rep.fitSeconds.toFixed(1)}s; ` +
      `prior strengths ${Object.entries(rep.strengths).map(([k, v]) => `${k} ${v}`).join(" ")}`);
  const cols = ["header", "position", "log_aniso", "θ", "log_tau", "radius", "c0", "c1", "c2", "color", "total"];
  log(`  bits/site               ${cols.map((c) => c.padStart(9)).join("")}   (c0–c2: Y Co Cg, or K0 K1 K2)`);
  for (const [label, b] of Object.entries(rep.rows)) {
    const v = Object.values(b.perSite); // header, position, 7 planes, total, color
    const cells = [...v.slice(0, 9), b.perSite.color, b.perSite.total];
    log(`  ${label.padEnd(22)}  ${cells.map((x) => x.toFixed(2).padStart(9)).join("")}   ${b.sites} sites`);
  }
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      mnist: { type: "string" }, parquet: { type: "string" },
      fit: { type: "string", default: "1024" }, holdout: { type: "string", default: "2048" },
      set: { type: "string", multiple: true, default: [] }, out: { type: "string" },
    },
  });
  const nFit = Number(cli.fit) | 0, nHold = Number(cli.holdout) | 0;
  const datasets = [];
  if (cli.mnist) datasets.push({ name: "mnist", load: (from, n) => loadMnist(cli.mnist, from, n) });
  if (cli.parquet) datasets.push({ name: "cifar10", load: (from, n) => loadParquet(cli.parquet, from, n) });
  if (!datasets.length) {
    process.stdout.write("Usage: node shared_codec.mjs [--mnist <idx3.gz>] [--parquet <file>] [--fit n] [--holdout n] [--set key=value …] [--out json]\n");
    process.exit(1);
  }
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label}`);
  const out = {};
  for (const ds of datasets) {
    const { args } = await argsFor(await loadParams(resolve(HERE, `../presets/${ds.name}.json`), cli.set));
    const fit = await sitesOf({ device, shaders }, args, await ds.load(0, nFit));
    const held = await sitesOf({ device, shaders }, args, await ds.load(nFit, nHold));
    console.error(`${ds.name}: fit on images 0–${nFit - 1} (median ${fit.medianPsnr.toFixed(2)} dB), ` +
                  `held out ${nFit}–${nFit + nHold - 1} (median ${held.medianPsnr.toFixed(2)} dB)`);
    const rep = await sharedAudit(fit, held);
    printShared(ds.name, rep);
    out[ds.name] = { fit: [0, nFit], holdout: [nFit, nFit + nHold], ...rep };
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/shared_codec_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(out, null, 1) + "\n");
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
