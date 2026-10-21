#!/usr/bin/env node
// Classification without training on labels: encode a dataset into sites
// (batch mode, a preset), cluster the converged sites into k groups, then
// look at ONE random label per cluster, call every member that label, and
// score it against the true labels.
//
//   node cluster.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 2000 \
//     --preset ../presets/mnist.json
//
// Clusterings:
//   pixels     k-means on the raw pixels (a reference, not the sites).
//   site map   k-means on a coarse map of each image's sites: every site adds
//              a Gaussian at its position (σ = --map-sigma px, default 2) weighted by its
//              color's first KLT coordinate (luma-like, about the mean), on a
//              --map × --map grid (default 14). The histogram layer is left out.
//   codelength The codec as the classifier: each cluster gets a shared coding
//              model (codec.js buildSharedModel) fit on its members, and each
//              image goes to the cluster whose model codes its sites in the
//              fewest bits; repeated (--rounds) like k-means, from the site-map
//              clusters. Images are the sites, the distance is bits.
// Scoring, per clustering: purity (every cluster named by its majority label,
// the best a one-label-per-cluster rule can do), and the peek protocol over
// --draws random draws (one random member's label names each cluster; the
// peeked images themselves are not scored): mean ± sd accuracy, and how many
// classes the peeks miss on average (two clusters can draw the same label).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=w2b-20261021";
import { encodeSites, buildSharedModel } from "../trainer/codec.js?v=rui-20261021";
import { kMeans } from "../web/analysis.js";
import { createHeadlessDevice, loadShadersFromDisk } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched, mean } from "./common.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Trains `images` and returns per image { sites (layer slot inactive), n,
// width, height, label }, plus the codec steps and KLT basis the rate model
// fit to the final sites (rate.js; the same steps for every image).
async function encode(ctx, args, images) {
  let steps = null, basis = null;
  const after = async (bt) => {
    await bt.trainer._rateRefit(bt.trainer.siteCount);
    steps ??= bt.trainer.rateLast.steps;
    basis ??= bt.trainer.rateBasis;
  };
  const { results, seconds } = await trainBatched({ ...ctx, args, quiet: true }, images, { after });
  const items = results.map((r, i) => {
    const sites = Float32Array.from(r.sites);
    if (r.layer !== null && r.layer !== undefined) sites[r.layer * SITE_FLOATS] = -1;
    return { sites, n: r.count, width: r.width, height: r.height, label: images[i].label, psnr: r.psnr };
  });
  return { items, steps, basis, seconds };
}

function siteMap(items, basis, grid, sigma) {
  const d = grid * grid;
  const X = new Float64Array(items.length * d);
  const a = basis.axes[0], m = basis.mean;
  items.forEach((im, k) => {
    const cell = im.width / grid;
    for (let i = 0; i < im.n; i += 1) {
      const b = i * SITE_FLOATS;
      if (!(im.sites[b] >= 0)) continue;
      const w = a[0] * (im.sites[b + 4] - m[0]) + a[1] * (im.sites[b + 5] - m[1]) + a[2] * (im.sites[b + 6] - m[2]);
      for (let gy = 0; gy < grid; gy += 1) {
        for (let gx = 0; gx < grid; gx += 1) {
          const dx = (gx + 0.5) * cell - im.sites[b], dy = (gy + 0.5) * cell - im.sites[b + 1];
          X[k * d + gy * grid + gx] += w * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
      }
    }
  });
  return { X, d };
}

function pixelFeatures(images) {
  const d = images[0].width * images[0].height;
  const X = new Float64Array(images.length * d);
  images.forEach((im, k) => {
    for (let p = 0; p < d; p += 1) X[k * d + p] = 0.299 * im.data[p * 4] + 0.587 * im.data[p * 4 + 1] + 0.114 * im.data[p * 4 + 2];
  });
  return { X, d };
}

// Codelength clustering from `init` labels: refit a shared model per cluster,
// reassign by fewest bits, until fewer than 0.2% move or `rounds`. An empty
// cluster is reseeded with the image coded worst under its own cluster.
function codelength(items, steps, basis, k, init, rounds, log) {
  const labels = Int32Array.from(init);
  const bits = new Float64Array(items.length * k);
  let models = null;
  for (let round = 1; round <= rounds; round += 1) {
    const t0 = performance.now();
    models = Array.from({ length: k }, (_, c) => {
      const members = items.filter((_, i) => labels[i] === c);
      return members.length ? buildSharedModel(members, steps, { colorBasis: basis }) : null;
    });
    items.forEach((im, i) => {
      for (let c = 0; c < k; c += 1) {
        bits[i * k + c] = models[c] ? encodeSites(im.sites, im.n, im.width, im.height, steps, { shared: models[c] }).bits : Infinity;
      }
    });
    let moved = 0, total = 0;
    items.forEach((_, i) => {
      let best = 0;
      for (let c = 1; c < k; c += 1) if (bits[i * k + c] < bits[i * k + best]) best = c;
      if (best !== labels[i]) { labels[i] = best; moved += 1; }
      total += bits[i * k + best];
    });
    for (let c = 0; c < k; c += 1) {
      if (labels.includes(c)) continue;
      let worst = 0;
      items.forEach((_, i) => { if (bits[i * k + labels[i]] > bits[worst * k + labels[worst]]) worst = i; });
      labels[worst] = c;
    }
    log(`         round ${round}: ${moved} moved, ${(total / items.length).toFixed(1)} bits/image, ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    if (moved < 0.002 * items.length) break;
  }
  return labels;
}

// Purity, and the peek protocol over `draws` draws.
function score(labels, truth, k, classes, draws, rand) {
  const n = labels.length;
  const members = Array.from({ length: k }, () => []);
  labels.forEach((c, i) => members[c].push(i));
  let pure = 0;
  for (const m of members) {
    const counts = new Map();
    for (const i of m) counts.set(truth[i], (counts.get(truth[i]) ?? 0) + 1);
    pure += Math.max(0, ...counts.values());
  }
  const accs = [], missing = [];
  for (let d = 0; d < draws; d += 1) {
    const name = new Int32Array(k).fill(-1), peeked = new Set();
    members.forEach((m, c) => {
      if (!m.length) return;
      const i = m[Math.floor(rand() * m.length)];
      name[c] = truth[i];
      peeked.add(i);
    });
    let right = 0, scored = 0;
    for (let i = 0; i < n; i += 1) {
      if (peeked.has(i)) continue;
      scored += 1;
      if (name[labels[i]] === truth[i]) right += 1;
    }
    accs.push(right / scored);
    missing.push(classes - new Set([...name].filter((x) => x >= 0)).size);
  }
  const m = mean(accs);
  return { purity: pure / n, peekMean: m, peekSd: Math.sqrt(mean(accs.map((a) => (a - m) ** 2))), missing: mean(missing),
           sizes: members.map((x) => x.length) };
}

async function main() {
  const { values: cli } = parseArgs({
    options: {
      ...DATASET_OPTIONS, k: { type: "string", default: "10" }, draws: { type: "string", default: "1000" },
      rounds: { type: "string", default: "8" }, map: { type: "string", default: "14" }, "map-sigma": { type: "string", default: "2" },
      seed: { type: "string", default: "1" }, out: { type: "string" },
    },
  });
  const { args, config } = await argsFor(await loadParams(cli.preset, cli.set));
  args.logFreq = args.iters;
  const images = await loadImages(cli, Number(config.MAX_DIM || 0));
  if (images.some((im) => im.label === null || im.label === undefined)) throw new Error("these images have no labels");
  const truth = images.map((im) => im.label);
  const classes = new Set(truth).size;
  const k = Number(cli.k) | 0, draws = Number(cli.draws) | 0;
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  const log = (s) => console.error(s);
  log(`Device | ${label}`);
  const enc = await encode({ device, shaders }, args, images);
  log(`Encode | ${images.length} images, ${classes} classes, ${args.sites} sites, ${enc.seconds.toFixed(1)}s, ` +
      `median PSNR ${[...enc.items.map((x) => x.psnr)].sort((a, b) => a - b)[enc.items.length >> 1].toFixed(2)} dB`);

  const out = { dataset: cli.mnist || cli.parquet || cli.images, count: images.length, k, draws, args, results: {} };
  const report = (name, labels) => {
    const r = score(labels, truth, k, classes, draws, rng(Number(cli.seed) | 0));
    out.results[name] = r;
    log(`${name.padEnd(11)}| purity ${(100 * r.purity).toFixed(1)}% · one peek per cluster ${(100 * r.peekMean).toFixed(1)} ` +
        `± ${(100 * r.peekSd).toFixed(1)}% · classes missed ${r.missing.toFixed(2)} · sizes ${r.sizes.join(" ")}`);
    return r;
  };

  const px = pixelFeatures(images);
  report("pixels", kMeans(px.X, images.length, px.d, k, { maxIters: 100 }).labels);
  const grid = Number(cli.map) | 0, sigma = Number(cli["map-sigma"]);
  const sm = siteMap(enc.items, enc.basis, grid, sigma);
  const smLabels = kMeans(sm.X, images.length, sm.d, k, { maxIters: 100 }).labels;
  report("site map", smLabels);
  log(`Codec  | codelength clustering from the site-map clusters`);
  report("codelength", codelength(enc.items, enc.steps, enc.basis, k, smLabels, Number(cli.rounds) | 0, log));

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = cli.out || resolve(HERE, `../results/cluster_${stamp}.json`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(out, null, 1) + "\n");
  log(`Wrote  | ${outPath}`);
  device.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
