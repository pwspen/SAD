#!/usr/bin/env node
// SAD side of the SAD vs CNN-autoencoder figure (../experiments/ae_compare):
// trains one image at each --sites count, codes the sites with the real codec
// at --db's steps (post-hoc, as qat.mjs's arm a), and writes per count the
// decoded render, the cell map ("hashed" view) and the decoded sites, plus
// the measured bits in summary.json.
//
//   node ae_compare_sad.mjs --images ../../comp/data/kodak --from 22 --count 1 \
//     --preset ../presets/kodak.json --sites 250,1000,4000 --out ../experiments/ae_compare/sad

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { SITE_FLOATS } from "../trainer/params.js?v=w2b-20261021";
import { codecSteps, encodeSites } from "../trainer/codec.js?v=rui-20261021";
import { encodeSiteColors } from "../trainer/color_spaces.js?v=w2b-20261021";
import { tilePsnrs, tileOrigin } from "../trainer/batch.js?v=rui-20261021";
import { createHeadlessDevice, loadShadersFromDisk, writeFloatPng } from "./headless.mjs";
import { DATASET_OPTIONS, loadParams, argsFor, loadImages, trainBatched } from "./common.mjs";

// The batch's one tile, cropped out of the atlas render.
function crop(rgba, layout, w, h) {
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y += 1) out.set(rgba.subarray(y * layout.width * 4, (y * layout.width + w) * 4), y * w * 4);
  return out;
}

async function evaluate(bt, db, dir) {
  const t = bt.trainer, n = t.siteCount, layout = bt.layout, im = bt.images[0];
  const sites = await t._readSitesRgb(n);
  const { F } = await t.siteSensitivity();
  const render = async (mode) => {
    t._rebuildCandidates(n);
    return t.renderView(mode);
  };
  const clean = await render("render");
  let sse = 0;
  for (let i = 0; i < t.width * t.height; i += 1) {
    if (t.maskCpu && !(t.maskCpu[i] > 0)) continue;
    for (let c = 0; c < 3; c += 1) sse += (clean[i * 4 + c] - t.target[i * 4 + c]) ** 2;
  }
  const cleanPsnr = tilePsnrs(clean, bt.images, layout)[0];
  const coded = Float32Array.from(sites);
  for (let i = 0; i < n; i += 1) if (t._uncoded(i)) coded[i * SITE_FLOATS] = -1;
  const steps = codecSteps(coded, n, F, { lossDb: db, sse0: sse });
  const enc = encodeSites(coded, n, t.width, t.height, steps, { colorBasis: t.rateBasis });
  const dec = Float32Array.from(sites);
  enc.slots.forEach((slot, k) => {
    const b = slot * SITE_FLOATS;
    dec.set(enc.decoded.subarray(k * SITE_FLOATS, (k + 1) * SITE_FLOATS), b);
    const [ox, oy] = tileOrigin(layout, 0);
    dec[b] = Math.min(ox + layout.tileW - 1, Math.max(ox, dec[b]));
    dec[b + 1] = Math.min(oy + layout.tileH - 1, Math.max(oy, dec[b + 1]));
  });
  t.device.queue.writeBuffer(t.sitesBuf, 0, encodeSiteColors(dec, n, t.colorSpace));
  const decoded = await render("render");
  const psnr = tilePsnrs(decoded, bt.images, layout)[0];
  await writeFloatPng(join(dir, "recon.png"), crop(decoded, layout, im.width, im.height), im.width, im.height);
  await writeFloatPng(join(dir, "cells.png"), crop(await render("hashed"), layout, im.width, im.height), im.width, im.height);
  const kept = enc.slots.map((slot) => Array.from(dec.subarray(slot * SITE_FLOATS, (slot + 1) * SITE_FLOATS)));
  await writeFile(join(dir, "sites.json"), JSON.stringify({ layout: "x y log_tau radius r g b dir_x dir_y log_aniso", sites: kept }));
  return { cleanPsnr, psnr, bits: enc.bits, bytes: Math.ceil(enc.bits / 8), coded: enc.n, bpp: enc.bits / (im.width * im.height) };
}

async function main() {
  const { values: cli } = parseArgs({
    options: { ...DATASET_OPTIONS, sites: { type: "string", default: "1000" }, db: { type: "string", default: "0.5" }, out: { type: "string" } },
  });
  const base = await loadParams(cli.preset, cli.set);
  const { config } = await argsFor(base);
  const images = (await loadImages(cli, Number(config.MAX_DIM || 0))).slice(0, 1);
  const { device, label } = await createHeadlessDevice();
  const shaders = await loadShadersFromDisk(device);
  console.error(`Device | ${label} | ${images[0].name} ${images[0].width}×${images[0].height}`);
  const db = Number(cli.db);
  const summary = { image: images[0].name, width: images[0].width, height: images[0].height, db, runs: [] };
  for (const s of cli.sites.split(",").map(Number)) {
    const dir = join(cli.out, `sites_${s}`);
    await mkdir(dir, { recursive: true });
    // Batched training needs a fixed count.
    const { args } = await argsFor({ ...base, startSites: s, endSites: s });
    args.logFreq = args.iters;
    let ev;
    const { results } = await trainBatched({ device, shaders, args, quiet: true }, images, {
      after: async (bt) => { ev = await evaluate(bt, db, dir); },
    });
    const r = { sites: s, trainPsnr: results[0].psnr, ...ev };
    summary.runs.push(r);
    console.error(`${String(s).padStart(5)} sites | clean ${ev.cleanPsnr.toFixed(2)} dB | decoded ${ev.psnr.toFixed(2)} dB ` +
                  `| ${ev.bits} bits (${ev.bytes} B, ${ev.bpp.toFixed(4)} bpp, ${ev.coded} coded sites)`);
  }
  await writeFile(join(cli.out, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  device.destroy?.();
}

main().catch((e) => { console.error(e); process.exit(1); });
