// Batched training: many images with the same fixed site count as one run.
// The images are packed into an atlas of equal tiles (one image each, top-left
// aligned, the rest of the tile masked out) and trained by a single Trainer
// with a Tiling (sad_shared.wgsl): tile b owns sites [b·n, (b + 1)·n), and
// candidate lists, rendering, gradients and Adam keep every site inside its
// tile. So one dispatch covers every image, and the GPU sees one large image
// instead of many tiny ones. Results come back per image in its own pixel
// coordinates, as a single-image run would give them.
//
// The logit's scale (1 / long side) and Adam's gradient normalization are the
// tile's, so images should share their long side (MNIST, Kodak's 768 × 512
// and 512 × 768) to train as they would alone.

import { Trainer } from "./trainer.js?v=m3-20261020";
import { SITE_FLOATS } from "./params.js?v=m3-20261020";

// Default atlas limits: the WebGPU default texture side, and a pixel budget
// that keeps the per-pixel buffers (≈132 B/px) near half a gigabyte.
export const ATLAS_MAX_SIDE = 8192;
export const ATLAS_MAX_PIXELS = 1 << 22;

// Tile size for images of these sizes: the largest of each side, rounded up to
// whole candidate cells so no cell straddles two tiles.
export function tileSize(sizes, candDownscale = 1) {
  const ds = Math.max(1, candDownscale | 0);
  const w = Math.max(...sizes.map((s) => s.width));
  const h = Math.max(...sizes.map((s) => s.height));
  return { tileW: Math.ceil(w / ds) * ds, tileH: Math.ceil(h / ds) * ds };
}

// Atlases for images of these sizes: images of one size share an atlas (a
// tile sized for the largest wastes the difference on every smaller image),
// each atlas holding at most atlasCapacity. Returns [[image index, …], …].
export function planBatches(sizes, candDownscale = 1, limits = {}) {
  const groups = new Map();
  sizes.forEach((s, i) => {
    const { tileW, tileH } = tileSize([s], candDownscale);
    const key = `${tileW}x${tileH}`;
    if (!groups.has(key)) groups.set(key, { tileW, tileH, items: [] });
    groups.get(key).items.push(i);
  });
  const batches = [];
  for (const { tileW, tileH, items } of groups.values()) {
    const cap = atlasCapacity(tileW, tileH, limits);
    for (let s = 0; s < items.length; s += cap) batches.push(items.slice(s, s + cap));
  }
  return batches;
}

// How many tiles of tileW × tileH fit one atlas.
export function atlasCapacity(tileW, tileH, { maxSide = ATLAS_MAX_SIDE, maxPixels = ATLAS_MAX_PIXELS } = {}) {
  const bySide = Math.floor(maxSide / tileW) * Math.floor(maxSide / tileH);
  return Math.max(1, Math.min(bySide, Math.floor(maxPixels / (tileW * tileH))));
}

// The Tiling for `count` tiles of tileW × tileH, near square, plus the atlas
// size: { tileW, tileH, tilesX, tilesY, tiles, count, sitesPerTile, width, height }.
export function atlasLayout(count, tileW, tileH, sitesPerTile, { maxSide = ATLAS_MAX_SIDE } = {}) {
  let tilesX = Math.max(1, Math.ceil(Math.sqrt((count * tileH) / tileW)));
  tilesX = Math.min(tilesX, Math.max(1, Math.floor(maxSide / tileW)), count);
  const tilesY = Math.ceil(count / tilesX);
  if (tilesY * tileH > maxSide) throw new Error(`${count} tiles of ${tileW}×${tileH} exceed a ${maxSide}px atlas`);
  return {
    tileW, tileH, tilesX, tilesY, tiles: tilesX * tilesY, count, sitesPerTile,
    width: tilesX * tileW, height: tilesY * tileH,
  };
}

// Top-left pixel of tile b.
export function tileOrigin(layout, b) {
  return [(b % layout.tilesX) * layout.tileW, Math.floor(b / layout.tilesX) * layout.tileH];
}

// The atlas target (rgba32f) and mask (rgba32f, r = 1 on image pixels) for
// `images` ({ data: rgba32f, width, height }, optional mask: f32 per pixel).
export function packAtlas(images, layout) {
  const { width } = layout;
  const target = new Float32Array(layout.width * layout.height * 4);
  const mask = new Float32Array(layout.width * layout.height * 4);
  for (let i = 3; i < mask.length; i += 4) mask[i] = 1;
  let maskSum = 0;
  images.forEach((img, b) => {
    const [ox, oy] = tileOrigin(layout, b);
    for (let y = 0; y < img.height; y += 1) {
      const row = (oy + y) * width + ox;
      target.set(img.data.subarray(y * img.width * 4, (y + 1) * img.width * 4), row * 4);
      for (let x = 0; x < img.width; x += 1) {
        const m = img.mask ? (img.mask[y * img.width + x] > 0 ? 1 : 0) : 1;
        mask[(row + x) * 4] = m;
        maskSum += m;
      }
    }
  });
  return { target, mask, maskSum };
}

// Masked PSNR of each image in an atlas render (rgba32f), as computePsnr gives
// it for that image alone.
export function tilePsnrs(render, images, layout) {
  const out = new Float64Array(images.length);
  images.forEach((img, b) => {
    const [ox, oy] = tileOrigin(layout, b);
    let acc = 0, count = 0;
    for (let y = 0; y < img.height; y += 1) {
      const row = (oy + y) * layout.width + ox;
      for (let x = 0; x < img.width; x += 1) {
        const i = y * img.width + x;
        if (img.mask && !(img.mask[i] > 0)) continue;
        const r = (row + x) * 4;
        for (let c = 0; c < 3; c += 1) {
          const d = render[r + c] - img.data[i * 4 + c];
          acc += d * d;
        }
        count += 3;
      }
    }
    const mse = acc / Math.max(1, count);
    out[b] = mse > 0 ? 10 * Math.log10(1 / mse) : 99;
  });
  return out;
}

// Tile b's sites (RGB colors) in its image's pixel coordinates: n · SITE_FLOATS.
export function tileSites(sites, layout, b) {
  const n = layout.sitesPerTile;
  const out = Float32Array.from(sites.subarray(b * n * SITE_FLOATS, (b + 1) * n * SITE_FLOATS));
  const [ox, oy] = tileOrigin(layout, b);
  for (let i = 0; i < n; i += 1) {
    if (!(out[i * SITE_FLOATS] >= 0)) continue;
    out[i * SITE_FLOATS] -= ox;
    out[i * SITE_FLOATS + 1] -= oy;
  }
  return out;
}

// Active sites of tile b.
export function tileActive(sites, layout, b) {
  const n = layout.sitesPerTile;
  let active = 0;
  for (let i = b * n; i < (b + 1) * n; i += 1) if (sites[i * SITE_FLOATS] >= 0) active += 1;
  return active;
}

export class BatchTrainer {
  // args as for Trainer, with a fixed site count: args.startSites per image
  // (the histogram layer included). onLog gets the Trainer's "msg" events and,
  // at each log tick, { type: "batchIter", it, iters, psnrs, sitesData, grads,
  // layer, lrScale, itsPerSec, elapsedSec } (grads: the atlas sites' raw
  // gradients, 10 per site as Trainer's "iter" event); tileSites(sitesData, layout, b) gives
  // image b's sites, tileActive(sitesData, layout, b) its active count.
  constructor(device, shaders, args, { onLog = null, signal = null } = {}) {
    this.device = device;
    this.args = args;
    this.onLog = onLog;
    this.trainer = new Trainer(device, shaders, args, { signal, onLog: (info) => this._onLog(info) });
  }

  // images: [{ data: rgba32f, width, height, mask? }], at most atlasCapacity.
  async prepare(images) {
    const args = this.args;
    const { tileW, tileH } = tileSize(images, args.candDownscale);
    this.images = images;
    this.layout = atlasLayout(images.length, tileW, tileH, args.startSites);
    const { target, mask, maskSum } = packAtlas(images, this.layout);
    await this.trainer.prepare({
      width: this.layout.width, height: this.layout.height,
      targetRgba32f: target, maskRgba32f: mask, maskSum,
      initSites: null, initSiteCount: 0, tiling: this.layout,
    });
  }

  // The histogram layer's slot in each image's sites, or null.
  get layerSlot() { return this.trainer.layerIndex; }

  _onLog(info) {
    if (!this.onLog) return;
    if (info.type !== "iter") {
      this.onLog(info);
      return;
    }
    this.onLog({
      type: "batchIter", it: info.it, iters: info.iters,
      psnrs: tilePsnrs(info.renderRgba, this.images, this.layout),
      sitesData: info.sitesData, grads: info.grads, layer: this.layerSlot, lrScale: info.lrScale,
      itsPerSec: info.itsPerSec, elapsedSec: info.elapsedSec,
    });
  }

  run() { return this.trainer.run(); }

  // Per image: { psnr, sites (its own coordinates), count, layer }.
  async finalize() {
    const final = await this.trainer.finalize();
    const psnrs = tilePsnrs(final.renderRgba, this.images, this.layout);
    return this.images.map((img, b) => {
      const sites = tileSites(final.sites, this.layout, b);
      let active = 0;
      for (let i = 0; i < this.layout.sitesPerTile; i += 1) if (sites[i * SITE_FLOATS] >= 0) active += 1;
      return { psnr: psnrs[b], sites, count: this.layout.sitesPerTile, activeSites: active, layer: this.layerSlot,
               width: img.width, height: img.height };
    }).map((r) => ({ ...r, trainSeconds: final.trainSeconds }));
  }

  destroy() { this.trainer.destroy(); }
}
