// Per-pixel candidate lists: the 8 sites each candidate pixel blends (plus the
// global site, see cand_ids in sad_shared.wgsl), kept current by
// vpt.wgsl passes. Shared by the trainer and the viewer (web/main.js).
//
// Lists live in two rgba32uint textures (cand0 = slots 0-3, cand1 = 4-7),
// double-buffered. A pass offers each pixel its own list, its 4 neighbours'
// lists at the current jump step, optional radius probes and a few random
// sites, and keeps the 8 with the highest logit. Jump steps
// halve from half the grid down to 1 px over the first passes (packJumpStep),
// then stay at 1 px.

import {
  createParamsBuffer,
  updateParamsBuffer,
  createPackParamsBuffer,
  updatePackParamsBuffer,
  packJumpStep,
} from "./params.js?v=20261019";
import { createPackedCandidatesBuffer } from "./buffers.js?v=20261019";
import { createCandidateTexture, readRgba32FloatTexture } from "./textures.js?v=20261019";
import { CandidatesEncoder, SeedCandidatesEncoder, CandidatePackEncoder } from "./encoders.js?v=20261019";

const NO_SITE = 0xffffffff;

export class CandidateLists {
  // opts: width, height (image), capacity (sites the buffers hold), and the
  // search settings candDownscale, candUpdatePasses, candRadiusScale,
  // candRadiusProbes, candInjectCount (as in buildArgs). opts.tiling: batched
  // training's atlas layout (trainer/batch.js); lists stay within a tile, and
  // jumps and scale follow the tile.
  constructor(device, shaders, opts) {
    this.device = device;
    this.width = opts.width;
    this.height = opts.height;
    this.opts = opts;
    this.downscale = Math.max(1, opts.candDownscale || 1);
    this.candWidth = Math.max(1, Math.ceil(this.width / this.downscale));
    this.candHeight = Math.max(1, Math.ceil(this.height / this.downscale));
    this.tiling = opts.tiling ?? null;
    // The grid the jump flood spans: one tile's cells when tiled.
    this.jumpWidth = this.tiling ? Math.ceil(this.tiling.tileW / this.downscale) : this.candWidth;
    this.jumpHeight = this.tiling ? Math.ceil(this.tiling.tileH / this.downscale) : this.candHeight;
    this.updatePasses = Math.max(1, opts.candUpdatePasses || 1);
    this.globalCand = NO_SITE; // the site every pixel blends besides its list
    this.jumpPassIndex = 0;

    this.updateEnc = new CandidatesEncoder(device, shaders);
    this.seedEnc = new SeedCandidatesEncoder(device, shaders);
    this.packEnc = new CandidatePackEncoder(device, shaders);
    this.tex = [0, 1, 2, 3].map(() => createCandidateTexture(device, this.candWidth, this.candHeight));
    this.packed = createPackedCandidatesBuffer(device, opts.capacity);
    this.packParams = createPackParamsBuffer(device);
    this.passParams = [];   // one per pass encoded before a submit
  }

  // Current lists.
  get cand0() { return this.tex[0]; }
  get cand1() { return this.tex[1]; }

  _invScale() {
    return 1 / (this.tiling ? Math.max(this.tiling.tileW, this.tiling.tileH) : Math.max(this.width, this.height));
  }

  _swap() {
    const [a0, a1, b0, b1] = this.tex;
    this.tex = [b0, b1, a0, a1];
  }

  // Params (sad_shared.wgsl) for `siteCount` sites at jump `step`.
  paramsArgs(siteCount, step = 0, seed = 0) {
    const o = this.opts;
    return {
      width: this.width, height: this.height, siteCount, step, seed,
      radiusScale: o.candRadiusScale, radiusProbes: o.candRadiusProbes, injectCount: o.candInjectCount,
      candDownscale: this.downscale, candWidth: this.candWidth, candHeight: this.candHeight,
      globalCand: this.globalCand, tiling: this.tiling,
    };
  }

  // The k-th params buffer of this submit, filled with `args`.
  _params(k, args) {
    this.passParams[k] ??= createParamsBuffer(this.device, args);
    updateParamsBuffer(this.device.queue, this.passParams[k], args);
    return this.passParams[k];
  }

  _stages() { return Math.max(1, Math.ceil(Math.log2(Math.max(this.jumpWidth, this.jumpHeight, 2)))); }

  // Repacks sites [0, n) for the search.
  _encodePack(encoder, sitesBuf, n) {
    updatePackParamsBuffer(this.device.queue, this.packParams, n, this._invScale(), this.tiling);
    this.packEnc.encode(encoder, sitesBuf, this.packed, this.packParams, n);
  }

  _encodePass(encoder, params) {
    this.updateEnc.encodeUpdate(encoder, this.packed, params, this.tex[0], this.tex[1], this.tex[2], this.tex[3],
      this.candWidth, this.candHeight);
    this._swap();
  }

  // Seeds sites [from, to) into their home pixels (slot 0).
  _encodeSeed(encoder, sitesBuf, from, to, k) {
    encoder.copyTextureToTexture({ texture: this.tex[0] }, { texture: this.tex[2] }, [this.candWidth, this.candHeight, 1]);
    this.seedEnc.encode(encoder, sitesBuf, this._params(k, this.paramsArgs(to, from)), this.tex[0], this.tex[2], to - from);
    [this.tex[0], this.tex[2]] = [this.tex[2], this.tex[0]];
  }

  // Jump-flood stages from half the grid down to 1 px, `rounds` times.
  _encodeFlood(encoder, sitesBuf, n, rounds, k) {
    this._encodePack(encoder, sitesBuf, n);
    for (let r = 0; r < rounds; r += 1) {
      for (let st = 0; st < this._stages(); st += 1) {
        this._encodePass(encoder, this._params(k++, this.paramsArgs(n, packJumpStep(st, this.jumpWidth, this.jumpHeight))));
      }
    }
    return k;
  }

  // Lists from scratch for sites [0, n): random ids, every site seeded at its
  // home pixel, then the jump flood twice.
  rebuild(sitesBuf, n) {
    const encoder = this.device.createCommandEncoder();
    this.updateEnc.encodeInit(encoder, this._params(0, this.paramsArgs(n)), this.tex[0], this.tex[1],
      this.candWidth, this.candHeight);
    this._encodeSeed(encoder, sitesBuf, 0, n, 1);
    this._encodeFlood(encoder, sitesBuf, n, 2, 2);
    this.device.queue.submit([encoder.finish()]);
  }

  // Sites [from, to) are new or moved (of n): seed them and run one jump flood
  // over the lists as they are, so they claim their regions now rather than
  // spreading 1 px per update.
  encodeAdded(encoder, sitesBuf, from, to, n = to) {
    this._encodeSeed(encoder, sitesBuf, from, to, 0);
    this._encodeFlood(encoder, sitesBuf, n, 1, 1);
  }

  // The regular update: repack sites [0, n), then updatePasses passes at the
  // next jump steps.
  encodeUpdate(encoder, sitesBuf, n) {
    this._encodePack(encoder, sitesBuf, n);
    for (let p = 0; p < this.updatePasses; p += 1) {
      const i = this.jumpPassIndex++;
      this._encodePass(encoder, this._params(p, this.paramsArgs(n,
        packJumpStep(i, this.jumpWidth, this.jumpHeight), (i >>> 16) & 0xffff)));
    }
  }

  // The lists as 8 u32 site ids per candidate pixel. (rgba32uint read back
  // through the float reader: same-type typed-array copies are bitwise.)
  async read() {
    const [c0, c1] = await Promise.all([
      readRgba32FloatTexture(this.device, this.tex[0], this.candWidth, this.candHeight),
      readRgba32FloatTexture(this.device, this.tex[1], this.candWidth, this.candHeight),
    ]);
    const u0 = new Uint32Array(c0.buffer), u1 = new Uint32Array(c1.buffer);
    const out = new Uint32Array(this.candWidth * this.candHeight * 8);
    for (let i = 0; i < this.candWidth * this.candHeight; i += 1) {
      out.set(u0.subarray(i * 4, i * 4 + 4), i * 8);
      out.set(u1.subarray(i * 4, i * 4 + 4), i * 8 + 4);
    }
    return out;
  }

  destroy() {
    const owned = [...this.tex, this.packed, this.packParams, ...this.passParams];
    for (const b of owned) b?.destroy();
  }
}
