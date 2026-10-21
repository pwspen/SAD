// Node host for the WebGPU JS trainer: Dawn device, file-backed shader reads,
// and PNG I/O. Everything training-related lives in ../trainer and is shared with
// train.html / bench.html; this file only replaces the browser-specific pieces.

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { create, globals } from "webgpu";
import { PNG } from "pngjs";

import { requestTrainingDevice, adapterLabel } from "../trainer/device.js?v=20261019";
import { loadShaders, tileSlotsForLimit } from "../trainer/shader_loader.js?v=rate-20261020";
import { floatImageToRgba8 } from "../trainer/io.js?v=20261019";

// GPUBufferUsage, GPUMapMode, GPUTextureUsage, ... are globals in the browser.
Object.assign(globalThis, globals);

// dawn.node tears down its instance when the GPU object is garbage collected,
// and the next queue call on a device from it segfaults. Hold every GPU/adapter
// for the life of the process.
const keepAlive = [];

export async function createHeadlessDevice(dawnFlags = []) {
  const gpu = create(dawnFlags);
  const { adapter, device } = await requestTrainingDevice(gpu);
  keepAlive.push(gpu, adapter);
  device.addEventListener?.("uncapturederror", (e) => {
    console.error(`GPU error: ${e.error?.message || e.error}`);
  });
  return { adapter, device, label: adapterLabel(adapter) };
}

export async function loadShadersFromDisk(device) {
  return await loadShaders({
    tileSlots: tileSlotsForLimit(device.limits.maxComputeWorkgroupStorageSize),
    readText: (url) => readFile(fileURLToPath(url), "utf8"),
  });
}

export async function loadJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function decodePng(buf, path) {
  try {
    return PNG.sync.read(buf); // always RGBA8, 16-bit inputs are rescaled
  } catch (err) {
    throw new Error(`${path}: only PNG input is supported in Node (${err.message})`);
  }
}

// Area-average downscale of RGBA8 to fit maxDim. The browser (createImageBitmap
// "high") and Python (PIL LANCZOS) resample slightly differently, so PSNR on a
// downscaled target is only comparable within the same host.
function downscaleRgba8(src, sw, sh, maxDim) {
  const scale = maxDim / Math.max(sw, sh);
  const dw = Math.max(1, Math.floor(sw * scale));
  const dh = Math.max(1, Math.floor(sh * scale));
  const out = new Uint8Array(dw * dh * 4);
  for (let y = 0; y < dh; y += 1) {
    const y0 = Math.floor((y * sh) / dh);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * sh) / dh));
    for (let x = 0; x < dw; x += 1) {
      const x0 = Math.floor((x * sw) / dw);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * sw) / dw));
      const acc = [0, 0, 0, 0];
      for (let yy = y0; yy < y1; yy += 1) {
        for (let xx = x0; xx < x1; xx += 1) {
          const p = (yy * sw + xx) * 4;
          for (let c = 0; c < 4; c += 1) acc[c] += src[p + c];
        }
      }
      const n = (y1 - y0) * (x1 - x0);
      for (let c = 0; c < 4; c += 1) out[(y * dw + x) * 4 + c] = Math.round(acc[c] / n);
    }
  }
  return { data: out, width: dw, height: dh };
}

// Same shape as textures.js loadImageToFloat32().
export async function loadPngToFloat32(path, maxDim = 0) {
  const png = decodePng(await readFile(path), path);
  let img = { data: png.data, width: png.width, height: png.height };
  if (maxDim > 0 && Math.max(img.width, img.height) > maxDim) {
    img = downscaleRgba8(img.data, img.width, img.height, maxDim);
  }
  const out = new Float32Array(img.width * img.height * 4);
  for (let i = 0; i < out.length; i += 1) out[i] = img.data[i] / 255.0;
  return { data: out, width: img.width, height: img.height, sourceWidth: png.width, sourceHeight: png.height };
}

// Same shape as textures.js loadMaskRgba32Float(): mask = 1 where gray > 0.
export async function loadPngMask(path, width, height) {
  const png = decodePng(await readFile(path), path);
  if (png.width !== width || png.height !== height) {
    throw new Error(`${path}: mask is ${png.width}x${png.height}, target is ${width}x${height}`);
  }
  const mask = new Float32Array(width * height);
  const rgba = new Float32Array(width * height * 4);
  let maskSum = 0;
  for (let i = 0; i < width * height; i += 1) {
    const m = png.data[i * 4] + png.data[i * 4 + 1] + png.data[i * 4 + 2] > 0 ? 1.0 : 0.0;
    mask[i] = m;
    rgba[i * 4] = m;
    rgba[i * 4 + 1] = m;
    rgba[i * 4 + 2] = m;
    rgba[i * 4 + 3] = 1.0;
    maskSum += m;
  }
  return { mask, rgba, maskSum };
}

export async function writeFloatPng(path, rgbaF, width, height) {
  const png = new PNG({ width, height });
  png.data = Buffer.from(floatImageToRgba8(rgbaF, width, height).buffer);
  await writeFile(path, PNG.sync.write(png));
}
