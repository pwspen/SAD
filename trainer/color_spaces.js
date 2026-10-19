// Site color parameterizations. A site's three color_* slots hold params in
// the run's color space; blending, the loss and every site file stay RGB.
//
// Each space exists twice and the two must agree:
//   - here: `encode` / `decode` for the CPU boundary (init sites in, readbacks
//     out), plus `vjp` used only by the tests;
//   - in shaders/sad_shared.wgsl: decode_color / encode_color / color_vjp,
//     selected by `const COLOR_SPACE` (the `id` below), which
//     specializeShaders() rewrites per run.
// node/color_spaces_test.mjs checks the round trip, the VJP against finite
// differences, and the WGSL against these functions on the GPU.
//
// To add a space: add an entry here with the next id, add a case to the three
// WGSL functions, and run the test.

import { SITE_FLOATS } from "./params.js?v=20261019";

const COLOR_OFFSET = 4; // color_r/g/b within a site row (see io.js)

// log(x + EPS) keeps black finite. Must match LOG_CHROMA_EPS in sad_shared.wgsl.
export const LOG_CHROMA_EPS = 1 / 255;

function logChromaLogs(r, g, b) {
  return [
    Math.log(Math.max(r, 0) + LOG_CHROMA_EPS),
    Math.log(Math.max(g, 0) + LOG_CHROMA_EPS),
    Math.log(Math.max(b, 0) + LOG_CHROMA_EPS),
  ];
}

export const COLOR_SPACES = {
  rgb: {
    id: 0,
    label: "RGB",
    encode: (r, g, b) => [r, g, b],
    decode: (p0, p1, p2) => [p0, p1, p2],
    vjp: (p, g) => [g[0], g[1], g[2]],
  },
  // (L, c1, c2) = (mean log RGB, log R/G, log B/G). An ideal shadow (a pure
  // intensity scale) moves only L.
  logchroma: {
    id: 1,
    label: "log intensity + chroma",
    encode: (r, g, b) => {
      const [lr, lg, lb] = logChromaLogs(r, g, b);
      return [(lr + lg + lb) / 3, lr - lg, lb - lg];
    },
    decode: (L, c1, c2) => {
      const lg = L - (c1 + c2) / 3;
      return [Math.exp(lg + c1) - LOG_CHROMA_EPS, Math.exp(lg) - LOG_CHROMA_EPS, Math.exp(lg + c2) - LOG_CHROMA_EPS];
    },
    vjp: ([L, c1, c2], g) => {
      const lg = L - (c1 + c2) / 3;
      const u = [g[0] * Math.exp(lg + c1), g[1] * Math.exp(lg), g[2] * Math.exp(lg + c2)];
      return [u[0] + u[1] + u[2], (2 * u[0] - u[1] - u[2]) / 3, (2 * u[2] - u[1] - u[0]) / 3];
    },
  },
};

export function colorSpaceFor(name) {
  const space = COLOR_SPACES[name];
  if (!space) {
    throw new Error(`Unknown color space "${name}". Options: ${Object.keys(COLOR_SPACES).join(", ")}`);
  }
  return space;
}

function mapSiteColors(data, count, fn) {
  for (let i = 0; i < count; i += 1) {
    const base = i * SITE_FLOATS;
    if (data[base] < 0) continue; // inactive
    const c = base + COLOR_OFFSET;
    const out = fn(data[c], data[c + 1], data[c + 2]);
    data[c] = out[0];
    data[c + 1] = out[1];
    data[c + 2] = out[2];
  }
  return data;
}

// In-place conversion of a site array's color slots (RGB <-> space params).
export function encodeSiteColors(data, count, space) {
  return space.id === 0 ? data : mapSiteColors(data, count, space.encode);
}
export function decodeSiteColors(data, count, space) {
  return space.id === 0 ? data : mapSiteColors(data, count, space.decode);
}

const COLOR_SPACE_RE = /const COLOR_SPACE : u32 = \d+u;/;

// Copy of a loadShaders() result with every kernel compiled for `space`.
export function specializeShaders(shaders, space) {
  const out = { ...shaders, colorSpaceId: space.id };
  for (const [key, src] of Object.entries(shaders)) {
    if (typeof src !== "string") continue;
    if (!COLOR_SPACE_RE.test(src)) {
      throw new Error(`Shader ${key} is missing its \`const COLOR_SPACE : u32 = Nu;\` line`);
    }
    out[key] = src.replace(COLOR_SPACE_RE, `const COLOR_SPACE : u32 = ${space.id}u;`);
  }
  return out;
}
