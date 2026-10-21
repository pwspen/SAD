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
//
// "klt" has no fixed transform: its basis (a mean and three orthonormal axes,
// pcaBasis()) comes from the data, colorSpaceFor(name, basis) binds it, and
// specializeShaders() writes it into the WGSL as constants. The trainer takes
// it from the masked target pixels at prepare(), so an atlas of images shares
// one, or from args.colorBasis.

import { SITE_FLOATS } from "./params.js?v=m3-20261020";

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
  // Real-valued YCoCg, the linear transform the codec's YCoCg-R approximates
  // on integers: Y = (R + 2G + B) / 4, Co = (R − B) / 2, Cg = (2G − R − B) / 4.
  // Luma and the two chroma axes get their own Adam scales.
  ycocg: {
    id: 2,
    label: "YCoCg",
    encode: (r, g, b) => [(r + 2 * g + b) / 4, (r - b) / 2, (2 * g - r - b) / 4],
    decode: (y, co, cg) => [y - cg + co, y + cg, y - cg - co],
    vjp: (p, g) => [g[0] + g[1] + g[2], g[0] - g[2], g[1] - g[0] - g[2]],
  },
};

export const IDENTITY_BASIS = { mean: [0, 0, 0], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], variances: [1, 1, 1] };

// Coordinates along orthonormal `axes` about `mean`: p = Aᵀ(rgb − mean),
// rgb = mean + A p, and the VJP is Aᵀ g.
function kltSpace(basis) {
  const { mean: m, axes: a } = basis;
  const dot3 = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  return {
    id: 3,
    label: "KLT (PCA of the target)",
    needsBasis: true,
    basis,
    encode: (r, g, b) => a.map((ax) => dot3(ax, [r - m[0], g - m[1], b - m[2]])),
    decode: (p0, p1, p2) => [0, 1, 2].map((c) => m[c] + a[0][c] * p0 + a[1][c] * p1 + a[2][c] * p2),
    vjp: (p, g) => a.map((ax) => dot3(ax, g)),
  };
}

COLOR_SPACES.klt = kltSpace(IDENTITY_BASIS);

// `basis` binds klt's ({ mean, axes }, pcaBasis()); other spaces ignore it.
export function colorSpaceFor(name, basis = null) {
  const space = COLOR_SPACES[name];
  if (!space) {
    throw new Error(`Unknown color space "${name}". Options: ${Object.keys(COLOR_SPACES).join(", ")}`);
  }
  return space.needsBasis && basis ? kltSpace(basis) : space;
}

// Eigen-decomposition of a symmetric 3×3 (cyclic Jacobi): { values, vectors }
// with vectors[k] the unit eigenvector of values[k], largest value first.
function symEigen3(C) {
  const A = C.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep += 1) {
    const off = A[0][1] ** 2 + A[0][2] ** 2 + A[1][2] ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < 3; k += 1) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - sn * akq;
        A[k][q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k += 1) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - sn * aqk;
        A[q][k] = sn * apk + c * aqk;
      }
      for (let k = 0; k < 3; k += 1) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - sn * vkq;
        V[k][q] = sn * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => A[j][j] - A[i][i]);
  return { values: order.map((i) => Math.max(0, A[i][i])), vectors: order.map((i) => [V[0][i], V[1][i], V[2][i]]) };
}

// The KLT of a set of colors: their mean and principal axes, largest variance
// first. rgba: interleaved RGBA (alpha ignored), `count` pixels; mask: one
// value per pixel, > 0 counts (null = all). Signs are fixed so each axis's
// largest component is positive (the first axis of grey-ish data then points
// along +luma), and the axes form a right-handed frame. Axes with no variance
// (e.g. chroma of a greyscale set) still come out orthonormal.
export function pcaBasis(rgba, count, mask = null) {
  const mean = [0, 0, 0];
  let n = 0;
  for (let i = 0; i < count; i += 1) {
    if (mask && !(mask[i] > 0)) continue;
    for (let c = 0; c < 3; c += 1) mean[c] += rgba[i * 4 + c];
    n += 1;
  }
  if (!n) return IDENTITY_BASIS;
  for (let c = 0; c < 3; c += 1) mean[c] /= n;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < count; i += 1) {
    if (mask && !(mask[i] > 0)) continue;
    const d = [0, 1, 2].map((c) => rgba[i * 4 + c] - mean[c]);
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) C[r][c] += d[r] * d[c];
  }
  for (const row of C) for (let c = 0; c < 3; c += 1) row[c] /= n;
  const { values, vectors } = symEigen3(C);
  const axes = vectors.map((v) => {
    const big = v.reduce((k, x, i) => (Math.abs(x) > Math.abs(v[k]) + 1e-12 ? i : k), 0);
    return v[big] < 0 ? v.map((x) => -x) : v;
  });
  // Third axis = first × second, so the frame is right-handed and exactly orthonormal.
  const [u, w] = axes;
  axes[2] = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  return { mean, axes, variances: values };
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
const BASIS_AXES_RE = /const COLOR_BASIS_AXES : mat3x3<f32> = mat3x3<f32>\([^;]*\);/;
const BASIS_MEAN_RE = /const COLOR_BASIS_MEAN : vec3<f32> = vec3<f32>\([^;]*\);/;
const wgslF32 = (x) => { const s = Number(x).toPrecision(9); return /[.e]/.test(s) ? s : `${s}.0`; };

// Copy of a loadShaders() result with every kernel compiled for `space`.
export function specializeShaders(shaders, space) {
  const out = { ...shaders, colorSpaceId: space.id };
  for (const [key, src] of Object.entries(shaders)) {
    if (typeof src !== "string") continue;
    if (!COLOR_SPACE_RE.test(src)) {
      throw new Error(`Shader ${key} is missing its \`const COLOR_SPACE : u32 = Nu;\` line`);
    }
    let code = src.replace(COLOR_SPACE_RE, `const COLOR_SPACE : u32 = ${space.id}u;`);
    if (space.needsBasis) {
      // mat3x3 takes columns: column k is axis k.
      const { mean, axes } = space.basis;
      code = code
        .replace(BASIS_AXES_RE, `const COLOR_BASIS_AXES : mat3x3<f32> = mat3x3<f32>(${axes.flat().map(wgslF32).join(", ")});`)
        .replace(BASIS_MEAN_RE, `const COLOR_BASIS_MEAN : vec3<f32> = vec3<f32>(${mean.map(wgslF32).join(", ")});`);
    }
    out[key] = code;
  }
  return out;
}
