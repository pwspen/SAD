// Viewer-only image analysis for train.html. Everything here runs on the CPU
// against copies of the target and the latest render; nothing feeds back into
// the optimizer. Images are Float32 RGBA in [0, 1]; masks are per-pixel
// weights (Float32Array, length w·h) or null for "everything counts".

export function luma(rgbaF, width, height) {
  const n = width * height;
  const out = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p += 1, i += 4) {
    out[p] = 0.299 * rgbaF[i] + 0.587 * rgbaF[i + 1] + 0.114 * rgbaF[i + 2];
  }
  return out;
}

// Mask weights from an RGBA mask (R channel), or null when there's no mask.
export function maskWeights(maskRgba, width, height) {
  if (!maskRgba || maskRgba.length !== width * height * 4) return null;
  const out = new Float32Array(width * height);
  for (let p = 0; p < out.length; p += 1) out[p] = maskRgba[p * 4];
  return out;
}

// ---- Laplacian pyramid ----

// Separable [1 4 6 4 1]/16 blur followed by 2× decimation (clamped edges).
export function pyrDown(src, w, h) {
  const k = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  const dw = Math.ceil(w / 2);
  const dh = Math.ceil(h / 2);
  const tmp = new Float32Array(dw * h);
  for (let y = 0; y < h; y += 1) {
    for (let dx = 0; dx < dw; dx += 1) {
      const x = dx * 2;
      let s = 0;
      for (let t = -2; t <= 2; t += 1) {
        const xx = Math.min(w - 1, Math.max(0, x + t));
        s += k[t + 2] * src[y * w + xx];
      }
      tmp[y * dw + dx] = s;
    }
  }
  const out = new Float32Array(dw * dh);
  for (let dy = 0; dy < dh; dy += 1) {
    const y = dy * 2;
    for (let dx = 0; dx < dw; dx += 1) {
      let s = 0;
      for (let t = -2; t <= 2; t += 1) {
        const yy = Math.min(h - 1, Math.max(0, y + t));
        s += k[t + 2] * tmp[yy * dw + dx];
      }
      out[dy * dw + dx] = s;
    }
  }
  return { data: out, width: dw, height: dh };
}

// Bilinear resample of a (sw×sh) grid onto (w×h), pixel-center aligned.
export function upsampleBilinear(src, sw, sh, w, h) {
  const out = new Float32Array(w * h);
  const sx = sw / w;
  const sy = sh / h;
  for (let y = 0; y < h; y += 1) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x += 1) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = fx - x0;
      const a = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
      const b = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
      out[y * w + x] = a + (b - a) * ty;
    }
  }
  return out;
}

// Laplacian bands L_k = G_k − up(G_{k+1}), finest first, each at its own
// level's resolution. Stops early when a level gets smaller than 2×2.
export function laplacianBands(img, width, height, levels) {
  const bands = [];
  let level = { data: img, width, height };
  for (let k = 0; k < levels; k += 1) {
    if (level.width < 2 || level.height < 2) break;
    const next = pyrDown(level.data, level.width, level.height);
    const up = upsampleBilinear(next.data, next.width, next.height, level.width, level.height);
    const band = new Float32Array(level.width * level.height);
    for (let p = 0; p < band.length; p += 1) band[p] = level.data[p] - up[p];
    bands.push({ data: band, width: level.width, height: level.height });
    level = next;
  }
  return bands;
}

// The mask blurred down to each band's resolution (null mask → null levels).
export function maskLevels(mask, width, height, levels) {
  if (!mask) return null;
  const out = [];
  let level = { data: mask, width, height };
  for (let k = 0; k < levels; k += 1) {
    out.push(level.data);
    if (level.width < 2 || level.height < 2) break;
    level = pyrDown(level.data, level.width, level.height);
  }
  return out;
}

// Mask-weighted mean of L_k² per band.
export function bandEnergies(bands, masks) {
  return bands.map((b, k) => {
    const m = masks?.[k];
    let s = 0, wsum = 0;
    for (let p = 0; p < b.data.length; p += 1) {
      const wt = m ? m[p] : 1;
      s += wt * b.data[p] * b.data[p];
      wsum += wt;
    }
    return wsum > 0 ? s / wsum : 0;
  });
}

// ---- Multi-scale importance map ----
// Each band's energy L_k² is upsampled to full res and combined as
//   w = mask · (ε + sqrt(Σ_k β_k · up(L_k²)))^γ
// so smooth large-scale structure (coarse bands) counts as well as fine edges.
export const IMPORTANCE = {
  levels: 6,
  bandWeights: [0.5, 1, 1, 1, 1, 1], // β_k, finest band first
  gamma: 1.0,
  eps: 1e-3,
};

export function importanceMap(rgbaF, width, height, mask) {
  const n = width * height;
  const bands = laplacianBands(luma(rgbaF, width, height), width, height, IMPORTANCE.levels);
  const energy = new Float32Array(n);
  bands.forEach((b, k) => {
    const sq = new Float32Array(b.data.length);
    for (let p = 0; p < sq.length; p += 1) sq[p] = b.data[p] * b.data[p];
    const full = k === 0 ? sq : upsampleBilinear(sq, b.width, b.height, width, height);
    const beta = IMPORTANCE.bandWeights[k] ?? 1;
    for (let p = 0; p < n; p += 1) energy[p] += beta * full[p];
  });
  const out = new Float32Array(n);
  for (let p = 0; p < n; p += 1) {
    const m = mask ? mask[p] : 1;
    out[p] = m * Math.pow(IMPORTANCE.eps + Math.sqrt(energy[p]), IMPORTANCE.gamma);
  }
  return out;
}

// ---- FFT ----

const fftTables = new Map();

function fftTable(n) {
  let t = fftTables.get(n);
  if (t) return t;
  const bits = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) {
    let r = 0;
    for (let b = 0; b < bits; b += 1) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k += 1) {
    cos[k] = Math.cos((2 * Math.PI * k) / n);
    sin[k] = Math.sin((2 * Math.PI * k) / n);
  }
  t = { rev, cos, sin };
  fftTables.set(n, t);
  return t;
}

// In-place iterative radix-2 forward FFT (e^{-2πi kn/N}); length must be 2^m.
function fft(re, im) {
  const n = re.length;
  if (n < 2) return;
  const { rev, cos, sin } = fftTable(n);
  for (let i = 0; i < n; i += 1) {
    const j = rev[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0, k = 0; j < half; j += 1, k += step) {
        const a = i + j, b = a + half;
        const tr = re[b] * cos[k] + im[b] * sin[k];
        const ti = im[b] * cos[k] - re[b] * sin[k];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

const nextPow2 = (v) => 2 ** Math.ceil(Math.log2(Math.max(2, v)));

// ---- Spectra ----
// A plan fixes everything that depends only on the image size and mask: the
// analysis window (2D Hann × mask), the zero-padded FFT size, and the log-spaced
// radial bins in cycles/pixel. Bins run from 2/min(w, h) (below that the
// window's own spread dominates) up to Nyquist, 0.5; corners past 0.5 are dropped.
export const SPECTRUM_BINS = 40;

export function spectralPlan(width, height, mask) {
  const nx = nextPow2(width);
  const ny = nextPow2(height);
  const hann = (i, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / n);
  const win = new Float32Array(width * height);
  let wSum = 0, w2Sum = 0;
  for (let y = 0; y < height; y += 1) {
    const hy = hann(y, height);
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      const v = hy * hann(x, width) * (mask ? mask[p] : 1);
      win[p] = v;
      wSum += v;
      w2Sum += v * v;
    }
  }
  const fMin = 2 / Math.min(width, height);
  const fMax = 0.5;
  const lMin = Math.log(fMin), lSpan = Math.log(fMax) - lMin;
  const binOf = new Int16Array(nx * ny).fill(-1);
  const counts = new Uint32Array(SPECTRUM_BINS);
  for (let ky = 0; ky < ny; ky += 1) {
    const fy = (ky < ny / 2 ? ky : ky - ny) / ny;
    for (let kx = 0; kx < nx; kx += 1) {
      const fx = (kx < nx / 2 ? kx : kx - nx) / nx;
      const r = Math.hypot(fx, fy);
      if (r < fMin || r > fMax) continue;
      const b = Math.min(SPECTRUM_BINS - 1, Math.floor(((Math.log(r) - lMin) / lSpan) * SPECTRUM_BINS));
      binOf[ky * nx + kx] = b;
      counts[b] += 1;
    }
  }
  // Geometric bin centres; empty bins (tiny images) are skipped downstream.
  const freq = Array.from({ length: SPECTRUM_BINS }, (_, b) => Math.exp(lMin + ((b + 0.5) / SPECTRUM_BINS) * lSpan));
  return { width, height, nx, ny, win, wSum, w2Sum, binOf, counts, freq };
}

// 2D FFT of (luma − windowed mean) · window, zero-padded to the plan size.
export function windowedSpectrum(plan, img) {
  const { width, height, nx, ny, win, wSum } = plan;
  let mean = 0;
  for (let p = 0; p < win.length; p += 1) mean += win[p] * img[p];
  mean = wSum > 0 ? mean / wSum : 0;
  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      re[y * nx + x] = (img[p] - mean) * win[p];
    }
  }
  // Rows past the image are all zero, so their transform is too.
  for (let y = 0; y < height; y += 1) {
    fft(re.subarray(y * nx, (y + 1) * nx), im.subarray(y * nx, (y + 1) * nx));
  }
  const cr = new Float64Array(ny);
  const ci = new Float64Array(ny);
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) { cr[y] = re[y * nx + x]; ci[y] = im[y * nx + x]; }
    fft(cr, ci);
    for (let y = 0; y < ny; y += 1) { re[y * nx + x] = cr[y]; im[y * nx + x] = ci[y]; }
  }
  return { re, im };
}

// The residual's spectrum, by linearity: F(recon − target) = F(recon) − F(target).
export function spectrumDiff(a, b) {
  const re = new Float64Array(a.re.length);
  const im = new Float64Array(a.im.length);
  for (let i = 0; i < re.length; i += 1) { re[i] = a.re[i] - b.re[i]; im[i] = a.im[i] - b.im[i]; }
  return { re, im };
}

// Per radial bin: power of target / recon / residual (|F|² / Σwindow², i.e.
// per-pixel power), spectral SNR in dB, and Fourier ring correlation
// FRC = Σ Re(F_r · F_t*) / sqrt(Σ|F_r|² · Σ|F_t|²).
export function radialStats(plan, Ft, Fr, Fe) {
  const nb = SPECTRUM_BINS;
  const pt = new Float64Array(nb), pr = new Float64Array(nb), pe = new Float64Array(nb), cross = new Float64Array(nb);
  const { binOf } = plan;
  for (let i = 0; i < binOf.length; i += 1) {
    const b = binOf[i];
    if (b < 0) continue;
    const tr = Ft.re[i], ti = Ft.im[i], rr = Fr.re[i], ri = Fr.im[i], er = Fe.re[i], ei = Fe.im[i];
    pt[b] += tr * tr + ti * ti;
    pr[b] += rr * rr + ri * ri;
    pe[b] += er * er + ei * ei;
    cross[b] += rr * tr + ri * ti;
  }
  const bins = [];
  for (let b = 0; b < nb; b += 1) {
    const c = plan.counts[b];
    if (!c) continue;
    const norm = 1 / (c * plan.w2Sum);
    bins.push({
      f: plan.freq[b],
      target: pt[b] * norm,
      recon: pr[b] * norm,
      residual: pe[b] * norm,
      ssnr: 10 * Math.log10(pt[b] / Math.max(pe[b], 1e-30)),
      frc: pt[b] > 0 && pr[b] > 0 ? cross[b] / Math.sqrt(pt[b] * pr[b]) : 0,
    });
  }
  return bins;
}

// Lowest frequency where FRC falls below `threshold` (log-interpolated between
// bins), or null if it never does.
export function frcCrossing(bins, threshold = 0.5) {
  for (let i = 1; i < bins.length; i += 1) {
    const a = bins[i - 1], b = bins[i];
    if (a.frc >= threshold && b.frc < threshold) {
      const t = (a.frc - threshold) / (a.frc - b.frc);
      return Math.exp(Math.log(a.f) + t * (Math.log(b.f) - Math.log(a.f)));
    }
  }
  return null;
}

// log10 power on a size×size grid with DC at the centre: each output pixel
// averages the FFT bins that fall in it (size ≤ min(nx, ny), so none is empty).
export function logPowerImage(plan, F, size) {
  const { nx, ny, w2Sum } = plan;
  const sum = new Float64Array(size * size);
  const cnt = new Uint32Array(size * size);
  for (let ky = 0; ky < ny; ky += 1) {
    const fy = (ky < ny / 2 ? ky : ky - ny) / ny;
    const v = Math.min(size - 1, Math.floor((fy + 0.5) * size));
    for (let kx = 0; kx < nx; kx += 1) {
      const fx = (kx < nx / 2 ? kx : kx - nx) / nx;
      const u = Math.min(size - 1, Math.floor((fx + 0.5) * size));
      const i = ky * nx + kx;
      sum[v * size + u] += F.re[i] * F.re[i] + F.im[i] * F.im[i];
      cnt[v * size + u] += 1;
    }
  }
  const out = new Float32Array(size * size);
  for (let p = 0; p < out.length; p += 1) out[p] = Math.log10(sum[p] / (cnt[p] * w2Sum) + 1e-20);
  return out;
}

// ---- Mutual information between per-site parameters ----
// Plug-in entropies from equal-width histograms with the Miller–Madow bias
// correction, H ≈ H_ML + (m − 1) / (2N ln 2) bits, m = occupied bins. Each
// column is a Float32Array over sites; NaN marks "undefined for this site"
// and drops it from every pair that uses that column.

export const MI_BINS = 16;

// Bin index per site (−1 for NaN). `range` = [lo, hi] fixes the bins (for
// circular values); otherwise they span the 1st–99th percentile and outliers
// fall into the edge bins.
function binColumn(values, bins, range) {
  let lo, hi;
  if (range) {
    [lo, hi] = range;
  } else {
    const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
    if (!finite.length) return new Int32Array(values.length).fill(-1);
    lo = finite[Math.floor(0.01 * (finite.length - 1))];
    hi = finite[Math.ceil(0.99 * (finite.length - 1))];
  }
  const scale = hi > lo ? bins / (hi - lo) : 0;
  const out = new Int32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    out[i] = Number.isFinite(v) ? Math.max(0, Math.min(bins - 1, Math.floor((v - lo) * scale))) : -1;
  }
  return out;
}

// Miller–Madow entropy in bits of a count table with total n.
function entropyMM(counts, n) {
  let h = 0, m = 0;
  for (const c of counts) {
    if (c > 0) { h -= (c / n) * Math.log2(c / n); m += 1; }
  }
  return h + (m - 1) / (2 * n * Math.LN2);
}

// columns = [{ values, range? }]. Returns { n, h, bits, norm } where h[i] is
// column i's entropy over its own defined sites, and for each pair, over the
// sites defined in both: bits[i][j] = H(X) + H(Y) − H(X,Y) and
// norm[i][j] = bits / min(H(X), H(Y)) clamped to [0, 1]. Diagonals are 1.
export function mutualInfoMatrix(columns, bins = MI_BINS) {
  const k = columns.length;
  const idx = columns.map((c) => binColumn(c.values, bins, c.range));
  const nSites = idx[0]?.length ?? 0;
  const n = [], h = [], bits = [], norm = [];
  for (let i = 0; i < k; i += 1) {
    const counts = new Float64Array(bins);
    let ni = 0;
    for (const b of idx[i]) if (b >= 0) { counts[b] += 1; ni += 1; }
    h.push(ni > 1 ? entropyMM(counts, ni) : 0);
    n.push(new Array(k).fill(0));
    bits.push(new Array(k).fill(0));
    norm.push(new Array(k).fill(1));
    n[i][i] = ni;
    bits[i][i] = h[i];
  }
  const joint = new Float64Array(bins * bins);
  const px = new Float64Array(bins), py = new Float64Array(bins);
  for (let i = 0; i < k; i += 1) {
    for (let j = i + 1; j < k; j += 1) {
      joint.fill(0); px.fill(0); py.fill(0);
      let nij = 0;
      for (let s = 0; s < nSites; s += 1) {
        const a = idx[i][s], b = idx[j][s];
        if (a < 0 || b < 0) continue;
        joint[a * bins + b] += 1; px[a] += 1; py[b] += 1; nij += 1;
      }
      let I = 0, r = 0;
      if (nij > 1) {
        const hx = entropyMM(px, nij), hy = entropyMM(py, nij);
        I = Math.max(0, hx + hy - entropyMM(joint, nij));
        const hMin = Math.min(hx, hy);
        r = hMin > 0 ? Math.min(1, I / hMin) : 0;
      }
      n[i][j] = n[j][i] = nij;
      bits[i][j] = bits[j][i] = I;
      norm[i][j] = norm[j][i] = r;
    }
  }
  return { n, h, bits, norm };
}
