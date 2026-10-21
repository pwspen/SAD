// Checks codec.js (JS only, no GPU):
//   1. round trip: decodeSites returns exactly the encoder's dequantized sites
//   2. the bitstream length matches the counted bits (within the coder's flush)
//   3. dequantized sites are within half a step of the originals
//   4. the histogram layer's index survives the reordering
//   5. coarser steps cost fewer bits; clustered sites cost less than noise
//   6. edge cases: one site, every site round, every site in one cell
//   7. colors along a KLT basis
//   8. a shared model (buildSharedModel) fit on other images: round trip, and cheaper
// Usage: node codec_test.mjs

import { encodeSites, decodeSites, codecSteps, buildSharedModel } from "../trainer/codec.js?v=shared-20261020";
import { pcaBasis } from "../trainer/color_spaces.js?v=klt-20261020";
import { SITE_FLOATS } from "../trainer/params.js?v=20261019";

let failures = 0;
function check(name, ok, detail) {
  if (!ok) {
    failures += 1;
    console.error(`FAIL ${name}: ${detail}`);
  }
}

// Deterministic PRNG (mulberry32).
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomSites(n, w, h, rand, { smooth = false, inactive = 0 } = {}) {
  const s = new Float32Array(n * SITE_FLOATS);
  for (let i = 0; i < n; i += 1) {
    const b = i * SITE_FLOATS;
    const x = rand() * w, y = rand() * h;
    const th = rand() * Math.PI * 2;
    // smooth: colors and τ follow position, as on a real image
    const c = (k) => (smooth ? 0.5 + 0.4 * Math.sin(x / 40 + k) * Math.cos(y / 50) : rand());
    s.set([x, y, smooth ? 3 + Math.sin(x / 60) : 1 + 4 * rand(), (rand() - 0.5) * 0.2,
           c(0), c(1), c(2), Math.cos(th), Math.sin(th), (rand() - 0.5) * 2], b);
    if (i < inactive) s[b] = -1;
  }
  return s;
}

const STEPS = { pos: 0.25, logTau: 0.05, radius: 0.01, color: 1 / 128, logAniso: 0.05, theta: 0.02 };

function roundTrip(name, sites, count, w, h, steps, opts = {}) {
  const enc = encodeSites(sites, count, w, h, steps, opts);
  const dec = decodeSites(enc.bytes, { colorBasis: opts.colorBasis ?? null, shared: opts.shared ?? null });
  check(`${name} count`, dec.count === enc.n, `${dec.count} vs ${enc.n}`);
  check(`${name} size`, dec.width === w && dec.height === h, `${dec.width}×${dec.height}`);
  let bad = -1;
  for (let i = 0; i < enc.decoded.length; i += 1) if (enc.decoded[i] !== dec.sites[i]) { bad = i; break; }
  check(`${name} exact`, bad < 0, `value ${bad}: ${enc.decoded[bad]} vs ${dec.sites[bad]}`);
  check(`${name} bits`, enc.bytes.length === Math.ceil(enc.bits / 8), `${enc.bytes.length} bytes, ${enc.bits} bits`);
  const counted = enc.headerBits + enc.positionBits + enc.planeBits.reduce((a, b) => a + b, 0);
  check(`${name} accounting`, Math.abs(enc.bits - counted) < 40, `${enc.bits} written vs ${counted.toFixed(1)} counted`);
  check(`${name} global`, dec.globalSite === enc.globalIndex, `${dec.globalSite} vs ${enc.globalIndex}`);
  return enc;
}

// 1–4: random sites with some inactive slots and a layer
{
  const rand = rng(1);
  const w = 256, h = 192, n = 500;
  const sites = randomSites(n, w, h, rand, { inactive: 20 });
  const enc = roundTrip("random", sites, n, w, h, STEPS, { globalSite: 37 });
  check("random active", enc.n === n - 20, enc.n);
  check("random layer", enc.globalIndex >= 0 && enc.slots[enc.globalIndex] === 37, enc.globalIndex);
  let worst = 0;
  enc.slots.forEach((slot, i) => {
    const a = slot * SITE_FLOATS, b = i * SITE_FLOATS;
    worst = Math.max(worst,
      Math.abs(sites[a] - enc.decoded[b]) / STEPS.pos, Math.abs(sites[a + 2] - enc.decoded[b + 2]) / STEPS.logTau,
      Math.abs(sites[a + 4] - enc.decoded[b + 4]) / STEPS.color);
  });
  // Steps are rounded to 2^(1/8) in the header: up to ~4.4% coarser.
  check("random error", worst <= 0.53, `worst ${worst.toFixed(3)} steps`);
  console.log(`random: ${enc.n} sites, ${(enc.bits / enc.n).toFixed(1)} bits/site ` +
              `(positions ${(enc.positionBits / enc.n).toFixed(1)}, header ${enc.headerBits})`);
}

// 5: coarser steps → fewer bits; smooth sites code shorter than random ones
{
  const w = 256, h = 256, n = 800;
  const smooth = randomSites(n, w, h, rng(2), { smooth: true });
  const noisy = randomSites(n, w, h, rng(2));
  const fine = roundTrip("smooth", smooth, n, w, h, STEPS);
  const coarse = roundTrip("coarse", smooth, n, w, h, Object.fromEntries(Object.entries(STEPS).map(([k, v]) => [k, v * 4])));
  const rnd = roundTrip("noisy", noisy, n, w, h, STEPS);
  check("coarser is cheaper", coarse.bits < fine.bits, `${coarse.bits} vs ${fine.bits}`);
  check("smooth is cheaper", fine.bits < rnd.bits, `${fine.bits} vs ${rnd.bits}`);
  check("smooth uses neighbours", fine.modes.slice(2).some((m) => m === 1), fine.modes.join(""));
  console.log(`smooth: ${(fine.bits / n).toFixed(1)} bits/site, 4× coarser ${(coarse.bits / n).toFixed(1)}, ` +
              `random colors ${(rnd.bits / n).toFixed(1)}`);
}

// 6: edge cases
{
  roundTrip("one site", randomSites(1, 28, 28, rng(3)), 1, 28, 28, STEPS);
  const round = randomSites(50, 64, 64, rng(4));
  for (let i = 0; i < 50; i += 1) round[i * SITE_FLOATS + 9] = 0;
  roundTrip("all round", round, 50, 64, 64, STEPS);
  const stacked = randomSites(30, 64, 64, rng(5));
  for (let i = 0; i < 30; i += 1) { stacked[i * SITE_FLOATS] = 10.1; stacked[i * SITE_FLOATS + 1] = 20.2; }
  roundTrip("one cell", stacked, 30, 64, 64, STEPS);
  roundTrip("infinite steps", randomSites(40, 64, 64, rng(6)), 40, 64, 64,
            { pos: 1, logTau: Infinity, radius: Infinity, color: Infinity, logAniso: Infinity, theta: Infinity });
}

// 7: colors along a KLT basis: exact round trip, error within half a step per
// axis (so ≤ √3/2 steps in RGB), and fewer color bits than YCoCg-R on colors
// that vary along one non-grey axis.
{
  const w = 128, h = 128, n = 400;
  const sites = randomSites(n, w, h, rng(8));
  const rand = rng(9);
  for (let i = 0; i < n; i += 1) {
    const t = rand(), b = i * SITE_FLOATS;
    sites.set([0.2 + 0.6 * t, 0.5 - 0.3 * t + 0.01 * rand(), 0.1 + 0.5 * t], b + 4);
  }
  const rgba = new Float32Array(n * 4);
  for (let i = 0; i < n; i += 1) rgba.set([...sites.subarray(i * SITE_FLOATS + 4, i * SITE_FLOATS + 7), 1], i * 4);
  const colorBasis = pcaBasis(rgba, n);
  const klt = roundTrip("klt", sites, n, w, h, STEPS, { colorBasis });
  const ycc = roundTrip("ycocg-r", sites, n, w, h, STEPS);
  let worst = 0;
  klt.slots.forEach((slot, i) => {
    for (let c = 0; c < 3; c += 1) worst = Math.max(worst, Math.abs(sites[slot * SITE_FLOATS + 4 + c] - klt.decoded[i * SITE_FLOATS + 4 + c]) / STEPS.color);
  });
  check("klt error", worst <= 0.9, `worst ${worst.toFixed(3)} steps`);
  const colorBits = (e) => e.planeBits.slice(4).reduce((a, b) => a + b, 0);
  check("klt codes one-axis colors shorter", colorBits(klt) < colorBits(ycc), `${colorBits(klt)} vs ${colorBits(ycc)}`);
  let threw = false;
  try { decodeSites(klt.bytes); } catch { threw = true; }
  check("klt stream needs its basis", threw, "decoded without the basis");
  console.log(`klt: color ${(colorBits(klt) / n).toFixed(2)} bits/site vs YCoCg-R ${(colorBits(ycc) / n).toFixed(2)}`);
}

// 8: shared model. Fit on one set of small "images" (few sites, shared
// statistics), coded on a disjoint held-out set: exact round trip (also after
// a JSON round trip of the model, with a color basis, an odd image size, no
// layer, one site, no sites), and fewer bits than coding each image alone.
{
  const image = (rand) => {
    const n = 6 + Math.floor(rand() * 6), s = randomSites(n, 28, 28, rand);
    for (let i = 0; i < n; i += 1) {
      const b = i * SITE_FLOATS, v = rand() < 0.3 ? 0 : 0.8 + 0.2 * rand();
      s.set([v, v, v], b + 4);
      s[b + 2] = 4 + 0.5 * rand();
      s[b + 3] = 3 + rand();
      if (rand() < 0.5) s[b + 9] = 0;
    }
    return { sites: s, n, width: 28, height: 28, globalSite: 0 };
  };
  const rand = rng(10);
  const train = Array.from({ length: 200 }, () => image(rand));
  const test = Array.from({ length: 100 }, () => image(rand));
  const steps = { ...STEPS, pos: 1, color: 1 / 32 };
  const shared = buildSharedModel(train, steps);
  const json = JSON.parse(JSON.stringify(shared));
  let alone = 0, withShared = 0, sites = 0, header = 0;
  test.forEach((im, k) => {
    const enc = roundTrip(`shared ${k}`, im.sites, im.n, 28, 28, null, { globalSite: im.globalSite, shared: k % 2 ? json : shared });
    alone += encodeSites(im.sites, im.n, 28, 28, steps, { globalSite: im.globalSite }).bits;
    withShared += enc.bits; sites += enc.n; header += enc.headerBits;
  });
  check("shared is cheaper", withShared < 0.8 * alone, `${withShared} vs ${alone} bits`);
  const one = randomSites(1, 28, 28, rng(11)), none = randomSites(3, 28, 28, rng(12), { inactive: 3 });
  roundTrip("shared odd size", randomSites(40, 61, 37, rng(13)), 40, 61, 37, null, { shared });
  roundTrip("shared one site", one, 1, 28, 28, null, { shared });
  roundTrip("shared no sites", none, 3, 28, 28, null, { shared, globalSite: 1 });
  const rgba = new Float32Array(train.length * 4);
  train.forEach((im, i) => rgba.set([...im.sites.subarray(4, 7), 1], i * 4));
  const kltShared = buildSharedModel(train, steps, { colorBasis: pcaBasis(rgba, train.length) });
  test.slice(0, 10).forEach((im, k) => roundTrip(`shared klt ${k}`, im.sites, im.n, 28, 28, null, { globalSite: 0, shared: kltShared }));
  console.log(`shared: ${(withShared / sites).toFixed(1)} bits/site held out (header ${(header / sites).toFixed(1)}) ` +
              `vs ${(alone / sites).toFixed(1)} per image alone; strengths ${JSON.stringify(shared.strengths)}`);
}

// codecSteps: equal predicted error per scalar, and a bigger budget, bigger steps
{
  const n = 100;
  const sites = randomSites(n, 64, 64, rng(7));
  const F = new Float32Array(n * 10).map((_, i) => 1 + (i % 10));
  const a = codecSteps(sites, n, F, { lossDb: 0.1, sse0: 10 });
  const b = codecSteps(sites, n, F, { lossDb: 0.5, sse0: 10 });
  check("steps grow with budget", Object.keys(a).every((k) => b[k] > a[k]), JSON.stringify({ a, b }));
  const added = n * (3 * a.logTau ** 2 + 4 * a.radius ** 2) / 12; // log τ and radius shares
  const want = 2 * (10 * (10 ** 0.01 - 1)) / 9;
  check("steps split the error", Math.abs(added - want) / want < 1e-9, `${added} vs ${want}`);
}

if (failures) {
  console.error(`${failures} failure(s)`);
  process.exit(1);
}
console.log("codec: all checks passed");
