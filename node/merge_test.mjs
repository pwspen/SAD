// Checks merge.js's parameter averaging (JS only, no GPU):
//   1. scalar weights: pixel-count average, as before precision weights
//   2. precision weights pull each parameter toward the site that cares
//   3. no precision on either site: falls back to the pixel counts
//   4. bias correction: a site whose moments are younger isn't underweighted
//   5. candidateEdges: owner/runner-up from the lists' first two usable sites
//   6. neighborMerge: the most alike neighbours merge, position ignored
//   7. shape distance: dir ≡ −dir, and the axis doesn't count when isotropic
//   8. chained merges reach the budget past half the sites
// Usage: node merge_test.mjs

import { mergedSite, mergePrecision, sitePrecision, candidateEdges, neighborMerge } from "../trainer/merge.js?v=20261019";
import { SITE_FLOATS, ADAM_FLOATS } from "../trainer/params.js?v=cmoment-20261020";

let failures = 0;
function check(name, ok, detail) {
  if (!ok) {
    failures += 1;
    console.error(`FAIL ${name}: ${detail}`);
  }
}
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol;

// Two sites: x, y, log τ, radius, RGB, dir, log aniso. Same axis, so the
// doubled-angle average leaves it alone.
const sites = new Float32Array([
  10, 20, 1, 2, 0.0, 0.0, 0.0, 1, 0, 0.5,
  30, 40, 3, 6, 1.0, 1.0, 1.0, 1, 0, 1.5,
]);

// Adam state with v (the slots sitePrecision reads) set per site.
function adamFor(vA, vB, bornA = 0, bornB = 0) {
  const adam = new Float32Array(2 * ADAM_FLOATS);
  [[vA, bornA], [vB, bornB]].forEach(([v, born], i) => {
    const b = i * ADAM_FLOATS;
    adam[b + 2] = adam[b + 3] = v.pos;
    adam[b + 5] = v.tau;
    adam[b + 7] = v.radius;
    adam[b + 11] = adam[b + 12] = adam[b + 13] = v.color;
    adam[b + 16] = adam[b + 17] = v.dir;
    adam[b + 19] = v.aniso;
    adam[b + 20] = born;
  });
  return adam;
}
const all = (x) => ({ pos: x, tau: x, radius: x, color: x, dir: x, aniso: x });

// 1. Scalar weights 3:1.
{
  const m = mergedSite(sites, 0, 1, 3, 1);
  const want = Array.from({ length: SITE_FLOATS }, (_, k) => 0.75 * sites[k] + 0.25 * sites[SITE_FLOATS + k]);
  want[7] = 1; want[8] = 0;
  check("area average", want.every((v, k) => near(m[k], v)), `${Array.from(m)} vs ${want}`);
}

// 2. Site a cares about position 9× more, site b about color 9× more; the
// rest equal. Pixel counts (b 100× larger) must not matter.
{
  const va = { ...all(1e-4), pos: 9e-4 }, vb = { ...all(1e-4), color: 9e-4 };
  const t = 1000, beta2 = 0.999;
  const [wa, wb] = mergePrecision(adamFor(va, vb), 0, 1, t, beta2, 1, 100);
  const m = mergedSite(sites, 0, 1, wa, wb);
  check("position follows a", near(m[0], 0.9 * 10 + 0.1 * 30, 1e-4), `x ${m[0]}`);
  check("color follows b", near(m[4], 0.1 * 0 + 0.9 * 1, 1e-5), `r ${m[4]}`);
  check("log τ split evenly", near(m[2], 2, 1e-5), `log τ ${m[2]}`);
}

// 3. Fresh moments on both sites (v = 0): pixel counts decide.
{
  const [wa, wb] = mergePrecision(adamFor(all(0), all(0)), 0, 1, 50, 0.999, 3, 1);
  const m = mergedSite(sites, 0, 1, wa, wb);
  check("fallback to area", near(m[0], 0.75 * 10 + 0.25 * 30, 1e-4) && near(m[2], 1.5, 1e-5), `x ${m[0]}, log τ ${m[2]}`);
}

// 4. Same true mean squared gradient, but b's moments were reset 10 steps ago
// so its raw v is still small: after bias correction the weights match.
{
  const beta2 = 0.999, t = 1000, g2 = 1e-4;
  const raw = (age) => g2 * (1 - beta2 ** age);
  const adam = adamFor(all(raw(t)), all(raw(10)), 0, t - 10);
  const pa = sitePrecision(adam, 0, t, beta2), pb = sitePrecision(adam, 1, t, beta2);
  check("bias correction", pa.every((v, k) => near(v / pb[k], 1, 1e-3)), `${pa} vs ${pb}`);
}

// Sites in a row, x = 10·i: [log τ, radius, RGB, dir, log aniso] per site.
function row(specs) {
  const out = new Float32Array(specs.length * SITE_FLOATS);
  specs.forEach(([tau, r, rgb, dir, aniso], i) => out.set([10 * i, 0, tau, r, ...rgb, ...dir, aniso], i * SITE_FLOATS));
  return out;
}
// A chain graph 0–1–2–… with equal counts.
function chain(n, count = 1) {
  const edges = new Map();
  for (let i = 0; i + 1 < n; i += 1) edges.set(i * n + i + 1, 1);
  return { count: new Float64Array(n).fill(count), edges };
}

// 5. Two candidate pixels: [0, 1, …] and [none, 2 (inactive), 1, 3, …];
// site 4 pinned. Edges 0–1 and 1–3; site 2 is skipped as inactive.
{
  const NONE = 0xffffffff;
  const cs = row([[0, 0, [0, 0, 0], [1, 0], 0], [0, 0, [0, 0, 0], [1, 0], 0], [0, 0, [0, 0, 0], [1, 0], 0],
    [0, 0, [0, 0, 0], [1, 0], 0], [0, 0, [0, 0, 0], [1, 0], 0]]);
  cs[2 * SITE_FLOATS] = -1;
  const cand = new Uint32Array([0, 1, 3, NONE, NONE, NONE, NONE, NONE, NONE, 2, 4, 1, 3, NONE, NONE, NONE]);
  const g = candidateEdges({ sites: cs, n: 5, cand, candWidth: 2, candHeight: 1, width: 4, height: 2, pinned: new Set([4]) });
  check("edges", g.edges.size === 2 && g.edges.get(0 * 5 + 1) === 1 && g.edges.get(1 * 5 + 3) === 1,
    `${[...g.edges]}`);
  check("owner counts", g.count[0] === 4 && g.count[1] === 4 && g.count[3] === 0, `${g.count}`);
}

// 6. Sites 0, 1 alike in all but position, 2, 3 different from them and each
// other: one merge joins 0 and 1, at their mean position.
{
  const s = row([[1, 2, [0.2, 0.2, 0.2], [1, 0], 0.5], [1, 2, [0.21, 0.2, 0.2], [1, 0], 0.5],
    [2, 4, [0.9, 0.1, 0.1], [0, 1], -0.5], [0, 1, [0.1, 0.9, 0.5], [1, 0], 1.0]]);
  const res = neighborMerge({ sites: s, n: 4, graph: chain(4), budget: 1 });
  const [m] = res.merges;
  check("alike pair merges", res.merges.length === 1 && Math.min(m.keep, m.drop) === 0 && Math.max(m.keep, m.drop) === 1,
    JSON.stringify(res.merges));
  check("merged position", near(res.sites[m.keep * SITE_FLOATS], 5, 1e-4), `x ${res.sites[m.keep * SITE_FLOATS]}`);
  check("input untouched", s[0] === 0 && s[SITE_FLOATS] === 10, "sites modified");
}

// 7. 0, 1: same anisotropic shape, axes d and −d. 2, 3: isotropic, axes 90°
// apart. 1–2 differ in shape. The two zero-shape-distance pairs merge first.
{
  const s = row([[1, 2, [0.5, 0.5, 0.5], [1, 0], 0.8], [1, 2, [0.5, 0.5, 0.5], [-1, 0], 0.8],
    [1, 2, [0.5, 0.5, 0.5], [1, 0], 0], [1, 2, [0.5, 0.5, 0.5], [0, 1], 0]]);
  const res = neighborMerge({ sites: s, n: 4, graph: chain(4), budget: 2 });
  const pairs = res.merges.map((m) => [m.keep, m.drop].sort().join("-")).sort();
  check("shape invariances", pairs.join(",") === "0-1,2-3" && res.merges.every((m) => m.weight < 1e-6),
    JSON.stringify(res.merges));
}

// 8. Ten alike sites in a chain, budget 9: chaining leaves one.
{
  const s = row(Array.from({ length: 10 }, (_, i) => [1, 2, [0.5 + 0.01 * i, 0.5, 0.5], [1, 0], 0]));
  const res = neighborMerge({ sites: s, n: 10, graph: chain(10), budget: 9 });
  check("chained merges", res.merges.length === 9 && res.changed.length === 1 && res.dropped.length === 9,
    `${res.merges.length} merges, changed ${res.changed}`);
  check("chained mean", near(res.sites[res.changed[0] * SITE_FLOATS], 45, 1e-3),
    `x ${res.sites[res.changed[0] * SITE_FLOATS]}`);
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("merge: all checks passed");
