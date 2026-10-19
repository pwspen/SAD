// Checks merge.js's parameter averaging (JS only, no GPU):
//   1. scalar weights: pixel-count average, as before precision weights
//   2. precision weights pull each parameter toward the site that cares
//   3. no precision on either site: falls back to the pixel counts
//   4. bias correction: a site whose moments are younger isn't underweighted
// Usage: node merge_test.mjs

import { mergedSite, mergePrecision, sitePrecision } from "../trainer/merge.js?v=20261019";
import { SITE_FLOATS, ADAM_FLOATS } from "../trainer/params.js?v=20261019";

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

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("merge: all checks passed");
