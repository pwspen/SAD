# Trainer

The WebGPU optimizer: the training loop, Adam schedule, GPU buffers and kernel dispatch.
It has no DOM dependencies, so the browser pages in `web/` and the Node runner in `node/`
both import it. Hyperparameter defaults come from `training_config.json`.

The site count goes from `startSites` (or a loaded sites file's count) to `endSites` by
`countEnd` × `iters`, on the `countSchedule` curve, then holds; these are the only settings that
set it. Every `growEvery` iterations the count moves to the curve's value (`countTarget`):

- `"linear"`: the same number of sites per step.
- `"exponential"`: the same factor per step.
- `"sawtooth"` (default): the exponential curve with `sawTeeth` teeth on top, each rising to
  (1 + `sawAmp`) × the curve over `sawRatio` / (1 + `sawRatio`) of the tooth, then merging back.
  Swept against exponential (both tuned) and checked on held-out images, it won by about 2 dB on
  MNIST (2 teeth, amplitude 2, 1:1, by 10%: 21.88 vs 19.85 dB for the old exponential preset) and
  0.7 dB on Kodak at 256 px (6 teeth, amplitude 2, 3:1, by 50%, a step every 5: 25.81 vs 25.08);
  these are the presets, and Kodak's are the defaults. Teeth shorter than a few `growEvery` steps
  never go down (`countPlan` shows it). A step merges at most half the sites, so a big amplitude
  may not be back down by `countEnd`; merging then goes on until the count is at `endSites`.

Start < end grows. Each batch splits the cells whose color structure a second site would explain
best (the variance explained by a linear color ramp across the cell): the site moves to one half, a
new site takes the other. Only splits whose gain beats `growLambda × BITS_PER_SITE` qualify (0 = any
positive gain), so λ can shrink a batch or end the schedule early.

Start > end merges: each step merges the cheapest neighbouring pairs (Ward's criterion on cells at
their mean colors, `merge.js`), each site in at most one merge per step, into the pixel-weighted
average of both sites' parameters (anisotropy axes as doubled angles). On Kodak at 256 px, 1600 →
400 beat a fixed 400 (28.79 vs 28.06 dB); on MNIST, 128 → 16 was still behind a fixed 16 (L1 6.97
vs 5.32 levels). Start = end is a fixed count.

`layer: "learn"` adds a histogram layer: a flat site behind all the others, colored by the peak of
the target's color histogram smoothed over ±16 levels (`colorPeak`), with a learned depth β (its
constant logit), starting from log(p / (1 − p)) of the peak's mass p. A common color starts in
front of the sites' edges and takes the background; a rare one starts behind and the gradient
decides. Every pixel blends it besides its 8 listed candidates (`Params.globalCand`), so it never
takes a list slot; it never splits or merges, counts against `endSites`, and is an ordinary site
in the saved file (`# Global site: k`). A per-site bias from p(color) would not do this: logits
are −τ(d − r), so a constant is exactly a radius change. On 40 held-out MNIST digits (5 repeats,
paired against no layer, ± standard error over digits) it gains 0.78 ± 0.28 dB at 8 sites and
3.06 ± 0.41 at 32 (the MNIST preset); on Kodak at 256 px it stays mostly behind (β ≈ −4.7, ~4%
of pixels) and costs 0.13 ± 0.03 dB.

There is no LR warmup. `lrSchedule` is `"cosine"` (1 to 0 over `iters`), `"fixed"` (default, constant) or
`"decay-after-growth"` (constant until growth ends, then cosine to 0 by the end of the run).

## Modules

- `trainer.js` - training loop, site-count schedule (growth by splitting, merging) and epilogue.
- `candidates.js` - per-pixel candidate lists (`CandidateLists`), shared with the viewer.
- `merge.js` - neighbouring-cell merges and the CPU copy of the site logit.
- `shader_loader.js` - fetches and assembles the WGSL in `shaders/`.
- `device.js` - adapter/device setup.
- `params.js` - uniform buffer writers and `packJumpStep`.
- `encoders.js` - one class per GPU kernel (pipelines cached per device).
- `buffers.js` - site/Adam/grad buffer factories.
- `textures.js` - target / mask / candidate texture helpers + image upload.
- `color_spaces.js` - site color parameterizations.
- `io.js` - sites `.txt` parsing + serialization + PNG download.

## Storage-buffer limits

The kernels bind at most 4 storage buffers per shader stage, within WebGPU's default of 8; the trainer still requests the adapter's maximum.
