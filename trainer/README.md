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
positive gain), so λ can shrink a batch or end the schedule early. `growMode` picks how sites are
added: `"split"` (default, the split above) or `"insert"` (`insert.js`): new sites are placed one at a
time where they cut the training loss most, computed exactly from the current fit (candidate
positions, median colors), and existing sites are left alone; λ is then in the training loss's units.

Start > end prunes: each step merges the cheapest neighbouring pairs (Ward's criterion on cells at
their mean colors, `merge.js`), each site in at most one merge per step, into the pixel-weighted
average of both sites' parameters (anisotropy axes as doubled angles). Merges don't age the other
sites for `growLrDecay`. On Kodak at 256 px, 1600 → 400 beat a fixed 400 (28.79 vs 28.06 dB); on
MNIST, 128 → 16 was still behind a fixed 16 (L1 6.97 vs 5.32 levels). Start = end is a fixed count.

`growCriterion` picks what ends growth early: `"lambda"` (default, predicted gain against
`growLambda`) or `"mdl"`: at each growth step the cells are scored by description length (sites ×
`BITS_PER_SITE` plus the squared error of every cell at its mean color, coded as a Gaussian at
8-bit precision), and growth ends once a batch hasn't lowered it. Judged on realized gains, so it can stop early
where a new batch needs longer than `growEvery` to pay off (MNIST stops at ~2 sites).

`pruneEvery` (0 = off) merges neighbouring sites that don't pay for their bits, every that many
iterations once the count schedule is over (`merge.js`). Pairs are scored by Ward's criterion on cells at their
mean colors and merged cheapest first while the cost stays under a site's bits; the merged site is
the pixel-weighted average of every parameter (anisotropy axes as doubled angles). Each batch is
kept only if the render's description length went down by the next step, else undone and pruning
ends. On MNIST, growing to 64 and pruning ended at ~18 sites, better in sites and error than a
fixed 16; on Kodak at 256 px it kept all 800, as the description length says it should.

`keepBest` scores every log tick and ends the run on the best state seen, restored exactly
(sites and Adam moments, candidate lists rebuilt): `"loss"` (default, lowest training loss),
`"mdl"` (description length: sites × `BITS_PER_SITE` plus the residual coded at 8-bit precision
under the loss's likelihood, so fewer sites at equal loss win) or `"off"`. With `growPatience` > 0,
growth also ends, and the run returns to the best state, after that many growth steps without a
better score. Mid-growth states are far from converged, so `"mdl"` and patience pick early
half-fitted states with the current presets; see the comment on `KEEP_BEST` in `trainer.js`.

There is no LR warmup. `lrSchedule` is `"cosine"` (1 to 0 over `iters`), `"fixed"` (default, constant) or
`"decay-after-growth"` (constant until growth ends, then cosine to 0 by the end of the run).

## Modules

- `trainer.js` - training loop (fixed site count, or rate-distortion growth; no prune) and epilogue.
- `shader_loader.js` - fetches and assembles the WGSL in `shaders/`.
- `device.js` - adapter/device setup.
- `params.js` - parameter buffer writers, `pack_jump_step`, and `hilbert_bits_for_size`.
- `radix_sort.js` - radix-sort GPU pass orchestration.
- `encoders.js` - one class per GPU kernel.
- `buffers.js` - site/Adam/grad buffer factories.
- `textures.js` - target / mask / candidate texture helpers + image upload.
- `io.js` - sites `.txt` parsing + serialization + PNG download.

## Storage-buffer limits

`gradients_tiled.wgsl` and `adam.wgsl` each bind 12 storage buffers per shader stage. The WebGPU default is often lower (8-10), so the trainer requests the adapter's maximum. On Chrome/Mac this is normally 12+ on Metal; if your browser reports fewer, training will fail at pipeline creation. Upgrade the browser.
