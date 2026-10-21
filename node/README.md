# Node Runner

The trainer in `trainer/` runs under Node via Dawn's `webgpu` bindings,
so parameter sweeps don't need a browser:

```bash
cd node
npm install
node sweep.mjs --image /path/to/image.png --set startSites=1 --set endSites=500,1000,2000 --set lrPosBase=0.025,0.05
node sweep.mjs --image /path/to/image.png --grid grid.json --repeat 3 --save-dir results/sweep
```

- The site count goes from `startSites` to `endSites` by `countEnd` × `iters` on the
  `countSchedule` curve (`linear`, `exponential` or `sawtooth`), stepping every `growEvery`
  iterations: growth when start < end, merging when start > end, fixed when equal. λ can end
  growth early. Each run reports where and why the schedule ended (`growStop`, `growDoneIter`
  in the JSON). Note: `--set` axes combine as a grid, so for fixed counts use `--grid` with
  explicit `{"startSites": n, "endSites": n}` runs.
- `--set key=v1,v2` sweeps a trainer arg (camelCase, as in `buildArgs` in `trainer/trainer.js`) or a
  `training_config.json` key (UPPER_CASE). Axes combine as a cartesian grid.
  `--grid` takes the same as JSON, or an explicit list of runs.
- Each run appends one JSON line (args, final/best PSNR, it/s, PSNR + gradient
  curve) to `--out` (default `results/sweep_<time>.jsonl`).
- Only PNG input is supported in Node. Images above `MAX_DIM` are area-downscaled,
  which differs slightly from the browser's resampling.
- Runs aren't bit-reproducible (GPU atomics sum in varying order); expect a few
  hundredths of a dB between repeats, and use `--repeat` when comparing close settings.
- `--jobs N` runs N trainers at once on one device. On an M4 at 768×512 the GPU is
  already saturated by one run, so total throughput stays the same. Parallelism
  only pays off across GPUs or machines.

## Batched training

`batch.mjs` trains many images with one fixed site count as a single run
(`trainer/batch.js`: one tile per image in a shared atlas), and writes one JSON line
per image with its final PSNR and sites in its own pixel coordinates:

```bash
node batch.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1000 \
  --preset ../presets/mnist.json --set startSites=8 --set endSites=8
node batch.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 --preset ../presets/mnist.json
node batch.mjs --images ../../comp/data/kodak --preset ../presets/kodak.json \
  --set startSites=1000 --set endSites=1000 --compare 2
```

- The count must be fixed (`startSites` = `endSites`, not sawtooth, no turnover).
  Images of one size share an atlas; others get their own.
- `--compare <jobs>` also trains every image alone (`<jobs>` at once) and prints both
  runs' speed and the PSNR difference. On an M4 with the MNIST preset at 8 sites,
  256 digits: 9.4× faster than 8 separate trainers, −0.18 dB mean (per-image noise
  ±6 dB). Kodak at 1000 sites: no speedup (one image already fills the GPU), −0.11 dB.

## Auto-tune and diagnostics (batch mode)

`tune.mjs` is train.html's auto-tune without the browser: each evaluation is one batched
run over the same images, scored by their median PSNR. One setting at a time, in log
space, it tries × and ÷ the step, plus the peak of a parabola through those three
medians; it moves to the best point, and when the center stays best the step decays
(r ← √r) instead of the search stopping, down to `--min-ratio`. Each setting keeps its
step across `--cycles`. Afterwards the start and tuned settings run on held-out images.

```bash
node tune.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 --preset ../presets/mnist.json
node tune.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 --preset ../presets/mnist.json --keys lrMult
```

`probe.mjs` runs one batch and reports three diagnostics for putting the LR groups in
common units (header comment for details): E1, candidate units per LR group (the codec
quantizer step Δ, the median 1/√F of the Gauss–Newton diagonal, the spread σ of the
values) and how well each explains the LRs; E2, how much of the per-site gradient a
per-value-bin mean explains, early and late in the run; E3, per-parameter entropy at the
codec steps against the total correlation a joint model could remove.
It also audits the codec: color bits per site under static RGB, YCoCg-R and joint
models, against the codec itself per image and over every image's sites in one stream.
`shared_codec.mjs` fits the codec's dataset-level shared model (`buildSharedModel` in
`trainer/codec.js`: steps, color basis, plane constants and primed model frequencies) on
images 0–1023 and reports bits per site by stage on held-out images 1024–3071: each image
coded alone, with the shared model, and pooled into one stream. MNIST / CIFAR-10 presets,
held out: 76.4 / 76.4 bits/site per image, 36.1 / 36.6 shared, 36.4 / 37.1 pooled.
`colorspaces.mjs` tunes the six LR bases separately for each training color space and
compares held-out PSNR and codec bits.
`e1.mjs` tunes the six LR bases on MNIST and CIFAR-10, replaces them with one constant
× each unit, and scores everything on held-out images, within and across datasets.

## Rate term

`rateLambda` > 0 trains on squared error + λ × estimated coded bits (`trainer/rate.js`,
`shaders/wgsl/rate.wgsl`): per coded plane a Gaussian mixture refit to the sites every
`rateEvery` iterations, at the codec's quantizer steps for `rateLossDb`, colors along the
KLT of the target. `rate.mjs` checks the estimate against the real codec per plane and
sweeps λ against the site count by median PSNR vs measured bits per pixel.

## LR bases for a new dataset (units_init.mjs)

`units_init.mjs` sets the six LR bases from one probe batch instead of tuning them: every
group's effective rate (base × `lrMult`) becomes K · Δ_g, Δ_g the codec quantizer step of
that group measured on the new data (`probe.mjs`' E1 unit), K one constant. `--tune-mult`
then tunes `lrMult` alone (one free number instead of six); `--holdout` scores the start
preset, K · Δ, K · Δ with `lrMult` tuned and any `--compare` presets on the images after the
probe ones. `--fit` / `--tuned` go the other way: from LRs already tuned on a dataset, K₀ =
geometric mean of lr / Δ, then `lrMult` tuned on K₀ · Δ gives K. The result JSON holds the
settings as a preset (`presetJson`).

```bash
node units_init.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
  --preset ../presets/mnist.json --tune-mult --holdout 2048 --compare ../presets/cifar10.json
node units_init.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 \
  --tuned ../presets/cifar10.json --preset ../presets/mnist.json        # K for CIFAR-10
```

K depends on the settings Δ is probed at (the probe run's final error and Gauss–Newton
diagonal), so fit and apply with the same probe preset. With Δ probed at
`presets/mnist.json` on test images 0–1023: K = 0.02046 on MNIST, 0.01126 on CIFAR-10; the
default `--k` is their geometric mean, 0.01518. Validation, each dataset with the other's K,
median PSNR (dB) on held-out test images 1024–3071, median of 3 repeats:

| target | probed at | K from | other dataset's tuned preset | K · Δ | K · Δ, lrMult tuned | fully tuned preset |
|---|---|---|---|---|---|---|
| CIFAR-10 | mnist.json | MNIST (0.02046) | 21.009 (mnist.json) | 20.492 | 20.976 (lrMult 4.14) | 21.330 (cifar10.json) |
| MNIST | mnist.json | CIFAR-10 (0.01126) | 14.747 (cifar10.json) | 15.944 | 16.317 (lrMult 10.4) | 17.527 (mnist.json) |
| MNIST | cifar10.json | CIFAR-10 (0.01385, Δ at cifar10.json) | 14.698 (cifar10.json) | 14.801 | 14.899 | 17.546 (mnist.json) |

- K · Δ is a starting point, not a substitute for tuning: with `lrMult` tuned it matched the
  MNIST-tuned bases on CIFAR-10 (−0.03 dB) and beat CIFAR's on MNIST by 1.6 dB, but stayed
  0.35 / 1.2 dB under the fully tuned presets. Untuned (K from the other dataset) it lost
  0.5 dB on CIFAR: K is only good to within about ×1.6 (the tuned `lrMult` moved it that
  much), so add `--tune-mult`.
- Probe at settings that already suit the data roughly. The last row probes MNIST at
  CIFAR's preset (2.8 dB under MNIST's own): Δ came out poor (Δ_dir 0.94 vs 5.4 at
  mnist.json, and Δ varied up to ±50% between repeats), and K · Δ gained only 0.2 dB.
  `--passes` (re-probing at the new bases) did not fix it.
- E1 (e1.mjs: K · Δ within −0.05 / −0.12 dB of each dataset's own K) probed both datasets
  at the same, then-untuned MNIST preset, which suited both moderately; the table agrees
  with it once Δ is probed at the same settings on both.


## Parameter-space diagnostics (E7, E8)

`lossmap.mjs` (E7) trains one batch and asks whether a loss map over parameter space
predicts what restricting the space costs. Each pixel's squared error goes to its owner
site; the sites are binned on chroma × luma, log τ × radius and |log aniso| × radius.
Every bin is then snapped into a restricted space (grey colors, log τ to the median or
clamped to its interquartile range, radius to the median, log aniso to 0) and the ΔPSNR
measured by render (trained and rebuilt candidate lists) and after a short projected
refit, against the Gauss–Newton prediction Σ F·δ² (and with the first-order term g·δ):

```bash
node lossmap.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --count 1024 --preset ../presets/mnist.json
node lossmap.mjs --parquet ../../comp/data/cifar10/test.parquet --count 1024 --preset ../presets/cifar10.json
```

On 1024 test images per dataset (5 × 5 bins, ~115–121 regions): Σ F·δ² ranks the measured
render cost well (Spearman 0.87 MNIST, 0.89 CIFAR, pooled; 0.80 / 0.77 after a 40-step
refit) but overstates it for the large geometric snaps (slope 0.14 / 0.54; refit 0.03 /
0.11), since the softmax saturates. On MNIST the sites' chroma is off-optimum, so greying
gains: there the first-order term is needed (g·δ ≈ −2F·δ², GN+g Spearman 0.92). The naive
loss map is confounded: a bin's SSE tracks its site count (Spearman 0.81 / 0.77) and ranks
the cost poorly (0.36 / 0.55); per site, SSE per pixel says nothing (−0.10 / 0.12) while F·δ²
per site does (0.87 / 0.86).

`clamps.mjs` (E8) lists every bound the trainer applies (log aniso ±5, position in the
tile, layer β ±8, the logchroma floor at −1/255; none on τ or radius), measures the share
of sites at each and their outward gradient at the end of the run, then extends each
bound by patching adam.wgsl's text in memory, retrains held-out images and compares the
ΔPSNR with the signal:

```bash
node clamps.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --from 1024 --count 2048 --preset ../presets/mnist.json
node clamps.mjs --parquet ../../comp/data/cifar10/test.parquet --from 1024 --count 2048 --preset ../presets/cifar10.json
```

With the presets (100 iterations) no stock bound binds much: log aniso never reaches ±5
(it learns from iteration 50), β ±8 holds 0–0.3% of layers, and only position sits at the
tile edge (1% MNIST, 8.8% CIFAR), where extending it by 4 px costs 0.026 ± 0.008 dB on
CIFAR (MNIST within noise). `--base aniso=0.5,posMargin=-2,beta=1` tightens the bounds so
they bind (21–52% of sites at them); restoring each to stock, the Newton-gain signal ranks
the gains correctly on both datasets (aniso +0.23 / +0.30 dB, position +0.01 / +0.07,
β +0.03 / +0.02), share × outward gradient on MNIST only.

## Structure search priced in bits (E9)

`structure.mjs` runs a greedy search over discrete structure decisions on J = SSE + λ · bits
(λ in rateLambda units; SSE of the decoded, codec-quantized sites; bits from the real codec,
every image's sites in one stream). Actions (pluggable `ACTIONS`): `drop` (per image, merge the
pair predicted cheapest, in every image where that pays), `coarsen:<g>` / `refine:<g>` (one
quantizer step group ×2 / ÷2), `restrict:<r>` (iso: log aniso → 0 and no θ; grey; log τ or
radius to the median; held by projected refits through the log hook, no trainer changes) and
`lift:<r>`. Each is priced by Gauss–Newton + g·δ (calibrated per action on images 0–1023) and
the codec on the previewed state; each step refits the best action and a control (40
iterations), and keeps the action if it beats the control's J. Baselines are refit as often as
the greedy accepted actions (k), since a 40-step refit alone gains 0.4–0.8 dB at 100 iterations.

```bash
node structure.mjs --mnist ../../comp/data/mnist/t10k-images-idx3-ubyte.gz --preset ../presets/mnist.json --lambdas 0.1,0.3,1 --sites 3,4,6,8,12
node structure.mjs --parquet ../../comp/data/cifar10/test.parquet --preset ../presets/cifar10.json --lambdas 0.1,0.3,1 --sites 3,4,6,8,12
```

Calibration (images 0–1023, per image; ρ = Spearman, slope through 0, pooled = Σ measured / Σ
predicted after a 40-step refit; ΔR predicted on the snapped state vs measured after the refit):

| action | MNIST ρ refit | MNIST pooled | MNIST ΔR slope | CIFAR ρ refit | CIFAR pooled | CIFAR ΔR slope |
|---|---|---|---|---|---|---|
| drop (merge) | 0.55 | 0.14 | 0.98 | 0.62 | 0.13 | 0.97 |
| coarsen pos / logτ / radius | 0.15 / 0.11 / 0.06 | 0.39 / 0.28 / 0.43 (render) | 0.95 / 0.83 / 0.79 | 0.60 / 0.33 / 0.30 | 0.68 / 1.01 / 0.67 | 0.92 / 0.84 / 0.92 |
| coarsen color / log aniso / θ | 0.57 / 0.27 / 0.23 | 1.00 / 0.45 / 0.31 | 0.80 / 0.97 / 1.00 | 0.72 / 0.59 / 0.44 | 1.02 / 0.83 / 0.35 | 0.93 / 0.97 / 0.98 |
| restrict iso | 0.57 | 0.075 | 1.00 | 0.80 | 0.21 | 0.99 |
| restrict τ / radius | 0.27 / 0.43 | 0.038 / 0.028 | 1.01 / 1.00 | 0.71 / 0.63 | 0.059 / 0.093 | 1.00 / 0.96 |
| restrict grey | — (MNIST sites already grey: ΔD = ΔR = 0) | | | 0.94 | 0.90 | 0.97 |

The codec on the previewed state predicts ΔR almost exactly (greedy steps: ρ 0.99–1.00, slope
0.97–1.00). Raw GN+g overstates ΔD after a refit by 13–35× for restrictions, ~4× for drops,
~3× for most quantizer steps (color is exact: the render is linear in color). Calibrated, the
greedy steps came out at slope (measured / calibrated, held out, 3 repeats) restrict 1.05 / 0.68,
coarsen 0.80 / 1.08, drop 1.80 / 1.65 (MNIST / CIFAR): once the cheapest merges are taken the
rest cost more than the all-images calibration says, so about half the drop attempts on MNIST
were rejected.

Held out (images 1024–3071, median of 3 repeats; dB = median per-image PSNR of the decoded
sites, J per image; "+k" = refit k times like the greedy; sweep = λ 0 at 3–12 sites + the
preset's 10, frontier PSNR linear in bpp):

| data | λ | greedy dB / bpp / J | rate term +k | sweep best J (+k) | sweep frontier dB at greedy bpp | accepted actions (repeat 1) |
|---|---|---|---|---|---|---|
| MNIST | 0.1 | 18.62 / 0.315 / 59.45 | 18.21 / 0.321 / 63.06 | 60.26 | 18.43 | drop, coarsen logτ pos θ log aniso, drop, coarsen radius |
| MNIST | 0.3 | 16.65 / 0.211 / 102.6 | 16.89 / 0.245 / 108.8 | 96.50 | 17.04 | iso, drop, pos×2, drop, logτ, radius, θ |
| MNIST | 1 | 13.59 / 0.067 / 158.0 | 15.44 / 0.177 / 208.0 | 155.4 | below 3 sites (2 sites: J 175) | iso, τ, radius, pos×3, color×2, drop×6–8, θ |
| CIFAR-10 | 0.1 | 20.61 / 0.216 / 52.00 | 20.57 / 0.203 / 50.27 | 54.19 | 20.30 | drop×8–9, coarsen θ (radius, log aniso) |
| CIFAR-10 | 0.3 | 18.26 / 0.084 / 74.79 | 19.10 / 0.133 / 81.23 | 82.63 | 17.82 | iso, radius, drop×6, pos, color |
| CIFAR-10 | 1 | 16.69 / 0.035 / 106.8 | 18.13 / 0.109 / 162.7 | 127.5 (2 sites, +0: 123.2) | below 3 sites | iso, τ, radius, color×2, pos×2, drop×6, logτ×2 |

The greedy beats the rate term at the same λ (both refit) everywhere but CIFAR λ 0.1 (+1.7 J,
3%), and the site-count frontier on CIFAR at λ ≥ 0.3 (−6.6 / −20.7 J; +0.44 dB at its bpp) and
on MNIST at λ 0.1 (−0.55 J, +0.19 dB). It loses to the sweep on MNIST at λ 0.3 / 1 (+6.1 / +2.4
J): one merge per image and step, then a refit, is worse than training fewer sites from the
start. What it picks: quantizer steps first at small λ (pos, log τ, θ, log aniso, radius all
×2; refining never paid), then iso (all sites round, which also drops θ), τ and radius
constant at larger λ; on CIFAR mostly site drops. Caveats: 100-iteration presets with
INIT_LOG_TAU 5 / INIT_RADIUS 4 (untuned; the start is far from converged, so which τ / radius
restrictions pay may shift with a tuned init, the calibration findings less so); direction and
anisotropy don't train in a 40-step refit (iso is one-way); calibration is fit at the start
state and drifts as the greedy goes.
