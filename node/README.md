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

`profile.mjs` takes the same dataset options (plus `--alone`) and breaks a run's time down:
wall clock by phase (prepare / run / finalize), the share spent on log ticks, and GPU time
per kernel from timestamp queries on every compute pass.

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

`--method cma` searches all `--keys` at once with CMA-ES instead: each generation samples
λ points (4 + ⌊3 ln n⌋ by default) from a Gaussian in log space, moves its mean to the
weighted best half, and adapts the covariance toward the directions that paid, so
settings that trade off against each other (lrMult against the bases) move together.
The mean is scored once per generation and is what it returns; the best single sample is
reported alongside. `--budget` caps either method at that many batches.

`tunecmp.mjs` runs both methods from the same preset at the same `--budget`, `--repeats`
times (cma with a new seed each time), and scores each one's incumbent on held-out images
every `--every` batches, so the two compare as held-out gain per tuning batch.

Held-out gain (median PSNR over 2048 images after the 1024 tuning ones) vs tuning
batches, mean (min…max) over 3 repeats, defaults (coord ×2 → ×1.1; cma σ 0.5):

| batches | CIFAR, 6 LR bases: coord | cma | MNIST, 9 keys: coord | cma |
|---|---|---|---|---|
| 20 | +0.18 (0.13…0.21) | +0.16 (0.04…0.27) | +0.03 (−0.00…0.08) | −0.30 (−0.42…−0.11) |
| 60 | +0.29 (0.27…0.31) | +0.31 (0.30…0.33) | +0.04 (0.01…0.07) | −0.17 (−0.32…0.05) |
| 100 | +0.29 (done) | +0.31 (0.30…0.32) | +0.13 (0.01…0.33) (done) | +0.24 (0.13…0.35) |
| 240 | +0.29 | +0.36 (0.34…0.38) | +0.13 | +0.39 (0.32…0.51) |

CIFAR-10 starts from presets/mnist.json and tunes the six LR bases; MNIST starts from
its own preset and tunes those plus lrMult, INIT_LOG_TAU and INIT_RADIUS. The coordinate
search stops on its own once every step is below ×1.1 (55–62 batches on CIFAR, 69–87 on
MNIST), so past that it is flat. Up to there the two are even on CIFAR; CMA-ES keeps
going and ends +0.07 dB higher. With 9 keys, CMA-ES's first generations (σ 0.5 on every
key at once, the init values included) land below the start for ~60 batches, then it
passes the coordinate search, which on MNIST gains +0.28 dB on the tuning images but
only +0.13 held out (its point is the single best measurement, so noise flatters it;
the same preset scores 24.78–24.84 dB across repeats). The best sample tracks CMA-ES's
mean within noise. `results/tunecmp/{cifar_lr,mnist_9keys}.{json,log}`.

The MNIST run again with `--sigma 0.25` (±1σ ≈ ×1.28; coord rerun alongside):

| batches | coord | cma σ 0.25 |
|---|---|---|
| 20 | +0.05 (−0.02…0.12) | +0.04 (−0.06…0.16) |
| 40 | +0.05 (0.00…0.10) | +0.18 (0.02…0.29) |
| 60 | +0.09 (0.07…0.10) | +0.23 (0.08…0.33) |
| 100 | +0.20 (0.11…0.25) (done) | +0.39 (0.27…0.51) |
| 240 | +0.20 | +0.44 (0.30…0.62) |

No dip: CMA-ES is level with the coordinate search at 20 batches and ahead from 40 on,
where at σ 0.5 it was still −0.13 dB. It also ends higher, though one repeat stopped at
210 batches (step below ×1.1). The coordinate search's +0.20 vs +0.13 before is the
repeat-to-repeat spread. `results/tunecmp/mnist_9keys_s025.{json,log}`.

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

## Color dims (colordims.mjs)

`colorDims` k (klt only, 1–3, default 3) keeps the first k principal axes of the KLT basis:
rgb = mean + Σ_{j<k} axis_j · c_j. The dropped coordinates are held at 0 (a mask compiled
into the WGSL next to the basis: encode zeroes them, so init colors are projected; decode
ignores them; their gradient is 0, so Adam never moves them), and the codec does not code
their planes at all when its `colorBasis` carries the same `dims`. `colordims.mjs` draws
rate–distortion curves per color configuration over the site count: median PSNR vs bits
per pixel measured with the codec (steps at `rateLossDb`, every image's sites pooled into
one stream, as rate.mjs), the Pareto frontier, and each point's PSNR against a reference
curve at equal bpp:

```bash
node colordims.mjs --parquet ../../comp/data/cifar10/test.parquet --from 1024 --count 2048 \
  --preset ../presets/cifar10.json --configs klt:1:0.01465,klt:2:0.01625,klt:3:0.01784 --sites 2,3,4,6,8,10,12,14 --repeats 3
```

Held-out images 1024–3071, median of 3 repeats (repeat spread ≤ 0.05 dB), presets'
settings with colorSpace klt; lrColorBase tuned per k on CIFAR images 0–1023 (tune.mjs
`--keys lrColorBase`: k = 1 / 2 / 3 → 0.01465 / 0.01625 / 0.01784, from 0.008919, each
+0.08–0.10 dB on the tuning images). Δ = PSNR minus the klt k = 3 curve's at the same bpp
(sites count the layer, which is not coded).

**MNIST** (presets/mnist.json): k is free and buys nothing. The targets are grey, so
under k = 3 the two chroma coordinates already get no gradient and code at ~0.01 bits/site
each; k = 1, 2, 3 agree within noise at every count (|Δ| ≤ 0.045 dB, 4–12 sites).

| sites | k = 1 dB / bpp | k = 2 dB / bpp | k = 3 dB / bpp |
|---|---|---|---|
| 4 | 15.238 / 0.1219 | 15.255 / 0.1219 | 15.242 / 0.1218 |
| 8 | 17.620 / 0.3230 | 17.602 / 0.3217 | 17.620 / 0.3219 |
| 12 | 18.776 / 0.5240 | 18.800 / 0.5243 | 18.821 / 0.5242 |

**CIFAR-10** (presets/cifar10.json): k < 3 is never on the frontier; the bits are better
spent on sites. The third axis costs 0.4–1.4 bits/site (more sites, more bits) and the
second 1.3–2.7, less than the ~1.7 / ~4.7 the variances suggested.

| sites | k = 1 dB / bpp (Δ) | k = 2 dB / bpp (Δ) | klt k = 3 dB / bpp | ycocg dB / bpp (Δ) |
|---|---|---|---|---|
| 2 | 15.481 / 0.0209 | 16.062 / 0.0224 | 16.164 / 0.0227 | |
| 3 | 16.481 / 0.0487 (−0.55) | 17.136 / 0.0543 (−0.08) | 17.236 / 0.0548 | |
| 4 | 17.238 / 0.0778 (−0.70) | 18.096 / 0.0849 (−0.06) | 18.293 / 0.0895 | 18.321 / 0.0888 (+0.05) |
| 6 | 18.348 / 0.1378 (−1.03) | 19.505 / 0.1539 (−0.17) | 19.787 / 0.1599 | 19.828 / 0.1596 (+0.05) |
| 8 | 18.929 / 0.1971 (−1.31) | 20.277 / 0.2240 (−0.28) | 20.726 / 0.2377 | 20.722 / 0.2375 (−0.00) |
| 10 | 19.422 / 0.2576 (−1.51) | 21.063 / 0.2963 (−0.26) | 21.493 / 0.3128 | 21.561 / 0.3125 (+0.07) |
| 12 | 19.734 / 0.3168 (−1.79) | 21.431 / 0.3695 (−0.45) | 22.006 / 0.3880 | 22.003 / 0.3933 (−0.03) |
| 14 | 20.055 / 0.3807 (−1.90) | 21.870 / 0.4439 (−0.43) | 22.458 / 0.4756 | 22.467 / 0.4748 (+0.01) |

- The frontier is k = 3 (klt and ycocg tie within noise, ycocg +0.03 dB on average). k = 2
  sits below it everywhere (−0.06 to −0.45 dB, the gap growing with the count) and k = 1
  far below (−0.55 to −1.9 dB). k < 3 points are only non-dominated in the gaps between
  integer site counts, never above the interpolated k = 3 curve. At 2 sites (one coded
  site) the k = 3 slope is ~33 dB/bpp, so k = 2's 0.0003 bpp saving is worth ~0.01 dB
  against its −0.10 dB.
- With the rate term on (`rateLambda` 0.03, 3–14 sites) the picture is the same: k = 2
  −0.15 to −0.38 dB, k = 1 −0.62 to −1.63 dB at equal bpp; the rate term already shrinks
  the third axis to 0.6–0.9 bits/site at k = 3.
- Batches share one basis per atlas (here all 2048 images, so a dataset KLT). With a basis
  per image (`--per-image`: each image trained and coded alone, 64 images, 8 sites, one
  repeat) k = 2 loses nothing (20.743 vs 20.726 dB) and k = 1 only 0.23 dB (vs 1.96 dB with
  the shared basis on the same images). At the pooled stream's ~3.7 bits/site for the two
  dropped planes that is roughly break-even with more sites before paying for the
  per-image basis (a line: ~4–5 numbers), which the shared basis needs none of; worth a
  look only with a cheap basis code.

`warp.mjs` (3a) adds a shared monotone warp per coded plane, φ(v) = v + Σ a_k b_k(v)
(Gaussian bumps), applied to every site of the atlas and stepped from the log hook on the
kernel-smoothed histogram of per-site gradients of squared error + λ · bits (no trainer
changes); `binarize.mjs` (3c) measures the MNIST luma (K0) distribution across λ and
projects K0 onto two levels. Both score median PSNR vs real-codec bpp on held-out images
1024–3071 (header comments for the method). Findings, 100 iterations, 10 sites:

- The presets' `INIT_LOG_TAU` 5 / `INIT_RADIUS` 4 were never tuned. tune.mjs on images
  0–1023: MNIST 2.862 / 1.981, 18.44 → 24.18 dB; CIFAR-10 2.822 / 4, 21.48 → 21.63 dB.
  The runs below use those (MNIST also with its LR bases retuned at that init where noted).
- Warp (lr 0.3 tuned on 0–1023, 16 bumps, every iteration), ΔPSNR at matched bpp against
  the rate-only frontier, 3 repeats: MNIST +0.9–1.3 dB over 0.21–0.41 bpp (LRs retuned;
  +1.4–1.5 at init-only), CIFAR-10 +0.04–0.20 dB (mean 0.10). Most of it is an optimizer
  effect, not a rate one: at λ = 0 the warp already gains +1.2 dB MNIST / +0.11 CIFAR, and at
  λ ≥ 1 it gains nothing. One bump (a per-plane shift) loses. Driven by the squared error
  alone (rate term still on per site) the warp undoes the rate term's clustering and is
  dominated by the λ · bits-driven warp at every bpp.
- Binarization (tuned init): as λ grows the K0 mixture collapses onto black and white
  (sites within 0.1 of either: 55% at λ 0, 83–95% at 0.1, 99% at 0.3); projecting K0 onto
  {0, 1} from mid-run costs no PSNR anywhere from λ 0 to 0.3 (+0.03…+0.23 dB at the same
  λ, −0.002…−0.04 bpp; K0 3.3 → 0.6 codec bits/site at λ 0) and is neutral at λ 1, where the
  free run is already near-binary. At the stock τ init the sites keep a grey mode (~35%)
  and snapping costs ~2 dB.

`quantNoise` > 0 trains quantization-aware (`shaders/wgsl/qnoise.wgsl`): each gradient pass
sees the sites perturbed by uniform noise U(−Δ/2, Δ/2) × `quantNoise` per coded plane, Δ the
rate model's codec steps at `rateLossDb` (refit every `rateEvery` iterations, with or without
the rate term): positions on their grid step, log τ, radius and log aniso on theirs, colors
along the codec's basis axes, θ by each site's level step π / m. The clean sites are copied
aside and back around the gradient pass, so Adam updates the clean values (`qnoise_test.mjs`).
`quantNoiseStart` delays it (fraction of `iters`): on images 0–1023 noise from iteration 0
cost up to 1 dB clean (the early steps, from a large error, are coarse); 0.7 was best.

`qat.mjs` compares, by median PSNR of the decoded sites (coded as one stream at codecSteps'
steps for a loss budget, rendered with rebuilt candidate lists) against measured codec bpp:
(a) post-hoc quantization at several budgets, (b) noise at each budget's steps, (c) the rate
term at each budget, (d) both. Held-out images 1024–3071, presets as shipped (INIT_LOG_TAU 5,
INIT_RADIUS 4), `quantNoise` 1 from 0.7, λ 0.01 / 0.03 / 0.1, budgets 0.25–4 dB, median of 3
repeats (repeat sd ≈ 0.015 dB). ΔPSNR of rate + noise over the rate-only curve of the same λ, at
equal bpp (* extrapolated):

| dataset | λ | 0.25 dB | 0.5 dB | 1 dB | 2 dB | 4 dB |
|---|---|---|---|---|---|---|
| MNIST | 0.01 | +0.03 | +0.08 | +0.37 | +0.20 | +0.18* |
| MNIST | 0.03 | −0.10 | +0.13 | +0.41 | +0.24 | +0.18* |
| MNIST | 0.1 | −0.18 | +0.02 | +0.28 | +0.11 | −0.00* |
| CIFAR-10 | 0.01 | −0.04 | −0.01 | +0.09 | +0.32 | +0.68* |
| CIFAR-10 | 0.03 | −0.04 | −0.03 | +0.11 | +0.43 | +1.02* |
| CIFAR-10 | 0.1 | −0.10 | −0.03 | +0.16 | +0.43 | +0.82* |

- Noise pays at coarse budgets (1–2 dB of quantization loss, ≲ 0.3 bpp MNIST, ≲ 0.24 bpp CIFAR)
  and costs a little at fine ones (0.25 dB). On the frontier over every λ and budget, rate +
  noise beats rate alone by +0.2–0.3 dB on MNIST at 0.28–0.33 bpp and +0.07 above 0.36; on
  CIFAR it is within ±0.15 dB from 0.18 bpp up, ahead only below (sparse points).
- Noise alone beats post-hoc quantization on MNIST (+0.07 to +0.35 dB) and is even on CIFAR
  above 0.25 bpp; the rate term matters more on CIFAR (λ 0.1: +0.6 to +1.4 dB over post-hoc).
- Init-dependent: with MNIST's tuned init (INIT_LOG_TAU 2.862, INIT_RADIUS 1.981; one repeat,
  λ 0.03, 0.5–2 dB) clean PSNR is 24.05 dB instead of 18.3 and noise no longer helps: noise
  alone −0.03 dB vs post-hoc, rate + noise −0.02 / 0.00 / ≈0 vs rate alone.

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

## Clustering with one peeked label (cluster.mjs)

Classification without training on labels: encode a labelled dataset into sites (batch
mode, a preset), cluster the converged sites into k = 10 groups, peek at one random
member's label per cluster, name every member by it, and score the rest against the
true labels (mean ± sd over 1000 draws; purity = the majority label per cluster, the
best any one-label-per-cluster rule can do). Clusterings: k-means on the raw pixels (a
reference), k-means on a 14 × 14 map of each image's sites (Gaussians, σ 2 px, weighted
by the first KLT color coordinate), and codelength clustering, the codec as the
classifier: a shared coding model per cluster, each image to the model that codes it in
the fewest bits, iterated from the site-map clusters. The loaders now return each
image's label (MNIST's labels file next to the images, the parquet `label` column).

2000 test images, presets, 10 sites (9 coded):

| | MNIST purity | MNIST one peek | CIFAR-10 purity | CIFAR-10 one peek |
|---|---|---|---|---|
| pixels (k-means) | 55.0% | 43.8 ± 6.0% | 21.3% | 13.0 ± 2.0% |
| site map (k-means) | 53.5–56.4% | 38.0–41.8% | 19.9% | 11.9 ± 1.9% |
| codelength | 37.8% | 22.7 ± 4.8% | 20.1% | 12.5 ± 2.0% |

The site map matches the pixels on MNIST (two runs, so ±3% from encoding and k-means
seeding); on CIFAR-10 every method is near chance, as k-means on pixels is known to be.
The codec does not classify: its shared model is factorized per plane (value histograms
and raster position gaps), with no model of where on the image which values sit, so the
clusters it settles into share value statistics, not shapes (325 → 322 bits/image).


## Gap to RECOMBINER on CIFAR-10 (qat.mjs)

Where SAD's ~2 dB gap to [RECOMBINER](https://arxiv.org/abs/2309.17182) (ICLR 2024; 23.59 dB at
0.297 bpp, its lowest CIFAR-10 point) comes from. `qat.mjs` arms a (post-hoc) and c (rate term,
λ 0.01 / 0.03 / 0.1) on held-out test images 1024–3071, presets/cifar10.json at 10 and 12 sites,
100 / 1000 / 3000 iterations (cosine LR stretched over the run, lrMult untuned past 100), budgets
0.25–4 dB; post-hoc 3 repeats (spread ≤ 0.07 dB), rate 1. Bits are the pooled stream (close to the
shared model: 36.4 vs 36.1 bits/site above). PSNR at a bpp is the upper hull over sites, arms and
budgets, interpolated; "clean" is the unquantized PSNR at the same point.

```bash
node qat.mjs --parquet ../../comp/data/cifar10/test.parquet --from 1024 --count 2048 \
  --preset ../presets/cifar10.json --set sites=10 --set iters=1000 --arms c --lambdas 0.01,0.03,0.1 --db 0.5,1,2
```

| frontier | 0.25 bpp | 0.297 bpp (clean) | 0.35 bpp | 0.45 bpp |
|---|---|---|---|---|
| 100 it, post-hoc | 20.33 | 20.97 (21.68) | 21.45 | — |
| 100 it, + rate term | 20.95 | 21.36 (22.03) | 21.74 | — |
| 1000 it, post-hoc | — | 20.72 (23.14) | 21.90 | 22.98 |
| 1000 it, + rate term | 21.44 | 22.03 (22.67) | 22.62 | 23.41 |
| 3000 it, post-hoc | — | 20.38 (23.28) | 21.71 | 22.97 |

Medians over images; the mean (`mean` in the rows, which RECOMBINER likely reports) is +0.22–0.25 dB
higher at the 0.297 bpp neighbours, so the best is ≈ 22.25 dB mean against 23.59.

- Training longer alone does not pay at matched rate: clean PSNR rises 1.46 / 1.64 dB (10 / 12 sites)
  from 100 to 1000 iterations and 0.14 more by 3000, but the converged sites cost ~11 more bits each
  at the same budget (30.8 → 41.7 bits/site at 1 dB, 10 sites), so post-hoc at 0.297 bpp falls
  (20.97 → 20.72 → 20.38). Priced by the rate term during training it does: +0.67 dB at 0.297.
- At 0.297 bpp, gap 2.62 dB (100 it, post-hoc, median) = rate term 0.39 + convergence 0.67 +
  remaining 1.56 (≈ 1.34 by means). Of the remainder, quantization loss at the frontier is 0.64 dB
  (clean − coded), so even lossless coding at the same bits leaves ≥ 0.7 dB: the representation and
  its lack of a learned prior, not rounding or iterations.
- Not run: the rate term at 3000 iterations (post-hoc clean gains only 0.14 past 1000), lrMult retuned
  for long runs.
