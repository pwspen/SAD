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
`colorspaces.mjs` tunes the six LR bases separately for each training color space and
compares held-out PSNR and codec bits.
`e1.mjs` tunes the six LR bases on MNIST and CIFAR-10, replaces them with one constant
× each unit, and scores everything on held-out images, within and across datasets.


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
