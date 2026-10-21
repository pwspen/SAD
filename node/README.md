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

