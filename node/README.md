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
node batch.mjs --images ../../comp/data/kodak --preset ../presets/kodak.json \
  --set startSites=1000 --set endSites=1000 --compare 2
```

- The count must be fixed (`startSites` = `endSites`, not sawtooth, no turnover).
  Images of one size share an atlas; others get their own.
- `--compare <jobs>` also trains every image alone (`<jobs>` at once) and prints both
  runs' speed and the PSNR difference. On an M4 with the MNIST preset at 8 sites,
  256 digits: 9.4× faster than 8 separate trainers, −0.18 dB mean (per-image noise
  ±6 dB). Kodak at 1000 sites: no speedup (one image already fills the GPU), −0.11 dB.
