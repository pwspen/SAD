# WebGPU JS

Browser tools backed by the shared WGSL kernels:

- `index.html` - **viewer** for `*_sites.txt` files (renderer-only)
- `train.html` - **trainer** (direct JS port of `backends/webgpu_py/train_wgpu.py`)
- `bench.html` - automated browser benchmark harness for the trainer modules

Both pages use the shared SAD WGSL kernels under `backends/shared/` and hyperparameters from `training_config.json`.

## Requirements

- WebGPU-capable browser (Chrome/Edge, Safari Tech Preview).
- HTTP(S) hosting. ES modules + `fetch` won't run from `file://`; GitHub Pages
  works because it serves the repository over HTTPS:

  ```bash
  python -m http.server 8000 -d /path/to/SAD
  # then open http://localhost:8000/backends/webgpu_js/train.html
  ```

## Viewer (`index.html`)

- Load a `*_sites.txt` file with an `# Image size: W H` header.
- Choose render mode, toggle site dots, and export PNG.

## Trainer (`train.html`)

- Pick a target image (and optional mask / init-sites file).
- Edit hyperparameters in the left panel (they mirror the CLI flags in `train_wgpu.py`; values are pre-filled from `training_config.json`).
- The site count is fixed for the whole run: the JS trainer has no densify/prune (and so no target-bpp). Defaults: `iters` 10000, `sites` 1000, `log-freq` 50, `viewer-freq` 50.
- Click **Train**. `viewer-freq` controls visual refreshes, while `log-freq` controls PSNR log lines (`Iter N | PSNR: X dB | Active: a/t | it/s | seconds`).
- **Export PNG** / **Export sites.txt** produce the same formats the Python and Metal trainers write.

## Headless sweeps (`node/sweep.mjs`)

The same trainer (`src/trainer.js`) runs under Node via Dawn's `webgpu` bindings,
so parameter sweeps don't need a browser:

```bash
cd backends/webgpu_js/node
npm install
node sweep.mjs --image /path/to/image.png --set sites=500,1000,2000 --set lrPosBase=0.025,0.05
node sweep.mjs --image /path/to/image.png --grid grid.json --repeat 3 --save-dir results/sweep
```

- `--set key=v1,v2` sweeps a trainer arg (camelCase, as in `buildArgs`) or a
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

## Benchmark (`bench.html`)

Run the browser trainer without manual file picking:

```bash
python -m http.server 8000 -d /path/to/SAD
# then open:
# http://localhost:8000/backends/webgpu_js/bench.html?image=../../test.png
```

The page logs `BENCH_RESULT` with final PSNR, active sites, training time, and total elapsed time.

### Storage-buffer limits

`gradients_tiled.wgsl` and `adam.wgsl` each bind 12 storage buffers per shader stage. The WebGPU default is often lower (8-10), so the trainer requests the adapter's maximum. On Chrome/Mac this is normally 12+ on Metal; if your browser reports fewer, training will fail at pipeline creation. Upgrade the browser or switch backends.

### Module layout

- `train.js` - DOM wiring + entry point.
- `src/shader_loader.js` - shared WGSL loader.
- `src/params.js` - parameter buffer writers, `pack_jump_step`, and `hilbert_bits_for_size`.
- `src/radix_sort.js` - radix-sort GPU pass orchestration.
- `src/encoders.js` - one class per GPU kernel.
- `src/buffers.js` - site/Adam/grad buffer factories.
- `src/textures.js` - target / mask / candidate texture helpers + image upload.
- `src/io.js` - sites `.txt` parsing + serialization + PNG download.
- `src/trainer.js` - training loop (fixed site count; no densify/prune) and epilogue.

## Shader Source

The viewer has `shaders.js` as a fallback with embedded WGSL. After editing any WGSL under `backends/shared/`, regenerate it with:

```bash
python backends/shared/gen_shaders_js.py
```

The trainer fetches WGSL at runtime and has no embedded fallback.
