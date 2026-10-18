# Web

Browser pages for SAD. They run the trainer in `trainer/` and the WGSL kernels in `shaders/`:

- `index.html` - **viewer** for `*_sites.txt` files (renderer-only)
- `train.html` - **trainer** (originally ported from the wgpu-py trainer at the `multi-backend` tag)
- `bench.html` - automated browser benchmark harness for the trainer modules

All pages read hyperparameters from `training_config.json`.

## Requirements

- WebGPU-capable browser (Chrome/Edge, Safari Tech Preview).
- HTTP(S) hosting. ES modules + `fetch` won't run from `file://`; GitHub Pages
  works because it serves the repository over HTTPS. Locally, run `./serve.sh`
  from the repo root and open the trainer link it prints. It serves the parent
  folder so the dataset picker can find `../comp/data/kodak` and
  `../comp/data/mnist` (the MNIST test split's `t10k-*-ubyte.gz` files);
  override the folders with `?kodak=<dir>` / `?mnist=<dir>`.

## Viewer (`index.html`)

- Load a `*_sites.txt` file with an `# Image size: W H` header.
- Choose render mode, toggle site dots, and export PNG.

## Trainer (`train.html`)

- Pick a target image (and optional mask / init-sites file).
- Edit hyperparameters in the left panel (values are pre-filled from `training_config.json`).
- The site count is fixed for the whole run: the JS trainer has no densify/prune (and so no target-bpp). Defaults: 100 warmup + 500 decay iterations, LR mult 3, `sites` 1000, `log-freq` 50, `viewer-freq` 50.
- Click **Train**. `viewer-freq` controls visual refreshes, while `log-freq` controls PSNR log lines (`Iter N | PSNR: X dB | Active: a/t | it/s | seconds`).
- Each viewer card carries a badge for where its data comes from: **Input**, **Model** (rendered by the trainer), **Training** (metrics the trainer reports) or **Viz only** (computed in the viewer by `analysis.js`; dashed card edge). Hover a card's label for exactly what it shows. The **Spectral** layout has the radial power spectrum, spectral SNR / FRC, 2D spectra and per-band SNR.
- **Edit layout** lets you drag cards by their label (onto an empty cell, or onto a same-size card to swap), add views from empty cells, remove them with ×, and add or remove rows / columns. Cards keep the size their layout gave them. Edits go to a **Custom** layout saved in the browser.
- **Export PNG** / **Export sites.txt** produce the same formats the Python and Metal trainers write.

## Benchmark (`bench.html`)

Run the browser trainer without manual file picking:

```bash
./serve.sh
# then open:
# http://127.0.0.1:8766/SAD/web/bench.html?image=../test.png
```

The page logs `BENCH_RESULT` with final PSNR, active sites, training time, and total elapsed time.

## Shader Source

The viewer has `shaders.js` as a fallback with embedded WGSL. After editing any WGSL under `shaders/`, regenerate it with:

```bash
python shaders/gen_shaders_js.py
```

The trainer fetches WGSL at runtime and has no embedded fallback.
