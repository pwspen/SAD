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
  folder so the dataset picker can find its Hugging Face parquet files:

  | Dataset | File | Source |
  | --- | --- | --- |
  | Kodak | `../comp/data/kodak/validation.parquet` | [danjacobellis/kodak](https://huggingface.co/datasets/danjacobellis/kodak) (lossless WebP, pixel-identical to the PNGs) |
  | MNIST | `../comp/data/mnist/test.parquet` | [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist), test split |
  | CIFAR-10 | `../comp/data/cifar10/test.parquet` | [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), test split |
  | CLIC 2025 test | `data/clic2025_image_test/*.png` (in this repo, gitignored) | [CLIC image task](https://clic2027.compression.cc/tasks/#image), 30 PNGs up to 2048 px |

  Override a folder with `?kodak=<dir>` / `?mnist=<dir>` / `?cifar10=<dir>` / `?clic2025=<dir>`.
  A PNG folder is listed through the server's directory index, so it needs a
  server that lists folders (`./serve.sh` does). CLIC uses the Kodak preset.
  The files are read with [hyparquet](https://github.com/hyparam/hyparquet)
  1.31.2, vendored in `web/vendor/hyparquet` (MIT). CIFAR-10 uses the MNIST
  preset (small images, fixed sites); it hasn't been tuned on CIFAR.

## Viewer (`index.html`)

- Load a `*_sites.txt` file with an `# Image size: W H` header (and `# Global site: k` for a
  histogram layer).
- Choose render mode, toggle site dots, and export PNG.
- It renders with the trainer's kernels (`trainer/candidates.js`, `render_compute.wgsl`), so a
  file looks as it did in training.

## Trainer (`train.html`)

- Pick a target image (and optional mask / init-sites file).
- Edit settings in the **Advanced** panel. It is ordered by what a setting changes:
  - **Training** (what is learned; filled by the preset, values from `training_config.json` otherwise): start state, budget (site count), objective, optimizer, schedule (run length, LR and count schedules), growth rules, candidate search (collapsed).
  - **Batch** (which images and how they run; speed and noise only). **Batch settings** in the top bar opens the panel here.
  - **Instrumentation** (nothing about the result): **refresh every** for single runs (blank = about 20 refreshes per run), **log every** and **snapshots** for batch runs. Not part of presets.
  - **Tuning**: the tune search settings and **optimize all**, which searches Training settings on the Batch images.
- The site count follows the count schedule (start → end sites, linear / exponential / sawtooth); see `trainer/README.md`.
- Click **Train**. **refresh every** sets both visual refreshes and PSNR log lines (`Iter N | PSNR: X dB | Active: a/t | it/s | seconds`).
- Each viewer card carries a badge for where its data comes from: **Input**, **Model** (rendered by the trainer), **Training** (metrics the trainer reports) or **Viz only** (computed in the viewer by `analysis.js`; dashed card edge). Hover a card's label for exactly what it shows; cards tagged **after run** are computed once a run has stopped. Views are grouped by stage, one layout each: **Batch**, **Run**, **Images**, **Growth & merge**, **Spectral**, **Site statistics**, **Rate** and **Site editor**. Batch mode shows the Batch layout and a single run (or a batch run you open) the Run layout, while one of those two is showing. The **Spectral** layout has the radial power spectrum, spectral SNR / FRC, 2D spectra and per-band SNR.
- After a run, the **Entropy coding** card (in the **Rate** layout) codes the sites with `trainer/codec.js` at several PSNR budgets, decodes them and re-renders, and lists the bits per site by stage. **PSNR vs bpp** then shows the coded rate (green) beside the assumed 128 bits/site.
- **Batch** trains a range of dataset images with one set of settings. Its layout shows PSNR vs iteration, final PSNR, and, for MNIST and CIFAR-10, the final PSNR of every run grouped by digit / class (box plot with one dot per run) once the batch finishes.
- **tune**, next to each float setting under Advanced, auto-tunes that one setting with the current batch settings: it runs a batch at the value × and ÷ **tune ratio** (default 2, on the same images as the baseline), moves to whichever raises the median final PSNR, shrinks the step when neither does, and stops below **tune min ratio** or after **tune max rounds** (default 20). **optimize all** goes through a fixed order (objective, optimizer, schedule, growth) and skips settings that have no effect with the current values (fields hidden for the count schedule; persist decay and cap while persist weight is 0). Click it again, or Stop, to end early and keep the best value so far.
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

Both pages fetch WGSL at runtime (serve over HTTP; ES modules and `fetch` don't work from
`file://`).
