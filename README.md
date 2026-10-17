# SAD: Soft Anisotropic Diagrams

Code for the paper _Soft Anisotropic Diagrams for Differentiable Image Representation_, implemented in WebGPU. It runs in the browser and headless under Node.

Project page: https://luckyiyi.github.io/SAD/

The Metal, CUDA, PyTorch and wgpu-py backends, along with the `run.sh` / `install.sh` /
`reproduce_table4.sh` scripts, were removed after the `multi-backend` tag. Check out that
tag to reproduce the paper's numbers with the native backends:

```bash
git checkout multi-backend
```

## Train in the Browser

Serve the repository over HTTP. ES modules and `fetch` don't work from `file://`.

```bash
python3 -m http.server 8000
```

Then open:

- http://localhost:8000/backends/webgpu_js/train.html: interactive trainer
- http://localhost:8000/backends/webgpu_js/index.html: viewer for saved `*_sites.txt` files
- http://localhost:8000/backends/webgpu_js/bench.html?image=../../test.png: scripted benchmark

## Headless Sweeps (Node)

```bash
cd backends/webgpu_js/node
npm install
node sweep.mjs --image /path/to/image.png --set sites=500,1000,2000 --set lrPosBase=0.025,0.05
```

Each run appends one JSON line with its args, PSNR curve and timing to `results/`.
Run `node sweep.mjs --help` for every option.

## Layout

- `backends/webgpu_js/` contains the trainer (`src/`), the browser pages, and the Node runner (`node/`).
  See `backends/webgpu_js/README.md`.
- `backends/shared/` contains the WGSL kernels.
- `training_config.json` holds the hyperparameter defaults read by both hosts.

## Browser Demo

The hosted demo is available from the project page: https://luckyiyi.github.io/SAD/
