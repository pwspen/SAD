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

Serve the pages over HTTP (ES modules and `fetch` don't work from `file://`):

```bash
./serve.sh
```

It serves the folder above the repo on port 8766 (`PORT=…` to change it), so the
trainer's Kodak picker can reach `../comp/data/kodak`, and prints the links:

- http://127.0.0.1:8766/SAD/web/train.html: interactive trainer
- http://127.0.0.1:8766/SAD/web/index.html: viewer for saved `*_sites.txt` files
- http://127.0.0.1:8766/SAD/web/bench.html?image=../test.png: scripted benchmark

## Headless Sweeps (Node)

```bash
cd node
npm install
node sweep.mjs --image /path/to/image.png --set sites=500,1000,2000 --set lrPosBase=0.025,0.05
```

Each run appends one JSON line with its args, PSNR curve and timing to `results/`.
Run `node sweep.mjs --help` for every option.

## Layout

- `web/` contains the browser pages: trainer UI, viewer and benchmark. See `web/README.md`.
- `trainer/` contains the WebGPU optimizer (training loop, Adam schedule, buffers, kernel
  dispatch). It has no DOM code, so the browser and Node both use it. See `trainer/README.md`.
- `shaders/` contains the WGSL kernels. See `shaders/README.md`.
- `node/` contains the headless sweep runner. See `node/README.md`.
- `training_config.json` holds the hyperparameter defaults read by both hosts.

## Browser Demo

The hosted demo is available from the project page: https://luckyiyi.github.io/SAD/
