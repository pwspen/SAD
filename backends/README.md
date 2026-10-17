# SAD Backends

- `webgpu_js/` contains the WebGPU trainer, the browser viewer, trainer and benchmark pages, and the headless Node sweep runner.
- `shared/` contains the WGSL kernels loaded by `webgpu_js/`.

The Metal, CUDA, PyTorch and wgpu-py backends live at the `multi-backend` git tag.
