# Shared WGSL

This folder holds the WGSL sources loaded by `backends/webgpu_js/`:

- `sad_shared.wgsl` contains common structs/helpers (the `COMMON` block) and the viewer kernels for SAD's soft anisotropic site scoring.
- `wgsl/` contains the per-kernel bodies that the trainer concatenates with `COMMON`. These are the real kernels; there are no JS-side overrides. `gradients_tiled` keeps its gradients in one interleaved buffer (WebGPU allows 10 storage buffers per stage), and `webgpu_js/src/shader_loader.js` rewrites its `const TILE_SLOTS` line to fit the device's workgroup storage.

## Regenerate JS Shader Bundle

The browser viewer falls back to `backends/webgpu_js/shaders.js` when it cannot fetch WGSL.
Regenerate that file after WGSL changes:

```bash
python backends/shared/gen_shaders_js.py
```
