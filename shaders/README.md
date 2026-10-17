# Shaders

WGSL kernels for SAD. `trainer/` and the viewer in `web/` load them at runtime.

- `sad_shared.wgsl` contains common structs/helpers (the `COMMON` block) and the viewer kernels for SAD's soft anisotropic site scoring.
- `wgsl/` contains the per-kernel bodies that the trainer concatenates with `COMMON`. These are the real kernels; there are no JS-side overrides. `gradients_tiled` keeps its gradients in one interleaved buffer (WebGPU allows 10 storage buffers per stage), and `trainer/shader_loader.js` rewrites its `const TILE_SLOTS` line to fit the device's workgroup storage.

## Regenerate JS Shader Bundle

The browser viewer falls back to `web/shaders.js` when it cannot fetch WGSL.
Regenerate that file after WGSL changes:

```bash
python shaders/gen_shaders_js.py
```
