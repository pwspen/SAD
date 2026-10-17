# Shared WGSL

This folder holds the WGSL sources loaded by `backends/webgpu_js/`:

- `sad_shared.wgsl` contains common structs/helpers (the `COMMON` block) and the viewer kernels for SAD's soft anisotropic site scoring.
- `wgsl/` contains the per-kernel bodies that the trainer concatenates with `COMMON`. `gradients_tiled`, `adam`, `tau_extract` and `tau_writeback` are currently superseded by the versions in `webgpu_js/src/wgsl_overrides.js`.

## Regenerate JS Shader Bundle

The browser viewer falls back to `backends/webgpu_js/shaders.js` when it cannot fetch WGSL.
Regenerate that file after WGSL changes:

```bash
python backends/shared/gen_shaders_js.py
```
