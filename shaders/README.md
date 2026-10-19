# Shaders

WGSL kernels for SAD. `trainer/` and the viewer in `web/` load them at runtime.

- `sad_shared.wgsl` contains the structs and helpers every kernel shares, including the one site blend (`site_logit`, `cand_ids`, `softmax_c`) that rendering, gradients, site statistics and the info pass all use.
- `wgsl/` contains the per-kernel bodies that `trainer/shader_loader.js` concatenates with `sad_shared.wgsl`. The trainer and the viewer in `web/` both run these. `gradients_tiled` keeps its gradients in one interleaved buffer (WebGPU allows 10 storage buffers per stage), and the loader rewrites its `const TILE_SLOTS` line to fit the device's workgroup storage.
