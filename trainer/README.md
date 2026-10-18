# Trainer

The WebGPU optimizer: the training loop, Adam schedule, GPU buffers and kernel dispatch.
It has no DOM dependencies, so the browser pages in `web/` and the Node runner in `node/`
both import it. Hyperparameter defaults come from `training_config.json`.

`siteMode` picks how many sites a run uses:

- `"grow"` (default): start from `growInitSites` and add sites by rate-distortion. Every
  `growEvery` iterations, each site-spacing cell gets a site if the best error one site could
  remove there (scored over the cell's dyadic sub-blocks) beats `growLambda × BITS_PER_SITE`, so
  growth runs coarse to fine and stops when sites no longer pay for their bits. `growMaxSites`
  is a hard cap (and sizes the buffers); if growth hits it, the trainer logs it and reports
  `growCapped`, since the cap rather than λ then sets the count. Sites are never removed.
- `"fixed"`: exactly `sites` for the whole run.

Without an explicit `siteMode`, `growLambda <= 0` means fixed, so older `growLambda=0` configs
still work.

## Modules

- `trainer.js` - training loop (fixed site count, or rate-distortion growth; no prune) and epilogue.
- `shader_loader.js` - fetches and assembles the WGSL in `shaders/`.
- `device.js` - adapter/device setup.
- `params.js` - parameter buffer writers, `pack_jump_step`, and `hilbert_bits_for_size`.
- `radix_sort.js` - radix-sort GPU pass orchestration.
- `encoders.js` - one class per GPU kernel.
- `buffers.js` - site/Adam/grad buffer factories.
- `textures.js` - target / mask / candidate texture helpers + image upload.
- `io.js` - sites `.txt` parsing + serialization + PNG download.

## Storage-buffer limits

`gradients_tiled.wgsl` and `adam.wgsl` each bind 12 storage buffers per shader stage. The WebGPU default is often lower (8-10), so the trainer requests the adapter's maximum. On Chrome/Mac this is normally 12+ on Metal; if your browser reports fewer, training will fail at pipeline creation. Upgrade the browser.
