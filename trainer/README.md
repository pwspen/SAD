# Trainer

The WebGPU optimizer: the training loop, Adam schedule, GPU buffers and kernel dispatch.
It has no DOM dependencies, so the browser pages in `web/` and the Node runner in `node/`
both import it. Hyperparameter defaults come from `training_config.json`.

`siteMode` picks how many sites a run uses:

- `"grow"` (default): start from `growInitSites` and add one site every `growEvery` iterations,
  greedily at the single best place (the dyadic block where one site would remove the most
  error), until `growMaxSites`. `growLambda` stops growth early once the best gain falls under
  `growLambda × BITS_PER_SITE` (0 = no early stop). Sites are never removed.
- `"fixed"`: exactly `sites` for the whole run.

`lrSchedule` is `"warmup-cosine"` (default), `"fixed"` (constant after warmup) or
`"decay-after-growth"` (constant until growth ends, then cosine to 0 by the end of the run).

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
