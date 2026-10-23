# SAD vs a CNN autoencoder at equal bits

`figure.png`: Kodak kodim23 (768×512) coded by SAD and by a textbook convolutional
autoencoder overfit to the same image. The autoencoder gets the same number of bits as
SAD, counting its latent plus its decoder weights.

| | decoded PSNR | bits | bpp |
|---|---|---|---|
| SAD, 1000 cells (real codec, post-hoc at 0.5 dB loss) | **30.14 dB** | 51,606 | 0.131 |
| CNN AE, c = 3, 2-bit latent (best under SAD's bits) | 23.11 dB | 48,314 = latent 35,897 + decoder 12,417 | 0.123 |

**SAD** (`node/ae_compare_sad.mjs`): kodak preset at a fixed 1000 sites, since batch mode
needs a fixed count. The sites are coded with `trainer/codec.js` at `codecSteps`' steps for
0.5 dB, then decoded and rendered. `cells.png` is the trainer's "hashed" view of the decoded
sites, one random color per cell.

```bash
cd node && node ae_compare_sad.mjs --images ../../comp/data/kodak --from 22 --count 1 \
  --preset ../presets/kodak.json --sites 1000 --out ../experiments/ae_compare/sad
```

**CNN** (`cnn_ae.py`): three 3×3 stride-2 convs, 3 → 4c → 2c → c channels, with ReLU. The
decoder mirrors them with transposed convs and a sigmoid output. The latent is
sigmoid-bounded and quantized to b bits, with uniform noise during training. Training runs
8000 Adam steps (lr 2e-3, cosine) on MSE. Counted bits are the latent symbols plus the
decoder weights (8-bit, per-tensor scale), each priced at its empirical entropy plus
headers. That is an ideal coder, so the count is generous to the AE. The encoder is not
counted. A run whose latent collapses to a constant is reseeded.

The sweep covered c ∈ {1, 2, 3, 4, 6} × b ∈ {2…6}.
- `cnn_sweep_all.log` and `cnn/runs_all.json` hold the first pass, without reseeding: c = 1
  and c = 3 collapsed there, and every c ≥ 4 run was over budget (c = 4 at 2 bits: 25.08 dB
  at 65k bits).
- `cnn_sweep.log` and `cnn/runs.json` rerun c ≤ 3 with reseeding.
- Even c = 6 at 6 bits (241k bits, 4.7× SAD's) reaches only 28.5 dB.

```bash
uv run --no-project --python 3.12 --with torch --with numpy --with pillow \
  python experiments/ae_compare/cnn_ae.py --image ../comp/data/kodak/kodim23.png --budget 51606 \
  --channels 1,2,3 --bits 2,3,4,5,6 --iters 8000
uv run --no-project --python 3.12 --with numpy --with pillow --with matplotlib \
  python experiments/ae_compare/figure.py --image ../comp/data/kodak/kodim23.png --sites 1000
```
