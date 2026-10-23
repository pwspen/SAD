"""Side-by-side figure from sad/summary.json and cnn/best.npz.

Top row: original, SAD decoded, SAD's encoding (its cells, one random color
each). Bottom row: CNN decoded, the CNN's encoding (latent channels as they are
stored) and its decoder weights (every 3x3 kernel, per layer).

  uv run --no-project --python 3.12 --with numpy --with pillow --with matplotlib \
    python experiments/ae_compare/figure.py --image ../comp/data/kodak/kodim23.png --sites 1000
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def kernel_mosaic(w, pad=1):
    """(in, out, k, k) weights as a grid of kernels, rows = outputs, NaN between."""
    cin, cout, k, _ = w.shape
    m = np.full((cout * (k + pad) - pad, cin * (k + pad) - pad), np.nan)
    for i in range(cin):
        for o in range(cout):
            m[o * (k + pad):o * (k + pad) + k, i * (k + pad):i * (k + pad) + k] = w[i, o]
    return m


def latent_mosaic(z, cols, pad=4):
    c, h, w = z.shape
    rows = -(-c // cols)
    m = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad), np.nan)
    for i in range(c):
        r, q = divmod(i, cols)
        m[r * (h + pad):r * (h + pad) + h, q * (w + pad):q * (w + pad) + w] = z[i]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sites", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE / "figure.png"))
    args = ap.parse_args()

    sad = json.load(open(HERE / "sad" / "summary.json"))
    sad_run = next(r for r in sad["runs"] if r["sites"] == args.sites)
    sad_dir = HERE / "sad" / f"sites_{args.sites}"
    cnn = np.load(HERE / "cnn" / "best.npz")
    cs = json.loads(str(cnn["stats"]))
    pixels = sad["width"] * sad["height"]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig = plt.figure(figsize=(15, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 3, wspace=0.04, hspace=0.18)

    def show(ax, path, title):
        ax.imshow(Image.open(path))
        ax.set_title(title, loc="left")
        ax.axis("off")

    show(fig.add_subplot(gs[0, 0]), args.image, f"Original ({sad['width']}×{sad['height']})")
    show(fig.add_subplot(gs[0, 1]), sad_dir / "recon.png",
         f"SAD decoded: {sad_run['psnr']:.2f} dB")
    show(fig.add_subplot(gs[0, 2]), sad_dir / "cells.png",
         f"SAD encoding: {sad_run['coded']} cells, {sad_run['bits'] / 8 / 1024:.1f} KiB ({sad_run['bpp']:.3f} bpp)")
    show(fig.add_subplot(gs[1, 0]), HERE / "cnn" / "recon.png",
         f"CNN autoencoder decoded: {cs['psnr']:.2f} dB")

    latent = cnn["latent"]
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(latent_mosaic(latent, cols=min(len(latent), 2)), cmap="gray", interpolation="nearest")
    ax.set_title(f"CNN encoding: {latent.shape[0]}×{latent.shape[1]}×{latent.shape[2]} latent, "
                 f"{cs['b']} bits/value", loc="left")
    ax.axis("off")

    sub = gs[1, 2].subgridspec(1, 3, wspace=0.15, width_ratios=[w.shape[0] for w in [cnn[f"dec{i}_w"] for i in range(3)]])
    layers = [cnn[f"dec{i}_w"] for i in range(3)]
    vmax = max(np.abs(w).max() for w in layers)
    for i, w in enumerate(layers):
        ax = fig.add_subplot(sub[i])
        ax.imshow(kernel_mosaic(w), cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"layer {i + 1}: {w.shape[0]}→{w.shape[1]}", fontsize=8)
    fig.text(0.0, 1.0, f"+ decoder weights: {cs['decoder_params']} params, 8-bit "
             f"(total {cs['bits'] / 8 / 1024:.1f} KiB, {cs['bpp']:.3f} bpp)",
             transform=fig.add_subplot(gs[1, 2], frameon=False, xticks=[], yticks=[]).transAxes, va="bottom")

    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
