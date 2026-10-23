"""CNN autoencoder side of the SAD vs CNN-autoencoder figure.

A textbook convolutional autoencoder overfit to one image: three stride-2
3x3 convs down (3 -> 4c -> 2c -> c channels, ReLU), three stride-2
transposed convs back up (c -> 2c -> 4c -> 3, sigmoid out). The latent is
squashed to [0, 1] by a sigmoid and quantized to `b` bits (uniform noise of
one step while training, rounding at test time).

What a decoder needs to rebuild the image is the latent plus the decoder's
weights, so that is what is counted: the latent's b-bit symbols and the
decoder's 8-bit weights (per-tensor scale), each priced at its empirical
entropy (an ideal static entropy coder, generous to the AE) plus a 32-bit
header per tensor. The encoder is not counted.

Sweeps c and b, keeps every run's bits and PSNR in cnn/runs.json and, for the
best PSNR under --budget bits, writes the reconstruction, latent and decoder
weights to cnn/best.npz and cnn/recon.png.

  uv run --no-project --python 3.12 --with torch --with numpy --with pillow \
    python experiments/ae_compare/cnn_ae.py --image ../comp/data/kodak/kodim23.png --budget 51606
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent


class AE(nn.Module):
    def __init__(self, c):
        super().__init__()
        ch = [3, 4 * c, 2 * c, c]
        self.enc = nn.ModuleList(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(3))
        self.dec = nn.ModuleList(
            nn.ConvTranspose2d(ch[3 - i], ch[2 - i], 3, stride=2, padding=1, output_padding=1) for i in range(3)
        )

    def encode(self, x):
        for i, conv in enumerate(self.enc):
            x = conv(x)
            if i < 2:
                x = F.relu(x)
        return torch.sigmoid(x)

    def decode(self, z):
        for i, conv in enumerate(self.dec):
            z = conv(z)
            z = F.relu(z) if i < 2 else torch.sigmoid(z)
        return z


def entropy_bits(symbols):
    """Ideal static entropy coding of an integer array, plus the symbol table."""
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum()) + 16 * len(counts) + 32


def quantize_weights(model, bits=8):
    """Per-tensor symmetric quantization of the decoder in place; returns its coded bits."""
    total = 0.0
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.dec.parameters():
            scale = p.abs().max().clamp_min(1e-12) / qmax
            q = torch.round(p / scale)
            p.copy_(q * scale)
            total += entropy_bits(q.cpu().numpy().astype(np.int64)) + 32
    return total


def psnr(a, b):
    return float(10 * torch.log10(1 / F.mse_loss(a, b)))


def run(x, c, b, iters, device, seed=0):
    torch.manual_seed(seed)
    model = AE(c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    levels = 2**b - 1
    for _ in range(iters):
        z = model.encode(x) * levels
        z = z + (torch.rand_like(z) - 0.5)
        loss = F.mse_loss(model.decode(z / levels), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        q = torch.round(model.encode(x) * levels)
        clean = psnr(model.decode(q / levels), x)
        latent_bits = entropy_bits(q.cpu().numpy().astype(np.int64)) + 32
        weight_bits = quantize_weights(model)
        recon = model.decode(q / levels)
    n_dec = sum(p.numel() for p in model.dec.parameters())
    n_enc = sum(p.numel() for p in model.enc.parameters())
    stats = dict(
        c=c, b=b, psnr=psnr(recon, x), psnr_float_weights=clean,
        latent_bits=latent_bits, weight_bits=weight_bits, bits=latent_bits + weight_bits,
        latent_values=q.numel(), decoder_params=n_dec, encoder_params=n_enc,
    )
    return stats, model, q, recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--budget", type=float, required=True, help="bits to match (SAD's coded size)")
    ap.add_argument("--channels", default="1,2,3,4,6,8")
    ap.add_argument("--bits", default="1,2,3,4,5,6")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default=str(HERE / "cnn"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
    w, h = x.shape[3], x.shape[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs, best = [], None
    for c in map(int, args.channels.split(",")):
        for b in map(int, args.bits.split(",")):
            # A run whose latent collapses to a constant (dead ReLUs / saturated
            # sigmoid) is a training failure, not a rate point: reseed.
            for seed in range(4):
                stats, model, q, recon = run(x, c, b, args.iters, device, seed)
                if len(torch.unique(q)) > 1:
                    break
            stats["seed"] = seed
            stats["bpp"] = stats["bits"] / (w * h)
            runs.append(stats)
            fits = stats["bits"] <= args.budget
            print(f"c={c} b={b} | {stats['psnr']:.2f} dB ({stats['psnr_float_weights']:.2f} float weights) | "
                  f"{stats['bits']:.0f} bits = latent {stats['latent_bits']:.0f} + decoder {stats['weight_bits']:.0f} "
                  f"({stats['decoder_params']} params) | {stats['bpp']:.4f} bpp{'' if fits else ' (over budget)'}",
                  flush=True)
            if fits and (best is None or stats["psnr"] > best[0]["psnr"]):
                best = (stats, model, q, recon)
            json.dump({"image": args.image, "budget": args.budget, "runs": runs}, open(out / "runs.json", "w"), indent=2)

    stats, model, q, recon = best
    print(f"best under {args.budget:.0f} bits: c={stats['c']} b={stats['b']} {stats['psnr']:.2f} dB at {stats['bits']:.0f} bits")
    Image.fromarray((recon[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(out / "recon.png")
    np.savez(
        out / "best.npz",
        latent=q[0].cpu().numpy(),
        **{f"dec{i}_w": m.weight.detach().cpu().numpy() for i, m in enumerate(model.dec)},
        **{f"dec{i}_b": m.bias.detach().cpu().numpy() for i, m in enumerate(model.dec)},
        stats=json.dumps(stats),
    )


if __name__ == "__main__":
    main()
