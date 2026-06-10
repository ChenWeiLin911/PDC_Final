import sys
import torch

sys.path.append("./afno_extension")

import afno_cuda
from networks.afnonet import AFNO2D

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

B, H, W, C = 1, 90, 180, 768
num_blocks = 8

print("Using device:", device)

afno = AFNO2D(
    hidden_size=C,
    num_blocks=num_blocks,
    sparsity_threshold=0.01,
    hard_thresholding_fraction=1.0,
    hidden_size_factor=1,
).to(device).eval()

x = torch.randn(B, H, W, C, device=device)

with torch.no_grad():
    x_float = x.float()

    x_fft = torch.fft.rfft2(x_float, dim=(1, 2), norm="ortho")
    x_fft = x_fft.reshape(B, H, W // 2 + 1, afno.num_blocks, afno.block_size)

    x_real = x_fft.real.contiguous()
    x_imag = x_fft.imag.contiguous()

    out_real, out_imag = afno_cuda.frequency_mlp_forward(
        x_real,
        x_imag,
        afno.w1.contiguous(),
        afno.b1.contiguous(),
        afno.w2.contiguous(),
        afno.b2.contiguous(),
        float(afno.sparsity_threshold),
    )

print("x_fft shape:", x_fft.shape)
print("x_real shape:", x_real.shape)
print("x_imag shape:", x_imag.shape)
print("out_real shape:", out_real.shape)
print("out_imag shape:", out_imag.shape)

print("out_real dtype:", out_real.dtype)
print("out_imag dtype:", out_imag.dtype)

print("interface max real error identity:", (out_real - x_real).abs().max().item())
print("interface max imag error identity:", (out_imag - x_imag).abs().max().item())

print("allclose real:", torch.allclose(out_real, x_real))
print("allclose imag:", torch.allclose(out_imag, x_imag))

print("done")
