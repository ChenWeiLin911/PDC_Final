import sys
import torch

sys.path.append("./afno_extension")

import afno_cuda
from networks.afnonet import AFNO2D

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

B, H, W, C = 1, 90, 180, 768
num_blocks = 8


def afno2d_cuda_frequency_mlp_forward(
    x,
    afno: AFNO2D,
):
    """
    Full AFNO2D flow, but replacing the frequency-domain MLP with CUDA extension.

    Original:
        x -> rfft2 -> PyTorch einsum/ReLU/einsum/softshrink -> irfft2 -> residual

    This version:
        x -> rfft2 -> afno_cuda.frequency_mlp_forward -> irfft2 -> residual
    """

    bias = x

    dtype = x.dtype
    x = x.float()
    B, H, W, C = x.shape

    # Same as original AFNO2D.forward()
    x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
    x_fft = x_fft.reshape(B, H, W // 2 + 1, afno.num_blocks, afno.block_size)

    # CUDA frequency MLP
    out_real, out_imag = afno_cuda.frequency_mlp_forward(
        x_fft.real.contiguous(),
        x_fft.imag.contiguous(),
        afno.w1.contiguous(),
        afno.b1.contiguous(),
        afno.w2.contiguous(),
        afno.b2.contiguous(),
        float(afno.sparsity_threshold),
    )

    x_fft_out = torch.complex(out_real, out_imag)
    x_fft_out = x_fft_out.reshape(B, H, W // 2 + 1, C)

    x_out = torch.fft.irfft2(
        x_fft_out,
        s=(H, W),
        dim=(1, 2),
        norm="ortho",
    )

    x_out = x_out.type(dtype)

    return x_out + bias


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
    y_original = afno(x)

    y_cuda = afno2d_cuda_frequency_mlp_forward(
        x=x,
        afno=afno,
    )

    torch.cuda.synchronize()

diff = (y_original - y_cuda).abs()
relative_error = diff.norm() / y_original.norm()

print("input:", x.shape)
print("original output:", y_original.shape)
print("cuda full output:", y_cuda.shape)

print("max error:", diff.max().item())
print("mean error:", diff.mean().item())
print("relative error:", relative_error.item())

print("allclose atol=1e-5 rtol=1e-5:", torch.allclose(y_original, y_cuda, atol=1e-5, rtol=1e-5))
print("allclose atol=1e-4 rtol=1e-4:", torch.allclose(y_original, y_cuda, atol=1e-4, rtol=1e-4))
print("allclose atol=1e-3 rtol=1e-3:", torch.allclose(y_original, y_cuda, atol=1e-3, rtol=1e-3))

print("done")
