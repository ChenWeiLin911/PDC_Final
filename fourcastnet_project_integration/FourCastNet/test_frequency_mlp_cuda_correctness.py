import sys
import torch
import torch.nn.functional as F

sys.path.append("./afno_extension")

import afno_cuda
from networks.afnonet import AFNO2D

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

B, H, W, C = 1, 90, 180, 768
num_blocks = 8


def afno_frequency_mlp_reference(
    x_fft,
    w1,
    b1,
    w2,
    b2,
    sparsity_threshold,
    hard_thresholding_fraction,
):
    B, H, W_freq, num_blocks, block_size = x_fft.shape
    hidden_size_factor = w1.shape[-1] // block_size

    o1_real = torch.zeros(
        [B, H, W_freq, num_blocks, block_size * hidden_size_factor],
        device=x_fft.device,
    )
    o1_imag = torch.zeros(
        [B, H, W_freq, num_blocks, block_size * hidden_size_factor],
        device=x_fft.device,
    )

    o2_real = torch.zeros(x_fft.shape, device=x_fft.device)
    o2_imag = torch.zeros(x_fft.shape, device=x_fft.device)

    total_modes = H // 2 + 1
    kept_modes = int(total_modes * hard_thresholding_fraction)

    h_slice = slice(total_modes - kept_modes, total_modes + kept_modes)
    w_slice = slice(0, kept_modes)

    x_selected = x_fft[:, h_slice, w_slice]

    o1_real[:, h_slice, w_slice] = F.relu(
        torch.einsum("...bi,bio->...bo", x_selected.real, w1[0])
        - torch.einsum("...bi,bio->...bo", x_selected.imag, w1[1])
        + b1[0]
    )

    o1_imag[:, h_slice, w_slice] = F.relu(
        torch.einsum("...bi,bio->...bo", x_selected.imag, w1[0])
        + torch.einsum("...bi,bio->...bo", x_selected.real, w1[1])
        + b1[1]
    )

    o1_real_selected = o1_real[:, h_slice, w_slice]
    o1_imag_selected = o1_imag[:, h_slice, w_slice]

    o2_real[:, h_slice, w_slice] = (
        torch.einsum("...bi,bio->...bo", o1_real_selected, w2[0])
        - torch.einsum("...bi,bio->...bo", o1_imag_selected, w2[1])
        + b2[0]
    )

    o2_imag[:, h_slice, w_slice] = (
        torch.einsum("...bi,bio->...bo", o1_imag_selected, w2[0])
        + torch.einsum("...bi,bio->...bo", o1_real_selected, w2[1])
        + b2[1]
    )

    x_out = torch.stack([o2_real, o2_imag], dim=-1)
    x_out = F.softshrink(x_out, lambd=sparsity_threshold)
    x_out = torch.view_as_complex(x_out)

    return x_out


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

    ref_complex = afno_frequency_mlp_reference(
        x_fft=x_fft,
        w1=afno.w1,
        b1=afno.b1,
        w2=afno.w2,
        b2=afno.b2,
        sparsity_threshold=afno.sparsity_threshold,
        hard_thresholding_fraction=afno.hard_thresholding_fraction,
    )

    out_real_cuda, out_imag_cuda = afno_cuda.frequency_mlp_forward(
        x_fft.real.contiguous(),
        x_fft.imag.contiguous(),
        afno.w1.contiguous(),
        afno.b1.contiguous(),
        afno.w2.contiguous(),
        afno.b2.contiguous(),
        float(afno.sparsity_threshold),
    )

    torch.cuda.synchronize()

    cuda_complex = torch.complex(out_real_cuda, out_imag_cuda)

diff = (ref_complex - cuda_complex).abs()
relative_error = diff.norm() / ref_complex.norm()

print("x_fft shape:", x_fft.shape)
print("ref_complex shape:", ref_complex.shape)
print("cuda_complex shape:", cuda_complex.shape)

print("max error:", diff.max().item())
print("mean error:", diff.mean().item())
print("relative error:", relative_error.item())

print("allclose atol=1e-4 rtol=1e-4:", torch.allclose(ref_complex, cuda_complex, atol=1e-4, rtol=1e-4))
print("allclose atol=1e-3 rtol=1e-3:", torch.allclose(ref_complex, cuda_complex, atol=1e-3, rtol=1e-3))

print("done")
