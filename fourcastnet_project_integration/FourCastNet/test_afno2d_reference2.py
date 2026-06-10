import torch
import torch.nn.functional as F
from networks.afnonet import AFNO2D

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

B, H, W, C = 1, 90, 180, 768
num_blocks = 8


def afno2d_reference_forward(
    x,
    w1,
    b1,
    w2,
    b2,
    num_blocks,
    block_size,
    hidden_size_factor,
    sparsity_threshold,
    hard_thresholding_fraction,
):
    bias = x

    dtype = x.dtype
    x = x.float()
    B, H, W, C = x.shape

    x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
    x = x.reshape(B, H, W // 2 + 1, num_blocks, block_size)

    o1_real = torch.zeros(
        [B, H, W // 2 + 1, num_blocks, block_size * hidden_size_factor],
        device=x.device,
    )
    o1_imag = torch.zeros(
        [B, H, W // 2 + 1, num_blocks, block_size * hidden_size_factor],
        device=x.device,
    )
    o2_real = torch.zeros(x.shape, device=x.device)
    o2_imag = torch.zeros(x.shape, device=x.device)

    total_modes = H // 2 + 1
    kept_modes = int(total_modes * hard_thresholding_fraction)

    h_slice = slice(total_modes - kept_modes, total_modes + kept_modes)
    w_slice = slice(0, kept_modes)

    x_selected = x[:, h_slice, w_slice]

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

    x = torch.stack([o2_real, o2_imag], dim=-1)
    x = F.softshrink(x, lambd=sparsity_threshold)
    x = torch.view_as_complex(x)
    x = x.reshape(B, H, W // 2 + 1, C)
    x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
    x = x.type(dtype)

    return x + bias


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

    y_ref = afno2d_reference_forward(
        x=x,
        w1=afno.w1,
        b1=afno.b1,
        w2=afno.w2,
        b2=afno.b2,
        num_blocks=afno.num_blocks,
        block_size=afno.block_size,
        hidden_size_factor=afno.hidden_size_factor,
        sparsity_threshold=afno.sparsity_threshold,
        hard_thresholding_fraction=afno.hard_thresholding_fraction,
    )

diff = (y_original - y_ref).abs()
relative_error = diff.norm() / y_original.norm()

print("input:", x.shape)
print("original output:", y_original.shape)
print("reference output:", y_ref.shape)
print("max error:", diff.max().item())
print("mean error:", diff.mean().item())
print("relative error:", relative_error.item())
print("allclose:", torch.allclose(y_original, y_ref, atol=1e-6, rtol=1e-6))
print("done")
