import sys
import torch

sys.path.append("./afno_extension")

import afno_cuda
from networks.afnonet import AFNO2D

torch.manual_seed(0)

device = "cuda"

B, H, W, C = 1, 90, 180, 768
num_blocks = 8


def afno2d_cuda_frequency_mlp_forward(x, afno: AFNO2D):
    bias = x

    dtype = x.dtype
    x = x.float()
    B, H, W, C = x.shape

    x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
    x_fft = x_fft.reshape(B, H, W // 2 + 1, afno.num_blocks, afno.block_size)

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


def measure_peak_memory(fn, name, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for _ in range(iters):
        y = fn()

    torch.cuda.synchronize()

    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()

    print(f"========== {name} ==========")
    print(f"max_memory_allocated_MB: {allocated / 1024 / 1024:.3f}")
    print(f"max_memory_reserved_MB:  {reserved / 1024 / 1024:.3f}")

    return allocated, reserved


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
    y_cuda = afno2d_cuda_frequency_mlp_forward(x, afno)

    diff = (y_original - y_cuda).abs()
    print("========== Correctness ==========")
    print("max error:", diff.max().item())
    print("mean error:", diff.mean().item())
    print("relative error:", (diff.norm() / y_original.norm()).item())
    print("allclose 1e-5:", torch.allclose(y_original, y_cuda, atol=1e-5, rtol=1e-5))

    measure_peak_memory(lambda: afno(x), "Original AFNO2D.forward")

    measure_peak_memory(
        lambda: afno2d_cuda_frequency_mlp_forward(x, afno),
        "CUDA frequency MLP prototype",
    )

print("done")
