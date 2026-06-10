import sys
import csv
import argparse
import statistics
import torch

sys.path.append("./afno_extension")

import afno_cuda
from networks.afnonet import AFNO2D


def afno2d_cuda_frequency_mlp_forward(x, afno: AFNO2D):
    """
    Full AFNO2D flow, but replacing the frequency-domain MLP with CUDA extension.

    Original:
        x -> rfft2 -> PyTorch einsum/ReLU/einsum/softshrink -> irfft2 -> residual

    CUDA prototype:
        x -> rfft2 -> afno_cuda.frequency_mlp_forward -> irfft2 -> residual
    """
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


def measure_latency_ms(fn, warmup=10, iters=30):
    # Warmup
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times = []

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def summarize(times):
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--csv", type=str, default="logs/afno2d_latency_results.csv")
    args = parser.parse_args()

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This benchmark requires CUDA."

    B, H, W, C = 1, 90, 180, 768
    num_blocks = 8

    print("Using device:", device)
    print(f"Shape: B={B}, H={H}, W={W}, C={C}, num_blocks={num_blocks}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")

    afno = AFNO2D(
        hidden_size=C,
        num_blocks=num_blocks,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
    ).to(device).eval()

    x = torch.randn(B, H, W, C, device=device)

    with torch.no_grad():
        # Correctness check
        y_original = afno(x)
        y_cuda = afno2d_cuda_frequency_mlp_forward(x, afno)
        torch.cuda.synchronize()

        diff = (y_original - y_cuda).abs()
        relative_error = diff.norm() / y_original.norm()

        print("========== Correctness ==========")
        print("original output:", y_original.shape)
        print("cuda output:", y_cuda.shape)
        print("max error:", diff.max().item())
        print("mean error:", diff.mean().item())
        print("relative error:", relative_error.item())
        print("allclose atol=1e-5 rtol=1e-5:", torch.allclose(y_original, y_cuda, atol=1e-5, rtol=1e-5))
        print("allclose atol=1e-4 rtol=1e-4:", torch.allclose(y_original, y_cuda, atol=1e-4, rtol=1e-4))

        # Latency
        print("========== Latency Benchmark ==========")

        original_times = measure_latency_ms(
            lambda: afno(x),
            warmup=args.warmup,
            iters=args.iters,
        )

        cuda_times = measure_latency_ms(
            lambda: afno2d_cuda_frequency_mlp_forward(x, afno),
            warmup=args.warmup,
            iters=args.iters,
        )

    original_stat = summarize(original_times)
    cuda_stat = summarize(cuda_times)

    speedup = original_stat["avg_ms"] / cuda_stat["avg_ms"]

    print("========== Original AFNO2D.forward ==========")
    for k, v in original_stat.items():
        print(f"{k}: {v:.6f}")

    print("========== CUDA frequency MLP prototype ==========")
    for k, v in cuda_stat.items():
        print(f"{k}: {v:.6f}")

    print("========== Comparison ==========")
    print(f"speedup_original_over_cuda: {speedup:.6f}x")
    if speedup >= 1.0:
        print("CUDA prototype is faster on average.")
    else:
        print("CUDA prototype is slower on average. This is expected for a naive correctness kernel.")

    # Save CSV
    rows = [
        {
            "version": "original_afno2d_forward",
            **original_stat,
        },
        {
            "version": "cuda_frequency_mlp_prototype",
            **cuda_stat,
        },
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["version", "avg_ms", "min_ms", "max_ms", "median_ms", "std_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("CSV saved to:", args.csv)
    print("done")


if __name__ == "__main__":
    main()
