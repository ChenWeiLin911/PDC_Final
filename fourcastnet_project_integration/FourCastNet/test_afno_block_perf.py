#!/usr/bin/env python
"""Benchmark one AFNO filter block: original AFNO2D vs optimized AFNO2DSeparableBMM.

This script tests the AFNO filter in networks/afnonet.py directly, not the full
FourCastNet model. It copies the same random weights into both implementations,
runs a few warmup forwards, then times N forwards on the same input.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from networks.afnonet import AFNO2D, AFNO2DSeparableBMM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=90, help="Patch-grid height. FourCastNet 720/8 = 90.")
    parser.add_argument("--width", type=int, default=180, help="Patch-grid width. FourCastNet 1440/8 = 180.")
    parser.add_argument("--hidden-size", type=int, default=768, help="AFNO embedding dimension.")
    parser.add_argument("--num-blocks", type=int, default=8, help="AFNO channel blocks.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5, help="Number of timed forward runs.")
    parser.add_argument("--seed", type=int, default=0, help="Default matches test_afno2d_latency.py.")
    parser.add_argument("--compile-original", action="store_true", help="Also torch.compile the original AFNO block.")
    parser.add_argument("--compile-optimized", action="store_true", help="torch.compile the optimized AFNO block.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def time_forward(name, module, x, warmup, iters):
    # Use no_grad, same as test_afno2d_latency.py, so original timing is comparable.
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()

        times_ms = []
        for idx in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y = module(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            times_ms.append(elapsed)
            print(f"{name} run {idx}: {elapsed:.6f} ms")

    t = torch.tensor(times_ms, dtype=torch.float64)
    print(
        f"{name} summary: runs={iters}, "
        f"mean={t.mean().item():.6f} ms, "
        f"min={t.min().item():.6f} ms, "
        f"max={t.max().item():.6f} ms"
    )
    return y, times_ms


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(0))
    print("gpu_capability", torch.cuda.get_device_capability(0))
    print(
        "shape",
        (args.batch_size, args.height, args.width, args.hidden_size),
        "num_blocks",
        args.num_blocks,
    )

    # Match test_afno2d_latency.py / run_afno2d_latency.sbatch original AFNO setup exactly.
    afno_kwargs = dict(
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
    )
    original = AFNO2D(**afno_kwargs).to(device).eval()
    optimized = AFNO2DSeparableBMM(**afno_kwargs).to(device).eval()
    optimized.load_state_dict(original.state_dict())
    if hasattr(optimized, "prepare_inference_cache"):
        optimized.prepare_inference_cache()

    x = torch.randn(args.batch_size, args.height, args.width, args.hidden_size, device=device)

    with torch.no_grad():
        y_original_ref = original(x)
        torch.cuda.synchronize()
        y_optimized_ref = optimized(x)
        torch.cuda.synchronize()

    diff = (y_original_ref - y_optimized_ref).abs()
    print("max_abs_diff", diff.max().item())
    print("mean_abs_diff", diff.mean().item())
    print("allclose", torch.allclose(y_original_ref, y_optimized_ref, atol=args.atol, rtol=args.rtol))

    if args.compile_original:
        print(f"Compiling original with mode={args.compile_mode}")
        original = torch.compile(original, mode=args.compile_mode, fullgraph=False)
    if args.compile_optimized:
        print(f"Compiling optimized with mode={args.compile_mode}")
        optimized = torch.compile(optimized, mode=args.compile_mode, fullgraph=False)

    y_original, original_times = time_forward("original_afno", original, x, args.warmup, args.iters)
    y_optimized, optimized_times = time_forward("optimized_sep_bmm_afno", optimized, x, args.warmup, args.iters)

    original_mean = sum(original_times) / len(original_times)
    optimized_mean = sum(optimized_times) / len(optimized_times)
    print(f"speedup_original_over_optimized: {original_mean / optimized_mean:.6f}x")

    final_diff = (y_original - y_optimized).abs()
    print("final_max_abs_diff", final_diff.max().item())
    print("final_mean_abs_diff", final_diff.mean().item())


if __name__ == "__main__":
    main()
