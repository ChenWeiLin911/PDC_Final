import sys
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original_full", "cuda_full"], required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    afno = AFNO2D(
        hidden_size=C,
        num_blocks=num_blocks,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
    ).to(device).eval()

    x = torch.randn(B, H, W, C, device=device)

    with torch.no_grad():
        # warmup
        for _ in range(2):
            if args.mode == "original_full":
                y = afno(x)
            else:
                y = afno2d_cuda_frequency_mlp_forward(x, afno)

        torch.cuda.synchronize()

        for _ in range(args.repeat):
            if args.mode == "original_full":
                y = afno(x)
            else:
                y = afno2d_cuda_frequency_mlp_forward(x, afno)

        torch.cuda.synchronize()

    print("mode:", args.mode)
    print("repeat:", args.repeat)
    print("output shape:", y.shape)
    print("done")


if __name__ == "__main__":
    main()
