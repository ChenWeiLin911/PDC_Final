import torch
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
    y = afno(x)

print("input:", x.shape)
print("output:", y.shape)
print("output dtype:", y.dtype)
print("max abs:", y.abs().max().item())
print("mean abs:", y.abs().mean().item())

assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"

print("done")
