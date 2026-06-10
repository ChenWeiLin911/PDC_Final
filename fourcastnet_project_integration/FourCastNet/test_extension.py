import torch
import sys

sys.path.append("./afno_extension")

import afno_cuda

device = "cuda"

x = torch.randn(1024, device=device).contiguous()
y = afno_cuda.identity_forward(x)

diff = (x - y).abs()

print("x shape:", x.shape)
print("y shape:", y.shape)
print("max error:", diff.max().item())
print("allclose:", torch.allclose(x, y))
print("done")
