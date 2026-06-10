#reference: https://github.com/NVlabs/AFNO-transformer

import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.img_utils import PeriodicPad2d

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        # residual
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        
        if not hasattr(self, "_debug_printed"):
            print("AFNO input shape:", x.shape)
            print("num_blocks:", self.num_blocks)
            print("block_size:", self.block_size)
            print("w1 shape:", self.w1.shape)
            print("w2 shape:", self.w2.shape)
            print("hard_thresholding_fraction:", self.hard_thresholding_fraction)
            print("total_modes:", H // 2 + 1)
            print("kept_modes:", int((H // 2 + 1) * self.hard_thresholding_fraction))
            self._debug_printed = True
            
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # [1, 90, 91, 768] -> [1, 90, 91, 8, 96]
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias



class AFNO2DSeparableBMM(AFNO2D):
    """Separable FFT + batched GEMM AFNO path for faster FP32 inference.

    This keeps the original checkpoint and model math for retained modes. It
    avoids unused width-frequency modes by doing rFFT over W, truncating to the
    modes AFNO actually consumes, then doing FFT over H only on that compact
    spectrum. The complex spectral MLP is implemented as one augmented real BMM
    per layer and cached in eval mode.
    """

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__(hidden_size, num_blocks, sparsity_threshold, hard_thresholding_fraction, hidden_size_factor)
        self._bmm_cache = None
        self._bmm_cache_key = None

    @torch.no_grad()
    def prepare_inference_cache(self):
        self._augmented_weights()

    def _forward_augmented_weights(self):
        if (not self.training) and self._bmm_cache is not None:
            return self._bmm_cache
        return self._augmented_weights()

    @torch._dynamo.disable
    def _augmented_weights(self):
        key = (
            self.w1.device,
            self.w1.dtype,
            self.w1._version,
            self.b1._version,
            self.w2._version,
            self.b2._version,
        )
        if (not self.training) and self._bmm_cache is not None and self._bmm_cache_key == key:
            return self._bmm_cache

        w1r, w1i = self.w1[0], self.w1[1]
        w2r, w2i = self.w2[0], self.w2[1]
        w1_aug = torch.cat(
            [
                torch.cat([w1r, w1i], dim=-1),
                torch.cat([-w1i, w1r], dim=-1),
            ],
            dim=-2,
        ).contiguous()
        w2_aug = torch.cat(
            [
                torch.cat([w2r, w2i], dim=-1),
                torch.cat([-w2i, w2r], dim=-1),
            ],
            dim=-2,
        ).contiguous()
        b1_aug = torch.cat([self.b1[0], self.b1[1]], dim=-1).contiguous()
        b2_aug = torch.cat([self.b2[0], self.b2[1]], dim=-1).contiguous()
        cache = (w1_aug, b1_aug, w2_aug, b2_aug)

        if not self.training:
            self._bmm_cache = cache
            self._bmm_cache_key = key
        return cache

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        h_start = max(total_modes - kept_modes, 0)
        h_end = min(total_modes + kept_modes, H)
        h_kept = h_end - h_start

        x_w = torch.fft.rfft(x, dim=2, norm="ortho")[:, :, :kept_modes]
        x_fft = torch.fft.fft(x_w, dim=1, norm="ortho")
        x_fft = x_fft.reshape(B, H, kept_modes, self.num_blocks, self.block_size)
        x_keep = x_fft[:, h_start:h_end]

        xr = x_keep.real.permute(3, 0, 1, 2, 4).reshape(self.num_blocks, -1, self.block_size)
        xi = x_keep.imag.permute(3, 0, 1, 2, 4).reshape(self.num_blocks, -1, self.block_size)
        x_aug = torch.cat([xr, xi], dim=-1)

        w1_aug, b1_aug, w2_aug, b2_aug = self._forward_augmented_weights()
        o1 = F.relu(torch.bmm(x_aug, w1_aug) + b1_aug.unsqueeze(1))
        o2 = torch.bmm(o1, w2_aug) + b2_aug.unsqueeze(1)
        o2 = F.softshrink(o2, lambd=self.sparsity_threshold)

        out_real = o2[..., :self.block_size]
        out_imag = o2[..., self.block_size:]
        out = torch.complex(out_real, out_imag)
        out = out.reshape(self.num_blocks, B, h_kept, kept_modes, self.block_size)
        out = out.permute(1, 2, 3, 0, 4)

        if h_start == 0 and h_kept == H:
            x_h = out.reshape(B, H, kept_modes, C)
        else:
            x_h_full = torch.zeros_like(x_fft)
            x_h_full[:, h_start:h_end] = out
            x_h = x_h_full.reshape(B, H, kept_modes, C)

        x_w_filtered = torch.fft.ifft(x_h, dim=1, norm="ortho")
        x = torch.fft.irfft(x_w_filtered, n=W, dim=2, norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            afno2d_impl='original'
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if afno2d_impl == 'sep_bmm':
            self.filter = AFNO2DSeparableBMM(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        elif afno2d_impl == 'original':
            self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        else:
            raise ValueError("Unknown afno2d_impl '{}'. Use 'original' or 'sep_bmm'.".format(afno2d_impl))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AFNONet(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks 
        self.afno2d_impl = getattr(params, 'afno2d_impl', 'original')
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction,
            afno2d_impl=self.afno2d_impl) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        B, h, w, _ = x.shape
        p1, p2 = self.patch_size
        x = x.reshape(B, h, w, p1, p2, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, h * p1, w * p2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    model = AFNONet(img_size=(720, 1440), patch_size=(4,4), in_chans=20, out_chans=20)
    sample = torch.randn(1, 20, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
