import time

import torch, torchvision
from thop import profile
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Dataloader_pavia import Datasat
from einops import rearrange, repeat
from tqdm.auto import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math, os, copy
from einops import rearrange
import numbers
import scipy.io as sio
from inspect import isfunction



# +--------------------------------
#       some global params
# +--------------------------------

ms_channels = 4
hs_channels = 102
layer = 4

"""

 interpretable fusion

"""

# +------------------------
#  InDNet，输入Xk，输出Mk
# +------------------------

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(int(inp), hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, int(oup), 1, bias=False),
            # nn.BatchNorm2d(oup),

        )

    def forward(self, x):
        x = self.bottleneckBlock(x)
        return x


class DetailNode(nn.Module):
    def __init__(self, channels):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=channels, oup=channels, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=channels, oup=channels, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=channels, oup=channels, expand_ratio=2)
        self.shffleconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1 = z1.to(torch.float32)
        z2 = z2.to(torch.float32)
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z3 = self.theta_phi(z1)
        z2 = z2 + z3
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class denoiser(nn.Module):
    def __init__(self, channels, num_layers=3):
        super(denoiser, self).__init__()
        INNmodules = [DetailNode(channels) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        self.conv = nn.Conv2d(hs_channels*2, hs_channels, 3, 1, 1)

    def forward(self, x):
        # x = torch.from_numpy(x)
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return self.conv(torch.cat((z1, z2), dim=1))


# +---------------------
#  CMUN
# +---------------------
class IFN_diff(nn.Module):
    def __init__(self, target_size, hs_ch, ms_ch):
        super(IFN_diff, self).__init__()
        # self.size = target_size
        self.R = nn.Conv2d(hs_ch, ms_ch, 1)
        self.RT = nn.Conv2d(ms_ch, hs_ch, 1)

        self.B = nn.Sequential(
            nn.Conv2d(hs_ch, hs_ch, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(hs_ch, hs_ch, 2, 2)
        )
        self.BT = nn.Sequential(
            nn.ConvTranspose2d(hs_ch, hs_ch, 2, 2),
            nn.PReLU(),
            nn.ConvTranspose2d(hs_ch, hs_ch, 3, 1, 1)
        )
        self.denoiser = denoiser(hs_ch)
        self.u = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def predict_M(self, hs, ms, xk, mk, u, beta):

        Qk1 = self.B(xk) - hs
        Qk2 = self.R(xk) - ms
        Uk1 = self.BT(Qk1)
        Uk2 = self.RT(Qk2)
        uk = xk - mk
        x_out = xk - u*(Uk1 + Uk2 + 2 * beta * uk)

        return x_out   #输出xk+1


    def forward(self, hs, ms):

        b_h, c_h, h_h, w_h = hs.shape
        b_m, c_m, h_m, w_m = ms.shape
        m0 = torch.zeros(size=(b_h, c_h, h_m, w_m)).to(device)
        x0 = torch.nn.functional.interpolate(hs, scale_factor=2, mode='bicubic')
        x1 = self.predict_M(hs, ms, x0, m0, self.u, self.beta)

        # start iteration
        # m1 = self.denoiser(x1)
        m1 = self.denoiser(torch.cat([x1, x1], dim=1))
        x2 = self.predict_M(hs, ms, x1, m1, self.u, self.beta)

        m2 = self.denoiser(torch.cat([x2, x2], dim=1))
        x3 = self.predict_M(hs, ms, x2, m2, self.u, self.beta)

        m3 = self.denoiser(torch.cat([x3, x3], dim=1))
        x4 = self.predict_M(hs, ms, x3, m3, self.u, self.beta)

        return x4

# +---------------------
#    P2MSRN
# +---------------------

class IFN_same(nn.Module):
    def __init__(self, hs_ch, ms_ch):
        super(IFN_same, self).__init__()

        self.R = nn.Conv2d(hs_ch, ms_ch, 1)
        self.RT = nn.Conv2d(ms_ch, hs_ch, 1)

        self.B = nn.Sequential(
            nn.Conv2d(hs_ch, hs_ch, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(hs_ch, hs_ch, 3, 1, 1)
        )
        self.BT = nn.Sequential(
            nn.ConvTranspose2d(hs_ch, hs_ch, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(hs_ch, hs_ch, 3, 1, 1)
        )
        self.denoiser = denoiser(hs_ch)
        self.u = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def predict_M(self, hs, ms, xk, mk, u, beta):

        Qk1 = self.B(xk) - hs
        Qk2 = self.R(xk) - ms
        Uk1 = self.BT(Qk1)
        Uk2 = self.RT(Qk2)
        uk = xk - mk
        x_out = xk - u*(Uk1 + Uk2 + 2 * beta * uk)

        return x_out   #输出xk+1


    def forward(self, xt, hs, ms):

        b_h, c_h, h_h, w_h = hs.shape
        b_m, c_m, h_m, w_m = ms.shape
        m0 = torch.zeros(size=(b_h, c_h, h_m, w_m)).to(ms.device)
        # x0 = torch.nn.functional.interpolate(hs, scale_factor=4, mode='bicubic')
        x0 = xt
        x1 = self.predict_M(hs, ms, x0, m0, self.u, self.beta)

        # start iteration
        m1 = self.denoiser(torch.cat([x1,x1], dim=1))
        x2 = self.predict_M(hs, ms, x1, m1, self.u, self.beta)

        m2 = self.denoiser(torch.cat([x2,x2], dim=1))
        x3 = self.predict_M(hs, ms, x2, m2, self.u, self.beta)

        m3 = self.denoiser(torch.cat([x3,x3], dim=1))
        x4 = self.predict_M(hs, ms, x3, m3, self.u, self.beta)

        return x4

# +--------------------------------
#   global information extractor
# +--------------------------------

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # x = torch.from_numpy(x)
        mu = torch.mean(x, dim=(1,), keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, hidden_features=64)

    def forward(self, x):
        # x = torch.from_numpy(x)
        x1 = self.attn(self.norm1(x))
        x = x + x1
        x2 = self.mlp(self.norm2(x))
        x = x + x2
        # x = np.concatenate(x, self.attn(self.norm1(x)))
        # x = np.concatenate(x, self.mlp(self.norm2(x)))
        return x


class ISPDiff(nn.Module):
    def __init__(self):
        super(ISPDiff, self).__init__()
        self.conv2 = nn.Conv2d(hs_channels * 2, hs_channels, 3, 1, 1)
        self.unet = UNet().to(device)
        self.ifn = IFN_same(hs_channels, ms_channels).to(device)


    def forward(self, ms, hs, xt, noise_level):

        hs_unet = self.unet(self.conv2(torch.cat([hs, xt], dim=1)), noise_level)

        out = self.ifn(hs_unet, hs, ms)


        return out



"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.att = att
        self.attn = SelfAtt(dim_out, num_heads=num_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        if self.att:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=hs_channels, out_channel=hs_channels, inner_channel=16, norm_groups=16,
                 channel_mults=[1, 2], res_blocks=2, dropout=0, img_size=160):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = img_size

        # Downsampling stage of U-net
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, noise_level):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats = []
        out = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
            out.append(x)
        x = self.final_conv(x)
        return x



"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=128, pyramid_list=[1, 0.5, 0.25, 0.125]):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device
        self.pyramid_list = pyramid_list
        self.ifn1 = IFN_diff(20, hs_channels, ms_channels).to(self.device)
        self.ifn2 = IFN_diff(40, hs_channels, ms_channels).to(self.device)


    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-3):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        # make downsampling schedule
        assert schedule_opt['n_timestep'] % len(self.pyramid_list) == 0
        segment_length = schedule_opt['n_timestep'] // len(self.pyramid_list)
        downsampling_schedule = []
        for downsampling_scale in self.pyramid_list:
            downsampling_schedule += [downsampling_scale] * segment_length
        downsampling_schedule = np.array(downsampling_schedule)
        downsampling_schedule = torch.from_numpy(downsampling_schedule).to(self.device)
        self.register_buffer('downsampling_schedule', downsampling_schedule)

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.register_buffer('pred_coef3', to_torch(np.sqrt(1. / 1 - alphas_cumprod)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, ms, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        pred_x = self.model(ms, condition_x, x, noise_level)
        in_coe = self.alphas_cumprod[t]

        noise = self.pred_coef3[t] * (x - torch.sqrt(in_coe) * pred_x)
        x_recon = self.predict_start(x_t=pred_x, t=t, noise=noise)

        x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, ms, clip_denoised=True, condition_x=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, ms=ms, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, x_in, ms):
        img = torch.rand_like(x_in, device=x_in.device)

        ms2 = torch.nn.functional.interpolate(ms, scale_factor=0.125)    # 8倍下采MS
        ms3 = torch.nn.functional.interpolate(ms, scale_factor=0.25)     # 4倍下采MS
        ms4 = torch.nn.functional.interpolate(ms, scale_factor=0.5)     # 2倍下采MS
        x_in2 = torch.nn.functional.interpolate(x_in, scale_factor=2)  # 2倍上采
        x_in4 = torch.nn.functional.interpolate(x_in, scale_factor=4)  # 4倍上采
        x_in8 = torch.nn.functional.interpolate(x_in, scale_factor=8)  # 8倍上采


        for i in reversed(range(0, self.num_timesteps)):
            # upsampling schedule
            if i == 1199:
                img = self.ifn1(img, ms3)
            elif i == 799:
                img = self.ifn2(img, ms4)
            elif i == 399:
                img = self.ifn2(img, ms)

            # elif 1600 <= i <= 1999:
            #     img = self.p_sample(img, i, ms1, condition_x=x_in)
            elif 1200 <= i <= 1599:
                img = self.p_sample(img, i, ms2, condition_x=x_in)
            elif 800 <= i <= 1199:
                img = self.p_sample(img, i, ms3, condition_x=x_in2)
            elif 400 <= i <= 799:
                img = self.p_sample(img, i, ms4, condition_x=x_in4)
            else:
                img = self.p_sample(img, i, ms, condition_x=x_in8)

        return img


    # Compute loss to train the model
    def p_losses(self, x_in, ms, gt): # x_in=lrHS

        t = np.random.randint(1, self.num_timesteps + 1)
        x_start = torch.nn.functional.interpolate(x_in, scale_factor=self.downsampling_schedule[t - 1].item()).to(
            device)
        ms = torch.nn.functional.interpolate(ms, scale_factor=self.downsampling_schedule[t - 1].item()).to(device)
        gt = torch.nn.functional.interpolate(gt, scale_factor=self.downsampling_schedule[t - 1].item()).to(device)
        b, c, h, w = x_start.shape
        noise = torch.randn_like(x_start).to(x_start.device)

        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha ** 2).sqrt() * noise
        pred_x = self.model(ms, x_start, x_noisy, noise_level=sqrt_alpha)

        return self.loss_func(gt, pred_x)

    def forward(self, ms, x, gt):
        return self.p_losses(x, ms, gt)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=False,
                 in_channel=102, out_channel=hs_channels, inner_channel=32, norm_groups=8,
                 channel_mults=(1, 2, 4, 8), res_blocks=2, dropout=0, lr=1e-5, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        model = ISPDiff()
        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel, pyramid_list=[1, 0.5, 0.25, 0.125])

        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        fixed_imgs = copy.deepcopy(next(iter(self.testloader)))
        fixed_imgs[1] = fixed_imgs[1].to(self.device)  #hs
        fixed_imgs[0] = fixed_imgs[0].to(self.device)  #ms

        best_loss = 100

        for i in tqdm(range(epoch)):
            train_loss = 0

            self.sr3.train()
            for _, (ms, hs, gt) in enumerate(self.dataloader):
                    # Initial imgs are high-resolution

                    ms = ms.to(self.device)
                    hs = torch.nn.functional.interpolate(hs, scale_factor=0.25)
                    hs = torch.nn.functional.interpolate(hs, scale_factor=4).to(device)
                    gt = gt.to(device)
                    b, c, h, w = gt.shape

                    self.optimizer.zero_grad()
                    loss = self.sr3(ms, hs, gt)
                    loss = (loss.sum() / int(b * c * h * w))
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * b
            if i % 1 == 0:
                if train_loss <= best_loss:
                    best_loss = train_loss
                    # Save model weight
                    self.save(self.save_path)

            if (i + 1) % verbose == 0:
                    test_imgs = next(iter(self.testloader))
                    gt = test_imgs[2].to(device)
                    test_imgs[0] = test_imgs[0].to(device)
                    test_imgs[1] = test_imgs[1].to(device)
                    b, c, h, w = test_imgs[1].shape

                    train_loss = train_loss / len(self.dataloader)
                    print(f'Epoch: {i + 1} / loss:{train_loss:.6f}')

                    plt.figure(figsize=(15, 10))
                    plt.subplot(1, 3, 1)
                    plt.axis("off")
                    plt.title("model3 LR")
                    plt.imshow(np.transpose(torchvision.utils.make_grid(fixed_imgs[1],
                                                                        nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0))[:, :, [10, 15, 20]])

                    plt.subplot(1, 3, 2)
                    plt.axis("off")
                    plt.title("model3 GT")
                    plt.imshow(np.transpose(torchvision.utils.make_grid(gt.detach().cpu(),
                                                                        nrow=2, padding=1, normalize=True).cpu(),
                                            (1, 2, 0))[:, :, [10, 15, 20]])

                    plt.subplot(1, 3, 3)
                    plt.axis("off")
                    plt.title("model3 SR")
                    test_imgs[1] = torch.nn.functional.interpolate(test_imgs[1], scale_factor=0.25)

                    test = self.test(torch.nn.functional.interpolate(test_imgs[1], scale_factor=0.5),test_imgs[0])
                    test = torch.squeeze(test)
                    plt.imshow(np.transpose(torchvision.utils.make_grid(test.detach().cpu(),
                                                                        nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,[10, 15, 20]])
                    # plt.savefig('SuperResolution_Result_hs.jpg')
                    plt.show()
                    # plt.close()

    def test(self, imgs, ms):
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs, ms)
            else:
                result_SR = self.sr3.super_resolution(imgs, ms)
        self.sr3.train()
        return result_SR

    def save(self, save_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

if __name__ == "__main__":

    batch_size = 4
    data = Datasat('train')
    testdata = Datasat('test')
    dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testdata, batch_size=1, shuffle=False)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if cuda else "cpu")
    schedule_opt = {'schedule': 'cosine', 'n_timestep': 1600, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=dataloader, testloader=testloader, schedule_opt=schedule_opt,
              save_path='./model/pavia_k_random.pt',
              load_path='./model/pavia_k_random.pt',
              load=False, inner_channel=32,
              norm_groups=16, channel_mults=(1, 2, 4, 8), dropout=0.2, res_blocks=2, lr=1e-4, distributed=False)
    sr3.train(epoch=1500, verbose=15)

    # +-----------------
    #       test
    # +-----------------
    # total_time = 0
    # for i, (MS, HS, GT) in enumerate(testloader):
    #     with torch.no_grad():
    #         print('+----------------- 测试第{}张图 -----------------+'.format(i + 1))
    #         start = time.time()
    #         HS = torch.nn.functional.interpolate(HS, scale_factor=0.125).to(device)
    #         MS = MS.to(device)
    #         # HS = gaussian_blur_4d_tensor(HS)
    #         # HS = torch.nn.functional.i+nterpolate(HS, scale_factor=0.25).to(device)
    #         # gt = gt.to(device)
    #         # HS = torch.nn.functional.interpolate(HS, scale_factor=4)
    #         test_out = sr3.test(imgs=HS, ms=MS)
    #         test_out = test_out.detach().cpu().numpy()
    #         sio.savemat(
    #             './%d.mat' % (i + 1),
    #             {'hs': test_out})
    #         end = time.time()
    #         time_use = end - start
    #         total_time = total_time + time_use
    #         print('test complete!    total time: {}'.format(end - start))
    #         print('total_mul time: {}'.format(total_time))
    #
    # print("total_aver_time = {}".format(total_time / 63))

