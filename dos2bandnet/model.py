#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== utils =====================
def sinusoidal_time_emb(timesteps: torch.Tensor, dim: int, T_total: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    t = timesteps.float() / max(1, (T_total - 1))
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1: emb = F.pad(emb, (0, 1))
    return emb

def map01(x): return (x.clamp(-1, 1) + 1.0) * 0.5


# ===================== VAE =====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_p: float = 0.0):
        super().__init__()
        Drop = nn.Identity if drop_p <= 0 else (lambda: nn.Dropout2d(drop_p))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(), Drop(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(), Drop(),
        )
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, base=64, z_ch=4, down=4, drop_p: float = 0.0):
        super().__init__()
        ch1 = base; ch2 = base * 2
        self.b1 = ConvBlock(in_ch, ch1, drop_p); self.d1 = nn.Conv2d(ch1, ch1, 4, stride=2, padding=1)
        self.b2 = ConvBlock(ch1, ch2, drop_p);   self.d2 = nn.Conv2d(ch2, ch2, 4, stride=2, padding=1)
        assert down == 4, "down must be 4"
        self.out_mu = nn.Conv2d(ch2, z_ch, 1); self.out_logvar = nn.Conv2d(ch2, z_ch, 1)
    def forward(self, x):
        h = self.b1(x); h = self.d1(h)
        h = self.b2(h); h = self.d2(h)
        return self.out_mu(h), self.out_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, base=64, z_ch=4, drop_p: float = 0.0):
        super().__init__()
        ch2 = base * 2; ch1 = base
        self.inp = nn.Conv2d(z_ch, ch2, 1); self.b2 = ConvBlock(ch2, ch2, drop_p)
        self.u2 = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.b1 = ConvBlock(ch1, ch1, drop_p)
        self.u1 = nn.ConvTranspose2d(ch1, ch1, 4, stride=2, padding=1)
        self.out = nn.Sequential(nn.GroupNorm(8, ch1), nn.SiLU(), nn.Conv2d(ch1, out_ch, 3, padding=1), nn.Tanh())
    def forward(self, z):
        h = self.inp(z); h = self.b2(h); h = self.u2(h); h = self.b1(h); h = self.u1(h)
        return self.out(h)

class SimpleVAE(nn.Module):
    def __init__(self, in_ch=1, base=64, z_ch=4, down=4, drop_p: float = 0.0):
        super().__init__()
        self.enc = Encoder(in_ch, base, z_ch, down, drop_p); self.dec = Decoder(in_ch, base, z_ch, drop_p)
    def encode(self, x, sample_posterior=False):
        mu, logvar = self.enc(x)
        if sample_posterior:
            std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); z = mu + std * eps
        else: z = mu
        return z, (mu, logvar)
    def decode(self, z): return self.dec(z)
    def forward(self, x, sample_posterior: bool = True):
        z, (mu, logvar) = self.encode(x, sample_posterior=sample_posterior)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, kl_beta=1e-3):
    rec = F.l1_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl_beta * kl, rec, kl


# ===================== 1D Encoders =====================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, drop_p: float = 0.0):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Identity() if drop_p <= 0 else nn.Dropout(drop_p)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Identity() if drop_p <= 0 else nn.Dropout(drop_p)
        self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else None
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True); h = self.drop1(h)
        h = self.bn2(self.conv2(h)); h = self.drop2(h)
        if self.proj is not None: x = self.proj(x)
        return F.relu(x + h, inplace=True)

def sinusoidal_posenc_1d(L: int, pe_dim: int, device):
    n = pe_dim // 2
    pos = torch.linspace(0, 1, L, device=device)
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), n, device=device))
    args = pos[None, :] * freqs[:, None] * (2 * math.pi)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=0)

class SimpleEnc1D(nn.Module):
    def __init__(self, in_ch=1, c1=32, c2=64, k1=7, k2=5, latent=128, out_dim=128, drop_p: float = 0.0):
        super().__init__()
        Drop = nn.Identity if drop_p <= 0 else (lambda: nn.Dropout(drop_p))
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, c1, k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c1, c1, k2, padding=k2 // 2), nn.BatchNorm1d(c1), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c1, c1, k2, padding=k2 // 2), nn.BatchNorm1d(c1), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c1, c1, k2, padding=k2 // 2), nn.BatchNorm1d(c1), nn.ReLU(inplace=True), Drop(),
            nn.AvgPool1d(2)
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(c1, c2, k2, padding=(k2 // 2) * 2, dilation=2), nn.BatchNorm1d(c2), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c2, c2, 3, padding=2, dilation=2), nn.BatchNorm1d(c2), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c2, c2, 3, padding=2, dilation=2), nn.BatchNorm1d(c2), nn.ReLU(inplace=True), Drop(),
            nn.Conv1d(c2, c2, 3, padding=2, dilation=2), nn.BatchNorm1d(c2), nn.ReLU(inplace=True), Drop(),
            nn.AvgPool1d(2)
        )
        self.proj = nn.Conv1d(c2, latent, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(latent, out_dim), nn.ReLU(inplace=True), Drop())
    def forward(self, x):
        h = self.b1(x); h = self.b2(h); z = self.proj(h); return self.head(z)

class PeakAware1DEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32, out_dim=128, pe_dim=16, drop_p: float = 0.0):
        super().__init__()
        self.pe_dim = pe_dim; c_in = in_ch + pe_dim
        self.stem = nn.Sequential(nn.Conv1d(c_in, base, 7, padding=3), nn.BatchNorm1d(base), nn.ReLU(inplace=True),
                                  nn.Dropout(drop_p) if drop_p > 0 else nn.Identity())
        self.l1 = ResBlock1D(base, base, stride=2, drop_p=drop_p)
        self.l2 = ResBlock1D(base, base * 2, stride=2, drop_p=drop_p)
        self.l3 = ResBlock1D(base * 2, base * 2, dilation=2, drop_p=drop_p)
        self.l4 = ResBlock1D(base * 2, base * 2, dilation=4, drop_p=drop_p)
        self.attn = nn.Conv1d(base * 2, 1, 1); self.fc = nn.Sequential(nn.Linear(base * 2, out_dim), nn.ReLU(inplace=True),
                                                                        nn.Dropout(drop_p) if drop_p > 0 else nn.Identity())
    def forward(self, x):
        B, _, L = x.shape
        pe = sinusoidal_posenc_1d(L, self.pe_dim, x.device).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, pe], dim=1)
        h = self.stem(x); h = self.l1(h); h = self.l2(h); h = self.l3(h); h = self.l4(h)
        w = torch.softmax(self.attn(h), dim=-1); feat = torch.sum(h * w, dim=-1)
        return self.fc(feat)

class Simple1DLight(nn.Module):
    def __init__(self, in_ch=1, base=32, out_dim=128, drop_p: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(base, base, 5, padding=2), nn.ReLU(inplace=True),
            nn.Dropout(drop_p) if drop_p > 0 else nn.Identity(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1); self.fc = nn.Sequential(nn.Linear(base, out_dim), nn.ReLU(inplace=True),
                                                                     nn.Dropout(drop_p) if drop_p > 0 else nn.Identity())
    def forward(self, x):
        h = self.stem(x); g = self.gap(h).squeeze(-1); return self.fc(g)

def build_cond_encoder(kind: str, in_ch: int = 1, out_dim: int = 128, drop_p: float = 0.0):
    k = (kind or "deep").lower()
    if k == "wide":   return SimpleEnc1D(in_ch=in_ch, out_dim=out_dim, drop_p=drop_p)
    if k == "deep":   return PeakAware1DEncoder(in_ch=in_ch, out_dim=out_dim, pe_dim=8, drop_p=drop_p)
    if k == "simple": return Simple1DLight(in_ch=in_ch, out_dim=out_dim, drop_p=drop_p)
    raise ValueError(f"Unknown encoder kind: {kind}")


# ===================== UNet (latent) =====================
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, num_channels); self.to_shift = nn.Linear(cond_dim, num_channels)
    def forward(self, h, cond):
        s = self.to_scale(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + s) + b

class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, num_groups=8, drop_p: float = 0.0):
        super().__init__()
        assert in_ch % num_groups == 0 and out_ch % num_groups == 0
        self.n1 = nn.GroupNorm(num_groups, in_ch); self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(num_groups, out_ch); self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.f1 = FiLM(cond_dim, in_ch); self.f2 = FiLM(cond_dim, out_ch)
        self.drop1 = nn.Identity() if drop_p <= 0 else nn.Dropout2d(drop_p)
        self.drop2 = nn.Identity() if drop_p <= 0 else nn.Dropout2d(drop_p)
    def forward(self, x, cond):
        h = F.silu(self.f1(self.n1(x), cond)); h = self.c1(h); h = self.drop1(h)
        h2 = F.silu(self.f2(self.n2(h), cond)); h2 = self.c2(h2); h2 = self.drop2(h2)
        if self.proj is not None: x = self.proj(x)
        return x + h2

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, drop_p: float = 0.0):
        super().__init__()
        self.b1 = ResBlock2D(in_ch, out_ch, cond_dim, drop_p=drop_p); self.b2 = ResBlock2D(out_ch, out_ch, cond_dim, drop_p=drop_p)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x, cond):
        x = self.b1(x, cond); x = self.b2(x, cond); skip = x; x = self.pool(x); return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, drop_p: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.b1 = ResBlock2D(out_ch + skip_ch, out_ch, cond_dim, drop_p=drop_p); self.b2 = ResBlock2D(out_ch, out_ch, cond_dim, drop_p=drop_p)
    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1); x = self.b1(x, cond); x = self.b2(x, cond); return x

class UNetLatent(nn.Module):
    def __init__(self, in_ch=4, base=64, cond_dim=256 + 128, T_total: int = 500, drop_p: float = 0.0):
        super().__init__()
        self.T_total = T_total
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = Down(base, base, cond_dim, drop_p=drop_p)
        self.down2 = Down(base, base * 2, cond_dim, drop_p=drop_p)
        self.mid1 = ResBlock2D(base * 2, base * 4, cond_dim, drop_p=drop_p)
        self.mid2 = ResBlock2D(base * 4, base * 2, cond_dim, drop_p=drop_p)
        self.up2 = Up(base * 2, base * 2, base, cond_dim, drop_p=drop_p)
        self.up1 = Up(base, base, base, cond_dim, drop_p=drop_p)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, padding=1))
        self.tproj = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x, t, cond_feat):
        t_emb = sinusoidal_time_emb(t, 128, self.T_total)
        tfe = self.tproj(t_emb); cond = torch.cat([cond_feat, tfe], dim=-1)
        x = self.in_conv(x); x, s1 = self.down1(x, cond); x, s2 = self.down2(x, cond)
        x = self.mid1(x, cond); x = self.mid2(x, cond); x = self.up2(x, s2, cond); x = self.up1(x, s1, cond)
        return self.out(x)

class LatentDiffusionModel(nn.Module):
    def __init__(self, vae: SimpleVAE, cond_dim: int = 256, z_ch: int = 4, model_ch: int = 64, T_total: int = 500,
                 drop_p: float = 0.0, cond_drop_prob: float = 0.0):
        super().__init__()
        self.vae = vae
        self.cond_dim = cond_dim
        self.cond_drop_prob = float(cond_drop_prob)
        self.enc_dos = None; self.enc_ed = None
        self.unet = UNetLatent(in_ch=z_ch, base=model_ch, cond_dim=cond_dim + 128, T_total=T_total, drop_p=drop_p)
    def set_encoders(self, enc_dos: nn.Module, enc_ed: Optional[nn.Module]):
        self.enc_dos = enc_dos; self.enc_ed = enc_ed
    def make_cond(self, dos_1d: torch.Tensor, ed_1d: torch.Tensor) -> torch.Tensor:
        f_dos = self.enc_dos(dos_1d)
        if self.enc_ed is None:
            ed_dim = max(0, int(self.cond_dim) - int(f_dos.size(-1)))
            f_ed = torch.zeros(f_dos.size(0), ed_dim, device=f_dos.device, dtype=f_dos.dtype)
        else:
            f_ed = self.enc_ed(ed_1d)
        cond = torch.cat([f_dos, f_ed], dim=-1)
        # classifier-free guidance style cond dropout (during training only)
        if self.training and self.cond_drop_prob > 0.0:
            mask = (torch.rand(cond.size(0), device=cond.device) < self.cond_drop_prob).float().unsqueeze(1)
            cond = cond * (1.0 - mask)  # zero-out some conditioning vectors
        return cond
    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor, dos_1d: torch.Tensor, ed_1d: torch.Tensor, uncond: bool = False):
        if uncond:
            cond = torch.zeros(z_noisy.size(0), self.cond_dim, device=z_noisy.device, dtype=z_noisy.dtype)
        else:
            cond = self.make_cond(dos_1d, ed_1d)
        return self.unet(z_noisy, t, cond)

class SimpleBandModel(nn.Module):
    """
    Simple baseline model that embeds DOS/ED conditions and directly predicts the band map with an MLP.
    """
    def __init__(self, cond_dim: int = 256, model_ch: int = 64, band_shape: Tuple[int, int] = (96, 256),
                 drop_p: float = 0.0, cond_drop_prob: float = 0.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_drop_prob = float(cond_drop_prob)
        self.enc_dos = None; self.enc_ed = None
        self.band_shape = tuple(band_shape)
        out_dim = int(self.band_shape[0] * self.band_shape[1])
        hid = max(128, int(model_ch) * 4)
        Drop = nn.Identity if drop_p <= 0 else (lambda: nn.Dropout(drop_p))
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hid), nn.SiLU(), Drop(),
            nn.Linear(hid, hid), nn.SiLU(), Drop(),
            nn.Linear(hid, out_dim), nn.Tanh(),
        )

    def set_encoders(self, enc_dos: nn.Module, enc_ed: Optional[nn.Module]):
        self.enc_dos = enc_dos; self.enc_ed = enc_ed

    def make_cond(self, dos_1d: torch.Tensor, ed_1d: torch.Tensor) -> torch.Tensor:
        f_dos = self.enc_dos(dos_1d)
        if self.enc_ed is None:
            ed_dim = max(0, int(self.cond_dim) - int(f_dos.size(-1)))
            f_ed = torch.zeros(f_dos.size(0), ed_dim, device=f_dos.device, dtype=f_dos.dtype)
        else:
            f_ed = self.enc_ed(ed_1d)
        cond = torch.cat([f_dos, f_ed], dim=-1)
        if self.training and self.cond_drop_prob > 0.0:
            mask = (torch.rand(cond.size(0), device=cond.device) < self.cond_drop_prob).float().unsqueeze(1)
            cond = cond * (1.0 - mask)
        return cond

    def forward(self, dos_1d: torch.Tensor, ed_1d: torch.Tensor, uncond: bool = False):
        if uncond:
            cond = torch.zeros(dos_1d.size(0), self.cond_dim, device=dos_1d.device, dtype=dos_1d.dtype)
        else:
            cond = self.make_cond(dos_1d, ed_1d)
        y = self.net(cond)
        nk, nE = self.band_shape
        return y.view(cond.size(0), 1, nk, nE)

# ===================== Diffusion schedule =====================
@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

@torch.no_grad()
def make_schedule(T: int, beta_start: float, beta_end: float, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = F.pad(abar[:-1], (1, 0), value=1.0)
    post_var = betas * (1.0 - abar_prev) / (1.0 - abar)
    return DiffusionSchedule(
        betas=betas, alphas=alphas, alphas_cumprod=abar, alphas_cumprod_prev=abar_prev,
        sqrt_alphas_cumprod=torch.sqrt(abar),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - abar),
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        posterior_variance=post_var.clamp_min(1e-20)
    )

@torch.no_grad()
def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule, noise: Optional[torch.Tensor] = None):
    if noise is None: noise = torch.randn_like(x0)
    sqrt_cum = sched.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_1mc = sched.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return sqrt_cum * x0 + sqrt_1mc * noise, noise


# ===================== EMA =====================
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_((self.decay)).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        state = model.state_dict()
        for k in self.shadow.keys():
            state[k].copy_(self.shadow[k])


# ===================== DDIM =====================
@torch.no_grad()
def ddim_timesteps(T: int, S: int):
    ts = np.linspace(T - 1, 0, S, dtype=np.int64).tolist()
    uniq = []
    for t in ts:
        if len(uniq) == 0 or t < uniq[-1]: uniq.append(int(t))
    return uniq

@torch.no_grad()
def sample_ddim(ldm, dos, ed, sched: DiffusionSchedule, device,
                band_shape: Tuple[int, int], downsample: int, z_ch: int,
                steps: int = 50, eta: float = 0.0, guidance: float = 1.0,
                latent_scale: float = 1.0, predict_type: str = "v"):
    B = dos.size(0)
    nk, nE = band_shape
    H = nk // downsample; W = nE // downsample
    z_t = torch.randn(B, z_ch, H, W, device=device)

    t_list = ddim_timesteps(len(sched.betas), steps)
    for i, t_scalar in enumerate(t_list):
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        if guidance != 1.0:
            v_uncond = ldm(z_t, t, dos, ed, uncond=True)
            v_cond   = ldm(z_t, t, dos, ed, uncond=False)
            v_hat    = v_uncond + guidance * (v_cond - v_uncond)
        else:
            v_hat = ldm(z_t, t, dos, ed, uncond=False)

        alpha_t = sched.sqrt_alphas_cumprod[t_scalar]
        sigma_t = sched.sqrt_one_minus_alphas_cumprod[t_scalar]
        if predict_type.lower() in ("v", "v_l1"):
            x0_hat = alpha_t * z_t - sigma_t * v_hat
            eps_hat= sigma_t * z_t + alpha_t * v_hat
        else:
            eps_hat = v_hat
            x0_hat  = (z_t - sigma_t * eps_hat) / (alpha_t + 1e-12)

        if i == len(t_list) - 1:
            z_t = x0_hat
            break

        t_prev   = t_list[i + 1]
        abar_t   = sched.alphas_cumprod[t_scalar]
        abar_prev= sched.alphas_cumprod[t_prev]
        sigma_ddim = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
        noise = torch.randn_like(z_t) if eta > 0 else 0.0
        z_t = torch.sqrt(abar_prev) * x0_hat + torch.sqrt(1 - abar_prev - sigma_ddim ** 2) * eps_hat + sigma_ddim * noise

    z = z_t / latent_scale
    band_hat = ldm.vae.decode(z).clamp(-1, 1)
    return (band_hat + 1.0) / 2.0
