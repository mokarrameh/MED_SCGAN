r"""
The authors' re-implementation of SPA GAN [40] (an approximation), used as a comparison model in the paper.

This is not the official SPA GAN code (Emami et al., "SPA-GAN: Spatial Attention GAN for
Image-to-Image Translation", IEEE Trans. Multimedia, 2021). The published SPA GAN computes the
attention map in its discriminator and adds a feature-map loss. Here, the generator predicts
an attention mask A(x) \in [0,1]^{H×W} indicating *where* to apply the translation, then
blends the translated content T(x) with the original input x:

    y_hat = A * T(x) + (1 - A) * x

Losses: adversarial (LSGAN), cycle-consistency, identity, attention sparsity
(L1 on A), and attention smoothness (total variation on A).

Usage (sketch):
---------------
1) Construct two generators (A→B, B→A) and two PatchGAN discriminators.
2) Train with unpaired minibatches from domain A and B (see spa_train.py).

Requirements: torch>=2.0, torchvision
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict

# ------------------------------
# Building blocks
# ------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(dim, dim, k=3, s=1, p=1),
            ConvBlock(dim, dim, k=3, s=1, p=1, activation=False),
        )

    def forward(self, x):
        return x + self.block(x)

class SpatialAttention(nn.Module):
    """Predict per-pixel attention mask A in [0,1].

    We condition A on mid-level features and a low-level summary of the input
    image to encourage alignment with content.
    """
    def __init__(self, in_c_feat: int, in_c_img: int = 3, mid: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBlock(in_c_feat + in_c_img, mid, k=3, s=1, p=1),
            ConvBlock(mid, mid, k=3, s=1, p=1),
            nn.Conv2d(mid, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, x_img):
        # Resize image to feature map size for fusion
        x_resized = F.interpolate(x_img, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([feat, x_resized], dim=1))

# ------------------------------
# Generators with attention
# ------------------------------

class AttnResnetGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            ConvBlock(ngf, ngf*2, k=3, s=2, p=1),  # 1/2
            ConvBlock(ngf*2, ngf*4, k=3, s=2, p=1),  # 1/4
        )
        # Bottleneck
        blocks = [ResnetBlock(ngf*4) for _ in range(n_blocks)]
        self.res = nn.Sequential(*blocks)
        # Decoder to translated image T(x)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, out_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )
        # Attention head on bottleneck features
        self.attn = SpatialAttention(in_c_feat=ngf*4, in_c_img=in_nc, mid=ngf)

    def forward(self, x):
        f = self.enc(x)
        f = self.res(f)
        T = self.dec(f)            # translated proposal
        A = self.attn(f, x)        # spatial mask in [0,1]

        if T.shape[2:] != x.shape[2:]:
            T = torch.nn.functional.interpolate(T, size=x.shape[2:], mode="bilinear", align_corners=False)

        if A.shape[2:] != x.shape[2:]:
            A = torch.nn.functional.interpolate(A, size=x.shape[2:], mode="bilinear", align_corners=False)
        y = A * T + (1.0 - A) * x  # blend
        return y, A, T

# ------------------------------
# PatchGAN Discriminator
# ------------------------------

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4; padw = 1
        sequence = [
            nn.Conv2d(in_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

# ------------------------------
# Losses
# ------------------------------

class GANLoss(nn.Module):
    """LSGAN (least-squares) adversarial loss."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target_is_real: bool):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse(pred, target)

def total_variation(x):
    """Total variation regularization for smooth attention maps."""
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw

@dataclass
class SpaGanLossWeights:
    gan: float = 1.0
    cycle: float = 10.0
    identity: float = 0.5
    attn_sparsity: float = 1.0
    attn_tv: float = 1.0

@dataclass
class SpaGanBatch:
    a: torch.Tensor  # images from domain A, normalized to [-1, 1]
    b: torch.Tensor  # images from domain B, normalized to [-1, 1]

class SpaGanModel(nn.Module):
    """Wraps two generators and two discriminators for A↔B translation."""
    def __init__(self, img_nc=3, ngf=64, ndf=64, n_res_blocks=9, device: str = "cuda"):
        super().__init__()
        self.G_AB = AttnResnetGenerator(img_nc, img_nc, ngf, n_res_blocks).to(device)
        self.G_BA = AttnResnetGenerator(img_nc, img_nc, ngf, n_res_blocks).to(device)
        self.D_A = NLayerDiscriminator(img_nc, ndf).to(device)
        self.D_B = NLayerDiscriminator(img_nc, ndf).to(device)
        self.crit_gan = GANLoss()
        self.l1 = nn.L1Loss()
        self.device = device

    def setup_optim(self, lr=2e-4, betas=(0.5, 0.999)):
        self.opt_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=lr, betas=betas
        )
        self.opt_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()), lr=lr, betas=betas
        )

    def forward_G(self, batch: SpaGanBatch) -> Dict[str, torch.Tensor]:
        a, b = batch.a, batch.b
        b_fake, a_attn_AB, _ = self.G_AB(a)
        a_recon, a_attn_cycle, _ = self.G_BA(b_fake)
        a_fake, b_attn_BA, _ = self.G_BA(b)
        b_recon, b_attn_cycle, _ = self.G_AB(a_fake)
        return {
            "b_fake": b_fake, "a_recon": a_recon,
            "a_fake": a_fake, "b_recon": b_recon,
            "A_AB": a_attn_AB, "A_cycle_a": a_attn_cycle,
            "B_BA": b_attn_BA, "A_cycle_b": b_attn_cycle,
        }

    def generator_loss(self, out: Dict[str, torch.Tensor], batch: SpaGanBatch, w: SpaGanLossWeights):
        a, b = batch.a, batch.b
        # GAN loss (G wants D to see fakes as real)
        gan_B = self.crit_gan(self.D_B(out["b_fake"]), True)
        gan_A = self.crit_gan(self.D_A(out["a_fake"]), True)
        # Cycle-consistency
        cycle_a = self.l1(out["a_recon"], a)
        cycle_b = self.l1(out["b_recon"], b)
        # Identity losses (optional but stabilizing)
        idt_b, A_idt_b, _ = self.G_AB(b)
        idt_a, A_idt_a, _ = self.G_BA(a)
        idt_loss = self.l1(idt_b, b) + self.l1(idt_a, a)
        # Attention regularization: sparsity + smoothness
        attn_maps = [out["A_AB"], out["B_BA"], out["A_cycle_a"], out["A_cycle_b"], A_idt_a, A_idt_b]
        attn_sparsity = sum(m.abs().mean() for m in attn_maps) / len(attn_maps)
        attn_tv = sum(total_variation(m) for m in attn_maps) / len(attn_maps)
        # Weighted sum
        loss_G = (
            w.gan * (gan_A + gan_B) +
            w.cycle * (cycle_a + cycle_b) +
            w.identity * idt_loss +
            w.attn_sparsity * attn_sparsity +
            w.attn_tv * attn_tv
        )
        loss_dict = {
            "G_total": loss_G.detach(),
            "gan": (gan_A + gan_B).detach(),
            "cycle": (cycle_a + cycle_b).detach(),
            "identity": idt_loss.detach(),
            "attn_sparsity": attn_sparsity.detach(),
            "attn_tv": attn_tv.detach(),
        }
        return loss_G, loss_dict

    def discriminator_loss(self, real: torch.Tensor, fake_detached: torch.Tensor, D: nn.Module):
        pred_real = D(real)
        pred_fake = D(fake_detached)
        loss_real = self.crit_gan(pred_real, True)
        loss_fake = self.crit_gan(pred_fake, False)
        return 0.5 * (loss_real + loss_fake)

    def training_step(self, batch: SpaGanBatch, w: SpaGanLossWeights) -> Dict[str, float]:
        # --- update G ---
        self.opt_G.zero_grad(set_to_none=True)
        out = self.forward_G(batch)
        loss_G, loss_g_dict = self.generator_loss(out, batch, w)
        loss_G.backward()
        self.opt_G.step()
        # --- update D ---
        self.opt_D.zero_grad(set_to_none=True)
        loss_D_A = self.discriminator_loss(batch.a, out["a_fake"].detach(), self.D_A)
        loss_D_B = self.discriminator_loss(batch.b, out["b_fake"].detach(), self.D_B)
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.opt_D.step()
        # metrics
        return {
            **{k: v.detach() for k, v in loss_g_dict.items()},
            "D_total": loss_D.detach(),
            "D_A": loss_D_A.detach(),
            "D_B": loss_D_B.detach(),
            "A_mean_AB": out["A_AB"].mean().detach(),
            "A_mean_BA": out["B_BA"].mean().detach(),
            "real_a": batch.a.detach().cpu(),
            "real_b": batch.b.detach().cpu(),
            "fake_a": out["a_fake"].detach().cpu(),
            "fake_b": out["b_fake"].detach().cpu()
        }

# ------------------------------
# Utilities
# ------------------------------

def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] for visualization"""
    return (x.clamp(-1, 1) + 1.0) / 2.0
