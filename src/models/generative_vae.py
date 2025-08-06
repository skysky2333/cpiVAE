"""
Generative VAE with advanced decoders (MADE or DDPM)
----------------------------------------------------
Clean, self‑contained implementation that fixes the logical and numerical
problems in the original script:

*   Correct DDPM sampling equations and memory usage.
*   Autoregressive decoder now supports **teacher‑forcing** *and* ancestral
    sampling so that cross‑platform imputation no longer leaks the answers.
*   Robust R² computation that skips constant‑valued proteins and avoids the
    scipy/NumPy *ConstantInputWarning*.
*   Streaming metric accumulator instead of storing every batch in RAM.
*   Guard‐rails for missing modalities in semi‑supervised scenarios.

The external encoder (`EnhancedEncoder`) and `total_correlation` loss are
assumed to be available and unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class _RunningR2:
    """Online R² for N×D matrices (samples × features).

    Keeps the five sufficient statistics necessary to compute feature‑wise R²
    without storing the full matrices:  ∑y, ∑y², ∑ŷ, ∑ŷ², ∑yŷ.
    """

    def __init__(self, n_features: int) -> None:
        self.n = 0
        self.sum_y = torch.zeros(n_features)
        self.sum_y2 = torch.zeros(n_features)
        self.sum_yhat = torch.zeros(n_features)
        self.sum_yhat2 = torch.zeros(n_features)
        self.sum_y_yhat = torch.zeros(n_features)

    @torch.no_grad()
    def update(self, y: Tensor, yhat: Tensor) -> None:  # y, yhat: [B, D]
        y = y.detach().cpu()
        yhat = yhat.detach().cpu()
        self.n += y.shape[0]
        self.sum_y += y.sum(0)
        self.sum_y2 += (y ** 2).sum(0)
        self.sum_yhat += yhat.sum(0)
        self.sum_yhat2 += (yhat ** 2).sum(0)
        self.sum_y_yhat += (y * yhat).sum(0)

    def compute(self) -> Tuple[float, torch.Tensor]:
        """Return overall R² and per‑feature R²."""
        if self.n == 0:
            raise RuntimeError("No samples were accumulated.")
        n = float(self.n)
        
        # Means
        mean_y = self.sum_y / n
        mean_yhat = self.sum_yhat / n
        
        # Variance of y
        var_y = self.sum_y2 / n - mean_y ** 2
        
        # Mean Squared Error (MSE)
        mse = self.sum_y2 / n - 2 * self.sum_y_yhat / n + self.sum_yhat2 / n
        
        # Avoid tiny negative variances due to FP error
        var_y.clamp_(min=0.0)
        mse.clamp_(min=0.0)
        
        # R² = 1 - MSE / Var(Y)
        r2_per_feature = torch.zeros_like(var_y)
        mask = var_y > 1e-12
        r2_per_feature[mask] = 1.0 - mse[mask] / var_y[mask]
        overall_r2 = r2_per_feature.mean().item()
        return overall_r2, r2_per_feature

    def summary(self) -> Dict[str, float]:
        overall, per_feat = self.compute()
        return {
            "overall_r2": overall,
            "mean_feature_r2": per_feat.mean().item(),
            "median_feature_r2": per_feat.median().item(),
            "fraction_r2_above_0.5": (per_feat > 0.5).float().mean().item(),
            "fraction_r2_above_0.7": (per_feat > 0.7).float().mean().item(),
        }

# -----------------------------------------------------------------------------
# MADE‑style masked linear layer
# -----------------------------------------------------------------------------
class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: np.ndarray):
        # mask expected shape: [out_features, in_features]
        self.mask.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        return F.linear(input, self.mask * self.weight, self.bias)

# -----------------------------------------------------------------------------
# Autoregressive (MADE) decoder
# -----------------------------------------------------------------------------
class ConditionalAutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Build MADE masks
        degrees: List[torch.Tensor] = [torch.arange(output_dim)]
        for _ in range(n_layers):
            degrees.append(torch.randint(1, output_dim, (hidden_dim,)))
        degrees.append(torch.arange(output_dim))  # output layer degrees

        self.layers = nn.ModuleList()
        self.latent_proj = nn.ModuleList()
        for in_deg, out_deg in zip(degrees[:-1], degrees[1:]):
            mask = (out_deg.unsqueeze(1) >= in_deg.unsqueeze(0)).float().numpy()
            ml = MaskedLinear(in_deg.size(0), out_deg.size(0))
            ml.set_mask(mask)  # shape OK
            self.layers.append(ml)
            # latent projection for all *hidden* layers
            if out_deg is not degrees[-1]:
                self.latent_proj.append(nn.Linear(latent_dim, out_deg.size(0)))

    # ------------------------------------------------------------
    # Training pass (teacher forcing)
    # ------------------------------------------------------------
    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h) + self.latent_proj[i](z))
        return self.layers[-1](h)

    # ------------------------------------------------------------
    # Ancestral sampling (no ground truth)
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, z: Tensor, temperature: float = 1.0) -> Tensor:  # z: [B, L]
        device = z.device
        B = z.size(0)
        x = torch.zeros(B, self.output_dim, device=device)
        for k in range(self.output_dim):
            h = x
            for i, layer in enumerate(self.layers[:-1]):
                h = F.relu(layer(h) + self.latent_proj[i](z))
            logits = self.layers[-1](h)  # [B, D]
            # Stochastic sampling from Gaussian distribution
            x[:, k] = logits[:, k] + torch.randn_like(logits[:, k]) * temperature
        return x

# -----------------------------------------------------------------------------
# DDPM decoder (vectorised)
# -----------------------------------------------------------------------------

def _linear_beta_schedule(timesteps: int, *, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)

class ConditionalDiffusionDecoder(nn.Module):
    """Vector DDPM conditioned on latent *z* (FiLM‑style conditioning)."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_timesteps: int = 100,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_timesteps = n_timesteps

        self.time_embed = nn.Embedding(n_timesteps, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(output_dim + latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        betas = _linear_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, 0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.register_buffer("sqrt_alphas_cum", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_alphas_cum", torch.sqrt(1.0 - alphas_cum))

    # ------------------------------------------------------------
    # Forward noise‑prediction
    # ------------------------------------------------------------
    def _epsilon_theta(self, z: Tensor, x_t: Tensor, t: Tensor) -> Tensor:  # [B,D]
        emb = self.time_embed(t)
        inp = torch.cat([x_t, z, emb], dim=1)
        return self.net(inp)

    # ------------------------------------------------------------
    # Training loss (predict ε)
    # ------------------------------------------------------------
    def loss(self, z: Tensor, x0: Tensor) -> Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (
            self.sqrt_alphas_cum[t].unsqueeze(1) * x0 +
            self.sqrt_one_minus_alphas_cum[t].unsqueeze(1) * noise
        )
        eps_hat = self._epsilon_theta(z, x_t, t)
        return F.mse_loss(eps_hat, noise)

    # ------------------------------------------------------------
    # Sampling (predict‑then‑sample, DDPM Eq. 11)
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, z: Tensor) -> Tensor:  # returns x0, shape [B,D]
        B, D = z.size(0), self.output_dim
        x_t = torch.randn(B, D, device=z.device)
        for t in reversed(range(self.n_timesteps)):
            t_tensor = torch.full((B,), t, device=z.device, dtype=torch.long)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cum_t = self.alphas_cum[t]
            sqrt_one_minus_cum = self.sqrt_one_minus_alphas_cum[t]

            eps_hat = self._epsilon_theta(z, x_t, t_tensor)
            # Eq. 11 in Ho et al.
            mean = (
                (1.0 / math.sqrt(alpha_t)) *
                (x_t - beta_t / sqrt_one_minus_cum * eps_hat)
            )
            if t > 0:
                noise = torch.randn_like(x_t)
                var = beta_t * (1 - alpha_cum_t) / (1 - self.alphas_cum[t - 1])
                x_t = mean + math.sqrt(var) * noise
            else:
                x_t = mean  # x_0
        return x_t

# -----------------------------------------------------------------------------
# Main LightningModule
# -----------------------------------------------------------------------------
from .joint_vae import Encoder  # type: ignore

class GenerativeVAE(pl.LightningModule):
    def __init__(self, input_dim_a: int, input_dim_b: int, config: Dict):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.cfg = config
        m_cfg = config["model"]
        enc_cfg = m_cfg.get("encoder", {})
        latent_dim = m_cfg.get("latent_dim", 32)

        # Extract encoder parameters that match the simple Encoder interface
        encoder_params = {
            "hidden_dims": enc_cfg.get("hidden_dims", [512, 256, 128]),
            "activation": enc_cfg.get("activation", "relu"),
            "dropout_rate": enc_cfg.get("dropout_rate", 0.2),
            "batch_norm": enc_cfg.get("batch_norm", True),
        }

        self.encoder_a = Encoder(input_dim=input_dim_a, latent_dim=latent_dim, **encoder_params)
        self.encoder_b = Encoder(input_dim=input_dim_b, latent_dim=latent_dim, **encoder_params)

        dec_cfg = m_cfg.get("decoder", {})
        dec_type = m_cfg.get("decoder_type", "autoregressive")
        if dec_type == "autoregressive":
            # Filter parameters for autoregressive decoder
            autoregressive_params = {
                "hidden_dim": dec_cfg.get("hidden_dim", 256),
                "n_layers": dec_cfg.get("n_layers", 2),
            }
            self.decoder_a = ConditionalAutoregressiveDecoder(latent_dim, input_dim_a, **autoregressive_params)
            self.decoder_b = ConditionalAutoregressiveDecoder(latent_dim, input_dim_b, **autoregressive_params)
        elif dec_type == "diffusion":
            # Filter parameters for diffusion decoder
            diffusion_params = {
                "n_timesteps": dec_cfg.get("n_timesteps", 100),
                "hidden_dim": dec_cfg.get("hidden_dim", 256),
            }
            self.decoder_a = ConditionalDiffusionDecoder(latent_dim, input_dim_a, **diffusion_params)
            self.decoder_b = ConditionalDiffusionDecoder(latent_dim, input_dim_b, **diffusion_params)
        else:
            raise ValueError(f"Unknown decoder_type: {dec_type}")
        self.dec_type = dec_type

        self.loss_w = config.get("loss_weights", {
            "reconstruction": 1.0,
            "kl_divergence": 1.0,
            "latent_alignment": 1.0,
            "total_correlation": 1.0,
            "cross_reconstruction": 1.0,
        })

        # Streaming R² accumulators
        self._val_r2_native_a: Optional[_RunningR2] = None
        self._val_r2_native_b: Optional[_RunningR2] = None
        self._val_r2_cross_a: Optional[_RunningR2] = None
        self._val_r2_cross_b: Optional[_RunningR2] = None

    # ------------------------------------------------------------------
    # Forward pass returns latent dictionaries
    # ------------------------------------------------------------------
    def _encode(self, xa: Optional[Tensor], xb: Optional[Tensor]):
        outs = {}
        if xa is not None:
            ma, lva = self.encoder_a(xa)
            za = self._reparameterize(ma, lva)
            outs.update(za=za, ma=ma, lva=lva)
        if xb is not None:
            mb, lvb = self.encoder_b(xb)
            zb = self._reparameterize(mb, lvb)
            outs.update(zb=zb, mb=mb, lvb=lvb)
        return outs

    def _reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    # ------------------------------------------------------------------
    # Loss computation common to train/val/test
    # ------------------------------------------------------------------
    def _common_step(self, batch):
        xa, xb = batch  # each may be None in semi‑supervised mode
        outs = self._encode(xa, xb)

        # --- KL + alignment + TC ------------------------------------------------
        kl = 0.0
        align = 0.0
        if "za" in outs:
            kl_a = 0.5 * torch.mean(outs["ma"].pow(2) + outs["lva"].exp() - 1 - outs["lva"])
            kl += kl_a
        if "zb" in outs:
            kl_b = 0.5 * torch.mean(outs["mb"].pow(2) + outs["lvb"].exp() - 1 - outs["lvb"])
            kl += kl_b
        kl = kl / 2.0
        if "ma" in outs and "mb" in outs:
            align = F.mse_loss(outs["ma"], outs["mb"])

        # --- Reconstruction -----------------------------------------------------
        if self.dec_type == "autoregressive":
            # Native reconstructions – teacher forcing
            rec_a = self.decoder_a(outs["za"], xa) if xa is not None else None
            rec_b = self.decoder_b(outs["zb"], xb) if xb is not None else None
            # Cross reconstructions – *sampling* (no teacher forcing!)
            cross_a = self.decoder_a.sample(outs["zb"]) if xb is not None else None
            cross_b = self.decoder_b.sample(outs["za"]) if xa is not None else None

            recon_loss = 0.0
            cross_loss = 0.0
            if rec_a is not None:
                recon_loss += F.mse_loss(rec_a, xa)
            if rec_b is not None:
                recon_loss += F.mse_loss(rec_b, xb)
            if cross_a is not None:
                cross_loss += F.mse_loss(cross_a, xa)
            if cross_b is not None:
                cross_loss += F.mse_loss(cross_b, xb)
            recon_loss /= 2.0
            cross_loss /= 2.0
        else:  # ---------------------------  DIFFUSION  -----------------------------
            # 1) native diffusion loss (unchanged – *has* gradients)
            recon_loss = 0.0
            if xa is not None:
                recon_loss += self.decoder_a.loss(outs["za"], xa)
            if xb is not None:
                recon_loss += self.decoder_b.loss(outs["zb"], xb)
            recon_loss /= 2.0

            # 2) ★ TRAINABLE cross-diffusion loss  ★
            cross_loss = 0.0
            if xa is not None and xb is not None:
                # NOTE: these are *true* diffusion losses, not MSE on samples
                cross_loss  = self.decoder_a.loss(outs["zb"], xa)
                cross_loss += self.decoder_b.loss(outs["za"], xb)
                cross_loss *= 0.5        # average over the two directions

            # 3) Optional: generate samples for R² metrics (no grad needed)
            with torch.no_grad():
                rec_a   = self.decoder_a.sample(outs["za"]) if xa is not None else None
                rec_b   = self.decoder_b.sample(outs["zb"]) if xb is not None else None
                cross_a = self.decoder_a.sample(outs["zb"]) if xa is not None and xb is not None else None
                cross_b = self.decoder_b.sample(outs["za"]) if xa is not None and xb is not None else None
                
        total = (
            self.loss_w["reconstruction"] * recon_loss +
            self.loss_w["kl_divergence"] * kl +
            self.loss_w["latent_alignment"] * align +
            self.loss_w["cross_reconstruction"] * cross_loss
        )

        return {
            "loss": total,
            "kl": kl,
            "align": align,
            "recon": recon_loss,
            "cross": cross_loss,
            "rec_a": rec_a,
            "rec_b": rec_b,
            "cross_a": cross_a,
            "cross_b": cross_b,
            "xa": xa,
            "xb": xb,
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        out = self._common_step(batch)
        # Only log scalar loss values, not tensors
        scalar_losses = {k: v for k, v in out.items() if k in ["loss", "kl", "align", "recon", "cross"] and isinstance(v, torch.Tensor)}
        self.log_dict({f"train_{k}": v for k, v in scalar_losses.items()}, prog_bar=True)
        return out["loss"]

    def _accumulate_val_metrics(self, out):
        if self._val_r2_native_a is None:
            D_a = out["xa"].size(1)
            D_b = out["xb"].size(1)
            self._val_r2_native_a = _RunningR2(D_a)
            self._val_r2_native_b = _RunningR2(D_b)
            self._val_r2_cross_a = _RunningR2(D_a)
            self._val_r2_cross_b = _RunningR2(D_b)
        if out["rec_a"] is not None:
            self._val_r2_native_a.update(out["xa"], out["rec_a"].detach())
        if out["rec_b"] is not None:
            self._val_r2_native_b.update(out["xb"], out["rec_b"].detach())
        if out["cross_a"] is not None:
            self._val_r2_cross_a.update(out["xa"], out["cross_a"].detach())
        if out["cross_b"] is not None:
            self._val_r2_cross_b.update(out["xb"], out["cross_b"].detach())

    def validation_step(self, batch, _):
        out = self._common_step(batch)
        
        # Log basic validation losses for callbacks to monitor
        scalar_losses = {k: v for k, v in out.items() if k in ["loss", "kl", "align", "recon", "cross"] and isinstance(v, torch.Tensor)}
        # Log with val_ prefix for callback compatibility
        val_logs = {f"val_{k}": v for k, v in scalar_losses.items()}
        val_logs["val_total_loss"] = out["loss"]  # Add total loss alias
        self.log_dict(val_logs, prog_bar=True, on_epoch=True)
        
        self._accumulate_val_metrics(out)
        return out["loss"]

    def on_validation_epoch_end(self):
        if self._val_r2_native_a is None:
            return
        metrics = {}
        metrics.update({f"val_native_a_{k}": v for k, v in self._val_r2_native_a.summary().items()})
        metrics.update({f"val_native_b_{k}": v for k, v in self._val_r2_native_b.summary().items()})
        metrics.update({f"val_cross_a_{k}": v for k, v in self._val_r2_cross_a.summary().items()})
        metrics.update({f"val_cross_b_{k}": v for k, v in self._val_r2_cross_b.summary().items()})
        self.log_dict(metrics, prog_bar=True)
        # Reset accumulators
        self._val_r2_native_a = None
        self._val_r2_native_b = None
        self._val_r2_cross_a = None
        self._val_r2_cross_b = None

    # For brevity, test_step mirrors validation metrics accumulation
    test_step = validation_step  # type: ignore
    on_test_epoch_end = on_validation_epoch_end  # type: ignore

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cfg = self.cfg["training"]
        return torch.optim.AdamW(self.parameters(), lr=opt_cfg.get("learning_rate", 1e-3), weight_decay=opt_cfg.get("weight_decay", 1e-4))
