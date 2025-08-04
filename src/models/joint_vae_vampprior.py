"""
Joint Variational Autoencoder (jVAE) for cross-platform metabolite data imputation -- *numerically-stable* version.

Key fixes over the previous revision
------------------------------------
* **Stable re-parameterisation** – clamp `logvar` inside `reparameterize` so `exp()` never over/under-flows.
* **Correct log-normal density** – always sum over the *last* dimension in `log_normal_diag`.
* **Deterministic VampPrior** – pseudo-inputs are encoded with the encoder in `eval` mode under `torch.no_grad()`; the encoder’s original mode is restored afterwards.
* **FP-safe constants** – use `math.log(2π)` with the tensor’s dtype/device.
* Minor clean-ups (static constants, type hints).

This file is a *drop-in replacement* for `src/models/joint_vae_vampprior.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Local utils (metrics)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_imputation_metrics  # noqa: E402

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

LOG2PI = math.log(2.0 * math.pi)


class MLP(nn.Module):
    """Simple MLP with optional BN / Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        final_activation: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add extras to *all* but the final layer unless requested otherwise
            is_last_linear = i == len(dims) - 2
            if not is_last_linear or final_activation:
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(dims[i + 1]))

                act = {
                    "relu": nn.ReLU(),
                    "leaky_relu": nn.LeakyReLU(0.2),
                    "tanh": nn.Tanh(),
                    "elu": nn.ELU(),
                }[activation]
                self.layers.append(act)

                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """VAE encoder -> μ, log σ²."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        **mlp_kwargs,
    ) -> None:
        super().__init__()

        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else input_dim,
            **mlp_kwargs,
        )
        feat_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.mean_layer = nn.Linear(feat_dim, latent_dim)
        self.logvar_layer = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        h = self.backbone(x)
        return self.mean_layer(h), self.logvar_layer(h)


class Decoder(nn.Module):
    """Latent → reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        **mlp_kwargs,
    ) -> None:
        super().__init__()
        self.network = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            final_activation=False,
            **mlp_kwargs,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(z)


# -----------------------------------------------------------------------------
# Joint VAE with VampPrior
# -----------------------------------------------------------------------------

class JointVAE(nn.Module):
    """jVAE with two encoders/decoders and an optional VampPrior."""

    LOGVAR_MAX = 10.0  # σ ≈ 148
    LOGVAR_MIN = -10.0  # σ ≈ 0.007

    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        *,
        latent_dim: int = 32,
        encoder_layers: List[int] | None = None,
        decoder_layers: List[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_vampprior: bool = False,
        num_pseudoinputs: int = 500,
    ) -> None:
        super().__init__()

        encoder_layers = encoder_layers or [512, 256, 128]
        decoder_layers = decoder_layers or [128, 256, 512]
        mlp_kwargs = dict(
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
        )

        # A-side
        self.encoder_a = Encoder(input_dim_a, encoder_layers, latent_dim, **mlp_kwargs)
        self.decoder_a = Decoder(latent_dim, decoder_layers, input_dim_a, **mlp_kwargs)
        # B-side
        self.encoder_b = Encoder(input_dim_b, encoder_layers, latent_dim, **mlp_kwargs)
        self.decoder_b = Decoder(latent_dim, decoder_layers, input_dim_b, **mlp_kwargs)

        # VampPrior
        self.use_vampprior = use_vampprior
        self.num_pseudoinputs = num_pseudoinputs
        if use_vampprior:
            self.pseudoinputs_a = nn.Parameter(torch.randn(num_pseudoinputs, input_dim_a))
            self.pseudoinputs_b = nn.Parameter(torch.randn(num_pseudoinputs, input_dim_b))
        else:
            self.register_parameter("pseudoinputs_a", None)
            self.register_parameter("pseudoinputs_b", None)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mean, exp(logvar)) with clamped variance."""
        logvar = logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _log_normal_diag(self, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """log N(x | μ, diag(exp(log_var))). Works with arbitrary leading dims."""
        return -0.5 * (
            log_var + ((x - mu) ** 2) / log_var.exp() + LOG2PI
        ).sum(dim=-1)  # ← sum over latent dim only

    # Public alias (kept for backward compat.)
    log_normal_diag = _log_normal_diag

    # ------------------------------------------------------------------
    # Encoder / decoder wrappers
    # ------------------------------------------------------------------

    def encode_a(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder_a(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def encode_b(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder_b(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_a(z)

    def decode_b(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_b(z)

    # ------------------------------------------------------------------
    # VampPrior log-densities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_pseudo(self, encoder: nn.Module, pseudo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper: encode pseudo-inputs under eval() so dropout/B-N are deterministic."""
        was_training = encoder.training
        encoder.eval()
        mu, logvar = encoder(pseudo)
        if was_training:
            encoder.train()
        # Clamp for numerical safety
        return mu, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)

    def vampprior_log_prob_a(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_vampprior:
            return torch.zeros(z.size(0), device=z.device)
        mu_p, logvar_p = self._encode_pseudo(self.encoder_a, self.pseudoinputs_a)
        log_probs = self._log_normal_diag(z.unsqueeze(1), mu_p.unsqueeze(0), logvar_p.unsqueeze(0))
        return torch.logsumexp(log_probs, dim=1) - math.log(self.num_pseudoinputs)

    def vampprior_log_prob_b(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_vampprior:
            return torch.zeros(z.size(0), device=z.device)
        mu_p, logvar_p = self._encode_pseudo(self.encoder_b, self.pseudoinputs_b)
        log_probs = self._log_normal_diag(z.unsqueeze(1), mu_p.unsqueeze(0), logvar_p.unsqueeze(0))
        return torch.logsumexp(log_probs, dim=1) - math.log(self.num_pseudoinputs)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        # Encode
        z_a, mu_a, logvar_a = self.encode_a(x_a)
        z_b, mu_b, logvar_b = self.encode_b(x_b)

        # Reconstructions
        recon_a = self.decode_a(z_a)
        recon_b = self.decode_b(z_b)

        # Cross-recon
        cross_recon_a = self.decode_a(z_b)
        cross_recon_b = self.decode_b(z_a)

        # Cycle recon
        z_cycle_b, _, _ = self.encode_b(cross_recon_b)
        z_cycle_a, _, _ = self.encode_a(cross_recon_a)
        cyclic_recon_a = self.decode_a(z_cycle_b)
        cyclic_recon_b = self.decode_b(z_cycle_a)

        return {
            "z_a": z_a,
            "z_b": z_b,
            "mean_a": mu_a,
            "mean_b": mu_b,
            "logvar_a": logvar_a,
            "logvar_b": logvar_b,
            "recon_a": recon_a,
            "recon_b": recon_b,
            "cross_recon_a": cross_recon_a,
            "cross_recon_b": cross_recon_b,
            "cyclic_recon_a": cyclic_recon_a,
            "cyclic_recon_b": cyclic_recon_b,
        }

    # Convenience imputation helpers (unused in training)
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        return self.decode_b(self.encode_a(x_a)[0])

    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        return self.decode_a(self.encode_b(x_b)[0])


# -----------------------------------------------------------------------------
# Lightning wrapper (unchanged except for the safer model)
# -----------------------------------------------------------------------------

class JointVAELightning(pl.LightningModule):
    """PyTorch-Lightning wrapper around :class:`JointVAE`."""

    def __init__(self, input_dim_a: int, input_dim_b: int, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        mcfg = config["model"]
        self.model = JointVAE(
            input_dim_a,
            input_dim_b,
            latent_dim=mcfg["latent_dim"],
            encoder_layers=mcfg["encoder_layers"],
            decoder_layers=mcfg["decoder_layers"],
            activation=mcfg["activation"],
            dropout_rate=mcfg["dropout_rate"],
            batch_norm=mcfg["batch_norm"],
            use_vampprior=mcfg.get("use_vampprior", False),
            num_pseudoinputs=mcfg.get("num_pseudoinputs", 500),
        )

        self.loss_weights = config["loss_weights"]
        self.validation_step_outputs: list[Dict] = []
        self.test_step_outputs: list[Dict] = []
        self.anneal_epochs = float(config["training"].get("anneal_epochs", 100.0))

    # ------------------------- loss & metrics -------------------------

    def _kl_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model.use_vampprior:
            # This path is correct, no changes needed here.
            mu_a, mu_b = outputs["mean_a"], outputs["mean_b"]
            logvar_a = outputs["logvar_a"].clamp(self.model.LOGVAR_MIN, self.model.LOGVAR_MAX)
            logvar_b = outputs["logvar_b"].clamp(self.model.LOGVAR_MIN, self.model.LOGVAR_MAX)

            log_q_a = self.model.log_normal_diag(outputs["z_a"], mu_a, logvar_a)
            log_q_b = self.model.log_normal_diag(outputs["z_b"], mu_b, logvar_b)
            log_p_a = self.model.vampprior_log_prob_a(outputs["z_a"])
            log_p_b = self.model.vampprior_log_prob_b(outputs["z_b"])
            return 0.5 * ((log_q_a - log_p_a).mean() + (log_q_b - log_p_b).mean())
        else:
            # --- CHANGE START: Stabilize Standard KL Loss ---
            # Clamp logvar before .exp() to prevent numerical overflow
            logvar_a = outputs["logvar_a"].clamp(self.model.LOGVAR_MIN, self.model.LOGVAR_MAX)
            logvar_b = outputs["logvar_b"].clamp(self.model.LOGVAR_MIN, self.model.LOGVAR_MAX)

            kl_a = 0.5 * (
                outputs["mean_a"].pow(2) + logvar_a.exp() - 1 - logvar_a
            ).mean()
            kl_b = 0.5 * (
                outputs["mean_b"].pow(2) + logvar_b.exp() - 1 - logvar_b
            ).mean()
            # --- CHANGE END ---
            return 0.5 * (kl_a + kl_b)

    def compute_loss(self, outputs: Dict[str, torch.Tensor], x_a: torch.Tensor, x_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = 0.5 * (
            F.mse_loss(outputs["recon_a"], x_a) + F.mse_loss(outputs["recon_b"], x_b)
        )
        cross_loss = 0.5 * (
            F.mse_loss(outputs["cross_recon_a"], x_a) + F.mse_loss(outputs["cross_recon_b"], x_b)
        )
        cyclic_loss = 0.5 * (
            F.mse_loss(outputs["cyclic_recon_a"], x_a) + F.mse_loss(outputs["cyclic_recon_b"], x_b)
        )
        align_loss = F.mse_loss(outputs["mean_a"], outputs["mean_b"])
        kl_loss = self._kl_loss(outputs)

        total = (
            self.loss_weights["reconstruction"] * recon_loss
            + self.loss_weights["kl_divergence"] * kl_loss
            + self.loss_weights["latent_alignment"] * align_loss
            + self.loss_weights["cross_reconstruction"] * cross_loss
            + self.loss_weights.get("cyclic", 0.0) * cyclic_loss
        )
        return dict(
            total_loss=total,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            align_loss=align_loss,
            cross_loss=cross_loss,
            cyclic_loss=cyclic_loss,
        )

    # ------------------------- PL hooks -------------------------

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):  # noqa: D401
        return self.model(x_a, x_b)

    def training_step(self, batch, *_):  # type: ignore[override]
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        loss_dict = self.compute_loss(outputs, x_a, x_b)

        # --- CHANGE START: KL Annealing Logic ---
        # Get target weight from config
        target_kl_weight = self.loss_weights['kl_divergence']

        # Anneal over the first 100 epochs (adjust as needed)
        current_kl_weight = target_kl_weight * min(1.0, self.current_epoch / self.anneal_epochs)

        # Re-calculate the total loss with the annealed KL weight
        total_loss = (
            self.loss_weights["reconstruction"] * loss_dict["recon_loss"]
            + current_kl_weight * loss_dict["kl_loss"]  # Use annealed weight
            + self.loss_weights["latent_alignment"] * loss_dict["align_loss"]
            + self.loss_weights["cross_reconstruction"] * loss_dict["cross_loss"]
            + self.loss_weights.get("cyclic", 0.0) * loss_dict["cyclic_loss"]
        )

        # Log all values for monitoring
        self.log("train_total_loss_annealed", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("current_kl_weight", current_kl_weight, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            # Log raw, unweighted losses
            if k != "total_loss":
                self.log(f"train_{k}", v, on_step=True, on_epoch=True)
        # --- CHANGE END ---

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b)

        # --- CHANGE START: KL Annealing Logic for Validation ---
        target_kl_weight = self.loss_weights['kl_divergence']
        current_kl_weight = target_kl_weight * min(1.0, self.current_epoch / self.anneal_epochs)

        total_loss_annealed = (
            self.loss_weights["reconstruction"] * losses["recon_loss"]
            + current_kl_weight * losses["kl_loss"]
            + self.loss_weights["latent_alignment"] * losses["align_loss"]
            + self.loss_weights["cross_reconstruction"] * losses["cross_loss"]
            + self.loss_weights.get("cyclic", 0.0) * losses["cyclic_loss"]
        )
        self.log('val_total_loss', total_loss_annealed, on_step=False, on_epoch=True, prog_bar=True)

        # Log other individual losses
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True)
        # --- CHANGE END ---

        # Store outputs for R² metrics at epoch end
        step_output = {
            'outputs': outputs,
            'x_a': x_a,
            'x_b': x_b
        }
        self.validation_step_outputs.append(step_output)
        return step_output
    
    def test_step(self, batch, batch_idx):
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'test_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        # Store outputs for epoch-end processing
        step_output = {
            'test_loss': losses['total_loss'],
            'outputs': outputs,
            'x_a': x_a,
            'x_b': x_b
        }
        self.test_step_outputs.append(step_output)
        
        return step_output
    
    def on_validation_epoch_end(self):
        """Compute additional metrics at the end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Concatenate all outputs from validation steps
        all_x_a = torch.cat([output['x_a'] for output in self.validation_step_outputs], dim=0)
        all_x_b = torch.cat([output['x_b'] for output in self.validation_step_outputs], dim=0)
        all_outputs = {}
        
        # Concatenate all output tensors
        for key in self.validation_step_outputs[0]['outputs'].keys():
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.validation_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = all_outputs['recon_a'].detach().cpu().numpy()
        recon_b_np = all_outputs['recon_b'].detach().cpu().numpy()
        cross_recon_a_np = all_outputs['cross_recon_a'].detach().cpu().numpy()
        cross_recon_b_np = all_outputs['cross_recon_b'].detach().cpu().numpy()
        
        # Compute R² metrics for reconstruction tasks
        recon_a_metrics = compute_imputation_metrics(x_a_np, recon_a_np)
        recon_b_metrics = compute_imputation_metrics(x_b_np, recon_b_np)
        
        # Compute R² metrics for cross-platform imputation
        cross_a_metrics = compute_imputation_metrics(x_a_np, cross_recon_a_np)
        cross_b_metrics = compute_imputation_metrics(x_b_np, cross_recon_b_np)
        
        # Log key R² metrics
        self.log('val_recon_a_r2', recon_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_recon_b_r2', recon_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_cross_a_r2', cross_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_cross_b_r2', cross_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log mean feature-wise R² scores
        self.log('val_recon_a_mean_feature_r2', recon_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_recon_b_mean_feature_r2', recon_b_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_cross_a_mean_feature_r2', cross_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_cross_b_mean_feature_r2', cross_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics
        self.log('val_recon_a_frac_r2_above_0.5', recon_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_recon_b_frac_r2_above_0.5', recon_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_cross_a_frac_r2_above_0.5', cross_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_cross_b_frac_r2_above_0.5', cross_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Compute additional metrics at the end of test epoch."""
        if not self.test_step_outputs:
            return
        
        # Concatenate all outputs from test steps
        all_x_a = torch.cat([output['x_a'] for output in self.test_step_outputs], dim=0)
        all_x_b = torch.cat([output['x_b'] for output in self.test_step_outputs], dim=0)
        all_outputs = {}
        
        # Concatenate all output tensors
        for key in self.test_step_outputs[0]['outputs'].keys():
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.test_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = all_outputs['recon_a'].detach().cpu().numpy()
        recon_b_np = all_outputs['recon_b'].detach().cpu().numpy()
        cross_recon_a_np = all_outputs['cross_recon_a'].detach().cpu().numpy()
        cross_recon_b_np = all_outputs['cross_recon_b'].detach().cpu().numpy()
        
        # Compute R² metrics for reconstruction tasks
        recon_a_metrics = compute_imputation_metrics(x_a_np, recon_a_np)
        recon_b_metrics = compute_imputation_metrics(x_b_np, recon_b_np)
        
        # Compute R² metrics for cross-platform imputation
        cross_a_metrics = compute_imputation_metrics(x_a_np, cross_recon_a_np)
        cross_b_metrics = compute_imputation_metrics(x_b_np, cross_recon_b_np)
        
        # Log key R² metrics
        self.log('test_recon_a_r2', recon_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_recon_b_r2', recon_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_cross_a_r2', cross_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_cross_b_r2', cross_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log mean feature-wise R² scores
        self.log('test_recon_a_mean_feature_r2', recon_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_recon_b_mean_feature_r2', recon_b_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_cross_a_mean_feature_r2', cross_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_cross_b_mean_feature_r2', cross_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics
        self.log('test_recon_a_frac_r2_above_0.5', recon_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_recon_b_frac_r2_above_0.5', recon_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_cross_a_frac_r2_above_0.5', cross_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_cross_b_frac_r2_above_0.5', cross_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
        # Log additional detailed metrics for final test evaluation
        self.log('test_recon_a_median_feature_r2', recon_a_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_recon_b_median_feature_r2', recon_b_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_cross_a_median_feature_r2', cross_a_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_cross_b_median_feature_r2', cross_b_metrics['median_feature_r2'], on_epoch=True)
        
        self.log('test_recon_a_frac_r2_above_0.7', recon_a_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_recon_b_frac_r2_above_0.7', recon_b_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_cross_a_frac_r2_above_0.7', cross_a_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_cross_b_frac_r2_above_0.7', cross_b_metrics['fraction_r2_above_0.7'], on_epoch=True)
        
        # Print summary for final test results
        print("\n" + "="*60)
        print("FINAL TEST R² METRICS SUMMARY")
        print("="*60)
        print(f"Platform A Reconstruction R²: {recon_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Reconstruction R²: {recon_b_metrics['overall_r2']:.4f}")
        print(f"Platform A Cross-Imputation R²: {cross_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Cross-Imputation R²: {cross_b_metrics['overall_r2']:.4f}")
        print(f"Features with R² > 0.5 (Recon A): {recon_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Recon B): {recon_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross A): {cross_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross B): {cross_b_metrics['fraction_r2_above_0.5']:.2%}")
        print("="*60)
        
        # Clear outputs
        self.test_step_outputs.clear()

    # ------------------------- Optimiser -------------------------

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Optimizer
        if training_config['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=training_config['learning_rate']
            )
        elif training_config['optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=training_config['learning_rate']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
        
        # Scheduler
        if training_config.get('scheduler') == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=training_config.get('scheduler_factor', 0.5),
                patience=training_config.get('scheduler_patience', 5)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_total_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
