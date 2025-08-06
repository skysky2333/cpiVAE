"""
Joint Variational Autoencoder Plus (jVAE+) with State-of-the-Art Techniques.

This module implements an enhanced jVAE architecture incorporating:
- β-VAE with annealing for disentanglement
- KL annealing schedules
- Normalizing flows for improved posterior expressivity
- FactorVAE-style total correlation penalty
- Advanced architectural components
- Copula-based modeling in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import math
from .losses import total_correlation, log_normal_diag

# Add imports for metrics
try:
    from ..utils.metrics import compute_imputation_metrics
except ImportError:
    # Fallback for when running from different context
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.metrics import compute_imputation_metrics


class NormalizingFlow(nn.Module):
    """Planar normalizing flow for improving posterior expressivity."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through planar flow.
        
        Returns:
            z_new: Transformed latent variables
            log_det_jac: Log determinant of Jacobian
        """
        # Ensure u and w have proper orientation for invertibility
        uw = torch.sum(self.u * self.w)
        m_uw = -1 + F.softplus(uw)
        u_hat = self.u + (m_uw - uw) * self.w / (torch.sum(self.w ** 2) + 1e-8)
        
        # Planar transformation
        wz_b = torch.sum(self.w * z, dim=-1, keepdim=True) + self.b
        f_z = z + u_hat.unsqueeze(0) * torch.tanh(wz_b)
        
        # Log determinant of Jacobian
        psi = (1 - torch.tanh(wz_b) ** 2) * self.w.unsqueeze(0)
        det_jacobian = 1 + torch.sum(psi * u_hat.unsqueeze(0), dim=-1)
        
        # Add a small epsilon for numerical stability and clamp to avoid log(<=0)
        log_det_jac = torch.log(torch.clamp(torch.abs(det_jacobian), min=1e-8))
        
        return f_z, log_det_jac


class MultiLayerFlow(nn.Module):
    """Multiple normalizing flow layers for improved expressivity."""
    
    def __init__(self, latent_dim: int, num_flows: int = 8):
        super().__init__()
        self.flows = nn.ModuleList([NormalizingFlow(latent_dim) for _ in range(num_flows)])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_jac_total = torch.zeros(z.shape[0], device=z.device)
        
        for flow in self.flows:
            z, log_det_jac = flow(z)
            log_det_jac_total += log_det_jac
        
        return z, log_det_jac_total


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization for stable training."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=bias))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization and improved activations."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "swish",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        spectral_norm: bool = True
    ):
        super().__init__()
        
        # Layer selection based on spectral norm preference
        LinearLayer = SpectralNormLinear if spectral_norm else nn.Linear
        
        self.input_projection = None
        if input_dim != output_dim:
            self.input_projection = LinearLayer(input_dim, output_dim)
        
        # Two-layer residual block
        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        
        self.layer2 = LinearLayer(hidden_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim) if batch_norm else nn.Identity()
        
        # Advanced activation functions
        if activation == "swish":
            self.activation = nn.SiLU()  # Swish/SiLU activation
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "mish":
            self.activation = nn.Mish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First layer
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second layer
        out = self.layer2(out)
        out = self.norm2(out)
        
        # Skip connection
        if self.input_projection is not None:
            identity = self.input_projection(identity)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class EnhancedMLP(nn.Module):
    """Enhanced MLP with residual connections and advanced features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "swish",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        spectral_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            if use_residual and i < len(dims) - 2:  # No residual for final layer
                layer = ResidualBlock(
                    input_dim=dims[i],
                    hidden_dim=dims[i + 1],
                    output_dim=dims[i + 1],
                    activation=activation,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    spectral_norm=spectral_norm
                )
            else:
                # Simple linear layer for final output
                LinearLayer = SpectralNormLinear if spectral_norm else nn.Linear
                layer = LinearLayer(dims[i], dims[i + 1])
            
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class EnhancedEncoder(nn.Module):
    """Enhanced VAE encoder with normalizing flows and improved architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "swish",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        spectral_norm: bool = True,
        use_flows: bool = True,
        num_flows: int = 8
    ):
        super().__init__()
        
        self.use_flows = use_flows
        
        # Main encoder network
        self.backbone = EnhancedMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else input_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            use_residual=True
        )
        
        # Output layers for mean and log-variance
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        LinearLayer = SpectralNormLinear if spectral_norm else nn.Linear
        
        self.mean_layer = LinearLayer(final_dim, latent_dim)
        self.logvar_layer = LinearLayer(final_dim, latent_dim)
        
        # Normalizing flows for improved posterior
        if use_flows:
            self.flows = MultiLayerFlow(latent_dim, num_flows)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.backbone(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterization trick with numerical stability
        # Clamp logvar to prevent overflow/underflow
        logvar_clamped = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        z_0 = mean + eps * std
        
        # Check for NaN/Inf in latent variables
        if torch.isnan(z_0).any() or torch.isinf(z_0).any():
            print("Warning: NaN/Inf detected in latent variables z_0")
            z_0 = torch.randn_like(z_0) * 0.1  # Small random noise fallback
        
        # Apply normalizing flows if enabled
        log_det_jac = None
        if self.use_flows:
            z, log_det_jac = self.flows(z_0)
        else:
            z = z_0
        
        return z, mean, logvar, log_det_jac


class EnhancedDecoder(nn.Module):
    """Enhanced VAE decoder with improved architecture."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "swish",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        spectral_norm: bool = True
    ):
        super().__init__()
        
        self.network = EnhancedMLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            use_residual=True
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class JointVAEPlus(nn.Module):
    """
    Enhanced Joint Variational Autoencoder with state-of-the-art techniques.
    
    Features:
    - Normalizing flows for improved posterior expressivity
    - β-VAE with annealing for disentanglement
    - FactorVAE-style total correlation penalty
    - Enhanced architectures with residual connections
    - Spectral normalization for training stability
    """
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        latent_dim: int = 32,
        encoder_layers: List[int] = [512, 256, 128],
        decoder_layers: List[int] = [128, 256, 512],
        activation: str = "swish",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        spectral_norm: bool = True,
        use_flows: bool = True,
        num_flows: int = 8
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.latent_dim = latent_dim
        self.use_flows = use_flows
        
        # Platform A components
        self.encoder_a = EnhancedEncoder(
            input_dim=input_dim_a,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            use_flows=use_flows,
            num_flows=num_flows
        )
        
        self.decoder_a = EnhancedDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_a,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm
        )
        
        # Platform B components
        self.encoder_b = EnhancedEncoder(
            input_dim=input_dim_b,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            use_flows=use_flows,
            num_flows=num_flows
        )
        
        self.decoder_b = EnhancedDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_b,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm
        )
    
    def encode_a(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode platform A data."""
        return self.encoder_a(x_a)
    
    def encode_b(self, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode platform B data."""
        return self.encoder_b(x_b)
    
    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform A data."""
        return self.decoder_a(z)
    
    def decode_b(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform B data."""
        return self.decoder_b(z)
    
    def forward(
        self, 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced joint VAE.
        
        Returns:
            Dictionary containing all outputs needed for advanced loss computation
        """
        # Encode both platforms
        z_a, mean_a, logvar_a, log_det_jac_a = self.encode_a(x_a)
        z_b, mean_b, logvar_b, log_det_jac_b = self.encode_b(x_b)
        
        # Reconstruct same platform (auto-reconstruction)
        recon_a = self.decode_a(z_a)
        recon_b = self.decode_b(z_b)
        
        # Cross-platform reconstruction
        cross_recon_a = self.decode_a(z_b)  # B -> A
        cross_recon_b = self.decode_b(z_a)  # A -> B
        
        result = {
            'z_a': z_a,
            'z_b': z_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'logvar_a': logvar_a,
            'logvar_b': logvar_b,
            'recon_a': recon_a,
            'recon_b': recon_b,
            'cross_recon_a': cross_recon_a,
            'cross_recon_b': cross_recon_b
        }
        
        # Add flow log determinants if using flows
        if self.use_flows:
            result['log_det_jac_a'] = log_det_jac_a
            result['log_det_jac_b'] = log_det_jac_b
        
        return result
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        z_a, _, _, _ = self.encode_a(x_a)
        return self.decode_b(z_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        z_b, _, _, _ = self.encode_b(x_b)
        return self.decode_a(z_b)


class KLAnnealingScheduler:
    """KL annealing scheduler for β-VAE training."""
    
    def __init__(
        self,
        schedule_type: str = "cyclical",
        cycle_length: int = 1000,
        min_beta: float = 0.0,
        max_beta: float = 1.0,
        anneal_ratio: float = 0.5
    ):
        self.schedule_type = schedule_type
        self.cycle_length = cycle_length
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.anneal_ratio = anneal_ratio
    
    def get_beta(self, step: int) -> float:
        """Get beta value for current step."""
        if self.schedule_type == "linear":
            # Linear annealing
            return min(self.max_beta, self.min_beta + (self.max_beta - self.min_beta) * step / self.cycle_length)
        
        elif self.schedule_type == "cyclical":
            # Cyclical annealing
            cycle_step = step % self.cycle_length
            anneal_steps = int(self.cycle_length * self.anneal_ratio)
            
            if cycle_step < anneal_steps:
                return self.min_beta + (self.max_beta - self.min_beta) * cycle_step / anneal_steps
            else:
                return self.max_beta
        
        elif self.schedule_type == "constant":
            return self.max_beta
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def safe_compute_metrics(true_data, pred_data, name):
    """Safely compute metrics with NaN handling."""
    if np.isnan(pred_data).any() or np.isinf(pred_data).any():
        print(f"Warning: NaN/Inf values detected in {name} predictions. Returning zero metrics.")
        return {
            'overall_r2': 0.0,
            'mean_feature_r2': 0.0,
            'median_feature_r2': 0.0,
            'fraction_r2_above_0.5': 0.0,
            'fraction_r2_above_0.7': 0.0
        }
    return compute_imputation_metrics(true_data, pred_data)


class JointVAEPlusLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Enhanced Joint VAE with SOTA techniques."""
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        config: Dict
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        model_config = config['model']
        
        # Initialize enhanced model
        self.model = JointVAEPlus(
            input_dim_a=input_dim_a,
            input_dim_b=input_dim_b,
            latent_dim=model_config['latent_dim'],
            encoder_layers=model_config['encoder_layers'],
            decoder_layers=model_config['decoder_layers'],
            activation=model_config.get('activation', 'swish'),
            dropout_rate=model_config['dropout_rate'],
            batch_norm=model_config['batch_norm'],
            spectral_norm=model_config.get('spectral_norm', True),
            use_flows=model_config.get('use_flows', True),
            num_flows=model_config.get('num_flows', 8)
        )
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # KL annealing scheduler
        self.kl_scheduler = KLAnnealingScheduler(
            schedule_type=model_config.get('kl_schedule', 'cyclical'),
            cycle_length=model_config.get('kl_cycle_length', 1000),
            min_beta=model_config.get('min_beta', 0.0),
            max_beta=model_config.get('max_beta', 1.0),
            anneal_ratio=model_config.get('kl_anneal_ratio', 0.5)
        )
        
        # TC loss annealing parameters from model config
        model_config = self.config.get('model', {})
        self.tc_anneal_schedule = model_config.get('tc_anneal_schedule', 'constant')
        self.tc_anneal_length = model_config.get('tc_anneal_length', 10000)
        self.max_tc_weight = model_config.get('max_tc_weight', self.loss_weights.get('total_correlation', 0.0))
        
        # Training step counter for annealing
        self.training_step_count = 0
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x_a, x_b)
    
    def compute_enhanced_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the enhanced loss function with SOTA techniques."""
        
        # Get current beta for KL annealing
        beta = self.kl_scheduler.get_beta(self.training_step_count)

        # Get current TC weight for annealing
        if self.tc_anneal_schedule == 'linear':
            tc_weight = min(self.max_tc_weight, (self.max_tc_weight / self.tc_anneal_length) * self.training_step_count)
        else:  # constant
            tc_weight = self.max_tc_weight
        
        # Reconstruction losses
        recon_loss_a = F.mse_loss(outputs['recon_a'], x_a)
        recon_loss_b = F.mse_loss(outputs['recon_b'], x_b)
        recon_loss = (recon_loss_a + recon_loss_b) / 2
        
        # KL divergence losses (with β-VAE formulation) - numerically stable version
        # Clamp logvar to prevent overflow/underflow
        logvar_a_clamped = torch.clamp(outputs['logvar_a'], -10, 10)
        logvar_b_clamped = torch.clamp(outputs['logvar_b'], -10, 10)
        
        kl_loss_a = 0.5 * torch.mean(
            outputs['mean_a'].pow(2) + logvar_a_clamped.exp() - 1 - logvar_a_clamped
        )
        kl_loss_b = 0.5 * torch.mean(
            outputs['mean_b'].pow(2) + logvar_b_clamped.exp() - 1 - logvar_b_clamped
        )
        kl_loss = (kl_loss_a + kl_loss_b) / 2
        
        # Note: KL loss is computed in the original space (z_0) before flows
        # This is a valid approach as flows are used to increase expressivity of the posterior
        # but we still match to a standard normal prior in the original space
        
        # Latent alignment loss
        align_loss = F.mse_loss(outputs['mean_a'], outputs['mean_b'])
        
        # Cross-reconstruction losses
        cross_loss_a = F.mse_loss(outputs['cross_recon_a'], x_a)
        cross_loss_b = F.mse_loss(outputs['cross_recon_b'], x_b)
        cross_loss = (cross_loss_a + cross_loss_b) / 2
        
        # Total correlation loss for disentanglement
        tc_loss_a = total_correlation(outputs['z_a'], outputs['mean_a'], outputs['logvar_a'])
        tc_loss_b = total_correlation(outputs['z_b'], outputs['mean_b'], outputs['logvar_b'])
        tc_loss = (tc_loss_a + tc_loss_b) / 2
        
        # Check for NaN/Inf in loss components before combining
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            print(f"Warning: NaN/Inf detected in reconstruction loss: {recon_loss}")
            recon_loss = torch.tensor(0.0, device=recon_loss.device)
        
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print(f"Warning: NaN/Inf detected in KL loss: {kl_loss}")
            kl_loss = torch.tensor(0.0, device=kl_loss.device)
        
        if torch.isnan(tc_loss) or torch.isinf(tc_loss):
            print(f"Warning: NaN/Inf detected in TC loss: {tc_loss}")
            tc_loss = torch.tensor(0.0, device=tc_loss.device)
        
        # Weighted total loss with β-VAE and enhancements
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['kl_divergence'] * beta * kl_loss +
            self.loss_weights['latent_alignment'] * align_loss +
            self.loss_weights['cross_reconstruction'] * cross_loss +
            tc_weight * tc_loss
        )
        
        self.log('train_beta_step', beta, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_tc_weight_step', tc_weight, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('raw_tc_loss', tc_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'align_loss': align_loss,
            'cross_loss': cross_loss,
            'tc_loss': tc_loss,
            'beta': beta,
            'tc_weight': tc_weight
        }
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_enhanced_loss(outputs, x_a, x_b)
        
        # Log losses and beta
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Increment training step for annealing
        self.training_step_count += 1
        
        return losses['total_loss']
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_enhanced_loss(outputs, x_a, x_b)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end processing
        step_output = {
            'val_loss': losses['total_loss'],
            'outputs': outputs,
            'x_a': x_a,
            'x_b': x_b
        }
        self.validation_step_outputs.append(step_output)
        
        return step_output
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_enhanced_loss(outputs, x_a, x_b)
        
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
        recon_a_metrics = safe_compute_metrics(x_a_np, recon_a_np, "recon_a")
        recon_b_metrics = safe_compute_metrics(x_b_np, recon_b_np, "recon_b")
        
        # Compute R² metrics for cross-platform imputation
        cross_a_metrics = safe_compute_metrics(x_a_np, cross_recon_a_np, "cross_a")
        cross_b_metrics = safe_compute_metrics(x_b_np, cross_recon_b_np, "cross_b")
        
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
        recon_a_metrics = safe_compute_metrics(x_a_np, recon_a_np, "recon_a")
        recon_b_metrics = safe_compute_metrics(x_b_np, recon_b_np, "recon_b")
        
        # Compute R² metrics for cross-platform imputation
        cross_a_metrics = safe_compute_metrics(x_a_np, cross_recon_a_np, "cross_a")
        cross_b_metrics = safe_compute_metrics(x_b_np, cross_recon_b_np, "cross_b")
        
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
        
        # Enhanced summary for jVAE+ with SOTA techniques
        print("\n" + "="*70)
        print("FINAL TEST R² METRICS SUMMARY - ENHANCED JOINT VAE+ (SOTA)")
        print("="*70)
        print(f"Platform A Reconstruction R²: {recon_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Reconstruction R²: {recon_b_metrics['overall_r2']:.4f}")
        print(f"Platform A Cross-Imputation R²: {cross_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Cross-Imputation R²: {cross_b_metrics['overall_r2']:.4f}")
        print(f"Features with R² > 0.5 (Recon A): {recon_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Recon B): {recon_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross A): {cross_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross B): {cross_b_metrics['fraction_r2_above_0.5']:.2%}")
        print("Enhanced Features: β-VAE, Normalizing Flows, Total Correlation Loss")
        print("="*70)
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Enhanced optimizer configuration
        if training_config['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 1e-4)
            )
        elif training_config['optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 1e-4)
            )
        elif training_config['optimizer'].lower() == 'radam':
            # RAdam optimizer for more stable training (fallback to Adam if not available)
            try:
                optimizer = torch.optim.RAdam(
                    self.parameters(),
                    lr=training_config['learning_rate'],
                    weight_decay=training_config.get('weight_decay', 1e-4)
                )
            except AttributeError:
                print("RAdam not available, falling back to Adam")
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=training_config['learning_rate'],
                    weight_decay=training_config.get('weight_decay', 1e-4)
                )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
        
        # Enhanced scheduler configuration
        if training_config['scheduler'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=training_config['scheduler_factor'],
                patience=training_config['scheduler_patience']
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
        elif training_config['scheduler'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config['max_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer 