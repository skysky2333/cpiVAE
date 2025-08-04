"""
Joint Variational Autoencoder (jVAE) for cross-platform omics data imputation.

This module implements both the standard Joint VAE and Mixture VAE (MMVAE) architectures 
with dual encoders/decoders and a shared latent space for learning platform-invariant representations.
Suitable for proteomics, metabolomics, transcriptomics, and other high-dimensional omics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from itertools import combinations

# Add imports for metrics and losses
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_imputation_metrics
from .losses import kl_divergence_alignment, mmd_loss


def get_mean(distribution):
    """Extract mean from a distribution or tensor."""
    if hasattr(distribution, 'mean'):
        return distribution.mean
    return distribution


class ResMLP(nn.Module):
    """
    A residual block that handles changes in dimension.
    
    This follows the post-activation residual pattern: 
    out = activation(F(x) + shortcut(x))
    
    Args:
        input_dim (int): Dimension of the input tensor.
        output_dim (int): Dimension of the output tensor.
        activation (str): The activation function name to use.
        dropout_rate (float): The dropout rate.
        batch_norm (bool): Whether to use batch normalization.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Main transformation path F(x)
        # Apply Linear -> BatchNorm -> (activation and dropout applied after residual)
        main_layers = [nn.Linear(input_dim, output_dim)]
        if batch_norm:
            main_layers.append(nn.BatchNorm1d(output_dim))
        
        self.main_path = nn.Sequential(*main_layers)

        # Shortcut connection path
        # If dimensions differ, we need a projection layer to match them.
        # Otherwise, we use an identity mapping.
        if self.input_dim != self.output_dim:
            shortcut_layers = [nn.Linear(input_dim, output_dim)]
            if batch_norm:
                shortcut_layers.append(nn.BatchNorm1d(output_dim))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()
            
        # Store activation and dropout for post-residual application
        self.activation_fn = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function from string name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU()  # Default fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The residual connection: F(x) + shortcut(x)
        shortcut_x = self.shortcut(x)
        main_x = self.main_path(x)
        
        # Apply activation after the residual addition
        out = self.activation_fn(main_x + shortcut_x)
        
        # Apply dropout after activation
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture and optional residual connections."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        final_activation: bool = False,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        if use_residual:
            # Use residual connections for hidden layers
            for i in range(len(dims) - 1):
                if i < len(dims) - 2:  # Hidden layers
                    self.layers.append(ResMLP(
                        input_dim=dims[i],
                        output_dim=dims[i + 1],
                        activation=activation,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm
                    ))
                else:  # Final layer
                    # Final layer is regular linear (no residual)
                    self.layers.append(nn.Linear(dims[i], dims[i + 1]))
                    
                    # Add final activation if specified
                    if final_activation:
                        # Batch normalization
                        if batch_norm:
                            self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                        
                        # Activation function
                        if activation == "relu":
                            self.layers.append(nn.ReLU())
                        elif activation == "leaky_relu":
                            self.layers.append(nn.LeakyReLU(0.2))
                        elif activation == "tanh":
                            self.layers.append(nn.Tanh())
                        elif activation == "elu":
                            self.layers.append(nn.ELU())
                        elif activation == "gelu":
                            self.layers.append(nn.GELU())
                        elif activation == "swish":
                            self.layers.append(nn.SiLU())
                        
                        # Dropout
                        if dropout_rate > 0:
                            self.layers.append(nn.Dropout(dropout_rate))
        else:
            # Original MLP implementation
            for i in range(len(dims) - 1):
                # Linear layer
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
                
                # Don't add activation/norm/dropout to final layer unless specified
                if i < len(dims) - 2 or final_activation:
                    # Batch normalization
                    if batch_norm:
                        self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                    
                    # Activation function
                    if activation == "relu":
                        self.layers.append(nn.ReLU())
                    elif activation == "leaky_relu":
                        self.layers.append(nn.LeakyReLU(0.2))
                    elif activation == "tanh":
                        self.layers.append(nn.Tanh())
                    elif activation == "elu":
                        self.layers.append(nn.ELU())
                    elif activation == "gelu":
                        self.layers.append(nn.GELU())
                    elif activation == "swish":
                        self.layers.append(nn.SiLU())
                    
                    # Dropout
                    if dropout_rate > 0:
                        self.layers.append(nn.Dropout(dropout_rate))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """VAE Encoder that outputs mean and log-variance for the latent distribution."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else input_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        # Output layers for mean and log-variance
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.mean_layer = nn.Linear(final_dim, latent_dim)
        self.logvar_layer = nn.Linear(final_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar


class Decoder(nn.Module):
    """VAE Decoder that reconstructs data from latent representations."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            final_activation=False,
            use_residual=use_residual
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class SingleVAE(nn.Module):
    """
    Single VAE module for one modality - used as component in MMVAE.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor, K: int = 1) -> Tuple:
        """
        Forward pass compatible with MMVAE interface.
        
        Args:
            x: Input data
            K: Number of samples (for importance sampling, currently not used)
            
        Returns:
            Tuple of (qz_x, px_z, zs) where:
            - qz_x: Approximate posterior distribution (mean, logvar)
            - px_z: Reconstruction distribution (mean tensor)
            - zs: Latent samples
        """
        # Encode
        mean, logvar = self.encoder(x)
        
        # Sample latent
        z = self.reparameterize(mean, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        # Return in MMVAE-compatible format
        qz_x = (mean, logvar)  # Posterior parameters
        px_z = reconstruction  # Reconstruction (acting as mean of distribution)
        zs = z  # Latent samples
        
        return qz_x, px_z, zs
    
    def px_z(self, *args):
        """Compatibility method for MMVAE interface."""
        if len(args) == 1:
            # Direct reconstruction tensor
            return args[0]
        else:
            # Assume decoder output
            return args[0]
    
    def dec(self, z: torch.Tensor):
        """Decode latent samples."""
        return (self.decoder(z),)


class JointVAE(nn.Module):
    """
    Joint Variational Autoencoder with support for both standard and MMVAE approaches.
    
    Architecture:
    - Two platform-specific VAEs (VAE_A, VAE_B)
    - Shared latent space
    - Cross-modal reconstruction matrix for MMVAE
    """
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        latent_dim: int = 32,
        encoder_layers: List[int] = [512, 256, 128],
        decoder_layers: List[int] = [128, 256, 512],
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_residual: bool = False,
        use_mmvae: bool = False
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.latent_dim = latent_dim
        self.use_mmvae = use_mmvae
        
        # Create individual VAE components
        self.vae_a = SingleVAE(
            input_dim=input_dim_a,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.vae_b = SingleVAE(
            input_dim=input_dim_b,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.vaes = [self.vae_a, self.vae_b]
        
        # For standard normal prior
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode_a(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform A data."""
        mean, logvar = self.vae_a.encoder(x_a)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def encode_b(self, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform B data."""
        mean, logvar = self.vae_b.encoder(x_b)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform A data."""
        return self.vae_a.decoder(z)
    
    def decode_b(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform B data."""
        return self.vae_b.decoder(z)
    
    def forward(
        self, 
        x_a: torch.Tensor, 
        x_b: torch.Tensor,
        K: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the joint VAE with optional MMVAE cross-modal matrix.
        
        Args:
            x_a: Platform A input data
            x_b: Platform B input data  
            K: Number of samples for importance sampling
            
        Returns:
            Dictionary containing all outputs needed for loss computation
        """
        if self.use_mmvae:
            return self._forward_mmvae([x_a, x_b], K)
        else:
            return self._forward_standard(x_a, x_b)
    
    def _forward_standard(
        self, 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Standard Joint VAE forward pass."""
        # Encode both platforms
        z_a, mean_a, logvar_a = self.encode_a(x_a)
        z_b, mean_b, logvar_b = self.encode_b(x_b)
        
        # Reconstruct same platform (auto-reconstruction)
        recon_a = self.decode_a(z_a)
        recon_b = self.decode_b(z_b)
        
        # Cross-platform reconstruction
        cross_recon_a = self.decode_a(z_b)  # B -> A
        cross_recon_b = self.decode_b(z_a)  # A -> B
        
        return {
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
    
    def _forward_mmvae(self, x: List[torch.Tensor], K: int = 1) -> Dict[str, torch.Tensor]:
        """MMVAE-style forward pass with cross-modal reconstruction matrix."""
        qz_xs, zss = [], []
        
        # Initialize cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        
        # Encode each modality and fill diagonal of reconstruction matrix
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # Fill diagonal (same-modality reconstruction)
        
        # Fill off-diagonal elements (cross-modal reconstructions)
        for src_idx, z_latent in enumerate(zss):
            for tgt_idx, vae in enumerate(self.vaes):
                if src_idx != tgt_idx:  # Cross-modal
                    px_zs[src_idx][tgt_idx] = vae.px_z(*vae.dec(z_latent))
        
        # Convert to standard format for compatibility
        mean_a, logvar_a = qz_xs[0]
        mean_b, logvar_b = qz_xs[1]
        z_a, z_b = zss[0], zss[1]
        
        return {
            'z_a': z_a,
            'z_b': z_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'logvar_a': logvar_a,
            'logvar_b': logvar_b,
            'recon_a': px_zs[0][0],  # A -> A
            'recon_b': px_zs[1][1],  # B -> B
            'cross_recon_a': px_zs[1][0],  # B -> A
            'cross_recon_b': px_zs[0][1],  # A -> B
            # Additional MMVAE outputs
            'qz_xs': qz_xs,
            'px_zs': px_zs,
            'zss': zss
        }
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        z_a, _, _ = self.encode_a(x_a)
        return self.decode_b(z_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        z_b, _, _ = self.encode_b(x_b)
        return self.decode_a(z_b)
    
    def generate(self, N: int, device: str = 'cpu', use_learned_prior: bool = False) -> List[torch.Tensor]:
        """Generate samples from prior for both modalities."""
        self.eval()
        with torch.no_grad():
            if use_learned_prior and hasattr(self, 'prior_mean') and hasattr(self, 'prior_logvar'):
                # Sample from learned prior if available
                std = torch.exp(0.5 * self.prior_logvar)
                eps = torch.randn(N, self.latent_dim, device=device)
                z = self.prior_mean + eps * std
            else:
                # Sample from standard normal prior
                z = torch.randn(N, self.latent_dim, device=device)
            
            # Generate for both modalities
            gen_a = self.decode_a(z)
            gen_b = self.decode_b(z)
            
        return [gen_a, gen_b]
    
    def reconstruct(self, data: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """Reconstruct data with cross-modal matrix (MMVAE style)."""
        self.eval()
        with torch.no_grad():
            if self.use_mmvae:
                outputs = self._forward_mmvae(data)
                px_zs = outputs['px_zs']
                # Convert to mean values
                recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
                return recons
            else:
                outputs = self._forward_standard(data[0], data[1])
                return [
                    [outputs['recon_a'], outputs['cross_recon_a']],
                    [outputs['cross_recon_b'], outputs['recon_b']]
                ]


class JointVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Joint VAE with MMVAE support."""
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        config: Dict,
        datamodule: Optional = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        
        self.config = config
        model_config = config['model']
        
        self.model = JointVAE(
            input_dim_a=input_dim_a,
            input_dim_b=input_dim_b,
            latent_dim=model_config['latent_dim'],
            encoder_layers=model_config['encoder_layers'],
            decoder_layers=model_config['decoder_layers'],
            activation=model_config['activation'],
            dropout_rate=model_config['dropout_rate'],
            batch_norm=model_config['batch_norm'],
            use_residual=model_config.get('use_residual_blocks', False),
            use_mmvae=model_config.get('use_mmvae', False)
        )
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # Store reference to datamodule for accessing preprocessor
        self.datamodule = datamodule
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Store original test data for inverse-transformed metrics computation
        self.original_test_data = None
    
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x_a, x_b)
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the composite loss function with optional MMVAE enhancements."""
        
        if self.model.use_mmvae:
            return self._compute_mmvae_loss(outputs, x_a, x_b)
        else:
            return self._compute_standard_loss(outputs, x_a, x_b)
    
    def _compute_standard_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Standard Joint VAE loss computation."""
        # Reconstruction losses
        recon_loss_a = F.mse_loss(outputs['recon_a'], x_a)
        recon_loss_b = F.mse_loss(outputs['recon_b'], x_b)
        recon_loss = (recon_loss_a + recon_loss_b) / 2
        
        # KL divergence losses (numerically stable version)
        kl_loss_a = 0.5 * torch.mean(
            outputs['mean_a'].pow(2) + outputs['logvar_a'].exp() - 1 - outputs['logvar_a']
        )
        kl_loss_b = 0.5 * torch.mean(
            outputs['mean_b'].pow(2) + outputs['logvar_b'].exp() - 1 - outputs['logvar_b']
        )
        kl_loss = (kl_loss_a + kl_loss_b) / 2
        
        # Latent alignment loss (force latent representations to be similar)
        alignment_type = self.config['loss_weights'].get('alignment_type', 'mse')
        
        if alignment_type == 'kl_divergence':
            # KL divergence between the two posterior distributions
            align_loss = kl_divergence_alignment(
                outputs['mean_a'], outputs['logvar_a'],
                outputs['mean_b'], outputs['logvar_b']
            )
        elif alignment_type == 'mmd':
            # Maximum Mean Discrepancy (MMD) loss on sampled latent variables
            align_loss = mmd_loss(outputs['z_a'], outputs['z_b'])
        else:
            # Default MSE alignment
            align_loss = F.mse_loss(outputs['mean_a'], outputs['mean_b'])
        
        # Cross-reconstruction losses
        cross_loss_a = F.mse_loss(outputs['cross_recon_a'], x_a)
        cross_loss_b = F.mse_loss(outputs['cross_recon_b'], x_b)
        cross_loss = (cross_loss_a + cross_loss_b) / 2
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['kl_divergence'] * kl_loss +
            self.loss_weights['latent_alignment'] * align_loss +
            self.loss_weights['cross_reconstruction'] * cross_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'align_loss': align_loss,
            'cross_loss': cross_loss
        }
    
    def _compute_mmvae_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """MMVAE loss computation with cross-modal considerations."""
        x = [x_a, x_b]
        qz_xs = outputs['qz_xs']
        px_zs = outputs['px_zs']
        zss = outputs['zss']
        
        # Reconstruction losses (diagonal elements of cross-modal matrix)
        recon_loss_a = F.mse_loss(px_zs[0][0], x_a)
        recon_loss_b = F.mse_loss(px_zs[1][1], x_b)
        recon_loss = (recon_loss_a + recon_loss_b) / 2
        
        # KL divergence losses
        mean_a, logvar_a = qz_xs[0]
        mean_b, logvar_b = qz_xs[1]
        
        kl_loss_a = 0.5 * torch.mean(
            mean_a.pow(2) + logvar_a.exp() - 1 - logvar_a
        )
        kl_loss_b = 0.5 * torch.mean(
            mean_b.pow(2) + logvar_b.exp() - 1 - logvar_b
        )
        kl_loss = (kl_loss_a + kl_loss_b) / 2
        
        # Cross-modal reconstruction losses (off-diagonal elements)
        cross_loss_a = F.mse_loss(px_zs[1][0], x_a)  # B -> A
        cross_loss_b = F.mse_loss(px_zs[0][1], x_b)  # A -> B
        cross_loss = (cross_loss_a + cross_loss_b) / 2
        
        # Latent alignment loss
        alignment_type = self.config['loss_weights'].get('alignment_type', 'mse')
        
        if alignment_type == 'kl_divergence':
            align_loss = kl_divergence_alignment(mean_a, logvar_a, mean_b, logvar_b)
        elif alignment_type == 'mmd':
            align_loss = mmd_loss(zss[0], zss[1])
        else:
            align_loss = F.mse_loss(mean_a, mean_b)
        
        # MMVAE-specific loss: encourage cross-modal coherence
        mmvae_coherence_loss = 0.0
        if self.loss_weights.get('mmvae_coherence', 0) > 0:
            # Penalize inconsistency in cross-modal reconstructions
            # E.g., A->B->A should be close to A
            # Use already computed latents to avoid double forward pass
            with torch.no_grad():
                # Detach cross-modal reconstructions to avoid gradient conflicts
                cross_recon_a_detached = px_zs[1][0].detach()  # B -> A
                cross_recon_b_detached = px_zs[0][1].detach()  # A -> B
            
            # Encode cross-modal reconstructions back to latent space
            z_a_via_b, _, _ = self.model.encode_a(cross_recon_a_detached)
            z_b_via_a, _, _ = self.model.encode_b(cross_recon_b_detached)
            
            # Decode back to original space
            recon_aa_via_b = self.model.decode_a(z_a_via_b)
            recon_bb_via_a = self.model.decode_b(z_b_via_a)
            
            mmvae_coherence_loss = (
                F.mse_loss(recon_aa_via_b, x_a) + 
                F.mse_loss(recon_bb_via_a, x_b)
            ) / 2
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['kl_divergence'] * kl_loss +
            self.loss_weights['latent_alignment'] * align_loss +
            self.loss_weights['cross_reconstruction'] * cross_loss +
            self.loss_weights.get('mmvae_coherence', 0) * mmvae_coherence_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'align_loss': align_loss,
            'cross_loss': cross_loss,
            'mmvae_coherence_loss': mmvae_coherence_loss
        }
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_a, x_b = batch
        
        # Apply data augmentation (Gaussian noise) during training
        aug_config = self.config['training'].get('data_augmentation', {})
        if aug_config.get('enabled', False):
            noise_std = aug_config.get('gaussian_noise_std', 0.01)
            x_a = x_a + torch.randn_like(x_a) * noise_std
            x_b = x_b + torch.randn_like(x_b) * noise_std
        
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b)
        
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
            if key not in ['qz_xs', 'px_zs', 'zss']:  # Skip MMVAE-specific complex structures
                all_outputs[key] = torch.cat([output['outputs'][key] for output in self.validation_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = all_outputs['recon_a'].detach().cpu().numpy()
        recon_b_np = all_outputs['recon_b'].detach().cpu().numpy()
        cross_recon_a_np = all_outputs['cross_recon_a'].detach().cpu().numpy()
        cross_recon_b_np = all_outputs['cross_recon_b'].detach().cpu().numpy()
        
        # Apply inverse transformations to get data in original scale
        if self.datamodule is not None:
            preprocessor = self.datamodule.get_preprocessor()
            
            # First inverse normalize
            if preprocessor.scalers['platform_a'] is not None:
                x_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(x_a_np)
                recon_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(recon_a_np)
                cross_recon_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(cross_recon_a_np)
            else:
                x_a_inv_norm = x_a_np
                recon_a_inv_norm = recon_a_np
                cross_recon_a_inv_norm = cross_recon_a_np
            
            if preprocessor.scalers['platform_b'] is not None:
                x_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(x_b_np)
                recon_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(recon_b_np)
                cross_recon_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(cross_recon_b_np)
            else:
                x_b_inv_norm = x_b_np
                recon_b_inv_norm = recon_b_np
                cross_recon_b_inv_norm = cross_recon_b_np
            
            # Then apply inverse log transformation if enabled
            x_a_orig, x_b_orig = preprocessor.inverse_log_transform_data(x_a_inv_norm, x_b_inv_norm)
            recon_a_orig, recon_b_orig = preprocessor.inverse_log_transform_data(recon_a_inv_norm, recon_b_inv_norm)
            cross_recon_a_orig, cross_recon_b_orig = preprocessor.inverse_log_transform_data(cross_recon_a_inv_norm, cross_recon_b_inv_norm)
        else:
            # If no preprocessor available, use the data as-is (this shouldn't happen in normal training)
            print("ATTENTION!!!! No preprocessor available, using data as-is")
            x_a_orig, x_b_orig = x_a_np, x_b_np
            recon_a_orig, recon_b_orig = recon_a_np, recon_b_np
            cross_recon_a_orig, cross_recon_b_orig = cross_recon_a_np, cross_recon_b_np
        
        # Compute R² metrics for reconstruction tasks on original scale
        recon_a_metrics = compute_imputation_metrics(x_a_orig, recon_a_orig)
        recon_b_metrics = compute_imputation_metrics(x_b_orig, recon_b_orig)
        
        # Compute R² metrics for cross-platform imputation on original scale
        cross_a_metrics = compute_imputation_metrics(x_a_orig, cross_recon_a_orig)
        cross_b_metrics = compute_imputation_metrics(x_b_orig, cross_recon_b_orig)
        
        # Log key R² metrics (original scale)
        self.log('val_recon_a_r2', recon_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_recon_b_r2', recon_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_cross_a_r2', cross_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_cross_b_r2', cross_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log Pearson correlation r metrics (original scale)
        self.log('val_recon_a_corr', recon_a_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('val_recon_b_corr', recon_b_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('val_cross_a_corr', cross_a_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('val_cross_b_corr', cross_b_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        
        # Log per-feature correlation statistics (original scale)
        self.log('val_recon_a_corr_mean', recon_a_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('val_recon_b_corr_mean', recon_b_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('val_cross_a_corr_mean', cross_a_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('val_cross_b_corr_mean', cross_b_metrics['mean_feature_correlation'], on_epoch=True)
        
        self.log('val_recon_a_corr_median', recon_a_metrics['median_feature_correlation'], on_epoch=True)
        self.log('val_recon_b_corr_median', recon_b_metrics['median_feature_correlation'], on_epoch=True)
        self.log('val_cross_a_corr_median', cross_a_metrics['median_feature_correlation'], on_epoch=True)
        self.log('val_cross_b_corr_median', cross_b_metrics['median_feature_correlation'], on_epoch=True)
        
        # Log mean feature-wise R² scores (original scale)
        self.log('val_recon_a_mean_feature_r2', recon_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_recon_b_mean_feature_r2', recon_b_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_cross_a_mean_feature_r2', cross_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_cross_b_mean_feature_r2', cross_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics (original scale)
        self.log('val_recon_a_frac_r2_above_0.5', recon_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_recon_b_frac_r2_above_0.5', recon_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_cross_a_frac_r2_above_0.5', cross_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_cross_b_frac_r2_above_0.5', cross_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
        # MMVAE-specific metrics if enabled
        if self.model.use_mmvae and len(self.validation_step_outputs) > 0:
            sample_output = self.validation_step_outputs[0]['outputs']
            if 'qz_xs' in sample_output:
                try:
                    # Analyze latent alignment - aggregate across all batches
                    all_mean_a, all_logvar_a = [], []
                    all_mean_b, all_logvar_b = [], []
                    
                    for output in self.validation_step_outputs:
                        if 'qz_xs' in output['outputs']:
                            qz_xs = output['outputs']['qz_xs']
                            mean_a, logvar_a = qz_xs[0]
                            mean_b, logvar_b = qz_xs[1]
                            all_mean_a.append(mean_a)
                            all_logvar_a.append(logvar_a)
                            all_mean_b.append(mean_b)
                            all_logvar_b.append(logvar_b)
                    
                    if all_mean_a:
                        # Concatenate across batches for comprehensive analysis
                        agg_mean_a = torch.cat(all_mean_a, dim=0)
                        agg_logvar_a = torch.cat(all_logvar_a, dim=0)
                        agg_mean_b = torch.cat(all_mean_b, dim=0)
                        agg_logvar_b = torch.cat(all_logvar_b, dim=0)
                        
                        agg_qz_xs = [(agg_mean_a, agg_logvar_a), (agg_mean_b, agg_logvar_b)]
                        
                        from utils.metrics import analyze_mmvae_latent_alignment
                        alignment_metrics = analyze_mmvae_latent_alignment(agg_qz_xs)
                        
                        for key, value in alignment_metrics.items():
                            self.log(f'val_{key}', value, on_epoch=True)
                except Exception as e:
                    print(f"MMVAE validation analysis failed: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def set_original_test_data(self, original_test_a: np.ndarray, original_test_b: np.ndarray):
        """Set the original test data for inverse-transformed metrics computation."""
        self.original_test_data = {
            'test_a': original_test_a,
            'test_b': original_test_b
        }
    
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
            if key not in ['qz_xs', 'px_zs', 'zss']:  # Skip MMVAE-specific complex structures
                all_outputs[key] = torch.cat([output['outputs'][key] for output in self.test_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = all_outputs['recon_a'].detach().cpu().numpy()
        recon_b_np = all_outputs['recon_b'].detach().cpu().numpy()
        cross_recon_a_np = all_outputs['cross_recon_a'].detach().cpu().numpy()
        cross_recon_b_np = all_outputs['cross_recon_b'].detach().cpu().numpy()
        
        # Apply inverse transformations to get data in original scale
        if self.datamodule is not None:
            preprocessor = self.datamodule.get_preprocessor()
            
            # First inverse normalize
            if preprocessor.scalers['platform_a'] is not None:
                x_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(x_a_np)
                recon_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(recon_a_np)
                cross_recon_a_inv_norm = preprocessor.scalers['platform_a'].inverse_transform(cross_recon_a_np)
            else:
                x_a_inv_norm = x_a_np
                recon_a_inv_norm = recon_a_np
                cross_recon_a_inv_norm = cross_recon_a_np
            
            if preprocessor.scalers['platform_b'] is not None:
                x_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(x_b_np)
                recon_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(recon_b_np)
                cross_recon_b_inv_norm = preprocessor.scalers['platform_b'].inverse_transform(cross_recon_b_np)
            else:
                x_b_inv_norm = x_b_np
                recon_b_inv_norm = recon_b_np
                cross_recon_b_inv_norm = cross_recon_b_np
            
            # Then apply inverse log transformation if enabled
            x_a_orig, x_b_orig = preprocessor.inverse_log_transform_data(x_a_inv_norm, x_b_inv_norm)
            recon_a_orig, recon_b_orig = preprocessor.inverse_log_transform_data(recon_a_inv_norm, recon_b_inv_norm)
            cross_recon_a_orig, cross_recon_b_orig = preprocessor.inverse_log_transform_data(cross_recon_a_inv_norm, cross_recon_b_inv_norm)
            
            print("\n" + "="*60)
            print("COMPUTING METRICS ON INVERSE-TRANSFORMED DATA (ORIGINAL SCALE)")
            print("="*60)
        else:
            # If no preprocessor available, use the data as-is (this shouldn't happen in normal training)
            x_a_orig, x_b_orig = x_a_np, x_b_np
            recon_a_orig, recon_b_orig = recon_a_np, recon_b_np
            cross_recon_a_orig, cross_recon_b_orig = cross_recon_a_np, cross_recon_b_np
            
            print("\n" + "="*60)
            print("NO PREPROCESSOR AVAILABLE - USING DATA AS-IS")
            print("="*60)
        
        # Compute R² metrics for reconstruction tasks on original scale
        recon_a_metrics = compute_imputation_metrics(x_a_orig, recon_a_orig)
        recon_b_metrics = compute_imputation_metrics(x_b_orig, recon_b_orig)
        
        # Compute R² metrics for cross-platform imputation on original scale
        cross_a_metrics = compute_imputation_metrics(x_a_orig, cross_recon_a_orig)
        cross_b_metrics = compute_imputation_metrics(x_b_orig, cross_recon_b_orig)
        
        # Log key R² metrics (original scale)
        self.log('test_recon_a_r2', recon_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_recon_b_r2', recon_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_cross_a_r2', cross_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_cross_b_r2', cross_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log Pearson correlation r metrics (original scale)
        self.log('test_recon_a_corr', recon_a_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('test_recon_b_corr', recon_b_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('test_cross_a_corr', cross_a_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('test_cross_b_corr', cross_b_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        
        # Log per-feature correlation statistics (original scale)
        self.log('test_recon_a_corr_mean', recon_a_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('test_recon_b_corr_mean', recon_b_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('test_cross_a_corr_mean', cross_a_metrics['mean_feature_correlation'], on_epoch=True)
        self.log('test_cross_b_corr_mean', cross_b_metrics['mean_feature_correlation'], on_epoch=True)
        
        self.log('test_recon_a_corr_median', recon_a_metrics['median_feature_correlation'], on_epoch=True)
        self.log('test_recon_b_corr_median', recon_b_metrics['median_feature_correlation'], on_epoch=True)
        self.log('test_cross_a_corr_median', cross_a_metrics['median_feature_correlation'], on_epoch=True)
        self.log('test_cross_b_corr_median', cross_b_metrics['median_feature_correlation'], on_epoch=True)
        
        # Log mean feature-wise R² scores (original scale)
        self.log('test_recon_a_mean_feature_r2', recon_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_recon_b_mean_feature_r2', recon_b_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_cross_a_mean_feature_r2', cross_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_cross_b_mean_feature_r2', cross_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics (original scale)
        self.log('test_recon_a_frac_r2_above_0.5', recon_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_recon_b_frac_r2_above_0.5', recon_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_cross_a_frac_r2_above_0.5', cross_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_cross_b_frac_r2_above_0.5', cross_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
        # Log additional detailed metrics for final test evaluation (original scale)
        self.log('test_recon_a_median_feature_r2', recon_a_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_recon_b_median_feature_r2', recon_b_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_cross_a_median_feature_r2', cross_a_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_cross_b_median_feature_r2', cross_b_metrics['median_feature_r2'], on_epoch=True)
        
        self.log('test_recon_a_frac_r2_above_0.7', recon_a_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_recon_b_frac_r2_above_0.7', recon_b_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_cross_a_frac_r2_above_0.7', cross_a_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_cross_b_frac_r2_above_0.7', cross_b_metrics['fraction_r2_above_0.7'], on_epoch=True)
        
        # MMVAE-specific test metrics
        if self.model.use_mmvae and len(self.test_step_outputs) > 0:
            try:
                # Compute cross-modal reconstruction matrix metrics
                sample_output = self.test_step_outputs[0]['outputs']
                if 'px_zs' in sample_output:
                    # Collect all cross-modal reconstructions
                    all_px_zs = []
                    for output in self.test_step_outputs:
                        if 'px_zs' in output['outputs']:
                            all_px_zs.append(output['outputs']['px_zs'])
                    
                    if all_px_zs:
                        # Use first batch for comprehensive analysis
                        px_zs = all_px_zs[0]
                        x_true = [all_x_a, all_x_b]
                        
                        from utils.metrics import compute_cross_modal_metrics
                        cross_modal_metrics = compute_cross_modal_metrics(
                            px_zs, x_true, modality_names=['platform_a', 'platform_b']
                        )
                        
                        # Log cross-modal metrics
                        for path_key, metrics in cross_modal_metrics.items():
                            self.log(f'test_{path_key}_r2', metrics['overall_r2'], on_epoch=True)
                            self.log(f'test_{path_key}_corr', metrics['overall_correlation'], on_epoch=True)
                            self.log(f'test_{path_key}_mean_feature_r2', metrics['mean_feature_r2'], on_epoch=True)
                
                # Analyze latent alignment - aggregate across all batches
                all_mean_a, all_logvar_a = [], []
                all_mean_b, all_logvar_b = [], []
                
                for output in self.test_step_outputs:
                    if 'qz_xs' in output['outputs']:
                        qz_xs = output['outputs']['qz_xs']
                        mean_a, logvar_a = qz_xs[0]
                        mean_b, logvar_b = qz_xs[1]
                        all_mean_a.append(mean_a)
                        all_logvar_a.append(logvar_a)
                        all_mean_b.append(mean_b)
                        all_logvar_b.append(logvar_b)
                
                if all_mean_a:
                    # Concatenate across batches for comprehensive analysis
                    agg_mean_a = torch.cat(all_mean_a, dim=0)
                    agg_logvar_a = torch.cat(all_logvar_a, dim=0)
                    agg_mean_b = torch.cat(all_mean_b, dim=0)
                    agg_logvar_b = torch.cat(all_logvar_b, dim=0)
                    
                    agg_qz_xs = [(agg_mean_a, agg_logvar_a), (agg_mean_b, agg_logvar_b)]
                    
                    from utils.metrics import analyze_mmvae_latent_alignment
                    alignment_metrics = analyze_mmvae_latent_alignment(agg_qz_xs)
                    
                    for key, value in alignment_metrics.items():
                        self.log(f'test_{key}', value, on_epoch=True)
                    
                    print("\nMMVAE LATENT ALIGNMENT ANALYSIS:")
                    print("="*40)
                    for key, value in alignment_metrics.items():
                        print(f"{key}: {value:.4f}")
                    
            except Exception as e:
                print(f"MMVAE test analysis failed: {e}")
        
        # Print summary for final test results (original scale)
        print("FINAL TEST METRICS SUMMARY (ORIGINAL DATA SCALE)")
        print("="*60)
        print(f"Platform A Reconstruction R²: {recon_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Reconstruction R²: {recon_b_metrics['overall_r2']:.4f}")
        print(f"Platform A Cross-Imputation R²: {cross_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Cross-Imputation R²: {cross_b_metrics['overall_r2']:.4f}")
        print(f"Platform A Reconstruction Corr (flattened): {recon_a_metrics['overall_correlation']:.4f}")
        print(f"Platform B Reconstruction Corr (flattened): {recon_b_metrics['overall_correlation']:.4f}")
        print(f"Platform A Cross-Imputation Corr (flattened): {cross_a_metrics['overall_correlation']:.4f}")
        print(f"Platform B Cross-Imputation Corr (flattened): {cross_b_metrics['overall_correlation']:.4f}")
        print(f"Platform A Reconstruction Corr (mean per-feature): {recon_a_metrics['mean_feature_correlation']:.4f}")
        print(f"Platform B Reconstruction Corr (mean per-feature): {recon_b_metrics['mean_feature_correlation']:.4f}")
        print(f"Platform A Cross-Imputation Corr (mean per-feature): {cross_a_metrics['mean_feature_correlation']:.4f}")
        print(f"Platform B Cross-Imputation Corr (mean per-feature): {cross_b_metrics['mean_feature_correlation']:.4f}")
        print(f"Platform A Reconstruction Corr (median per-feature): {recon_a_metrics['median_feature_correlation']:.4f}")
        print(f"Platform B Reconstruction Corr (median per-feature): {recon_b_metrics['median_feature_correlation']:.4f}")
        print(f"Platform A Cross-Imputation Corr (median per-feature): {cross_a_metrics['median_feature_correlation']:.4f}")
        print(f"Platform B Cross-Imputation Corr (median per-feature): {cross_b_metrics['median_feature_correlation']:.4f}")
        print(f"Features with R² > 0.5 (Recon A): {recon_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Recon B): {recon_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross A): {cross_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross B): {cross_b_metrics['fraction_r2_above_0.5']:.2%}")
        print("="*60)
        
        # Clear outputs
        self.test_step_outputs.clear()
    
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
        else:
            return optimizer 