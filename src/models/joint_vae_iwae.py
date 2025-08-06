"""
Joint Variational Autoencoder (jVAE) for cross-platform metabolite data imputation.

This module implements the core jVAE architecture with dual encoders/decoders
and a shared latent space for learning platform-invariant representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add imports for metrics
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_imputation_metrics


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        final_activation: bool = False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
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
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else input_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
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
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            final_activation=False
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class JointVAE(nn.Module):
    """
    Joint Variational Autoencoder with IWAE support for cross-platform data imputation.
    
    Architecture:
    - Two platform-specific encoders (Encoder_A, Encoder_B)
    - Shared latent space
    - Two platform-specific decoders (Decoder_A, Decoder_B)
    - Support for IWAE training with importance sampling
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
        k_samples: int = 5  # Number of importance samples for IWAE
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.latent_dim = latent_dim
        self.k_samples = k_samples
        
        # Platform A components
        self.encoder_a = Encoder(
            input_dim=input_dim_a,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
        
        self.decoder_a = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_a,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
        
        # Platform B components
        self.encoder_b = Encoder(
            input_dim=input_dim_b,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
        
        self.decoder_b = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_b,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor, k_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for VAE with support for multiple samples.
        
        Args:
            mean: Mean of the latent distribution [batch_size, latent_dim]
            logvar: Log variance of the latent distribution [batch_size, latent_dim]
            k_samples: Number of samples to draw (defaults to self.k_samples)
        
        Returns:
            z: Latent samples [k_samples, batch_size, latent_dim]
            eps: Random noise used for sampling [k_samples, batch_size, latent_dim]
        """
        if k_samples is None:
            k_samples = self.k_samples
            
        # Clamp logvar for numerical stability
        logvar_clamped = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar_clamped)  # [batch_size, latent_dim]
        
        # Expand dimensions for k samples
        mean_expanded = mean.unsqueeze(0).expand(k_samples, -1, -1)  # [k_samples, batch_size, latent_dim]
        std_expanded = std.unsqueeze(0).expand(k_samples, -1, -1)    # [k_samples, batch_size, latent_dim]
        
        # Sample noise
        eps = torch.randn_like(std_expanded)  # [k_samples, batch_size, latent_dim]
        z = mean_expanded + std_expanded * eps
        
        return z, eps
    
    def encode_a(self, x_a: torch.Tensor, k_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform A data with k samples."""
        mean, logvar = self.encoder_a(x_a)
        z, eps = self.reparameterize(mean, logvar, k_samples)
        return z, mean, logvar, eps
    
    def encode_b(self, x_b: torch.Tensor, k_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform B data with k samples."""
        mean, logvar = self.encoder_b(x_b)
        z, eps = self.reparameterize(mean, logvar, k_samples)
        return z, mean, logvar, eps
    
    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform A data."""
        # Handle multiple samples: reshape from [k_samples, batch_size, latent_dim] to [k_samples*batch_size, latent_dim]
        original_shape = z.shape
        if len(original_shape) == 3:  # [k_samples, batch_size, latent_dim]
            k_samples, batch_size, latent_dim = original_shape
            z_reshaped = z.view(k_samples * batch_size, latent_dim)
            decoded = self.decoder_a(z_reshaped)
            # Reshape back to [k_samples, batch_size, output_dim]
            return decoded.view(k_samples, batch_size, -1)
        else:
            return self.decoder_a(z)
    
    def decode_b(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to platform B data."""
        # Handle multiple samples: reshape from [k_samples, batch_size, latent_dim] to [k_samples*batch_size, latent_dim]
        original_shape = z.shape
        if len(original_shape) == 3:  # [k_samples, batch_size, latent_dim]
            k_samples, batch_size, latent_dim = original_shape
            z_reshaped = z.view(k_samples * batch_size, latent_dim)
            decoded = self.decoder_b(z_reshaped)
            # Reshape back to [k_samples, batch_size, output_dim]
            return decoded.view(k_samples, batch_size, -1)
        else:
            return self.decoder_b(z)
    
    def forward(
        self, 
        x_a: torch.Tensor, 
        x_b: torch.Tensor,
        k_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the joint VAE with IWAE sampling.
        
        Args:
            x_a: Platform A data [batch_size, input_dim_a]
            x_b: Platform B data [batch_size, input_dim_b]
            k_samples: Number of importance samples
        
        Returns:
            Dictionary containing all outputs needed for loss computation
        """
        if k_samples is None:
            k_samples = self.k_samples
            
        # Encode both platforms with k samples
        z_a, mean_a, logvar_a, eps_a = self.encode_a(x_a, k_samples)
        z_b, mean_b, logvar_b, eps_b = self.encode_b(x_b, k_samples)
        
        # Reconstruct same platform (auto-reconstruction)
        recon_a = self.decode_a(z_a)  # [k_samples, batch_size, input_dim_a]
        recon_b = self.decode_b(z_b)  # [k_samples, batch_size, input_dim_b]
        
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
            'eps_a': eps_a,
            'eps_b': eps_b,
            'recon_a': recon_a,
            'recon_b': recon_b,
            'cross_recon_a': cross_recon_a,
            'cross_recon_b': cross_recon_b,
            'k_samples': k_samples
        }
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        z_a, _, _, _ = self.encode_a(x_a, k_samples=1)
        z_a = z_a.squeeze(0)  # Remove k_samples dimension
        return self.decode_b(z_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        z_b, _, _, _ = self.encode_b(x_b, k_samples=1)
        z_b = z_b.squeeze(0)  # Remove k_samples dimension
        return self.decode_a(z_b)


class JointVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Joint VAE with IWAE support."""
    
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
        
        # Extract k_samples from config or use default
        k_samples = model_config.get('k_samples', 5)
        
        self.model = JointVAE(
            input_dim_a=input_dim_a,
            input_dim_b=input_dim_b,
            latent_dim=model_config['latent_dim'],
            encoder_layers=model_config['encoder_layers'],
            decoder_layers=model_config['decoder_layers'],
            activation=model_config['activation'],
            dropout_rate=model_config['dropout_rate'],
            batch_norm=model_config['batch_norm'],
            k_samples=k_samples
        )
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, k_samples: int = None) -> Dict[str, torch.Tensor]:
        return self.model(x_a, x_b, k_samples)
    
    def compute_iwae_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x_a: torch.Tensor, 
        x_b: torch.Tensor,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the IWAE loss function with importance weighting.
        
        Args:
            outputs: Model outputs containing samples and reconstructions
            x_a: Original platform A data [batch_size, input_dim_a]
            x_b: Original platform B data [batch_size, input_dim_b]
            is_training: Whether in training mode (affects loss computation)
        """
        k_samples = outputs['k_samples']
        batch_size = x_a.shape[0]
        
        # Expand input data to match k_samples dimension
        x_a_expanded = x_a.unsqueeze(0).expand(k_samples, -1, -1)  # [k_samples, batch_size, input_dim_a]
        x_b_expanded = x_b.unsqueeze(0).expand(k_samples, -1, -1)  # [k_samples, batch_size, input_dim_b]
        
        # Extract variables
        z_a = outputs['z_a']  # [k_samples, batch_size, latent_dim]
        z_b = outputs['z_b']  # [k_samples, batch_size, latent_dim]
        eps_a = outputs['eps_a']  # [k_samples, batch_size, latent_dim]
        eps_b = outputs['eps_b']  # [k_samples, batch_size, latent_dim]
        mean_a = outputs['mean_a']  # [batch_size, latent_dim]
        mean_b = outputs['mean_b']  # [batch_size, latent_dim]
        logvar_a = outputs['logvar_a']  # [batch_size, latent_dim]
        logvar_b = outputs['logvar_b']  # [batch_size, latent_dim]
        recon_a = outputs['recon_a']  # [k_samples, batch_size, input_dim_a]
        recon_b = outputs['recon_b']  # [k_samples, batch_size, input_dim_b]
        cross_recon_a = outputs['cross_recon_a']  # [k_samples, batch_size, input_dim_a]
        cross_recon_b = outputs['cross_recon_b']  # [k_samples, batch_size, input_dim_b]
        
        # Expand mean and logvar for k_samples dimension
        mean_a_expanded = mean_a.unsqueeze(0).expand(k_samples, -1, -1)
        mean_b_expanded = mean_b.unsqueeze(0).expand(k_samples, -1, -1)
        logvar_a_expanded = logvar_a.unsqueeze(0).expand(k_samples, -1, -1)
        logvar_b_expanded = logvar_b.unsqueeze(0).expand(k_samples, -1, -1)
        
        # Compute log probabilities
        # Prior: log p(z) = -0.5 * z^2 - 0.5 * log(2π) (standard normal prior)
        log_prior_a = torch.sum(-0.5 * z_a**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        log_prior_b = torch.sum(-0.5 * z_b**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        
        # Encoder: log q(z|x) = -0.5 * eps^2 - 0.5 * log(2π) - log(sigma)
        log_encoder_a = torch.sum(-0.5 * eps_a**2 - 0.5 * np.log(2 * np.pi) - 0.5 * logvar_a_expanded, dim=-1)  # [k_samples, batch_size]
        log_encoder_b = torch.sum(-0.5 * eps_b**2 - 0.5 * np.log(2 * np.pi) - 0.5 * logvar_b_expanded, dim=-1)  # [k_samples, batch_size]
        
        # Reconstruction likelihood: log p(x|z) assuming unit variance Gaussian
        # For simplicity, we use -0.5 * (x - x_recon)^2 which corresponds to unit variance
        log_likelihood_a = torch.sum(-0.5 * (x_a_expanded - recon_a)**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        log_likelihood_b = torch.sum(-0.5 * (x_b_expanded - recon_b)**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        
        # Cross-reconstruction likelihood
        log_cross_likelihood_a = torch.sum(-0.5 * (x_a_expanded - cross_recon_a)**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        log_cross_likelihood_b = torch.sum(-0.5 * (x_b_expanded - cross_recon_b)**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        
        # Latent alignment term (force latent representations to be similar)
        # Using negative squared distance as a log probability
        log_alignment = torch.sum(-0.5 * (mean_a_expanded - mean_b_expanded)**2 - 0.5 * np.log(2 * np.pi), dim=-1)  # [k_samples, batch_size]
        
        # Compute importance weights for each platform
        # log_weight = log p(z) + log p(x|z) - log q(z|x)
        log_weight_a = (self.loss_weights['reconstruction'] * log_likelihood_a + 
                       log_prior_a - log_encoder_a + 
                       self.loss_weights['latent_alignment'] * log_alignment +
                       self.loss_weights['cross_reconstruction'] * log_cross_likelihood_b)  # [k_samples, batch_size]
        
        log_weight_b = (self.loss_weights['reconstruction'] * log_likelihood_b + 
                       log_prior_b - log_encoder_b + 
                       self.loss_weights['latent_alignment'] * log_alignment +
                       self.loss_weights['cross_reconstruction'] * log_cross_likelihood_a)  # [k_samples, batch_size]
        
        if is_training:
            # Training loss: use importance sampling with normalized weights
            # Normalize weights to prevent overflow
            log_weight_a_norm = log_weight_a - torch.max(log_weight_a, dim=0, keepdim=True)[0]
            log_weight_b_norm = log_weight_b - torch.max(log_weight_b, dim=0, keepdim=True)[0]
            
            weight_a = torch.exp(log_weight_a_norm)
            weight_b = torch.exp(log_weight_b_norm)
            
            # Normalize weights (add small epsilon for numerical stability)
            weight_a = weight_a / (torch.sum(weight_a, dim=0, keepdim=True) + 1e-8)
            weight_b = weight_b / (torch.sum(weight_b, dim=0, keepdim=True) + 1e-8)
            
            # Detach weights to prevent gradient flow
            weight_a = weight_a.detach()
            weight_b = weight_b.detach()
            
            # Compute weighted loss
            loss_a = -torch.mean(torch.sum(weight_a * log_weight_a, dim=0))
            loss_b = -torch.mean(torch.sum(weight_b * log_weight_b, dim=0))
            
            total_loss = (loss_a + loss_b) / 2
            
        else:
            # Test loss: use log mean of unnormalized weights
            # Normalize weights first to prevent overflow
            log_weight_a_norm = log_weight_a - torch.max(log_weight_a, dim=0, keepdim=True)[0]
            log_weight_b_norm = log_weight_b - torch.max(log_weight_b, dim=0, keepdim=True)[0]
            
            weight_a = torch.exp(log_weight_a_norm)
            weight_b = torch.exp(log_weight_b_norm)
            
            # Compute log of mean of weights (add back the max for proper scaling)
            loss_a = -torch.mean(torch.log(torch.mean(weight_a, dim=0) + 1e-8) + torch.max(log_weight_a, dim=0)[0])
            loss_b = -torch.mean(torch.log(torch.mean(weight_b, dim=0) + 1e-8) + torch.max(log_weight_b, dim=0)[0])
            
            total_loss = (loss_a + loss_b) / 2
        
        # Compute individual loss components for monitoring (using mean of k samples)
        recon_loss_a = F.mse_loss(torch.mean(recon_a, dim=0), x_a)
        recon_loss_b = F.mse_loss(torch.mean(recon_b, dim=0), x_b)
        recon_loss = (recon_loss_a + recon_loss_b) / 2
        
        # KL divergence (averaged over k samples)
        kl_loss_a = 0.5 * torch.mean(mean_a.pow(2) + logvar_a.exp() - 1 - logvar_a)
        kl_loss_b = 0.5 * torch.mean(mean_b.pow(2) + logvar_b.exp() - 1 - logvar_b)
        kl_loss = (kl_loss_a + kl_loss_b) / 2
        
        # Alignment loss
        align_loss = F.mse_loss(mean_a, mean_b)
        
        # Cross-reconstruction loss
        cross_loss_a = F.mse_loss(torch.mean(cross_recon_a, dim=0), x_a)
        cross_loss_b = F.mse_loss(torch.mean(cross_recon_b, dim=0), x_b)
        cross_loss = (cross_loss_a + cross_loss_b) / 2
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'align_loss': align_loss,
            'cross_loss': cross_loss
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.
        - For k_samples > 1, uses a principled IWAE objective for reconstruction and KL,
        with other losses added as separate penalties.
        - For k_samples = 1, falls back to the standard VAE loss.
        """
        k_samples = outputs.get('k_samples', 1)

        # --- Standard VAE Loss (k=1) ---
        if k_samples <= 1:
            # This part remains the same as your original code
            for key in ['recon_a', 'recon_b', 'cross_recon_a', 'cross_recon_b']:
                if key in outputs and len(outputs[key].shape) == 3:
                    outputs[key] = outputs[key].squeeze(0)

            recon_loss_a = F.mse_loss(outputs['recon_a'], x_a)
            recon_loss_b = F.mse_loss(outputs['recon_b'], x_b)
            recon_loss = (recon_loss_a + recon_loss_b) / 2

            logvar_a_clamped = torch.clamp(outputs['logvar_a'], -10, 10)
            logvar_b_clamped = torch.clamp(outputs['logvar_b'], -10, 10)
            kl_loss_a = 0.5 * torch.mean(
                outputs['mean_a'].pow(2) + logvar_a_clamped.exp() - 1 - logvar_a_clamped
            )
            kl_loss_b = 0.5 * torch.mean(
                outputs['mean_b'].pow(2) + logvar_b_clamped.exp() - 1 - logvar_b_clamped
            )
            kl_loss = (kl_loss_a + kl_loss_b) / 2

            align_loss = F.mse_loss(outputs['mean_a'], outputs['mean_b'])

            cross_loss_a = F.mse_loss(outputs['cross_recon_a'], x_a)
            cross_loss_b = F.mse_loss(outputs['cross_recon_b'], x_b)
            cross_loss = (cross_loss_a + cross_loss_b) / 2

            total_loss = (
                self.loss_weights['reconstruction'] * recon_loss +
                self.loss_weights['kl_divergence'] * kl_loss +
                self.loss_weights['latent_alignment'] * align_loss +
                self.loss_weights['cross_reconstruction'] * cross_loss
            )
            
            return {
                'total_loss': total_loss, 'recon_loss': recon_loss,
                'kl_loss': kl_loss, 'align_loss': align_loss, 'cross_loss': cross_loss
            }

        # --- Principled IWAE Loss (k > 1) ---
        
        # 1. Calculate the standard IWAE log-weights for the data likelihood
        # These terms together form the IWAE bound on log p(x)
        
        # Expand inputs and params for k_samples dimension
        x_a_expanded = x_a.unsqueeze(0).expand(k_samples, -1, -1)
        x_b_expanded = x_b.unsqueeze(0).expand(k_samples, -1, -1)
        mean_a_expanded = outputs['mean_a'].unsqueeze(0)
        logvar_a_expanded = outputs['logvar_a'].unsqueeze(0)
        mean_b_expanded = outputs['mean_b'].unsqueeze(0)
        logvar_b_expanded = outputs['logvar_b'].unsqueeze(0)
        
        # log p(x|z): Reconstruction log-likelihood (assuming Gaussian with variance 1)
        # Complete Gaussian log-pdf: -D/2 * log(2π) - 0.5 * MSE
        D_a = x_a.shape[-1]  # dimensionality of platform A
        D_b = x_b.shape[-1]  # dimensionality of platform B
        log_p_x_given_z_a = -0.5 * F.mse_loss(outputs['recon_a'], x_a_expanded, reduction='none').sum(dim=-1) - 0.5 * D_a * np.log(2 * np.pi)
        log_p_x_given_z_b = -0.5 * F.mse_loss(outputs['recon_b'], x_b_expanded, reduction='none').sum(dim=-1) - 0.5 * D_b * np.log(2 * np.pi)

        # log p(z): Prior log-likelihood (standard normal)
        # Complete Gaussian log-pdf: -D_z/2 * log(2π) - 0.5 * ||z||²
        latent_dim = outputs['z_a'].shape[-1]
        log_p_z_a = torch.sum(-0.5 * outputs['z_a']**2, dim=-1) - 0.5 * latent_dim * np.log(2 * np.pi)
        log_p_z_b = torch.sum(-0.5 * outputs['z_b']**2, dim=-1) - 0.5 * latent_dim * np.log(2 * np.pi)
        
        # log q(z|x): Variational posterior log-likelihood
        # Complete Gaussian log-pdf: -D_z/2 * log(2π) - 0.5 * Σ[log(σ²) + ε²]
        log_q_z_given_x_a = torch.sum(-0.5 * outputs['eps_a']**2 - 0.5 * logvar_a_expanded, dim=-1) - 0.5 * latent_dim * np.log(2 * np.pi)
        log_q_z_given_x_b = torch.sum(-0.5 * outputs['eps_b']**2 - 0.5 * logvar_b_expanded, dim=-1) - 0.5 * latent_dim * np.log(2 * np.pi)

        log_weight_a = log_p_x_given_z_a + log_p_z_a - log_q_z_given_x_a
        log_weight_b = log_p_x_given_z_b + log_p_z_b - log_q_z_given_x_b

        # 2. Calculate the main IWAE objective (negative ELBO)
        if is_training:
            # For training, use the re-weighted wake-sleep update (lower variance gradients)
            # Detach the weights to stop gradients from flowing through them
            w_a_norm = F.softmax(log_weight_a, dim=0).detach()
            w_b_norm = F.softmax(log_weight_b, dim=0).detach()
            iwae_loss_a = -torch.mean(torch.sum(w_a_norm * log_weight_a, dim=0))
            iwae_loss_b = -torch.mean(torch.sum(w_b_norm * log_weight_b, dim=0))
        else:
            # For validation/testing, compute the true IWAE bound
            iwae_elbo_a = torch.mean(torch.logsumexp(log_weight_a, dim=0) - np.log(k_samples))
            iwae_elbo_b = torch.mean(torch.logsumexp(log_weight_b, dim=0) - np.log(k_samples))
            iwae_loss_a = -iwae_elbo_a
            iwae_loss_b = -iwae_elbo_b

        iwae_main_loss = (iwae_loss_a + iwae_loss_b) / 2
        
        # 3. Calculate auxiliary losses as separate, deterministic penalties
        align_loss = F.mse_loss(outputs['mean_a'], outputs['mean_b'])
        
        # Use the mean of the k reconstructions for a stable cross-loss
        cross_recon_a_mean = torch.mean(outputs['cross_recon_a'], dim=0)
        cross_recon_b_mean = torch.mean(outputs['cross_recon_b'], dim=0)
        cross_loss = (F.mse_loss(cross_recon_a_mean, x_a) + F.mse_loss(cross_recon_b_mean, x_b)) / 2
        
        # 4. Combine the IWAE objective with auxiliary penalties
        # Note: The IWAE loss already contains the reconstruction and KL terms.
        # We no longer need separate weights for them.
        total_loss = (
            iwae_main_loss +
            self.loss_weights['latent_alignment'] * align_loss +
            self.loss_weights['cross_reconstruction'] * cross_loss
        )
        
        # 5. Calculate individual components for logging purposes
        recon_a_mean = torch.mean(outputs['recon_a'], dim=0)
        recon_b_mean = torch.mean(outputs['recon_b'], dim=0)
        recon_loss_log = (F.mse_loss(recon_a_mean, x_a) + F.mse_loss(recon_b_mean, x_b)) / 2
        
        logvar_a_clamped = torch.clamp(outputs['logvar_a'], -10, 10)
        logvar_b_clamped = torch.clamp(outputs['logvar_b'], -10, 10)
        kl_loss_a_log = 0.5 * torch.mean(outputs['mean_a'].pow(2) + logvar_a_clamped.exp() - 1 - logvar_a_clamped)
        kl_loss_b_log = 0.5 * torch.mean(outputs['mean_b'].pow(2) + logvar_b_clamped.exp() - 1 - logvar_b_clamped)
        kl_loss_log = (kl_loss_a_log + kl_loss_b_log) / 2

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss_log,
            'kl_loss': kl_loss_log,
            'align_loss': align_loss,
            'cross_loss': cross_loss
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b, is_training=True)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x_a, x_b = batch
        outputs = self.forward(x_a, x_b)
        losses = self.compute_loss(outputs, x_a, x_b, is_training=False)
        
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
        losses = self.compute_loss(outputs, x_a, x_b, is_training=False)
        
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
            if key == 'k_samples':
                continue  # Skip k_samples, it's just an integer
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.validation_step_outputs], dim=-2)  # Concatenate along batch dimension
        
        # Handle k_samples dimension - take mean across k samples for metrics computation
        k_samples = self.validation_step_outputs[0]['outputs'].get('k_samples', 1)
        if k_samples > 1:
            # Average across k samples for metrics computation
            recon_a_for_metrics = torch.mean(all_outputs['recon_a'], dim=0)  # [batch_size, input_dim_a]
            recon_b_for_metrics = torch.mean(all_outputs['recon_b'], dim=0)  # [batch_size, input_dim_b]
            cross_recon_a_for_metrics = torch.mean(all_outputs['cross_recon_a'], dim=0)  # [batch_size, input_dim_a]
            cross_recon_b_for_metrics = torch.mean(all_outputs['cross_recon_b'], dim=0)  # [batch_size, input_dim_b]
        else:
            # Remove k_samples dimension if it exists
            recon_a_for_metrics = all_outputs['recon_a'].squeeze(0) if len(all_outputs['recon_a'].shape) == 3 else all_outputs['recon_a']
            recon_b_for_metrics = all_outputs['recon_b'].squeeze(0) if len(all_outputs['recon_b'].shape) == 3 else all_outputs['recon_b']
            cross_recon_a_for_metrics = all_outputs['cross_recon_a'].squeeze(0) if len(all_outputs['cross_recon_a'].shape) == 3 else all_outputs['cross_recon_a']
            cross_recon_b_for_metrics = all_outputs['cross_recon_b'].squeeze(0) if len(all_outputs['cross_recon_b'].shape) == 3 else all_outputs['cross_recon_b']
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = recon_a_for_metrics.detach().cpu().numpy()
        recon_b_np = recon_b_for_metrics.detach().cpu().numpy()
        cross_recon_a_np = cross_recon_a_for_metrics.detach().cpu().numpy()
        cross_recon_b_np = cross_recon_b_for_metrics.detach().cpu().numpy()
        
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
            if key == 'k_samples':
                continue  # Skip k_samples, it's just an integer
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.test_step_outputs], dim=-2)  # Concatenate along batch dimension
        
        # Handle k_samples dimension - take mean across k samples for metrics computation
        k_samples = self.test_step_outputs[0]['outputs'].get('k_samples', 1)
        if k_samples > 1:
            # Average across k samples for metrics computation
            recon_a_for_metrics = torch.mean(all_outputs['recon_a'], dim=0)  # [batch_size, input_dim_a]
            recon_b_for_metrics = torch.mean(all_outputs['recon_b'], dim=0)  # [batch_size, input_dim_b]
            cross_recon_a_for_metrics = torch.mean(all_outputs['cross_recon_a'], dim=0)  # [batch_size, input_dim_a]
            cross_recon_b_for_metrics = torch.mean(all_outputs['cross_recon_b'], dim=0)  # [batch_size, input_dim_b]
        else:
            # Remove k_samples dimension if it exists
            recon_a_for_metrics = all_outputs['recon_a'].squeeze(0) if len(all_outputs['recon_a'].shape) == 3 else all_outputs['recon_a']
            recon_b_for_metrics = all_outputs['recon_b'].squeeze(0) if len(all_outputs['recon_b'].shape) == 3 else all_outputs['recon_b']
            cross_recon_a_for_metrics = all_outputs['cross_recon_a'].squeeze(0) if len(all_outputs['cross_recon_a'].shape) == 3 else all_outputs['cross_recon_a']
            cross_recon_b_for_metrics = all_outputs['cross_recon_b'].squeeze(0) if len(all_outputs['cross_recon_b'].shape) == 3 else all_outputs['cross_recon_b']
        
        # Convert to numpy for metrics computation
        x_a_np = all_x_a.detach().cpu().numpy()
        x_b_np = all_x_b.detach().cpu().numpy()
        recon_a_np = recon_a_for_metrics.detach().cpu().numpy()
        recon_b_np = recon_b_for_metrics.detach().cpu().numpy()
        cross_recon_a_np = cross_recon_a_for_metrics.detach().cpu().numpy()
        cross_recon_b_np = cross_recon_b_for_metrics.detach().cpu().numpy()
        
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
        print("FINAL TEST R² METRICS SUMMARY (IWAE)")
        print("="*60)
        print(f"Platform A Reconstruction R²: {recon_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Reconstruction R²: {recon_b_metrics['overall_r2']:.4f}")
        print(f"Platform A Cross-Imputation R²: {cross_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Cross-Imputation R²: {cross_b_metrics['overall_r2']:.4f}")
        print(f"Features with R² > 0.5 (Recon A): {recon_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Recon B): {recon_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross A): {cross_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (Cross B): {cross_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Number of importance samples (k): {k_samples}")
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