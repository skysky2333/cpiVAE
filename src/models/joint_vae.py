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

# Add imports for metrics and losses
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_imputation_metrics
from .losses import kl_divergence_alignment, mmd_loss


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


class JointVAE(nn.Module):
    """
    Joint Variational Autoencoder for cross-platform data imputation.
    
    Architecture:
    - Two platform-specific encoders (Encoder_A, Encoder_B)
    - Shared latent space
    - Two platform-specific decoders (Decoder_A, Decoder_B)
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
        use_residual: bool = False
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.latent_dim = latent_dim
        
        # Platform A components
        self.encoder_a = Encoder(
            input_dim=input_dim_a,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.decoder_a = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_a,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        # Platform B components
        self.encoder_b = Encoder(
            input_dim=input_dim_b,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        self.decoder_b = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim_b,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode_a(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform A data."""
        mean, logvar = self.encoder_a(x_a)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def encode_b(self, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode platform B data."""
        mean, logvar = self.encoder_b(x_b)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
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
        Forward pass through the joint VAE.
        
        Returns:
            Dictionary containing all outputs needed for loss computation
        """
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
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        z_a, _, _ = self.encode_a(x_a)
        return self.decode_b(z_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        z_b, _, _ = self.encode_b(x_b)
        return self.decode_a(z_b)


class JointVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Joint VAE."""
    
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
            use_residual=model_config.get('use_residual_blocks', False)  # Default to False for backward compatibility
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
        """Compute the composite loss function."""
        
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