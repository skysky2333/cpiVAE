"""
Quality Control Variational Autoencoder (QC-VAE) for anomaly detection in metabolite data.

This module implements a single VAE for self-reconstruction quality control,
designed to identify anomalous samples based on reconstruction error and latent likelihood.
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
        main_layers = [nn.Linear(input_dim, output_dim)]
        if batch_norm:
            main_layers.append(nn.BatchNorm1d(output_dim))
        
        self.main_path = nn.Sequential(*main_layers)

        # Shortcut connection path
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
                        if batch_norm:
                            self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                        
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
                        
                        if dropout_rate > 0:
                            self.layers.append(nn.Dropout(dropout_rate))
        else:
            # Original MLP implementation
            for i in range(len(dims) - 1):
                # Linear layer
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
                
                # Don't add activation/norm/dropout to final layer unless specified
                if i < len(dims) - 2 or final_activation:
                    if batch_norm:
                        self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                    
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


class QCVAE(nn.Module):
    """
    Quality Control Variational Autoencoder for anomaly detection.
    
    Architecture:
    - Single encoder
    - Single decoder  
    - Self-reconstruction for quality control
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_layers: List[int] = [512, 256, 128],
        decoder_layers: List[int] = [128, 256, 512],
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        # Clamp logvar to prevent extreme values that could cause NaN
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data."""
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the QC VAE.
        
        Returns:
            Dictionary containing all outputs needed for loss computation and QC
        """
        # Encode
        z, mean, logvar = self.encode(x)
        
        # Reconstruct
        recon = self.decode(z)
        
        return {
            'z': z,
            'mean': mean,
            'logvar': logvar,
            'recon': recon
        }
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Get reconstruction error for each sample."""
        outputs = self.forward(x)
        # Per-sample MSE
        recon_error = torch.mean((outputs['recon'] - x) ** 2, dim=1)
        return recon_error
    
    def get_latent_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """Get negative KL divergence (latent likelihood) for each sample."""
        outputs = self.forward(x)
        # Per-sample KL divergence (higher = more anomalous)
        # Clamp logvar for numerical stability
        logvar_clamped = torch.clamp(outputs['logvar'], min=-20, max=20)
        kl_div = 0.5 * torch.sum(
            outputs['mean'].pow(2) + logvar_clamped.exp() - 1 - logvar_clamped, 
            dim=1
        )
        # Return negative KL (higher = more likely under prior)
        return -kl_div


class QCVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for QC VAE."""
    
    def __init__(
        self,
        input_dim: int,
        config: Dict,
        datamodule: Optional = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'])
        
        self.config = config
        model_config = config['model']
        
        self.model = QCVAE(
            input_dim=input_dim,
            latent_dim=model_config['latent_dim'],
            encoder_layers=model_config['encoder_layers'],
            decoder_layers=model_config['decoder_layers'],
            activation=model_config['activation'],
            dropout_rate=model_config['dropout_rate'],
            batch_norm=model_config['batch_norm'],
            use_residual=model_config.get('use_residual_blocks', False)
        )
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
        # Store reference to datamodule for accessing preprocessor
        self.datamodule = datamodule
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss function."""
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['recon'], x)
        
        # KL divergence loss (numerically stable version)
        # Clamp logvar to prevent numerical instability
        logvar_clamped = torch.clamp(outputs['logvar'], min=-20, max=20)
        kl_loss = 0.5 * torch.mean(
            outputs['mean'].pow(2) + logvar_clamped.exp() - 1 - logvar_clamped
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['kl_divergence'] * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        
        # Apply data augmentation (Gaussian noise) during training
        aug_config = self.config['training'].get('data_augmentation', {})
        if aug_config.get('enabled', False):
            noise_std = aug_config.get('gaussian_noise_std', 0.01)
            x = x + torch.randn_like(x) * noise_std
        
        outputs = self.forward(x)
        losses = self.compute_loss(outputs, batch)  # Use original batch for loss
        
        # Check for NaN values and skip if found
        if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
            print(f"Warning: NaN/Inf detected in training step {batch_idx}")
            print(f"Recon loss: {losses['recon_loss']}")
            print(f"KL loss: {losses['kl_loss']}")
            print(f"Mean range: [{outputs['mean'].min():.4f}, {outputs['mean'].max():.4f}]")
            print(f"Logvar range: [{outputs['logvar'].min():.4f}, {outputs['logvar'].max():.4f}]")
            print(f"Recon range: [{outputs['recon'].min():.4f}, {outputs['recon'].max():.4f}]")
            # Return a small dummy loss to prevent crash
            return torch.tensor(0.001, requires_grad=True, device=self.device)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        x = batch
        outputs = self.forward(x)
        losses = self.compute_loss(outputs, x)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end processing
        step_output = {
            'val_loss': losses['total_loss'],
            'outputs': outputs,
            'x': x
        }
        self.validation_step_outputs.append(step_output)
        
        return step_output
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        x = batch
        outputs = self.forward(x)
        losses = self.compute_loss(outputs, x)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'test_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        # Store outputs for epoch-end processing
        step_output = {
            'test_loss': losses['total_loss'],
            'outputs': outputs,
            'x': x
        }
        self.test_step_outputs.append(step_output)
        
        return step_output
    
    def on_validation_epoch_end(self):
        """Compute additional metrics at the end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Concatenate all outputs from validation steps
        all_x = torch.cat([output['x'] for output in self.validation_step_outputs], dim=0)
        all_outputs = {}
        
        # Concatenate all output tensors
        for key in self.validation_step_outputs[0]['outputs'].keys():
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.validation_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_np = all_x.detach().cpu().numpy()
        recon_np = all_outputs['recon'].detach().cpu().numpy()
        
        # Apply inverse transformations to get data in original scale
        if self.datamodule is not None:
            preprocessor = self.datamodule.get_preprocessor()
            
            # First inverse normalize
            if preprocessor.scaler is not None:
                x_inv_norm = preprocessor.scaler.inverse_transform(x_np)
                recon_inv_norm = preprocessor.scaler.inverse_transform(recon_np)
            else:
                x_inv_norm = x_np
                recon_inv_norm = recon_np
            
            # Then apply inverse log transformation if enabled
            x_orig = preprocessor.inverse_log_transform_single(x_inv_norm)
            recon_orig = preprocessor.inverse_log_transform_single(recon_inv_norm)
        else:
            x_orig, recon_orig = x_np, recon_np
        
        # Compute R² metrics for reconstruction on original scale
        recon_metrics = compute_imputation_metrics(x_orig, recon_orig)
        
        # Log key R² metrics (original scale)
        self.log('val_recon_r2', recon_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_recon_corr', recon_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('val_recon_mean_feature_r2', recon_metrics['mean_feature_r2'], on_epoch=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Compute additional metrics at the end of test epoch."""
        if not self.test_step_outputs:
            return
        
        # Concatenate all outputs from test steps
        all_x = torch.cat([output['x'] for output in self.test_step_outputs], dim=0)
        all_outputs = {}
        
        # Concatenate all output tensors
        for key in self.test_step_outputs[0]['outputs'].keys():
            all_outputs[key] = torch.cat([output['outputs'][key] for output in self.test_step_outputs], dim=0)
        
        # Convert to numpy for metrics computation
        x_np = all_x.detach().cpu().numpy()
        recon_np = all_outputs['recon'].detach().cpu().numpy()
        
        # Apply inverse transformations to get data in original scale
        if self.datamodule is not None:
            preprocessor = self.datamodule.get_preprocessor()
            
            # First inverse normalize
            if preprocessor.scaler is not None:
                x_inv_norm = preprocessor.scaler.inverse_transform(x_np)
                recon_inv_norm = preprocessor.scaler.inverse_transform(recon_np)
            else:
                x_inv_norm = x_np
                recon_inv_norm = recon_np
            
            # Then apply inverse log transformation if enabled
            x_orig = preprocessor.inverse_log_transform_single(x_inv_norm)
            recon_orig = preprocessor.inverse_log_transform_single(recon_inv_norm)
        else:
            x_orig, recon_orig = x_np, recon_np
        
        # Compute R² metrics for reconstruction on original scale
        recon_metrics = compute_imputation_metrics(x_orig, recon_orig)
        
        # Log key R² metrics (original scale)
        self.log('test_recon_r2', recon_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_recon_corr', recon_metrics['overall_correlation'], on_epoch=True, prog_bar=True)
        self.log('test_recon_mean_feature_r2', recon_metrics['mean_feature_r2'], on_epoch=True)
        
        # Print summary for final test results (original scale)
        print("\nFINAL QC VAE TEST METRICS SUMMARY (ORIGINAL DATA SCALE)")
        print("="*60)
        print(f"Reconstruction R²: {recon_metrics['overall_r2']:.4f}")
        print(f"Reconstruction Correlation: {recon_metrics['overall_correlation']:.4f}")
        print(f"Mean Feature R²: {recon_metrics['mean_feature_r2']:.4f}")
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