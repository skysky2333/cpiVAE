"""
Joint Variational Autoencoder with Vector Quantization (jVQ-VAE) for cross-platform data imputation.

This module implements the core jVQ-VAE architecture with dual encoders/decoders,
a shared quantized latent space for learning platform-invariant representations.
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


def _get_activation_fn(activation: str) -> nn.Module:
    """Helper to get activation function from string."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class ResidualBlock(nn.Module):
    """A simple residual block for MLPs with a skip connection."""
    def __init__(self, dim: int, activation_fn: nn.Module, dropout_rate: float, batch_norm: bool):
        super().__init__()
        
        layers = [nn.Linear(dim, dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        
        layers.append(activation_fn)
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


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
                self.layers.append(_get_activation_fn(activation))
                
                # Dropout
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """Encoder that outputs a latent representation for VQ-VAE."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        num_res_blocks: int = 0
    ):
        super().__init__()
        
        # Project input to latent dimension
        self.initial_projection = MLP(
            input_dim=input_dim,
            hidden_dims=[],
            output_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            final_activation=True
        )

        # Series of residual blocks in the latent dimension
        act_fn = _get_activation_fn(activation)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, act_fn, dropout_rate, batch_norm) for _ in range(num_res_blocks)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.initial_projection(x)
        if self.res_blocks:
            h = self.res_blocks(h)
        return h


class Decoder(nn.Module):
    """VAE Decoder that reconstructs data from latent representations."""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        num_res_blocks: int = 0
    ):
        super().__init__()

        # Series of residual blocks in the latent dimension
        act_fn = _get_activation_fn(activation)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, act_fn, dropout_rate, batch_norm) for _ in range(num_res_blocks)]
        )
        
        # Final MLP to reconstruct data
        self.final_projection = MLP(
            input_dim=latent_dim,
            hidden_dims=[],
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            final_activation=False
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        if self.res_blocks:
            h = self.res_blocks(h)
        return self.final_projection(h)


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for VQ-VAE.
    Adapted from sonnet.src.nets.vqvae.py for 1D data.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = (
            torch.sum(latents ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(latents, self.embedding.weight.t())
        )

        encoding_inds = torch.argmin(dist, dim=1)
        quantized_latents = self.embedding(encoding_inds)

        embedding_loss = F.mse_loss(quantized_latents, latents.detach(), reduction='none').sum(dim=1)
        commitment_loss = F.mse_loss(latents, quantized_latents.detach(), reduction='none').sum(dim=1)
        vq_loss = embedding_loss + self.beta * commitment_loss

        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents, vq_loss


class JointVAE(nn.Module):
    """
    Joint Vector-Quantized Variational Autoencoder for cross-platform data imputation.
    
    Architecture:
    - Two platform-specific encoders (Encoder_A, Encoder_B)
    - Shared Vector-Quantized latent space
    - Two platform-specific decoders (Decoder_A, Decoder_B)
    """
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        latent_dim: int = 32,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        num_embeddings: int = 512,
        vq_beta: float = 0.25,
        num_res_blocks: int = 0
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.latent_dim = latent_dim
        
        # Platform A components
        self.encoder_a = Encoder(
            input_dim=input_dim_a,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            num_res_blocks=num_res_blocks
        )
        
        self.decoder_a = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim_a,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            num_res_blocks=num_res_blocks
        )
        
        # Platform B components
        self.encoder_b = Encoder(
            input_dim=input_dim_b,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            num_res_blocks=num_res_blocks
        )
        
        self.decoder_b = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim_b,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            num_res_blocks=num_res_blocks
        )

        # Shared Vector Quantizer
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            beta=vq_beta
        )
    
    def encode_a(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode platform A data and quantize."""
        z_e_a = self.encoder_a(x_a)
        z_q_a, vq_loss_a = self.vq_layer(z_e_a)
        return z_q_a, vq_loss_a
    
    def encode_b(self, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode platform B data and quantize."""
        z_e_b = self.encoder_b(x_b)
        z_q_b, vq_loss_b = self.vq_layer(z_e_b)
        return z_q_b, vq_loss_b
    
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
        Forward pass through the joint VQ-VAE.
        
        Returns:
            Dictionary containing all outputs needed for loss computation
        """
        # Encode both platforms
        z_a, vq_loss_a = self.encode_a(x_a)
        z_b, vq_loss_b = self.encode_b(x_b)
        
        # Reconstruct same platform (auto-reconstruction)
        recon_a = self.decode_a(z_a)
        recon_b = self.decode_b(z_b)
        
        # Cross-platform reconstruction
        cross_recon_a = self.decode_a(z_b)  # B -> A
        cross_recon_b = self.decode_b(z_a)  # A -> B
        
        return {
            'z_a': z_a,
            'z_b': z_b,
            'vq_loss_a': vq_loss_a,
            'vq_loss_b': vq_loss_b,
            'recon_a': recon_a,
            'recon_b': recon_b,
            'cross_recon_a': cross_recon_a,
            'cross_recon_b': cross_recon_b
        }
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        z_a, _ = self.encode_a(x_a)
        return self.decode_b(z_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        z_b, _ = self.encode_b(x_b)
        return self.decode_a(z_b)


class JointVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Joint VQ-VAE."""
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        config: Dict
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        model_config = config['model'].copy()
        model_config.pop('model_type', None) # Remove model_type as it's not a model parameter
        
        self.model = JointVAE(
            input_dim_a=input_dim_a,
            input_dim_b=input_dim_b,
            **model_config
        )
        
        # Loss weights
        self.loss_weights = config.get('loss_weights', {})
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
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
        
        # VQ loss - model returns per-sample loss, so we average here
        vq_loss = (outputs['vq_loss_a'].mean() + outputs['vq_loss_b'].mean()) / 2
        
        # Latent alignment loss (force quantized latent representations to be similar)
        align_loss = F.mse_loss(outputs['z_a'], outputs['z_b'])
        
        # Cross-reconstruction losses
        cross_loss_a = F.mse_loss(outputs['cross_recon_a'], x_a)
        cross_loss_b = F.mse_loss(outputs['cross_recon_b'], x_b)
        cross_loss = (cross_loss_a + cross_loss_b) / 2
        
        # Weighted total loss
        total_loss = (
            self.loss_weights.get('reconstruction', 1.0) * recon_loss +
            self.loss_weights.get('vq_loss', 1.0) * vq_loss +
            self.loss_weights.get('latent_alignment', 1.0) * align_loss +
            self.loss_weights.get('cross_reconstruction', 1.0) * cross_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'align_loss': align_loss,
            'cross_loss': cross_loss
        }
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_a, x_b = batch
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
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        training_config = self.config.get('training', {})
        
        # Optimizer
        optimizer_name = training_config.get('optimizer', 'adamw').lower()
        learning_rate = training_config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=learning_rate
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config.get('optimizer')}")
        
        # Scheduler
        scheduler_name = training_config.get('scheduler', 'none').lower()
        if scheduler_name == 'reduce_on_plateau':
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