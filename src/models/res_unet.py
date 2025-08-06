"""
Residual U-Net for direct cross-platform metabolite data imputation.

This module implements a ResNet-UNet architecture for direct mapping
between platform A and platform B data without latent space representation.
Enhanced with self-attention mechanisms similar to TabTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
import numpy as np
import math

# Add imports for metrics
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_imputation_metrics


class SAINTLikeAttention(nn.Module):
    """SAINT-like attention for feature interactions with position encoding."""
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Each feature becomes a token with value + position encoding
        self.value_embedding = nn.Linear(1, embed_dim)  # Embed the feature value
        self.position_embedding = nn.Embedding(input_dim, embed_dim)  # Position encoding for each feature
        
        # Transformer layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection back to feature values
        self.output_proj = nn.Linear(embed_dim, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        batch_size, num_features = x.shape
        
        # Convert each feature to a token: [batch_size, num_features, 1]
        feature_values = x.unsqueeze(-1)
        
        # Embed feature values: [batch_size, num_features, embed_dim]
        value_embeddings = self.value_embedding(feature_values)
        
        # Add position embeddings for each feature
        positions = torch.arange(num_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Combine value and position embeddings
        tokens = value_embeddings + position_embeddings  # [batch_size, num_features, embed_dim]
        
        # Self-attention across features
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        
        # Project back to feature values: [batch_size, num_features, 1] -> [batch_size, num_features]
        output = self.output_proj(tokens).squeeze(-1)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with skip connections and optional self-attention."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_embed_dim: int = None
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Projection layer for skip connection if dimensions don't match
        self.input_projection = None
        if input_dim != output_dim:
            self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Self-attention layer (optional)
        if use_attention:
            self.attention = SAINTLikeAttention(
                input_dim=input_dim,
                embed_dim=attention_embed_dim or min(128, input_dim//2),
                num_heads=attention_heads,
                dropout=dropout_rate
            )
        
        # Two-layer residual block
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim) if batch_norm else nn.Identity()
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply self-attention if enabled (to both paths for proper residual connection)
        if self.use_attention:
            x = self.attention(x)
        
        identity = x  # Save identity AFTER attention processing
        
        # First layer
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second layer
        out = self.layer2(out)
        out = self.norm2(out)
        
        # Skip connection (project identity if needed)
        if self.input_projection is not None:
            identity = self.input_projection(identity)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class ResUNetEncoder(nn.Module):
    """ResNet-based encoder with skip connections and optional self-attention."""
    
    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_embed_dim: int = None
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for i, layer_dim in enumerate(layer_dims):
            # Add attention to middle layers
            
            #use_attn = use_attention and (i > 0)
            use_attn = use_attention
            
            block = ResidualBlock(
                input_dim=current_dim,
                hidden_dim=layer_dim,
                output_dim=layer_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                use_attention=use_attn,
                attention_heads=attention_heads,
                attention_embed_dim=attention_embed_dim
            )
            self.layers.append(block)
            current_dim = layer_dim
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip_connections = []
        
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
        
        return skip_connections


class ResUNetDecoder(nn.Module):
    """ResNet-based decoder with skip connections from encoder and optional self-attention."""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_embed_dim: int = None
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Decoder processes encoder features in reverse order
        # For encoder_dims=[512, 256, 128], decoder goes [128, 256, 512]
        # Skip connections will be passed as [256, 512] (encoder[:-1] reversed)
        decoder_dims = encoder_dims[::-1]  # [128, 256, 512]
        num_skip_layers = len(encoder_dims) - 1  # Number of skip connections available
        
        current_dim = input_dim  # Start with bottleneck dimension
        
        for i, decoder_dim in enumerate(decoder_dims):
            # Calculate input dimension (current + skip connection if available)
            if i < num_skip_layers:
                # We have a skip connection for this layer
                skip_dim = encoder_dims[-(i+2)]  # Get corresponding encoder layer dimension
                actual_input_dim = current_dim + skip_dim
            else:
                # No skip connection for this layer
                actual_input_dim = current_dim
            
            # Add attention to middle layers
            
            #use_attn = use_attention and (i > 0)
            use_attn = use_attention
            
            block = ResidualBlock(
                input_dim=actual_input_dim,
                hidden_dim=decoder_dim,
                output_dim=decoder_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                use_attention=use_attn,
                attention_heads=attention_heads,
                attention_embed_dim=attention_embed_dim
            )
            self.layers.append(block)
            current_dim = decoder_dim
        
        # Final output layer
        self.output_layer = nn.Linear(current_dim, output_dim)
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder with skip connections from encoder.
        
        Args:
            x: Bottleneck output tensor
            skip_connections: List of skip connections from encoder (already in reverse order, excluding last layer)
        """
        current = x
        
        for i, layer in enumerate(self.layers):
            # Concatenate with corresponding skip connection
            if i < len(skip_connections):
                current = torch.cat([current, skip_connections[i]], dim=1)
            
            current = layer(current)
        
        # Final output layer
        output = self.output_layer(current)
        
        return output


class ResUNet(nn.Module):
    """
    Residual U-Net for direct cross-platform imputation with self-attention.
    
    Architecture:
    - ResNet-based encoder with skip connections and self-attention
    - Bottleneck layer with self-attention
    - ResNet-based decoder with skip connections from encoder and self-attention
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_layers: List[int] = [512, 256, 128],
        decoder_layers: Optional[List[int]] = None,
        bottleneck_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_embed_dim: int = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if decoder_layers is None:
            decoder_layers = encoder_layers.copy()
        
        # Encoder
        self.encoder = ResUNetEncoder(
            input_dim=input_dim,
            layer_dims=encoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim
        )
        
        # Bottleneck with attention
        self.bottleneck = ResidualBlock(
            input_dim=encoder_layers[-1],
            hidden_dim=bottleneck_dim,
            output_dim=bottleneck_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim
        )
        
        # Decoder
        self.decoder = ResUNetDecoder(
            input_dim=bottleneck_dim,
            encoder_dims=encoder_layers,  # Pass encoder dims for skip connection compatibility
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode with skip connections
        skip_connections = self.encoder(x)
        
        # Bottleneck uses the output of the last encoder layer
        bottleneck_input = skip_connections[-1]
        bottleneck = self.bottleneck(bottleneck_input)
        
        # Decode with skip connections (pass all but the last one which was used by bottleneck)
        # Reverse the skip connections to match decoder expectation
        decoder_skips = skip_connections[:-1][::-1]  # Reverse [512, 256] to [256, 512]
        output = self.decoder(bottleneck, decoder_skips)
        
        return output


class DirectImputationModel(nn.Module):
    """
    Direct imputation model with two ResUNet networks for bidirectional mapping.
    Enhanced with self-attention mechanisms similar to TabTransformer.
    """
    
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        encoder_layers: List[int] = [512, 256, 128],
        decoder_layers: Optional[List[int]] = None,
        bottleneck_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_embed_dim: int = None
    ):
        super().__init__()
        
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        
        # A -> B imputation network
        self.net_a_to_b = ResUNet(
            input_dim=input_dim_a,
            output_dim=input_dim_b,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim
        )
        
        # B -> A imputation network
        self.net_b_to_a = ResUNet(
            input_dim=input_dim_b,
            output_dim=input_dim_a,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim
        )
    
    def impute_a_to_b(self, x_a: torch.Tensor) -> torch.Tensor:
        """Impute platform B data from platform A data."""
        return self.net_a_to_b(x_a)
    
    def impute_b_to_a(self, x_b: torch.Tensor) -> torch.Tensor:
        """Impute platform A data from platform B data."""
        return self.net_b_to_a(x_b)
    
    def forward(
        self, 
        x_a: torch.Tensor, 
        x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both imputation networks.
        
        Returns:
            Dictionary containing imputation outputs
        """
        # Direct imputation
        imputed_b = self.impute_a_to_b(x_a)  # A -> B
        imputed_a = self.impute_b_to_a(x_b)  # B -> A
        
        # Cycle consistency (optional for additional loss)
        cycle_a = self.impute_b_to_a(imputed_b)  # A -> B -> A
        cycle_b = self.impute_a_to_b(imputed_a)  # B -> A -> B
        
        return {
            'imputed_a': imputed_a,
            'imputed_b': imputed_b,
            'cycle_a': cycle_a,
            'cycle_b': cycle_b
        }


class DirectImputationLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for Direct Imputation ResUNet with Self-Attention."""
    
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
        
        self.model = DirectImputationModel(
            input_dim_a=input_dim_a,
            input_dim_b=input_dim_b,
            encoder_layers=model_config['encoder_layers'],
            decoder_layers=model_config.get('decoder_layers', None),
            bottleneck_dim=model_config.get('bottleneck_dim', 64),
            activation=model_config['activation'],
            dropout_rate=model_config['dropout_rate'],
            batch_norm=model_config['batch_norm'],
            use_attention=model_config.get('use_attention', False),
            attention_heads=model_config.get('attention_heads', 8),
            attention_embed_dim=model_config.get('attention_embed_dim', None)
        )
        
        # Loss weights
        self.loss_weights = config['loss_weights']
        
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
        
        # Direct imputation losses
        impute_loss_a = F.mse_loss(outputs['imputed_a'], x_a)  # B -> A loss
        impute_loss_b = F.mse_loss(outputs['imputed_b'], x_b)  # A -> B loss
        imputation_loss = (impute_loss_a + impute_loss_b) / 2
        
        # Cycle consistency losses (optional)
        cycle_loss_a = F.mse_loss(outputs['cycle_a'], x_a)  # A -> B -> A
        cycle_loss_b = F.mse_loss(outputs['cycle_b'], x_b)  # B -> A -> B
        cycle_loss = (cycle_loss_a + cycle_loss_b) / 2
        
        # Weighted total loss
        total_loss = (
            self.loss_weights.get('imputation', 1.0) * imputation_loss +
            self.loss_weights.get('cycle_consistency', 0.1) * cycle_loss
        )
        
        return {
            'total_loss': total_loss,
            'imputation_loss': imputation_loss,
            'cycle_loss': cycle_loss,
            'impute_loss_a': impute_loss_a,
            'impute_loss_b': impute_loss_b
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
        imputed_a_np = all_outputs['imputed_a'].detach().cpu().numpy()
        imputed_b_np = all_outputs['imputed_b'].detach().cpu().numpy()
        
        # Compute R² metrics for imputation tasks
        impute_a_metrics = compute_imputation_metrics(x_a_np, imputed_a_np)
        impute_b_metrics = compute_imputation_metrics(x_b_np, imputed_b_np)
        
        # Log key R² metrics
        self.log('val_impute_a_r2', impute_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('val_impute_b_r2', impute_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log mean feature-wise R² scores
        self.log('val_impute_a_mean_feature_r2', impute_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('val_impute_b_mean_feature_r2', impute_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics
        self.log('val_impute_a_frac_r2_above_0.5', impute_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('val_impute_b_frac_r2_above_0.5', impute_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
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
        imputed_a_np = all_outputs['imputed_a'].detach().cpu().numpy()
        imputed_b_np = all_outputs['imputed_b'].detach().cpu().numpy()
        
        # Compute R² metrics for imputation tasks
        impute_a_metrics = compute_imputation_metrics(x_a_np, imputed_a_np)
        impute_b_metrics = compute_imputation_metrics(x_b_np, imputed_b_np)
        
        # Log key R² metrics
        self.log('test_impute_a_r2', impute_a_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        self.log('test_impute_b_r2', impute_b_metrics['overall_r2'], on_epoch=True, prog_bar=True)
        
        # Log mean feature-wise R² scores
        self.log('test_impute_a_mean_feature_r2', impute_a_metrics['mean_feature_r2'], on_epoch=True)
        self.log('test_impute_b_mean_feature_r2', impute_b_metrics['mean_feature_r2'], on_epoch=True)
        
        # Log quality metrics
        self.log('test_impute_a_frac_r2_above_0.5', impute_a_metrics['fraction_r2_above_0.5'], on_epoch=True)
        self.log('test_impute_b_frac_r2_above_0.5', impute_b_metrics['fraction_r2_above_0.5'], on_epoch=True)
        
        # Log additional detailed metrics for final test evaluation
        self.log('test_impute_a_median_feature_r2', impute_a_metrics['median_feature_r2'], on_epoch=True)
        self.log('test_impute_b_median_feature_r2', impute_b_metrics['median_feature_r2'], on_epoch=True)
        
        self.log('test_impute_a_frac_r2_above_0.7', impute_a_metrics['fraction_r2_above_0.7'], on_epoch=True)
        self.log('test_impute_b_frac_r2_above_0.7', impute_b_metrics['fraction_r2_above_0.7'], on_epoch=True)
        
        # Print summary for final test results
        print("\n" + "="*60)
        print("FINAL TEST R² METRICS SUMMARY - DIRECT IMPUTATION")
        print("="*60)
        print(f"Platform A Imputation R² (B->A): {impute_a_metrics['overall_r2']:.4f}")
        print(f"Platform B Imputation R² (A->B): {impute_b_metrics['overall_r2']:.4f}")
        print(f"Features with R² > 0.5 (B->A): {impute_a_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.5 (A->B): {impute_b_metrics['fraction_r2_above_0.5']:.2%}")
        print(f"Features with R² > 0.7 (B->A): {impute_a_metrics['fraction_r2_above_0.7']:.2%}")
        print(f"Features with R² > 0.7 (A->B): {impute_b_metrics['fraction_r2_above_0.7']:.2%}")
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