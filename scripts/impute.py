#!/usr/bin/env python3
"""
Imputation script for Joint VAE models.

This script loads a trained model from an experiment directory and performs 
cross-platform imputation with proper handling of all preprocessing steps.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
from typing import Dict, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent / 'src'))
try:
    from captum.attr import IntegratedGradients, DeepLift, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Warning: Captum not available. Feature importance analysis will be disabled.")
    CAPTUM_AVAILABLE = False

from models import (
    JointVAELightning, JointVAEPlusLightning, DirectImputationLightning, 
    GenerativeVAE, JointIWAELightning, JointVAEVampPriorLightning, JointVQLightning
)
from utils import load_scalers_and_features, preprocess_input_data, inverse_transform_output


def find_best_checkpoint(exp_dir: str) -> str:
    """Find the best checkpoint file in the experiment directory."""
    checkpoint_dir = Path(exp_dir) / 'checkpoints'
    
    # Look for best checkpoint first
    best_checkpoints = list(checkpoint_dir.glob("*best*.ckpt"))
    if best_checkpoints:
        return str(best_checkpoints[0])
    
    # Look for last checkpoint
    last_checkpoints = list(checkpoint_dir.glob("last.ckpt"))
    if last_checkpoints:
        return str(last_checkpoints[0])
    
    # Fallback to latest checkpoint
    all_checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if all_checkpoints:
        return str(all_checkpoints[-1])
    
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle PyTorch Lightning hparams format where config is nested
    if 'config' in data and isinstance(data['config'], dict):
        print("Detected PyTorch Lightning hparams format, extracting config...")
        return data['config']
    
    return data


def load_model(model_path: str, config: Dict) -> torch.nn.Module:
    """Load the trained model from checkpoint."""
    model_type = config['model'].get('model_type', 'joint_vae')
    
    # Model type mapping
    model_classes = {
        'joint_vae': JointVAELightning,
        'joint_vae_plus': JointVAEPlusLightning,
        'JointVAEVampPrior': JointVAEVampPriorLightning,
        'JointIWAE': JointIWAELightning,
        'JointVQ': JointVQLightning,
        'res_unet': DirectImputationLightning,
        'generative_vae': GenerativeVAE
    }
    
    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Loading {model_type} model from checkpoint...")
    model = model_class.load_from_checkpoint(model_path)
    model.eval()
    
    return model


def perform_imputation(
    model: torch.nn.Module,
    input_data: np.ndarray,
    source_platform: str,
    target_platform: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform imputation using the trained model.
    
    Args:
        model: Trained model
        input_data: Preprocessed input data
        source_platform: Source platform ('a' or 'b')
        target_platform: Target platform ('a' or 'b')
    
    Returns:
        Tuple of (imputed data (still normalized), latent vectors)
    """
    print(f"Performing imputation from platform {source_platform.upper()} to platform {target_platform.upper()}")
    
    # Convert to tensor and move to same device as model
    input_tensor = torch.FloatTensor(input_data)
    
    # Get model device
    model_device = next(model.parameters()).device
    input_tensor = input_tensor.to(model_device)
    print(f"Using device: {model_device}")
    
    with torch.no_grad():
        latent_vectors = None
        
        # Use the model's imputation methods if available
        if hasattr(model.model, 'impute_a_to_b') and source_platform == 'a' and target_platform == 'b':
            output_tensor = model.model.impute_a_to_b(input_tensor)
            # Try to extract latent vectors
            try:
                latent_vectors = extract_latent_from_model(model, input_tensor, source_platform)
            except Exception as e:
                print(f"Warning: Could not extract latent vectors: {e}")
                
        elif hasattr(model.model, 'impute_b_to_a') and source_platform == 'b' and target_platform == 'a':
            output_tensor = model.model.impute_b_to_a(input_tensor)
            # Try to extract latent vectors
            try:
                latent_vectors = extract_latent_from_model(model, input_tensor, source_platform)
            except Exception as e:
                print(f"Warning: Could not extract latent vectors: {e}")
                
        else:
            # Fallback: use forward pass with dummy data for the other platform
            try:
                if source_platform == 'a':
                    # Create dummy platform B data with same batch size
                    if hasattr(model.model, 'input_dim_b'):
                        dummy_b = torch.zeros(input_tensor.shape[0], model.model.input_dim_b, device=model_device)
                    else:
                        # Try to infer from model structure
                        dummy_b = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], device=model_device)
                        print("Warning: Could not determine input_dim_b, using same as input")
                    outputs = model.model(input_tensor, dummy_b)
                    output_tensor = outputs['cross_recon_b']  # A -> B
                    
                    # Extract latent vectors from outputs
                    if 'latent' in outputs:
                        latent_vectors = outputs['latent'].cpu().numpy()
                    elif 'mu' in outputs:
                        latent_vectors = outputs['mu'].cpu().numpy()
                    elif 'z' in outputs:
                        latent_vectors = outputs['z'].cpu().numpy()
                else:
                    # Create dummy platform A data with same batch size
                    if hasattr(model.model, 'input_dim_a'):
                        dummy_a = torch.zeros(input_tensor.shape[0], model.model.input_dim_a, device=model_device)
                    else:
                        # Try to infer from model structure
                        dummy_a = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], device=model_device)
                        print("Warning: Could not determine input_dim_a, using same as input")
                    outputs = model.model(dummy_a, input_tensor)
                    output_tensor = outputs['cross_recon_a']  # B -> A
                    
                    # Extract latent vectors from outputs
                    if 'latent' in outputs:
                        latent_vectors = outputs['latent'].cpu().numpy()
                    elif 'mu' in outputs:
                        latent_vectors = outputs['mu'].cpu().numpy()
                    elif 'z' in outputs:
                        latent_vectors = outputs['z'].cpu().numpy()
                        
            except Exception as e:
                raise RuntimeError(f"Failed to perform imputation using model fallback: {e}. "
                                 f"The model may not support the expected imputation interface.")
    
    return output_tensor.cpu().numpy(), latent_vectors


def extract_latent_from_model(model: torch.nn.Module, input_tensor: torch.Tensor, platform: str) -> Optional[np.ndarray]:
    """
    Extract latent vectors from the model for the given input.
    
    Args:
        model: Trained model
        input_tensor: Input data tensor
        platform: Platform identifier ('a' or 'b')
        
    Returns:
        Latent vectors as numpy array or None if extraction fails
    """
    try:
        # Different VAE models might have different ways to access the encoder
        if hasattr(model.model, 'encoder'):
            # Standard VAE encoder pattern
            if hasattr(model.model.encoder, 'encode'):
                encoded = model.model.encoder.encode(input_tensor)
            else:
                encoded = model.model.encoder(input_tensor)
                
            if isinstance(encoded, tuple):
                mu, logvar = encoded
                return mu.cpu().numpy()
            else:
                return encoded.cpu().numpy()
                
        elif hasattr(model.model, f'encoder_{platform}'):
            # Platform-specific encoder
            encoder = getattr(model.model, f'encoder_{platform}')
            encoded = encoder(input_tensor)
            if isinstance(encoded, tuple):
                mu, logvar = encoded
                return mu.cpu().numpy()
            else:
                return encoded.cpu().numpy()
                
        elif hasattr(model.model, 'encode'):
            # Direct encode method
            encoded = model.model.encode(input_tensor)
            if isinstance(encoded, tuple):
                mu, logvar = encoded
                return mu.cpu().numpy()
            else:
                return encoded.cpu().numpy()
                
        else:
            # Try to extract from forward pass
            if platform == 'a':
                dummy_b = torch.zeros(input_tensor.shape[0], 
                                    getattr(model.model, 'input_dim_b', input_tensor.shape[1]), 
                                    device=input_tensor.device)
                outputs = model.model(input_tensor, dummy_b)
            else:
                dummy_a = torch.zeros(input_tensor.shape[0], 
                                    getattr(model.model, 'input_dim_a', input_tensor.shape[1]), 
                                    device=input_tensor.device)
                outputs = model.model(dummy_a, input_tensor)
            
            # Look for latent representation in outputs
            for key in ['latent', 'mu', 'z', f'latent_{platform}', f'mu_{platform}']:
                if key in outputs:
                    return outputs[key].cpu().numpy()
            
            print("Warning: Could not find latent representation in model outputs")
            return None
            
    except Exception as e:
        print(f"Warning: Failed to extract latent vectors: {e}")
        return None


class ImputationWrapper(torch.nn.Module):
    """Wrapper for Joint VAE models to enable Captum analysis."""
    
    def __init__(self, model: torch.nn.Module, source_platform: str, target_platform: str, target_index: int = None):
        super().__init__()
        self.model = model.model  # Extract the actual model from Lightning wrapper
        self.source_platform = source_platform
        self.target_platform = target_platform
        self.target_index = target_index  # For feature-specific attribution
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for imputation from source to target platform."""
        try:
            # Use the model's imputation methods if available
            if (hasattr(self.model, 'impute_a_to_b') and 
                self.source_platform == 'a' and self.target_platform == 'b'):
                result = self.model.impute_a_to_b(x)
            elif (hasattr(self.model, 'impute_b_to_a') and 
                  self.source_platform == 'b' and self.target_platform == 'a'):
                result = self.model.impute_b_to_a(x)
            else:
                # Fallback: use forward pass with dummy data
                if self.source_platform == 'a':
                    dummy_b = torch.zeros(x.shape[0], 
                                        getattr(self.model, 'input_dim_b', x.shape[1]), 
                                        device=x.device)
                    outputs = self.model(x, dummy_b)
                    result = outputs.get('cross_recon_b', outputs.get('recon_b', outputs.get('x_recon_b')))
                    if result is None:
                        # If no cross reconstruction, try using the latent space
                        z = outputs.get('z', outputs.get('latent', outputs.get('mean_a')))
                        if z is not None and hasattr(self.model, 'decode_b'):
                            result = self.model.decode_b(z)
                        else:
                            raise ValueError("Cannot find appropriate output for cross-platform imputation")
                else:
                    dummy_a = torch.zeros(x.shape[0], 
                                        getattr(self.model, 'input_dim_a', x.shape[1]), 
                                        device=x.device)
                    outputs = self.model(dummy_a, x)
                    result = outputs.get('cross_recon_a', outputs.get('recon_a', outputs.get('x_recon_a')))
                    if result is None:
                        # If no cross reconstruction, try using the latent space
                        z = outputs.get('z', outputs.get('latent', outputs.get('mean_b')))
                        if z is not None and hasattr(self.model, 'decode_a'):
                            result = self.model.decode_a(z)
                        else:
                            raise ValueError("Cannot find appropriate output for cross-platform imputation")
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Ultimate fallback: return a simple linear transformation
            if self.source_platform == 'a':
                output_dim = getattr(self.model, 'input_dim_b', x.shape[1])
            else:
                output_dim = getattr(self.model, 'input_dim_a', x.shape[1])
            
            # Simple linear projection as fallback
            result = torch.zeros(x.shape[0], output_dim, device=x.device)
            min_dim = min(x.shape[1], output_dim)
            result[:, :min_dim] = x[:, :min_dim]
        
        # If target_index is specified, return only that feature
        if self.target_index is not None:
            if self.target_index < result.shape[1]:
                return result[:, self.target_index]
            else:
                return torch.zeros(result.shape[0], device=result.device)
        else:
            # Return the mean of all outputs to get a scalar for each sample
            # This allows Captum to compute gradients properly
            return result.mean(dim=1)


def compute_feature_importance(
    model: torch.nn.Module,
    input_data: np.ndarray,
    source_platform: str,
    target_platform: str,
    feature_names: list,
    method: str = 'integrated_gradients',
    baseline: str = 'zero',
    n_steps: int = 50,
    per_output_feature: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature importance using Captum.
    
    Args:
        model: Trained model
        input_data: Preprocessed input data
        source_platform: Source platform ('a' or 'b')
        target_platform: Target platform ('a' or 'b')
        feature_names: List of feature names
        method: Attribution method ('integrated_gradients', 'deeplift', 'gradient_shap')
        baseline: Baseline for attribution ('zero', 'mean')
        n_steps: Number of steps for integrated gradients
        per_output_feature: If True, compute importance for each output feature separately
        
    Returns:
        Tuple of (sample-wise attributions, global importance scores)
        If per_output_feature=True:
            attributions shape: (n_samples, n_input_features) for global importance
            OR (n_output_features, n_samples, n_input_features) for per-output
        If per_output_feature=False:
            attributions shape: (n_samples, n_input_features)
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("Captum is required for feature importance analysis")
    
    print(f"Computing feature importance using {method} with {baseline} baseline...")
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(input_data)
    model_device = next(model.parameters()).device
    input_tensor = input_tensor.to(model_device)
    input_tensor.requires_grad_(True)
    
    # Create baseline
    if baseline == 'zero':
        baseline_tensor = torch.zeros_like(input_tensor)
    elif baseline == 'mean':
        baseline_tensor = torch.mean(input_tensor, dim=0, keepdim=True).expand_as(input_tensor)
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")
    
    if per_output_feature:
        # Compute importance for each output feature separately
        print("Computing per-output-feature importance...")
        
        # First, get the output dimensions by doing a forward pass
        wrapper_test = ImputationWrapper(model, source_platform, target_platform, target_index=None)
        wrapper_test.eval()
        wrapper_test = wrapper_test.to(model_device)
        
        with torch.no_grad():
            # Get a small batch to determine output dimensions
            test_input = input_tensor[:1]
            test_output = wrapper_test.model.impute_a_to_b(test_input) if source_platform == 'a' else wrapper_test.model.impute_b_to_a(test_input)
            n_output_features = test_output.shape[1] if len(test_output.shape) > 1 else 1
        
        print(f"Computing importance for {n_output_features} output features...")
        
        all_attributions = []
        all_importance = []
        
        # Compute importance for each output feature
        for output_idx in range(n_output_features):
            if output_idx % 100 == 0:
                print(f"  Processing output feature {output_idx+1}/{n_output_features}")
            
            # Create wrapper for this specific output feature
            wrapper = ImputationWrapper(model, source_platform, target_platform, target_index=output_idx)
            wrapper.eval()
            wrapper = wrapper.to(model_device)
            
            try:
                # Initialize attribution method for this output feature
                if method == 'integrated_gradients':
                    attr_method = IntegratedGradients(wrapper)
                    attributions = attr_method.attribute(
                        input_tensor, 
                        baselines=baseline_tensor, 
                        n_steps=n_steps
                    )
                elif method == 'deeplift':
                    attr_method = DeepLift(wrapper)
                    attributions = attr_method.attribute(
                        input_tensor, 
                        baselines=baseline_tensor
                    )
                elif method == 'gradient_shap':
                    attr_method = GradientShap(wrapper)
                    n_baseline = min(50, input_tensor.shape[0])
                    baseline_dist = input_tensor[:n_baseline]
                    attributions = attr_method.attribute(
                        input_tensor,
                        baselines=baseline_dist,
                        n_samples=10
                    )
                else:
                    raise ValueError(f"Unsupported attribution method: {method}")
                
                # Convert to numpy
                attributions_np = attributions.detach().cpu().numpy()
                all_attributions.append(attributions_np)
                
                # Compute importance for this output feature
                feature_importance = np.median(np.abs(attributions_np), axis=0)
                all_importance.append(feature_importance)
                
            except Exception as e:
                print(f"Warning: Failed to compute importance for output feature {output_idx}: {e}")
                # Add zeros as fallback
                zero_attributions = np.zeros((input_tensor.shape[0], input_tensor.shape[1]))
                all_attributions.append(zero_attributions)
                all_importance.append(np.zeros(input_tensor.shape[1]))
        
        # Stack all attributions: shape (n_output_features, n_samples, n_input_features)
        attributions_per_output = np.stack(all_attributions, axis=0)
        importance_per_output = np.stack(all_importance, axis=0)
        
        # Compute global importance across all output features
        global_importance = np.median(importance_per_output, axis=0)
        
        print(f"Per-output feature importance completed. Shape: {attributions_per_output.shape}")
        
        return attributions_per_output, global_importance
        
    else:
        # Compute global importance (mean across all output features)
        wrapper = ImputationWrapper(model, source_platform, target_platform, target_index=None)
        wrapper.eval()
        wrapper = wrapper.to(model_device)
        
        # Initialize attribution method
        try:
            if method == 'integrated_gradients':
                attr_method = IntegratedGradients(wrapper)
                attributions = attr_method.attribute(
                    input_tensor, 
                    baselines=baseline_tensor, 
                    n_steps=n_steps
                )
            elif method == 'deeplift':
                attr_method = DeepLift(wrapper)
                attributions = attr_method.attribute(
                    input_tensor, 
                    baselines=baseline_tensor
                )
            elif method == 'gradient_shap':
                attr_method = GradientShap(wrapper)
                # Use subset of input as baseline distribution for GradientShap
                n_baseline = min(50, input_tensor.shape[0])
                baseline_dist = input_tensor[:n_baseline]
                attributions = attr_method.attribute(
                    input_tensor,
                    baselines=baseline_dist,
                    n_samples=10
                )
            else:
                raise ValueError(f"Unsupported attribution method: {method}")
            
            # Convert to numpy
            attributions_np = attributions.detach().cpu().numpy()
            
            # Compute global importance (median absolute attribution)
            global_importance = np.median(np.abs(attributions_np), axis=0)
            
            print(f"Feature importance computation completed. Shape: {attributions_np.shape}")
            
            return attributions_np, global_importance
            
        except Exception as e:
            # If the above fails, try a simpler approach with gradient computation
            print(f"Standard Captum attribution failed ({e}), trying manual gradient computation...")
            
            # Simple gradient-based importance
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = wrapper(input_tensor_grad)
            
            # Compute gradients for each sample
            attributions_list = []
            for i in range(output.shape[0]):
                if input_tensor_grad.grad is not None:
                    input_tensor_grad.grad.zero_()
                
                # Backward pass for this sample
                output[i].backward(retain_graph=True)
                
                # Get gradients
                if input_tensor_grad.grad is not None:
                    grad = input_tensor_grad.grad[i].detach().cpu().numpy()
                    attributions_list.append(grad)
                else:
                    # Fallback to zeros if gradient computation fails
                    attributions_list.append(np.zeros(input_tensor_grad.shape[1]))
            
            # Stack attributions
            attributions_np = np.array(attributions_list)
            
            # Compute global importance (median absolute attribution)
            global_importance = np.median(np.abs(attributions_np), axis=0)
            
            print(f"Manual gradient computation completed. Shape: {attributions_np.shape}")
            
            return attributions_np, global_importance


def save_feature_importance(
    attributions: np.ndarray,
    global_importance: np.ndarray,
    feature_names: list,
    sample_ids: list,
    output_dir: Path,
    source_platform: str,
    target_platform: str,
    method: str,
    baseline: str,
    per_output_feature: bool = False,
    target_feature_names: list = None,
    save_detailed: bool = False
):
    """Save feature importance results to CSV files."""
    
    print("Saving feature importance results...")
    
    # Create importance subdirectory
    importance_dir = output_dir / "importance"
    importance_dir.mkdir(exist_ok=True)
    
    direction = f"{source_platform}to{target_platform}"
    
    if per_output_feature:
        # attributions shape: (n_output_features, n_samples, n_input_features)
        n_output_features, n_samples, n_input_features = attributions.shape
        
        # Create target feature names if not provided
        if target_feature_names is None:
            target_feature_names = [f"output_feature_{i:04d}" for i in range(n_output_features)]
        
        print(f"  Saving per-output-feature importance for {n_output_features} output features...")
        
        # Save aggregated sample-wise attributions (mean across output features)
        mean_attributions = np.mean(attributions, axis=0)  # Shape: (n_samples, n_input_features)
        samplewise_df = pd.DataFrame(
            mean_attributions,
            index=sample_ids,
            columns=feature_names
        )
        samplewise_path = importance_dir / f"importance_samplewise_{direction}_aggregated.csv"
        samplewise_df.to_csv(samplewise_path)
        print(f"  Aggregated sample-wise attributions saved to: {samplewise_path}")
        
        # SKIP THE MASSIVE PER-SAMPLE-PER-OUTPUT FILE - it's 25GB+ and not needed
        # Only save if explicitly requested via --save_detailed_attributions flag
        
        if save_detailed:
            # Save per-output-feature importance matrix
            # Reshape to 2D: (n_output_features * n_samples, n_input_features)
            reshaped_attributions = attributions.reshape(-1, n_input_features)
            
            # Create multi-index for rows (output_feature, sample)
            output_indices = []
            sample_indices = []
            for out_idx in range(n_output_features):
                for sample_id in sample_ids:
                    output_indices.append(target_feature_names[out_idx])
                    sample_indices.append(sample_id)
            
            multi_index = pd.MultiIndex.from_arrays([output_indices, sample_indices], 
                                                   names=['output_feature', 'sample'])
            
            per_output_df = pd.DataFrame(
                reshaped_attributions,
                index=multi_index,
                columns=feature_names
            )
            per_output_path = importance_dir / f"importance_per_output_{direction}.csv"
            per_output_df.to_csv(per_output_path)
            print(f"  Per-output-feature attributions saved to: {per_output_path}")
        else:
            print(f"  Skipping detailed per-sample-per-output file (would be {n_output_features * n_samples} rows)")
            per_output_path = None
        
        # Save per-output-feature global importance
        per_output_importance = np.median(np.abs(attributions), axis=1)  # Shape: (n_output_features, n_input_features)
        per_output_importance_df = pd.DataFrame(
            per_output_importance,
            index=target_feature_names,
            columns=feature_names
        )
        per_output_global_path = importance_dir / f"importance_per_output_global_{direction}.csv"
        per_output_importance_df.to_csv(per_output_global_path)
        print(f"  Per-output-feature global importance saved to: {per_output_global_path}")
        
        # Save input-feature × output-feature importance matrix (transposed for easier reading)
        # Transpose so rows=input_features, columns=output_features
        importance_matrix = per_output_importance.T  # Shape: (n_input_features, n_output_features)
        importance_matrix_df = pd.DataFrame(
            importance_matrix,
            index=feature_names,  # Input features as rows
            columns=target_feature_names  # Output features as columns
        )
        importance_matrix_path = importance_dir / f"importance_matrix_{direction}.csv"
        importance_matrix_df.to_csv(importance_matrix_path)
        print(f"  Input×Output importance matrix saved to: {importance_matrix_path}")
        
        # Also save top important input features for each output feature
        top_inputs_per_output = {}
        for out_idx, out_feature in enumerate(target_feature_names):
            # Get top 10 input features for this output feature
            importance_scores = per_output_importance[out_idx]
            top_indices = np.argsort(importance_scores)[-10:][::-1]
            top_inputs_per_output[out_feature] = [
                {'input_feature': feature_names[idx], 'importance': importance_scores[idx]}
                for idx in top_indices
            ]
        
        top_inputs_path = importance_dir / f"top_inputs_per_output_{direction}.yaml"
        with open(top_inputs_path, 'w') as f:
            yaml.dump(top_inputs_per_output, f, default_flow_style=False)
        print(f"  Top inputs per output feature saved to: {top_inputs_path}")
        
        # Save standard global importance (aggregated)
        global_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        global_path = importance_dir / f"importance_global_{direction}.csv"
        global_df.to_csv(global_path, index=False)
        print(f"  Aggregated global importance saved to: {global_path}")
        
        # Enhanced metadata
        metadata = {
            'method': method,
            'baseline': baseline,
            'source_platform': source_platform,
            'target_platform': target_platform,
            'per_output_feature': True,
            'n_samples': len(sample_ids),
            'n_input_features': len(feature_names),
            'n_output_features': n_output_features,
            'top_features_aggregated': global_df.head(10)['feature'].tolist(),
            'attribution_shape': list(attributions.shape)
        }
        
        paths = {
            'samplewise_path': samplewise_path,
            'global_path': global_path,
            'per_output_global_path': per_output_global_path,
            'importance_matrix_path': importance_matrix_path,
            'top_inputs_path': top_inputs_path,
            'metadata_path': importance_dir / f"importance_metadata_{direction}.yaml"
        }
        
        # Add per_output_path only if it was created
        if per_output_path is not None:
            paths['per_output_path'] = per_output_path
        
    else:
        # Standard case: attributions shape (n_samples, n_input_features)
        # Save sample-wise attributions
        samplewise_df = pd.DataFrame(
            attributions,
            index=sample_ids,
            columns=feature_names
        )
        samplewise_path = importance_dir / f"importance_samplewise_{direction}.csv"
        samplewise_df.to_csv(samplewise_path)
        print(f"  Sample-wise attributions saved to: {samplewise_path}")
        
        # Save global importance
        global_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        global_path = importance_dir / f"importance_global_{direction}.csv"
        global_df.to_csv(global_path, index=False)
        print(f"  Global importance saved to: {global_path}")
        
        # Standard metadata
        metadata = {
            'method': method,
            'baseline': baseline,
            'source_platform': source_platform,
            'target_platform': target_platform,
            'per_output_feature': False,
            'n_samples': len(sample_ids),
            'n_features': len(feature_names),
            'top_features': global_df.head(10)['feature'].tolist()
        }
        
        paths = {
            'samplewise_path': samplewise_path,
            'global_path': global_path,
            'per_output_global_path': per_output_global_path,
            'importance_matrix_path': importance_matrix_path,
            'top_inputs_path': top_inputs_path,
            'metadata_path': importance_dir / f"importance_metadata_{direction}.yaml"
        }
    
    # Save metadata
    with open(paths['metadata_path'], 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"  Metadata saved to: {paths['metadata_path']}")
    
    return paths


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Impute data using trained Joint VAE model')
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Path to experiment directory containing config, checkpoint, and preprocessing artifacts'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        required=True,
        help='Path to input data file to be imputed (CSV or TXT)'
    )
    parser.add_argument(
        '--source_platform',
        type=str,
        choices=['a', 'b'],
        required=True,
        help='Source platform of input data (a or b)'
    )
    parser.add_argument(
        '--target_platform',
        type=str,
        choices=['a', 'b'],
        required=True,
        help='Target platform for imputation (a or b)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path for output imputed CSV file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Specific checkpoint file to use (if not provided, will find best checkpoint)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Specific config file to use (if not provided, will search for config files in experiment_dir)'
    )
    parser.add_argument(
        '--output_latent',
        type=str,
        default=None,
        help='Path to save latent vectors (optional)'
    )
    parser.add_argument(
        '--id_column',
        type=str,
        default=None,
        help='Name of ID column in input data (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output_importance',
        type=str,
        default=None,
        help='Path to save importance matrix CSV file (input features × output features). If provided, importance will be computed.'
    )
    parser.add_argument(
        '--importance_method',
        type=str,
        choices=['integrated_gradients', 'deeplift', 'gradient_shap'],
        default='integrated_gradients',
        help='Attribution method for feature importance'
    )
    parser.add_argument(
        '--importance_baseline',
        type=str,
        choices=['zero', 'mean'],
        default='zero',
        help='Baseline for feature importance attribution'
    )
    parser.add_argument(
        '--importance_steps',
        type=int,
        default=50,
        help='Number of steps for integrated gradients (only used with integrated_gradients method)'
    )
    
    return parser.parse_args()


def main():
    """
    Main imputation function for cross-platform data transformation.
    
    Loads trained Joint VAE model, preprocesses input data, performs imputation,
    and optionally computes feature importance using Captum.
    """
    args = parse_arguments()
    
    # Validate arguments
    if args.source_platform == args.target_platform:
        raise ValueError("Source and target platforms must be different")
    
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Find and load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        # Try to find config file in experiment directory
        # Prefer original config files over hparams.yaml
        config_candidates = (
            list(exp_dir.glob("config.yaml")) + 
            list(exp_dir.glob("config.yml")) +
            list(exp_dir.glob("*_config.yaml")) +
            list(exp_dir.glob("*_config.yml")) +
            list(exp_dir.glob("hparams.yaml")) +
            list(exp_dir.glob("*.yaml")) + 
            list(exp_dir.glob("*.yml"))
        )
        
        if not config_candidates:
            raise FileNotFoundError(f"No config file found in {exp_dir}")
        
        config_path = config_candidates[0]
    
    print(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    
    # Load preprocessing artifacts using utility function
    print("Loading preprocessing artifacts...")
    scalers, feature_names, log_transform_params = load_scalers_and_features(experiment_dir=str(exp_dir))
    
    # Validate that all required components were loaded
    if scalers is None or feature_names is None or log_transform_params is None:
        raise ValueError("Could not load all required preprocessing artifacts. "
                        "Make sure the experiment directory contains scalers.pkl, "
                        "feature_names.pkl, and log_transform_params.pkl files.")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_best_checkpoint(str(exp_dir))
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, config)
    
    # Load input data
    print(f"Loading input data from {args.input_data}")
    sep_input = '\t' if Path(args.input_data).suffix.lower() == '.txt' else ','
    input_df = pd.read_csv(args.input_data, sep=sep_input)
    print(f"Input data shape: {input_df.shape}")
    
    # Preprocess input data using utility function
    source_platform_key = f'platform_{args.source_platform}'
    target_platform_key = f'platform_{args.target_platform}'
    
    preprocessed_data, id_column = preprocess_input_data(
        input_df, config, scalers, log_transform_params, source_platform_key
    )
    
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    
    # Perform imputation
    imputed_data_normalized, latent_vectors = perform_imputation(
        model, preprocessed_data, args.source_platform, args.target_platform
    )
    
    print(f"Imputed data shape: {imputed_data_normalized.shape}")
    
    # Save latent vectors if requested
    if args.output_latent and latent_vectors is not None:
        print(f"Latent vectors shape: {latent_vectors.shape}")
        
        # Create latent DataFrame
        latent_columns = [f"latent_dim_{i+1}" for i in range(latent_vectors.shape[1])]
        
        # Use provided ID column or detect it
        if args.id_column and args.id_column in input_df.columns:
            sample_ids = input_df[args.id_column].values
        elif id_column in input_df.columns:
            sample_ids = input_df[id_column].values
        else:
            sample_ids = input_df.index.values
        
        latent_df = pd.DataFrame(latent_vectors, 
                                columns=latent_columns, 
                                index=sample_ids)
        
        # Save latent vectors
        latent_path = Path(args.output_latent)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        latent_df.to_csv(latent_path)
        
        print(f"Latent vectors saved to: {latent_path}")
        
        # Save metadata
        metadata_path = latent_path.with_suffix('.metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Latent Vector Extraction Metadata\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Input: {args.input_data}\n")
            f.write(f"Source platform: {args.source_platform}\n")
            f.write(f"Input shape: {preprocessed_data.shape}\n")
            f.write(f"Latent shape: {latent_vectors.shape}\n")
            f.write(f"Latent dimensions: {latent_vectors.shape[1]}\n")
            f.write(f"Samples processed: {len(sample_ids)}\n")
        
        print(f"Latent metadata saved to: {metadata_path}")
    elif args.output_latent:
        print("Warning: Latent vector output requested but extraction failed")
    
    print(f"Imputed data shape: {imputed_data_normalized.shape}")
    
    # Compute feature importance if output path is provided
    if args.output_importance:
        if not CAPTUM_AVAILABLE:
            print("Warning: Captum not available. Skipping feature importance computation.")
        else:
            try:
                print("\nComputing feature importance...")
                
                # Get source feature names
                source_features = None
                if feature_names and source_platform_key in feature_names:
                    source_features = feature_names[source_platform_key]
                else:
                    # Generate feature names for source platform
                    n_source_features = preprocessed_data.shape[1]
                    source_features = [f"source_feature_{i}" for i in range(n_source_features)]
                
                # Get target feature names
                target_features = None
                if feature_names and target_platform_key in feature_names:
                    target_features = feature_names[target_platform_key]
                else:
                    # Will be determined during importance computation
                    target_features = None
                
                # Compute feature importance with per_output_feature=True by default
                attributions, global_importance = compute_feature_importance(
                    model=model,
                    input_data=preprocessed_data,
                    source_platform=args.source_platform,
                    target_platform=args.target_platform,
                    feature_names=source_features,
                    method=args.importance_method,
                    baseline=args.importance_baseline,
                    n_steps=args.importance_steps,
                    per_output_feature=True  # Always True now
                )
                
                # Save only the importance matrix (input features × output features)
                print("Saving importance matrix...")
                
                # attributions shape: (n_output_features, n_samples, n_input_features)
                n_output_features, n_samples, n_input_features = attributions.shape
                
                # Create target feature names if not provided
                if target_features is None or len(target_features) != n_output_features:
                    target_features = [f"output_feature_{i:04d}" for i in range(n_output_features)]
                
                # Compute per-output-feature global importance
                per_output_importance = np.median(np.abs(attributions), axis=1)  # Shape: (n_output_features, n_input_features)
                
                # Create importance matrix DataFrame (input features × output features)
                # Transpose so rows=input_features, columns=output_features
                importance_matrix = per_output_importance.T  # Shape: (n_input_features, n_output_features)
                importance_matrix_df = pd.DataFrame(
                    importance_matrix,
                    index=source_features,  # Input features as rows
                    columns=target_features  # Output features as columns
                )
                
                # Save the matrix
                importance_path = Path(args.output_importance)
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                importance_matrix_df.to_csv(importance_path)
                
                print(f"Importance matrix saved to: {importance_path}")
                print(f"Matrix shape: {importance_matrix_df.shape} (input features × output features)")
                
                # Print summary statistics
                print(f"Importance statistics:")
                print(f"  Mean importance: {importance_matrix.mean():.6f}")
                print(f"  Std importance: {importance_matrix.std():.6f}")
                print(f"  Max importance: {importance_matrix.max():.6f}")
                
                # Find most important input feature overall
                overall_importance = importance_matrix.mean(axis=1)
                top_input_idx = np.argmax(overall_importance)
                print(f"  Most important input feature overall: {source_features[top_input_idx]} (avg importance: {overall_importance[top_input_idx]:.6f})")
                
            except Exception as e:
                print(f"Error computing feature importance: {e}")
                print("Continuing with imputation without feature importance...")
    
    # Apply inverse transformations using utility function
    imputed_data_original = inverse_transform_output(
        imputed_data_normalized, scalers, log_transform_params, target_platform_key
    )
    
    # Create output DataFrame
    if feature_names and target_platform_key in feature_names:
        target_features = feature_names[target_platform_key]
        print(f"Using {len(target_features)} original feature names for {target_platform_key}")
    else:
        # Generate feature names for target platform
        n_target_features = imputed_data_original.shape[1]
        target_features = [f"imputed_feature_{i}" for i in range(n_target_features)]
        print(f"Generated {n_target_features} generic feature names")
    
    # Handle potential feature count mismatch
    if imputed_data_original.shape[1] != len(target_features):
        print(f"Warning: Imputed data has {imputed_data_original.shape[1]} features, "
              f"but expected {len(target_features)} for {target_platform_key}")
        n_features = imputed_data_original.shape[1]
        target_features = [f"imputed_feature_{i}" for i in range(n_features)]
    
    output_df = pd.DataFrame(imputed_data_original, columns=target_features)
    
    # Add ID column
    output_df.insert(0, id_column, input_df[id_column].values)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving imputed data to {output_path}")
    output_df.to_csv(output_path, index=False)
    
    print(f"\nImputation completed successfully!")
    print(f"Input: {args.input_data} ({args.source_platform.upper()} -> {args.target_platform.upper()})")
    print(f"Output: {output_path}")
    print(f"Output shape: {output_df.shape}")
    
    # Print summary statistics
    try:
        print(f"\nSummary statistics for imputed data:")
        print(f"Mean: {imputed_data_original.mean():.4f}")
        print(f"Std: {imputed_data_original.std():.4f}")
        print(f"Min: {imputed_data_original.min():.4f}")
        print(f"Max: {imputed_data_original.max():.4f}")
        
        # Check for potential issues
        if np.any(np.isnan(imputed_data_original)):
            print("Warning: Imputed data contains NaN values!")
        if np.any(np.isinf(imputed_data_original)):
            print("Warning: Imputed data contains infinite values!")
        if np.any(imputed_data_original < 0):
            n_negative = np.sum(imputed_data_original < 0)
            print(f"Note: {n_negative} imputed values are negative (may be expected depending on data type)")
            
    except Exception as e:
        print(f"Could not compute summary statistics: {e}")


if __name__ == "__main__":
    main() 