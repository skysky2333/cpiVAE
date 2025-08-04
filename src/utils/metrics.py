"""Evaluation metrics for Joint VAE imputation performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from typing import Dict, Tuple, List
import logging
import torch
from itertools import combinations

logger = logging.getLogger(__name__)


def kl_divergence(p_mean, p_logvar, q_mean, q_logvar):
    """
    Compute KL divergence between two Gaussian distributions.
    KL(p || q) where p ~ N(p_mean, exp(p_logvar)) and q ~ N(q_mean, exp(q_logvar))
    """
    kl = 0.5 * (
        q_logvar - p_logvar + 
        (torch.exp(p_logvar) + (p_mean - q_mean).pow(2)) / torch.exp(q_logvar) - 1
    )
    return kl.sum(dim=-1)


def tensors_to_df(tensors, head='', keys=None, ax_names=None):
    """
    Convert list of tensors to pandas DataFrame for analysis.
    
    Args:
        tensors: List of tensors to convert
        head: Prefix for column names
        keys: List of keys/names for each tensor
        ax_names: Names for axes
    
    Returns:
        pandas DataFrame
    """
    if keys is None:
        keys = [f'{head}_{i}' for i in range(len(tensors))]
    
    data = {}
    for i, (tensor, key) in enumerate(zip(tensors, keys)):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        
        if tensor.ndim == 1:
            data[key] = tensor
        else:
            # Flatten multi-dimensional tensors
            data[key] = tensor.flatten()
    
    df = pd.DataFrame(data)
    
    if ax_names and len(ax_names) >= 2:
        df.index.name = ax_names[0]
        # Note: ax_names[1] would be used for column naming in more complex cases
    
    return df


def compute_cross_modal_metrics(
    px_zs: List[List[torch.Tensor]], 
    x_true: List[torch.Tensor],
    modality_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute cross-modal reconstruction metrics from MMVAE cross-modal matrix.
    
    Args:
        px_zs: Cross-modal reconstruction matrix [encoder_modality][decoder_modality]
        x_true: List of true data for each modality
        modality_names: Names for each modality
    
    Returns:
        Dictionary of metrics for each reconstruction path
    """
    if modality_names is None:
        modality_names = [f'modality_{i}' for i in range(len(x_true))]
    
    metrics = {}
    
    for e in range(len(px_zs)):  # Encoder modality
        for d in range(len(px_zs[e])):  # Decoder modality
            if px_zs[e][d] is not None:
                # Convert to numpy
                if isinstance(px_zs[e][d], torch.Tensor):
                    pred = px_zs[e][d].detach().cpu().numpy()
                else:
                    pred = px_zs[e][d]
                
                if isinstance(x_true[d], torch.Tensor):
                    true = x_true[d].detach().cpu().numpy()
                else:
                    true = x_true[d]
                
                # Compute metrics
                reconstruction_metrics = compute_imputation_metrics(true, pred)
                
                # Create key for this reconstruction path
                path_key = f"{modality_names[e]}_to_{modality_names[d]}"
                metrics[path_key] = reconstruction_metrics
    
    return metrics


def analyze_mmvae_latent_alignment(qz_xs: List[Tuple], prior_params: Tuple = None):
    """
    Analyze latent space alignment in MMVAE.
    
    Args:
        qz_xs: List of (mean, logvar) tuples for each modality
        prior_params: (prior_mean, prior_logvar) if using non-standard prior
    
    Returns:
        Dictionary containing KL divergence analysis
    """
    if prior_params is None:
        # Standard normal prior
        prior_mean = torch.zeros_like(qz_xs[0][0])
        prior_logvar = torch.zeros_like(qz_xs[0][1])
    else:
        prior_mean, prior_logvar = prior_params
    
    analysis = {}
    
    # KL divergence from each posterior to prior
    for i, (mean, logvar) in enumerate(qz_xs):
        kl_to_prior = kl_divergence(mean, logvar, prior_mean, prior_logvar)
        analysis[f'kl_q{i}_to_prior'] = kl_to_prior.mean().item()
    
    # Jensen-Shannon divergence between posteriors
    for i, j in combinations(range(len(qz_xs)), 2):
        mean_i, logvar_i = qz_xs[i]
        mean_j, logvar_j = qz_xs[j]
        
        # Symmetric KL divergence (Jensen-Shannon approximation)
        kl_ij = kl_divergence(mean_i, logvar_i, mean_j, logvar_j)
        kl_ji = kl_divergence(mean_j, logvar_j, mean_i, logvar_i)
        js_div = 0.5 * (kl_ij + kl_ji)
        
        analysis[f'js_q{i}_q{j}'] = js_div.mean().item()
        analysis[f'kl_q{i}_to_q{j}'] = kl_ij.mean().item()
        analysis[f'kl_q{j}_to_q{i}'] = kl_ji.mean().item()
    
    return analysis


def compute_imputation_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive imputation metrics.
    
    Args:
        true_values: Ground truth values
        predicted_values: Imputed/predicted values
        feature_names: Names of features (optional)
        
    Returns:
        Dictionary of metrics
    """
    if true_values.shape != predicted_values.shape:
        raise ValueError("True and predicted values must have the same shape")
    
    n_samples, n_features = true_values.shape
    
    # Overall metrics
    overall_r2 = r2_score(true_values.flatten(), predicted_values.flatten())
    overall_rmse = np.sqrt(mean_squared_error(true_values.flatten(), predicted_values.flatten()))
    overall_corr, overall_corr_p = pearsonr(true_values.flatten(), predicted_values.flatten())
    
    # Per-feature metrics
    feature_r2_scores = []
    feature_correlations = []
    feature_rmse_scores = []
    
    for i in range(n_features):
        true_feat = true_values[:, i]
        pred_feat = predicted_values[:, i]
        
        # R² score
        r2 = r2_score(true_feat, pred_feat)
        feature_r2_scores.append(r2)
        
        # Correlation
        corr, _ = pearsonr(true_feat, pred_feat)
        feature_correlations.append(corr)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(true_feat, pred_feat))
        feature_rmse_scores.append(rmse)
    
    feature_r2_scores = np.array(feature_r2_scores)
    feature_correlations = np.array(feature_correlations)
    feature_rmse_scores = np.array(feature_rmse_scores)
    
    # Summary statistics
    metrics = {
        # Overall metrics
        'overall_r2': overall_r2,
        'overall_rmse': overall_rmse,
        'overall_correlation': overall_corr,
        'overall_correlation_pvalue': overall_corr_p,
        
        # Per-feature statistics
        'mean_feature_r2': np.mean(feature_r2_scores),
        'std_feature_r2': np.std(feature_r2_scores),
        'median_feature_r2': np.median(feature_r2_scores),
        'min_feature_r2': np.min(feature_r2_scores),
        'max_feature_r2': np.max(feature_r2_scores),
        
        'mean_feature_correlation': np.mean(feature_correlations),
        'std_feature_correlation': np.std(feature_correlations),
        'median_feature_correlation': np.median(feature_correlations),
        'min_feature_correlation': np.min(feature_correlations),
        'max_feature_correlation': np.max(feature_correlations),
        
        'mean_feature_rmse': np.mean(feature_rmse_scores),
        'std_feature_rmse': np.std(feature_rmse_scores),
        'median_feature_rmse': np.median(feature_rmse_scores),
        'min_feature_rmse': np.min(feature_rmse_scores),
        'max_feature_rmse': np.max(feature_rmse_scores),
        
        # Quality thresholds
        'features_r2_above_0.3': np.sum(feature_r2_scores > 0.3),
        'features_r2_above_0.5': np.sum(feature_r2_scores > 0.5),
        'features_r2_above_0.7': np.sum(feature_r2_scores > 0.7),
        'fraction_r2_above_0.3': np.mean(feature_r2_scores > 0.3),
        'fraction_r2_above_0.5': np.mean(feature_r2_scores > 0.5),
        'fraction_r2_above_0.7': np.mean(feature_r2_scores > 0.7),
        
        'features_corr_above_0.6': np.sum(feature_correlations > 0.6),
        'features_corr_above_0.8': np.sum(feature_correlations > 0.8),
        'fraction_corr_above_0.6': np.mean(feature_correlations > 0.6),
        'fraction_corr_above_0.8': np.mean(feature_correlations > 0.8),
    }
    
    return metrics


def create_detailed_feature_report(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Create detailed per-feature evaluation report.
    
    Args:
        true_values: Ground truth values
        predicted_values: Imputed/predicted values  
        feature_names: Names of features
        
    Returns:
        DataFrame with per-feature metrics
    """
    n_features = true_values.shape[1]
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    results = []
    
    for i, feat_name in enumerate(feature_names):
        true_feat = true_values[:, i]
        pred_feat = predicted_values[:, i]
        
        # Metrics
        r2 = r2_score(true_feat, pred_feat)
        corr, corr_p = pearsonr(true_feat, pred_feat)
        rmse = np.sqrt(mean_squared_error(true_feat, pred_feat))
        mae = np.mean(np.abs(true_feat - pred_feat))
        
        # Value ranges
        true_range = np.max(true_feat) - np.min(true_feat)
        pred_range = np.max(pred_feat) - np.min(pred_feat)
        
        results.append({
            'feature': feat_name,
            'r2_score': r2,
            'correlation': corr,
            'correlation_pvalue': corr_p,
            'rmse': rmse,
            'mae': mae,
            'true_mean': np.mean(true_feat),
            'pred_mean': np.mean(pred_feat),
            'true_std': np.std(true_feat),
            'pred_std': np.std(pred_feat),
            'true_range': true_range,
            'pred_range': pred_range,
            'range_ratio': pred_range / true_range if true_range > 0 else np.nan
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('r2_score', ascending=False).reset_index(drop=True)
    
    return df


 