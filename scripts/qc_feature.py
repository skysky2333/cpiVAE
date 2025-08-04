#!/usr/bin/env python3
"""
Feature Quality Control script using QC VAE for feature filtering.

This script trains a QC VAE on metabolite data and identifies poor-quality features
based on reconstruction error and correlation across samples. Optionally, it can analyze
the relationship between feature quality metrics and feature annotations (categorical
or continuous) to understand patterns in feature quality.

Additional validation capabilities include:
- Binary classification analysis to prove QC improves biological discovery
- Comparison of VAE-based vs classic QC methods
- Statistical testing of feature set performance (DeLong tests)
- Pathway enrichment and network analysis for biological validation
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from scipy import stats
import umap
import warnings
import logging
warnings.filterwarnings("ignore")

# Suppress fontTools subset logging messages
logging.getLogger('fontTools.subset').setLevel(logging.WARNING)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.qc_vae import QCVAELightning
from data.datamodule import SingleDataModule

from qc_feature_helper import run_biological_analysis, run_classification_analysis
BIOLOGICAL_ANALYSIS_AVAILABLE = True

# ============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# ============================================================================

QC_CONFIG = {
    # Model Architecture
    'model': {
        'model_type': "qc_vae",
        'latent_dim': 128,
        'encoder_layers': [1024],  # Hidden layer sizes for encoder
        'decoder_layers': [512],   # Hidden layer sizes for decoder
        'activation': "leaky_relu",  # Options: "relu", "leaky_relu", "gelu", "swish", "tanh", "elu"
        'dropout_rate': 0.15,
        'batch_norm': True,
        'use_residual_blocks': True,  # Use residual connections in MLP layers
    },
    
    # Training Parameters
    'training': {
        'max_epochs': 200,
        'learning_rate': 0.001,  # Reduced from 0.01 to prevent instability
        'batch_size': 256,
        'optimizer': "adamw",
        'scheduler': "reduce_on_plateau",
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'early_stopping_patience': 15,
        'gradient_clip_val': 0.5,  # Reduced for more aggressive clipping
        
        # Data Augmentation
        'data_augmentation': {
            'enabled': True,
            'gaussian_noise_std': 0.01,
        }
    },
    
    # Loss Function Weights
    'loss_weights': {
        'reconstruction': 1.0,
        'kl_divergence': 0.01,
    },
    
    # Data Processing
    'data': {
        'train_split': 0.8,
        'test_split': 0.2,
        'normalization_method': "zscore",  # Options: zscore, minmax, robust
        'missing_value_strategy': "mean",  # Options: mean, median, knn, drop
        'random_seed': 42,
        
        # Log Transformation Configuration
        'log_transform': {
            'enabled': False,  # Set to True to apply log transformation
            'epsilon': 1e-8,  # Small value added to ensure positive values
        }
    },
    
    # Feature Quality Control Parameters
    'feature_qc': {
        # QC Metric Selection
        'qc_metric': 'r2_baseline',  # Options: correlation, normalized_rmse, nash_sutcliffe, normalized_mse, r2_baseline
        
        # Legacy threshold (maintained for backward compatibility)
        'correlation_threshold': 0.7,  # Features with correlation < this threshold are filtered
        
        # Thresholds for different QC metrics
        'thresholds': {
            'correlation': 0.7,       # Higher is better (Pearson correlation)
            'normalized_rmse': 0.7,   # Lower is better (RMSE / feature_std)
            'nash_sutcliffe': 0.4,    # Higher is better (Nash-Sutcliffe efficiency)
            'normalized_mse': 0.7,    # Lower is better (MSE / feature_var)
            'r2_baseline': 0.5,       # Higher is better (R² against no-skill baseline)
        },
        
        # Normalization options
        'normalization': {
            'method': 'std',          # Options: 'std' (standard deviation), 'var' (variance), 'iqr' (interquartile range)
            'handle_zero_variance': True,  # Replace zero variance with small epsilon
            'epsilon': 1e-8,          # Small value for zero variance replacement
        },
        
        'plot_sample_size': 5000,      # Max number of samples to plot (for performance)
    },
    
    # Visualization
    'visualization': {
        'palette': {
            'primary': '#4dbbd5',
            'secondary': '#00a087',
            'accent': '#3c5488',
            'error': '#e64b35',
            'highlight': '#f39b7f',
            'good': '#00a087',
            'warning': '#f39b7f'
        },
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.1,
        'umap_random_state': 42,
        'pca_n_components': 2,
        'plot_style': "seaborn-v0_8",
        'figsize': [12, 8],
        'dpi': 300,
    },
    
    # Hardware
    'hardware': {
        'accelerator': "auto",  # auto, cpu, gpu, mps
        'devices': "auto",
        'precision': 32,
    }
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Feature Quality Control using QC VAE')
    
    parser.add_argument(
        '--data_file', 
        type=str, 
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs_feature_qc',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default='feature_qc_vae_experiment',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--fast_dev_run', 
        action='store_true',
        help='Run a fast development run for debugging'
    )
    parser.add_argument(
        '--feature_annot',
        type=str,
        help='Path to tab-delimited annotation file for features (first column should match data column names)'
    )
    parser.add_argument(
        '--categorical_annot',
        type=str,
        nargs='*',
        help='List of column names in annotation file that contain categorical annotations'
    )
    parser.add_argument(
        '--continuous_annot',
        type=str,
        nargs='*',
        help='List of column names in annotation file that contain continuous annotations'
    )
    
    # Classification analysis arguments
    parser.add_argument(
        '--sample_pheno',
        type=str,
        help='Path to phenotype table with sample IDs as index and phenotypes as columns'
    )
    parser.add_argument(
        '--binary_pheno',
        type=str,
        nargs='+',
        help='List of column names in phenotype file for binary classification'
    )
    parser.add_argument(
        '--classic_qc_flag',
        type=str,
        help='Column name in annotation file that contains categorical annotations for classic QC'
    )
    
    # Biological analysis arguments
    parser.add_argument(
        '--run_biological_analysis',
        action='store_true',
        help='Run biological validation analysis (pathway enrichment and network analysis)'
    )
    parser.add_argument(
        '--run_enrichment',
        action='store_true',
        help='Run pathway enrichment analysis comparing best vs worst features'
    )
    parser.add_argument(
        '--run_network',
        action='store_true',
        help='Run STRING protein interaction network analysis'
    )
    parser.add_argument(
        '--organism',
        type=str,
        default='hsapiens',
        help='Organism for enrichment analysis (hsapiens, mmusculus, etc.)'
    )
    parser.add_argument(
        '--decile_cutoff',
        type=float,
        default=0.1,
        help='Fraction of features to include in top/bottom groups for enrichment (default: 0.1)'
    )
    parser.add_argument(
        '--network_top_n',
        type=int,
        default=100,
        help='Number of top features to include in STRING network analysis (default: 100)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=int,
        default=400,
        help='STRING confidence threshold 0-1000 (default: 400)'
    )
    
    return parser.parse_args()


def setup_logging_and_checkpointing(output_dir, experiment_name):
    """Setup logging and checkpointing callbacks."""
    import time
    version = time.strftime("version_%Y%m%d-%H%M%S")

    exp_dir = Path(output_dir) / experiment_name / version
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=str(exp_dir.parent.parent),
        name=experiment_name,
        version=version
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f'{experiment_name}-{{epoch:02d}}-{{val_total_loss:.3f}}',
        monitor='val_total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_total_loss',
        mode='min',
        patience=QC_CONFIG['training']['early_stopping_patience'],
        min_delta=0.0,
        verbose=True,
        strict=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    
    return logger, callbacks, exp_dir


def train_qc_vae(datamodule, output_dir, experiment_name, fast_dev_run=False):
    """Train the QC VAE model."""
    print("\n" + "="*60)
    print("TRAINING QC VAE MODEL FOR FEATURE QC")
    print("="*60)
    
    # Setup logging and callbacks
    logger, callbacks, exp_dir = setup_logging_and_checkpointing(output_dir, experiment_name)
    
    # Get data dimensions
    dims = datamodule.get_dims()
    print(f"Input dimensions: {dims['input_dim']}")
    
    # Initialize model
    model = QCVAELightning(
        input_dim=dims['input_dim'],
        config=QC_CONFIG,
        datamodule=datamodule
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=QC_CONFIG['training']['max_epochs'],
        accelerator=QC_CONFIG['hardware']['accelerator'],
        devices=QC_CONFIG['hardware']['devices'],
        precision=QC_CONFIG['hardware']['precision'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=QC_CONFIG['training']['gradient_clip_val'],
        gradient_clip_algorithm='norm',
        log_every_n_steps=50,
        fast_dev_run=fast_dev_run,
        deterministic=True,
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    if not fast_dev_run:
        print("Training completed successfully!")
        print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        
        # Load best model
        best_model = QCVAELightning.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            input_dim=dims['input_dim'],
            config=QC_CONFIG,
            datamodule=datamodule
        )
        return best_model, exp_dir
    else:
        print("Fast dev run completed!")
        return model, exp_dir




def get_normalization_factor(data, method='std', epsilon=1e-8, handle_zero_variance=True):
    """
    Compute normalization factor for features.
    
    Args:
        data: Array of shape (n_samples, n_features)
        method: 'std', 'var', or 'iqr'
        epsilon: Small value for zero variance replacement
        handle_zero_variance: Whether to replace zero variance with epsilon
    
    Returns:
        Array of normalization factors for each feature
    """
    if method == 'std':
        factors = np.std(data, axis=0)
    elif method == 'var':
        factors = np.var(data, axis=0)
    elif method == 'iqr':
        factors = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if handle_zero_variance:
        factors = np.where(factors == 0, epsilon, factors)
    
    return factors


def compute_normalized_rmse(original_data, reconstructed_data, normalization_config):
    """
    Compute normalized RMSE (RMSE divided by feature normalization factor).
    
    Args:
        original_data: Original data array (n_samples, n_features)
        reconstructed_data: Reconstructed data array (n_samples, n_features)
        normalization_config: Normalization configuration from QC_CONFIG
    
    Returns:
        Array of normalized RMSE values for each feature
    """
    rmse = np.sqrt(np.mean((reconstructed_data - original_data) ** 2, axis=0))
    
    normalization_factors = get_normalization_factor(
        original_data,
        method=normalization_config['method'],
        epsilon=normalization_config['epsilon'],
        handle_zero_variance=normalization_config['handle_zero_variance']
    )
    
    normalized_rmse = rmse / normalization_factors
    return normalized_rmse


def compute_nash_sutcliffe_efficiency(original_data, reconstructed_data):
    """
    Compute Nash-Sutcliffe efficiency: 1 - (SS_res / SS_tot).
    
    Args:
        original_data: Original data array (n_samples, n_features)
        reconstructed_data: Reconstructed data array (n_samples, n_features)
    
    Returns:
        Array of Nash-Sutcliffe efficiency values for each feature
    """
    # Residual sum of squares
    ss_res = np.sum((original_data - reconstructed_data) ** 2, axis=0)
    
    # Total sum of squares (variance around mean)
    ss_tot = np.sum((original_data - np.mean(original_data, axis=0)) ** 2, axis=0)
    
    # Nash-Sutcliffe efficiency
    nash_sutcliffe = 1 - (ss_res / ss_tot)
    
    # Handle cases where total sum of squares is zero (constant features)
    nash_sutcliffe = np.where(ss_tot == 0, 0.0, nash_sutcliffe)
    
    return nash_sutcliffe


def compute_normalized_mse(original_data, reconstructed_data, normalization_config):
    """
    Compute normalized MSE (MSE divided by feature variance or other normalization factor).
    
    Args:
        original_data: Original data array (n_samples, n_features)
        reconstructed_data: Reconstructed data array (n_samples, n_features)
        normalization_config: Normalization configuration from QC_CONFIG
    
    Returns:
        Array of normalized MSE values for each feature
    """
    mse = np.mean((reconstructed_data - original_data) ** 2, axis=0)
    
    # For MSE normalization, we typically use variance
    normalization_factors = get_normalization_factor(
        original_data,
        method='var',  # Use variance for MSE normalization
        epsilon=normalization_config['epsilon'],
        handle_zero_variance=normalization_config['handle_zero_variance']
    )
    
    normalized_mse = mse / normalization_factors
    return normalized_mse


def compute_r2_baseline(original_data, reconstructed_data):
    """
    Compute R² against no-skill baseline (same as Nash-Sutcliffe efficiency).
    
    Args:
        original_data: Original data array (n_samples, n_features)
        reconstructed_data: Reconstructed data array (n_samples, n_features)
    
    Returns:
        Array of R² baseline values for each feature
    """
    # R² against no-skill baseline is equivalent to Nash-Sutcliffe efficiency
    return compute_nash_sutcliffe_efficiency(original_data, reconstructed_data)


def perform_feature_qc_analysis(model, datamodule, output_dir):
    """Perform feature quality control analysis."""
    print("\n" + "="*60)
    print("PERFORMING FEATURE QUALITY CONTROL ANALYSIS")
    print("="*60)
    
    model.eval()
    
    # Ensure data and model are on the same device
    device = next(model.parameters()).device
    print(f"Analyzing on device: {device}")

    full_data = datamodule.full_data.to(device)
    original_data = datamodule.original_data
    
    print(f"Analyzing {full_data.shape[1]} features across {full_data.shape[0]} samples...")
    
    # Compute reconstructions for all samples on processed data
    with torch.no_grad():
        outputs = model(full_data)
        reconstructions_processed = outputs['recon'].detach().cpu().numpy()
    
    full_data_processed = full_data.detach().cpu().numpy()
    
    # Apply inverse transformations to get data back to original scale
    preprocessor = datamodule.get_preprocessor()
    
    # First inverse normalize
    if preprocessor.scaler is not None:
        full_data_inv_norm = preprocessor.scaler.inverse_transform(full_data_processed)
        reconstructions_inv_norm = preprocessor.scaler.inverse_transform(reconstructions_processed)
    else:
        full_data_inv_norm = full_data_processed
        reconstructions_inv_norm = reconstructions_processed
    
    # Then apply inverse log transformation if enabled
    full_data_original = preprocessor.inverse_log_transform_single(full_data_inv_norm)
    reconstructions_original = preprocessor.inverse_log_transform_single(reconstructions_inv_norm)
    
    print("Computing feature-wise metrics on original data scale...")
    
    # Compute per-feature reconstruction errors on original scale (MSE across samples)
    feature_errors = np.mean((reconstructions_original - full_data_original) ** 2, axis=0)
    
    # Compute traditional feature-wise metrics on original scale
    feature_correlations = []
    feature_r2_scores = []
    
    for i in range(full_data_original.shape[1]):
        # Correlation between original and reconstructed
        corr = np.corrcoef(full_data_original[:, i], reconstructions_original[:, i])[0, 1]
        feature_correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # R² score
        ss_res = np.sum((full_data_original[:, i] - reconstructions_original[:, i]) ** 2)
        ss_tot = np.sum((full_data_original[:, i] - np.mean(full_data_original[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        feature_r2_scores.append(r2)
    
    feature_correlations = np.array(feature_correlations)
    feature_r2_scores = np.array(feature_r2_scores)
    
    # Compute new QC metrics using helper functions
    print("Computing new normalized QC metrics...")
    normalization_config = QC_CONFIG['feature_qc']['normalization']
    
    normalized_rmse = compute_normalized_rmse(full_data_original, reconstructions_original, normalization_config)
    nash_sutcliffe = compute_nash_sutcliffe_efficiency(full_data_original, reconstructions_original)
    normalized_mse = compute_normalized_mse(full_data_original, reconstructions_original, normalization_config)
    r2_baseline = compute_r2_baseline(full_data_original, reconstructions_original)
    
    # Get QC metric selection and threshold from config
    qc_metric = QC_CONFIG['feature_qc']['qc_metric']
    qc_thresholds = QC_CONFIG['feature_qc']['thresholds']
    
    # Map metric names to computed values
    qc_metrics = {
        'correlation': feature_correlations,
        'normalized_rmse': normalized_rmse,
        'nash_sutcliffe': nash_sutcliffe,
        'normalized_mse': normalized_mse,
        'r2_baseline': r2_baseline
    }
    
    # Check if selected metric is valid
    if qc_metric not in qc_metrics:
        raise ValueError(f"Unknown QC metric '{qc_metric}'. Available: {list(qc_metrics.keys())}")
    
    # Get the selected metric values and threshold
    selected_metric_values = qc_metrics[qc_metric]
    selected_threshold = qc_thresholds[qc_metric]
    
    higher_is_better = {'correlation', 'nash_sutcliffe', 'r2_baseline'}
    lower_is_better = {'normalized_rmse', 'normalized_mse'}
    
    # Determine poor-quality features based on selected metric
    if qc_metric in higher_is_better:
        is_poor_feature = selected_metric_values < selected_threshold
        threshold_direction = "below"
    elif qc_metric in lower_is_better:
        is_poor_feature = selected_metric_values > selected_threshold
        threshold_direction = "above"
    else:
        raise ValueError(f"Unknown metric direction for '{qc_metric}'")
    
    # Legacy threshold for backward compatibility
    correlation_threshold = QC_CONFIG['feature_qc']['correlation_threshold']
    
    print(f"Selected QC metric: {qc_metric}")
    print(f"QC threshold: {selected_threshold:.2f} ({threshold_direction} threshold = poor quality)")
    print(f"Number of poor-quality features: {np.sum(is_poor_feature)} / {len(feature_errors)} ({100*np.sum(is_poor_feature)/len(feature_errors):.1f}%)")
    print(f"Metric value range: {selected_metric_values.min():.4f} to {selected_metric_values.max():.4f}")
    
    # Create results dataframe with all QC metrics
    results_df = pd.DataFrame({
        'feature_name': original_data.columns,
        'reconstruction_error': feature_errors,
        'correlation': feature_correlations,
        'r2_score': feature_r2_scores,
        'normalized_rmse': normalized_rmse,
        'nash_sutcliffe': nash_sutcliffe,
        'normalized_mse': normalized_mse,
        'r2_baseline': r2_baseline,
        'selected_qc_metric': selected_metric_values,
        'selected_qc_metric_name': qc_metric,
        'qc_threshold': selected_threshold,
        'is_poor_quality': is_poor_feature,
        'quality_category': ['Poor Quality' if poor else 'Good Quality' for poor in is_poor_feature]
    })
    
    # Sort by selected QC metric (worst first)
    # For "higher is better" metrics, ascending=True shows worst (lowest) first
    # For "lower is better" metrics, ascending=False shows worst (highest) first
    sort_ascending = qc_metric in higher_is_better
    results_df = results_df.sort_values('selected_qc_metric', ascending=sort_ascending)
    
    # Save results
    results_path = output_dir / 'feature_qc_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Feature QC results saved to: {results_path}")
    
    # Print summary statistics
    print("\nFEATURE QC ANALYSIS SUMMARY:")
    print("="*50)
    print(f"Selected QC Metric: {qc_metric}")
    print(f"QC Threshold: {selected_threshold:.4f} ({threshold_direction} = poor quality)")
    print(f"Selected Metric Range: {selected_metric_values.min():.4f} to {selected_metric_values.max():.4f}")
    print(f"Selected Metric - Mean: {selected_metric_values.mean():.4f}, Std: {selected_metric_values.std():.4f}")
    print("\nALL QC METRICS SUMMARY:")
    print(f"Reconstruction Error (MSE) - Mean: {feature_errors.mean():.6f}, Std: {feature_errors.std():.6f}")
    print(f"Correlation - Mean: {feature_correlations.mean():.4f}, Std: {feature_correlations.std():.4f}")
    print(f"R² Score - Mean: {feature_r2_scores.mean():.4f}, Std: {feature_r2_scores.std():.4f}")
    print(f"Normalized RMSE - Mean: {normalized_rmse.mean():.4f}, Std: {normalized_rmse.std():.4f}")
    print(f"Nash-Sutcliffe Efficiency - Mean: {nash_sutcliffe.mean():.4f}, Std: {nash_sutcliffe.std():.4f}")
    print(f"Normalized MSE - Mean: {normalized_mse.mean():.4f}, Std: {normalized_mse.std():.4f}")
    print(f"R² Baseline - Mean: {r2_baseline.mean():.4f}, Std: {r2_baseline.std():.4f}")
    
    # Quality category breakdown
    category_counts = results_df['quality_category'].value_counts()
    print("\nQUALITY CATEGORY BREAKDOWN:")
    print("="*40)
    total_features = len(results_df)
    for cat, count in category_counts.items():
        print(f"{cat}: {count} features ({100*count/total_features:.1f}%)")
    
    # Top and bottom features based on selected metric
    worst_label = "lowest" if qc_metric in higher_is_better else "highest"
    best_label = "highest" if qc_metric in higher_is_better else "lowest"
    
    print(f"\nTOP 10 WORST FEATURES ({worst_label} {qc_metric}):")
    print("-" * 80)
    for idx, row in results_df.head(10).iterrows():
        print(f"{row['feature_name']}: {qc_metric}={row['selected_qc_metric']:.4f}, corr={row['correlation']:.3f}, MSE={row['reconstruction_error']:.6f}")
    
    print(f"\nTOP 10 BEST FEATURES ({best_label} {qc_metric}):")
    print("-" * 80)
    for idx, row in results_df.tail(10).iterrows():
        print(f"{row['feature_name']}: {qc_metric}={row['selected_qc_metric']:.4f}, corr={row['correlation']:.3f}, MSE={row['reconstruction_error']:.6f}")
    
    return results_df, reconstructions_original, full_data_original


def create_feature_qc_plots(results_df, reconstructions, full_data, original_data, output_dir):
    """Create visualization plots for feature QC analysis."""
    print("\n" + "="*60)
    print("CREATING FEATURE QC VISUALIZATION PLOTS")
    print("="*60)
    
    # Set up plotting style and colors
    plt.style.use(QC_CONFIG['visualization']['plot_style'])
    palette = QC_CONFIG['visualization']['palette']
    
    # Sample data for plotting if too large
    max_samples = QC_CONFIG['feature_qc']['plot_sample_size']
    if len(full_data) > max_samples:
        sample_indices = np.random.choice(len(full_data), max_samples, replace=False)
        data_sample = full_data[sample_indices]
        recon_sample = reconstructions[sample_indices]
    else:
        data_sample = full_data
        recon_sample = reconstructions
    
    print(f"Creating plots with {len(data_sample)} samples...")
    
    # Get QC metric information from results dataframe
    qc_metric = results_df['selected_qc_metric_name'].iloc[0]  # All rows have same metric name
    qc_threshold = results_df['qc_threshold'].iloc[0]  # All rows have same threshold
    
    higher_is_better = {'correlation', 'nash_sutcliffe', 'r2_baseline'}
    lower_is_better = {'normalized_rmse', 'normalized_mse'}
    
    # 1. Joint distribution plot (separate figure, similar to qc.py)
    g = sns.jointplot(
        x='selected_qc_metric',
        y='reconstruction_error',
        data=results_df,
        kind='scatter',
        hue='quality_category',
        palette={
            "Good Quality": palette['good'],
            "Poor Quality": palette['error']
        },
        marginal_kws=dict(fill=True),
        height=7
    )
    
    # Add threshold line
    g.ax_joint.axvline(x=qc_threshold, color=palette['error'], linestyle='--', 
                      label=f'{qc_metric.title()} Threshold ({qc_threshold:.3f})')
    
    if qc_metric in higher_is_better:
        fill_x = [g.ax_joint.get_xlim()[0], qc_threshold]
        region_label = f'Poor Quality Region (Low {qc_metric.title()})'
    else:
        fill_x = [qc_threshold, g.ax_joint.get_xlim()[1]]
        region_label = f'Poor Quality Region (High {qc_metric.title()})'
    
    g.ax_joint.fill_between(
        fill_x,
        g.ax_joint.get_ylim()[0],
        g.ax_joint.get_ylim()[1],
        color=palette['error'],
        alpha=0.2,
        label=region_label
    )
    g.ax_joint.legend()
    g.ax_joint.set_yscale('log')
    
    metric_labels = {
        'correlation': 'Correlation (Original vs Reconstructed)',
        'normalized_rmse': 'Normalized RMSE',
        'nash_sutcliffe': 'Nash-Sutcliffe Efficiency',
        'normalized_mse': 'Normalized MSE',
        'r2_baseline': 'R² vs Baseline'
    }
    
    x_label = metric_labels.get(qc_metric, qc_metric.title())
    g.set_axis_labels(x_label, 'Reconstruction Error (Log Scale)', fontsize=12)
    g.fig.suptitle(f'Feature Quality: {x_label} vs. Reconstruction Error', y=1.02, fontsize=14)
    
    joint_plot_path = output_dir / 'feature_qc_joint_distribution.png'
    plt.savefig(joint_plot_path, dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].hist(results_df['selected_qc_metric'], bins=50, alpha=0.7, 
                    color=palette['primary'], edgecolor='black')
    axes[0, 0].axvline(qc_threshold, color=palette['error'], linestyle='--', 
                      label=f'{qc_metric.title()} Threshold ({qc_threshold:.3f})')
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of {x_label}')
    axes[0, 0].legend()
    
    # Placeholder for the jointplot
    axes[0, 1].text(0.5, 0.5, f'Joint distribution plot saved to:\nfeature_qc_joint_distribution.png', 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="black", lw=1))
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f'{x_label} vs Reconstruction Error')
    
    # R² vs Selected QC Metric scatter
    good_features = results_df['quality_category'] == 'Good Quality'
    axes[1, 0].scatter(results_df[good_features]['selected_qc_metric'], 
                      results_df[good_features]['r2_score'], 
                      c=palette['good'], alpha=0.6, s=20, label='Good Quality')
    axes[1, 0].scatter(results_df[~good_features]['selected_qc_metric'], 
                      results_df[~good_features]['r2_score'], 
                      c=palette['error'], alpha=0.8, s=20, label='Poor Quality')
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title(f'{x_label} vs R² Score')
    axes[1, 0].legend()
    
    # Quality category pie chart
    category_counts = results_df['quality_category'].value_counts()
    colors = [palette['good'] if 'Good' in cat else palette['error'] for cat in category_counts.index]
    axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[1, 1].set_title('Feature Quality Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_qc_overview.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # 2. Feature comparison plots (best vs worst features)
    n_examples = 6
    worst_features = results_df.head(n_examples)
    best_features = results_df.tail(n_examples)
    
    fig, axes = plt.subplots(2, n_examples, figsize=(20, 8))
    
    for i, (idx, row) in enumerate(worst_features.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        axes[0, i].scatter(data_sample[:, feature_idx], recon_sample[:, feature_idx], 
                          alpha=0.5, s=10, color=palette['error'])
        
        # Add diagonal line
        min_val = min(data_sample[:, feature_idx].min(), recon_sample[:, feature_idx].min())
        max_val = max(data_sample[:, feature_idx].max(), recon_sample[:, feature_idx].max())
        axes[0, i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        
        axes[0, i].set_xlabel('Original')
        axes[0, i].set_ylabel('Reconstructed')
        # Create abbreviated metric label for title
        qc_metric_abbrev = {
            'correlation': 'Corr',
            'normalized_rmse': 'NRMSE',
            'nash_sutcliffe': 'NSE',
            'normalized_mse': 'NMSE',
            'r2_baseline': 'R²'
        }
        qc_abbrev = qc_metric_abbrev.get(qc_metric, qc_metric[:6])
        axes[0, i].set_title(f'WORST: {row["feature_name"][:15]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
    
    for i, (idx, row) in enumerate(best_features.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        axes[1, i].scatter(data_sample[:, feature_idx], recon_sample[:, feature_idx], 
                          alpha=0.5, s=10, color=palette['good'])
        
        # Add diagonal line
        min_val = min(data_sample[:, feature_idx].min(), recon_sample[:, feature_idx].min())
        max_val = max(data_sample[:, feature_idx].max(), recon_sample[:, feature_idx].max())
        axes[1, i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        
        axes[1, i].set_xlabel('Original')
        axes[1, i].set_ylabel('Reconstructed')
        axes[1, i].set_title(f'BEST: {row["feature_name"][:15]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_comparison_examples.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # 3. Feature profiles for worst and best features
    n_profile_features = 5
    worst_features_profile = results_df.head(n_profile_features)
    best_features_profile = results_df.tail(n_profile_features)
    
    fig, axes = plt.subplots(2, n_profile_features, figsize=(25, 8))
    
    for i, (idx, row) in enumerate(worst_features_profile.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        # Plot first 100 samples for clarity
        n_samples = min(100, len(data_sample))
        sample_range = range(n_samples)
        
        axes[0, i].plot(sample_range, data_sample[:n_samples, feature_idx], 
                       'o-', label='Original', color=palette['primary'], markersize=3, alpha=0.7)
        axes[0, i].plot(sample_range, recon_sample[:n_samples, feature_idx], 
                       'o-', label='Reconstructed', color=palette['error'], markersize=3, alpha=0.7)
        
        axes[0, i].set_xlabel('Sample Index')
        axes[0, i].set_ylabel('Value')
        axes[0, i].set_title(f'WORST: {row["feature_name"][:20]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    for i, (idx, row) in enumerate(best_features_profile.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        # Plot first 100 samples for clarity
        n_samples = min(100, len(data_sample))
        sample_range = range(n_samples)
        
        axes[1, i].plot(sample_range, data_sample[:n_samples, feature_idx], 
                       'o-', label='Original', color=palette['primary'], markersize=3, alpha=0.7)
        axes[1, i].plot(sample_range, recon_sample[:n_samples, feature_idx], 
                       'o-', label='Reconstructed', color=palette['good'], markersize=3, alpha=0.7)
        
        axes[1, i].set_xlabel('Sample Index')
        axes[1, i].set_ylabel('Value')
        axes[1, i].set_title(f'BEST: {row["feature_name"][:20]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_profile_examples.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # 4. Feature profiles for worst and best features (ranked by original values)
    fig, axes = plt.subplots(2, n_profile_features, figsize=(25, 8))
    
    for i, (idx, row) in enumerate(worst_features_profile.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        # Plot first 100 samples for clarity
        n_samples = min(100, len(data_sample))
        
        # Get original values for this feature and create ranking
        original_values = data_sample[:n_samples, feature_idx]
        recon_values = recon_sample[:n_samples, feature_idx]
        
        # Create ranking based on original values (sort by original values)
        sorted_indices = np.argsort(original_values)
        ranked_original = original_values[sorted_indices]
        ranked_recon = recon_values[sorted_indices]
        rank_range = range(1, n_samples + 1)  # Rank from 1 to n_samples
        
        axes[0, i].plot(rank_range, ranked_original, 
                       'o-', label='Original', color=palette['primary'], markersize=3, alpha=0.7)
        axes[0, i].plot(rank_range, ranked_recon, 
                       'o-', label='Reconstructed', color=palette['error'], markersize=3, alpha=0.7)
        
        axes[0, i].set_xlabel('Rank (by Original Value)')
        axes[0, i].set_ylabel('Value')
        axes[0, i].set_title(f'WORST: {row["feature_name"][:20]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    for i, (idx, row) in enumerate(best_features_profile.iterrows()):
        feature_idx = original_data.columns.get_loc(row['feature_name'])
        
        # Plot first 100 samples for clarity
        n_samples = min(100, len(data_sample))
        
        # Get original values for this feature and create ranking
        original_values = data_sample[:n_samples, feature_idx]
        recon_values = recon_sample[:n_samples, feature_idx]
        
        # Create ranking based on original values (sort by original values)
        sorted_indices = np.argsort(original_values)
        ranked_original = original_values[sorted_indices]
        ranked_recon = recon_values[sorted_indices]
        rank_range = range(1, n_samples + 1)  # Rank from 1 to n_samples
        
        axes[1, i].plot(rank_range, ranked_original, 
                       'o-', label='Original', color=palette['primary'], markersize=3, alpha=0.7)
        axes[1, i].plot(rank_range, ranked_recon, 
                       'o-', label='Reconstructed', color=palette['good'], markersize=3, alpha=0.7)
        
        axes[1, i].set_xlabel('Rank (by Original Value)')
        axes[1, i].set_ylabel('Value')
        axes[1, i].set_title(f'BEST: {row["feature_name"][:20]}...\n{qc_abbrev}: {row["selected_qc_metric"]:.3f}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_profile_examples_ranked.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    print("Feature QC visualization plots saved successfully!")
    print(f"Plots saved to:")
    print(f"  - {output_dir / 'feature_qc_joint_distribution.png'}")
    print(f"  - {output_dir / 'feature_qc_overview.png'}")
    print(f"  - {output_dir / 'feature_comparison_examples.png'}")
    print(f"  - {output_dir / 'feature_profile_examples.png'}")
    print(f"  - {output_dir / 'feature_profile_examples_ranked.png'}")


def create_pca_variance_analysis(results_df, full_data_original, original_data, output_dir):
    """Create PCA variance analysis to validate feature filtering."""
    print("\n" + "="*60)
    print("CREATING PCA VARIANCE ANALYSIS")
    print("="*60)
    
    palette = QC_CONFIG['visualization']['palette']
    
    # Filter features
    good_features = results_df[results_df['quality_category'] == 'Good Quality']
    poor_features = results_df[results_df['quality_category'] == 'Poor Quality']
    
    good_feature_indices = [original_data.columns.get_loc(name) for name in good_features['feature_name']]
    poor_feature_indices = [original_data.columns.get_loc(name) for name in poor_features['feature_name']]
    
    # Standardize data for PCA (important for comparing variance explained)
    from sklearn.preprocessing import StandardScaler
    scaler_full = StandardScaler()
    full_data_std = scaler_full.fit_transform(full_data_original)
    
    if len(good_feature_indices) > 0:
        good_data = full_data_original[:, good_feature_indices]
        scaler_good = StandardScaler()
        good_data_std = scaler_good.fit_transform(good_data)
    else:
        good_data_std = np.array([]).reshape(len(full_data_original), 0)
    
    # Compute PCA for full dataset
    from sklearn.decomposition import PCA
    max_components = min(50, full_data_std.shape[1], full_data_std.shape[0] - 1)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(full_data_std)
    
    # Compute PCA for filtered dataset (if we have good features)
    if len(good_feature_indices) > 0:
        max_components_filtered = min(50, good_data_std.shape[1], good_data_std.shape[0] - 1)
        pca_filtered = PCA(n_components=max_components_filtered)
        pca_filtered.fit(good_data_std)
    else:
        pca_filtered = None
        max_components_filtered = 0
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative variance explained comparison
    components_full = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)
    cumvar_full = np.cumsum(pca_full.explained_variance_ratio_)
    
    axes[0, 0].plot(components_full, cumvar_full, 'o-', color=palette['primary'], 
                   label=f'Before filtering ({full_data_std.shape[1]} features)', linewidth=2, markersize=4)
    
    if pca_filtered is not None:
        components_filtered = np.arange(1, len(pca_filtered.explained_variance_ratio_) + 1)
        cumvar_filtered = np.cumsum(pca_filtered.explained_variance_ratio_)
        axes[0, 0].plot(components_filtered, cumvar_filtered, 'o-', color=palette['good'], 
                       label=f'After filtering ({good_data_std.shape[1]} features)', linewidth=2, markersize=4)
    
    axes[0, 0].set_xlabel('Number of Principal Components')
    axes[0, 0].set_ylabel('Cumulative Variance Explained')
    axes[0, 0].set_title('Cumulative Variance Explained by PCA')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Add lines at common variance thresholds
    for var_threshold in [0.8, 0.9, 0.95]:
        axes[0, 0].axhline(y=var_threshold, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].text(0.02, var_threshold + 0.01, f'{var_threshold:.0%}', transform=axes[0, 0].get_yaxis_transform())
    
    # 2. Individual PC variance explained
    n_show = min(20, len(pca_full.explained_variance_ratio_))
    x_pos = np.arange(1, n_show + 1)
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, pca_full.explained_variance_ratio_[:n_show], width,
                   label=f'Before filtering', color=palette['primary'], alpha=0.7)
    
    if pca_filtered is not None and len(pca_filtered.explained_variance_ratio_) > 0:
        n_show_filtered = min(n_show, len(pca_filtered.explained_variance_ratio_))
        axes[0, 1].bar(x_pos[:n_show_filtered] + width/2, pca_filtered.explained_variance_ratio_[:n_show_filtered], width,
                       label=f'After filtering', color=palette['good'], alpha=0.7)
    
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Variance Explained')
    axes[0, 1].set_title(f'Individual PC Variance Explained (First {n_show} PCs)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PCs needed for variance thresholds
    variance_thresholds = [0.8, 0.9, 0.95, 0.99]
    pcs_needed_full = []
    pcs_needed_filtered = []
    
    for threshold in variance_thresholds:
        # Full dataset
        pcs_full = np.where(cumvar_full >= threshold)[0]
        pcs_needed_full.append(pcs_full[0] + 1 if len(pcs_full) > 0 else len(cumvar_full))
        
        # Filtered dataset
        if pca_filtered is not None:
            pcs_filtered = np.where(cumvar_filtered >= threshold)[0]
            pcs_needed_filtered.append(pcs_filtered[0] + 1 if len(pcs_filtered) > 0 else len(cumvar_filtered))
        else:
            pcs_needed_filtered.append(0)
    
    x = np.arange(len(variance_thresholds))
    axes[1, 0].bar(x - width/2, pcs_needed_full, width, label='Before filtering', 
                   color=palette['primary'], alpha=0.7)
    if pca_filtered is not None:
        axes[1, 0].bar(x + width/2, pcs_needed_filtered, width, label='After filtering', 
                       color=palette['good'], alpha=0.7)
    
    axes[1, 0].set_xlabel('Variance Threshold')
    axes[1, 0].set_ylabel('PCs Needed')
    axes[1, 0].set_title('PCs Needed to Reach Variance Thresholds')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{t:.0%}' for t in variance_thresholds])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary metrics
    summary_text = f"""PCA Variance Analysis Summary:

Before Filtering:
• Total features: {full_data_std.shape[1]}
• PCs for 80% variance: {pcs_needed_full[0]}
• PCs for 90% variance: {pcs_needed_full[1]}
• PCs for 95% variance: {pcs_needed_full[2]}

After Filtering:
• Total features: {good_data_std.shape[1] if pca_filtered is not None else 0}
• PCs for 80% variance: {pcs_needed_filtered[0] if pca_filtered is not None else 'N/A'}
• PCs for 90% variance: {pcs_needed_filtered[1] if pca_filtered is not None else 'N/A'}
• PCs for 95% variance: {pcs_needed_filtered[2] if pca_filtered is not None else 'N/A'}

Noise Reduction:
• Features removed: {len(poor_feature_indices)} ({100*len(poor_feature_indices)/full_data_std.shape[1]:.1f}%)
• Features retained: {len(good_feature_indices)} ({100*len(good_feature_indices)/full_data_std.shape[1]:.1f}%)"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_qc_pca_variance_analysis.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Save detailed metrics
    pca_summary = {
        'total_features_before': full_data_std.shape[1],
        'total_features_after': good_data_std.shape[1] if pca_filtered is not None else 0,
        'features_removed': len(poor_feature_indices),
        'filtering_rate': len(poor_feature_indices) / full_data_std.shape[1],
        'pcs_for_80_var_before': pcs_needed_full[0],
        'pcs_for_90_var_before': pcs_needed_full[1],
        'pcs_for_95_var_before': pcs_needed_full[2],
        'pcs_for_80_var_after': pcs_needed_filtered[0] if pca_filtered is not None else None,
        'pcs_for_90_var_after': pcs_needed_filtered[1] if pca_filtered is not None else None,
        'pcs_for_95_var_after': pcs_needed_filtered[2] if pca_filtered is not None else None,
    }
    
    pca_summary_df = pd.DataFrame([pca_summary])
    pca_summary_df.to_csv(output_dir / 'feature_qc_pca_summary.csv', index=False)
    
    print("PCA variance analysis completed!")
    print(f"Summary: {len(poor_feature_indices)} features removed ({100*len(poor_feature_indices)/full_data_std.shape[1]:.1f}%)")
    if pca_filtered is not None:
        print(f"PCs needed for 90% variance: {pcs_needed_full[1]} → {pcs_needed_filtered[1]} (reduction: {pcs_needed_full[1] - pcs_needed_filtered[1]})")


def create_validation_analysis(results_df, reconstructions, full_data, original_data, output_dir):
    """Create validation analysis to prove feature QC is working."""
    print("\n" + "="*60)
    print("CREATING VALIDATION ANALYSIS")
    print("="*60)
    
    palette = QC_CONFIG['visualization']['palette']
    
    # Filter out poor quality features
    good_features = results_df[results_df['quality_category'] == 'Good Quality']
    poor_features = results_df[results_df['quality_category'] == 'Poor Quality']
    
    good_feature_indices = [original_data.columns.get_loc(name) for name in good_features['feature_name']]
    poor_feature_indices = [original_data.columns.get_loc(name) for name in poor_features['feature_name']]
    
    # 1. Overall reconstruction quality comparison
    # Before filtering (all features)
    overall_mse_before = np.mean((reconstructions - full_data) ** 2)
    overall_corr_before = np.corrcoef(full_data.flatten(), reconstructions.flatten())[0, 1]
    
    # After filtering (only good features)
    if len(good_feature_indices) > 0:
        good_data = full_data[:, good_feature_indices]
        good_recon = reconstructions[:, good_feature_indices]
        overall_mse_after = np.mean((good_recon - good_data) ** 2)
        overall_corr_after = np.corrcoef(good_data.flatten(), good_recon.flatten())[0, 1]
    else:
        overall_mse_after = np.inf
        overall_corr_after = 0
    
    # Get QC metric information
    qc_metric = results_df['selected_qc_metric_name'].iloc[0]
    qc_column = 'selected_qc_metric'
    
    # Create readable metric names for labels
    metric_labels = {
        'correlation': 'Pearson Correlation',
        'normalized_rmse': 'Normalized RMSE',
        'nash_sutcliffe': 'Nash-Sutcliffe Efficiency', 
        'normalized_mse': 'Normalized MSE',
        'r2_baseline': 'R² vs Baseline'
    }
    metric_label = metric_labels.get(qc_metric, qc_metric.title())
    
    # 2. Feature-wise metrics comparison - use selected QC metric
    good_qc_values = good_features[qc_column].values
    poor_qc_values = poor_features[qc_column].values
    
    # Still include correlation for comparison
    good_correlations = good_features['correlation'].values
    poor_correlations = poor_features['correlation'].values
    
    good_r2_scores = good_features['r2_score'].values
    poor_r2_scores = poor_features['r2_score'].values
    
    good_recon_errors = good_features['reconstruction_error'].values
    poor_recon_errors = poor_features['reconstruction_error'].values
    
    # Create validation plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Overall metrics comparison
    metrics_comparison = pd.DataFrame({
        'Metric': ['MSE', 'Correlation'],
        'Before Filtering': [overall_mse_before, overall_corr_before],
        'After Filtering': [overall_mse_after, overall_corr_after]
    })
    
    x = np.arange(len(metrics_comparison))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, metrics_comparison['Before Filtering'], width, 
                   label='Before Filtering', color=palette['primary'])
    axes[0, 0].bar(x + width/2, metrics_comparison['After Filtering'], width, 
                   label='After Filtering', color=palette['good'])
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Overall Reconstruction Quality')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_comparison['Metric'])
    axes[0, 0].legend()
    
    # Selected QC metric distribution comparison
    if len(poor_qc_values) > 0:
        axes[0, 1].hist(poor_qc_values, bins=20, alpha=0.7, color=palette['error'], 
                       label=f'Poor Quality (n={len(poor_qc_values)})')
    if len(good_qc_values) > 0:
        axes[0, 1].hist(good_qc_values, bins=20, alpha=0.7, color=palette['good'], 
                       label=f'Good Quality (n={len(good_qc_values)})')
    
    # Add QC threshold line
    qc_threshold = results_df['qc_threshold'].iloc[0]
    axes[0, 1].axvline(qc_threshold, color=palette['warning'], linestyle='--', 
                      label=f'{metric_label} Threshold ({qc_threshold:.3f})')
    
    axes[0, 1].set_xlabel(metric_label)
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{metric_label} Distribution')
    axes[0, 1].legend()
    
    # Reconstruction error distribution comparison
    if len(poor_recon_errors) > 0:
        axes[0, 2].hist(poor_recon_errors, bins=20, alpha=0.7, color=palette['error'], 
                       label=f'Poor Quality (n={len(poor_recon_errors)})', log=True)
    if len(good_recon_errors) > 0:
        axes[0, 2].hist(good_recon_errors, bins=20, alpha=0.7, color=palette['good'], 
                       label=f'Good Quality (n={len(good_recon_errors)})', log=True)
    axes[0, 2].set_xlabel('Reconstruction Error')
    axes[0, 2].set_ylabel('Frequency (Log Scale)')
    axes[0, 2].set_title('Feature Reconstruction Error Distribution')
    axes[0, 2].legend()
    
    # Box plots for selected QC metric comparison
    qc_data = []
    qc_labels = []
    if len(poor_qc_values) > 0:
        qc_data.append(poor_qc_values)
        qc_labels.append('Poor Quality')
    if len(good_qc_values) > 0:
        qc_data.append(good_qc_values)
        qc_labels.append('Good Quality')
    
    if qc_data:
        bp1 = axes[1, 0].boxplot(qc_data, labels=qc_labels, patch_artist=True)
        colors = [palette['error'] if 'Poor' in label else palette['good'] for label in qc_labels]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 0].axhline(qc_threshold, color=palette['warning'], linestyle='--', 
                          label=f'{metric_label} Threshold ({qc_threshold:.3f})')
        axes[1, 0].set_ylabel(metric_label)
        axes[1, 0].set_title(f'{metric_label} Distribution by Quality')
        axes[1, 0].legend()
    
    # R² box plots
    r2_data = []
    r2_labels = []
    if len(poor_r2_scores) > 0:
        r2_data.append(poor_r2_scores)
        r2_labels.append('Poor Quality')
    if len(good_r2_scores) > 0:
        r2_data.append(good_r2_scores)
        r2_labels.append('Good Quality')
    
    if r2_data:
        bp2 = axes[1, 1].boxplot(r2_data, labels=r2_labels, patch_artist=True)
        colors = [palette['error'] if 'Poor' in label else palette['good'] for label in r2_labels]
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('R² Score Distribution by Quality')
    
    # Feature count summary
    count_data = [len(poor_features), len(good_features)]
    count_labels = ['Poor Quality', 'Good Quality']
    colors = [palette['error'], palette['good']]
    
    axes[1, 2].bar(count_labels, count_data, color=colors)
    axes[1, 2].set_ylabel('Number of Features')
    axes[1, 2].set_title('Feature Count by Quality Category')
    
    # Add percentage labels
    total_features = len(poor_features) + len(good_features)
    for i, (count, label) in enumerate(zip(count_data, count_labels)):
        percentage = 100 * count / total_features
        axes[1, 2].text(i, count + max(count_data) * 0.01, f'{percentage:.1f}%', 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_qc_validation.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    validation_summary = {
        'total_features': len(results_df),
        'good_features': len(good_features),
        'poor_features': len(poor_features),
        'filtering_rate': len(poor_features) / len(results_df),
        'qc_metric': qc_metric,
        'qc_threshold': qc_threshold,
        'overall_mse_before': overall_mse_before,
        'overall_mse_after': overall_mse_after,
        'overall_corr_before': overall_corr_before,
        'overall_corr_after': overall_corr_after,
        f'good_features_mean_{qc_metric}': good_qc_values.mean() if len(good_qc_values) > 0 else 0,
        f'poor_features_mean_{qc_metric}': poor_qc_values.mean() if len(poor_qc_values) > 0 else 0,
        'good_features_mean_corr': good_correlations.mean() if len(good_correlations) > 0 else 0,
        'poor_features_mean_corr': poor_correlations.mean() if len(poor_correlations) > 0 else 0,
        'good_features_mean_r2': good_r2_scores.mean() if len(good_r2_scores) > 0 else 0,
        'poor_features_mean_r2': poor_r2_scores.mean() if len(poor_r2_scores) > 0 else 0,
        'good_features_mean_mse': good_recon_errors.mean() if len(good_recon_errors) > 0 else 0,
        'poor_features_mean_mse': poor_recon_errors.mean() if len(poor_recon_errors) > 0 else 0,
    }
    
    # Save validation summary
    validation_df = pd.DataFrame([validation_summary])
    validation_df.to_csv(output_dir / 'feature_qc_validation_summary.csv', index=False)
    
    print("VALIDATION ANALYSIS SUMMARY:")
    print("="*40)
    print(f"Total features: {validation_summary['total_features']}")
    print(f"Good quality features: {validation_summary['good_features']} ({100*(1-validation_summary['filtering_rate']):.1f}%)")
    print(f"Poor quality features: {validation_summary['poor_features']} ({100*validation_summary['filtering_rate']:.1f}%)")
    print(f"QC metric: {validation_summary['qc_metric']}")
    print(f"QC threshold: {validation_summary['qc_threshold']}")
    print(f"Overall MSE: {overall_mse_before:.6f} -> {overall_mse_after:.6f}")
    print(f"Overall correlation: {overall_corr_before:.4f} -> {overall_corr_after:.4f}")
    print(f"Good features mean {qc_metric}: {validation_summary[f'good_features_mean_{qc_metric}']:.4f}")
    print(f"Poor features mean {qc_metric}: {validation_summary[f'poor_features_mean_{qc_metric}']:.4f}")
    print(f"Good features mean correlation: {validation_summary['good_features_mean_corr']:.4f}")
    print(f"Poor features mean correlation: {validation_summary['poor_features_mean_corr']:.4f}")
    
    print("Validation analysis completed!")


def load_and_validate_annotations(annotation_file, feature_names, categorical_cols=None, continuous_cols=None):
    """Load and validate feature annotation file."""
    print(f"\nLoading annotation file: {annotation_file}")
    
    try:
        # Load annotation file (tab-delimited)
        annotations_df = pd.read_csv(annotation_file, sep='\t')
        print(f"Annotation file loaded: {annotations_df.shape}")
        
        # Check if first column contains feature names
        first_col = annotations_df.columns[0]
        # Strip whitespace from annotation values
        annotation_features = set(str(x).strip() for x in annotations_df[first_col].values if pd.notna(x))
        data_features = set(feature_names)
        
        # Find intersection with primary column
        common_features = annotation_features.intersection(data_features)
        missing_in_annot = data_features - annotation_features
        
        # Check secondary index if available and if there are missing features
        secondary_common = set()
        if len(missing_in_annot) > 0 and 'entrezgenesymbol' in annotations_df.columns:
            print(f"Looking for {len(missing_in_annot)} missing features in 'entrezgenesymbol' column...")
            # Strip whitespace from secondary column values
            secondary_features = set(str(x).strip() for x in annotations_df['entrezgenesymbol'].values if pd.notna(x))
            secondary_common = missing_in_annot.intersection(secondary_features)
            print(f"  - Found {len(secondary_common)} additional matches in 'entrezgenesymbol'")
        
        # Combine primary and secondary matches
        all_common_features = common_features.union(secondary_common)
        missing_in_data = annotation_features - data_features
        
        print(f"Feature matching summary:")
        print(f"  - Features in data: {len(data_features)}")
        print(f"  - Features in annotations (primary): {len(annotation_features)}")
        print(f"  - Common features (primary): {len(common_features)}")
        if secondary_common:
            print(f"  - Common features (secondary): {len(secondary_common)}")
        print(f"  - Total common features: {len(all_common_features)}")
        print(f"  - Features in data but not in annotations: {len(data_features - all_common_features)}")
        print(f"  - Features in annotations but not in data: {len(missing_in_data)}")
        
        if len(all_common_features) == 0:
            print("WARNING: No common features found between data and annotations!")
            return None, None, None
        
        # Create unified annotations dataframe with proper indexing
        if len(secondary_common) > 0:
            # Create separate dataframes for primary and secondary matches
            # Strip whitespace in the filtering conditions
            primary_mask = annotations_df[first_col].astype(str).str.strip().isin(common_features)
            secondary_mask = annotations_df['entrezgenesymbol'].astype(str).str.strip().isin(secondary_common)
            
            primary_df = annotations_df[primary_mask].copy()
            primary_df[first_col] = primary_df[first_col].astype(str).str.strip()
            primary_df = primary_df.set_index(first_col)
            
            secondary_df = annotations_df[secondary_mask].copy()
            secondary_df['entrezgenesymbol'] = secondary_df['entrezgenesymbol'].astype(str).str.strip()
            secondary_df = secondary_df.set_index('entrezgenesymbol')
            
            # Combine them
            annotations_df = pd.concat([primary_df, secondary_df])
        else:
            # Set feature names as index for easier lookup (original behavior)
            annotations_df[first_col] = annotations_df[first_col].astype(str).str.strip()
            annotations_df = annotations_df.set_index(first_col)
        
        # Validate categorical and continuous columns
        available_cols = list(annotations_df.columns)
        
        valid_categorical = []
        if categorical_cols:
            for col in categorical_cols:
                if col in available_cols:
                    valid_categorical.append(col)
                    print(f"  - Found categorical annotation: {col}")
                else:
                    print(f"WARNING: Categorical column '{col}' not found in annotation file")
        
        valid_continuous = []
        if continuous_cols:
            for col in continuous_cols:
                if col in available_cols:
                    valid_continuous.append(col)
                    print(f"  - Found continuous annotation: {col}")
                else:
                    print(f"WARNING: Continuous column '{col}' not found in annotation file")
        
        print(f"Valid annotation columns: {len(valid_categorical)} categorical, {len(valid_continuous)} continuous")
        
        return annotations_df, valid_categorical, valid_continuous
        
    except Exception as e:
        print(f"ERROR loading annotation file: {e}")
        return None, None, None


def create_annotation_analysis_plots(results_df, annotations_df, categorical_cols, continuous_cols, output_dir):
    """Create annotation analysis plots."""
    print("\n" + "="*60)
    print("CREATING ANNOTATION ANALYSIS PLOTS")
    print("="*60)
    
    if annotations_df is None:
        print("No valid annotations found. Skipping annotation analysis.")
        return
    
    palette = QC_CONFIG['visualization']['palette']
    
    # Merge results with annotations
    results_with_annot = results_df.set_index('feature_name').join(annotations_df, how='inner')
    
    if len(results_with_annot) == 0:
        print("No features with annotations found. Skipping annotation analysis.")
        return
    
    print(f"Creating annotation plots for {len(results_with_annot)} features with annotations...")
    
    # Create plots for categorical annotations
    if categorical_cols:
        for col in categorical_cols:
            create_categorical_annotation_plots(results_with_annot, col, output_dir, palette)
    
    # Create plots for continuous annotations
    if continuous_cols:
        for col in continuous_cols:
            create_continuous_annotation_plots(results_with_annot, col, output_dir, palette)
    
    # Create summary correlation matrix if we have multiple annotations
    all_annot_cols = (categorical_cols or []) + (continuous_cols or [])
    if len(all_annot_cols) > 1:
        create_annotation_correlation_matrix(results_with_annot, all_annot_cols, categorical_cols or [], output_dir, palette)
    
    print("Annotation analysis plots completed!")


def create_categorical_annotation_plots(results_with_annot, annotation_col, output_dir, palette):
    """Create violin plots for categorical annotations."""
    print(f"Creating categorical annotation plots for: {annotation_col}")
    
    # Filter out missing values
    data_subset = results_with_annot.dropna(subset=[annotation_col])
    
    if len(data_subset) == 0:
        print(f"No valid data for annotation column: {annotation_col}")
        return
    
    # Get unique categories
    categories = data_subset[annotation_col].unique()
    print(f"  - Found {len(categories)} categories: {categories}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Violin plot for reconstruction error
    sns.violinplot(data=data_subset, x=annotation_col, y='reconstruction_error', ax=axes[0, 0])
    axes[0, 0].set_title(f'Reconstruction Error by {annotation_col}')
    axes[0, 0].set_ylabel('Reconstruction Error (Log Scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Violin plot for selected QC metric
    qc_metric = results_with_annot['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in results_with_annot.columns else 'correlation'
    qc_threshold = results_with_annot['qc_threshold'].iloc[0] if 'qc_threshold' in results_with_annot.columns else QC_CONFIG['feature_qc']['correlation_threshold']
    qc_column = 'selected_qc_metric' if 'selected_qc_metric' in results_with_annot.columns else 'correlation'
    
    # Create readable metric names for labels
    metric_labels = {
        'correlation': 'Pearson Correlation',
        'normalized_rmse': 'Normalized RMSE',
        'nash_sutcliffe': 'Nash-Sutcliffe Efficiency', 
        'normalized_mse': 'Normalized MSE',
        'r2_baseline': 'R² vs Baseline'
    }
    metric_label = metric_labels.get(qc_metric, qc_metric.title())
    
    sns.violinplot(data=data_subset, x=annotation_col, y=qc_column, ax=axes[0, 1])
    axes[0, 1].set_title(f'{metric_label} by {annotation_col}')
    axes[0, 1].set_ylabel(metric_label)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add QC threshold line
    axes[0, 1].axhline(y=qc_threshold, color=palette['error'], linestyle='--', 
                      label=f'{metric_label} Threshold ({qc_threshold:.3f})')
    axes[0, 1].legend()
    
    # Violin plot for R² score
    sns.violinplot(data=data_subset, x=annotation_col, y='r2_score', ax=axes[1, 0])
    axes[1, 0].set_title(f'R² Score by {annotation_col}')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Quality distribution by category
    quality_counts = data_subset.groupby([annotation_col, 'quality_category']).size().unstack(fill_value=0)
    quality_counts_pct = quality_counts.div(quality_counts.sum(axis=1), axis=0) * 100
    
    quality_counts_pct.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                           color=[palette['good'], palette['error']])
    axes[1, 1].set_title(f'Quality Distribution by {annotation_col}')
    axes[1, 1].set_ylabel('Percentage of Features (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Quality Category')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'annotation_categorical_{annotation_col}.png', 
                dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = data_subset.groupby(annotation_col).agg({
        'reconstruction_error': ['mean', 'std', 'median'],
        'correlation': ['mean', 'std', 'median'],
        'r2_score': ['mean', 'std', 'median'],
        'is_poor_quality': ['sum', 'count']
    }).round(4)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats['poor_quality_rate'] = summary_stats['is_poor_quality_sum'] / summary_stats['is_poor_quality_count']
    
    summary_stats.to_csv(output_dir / f'annotation_categorical_{annotation_col}_summary.csv')
    print(f"  - Saved categorical summary: annotation_categorical_{annotation_col}_summary.csv")


def create_continuous_annotation_plots(results_with_annot, annotation_col, output_dir, palette):
    """Create scatter plots for continuous annotations."""
    print(f"Creating continuous annotation plots for: {annotation_col}")
    
    # Filter out missing values and convert to numeric
    data_subset = results_with_annot.dropna(subset=[annotation_col]).copy()
    
    # Try to convert to numeric
    try:
        data_subset[annotation_col] = pd.to_numeric(data_subset[annotation_col])
    except:
        print(f"WARNING: Could not convert {annotation_col} to numeric. Treating as categorical.")
        create_categorical_annotation_plots(results_with_annot, annotation_col, output_dir, palette)
        return
    
    if len(data_subset) == 0:
        print(f"No valid data for annotation column: {annotation_col}")
        return
    
    print(f"  - Found {len(data_subset)} features with numeric values")
    print(f"  - Value range: {data_subset[annotation_col].min():.3f} to {data_subset[annotation_col].max():.3f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scatter plot for reconstruction error
    good_mask = data_subset['quality_category'] == 'Good Quality'
    
    axes[0, 0].scatter(data_subset[~good_mask][annotation_col], 
                      data_subset[~good_mask]['reconstruction_error'],
                      c=palette['error'], alpha=0.6, s=20, label='Poor Quality')
    axes[0, 0].scatter(data_subset[good_mask][annotation_col], 
                      data_subset[good_mask]['reconstruction_error'],
                      c=palette['good'], alpha=0.6, s=20, label='Good Quality')
    
    axes[0, 0].set_xlabel(annotation_col)
    axes[0, 0].set_ylabel('Reconstruction Error (Log Scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(f'Reconstruction Error vs {annotation_col}')
    axes[0, 0].legend()
    
    # Calculate and display correlation
    corr_recon = data_subset[annotation_col].corr(data_subset['reconstruction_error'])
    spearman_recon = data_subset[annotation_col].corr(data_subset['reconstruction_error'], method='spearman')
    axes[0, 0].text(0.05, 0.95, f'Pearson r: {corr_recon:.3f}\nSpearman ρ: {spearman_recon:.3f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Scatter plot for selected QC metric
    qc_metric = results_with_annot['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in results_with_annot.columns else 'correlation'
    qc_column = 'selected_qc_metric' if 'selected_qc_metric' in results_with_annot.columns else 'correlation'
    
    # Create readable metric names for labels
    metric_labels = {
        'correlation': 'Pearson Correlation',
        'normalized_rmse': 'Normalized RMSE',
        'nash_sutcliffe': 'Nash-Sutcliffe Efficiency', 
        'normalized_mse': 'Normalized MSE',
        'r2_baseline': 'R² vs Baseline'
    }
    metric_label = metric_labels.get(qc_metric, qc_metric.title())
    
    axes[0, 1].scatter(data_subset[~good_mask][annotation_col], 
                      data_subset[~good_mask][qc_column],
                      c=palette['error'], alpha=0.6, s=20, label='Poor Quality')
    axes[0, 1].scatter(data_subset[good_mask][annotation_col], 
                      data_subset[good_mask][qc_column],
                      c=palette['good'], alpha=0.6, s=20, label='Good Quality')
    
    axes[0, 1].set_xlabel(annotation_col)
    axes[0, 1].set_ylabel(metric_label)
    axes[0, 1].set_title(f'{metric_label} vs {annotation_col}')
    axes[0, 1].legend()
    
    # Add QC threshold line
    qc_threshold = results_with_annot['qc_threshold'].iloc[0] if 'qc_threshold' in results_with_annot.columns else QC_CONFIG['feature_qc']['correlation_threshold']
    axes[0, 1].axhline(y=qc_threshold, color=palette['error'], linestyle='--', 
                      label=f'{metric_label} Threshold ({qc_threshold:.3f})')
    
    # Calculate and display correlation with selected QC metric
    corr_qc = data_subset[annotation_col].corr(data_subset[qc_column])
    spearman_qc = data_subset[annotation_col].corr(data_subset[qc_column], method='spearman')
    axes[0, 1].text(0.05, 0.95, f'Pearson r: {corr_qc:.3f}\nSpearman ρ: {spearman_qc:.3f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Scatter plot for R² score
    axes[1, 0].scatter(data_subset[~good_mask][annotation_col], 
                      data_subset[~good_mask]['r2_score'],
                      c=palette['error'], alpha=0.6, s=20, label='Poor Quality')
    axes[1, 0].scatter(data_subset[good_mask][annotation_col], 
                      data_subset[good_mask]['r2_score'],
                      c=palette['good'], alpha=0.6, s=20, label='Good Quality')
    
    axes[1, 0].set_xlabel(annotation_col)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title(f'R² Score vs {annotation_col}')
    axes[1, 0].legend()
    
    # Calculate and display correlation
    corr_r2 = data_subset[annotation_col].corr(data_subset['r2_score'])
    spearman_r2 = data_subset[annotation_col].corr(data_subset['r2_score'], method='spearman')
    axes[1, 0].text(0.05, 0.95, f'Pearson r: {corr_r2:.3f}\nSpearman ρ: {spearman_r2:.3f}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Binned quality distribution
    # Create bins for the continuous variable
    n_bins = min(10, len(data_subset[annotation_col].unique()))
    data_subset['binned_annot'] = pd.cut(data_subset[annotation_col], bins=n_bins, duplicates='drop')
    
    if data_subset['binned_annot'].notna().sum() > 0:
        quality_by_bin = data_subset.groupby('binned_annot')['is_poor_quality'].agg(['sum', 'count'])
        quality_by_bin['poor_rate'] = quality_by_bin['sum'] / quality_by_bin['count']
        
        bin_centers = [interval.mid for interval in quality_by_bin.index]
        axes[1, 1].bar(range(len(bin_centers)), quality_by_bin['poor_rate'] * 100, 
                      color=palette['primary'], alpha=0.7)
        axes[1, 1].set_xlabel(f'{annotation_col} (Binned)')
        axes[1, 1].set_ylabel('Poor Quality Rate (%)')
        axes[1, 1].set_title(f'Poor Quality Rate by {annotation_col} Bins')
        axes[1, 1].set_xticks(range(len(bin_centers)))
        axes[1, 1].set_xticklabels([f'{x:.2f}' for x in bin_centers], rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Unable to create bins\n(insufficient data)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'Poor Quality Rate by {annotation_col} Bins')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'annotation_continuous_{annotation_col}.png', 
                dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Save correlation summary
    correlation_summary = {
        'annotation_column': annotation_col,
        'qc_metric': qc_metric,
        'n_features': len(data_subset),
        'annotation_range': [data_subset[annotation_col].min(), data_subset[annotation_col].max()],
        'reconstruction_error_pearson': corr_recon,
        'reconstruction_error_spearman': spearman_recon,
        f'{qc_metric}_pearson': corr_qc,
        f'{qc_metric}_spearman': spearman_qc,
        'r2_score_pearson': corr_r2,
        'r2_score_spearman': spearman_r2,
    }
    
    correlation_df = pd.DataFrame([correlation_summary])
    correlation_df.to_csv(output_dir / f'annotation_continuous_{annotation_col}_correlations.csv', index=False)
    print(f"  - Saved correlation summary: annotation_continuous_{annotation_col}_correlations.csv")


def create_annotation_correlation_matrix(results_with_annot, annotation_cols, categorical_cols, output_dir, palette):
    """Create correlation matrix between quality metrics and annotations."""
    print("Creating annotation correlation matrix...")
    
    # Prepare data for correlation analysis
    corr_data = results_with_annot[['reconstruction_error', 'correlation', 'r2_score'] + annotation_cols].copy()
    
    # Convert categorical variables to numeric (if possible) or use label encoding
    for col in categorical_cols:
        if col in corr_data.columns:
            le = LabelEncoder()
            # Fill missing values with a placeholder before encoding
            corr_data[col] = corr_data[col].fillna('Unknown')
            corr_data[f'{col}_encoded'] = le.fit_transform(corr_data[col])
            corr_data = corr_data.drop(columns=[col])
    
    # Calculate Pearson and Spearman correlations
    pearson_corr = corr_data.corr(method='pearson')
    spearman_corr = corr_data.corr(method='spearman')
    
    # Create correlation plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0], cbar_kws={'label': 'Pearson Correlation'})
    axes[0].set_title('Pearson Correlation Matrix')
    
    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1], cbar_kws={'label': 'Spearman Correlation'})
    axes[1].set_title('Spearman Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'annotation_correlation_matrix.png', 
                dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Save correlation matrices
    pearson_corr.to_csv(output_dir / 'annotation_pearson_correlations.csv')
    spearman_corr.to_csv(output_dir / 'annotation_spearman_correlations.csv')
    
    print("  - Saved correlation matrices: annotation_pearson_correlations.csv, annotation_spearman_correlations.csv")


def main():
    """Main feature QC analysis function."""
    args = parse_arguments()
    
    print("="*80)
    print("FEATURE QUALITY CONTROL ANALYSIS USING QC VAE")
    print("="*80)
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Set up data module
    print("\nSetting up data module...")
    datamodule = SingleDataModule(
        config=QC_CONFIG,
        data_file=args.data_file
    )
    
    # Setup data
    datamodule.setup()
    
    # Train QC VAE
    model, exp_dir = train_qc_vae(
        datamodule=datamodule,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        fast_dev_run=args.fast_dev_run
    )
    
    if not args.fast_dev_run:
        # Perform feature QC analysis
        results_df, reconstructions, full_data = perform_feature_qc_analysis(
            model=model,
            datamodule=datamodule,
            output_dir=exp_dir
        )
        
        # Create feature QC plots
        create_feature_qc_plots(
            results_df=results_df,
            reconstructions=reconstructions,
            full_data=full_data,
            original_data=datamodule.original_data,
            output_dir=exp_dir
        )
        
        # Create validation analysis
        create_validation_analysis(
            results_df=results_df,
            reconstructions=reconstructions,
            full_data=full_data,
            original_data=datamodule.original_data,
            output_dir=exp_dir
        )
        
        # Create PCA variance analysis
        create_pca_variance_analysis(
            results_df=results_df,
            full_data_original=full_data,
            original_data=datamodule.original_data,
            output_dir=exp_dir
        )
        
        # Annotation analysis (if provided)
        annotations_df, valid_categorical, valid_continuous = None, None, None
        if args.feature_annot:
            annotations_df, valid_categorical, valid_continuous = load_and_validate_annotations(
                annotation_file=args.feature_annot,
                feature_names=datamodule.original_data.columns.tolist(),
                categorical_cols=args.categorical_annot,
                continuous_cols=args.continuous_annot
            )
            
            if annotations_df is not None:
                create_annotation_analysis_plots(
                    results_df=results_df,
                    annotations_df=annotations_df,
                    categorical_cols=valid_categorical,
                    continuous_cols=valid_continuous,
                    output_dir=exp_dir
                )
        
        # Biological analysis (if requested and available)
        biological_results = None
        if (args.run_biological_analysis or args.run_enrichment or args.run_network) and BIOLOGICAL_ANALYSIS_AVAILABLE:
            try:
                biological_results = run_biological_analysis(
                    results_df=results_df,
                    output_dir=exp_dir,
                    run_enrichment=args.run_biological_analysis or args.run_enrichment,
                    run_network=args.run_biological_analysis or args.run_network,
                    organism=args.organism,
                    decile_cutoff=args.decile_cutoff,
                    network_top_n=args.network_top_n,
                    confidence_threshold=args.confidence_threshold
                )
            except Exception as e:
                print(f"WARNING: Biological analysis failed: {e}")
                print("This may be due to missing dependencies (gseapy, networkx, plotly) or API issues.")
        elif (args.run_biological_analysis or args.run_enrichment or args.run_network) and not BIOLOGICAL_ANALYSIS_AVAILABLE:
            print("WARNING: Biological analysis requested but helper module not available.")
            print("Make sure qc_feature_helper.py is in the same directory and dependencies are installed.")
        
        # Classification analysis (if phenotype data provided)
        classification_results = None
        if args.sample_pheno is not None and args.binary_pheno is not None:
            if BIOLOGICAL_ANALYSIS_AVAILABLE:
                try:
                    print("\nRunning classification analysis to validate QC efficacy...")
                    classification_results = run_classification_analysis(
                        results_df=results_df,
                        original_data=datamodule.original_data,
                        pheno_file=args.sample_pheno,
                        binary_pheno_cols=args.binary_pheno,
                        annotations_df=annotations_df,
                        classic_qc_flag=args.classic_qc_flag,
                        output_dir=exp_dir
                    )
                except Exception as e:
                    print(f"WARNING: Classification analysis failed: {e}")
                    print("This may be due to missing dependencies (scikit-learn, scipy) or data alignment issues.")
            else:
                print("WARNING: Classification analysis requested but helper module not available.")
        elif args.sample_pheno is not None and args.binary_pheno is None:
            print("WARNING: Phenotype file provided but no binary phenotype columns specified.")
            print("Use --binary_pheno to specify column names for binary classification.")
        elif args.binary_pheno is not None and args.sample_pheno is None:
            print("WARNING: Binary phenotype columns specified but no phenotype file provided.")
            print("Use --sample_pheno to specify the phenotype file path.")
        
        # Save filtered data (poor quality features removed)
        poor_feature_names = results_df[results_df['is_poor_quality']]['feature_name'].tolist()
        good_feature_names = results_df[~results_df['is_poor_quality']]['feature_name'].tolist()
        
        filtered_data = datamodule.original_data[good_feature_names]
        filtered_path = exp_dir / 'filtered_data_feature_qc_passed.csv'
        filtered_data.to_csv(filtered_path)
        
        # Save poor quality features list for inspection
        poor_features_df = results_df[results_df['is_poor_quality']]
        poor_features_path = exp_dir / 'poor_quality_features.csv'
        poor_features_df.to_csv(poor_features_path, index=False)
        
        # Final summary
        print("\n" + "="*80)
        print("FEATURE QC ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {exp_dir}")
        print(f"Feature QC results CSV: {exp_dir / 'feature_qc_results.csv'}")
        print(f"Validation summary: {exp_dir / 'feature_qc_validation_summary.csv'}")
        print(f"PCA variance analysis: {exp_dir / 'feature_qc_pca_summary.csv'}")
        print(f"Visualization plots:")
        print(f"  - Joint distribution: {exp_dir / 'feature_qc_joint_distribution.png'}")
        print(f"  - Overview plots: {exp_dir / 'feature_qc_overview.png'}")
        print(f"  - Validation plots: {exp_dir / 'feature_qc_validation.png'}")
        print(f"  - PCA variance analysis: {exp_dir / 'feature_qc_pca_variance_analysis.png'}")
        print(f"  - Feature examples: {exp_dir / 'feature_comparison_examples.png'}")
        print(f"  - Feature profiles: {exp_dir / 'feature_profile_examples.png'}")
        print(f"  - Feature profiles (ranked): {exp_dir / 'feature_profile_examples_ranked.png'}")
        print(f"Filtered data (good features): {filtered_path}")
        print(f"Poor quality features list: {poor_features_path}")
        
        # Print annotation analysis summary if performed
        if args.feature_annot and annotations_df is not None:
            print(f"\nANNOTATION ANALYSIS:")
            print(f"Annotation file: {args.feature_annot}")
            if valid_categorical:
                print(f"Categorical annotations analyzed: {', '.join(valid_categorical)}")
                for col in valid_categorical:
                    print(f"  - Plots saved: annotation_categorical_{col}.png")
                    print(f"  - Summary saved: annotation_categorical_{col}_summary.csv")
            if valid_continuous:
                print(f"Continuous annotations analyzed: {', '.join(valid_continuous)}")
                for col in valid_continuous:
                    print(f"  - Plots saved: annotation_continuous_{col}.png")
                    print(f"  - Correlations saved: annotation_continuous_{col}_correlations.csv")
            if len((valid_categorical or []) + (valid_continuous or [])) > 1:
                print(f"  - Correlation matrix saved: annotation_correlation_matrix.png")
        
        # Print biological analysis summary if performed
        if biological_results is not None:
            print(f"\nBIOLOGICAL VALIDATION ANALYSIS:")
            print(f"Organism: {args.organism}")
            bio_dir = exp_dir / "biological_analysis"
            
            if 'enrichment' in biological_results and biological_results['enrichment']:
                print(f"Pathway Enrichment Analysis:")
                enrichment_results = biological_results['enrichment']
                total_databases = len(enrichment_results)
                total_worst_terms = sum(len(db_results['worst']) if db_results['worst'] is not None else 0 
                                      for db_results in enrichment_results.values())
                total_best_terms = sum(len(db_results['best']) if db_results['best'] is not None else 0 
                                     for db_results in enrichment_results.values())
                print(f"  - Databases analyzed: {total_databases}")
                print(f"  - Enriched terms (worst features): {total_worst_terms}")
                print(f"  - Enriched terms (best features): {total_best_terms}")
                print(f"  - Results saved to: {bio_dir}/enrichment_*")
            
            if 'network' in biological_results and biological_results['network']:
                print(f"Protein Interaction Network Analysis:")
                network_results = biological_results['network']
                if 'network_stats' in network_results:
                    stats = network_results['network_stats']
                    print(f"  - Network nodes: {stats['num_nodes']}")
                    print(f"  - Network edges: {stats['num_edges']}")
                    print(f"  - Network density: {stats['density']:.4f}")
                print(f"  - Results saved to: {bio_dir}/string_*")
            
            print(f"  - Analysis summary: {bio_dir}/biological_analysis_summary.md")
        
        # Print classification analysis summary if performed
        if classification_results is not None:
            print(f"\nCLASSIFICATION VALIDATION ANALYSIS:")
            print(f"Phenotype file: {args.sample_pheno}")
            print(f"Binary phenotypes analyzed: {', '.join(args.binary_pheno)}")
            if args.classic_qc_flag:
                print(f"Classic QC flag: {args.classic_qc_flag}")
            
            class_dir = exp_dir / "classification_analysis"
            print(f"Feature sets compared:")
            
            # Count total comparisons
            total_phenotypes = len(args.binary_pheno)
            total_feature_sets = 0
            
            if classification_results:
                sample_pheno = list(classification_results.keys())[0]
                if sample_pheno in classification_results:
                    total_feature_sets = len(classification_results[sample_pheno])
            
            print(f"  - Unfiltered features (all original features)")
            print(f"  - VAE good features (passed VAE QC)")
            print(f"  - VAE bad features (failed VAE QC)")
            if args.classic_qc_flag:
                print(f"  - Classic good features (passed classic QC)")
                print(f"  - Classic bad features (failed classic QC)")
            
            print(f"Analysis summary:")
            print(f"  - Phenotypes tested: {total_phenotypes}")
            print(f"  - Feature sets compared: {total_feature_sets}")
            print(f"  - Classifiers used: LogisticRegression")
            print(f"  - Results saved to: {class_dir}/")
            print(f"  - Summary plots: {class_dir}/classification_summary.png")
            print(f"  - Statistical tests: {class_dir}/statistical_tests.csv")
            print(f"  - Detailed results: {class_dir}/classification_results.json")
        
        # Print final feature summary
        n_total = len(results_df)
        n_poor = np.sum(results_df['is_poor_quality'])
        n_good = n_total - n_poor
        selected_qc_metric = QC_CONFIG['feature_qc']['qc_metric']
        selected_threshold = QC_CONFIG['feature_qc']['thresholds'][selected_qc_metric]
        print(f"\nFINAL FEATURE QC SUMMARY:")
        print(f"QC metric used: {selected_qc_metric}")
        print(f"QC threshold used: {selected_threshold}")
        print(f"Total features analyzed: {n_total}")
        print(f"Poor quality features filtered: {n_poor} ({100*n_poor/n_total:.1f}%)")
        print(f"Good quality features retained: {n_good} ({100*n_good/n_total:.1f}%)")
        print(f"Filtered dataset shape: {filtered_data.shape}")
        print(f"Reconstruction errors computed on original data scale")
    
    print("\nFeature QC analysis pipeline completed!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(QC_CONFIG['data']['random_seed'], workers=True)
    
    # Run main function
    main()