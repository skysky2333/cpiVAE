#!/usr/bin/env python3
"""
Quality Control script using QC VAE for anomaly detection.

This script trains a QC VAE on metabolite data and identifies anomalous samples
based on reconstruction error and latent likelihood.
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import umap
import warnings
warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.qc_vae import QCVAELightning
from data.datamodule import SingleDataModule

QC_CONFIG = {
    'model': {
        'model_type': "qc_vae",
        'latent_dim': 128,
        'encoder_layers': [1024],
        'decoder_layers': [512],
        'activation': "leaky_relu",
        'dropout_rate': 0.15,
        'batch_norm': True,
        'use_residual_blocks': True,
    },
    
    'training': {
        'max_epochs': 200,
        'learning_rate': 0.001,
        'batch_size': 256,
        'optimizer': "adamw",
        'scheduler': "reduce_on_plateau",
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'early_stopping_patience': 15,
        'gradient_clip_val': 0.5,
        
        'data_augmentation': {
            'enabled': True,
            'gaussian_noise_std': 0.01,
        }
    },
    
    'loss_weights': {
        'reconstruction': 1.0,
        'kl_divergence': 0.01,
    },
    
    'data': {
        'train_split': 0.8,
        'test_split': 0.2,
        'normalization_method': "zscore",
        'missing_value_strategy': "mean",
        'random_seed': 42,
        
        'log_transform': {
            'enabled': True,
            'epsilon': 1e-8,
        }
    },
    
    'qc': {
        'anomaly_threshold_percentile': 80,
        'plot_sample_size': 5000,
    },
    
    'visualization': {
        'palette': {
            'primary': '#4dbbd5',
            'secondary': '#00a087',
            'accent': '#3c5488',
            'error': '#e64b35',
            'highlight': '#f39b7f'
        },
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.1,
        'umap_random_state': 42,
        'pca_n_components': 2,
        'plot_style': "seaborn-v0_8",
        'figsize': [12, 8],
        'dpi': 300,
    },
    
    'hardware': {
        'accelerator': "auto",
        'devices': "auto",
        'precision': 32,
    }
}


def parse_arguments():
    """Parse command line arguments for quality control analysis.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing data_file,
            output_dir, experiment_name, and fast_dev_run parameters.
    """
    parser = argparse.ArgumentParser(description='Quality Control using QC VAE')
    
    parser.add_argument(
        '--data_file', 
        type=str, 
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs_qc',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default='qc_vae_experiment',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--fast_dev_run', 
        action='store_true',
        help='Run a fast development run for debugging'
    )
    
    return parser.parse_args()


def setup_logging_and_checkpointing(output_dir, experiment_name):
    """Setup logging and checkpointing callbacks for training.
    
    Args:
        output_dir (str): Base output directory for experiment results.
        experiment_name (str): Name of the experiment for organizing outputs.
        
    Returns:
        tuple: A tuple containing (logger, callbacks, exp_dir) where:
            - logger: TensorBoard logger instance
            - callbacks: List of PyTorch Lightning callbacks
            - exp_dir: Path to the experiment directory
    """
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
    """Train the QC VAE model for anomaly detection.
    
    Args:
        datamodule: Data module containing the preprocessed data.
        output_dir (str): Directory to save training outputs.
        experiment_name (str): Name of the experiment.
        fast_dev_run (bool): Whether to run a fast development run for debugging.
        
    Returns:
        tuple: A tuple containing (model, exp_dir) where:
            - model: Trained QC VAE model
            - exp_dir: Path to the experiment directory
    """
    print("\n" + "="*60)
    print("TRAINING QC VAE MODEL")
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


def perform_qc_analysis(model, datamodule, output_dir):
    """Perform quality control analysis on the full dataset.
    
    Analyzes reconstruction errors and latent likelihoods to identify
    anomalous samples that may represent poor quality data.
    
    Args:
        model: Trained QC VAE model.
        datamodule: Data module containing the full dataset.
        output_dir (Path): Directory to save analysis results.
        
    Returns:
        tuple: A tuple containing (results_df, latent_reps, reconstructions, full_data_cpu)
            - results_df: DataFrame with QC metrics for each sample
            - latent_reps: Latent space representations
            - reconstructions: Reconstructed data
            - full_data_cpu: Original data on CPU
    """
    print("\n" + "="*60)
    print("PERFORMING QUALITY CONTROL ANALYSIS")
    print("="*60)
    
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Analyzing on device: {device}")
    full_data = datamodule.full_data.to(device)
    original_data = datamodule.original_data
    
    print(f"Analyzing {len(full_data)} samples...")
    
    # Compute QC metrics for all samples
    with torch.no_grad():
        # Get model outputs
        outputs = model(full_data)
        
        recon_errors = torch.mean((outputs['recon'] - full_data) ** 2, dim=1).detach().cpu().numpy()
        
        kl_divs = 0.5 * torch.sum(
            outputs['mean'].pow(2) + outputs['logvar'].exp() - 1 - outputs['logvar'], 
            dim=1
        )
        latent_likelihoods = (-kl_divs).detach().cpu().numpy()
        
        latent_reps = outputs['z'].detach().cpu().numpy()
        reconstructions = outputs['recon'].detach().cpu().numpy()
    full_data_cpu = full_data.detach().cpu().numpy()
    
    upper_percentile = QC_CONFIG['qc']['anomaly_threshold_percentile']
    lower_percentile = 100 - upper_percentile
    recon_threshold = np.percentile(recon_errors, upper_percentile)
    likelihood_threshold = np.percentile(latent_likelihoods, upper_percentile)
    is_anomaly = (recon_errors > recon_threshold) & (latent_likelihoods > likelihood_threshold)

    print(f"Reconstruction error threshold (top {lower_percentile}%): {recon_threshold:.4f}")
    print(f"Latent likelihood threshold (top {lower_percentile}%): {likelihood_threshold:.4f}")
    print(f"Number of anomalous samples (high likelihood + high recon error): {np.sum(is_anomaly)} / {len(full_data_cpu)} ({100*np.sum(is_anomaly)/len(full_data_cpu):.1f}%)")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'sample_id': original_data.index,
        'reconstruction_error': recon_errors,
        'latent_likelihood': latent_likelihoods,
        'is_anomaly': is_anomaly,
        'category': ['Anomaly' if anom else 'Normal' for anom in is_anomaly]
    })
    
    # Save results
    results_path = output_dir / 'qc_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"QC results saved to: {results_path}")
    
    # Print summary statistics
    print("\nQC ANALYSIS SUMMARY:")
    print("="*40)
    print(f"Reconstruction Error - Mean: {recon_errors.mean():.4f}, Std: {recon_errors.std():.4f}")
    print(f"Latent Likelihood - Mean: {latent_likelihoods.mean():.4f}, Std: {latent_likelihoods.std():.4f}")
    upper_percentile = QC_CONFIG['qc']['anomaly_threshold_percentile']
    lower_percentile = 100 - upper_percentile
    high_recon_threshold = np.percentile(recon_errors, upper_percentile)
    high_likelihood_threshold = np.percentile(latent_likelihoods, upper_percentile)

    categories = []
    for i in range(len(full_data_cpu)):
        high_recon = recon_errors[i] >= high_recon_threshold
        high_likelihood = latent_likelihoods[i] >= high_likelihood_threshold

        if not high_likelihood and not high_recon:
            categories.append("✅ Normal")
        elif not high_likelihood and high_recon:
            categories.append("🔬 Interesting Unusual")
        elif high_likelihood and high_recon:
            categories.append("❌ Poor Quality")
        else:
            categories.append("✅ Well-Captured")
    
    results_df['detailed_category'] = categories
    
    # Print category statistics
    category_counts = pd.Series(categories).value_counts()
    print("\nDETAILED CATEGORY BREAKDOWN:")
    print("="*40)
    total_samples = len(full_data_cpu)
    for cat, count in category_counts.items():
        print(f"{cat}: {count} samples ({100*count/total_samples:.1f}%)")
    
    total_percentage = sum(100*count/total_samples for count in category_counts.values)
    print(f"\nTotal: {total_samples} samples ({total_percentage:.1f}%)")
    
    true_anomaly_count = category_counts.get("❌ Poor Quality", 0)
    intersection_anomaly_count = np.sum(is_anomaly)
    print(f"\nVERIFICATION:")
    print(f"Intersection anomalies: {intersection_anomaly_count}")
    print(f"Poor Quality category: {true_anomaly_count}")
    print(f"Match: {'✅ Yes' if true_anomaly_count == intersection_anomaly_count else '❌ No'}")
    
    return results_df, latent_reps, reconstructions, full_data_cpu


def create_qc_plots(results_df, latent_reps, reconstructions, full_data, output_dir):
    """Create visualization plots for QC analysis.
    
    Generates comprehensive visualization plots including joint distributions,
    latent space representations, and reconstruction quality assessments.
    
    Args:
        results_df (pd.DataFrame): QC analysis results.
        latent_reps (np.ndarray): Latent space representations.
        reconstructions (np.ndarray): Reconstructed data.
        full_data (np.ndarray): Original data.
        output_dir (Path): Directory to save plots.
    """
    print("\n" + "="*60)
    print("CREATING QC VISUALIZATION PLOTS")
    print("="*60)
    
    plt.style.use(QC_CONFIG['visualization']['plot_style'])
    palette = QC_CONFIG['visualization']['palette']
    max_samples = QC_CONFIG['qc']['plot_sample_size']
    if len(results_df) > max_samples:
        sample_indices = np.random.choice(len(results_df), max_samples, replace=False)
        results_sample = results_df.iloc[sample_indices].copy()
        latent_sample = latent_reps[sample_indices]
        recon_sample = reconstructions[sample_indices]
        data_sample = full_data[sample_indices]
    else:
        results_sample = results_df.copy()
        latent_sample = latent_reps
        recon_sample = reconstructions
        data_sample = full_data
    
    print(f"Creating plots with {len(results_sample)} samples...")
    
    # Joint distribution plot
    results_sample_log = results_sample.copy()
    results_sample_log['latent_likelihood_log'] = np.log(-results_sample['latent_likelihood'])
    
    g = sns.jointplot(
        x='latent_likelihood_log',
        y='reconstruction_error',
        data=results_sample_log,
        kind='scatter',
        hue='detailed_category',
        palette={
            "✅ Normal": palette['primary'],
            "✅ Well-Captured": palette['secondary'],
            "❌ Poor Quality": palette['error'],
            "🔬 Interesting Unusual": palette['highlight']
        },
        marginal_kws=dict(fill=True),
        height=7
    )
    
    # Add threshold lines
    upper_percentile = QC_CONFIG['qc']['anomaly_threshold_percentile']
    lower_percentile = 100 - upper_percentile
    recon_threshold = np.percentile(results_sample['reconstruction_error'], upper_percentile)
    likelihood_threshold = np.percentile(results_sample['latent_likelihood'], upper_percentile)  # HIGH likelihood
    likelihood_threshold_log = np.log(-likelihood_threshold)  # Convert to log scale
    
    g.ax_joint.axhline(y=recon_threshold, color=palette['error'], linestyle='--', label=f'Recon Error {upper_percentile}th Pct')
    g.ax_joint.axvline(x=likelihood_threshold_log, color=palette['error'], linestyle='--', label=f'Log(-Likelihood) {upper_percentile}th Pct')

    g.ax_joint.fill_between(
        [g.ax_joint.get_xlim()[0], likelihood_threshold_log],
        recon_threshold,
        g.ax_joint.get_ylim()[1],
        color=palette['error'],
        alpha=0.2,
        label='Poor Quality Region (High Likelihood + High Recon Error)'
    )
    g.ax_joint.legend()
    g.ax_joint.set_yscale('log')

    g.set_axis_labels('Log(-Latent Likelihood)', 'Reconstruction Error (Log Scale)', fontsize=12)
    g.fig.suptitle('Log(-Latent Likelihood) vs. Reconstruction Error', y=1.02, fontsize=14)
    
    joint_plot_path = output_dir / 'qc_joint_distribution.png'
    plt.savefig(joint_plot_path, dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()

    # Overview plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(results_sample['reconstruction_error'], bins=50, alpha=0.7, color=palette['primary'], edgecolor='black')
    axes[0, 0].axvline(np.percentile(results_sample['reconstruction_error'], upper_percentile), 
                       color=palette['error'], linestyle='--', label=f'{upper_percentile}th Percentile Threshold')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 1].text(0.5, 0.5, 'Joint distribution plot saved to:\\nqc_joint_distribution.png', 
                    ha='center', va='center', fontsize=12, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="black", lw=1))
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Latent Likelihood vs Reconstruction Error')

    # Category distribution pie chart
    category_counts = results_sample['category'].value_counts()
    colors = [palette['error'] if 'Anomaly' in cat else palette['primary'] for cat in category_counts.index]
    axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[1, 0].set_title('Sample Classification')
    
    # Detailed category distribution
    detailed_counts = results_sample['detailed_category'].value_counts()
    detailed_colors = {
        "✅ Well-Captured": palette['secondary'],
        "❌ Poor Quality": palette['error'],
        "✅ Normal": palette['primary'],
        "🔬 Interesting Unusual": palette['highlight']
    }
    bar_colors = [detailed_colors.get(cat, palette['accent']) for cat in detailed_counts.index]
    
    axes[1, 1].bar(range(len(detailed_counts)), detailed_counts.values, color=bar_colors)
    axes[1, 1].set_xticks(range(len(detailed_counts)))
    axes[1, 1].set_xticklabels([cat.split(' ', 1)[1] if ' ' in cat else cat for cat in detailed_counts.index], rotation=45, ha="right")
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Detailed Category Breakdown')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'qc_overview.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Latent Space Visualization
    print("Computing PCA and UMAP embeddings...")
    
    pca = PCA(n_components=QC_CONFIG['visualization']['pca_n_components'])
    latent_pca = pca.fit_transform(latent_sample)
    umap_reducer = umap.UMAP(
        n_neighbors=QC_CONFIG['visualization']['umap_n_neighbors'],
        min_dist=QC_CONFIG['visualization']['umap_min_dist'],
        random_state=QC_CONFIG['visualization']['umap_random_state']
    )
    latent_umap = umap_reducer.fit_transform(latent_sample)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    normal_mask = results_sample['category'] == 'Normal'
    axes[0, 0].scatter(latent_pca[normal_mask, 0], latent_pca[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[0, 0].scatter(latent_pca[~normal_mask, 0], latent_pca[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].set_title('PCA of Latent Space (Anomaly Status)')
    axes[0, 0].legend()
    axes[0, 1].scatter(latent_umap[normal_mask, 0], latent_umap[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[0, 1].scatter(latent_umap[~normal_mask, 0], latent_umap[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    axes[0, 1].set_title('UMAP of Latent Space (Anomaly Status)')
    axes[0, 1].legend()
    scatter1 = axes[1, 0].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                 c=results_sample['reconstruction_error'], cmap='viridis', alpha=0.6, s=20)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1, 0].set_title('PCA of Latent Space (Reconstruction Error)')
    plt.colorbar(scatter1, ax=axes[1, 0], label='Reconstruction Error')
    scatter2 = axes[1, 1].scatter(latent_umap[:, 0], latent_umap[:, 1], 
                                 c=results_sample['reconstruction_error'], cmap='viridis', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    axes[1, 1].set_title('UMAP of Latent Space (Reconstruction Error)')
    plt.colorbar(scatter2, ax=axes[1, 1], label='Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_qc.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    # Reconstruction Space Visualization
    print("Computing PCA and UMAP for reconstruction space...")
    
    data_pca = PCA(n_components=2).fit_transform(data_sample)
    recon_pca = PCA(n_components=2).fit_transform(recon_sample)
    data_umap = umap.UMAP(
        n_neighbors=QC_CONFIG['visualization']['umap_n_neighbors'],
        min_dist=QC_CONFIG['visualization']['umap_min_dist'],
        random_state=QC_CONFIG['visualization']['umap_random_state']
    ).fit_transform(data_sample)
    recon_umap = umap.UMAP(
        n_neighbors=QC_CONFIG['visualization']['umap_n_neighbors'],
        min_dist=QC_CONFIG['visualization']['umap_min_dist'],
        random_state=QC_CONFIG['visualization']['umap_random_state']
    ).fit_transform(recon_sample)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].scatter(data_pca[normal_mask, 0], data_pca[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[0, 0].scatter(data_pca[~normal_mask, 0], data_pca[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA of Original Data (Anomaly Status)')
    axes[0, 0].legend()
    axes[0, 1].scatter(recon_pca[normal_mask, 0], recon_pca[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[0, 1].scatter(recon_pca[~normal_mask, 0], recon_pca[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA of Reconstructed Data (Anomaly Status)')
    axes[0, 1].legend()
    axes[1, 0].scatter(data_umap[normal_mask, 0], data_umap[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[1, 0].scatter(data_umap[~normal_mask, 0], data_umap[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].set_title('UMAP of Original Data (Anomaly Status)')
    axes[1, 0].legend()
    axes[1, 1].scatter(recon_umap[normal_mask, 0], recon_umap[normal_mask, 1], 
                      c=palette['primary'], alpha=0.6, s=20, label='Normal')
    axes[1, 1].scatter(recon_umap[~normal_mask, 0], recon_umap[~normal_mask, 1], 
                      c=palette['error'], alpha=0.8, s=20, label='Anomaly')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    axes[1, 1].set_title('UMAP of Reconstructed Data (Anomaly Status)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_space_qc.png', dpi=QC_CONFIG['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    print("QC visualization plots saved successfully!")
    print(f"Plots saved to:")
    print(f"  - {output_dir / 'qc_overview.png'}")
    print(f"  - {output_dir / 'latent_space_qc.png'}")
    print(f"  - {output_dir / 'reconstruction_space_qc.png'}")
    print(f"  - {output_dir / 'qc_joint_distribution.png'}")


def create_additional_qc_plots(results_df, reconstructions, full_data, original_data, output_dir):
    """Create additional detailed QC plots.
    
    Generates feature-wise reconstruction error analysis and individual
    anomalous sample profiles for detailed inspection.
    
    Args:
        results_df (pd.DataFrame): QC analysis results.
        reconstructions (np.ndarray): Reconstructed data.
        full_data (np.ndarray): Original data.
        original_data (pd.DataFrame): Original data with feature names.
        output_dir (Path): Directory to save plots.
    """
    print("\n" + "="*60)
    print("CREATING ADDITIONAL QC PLOTS")
    print("="*60)

    plt.style.use(QC_CONFIG['visualization']['plot_style'])
    palette = QC_CONFIG['visualization']['palette']
    feature_errors = np.mean((reconstructions - full_data) ** 2, axis=0)
    error_df = pd.DataFrame({
        'feature': original_data.columns,
        'error': feature_errors
    }).sort_values('error', ascending=False)

    top_n_features = 25
    
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='error', 
        y='feature', 
        data=error_df.head(top_n_features), 
        color=palette['primary']
    )
    plt.title(f'Top {top_n_features} Features by Mean Reconstruction Error')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    feature_error_path = output_dir / 'feature_reconstruction_error.png'
    plt.savefig(feature_error_path, dpi=QC_CONFIG['visualization']['dpi'])
    plt.close()
    print(f"Feature reconstruction error plot saved to: {feature_error_path}")
    top_anomalies = results_df.sort_values('reconstruction_error', ascending=False).head(5)
    anomaly_plot_dir = output_dir / 'top_anomalous_samples'
    anomaly_plot_dir.mkdir(exist_ok=True)
    
    print(f"Saving top anomaly profile plots to: {anomaly_plot_dir}")

    for idx, row in top_anomalies.iterrows():
        sample_id = row['sample_id']
        sample_idx = original_data.index.get_loc(sample_id)
        
        original_sample = full_data[sample_idx]
        recon_sample = reconstructions[sample_idx]
        
        plt.figure(figsize=(15, 6))
        
        plt.plot(original_sample, 'o-', label='Original Data', color=palette['primary'], markersize=4, alpha=0.7)
        plt.plot(recon_sample, 'o-', label='Reconstructed Data', color=palette['error'], markersize=4, alpha=0.7)
        
        plt.title(f'Profile of Anomalous Sample: {sample_id}\nReconstruction Error: {row["reconstruction_error"]:.4f}')
        plt.xlabel('Feature Index')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plot_path = anomaly_plot_dir / f'anomaly_profile_{sample_id}.png'
        plt.savefig(plot_path, dpi=QC_CONFIG['visualization']['dpi'])
        plt.close()

    print("Additional QC plots created successfully.")


def main():
    """Main QC analysis function.
    
    Orchestrates the complete quality control analysis pipeline including
    data loading, model training, anomaly detection, and visualization.
    """
    args = parse_arguments()
    
    print("="*80)
    print("QUALITY CONTROL ANALYSIS USING QC VAE")
    print("="*80)
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    
    print("\nSetting up data module...")
    datamodule = SingleDataModule(
        config=QC_CONFIG,
        data_file=args.data_file
    )
    datamodule.setup()
    model, exp_dir = train_qc_vae(
        datamodule=datamodule,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        fast_dev_run=args.fast_dev_run
    )
    
    if not args.fast_dev_run:
        results_df, latent_reps, reconstructions, full_data = perform_qc_analysis(
            model=model,
            datamodule=datamodule,
            output_dir=exp_dir
        )
        create_qc_plots(
            results_df=results_df,
            latent_reps=latent_reps,
            reconstructions=reconstructions,
            full_data=full_data,
            output_dir=exp_dir
        )
        create_additional_qc_plots(
            results_df=results_df,
            reconstructions=reconstructions,
            full_data=full_data,
            original_data=datamodule.original_data,
            output_dir=exp_dir
        )
        print("\n" + "="*80)
        print("QC ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {exp_dir}")
        print(f"QC results CSV: {exp_dir / 'qc_results.csv'}")
        print(f"Visualization plots: {exp_dir / '*.png'}")
        print(f"Detailed anomaly plots: {exp_dir / 'top_anomalous_samples/'}")
        poor_quality_sample_ids = results_df[results_df['is_anomaly']]['sample_id'].tolist()
        filtered_data = datamodule.original_data[~datamodule.original_data.index.isin(poor_quality_sample_ids)]
        filtered_path = exp_dir / 'filtered_data_qc_passed.csv'
        filtered_data.to_csv(filtered_path)
        print(f"Filtered data (QC passed): {filtered_path}")
        poor_quality_data = datamodule.original_data[datamodule.original_data.index.isin(poor_quality_sample_ids)]
        poor_quality_path = exp_dir / 'poor_quality_samples.csv'
        poor_quality_data.to_csv(poor_quality_path)
        print(f"Poor quality samples: {poor_quality_path}")
        n_total = len(results_df)
        n_anomalies = np.sum(results_df['is_anomaly'])
        print(f"\nFINAL QC SUMMARY:")
        print(f"Total samples analyzed: {n_total}")
        print(f"Anomalous samples detected: {n_anomalies} ({100*n_anomalies/n_total:.1f}%)")
        print(f"Normal samples: {n_total - n_anomalies} ({100*(n_total - n_anomalies)/n_total:.1f}%)")
    
    print("\nQC analysis pipeline completed!")


if __name__ == "__main__":
    pl.seed_everything(QC_CONFIG['data']['random_seed'], workers=True)
    main() 