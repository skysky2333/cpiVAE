#!/usr/bin/env python3
"""
Hyperparameter tuning script for Joint VAE model using Optuna.

This script optimizes hyperparameters based on cross-imputation correlation 
mean per-feature metrics, which is the key performance indicator for the 
Joint VAE model's ability to perform cross-platform imputation.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
import copy
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import (
    JointVAELightning, 
    JointVAEPlusLightning, 
    DirectImputationLightning, 
    GenerativeVAE, 
    JointIWAELightning, 
    JointVAEVampPriorLightning, 
    JointVQLightning
)
from data import JointVAEDataModule
from utils import load_config


def parse_arguments():
    """Parse command line arguments for hyperparameter tuning.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing config path,
            platform data files, output directory, study settings, and training parameters.
    """
    parser = argparse.ArgumentParser(description='Tune Joint VAE model with Optuna')
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the base configuration YAML file'
    )
    parser.add_argument(
        '--platform_a', 
        type=str, 
        required=True,
        help='Path to platform A CSV file'
    )
    parser.add_argument(
        '--platform_b', 
        type=str, 
        required=True,
        help='Path to platform B CSV file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs_tune',
        help='Output directory for Optuna study and logs'
    )
    parser.add_argument(
        '--study_name', 
        type=str, 
        default='joint_vae_study',
        help='Name of the Optuna study'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of trials for Optuna study'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum epochs per trial (reduced for faster tuning)'
    )
    
    return parser.parse_args()


def objective(trial, args, base_config, datamodule):
    """
    Optuna objective function.
    
    Optimizes hyperparameters based on the average of cross-imputation 
    correlation mean per-feature for both platforms (A->B and B->A).
    """
    config = copy.deepcopy(base_config)

    config['training']['max_epochs'] = args.max_epochs

    config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config['model']['latent_dim'] = trial.suggest_categorical('latent_dim', [16, 32, 64, 128, 256])
    config['model']['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
    config['model']['activation'] = trial.suggest_categorical('activation', 
        ['relu', 'leaky_relu', 'gelu', 'swish'])
    config['model']['batch_norm'] = trial.suggest_categorical('batch_norm', [True, False])
    config['model']['use_residual_blocks'] = trial.suggest_categorical('use_residual_blocks', [True, False])
    
    config['training']['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    config['training']['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.5, 2.0)
    
    config['training']['data_augmentation']['enabled'] = True
    config['training']['data_augmentation']['gaussian_noise_std'] = trial.suggest_float('gaussian_noise_std', 0.001, 0.5, log=True)
    
    config['loss_weights']['reconstruction'] = trial.suggest_float('reconstruction', 0.5, 2.0)
    config['loss_weights']['kl_divergence'] = trial.suggest_float('kl_divergence', 1e-4, 1e-1, log=True)
    config['loss_weights']['cross_reconstruction'] = trial.suggest_float('cross_reconstruction', 0.5, 2.0)
    config['loss_weights']['latent_alignment'] = trial.suggest_float('latent_alignment', 0.5, 2.0)
    config['loss_weights']['alignment_type'] = trial.suggest_categorical('alignment_type', 
        ['mse', 'kl_divergence', 'mmd'])

    n_encoder_layers = trial.suggest_int('n_encoder_layers', 1, 3)
    encoder_layers = []
    for i in range(n_encoder_layers):
        layer_size = trial.suggest_categorical(f'encoder_layer_{i}', [64, 128, 256, 512, 1024])
        encoder_layers.append(layer_size)
    config['model']['encoder_layers'] = encoder_layers
    
    n_decoder_layers = trial.suggest_int('n_decoder_layers', 1, 3)
    decoder_layers = []
    for i in range(n_decoder_layers):
        layer_size = trial.suggest_categorical(f'decoder_layer_{i}', [64, 128, 256, 512, 1024])
        decoder_layers.append(layer_size)
    config['model']['decoder_layers'] = decoder_layers
    logger = TensorBoardLogger(
        save_dir=str(Path(args.output_dir) / "tensorboard_logs"),
        name=args.study_name,
        version=f"trial_{trial.number}"
    )

    early_stopping = EarlyStopping(
        monitor="val_cross_a_corr_mean",
        mode="max",
        patience=10,
        verbose=False
    )

    callbacks = [early_stopping]
    dims = datamodule.get_dims()
    model_type = config['model'].get('model_type', 'joint_vae')
    
    model_map = {
        'joint_vae': JointVAELightning,
        'joint_vae_plus': JointVAEPlusLightning,
        'JointVAEVampPrior': JointVAEVampPriorLightning,
        'JointIWAE': JointIWAELightning,
        'JointVQ': JointVQLightning,
        'res_unet': DirectImputationLightning,
        'generative_vae': GenerativeVAE
    }

    model_class = model_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_kwargs = {
        'input_dim_a': dims['input_dim_a'],
        'input_dim_b': dims['input_dim_b'],
        'config': config
    }
    if model_type == 'joint_vae':
        model_kwargs['datamodule'] = datamodule

    model = model_class(**model_kwargs)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        gradient_clip_val=config['training'].get('gradient_clip_val', 0.5),
        log_every_n_steps=config['logging']['log_every_n_steps'],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    try:
        trainer.fit(model, datamodule=datamodule)
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf')  # Return negative infinity for failed trials

    val_cross_a_corr_mean = trainer.callback_metrics.get('val_cross_a_corr_mean')
    val_cross_b_corr_mean = trainer.callback_metrics.get('val_cross_b_corr_mean')
    
    if val_cross_a_corr_mean is None or val_cross_b_corr_mean is None:
        print(f"Warning: Cross-imputation correlation metrics not found for trial {trial.number}")
        return float('-inf')
    
    avg_cross_corr = (val_cross_a_corr_mean.item() + val_cross_b_corr_mean.item()) / 2.0
    
    return avg_cross_corr


def main():
    """Main hyperparameter tuning function using Optuna.
    
    Loads configuration, sets up data module, creates Optuna study, and runs
    optimization trials to find the best hyperparameters for Joint VAE model.
    Results are saved to database and best configuration is exported to YAML.
    """
    args = parse_arguments()
    
    print(f"Loading base configuration from {args.config}")
    config = load_config(args.config)
    
    print("Setting up data module...")
    datamodule = JointVAEDataModule(
        config=config,
        file_a=args.platform_a,
        file_b=args.platform_b
    )
    datamodule.setup()
    
    dims = datamodule.get_dims()
    print(f"Data loaded successfully:")
    print(f"  Platform A dimensions: {dims['input_dim_a']}")
    print(f"  Platform B dimensions: {dims['input_dim_b']}")
    
    study_path = Path(args.output_dir) / f"{args.study_name}.db"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',  # Maximize cross-imputation correlation
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        ),
        load_if_exists=True
    )

    print(f"\nStarting Optuna study: {args.study_name}")
    print(f"Optimization target: Average cross-imputation correlation (mean per-feature)")
    print(f"Number of trials: {args.n_trials}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Study database: {study_path}")

    study.optimize(
        lambda trial: objective(trial, args, config, datamodule), 
        n_trials=args.n_trials,
        timeout=None,
        n_jobs=1,
    )

    print("\n" + "="*60)
    print("OPTUNA STUDY RESULTS")
    print("="*60)
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if len(study.trials) > 0:
        print(f"\nBest trial:")
        best_trial = study.best_trial
        print(f"  Cross-imputation correlation (avg): {best_trial.value:.4f}")
        
        print(f"\n  Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best parameters to a config file
        best_config_path = Path(args.output_dir) / f"{args.study_name}_best_config.yaml"
        best_config = copy.deepcopy(config)
        
        best_config['training']['learning_rate'] = best_trial.params['learning_rate']
        best_config['model']['latent_dim'] = best_trial.params['latent_dim']
        best_config['model']['dropout_rate'] = best_trial.params['dropout_rate']
        best_config['model']['activation'] = best_trial.params['activation']
        best_config['model']['batch_norm'] = best_trial.params['batch_norm']
        best_config['model']['use_residual_blocks'] = best_trial.params['use_residual_blocks']
        
        best_config['training']['optimizer'] = best_trial.params['optimizer']
        best_config['training']['batch_size'] = best_trial.params['batch_size']
        best_config['training']['gradient_clip_val'] = best_trial.params['gradient_clip_val']
        best_config['training']['data_augmentation']['enabled'] = True
        best_config['training']['data_augmentation']['gaussian_noise_std'] = best_trial.params['gaussian_noise_std']
        
        best_config['loss_weights']['reconstruction'] = best_trial.params['reconstruction']
        best_config['loss_weights']['kl_divergence'] = best_trial.params['kl_divergence']
        best_config['loss_weights']['cross_reconstruction'] = best_trial.params['cross_reconstruction']
        best_config['loss_weights']['latent_alignment'] = best_trial.params['latent_alignment']
        best_config['loss_weights']['alignment_type'] = best_trial.params['alignment_type']
        
        encoder_layers = []
        decoder_layers = []
        for key, value in best_trial.params.items():
            if key.startswith('encoder_layer_'):
                encoder_layers.append(value)
            elif key.startswith('decoder_layer_'):
                decoder_layers.append(value)
        
        if encoder_layers:
            best_config['model']['encoder_layers'] = encoder_layers
        if decoder_layers:
            best_config['model']['decoder_layers'] = decoder_layers
        
        import yaml
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)
        
        print(f"\nBest configuration saved to: {best_config_path}")
        print(f"You can use this config for training with:")
        print(f"python scripts/train.py --config {best_config_path} --platform_a {args.platform_a} --platform_b {args.platform_b}")
    
    print("="*60)


if __name__ == '__main__':
    main() 