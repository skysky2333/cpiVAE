#!/usr/bin/env python3
"""
Training script for Joint VAE model.

This script handles the complete training pipeline including data loading,
model initialization, training loop, and checkpoint saving.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import JointVAELightning, JointVAEPlusLightning, DirectImputationLightning, GenerativeVAE, JointIWAELightning, JointVAEVampPriorLightning, JointVQLightning, JointMMLightning
from data import JointVAEDataModule
from utils import load_config, save_scalers_and_features


def parse_arguments():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Path to configuration YAML file
            - platform_a: Path to platform A CSV file
            - platform_b: Path to platform B CSV file
            - output_dir: Output directory for logs and checkpoints
            - experiment_name: Name of the experiment
            - version: Experiment version (timestamp if None)
            - resume_from_checkpoint: Path to checkpoint for resuming training
            - fast_dev_run: Flag for debugging mode
    """
    parser = argparse.ArgumentParser(description='Train Joint VAE model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
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
        default='outputs',
        help='Output directory for logs and checkpoints'
    )
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default='joint_vae_experiment',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Experiment version. If None, uses a timestamp.'
    )
    parser.add_argument(
        '--resume_from_checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--fast_dev_run', 
        action='store_true',
        help='Run a fast development run for debugging'
    )
    
    return parser.parse_args()


def setup_logging_and_checkpointing(output_dir, experiment_name, version, config):
    """
    Setup logging and checkpointing callbacks with flexible configuration parsing.
    
    Args:
        output_dir (str): Base output directory for experiments
        experiment_name (str): Name of the experiment
        version (str or None): Experiment version, timestamp if None
        config (dict): Configuration dictionary containing callback settings
        
    Returns:
        tuple: (logger, callbacks, exp_dir) containing:
            - TensorBoardLogger instance
            - List of PyTorch Lightning callbacks
            - Path to experiment directory
    """
    if version is None:
        import time
        version = time.strftime("version_%Y%m%d-%H%M%S")

    exp_dir = Path(output_dir) / experiment_name / version
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=str(exp_dir.parent.parent),
        name=experiment_name,
        version=version
    )
    
    callbacks_config = config.get('callbacks', {})
    mc_config = callbacks_config.get('model_checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=mc_config.get('filename', f'{experiment_name}-{{epoch:02d}}-{{val_total_loss:.3f}}'),
        monitor=mc_config.get('monitor', config.get('logging', {}).get('monitor_metric', 'val_total_loss')),
        mode=mc_config.get('mode', config.get('logging', {}).get('monitor_mode', 'min')),
        save_top_k=mc_config.get('save_top_k', config.get('logging', {}).get('save_top_k', 3)),
        save_last=mc_config.get('save_last', True),
        verbose=True
    )
    
    es_config = callbacks_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        monitor=es_config.get('monitor', config.get('logging', {}).get('monitor_metric', 'val_total_loss')),
        mode=es_config.get('mode', config.get('logging', {}).get('monitor_mode', 'min')),
        patience=es_config.get('patience', config.get('training', {}).get('early_stopping_patience', 10)),
        min_delta=es_config.get('min_delta', 0.0),
        verbose=True,
        strict=False
    )
    
    lr_config = callbacks_config.get('lr_monitor', {})
    lr_monitor = LearningRateMonitor(
        logging_interval=lr_config.get('logging_interval', 'step')
    )
    
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    
    return logger, callbacks, exp_dir


def main():
    """
    Main training function that orchestrates the complete training pipeline.
    
    This function handles:
    - Configuration loading and argument parsing
    - Data module setup and dimension extraction
    - Model initialization based on configuration
    - Training loop execution with callbacks
    - Final model evaluation and artifact saving
    """
    args = parse_arguments()
    
    print(f"Loading configuration from {args.config}")
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
    
    logger, callbacks, exp_dir = setup_logging_and_checkpointing(
        args.output_dir, args.experiment_name, args.version, config
    )
    
    print("Saving scalers, feature names, and log transformation parameters for inference...")
    scaler_paths = save_scalers_and_features(datamodule, str(exp_dir))
    print(f"Scalers saved to: {scaler_paths['scalers_path']}")
    print(f"Feature names saved to: {scaler_paths['features_path']}")
    print(f"Log transformation parameters saved to: {scaler_paths['log_params_path']}")
    
    print("Initializing model...")
    model_type = config['model'].get('model_type', 'joint_vae')
    
    if model_type == 'joint_vae_plus':
        print("Using Enhanced Joint VAE (joint_vae_plus)")
        model = JointVAEPlusLightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'joint_vae':
        print("Using Original Joint VAE (joint_vae)")
        model = JointVAELightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config,
            datamodule=datamodule
        )
    elif model_type == 'JointVAEVampPrior':
        print("Using Joint VAE with VampPrior (JointVAEVampPrior)")
        model = JointVAEVampPriorLightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'JointIWAE':
        print("Using Joint VAE with IWAE")
        model = JointIWAELightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'JointVQ':
        print("Using Joint VAE with VQ-VAE")
        model = JointVQLightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'res_unet':
        print("Using ResNet-UNet Direct Imputation (res_unet)")
        model = DirectImputationLightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'generative_vae':
        print(f"Using Generative VAE with {config['model']['decoder_type']} decoder")
        model = GenerativeVAE(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config
        )
    elif model_type == 'JointMM':
        print("Using Joint VAE with MMVAE (Mixture Model)")
        model = JointMMLightning(
            input_dim_a=dims['input_dim_a'],
            input_dim_b=dims['input_dim_b'],
            config=config,
            datamodule=datamodule
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'joint_vae', 'joint_vae_plus', 'JointVAEVampPrior', 'res_unet', or 'generative_vae'")
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config['training'].get('gradient_clip_val', 0.5),
        gradient_clip_algorithm=config['training'].get('gradient_clip_algorithm', 'norm'),
        log_every_n_steps=config['logging']['log_every_n_steps'],
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )
    
    print("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )
    
    if not args.fast_dev_run:
        print("Running final evaluation on the best checkpoint...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
    else:
        print("Skipping final evaluation in fast_dev_run mode.")
    final_model_path = exp_dir / 'final_model.ckpt'
    trainer.save_checkpoint(str(final_model_path))
    
    print(f"Training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main() 