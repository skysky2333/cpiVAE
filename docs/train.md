# Model Training (`train.py`)

## Overview

The `train.py` script handles the complete training pipeline for cpiVAE models using PyTorch Lightning. It supports multiple model architectures, comprehensive logging, checkpointing, and automatic preprocessing artifact saving for downstream inference.

## Usage

```bash
python scripts/train.py --config CONFIG_FILE --platform_a PLATFORM_A_FILE --platform_b PLATFORM_B_FILE [OPTIONS]
```

### Required Arguments

- `--config`: Path to configuration YAML file (e.g., `configs/default.yaml`)
- `--platform_a`: Path to platform A CSV file (training data)
- `--platform_b`: Path to platform B CSV file (training data)

### Optional Arguments

- `--output_dir`: Output directory for logs and checkpoints (default: `outputs`)
- `--experiment_name`: Name of the experiment (default: `joint_vae_experiment`)
- `--version`: Experiment version (default: timestamp `version_YYYYMMDD-HHMMSS`)
- `--resume_from_checkpoint`: Path to checkpoint to resume training from
- `--fast_dev_run`: Run a fast development run for debugging

## Supported Model Architectures

### Models
- **`joint_vae`**: jointVAE with dual encoders/decoders (Recommanded)
- **`joint_vae_plus`**: Enhanced version with additional tricks
- **`JointVAEVampPrior`**: VampPrior variant for improved posterior approximation
- **`JointIWAE`**: Importance Weighted Autoencoder variant
- **`JointVQ`**: Vector Quantized VAE variant
- **`JointMM`**: Mixture Model VAE variant
- **`res_unet`**: ResNet-UNet based direct imputation model
- **`generative_vae`**: Generative VAE with DDPM or GAN decoder

Note that the base JointVAE performs the best in our testing.

## Training Pipeline

### 1. Data Loading and Preprocessing
- Loads paired proteomics data from CSV files
- Automatic feature-wise normalization (z-score, min-max, or robust)
- Optional log transformation per platform
- Missing value handling and sample alignment

### 2. Model Initialization
- Automatic input dimension detection
- Model architecture selected via configuration
- Parameter count reporting

### 3. Training Loop
- PyTorch Lightning trainer with GPU/CPU acceleration
- Gradient clipping and learning rate monitoring
- Early stopping based on validation metrics
- Model checkpointing (best and last models)

### 4. Artifact Saving
- Preprocessing scalers (`scaler_a.pkl`, `scaler_b.pkl`)
- Feature names (`feature_names_a.txt`, `feature_names_b.txt`)
- Log transformation parameters (`log_params.yaml`)
- Configuration backup (`config.yaml`)

## Output Structure

Training creates the following directory structure:

```
{output_dir}/
└── {experiment_name}/
    └── {version}/
        ├── checkpoints/
        │   ├── {experiment_name}-epoch=XX-val_total_loss=X.XXX.ckpt
        │   └── last.ckpt
        ├── config.yaml
        ├── scaler_a.pkl
        ├── scaler_b.pkl
        ├── feature_names_a.txt
        ├── feature_names_b.txt
        ├── log_params.yaml
        ├── final_model.ckpt
        └── tensorboard_logs/
```

## Configuration System

### Model Configuration
```yaml
model:
  model_type: "joint_vae"  # Architecture selection
  latent_dim: 64           # Latent space dimensionality
  encoder_layers: [512, 256, 128]  # Encoder architecture
  decoder_layers: [128, 256, 512]  # Decoder architecture
  dropout_rate: 0.2        # Dropout for regularization
  activation: "relu"       # Activation function
  batch_norm: true         # Batch normalization
```

### Training Configuration
```yaml
training:
  max_epochs: 100
  learning_rate: 0.001
  batch_size: 64
  optimizer: "adam"
  gradient_clip_val: 0.5
  early_stopping_patience: 10
```

### Hardware Configuration
```yaml
hardware:
  accelerator: "auto"      # "auto", "gpu", "cpu"
  devices: "auto"          # Device selection
  precision: 32            # Mixed precision training
```

## Examples

### Basic Training
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv
```

### Custom Experiment
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --output_dir outputs_custom \
    --experiment_name my_joint_vae \
    --version v1.0
```

### Resume Training
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --resume_from_checkpoint outputs/joint_vae_experiment/version_20250804-120000/checkpoints/last.ckpt
```

### Debug Mode
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --fast_dev_run
```

## Monitoring and Callbacks

### TensorBoard Logging
View training progress:
```bash
tensorboard --logdir outputs/joint_vae_experiment/version_YYYYMMDD-HHMMSS/
```

Logged metrics include:
- Training/validation losses (total, reconstruction, KL, cross-reconstruction)
- Cross-imputation correlations
- Learning rate schedules
- Model gradients and weights

### Model Checkpointing
- **Best model**: Saved based on monitored metric (default: `val_total_loss`)
- **Last model**: Most recent checkpoint
- **Final model**: Saved at training completion
- Configurable via `callbacks.model_checkpoint` settings

### Early Stopping
- Monitors validation metrics for convergence
- Configurable patience and minimum delta
- Prevents overfitting on small datasets

## Advanced Features

### Multi-GPU Training
```yaml
hardware:
  accelerator: "gpu"
  devices: [0, 1, 2, 3]  # Use specific GPUs
```

### Mixed Precision Training
```yaml
hardware:
  precision: 16  # Use 16-bit precision for faster training
```

### Custom Callbacks
```yaml
callbacks:
  model_checkpoint:
    monitor: "val_cross_a_corr_mean"
    mode: "max"
    save_top_k: 5
  early_stopping:
    monitor: "val_cross_a_corr_mean"
    mode: "max"
    patience: 15
```

## Loss Function Components

The composite loss function includes:

1. **Reconstruction Loss**: Platform-specific data reconstruction
2. **KL Divergence**: Regularization of latent space
3. **Cross-reconstruction Loss**: Cross-platform imputation quality
4. **Latent Alignment Loss**: Shared representation learning

Loss weights are configurable via `loss_weights` in the config file.

## Data Format Requirements

### Input CSV Format
- **First column**: Sample IDs (must match between platforms)
- **Remaining columns**: Feature measurements
- **Missing values**: Handled automatically during preprocessing

### Sample Alignment
- Samples are automatically aligned by ID
- Mismatched samples are excluded with warnings

## Performance Optimization

### Memory Management
- Custom and efficient data loading with PyTorch DataLoader
- Gradient accumulation for large batch simulation


## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size in configuration
- Use mixed precision training (precision: 16)
- Reduce model architecture size

**Slow Convergence**
- Increase learning rate
- Adjust loss function weights
- Check data normalization

**Poor Cross-imputation Performance**
- Increase cross-reconstruction loss weight
- Adjust latent alignment parameters
- Verify data quality and preprocessing

### Model Selection
- Monitor validation metrics, not training loss
- Use cross-validation for small datasets
- Compare multiple model architectures
