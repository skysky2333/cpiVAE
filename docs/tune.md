# Hyperparameter Tuning (`tune.py`)

## Overview

The `tune.py` script performs Bayesian hyperparameter optimization for cpiVAE models using Optuna. It optimizes model parameters based on cross-imputation correlation performance, which is the key metric for evaluating cross-platform proteomics data translation quality.

## Usage

```bash
python scripts/tune.py --config CONFIG_FILE --platform_a PLATFORM_A_FILE --platform_b PLATFORM_B_FILE [OPTIONS]
```

### Required Arguments

- `--config`: Path to base configuration YAML file (e.g., `configs/default.yaml`)
- `--platform_a`: Path to platform A CSV file (training data)
- `--platform_b`: Path to platform B CSV file (training data)

### Optional Arguments

- `--output_dir`: Output directory for Optuna study and logs (default: `outputs_tune`)
- `--study_name`: Name of the Optuna study (default: `joint_vae_study`)
- `--n_trials`: Number of optimization trials (default: 50)
- `--max_epochs`: Maximum epochs per trial (default: 50, reduced for faster tuning)

## Optimization Strategy

### Objective Function
The hyperparameter optimizer maximizes the average cross-imputation mean per-feature correlation across both platforms

### Hyperparameter Search Space

#### Model Architecture
- **Learning rate**: Log-uniform distribution [1e-5, 1e-2]
- **Latent dimensions**: Categorical [16, 32, 64, 128, 256]
- **Dropout rate**: Uniform [0.1, 0.5]
- **Activation function**: Categorical ['relu', 'leaky_relu', 'gelu', 'swish']
- **Batch normalization**: Boolean [True, False]
- **Residual connections**: Boolean [True, False]

#### Training Parameters
- **Optimizer**: Categorical ['adam', 'adamw']
- **Batch size**: Categorical [32, 64, 128, 256]
- **Gradient clipping**: Uniform [0.5, 2.0]
- **Gaussian noise std**: Log-uniform [0.001, 0.5]

#### Network Architecture
- **Encoder layers**: 1-3 layers, each with size [64, 128, 256, 512, 1024]
- **Decoder layers**: 1-3 layers, each with size [64, 128, 256, 512, 1024]

#### Loss Function Weights
- **Reconstruction weight**: Uniform [0.5, 2.0]
- **KL divergence weight**: Log-uniform [1e-4, 1e-1]
- **Cross-reconstruction weight**: Uniform [0.5, 2.0]
- **Latent alignment weight**: Uniform [0.5, 2.0]
- **Alignment type**: Categorical ['mse', 'kl_divergence', 'mmd']

## Output Files

### Study Database
- `{output_dir}/{study_name}.db`: SQLite database containing all trial results
- Allows resuming interrupted studies with `load_if_exists=True`

### Best Configuration
- `{output_dir}/{study_name}_best_config.yaml`: Optimized configuration file
- Ready to use with `train.py` script

### TensorBoard Logs
- `{output_dir}/tensorboard_logs/{study_name}/trial_{N}/`: Per-trial training logs
- View with: `tensorboard --logdir {output_dir}/tensorboard_logs`

## Examples

### Basic Tuning
```bash
python scripts/tune.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv
```

### Extended Tuning with Custom Parameters
```bash
python scripts/tune.py \
    --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --output_dir outputs_extensive_tune \
    --study_name extensive_joint_vae_study \
    --n_trials 100 \
    --max_epochs 30
```

## Features

### Pruning Strategy
- Uses MedianPruner to terminate unpromising trials early
- Startup trials: 5
- Warmup steps: 10
- Evaluation interval: 5 steps

### Early Stopping
- Monitors `val_cross_a_corr_mean` (validation cross-imputation correlation)
- Mode: maximize correlation
- Patience: 10 epochs

### Error Handling
- Failed trials return `-inf` to exclude from optimization
- Study can be resumed if interrupted
- Detailed error logging for debugging

## Performance Considerations

### Computational Requirements
- Each trial trains a full model (up to `max_epochs`)
- Memory usage scales with batch size and model architecture
- GPU acceleration recommended for faster trials

### Tuning Strategy
- Start with 50 trials for initial exploration
- Use shorter `max_epochs` (30-50) for faster iteration
- Increase trials (100+) for production hyperparameters

### Study Management
- Study results persist in SQLite database
- Multiple studies can run in parallel with different names
- Resume interrupted studies automatically

## Best Practices

1. **Data Preparation**: Ensure training data is properly split and preprocessed
2. **Configuration Base**: Start with a reasonable base configuration
3. **Resource Planning**: Allocate sufficient GPU memory and time
4. **Study Naming**: Use descriptive study names for organization
5. **Result Analysis**: Review TensorBoard logs to understand trial progression



### Study Analysis
```python
import optuna

# Load and analyze study
study = optuna.load_study(study_name="joint_vae_study", 
                         storage="sqlite:///outputs_tune/joint_vae_study.db")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Plot optimization history
optuna.visualization.plot_optimization_history(study)
```

### Common Issues
- **Out of memory**: Reduce `max_epochs` or trial batch sizes
- **Slow convergence**: Increase `n_trials` or adjust search space
- **Database conflicts**: Use unique study names for parallel runs
