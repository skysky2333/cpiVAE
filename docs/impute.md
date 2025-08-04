# Cross-Platform Imputation (`impute.py`)

## Overview

The `impute.py` script performs cross-platform imputation using trained cpiVAE models. It loads preprocessing artifacts, applies the trained model to transform data from one platform to another, and optionally computes feature importance using Captum. The script ensures consistent preprocessing between training and inference phases.

## Usage

```bash
python scripts/impute.py --experiment_dir EXPERIMENT_DIR --input_data INPUT_FILE --source_platform {a,b} --target_platform {a,b} --output OUTPUT_FILE [OPTIONS]
```

### Required Arguments

- `--experiment_dir`: Path to experiment directory containing model checkpoint and preprocessing artifacts
- `--input_data`: Path to input CSV/TXT file to be imputed  
- `--source_platform`: Source platform of input data (`a` or `b`)
- `--target_platform`: Target platform for imputation (`a` or `b`, must differ from source)
- `--output`: Path for output imputed CSV file

### Optional Arguments

- `--checkpoint`: Specific checkpoint file to use (auto-detects best checkpoint if not provided)
- `--config`: Specific config file to use (auto-searches in experiment directory if not provided)
- `--output_latent`: Path to save latent representations CSV file
- `--output_importance`: Path to save feature importance matrix CSV file
- `--id_column`: Name of ID column in input data (auto-detected if not provided)

### Feature Importance Options

- `--importance_method`: Attribution method (`integrated_gradients`, `deeplift`, `gradient_shap`)
- `--importance_baseline`: Baseline for attribution (`zero`, `mean`)
- `--importance_steps`: Number of steps for integrated gradients (default: 50)

## Input Data Format

### CSV/TXT File Structure
- **First column**: Sample IDs (must match ID format used during training)
- **Remaining columns**: Feature measurements for the source platform
- **Separators**: Comma for `.csv`, tab for `.txt` files
- **Missing values**: Handled automatically during preprocessing

### Example Input
```csv
SampleID,Protein1,Protein2,Protein3,...
Sample001,1.23,2.45,0.89,...
Sample002,1.67,2.91,1.12,...
```

## Preprocessing Pipeline

The script automatically applies the same preprocessing used during training:

1. **Sample Alignment**: Matches samples with training data ID format
2. **Log Transformation**: Applied if configured during training
3. **Feature Normalization**: Uses saved scalers (z-score, min-max, or robust)
4. **Missing Value Handling**: Consistent with training preprocessing

## Model Loading and Inference

### Checkpoint Detection
- **Best checkpoint**: Searches for `*best*.ckpt` files first
- **Last checkpoint**: Falls back to `last.ckpt`
- **Latest checkpoint**: Uses most recent if others unavailable

## Output Files

### Imputed Data (`--output`)
```csv
SampleID,ImputedProtein1,ImputedProtein2,...
Sample001,2.31,1.87,...
Sample002,1.94,2.15,...
```

### Latent Representations (`--output_latent`)
```csv
SampleID,latent_dim_1,latent_dim_2,...
Sample001,0.45,-0.23,...
Sample002,-0.12,0.67,...
```

### Feature Importance Matrix (`--output_importance`)
Input features (rows) × Output features (columns) importance matrix:
```csv
,output_feature_0000,output_feature_0001,...
source_feature_0,0.0234,0.0156,...
source_feature_1,0.0445,0.0289,...
```

## Examples

### Basic Cross-Platform Imputation

**Olink to SomaScan:**
```bash
python scripts/impute.py \
    --experiment_dir outputs_vae/joint_vae_experiment/version_20250804-120000 \
    --input_data data/olink_test.csv \
    --source_platform a \
    --target_platform b \
    --output data/somascan_imputed.csv
```

**SomaScan to Olink:**
```bash
python scripts/impute.py \
    --experiment_dir outputs_vae/joint_vae_experiment/version_20250804-120000 \
    --input_data data/somascan_test.csv \
    --source_platform b \
    --target_platform a \
    --output data/olink_imputed.csv
```

### Imputation with Latent Space Extraction
```bash
python scripts/impute.py \
    --experiment_dir outputs_vae/joint_vae_experiment/version_20250804-120000 \
    --input_data data/olink_test.csv \
    --source_platform a \
    --target_platform b \
    --output data/somascan_imputed.csv \
    --output_latent data/olink_latent_vectors.csv
```

### Imputation with Feature Importance Analysis
```bash
python scripts/impute.py \
    --experiment_dir outputs_vae/joint_vae_experiment/version_20250804-120000 \
    --input_data data/olink_test.csv \
    --source_platform a \
    --target_platform b \
    --output data/somascan_imputed.csv \
    --output_importance data/importance_matrix.csv \
    --importance_method deeplift
```

### Custom Configuration and Checkpoint
```bash
python scripts/impute.py \
    --experiment_dir outputs_vae/joint_vae_experiment/version_20250804-120000 \
    --config configs/custom_config.yaml \
    --checkpoint outputs_vae/joint_vae_experiment/version_20250804-120000/checkpoints/best_model.ckpt \
    --input_data data/test_data.csv \
    --source_platform a \
    --target_platform b \
    --output data/imputed_custom.csv
```

## Feature Importance Analysis

### Attribution Methods

**Integrated Gradients** (default):
- Computes gradients along straight-line path from baseline to input
- Provides stable and theoretically grounded attributions
- Suitable for all architectures

**DeepLift**:
- Computes importance based on differences from reference baseline
- Faster than Integrated Gradients
- Good for ReLU-based networks

**GradientShap**:
- Combines gradients with Shapley values
- Uses input distribution as baseline
- Most computationally intensive but theoretically robust

### Baseline Selection

**Zero Baseline** (default):
- Uses all-zero input as reference point
- Suitable when zero represents meaningful "absence" of signal

**Mean Baseline**:
- Uses sample mean as reference point
- Better when zero is not meaningful baseline
- Represents "average" input sample

### Output Format

Feature importance generates multiple files:
- **Importance matrix**: Input features × Output features importance scores
- **Global importance**: Aggregated importance per input feature
- **Metadata**: Analysis parameters and summary statistics


## Error Handling and Troubleshooting

### Common Issues

**Checkpoint not found**:
```bash
# Specify checkpoint explicitly
--checkpoint path/to/specific/checkpoint.ckpt
```

**Preprocessing artifacts missing**:
- Ensure experiment directory contains `scaler_a.pkl`, `scaler_b.pkl`, `feature_names_a.txt`, etc.
- Re-run training if artifacts are missing

**Feature count mismatch**:
- Verify input data has same features as training data
- Check platform assignment (a vs b)

**Memory errors during importance analysis**:
```bash
# Use simpler attribution method
--importance_method deeplift

# Reduce integration steps
--importance_steps 20
```
