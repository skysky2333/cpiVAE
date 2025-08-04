# cpiVAE: Cross-Platform Proteomics Data Imputation

A PyTorch-based implementation of Joint Variational Autoencoders for cross-platform proteomics data harmonization and imputation, specifically designed for integrating measurements between different proteomic platforms like Olink and SomaScan.

## Overview

cpiVAE learns a shared latent representation across different proteomics platforms, enabling cross-platform imputation and data harmonization. The model consists of platform-specific encoders and decoders connected through a shared latent space, allowing for bidirectional translation between platforms while preserving biological signals.

### Key Features

- **Cross-platform imputation**: Translate measurements between Olink, SomaScan, and other proteomics platforms
- **Multiple model architectures**: Support for VAE, VampPrior, Vector Quantized VAE, and more
- **Comprehensive evaluation**: Built-in comparison with KNN and WNN baselines
- **Feature importance analysis**: Integration with Captum for interpretability
- **Hyperparameter optimization**: Bayesian optimization with Optuna

Check the website for [full documentations](https://skysky2333.github.io/cpiVAE/).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training Workflow

1. **Prepare your data**: CSV files with matching sample IDs between platforms
```bash
python scripts/split_data.py --platform_a data/olink.csv --platform_b data/somascan.csv --output_dir data/
```

2. **Hyperparameter tuning** (optional but highly recommended):
```bash
python scripts/tune.py --config configs/default.yaml \
    --platform_a data/olink_tune.csv \
    --platform_b data/somascan_tune.csv
```

3. **Train the model**:
```bash
python scripts/train.py --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --output_dir outputs_vae
```

4. **Perform cross-platform imputation**:
```bash
python scripts/impute.py --input_data data/olink_test.csv \
    --source_platform a --target_platform b \
    --experiment_dir outputs_vae/joint_vae_experiment/version_X \
    --output data/somascan_imputed.csv
```

### Quick Imputation with Pre-trained Weights

For immediate testing, you can use our pre-trained model to perform cross-platform imputation:
- Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1u5j3JBGjkvZrRHIDL3u1YNTqFM2JoTIO?usp=sharing).
- Determine the direction imputation. In our case, platform a is Olink and platform b is Somascan

```bash
python scripts/impute.py --input_data your_data.csv \
    --source_platform a --target_platform b \
    --experiment_dir public_weight/version_20250715-124150/ \
    --output imputed_output.csv
```

**Important**: Input data must be preprocessed following standard pipelines:
- **SomaScan**: Median normalized format
- **Olink**: Standard QC pipeline
- **Both platforms**: Log2 transformed and z-standardized feature-wise
- **Features**: Must match those in `feature_names.pkl` (no missing features allowed)

For optimal results, we recommend training and tuning your own model. See [full imputation documentation](https://skysky2333.github.io/cpiVAE/impute) for details.

### Complete Analysis Pipeline

Recreate manuscript figures with the full analysis pipeline including baseline comparisons:

```bash
bash scripts/manuscript_recreate.sh
```

This will:
- Train the jointVAE model
- Run KNN and WNN baselines
- Perform cross-platform imputation
- Generate comparison reports
- Generate latent space and feature importance reports


## Documentation

For detailed documentation of individual scripts, please check out [https://skysky2333.github.io/cpiVAE/](https://skysky2333.github.io/cpiVAE/).




## Data Format

Input CSV files should have (the following works with --transpose argument. See manuscript_recreate.sh for examples usage):
- **First column**: Sample IDs (must have overlaps between platforms)
- **Remaining columns**: Feature measurements (proteins/analytes)
- **Missing values**: Allowed and handled during preprocessing

Example:
```
SampleID,Protein1,Protein2,Protein3,...
Sample001,1.23,2.45,0.89,...
Sample002,1.67,2.91,1.12,...
```

## Configuration

Models are configured via YAML files in `configs/`:
- `default.yaml`: Standard jointVAE configuration

Key parameters:
- `model_type`: Architecture
- `latent_dim`: Latent space dimension
- `learning_rate`, `batch_size`, `max_epochs`: Training parameters
- `loss_weights`: Balance between loss components

## Output Structure

Training produces:
```
outputs_vae/
├── joint_vae_experiment/
│   └── version_YYYYMMDD-HHMMSS/
│       ├── checkpoints/
│       ├── config.yaml
│       ├── feature_names_a.txt
│       ├── feature_names_b.txt
│       ├── scaler_a.pkl
│       └── scaler_b.pkl
```

## Evaluation Metrics

The framework provides comprehensive evaluation:
- **R²**: Global prediction accuracy
- **Feature-wise correlation**: Per-protein imputation quality  
- **Sample-wise correlation**: Per-sample prediction quality
- **Cross-platform concordance**: Between-platform agreement

## Advanced Features

### Feature Importance Analysis
```bash
python scripts/feature_importance_analysis.py \
    --importance_a_to_b importance_scores.csv \
    --network_type directed \
    --threshold_method absolute_importance
```

### Latent Space Analysis
```bash
python scripts/latent_space_analysis.py \
    --latent_a latent_representations_a.csv \
    --latent_b latent_representations_b.csv
```

### Quality Control
```bash
python scripts/qc.py --data_file data/platform_data.csv
python scripts/qc_feature.py --data_file data/platform_data.csv
```

## Citation

If you use this software in your research, please cite:

```bibtex
[To be added]
```

## Acknowledgement

A portion of the code in this repository was generated by Claude 4.0, a generative model from Anthropic, and subsequently reviewed and refined by human developers.


## License

[To be added]