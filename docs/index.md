# cpiVAE: Cross-platform Proteomics Imputation using Variational Autoencoders

Welcome to the comprehensive documentation for **cpiVAE**, a highly customizable and user friendly framework for cross-platform proteomics data imputation using Variational Autoencoders.

## Overview

cpiVAE addresses the critical challenge of data harmonization in proteomics research by enabling accurate imputation between different measurement platforms (e.g., Olink and SomaScan). Our approach leverages deep generative models to learn shared latent representations that capture the underlying biological signals across platforms.

## Key Features

- **Multi-Architecture Support**: Joint VAE, VampPrior, IWAE, VQ-VAE, and easy addition of custom models
- **Robust Baseline Methods**: KNN and WNN implementations for comparison
- **Comprehensive Evaluation**: Feature importance analysis and quality control metrics
- **Production Ready**: PyTorch Lightning framework with full experiment tracking
- **Flexible Configuration**: YAML-based configuration setup for easy customization

## Quick Start

Get started with cpiVAE in three simple steps:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a Model**
   ```bash
   python scripts/train.py --config configs/default.yaml \
       --platform_a data/olink_train.csv \
       --platform_b data/somascan_train.csv
   ```

3. **Perform Cross-Platform Imputation**
   ```bash
   python scripts/impute.py --checkpoint outputs/model.ckpt \
       --platform_impute data/test.csv --impute_target b
   ```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [**Training**](train.md) | Model training with PyTorch Lightning |
| [**Baselines**](baselines.md) | KNN and WNN baseline implementations |
| [**Imputation**](impute.md) | Cross-platform imputation pipeline |
| [**Model Tuning**](tune.md) | Hyperparameter optimization and model selection |
| [**Quality Control**](quality_control.md) | Data validation and preprocessing |
| [**Feature Importance**](feature_importance.md) | Analyzing learned representations |
| [**Latent Space**](latent_space.md) | Visualization and interpretation |
| [**Comparison**](comparison.md) | Benchmarking against baselines |

## Architecture

cpiVAE employs a dual-encoder, dual-decoder architecture that learns platform-specific representations while maintaining a shared latent space. This design enables:

- **Bidirectional Imputation**: A→B and B→A cross-platform prediction
- **Latent Alignment**: Shared biological signal representation
- **Platform Adaptation**: Handling platform-specific measurement characteristics

## Research Applications

- **Multi-platform Studies**: Harmonize data across different proteomics platforms
- **Data Integration**: Combine measurements from multiple cohorts or studies  
- **Method Validation**: Compare measurement consistency across platforms

## Citation

If you use cpiVAE in your research, please cite:

```bibtex
TBD
```

---
