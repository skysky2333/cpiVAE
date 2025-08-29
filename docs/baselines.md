# Baseline Methods

## Overview

The cpiVAE framework includes implementations of established baseline methods for cross-platform proteomics imputation. These methods serve as benchmarks for evaluating the performance of the cpiVAE model and provide alternative approaches for cross-platform data harmonization.

## K-Nearest Neighbors (KNN) Baseline

### Script: `run_knn_comparison.py`

Implements K-nearest neighbors regression for cross-platform imputation with comprehensive parameter optimization.

#### Usage
```bash
python scripts/run_knn_comparison.py --platform_a PLATFORM_A_FILE --platform_b PLATFORM_B_FILE [OPTIONS]
```

#### Key Parameters
- `--platform_a`, `--platform_b`: Training data files for both platforms
- `--platform_impute`: Test data file for cross-platform imputation
- `--impute_target`: Target platform (`a` or `b`)
- `--k_values`: List of k values to test (default: [3,5,7,10,15,30,50,100,200])
- `--kernel`: Weighting function (`uniform`, `distance`, `gaussian`, `exponential`, `tricube`)
- `--cv_folds`: Cross-validation folds for parameter optimization

#### Details
1. **Data Preprocessing**: Optional log transformation and standardization
2. **Parameter Search**: Grid search over k values and kernel functions
3. **Cross-Validation**: k-fold CV for robust parameter selection
4. **Imputation**: Weighted average of k nearest neighbors in the source platform
5. **Kernel Functions**: Multiple weighting schemes including Gaussian and polynomial

#### Example
```bash
python scripts/run_knn_comparison.py \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --platform_impute data/olink_overlap_test.csv \
    --impute_target b \
    --kernel gaussian \
    --output_dir outputs_knn
```

---

## Weighted Nearest Neighbors (WNN) Baseline

### Script: `wnn_baseline.py`

Implements the Weighted Nearest Neighbors algorithm adapted from Hao et al. (2021), originally developed for single-cell multimodal data integration.

#### Usage
```bash
python scripts/wnn_baseline.py --platform_a PLATFORM_A_FILE --platform_b PLATFORM_B_FILE [OPTIONS]
```

#### Key Parameters
- `--platform_a`, `--platform_b`: Training data files
- `--platform_impute`: Test data for imputation
- `--impute_target`: Target platform for imputation
- `--n_neighbors`: Number of neighbors for graph construction (default: 20)
- `--n_components`: PCA components for dimensionality reduction (default: 50)
- `--sigma`: Bandwidth parameter for weight computation (default: 1.0)


#### Example
```bash
# WNN imputation from SomaScan to Olink
python scripts/wnn_baseline.py \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --platform_impute data/somascan_overlap_test.csv \
    --impute_target a \
    --output_dir outputs_wnn
```

---
