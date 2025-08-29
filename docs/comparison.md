# Results Comparison and Analysis (`compare_result.py`)

## Overview

The `compare_result.py` script provides a comprehensive framework for evaluating and comparing cross-platform proteomics imputation methods. It generates figures, statistical analyses, and detailed performance metrics.

## Usage

```bash
python scripts/compare_result.py --truth_a TRUTH_A --truth_b TRUTH_B --imp_a_m1 IMP_A_M1 --imp_b_m1 IMP_B_M1 [OPTIONS]
```

### Required Arguments

- `--truth_a`: Ground truth data for platform A (CSV/TXT)
- `--truth_b`: Ground truth data for platform B (CSV/TXT)
- `--imp_a_m1`: Method 1 imputed data for platform A
- `--imp_b_m1`: Method 1 imputed data for platform B

### Optional Method Comparisons

- `--imp_a_m2`, `--imp_b_m2`: Method 2 imputed data
- `--imp_a_m3`, `--imp_b_m3`: Method 3 imputed data  
- `--imp_a_m4`, `--imp_b_m4`: Method 4 imputed data
- `--method1_name`, `--method2_name`, etc.: Names for methods in plots

### Platform and Network Options

- `--platform_a_name`, `--platform_b_name`: Platform names for labels
- `--ppi_file`: Protein-protein interaction network file
- `--gri_file`: Gene regulatory interaction network file
- `--transpose`: Transpose data if samples are in rows

### Output and Display Options

- `--output_dir`: Output directory for results (default: `outputs_comparison`)
- `--figsize`: Figure dimensions as "width,height" (default: "12,10")
- `--dpi`: Figure resolution (default: 300)
- `--save_data`: Save processed data matrices

## Analysis

### 1. Overall Performance Analysis

**Metrics Computed:**
- **R² Score**: Coefficient of determination
- **Pearson Correlation**: Linear correlation coefficient
- **Spearman Correlation**: Rank-based correlation
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

### 2. Feature-wise Analysis

**Per-protein/analyte evaluation:**
- Individual feature correlation distributions
- Feature-wise R² score comparisons
- Identification of challenging vs. well-imputed features

**Outputs:**
- `feature_metrics.csv`: Per-feature performance statistics
- Feature correlation heatmaps
- Distribution plots of feature-wise correlations

### 3. Sample-wise Analysis

**Per-sample evaluation:**
- Sample-wise correlation across methods
- Identification of outlier samples
- Sample quality assessment
- Correlation with sample metadata (if provided)

**Outputs:**
- `sample_metrics.csv`: Per-sample performance statistics
- Sample correlation scatter plots
- Sample quality rankings

### 4. Cross-platform Concordance

**Between-platform consistency:**
- Platform A vs Platform B correlation in latent space
- Cross-platform feature relationships
- Platform-specific bias detection

### 5. Distribution Preservation

**Data distribution analysis:**
- Value range preservation
- Distribution shape comparison (KS tests)
- Outlier detection and handling
- Log-transformation effects

### 6. Dimensionality Reduction Analysis

**Low-dimensional visualization:**
- **PCA**: Principal component analysis
- **UMAP**: Uniform manifold approximation
- Cluster separation analysis

### 7. Network-based Biological Validation

**When PPI/GRI files provided:**
- Protein interaction network analysis
- Gene regulatory network validation
- Functional enrichment analysis
- Network topology preservation

## Statistical Testing

### Significance Tests

1. **Paired t-tests**: Between methods on same samples
2. **Wilcoxon signed-rank**: Non-parametric method comparison
3. **Kolmogorov-Smirnov**: Distribution shape comparison
4. **Chi-square**: Categorical association testing
5. **ANOVA**: Multi-method performance comparison

### Multiple Testing Correction

- **Bonferroni correction**: Conservative correction for multiple comparisons
- **False Discovery Rate (FDR)**: Benjamini-Hochberg procedure
- **Family-wise Error Rate (FWER)**: Holm-Bonferroni method

### Effect Size Reporting

- **Cohen's d**: Standardized mean difference
- **Eta-squared**: Proportion of variance explained
- **Confidence intervals**: 95% CI for all major metrics


## Examples

### Four-Method Comparison

```bash
python scripts/compare_result.py \
    --truth_a data/olink_overlap_test.csv \
    --truth_b data/somascan_overlap_test.csv \
    --imp_a_m1 data/olink_overlap_test_imputed_vae.csv \
    --imp_a_m2 data/olink_overlap_test_imputed_wnn.csv \
    --imp_a_m3 data/olink_overlap_test_imputed_knn.csv \
    --imp_a_m4 data/olink_overlap_test_imputed_vae_shuffled.csv \
    --imp_b_m1 data/somascan_overlap_test_imputed_vae.csv \
    --imp_b_m2 data/somascan_overlap_test_imputed_wnn.csv \
    --imp_b_m3 data/somascan_overlap_test_imputed_knn.csv \
    --imp_b_m4 data/somascan_overlap_test_imputed_vae_shuffled.csv \
    --method1_name "jointVAE" \
    --method2_name "WNN" \
    --method3_name "KNN" \
    --method4_name "Permuted Control" \
    --platform_a_name "Olink" \
    --platform_b_name "SomaScan" \
    --ppi_file data/human_annotated_PPIs.txt \
    --gri_file data/trrust.human.txt \
    --transpose \
    --output_dir outputs_comprehensive_comparison
```

