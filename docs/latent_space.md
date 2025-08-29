# Latent Space Analysis

The `latent_space_analysis.py` script provides comprehensive analysis of latent representations learned by cpiVAE for cross-platform proteomics data. It generates detailed visualizations and statistics to understand what biological patterns the model has captured in its latent space.

## Overview

The latent space analysis helps answer key questions about the cpiVAE's learned representations:
- How well do latent representations align between platforms?
- What biological patterns are captured in different latent dimensions?
- How can we interpret and validate the learned latent space structure?


## Usage

### Basic Usage

```bash
python scripts/latent_space_analysis.py \
    --latent_a data/latent_platform_a.csv \
    --latent_b data/latent_platform_b.csv \
    --truth_a data/truth_platform_a.csv \
    --truth_b data/truth_platform_b.csv \
    --platform_a_name "Olink" \
    --platform_b_name "SomaScan" \
    --output_dir latent_analysis_results
```

### With Biological Groups

```bash
python scripts/latent_space_analysis.py \
    --latent_a data/latent_platform_a.csv \
    --latent_b data/latent_platform_b.csv \
    --truth_a data/truth_platform_a.csv \
    --truth_b data/truth_platform_b.csv \
    --groups data/sample_groups.csv \
    --platform_a_name "Olink" \
    --platform_b_name "SomaScan" \
    --output_dir latent_analysis_results
```


## Arguments

### Required

- `--latent_a`: Path to latent space CSV file for platform A (samples × latent dimensions)
- `--latent_b`: Path to latent space CSV file for platform B (samples × latent dimensions)
- `--truth_a`: Path to truth data CSV file for platform A (samples × features)
- `--truth_b`: Path to truth data CSV file for platform B (samples × features)
- `--platform_a_name`: Display name for platform A (e.g., "Olink")
- `--platform_b_name`: Display name for platform B (e.g., "SomaScan")

### Optional

- `--groups`: Path to biological groups/metadata CSV file (samples × group labels)
- `--output_dir`: Output directory for results (default: `latent_analysis_output`)
- `--transpose_latent`: Transpose latent files if dimensions are stored as rows

## Input File Formats

### Latent Space Files
CSV format with samples as rows and latent dimensions as columns:
```csv
SampleID,Dim_1,Dim_2,Dim_3,...
Sample001,0.123,-0.456,0.789,...
Sample002,-0.234,0.567,-0.890,...
```

### Truth Data Files
CSV format with samples as rows and features as columns:
```csv
SampleID,Protein1,Protein2,Protein3,...
Sample001,1.23,2.45,0.89,...
Sample002,1.67,2.91,1.12,...
```

### Groups File (Optional)
CSV format with samples as rows and group labels:
```csv
SampleID,Group
Sample001,Disease
Sample002,Control
Sample003,Disease
```


## Dependencies

### Required Packages
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `scikit-learn`: PCA and t-SNE
- `scipy`: Statistical functions
- `umap-learn`: UMAP dimensionality reduction

## Troubleshooting

### Common Issues

1. **"No common samples found"**
   - Check that sample IDs match between latent files and truth files
   - Verify that files use consistent indexing

2. **"UMAP not available"**
   - Install UMAP: `pip install umap-learn`
   - Analysis will continue with PCA and t-SNE only

3. **Memory issues with large datasets**
   - The script automatically subsets large matrices for visualization
   - Consider reducing the number of samples for very large datasets

4. **Empty or incorrect latent files**
   - Verify that latent files contain numerical data
   - Use `--transpose_latent` if dimensions are stored as rows
