# Quality Control

## Overview

The cpiVAE framework includes quality control to ensure data integrity and identify potential issues before model training. These tools help detect anomalous samples, assess feature quality, and prepare data for cross-platform analysis.

---

## Quality Control VAE (`qc.py`)

### Overview

The QC VAE script uses a variational autoencoder to detect anomalous samples in proteomics data based on reconstruction error and latent space likelihood. This unsupervised approach identifies outliers without requiring labeled data.

### Usage

```bash
python scripts/qc.py --data_file DATA_FILE [OPTIONS]
```

### Key Parameters

- `--data_file`: Path to CSV/TXT data file for quality control analysis
- `--output_dir`: Output directory for QC results (default: `outputs_qc`)
- `--config_file`: Optional custom configuration file
- `--gpu`: Use GPU acceleration if available
- `--fast_run`: Quick debug dev run

### Algorithm Details

1. **Data Preprocessing**: Log transformation and normalization
2. **Model Training**: Single-platform VAE with reconstruction and KL losses
3. **Anomaly Detection**: Samples with high reconstruction error flagged as anomalous
4. **Visualization**: PCA, UMAP, and latent space plots
5. **Reporting**: Detailed QC metrics and sample rankings

### Configuration Options

**Model Architecture:**
```python
'model': {
    'latent_dim': 128,
    'encoder_layers': [1024],
    'decoder_layers': [512],
    'activation': "leaky_relu",
    'dropout_rate': 0.15,
    'batch_norm': True
}
```

**Quality Control Parameters:**
```python
'qc': {
    'anomaly_threshold_percentile': 80,  # Samples above this percentile flagged
    'plot_sample_size': 5000,           # Max samples for visualization
}
```

### Output Files

```
outputs_qc/
├── checkpoints
├── top_anomalous_samples
├── reconstruction_error.png
├── filted_data_qc_passed.csv
├── hparams.yaml
├── latent_space_qc.png
├── qc_joint_distribution.png
├── qc_overview.png
├── poor_quality_samples.cvs
├── qc_results.cvs
└── reconstruction_space_qc.png
```

### Example

```bash
# Basic QC analysis
python scripts/qc.py --data_file data/olink_overlap.csv

# Comprehensive QC with custom config
python scripts/qc.py \
    --data_file data/somascan_overlap.csv \
    --output_dir outputs_qc_somascan \
    --config_file configs/qc_custom.yaml
```

---

## Feature Quality Control (`qc_feature.py`)

### Overview

Feature-level quality control with biological validation, phenotype association analysis, and technical quality assessment. Designed specifically for proteomics data with support for SomaScan and Olink platforms.

### Usage

```bash
python scripts/qc_feature.py --data_file DATA_FILE [OPTIONS]
```

### Key Parameters  

**Required:**
- `--data_file`: Path to feature data file

**Optional:**
- `--feature_annot`: Feature annotation file with metadata
- `--sample_pheno`: Sample phenotype file for association testing
- `--categorical_annot`: Categorical annotation columns (e.g., QC flags)
- `--continuous_annot`: Continuous annotation columns (e.g., CV values)
- `--binary_pheno`: Binary phenotype columns for association testing
- `--output_dir`: Output directory (default: `outputs_feature_qc`)

### Analysis Components

#### 1. Technical Quality Assessment
- **Coefficient of Variation (CV)**: Measurement precision assessment
- **Missing Value Analysis**: Completeness evaluation
- **Dynamic Range**: Signal intensity range analysis
- **Batch Effects**: Technical variation detection

#### 2. Biological Validation
- **Pathway Enrichment**: GO term and KEGG pathway analysis (when `--run_enrichment` specified)
- **Protein-Protein Interactions**: Network connectivity analysis (when `--run_network` specified)
- **Functional Annotation**: Biological process categorization

#### 3. Phenotype Association
- **Binary Traits**: Association with disease/treatment status
- **Continuous Traits**: Correlation with quantitative phenotypes
- **Multiple Testing Correction**: FDR and Bonferroni adjustment

### Configuration Examples

#### Basic Feature QC
```bash
python scripts/qc_feature.py \
    --data_file data/olink_overlap.csv \
    --output_dir outputs_feature_qc_basic
```

#### QC with Annotations
```bash
python scripts/qc_feature.py \
    --data_file data/somascan_overlap.csv \
    --feature_annot data/somascan_annotations.txt \
    --categorical_annot flag1 flag2 flag3 \
    --continuous_annot CV_intra CV_inter \
    --output_dir outputs_feature_qc_comprehensive \
    --run_biological_analysis \
    --run_enrichment \
    --run_network
```