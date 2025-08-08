## Phenotype Association Analysis

This page describes two complementary ways to run phenotype-protein association analyses with covariate adjustment:

- Discovery mode over all available phenotypes using `scripts/pheno_discovery.py`
- Targeted analysis for specified phenotypes integrated into the broader comparison workflow via `scripts/compare_result_oneplatform.py`

### Discovery: pheno_discovery.py

The discovery script scans all phenotype columns, classifies each as binary or continuous, and performs per-protein association tests using the truth matrix. All tests adjust for the specified age and gender covariates, with the exception that a phenotype does not adjust for itself (gender phenotypes adjust only for age; age phenotypes adjust only for gender). Multiple testing correction is performed using FDR (Benjamini–Hochberg).

Usage:

```bash
python scripts/pheno_discovery.py \
  --truth_a PATH/to/truth.csv \
  --phenotype_file PATH/to/phenotypes.csv \
  --output_dir results_pheno \
  --gender_col GENDER --age_col V5AGE52 \
  [--transpose]
```

- truth_a: Features × samples by default (use `--transpose` if your input is samples × features). First column is the index.
- phenotype_file: Samples as index in the first column; remaining columns are phenotypes.
- gender_col / age_col: Covariate column names in the phenotype file.

Outputs:

- results_pheno/summary_binary_all.csv and summary_continuous_all.csv
- results_pheno/summary_binary_top5.csv and summary_continuous_top5.csv
- results_pheno/summary.txt (concise report)
- Per-phenotype result tables:
  - results_pheno/associations_binary/binary_associations_<PHENO>.csv
  - results_pheno/associations_continuous/continuous_associations_<PHENO>.csv

Each per-phenotype table contains effect estimates (odds ratios for binary; beta for continuous), standard errors, p-values, and FDR-adjusted p-values (`p_adj`).

### Targeted: compare_result_oneplatform.py

The `scripts/compare_result_oneplatform.py` workflow supports phenotype association analysis for a user-specified set of phenotype columns alongside comprehensive method comparison and figure generation. Provide a phenotype file and explicit phenotype columns via `--binary_pheno` and `--continuous_pheno`. The analysis adjusts for age and gender as covariates and will generate per-phenotype result tables and summary figures.

Key flags (excerpt):

```bash
python scripts/compare_result_oneplatform.py \
  --truth_a TRUTH_A --imp_a_m1 IMP_A_M1 --imp_a_m2 IMP_A_M2 \
  --platform_a_name "Platform A" \
  --phenotype_file data/phenotypes.csv \
  --binary_pheno DIABETES HYPERTENSION \
  --continuous_pheno AGE BMI \
  [--transpose] \
  --output_dir outputs
```

Refer to the Comparison documentation for full usage and outputs.


