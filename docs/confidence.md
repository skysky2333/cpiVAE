# Confidence & Analysis (confidence.py, confidence_analysis.py)

## Overview

This page covers both generating confidence (uncertainty) and analyzing how confidence relates to imputation performance.

---

## Part I — Confidence Estimation (`confidence.py`)

`confidence.py` estimates per-cell uncertainty/confidence for cross-platform imputation with trained cpiVAE models. It reuses the same preprocessing and model-loading pipeline as `impute.py` and supports three methods:

- mc (Monte Carlo, default): Run K stochastic imputations and convert the coefficient of variation (CV) into a confidence score.
- delta: Single-pass analytical variance propagation from latent space through the decoder (delta method), optionally scaling to original feature space.
- latent: Directly export the encoder’s latent variance per sample (no propagation).

Interpretation:
- mc: Higher score means higher confidence (confidence = 1 / (1 + CV)).
- delta: Outputs variance in the target feature space. Higher variance indicates lower confidence.
- latent: Outputs variance in the latent space per sample; useful as an intrinsic uncertainty signal.

### Usage

```bash
python scripts/confidence.py \
  --experiment_dir EXPERIMENT_DIR \
  --input_data INPUT_FILE \
  --source_platform {a,b} --target_platform {a,b} \
  --output OUTPUT_FILE [OPTIONS]
```

#### Required Arguments
- `--experiment_dir`: Path to experiment directory with checkpoint and preprocessing artifacts
- `--input_data`: Input CSV/TXT (first column sample IDs, remaining columns features)
- `--source_platform`: `a` or `b`
- `--target_platform`: `a` or `b` (must differ from source)
- `--output`: Output CSV path

#### Method Selection
- `--method {mc,delta,latent}` (default: mc)
  - `mc`: K runs → CV → confidence score (1/(1+CV)) in original target scale
  - `delta`: Variance via Jacobian-based delta method in target scale
  - `latent`: Latent variance per sample (no propagation)

#### MC Options
- `--n_runs`: Number of stochastic imputations (default: 5). Larger values yield more stable CV estimates.

#### Delta Options
- `--delta_backend {hutchinson,exact}` (default: `hutchinson`)
  - `hutchinson`: Fast estimator of diag(J Σ J^T) using jvp/vjp, scalable to large output dims
  - `exact`: Full Jacobian per sample (very slow; use only for small tests)
- `--delta_probes`: Number of probe vectors for Hutchinson estimator (default: 16). Increase for smoother estimates.

Notes:
- The script attempts to map propagated variance back to original target feature scale if the saved scaler exposes `scale_` (std); variance scales with std^2.
- All inputs are assumed to be in log space already for this project; we do not add extra log-domain adjustments.

### Output Format

- `mc` (confidence score): CSV with shape (n_samples × n_target_features)
  - Columns: target features (or generated names)
  - Values in (0, 1]; higher is more confident
- `delta` (variance): CSV with shape (n_samples × n_target_features)
  - Columns: target features (or generated names)
  - Values: estimated output variance; higher means less confident
- `latent` (variance): CSV with shape (n_samples × latent_dim)
  - Columns: `latent_var_dim_1`, `latent_var_dim_2`, ...
  - Values: exp(logvar); higher means less confident

All outputs include the sample ID column as the first column.

### Examples

#### Monte Carlo Confidence (Olink → SomaScan)
```bash
python scripts/confidence.py \
  --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 \
  --input_data data/olink_overlap_test.csv \
  --source_platform a --target_platform b \
  --output data/confidence_mc_a2b.csv \
  --method mc --n_runs 20
```

#### Delta (Fast Hutchinson)
```bash
python scripts/confidence.py \
  --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 \
  --input_data data/olink_overlap_test.csv \
  --source_platform a --target_platform b \
  --output data/confidence_delta_a2b.csv \
  --method delta --delta_backend hutchinson --delta_probes 16
```

#### Latent Variance
```bash
python scripts/confidence.py \
  --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 \
  --input_data data/olink_overlap_test.csv \
  --source_platform a --target_platform b \
  --output data/confidence_latent_a2b.csv \
  --method latent
```

### Performance Tips
- Prefer `delta_backend=hutchinson` for large models/feature sets; the exact Jacobian backend is O(output_dim × latent_dim) per sample.
- Increase `--delta_probes` to reduce estimator noise; decrease for speed.
- For `mc`, balance `--n_runs` with runtime; 10–20 is often sufficient.

### Interpretation
- Use `mc` confidence when you want a bounded score (0–1) that inversely reflects output variability.
- Use `delta` when you need an analytical, single-pass variance in the target feature space.
- Use `latent` for a quick, model-intrinsic uncertainty signal independent of the decoder.

---

## Part II — Confidence Analysis (`confidence_analysis.py` / `confidence_analysis_oneplatform.py`)

Analyze how confidence relates to performance by comparing confidence matrices against ground-truth and imputed matrices.

### Modes
- Single platform: use `confidence_analysis_oneplatform.py`
- Two platforms: use `confidence_analysis.py` (plots split into two subplots, each with its own colorbar)

### Inputs

Single platform:
- `--truth`: Ground truth matrix (features × samples)
- `--imputed`: Imputed matrix (samples × features)
- `--confidence`: Confidence matrix (samples × features)
- `--platform_name`: Name used in figures

Two platforms additionally accept:
- `--truth_a`, `--truth_b`
- `--imputed_a`, `--imputed_b`
- `--confidence_a`, `--confidence_b`
- `--platform_a_name`, `--platform_b_name`

Common options:
- `--correlation {pearson,spearman}` (default: pearson)
- `--output_dir` for figures and tables
- `--max_points` to downsample cell-wise scatter (default: 50,000)

### What It Computes
- Feature-wise: r between truth and imputed per feature; plot mean confidence (X) vs r (Y)
- Sample-wise: r per sample; plot mean confidence (X) vs r (Y)
- Cell-wise: confidence (X, log) vs |truth − imputed| (Y, log), with regression line

Aggregation: uses mean confidence for feature-wise and sample-wise.

Coloring:
- Feature-wise: colored by mean feature value
- Sample-wise: colored by mean sample value
- Cell-wise: colored by mean feature value; each subplot has its own colorbar; subplots are square

### Outputs
Tables under `output_dir/data`:
- `feature_confidence_vs_r.csv`, `sample_confidence_vs_r.csv`
- optional downsample previews for cell-wise scatter

Figures under `output_dir/figures`:
- Single platform: `confidence_vs_r_featurewise.(pdf|png)`, `confidence_vs_r_samplewise.(pdf|png)`, `confidence_vs_abs_error_cells.(pdf|png)`
- Two platforms: `*_split.(pdf|png)` variants for feature-wise, sample-wise, and cell-wise

### Examples

Single platform:
```bash
python scripts/confidence_analysis_oneplatform.py \
  --truth data/truth_platform_b.csv \
  --imputed outputs_vae/.../imputed_a2b.csv \
  --confidence outputs_vae/.../confidence_a2b.csv \
  --platform_name "Platform B" \
  --output_dir outputs_vae/confidence_analysis_b \
  --correlation pearson
```

Two platforms:
```bash
python scripts/confidence_analysis.py \
  --truth_a data/truth_platform_a.csv --truth_b data/truth_platform_b.csv \
  --imputed_a outputs_vae/.../imputed_a.csv --imputed_b outputs_vae/.../imputed_b.csv \
  --confidence_a outputs_vae/.../confidence_a.csv --confidence_b outputs_vae/.../confidence_b.csv \
  --platform_a_name "Platform A" --platform_b_name "Platform B" \
  --output_dir outputs_vae/confidence_analysis_both \
  --correlation pearson
```

### Notes & Tips
- Axes: confidence (X) and r (Y) for feature/sample-wise; log-log for cell-wise abs error
- The scripts auto-orient/align matrices by overlapping samples and features
- Use `--max_points` to keep cell-wise scatter fast on large datasets
