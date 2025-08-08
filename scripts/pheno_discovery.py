import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def read_matrix(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)


def adjust_pvalues_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ordered_p = p[order]
    ranks = np.arange(1, m + 1)
    adjusted = ordered_p * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def detect_binary_and_continuous_phenotypes(phenotypes: pd.DataFrame,
                                            exclude_cols: List[str]) -> Tuple[List[str], List[str]]:
    binary_cols: List[str] = []
    continuous_cols: List[str] = []
    for col in phenotypes.columns:
        if col in exclude_cols:
            continue
        series = phenotypes[col]
        non_null = series.dropna()
        unique_values = pd.unique(non_null)
        if unique_values.size == 0:
            continue
        if unique_values.size == 2:
            binary_cols.append(col)
            continue
        # Try continuous numeric
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric = numeric.dropna()
        if numeric.size == 0:
            continue
        if pd.unique(numeric).size >= 3:
            continuous_cols.append(col)
    return binary_cols, continuous_cols


def encode_binary(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    non_null = series.dropna()
    # If numeric with exactly two unique values, map by numeric equality
    values_full = pd.to_numeric(series, errors="coerce")
    unique_numeric = pd.unique(values_full.dropna())
    if unique_numeric.size == 2:
        uniq = np.sort(unique_numeric)
        encoded = values_full.copy()
        encoded[:] = np.nan
        encoded[values_full == uniq[0]] = 0.0
        encoded[values_full == uniq[1]] = 1.0
        mapping_num = {float(uniq[0]): 0, float(uniq[1]): 1}
        return encoded.values.astype(float), {str(k): v for k, v in mapping_num.items()}

    # Otherwise, treat as strings and factorize (must be exactly two classes)
    categories = pd.Index(sorted(pd.unique(non_null.astype(str))))
    mapping = {cat: i for i, cat in enumerate(categories)}
    if len(mapping) != 2:
        raise ValueError("Binary encoding requested for non-binary series")
    encoded = series.astype(str).map(mapping)
    return encoded.values.astype(float), mapping


def prepare_covariates(phenotypes: pd.DataFrame, gender_col: str, age_col: str) -> Tuple[pd.Series, pd.Series, Dict[str, int]]:
    if gender_col not in phenotypes.columns or age_col not in phenotypes.columns:
        raise ValueError(f"Phenotype file must contain columns '{gender_col}' and '{age_col}' for covariate adjustment.")
    gender_raw = phenotypes[gender_col]
    age_raw = phenotypes[age_col]
    gender_encoded_values, gender_mapping = encode_binary(gender_raw)
    gender_series = pd.Series(gender_encoded_values, index=phenotypes.index, name=gender_col)
    age_series = pd.to_numeric(age_raw, errors="coerce")
    return gender_series, age_series, gender_mapping


def fit_logistic_with_covariates(y_binary: np.ndarray, X_protein: np.ndarray,
                                 covariates: np.ndarray) -> Tuple[float, float, float, float, int]:
    # Build design without explicit intercept; let the model handle it
    protein_z = StandardScaler().fit_transform(X_protein.reshape(-1, 1)).flatten()
    covariates_z = []
    for j in range(covariates.shape[1]):
        cov = covariates[:, j]
        if np.all(np.isfinite(cov)) and (np.unique(cov).size > 2):
            covariates_z.append(StandardScaler().fit_transform(cov.reshape(-1, 1)).flatten())
        else:
            covariates_z.append(cov)
    if covariates_z:
        X_design = np.column_stack([protein_z] + covariates_z)
    else:
        X_design = protein_z.reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs", max_iter=1000, fit_intercept=True)
    lr.fit(X_design, y_binary)
    beta = float(lr.coef_.flatten()[0])

    # Compute Fisher information on augmented design (with intercept)
    probs = lr.predict_proba(X_design)[:, 1]
    W = probs * (1.0 - probs)
    X_aug = np.column_stack([X_design, np.ones_like(protein_z)])
    X_weighted = X_aug * np.sqrt(W)[:, None]
    try:
        fisher_info_inv = np.linalg.inv(X_weighted.T @ X_weighted)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, X_design.shape[0]
    se = float(np.sqrt(np.maximum(fisher_info_inv[0, 0], 0.0)))
    if se == 0 or not np.isfinite(se):
        return np.nan, np.nan, np.nan, np.nan, X_design.shape[0]
    z = beta / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    odds_ratio = float(np.exp(beta))
    return odds_ratio, beta, se, p_value, X_design.shape[0]


def fit_linear_with_covariates(y_cont: np.ndarray, X_protein: np.ndarray,
                               covariates: np.ndarray) -> Tuple[float, float, float, float, float, int]:
    # X matrix includes [protein_z, covariates_z..., intercept]
    y = StandardScaler().fit_transform(y_cont.reshape(-1, 1)).flatten()
    protein_z = StandardScaler().fit_transform(X_protein.reshape(-1, 1)).flatten()
    covariates_z = []
    for j in range(covariates.shape[1]):
        cov = covariates[:, j]
        if np.all(np.isfinite(cov)) and (np.unique(cov).size > 2):
            covariates_z.append(StandardScaler().fit_transform(cov.reshape(-1, 1)).flatten())
        else:
            covariates_z.append(cov)
    if covariates_z:
        X = np.column_stack([protein_z] + covariates_z + [np.ones_like(protein_z)])
    else:
        X = np.column_stack([protein_z, np.ones_like(protein_z)])

    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, np.nan, X.shape[0]

    beta_hat = XtX_inv @ (X.T @ y)
    y_pred = X @ beta_hat
    residuals = y - y_pred
    n = X.shape[0]
    p = X.shape[1]
    if n <= p:
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    sigma2 = float(residuals.T @ residuals) / (n - p)
    var_beta = sigma2 * XtX_inv
    se = float(np.sqrt(np.maximum(var_beta[0, 0], 0.0)))
    beta = float(beta_hat[0])
    if se == 0 or not np.isfinite(se):
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    t_stat = beta / se
    df = n - p
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df))
    r_squared = float(1.0 - (residuals @ residuals) / np.maximum(((y - y.mean()) @ (y - y.mean())), 1e-12))
    return beta, se, t_stat, p_value, r_squared, n


def run_binary_associations(truth: pd.DataFrame, phenotypes: pd.DataFrame,
                            binary_cols: List[str], cov_gender: pd.Series,
                            cov_age: pd.Series, output_dir: Path,
                            gender_col: str, age_col: str) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    assoc_dir = output_dir / "associations_binary"
    assoc_dir.mkdir(parents=True, exist_ok=True)

    for phenotype in binary_cols:
        y_series = phenotypes[phenotype].dropna()
        # Align common samples with proteomics and covariates
        common_samples = list(set(truth.columns) & set(y_series.index) & set(cov_gender.dropna().index) & set(cov_age.dropna().index))
        if len(common_samples) < 10:
            continue
        y_raw = y_series.loc[common_samples]
        y_encoded, _ = encode_binary(y_raw)
        gender_vals = cov_gender.loc[common_samples].values.astype(float)
        age_vals = pd.to_numeric(cov_age.loc[common_samples], errors="coerce").values.astype(float)

        phenotype_results: List[Dict[str, object]] = []
        for feature in truth.index:
            x_vals = truth.loc[feature, common_samples].astype(float).values
            mask = np.isfinite(x_vals) & np.isfinite(y_encoded) & np.isfinite(gender_vals) & np.isfinite(age_vals)
            if np.sum(mask) < 10 or (np.unique(y_encoded[mask]).size < 2):
                continue
            y_clean = y_encoded[mask]
            x_clean = x_vals[mask]
            # Adjust for covariates: for GENDER phenotype, DO NOT adjust for gender
            if phenotype == gender_col:
                cov_cols = [age_vals[mask]]
            else:
                cov_cols = [age_vals[mask], gender_vals[mask]]
            covs = np.column_stack(cov_cols) if len(cov_cols) > 0 else np.empty((int(np.sum(mask)), 0))
            try:
                or_value, beta, se, p_value, n_used = fit_logistic_with_covariates(y_clean, x_clean, covs)
            except Exception:
                or_value, beta, se, p_value, n_used = np.nan, np.nan, np.nan, np.nan, int(np.sum(mask))
            phenotype_results.append({
                "feature": feature,
                "odds_ratio": or_value,
                "beta": beta,
                "se": se,
                "p_value": p_value,
                "n_samples": int(n_used),
            })

        if not phenotype_results:
            continue
        results_df = pd.DataFrame(phenotype_results)
        results_df["p_adj"] = adjust_pvalues_bh(results_df["p_value"].fillna(1.0).values)
        results_df.sort_values("p_adj", inplace=True)
        results_df.to_csv(assoc_dir / f"binary_associations_{phenotype}.csv", index=False)
        num_significant = int((results_df["p_adj"] < 0.05).sum())
        summary_rows.append({
            "phenotype": phenotype,
            "num_tested": int((results_df["p_value"].notna()).sum()),
            "num_significant": num_significant,
        })

    return pd.DataFrame(summary_rows).sort_values(["num_significant", "num_tested"], ascending=[False, False])


def run_continuous_associations(truth: pd.DataFrame, phenotypes: pd.DataFrame,
                                continuous_cols: List[str], cov_gender: pd.Series,
                                cov_age: pd.Series, output_dir: Path,
                                gender_col: str, age_col: str) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    assoc_dir = output_dir / "associations_continuous"
    assoc_dir.mkdir(parents=True, exist_ok=True)

    for phenotype in continuous_cols:
        y_series_raw = phenotypes[phenotype]
        # Convert to numeric (drop rows that cannot be converted later)
        y_numeric = pd.to_numeric(y_series_raw, errors="coerce").dropna()
        common_samples = list(set(truth.columns) & set(y_numeric.index) & set(cov_gender.dropna().index) & set(cov_age.dropna().index))
        if len(common_samples) < 10:
            continue
        y_vals = y_numeric.loc[common_samples].values.astype(float)
        gender_vals = cov_gender.loc[common_samples].values.astype(float)
        age_vals = pd.to_numeric(cov_age.loc[common_samples], errors="coerce").values.astype(float)

        phenotype_results: List[Dict[str, object]] = []
        for feature in truth.index:
            x_vals = truth.loc[feature, common_samples].astype(float).values
            mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(gender_vals) & np.isfinite(age_vals)
            if np.sum(mask) < 10:
                continue
            y_clean = y_vals[mask]
            x_clean = x_vals[mask]
            # Adjust for covariates: for V5AGE52 phenotype, DO NOT adjust for age
            if phenotype == age_col:
                cov_cols = [gender_vals[mask]]
            else:
                cov_cols = [age_vals[mask], gender_vals[mask]]
            covs = np.column_stack(cov_cols) if len(cov_cols) > 0 else np.empty((int(np.sum(mask)), 0))
            try:
                beta, se, t_stat, p_value, r2, n_used = fit_linear_with_covariates(y_clean, x_clean, covs)
            except Exception:
                beta, se, t_stat, p_value, r2, n_used = np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(mask))
            phenotype_results.append({
                "feature": feature,
                "beta": beta,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "r_squared": r2,
                "n_samples": int(n_used),
            })

        if not phenotype_results:
            continue
        results_df = pd.DataFrame(phenotype_results)
        results_df["p_adj"] = adjust_pvalues_bh(results_df["p_value"].fillna(1.0).values)
        results_df.sort_values("p_adj", inplace=True)
        results_df.to_csv(assoc_dir / f"continuous_associations_{phenotype}.csv", index=False)
        num_significant = int((results_df["p_adj"] < 0.05).sum())
        summary_rows.append({
            "phenotype": phenotype,
            "num_tested": int((results_df["p_value"].notna()).sum()),
            "num_significant": num_significant,
        })

    return pd.DataFrame(summary_rows).sort_values(["num_significant", "num_tested"], ascending=[False, False])


def main():
    parser = argparse.ArgumentParser(description="Discover phenotypes associated with proteomics (Truth A) with covariate adjustment.")
    parser.add_argument("--truth_a", required=True, help="Path to truth matrix. Default expects features x samples (rows=features, cols=samples). CSV/TSV with index in first column.")
    parser.add_argument("--phenotype_file", required=True, help="Path to phenotype table. Samples as index in first column.")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--gender_col", default="GENDER", help="Column name for gender in phenotype file (default: GENDER)")
    parser.add_argument("--age_col", default="V5AGE52", help="Column name for age in phenotype file (default: V5AGE52)")
    parser.add_argument("--transpose", action="store_true", help="Transpose truth matrix on load (use if rows=samples, cols=features)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_matrix(args.truth_a)
    if args.transpose:
        truth = truth.T

    # Remove optional 'groups' row/column if present
    if 'groups' in truth.index:
        truth = truth.drop(index='groups')
    if 'groups' in truth.columns:
        truth = truth.drop(columns=['groups'])

    # Ensure numeric
    truth = truth.apply(pd.to_numeric, errors='coerce')
    phenotypes = read_matrix(args.phenotype_file)

    # Align sample overlap upfront for speed (keep all, we filter per-phenotype later)
    common_samples_all = list(set(truth.columns) & set(phenotypes.index))
    if len(common_samples_all) < 10:
        raise ValueError("Fewer than 10 overlapping samples between truth matrix and phenotype file.")
    truth = truth.loc[:, common_samples_all]
    phenotypes = phenotypes.loc[common_samples_all]

    # Prepare covariates (gender_col, age_col)
    cov_gender, cov_age, gender_mapping = prepare_covariates(phenotypes, args.gender_col, args.age_col)

    # Detect phenotype column types (excluding covariates)
    # Include all phenotype columns (no exclusions) so that gender/age are also analyzed
    binary_cols, continuous_cols = detect_binary_and_continuous_phenotypes(phenotypes, exclude_cols=[])

    # Run associations
    binary_summary = run_binary_associations(truth, phenotypes, binary_cols, cov_gender, cov_age, output_dir, args.gender_col, args.age_col)
    continuous_summary = run_continuous_associations(truth, phenotypes, continuous_cols, cov_gender, cov_age, output_dir, args.gender_col, args.age_col)

    # Save summaries and top-5 reports
    if not binary_summary.empty:
        binary_summary.to_csv(output_dir / "summary_binary_all.csv", index=False)
        top5_binary = binary_summary.head(5)
        top5_binary.to_csv(output_dir / "summary_binary_top5.csv", index=False)
    else:
        top5_binary = pd.DataFrame(columns=["phenotype", "num_tested", "num_significant"])  # empty

    if not continuous_summary.empty:
        continuous_summary.to_csv(output_dir / "summary_continuous_all.csv", index=False)
        top5_cont = continuous_summary.head(5)
        top5_cont.to_csv(output_dir / "summary_continuous_top5.csv", index=False)
    else:
        top5_cont = pd.DataFrame(columns=["phenotype", "num_tested", "num_significant"])  # empty

    # Simple text report
    report_lines: List[str] = []
    report_lines.append("Top 5 binary phenotypes by number of significant protein hits (FDR<0.05):")
    if not top5_binary.empty:
        for _, row in top5_binary.iterrows():
            report_lines.append(f"  - {row['phenotype']}: {int(row['num_significant'])} of {int(row['num_tested'])}")
    else:
        report_lines.append("  (none)")
    report_lines.append("")
    report_lines.append("Top 5 continuous phenotypes by number of significant protein hits (FDR<0.05):")
    if not top5_cont.empty:
        for _, row in top5_cont.iterrows():
            report_lines.append(f"  - {row['phenotype']}: {int(row['num_significant'])} of {int(row['num_tested'])}")
    else:
        report_lines.append("  (none)")

    (output_dir / "summary.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()


