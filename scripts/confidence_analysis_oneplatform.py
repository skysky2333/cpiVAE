#!/usr/bin/env python3
"""
Confidence vs Performance (r) Analysis - Single Platform

Reads ground-truth matrix, imputed matrix, and per-cell confidence (CV) matrix
for a single platform, aligns them, computes r (feature-wise and sample-wise),
aggregates confidence, and plots confidence vs r.

Expected matrix orientations:
- Truth: features as rows, samples as columns (common in this repo)
- Imputed: samples as rows (first column is sample ID), features as columns
- Confidence: samples as rows (first column is sample ID), features as columns
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def read_csv(path: str) -> pd.DataFrame:
    sep = "\t" if str(path).lower().endswith(".txt") else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def auto_orient_truth(truth_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    imputed_samples = set(imputed_df.index)
    overlap_cols = len(imputed_samples & set(truth_df.columns))
    overlap_index = len(imputed_samples & set(truth_df.index))
    if overlap_index > overlap_cols:
        return truth_df.T
    return truth_df


def auto_orient_like_imputed(df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    overlap_index = len(set(df.index) & set(imputed_df.index))
    overlap_cols = len(set(df.columns) & set(imputed_df.index))
    if overlap_cols > overlap_index:
        return df.T
    return df


def align_matrices(
    truth_fxs: pd.DataFrame,
    imputed_sxf: pd.DataFrame,
    conf_sxf: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_samples = (
        set(truth_fxs.columns) & set(imputed_sxf.index) & set(conf_sxf.index)
    )
    common_features = (
        set(truth_fxs.index) & set(imputed_sxf.columns) & set(conf_sxf.columns)
    )
    if not common_samples:
        raise ValueError("No common samples among truth, imputed, and confidence matrices")
    if not common_features:
        raise ValueError("No common features among truth, imputed, and confidence matrices")

    samples = sorted(common_samples)
    features = sorted(common_features)

    truth_fxs = truth_fxs.loc[features, samples]
    imputed_sxf = imputed_sxf.loc[samples, features]
    conf_sxf = conf_sxf.loc[samples, features]
    return truth_fxs, imputed_sxf, conf_sxf


def compute_feature_wise_r(truth_fxs: pd.DataFrame, imputed_sxf: pd.DataFrame, corr: str = "pearson") -> pd.Series:
    vals = {}
    for feat in truth_fxs.index:
        t = truth_fxs.loc[feat].values
        p = imputed_sxf[feat].values
        mask = ~(np.isnan(t) | np.isnan(p))
        if mask.sum() < 3:
            vals[feat] = np.nan
            continue
        t_c, p_c = t[mask], p[mask]
        if corr == "spearman":
            r, _ = spearmanr(t_c, p_c)
        else:
            r, _ = pearsonr(t_c, p_c)
        vals[feat] = r
    return pd.Series(vals)


def compute_sample_wise_r(truth_fxs: pd.DataFrame, imputed_sxf: pd.DataFrame, corr: str = "pearson") -> pd.Series:
    vals = {}
    for sample in truth_fxs.columns:
        t = truth_fxs[sample].values
        p = imputed_sxf.loc[sample].values
        mask = ~(np.isnan(t) | np.isnan(p))
        if mask.sum() < 3:
            vals[sample] = np.nan
            continue
        t_c, p_c = t[mask], p[mask]
        if corr == "spearman":
            r, _ = spearmanr(t_c, p_c)
        else:
            r, _ = pearsonr(t_c, p_c)
        vals[sample] = r
    return pd.Series(vals)


def scatter_conf_vs_r(r_vals: pd.Series, conf_vals: pd.Series, title: str, xlabel: str, ylabel: str, out_base: Path):
    df = pd.DataFrame({"r": r_vals, "confidence": conf_vals}).dropna()
    plt.figure(figsize=(6, 5))
    # Confidence on X-axis, correlation on Y-axis
    sns.scatterplot(data=df, x="confidence", y="r", s=12, alpha=0.6, edgecolor="none")
    # Add linear regression line
    try:
        sns.regplot(data=df, x="confidence", y="r", scatter=False, ci=None, color="black", line_kws={"linewidth": 1.5})
    except Exception:
        pass
    if len(df) >= 3:
        try:
            pr, _ = pearsonr(df["r"], df["confidence"])
            sr, _ = spearmanr(df["r"], df["confidence"])
            annot = f"Pearson r={pr:.3f}\nSpearman rho={sr:.3f}\nN={len(df)}"
        except Exception:
            annot = f"N={len(df)}"
        plt.gca().text(0.02, 0.98, annot, transform=plt.gca().transAxes, va="top", ha="left",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Confidence vs Performance (r) Analysis - Single Platform")
    p.add_argument("--truth", required=True, help="Ground truth matrix (features x samples)")
    p.add_argument("--imputed", required=True, help="Imputed matrix (samples x features; ID in first column)")
    p.add_argument("--confidence", required=True, help="Confidence (CV) matrix (samples x features; ID in first column)")
    p.add_argument("--platform_name", required=True, help="Display name for platform")
    p.add_argument("--method_name", default="VAE Imputation", help="Method display name")
    p.add_argument("--output_dir", default="confidence_analysis_oneplatform_output", help="Directory to save outputs")
    p.add_argument("--correlation", choices=["pearson", "spearman"], default="pearson", help="Correlation type for r")
    p.add_argument("--max_points", type=int, default=50000, help="Max datapoints to plot in cell-wise scatter (downsample if larger)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    print("Loading matrices...")
    truth_df = read_csv(args.truth)
    imputed_df = read_csv(args.imputed)
    conf_df = read_csv(args.confidence)

    imputed_df = auto_orient_like_imputed(imputed_df, imputed_df)
    conf_df = auto_orient_like_imputed(conf_df, imputed_df)
    truth_df = auto_orient_truth(truth_df, imputed_df)

    print(f"Truth shape (fxs): {truth_df.shape}")
    print(f"Imputed shape (sxf): {imputed_df.shape}")
    print(f"Confidence shape (sxf): {conf_df.shape}")

    truth_fxs, imputed_sxf, conf_sxf = align_matrices(truth_df, imputed_df, conf_df)
    print(f"Aligned shapes -> truth: {truth_fxs.shape}, imputed: {imputed_sxf.shape}, confidence: {conf_sxf.shape}")

    print("Computing r...")
    feat_r = compute_feature_wise_r(truth_fxs, imputed_sxf, corr=args.correlation)
    samp_r = compute_sample_wise_r(truth_fxs, imputed_sxf, corr=args.correlation)

    print("Aggregating confidence (mean)...")
    feat_conf = conf_sxf.mean(axis=0)
    samp_conf = conf_sxf.mean(axis=1)

    feat_table = pd.DataFrame({"r": feat_r, "confidence_mean": feat_conf}).dropna()
    samp_table = pd.DataFrame({"r": samp_r, "confidence_mean": samp_conf}).dropna()

    # Per-cell absolute error vs confidence (long form)
    truth_vals = truth_fxs.T.values
    imp_vals = imputed_sxf.values
    conf_vals = conf_sxf.values
    abs_err = np.abs(imp_vals - truth_vals)
    sample_ids = list(truth_fxs.columns)
    feature_ids = list(truth_fxs.index)
    cell_df = pd.DataFrame({
        "confidence": conf_vals.flatten(),
        "abs_error": abs_err.flatten(),
        "sample": np.repeat(sample_ids, len(feature_ids)),
        "feature": np.tile(feature_ids, len(sample_ids)),
    })
    cell_df = cell_df.replace([np.inf, -np.inf], np.nan).dropna()
    feat_table.to_csv(out_dir / "data" / "feature_confidence_vs_r.csv")
    samp_table.to_csv(out_dir / "data" / "sample_confidence_vs_r.csv")
    # Also save a small downsampled cell-level table for quick inspection
    try:
        cell_preview = cell_df.sample(min(len(cell_df), args.max_points), random_state=42)
        cell_preview.to_csv(out_dir / "data" / "cell_confidence_vs_abs_error_preview.csv", index=False)
    except Exception:
        pass

    print("Creating plots...")
    scatter_conf_vs_r(
        r_vals=feat_table["r"],
        conf_vals=feat_table["confidence_mean"],
        title=f"Feature-wise: Confidence vs {args.correlation.title()} r\n{args.method_name} on {args.platform_name}",
        xlabel="Confidence (CV)",
        ylabel=f"{args.correlation.title()} r (truth vs imputed)",
        out_base=out_dir / "figures" / "confidence_vs_r_featurewise",
    )

    scatter_conf_vs_r(
        r_vals=samp_table["r"],
        conf_vals=samp_table["confidence_mean"],
        title=f"Sample-wise: Confidence vs {args.correlation.title()} r\n{args.method_name} on {args.platform_name}",
        xlabel="Confidence (CV)",
        ylabel=f"{args.correlation.title()} r (truth vs imputed)",
        out_base=out_dir / "figures" / "confidence_vs_r_samplewise",
    )

    # Cell-wise confidence vs absolute error plot
    try:
        plt.figure(figsize=(6, 5))
        # Filter positive values for log-log plotting
        plot_df = cell_df[(cell_df["confidence"] > 0) & (cell_df["abs_error"] > 0)]
        sample_df = plot_df.sample(min(len(plot_df), args.max_points), random_state=42)
        sns.scatterplot(data=sample_df, x="confidence", y="abs_error", s=8, alpha=0.4, edgecolor="none")
        try:
            sns.regplot(data=plot_df, x="confidence", y="abs_error", scatter=False, ci=None, color="black", line_kws={"linewidth": 1.5})
        except Exception:
            pass
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Cell-wise: Confidence vs Absolute Error")
        plt.xlabel("Confidence (CV)")
        plt.ylabel("Absolute Error |truth - imputed|")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "confidence_vs_abs_error_cells.pdf")
        plt.savefig(out_dir / "figures" / "confidence_vs_abs_error_cells.png", dpi=300)
        plt.close()
    except Exception:
        pass

    print(f"Saved outputs to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()


