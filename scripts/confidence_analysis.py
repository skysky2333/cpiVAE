#!/usr/bin/env python3
"""
Confidence vs Performance (r) Analysis - Cross-Platform

Reads ground-truth matrix, imputed matrix, and per-cell confidence (CV) matrix,
aligns them, computes performance r (feature-wise and sample-wise), aggregates
confidence, and plots confidence vs r.

Expected matrix orientations:
- Truth: features as rows, samples as columns (common in this repo)
- Imputed: samples as rows (first column is sample ID), features as columns
- Confidence: samples as rows (first column is sample ID), features as columns

The script will attempt to auto-detect and transpose as needed based on overlap
of sample IDs and feature names.
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
    """Orient truth so that rows=features, cols=samples to match the code's expectations.
    Chooses orientation by maximizing sample ID overlap with imputed index.
    """
    imputed_samples = set(imputed_df.index)
    overlap_cols = len(imputed_samples & set(truth_df.columns))
    overlap_index = len(imputed_samples & set(truth_df.index))
    if overlap_index > overlap_cols:
        return truth_df.T
    return truth_df


def auto_orient_like_imputed(df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    """Orient df to match imputed_df (samples as rows, features as columns)."""
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
    """Align truth (features x samples), imputed (samples x features), confidence (samples x features)."""
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
    # Correlation between r and confidence
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
    p = argparse.ArgumentParser(description="Confidence vs Performance (r) Analysis - Cross-Platform (supports single or two-platform mode)")
    # Single-platform args (backwards compatible)
    p.add_argument("--truth", help="Ground truth matrix for target platform (features x samples)")
    p.add_argument("--imputed", help="Imputed matrix for target platform (samples x features; ID in first column)")
    p.add_argument("--confidence", help="Confidence (CV) matrix (samples x features; ID in first column)")
    p.add_argument("--platform_name", help="Display name for target platform")

    # Two-platform args
    p.add_argument("--truth_a", help="Ground truth matrix for Platform A (features x samples)")
    p.add_argument("--truth_b", help="Ground truth matrix for Platform B (features x samples)")
    p.add_argument("--imputed_a", help="Imputed matrix for Platform A (samples x features; ID in first column)")
    p.add_argument("--imputed_b", help="Imputed matrix for Platform B (samples x features; ID in first column)")
    p.add_argument("--confidence_a", help="Confidence (CV) matrix for Platform A (samples x features; ID in first column)")
    p.add_argument("--confidence_b", help="Confidence (CV) matrix for Platform B (samples x features; ID in first column)")
    p.add_argument("--platform_a_name", help="Display name for Platform A")
    p.add_argument("--platform_b_name", help="Display name for Platform B")

    p.add_argument("--method_name", default="VAE Imputation", help="Method display name")
    p.add_argument("--output_dir", default="confidence_analysis_output", help="Directory to save figures and tables")
    p.add_argument("--correlation", choices=["pearson", "spearman"], default="pearson", help="Correlation type for r")
    p.add_argument("--max_points", type=int, default=50000, help="Max datapoints to plot in cell-wise scatter (downsample if larger)")
    return p.parse_args()


def _compute_tables_for_platform(truth_df: pd.DataFrame, imputed_df: pd.DataFrame, conf_df: pd.DataFrame, correlation: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Auto-orient to: truth (features x samples), imputed/conf (samples x features)
    imputed_df = auto_orient_like_imputed(imputed_df, imputed_df)
    conf_df = auto_orient_like_imputed(conf_df, imputed_df)
    truth_df = auto_orient_truth(truth_df, imputed_df)

    # Align
    truth_fxs, imputed_sxf, conf_sxf = align_matrices(truth_df, imputed_df, conf_df)

    # Compute r
    feat_r = compute_feature_wise_r(truth_fxs, imputed_sxf, corr=correlation)
    samp_r = compute_sample_wise_r(truth_fxs, imputed_sxf, corr=correlation)

    # Aggregate confidence (mean)
    feat_conf = conf_sxf.mean(axis=0)  # per feature
    samp_conf = conf_sxf.mean(axis=1)  # per sample

    feat_table = pd.DataFrame({"r": feat_r, "confidence_mean": feat_conf}).dropna()
    samp_table = pd.DataFrame({"r": samp_r, "confidence_mean": samp_conf}).dropna()

    # Per-cell absolute error vs confidence (long form)
    truth_vals = truth_fxs.T.values  # (samples x features)
    imp_vals = imputed_sxf.values    # (samples x features)
    conf_vals = conf_sxf.values      # (samples x features)
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

    return feat_table, samp_table, truth_fxs, imputed_sxf, cell_df


def _scatter_dual(feat_table_a: pd.DataFrame, feat_table_b: pd.DataFrame, name_a: str, name_b: str, title_prefix: str, xlabel: str, ylabel: str, out_base: Path):
    combined = pd.concat([
        feat_table_a.assign(platform=name_a),
        feat_table_b.assign(platform=name_b),
    ])
    plt.figure(figsize=(7, 5))
    # Confidence on X-axis, correlation on Y-axis
    palette = sns.color_palette(n_colors=2)
    plat_to_color = {name_a: palette[0], name_b: palette[1]}
    sns.scatterplot(data=combined, x="confidence_median", y="r", hue="platform", s=12, alpha=0.6, edgecolor="none", palette=plat_to_color)
    # Add per-platform regression lines
    for plat_name, sub in combined.groupby("platform"):
        try:
            sns.regplot(data=sub, x="confidence_median", y="r", scatter=False, ci=None, color=plat_to_color.get(plat_name, "black"), line_kws={"linewidth": 1.5})
        except Exception:
            pass
    # Per-platform correlations between r and confidence
    y0 = 0.98
    for plat_name, sub in combined.groupby("platform"):
        if len(sub) >= 3:
            try:
                pr, _ = pearsonr(sub["r"], sub["confidence_median"])
                sr, _ = spearmanr(sub["r"], sub["confidence_median"])
                annot = f"{plat_name}: r={pr:.3f}, rho={sr:.3f}, N={len(sub)}"
            except Exception:
                annot = f"{plat_name}: N={len(sub)}"
            plt.gca().text(0.02, y0, annot, transform=plt.gca().transAxes, va="top", ha="left",
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
            y0 -= 0.12
    plt.title(title_prefix)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close()


def _scatter_dual_error(cell_df_a: pd.DataFrame, cell_df_b: pd.DataFrame, name_a: str, name_b: str, max_points: int, out_base: Path):
    # Downsample if too large
    def downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        return df.sample(n, random_state=42)

    cell_a = downsample(cell_df_a, max_points)
    cell_b = downsample(cell_df_b, max_points)

    combined = pd.concat([
        cell_a.assign(platform=name_a),
        cell_b.assign(platform=name_b),
    ])

    plt.figure(figsize=(7, 5))
    palette = sns.color_palette(n_colors=2)
    plat_to_color = {name_a: palette[0], name_b: palette[1]}
    sns.scatterplot(data=combined, x="confidence", y="abs_error", hue="platform", s=8, alpha=0.4, edgecolor="none", palette=plat_to_color)
    # Regression per platform
    for plat_name, sub in combined.groupby("platform"):
        try:
            sns.regplot(data=sub, x="confidence", y="abs_error", scatter=False, ci=None, color=plat_to_color.get(plat_name, "black"), line_kws={"linewidth": 1.5})
        except Exception:
            pass
    # Annotation per platform
    y0 = 0.98
    for plat_name, sub in combined.groupby("platform"):
        if len(sub) >= 3:
            try:
                pr, _ = pearsonr(sub["abs_error"], sub["confidence"])
                sr, _ = spearmanr(sub["abs_error"], sub["confidence"])
                annot = f"{plat_name}: r={pr:.3f}, rho={sr:.3f}, N={len(sub)}"
            except Exception:
                annot = f"{plat_name}: N={len(sub)}"
            plt.gca().text(0.02, y0, annot, transform=plt.gca().transAxes, va="top", ha="left",
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
            y0 -= 0.12
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Cell-wise: Confidence vs Absolute Error")
    plt.xlabel("Confidence (CV)")
    plt.ylabel("Absolute Error |truth - imputed|")
    plt.grid(alpha=0.3)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    # Two-platform mode if both sets provided
    two_platform = all([
        args.truth_a, args.truth_b, args.imputed_a, args.imputed_b, args.confidence_a, args.confidence_b,
        args.platform_a_name, args.platform_b_name,
    ])

    if two_platform:
        print("Loading matrices for both platforms...")
        truth_a = read_csv(args.truth_a)
        truth_b = read_csv(args.truth_b)
        imputed_a = read_csv(args.imputed_a)
        imputed_b = read_csv(args.imputed_b)
        conf_a = read_csv(args.confidence_a)
        conf_b = read_csv(args.confidence_b)

        print("Computing tables for Platform A...")
        feat_a, samp_a, truth_fxs_a, imputed_sxf_a, cell_a = _compute_tables_for_platform(truth_a, imputed_a, conf_a, args.correlation)
        feat_a.to_csv(out_dir / "data" / "feature_confidence_vs_r_platform_A.csv")
        samp_a.to_csv(out_dir / "data" / "sample_confidence_vs_r_platform_A.csv")

        print("Computing tables for Platform B...")
        feat_b, samp_b, truth_fxs_b, imputed_sxf_b, cell_b = _compute_tables_for_platform(truth_b, imputed_b, conf_b, args.correlation)
        feat_b.to_csv(out_dir / "data" / "feature_confidence_vs_r_platform_B.csv")
        samp_b.to_csv(out_dir / "data" / "sample_confidence_vs_r_platform_B.csv")
        # Save downsampled cell-level previews
        try:
            cell_a.sample(min(len(cell_a), args.max_points), random_state=42).to_csv(out_dir / "data" / "cell_confidence_vs_abs_error_preview_platform_A.csv", index=False)
            cell_b.sample(min(len(cell_b), args.max_points), random_state=42).to_csv(out_dir / "data" / "cell_confidence_vs_abs_error_preview_platform_B.csv", index=False)
        except Exception:
            pass

        print("Creating combined plots (both platforms)...")
        # Attach mean values for coloring
        feat_a = feat_a.copy()
        feat_b = feat_b.copy()
        samp_a = samp_a.copy()
        samp_b = samp_b.copy()
        feat_a["mean_value"] = truth_fxs_a.mean(axis=1).reindex(feat_a.index)
        feat_b["mean_value"] = truth_fxs_b.mean(axis=1).reindex(feat_b.index)
        samp_a["mean_value"] = truth_fxs_a.mean(axis=0).reindex(samp_a.index)
        samp_b["mean_value"] = truth_fxs_b.mean(axis=0).reindex(samp_b.index)

        # Create two-panel subplots per plot type (A and B split), colored by mean_value
        def subplot_two_platform(feat_or_samp: str, table_a: pd.DataFrame, table_b: pd.DataFrame, fig_base: str):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            for ax, (plat_name, tbl) in zip(axes, [(args.platform_a_name, table_a), (args.platform_b_name, table_b)]):
                scatter_df = tbl.rename(columns={"confidence_mean": "confidence"}).dropna(subset=["confidence", "r", "mean_value"]) 
                sc = ax.scatter(scatter_df["confidence"], scatter_df["r"], c=scatter_df["mean_value"], cmap="viridis", s=12, alpha=0.7, edgecolors='none')
                try:
                    sns.regplot(data=scatter_df, x="confidence", y="r", scatter=False, ci=None, color="black", line_kws={"linewidth": 1.5}, ax=ax)
                except Exception:
                    pass
                ax.set_title(f"{plat_name}")
                ax.set_xlabel("Confidence (CV)")
                ax.set_ylabel(f"{args.correlation.title()} r" if ax is axes[0] else "")
                ax.grid(alpha=0.3)
                # Make subplot square (excluding colorbar)
                try:
                    ax.set_box_aspect(1)
                except Exception:
                    pass
                # Individual colorbar per platform
                cbar = fig.colorbar(sc, ax=ax, location='right', fraction=0.046, pad=0.04)
                cbar.set_label("Mean value")
            fig.suptitle(f"{feat_or_samp.title()}-wise: Confidence vs {args.correlation.title()} r\n{args.method_name}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_dir / "figures" / f"{fig_base}.pdf")
            fig.savefig(out_dir / "figures" / f"{fig_base}.png", dpi=300)
            plt.close(fig)

        subplot_two_platform("Feature", feat_a.rename(columns={"confidence_mean": "confidence_mean"}), feat_b.rename(columns={"confidence_mean": "confidence_mean"}), "confidence_vs_r_featurewise_split")
        subplot_two_platform("Sample", samp_a.rename(columns={"confidence_mean": "confidence_mean"}), samp_b.rename(columns={"confidence_mean": "confidence_mean"}), "confidence_vs_r_samplewise_split")

        # Cell-wise split plot (two subplots), color by feature mean
        def subplot_two_platform_error(cell_df_a: pd.DataFrame, cell_df_b: pd.DataFrame, truth_fxs_a: pd.DataFrame, truth_fxs_b: pd.DataFrame, fig_base: str):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
            for ax, (plat_name, cell_df, truth_fxs) in zip(axes, [
                (args.platform_a_name, cell_df_a, truth_fxs_a),
                (args.platform_b_name, cell_df_b, truth_fxs_b),
            ]):
                # Filter positives for log-log
                df = cell_df[(cell_df["confidence"] > 0) & (cell_df["abs_error"] > 0)].copy()
                # Map feature mean
                feature_mean = truth_fxs.mean(axis=1)
                df["mean_value"] = df["feature"].map(feature_mean)
                # Downsample
                if len(df) > args.max_points:
                    df = df.sample(args.max_points, random_state=42)
                sc = ax.scatter(df["confidence"], df["abs_error"], c=df["mean_value"], cmap="viridis", s=8, alpha=0.4, edgecolors='none')
                try:
                    sns.regplot(data=df, x="confidence", y="abs_error", scatter=False, ci=None, color="black", line_kws={"linewidth": 1.5}, ax=ax)
                except Exception:
                    pass
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{plat_name}")
                ax.set_xlabel("Confidence (CV)")
                ax.set_ylabel("Absolute Error |truth - imputed|" if ax is axes[0] else "")
                ax.grid(alpha=0.3)
                # Make subplot square and add individual colorbar
                try:
                    ax.set_box_aspect(1)
                except Exception:
                    pass
                cbar = fig.colorbar(sc, ax=ax, location='right', fraction=0.046, pad=0.04)
                cbar.set_label("Mean feature value")
            fig.suptitle("Cell-wise: Confidence vs Absolute Error")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_dir / "figures" / f"{fig_base}.pdf")
            fig.savefig(out_dir / "figures" / f"{fig_base}.png", dpi=300)
            plt.close(fig)

        subplot_two_platform_error(cell_a, cell_b, truth_fxs_a, truth_fxs_b, "confidence_vs_abs_error_cells_split")

        print(f"Saved outputs to: {out_dir}")
        print("Done.")
        return

    # Fallback: single-platform mode
    required_single = [args.truth, args.imputed, args.confidence, args.platform_name]
    if not all(required_single):
        raise ValueError("Provide either two-platform arguments or all single-platform arguments.")

    print("Loading matrices (single platform)...")
    truth_df = read_csv(args.truth)
    imputed_df = read_csv(args.imputed)
    conf_df = read_csv(args.confidence)

    feat_table, samp_table, truth_fxs, imputed_sxf, cell_df = _compute_tables_for_platform(truth_df, imputed_df, conf_df, args.correlation)
    feat_table.to_csv(out_dir / "data" / "feature_confidence_vs_r.csv")
    samp_table.to_csv(out_dir / "data" / "sample_confidence_vs_r.csv")
    try:
        cell_df.sample(min(len(cell_df), args.max_points), random_state=42).to_csv(out_dir / "data" / "cell_confidence_vs_abs_error_preview.csv", index=False)
    except Exception:
        pass

    print("Creating plots (single platform)...")
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

    # Cell-wise plot for single platform
    try:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=cell_df.sample(min(len(cell_df), args.max_points), random_state=42),
                        x="confidence", y="abs_error", s=8, alpha=0.4, edgecolor="none")
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


