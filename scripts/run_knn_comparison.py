#!/usr/bin/env python3
"""
KNN Baseline Comparison Tool for Joint VAE Cross-Platform Proteomics Imputation.

This script orchestrates comprehensive KNN baseline experiments across multiple
k values and parameter configurations, enabling systematic comparison with Joint VAE
models for cross-platform proteomics data imputation tasks.

Key Features:
    - Multi-parameter grid search across k values and kernel functions
    - Cross-validation and train/test split evaluation modes
    - Automated result collection and performance reporting
    - Cross-platform imputation using optimal parameters
    - Support for log transformations and advanced kernel weighting

Usage:
    python run_knn_comparison.py --platform_a data/olink.csv --platform_b data/somascan.csv
        --k_values 5 10 20 50 --kernel gaussian --cv_folds 5
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path
import json
import sys
import numpy as np


def parse_arguments():
    """Parse and validate command line arguments for KNN comparison experiments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments including platform files,
            k values, kernel parameters, and cross-validation settings.
    """
    parser = argparse.ArgumentParser(description='Run KNN baseline with multiple configurations')
    
    parser.add_argument(
        '--platform_a', 
        type=str, 
        required=True,
        help='Path to platform A data file (CSV or TXT)'
    )
    parser.add_argument(
        '--platform_b', 
        type=str, 
        required=True,
        help='Path to platform B data file (CSV or TXT)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/knn_comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--k_values', 
        type=int,
        nargs='+',
        default=[3, 5, 7, 10, 15, 30, 50, 100, 200],
        help='List of k values to test (default: 3 5 7 10 15 30 50 100 200)'
    )
    parser.add_argument(
        '--cv_folds', 
        type=int, 
        default=5,
        help='Number of cross validation folds (default: 5, set to 0 to use train/test split)'
    )
    parser.add_argument(
        '--kernel', 
        type=str, 
        default='distance',
        choices=['uniform', 'distance', 'gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial'],
        help='Kernel/weighting function for KNN (default: distance)'
    )
    parser.add_argument(
        '--bandwidth_values', 
        type=float,
        nargs='+',
        default=None,
        help='List of bandwidth values to test for advanced kernels (default: automatic range based on kernel)'
    )
    parser.add_argument(
        '--bandwidth', 
        type=float, 
        default=None,
        help='Single bandwidth value (deprecated, use --bandwidth_values instead)'
    )
    parser.add_argument(
        '--polynomial_degree', 
        type=int, 
        default=2,
        help='Degree parameter for polynomial kernel (default: 2)'
    )
    parser.add_argument(
        '--log_transform_a', 
        action='store_true',
        help='Apply log transformation to platform A data'
    )
    parser.add_argument(
        '--log_transform_b', 
        action='store_true',
        help='Apply log transformation to platform B data'
    )
    parser.add_argument(
        '--log_epsilon', 
        type=float, 
        default=1e-8,
        help='Small value added to ensure positive values for log transformation (default: 1e-8)'
    )
    parser.add_argument(
        '--platform_impute', 
        type=str, 
        default=None,
        help='Path to platform file that needs cross-imputation (CSV or TXT, same format as platform A or B)'
    )
    parser.add_argument(
        '--impute_target', 
        type=str, 
        choices=['a', 'b'],
        default=None,
        help='Target platform for imputation: "a" to impute as platform A, "b" to impute as platform B'
    )
    
    return parser.parse_args()


def run_knn_experiment(platform_a, platform_b, output_dir, k_value, bandwidth_value, cv_folds=None, kernel='distance', polynomial_degree=2, log_transform_a=False, log_transform_b=False, log_epsilon=1e-8):
    """Execute a single KNN experiment with specified parameters.
    
    Args:
        platform_a (str): Path to platform A data file
        platform_b (str): Path to platform B data file
        output_dir (str): Directory to save experiment results
        k_value (int): Number of neighbors for KNN
        bandwidth_value (float): Bandwidth parameter for advanced kernels
        cv_folds (int, optional): Number of cross-validation folds
        kernel (str): Kernel function type for weighting neighbors
        polynomial_degree (int): Degree for polynomial kernel
        log_transform_a (bool): Whether to log-transform platform A data
        log_transform_b (bool): Whether to log-transform platform B data
        log_epsilon (float): Small value for log transformation stability
    
    Returns:
        tuple: (success_status, experiment_directory_path)
    """
    # Determine if bandwidth is relevant for this kernel
    uses_bandwidth = kernel in ['gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial']
    
    if cv_folds and cv_folds > 0:
        if uses_bandwidth:
            print(f"Running KNN experiment with k={k_value}, bandwidth={bandwidth_value}, {cv_folds}-fold CV, kernel={kernel}...")
        else:
            print(f"Running KNN experiment with k={k_value}, {cv_folds}-fold CV, kernel={kernel}...")
    else:
        if uses_bandwidth:
            print(f"Running KNN experiment with k={k_value}, bandwidth={bandwidth_value}, train/test split, kernel={kernel}...")
        else:
            print(f"Running KNN experiment with k={k_value}, train/test split, kernel={kernel}...")
    
    if uses_bandwidth:
        exp_output_dir = Path(output_dir) / f'k_{k_value}_bw_{bandwidth_value}'
    else:
        exp_output_dir = Path(output_dir) / f'k_{k_value}'
    cmd = [
        'python', 'scripts/knn_baseline.py',
        '--platform_a', platform_a,
        '--platform_b', platform_b,
        '--output_dir', str(exp_output_dir),
        '--n_neighbors', str(k_value),
        '--kernel', kernel,
        '--bandwidth', str(bandwidth_value),
        '--polynomial_degree', str(polynomial_degree),
        '--log_epsilon', str(log_epsilon)
    ]
    
    if log_transform_a:
        cmd.append('--log_transform_a')
    if log_transform_b:
        cmd.append('--log_transform_b')
    
    if cv_folds and cv_folds > 0:
        cmd.extend(['--cv_folds', str(cv_folds)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ k={k_value} completed successfully")
        return True, exp_output_dir
    except subprocess.CalledProcessError as e:
        print(f"✗ k={k_value} failed:")
        print(f"  Error: {e}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return False, None


def collect_results(output_dir, experiment_configs, is_cv=False):
    """Aggregate results from all completed KNN experiments.
    
    Args:
        output_dir (str): Directory containing experiment results
        experiment_configs (list): List of (k, bandwidth) tuples that were tested
        is_cv (bool): Whether cross-validation was used
    
    Returns:
        pd.DataFrame or None: Combined results from all experiments, or None if no results found
    """
    results = []
    
    for config in experiment_configs:
        if isinstance(config, tuple):
            k, bandwidth = config
            exp_dir = Path(output_dir) / f'k_{k}_bw_{bandwidth}'
        else:
            # Backward compatibility for k-only experiments
            k = config
            bandwidth = None
            exp_dir = Path(output_dir) / f'k_{k}'
        
        if is_cv:
            summary_file = exp_dir / 'knn_baseline_cv_summary.csv'
        else:
            summary_file = exp_dir / 'knn_baseline_summary.csv'
        
        if summary_file.exists():
            summary = pd.read_csv(summary_file)
            summary['k_value'] = k
            if bandwidth is not None:
                summary['bandwidth'] = bandwidth
            results.append(summary)
        else:
            if bandwidth is not None:
                print(f"Warning: Results for k={k}, bandwidth={bandwidth} not found")
            else:
                print(f"Warning: Results for k={k} not found")
    
    if results:
        combined_results = pd.concat(results, ignore_index=True)
        return combined_results
    else:
        return None


def create_comparison_report(results_df, output_dir, is_cv=False):
    """Generate comprehensive comparison report with best performing configurations.
    
    Args:
        results_df (pd.DataFrame): Combined results from all experiments
        output_dir (str): Directory to save the report files
        is_cv (bool): Whether cross-validation was used
    
    Returns:
        dict: Detailed report containing best configurations and performance metrics
    """
    if results_df is None or len(results_df) == 0:
        print("No results to create report")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if is_cv:
        results_df.to_csv(output_path / 'knn_comparison_cv_summary.csv', index=False)
    else:
        results_df.to_csv(output_path / 'knn_comparison_summary.csv', index=False)
    
    # Check if bandwidth column exists (for advanced kernels)
    has_bandwidth = 'bandwidth' in results_df.columns
    
    if is_cv:
        a_to_b_col = 'A_to_B_R2_mean'
        b_to_a_col = 'B_to_A_R2_mean'
        a_to_b_corr_col = 'A_to_B_correlation_mean'
        b_to_a_corr_col = 'B_to_A_correlation_mean'
        a_to_b_feature_col = 'A_to_B_mean_feature_R2_mean'
        b_to_a_feature_col = 'B_to_A_mean_feature_R2_mean'
        a_to_b_feature_corr_col = 'A_to_B_mean_feature_correlation_mean'
        b_to_a_feature_corr_col = 'B_to_A_mean_feature_correlation_mean'
        a_to_b_frac_col = 'A_to_B_fraction_R2_above_0.5_mean'
        b_to_a_frac_col = 'B_to_A_fraction_R2_above_0.5_mean'
        a_to_b_frac_corr_col = 'A_to_B_fraction_corr_above_0.6_mean'
        b_to_a_frac_corr_col = 'B_to_A_fraction_corr_above_0.6_mean'
    else:
        a_to_b_col = 'A_to_B_R2'
        b_to_a_col = 'B_to_A_R2'
        a_to_b_corr_col = 'A_to_B_correlation'
        b_to_a_corr_col = 'B_to_A_correlation'
        a_to_b_feature_col = 'A_to_B_mean_feature_R2'
        b_to_a_feature_col = 'B_to_A_mean_feature_R2'
        a_to_b_feature_corr_col = 'A_to_B_mean_feature_correlation'
        b_to_a_feature_corr_col = 'B_to_A_mean_feature_correlation'
        a_to_b_frac_col = 'A_to_B_fraction_R2_above_0.5'
        b_to_a_frac_col = 'B_to_A_fraction_R2_above_0.5'
        a_to_b_frac_corr_col = 'A_to_B_fraction_corr_above_0.6'
        b_to_a_frac_corr_col = 'B_to_A_fraction_corr_above_0.6'
    best_a_to_b = results_df.loc[results_df[a_to_b_col].idxmax()]
    best_b_to_a = results_df.loc[results_df[b_to_a_col].idxmax()]
    best_a_to_b_corr = results_df.loc[results_df[a_to_b_corr_col].idxmax()]
    best_b_to_a_corr = results_df.loc[results_df[b_to_a_corr_col].idxmax()]
    
    report = {
        'evaluation_type': 'cross_validation' if is_cv else 'train_test_split',
        'has_bandwidth_search': has_bandwidth,
        'best_a_to_b_r2': {
            'k_value': int(best_a_to_b['k_value']),
            'bandwidth': float(best_a_to_b['bandwidth']) if has_bandwidth else None,
            'r2_score': float(best_a_to_b[a_to_b_col]),
            'correlation': float(best_a_to_b[a_to_b_corr_col]),
            'mean_feature_r2': float(best_a_to_b[a_to_b_feature_col]),
            'mean_feature_correlation': float(best_a_to_b.get(a_to_b_feature_corr_col, 0.0)),
            'fraction_r2_above_0.5': float(best_a_to_b[a_to_b_frac_col]),
            'fraction_corr_above_0.6': float(best_a_to_b.get(a_to_b_frac_corr_col, 0.0))
        },
        'best_b_to_a_r2': {
            'k_value': int(best_b_to_a['k_value']),
            'bandwidth': float(best_b_to_a['bandwidth']) if has_bandwidth else None,
            'r2_score': float(best_b_to_a[b_to_a_col]),
            'correlation': float(best_b_to_a[b_to_a_corr_col]),
            'mean_feature_r2': float(best_b_to_a[b_to_a_feature_col]),
            'mean_feature_correlation': float(best_b_to_a.get(b_to_a_feature_corr_col, 0.0)),
            'fraction_r2_above_0.5': float(best_b_to_a[b_to_a_frac_col]),
            'fraction_corr_above_0.6': float(best_b_to_a.get(b_to_a_frac_corr_col, 0.0))
        },
        'best_a_to_b_correlation': {
            'k_value': int(best_a_to_b_corr['k_value']),
            'bandwidth': float(best_a_to_b_corr['bandwidth']) if has_bandwidth else None,
            'r2_score': float(best_a_to_b_corr[a_to_b_col]),
            'correlation': float(best_a_to_b_corr[a_to_b_corr_col]),
            'mean_feature_r2': float(best_a_to_b_corr[a_to_b_feature_col]),
            'mean_feature_correlation': float(best_a_to_b_corr.get(a_to_b_feature_corr_col, 0.0)),
            'fraction_r2_above_0.5': float(best_a_to_b_corr[a_to_b_frac_col]),
            'fraction_corr_above_0.6': float(best_a_to_b_corr.get(a_to_b_frac_corr_col, 0.0))
        },
        'best_b_to_a_correlation': {
            'k_value': int(best_b_to_a_corr['k_value']),
            'bandwidth': float(best_b_to_a_corr['bandwidth']) if has_bandwidth else None,
            'r2_score': float(best_b_to_a_corr[b_to_a_col]),
            'correlation': float(best_b_to_a_corr[b_to_a_corr_col]),
            'mean_feature_r2': float(best_b_to_a_corr[b_to_a_feature_col]),
            'mean_feature_correlation': float(best_b_to_a_corr.get(b_to_a_feature_corr_col, 0.0)),
            'fraction_r2_above_0.5': float(best_b_to_a_corr[b_to_a_frac_col]),
            'fraction_corr_above_0.6': float(best_b_to_a_corr.get(b_to_a_frac_corr_col, 0.0))
        },
        'all_results': results_df.to_dict('records')
    }
    
    if is_cv:
        report['cv_info'] = {
            'n_folds': int(results_df['n_folds'].iloc[0]) if 'n_folds' in results_df.columns else 5,
            'best_a_to_b_r2_std': float(best_a_to_b['A_to_B_R2_std']) if 'A_to_B_R2_std' in best_a_to_b else None,
            'best_b_to_a_r2_std': float(best_b_to_a['B_to_A_R2_std']) if 'B_to_A_R2_std' in best_b_to_a else None,
            'best_a_to_b_corr_std': float(best_a_to_b_corr['A_to_B_correlation_std']) if 'A_to_B_correlation_std' in best_a_to_b_corr else None,
            'best_b_to_a_corr_std': float(best_b_to_a_corr['B_to_A_correlation_std']) if 'B_to_A_correlation_std' in best_b_to_a_corr else None,
        }
    
    report_filename = 'comparison_report_cv.json' if is_cv else 'comparison_report.json'
    with open(output_path / report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    if is_cv:
        print("KNN CROSS VALIDATION COMPARISON SUMMARY")
    else:
        print("KNN COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nBest A → B Performance (by R²):")
    print(f"  k = {best_a_to_b['k_value']}")
    if has_bandwidth:
        print(f"  bandwidth = {best_a_to_b['bandwidth']}")
    if is_cv:
        print(f"  Overall R² = {best_a_to_b[a_to_b_col]:.4f} ± {best_a_to_b['A_to_B_R2_std']:.4f}")
        print(f"  Overall Correlation = {best_a_to_b[a_to_b_corr_col]:.4f} ± {best_a_to_b['A_to_B_correlation_std']:.4f}")
        print(f"  Mean Feature R² = {best_a_to_b[a_to_b_feature_col]:.4f} ± {best_a_to_b['A_to_B_mean_feature_R2_std']:.4f}")
        print(f"  Mean Feature Correlation = {best_a_to_b.get(a_to_b_feature_corr_col, 0.0):.4f} ± {best_a_to_b.get('A_to_B_mean_feature_correlation_std', 0.0):.4f}")
        print(f"  Features with R² > 0.5 = {best_a_to_b[a_to_b_frac_col]:.2%} ± {best_a_to_b['A_to_B_fraction_R2_above_0.5_std']:.2%}")
        print(f"  Features with Correlation > 0.6 = {best_a_to_b.get(a_to_b_frac_corr_col, 0.0):.2%} ± {best_a_to_b.get('A_to_B_fraction_corr_above_0.6_std', 0.0):.2%}")
    else:
        print(f"  Overall R² = {best_a_to_b[a_to_b_col]:.4f}")
        print(f"  Overall Correlation = {best_a_to_b[a_to_b_corr_col]:.4f}")
        print(f"  Mean Feature R² = {best_a_to_b[a_to_b_feature_col]:.4f}")
        print(f"  Mean Feature Correlation = {best_a_to_b.get(a_to_b_feature_corr_col, 0.0):.4f}")
        print(f"  Features with R² > 0.5 = {best_a_to_b[a_to_b_frac_col]:.2%}")
        print(f"  Features with Correlation > 0.6 = {best_a_to_b.get(a_to_b_frac_corr_col, 0.0):.2%}")
    
    print(f"\nBest B → A Performance (by R²):")
    print(f"  k = {best_b_to_a['k_value']}")
    if has_bandwidth:
        print(f"  bandwidth = {best_b_to_a['bandwidth']}")
    if is_cv:
        print(f"  Overall R² = {best_b_to_a[b_to_a_col]:.4f} ± {best_b_to_a['B_to_A_R2_std']:.4f}")
        print(f"  Overall Correlation = {best_b_to_a[b_to_a_corr_col]:.4f} ± {best_b_to_a['B_to_A_correlation_std']:.4f}")
        print(f"  Mean Feature R² = {best_b_to_a[b_to_a_feature_col]:.4f} ± {best_b_to_a['B_to_A_mean_feature_R2_std']:.4f}")
        print(f"  Mean Feature Correlation = {best_b_to_a.get(b_to_a_feature_corr_col, 0.0):.4f} ± {best_b_to_a.get('B_to_A_mean_feature_correlation_std', 0.0):.4f}")
        print(f"  Features with R² > 0.5 = {best_b_to_a[b_to_a_frac_col]:.2%} ± {best_b_to_a['B_to_A_fraction_R2_above_0.5_std']:.2%}")
        print(f"  Features with Correlation > 0.6 = {best_b_to_a.get(b_to_a_frac_corr_col, 0.0):.2%} ± {best_b_to_a.get('B_to_A_fraction_corr_above_0.6_std', 0.0):.2%}")
    else:
        print(f"  Overall R² = {best_b_to_a[b_to_a_col]:.4f}")
        print(f"  Overall Correlation = {best_b_to_a[b_to_a_corr_col]:.4f}")
        print(f"  Mean Feature R² = {best_b_to_a[b_to_a_feature_col]:.4f}")
        print(f"  Mean Feature Correlation = {best_b_to_a.get(b_to_a_feature_corr_col, 0.0):.4f}")
        print(f"  Features with R² > 0.5 = {best_b_to_a[b_to_a_frac_col]:.2%}")
        print(f"  Features with Correlation > 0.6 = {best_b_to_a.get(b_to_a_frac_corr_col, 0.0):.2%}")
    
    if best_a_to_b_corr['k_value'] != best_a_to_b['k_value'] or (has_bandwidth and best_a_to_b_corr.get('bandwidth') != best_a_to_b.get('bandwidth')):
        print(f"\nBest A → B Performance (by Correlation):")
        print(f"  k = {best_a_to_b_corr['k_value']}")
        if has_bandwidth:
            print(f"  bandwidth = {best_a_to_b_corr['bandwidth']}")
        print(f"  Overall Correlation = {best_a_to_b_corr[a_to_b_corr_col]:.4f}")
        print(f"  Overall R² = {best_a_to_b_corr[a_to_b_col]:.4f}")
    
    if best_b_to_a_corr['k_value'] != best_b_to_a['k_value'] or (has_bandwidth and best_b_to_a_corr.get('bandwidth') != best_b_to_a.get('bandwidth')):
        print(f"\nBest B → A Performance (by Correlation):")
        print(f"  k = {best_b_to_a_corr['k_value']}")
        if has_bandwidth:
            print(f"  bandwidth = {best_b_to_a_corr['bandwidth']}")
        print(f"  Overall Correlation = {best_b_to_a_corr[b_to_a_corr_col]:.4f}")
        print(f"  Overall R² = {best_b_to_a_corr[b_to_a_col]:.4f}")
    
    print("\nAll Results:")
    if has_bandwidth:
        display_cols = ['k_value', 'bandwidth', a_to_b_col, b_to_a_col, a_to_b_corr_col, b_to_a_corr_col]
    else:
        display_cols = ['k_value', a_to_b_col, b_to_a_col, a_to_b_corr_col, b_to_a_corr_col]
    print(results_df[display_cols].to_string(index=False))
    print("="*80)
    
    return report


def perform_cross_imputation(platform_a, platform_b, platform_impute, impute_target, best_k, best_bandwidth,
                            kernel='distance', polynomial_degree=2,
                            log_transform_a=False, log_transform_b=False, log_epsilon=1e-8,
                            output_dir=None):
    """
    Perform cross-imputation using the best k parameters.
    
    Args:
        platform_a: Path to platform A file
        platform_b: Path to platform B file  
        platform_impute: Path to file that needs imputation
        impute_target: 'a' or 'b' - which platform to impute as
        best_k: Best k value from comparison
        best_bandwidth: Best bandwidth value from comparison
        kernel: Kernel type
        polynomial_degree: Polynomial degree
        log_transform_a: Log transform platform A
        log_transform_b: Log transform platform B
        log_epsilon: Log epsilon value
        output_dir: Output directory
    
    Returns:
        Path to the imputed file
    """
    sys.path.append(str(Path(__file__).parent))
    from knn_baseline import (
        load_and_merge_data, handle_missing_values, apply_log_transformation,
        normalize_data, train_knn_models, create_kernel_function
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    print(f"\nPerforming cross-imputation with k={best_k} for target platform {impute_target.upper()}...")
    
    data_a, data_b, features_a, features_b = load_and_merge_data(platform_a, platform_b)
    data_a, data_b = handle_missing_values(data_a, data_b, 'mean')  # Use mean imputation for training
    
    train_a, train_b, log_params = apply_log_transformation(
        data_a, data_b, log_transform_a, log_transform_b, log_epsilon
    )
    
    scaler_class = StandardScaler
    
    scaler_a = scaler_class()
    scaler_b = scaler_class()
    
    train_a_norm = scaler_a.fit_transform(train_a)
    train_b_norm = scaler_b.fit_transform(train_b)
    
    weights = create_kernel_function(kernel, best_bandwidth, polynomial_degree)
    
    if impute_target == 'a':
        # We want to impute platform A data, so we need B->A model
        knn_model = KNeighborsRegressor(n_neighbors=best_k, weights=weights)
        knn_model.fit(train_b_norm, train_a_norm)
        target_scaler = scaler_a
        target_log_params = log_params['platform_a']
        target_features = features_a
    else:
        # We want to impute platform B data, so we need A->B model  
        knn_model = KNeighborsRegressor(n_neighbors=best_k, weights=weights)
        knn_model.fit(train_a_norm, train_b_norm)
        target_scaler = scaler_b
        target_log_params = log_params['platform_b']
        target_features = features_b
    
    print(f"Loading imputation file: {platform_impute}")
    sep_impute = '\t' if Path(platform_impute).suffix.lower() == '.txt' else ','
    impute_df = pd.read_csv(platform_impute, sep=sep_impute)
    
    id_column = impute_df.columns[0]
    
    if impute_target == 'a':
        # Input should be platform B format, output will be platform A format
        input_features = features_b
        expected_input_features = [col for col in impute_df.columns if col != id_column]
        if len(expected_input_features) != len(input_features):
            print(f"Warning: Expected {len(input_features)} features for platform B, got {len(expected_input_features)}")
        impute_data = impute_df[expected_input_features]
    else:
        # Input should be platform A format, output will be platform B format
        input_features = features_a
        expected_input_features = [col for col in impute_df.columns if col != id_column]
        if len(expected_input_features) != len(input_features):
            print(f"Warning: Expected {len(input_features)} features for platform A, got {len(expected_input_features)}")
        impute_data = impute_df[expected_input_features]
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    impute_data_clean = pd.DataFrame(
        imputer.fit_transform(impute_data),
        columns=impute_data.columns,
        index=impute_data.index
    )
    
    if impute_target == 'a':
        # Input is platform B data, so apply platform B transformations
        if log_params['platform_b']['enabled']:
            shift_b = log_params['platform_b']['shift_value']
            impute_data_log = pd.DataFrame(
                np.log(impute_data_clean + shift_b), 
                columns=impute_data_clean.columns, 
                index=impute_data_clean.index
            )
        else:
            impute_data_log = impute_data_clean
        input_scaler = scaler_b
    else:
        # Input is platform A data, so apply platform A transformations
        if log_params['platform_a']['enabled']:
            shift_a = log_params['platform_a']['shift_value']
            impute_data_log = pd.DataFrame(
                np.log(impute_data_clean + shift_a), 
                columns=impute_data_clean.columns, 
                index=impute_data_clean.index
            )
        else:
            impute_data_log = impute_data_clean
        input_scaler = scaler_a
    
    impute_data_norm = input_scaler.transform(impute_data_log)
    
    print(f"Performing imputation using KNN with k={best_k}...")
    imputed_norm = knn_model.predict(impute_data_norm)
    
    # Inverse transform predictions to original scale for target platform
    imputed_denorm = target_scaler.inverse_transform(imputed_norm)
    
    if target_log_params['enabled']:
        shift_value = target_log_params['shift_value']
        imputed_original = np.exp(imputed_denorm) - shift_value
    else:
        imputed_original = imputed_denorm
    
    imputed_df = pd.DataFrame(imputed_original, columns=target_features, index=impute_df.index)
    
    output_df = pd.concat([impute_df[[id_column]], imputed_df], axis=1)
    
    impute_path = Path(platform_impute)
    output_filename = f"{impute_path.stem}_cross_imputed_{impute_target}{impute_path.suffix}"
    
    if output_dir:
        output_path = Path(output_dir) / output_filename
    else:
        output_path = impute_path.parent / output_filename
    
    output_df.to_csv(output_path, index=False)
    print(f"Cross-imputed file saved to: {output_path}")
    
    return output_path


def main():
    """Execute the complete KNN comparison workflow.
    
    Orchestrates the full pipeline including argument parsing, experiment execution,
    result collection, report generation, and optional cross-imputation.
    """
    args = parse_arguments()
    
    # Determine bandwidth values to test
    if args.bandwidth_values is None:
        if args.bandwidth is not None:
            # Backward compatibility: use single bandwidth value
            bandwidth_values = [args.bandwidth]
        elif args.kernel in ['gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial']:
            # Default bandwidth search for advanced kernels
            bandwidth_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        else:
            # No bandwidth needed for uniform/distance kernels
            bandwidth_values = [1.0]  # Dummy value, won't be used
    else:
        bandwidth_values = args.bandwidth_values
    
    if not Path(args.platform_a).exists():
        print(f"Error: Platform A file not found: {args.platform_a}")
        sys.exit(1)
    if not Path(args.platform_b).exists():
        print(f"Error: Platform B file not found: {args.platform_b}")
        sys.exit(1)
    
    if args.platform_impute is not None and args.impute_target is None:
        print(f"Error: --impute_target must be specified when --platform_impute is provided")
        sys.exit(1)
    if args.impute_target is not None and args.platform_impute is None:
        print(f"Error: --platform_impute must be specified when --impute_target is provided")
        sys.exit(1)
    if args.platform_impute is not None and not Path(args.platform_impute).exists():
        print(f"Error: Platform impute file not found: {args.platform_impute}")
        sys.exit(1)
    
    if not args.k_values or any(k < 1 for k in args.k_values):
        print(f"Error: All k values must be >= 1, got: {args.k_values}")
        sys.exit(1)
    
    if args.cv_folds < 0:
        print(f"Error: cv_folds must be >= 0, got: {args.cv_folds}")
        sys.exit(1)
    
    if any(bw <= 0 for bw in bandwidth_values):
        print(f"Error: all bandwidth values must be > 0, got: {bandwidth_values}")
        sys.exit(1)
    
    if args.polynomial_degree < 1:
        print(f"Error: polynomial_degree must be >= 1, got: {args.polynomial_degree}")
        sys.exit(1)
    
    use_cv = args.cv_folds > 0
    
    print(f"Starting KNN baseline comparison...")
    print(f"Platform A file: {args.platform_a}")
    print(f"Platform B file: {args.platform_b}")
    print(f"Testing k values: {args.k_values}")
    print(f"Using kernel: {args.kernel}")
    if args.kernel in ['gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial']:
        print(f"Testing bandwidth values: {bandwidth_values}")
        if args.kernel == 'polynomial':
            print(f"Polynomial degree: {args.polynomial_degree}")
    
    if args.log_transform_a or args.log_transform_b:
        print(f"Log transformation: Platform A={'enabled' if args.log_transform_a else 'disabled'}, Platform B={'enabled' if args.log_transform_b else 'disabled'}, epsilon={args.log_epsilon}")
    
    if args.platform_impute is not None:
        print(f"Cross-imputation enabled:")
        print(f"  Input file: {args.platform_impute}")
        print(f"  Target platform: {args.impute_target.upper()}")
    
    if use_cv:
        print(f"Using {args.cv_folds}-fold cross validation")
    else:
        print(f"Using train/test split")
    print(f"Output directory: {args.output_dir}")
    
    successful_runs = []
    experiment_configs = []
    
    # Determine if we need bandwidth search
    uses_bandwidth = args.kernel in ['gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial']
    
    for k in args.k_values:
        if uses_bandwidth:
            # Grid search over both k and bandwidth
            for bandwidth in bandwidth_values:
                success, exp_dir = run_knn_experiment(
                    args.platform_a, args.platform_b, args.output_dir, k, bandwidth,
                    args.cv_folds if use_cv else None,
                    args.kernel, args.polynomial_degree,
                    args.log_transform_a, args.log_transform_b, args.log_epsilon
                )
                if success:
                    successful_runs.append((k, bandwidth))
                    experiment_configs.append((k, bandwidth))
        else:
            # Only k search for uniform/distance kernels
            success, exp_dir = run_knn_experiment(
                args.platform_a, args.platform_b, args.output_dir, k, 1.0,  # dummy bandwidth
                args.cv_folds if use_cv else None,
                args.kernel, args.polynomial_degree,
                args.log_transform_a, args.log_transform_b, args.log_epsilon
            )
            if success:
                successful_runs.append(k)
                experiment_configs.append(k)
    
    if not successful_runs:
        print("No experiments completed successfully!")
        sys.exit(1)
    
    if uses_bandwidth:
        print(f"\nCompleted experiments for (k, bandwidth) pairs: {successful_runs}")
    else:
        print(f"\nCompleted experiments for k values: {successful_runs}")
    
    print("\nCollecting results...")
    results_df = collect_results(args.output_dir, experiment_configs, use_cv)
    
    if results_df is not None:
        report = create_comparison_report(results_df, args.output_dir, use_cv)
        print(f"\nComparison completed! Results saved to: {args.output_dir}")
        
        if args.platform_impute is not None:
            has_bandwidth = 'bandwidth' in results_df.columns
            if use_cv:
                if args.impute_target == 'a':
                    best_k_row = results_df.loc[results_df['B_to_A_R2_mean'].idxmax()]
                    best_k = int(best_k_row['k_value'])
                    best_bandwidth = float(best_k_row['bandwidth']) if has_bandwidth else 1.0
                    best_r2 = best_k_row['B_to_A_R2_mean']
                    if has_bandwidth:
                        print(f"\nUsing best k={best_k}, bandwidth={best_bandwidth} for B→A imputation (R²={best_r2:.4f})")
                    else:
                        print(f"\nUsing best k={best_k} for B→A imputation (R²={best_r2:.4f})")
                else:
                    best_k_row = results_df.loc[results_df['A_to_B_R2_mean'].idxmax()]
                    best_k = int(best_k_row['k_value'])
                    best_bandwidth = float(best_k_row['bandwidth']) if has_bandwidth else 1.0
                    best_r2 = best_k_row['A_to_B_R2_mean']
                    if has_bandwidth:
                        print(f"\nUsing best k={best_k}, bandwidth={best_bandwidth} for A→B imputation (R²={best_r2:.4f})")
                    else:
                        print(f"\nUsing best k={best_k} for A→B imputation (R²={best_r2:.4f})")
            else:
                if args.impute_target == 'a':
                    best_k_row = results_df.loc[results_df['B_to_A_R2'].idxmax()]
                    best_k = int(best_k_row['k_value'])
                    best_bandwidth = float(best_k_row['bandwidth']) if has_bandwidth else 1.0
                    best_r2 = best_k_row['B_to_A_R2']
                    if has_bandwidth:
                        print(f"\nUsing best k={best_k}, bandwidth={best_bandwidth} for B→A imputation (R²={best_r2:.4f})")
                    else:
                        print(f"\nUsing best k={best_k} for B→A imputation (R²={best_r2:.4f})")
                else:
                    best_k_row = results_df.loc[results_df['A_to_B_R2'].idxmax()]
                    best_k = int(best_k_row['k_value'])
                    best_bandwidth = float(best_k_row['bandwidth']) if has_bandwidth else 1.0
                    best_r2 = best_k_row['A_to_B_R2']
                    if has_bandwidth:
                        print(f"\nUsing best k={best_k}, bandwidth={best_bandwidth} for A→B imputation (R²={best_r2:.4f})")
                    else:
                        print(f"\nUsing best k={best_k} for A→B imputation (R²={best_r2:.4f})")
            try:
                imputed_file = perform_cross_imputation(
                    args.platform_a, args.platform_b, args.platform_impute, 
                    args.impute_target, best_k, best_bandwidth,
                    args.kernel, args.polynomial_degree,
                    args.log_transform_a, args.log_transform_b, args.log_epsilon,
                    args.output_dir
                )
                print(f"\n✓ Cross-imputation completed successfully!")
                print(f"Imputed file: {imputed_file}")
            except Exception as e:
                print(f"\n✗ Cross-imputation failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Failed to collect results!")
        sys.exit(1)


if __name__ == "__main__":
    main() 