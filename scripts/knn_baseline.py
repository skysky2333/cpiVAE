#!/usr/bin/env python3
"""
KNN Baseline for Joint VAE comparison.

This script implements a simple K-Nearest Neighbors regression baseline
for cross-platform metabolite data imputation to compare against the Joint VAE model.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
import logging

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.metrics import compute_imputation_metrics, create_detailed_feature_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def gaussian_kernel(distances, bandwidth=1.0):
    """
    Gaussian (RBF) kernel for KNN weighting.
    
    Args:
        distances: Array of distances to neighbors
        bandwidth: Bandwidth parameter (sigma)
        
    Returns:
        Array of weights
    """
    # Add small epsilon to avoid numerical issues with zero distances
    distances = np.maximum(distances, 1e-10)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    # Ensure no NaN or infinite values
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
    # Ensure weights are positive and non-zero
    weights = np.maximum(weights, 1e-10)
    return weights


def exponential_kernel(distances, bandwidth=1.0):
    """
    Exponential kernel for KNN weighting.
    
    Args:
        distances: Array of distances to neighbors
        bandwidth: Bandwidth parameter
        
    Returns:
        Array of weights
    """
    # Add small epsilon to avoid numerical issues with zero distances
    distances = np.maximum(distances, 1e-10)
    weights = np.exp(-distances / bandwidth)
    # Ensure no NaN or infinite values
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
    # Ensure weights are positive and non-zero
    weights = np.maximum(weights, 1e-10)
    return weights


def tricube_kernel(distances, bandwidth=1.0):
    """
    Tricube kernel for KNN weighting.
    
    Args:
        distances: Array of distances to neighbors
        bandwidth: Bandwidth parameter
        
    Returns:
        Array of weights
    """
    # Add small epsilon to avoid division by zero
    bandwidth = max(bandwidth, 1e-10)
    normalized_distances = distances / bandwidth
    # Tricube: (1 - |u|^3)^3 for |u| <= 1, 0 otherwise
    weights = np.where(
        normalized_distances <= 1.0,
        (1 - np.abs(normalized_distances) ** 3) ** 3,
        1e-10  # Small positive value instead of 0 to avoid division issues
    )
    # Ensure no NaN or infinite values
    weights = np.nan_to_num(weights, nan=1e-10, posinf=1.0, neginf=1e-10)
    # Ensure weights are positive and non-zero
    weights = np.maximum(weights, 1e-10)
    return weights


def epanechnikov_kernel(distances, bandwidth=1.0):
    """
    Epanechnikov kernel for KNN weighting.
    
    Args:
        distances: Array of distances to neighbors
        bandwidth: Bandwidth parameter
        
    Returns:
        Array of weights
    """
    # Add small epsilon to avoid division by zero
    bandwidth = max(bandwidth, 1e-10)
    normalized_distances = distances / bandwidth
    # Epanechnikov: 3/4 * (1 - u^2) for |u| <= 1, 0 otherwise
    weights = np.where(
        normalized_distances <= 1.0,
        0.75 * (1 - normalized_distances ** 2),
        1e-10  # Small positive value instead of 0
    )
    # Ensure no NaN or infinite values
    weights = np.nan_to_num(weights, nan=1e-10, posinf=1.0, neginf=1e-10)
    # Ensure weights are positive and non-zero
    weights = np.maximum(weights, 1e-10)
    return weights


def polynomial_kernel(distances, bandwidth=1.0, degree=2):
    """
    Polynomial kernel for KNN weighting.
    
    Args:
        distances: Array of distances to neighbors
        bandwidth: Bandwidth parameter
        degree: Polynomial degree
        
    Returns:
        Array of weights
    """
    # Add small epsilon to avoid division by zero
    bandwidth = max(bandwidth, 1e-10)
    weights = 1.0 / (1.0 + (distances / bandwidth) ** degree)
    # Ensure no NaN or infinite values
    weights = np.nan_to_num(weights, nan=1e-10, posinf=1.0, neginf=1e-10)
    # Ensure weights are positive and non-zero
    weights = np.maximum(weights, 1e-10)
    return weights


def create_kernel_function(kernel_type, bandwidth=1.0, degree=2):
    """
    Create a kernel function based on the specified type.
    
    Args:
        kernel_type: Type of kernel ('uniform', 'distance', 'gaussian', 'exponential', 
                    'tricube', 'epanechnikov', 'polynomial')
        bandwidth: Bandwidth parameter for the kernel
        degree: Degree parameter for polynomial kernel
        
    Returns:
        Kernel function or string for sklearn built-ins
    """
    if kernel_type == 'uniform':
        return 'uniform'
    elif kernel_type == 'distance':
        return 'distance'
    elif kernel_type == 'gaussian':
        return lambda distances: gaussian_kernel(distances, bandwidth)
    elif kernel_type == 'exponential':
        return lambda distances: exponential_kernel(distances, bandwidth)
    elif kernel_type == 'tricube':
        return lambda distances: tricube_kernel(distances, bandwidth)
    elif kernel_type == 'epanechnikov':
        return lambda distances: epanechnikov_kernel(distances, bandwidth)
    elif kernel_type == 'polynomial':
        return lambda distances: polynomial_kernel(distances, bandwidth, degree)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='KNN Baseline for Joint VAE comparison')
    
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
        default='outputs/knn_baseline',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_neighbors', 
        type=int, 
        default=5,
        help='Number of neighbors for KNN (default: 5)'
    )
    parser.add_argument(
        '--test_split', 
        type=float, 
        default=0.2,
        help='Test split ratio (default: 0.2 for 80:20 split)'
    )
    parser.add_argument(
        '--cv_folds', 
        type=int, 
        default=None,
        help='Number of cross validation folds (if specified, overrides test_split)'
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--missing_strategy', 
        type=str, 
        default='mean',
        choices=['mean', 'median', 'drop'],
        help='Strategy for handling missing values'
    )
    parser.add_argument(
        '--normalization', 
        type=str, 
        default='zscore',
        choices=['zscore', 'minmax', 'robust'],
        help='Normalization method'
    )
    parser.add_argument(
        '--id_column', 
        type=str, 
        default=None,
        help='Name of ID column (defaults to first column)'
    )
    parser.add_argument(
        '--kernel', 
        type=str, 
        default='distance',
        choices=['uniform', 'distance', 'gaussian', 'exponential', 'tricube', 'epanechnikov', 'polynomial'],
        help='Kernel/weighting function for KNN (default: distance)'
    )
    parser.add_argument(
        '--bandwidth', 
        type=float, 
        default=1.0,
        help='Bandwidth parameter for advanced kernels (default: 1.0)'
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
    
    return parser.parse_args()


def load_and_merge_data(file_a: str, file_b: str, id_column: str = None):
    """
    Load two data files and merge them on a shared ID column.
    
    Args:
        file_a: Path to platform A data file (CSV or TXT)
        file_b: Path to platform B data file (CSV or TXT)
        id_column: Name of the ID column (defaults to first column)
        
    Returns:
        Tuple of (platform_a_data, platform_b_data, feature_names_a, feature_names_b)
    """
    logger.info(f"Loading data from {file_a} and {file_b}")
    
    # Load data files with appropriate separator
    sep_a = '\t' if Path(file_a).suffix.lower() == '.txt' else ','
    sep_b = '\t' if Path(file_b).suffix.lower() == '.txt' else ','
    df_a = pd.read_csv(file_a, sep=sep_a)
    df_b = pd.read_csv(file_b, sep=sep_b)
    
    # Use first column as ID if not specified
    if id_column is None:
        id_column = df_a.columns[0]
    
    logger.info(f"Using '{id_column}' as the ID column")
    
    # Extract feature columns (all except ID column)
    features_a = [col for col in df_a.columns if col != id_column]
    features_b = [col for col in df_b.columns if col != id_column]
    
    logger.info(f"Platform A: {len(features_a)} features, {len(df_a)} samples")
    logger.info(f"Platform B: {len(features_b)} features, {len(df_b)} samples")
    
    # Perform inner join on ID column
    merged_ids = pd.merge(
        df_a[[id_column]], 
        df_b[[id_column]], 
        on=id_column, 
        how='inner'
    )
    
    logger.info(f"After merging: {len(merged_ids)} shared samples")
    
    if len(merged_ids) == 0:
        raise ValueError("No shared samples found between the two datasets!")
    
    # Filter both datasets to only include shared samples
    df_a_filtered = df_a[df_a[id_column].isin(merged_ids[id_column])]
    df_b_filtered = df_b[df_b[id_column].isin(merged_ids[id_column])]
    
    # Sort by ID to ensure alignment
    df_a_filtered = df_a_filtered.sort_values(id_column).reset_index(drop=True)
    df_b_filtered = df_b_filtered.sort_values(id_column).reset_index(drop=True)
    
    # Extract feature data only
    data_a = df_a_filtered[features_a]
    data_b = df_b_filtered[features_b]
    
    logger.info("Data loading and merging completed successfully")
    
    return data_a, data_b, features_a, features_b


def handle_missing_values(data_a: pd.DataFrame, data_b: pd.DataFrame, strategy: str):
    """
    Handle missing values in both datasets.
    
    Args:
        data_a: Platform A feature data
        data_b: Platform B feature data
        strategy: Missing value handling strategy
        
    Returns:
        Tuple of cleaned DataFrames
    """
    logger.info(f"Handling missing values using strategy: {strategy}")
    
    # Report missing value statistics
    missing_a = data_a.isnull().sum().sum()
    missing_b = data_b.isnull().sum().sum()
    logger.info(f"Platform A missing values: {missing_a}")
    logger.info(f"Platform B missing values: {missing_b}")
    
    if strategy == 'drop':
        # Drop samples with any missing values
        mask_a = ~data_a.isnull().any(axis=1)
        mask_b = ~data_b.isnull().any(axis=1)
        combined_mask = mask_a & mask_b
        
        data_a = data_a[combined_mask].reset_index(drop=True)
        data_b = data_b[combined_mask].reset_index(drop=True)
        
    elif strategy in ['mean', 'median']:
        # Simple imputation
        imputer = SimpleImputer(strategy=strategy)
        data_a = pd.DataFrame(
            imputer.fit_transform(data_a),
            columns=data_a.columns
        )
        data_b = pd.DataFrame(
            imputer.fit_transform(data_b),
            columns=data_b.columns
        )
    
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    logger.info("Missing value handling completed")
    return data_a, data_b


def apply_log_transformation(data_a, data_b, log_transform_a=False, log_transform_b=False, epsilon=1e-8):
    """
    Apply log transformation to datasets.
    
    Args:
        data_a: Platform A feature data
        data_b: Platform B feature data
        log_transform_a: Whether to apply log transformation to platform A
        log_transform_b: Whether to apply log transformation to platform B
        epsilon: Small value added to ensure positive values
        
    Returns:
        Tuple of (transformed_data_a, transformed_data_b, log_params)
    """
    log_params = {
        'platform_a': {'enabled': False, 'shift_value': 0.0},
        'platform_b': {'enabled': False, 'shift_value': 0.0}
    }
    
    results = []
    
    for data, platform, apply_log in [(data_a, 'platform_a', log_transform_a), 
                                      (data_b, 'platform_b', log_transform_b)]:
        if apply_log and data is not None:
            logger.info(f"Applying log transformation to {platform}")
            
            # Calculate shift value to handle non-positive values
            min_val = data.min().min()
            if min_val <= 0:
                shift_value = -min_val + epsilon
                logger.info(f"  {platform}: Shifting data by {shift_value:.6f} to handle non-positive values")
            else:
                shift_value = 0.0
            
            # Store transformation parameters
            log_params[platform] = {
                'enabled': True,
                'shift_value': shift_value
            }
            
            # Apply transformation
            shifted_data = data + shift_value
            log_data = np.log(shifted_data)
            log_data = pd.DataFrame(log_data, columns=data.columns, index=data.index)
            results.append(log_data)
            
            logger.info(f"  {platform}: Log transformation completed")
        else:
            # No transformation needed
            results.append(data)
    
    return results[0], results[1], log_params


def normalize_data(train_a, train_b, test_a, test_b, method='zscore'):
    """
    Normalize data using scalers fitted on training data.
    
    Args:
        train_a, train_b: Training data
        test_a, test_b: Test data
        method: Normalization method
        
    Returns:
        Normalized data and fitted scalers
    """
    logger.info(f"Normalizing data using method: {method}")
    
    if method == 'zscore':
        scaler_class = StandardScaler
    elif method == 'minmax':
        scaler_class = MinMaxScaler
    elif method == 'robust':
        scaler_class = RobustScaler
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit scalers on training data
    scaler_a = scaler_class()
    scaler_b = scaler_class()
    
    train_a_norm = scaler_a.fit_transform(train_a)
    train_b_norm = scaler_b.fit_transform(train_b)
    
    # Transform test data
    test_a_norm = scaler_a.transform(test_a)
    test_b_norm = scaler_b.transform(test_b)
    
    logger.info("Data normalization completed")
    return train_a_norm, train_b_norm, test_a_norm, test_b_norm, scaler_a, scaler_b


def train_knn_models(train_a, train_b, n_neighbors=5, kernel='distance', bandwidth=1.0, degree=2):
    """
    Train KNN models for both directions.
    
    Args:
        train_a: Training data for platform A
        train_b: Training data for platform B
        n_neighbors: Number of neighbors for KNN
        kernel: Kernel/weighting function type
        bandwidth: Bandwidth parameter for advanced kernels
        degree: Degree parameter for polynomial kernel
        
    Returns:
        Tuple of (knn_a_to_b, knn_b_to_a) models
    """
    logger.info(f"Training KNN models with {n_neighbors} neighbors and {kernel} kernel")
    if kernel not in ['uniform', 'distance']:
        logger.info(f"Using bandwidth={bandwidth}" + (f", degree={degree}" if kernel == 'polynomial' else ""))
    
    # Create kernel function
    weights = create_kernel_function(kernel, bandwidth, degree)
    
    # Model for A -> B prediction
    knn_a_to_b = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn_a_to_b.fit(train_a, train_b)
    
    # Model for B -> A prediction  
    knn_b_to_a = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn_b_to_a.fit(train_b, train_a)
    
    logger.info("KNN models training completed")
    return knn_a_to_b, knn_b_to_a


def evaluate_models(knn_a_to_b, knn_b_to_a, test_a_norm, test_b_norm,
                    original_test_a, original_test_b, scaler_a, scaler_b,
                    log_params, features_a, features_b):
    """
    Evaluate KNN models and compute metrics after inverse-transforming predictions
    back to the original data space.
    
    Args:
        knn_a_to_b: KNN model for A -> B prediction
        knn_b_to_a: KNN model for B -> A prediction
        test_a_norm, test_b_norm: Normalized test data for making predictions
        original_test_a, original_test_b: Test data in the original data space for evaluation
        scaler_a, scaler_b: Fitted scalers for inverse normalization
        log_params: Log transformation parameters for inverse log transform
        features_a, features_b: Feature names
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating KNN models on original data scale...")
    
    # 1. Make predictions in the transformed (normalized) space
    pred_b_from_a_norm = knn_a_to_b.predict(test_a_norm)  # A -> B
    pred_a_from_b_norm = knn_b_to_a.predict(test_b_norm)  # B -> A
    
    # 2. Inverse transform predictions back to the original space
    
    # Inverse transform A -> B predictions
    pred_b_from_a_inv_norm = scaler_b.inverse_transform(pred_b_from_a_norm)
    if log_params['platform_b']['enabled']:
        shift_b = log_params['platform_b']['shift_value']
        pred_b_from_a_orig = np.exp(pred_b_from_a_inv_norm) - shift_b
    else:
        pred_b_from_a_orig = pred_b_from_a_inv_norm

    # Inverse transform B -> A predictions
    pred_a_from_b_inv_norm = scaler_a.inverse_transform(pred_a_from_b_norm)
    if log_params['platform_a']['enabled']:
        shift_a = log_params['platform_a']['shift_value']
        pred_a_from_b_orig = np.exp(pred_a_from_b_inv_norm) - shift_a
    else:
        pred_a_from_b_orig = pred_a_from_b_inv_norm

    # --- FIX START ---
    # 3. Convert original test DataFrames to NumPy arrays before computing metrics
    logger.info("Computing metrics against original test data.")
    metrics_a_to_b = compute_imputation_metrics(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    metrics_b_to_a = compute_imputation_metrics(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    
    # Also convert for the detailed report function to be safe
    report_a_to_b = create_detailed_feature_report(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    report_b_to_a = create_detailed_feature_report(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    # --- FIX END ---
    
    logger.info("Model evaluation completed")
    
    return {
        'metrics_a_to_b': metrics_a_to_b,
        'metrics_b_to_a': metrics_b_to_a,
        'report_a_to_b': report_a_to_b,
        'report_b_to_a': report_b_to_a,
    }


def evaluate_models_cv_fold(knn_a_to_b, knn_b_to_a, test_a_norm, test_b_norm,
                            original_test_a, original_test_b, scaler_a, scaler_b,
                            log_params, features_a, features_b):
    """
    Evaluate KNN models for a single CV fold after inverse-transforming predictions.
    
    Args:
        knn_a_to_b: KNN model for A -> B prediction
        knn_b_to_a: KNN model for B -> A prediction
        test_a_norm, test_b_norm: Normalized test data for making predictions
        original_test_a, original_test_b: Test data in the original data space for evaluation
        scaler_a, scaler_b: Fitted scalers for inverse normalization
        log_params: Log transformation parameters for inverse log transform
        features_a, features_b: Feature names
        
    Returns:
        Dictionary of evaluation results for this fold
    """
    # 1. Make predictions in the transformed (normalized) space
    pred_b_from_a_norm = knn_a_to_b.predict(test_a_norm)  # A -> B
    pred_a_from_b_norm = knn_b_to_a.predict(test_b_norm)  # B -> A

    # 2. Inverse transform predictions back to the original space
    pred_b_from_a_inv_norm = scaler_b.inverse_transform(pred_b_from_a_norm)
    if log_params['platform_b']['enabled']:
        shift_b = log_params['platform_b']['shift_value']
        pred_b_from_a_orig = np.exp(pred_b_from_a_inv_norm) - shift_b
    else:
        pred_b_from_a_orig = pred_b_from_a_inv_norm

    pred_a_from_b_inv_norm = scaler_a.inverse_transform(pred_a_from_b_norm)
    if log_params['platform_a']['enabled']:
        shift_a = log_params['platform_a']['shift_value']
        pred_a_from_b_orig = np.exp(pred_a_from_b_inv_norm) - shift_a
    else:
        pred_a_from_b_orig = pred_a_from_b_inv_norm

    # --- FIX START ---
    # 3. Convert original test DataFrames to NumPy arrays before computing metrics
    metrics_a_to_b = compute_imputation_metrics(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    metrics_b_to_a = compute_imputation_metrics(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    # --- FIX END ---
    
    return {
        'metrics_a_to_b': metrics_a_to_b,
        'metrics_b_to_a': metrics_b_to_a,
    }

def run_cross_validation(data_a, data_b, features_a, features_b, args):
    """
    Run cross validation experiment.
    
    Args:
        data_a, data_b: Feature data for both platforms
        features_a, features_b: Feature names
        args: Command line arguments
        
    Returns:
        Dictionary of cross validation results
    """
    logger.info(f"Running {args.cv_folds}-fold cross validation")
    
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data_a)):
        logger.info(f"Processing fold {fold_idx + 1}/{args.cv_folds}")
        
        # Split data into train and test for this fold
        train_a, test_a = data_a.iloc[train_idx], data_a.iloc[test_idx]
        train_b, test_b = data_b.iloc[train_idx], data_b.iloc[test_idx]

        # --- MODIFICATION START ---
        # Keep the original test sets for final evaluation
        original_fold_test_a = test_a.copy()
        original_fold_test_b = test_b.copy()
        # --- MODIFICATION END ---
        
        # Apply log transformation (correctly, only using train data to find params)
        log_params = {'platform_a': {'enabled': False, 'shift_value': 0.0}, 'platform_b': {'enabled': False, 'shift_value': 0.0}}
        if args.log_transform_a or args.log_transform_b:
            train_a, train_b, log_params = apply_log_transformation(
                train_a, train_b, args.log_transform_a, args.log_transform_b, args.log_epsilon
            )
            # Apply same transformation to test data using parameters from training data
            if log_params['platform_a']['enabled']:
                shift_a = log_params['platform_a']['shift_value']
                test_a = pd.DataFrame(np.log(test_a + shift_a), columns=test_a.columns, index=test_a.index)
            if log_params['platform_b']['enabled']:
                shift_b = log_params['platform_b']['shift_value']
                test_b = pd.DataFrame(np.log(test_b + shift_b), columns=test_b.columns, index=test_b.index)

        if len(train_a) < args.n_neighbors:
            logger.warning(f"Fold {fold_idx + 1}: Not enough training samples ({len(train_a)}) for k={args.n_neighbors}")
            continue
        
        # --- MODIFICATION START ---
        # Normalize data and capture the fitted scalers
        train_a_norm, train_b_norm, test_a_norm, test_b_norm, scaler_a, scaler_b = normalize_data(
            train_a, train_b, test_a, test_b, args.normalization
        )
        # --- MODIFICATION END ---
        
        # Train KNN models
        knn_a_to_b, knn_b_to_a = train_knn_models(
            train_a_norm, train_b_norm, args.n_neighbors, 
            args.kernel, args.bandwidth, args.polynomial_degree
        )
        
        # --- MODIFICATION START ---
        # Evaluate models for this fold, passing all necessary components
        fold_result = evaluate_models_cv_fold(
            knn_a_to_b, knn_b_to_a, 
            test_a_norm, test_b_norm, 
            original_fold_test_a, original_fold_test_b,
            scaler_a, scaler_b,
            log_params,
            features_a, features_b
        )
        # --- MODIFICATION END ---
        
        fold_result['fold'] = fold_idx + 1
        fold_results.append(fold_result)
    
    return aggregate_cv_results(fold_results, features_a, features_b)

def aggregate_cv_results(fold_results, features_a, features_b):
    """
    Aggregate cross validation results across folds.
    
    Args:
        fold_results: List of results from each fold
        features_a, features_b: Feature names
        
    Returns:
        Dictionary of aggregated results
    """
    if not fold_results:
        raise ValueError("No valid folds completed")
    
    logger.info(f"Aggregating results from {len(fold_results)} folds")
    
    # Extract metrics for each direction
    a_to_b_metrics = []
    b_to_a_metrics = []
    
    for result in fold_results:
        a_to_b_metrics.append(result['metrics_a_to_b'])
        b_to_a_metrics.append(result['metrics_b_to_a'])
    
    # Aggregate key metrics
    def aggregate_metrics(metrics_list):
        """Aggregate metrics across folds."""
        aggregated = {}
        
        # Overall metrics (simple average)
        overall_keys = ['overall_r2', 'overall_mse', 'overall_mae', 'overall_correlation']
        for key in overall_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        # Feature-wise metrics
        feature_keys = ['mean_feature_r2', 'median_feature_r2', 'fraction_r2_above_0.5', 'fraction_r2_above_0.7',
                       'mean_feature_correlation', 'median_feature_correlation', 'fraction_corr_above_0.6', 'fraction_corr_above_0.8']
        for key in feature_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        # Feature-specific R2 scores (average across folds)
        if 'feature_r2_scores' in metrics_list[0]:
            feature_r2_all_folds = [m['feature_r2_scores'] for m in metrics_list]
            feature_r2_mean = np.mean(feature_r2_all_folds, axis=0)
            feature_r2_std = np.std(feature_r2_all_folds, axis=0)
            aggregated['feature_r2_scores_mean'] = feature_r2_mean
            aggregated['feature_r2_scores_std'] = feature_r2_std
        
        # Feature-specific correlation scores (average across folds)  
        if 'feature_correlations' in metrics_list[0]:
            feature_corr_all_folds = [m['feature_correlations'] for m in metrics_list]
            feature_corr_mean = np.mean(feature_corr_all_folds, axis=0)
            feature_corr_std = np.std(feature_corr_all_folds, axis=0)
            aggregated['feature_correlations_mean'] = feature_corr_mean
            aggregated['feature_correlations_std'] = feature_corr_std
        
        return aggregated
    
    # Aggregate metrics for both directions
    aggregated_a_to_b = aggregate_metrics(a_to_b_metrics)
    aggregated_b_to_a = aggregate_metrics(b_to_a_metrics)
    
    # Create detailed reports with mean values
    if 'feature_r2_scores_mean' in aggregated_a_to_b:
        report_a_to_b = create_detailed_feature_report_cv(
            aggregated_a_to_b['feature_r2_scores_mean'],
            aggregated_a_to_b['feature_r2_scores_std'],
            aggregated_a_to_b.get('feature_correlations_mean'),
            aggregated_a_to_b.get('feature_correlations_std'),
            features_b
        )
    else:
        report_a_to_b = pd.DataFrame()
    
    if 'feature_r2_scores_mean' in aggregated_b_to_a:
        report_b_to_a = create_detailed_feature_report_cv(
            aggregated_b_to_a['feature_r2_scores_mean'],
            aggregated_b_to_a['feature_r2_scores_std'],
            aggregated_b_to_a.get('feature_correlations_mean'),
            aggregated_b_to_a.get('feature_correlations_std'),
            features_a
        )
    else:
        report_b_to_a = pd.DataFrame()
    
    return {
        'metrics_a_to_b': aggregated_a_to_b,
        'metrics_b_to_a': aggregated_b_to_a,
        'report_a_to_b': report_a_to_b,
        'report_b_to_a': report_b_to_a,
        'fold_results': fold_results,
        'n_folds': len(fold_results)
    }


def create_detailed_feature_report_cv(feature_r2_mean, feature_r2_std, feature_corr_mean=None, feature_corr_std=None, feature_names=None):
    """
    Create detailed feature report for cross validation results.
    
    Args:
        feature_r2_mean: Mean R2 scores across folds
        feature_r2_std: Standard deviation of R2 scores across folds
        feature_corr_mean: Mean correlation scores across folds (optional)
        feature_corr_std: Standard deviation of correlation scores across folds (optional)
        feature_names: Feature names
        
    Returns:
        DataFrame with detailed feature performance
    """
    report_data = []
    for i, feature in enumerate(feature_names):
        row_data = {
            'feature': feature,
            'r2_mean': feature_r2_mean[i],
            'r2_std': feature_r2_std[i],
            'r2_mean_above_0.5': feature_r2_mean[i] > 0.5,
            'r2_mean_above_0.7': feature_r2_mean[i] > 0.7,
        }
        
        # Add correlation metrics if available
        if feature_corr_mean is not None and feature_corr_std is not None:
            row_data.update({
                'correlation_mean': feature_corr_mean[i],
                'correlation_std': feature_corr_std[i],
                'correlation_mean_above_0.6': feature_corr_mean[i] > 0.6,
                'correlation_mean_above_0.8': feature_corr_mean[i] > 0.8,
            })
        
        report_data.append(row_data)
    
    return pd.DataFrame(report_data)


def save_results(results, output_dir, is_cv=False):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if is_cv:
        # Save cross validation summary
        summary = {
            'A_to_B_R2_mean': results['metrics_a_to_b']['overall_r2_mean'],
            'A_to_B_R2_std': results['metrics_a_to_b']['overall_r2_std'],
            'B_to_A_R2_mean': results['metrics_b_to_a']['overall_r2_mean'],
            'B_to_A_R2_std': results['metrics_b_to_a']['overall_r2_std'],
            'A_to_B_correlation_mean': results['metrics_a_to_b']['overall_correlation_mean'],
            'A_to_B_correlation_std': results['metrics_a_to_b']['overall_correlation_std'],
            'B_to_A_correlation_mean': results['metrics_b_to_a']['overall_correlation_mean'],
            'B_to_A_correlation_std': results['metrics_b_to_a']['overall_correlation_std'],
            'A_to_B_mean_feature_R2_mean': results['metrics_a_to_b']['mean_feature_r2_mean'],
            'A_to_B_mean_feature_R2_std': results['metrics_a_to_b']['mean_feature_r2_std'],
            'B_to_A_mean_feature_R2_mean': results['metrics_b_to_a']['mean_feature_r2_mean'],
            'B_to_A_mean_feature_R2_std': results['metrics_b_to_a']['mean_feature_r2_std'],
            'A_to_B_mean_feature_correlation_mean': results['metrics_a_to_b'].get('mean_feature_correlation_mean', 0.0),
            'A_to_B_mean_feature_correlation_std': results['metrics_a_to_b'].get('mean_feature_correlation_std', 0.0),
            'B_to_A_mean_feature_correlation_mean': results['metrics_b_to_a'].get('mean_feature_correlation_mean', 0.0),
            'B_to_A_mean_feature_correlation_std': results['metrics_b_to_a'].get('mean_feature_correlation_std', 0.0),
            'A_to_B_fraction_R2_above_0.5_mean': results['metrics_a_to_b']['fraction_r2_above_0.5_mean'],
            'A_to_B_fraction_R2_above_0.5_std': results['metrics_a_to_b']['fraction_r2_above_0.5_std'],
            'B_to_A_fraction_R2_above_0.5_mean': results['metrics_b_to_a']['fraction_r2_above_0.5_mean'],
            'B_to_A_fraction_R2_above_0.5_std': results['metrics_b_to_a']['fraction_r2_above_0.5_std'],
            'A_to_B_fraction_corr_above_0.6_mean': results['metrics_a_to_b'].get('fraction_corr_above_0.6_mean', 0.0),
            'A_to_B_fraction_corr_above_0.6_std': results['metrics_a_to_b'].get('fraction_corr_above_0.6_std', 0.0),
            'B_to_A_fraction_corr_above_0.6_mean': results['metrics_b_to_a'].get('fraction_corr_above_0.6_mean', 0.0),
            'B_to_A_fraction_corr_above_0.6_std': results['metrics_b_to_a'].get('fraction_corr_above_0.6_std', 0.0),
            'A_to_B_fraction_corr_above_0.8_mean': results['metrics_a_to_b'].get('fraction_corr_above_0.8_mean', 0.0),
            'A_to_B_fraction_corr_above_0.8_std': results['metrics_a_to_b'].get('fraction_corr_above_0.8_std', 0.0),
            'B_to_A_fraction_corr_above_0.8_mean': results['metrics_b_to_a'].get('fraction_corr_above_0.8_mean', 0.0),
            'B_to_A_fraction_corr_above_0.8_std': results['metrics_b_to_a'].get('fraction_corr_above_0.8_std', 0.0),
            'n_folds': results['n_folds']
        }
        filename = 'knn_baseline_cv_summary.csv'
    else:
        # Save single run summary
        summary = {
            'A_to_B_R2': results['metrics_a_to_b']['overall_r2'],
            'B_to_A_R2': results['metrics_b_to_a']['overall_r2'],
            'A_to_B_correlation': results['metrics_a_to_b']['overall_correlation'],
            'B_to_A_correlation': results['metrics_b_to_a']['overall_correlation'],
            'A_to_B_mean_feature_R2': results['metrics_a_to_b']['mean_feature_r2'],
            'B_to_A_mean_feature_R2': results['metrics_b_to_a']['mean_feature_r2'],
            'A_to_B_mean_feature_correlation': results['metrics_a_to_b']['mean_feature_correlation'],
            'B_to_A_mean_feature_correlation': results['metrics_b_to_a']['mean_feature_correlation'],
            'A_to_B_fraction_R2_above_0.5': results['metrics_a_to_b']['fraction_r2_above_0.5'],
            'B_to_A_fraction_R2_above_0.5': results['metrics_b_to_a']['fraction_r2_above_0.5'],
            'A_to_B_fraction_corr_above_0.6': results['metrics_a_to_b']['fraction_corr_above_0.6'],
            'B_to_A_fraction_corr_above_0.6': results['metrics_b_to_a']['fraction_corr_above_0.6'],
            'A_to_B_fraction_corr_above_0.8': results['metrics_a_to_b']['fraction_corr_above_0.8'],
            'B_to_A_fraction_corr_above_0.8': results['metrics_b_to_a']['fraction_corr_above_0.8'],
        }
        filename = 'knn_baseline_summary.csv'
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / filename, index=False)
    
    # Save detailed feature reports
    results['report_a_to_b'].to_csv(output_path / 'detailed_report_a_to_b.csv', index=False)
    results['report_b_to_a'].to_csv(output_path / 'detailed_report_b_to_a.csv', index=False)
    
    # Save all metrics (convert numpy types to native Python types for JSON)
    import json
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    metrics_a_to_b_serializable = convert_numpy_types(results['metrics_a_to_b'])
    metrics_b_to_a_serializable = convert_numpy_types(results['metrics_b_to_a'])
    
    with open(output_path / 'all_metrics_a_to_b.json', 'w') as f:
        json.dump(metrics_a_to_b_serializable, f, indent=2)
    
    with open(output_path / 'all_metrics_b_to_a.json', 'w') as f:
        json.dump(metrics_b_to_a_serializable, f, indent=2)
    
    # Save fold-level results if cross validation
    if is_cv and 'fold_results' in results:
        fold_results_serializable = convert_numpy_types(results['fold_results'])
        with open(output_path / 'fold_results.json', 'w') as f:
            json.dump(fold_results_serializable, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return summary


def print_results(results, is_cv=False):
    """Print evaluation results to console."""
    print("="*80)
    if is_cv:
        print(f"KNN BASELINE CROSS VALIDATION RESULTS ({results['n_folds']} folds)")
    else:
        print("KNN BASELINE EVALUATION RESULTS")
    print("="*80)
    
    # Print fold-level summary if cross validation
    if is_cv and 'fold_results' in results:
        print("\nFold-level Results:")
        print("Fold | A→B R² | B→A R² | A→B Corr | B→A Corr")
        print("-" * 45)
        for fold_result in results['fold_results']:
            fold_num = fold_result['fold']
            a_to_b_r2 = fold_result['metrics_a_to_b']['overall_r2']
            b_to_a_r2 = fold_result['metrics_b_to_a']['overall_r2']
            a_to_b_corr = fold_result['metrics_a_to_b']['overall_correlation']
            b_to_a_corr = fold_result['metrics_b_to_a']['overall_correlation']
            print(f"{fold_num:4d} | {a_to_b_r2:6.4f} | {b_to_a_r2:6.4f} | {a_to_b_corr:8.4f} | {b_to_a_corr:8.4f}")
        print("-" * 45)
    
    # A -> B results
    metrics_a_to_b = results['metrics_a_to_b']
    print(f"\nPlatform A → Platform B Prediction:")
    if is_cv:
        print(f"  Overall R²: {metrics_a_to_b['overall_r2_mean']:.4f} ± {metrics_a_to_b['overall_r2_std']:.4f}")
        print(f"  Overall Correlation: {metrics_a_to_b['overall_correlation_mean']:.4f} ± {metrics_a_to_b['overall_correlation_std']:.4f}")
        print(f"  Mean Feature R²: {metrics_a_to_b['mean_feature_r2_mean']:.4f} ± {metrics_a_to_b['mean_feature_r2_std']:.4f}")
        print(f"  Mean Feature Correlation: {metrics_a_to_b.get('mean_feature_correlation_mean', 0.0):.4f} ± {metrics_a_to_b.get('mean_feature_correlation_std', 0.0):.4f}")
        print(f"  Median Feature R²: {metrics_a_to_b['median_feature_r2_mean']:.4f} ± {metrics_a_to_b['median_feature_r2_std']:.4f}")
        print(f"  Median Feature Correlation: {metrics_a_to_b.get('median_feature_correlation_mean', 0.0):.4f} ± {metrics_a_to_b.get('median_feature_correlation_std', 0.0):.4f}")
        print(f"  Features with R² > 0.5: {metrics_a_to_b['fraction_r2_above_0.5_mean']:.2%} ± {metrics_a_to_b['fraction_r2_above_0.5_std']:.2%}")
        print(f"  Features with R² > 0.7: {metrics_a_to_b['fraction_r2_above_0.7_mean']:.2%} ± {metrics_a_to_b['fraction_r2_above_0.7_std']:.2%}")
        print(f"  Features with Correlation > 0.6: {metrics_a_to_b.get('fraction_corr_above_0.6_mean', 0.0):.2%} ± {metrics_a_to_b.get('fraction_corr_above_0.6_std', 0.0):.2%}")
        print(f"  Features with Correlation > 0.8: {metrics_a_to_b.get('fraction_corr_above_0.8_mean', 0.0):.2%} ± {metrics_a_to_b.get('fraction_corr_above_0.8_std', 0.0):.2%}")
    else:
        print(f"  Overall R²: {metrics_a_to_b['overall_r2']:.4f}")
        print(f"  Overall Correlation: {metrics_a_to_b['overall_correlation']:.4f}")
        print(f"  Mean Feature R²: {metrics_a_to_b['mean_feature_r2']:.4f}")
        print(f"  Mean Feature Correlation: {metrics_a_to_b['mean_feature_correlation']:.4f}")
        print(f"  Median Feature R²: {metrics_a_to_b['median_feature_r2']:.4f}")
        print(f"  Median Feature Correlation: {metrics_a_to_b['median_feature_correlation']:.4f}")
        print(f"  Features with R² > 0.5: {metrics_a_to_b['fraction_r2_above_0.5']:.2%}")
        print(f"  Features with R² > 0.7: {metrics_a_to_b['fraction_r2_above_0.7']:.2%}")
        print(f"  Features with Correlation > 0.6: {metrics_a_to_b['fraction_corr_above_0.6']:.2%}")
        print(f"  Features with Correlation > 0.8: {metrics_a_to_b['fraction_corr_above_0.8']:.2%}")
    
    # B -> A results
    metrics_b_to_a = results['metrics_b_to_a']
    print(f"\nPlatform B → Platform A Prediction:")
    if is_cv:
        print(f"  Overall R²: {metrics_b_to_a['overall_r2_mean']:.4f} ± {metrics_b_to_a['overall_r2_std']:.4f}")
        print(f"  Overall Correlation: {metrics_b_to_a['overall_correlation_mean']:.4f} ± {metrics_b_to_a['overall_correlation_std']:.4f}")
        print(f"  Mean Feature R²: {metrics_b_to_a['mean_feature_r2_mean']:.4f} ± {metrics_b_to_a['mean_feature_r2_std']:.4f}")
        print(f"  Mean Feature Correlation: {metrics_b_to_a.get('mean_feature_correlation_mean', 0.0):.4f} ± {metrics_b_to_a.get('mean_feature_correlation_std', 0.0):.4f}")
        print(f"  Median Feature R²: {metrics_b_to_a['median_feature_r2_mean']:.4f} ± {metrics_b_to_a['median_feature_r2_std']:.4f}")
        print(f"  Median Feature Correlation: {metrics_b_to_a.get('median_feature_correlation_mean', 0.0):.4f} ± {metrics_b_to_a.get('median_feature_correlation_std', 0.0):.4f}")
        print(f"  Features with R² > 0.5: {metrics_b_to_a['fraction_r2_above_0.5_mean']:.2%} ± {metrics_b_to_a['fraction_r2_above_0.5_std']:.2%}")
        print(f"  Features with R² > 0.7: {metrics_b_to_a['fraction_r2_above_0.7_mean']:.2%} ± {metrics_b_to_a['fraction_r2_above_0.7_std']:.2%}")
        print(f"  Features with Correlation > 0.6: {metrics_b_to_a.get('fraction_corr_above_0.6_mean', 0.0):.2%} ± {metrics_b_to_a.get('fraction_corr_above_0.6_std', 0.0):.2%}")
        print(f"  Features with Correlation > 0.8: {metrics_b_to_a.get('fraction_corr_above_0.8_mean', 0.0):.2%} ± {metrics_b_to_a.get('fraction_corr_above_0.8_std', 0.0):.2%}")
    else:
        print(f"  Overall R²: {metrics_b_to_a['overall_r2']:.4f}")
        print(f"  Overall Correlation: {metrics_b_to_a['overall_correlation']:.4f}")
        print(f"  Mean Feature R²: {metrics_b_to_a['mean_feature_r2']:.4f}")
        print(f"  Mean Feature Correlation: {metrics_b_to_a['mean_feature_correlation']:.4f}")
        print(f"  Median Feature R²: {metrics_b_to_a['median_feature_r2']:.4f}")
        print(f"  Median Feature Correlation: {metrics_b_to_a['median_feature_correlation']:.4f}")
        print(f"  Features with R² > 0.5: {metrics_b_to_a['fraction_r2_above_0.5']:.2%}")
        print(f"  Features with R² > 0.7: {metrics_b_to_a['fraction_r2_above_0.7']:.2%}")
        print(f"  Features with Correlation > 0.6: {metrics_b_to_a['fraction_corr_above_0.6']:.2%}")
        print(f"  Features with Correlation > 0.8: {metrics_b_to_a['fraction_corr_above_0.8']:.2%}")
    
    print("="*80)

def main():
    """
    Main function to run KNN baseline imputation analysis.
    
    Performs K-nearest neighbors regression for cross-platform imputation
    and compares performance against Joint VAE models.
    """
    args = parse_arguments()
    
    # ... (Argument validation code remains the same) ...
    
    np.random.seed(args.random_seed)
    
    # ... (Logging and initial setup code remains the same) ...
    
    data_a, data_b, features_a, features_b = load_and_merge_data(
        args.platform_a, args.platform_b, args.id_column
    )
    
    data_a, data_b = handle_missing_values(data_a, data_b, args.missing_strategy)
    
    # NOTE: Log transformation is now handled AFTER the train/test split to prevent data leakage.
    # The `apply_log_transformation` function is no longer called here directly on the whole dataset.

    if len(data_a) != len(data_b):
        raise ValueError(f"Sample count mismatch after missing value handling...")
    
    if len(data_a) < 10:
        logger.warning(f"Very few samples remaining ({len(data_a)}). Results may be unreliable.")
    
    if len(data_a) < args.n_neighbors:
        raise ValueError(f"Not enough samples ({len(data_a)}) for k={args.n_neighbors}...")
    
    if args.cv_folds and args.cv_folds > 0:
        # Cross validation path remains largely the same, as the changes are inside run_cross_validation
        if len(data_a) < args.cv_folds:
            raise ValueError(f"Not enough samples ({len(data_a)}) for {args.cv_folds}-fold CV...")
        
        results = run_cross_validation(data_a, data_b, features_a, features_b, args)
        is_cv = True
        
    else:
        # --- MODIFICATION START: Corrected logic for single train/test split ---
        
        # 1. First, split the original, untouched data. These are the final evaluation sets.
        original_train_a, original_test_a, original_train_b, original_test_b = train_test_split(
            data_a, data_b, 
            test_size=args.test_split,
            random_state=args.random_seed
        )
        
        logger.info(f"Data split: {len(original_train_a)} train, {len(original_test_a)} test samples")
        
        # 2. Create copies of the split data for transformation
        train_a, test_a = original_train_a.copy(), original_test_a.copy()
        train_b, test_b = original_train_b.copy(), original_test_b.copy()

        # 3. Apply log transformation (FIXED: Fit on train, apply to both train and test)
        train_a, train_b, log_params = apply_log_transformation(
            train_a, train_b, args.log_transform_a, args.log_transform_b, args.log_epsilon
        )
        # Apply the same shift derived from the training set to the test set
        if log_params['platform_a']['enabled']:
            shift_a = log_params['platform_a']['shift_value']
            test_a = pd.DataFrame(np.log(test_a + shift_a), columns=test_a.columns, index=test_a.index)
        if log_params['platform_b']['enabled']:
            shift_b = log_params['platform_b']['shift_value']
            test_b = pd.DataFrame(np.log(test_b + shift_b), columns=test_b.columns, index=test_b.index)
        
        if len(train_a) < args.n_neighbors:
            raise ValueError(f"Not enough training samples ({len(train_a)}) for k={args.n_neighbors}...")
        
        # 4. Normalize data using the (potentially log-transformed) data and get scalers
        train_a_norm, train_b_norm, test_a_norm, test_b_norm, scaler_a, scaler_b = normalize_data(
            train_a, train_b, test_a, test_b, args.normalization
        )
        
        # 5. Train KNN models
        knn_a_to_b, knn_b_to_a = train_knn_models(
            train_a_norm, train_b_norm, args.n_neighbors,
            args.kernel, args.bandwidth, args.polynomial_degree
        )
        
        # 6. Evaluate models, passing all the necessary pieces
        results = evaluate_models(
            knn_a_to_b, knn_b_to_a, 
            test_a_norm, test_b_norm, 
            original_test_a, original_test_b, # Pass original test data for evaluation
            scaler_a, scaler_b,              # Pass fitted scalers
            log_params,                       # Pass log parameters
            features_a, features_b
        )
        is_cv = False
        # --- MODIFICATION END ---
    
    # Save and print results (this part does not need to change)
    summary = save_results(results, args.output_dir, is_cv)
    print_results(results, is_cv)
    
    print(f"\nKNN baseline evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()