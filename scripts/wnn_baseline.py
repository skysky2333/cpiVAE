#!/usr/bin/env python3
"""
WNN (Weighted Nearest Neighbors) Baseline for Joint VAE comparison.

This script implements a Weighted Nearest Neighbors approach for cross-platform 
metabolite data imputation, adapting the WNN algorithm from Hao et al. 2021
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import csr_matrix, lil_matrix, diags
import logging
import time
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.metrics import compute_imputation_metrics, create_detailed_feature_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_nearestneighbor(knn_distances, neighbor=1):
    """
    For each row of knn distance matrix, returns the column with the lowest value.
    I.e. the nearest neighbor.
    
    Args:
        knn_distances: Sparse CSR matrix of KNN distances
        neighbor: Which nearest neighbor to return (1-indexed)
        
    Returns:
        Array of nearest neighbor indices
    """
    indices = knn_distances.indices
    indptr = knn_distances.indptr
    data = knn_distances.data
    nn_idx = []
    
    for i in range(knn_distances.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        if len(idx) >= neighbor:
            nn_idx.append(cols[idx[neighbor-1]])
        else:
            nn_idx.append(cols[idx[-1]])
    
    return np.array(nn_idx)


def compute_bandwidths(knn_adj, embedding, n_neighbors=20):
    """
    Compute bandwidth for each sample based on Jaccard similarity of neighborhoods.
    
    Args:
        knn_adj: Binary adjacency matrix of KNN graph
        embedding: Data embedding/features
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Array of bandwidth values for each sample
    """
    intersect = knn_adj.dot(knn_adj.T)
    indices = intersect.indices
    indptr = intersect.indptr
    data = intersect.data
    
    data = data / ((n_neighbors * 2) - data)
    
    bandwidth = []
    for i in range(intersect.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        valssort = rowvals[idx]
        numinset = len(cols)
        
        if numinset < n_neighbors:
            logger.error(f'Sample {i}: Fewer than {n_neighbors} cells with Jaccard sim > 0')
            sys.exit(1)
        
        curval = valssort[n_neighbors - 1]
        num = n_neighbors
        while num < numinset and valssort[num] == curval:
            num += 1
        minjacinset = cols[idx][:num]
        
        euc_dist = ((embedding[minjacinset, :] - embedding[i, :]) ** 2).sum(axis=1) ** 0.5
        euc_dist_sorted = np.sort(euc_dist)[::-1]
        
        bandwidth.append(np.mean(euc_dist_sorted[:n_neighbors]))

    return np.array(bandwidth)


class GaussianKernelRegressor(BaseEstimator, RegressorMixin):
    """
    A k-NN based regressor that uses a Gaussian kernel for weighting,
    mimicking Seurat's anchor weighting scheme in `TransferData`.
    """
    def __init__(self, k=50, sd_weight=1):
        self.k = k
        self.sd_weight = sd_weight

    def fit(self, X, y):
        """
        Fit the regressor. Stores the training data.
        
        Args:
            X (np.ndarray): Training data (features).
            y (np.ndarray): Target values.
        """
        self.X_train_ = X
        self.y_train_ = y
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        self.nn_.fit(self.X_train_)
        return self

    def predict(self, X):
        """
        Predict target values for query points.
        
        Args:
            X (np.ndarray): Query data.
            
        Returns:
            np.ndarray: Predicted target values.
        """
        # Find k nearest neighbors for each query point
        distances, indices = self.nn_.kneighbors(X)
        
        sigma = distances[:, -1] * self.sd_weight
        sigma[sigma == 0] = 1e-8
        
        weights = np.exp(-(distances ** 2) / (sigma[:, np.newaxis] ** 2))
        
        # Normalize weights to sum to 1
        norm = weights.sum(axis=1)
        norm[norm == 0] = 1e-8
        weights = weights / norm[:, np.newaxis]
        
        # Predict by taking the weighted average of neighbor target values
        pred = np.zeros((X.shape[0], self.y_train_.shape[1]))
        for i in range(X.shape[0]):
            neighbor_indices = indices[i]
            neighbor_weights = weights[i]
            pred[i, :] = np.dot(neighbor_weights, self.y_train_[neighbor_indices, :])
            
        return pred


def compute_affinity(dist_to_predict, dist_to_nn, bw):
    """
    Compute affinity based on prediction distance and bandwidth.
    
    Args:
        dist_to_predict: Distance to prediction
        dist_to_nn: Distance to nearest neighbor
        bw: Bandwidth parameter
        
    Returns:
        Affinity values
    """
    affinity = dist_to_predict - dist_to_nn
    affinity[affinity < 0] = 0
    affinity = affinity * -1
    
    denominator = np.maximum(bw - dist_to_nn, 1e-8)
    affinity = np.exp(affinity / denominator)
    
    return affinity


def build_knn_graph(data, n_neighbors, metric='euclidean'):
    """
    Build KNN graph using sklearn NearestNeighbors.
    
    Args:
        data: Input data matrix
        n_neighbors: Number of neighbors
        metric: Distance metric
        
    Returns:
        Tuple of (distances, adjacency_matrix)
    """
    data_norm = normalize(data, norm='l2')
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(data_norm)
    distances, indices = nbrs.kneighbors(data_norm)
    
    n_samples = data.shape[0]
    row_ind = np.repeat(np.arange(n_samples), n_neighbors)
    col_ind = indices.flatten()
    dist_data = distances.flatten()
    
    knn_distances = csr_matrix((dist_data, (row_ind, col_ind)), shape=(n_samples, n_samples))
    
    knn_adj = (knn_distances > 0).astype(int)
    
    logger.info("Data normalization completed")
    return knn_distances, knn_adj


def compute_modality_weights(data_a, data_b, n_neighbors=20, regularization=1e-4):
    """
    Compute modality weights for WNN based on within vs cross-modality predictions.
    
    Args:
        data_a: Platform A data (normalized)
        data_b: Platform B data (normalized)
        n_neighbors: Number of neighbors for KNN graphs
        regularization: Small value for numerical stability
        
    Returns:
        Tuple of (weights_a, weights_b, debug_info)
    """
    logger.info("Computing modality weights...")
    
    knn_dist_a, knn_adj_a = build_knn_graph(data_a, n_neighbors)
    knn_dist_b, knn_adj_b = build_knn_graph(data_b, n_neighbors)
    
    nn_idx_a = get_nearestneighbor(knn_dist_a, neighbor=1)
    nn_idx_b = get_nearestneighbor(knn_dist_b, neighbor=1)
    
    nn_dist_a = ((data_a - data_a[nn_idx_a, :]) ** 2).sum(axis=1) ** 0.5
    nn_dist_b = ((data_b - data_b[nn_idx_b, :]) ** 2).sum(axis=1) ** 0.5
    
    knn_adj_a_diag = knn_adj_a.copy()
    knn_adj_a_diag.setdiag(1)
    knn_adj_b_diag = knn_adj_b.copy()
    knn_adj_b_diag.setdiag(1)
    
    logger.info("Computing bandwidths on dense k-NN graphs...")
    bw_a = compute_bandwidths(knn_adj_a_diag, data_a, n_neighbors)
    bw_b = compute_bandwidths(knn_adj_b_diag, data_b, n_neighbors)

    logger.info("Filtering for mutual nearest neighbors for prediction step...")
    knn_adj_a = knn_adj_a.multiply(knn_adj_a.T)
    knn_adj_b = knn_adj_b.multiply(knn_adj_b.T)
    
    affinity_ratios = []
    
    for i, (data, knn_adj, nn_dist, bw) in enumerate([(data_a, knn_adj_a, nn_dist_a, bw_a), 
                                                      (data_b, knn_adj_b, nn_dist_b, bw_b)]):
        with np.errstate(divide='ignore', invalid='ignore'):
            within_predict = knn_adj.dot(data)
            row_sums = np.array(knn_adj.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            within_predict = within_predict / row_sums[:, np.newaxis]
            
        within_predict_dist = ((data - within_predict) ** 2).sum(axis=1) ** 0.5
        
        other_knn_adj = knn_adj_b if i == 0 else knn_adj_a
        with np.errstate(divide='ignore', invalid='ignore'):
            cross_predict = other_knn_adj.dot(data)
            row_sums_cross = np.array(other_knn_adj.sum(axis=1)).flatten()
            row_sums_cross[row_sums_cross == 0] = 1
            cross_predict = cross_predict / row_sums_cross[:, np.newaxis]
            
        cross_predict_dist = ((data - cross_predict) ** 2).sum(axis=1) ** 0.5
        
        within_affinity = compute_affinity(within_predict_dist, nn_dist, bw)
        cross_affinity = compute_affinity(cross_predict_dist, nn_dist, bw)
        
        affinity_ratio = within_affinity / (cross_affinity + regularization)
        affinity_ratios.append(affinity_ratio)
    
    # Compute final weights using sigmoid function
    weight_a = 1 / (1 + np.exp(affinity_ratios[1] - affinity_ratios[0]))
    weight_b = 1 - weight_a
    
    debug_info = {
        'affinity_ratios': affinity_ratios,
        'bandwidths': [bw_a, bw_b],
        'nn_distances': [nn_dist_a, nn_dist_b]
    }
    
    logger.info(f"Modality weights computed. Mean weight A: {weight_a.mean():.4f}, Mean weight B: {weight_b.mean():.4f}")
    
    return weight_a, weight_b, debug_info


def dist_from_adjacency(adjacency, embed1, embed2, nndist1, nndist2):
    """
    Compute distances between embeddings for neighbors defined by adjacency matrix.
    
    Args:
        adjacency: Sparse adjacency matrix defining neighbors
        embed1: First embedding
        embed2: Second embedding
        nndist1: Nearest neighbor distances for first embedding
        nndist2: Nearest neighbor distances for second embedding
        
    Returns:
        Tuple of (dist1, dist2) sparse matrices
    """
    dist1 = lil_matrix(adjacency.shape)
    dist2 = lil_matrix(adjacency.shape)
    
    indices = adjacency.indices
    indptr = adjacency.indptr
    ncells = adjacency.shape[0]
    
    tic = time.perf_counter()
    for i in range(ncells):
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
            
            # Compute distance for first embedding
            a = (((embed1[i, :] - embed1[col, :]) ** 2).sum() ** 0.5) - nndist1[i]
            if a == 0:
                dist1[i, col] = np.nan
            else:
                dist1[i, col] = a
            
            # Compute distance for second embedding
            b = (((embed2[i, :] - embed2[col, :]) ** 2).sum() ** 0.5) - nndist2[i]
            if b == 0:
                dist2[i, col] = np.nan
            else:
                dist2[i, col] = b
        
        if (i % 1000) == 0 and i > 0:
            toc = time.perf_counter()
            logger.info(f'{i} out of {ncells} processed, {toc-tic:.2f} seconds elapsed')
    
    return csr_matrix(dist1), csr_matrix(dist2)


def select_top_k_neighbors(dist_matrix, n_neighbors=20):
    """
    Select top K neighbors from a similarity/affinity matrix.
    
    Args:
        dist_matrix: Sparse similarity matrix (higher is better).
        n_neighbors: Number of neighbors to select.
        
    Returns:
        Sparse matrix with only top K neighbors.
    """
    indices = dist_matrix.indices
    indptr = dist_matrix.indptr
    data = dist_matrix.data
    nrows = dist_matrix.shape[0]
    
    final_data = []
    final_row_ind = []
    final_col_ind = []
    
    for i in range(nrows):
        # Get all non-zero similarities for the current row
        row_start = indptr[i]
        row_end = indptr[i+1]
        
        row_cols = indices[row_start:row_end]
        row_vals = data[row_start:row_end]
        
        if len(row_vals) == 0:
            continue
        
        # Sort by similarity (descending) and take top K
        # np.argsort returns indices for an ascending sort, so we take from the end.
        idx = np.argsort(row_vals)
        n_to_take = min(n_neighbors, len(idx))
        
        # Take the indices corresponding to the highest values
        top_k_local_indices = idx[-n_to_take:]
        
        # Append the data and column indices for these top k neighbors
        final_data.append(row_vals[top_k_local_indices])
        final_col_ind.append(row_cols[top_k_local_indices])
        
        # Append the current row index, repeated for each neighbor found
        final_row_ind.append(np.full(n_to_take, i, dtype=int))

    if not final_data:
        # Return an empty matrix if no neighbors were found for any sample
        return csr_matrix((nrows, dist_matrix.shape[1]))
    
    # Concatenate all lists into flat NumPy arrays
    final_data = np.concatenate(final_data)
    final_row_ind = np.concatenate(final_row_ind)
    final_col_ind = np.concatenate(final_col_ind)

    # Reconstruct the final sparse matrix
    result = csr_matrix((final_data, (final_row_ind, final_col_ind)), 
                       shape=(nrows, dist_matrix.shape[1]))
    
    return result


def build_wnn_graph(data_a, data_b, n_neighbors=20, n_neighbors_union=200, regularization=1e-4):
    """
    Build Weighted Nearest Neighbors graph from two modalities.
    
    Args:
        data_a: Platform A data (normalized)
        data_b: Platform B data (normalized)
        n_neighbors: Final number of neighbors in WNN graph
        n_neighbors_union: Number of neighbors for union graph
        regularization: Small value for numerical stability
        
    Returns:
        Sparse WNN adjacency matrix (row-normalized for smoothing)
    """
    logger.info("Building WNN graph...")
    
    # Step 1: Compute modality weights
    weights_a, weights_b, debug_info = compute_modality_weights(
        data_a, data_b, n_neighbors, regularization
    )
    
    # Step 2: Build union graph with larger neighborhood
    logger.info(f"Building union graph with {n_neighbors_union} neighbors...")
    knn_dist_a_union, knn_adj_a_union = build_knn_graph(data_a, n_neighbors_union)
    knn_dist_b_union, knn_adj_b_union = build_knn_graph(data_b, n_neighbors_union)
    
    # Create union adjacency matrix
    union_adj = ((knn_adj_a_union + knn_adj_b_union) > 0).astype(int)
    
    # Step 3: Compute distances for union neighbors
    logger.info("Computing weighted distances for union neighbors...")
    nn_dist_a = debug_info['nn_distances'][0]
    nn_dist_b = debug_info['nn_distances'][1]
    bw_a = debug_info['bandwidths'][0]
    bw_b = debug_info['bandwidths'][1]
    
    # Compute distances between embeddings for union neighbors
    full_dists = dist_from_adjacency(union_adj, data_a, data_b, nn_dist_a, nn_dist_b)
    
    weighted_dist = csr_matrix(union_adj.shape)
    
    for i, (dist, bw, nn_dist, weight) in enumerate(zip(full_dists, [bw_a, bw_b], 
                                                        [nn_dist_a, nn_dist_b], 
                                                        [weights_a, weights_b])):
        denominator = np.maximum(bw - nn_dist, 1e-8)
        scaling_diag = diags(-1 / denominator, format='csr')
        scaled_dist = scaling_diag.dot(dist)
        
        scaled_dist.data = np.exp(scaled_dist.data)
        
        nan_mask = np.isnan(scaled_dist.data)
        scaled_dist.data[nan_mask] = 1.0
        
        weighted_scaled_dist = scaled_dist.multiply(weight[:, np.newaxis])
        
        weighted_dist += weighted_scaled_dist
    
    # Step 5: Select top K neighbors
    logger.info(f"Selecting top {n_neighbors} neighbors...")
    wnn_graph = select_top_k_neighbors(weighted_dist, n_neighbors)
    
    row_sums = np.array(wnn_graph.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    row_diag = diags(1.0 / row_sums, format='csr')
    wnn_graph_normalized = row_diag.dot(wnn_graph)
    
    logger.info("WNN graph construction completed")
    
    return wnn_graph_normalized


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WNN Baseline for Joint VAE comparison')
    
    parser.add_argument(
        '--platform_a', 
        type=str, 
        required=True,
        help='Path to platform A data file (CSV or TXT)'
    )
    parser.add_argument(
        '--grid_search',
        action='store_true',
        help='Enable hyperparameter grid search using the existing train/test split'
    )
    parser.add_argument(
        '--grid_metric',
        type=str,
        default='overall_r2',
        choices=['overall_r2', 'overall_correlation', 'overall_mse', 'overall_mae'],
        help='Metric to optimize during grid search'
    )
    parser.add_argument(
        '--grid_direction',
        type=str,
        default='a_to_b',
        choices=['a_to_b', 'b_to_a', 'mean'],
        help='Which direction to optimize: A→B, B→A, or mean of both'
    )
    parser.add_argument(
        '--grid_n_neighbors',
        type=int,
        nargs='+',
        default=None,
        help='Grid values for n_neighbors'
    )
    parser.add_argument(
        '--grid_n_neighbors_union',
        type=int,
        nargs='+',
        default=None,
        help='Grid values for n_neighbors_union'
    )
    parser.add_argument(
        '--grid_n_pca_components',
        type=int,
        nargs='+',
        default=None,
        help='Grid values for n_pca_components'
    )
    parser.add_argument(
        '--grid_k_weight',
        type=int,
        nargs='+',
        default=None,
        help='Grid values for k_weight (neighbors for imputation)'
    )
    parser.add_argument(
        '--grid_wnn_regularization',
        type=float,
        nargs='+',
        default=None,
        help='Grid values for wnn_regularization'
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
        default='outputs/wnn_baseline',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_neighbors', 
        type=int, 
        default=20,
        help='Number of neighbors for final WNN graph (default: 20)'
    )
    parser.add_argument(
        '--n_neighbors_union', 
        type=int, 
        default=200,
        help='Number of neighbors for union graph construction (default: 200)'
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
        '--wnn_regularization', 
        type=float, 
        default=1e-4,
        help='Regularization parameter for WNN computations (default: 1e-4)'
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
        '--n_pca_components',
        type=int,
        default=50,
        help='Number of PCA components for dimensionality reduction. Must be > 0. (default: 50)'
    )
    parser.add_argument(
        '--k_weight',
        type=int,
        default=50,
        help='Number of neighbors for feature transfer/imputation (like Seurat k.weight). (default: 50)'
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


# Data loading and preprocessing functions (adapted from knn_baseline.py)

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


def train_seurat_like_models(train_a_norm, train_b_norm, n_pca_components,
                             n_neighbors, n_neighbors_union, regularization, random_seed,
                             k_weight):
    """
    Trains models for imputation in a way that mimics the Seurat v4 reference mapping workflow.
    This involves:
    1. Building a WNN graph on PCA-reduced data. This uses an exponential affinity kernel
       based on the Hao et al. 2021 paper.
    2. Using the WNN graph to create a "supervised" PCA (sPCA) transformation.
       NOTE: This is an approximation of Seurat's RunSPCA, which solves a generalized
       eigenvalue problem on the graph Laplacian. This implementation uses the simpler
       and empirically effective approach of performing PCA on WNN-smoothed data.
    3. Training k-NN regressors in the sPCA space on the original (non-smoothed) data.

    Args:
        train_a_norm (np.ndarray): Normalized training data for platform A.
        train_b_norm (np.ndarray): Normalized training data for platform B.
        n_pca_components (int): Number of components for both standard PCA and sPCA.
        n_neighbors (int): Number of neighbors for final WNN graph.
        n_neighbors_union (int): Number of neighbors for union graph.
        regularization (float): Regularization for WNN weight calculation.
        random_seed (int): Random seed for reproducibility.
        k_weight (int): Number of neighbors for the downstream k-NN regressor.

    Returns:
        tuple: A tuple containing the models for A->B and B->A predictions.
               Each element is a tuple of (GaussianKernelRegressor, PCA).
    """
    logger.info("Training Seurat-like models with sPCA...")

    # Step 1: Standard PCA for WNN graph construction
    # Use fewer components for WNN construction if the feature count is low
    pca_comps_for_wnn = min(n_pca_components, train_a_norm.shape[1], train_b_norm.shape[1])
    if pca_comps_for_wnn < n_pca_components:
        logger.warning(f"Using {pca_comps_for_wnn} PCA components for WNN graph instead of {n_pca_components} due to low feature count.")
    
    pca_a_wnn = PCA(n_components=pca_comps_for_wnn, random_state=random_seed)
    pca_b_wnn = PCA(n_components=pca_comps_for_wnn, random_state=random_seed)
    train_a_pca = pca_a_wnn.fit_transform(train_a_norm)
    train_b_pca = pca_b_wnn.fit_transform(train_b_norm)
    
    # Step 2: Build WNN graph
    wnn_graph = build_wnn_graph(
        train_a_pca, train_b_pca,
        n_neighbors=n_neighbors,
        n_neighbors_union=n_neighbors_union,
        regularization=regularization
    )

    # --- Train models for A -> B prediction ---
    # 3a: Smooth data A to define the sPCA space
    train_a_smooth = wnn_graph.dot(train_a_norm)
    # 3b: Compute the sPCA transformation for modality A
    n_spca_a = min(n_pca_components, train_a_norm.shape[1])
    if n_spca_a < n_pca_components:
        logger.warning(f"sPCA for A is using {n_spca_a} components instead of {n_pca_components}.")
    spca_a = PCA(n_components=n_spca_a, random_state=random_seed)
    spca_a.fit(train_a_smooth)
    # 3c: Transform modality A data into the sPCA space
    train_a_spca = spca_a.transform(train_a_norm)
    # 3d: Train k-NN regressor in sPCA space to predict original modality B data
    gkr_a_to_b = GaussianKernelRegressor(k=k_weight)
    gkr_a_to_b.fit(train_a_spca, train_b_norm)

    # --- Train models for B -> A prediction ---
    # 4a: Smooth data B to define the sPCA space
    train_b_smooth = wnn_graph.dot(train_b_norm)
    # 4b: Compute the sPCA transformation for modality B
    n_spca_b = min(n_pca_components, train_b_norm.shape[1])
    if n_spca_b < n_pca_components:
        logger.warning(f"sPCA for B is using {n_spca_b} components instead of {n_pca_components}.")
    spca_b = PCA(n_components=n_spca_b, random_state=random_seed)
    spca_b.fit(train_b_smooth)
    # 4c: Transform modality B data into the sPCA space
    train_b_spca = spca_b.transform(train_b_norm)
    # 4d: Train k-NN regressor in sPCA space to predict original modality A data
    gkr_b_to_a = GaussianKernelRegressor(k=k_weight)
    gkr_b_to_a.fit(train_b_spca, train_a_norm)

    logger.info("Seurat-like models training completed.")
    return (gkr_a_to_b, spca_a), (gkr_b_to_a, spca_b)


def evaluate_models(models_a_to_b, models_b_to_a, test_a_norm, test_b_norm,
                    original_test_a, original_test_b, scaler_a, scaler_b,
                    log_params, features_a, features_b):
    """
    Evaluate sPCA-based models and compute metrics after inverse-transforming
    predictions back to the original data space.
    
    Args:
        models_a_to_b: Tuple of (GaussianKernelRegressor for A->B, sPCA model for A)
        models_b_to_a: Tuple of (GaussianKernelRegressor for B->A, sPCA model for B)
        test_a_norm, test_b_norm: Normalized test data for making predictions
        original_test_a, original_test_b: Test data in the original data space
        scaler_a, scaler_b: Fitted scalers for inverse normalization
        log_params: Log transformation parameters for inverse log transform
        features_a, features_b: Feature names
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating sPCA-based models on original data scale...")
    
    gkr_a_to_b, spca_a = models_a_to_b
    gkr_b_to_a, spca_b = models_b_to_a

    # 1. Make predictions using the sPCA -> kNN pipeline
    
    # A -> B Prediction
    test_a_spca = spca_a.transform(test_a_norm)
    pred_b_from_a_norm = gkr_a_to_b.predict(test_a_spca)
    
    # B -> A Prediction
    test_b_spca = spca_b.transform(test_b_norm)
    pred_a_from_b_norm = gkr_b_to_a.predict(test_b_spca)
    
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

    # 3. Convert original test DataFrames to NumPy arrays before computing metrics
    logger.info("Computing metrics against original test data.")
    metrics_a_to_b = compute_imputation_metrics(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    metrics_b_to_a = compute_imputation_metrics(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    
    # Also convert for the detailed report function
    report_a_to_b = create_detailed_feature_report(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    report_b_to_a = create_detailed_feature_report(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    
    logger.info("Model evaluation completed")
    
    return {
        'metrics_a_to_b': metrics_a_to_b,
        'metrics_b_to_a': metrics_b_to_a,
        'report_a_to_b': report_a_to_b,
        'report_b_to_a': report_b_to_a,
    }


def evaluate_models_cv_fold(models_a_to_b, models_b_to_a, test_a_norm, test_b_norm,
                            original_test_a, original_test_b, scaler_a, scaler_b,
                            log_params, features_a, features_b):
    """
    Evaluate sPCA-based models for a single CV fold.
    
    Args:
        models_a_to_b: Tuple of (GaussianKernelRegressor for A->B, sPCA model for A)
        models_b_to_a: Tuple of (GaussianKernelRegressor for B->A, sPCA model for B)
        test_a_norm, test_b_norm: Normalized test data for making predictions
        original_test_a, original_test_b: Test data in the original data space
        scaler_a, scaler_b: Fitted scalers for inverse normalization
        log_params: Log transformation parameters for inverse log transform
        features_a, features_b: Feature names
        
    Returns:
        Dictionary of evaluation results for this fold
    """
    gkr_a_to_b, spca_a = models_a_to_b
    gkr_b_to_a, spca_b = models_b_to_a

    # 1. Make predictions using the sPCA -> kNN pipeline
    test_a_spca = spca_a.transform(test_a_norm)
    pred_b_from_a_norm = gkr_a_to_b.predict(test_a_spca)

    test_b_spca = spca_b.transform(test_b_norm)
    pred_a_from_b_norm = gkr_b_to_a.predict(test_b_spca)

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

    # 3. Convert original test DataFrames to NumPy arrays before computing metrics
    metrics_a_to_b = compute_imputation_metrics(original_test_b.to_numpy(), pred_b_from_a_orig, features_b)
    metrics_b_to_a = compute_imputation_metrics(original_test_a.to_numpy(), pred_a_from_b_orig, features_a)
    
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

        # Keep the original test sets for final evaluation
        original_fold_test_a = test_a.copy()
        original_fold_test_b = test_b.copy()
        
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
        
        # Normalize data and capture the fitted scalers
        train_a_norm, train_b_norm, test_a_norm, test_b_norm, scaler_a, scaler_b = normalize_data(
            train_a, train_b, test_a, test_b, args.normalization
        )
        
        # Train Seurat-like models (WNN -> sPCA -> kNN)
        models_a_to_b, models_b_to_a = train_seurat_like_models(
            train_a_norm, train_b_norm,
            n_pca_components=args.n_pca_components,
            n_neighbors=args.n_neighbors,
            n_neighbors_union=args.n_neighbors_union,
            regularization=args.wnn_regularization,
            random_seed=args.random_seed,
            k_weight=args.k_weight
        )
        
        # Evaluate models for this fold
        fold_result = evaluate_models_cv_fold(
            models_a_to_b, models_b_to_a,
            test_a_norm, test_b_norm,
            original_fold_test_a, original_fold_test_b,
            scaler_a, scaler_b,
            log_params,
            features_a, features_b
        )
        
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
        filename = 'wnn_baseline_cv_summary.csv'
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
        filename = 'wnn_baseline_summary.csv'
    
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
        print(f"WNN BASELINE CROSS VALIDATION RESULTS ({results['n_folds']} folds)")
    else:
        print("WNN BASELINE EVALUATION RESULTS")
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


def perform_cross_imputation(platform_a, platform_b, platform_impute, impute_target,
                             n_pca_components=50, n_neighbors=20, n_neighbors_union=200,
                             wnn_regularization=1e-4, k_weight=50, random_seed=42,
                             log_transform_a=False, log_transform_b=False, log_epsilon=1e-8,
                             output_dir=None):
    """
    Perform cross-imputation using the WNN-based model.
    
    Args:
        platform_a: Path to platform A file
        platform_b: Path to platform B file  
        platform_impute: Path to file that needs imputation
        impute_target: 'a' or 'b' - which platform to impute as
        n_pca_components: Number of PCA components
        n_neighbors: Number of neighbors for WNN graph
        n_neighbors_union: Number of neighbors for union graph
        wnn_regularization: WNN regularization parameter
        k_weight: Number of neighbors for imputation
        random_seed: Random seed
        log_transform_a: Log transform platform A
        log_transform_b: Log transform platform B
        log_epsilon: Log epsilon value
        output_dir: Output directory
    
    Returns:
        Path to the imputed file
    """
    print(f"\nPerforming WNN cross-imputation for target platform {impute_target.upper()}...")
    
    # Load and prepare training data (same as in main experiments)
    data_a, data_b, features_a, features_b = load_and_merge_data(platform_a, platform_b)
    data_a, data_b = handle_missing_values(data_a, data_b, 'mean')  # Use mean imputation for training
    
    # Apply log transformation to training data
    train_a, train_b, log_params = apply_log_transformation(
        data_a, data_b, log_transform_a, log_transform_b, log_epsilon
    )
    
    # Normalize training data (using zscore as default like in main script)
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    train_a_norm = scaler_a.fit_transform(train_a)
    train_b_norm = scaler_b.fit_transform(train_b)
    
    # Train WNN models
    models_a_to_b, models_b_to_a = train_seurat_like_models(
        train_a_norm, train_b_norm,
        n_pca_components=n_pca_components,
        n_neighbors=n_neighbors,
        n_neighbors_union=n_neighbors_union,
        regularization=wnn_regularization,
        random_seed=random_seed,
        k_weight=k_weight
    )
    
    # Extract the appropriate model and components based on impute target
    if impute_target == 'a':
        # We want to impute platform A data, so we need B->A model
        gkr_model, spca_transform = models_b_to_a
        target_scaler = scaler_a
        target_log_params = log_params['platform_a']
        target_features = features_a
        input_scaler = scaler_b
        input_log_params = log_params['platform_b']
    else:
        # We want to impute platform B data, so we need A->B model  
        gkr_model, spca_transform = models_a_to_b
        target_scaler = scaler_b
        target_log_params = log_params['platform_b']
        target_features = features_b
        input_scaler = scaler_a
        input_log_params = log_params['platform_a']
    
    # Load the file to be imputed
    print(f"Loading imputation file: {platform_impute}")
    sep_impute = '\t' if Path(platform_impute).suffix.lower() == '.txt' else ','
    impute_df = pd.read_csv(platform_impute, sep=sep_impute)
    
    # Use first column as ID if not specified
    id_column = impute_df.columns[0]
    
    # Extract feature columns (all except ID column)
    expected_input_features = [col for col in impute_df.columns if col != id_column]
    impute_data = impute_df[expected_input_features]
    
    # Handle missing values in impute data (using mean imputation)
    imputer = SimpleImputer(strategy='mean')
    impute_data_clean = pd.DataFrame(
        imputer.fit_transform(impute_data),
        columns=impute_data.columns,
        index=impute_data.index
    )
    
    # Apply transformations to impute data based on input platform
    if input_log_params['enabled']:
        shift_value = input_log_params['shift_value']
        impute_data_log = pd.DataFrame(
            np.log(impute_data_clean + shift_value), 
            columns=impute_data_clean.columns, 
            index=impute_data_clean.index
        )
    else:
        impute_data_log = impute_data_clean
    
    # Normalize impute data
    impute_data_norm = input_scaler.transform(impute_data_log)
    
    # Transform to sPCA space and perform imputation
    print(f"Performing WNN-based imputation...")
    impute_data_spca = spca_transform.transform(impute_data_norm)
    imputed_norm = gkr_model.predict(impute_data_spca)
    
    # Inverse transform predictions to original scale for target platform
    # Step 1: Inverse normalize (normalized -> log space or original space)
    imputed_denorm = target_scaler.inverse_transform(imputed_norm)
    
    # Step 2: Inverse log transform if target platform had log transformation
    if target_log_params['enabled']:
        shift_value = target_log_params['shift_value']
        imputed_original = np.exp(imputed_denorm) - shift_value
    else:
        imputed_original = imputed_denorm
    
    # Create output DataFrame
    imputed_df = pd.DataFrame(imputed_original, columns=target_features, index=impute_df.index)
    
    # Add ID column back
    output_df = pd.concat([impute_df[[id_column]], imputed_df], axis=1)
    
    # Generate output filename
    impute_path = Path(platform_impute)
    output_filename = f"{impute_path.stem}_cross_imputed_{impute_target}{impute_path.suffix}"
    
    if output_dir:
        output_path = Path(output_dir) / output_filename
    else:
        output_path = impute_path.parent / output_filename
    
    # Save imputed file
    output_df.to_csv(output_path, index=False)
    print(f"Cross-imputed file saved to: {output_path}")
    
    return output_path


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Validate arguments
    if not Path(args.platform_a).exists():
        logger.error(f"Platform A file not found: {args.platform_a}")
        sys.exit(1)
    if not Path(args.platform_b).exists():
        logger.error(f"Platform B file not found: {args.platform_b}")
        sys.exit(1)
    
    # Validate imputation arguments
    if args.platform_impute is not None and args.impute_target is None:
        logger.error(f"--impute_target must be specified when --platform_impute is provided")
        sys.exit(1)
    if args.impute_target is not None and args.platform_impute is None:
        logger.error(f"--platform_impute must be specified when --impute_target is provided")
        sys.exit(1)
    if args.platform_impute is not None and not Path(args.platform_impute).exists():
        logger.error(f"Platform impute file not found: {args.platform_impute}")
        sys.exit(1)
    
    if args.n_neighbors < 1:
        logger.error(f"n_neighbors must be >= 1, got: {args.n_neighbors}")
        sys.exit(1)
    
    if args.n_neighbors_union < args.n_neighbors:
        logger.error(f"n_neighbors_union ({args.n_neighbors_union}) must be >= n_neighbors ({args.n_neighbors})")
        sys.exit(1)
    
    if args.wnn_regularization <= 0:
        logger.error(f"wnn_regularization must be > 0, got: {args.wnn_regularization}")
        sys.exit(1)

    if args.n_pca_components <= 0:
        logger.error(f"n_pca_components must be > 0 for the Seurat-like workflow, got: {args.n_pca_components}")
        sys.exit(1)
    
    if args.k_weight < 1:
        logger.error(f"k_weight must be >= 1, got: {args.k_weight}")
        sys.exit(1)
    
    # Log experiment parameters
    logger.info("Starting WNN baseline experiment...")
    logger.info(f"Platform A file: {args.platform_a}")
    logger.info(f"Platform B file: {args.platform_b}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"n_neighbors: {args.n_neighbors}")
    logger.info(f"n_neighbors_union: {args.n_neighbors_union}")
    logger.info(f"WNN regularization: {args.wnn_regularization}")
    logger.info(f"Normalization: {args.normalization}")
    logger.info(f"Missing value strategy: {args.missing_strategy}")
    logger.info(f"Imputation k (k_weight): {args.k_weight}")
    
    if args.log_transform_a or args.log_transform_b:
        logger.info(f"Log transformation: Platform A={'enabled' if args.log_transform_a else 'disabled'}, Platform B={'enabled' if args.log_transform_b else 'disabled'}, epsilon={args.log_epsilon}")
    
    # Show cross-imputation settings
    if args.platform_impute is not None:
        logger.info(f"Cross-imputation enabled:")
        logger.info(f"  Input file: {args.platform_impute}")
        logger.info(f"  Target platform: {args.impute_target.upper()}")
    
    if args.n_pca_components > 0:
        logger.info(f"PCA components: {args.n_pca_components}")
    
    # Load and preprocess data
    data_a, data_b, features_a, features_b = load_and_merge_data(
        args.platform_a, args.platform_b, args.id_column
    )
    
    data_a, data_b = handle_missing_values(data_a, data_b, args.missing_strategy)
    
    if len(data_a) != len(data_b):
        logger.error(f"Sample count mismatch after missing value handling: {len(data_a)} vs {len(data_b)}")
        sys.exit(1)
    
    if len(data_a) < 10:
        logger.warning(f"Very few samples remaining ({len(data_a)}). Results may be unreliable.")
    
    if len(data_a) < args.n_neighbors:
        logger.error(f"Not enough samples ({len(data_a)}) for k={args.n_neighbors}")
        sys.exit(1)
    
    # Run experiment
    if args.cv_folds and args.cv_folds > 0:
        # Cross validation
        if len(data_a) < args.cv_folds:
            logger.error(f"Not enough samples ({len(data_a)}) for {args.cv_folds}-fold CV")
            sys.exit(1)
        
        logger.info(f"Using {args.cv_folds}-fold cross validation")
        results = run_cross_validation(data_a, data_b, features_a, features_b, args)
        is_cv = True
        
    else:
        # Single train/test split
        logger.info(f"Using train/test split with test ratio: {args.test_split}")
        
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
            logger.error(f"Not enough training samples ({len(train_a)}) for k={args.n_neighbors}")
            sys.exit(1)
        
        # 4. Normalize data using the (potentially log-transformed) data and get scalers
        train_a_norm, train_b_norm, test_a_norm, test_b_norm, scaler_a, scaler_b = normalize_data(
            train_a, train_b, test_a, test_b, args.normalization
        )
        
        best_params = None
        if args.grid_search:
            logger.info("Grid search enabled. Exploring hyperparameter combinations...")

            # Build grids with sensible defaults if not provided
            grid_n_neighbors = args.grid_n_neighbors or [args.n_neighbors, max(10, args.n_neighbors // 2), args.n_neighbors * 2]
            grid_n_neighbors = sorted(set([n for n in grid_n_neighbors if n >= 1]))

            grid_n_neighbors_union = args.grid_n_neighbors_union or [max(args.n_neighbors_union // 2, args.n_neighbors), args.n_neighbors_union, args.n_neighbors_union * 2]
            grid_n_neighbors_union = sorted(set([n for n in grid_n_neighbors_union if n >= 1]))

            grid_n_pca_components = args.grid_n_pca_components or [max(10, args.n_pca_components // 2), args.n_pca_components, args.n_pca_components * 2]
            grid_n_pca_components = sorted(set([n for n in grid_n_pca_components if n >= 1]))

            grid_k_weight = args.grid_k_weight or [max(10, args.k_weight // 2), args.k_weight, args.k_weight * 2]
            grid_k_weight = sorted(set([n for n in grid_k_weight if n >= 1]))

            grid_wnn_regularization = args.grid_wnn_regularization or [1e-5, args.wnn_regularization, 1e-3]
            grid_wnn_regularization = sorted(set([float(x) for x in grid_wnn_regularization if x > 0]))

            minimize_metrics = {'overall_mse', 'overall_mae'}

            def transform_score(metric_name: str, value: float) -> float:
                return -value if metric_name in minimize_metrics else value

            n_train_samples = train_a_norm.shape[0]
            max_valid_neighbors = n_train_samples
            max_valid_k = n_train_samples

            # Infer optimization direction from impute_target if provided
            effective_direction = args.grid_direction
            if args.impute_target is not None:
                if args.impute_target == 'a':
                    effective_direction = 'b_to_a'
                elif args.impute_target == 'b':
                    effective_direction = 'a_to_b'
                logger.info(f"Optimization direction inferred from impute_target='{args.impute_target}': {effective_direction}")

            all_rows = []
            best_objective = -np.inf
            best_result = None
            best_params = None

            for (n_neighbors_cur,
                 n_neighbors_union_cur,
                 n_pca_components_cur,
                 k_weight_cur,
                 wnn_reg_cur) in itertools.product(
                     grid_n_neighbors,
                     grid_n_neighbors_union,
                     grid_n_pca_components,
                     grid_k_weight,
                     grid_wnn_regularization
                 ):

                if n_neighbors_cur > max_valid_neighbors:
                    continue
                if n_neighbors_union_cur < n_neighbors_cur or n_neighbors_union_cur > max_valid_neighbors:
                    continue
                if k_weight_cur > max_valid_k:
                    continue

                try:
                    models_a_to_b, models_b_to_a = train_seurat_like_models(
                        train_a_norm, train_b_norm,
                        n_pca_components=n_pca_components_cur,
                        n_neighbors=n_neighbors_cur,
                        n_neighbors_union=n_neighbors_union_cur,
                        regularization=wnn_reg_cur,
                        random_seed=args.random_seed,
                        k_weight=k_weight_cur
                    )

                    res = evaluate_models(
                        models_a_to_b, models_b_to_a,
                        test_a_norm, test_b_norm,
                        original_test_a, original_test_b,
                        scaler_a, scaler_b,
                        log_params,
                        features_a, features_b
                    )

                    ma = res['metrics_a_to_b']
                    mb = res['metrics_b_to_a']

                    score_a_raw = ma.get(args.grid_metric, None)
                    score_b_raw = mb.get(args.grid_metric, None)
                    if score_a_raw is None or score_b_raw is None:
                        continue

                    score_a = transform_score(args.grid_metric, score_a_raw)
                    score_b = transform_score(args.grid_metric, score_b_raw)

                    if effective_direction == 'a_to_b':
                        objective = score_a
                    elif effective_direction == 'b_to_a':
                        objective = score_b
                    else:
                        objective = 0.5 * (score_a + score_b)

                    all_rows.append({
                        'n_neighbors': n_neighbors_cur,
                        'n_neighbors_union': n_neighbors_union_cur,
                        'n_pca_components': n_pca_components_cur,
                        'k_weight': k_weight_cur,
                        'wnn_regularization': wnn_reg_cur,
                        'a_to_b_overall_r2': ma.get('overall_r2'),
                        'b_to_a_overall_r2': mb.get('overall_r2'),
                        'a_to_b_overall_correlation': ma.get('overall_correlation'),
                        'b_to_a_overall_correlation': mb.get('overall_correlation'),
                        'metric_a': score_a_raw,
                        'metric_b': score_b_raw,
                        'objective': objective,
                    })

                    if objective > best_objective:
                        best_objective = objective
                        best_result = res
                        best_params = {
                            'n_neighbors': n_neighbors_cur,
                            'n_neighbors_union': n_neighbors_union_cur,
                            'n_pca_components': n_pca_components_cur,
                            'k_weight': k_weight_cur,
                            'wnn_regularization': wnn_reg_cur,
                            'grid_metric': args.grid_metric,
                            'grid_direction': effective_direction,
                        }
                        logger.info(f"New best objective {best_objective:.6f} with params: {best_params}")

                except SystemExit:
                    raise
                except Exception as e:
                    logger.warning(f"Skipping combination due to error: n_neighbors={n_neighbors_cur}, n_neighbors_union={n_neighbors_union_cur}, n_pca_components={n_pca_components_cur}, k_weight={k_weight_cur}, wnn_reg={wnn_reg_cur}. Error: {e}")
                    continue

            if not all_rows or best_result is None:
                logger.error("Grid search failed to find a valid configuration.")
                sys.exit(1)

            # Save grid results and best params
            grid_df = pd.DataFrame(all_rows)
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            grid_csv_path = output_path / 'grid_search_results.csv'
            grid_df.to_csv(grid_csv_path, index=False)

            best_json_path = output_path / 'best_grid_params.json'
            with open(best_json_path, 'w') as f:
                json.dump(best_params, f, indent=2)

            logger.info(f"Grid search completed. Results saved to {grid_csv_path}. Best params saved to {best_json_path}.")

            results = best_result
            is_cv = False

        else:
            # 5. Train Seurat-like models (WNN -> sPCA -> kNN)
            models_a_to_b, models_b_to_a = train_seurat_like_models(
                train_a_norm, train_b_norm,
                n_pca_components=args.n_pca_components,
                n_neighbors=args.n_neighbors,
                n_neighbors_union=args.n_neighbors_union,
                regularization=args.wnn_regularization,
                random_seed=args.random_seed,
                k_weight=args.k_weight
            )
            
            # 6. Evaluate models, passing all the necessary pieces
            results = evaluate_models(
                models_a_to_b, models_b_to_a,
                test_a_norm, test_b_norm,
                original_test_a, original_test_b, # Pass original test data for evaluation
                scaler_a, scaler_b,              # Pass fitted scalers
                log_params,                       # Pass log parameters
                features_a, features_b
            )
            is_cv = False
    
    # Save and print results
    summary = save_results(results, args.output_dir, is_cv)
    print_results(results, is_cv)
    
    # Perform cross-imputation if requested
    if args.platform_impute is not None:
        print(f"\nPerforming cross-imputation...")
        try:
            # Use best params from grid search if available, else fallback to args
            n_neighbors_imp = best_params['n_neighbors'] if ('best_params' in locals() and best_params) else args.n_neighbors
            n_neighbors_union_imp = best_params['n_neighbors_union'] if ('best_params' in locals() and best_params) else args.n_neighbors_union
            n_pca_components_imp = best_params['n_pca_components'] if ('best_params' in locals() and best_params) else args.n_pca_components
            k_weight_imp = best_params['k_weight'] if ('best_params' in locals() and best_params) else args.k_weight
            wnn_regularization_imp = best_params['wnn_regularization'] if ('best_params' in locals() and best_params) else args.wnn_regularization

            if best_params is not None:
                logger.info(f"Cross-imputation will use best grid params: {best_params}")

            imputed_file = perform_cross_imputation(
                args.platform_a, args.platform_b, args.platform_impute, 
                args.impute_target,
                n_pca_components=n_pca_components_imp,
                n_neighbors=n_neighbors_imp,
                n_neighbors_union=n_neighbors_union_imp,
                wnn_regularization=wnn_regularization_imp,
                k_weight=k_weight_imp,
                random_seed=args.random_seed,
                log_transform_a=args.log_transform_a,
                log_transform_b=args.log_transform_b,
                log_epsilon=args.log_epsilon,
                output_dir=args.output_dir
            )
            print(f"\n✓ Cross-imputation completed successfully!")
            print(f"Imputed file: {imputed_file}")
        except Exception as e:
            print(f"\n✗ Cross-imputation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nWNN baseline evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 