#!/usr/bin/env python3
"""
Comprehensive Cross-Platform Proteomics Imputation Analysis Script
Generates Nature-ready figures for comparing imputation methods across platforms.
"""

import argparse
import os
import sys
import warnings
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress, binomtest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import network analysis module
try:
    from network_analysis import NetworkAnalyzer
    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    print("⚠️  Network analysis module not available. Network features will be skipped.")
    NETWORK_ANALYSIS_AVAILABLE = False

# Set up plotting style for Nature journal standards
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    # Fallback for older seaborn versions
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.figsize': (5, 5)
})

NATURE_COLORS = {
    'primary': '#e64b35',
    'secondary': '#4dbbd5',
    'accent': '#00a087',
    'neutral': '#3c5488',
    'highlight': '#f39b7f',
    'alternative_1': '#bc3c29',
    'alternative_2': '#0072b5',
    'alternative_3': '#e18727',
    'alternative_4': '#20854e',
    'alternative_5': '#7876b1'
}

@dataclass
class AnalysisData:
    """
    Container for all analysis data and metadata.
    
    Contains raw data matrices, method metadata, optional additional methods,
    computed metrics, and phenotype analysis data for cross-platform comparison.
    """
    truth_a: pd.DataFrame
    truth_b: pd.DataFrame
    imp_a_m1: pd.DataFrame
    imp_a_m2: pd.DataFrame
    imp_b_m1: pd.DataFrame
    imp_b_m2: pd.DataFrame
    method1_name: str
    method2_name: str
    platform_a_name: str
    platform_b_name: str
    imp_a_m3: Optional[pd.DataFrame] = None
    imp_a_m4: Optional[pd.DataFrame] = None
    imp_b_m3: Optional[pd.DataFrame] = None
    imp_b_m4: Optional[pd.DataFrame] = None
    method3_name: Optional[str] = None
    method4_name: Optional[str] = None
    groups: Optional[pd.Series] = None
    metrics: Dict[str, pd.DataFrame] = None
    cross_platform_r2: pd.Series = None
    spearman_metrics: Dict[str, pd.DataFrame] = None
    cross_platform_rho: pd.Series = None
    phenotype_data: Optional[pd.DataFrame] = None
    binary_pheno_cols: Optional[List[str]] = None
    continuous_pheno_cols: Optional[List[str]] = None

class ComparativeAnalyzer:
    """Main class for comprehensive cross-platform imputation analysis"""
    
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.git_hash = self._get_git_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize network analyzer if available
        self.network_analyzer = None
        if NETWORK_ANALYSIS_AVAILABLE:
            self.network_analyzer = NetworkAnalyzer(output_dir=str(self.output_dir))
        
    def _get_git_hash(self) -> str:
        """Get current git commit hash for reproducibility"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def load_and_validate_data(self, file_paths: Dict[str, str], 
                              method1_name: str, method2_name: str,
                              platform_a_name: str, platform_b_name: str,
                              method3_name: Optional[str] = None, method4_name: Optional[str] = None,
                              transpose: bool = False) -> AnalysisData:
        """Load and validate all input files"""
        print("Loading and validating input files...")
        
        # Load all matrices
        data_frames = {}
        for key, path in file_paths.items():
            print(f"  Loading {key}: {path}")
            df = pd.read_csv(path, index_col=0)
            
            # Transpose if needed (for data where rows=samples, columns=features)
            if transpose:
                df = df.T
                print(f"    Transposed {key}: {df.shape}")
            
            data_frames[key] = df
        
        # Check for groups (could be row or column)
        # Note: Only checking truth_a by convention - if groups exist in other matrices,
        # they should be consistent or provided via separate group file
        groups = None
        
        # Check for groups row (feature-level grouping)
        first_col_values = data_frames['truth_a'].index.tolist()
        if 'groups' in first_col_values:
            print("  Found 'groups' row - extracting feature grouping information")
            groups_idx = first_col_values.index('groups')
            groups = data_frames['truth_a'].iloc[groups_idx, :].copy()
            
            # Remove groups row from all dataframes
            for key in data_frames:
                data_frames[key] = data_frames[key].drop(data_frames[key].index[groups_idx])
        
        # Check for groups column (sample-level grouping)
        elif 'groups' in data_frames['truth_a'].columns:
            print("  Found 'groups' column - extracting sample grouping information")
            groups = data_frames['truth_a']['groups'].copy()
            
            # Remove groups column from all dataframes
            for key in data_frames:
                if 'groups' in data_frames[key].columns:
                    data_frames[key] = data_frames[key].drop('groups', axis=1)
        
        # Harmonize dimensions within each platform (not across platforms)
        print("  Harmonizing matrix dimensions within each platform...")
        
        # Platform A harmonization - include ALL available methods
        platform_a_keys = ['truth_a', 'imp_a_m1', 'imp_a_m2', 'imp_a_m3', 'imp_a_m4']
        platform_a_common_features = None
        platform_a_common_samples = None
        
        for key in platform_a_keys:
            if key in data_frames:
                df = data_frames[key]
                if platform_a_common_features is None:
                    platform_a_common_features = set(df.index)
                    platform_a_common_samples = set(df.columns)
                else:
                    platform_a_common_features &= set(df.index)
                    platform_a_common_samples &= set(df.columns)
        
        # Platform B harmonization - include ALL available methods
        platform_b_keys = ['truth_b', 'imp_b_m1', 'imp_b_m2', 'imp_b_m3', 'imp_b_m4']
        platform_b_common_features = None
        platform_b_common_samples = None
        
        for key in platform_b_keys:
            if key in data_frames:
                df = data_frames[key]
                if platform_b_common_features is None:
                    platform_b_common_features = set(df.index)
                    platform_b_common_samples = set(df.columns)
                else:
                    platform_b_common_features &= set(df.index)
                    platform_b_common_samples &= set(df.columns)
        
        # Find common samples across all platforms (samples should be the same)
        all_common_samples = platform_a_common_samples & platform_b_common_samples
        
        print(f"  Platform A features: {len(platform_a_common_features)}")
        print(f"  Platform B features: {len(platform_b_common_features)}")
        print(f"  Common samples across platforms: {len(all_common_samples)}")
        
        # Find overlapping features for cross-platform analysis
        overlapping_features = platform_a_common_features & platform_b_common_features
        print(f"  Overlapping features (for cross-platform R²): {len(overlapping_features)}")
        
        # Reorder and subset Platform A dataframes
        platform_a_features = sorted(list(platform_a_common_features))
        common_samples_sorted = sorted(list(all_common_samples))
        
        print(f"  Harmonizing Platform A data to {len(platform_a_features)} features x {len(common_samples_sorted)} samples")
        for key in platform_a_keys:
            if key in data_frames:
                original_shape = data_frames[key].shape
                data_frames[key] = data_frames[key].loc[platform_a_features, common_samples_sorted]
                print(f"    {key}: {original_shape} -> {data_frames[key].shape}")
        
        # Reorder and subset Platform B dataframes  
        platform_b_features = sorted(list(platform_b_common_features))
        
        print(f"  Harmonizing Platform B data to {len(platform_b_features)} features x {len(common_samples_sorted)} samples")
        for key in platform_b_keys:
            if key in data_frames:
                original_shape = data_frames[key].shape
                data_frames[key] = data_frames[key].loc[platform_b_features, common_samples_sorted]
                print(f"    {key}: {original_shape} -> {data_frames[key].shape}")
        
        # Convert to float64
        for key in data_frames:
            data_frames[key] = data_frames[key].astype(np.float64)
        
        # Store overlapping features for cross-platform analysis
        self.overlapping_features = sorted(list(overlapping_features))
        
        # Validate data consistency across methods
        print("  Validating data consistency across methods...")
        self._validate_data_consistency(data_frames)
        
        # Create AnalysisData object
        analysis_data = AnalysisData(
            truth_a=data_frames['truth_a'],
            truth_b=data_frames['truth_b'],
            imp_a_m1=data_frames['imp_a_m1'],
            imp_a_m2=data_frames['imp_a_m2'],
            imp_b_m1=data_frames['imp_b_m1'],
            imp_b_m2=data_frames['imp_b_m2'],
            imp_a_m3=data_frames.get('imp_a_m3'),
            imp_a_m4=data_frames.get('imp_a_m4'),
            imp_b_m3=data_frames.get('imp_b_m3'),
            imp_b_m4=data_frames.get('imp_b_m4'),
            method1_name=method1_name,
            method2_name=method2_name,
            method3_name=method3_name,
            method4_name=method4_name,
            platform_a_name=platform_a_name,
            platform_b_name=platform_b_name,
            groups=groups
        )
        
        return analysis_data
    
    def _validate_data_consistency(self, data_frames: Dict[str, pd.DataFrame]):
        """Validate that all data matrices have consistent preprocessing"""
        
        # Group by platform for validation
        platform_a_matrices = {k: v for k, v in data_frames.items() if k.endswith('_a')}
        platform_b_matrices = {k: v for k, v in data_frames.items() if k.endswith('_b')}
        
        def validate_platform_consistency(matrices, platform_name):
            if not matrices:
                return
                
            reference_key = list(matrices.keys())[0]
            reference_df = matrices[reference_key]
            ref_features = set(reference_df.index)
            ref_samples = set(reference_df.columns)
            
            inconsistencies = []
            
            for key, df in matrices.items():
                # Check features
                if set(df.index) != ref_features:
                    missing_features = ref_features - set(df.index)
                    extra_features = set(df.index) - ref_features
                    inconsistencies.append(f"    {key}: features differ from {reference_key}")
                    if missing_features:
                        inconsistencies.append(f"      Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
                    if extra_features:
                        inconsistencies.append(f"      Extra {len(extra_features)} features: {list(extra_features)[:5]}...")
                
                # Check samples
                if set(df.columns) != ref_samples:
                    missing_samples = ref_samples - set(df.columns)
                    extra_samples = set(df.columns) - ref_samples
                    inconsistencies.append(f"    {key}: samples differ from {reference_key}")
                    if missing_samples:
                        inconsistencies.append(f"      Missing {len(missing_samples)} samples: {list(missing_samples)[:5]}...")
                    if extra_samples:
                        inconsistencies.append(f"      Extra {len(extra_samples)} samples: {list(extra_samples)[:5]}...")
                
                # Check for potential log transformation differences
                if df.min().min() <= 0 and reference_df.min().min() > 0:
                    inconsistencies.append(f"    {key}: contains non-positive values while {reference_key} doesn't (potential log transform difference)")
                elif df.min().min() > 0 and reference_df.min().min() <= 0:
                    inconsistencies.append(f"    {key}: all positive values while {reference_key} has non-positive (potential log transform difference)")
                
                # Check data ranges for potential scaling differences
                df_range = df.max().max() - df.min().min()
                ref_range = reference_df.max().max() - reference_df.min().min()
                if abs(df_range - ref_range) / max(df_range, ref_range) > 0.1:  # >10% difference
                    inconsistencies.append(f"    {key}: data range significantly different from {reference_key} (potential scaling difference)")
            
            if inconsistencies:
                print(f"  ⚠️  WARNING: Data inconsistencies found in {platform_name}:")
                for issue in inconsistencies:
                    print(issue)
            else:
                print(f"  ✅ {platform_name}: All matrices consistent")
        
        validate_platform_consistency(platform_a_matrices, "Platform A")
        validate_platform_consistency(platform_b_matrices, "Platform B")
    
    def _get_available_methods(self, data: AnalysisData) -> List[Tuple[str, str, str, pd.DataFrame, pd.DataFrame]]:
        """Get list of available methods and their data
        Returns: list of (method_key, method_name, platform, truth_data, imputed_data)
        """
        methods = []
        
        # Always include methods 1 and 2
        if data.imp_a_m1 is not None:
            methods.append(('Method_1', data.method1_name, 'Platform_A', data.truth_a, data.imp_a_m1))
        if data.imp_a_m2 is not None:
            methods.append(('Method_2', data.method2_name, 'Platform_A', data.truth_a, data.imp_a_m2))
        if data.imp_b_m1 is not None:
            methods.append(('Method_1', data.method1_name, 'Platform_B', data.truth_b, data.imp_b_m1))
        if data.imp_b_m2 is not None:
            methods.append(('Method_2', data.method2_name, 'Platform_B', data.truth_b, data.imp_b_m2))
        
        # Include methods 3 and 4 if available
        if data.imp_a_m3 is not None and data.method3_name is not None:
            methods.append(('Method_3', data.method3_name, 'Platform_A', data.truth_a, data.imp_a_m3))
        if data.imp_a_m4 is not None and data.method4_name is not None:
            methods.append(('Method_4', data.method4_name, 'Platform_A', data.truth_a, data.imp_a_m4))
        if data.imp_b_m3 is not None and data.method3_name is not None:
            methods.append(('Method_3', data.method3_name, 'Platform_B', data.truth_b, data.imp_b_m3))
        if data.imp_b_m4 is not None and data.method4_name is not None:
            methods.append(('Method_4', data.method4_name, 'Platform_B', data.truth_b, data.imp_b_m4))
        
        return methods
    
    def compute_all_metrics(self, data: AnalysisData) -> AnalysisData:
        """Compute all primary metrics for analysis"""
        print("Computing comprehensive metrics...")
        
        metrics = {}
        
        # Feature-wise metrics (Pearson)
        print("  Computing feature-wise metrics...")
        metrics['feature_wise'] = self._compute_feature_wise_metrics(data)
        
        # Sample-wise metrics (Pearson)
        print("  Computing sample-wise metrics...")
        metrics['sample_wise'] = self._compute_sample_wise_metrics(data)
        
        # Cross-platform R² for overlapping features (Pearson)
        print("  Computing cross-platform correlations...")
        cross_platform_r2 = self._compute_cross_platform_r2(data)
        
        # Spearman correlation metrics
        print("  Computing Spearman correlation metrics...")
        spearman_metrics = {}
        spearman_metrics['feature_wise'] = self._compute_feature_wise_metrics_spearman(data)
        spearman_metrics['sample_wise'] = self._compute_sample_wise_metrics_spearman(data)
        cross_platform_rho = self._compute_cross_platform_rho(data)
        
        data.metrics = metrics
        data.cross_platform_r2 = cross_platform_r2
        data.spearman_metrics = spearman_metrics
        data.cross_platform_rho = cross_platform_rho
        
        # Save intermediate results
        self._save_metrics_to_csv(data)
        
        return data
    
    def _compute_feature_wise_metrics(self, data: AnalysisData) -> pd.DataFrame:
        """Compute feature-wise performance metrics"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        for platform, method, truth, imputed in comparisons:
            for feature in truth.index:
                truth_vals = truth.loc[feature].values
                imp_vals = imputed.loc[feature].values
                
                # Skip if all values are NaN
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                    
                truth_clean = truth_vals[mask]
                imp_clean = imp_vals[mask]
                
                # Compute metrics
                r, r_p = pearsonr(truth_clean, imp_clean)
                rmse = np.sqrt(mean_squared_error(truth_clean, imp_clean))
                mae = mean_absolute_error(truth_clean, imp_clean)
                bias = np.mean(imp_clean - truth_clean)
                
                results.append({
                    'feature': feature,
                    'platform': platform,
                    'method': method,
                    'r': r,
                    'r_pvalue': r_p,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'n_samples': np.sum(mask)
                })
        
        return pd.DataFrame(results)
    
    def _compute_sample_wise_metrics(self, data: AnalysisData) -> pd.DataFrame:
        """Compute sample-wise performance metrics"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        for platform, method, truth, imputed in comparisons:
            for sample in truth.columns:
                truth_vals = truth[sample].values
                imp_vals = imputed[sample].values
                
                # Skip if all values are NaN
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                    
                truth_clean = truth_vals[mask]
                imp_clean = imp_vals[mask]
                
                # Compute metrics
                r, r_p = pearsonr(truth_clean, imp_clean)
                rmse = np.sqrt(mean_squared_error(truth_clean, imp_clean))
                mae = mean_absolute_error(truth_clean, imp_clean)
                bias = np.mean(imp_clean - truth_clean)
                
                results.append({
                    'sample': sample,
                    'platform': platform,
                    'method': method,
                    'r': r,
                    'r_pvalue': r_p,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'n_features': np.sum(mask)
                })
        
        return pd.DataFrame(results)
    
    def _compute_cross_platform_r2(self, data: AnalysisData) -> pd.Series:
        """Compute cross-platform correlation (r) between platforms for each overlapping feature
        Note: Despite the function name, we return correlation r, not R²"""
        results = {}
        
        # Only compute for overlapping features
        overlapping_features = getattr(self, 'overlapping_features', [])
        
        if not overlapping_features:
            print("    No overlapping features for cross-platform correlation calculation")
            return pd.Series(dtype=float)
        
        print(f"    Computing cross-platform correlation for {len(overlapping_features)} overlapping features")
        
        for feature in overlapping_features:
            # Check if feature exists in both platforms
            if feature in data.truth_a.index and feature in data.truth_b.index:
                vals_a = data.truth_a.loc[feature].values
                vals_b = data.truth_b.loc[feature].values
                
                # Find common samples (non-NaN in both)
                mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
                if np.sum(mask) < 3:
                    results[feature] = np.nan
                    continue
                    
                clean_a = vals_a[mask]
                clean_b = vals_b[mask]
                
                # Compute correlation r (preserve sign)
                r, _ = pearsonr(clean_a, clean_b)
                results[feature] = r
            else:
                results[feature] = np.nan
        
        return pd.Series(results)
    
    def _compute_feature_wise_metrics_spearman(self, data: AnalysisData) -> pd.DataFrame:
        """Compute feature-wise performance metrics using Spearman correlation"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        for platform, method, truth, imputed in comparisons:
            for feature in truth.index:
                truth_vals = truth.loc[feature].values
                imp_vals = imputed.loc[feature].values
                
                # Skip if all values are NaN
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                    
                truth_clean = truth_vals[mask]
                imp_clean = imp_vals[mask]
                
                # Compute Spearman correlation
                rho, rho_p = spearmanr(truth_clean, imp_clean)
                
                results.append({
                    'feature': feature,
                    'platform': platform,
                    'method': method,
                    'rho': rho,
                    'rho_pvalue': rho_p,
                    'n_samples': np.sum(mask)
                })
        
        return pd.DataFrame(results)
    
    def _compute_sample_wise_metrics_spearman(self, data: AnalysisData) -> pd.DataFrame:
        """Compute sample-wise performance metrics using Spearman correlation"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        for platform, method, truth, imputed in comparisons:
            for sample in truth.columns:
                truth_vals = truth[sample].values
                imp_vals = imputed[sample].values
                
                # Skip if all values are NaN
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                    
                truth_clean = truth_vals[mask]
                imp_clean = imp_vals[mask]
                
                # Compute Spearman correlation
                rho, rho_p = spearmanr(truth_clean, imp_clean)
                
                results.append({
                    'sample': sample,
                    'platform': platform,
                    'method': method,
                    'rho': rho,
                    'rho_pvalue': rho_p,
                    'n_features': np.sum(mask)
                })
        
        return pd.DataFrame(results)
    
    def _compute_cross_platform_rho(self, data: AnalysisData) -> pd.Series:
        """Compute cross-platform Spearman correlation between platforms for each overlapping feature"""
        results = {}
        
        # Only compute for overlapping features
        overlapping_features = getattr(self, 'overlapping_features', [])
        
        if not overlapping_features:
            print("    No overlapping features for cross-platform Spearman correlation calculation")
            return pd.Series(dtype=float)
        
        print(f"    Computing cross-platform Spearman correlation for {len(overlapping_features)} overlapping features")
        
        for feature in overlapping_features:
            # Check if feature exists in both platforms
            if feature in data.truth_a.index and feature in data.truth_b.index:
                vals_a = data.truth_a.loc[feature].values
                vals_b = data.truth_b.loc[feature].values
                
                # Find common samples (non-NaN in both)
                mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
                if np.sum(mask) < 3:
                    results[feature] = np.nan
                    continue
                    
                clean_a = vals_a[mask]
                clean_b = vals_b[mask]
                
                # Compute Spearman correlation
                rho, _ = spearmanr(clean_a, clean_b)
                results[feature] = rho
            else:
                results[feature] = np.nan
        
        return pd.Series(results)

    def _save_metrics_to_csv(self, data: AnalysisData):
        """Save computed metrics to CSV files"""
        data_dir = self.output_dir / "data"
        
        try:
            # Save Pearson metrics
            if data.metrics is None:
                raise ValueError("Pearson metrics not computed - run compute_all_metrics() first")
            
            data.metrics['feature_wise'].to_csv(data_dir / "feature_wise_metrics.csv", index=False)
            data.metrics['sample_wise'].to_csv(data_dir / "sample_wise_metrics.csv", index=False)
            
            # Save cross-platform correlation (note: despite variable name, this is r not R²)
            if data.cross_platform_r2 is not None:
                data.cross_platform_r2.to_csv(data_dir / "cross_platform_correlation.csv", header=['correlation'], index_label='feature')
            else:
                print("    Warning: Cross-platform Pearson correlation not available")
            
            # Save Spearman metrics if available
            if data.spearman_metrics is not None:
                if 'feature_wise' not in data.spearman_metrics or 'sample_wise' not in data.spearman_metrics:
                    raise ValueError("Incomplete Spearman metrics - missing feature_wise or sample_wise data")
                
                data.spearman_metrics['feature_wise'].to_csv(data_dir / "feature_wise_metrics_spearman.csv", index=False)
                data.spearman_metrics['sample_wise'].to_csv(data_dir / "sample_wise_metrics_spearman.csv", index=False)
            else:
                print("    Warning: Spearman metrics not available - they may not have been computed")
            
            if data.cross_platform_rho is not None:
                data.cross_platform_rho.to_csv(data_dir / "cross_platform_correlation_spearman.csv", header=['spearman_correlation'], index_label='feature')
            else:
                print("    Warning: Cross-platform Spearman correlation not available")
            
            print(f"  Metrics saved to {data_dir}")
            
        except Exception as e:
            print(f"    Error saving metrics: {e}")
            raise  # Re-raise to expose the error instead of hiding it
    
    def load_phenotype_data(self, phenotype_file: str, sample_id_col: str = None) -> pd.DataFrame:
        """Load and validate phenotype data"""
        print(f"Loading phenotype data from {phenotype_file}...")
        
        try:
            # Load phenotype data
            pheno_df = pd.read_csv(phenotype_file)
            
            # If sample_id_col is specified, use it as index
            if sample_id_col and sample_id_col in pheno_df.columns:
                pheno_df = pheno_df.set_index(sample_id_col)
            elif 'sample_id' in pheno_df.columns:
                pheno_df = pheno_df.set_index('sample_id')
            elif 'Sample_ID' in pheno_df.columns:
                pheno_df = pheno_df.set_index('Sample_ID')
            else:
                # Assume first column is sample ID
                pheno_df = pheno_df.set_index(pheno_df.columns[0])
            
            print(f"  Loaded phenotype data with shape: {pheno_df.shape}")
            print(f"  Available phenotypes: {list(pheno_df.columns)}")
            
            # Basic validation
            if pheno_df.shape[0] == 0:
                raise ValueError("Phenotype data has no samples")
            
            if pheno_df.shape[1] == 0:
                raise ValueError("Phenotype data has no phenotype columns")
            
            # Check for missing values
            missing_summary = pheno_df.isnull().sum()
            if missing_summary.sum() > 0:
                print("  Warning: Missing values detected in phenotype data:")
                for col in missing_summary[missing_summary > 0].index:
                    print(f"    - {col}: {missing_summary[col]} missing values")
            
            return pheno_df
            
        except FileNotFoundError:
            print(f"  Error: Phenotype file not found: {phenotype_file}")
            raise
        except Exception as e:
            print(f"  Error loading phenotype data: {str(e)}")
            raise
    
    def calculate_binary_associations(self, data: AnalysisData, phenotype_col: str) -> Dict[str, pd.DataFrame]:
        """Calculate associations between features and binary phenotype using logistic regression"""
        from sklearn.linear_model import LogisticRegression
        from statsmodels.stats.multitest import multipletests
        
        print(f"  Calculating binary associations for: {phenotype_col}")
        
        results = {}
        
        # Get phenotype data
        pheno_data = data.phenotype_data[phenotype_col].dropna()
        
        # Get unique values to check if binary
        unique_vals = pheno_data.unique()
        if len(unique_vals) != 2:
            print(f"    Warning: {phenotype_col} has {len(unique_vals)} unique values, expected 2")
            return results
        
        # Process each dataset
        datasets = [
            ('Truth_A', data.truth_a, data.platform_a_name),
            ('Truth_B', data.truth_b, data.platform_b_name),
            ('Method1_A', data.imp_a_m1, f"{data.method1_name}_{data.platform_a_name}"),
            ('Method1_B', data.imp_b_m1, f"{data.method1_name}_{data.platform_b_name}"),
            ('Method2_A', data.imp_a_m2, f"{data.method2_name}_{data.platform_a_name}"),
            ('Method2_B', data.imp_b_m2, f"{data.method2_name}_{data.platform_b_name}")
        ]
        
        if data.imp_a_m3 is not None:
            datasets.extend([
                ('Method3_A', data.imp_a_m3, f"{data.method3_name}_{data.platform_a_name}"),
                ('Method3_B', data.imp_b_m3, f"{data.method3_name}_{data.platform_b_name}")
            ])
        
        if data.imp_a_m4 is not None:
            datasets.extend([
                ('Method4_A', data.imp_a_m4, f"{data.method4_name}_{data.platform_a_name}"),
                ('Method4_B', data.imp_b_m4, f"{data.method4_name}_{data.platform_b_name}")
            ])
        
        for dataset_key, dataset, dataset_name in datasets:
            if dataset is None:
                continue
                
            # Find common samples
            common_samples = list(set(dataset.columns) & set(pheno_data.index))
            if len(common_samples) < 10:
                print(f"    Warning: Only {len(common_samples)} samples for {dataset_name}")
                continue
            
            # Align data
            X = dataset[common_samples].T
            y = pheno_data.loc[common_samples]
            
            # Calculate associations for each feature
            associations = []
            
            for feature in X.columns:
                try:
                    # Prepare feature data
                    feature_data = X[feature].values.reshape(-1, 1)
                    
                    # Standardize feature
                    feature_data = (feature_data - np.mean(feature_data)) / (np.std(feature_data) + 1e-8)
                    
                    # Fit logistic regression
                    lr = LogisticRegression(solver='liblinear', random_state=42)
                    lr.fit(feature_data, y)
                    
                    # Calculate odds ratio and CI
                    coef = lr.coef_[0, 0]
                    odds_ratio = np.exp(coef)
                    
                    # Bootstrap for confidence intervals
                    n_bootstrap = 100
                    bootstrap_ors = []
                    
                    for _ in range(n_bootstrap):
                        idx = np.random.choice(len(y), len(y), replace=True)
                        try:
                            lr_boot = LogisticRegression(solver='liblinear', random_state=42)
                            lr_boot.fit(feature_data[idx], y.iloc[idx])
                            bootstrap_ors.append(np.exp(lr_boot.coef_[0, 0]))
                        except:
                            continue
                    
                    if len(bootstrap_ors) > 10:
                        ci_lower = np.percentile(bootstrap_ors, 2.5)
                        ci_upper = np.percentile(bootstrap_ors, 97.5)
                    else:
                        ci_lower = odds_ratio * 0.5
                        ci_upper = odds_ratio * 2.0
                    
                    # Calculate p-value using permutation test
                    n_perm = 100
                    null_coefs = []
                    
                    for _ in range(n_perm):
                        y_perm = np.random.permutation(y)
                        try:
                            lr_perm = LogisticRegression(solver='liblinear', random_state=42)
                            lr_perm.fit(feature_data, y_perm)
                            null_coefs.append(abs(lr_perm.coef_[0, 0]))
                        except:
                            continue
                    
                    if len(null_coefs) > 10:
                        p_value = np.mean([abs(coef) <= nc for nc in null_coefs])
                    else:
                        p_value = 0.5
                    
                    associations.append({
                        'feature': feature,
                        'odds_ratio': odds_ratio,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'coef': coef,
                        'p_value': p_value,
                        'n_samples': len(y)
                    })
                    
                except Exception as e:
                    # Skip problematic features
                    continue
            
            if associations:
                assoc_df = pd.DataFrame(associations)
                
                # FDR correction
                if len(assoc_df) > 1:
                    _, fdr_pvals, _, _ = multipletests(assoc_df['p_value'], method='fdr_bh')
                    assoc_df['fdr_pvalue'] = fdr_pvals
                else:
                    assoc_df['fdr_pvalue'] = assoc_df['p_value']
                
                # Sort by significance
                assoc_df = assoc_df.sort_values('p_value')
                
                results[dataset_name] = assoc_df
        
        return results
    
    def calculate_continuous_associations(self, data: AnalysisData, phenotype_col: str) -> Dict[str, pd.DataFrame]:
        """Calculate associations between features and continuous phenotype using linear regression"""
        from sklearn.linear_model import LinearRegression
        from statsmodels.stats.multitest import multipletests
        
        print(f"  Calculating continuous associations for: {phenotype_col}")
        
        results = {}
        
        # Get phenotype data
        pheno_data = data.phenotype_data[phenotype_col].dropna()
        
        # Process each dataset
        datasets = [
            ('Truth_A', data.truth_a, data.platform_a_name),
            ('Truth_B', data.truth_b, data.platform_b_name),
            ('Method1_A', data.imp_a_m1, f"{data.method1_name}_{data.platform_a_name}"),
            ('Method1_B', data.imp_b_m1, f"{data.method1_name}_{data.platform_b_name}"),
            ('Method2_A', data.imp_a_m2, f"{data.method2_name}_{data.platform_a_name}"),
            ('Method2_B', data.imp_b_m2, f"{data.method2_name}_{data.platform_b_name}")
        ]
        
        if data.imp_a_m3 is not None:
            datasets.extend([
                ('Method3_A', data.imp_a_m3, f"{data.method3_name}_{data.platform_a_name}"),
                ('Method3_B', data.imp_b_m3, f"{data.method3_name}_{data.platform_b_name}")
            ])
        
        if data.imp_a_m4 is not None:
            datasets.extend([
                ('Method4_A', data.imp_a_m4, f"{data.method4_name}_{data.platform_a_name}"),
                ('Method4_B', data.imp_b_m4, f"{data.method4_name}_{data.platform_b_name}")
            ])
        
        for dataset_key, dataset, dataset_name in datasets:
            if dataset is None:
                continue
                
            # Find common samples
            common_samples = list(set(dataset.columns) & set(pheno_data.index))
            if len(common_samples) < 10:
                print(f"    Warning: Only {len(common_samples)} samples for {dataset_name}")
                continue
            
            # Align data
            X = dataset[common_samples].T
            y = pheno_data.loc[common_samples]
            
            # Calculate associations for each feature
            associations = []
            
            for feature in X.columns:
                try:
                    # Prepare feature data
                    feature_data = X[feature].values.reshape(-1, 1)
                    
                    # Standardize feature
                    feature_data = (feature_data - np.mean(feature_data)) / (np.std(feature_data) + 1e-8)
                    
                    # Fit linear regression
                    lr = LinearRegression()
                    lr.fit(feature_data, y)
                    
                    # Get coefficient and R²
                    beta = lr.coef_[0]
                    r2 = lr.score(feature_data, y)
                    
                    # Bootstrap for confidence intervals
                    n_bootstrap = 100
                    bootstrap_betas = []
                    
                    for _ in range(n_bootstrap):
                        idx = np.random.choice(len(y), len(y), replace=True)
                        lr_boot = LinearRegression()
                        lr_boot.fit(feature_data[idx], y.iloc[idx])
                        bootstrap_betas.append(lr_boot.coef_[0])
                    
                    ci_lower = np.percentile(bootstrap_betas, 2.5)
                    ci_upper = np.percentile(bootstrap_betas, 97.5)
                    
                    # Calculate p-value using permutation test
                    n_perm = 100
                    null_betas = []
                    
                    for _ in range(n_perm):
                        y_perm = np.random.permutation(y)
                        lr_perm = LinearRegression()
                        lr_perm.fit(feature_data, y_perm)
                        null_betas.append(abs(lr_perm.coef_[0]))
                    
                    p_value = np.mean([abs(beta) <= nb for nb in null_betas])
                    
                    associations.append({
                        'feature': feature,
                        'beta': beta,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'r2': r2,
                        'p_value': p_value,
                        'n_samples': len(y)
                    })
                    
                except Exception as e:
                    # Skip problematic features
                    continue
            
            if associations:
                assoc_df = pd.DataFrame(associations)
                
                # FDR correction
                if len(assoc_df) > 1:
                    _, fdr_pvals, _, _ = multipletests(assoc_df['p_value'], method='fdr_bh')
                    assoc_df['fdr_pvalue'] = fdr_pvals
                else:
                    assoc_df['fdr_pvalue'] = assoc_df['p_value']
                
                # Sort by significance
                assoc_df = assoc_df.sort_values('p_value')
                
                results[dataset_name] = assoc_df
        
        return results
    
    def generate_figure_26_comprehensive_method_comparison(self, data: AnalysisData):
        """Figure 26: Comprehensive comparison of all available methods (mean and median correlations)"""
        print("Generating Figure 26: Comprehensive method comparison...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Check how many methods are available
        available_methods = self._get_available_methods(data)
        unique_methods = list(set([(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods]))
        
        if len(unique_methods) < 2:
            print("    ⚠️  Less than 3 methods available - skipping comprehensive comparison")
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison',
                f'Only {len(unique_methods)} methods available. Need at least 3 for comprehensive comparison.'
            )
        
        # Get feature-wise and sample-wise metrics
        feat_metrics = data.metrics['feature_wise']
        sample_metrics = data.metrics['sample_wise']
        
        if feat_metrics.empty and sample_metrics.empty:
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison',
                'No metrics available for comparison'
            )
        
        # Create figure with simplified 2x1 layout (feature-wise and sample-wise)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Comprehensive Method Comparison: All Available Methods', fontsize=16, fontweight='bold')
        
        # Colors for up to 4 methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # 1. Feature-wise performance comparison (Mean and Median combined)
        if not feat_metrics.empty:
            feat_summary = feat_metrics.groupby(['method', 'platform'])['r'].agg(['mean', 'median', 'std']).reset_index()
            
            platforms = feat_summary['platform'].unique()
            methods = feat_summary['method'].unique()
            
            # Create combined data for mean and median
            x_labels = []
            mean_data = []
            median_data = []
            
            for platform in platforms:
                platform_label = platform_name_map.get(platform, platform)
                for method in methods:
                    method_data = feat_summary[(feat_summary['method'] == method) & (feat_summary['platform'] == platform)]
                    if not method_data.empty:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels.append(f'{method_name}\n({platform_label})')
                        mean_data.append(method_data['mean'].iloc[0])
                        median_data.append(method_data['median'].iloc[0])
                    else:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels.append(f'{method_name}\n({platform_label})')
                        mean_data.append(0)
                        median_data.append(0)
            
            x = np.arange(len(x_labels))
            width = 0.35
            
            ax1.bar(x - width/2, mean_data, width, label='Mean', 
                   color=colors[0], alpha=0.8)
            ax1.bar(x + width/2, median_data, width, label='Median', 
                   color=colors[1], alpha=0.8)
            
            ax1.set_xlabel('Method (Platform)')
            ax1.set_ylabel('Feature-wise Correlation (r)')
            ax1.set_title('Feature-wise Performance', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, 'No feature-wise metrics available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Feature-wise Performance')
        
        # 2. Sample-wise performance comparison (Mean and Median combined)
        if not sample_metrics.empty:
            sample_summary = sample_metrics.groupby(['method', 'platform'])['r'].agg(['mean', 'median', 'std']).reset_index()
            
            platforms_sample = sample_summary['platform'].unique()
            methods_sample = sample_summary['method'].unique()
            
            # Create combined data for mean and median
            x_labels_sample = []
            mean_data_sample = []
            median_data_sample = []
            
            for platform in platforms_sample:
                platform_label = platform_name_map.get(platform, platform)
                for method in methods_sample:
                    method_data = sample_summary[(sample_summary['method'] == method) & (sample_summary['platform'] == platform)]
                    if not method_data.empty:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels_sample.append(f'{method_name}\n({platform_label})')
                        mean_data_sample.append(method_data['mean'].iloc[0])
                        median_data_sample.append(method_data['median'].iloc[0])
                    else:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels_sample.append(f'{method_name}\n({platform_label})')
                        mean_data_sample.append(0)
                        median_data_sample.append(0)
            
            x_sample = np.arange(len(x_labels_sample))
            width = 0.35
            
            ax2.bar(x_sample - width/2, mean_data_sample, width, label='Mean', 
                   color=colors[0], alpha=0.8)
            ax2.bar(x_sample + width/2, median_data_sample, width, label='Median', 
                   color=colors[1], alpha=0.8)
            
            ax2.set_xlabel('Method (Platform)')
            ax2.set_ylabel('Sample-wise Correlation (r)')
            ax2.set_title('Sample-wise Performance', fontweight='bold')
            ax2.set_xticks(x_sample)
            ax2.set_xticklabels(x_labels_sample, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No sample-wise metrics available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Sample-wise Performance')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_27_comprehensive_method_comparison_spearman(self, data: AnalysisData):
        """Figure 27: Comprehensive comparison of all available methods using Spearman correlation (mean and median correlations)"""
        print("Generating Figure 27: Comprehensive method comparison (Spearman correlation)...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Check if Spearman metrics are available
        if data.spearman_metrics is None:
            print("    ⚠️  Spearman metrics not available - skipping Spearman comparison")
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison (Spearman)',
                'Spearman metrics not computed. Run compute_all_metrics() first.'
            )
        
        # Check how many methods are available
        available_methods = self._get_available_methods(data)
        unique_methods = list(set([(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods]))
        
        if len(unique_methods) < 2:
            print("    ⚠️  Less than 3 methods available - skipping comprehensive comparison")
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison (Spearman)',
                f'Only {len(unique_methods)} methods available. Need at least 3 for comprehensive comparison.'
            )
        
        # Get feature-wise and sample-wise Spearman metrics
        feat_metrics = data.spearman_metrics['feature_wise']
        sample_metrics = data.spearman_metrics['sample_wise']
        
        if feat_metrics.empty and sample_metrics.empty:
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison (Spearman)',
                'No Spearman metrics available for comparison'
            )
        
        # Create figure with simplified 2x1 layout (feature-wise and sample-wise)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Comprehensive Method Comparison: All Available Methods (Spearman Correlation)', fontsize=16, fontweight='bold')
        
        # Colors for up to 4 methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # 1. Feature-wise performance comparison (Mean and Median combined)
        if not feat_metrics.empty:
            feat_summary = feat_metrics.groupby(['method', 'platform'])['rho'].agg(['mean', 'median', 'std']).reset_index()
            
            platforms = feat_summary['platform'].unique()
            methods = feat_summary['method'].unique()
            
            # Create combined data for mean and median
            x_labels = []
            mean_data = []
            median_data = []
            
            for platform in platforms:
                platform_label = platform_name_map.get(platform, platform)
                for method in methods:
                    method_data = feat_summary[(feat_summary['method'] == method) & (feat_summary['platform'] == platform)]
                    if not method_data.empty:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels.append(f'{method_name}\n({platform_label})')
                        mean_data.append(method_data['mean'].iloc[0])
                        median_data.append(method_data['median'].iloc[0])
                    else:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels.append(f'{method_name}\n({platform_label})')
                        mean_data.append(0)
                        median_data.append(0)
            
            x = np.arange(len(x_labels))
            width = 0.35
            
            ax1.bar(x - width/2, mean_data, width, label='Mean', 
                   color=colors[0], alpha=0.8)
            ax1.bar(x + width/2, median_data, width, label='Median', 
                   color=colors[1], alpha=0.8)
            
            ax1.set_xlabel('Method (Platform)')
            ax1.set_ylabel('Feature-wise Correlation (ρ)')
            ax1.set_title('Feature-wise Performance', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, 'No feature-wise Spearman metrics available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Feature-wise Performance')
        
        # 2. Sample-wise performance comparison (Mean and Median combined)
        if not sample_metrics.empty:
            sample_summary = sample_metrics.groupby(['method', 'platform'])['rho'].agg(['mean', 'median', 'std']).reset_index()
            
            platforms_sample = sample_summary['platform'].unique()
            methods_sample = sample_summary['method'].unique()
            
            # Create combined data for mean and median
            x_labels_sample = []
            mean_data_sample = []
            median_data_sample = []
            
            for platform in platforms_sample:
                platform_label = platform_name_map.get(platform, platform)
                for method in methods_sample:
                    method_data = sample_summary[(sample_summary['method'] == method) & (sample_summary['platform'] == platform)]
                    if not method_data.empty:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels_sample.append(f'{method_name}\n({platform_label})')
                        mean_data_sample.append(method_data['mean'].iloc[0])
                        median_data_sample.append(method_data['median'].iloc[0])
                    else:
                        method_name = next((name for key, name in unique_methods if key == method), method)
                        x_labels_sample.append(f'{method_name}\n({platform_label})')
                        mean_data_sample.append(0)
                        median_data_sample.append(0)
            
            x_sample = np.arange(len(x_labels_sample))
            width = 0.35
            
            ax2.bar(x_sample - width/2, mean_data_sample, width, label='Mean', 
                   color=colors[0], alpha=0.8)
            ax2.bar(x_sample + width/2, median_data_sample, width, label='Median', 
                   color=colors[1], alpha=0.8)
            
            ax2.set_xlabel('Method (Platform)')
            ax2.set_ylabel('Sample-wise Correlation (ρ)')
            ax2.set_title('Sample-wise Performance', fontweight='bold')
            ax2.set_xticks(x_sample)
            ax2.set_xticklabels(x_labels_sample, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No sample-wise Spearman metrics available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Sample-wise Performance')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_28_phenotype_forest_plots_binary(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 28: Forest plots for binary phenotype associations (both platforms)"""
        print("  Generating Figure 28: Binary phenotype forest plots...")
        
        if not association_results:
            return self._create_insufficient_data_figure(
                'Binary Phenotype Forest Plots',
                'No binary phenotype associations calculated'
            )
        
        # Get unique phenotypes from all results
        all_phenotypes = set()
        for pheno_results in association_results.values():
            all_phenotypes.update(pheno_results.keys())
        
        if not all_phenotypes:
            return self._create_insufficient_data_figure(
                'Binary Phenotype Forest Plots', 
                'No phenotype data available'
            )
        
        # Create figure with subplots for each phenotype and platform
        n_phenotypes = len(all_phenotypes)
        fig, axes = plt.subplots(n_phenotypes, 2, figsize=(10, 5 * n_phenotypes))
        
        if n_phenotypes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Binary Phenotype Associations: Forest Plots', fontsize=16, fontweight='bold')
        
        # Colors and markers for different methods
        method_styles = {
            data.platform_a_name: {'color': NATURE_COLORS['primary'], 'marker': 'o', 'label': 'Truth'},
            data.platform_b_name: {'color': NATURE_COLORS['secondary'], 'marker': 'o', 'label': 'Truth'},
            f"{data.method1_name}_{data.platform_a_name}": {'color': NATURE_COLORS['accent'], 'marker': 's', 'label': data.method1_name},
            f"{data.method1_name}_{data.platform_b_name}": {'color': NATURE_COLORS['accent'], 'marker': 's', 'label': data.method1_name},
            f"{data.method2_name}_{data.platform_a_name}": {'color': NATURE_COLORS['neutral'], 'marker': '^', 'label': data.method2_name},
            f"{data.method2_name}_{data.platform_b_name}": {'color': NATURE_COLORS['neutral'], 'marker': '^', 'label': data.method2_name}
        }
        
        if data.method3_name:
            method_styles[f"{data.method3_name}_{data.platform_a_name}"] = {'color': NATURE_COLORS['highlight'], 'marker': 'D', 'label': data.method3_name}
            method_styles[f"{data.method3_name}_{data.platform_b_name}"] = {'color': NATURE_COLORS['highlight'], 'marker': 'D', 'label': data.method3_name}
        
        if data.method4_name:
            method_styles[f"{data.method4_name}_{data.platform_a_name}"] = {'color': NATURE_COLORS['alternative_1'], 'marker': 'v', 'label': data.method4_name}
            method_styles[f"{data.method4_name}_{data.platform_b_name}"] = {'color': NATURE_COLORS['alternative_1'], 'marker': 'v', 'label': data.method4_name}
        
        for pheno_idx, phenotype in enumerate(sorted(all_phenotypes)):
            # Plot for each platform
            for platform_idx, platform_name in enumerate([data.platform_a_name, data.platform_b_name]):
                ax = axes[pheno_idx, platform_idx]
                
                # Collect data for this phenotype and platform
                plot_data = []
                
                for pheno_key, pheno_results in association_results.items():
                    if phenotype in pheno_results:
                        result_df = pheno_results[phenotype]
                        
                        # Filter for this platform
                        platform_methods = [key for key in result_df.keys() if platform_name in key]
                        
                        for method_key in platform_methods:
                            if method_key in result_df:
                                # Get top 10 significant features
                                top_features = result_df[method_key].nsmallest(10, 'fdr_pvalue')
                                
                                for idx, (feature, row) in enumerate(top_features.iterrows()):
                                    plot_data.append({
                                        'feature': feature,
                                        'odds_ratio': row['odds_ratio'],
                                        'ci_lower': row['ci_lower'],
                                        'ci_upper': row['ci_upper'],
                                        'method': method_key,
                                        'y_pos': idx
                                    })
                
                if not plot_data:
                    ax.text(0.5, 0.5, f'No significant associations\nfor {phenotype}\nin {platform_name}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{phenotype} - {platform_name}')
                    continue
                
                # Create forest plot
                plot_df = pd.DataFrame(plot_data)
                
                # Plot each method
                for method in plot_df['method'].unique():
                    if method in method_styles:
                        method_data = plot_df[plot_df['method'] == method]
                        style = method_styles[method]
                        
                        # Plot confidence intervals
                        for _, row in method_data.iterrows():
                            ax.plot([row['ci_lower'], row['ci_upper']], 
                                   [row['y_pos'], row['y_pos']], 
                                   color=style['color'], alpha=0.5, linewidth=2)
                        
                        # Plot odds ratios
                        ax.scatter(method_data['odds_ratio'], method_data['y_pos'],
                                 color=style['color'], marker=style['marker'], 
                                 s=100, label=style['label'], zorder=5)
                
                # Add vertical line at OR=1
                ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
                
                # Set labels
                ax.set_xlabel('Odds Ratio (95% CI)')
                ax.set_ylabel('Features')
                ax.set_title(f'{phenotype} - {platform_name}')
                
                # Set y-tick labels to feature names
                unique_features = plot_df['feature'].unique()[:10]
                ax.set_yticks(range(len(unique_features)))
                ax.set_yticklabels(unique_features)
                
                # Set x-axis to log scale
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                
                # Add legend
                if pheno_idx == 0 and platform_idx == 0:
                    ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_29_phenotype_forest_plots_continuous(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 29: Forest plots for continuous phenotype associations (both platforms)"""
        print("  Generating Figure 29: Continuous phenotype forest plots...")
        
        if not association_results:
            return self._create_insufficient_data_figure(
                'Continuous Phenotype Forest Plots',
                'No continuous phenotype associations calculated'
            )
        
        # Get unique phenotypes from all results
        all_phenotypes = set()
        for pheno_results in association_results.values():
            all_phenotypes.update(pheno_results.keys())
        
        if not all_phenotypes:
            return self._create_insufficient_data_figure(
                'Continuous Phenotype Forest Plots', 
                'No phenotype data available'
            )
        
        # Create figure with subplots for each phenotype and platform
        n_phenotypes = len(all_phenotypes)
        fig, axes = plt.subplots(n_phenotypes, 2, figsize=(10, 5 * n_phenotypes))
        
        if n_phenotypes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Continuous Phenotype Associations: Forest Plots', fontsize=16, fontweight='bold')
        
        # Colors and markers for different methods
        method_styles = {
            data.platform_a_name: {'color': NATURE_COLORS['primary'], 'marker': 'o', 'label': 'Truth'},
            data.platform_b_name: {'color': NATURE_COLORS['secondary'], 'marker': 'o', 'label': 'Truth'},
            f"{data.method1_name}_{data.platform_a_name}": {'color': NATURE_COLORS['accent'], 'marker': 's', 'label': data.method1_name},
            f"{data.method1_name}_{data.platform_b_name}": {'color': NATURE_COLORS['accent'], 'marker': 's', 'label': data.method1_name},
            f"{data.method2_name}_{data.platform_a_name}": {'color': NATURE_COLORS['neutral'], 'marker': '^', 'label': data.method2_name},
            f"{data.method2_name}_{data.platform_b_name}": {'color': NATURE_COLORS['neutral'], 'marker': '^', 'label': data.method2_name}
        }
        
        if data.method3_name:
            method_styles[f"{data.method3_name}_{data.platform_a_name}"] = {'color': NATURE_COLORS['highlight'], 'marker': 'D', 'label': data.method3_name}
            method_styles[f"{data.method3_name}_{data.platform_b_name}"] = {'color': NATURE_COLORS['highlight'], 'marker': 'D', 'label': data.method3_name}
        
        if data.method4_name:
            method_styles[f"{data.method4_name}_{data.platform_a_name}"] = {'color': NATURE_COLORS['alternative_1'], 'marker': 'v', 'label': data.method4_name}
            method_styles[f"{data.method4_name}_{data.platform_b_name}"] = {'color': NATURE_COLORS['alternative_1'], 'marker': 'v', 'label': data.method4_name}
        
        for pheno_idx, phenotype in enumerate(sorted(all_phenotypes)):
            # Plot for each platform
            for platform_idx, platform_name in enumerate([data.platform_a_name, data.platform_b_name]):
                ax = axes[pheno_idx, platform_idx]
                
                # Collect data for this phenotype and platform
                plot_data = []
                
                for pheno_key, pheno_results in association_results.items():
                    if phenotype in pheno_results:
                        result_df = pheno_results[phenotype]
                        
                        # Filter for this platform
                        platform_methods = [key for key in result_df.keys() if platform_name in key]
                        
                        for method_key in platform_methods:
                            if method_key in result_df:
                                # Get top 10 significant features
                                top_features = result_df[method_key].nsmallest(10, 'fdr_pvalue')
                                
                                for idx, (feature, row) in enumerate(top_features.iterrows()):
                                    plot_data.append({
                                        'feature': feature,
                                        'beta': row['beta'],
                                        'ci_lower': row['ci_lower'],
                                        'ci_upper': row['ci_upper'],
                                        'r2': row['r2'],
                                        'method': method_key,
                                        'y_pos': idx
                                    })
                
                if not plot_data:
                    ax.text(0.5, 0.5, f'No significant associations\nfor {phenotype}\nin {platform_name}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{phenotype} - {platform_name}')
                    continue
                
                # Create forest plot
                plot_df = pd.DataFrame(plot_data)
                
                # Plot each method
                for method in plot_df['method'].unique():
                    if method in method_styles:
                        method_data = plot_df[plot_df['method'] == method]
                        style = method_styles[method]
                        
                        # Plot confidence intervals
                        for _, row in method_data.iterrows():
                            ax.plot([row['ci_lower'], row['ci_upper']], 
                                   [row['y_pos'], row['y_pos']], 
                                   color=style['color'], alpha=0.5, linewidth=2)
                        
                        # Plot beta coefficients
                        ax.scatter(method_data['beta'], method_data['y_pos'],
                                 color=style['color'], marker=style['marker'], 
                                 s=100, label=style['label'], zorder=5)
                        
                        # Add R² values as text
                        for _, row in method_data.iterrows():
                            ax.text(row['ci_upper'] + 0.05, row['y_pos'], 
                                   f"R²={row['r2']:.3f}", fontsize=8, 
                                   va='center', color=style['color'])
                
                # Add vertical line at beta=0
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                # Set labels
                ax.set_xlabel('Beta Coefficient (95% CI)')
                ax.set_ylabel('Features')
                ax.set_title(f'{phenotype} - {platform_name}')
                
                # Set y-tick labels to feature names
                unique_features = plot_df['feature'].unique()[:10]
                ax.set_yticks(range(len(unique_features)))
                ax.set_yticklabels(unique_features)
                
                # Set reasonable x limits
                if not plot_df.empty:
                    x_margin = (plot_df['ci_upper'].max() - plot_df['ci_lower'].min()) * 0.1
                    ax.set_xlim(plot_df['ci_lower'].min() - x_margin, 
                               plot_df['ci_upper'].max() + x_margin)
                
                ax.grid(True, alpha=0.3)
                
                # Add legend
                if pheno_idx == 0 and platform_idx == 0:
                    ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, name: str, **kwargs):
        """Save figure with metadata and proper formatting"""
        fig_path = self.output_dir / "figures" / f"{name}.pdf"
        png_path = self.output_dir / "figures" / f"{name}.png"
        
        # Add metadata
        metadata = {
            'Title': name,
            'Creator': 'Cross-Platform Proteomics Analysis',
            'Subject': 'Imputation Method Comparison',
            'Keywords': 'proteomics, imputation, cross-platform',
            'Git_Hash': self.git_hash,
            'Timestamp': self.timestamp
        }

        # Extract PNG DPI if provided, avoid passing duplicate 'dpi'
        png_dpi = kwargs.pop('dpi', 300)

        # Save PDF (vector format for publication)
        fig.savefig(fig_path, format='pdf', metadata=metadata, **kwargs)

        # Save PNG (for quick viewing) at requested/ default DPI
        fig.savefig(png_path, format='png', dpi=png_dpi, **kwargs)
        
        print(f"  Figure saved: {fig_path}")
    
    def generate_figure_1_feature_r_scatter(self, data: AnalysisData):
        """Figure 1: Feature-wise r scatter with marginals and mean feature value coloring"""
        print("Generating Figure 1: Feature-wise r scatter...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Get feature-wise correlations for both platforms
        feat_metrics = data.metrics['feature_wise']
        
        # Check if we have any feature metrics
        if feat_metrics.empty:
            print("    No feature metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No feature metrics available\n(No common features between platforms)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Feature-wise Performance Scatter Plot')
            return fig
        
        # Platform A comparison
        platform_a_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        platform_b_data = feat_metrics[feat_metrics['platform'] == 'Platform_B']
        
        if len(platform_a_data) == 0 and len(platform_b_data) == 0:
            print("    No data for either platform - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No feature data available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Feature-wise Performance Scatter Plot')
            return fig
        
        # Get aligned method data for each platform with mean feature values
        def get_aligned_data_with_means(platform_data, truth_data):
            if len(platform_data) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Create separate Series for each method, indexed by feature
            m1_series = platform_data[platform_data['method'] == 'Method_1'].set_index('feature')['r']
            m2_series = platform_data[platform_data['method'] == 'Method_2'].set_index('feature')['r']
            
            # Combine and align by feature index, drop features not present in both methods
            combined = pd.concat([m1_series, m2_series], axis=1, keys=['m1', 'm2']).dropna()
            
            if combined.empty:
                return np.array([]), np.array([]), np.array([])
            
            # Calculate mean feature values (across all samples) for color coding
            feature_means = []
            for feature in combined.index:
                if feature in truth_data.index:
                    mean_val = np.nanmean(truth_data.loc[feature].values)
                    feature_means.append(mean_val)
                else:
                    feature_means.append(np.nan)
            
            return combined['m1'].values, combined['m2'].values, np.array(feature_means)
        
        m1_a, m2_a, means_a = get_aligned_data_with_means(platform_a_data, data.truth_a)
        m1_b, m2_b, means_b = get_aligned_data_with_means(platform_b_data, data.truth_b)
        
        # Create figure with subplots for both platforms (adjusted for two separate colorbars)
        fig = plt.figure(figsize=(17, 5))
        
        for i, (platform_key, m1_r, m2_r, feature_means) in enumerate([('Platform_A', m1_a, m2_a, means_a), 
                                                                       ('Platform_B', m1_b, m2_b, means_b)]):
            # Create subplot with marginals (adjusted spacing for colorbar)
            gs = GridSpec(3, 3, figure=fig, 
                         left=0.08 + i*0.40, right=0.32 + i*0.40,
                         bottom=0.1, top=0.9,
                         hspace=0.05, wspace=0.05)
            
            ax_main = fig.add_subplot(gs[1:, :-1])
            ax_top = fig.add_subplot(gs[0, :-1])
            ax_right = fig.add_subplot(gs[1:, -1])
            
            if len(m1_r) > 0 and len(m2_r) > 0 and len(m1_r) == len(m2_r):
                # Main scatter plot with color coding by mean feature values
                valid_means = ~np.isnan(feature_means)
                
                if np.any(valid_means) and np.sum(valid_means) > 1:
                    # Use consistent scaling for both platforms (original values)
                    means_for_color = feature_means[valid_means]
                    colorbar_label = 'Mean Feature Value'
                    
                    scatter = ax_main.scatter(m2_r[valid_means], m1_r[valid_means], 
                                            c=means_for_color, alpha=0.7, s=6, 
                                            cmap='viridis', edgecolors='white', linewidth=0.3)
                    
                    # Add separate colorbar for each platform (positioned to the right of each plot)
                    cbar_left = 0.34 + i*0.40  # Position relative to each platform's plot
                    cbar_ax = fig.add_axes([cbar_left, 0.2, 0.015, 0.6])  # [left, bottom, width, height]
                    cbar = plt.colorbar(scatter, cax=cbar_ax)
                    cbar.set_label(colorbar_label, rotation=270, labelpad=15)
                    
                    # Plot points with NaN means in gray
                    if np.any(~valid_means):
                        ax_main.scatter(m2_r[~valid_means], m1_r[~valid_means], 
                                      alpha=0.5, s=8, color='gray', 
                                      edgecolors='white', linewidth=0.3)
                else:
                    # Fallback to original coloring if mean calculation fails
                    ax_main.scatter(m2_r, m1_r, alpha=0.6, s=8, 
                                  color=NATURE_COLORS['accent'])
                
                # Add 1:1 line
                min_r = min(np.min(m1_r), np.min(m2_r))
                max_r = max(np.max(m1_r), np.max(m2_r))
                ax_main.plot([min_r, max_r], [min_r, max_r], 
                            'k--', alpha=0.5, linewidth=1)
                
                # Add correlation coefficient
                spearman_r, spearman_p = spearmanr(m1_r, m2_r)
                
                # Add binomial test
                n_m1_better = np.sum(m1_r > m2_r)
                n_total = len(m1_r)
                binom_result = binomtest(n_m1_better, n_total, p=0.5, alternative='two-sided')
                binom_p = binom_result.pvalue
                
                # Format p-value text
                if binom_p < 0.001:
                    binom_text = 'Binomial test p < 0.001'
                else:
                    binom_text = f'Binomial test p = {binom_p:.3f}'
                
                # Display both statistics
                stats_text = f'ρ = {spearman_r:.3f}\n{binom_text}'
                ax_main.text(0.05, 0.95, stats_text, 
                            transform=ax_main.transAxes, fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            verticalalignment='top')
                
                # Top marginal (Method 2 now on X-axis)
                ax_top.hist(m2_r, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], 
                           density=True)
                ax_top.set_xlim(ax_main.get_xlim())
                ax_top.set_xticks([])
                ax_top.set_ylabel('Density', fontsize=8)
                
                # Right marginal (Method 1 now on Y-axis)
                ax_right.hist(m1_r, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                             orientation='horizontal', density=True)
                ax_right.set_ylim(ax_main.get_ylim())
                ax_right.set_yticks([])
                ax_right.set_xlabel('Density', fontsize=8)
                
                # Find and label extreme features correctly (note: now m2 - m1 since axes switched)
                platform_feat_data = feat_metrics[feat_metrics['platform'] == platform_key]
                
                if len(platform_feat_data) > 0:
                    # Get aligned data for both methods
                    m1_data = platform_feat_data[platform_feat_data['method'] == 'Method_1'][['feature', 'r']].set_index('feature')
                    m2_data = platform_feat_data[platform_feat_data['method'] == 'Method_2'][['feature', 'r']].set_index('feature')
                    
                    # Merge to ensure alignment
                    merged_r = m1_data.join(m2_data, lsuffix='_m1', rsuffix='_m2').dropna()
                    
                    if len(merged_r) > 0:
                        # Calculate differences correctly aligned with feature names (m1 - m2 since Y is m1, X is m2)
                        merged_r['diff'] = merged_r['r_m1'] - merged_r['r_m2']
                        
                        # Find extreme features
                        top_features = merged_r.nlargest(2, 'diff')  # Top 2 improved
                        bottom_features = merged_r.nsmallest(2, 'diff')  # Top 2 worsened
                        
                        # Label extreme points (note coordinates switched: X=m2, Y=m1)
                        for feature_name, row in pd.concat([top_features, bottom_features]).iterrows():
                            ax_main.annotate(feature_name[:10], 
                                           (row['r_m2'], row['r_m1']),
                                           fontsize=6, alpha=0.8,
                                           xytext=(5, 5), textcoords='offset points')
            else:
                # No data for this platform
                ax_main.text(0.5, 0.5, f'No data available\nfor {platform_name_map[platform_key]}', 
                           ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
                ax_top.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_top.transAxes)
                ax_right.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_right.transAxes)
            
            # Labels and title (note: axes switched)
            ax_main.set_xlabel(f'r ({data.method2_name})')
            ax_main.set_ylabel(f'r ({data.method1_name})')
            ax_main.set_title(f'{platform_name_map[platform_key]}')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_2_sample_r_scatter(self, data: AnalysisData):
        """Figure 2: Sample-wise r scatter with marginals"""
        print("Generating Figure 2: Sample-wise r scatter...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Get sample-wise correlations
        samp_metrics = data.metrics['sample_wise']
        
        # Check if we have any sample metrics
        if samp_metrics.empty:
            print("    No sample metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No sample metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Sample-wise Performance Scatter Plot')
            return fig
        
        # Platform A comparison
        platform_a_data = samp_metrics[samp_metrics['platform'] == 'Platform_A']
        platform_b_data = samp_metrics[samp_metrics['platform'] == 'Platform_B']
        
        if len(platform_a_data) == 0 and len(platform_b_data) == 0:
            print("    No data for either platform - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No sample data available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Sample-wise Performance Scatter Plot')
            return fig
        
        # Get aligned method data for each platform (FIX: Critical data alignment bug)
        def get_aligned_sample_data(platform_data):
            if len(platform_data) == 0:
                return np.array([]), np.array([])
            
            # Create separate Series for each method, indexed by sample
            m1_series = platform_data[platform_data['method'] == 'Method_1'].set_index('sample')['r']
            m2_series = platform_data[platform_data['method'] == 'Method_2'].set_index('sample')['r']
            
            # Combine and align by sample index, drop samples not present in both methods
            combined = pd.concat([m1_series, m2_series], axis=1, keys=['m1', 'm2']).dropna()
            
            if combined.empty:
                return np.array([]), np.array([])
            
            return combined['m1'].values, combined['m2'].values
        
        m1_a, m2_a = get_aligned_sample_data(platform_a_data)
        m1_b, m2_b = get_aligned_sample_data(platform_b_data)
        
        # Create figure (similar structure to Figure 1)
        fig = plt.figure(figsize=(13, 5))
        
        for i, (platform_key, m1_r, m2_r) in enumerate([('Platform_A', m1_a, m2_a), 
                                                        ('Platform_B', m1_b, m2_b)]):
            gs = GridSpec(3, 3, figure=fig, 
                         left=0.1 + i*0.45, right=0.4 + i*0.45,
                         bottom=0.1, top=0.9,
                         hspace=0.05, wspace=0.05)
            
            ax_main = fig.add_subplot(gs[1:, :-1])
            ax_top = fig.add_subplot(gs[0, :-1])
            ax_right = fig.add_subplot(gs[1:, -1])
            
            if len(m1_r) > 0 and len(m2_r) > 0 and len(m1_r) == len(m2_r):
                # Main scatter plot (switched axes: X=m2, Y=m1)
                ax_main.scatter(m2_r, m1_r, alpha=0.6, s=6, 
                               color=NATURE_COLORS['accent'])
                
                # Add 1:1 line
                min_r = min(np.min(m1_r), np.min(m2_r))
                max_r = max(np.max(m1_r), np.max(m2_r))
                ax_main.plot([min_r, max_r], [min_r, max_r], 
                            'k--', alpha=0.5, linewidth=1)
                
                # Add correlation coefficient
                spearman_r, spearman_p = spearmanr(m1_r, m2_r)
                
                # Add binomial test
                n_m1_better = np.sum(m1_r > m2_r)
                n_total = len(m1_r)
                binom_result = binomtest(n_m1_better, n_total, p=0.5, alternative='two-sided')
                binom_p = binom_result.pvalue
                
                # Format p-value text
                if binom_p < 0.001:
                    binom_text = 'Binomial test p < 0.001'
                else:
                    binom_text = f'Binomial test p = {binom_p:.3f}'
                
                # Display both statistics
                stats_text = f'ρ = {spearman_r:.3f}\n{binom_text}'
                ax_main.text(0.05, 0.95, stats_text, 
                            transform=ax_main.transAxes, fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            verticalalignment='top')
                
                # Top marginal (Method 2 now on X-axis)
                ax_top.hist(m2_r, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], 
                           density=True)
                ax_top.set_xlim(ax_main.get_xlim())
                ax_top.set_xticks([])
                ax_top.set_ylabel('Density', fontsize=8)
                
                # Right marginal (Method 1 now on Y-axis)
                ax_right.hist(m1_r, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                             orientation='horizontal', density=True)
                ax_right.set_ylim(ax_main.get_ylim())
                ax_right.set_yticks([])
                ax_right.set_xlabel('Density', fontsize=8)
            else:
                # No data for this platform
                ax_main.text(0.5, 0.5, f'No data available\nfor {platform_name_map[platform_key]}', 
                           ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
                ax_top.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_top.transAxes)
                ax_right.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_right.transAxes)
            
            # Labels and title
            ax_main.set_xlabel(f'r ({data.method2_name})')
            ax_main.set_ylabel(f'r ({data.method1_name})')
            ax_main.set_title(f'{platform_name_map[platform_key]} (Sample-wise)')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_3_r_distribution_ridge(self, data: AnalysisData):
        """Figure 3: Distribution ridge/violin of feature-wise and sample-wise r with comprehensive statistics"""
        print("Generating Figure 3: Feature-wise and sample-wise r distributions...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        feat_metrics = data.metrics['feature_wise']
        sample_metrics = data.metrics['sample_wise']
        
        # Check if we have any metrics
        if feat_metrics.empty and sample_metrics.empty:
            print("    No metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No metrics available\n(No features in common between platforms)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Distributions')
            return fig
        
        # Create 2 rows (feature-wise and sample-wise), each subplot is square
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        fig.suptitle('Comprehensive Distribution Analysis: Feature-wise and Sample-wise', fontsize=16, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1'],
                 NATURE_COLORS['alternative_2'], NATURE_COLORS['alternative_3'],
                 NATURE_COLORS['highlight'], NATURE_COLORS['alternative_4']]
        
        # Function to create combined violin plot with both platforms merged
        def create_combined_violin_plot(ax, feat_data_a, feat_data_b, metric_name, plot_title):
            methods_data = []
            labels = []
            colors_used = []
            
            # First, collect Platform A methods (first 4 columns)
            platform_a_data = feat_data_a if len(feat_data_a) > 0 else pd.DataFrame()
            if len(platform_a_data) > 0:
                method_mapping = {
                    'Method_1': data.method1_name,
                    'Method_2': data.method2_name,
                    'Method_3': data.method3_name,
                    'Method_4': data.method4_name
                }
                
                for i, (method_key, method_name) in enumerate(method_mapping.items()):
                    if method_name is not None:
                        method_subset = platform_a_data[platform_a_data['method'] == method_key]
                        if len(method_subset) > 0:
                            method_data = method_subset['r'].values
                            methods_data.append(method_data)
                            labels.append(f'{method_name}\n({data.platform_a_name})')
                            colors_used.append(colors[i])
            
            # Then, collect Platform B methods (last 4 columns)
            platform_b_data = feat_data_b if len(feat_data_b) > 0 else pd.DataFrame()
            if len(platform_b_data) > 0:
                method_mapping = {
                    'Method_1': data.method1_name,
                    'Method_2': data.method2_name,
                    'Method_3': data.method3_name,
                    'Method_4': data.method4_name
                }
                
                for i, (method_key, method_name) in enumerate(method_mapping.items()):
                    if method_name is not None:
                        method_subset = platform_b_data[platform_b_data['method'] == method_key]
                        if len(method_subset) > 0:
                            method_data = method_subset['r'].values
                            methods_data.append(method_data)
                            labels.append(f'{method_name}\n({data.platform_b_name})')
                            colors_used.append(colors[i + 4])  # Use different colors for Platform B
            
            if len(methods_data) > 0 and all(len(md) > 0 for md in methods_data):
                # Create violin plot with enhanced features
                positions = range(1, len(methods_data)+1)
                violin_parts = ax.violinplot(methods_data, positions=positions, widths=0.6, 
                                            showmeans=True, showmedians=True, showextrema=True)
                
                # Color the violins and enhance appearance
                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors_used[i] if i < len(colors_used) else colors[i % len(colors)])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.8)
                
                # Style the statistical elements
                if 'cmeans' in violin_parts:
                    violin_parts['cmeans'].set_color('red')
                    violin_parts['cmeans'].set_linewidth(2)
                if 'cmedians' in violin_parts:
                    violin_parts['cmedians'].set_color('blue')
                    violin_parts['cmedians'].set_linewidth(2)
                if 'cbars' in violin_parts:
                    violin_parts['cbars'].set_color('black')
                    violin_parts['cbars'].set_linewidth(1.5)
                if 'cmins' in violin_parts:
                    violin_parts['cmins'].set_color('black')
                    violin_parts['cmins'].set_linewidth(1.5)
                if 'cmaxes' in violin_parts:
                    violin_parts['cmaxes'].set_color('black')
                    violin_parts['cmaxes'].set_linewidth(1.5)
                
                # Add statistical annotations and quartile lines
                for i, (pos, method_data, label) in enumerate(zip(positions, methods_data, labels)):
                    if len(method_data) > 0:
                        # Calculate comprehensive statistics
                        mean_val = np.mean(method_data)
                        median_val = np.median(method_data)
                        q25 = np.percentile(method_data, 25)
                        q75 = np.percentile(method_data, 75)
                        std_val = np.std(method_data)
                        n_samples = len(method_data)
                        
                        # Add Q1 and Q3 horizontal lines across the violin width
                        violin_width = 0.3  # Half width of violin
                        ax.hlines(q25, pos - violin_width, pos + violin_width, 
                                 colors='green', linestyles='--', linewidth=2, alpha=0.8,
                                 label='Q1' if i == 0 else "")
                        ax.hlines(q75, pos - violin_width, pos + violin_width, 
                                 colors='orange', linestyles='--', linewidth=2, alpha=0.8,
                                 label='Q3' if i == 0 else "")
                
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                return True  # Successfully created plot
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_ylabel(f'{metric_name} Correlation (r)', fontweight='bold')
            ax.set_title(plot_title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0 for correlations
            return False  # No plot created
        
        # Create merged plots: platforms combined in same subplot
        # Feature-wise plot (both platforms combined)
        platform_a_feat_data = feat_metrics[feat_metrics['platform'] == 'Platform_A'] if not feat_metrics.empty else pd.DataFrame()
        platform_b_feat_data = feat_metrics[feat_metrics['platform'] == 'Platform_B'] if not feat_metrics.empty else pd.DataFrame()
        create_combined_violin_plot(ax1, platform_a_feat_data, platform_b_feat_data, 'Feature-wise', 
                                   'Feature-wise Performance Distribution (Both Platforms)')
        
        # Sample-wise plot (both platforms combined) 
        platform_a_samp_data = sample_metrics[sample_metrics['platform'] == 'Platform_A'] if not sample_metrics.empty else pd.DataFrame()
        platform_b_samp_data = sample_metrics[sample_metrics['platform'] == 'Platform_B'] if not sample_metrics.empty else pd.DataFrame()
        create_combined_violin_plot(ax2, platform_a_samp_data, platform_b_samp_data, 'Sample-wise', 
                                   'Sample-wise Performance Distribution (Both Platforms)')
        
        # Add overall legend (only once)
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Median'),
            plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='Q1'),
            plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Q3'),
            plt.Line2D([0], [0], color='black', lw=1.5, label='Min/Max')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_3b_vertical_violin_plots(self, data: AnalysisData):
        """Figure 3b: Vertically stacked violin plots with Method 4 as horizontal baseline and statistical testing (cross-platform)"""
        print("Generating Figure 3b: Vertical violin plots with statistical testing (cross-platform)...")
        
        from scipy.stats import mannwhitneyu
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        feat_metrics = data.metrics['feature_wise']
        sample_metrics = data.metrics['sample_wise']
        
        # Check if we have any metrics
        if feat_metrics.empty and sample_metrics.empty:
            print("    No metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No metrics available\n(No features in common between platforms)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Distributions')
            return fig
        
        # Create 2x2 layout for feature-wise and sample-wise, each platform separate
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle('Cross-Platform Vertical Distribution Analysis with Statistical Testing', 
                     fontsize=16, fontweight='bold')
        # Make subplots square
        for ax in (ax1, ax2, ax3, ax4):
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
        
        platforms = ['Platform_A', 'Platform_B']
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1'],
                 NATURE_COLORS['alternative_2'], NATURE_COLORS['alternative_3'],
                 NATURE_COLORS['highlight'], NATURE_COLORS['alternative_4']]
        
        def create_single_platform_vertical_violin_plot_with_stats(ax, platform_data, metric_name, plot_title, platform_name):
            """Create vertical violin plots with baseline and top significance bars for a single platform"""
            methods_data = []
            labels = []
            method_keys = []
            baseline_data = None
            baseline_name = None
            
            if len(platform_data) > 0:
                # Collect all available methods for this platform
                method_mapping = {
                    'Method_1': data.method1_name,
                    'Method_2': data.method2_name,
                    'Method_3': data.method3_name,
                    'Method_4': data.method4_name
                }
                
                for method_key, method_name in method_mapping.items():
                    if method_name is not None:
                        method_subset = platform_data[platform_data['method'] == method_key]
                        if len(method_subset) > 0:
                            method_data = method_subset['r'].values
                            
                            # Check if this is Method 4 (baseline)
                            if method_key == 'Method_4':
                                baseline_data = method_data
                                baseline_name = method_name
                            else:
                                methods_data.append(method_data)
                                labels.append(method_name)
                                method_keys.append(method_key)
            
            if len(methods_data) > 0:
                # Create vertical violin plot
                n_methods = len(methods_data)
                positions = list(range(1, n_methods + 1))

                # Preserve method order: Method 1, Method 2, Method 3
                methods_data_plot = methods_data
                labels_plot = labels

                violin_parts = ax.violinplot(
                    methods_data_plot,
                    positions=positions,
                    widths=0.6,
                    vert=True,
                    showmeans=True,
                    showmedians=True,
                    showextrema=True,
                )

                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.8)

                if 'cmeans' in violin_parts:
                    violin_parts['cmeans'].set_color('red')
                    violin_parts['cmeans'].set_linewidth(2)
                if 'cmedians' in violin_parts:
                    violin_parts['cmedians'].set_color('blue')
                    violin_parts['cmedians'].set_linewidth(2)
                if 'cbars' in violin_parts:
                    violin_parts['cbars'].set_color('black')
                    violin_parts['cbars'].set_linewidth(1.5)
                if 'cmins' in violin_parts:
                    violin_parts['cmins'].set_color('black')
                    violin_parts['cmins'].set_linewidth(1.5)
                if 'cmaxes' in violin_parts:
                    violin_parts['cmaxes'].set_color('black')
                    violin_parts['cmaxes'].set_linewidth(1.5)

                baseline_mean = None
                if baseline_data is not None:
                    baseline_mean = np.mean(baseline_data)
                    ax.axhline(
                        baseline_mean,
                        color=colors[3],
                        linestyle='--',
                        linewidth=3,
                        alpha=0.9,
                        label=f'{baseline_name} (baseline): {baseline_mean:.3f}',
                        zorder=10,
                    )

                # Pairwise statistical tests with top significance bars
                data_vals = [float(v) for d in methods_data for v in np.atleast_1d(d)]
                data_min = (min(data_vals) if len(data_vals) > 0 else 0.0)
                if baseline_mean is not None:
                    data_min = min(data_min, float(baseline_mean))
                if len(methods_data) > 1:
                    data_max = max([float(np.max(d)) for d in methods_data])
                    if baseline_mean is not None:
                        data_max = max(data_max, float(baseline_mean))
                    top_base = max(1.0, data_max)
                    y_start = top_base + 0.03
                    y_step = 0.04
                    comparison_idx = 0

                    for i in range(len(methods_data_plot)):
                        for j in range(i + 1, len(methods_data_plot)):
                            try:
                                _, p_value = mannwhitneyu(
                                    methods_data_plot[i],
                                    methods_data_plot[j],
                                    alternative='two-sided',
                                )
                                if p_value < 0.001:
                                    sig_text = '***'
                                elif p_value < 0.01:
                                    sig_text = '**'
                                elif p_value < 0.05:
                                    sig_text = '*'
                                else:
                                    sig_text = 'ns'

                                y = y_start + y_step * comparison_idx
                                x1, x2 = positions[i], positions[j]
                                ax.plot([x1, x1, x2, x2], [y - 0.005, y, y, y - 0.005], 'k-', linewidth=1)
                                ax.text((x1 + x2) / 2.0, y + 0.01, sig_text, ha='center', va='bottom', fontsize=8)
                                comparison_idx += 1
                            except Exception as e:
                                print(f"    Warning: Statistical test failed for {labels_plot[i]} vs {labels_plot[j]}: {e}")

                    lower_bound = min(0.0, data_min - 0.05)
                    ax.set_ylim(lower_bound, y_start + y_step * (comparison_idx + 1))
                else:
                    top_base = max(1.0, (baseline_mean + 0.1) if baseline_mean is not None else 1.0)
                    lower_bound = min(0.0, data_min - 0.05)
                    ax.set_ylim(lower_bound, top_base)

                ax.set_xticks(positions)
                ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel(f'{metric_name} Correlation (r)', fontweight='bold')
                ax.set_title(plot_title, fontweight='bold')
                ax.set_xlim(0.5, n_methods + 0.5)
                ax.grid(False)

                if baseline_data is not None:
                    ax.legend(loc='upper left', fontsize=8)

                return True
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                
            ax.set_xlabel(f'{metric_name} Correlation (r)', fontweight='bold')
            ax.set_title(plot_title, fontweight='bold')
            ax.grid(False)
            ax.set_xlim(0, 1)  # Set correlation range to 0-1
            return False
        
        # Get data for both platforms
        platform_a_feat_data = feat_metrics[feat_metrics['platform'] == 'Platform_A'] if not feat_metrics.empty else pd.DataFrame()
        platform_b_feat_data = feat_metrics[feat_metrics['platform'] == 'Platform_B'] if not feat_metrics.empty else pd.DataFrame()
        platform_a_samp_data = sample_metrics[sample_metrics['platform'] == 'Platform_A'] if not sample_metrics.empty else pd.DataFrame()
        platform_b_samp_data = sample_metrics[sample_metrics['platform'] == 'Platform_B'] if not sample_metrics.empty else pd.DataFrame()
        
        # Create 4 separate vertical violin plots
        create_single_platform_vertical_violin_plot_with_stats(
            ax1, platform_a_feat_data, 'Feature-wise', 
            f'Feature-wise: {data.platform_a_name}', data.platform_a_name)
        create_single_platform_vertical_violin_plot_with_stats(
            ax2, platform_b_feat_data, 'Feature-wise', 
            f'Feature-wise: {data.platform_b_name}', data.platform_b_name)
        create_single_platform_vertical_violin_plot_with_stats(
            ax3, platform_a_samp_data, 'Sample-wise', 
            f'Sample-wise: {data.platform_a_name}', data.platform_a_name)
        create_single_platform_vertical_violin_plot_with_stats(
            ax4, platform_b_samp_data, 'Sample-wise', 
            f'Sample-wise: {data.platform_b_name}', data.platform_b_name)
        
        # Add overall legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Median'),
            plt.Line2D([0], [0], color='gray', lw=1.5, label='Min/Max'),
            plt.Line2D([0], [0], color=colors[3], lw=3, linestyle='--', label='Method 4 Baseline')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4_bland_altman(self, data: AnalysisData):
        """Figure 4: Bland-Altman plots for bias assessment - separate subplot for each method"""
        print("Generating Figure 4: Bland-Altman plots...")
        
        # Get all available methods
        available_methods = self._get_available_methods(data)
        
        # Group methods by platform to ensure consistent scaling
        platform_a_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_A']
        platform_b_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_B']
        
        n_methods_a = len(platform_a_methods)
        n_methods_b = len(platform_b_methods)
        total_methods = n_methods_a + n_methods_b
        
        if total_methods == 0:
            print("No methods available for Bland-Altman analysis")
            return plt.figure()
        
        # Create subplots: 2 rows (one for each platform), methods as columns
        max_methods_per_platform = max(n_methods_a, n_methods_b, 1)
        fig, axes = plt.subplots(2, max_methods_per_platform, figsize=(5*max_methods_per_platform, 10))
        
        # Handle case where we have only one column
        if max_methods_per_platform == 1:
            axes = axes.reshape(2, 1)
        
        # Make all plots square
        for i in range(2):
            for j in range(max_methods_per_platform):
                axes[i, j].set_aspect('equal', adjustable='box')
        
        fig.suptitle('Bland-Altman Analysis: Individual Method Comparison', fontsize=16, fontweight='bold')
        
        # Colors for methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Calculate global limits for each platform to ensure consistent scaling
        def get_platform_limits(methods):
            all_truth_vals = []
            all_diff_vals = []
            
            for method_key, method_name, truth, imputed in methods:
                truth_flat = truth.values.flatten()
                imp_flat = imputed.values.flatten()
                mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
                truth_clean = truth_flat[mask]
                imp_clean = imp_flat[mask]
                diff_clean = imp_clean - truth_clean
                
                all_truth_vals.extend(truth_clean)
                all_diff_vals.extend(diff_clean)
            
            if all_truth_vals:
                truth_min, truth_max = np.min(all_truth_vals), np.max(all_truth_vals)
                diff_min, diff_max = np.min(all_diff_vals), np.max(all_diff_vals)
                # Add some padding
                truth_range = truth_max - truth_min
                diff_range = diff_max - diff_min
                truth_limits = (truth_min - 0.05*truth_range, truth_max + 0.05*truth_range)
                diff_limits = (diff_min - 0.05*diff_range, diff_max + 0.05*diff_range)
                return truth_limits, diff_limits
            return (0, 1), (-1, 1)
        
        # Get limits for consistent scaling within each platform
        if platform_a_methods:
            truth_limits_a, diff_limits_a = get_platform_limits(platform_a_methods)
        else:
            truth_limits_a, diff_limits_a = (0, 1), (-1, 1)
            
        if platform_b_methods:
            truth_limits_b, diff_limits_b = get_platform_limits(platform_b_methods)
        else:
            truth_limits_b, diff_limits_b = (0, 1), (-1, 1)
        
        # Plot Platform A methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            ax = axes[0, i]
            
            # Flatten data for BA plot
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            # Remove NaN values
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            # Bland-Altman calculations
            diff_vals = imp_clean - truth_clean
            mean_diff = np.mean(diff_vals)
            std_diff = np.std(diff_vals)
            
            # Create scatter plot
            ax.scatter(truth_clean, diff_vals, alpha=0.3, s=0.1, 
                      color=colors[i % len(colors)], rasterized=True)
            
            # Add mean line and limits of agreement
            ax.axhline(mean_diff, color=colors[i % len(colors)], linestyle='-', 
                      linewidth=2, alpha=0.8, label=f'Bias: {mean_diff:.3f}')
            ax.axhline(mean_diff + 1.96*std_diff, color=colors[i % len(colors)], 
                      linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(mean_diff - 1.96*std_diff, color=colors[i % len(colors)], 
                      linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # Set consistent limits
            ax.set_xlim(truth_limits_a)
            ax.set_ylim(diff_limits_a)
            
            ax.set_title(f'{data.platform_a_name}\n{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add statistics text
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}\n95% LoA=[{mean_diff-1.96*std_diff:.3f}, {mean_diff+1.96*std_diff:.3f}]', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Hide unused Platform A subplots
        for i in range(n_methods_a, max_methods_per_platform):
            axes[0, i].set_visible(False)
        
        # Plot Platform B methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_b_methods):
            ax = axes[1, i]
            
            # Flatten data for BA plot
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            # Remove NaN values
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            # Bland-Altman calculations
            diff_vals = imp_clean - truth_clean
            mean_diff = np.mean(diff_vals)
            std_diff = np.std(diff_vals)
            
            # Create scatter plot
            ax.scatter(truth_clean, diff_vals, alpha=0.3, s=0.1, 
                      color=colors[i % len(colors)], rasterized=True)
            
            # Add mean line and limits of agreement
            ax.axhline(mean_diff, color=colors[i % len(colors)], linestyle='-', 
                      linewidth=2, alpha=0.8, label=f'Bias: {mean_diff:.3f}')
            ax.axhline(mean_diff + 1.96*std_diff, color=colors[i % len(colors)], 
                      linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(mean_diff - 1.96*std_diff, color=colors[i % len(colors)], 
                      linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # Set consistent limits
            ax.set_xlim(truth_limits_b)
            ax.set_ylim(diff_limits_b)
            
            ax.set_title(f'{data.platform_b_name}\n{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add statistics text
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}\n95% LoA=[{mean_diff-1.96*std_diff:.3f}, {mean_diff+1.96*std_diff:.3f}]', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Hide unused Platform B subplots
        for i in range(n_methods_b, max_methods_per_platform):
            axes[1, i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4b_bias_comparison(self, data: AnalysisData):
        """Figure 4b: Cross-platform bias comparison panel across all methods"""
        print("Generating Figure 4b: Cross-platform bias comparison...")
        
        # Get all available methods
        available_methods = self._get_available_methods(data)
        
        # Group methods by platform
        platform_a_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_A']
        platform_b_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_B']
        
        # Exclude Method 4 (permuted baseline) from bias comparison
        platform_a_methods = [
            (method_key, method_name, truth, imputed)
            for (method_key, method_name, truth, imputed) in platform_a_methods
            if method_key != 'Method_4'
        ]
        platform_b_methods = [
            (method_key, method_name, truth, imputed)
            for (method_key, method_name, truth, imputed) in platform_b_methods
            if method_key != 'Method_4'
        ]
        
        if len(platform_a_methods) == 0 and len(platform_b_methods) == 0:
            print("No methods available for bias comparison")
            return plt.figure()
        
        # Create figure with 2 subfigures (one for each platform) - 10x5 total (two 5x5 subfigures)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Colors for methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Collect bias data for Platform A
        bias_data_a = []
        method_labels_a = []
        bias_colors_a = []
        bias_errors_a = []
        sample_sizes_a = []
        
        # Collect bias data for Platform B
        bias_data_b = []
        method_labels_b = []
        bias_colors_b = []
        bias_errors_b = []
        sample_sizes_b = []
        
        # Platform A methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            # Calculate bias statistics
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            
            if np.sum(mask) > 0:
                truth_clean = truth_flat[mask]
                imp_clean = imp_flat[mask]
                diff_vals = imp_clean - truth_clean
                
                mean_bias = np.mean(diff_vals)
                se_bias = np.std(diff_vals) / np.sqrt(len(diff_vals))  # Standard error
                ci_95 = 1.96 * se_bias  # 95% confidence interval
                
                bias_data_a.append(mean_bias)
                bias_errors_a.append(ci_95)
                sample_sizes_a.append(len(diff_vals))
            else:
                bias_data_a.append(0)
                bias_errors_a.append(0)
                sample_sizes_a.append(0)
            
            method_labels_a.append(method_name)
            bias_colors_a.append(colors[i % len(colors)])
        
        # Platform B methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_b_methods):
            # Calculate bias statistics
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            
            if np.sum(mask) > 0:
                truth_clean = truth_flat[mask]
                imp_clean = imp_flat[mask]
                diff_vals = imp_clean - truth_clean
                
                mean_bias = np.mean(diff_vals)
                se_bias = np.std(diff_vals) / np.sqrt(len(diff_vals))  # Standard error
                ci_95 = 1.96 * se_bias  # 95% confidence interval
                
                bias_data_b.append(mean_bias)
                bias_errors_b.append(ci_95)
                sample_sizes_b.append(len(diff_vals))
            else:
                bias_data_b.append(0)
                bias_errors_b.append(0)
                sample_sizes_b.append(0)
            
            method_labels_b.append(method_name)
            bias_colors_b.append(colors[i % len(colors)])
        
        # Plot Platform A bias comparison (left subplot)
        if bias_data_a:
            x_pos_a = np.arange(len(method_labels_a))
            bars_a = ax1.bar(x_pos_a, bias_data_a, yerr=bias_errors_a, capsize=5,
                           color=bias_colors_a, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add horizontal reference line at bias = 0
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Add value labels on bars
            if len(bias_data_a) > 0:
                data_range_a = max(bias_data_a) - min(bias_data_a) if max(bias_data_a) != min(bias_data_a) else 0.1
                for i, (bar, bias_val, n_samples) in enumerate(zip(bars_a, bias_data_a, sample_sizes_a)):
                    height = bar.get_height()
                    label_y = height + bias_errors_a[i] + 0.02 * data_range_a
                    ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{bias_val:.3f}\n(n={n_samples:,})',
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Bias (Imputed - Truth)')
            ax1.set_title(f'{data.platform_a_name} Bias Comparison', fontweight='bold', fontsize=12)
            ax1.set_xticks(x_pos_a)
            ax1.set_xticklabels(method_labels_a, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add summary statistics
            mean_abs_bias_a = np.mean([abs(b) for b in bias_data_a])
            ax1.text(0.02, 0.98, f'Mean |Bias|: {mean_abs_bias_a:.3f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, f'No bias data for {data.platform_a_name}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title(f'{data.platform_a_name} Bias Comparison', fontweight='bold', fontsize=12)
        
        # Plot Platform B bias comparison (right subplot)
        if bias_data_b:
            x_pos_b = np.arange(len(method_labels_b))
            bars_b = ax2.bar(x_pos_b, bias_data_b, yerr=bias_errors_b, capsize=5,
                           color=bias_colors_b, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add horizontal reference line at bias = 0
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Add value labels on bars
            if len(bias_data_b) > 0:
                data_range_b = max(bias_data_b) - min(bias_data_b) if max(bias_data_b) != min(bias_data_b) else 0.1
                for i, (bar, bias_val, n_samples) in enumerate(zip(bars_b, bias_data_b, sample_sizes_b)):
                    height = bar.get_height()
                    label_y = height + bias_errors_b[i] + 0.02 * data_range_b
                    ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{bias_val:.3f}\n(n={n_samples:,})',
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Bias (Imputed - Truth)')
            ax2.set_title(f'{data.platform_b_name} Bias Comparison', fontweight='bold', fontsize=12)
            ax2.set_xticks(x_pos_b)
            ax2.set_xticklabels(method_labels_b, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add summary statistics
            mean_abs_bias_b = np.mean([abs(b) for b in bias_data_b])
            ax2.text(0.02, 0.98, f'Mean |Bias|: {mean_abs_bias_b:.3f}', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        else:
            ax2.text(0.5, 0.5, f'No bias data for {data.platform_b_name}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'{data.platform_b_name} Bias Comparison', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4c_bland_altman_density(self, data: AnalysisData):
        """Figure 4c: Bland-Altman density plots (high-resolution) - one subplot per method"""
        print("Generating Figure 4c: Bland-Altman density plots...")

        # Get all available methods
        available_methods = self._get_available_methods(data)

        # Group methods by platform to ensure consistent scaling
        platform_a_methods = [(method_key, method_name, truth, imputed)
                              for method_key, method_name, platform, truth, imputed in available_methods
                              if platform == 'Platform_A']
        platform_b_methods = [(method_key, method_name, truth, imputed)
                              for method_key, method_name, platform, truth, imputed in available_methods
                              if platform == 'Platform_B']

        n_methods_a = len(platform_a_methods)
        n_methods_b = len(platform_b_methods)
        total_methods = n_methods_a + n_methods_b

        if total_methods == 0:
            print("No methods available for Bland-Altman density analysis")
            return plt.figure()

        # Create subplots: 2 rows (one for each platform), methods as columns
        max_methods_per_platform = max(n_methods_a, n_methods_b, 1)
        fig, axes = plt.subplots(2, max_methods_per_platform, figsize=(5*max_methods_per_platform, 10))

        # Handle single column case
        if max_methods_per_platform == 1:
            axes = axes.reshape(2, 1)

        # Make all plots square
        for i in range(2):
            for j in range(max_methods_per_platform):
                axes[i, j].set_aspect('equal', adjustable='box')

        fig.suptitle('Bland-Altman Analysis (Density): Individual Method Comparison', fontsize=16, fontweight='bold')

        # Calculate global limits for each platform
        def get_platform_limits(methods):
            all_truth_vals = []
            all_diff_vals = []
            for method_key, method_name, truth, imputed in methods:
                truth_flat = truth.values.flatten()
                imp_flat = imputed.values.flatten()
                mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
                truth_clean = truth_flat[mask]
                imp_clean = imp_flat[mask]
                diff_clean = imp_clean - truth_clean
                if truth_clean.size:
                    all_truth_vals.extend(truth_clean)
                    all_diff_vals.extend(diff_clean)
            if all_truth_vals:
                truth_min, truth_max = np.min(all_truth_vals), np.max(all_truth_vals)
                diff_min, diff_max = np.min(all_diff_vals), np.max(all_diff_vals)
                truth_range = truth_max - truth_min
                diff_range = diff_max - diff_min
                truth_limits = (truth_min - 0.05*truth_range, truth_max + 0.05*truth_range)
                diff_limits = (diff_min - 0.05*diff_range, diff_max + 0.05*diff_range)
                return truth_limits, diff_limits
            return (0, 1), (-1, 1)

        if platform_a_methods:
            truth_limits_a, diff_limits_a = get_platform_limits(platform_a_methods)
        else:
            truth_limits_a, diff_limits_a = (0, 1), (-1, 1)

        if platform_b_methods:
            truth_limits_b, diff_limits_b = get_platform_limits(platform_b_methods)
        else:
            truth_limits_b, diff_limits_b = (0, 1), (-1, 1)

        # Internal helper to render 2D density with high resolution
        def plot_density(ax, x_vals, y_vals, xlim, ylim):
            import matplotlib.colors as mcolors
            # High-resolution binning
            bins = (400, 400)
            # Clip to limits
            x = np.clip(x_vals, xlim[0], xlim[1])
            y = np.clip(y_vals, ylim[0], ylim[1])
            # 2D histogram
            H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
            H = H.T
            # Mask zero-density bins to show as white background
            H_masked = np.ma.masked_where(H == 0, H)
            # Build colormap with white for masked (bad) values
            base = plt.cm.get_cmap('magma', 256)
            cmap = mcolors.ListedColormap(base(np.linspace(0, 1, 256)))
            cmap.set_bad('white')
            # Log normalization on positive densities
            vmin = H_masked.min() if H_masked.count() > 0 else 1e-6
            vmax = H_masked.max() if H_masked.count() > 0 else 1.0
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.set_facecolor('white')
            im = ax.imshow(H_masked, extent=extent, origin='lower', aspect='auto',
                           cmap=cmap, norm=norm)
            return im

        # Plot Platform A methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            ax = axes[0, i]
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            diff_vals = imp_clean - truth_clean

            # Render density
            plot_density(ax, truth_clean, diff_vals, truth_limits_a, diff_limits_a)

            # Stats and reference lines
            mean_diff = float(np.mean(diff_vals)) if diff_vals.size else 0.0
            std_diff = float(np.std(diff_vals)) if diff_vals.size else 0.0
            ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, alpha=0.9)
            ax.axhline(mean_diff + 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(mean_diff - 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

            ax.set_xlim(truth_limits_a)
            ax.set_ylim(diff_limits_a)
            ax.set_title(f'{data.platform_a_name}\n{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(False)
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

        # Hide unused Platform A subplots
        for i in range(n_methods_a, max_methods_per_platform):
            axes[0, i].set_visible(False)

        # Plot Platform B methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_b_methods):
            ax = axes[1, i]
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            diff_vals = imp_clean - truth_clean

            plot_density(ax, truth_clean, diff_vals, truth_limits_b, diff_limits_b)

            mean_diff = float(np.mean(diff_vals)) if diff_vals.size else 0.0
            std_diff = float(np.std(diff_vals)) if diff_vals.size else 0.0
            ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, alpha=0.9)
            ax.axhline(mean_diff + 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(mean_diff - 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

            ax.set_xlim(truth_limits_b)
            ax.set_ylim(diff_limits_b)
            ax.set_title(f'{data.platform_b_name}\n{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(False)
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

        # Hide unused Platform B subplots
        for i in range(n_methods_b, max_methods_per_platform):
            axes[1, i].set_visible(False)

        plt.tight_layout()
        return fig

    def generate_figure_4d_variance_vs_mean(self, data: AnalysisData):
        """Figure 4d: Variance vs Mean plots (feature-wise and sample-wise) for both platforms"""
        print("Generating Figure 4d: Variance vs Mean plots...")

        # Prepare subplots: rows = platforms, cols = [Feature-wise, Sample-wise]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Variance vs Mean (Feature-wise and Sample-wise)', fontsize=16, fontweight='bold')

        # Colors for up to 4 methods
        method_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'],
                         NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        truth_color = 'black'

        # Helper to compute and plot mean-variance
        def plot_mv(ax, matrices, labels, title):
            eps = 1e-8
            for idx, (mat, label) in enumerate(zip(matrices, labels)):
                if mat is None:
                    continue
                # Flatten per-entity (rows for features, columns for samples handled by caller)
                means = np.nanmean(mat, axis=1)
                vars_ = np.nanvar(mat, axis=1)
                color = truth_color if label == 'Truth' else method_colors[(idx-1) % len(method_colors)] if idx > 0 else method_colors[0]
                ax.scatter(np.log10(means + eps), np.log10(vars_ + eps), s=6, alpha=0.35, c=color, edgecolors='none', rasterized=True, label=label)
            ax.set_xlabel('log10(Mean)')
            ax.set_ylabel('log10(Variance)')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Platform A - Only include Method 1 and Method 2
        platform_a_methods = []
        if data.imp_a_m1 is not None:
            platform_a_methods.append((data.imp_a_m1, data.method1_name))
        if data.imp_a_m2 is not None:
            platform_a_methods.append((data.imp_a_m2, data.method2_name))

        # Feature-wise (rows = features)
        a_feat_matrices = [data.truth_a.values] + [m.values for m, _ in platform_a_methods]
        a_feat_labels = ['Truth'] + [name for _, name in platform_a_methods]
        plot_mv(axes[0, 0], a_feat_matrices, a_feat_labels, f'{data.platform_a_name} - Feature-wise')

        # Sample-wise (rows = samples) -> transpose
        a_samp_matrices = [data.truth_a.values.T] + [m.values.T for m, _ in platform_a_methods]
        a_samp_labels = ['Truth'] + [name for _, name in platform_a_methods]
        plot_mv(axes[0, 1], a_samp_matrices, a_samp_labels, f'{data.platform_a_name} - Sample-wise')

        # Platform B - Only include Method 1 and Method 2
        platform_b_methods = []
        if data.imp_b_m1 is not None:
            platform_b_methods.append((data.imp_b_m1, data.method1_name))
        if data.imp_b_m2 is not None:
            platform_b_methods.append((data.imp_b_m2, data.method2_name))

        if len(platform_b_methods) or data.truth_b is not None:
            b_feat_matrices = [data.truth_b.values] + [m.values for m, _ in platform_b_methods]
            b_feat_labels = ['Truth'] + [name for _, name in platform_b_methods]
            plot_mv(axes[1, 0], b_feat_matrices, b_feat_labels, f'{data.platform_b_name} - Feature-wise')

            b_samp_matrices = [data.truth_b.values.T] + [m.values.T for m, _ in platform_b_methods]
            b_samp_labels = ['Truth'] + [name for _, name in platform_b_methods]
            plot_mv(axes[1, 1], b_samp_matrices, b_samp_labels, f'{data.platform_b_name} - Sample-wise')
        else:
            # If Platform B missing, hide bottom row
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)

        plt.tight_layout()
        return fig
    def generate_figure_5_error_ecdfs(self, data: AnalysisData):
        """Figure 5: Absolute-error empirical CDFs"""
        print("Generating Figure 5: Error ECDFs...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Error Distribution Analysis: Empirical CDFs', fontsize=14, fontweight='bold')
        
        # Calculate absolute errors for all methods
        errors = {}
        comparisons = [
            ('Method_1_A', data.truth_a, data.imp_a_m1),
            ('Method_2_A', data.truth_a, data.imp_a_m2),
            ('Method_1_B', data.truth_b, data.imp_b_m1),
            ('Method_2_B', data.truth_b, data.imp_b_m2),
        ]
        
        for name, truth, imputed in comparisons:
            abs_errors = np.abs(truth.values - imputed.values).flatten()
            abs_errors = abs_errors[~np.isnan(abs_errors)]
            # Filter out exact zeros for log scale plotting
            abs_errors = abs_errors[abs_errors > 0]
            errors[name] = abs_errors
        
        # Plot ECDFs for Platform A
        for method, color in [('Method_1_A', NATURE_COLORS['primary']), 
                             ('Method_2_A', NATURE_COLORS['secondary'])]:
            if method in errors:
                sorted_errors = np.sort(errors[method])
                y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                method_name = data.method1_name if '1' in method else data.method2_name
                ax1.plot(sorted_errors, y_vals, color=color, linewidth=2, 
                        label=method_name)
        
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title(f'{data.platform_a_name} - Error ECDF')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot ECDFs for Platform B
        for method, color in [('Method_1_B', NATURE_COLORS['primary']), 
                             ('Method_2_B', NATURE_COLORS['secondary'])]:
            if method in errors:
                sorted_errors = np.sort(errors[method])
                y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                method_name = data.method1_name if '1' in method else data.method2_name
                ax2.plot(sorted_errors, y_vals, color=color, linewidth=2, 
                        label=method_name)
        
        ax2.set_xlabel('Absolute Error')  
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'{data.platform_b_name} - Error ECDF')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_6_sample_error_heatmap(self, data: AnalysisData):
        """Figure 6: Heatmap of sample-wise and feature-wise error landscape"""
        print("Generating Figure 6: Sample-wise and feature-wise error heatmap...")
        
        # Create 2x2 grid: sample-wise (top row) and feature-wise (bottom row)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Error Landscape Analysis (MAE)', fontsize=16, fontweight='bold')
        
        # Platform A - get all available methods
        platform_a_comparisons = [
            (f'{data.method1_name}', data.truth_a, data.imp_a_m1),
            (f'{data.method2_name}', data.truth_a, data.imp_a_m2),
        ]
        if data.imp_a_m3 is not None and data.method3_name is not None:
            platform_a_comparisons.append((f'{data.method3_name}', data.truth_a, data.imp_a_m3))
        if data.imp_a_m4 is not None and data.method4_name is not None:
            platform_a_comparisons.append((f'{data.method4_name}', data.truth_a, data.imp_a_m4))
        
        platform_a_matrix = []
        platform_a_columns = []
        
        for name, truth, imputed in platform_a_comparisons:
            # Calculate MAE per sample
            sample_mae = []
            for sample in truth.columns:
                truth_vals = truth[sample].values
                imp_vals = imputed[sample].values
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(truth_vals[mask] - imp_vals[mask]))
                else:
                    mae = np.nan
                sample_mae.append(mae)
            platform_a_matrix.append(sample_mae)
            platform_a_columns.append(name)
        
        if len(platform_a_matrix) > 0:
            platform_a_df = pd.DataFrame(np.array(platform_a_matrix).T, 
                                        index=data.truth_a.columns, 
                                        columns=platform_a_columns)
            
            # Hierarchical clustering on samples (rows) for Platform A
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            
            error_for_clustering = platform_a_df.dropna()
            if len(error_for_clustering) > 3:
                try:
                    linkage_matrix = linkage(pdist(error_for_clustering), method='ward')
                    dendro = dendrogram(linkage_matrix, labels=error_for_clustering.index, no_plot=True)
                    sample_order = dendro['ivl']
                    platform_a_df_ordered = platform_a_df.loc[sample_order]
                except Exception:
                    platform_a_df_ordered = platform_a_df
            else:
                platform_a_df_ordered = platform_a_df
            
            # Create heatmap for Platform A
            # Apply log transformation for better visibility
            platform_a_df_logged = platform_a_df_ordered.copy()
            platform_a_df_logged = np.log10(platform_a_df_logged + 1e-6)  # Add small constant to avoid log(0)
            
            im1 = ax1.imshow(platform_a_df_logged.values, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax1.set_xticks(range(len(platform_a_columns)))
            ax1.set_xticklabels(platform_a_columns, rotation=45, ha='right')
            ax1.set_yticks(range(len(platform_a_df_ordered)))
            ax1.set_yticklabels(platform_a_df_ordered.index, fontsize=6)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('log10(Mean Absolute Error)', rotation=270, labelpad=15)
            
            ax1.set_title(f'{data.platform_a_name} - Sample-wise', fontweight='bold')
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Samples (Hierarchically Clustered)')
        else:
            ax1.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title(f'{data.platform_a_name} - Sample-wise')
        
        # Platform B - get all available methods
        platform_b_comparisons = [
            (f'{data.method1_name}', data.truth_b, data.imp_b_m1),
            (f'{data.method2_name}', data.truth_b, data.imp_b_m2),
        ]
        if data.imp_b_m3 is not None and data.method3_name is not None:
            platform_b_comparisons.append((f'{data.method3_name}', data.truth_b, data.imp_b_m3))
        if data.imp_b_m4 is not None and data.method4_name is not None:
            platform_b_comparisons.append((f'{data.method4_name}', data.truth_b, data.imp_b_m4))
        
        platform_b_matrix = []
        platform_b_columns = []
        
        for name, truth, imputed in platform_b_comparisons:
            # Calculate MAE per sample
            sample_mae = []
            for sample in truth.columns:
                truth_vals = truth[sample].values
                imp_vals = imputed[sample].values
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(truth_vals[mask] - imp_vals[mask]))
                else:
                    mae = np.nan
                sample_mae.append(mae)
            platform_b_matrix.append(sample_mae)
            platform_b_columns.append(name)
        
        if len(platform_b_matrix) > 0:
            platform_b_df = pd.DataFrame(np.array(platform_b_matrix).T, 
                                        index=data.truth_b.columns, 
                                        columns=platform_b_columns)
            
            # Hierarchical clustering on samples (rows) for Platform B
            error_for_clustering = platform_b_df.dropna()
            if len(error_for_clustering) > 3:
                try:
                    linkage_matrix = linkage(pdist(error_for_clustering), method='ward')
                    dendro = dendrogram(linkage_matrix, labels=error_for_clustering.index, no_plot=True)
                    sample_order = dendro['ivl']
                    platform_b_df_ordered = platform_b_df.loc[sample_order]
                except Exception:
                    platform_b_df_ordered = platform_b_df
            else:
                platform_b_df_ordered = platform_b_df
            
            # Create heatmap for Platform B
            # Apply log transformation for better visibility
            platform_b_df_logged = platform_b_df_ordered.copy()
            platform_b_df_logged = np.log10(platform_b_df_logged + 1e-6)  # Add small constant to avoid log(0)
            
            im2 = ax2.imshow(platform_b_df_logged.values, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax2.set_xticks(range(len(platform_b_columns)))
            ax2.set_xticklabels(platform_b_columns, rotation=45, ha='right')
            ax2.set_yticks(range(len(platform_b_df_ordered)))
            ax2.set_yticklabels(platform_b_df_ordered.index, fontsize=6)
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('log10(Mean Absolute Error)', rotation=270, labelpad=15)
            
            ax2.set_title(f'{data.platform_b_name} - Sample-wise', fontweight='bold')
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Samples (Hierarchically Clustered)')
        else:
            ax2.text(0.5, 0.5, f'No data for {data.platform_b_name}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'{data.platform_b_name} - Sample-wise')
        
        # Now add feature-wise error maps (bottom row)
        # Platform A - Feature-wise error map
        platform_a_feature_matrix = []
        
        for name, truth, imputed in platform_a_comparisons:
            # Calculate MAE per feature
            feature_mae = []
            for feature in truth.index:
                truth_vals = truth.loc[feature].values
                imp_vals = imputed.loc[feature].values
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(truth_vals[mask] - imp_vals[mask]))
                else:
                    mae = np.nan
                feature_mae.append(mae)
            platform_a_feature_matrix.append(feature_mae)
        
        if len(platform_a_feature_matrix) > 0:
            platform_a_feature_df = pd.DataFrame(np.array(platform_a_feature_matrix).T, 
                                                index=data.truth_a.index, 
                                                columns=platform_a_columns)
            
            # Hierarchical clustering on features for Platform A
            feature_error_for_clustering = platform_a_feature_df.dropna()
            if len(feature_error_for_clustering) > 3:
                try:
                    linkage_matrix = linkage(pdist(feature_error_for_clustering), method='ward')
                    dendro = dendrogram(linkage_matrix, labels=feature_error_for_clustering.index, no_plot=True)
                    feature_order = dendro['ivl']
                    platform_a_feature_df_ordered = platform_a_feature_df.loc[feature_order]
                except Exception:
                    platform_a_feature_df_ordered = platform_a_feature_df
            else:
                platform_a_feature_df_ordered = platform_a_feature_df
            
            # Create heatmap for Platform A features
            platform_a_feature_df_logged = platform_a_feature_df_ordered.copy()
            platform_a_feature_df_logged = np.log10(platform_a_feature_df_logged + 1e-6)
            
            im3 = ax3.imshow(platform_a_feature_df_logged.values, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels (limit to manageable number for readability)
            n_features = len(platform_a_feature_df_ordered)
            if n_features > 50:
                # Show every nth feature label to avoid overcrowding
                step = max(1, n_features // 20)
                ytick_positions = range(0, n_features, step)
                ytick_labels = [platform_a_feature_df_ordered.index[i][:15] + ('...' if len(platform_a_feature_df_ordered.index[i]) > 15 else '') for i in ytick_positions]
            else:
                ytick_positions = range(n_features)
                ytick_labels = [name[:15] + ('...' if len(name) > 15 else '') for name in platform_a_feature_df_ordered.index]
            
            ax3.set_xticks(range(len(platform_a_columns)))
            ax3.set_xticklabels(platform_a_columns, rotation=45, ha='right')
            ax3.set_yticks(ytick_positions)
            ax3.set_yticklabels(ytick_labels, fontsize=6)
            
            # Add colorbar
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('log10(Mean Absolute Error)', rotation=270, labelpad=15)
            
            ax3.set_title(f'{data.platform_a_name} - Feature-wise', fontweight='bold')
            ax3.set_xlabel('Method')
            ax3.set_ylabel('Features (Hierarchically Clustered)')
        else:
            ax3.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'{data.platform_a_name} - Feature-wise')
        
        # Platform B - Feature-wise error map
        platform_b_feature_matrix = []
        
        for name, truth, imputed in platform_b_comparisons:
            # Calculate MAE per feature
            feature_mae = []
            for feature in truth.index:
                truth_vals = truth.loc[feature].values
                imp_vals = imputed.loc[feature].values
                mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(truth_vals[mask] - imp_vals[mask]))
                else:
                    mae = np.nan
                feature_mae.append(mae)
            platform_b_feature_matrix.append(feature_mae)
        
        if len(platform_b_feature_matrix) > 0:
            platform_b_feature_df = pd.DataFrame(np.array(platform_b_feature_matrix).T, 
                                                index=data.truth_b.index, 
                                                columns=platform_b_columns)
            
            # Hierarchical clustering on features for Platform B
            feature_error_for_clustering = platform_b_feature_df.dropna()
            if len(feature_error_for_clustering) > 3:
                try:
                    linkage_matrix = linkage(pdist(feature_error_for_clustering), method='ward')
                    dendro = dendrogram(linkage_matrix, labels=feature_error_for_clustering.index, no_plot=True)
                    feature_order = dendro['ivl']
                    platform_b_feature_df_ordered = platform_b_feature_df.loc[feature_order]
                except Exception:
                    platform_b_feature_df_ordered = platform_b_feature_df
            else:
                platform_b_feature_df_ordered = platform_b_feature_df
            
            # Create heatmap for Platform B features
            platform_b_feature_df_logged = platform_b_feature_df_ordered.copy()
            platform_b_feature_df_logged = np.log10(platform_b_feature_df_logged + 1e-6)
            
            im4 = ax4.imshow(platform_b_feature_df_logged.values, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels (limit to manageable number for readability)
            n_features = len(platform_b_feature_df_ordered)
            if n_features > 50:
                # Show every nth feature label to avoid overcrowding
                step = max(1, n_features // 20)
                ytick_positions = range(0, n_features, step)
                ytick_labels = [platform_b_feature_df_ordered.index[i][:15] + ('...' if len(platform_b_feature_df_ordered.index[i]) > 15 else '') for i in ytick_positions]
            else:
                ytick_positions = range(n_features)
                ytick_labels = [name[:15] + ('...' if len(name) > 15 else '') for name in platform_b_feature_df_ordered.index]
            
            ax4.set_xticks(range(len(platform_b_columns)))
            ax4.set_xticklabels(platform_b_columns, rotation=45, ha='right')
            ax4.set_yticks(ytick_positions)
            ax4.set_yticklabels(ytick_labels, fontsize=6)
            
            # Add colorbar
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('log10(Mean Absolute Error)', rotation=270, labelpad=15)
            
            ax4.set_title(f'{data.platform_b_name} - Feature-wise', fontweight='bold')
            ax4.set_xlabel('Method')
            ax4.set_ylabel('Features (Hierarchically Clustered)')
        else:
            ax4.text(0.5, 0.5, f'No data for {data.platform_b_name}', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title(f'{data.platform_b_name} - Feature-wise')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_7_hexbin_error_abundance(self, data: AnalysisData):
        """Figure 7: Hexbin density plots showing error vs truth abundance"""
        print("Generating Figure 7: Hexbin error vs abundance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Error vs Truth Abundance Relationship', fontsize=14, fontweight='bold')
        
        comparisons = [
            (data.method1_name + f' - {data.platform_a_name}', data.truth_a, data.imp_a_m1, 0),
            (data.method2_name + f' - {data.platform_a_name}', data.truth_a, data.imp_a_m2, 1),
            (data.method1_name + f' - {data.platform_b_name}', data.truth_b, data.imp_b_m1, 2),
            (data.method2_name + f' - {data.platform_b_name}', data.truth_b, data.imp_b_m2, 3),
        ]
        
        # First pass: collect data ranges for each platform to determine consistent axis limits
        platform_a_log_truth_range = [float('inf'), float('-inf')]
        platform_a_error_range = [float('inf'), float('-inf')]
        platform_b_log_truth_range = [float('inf'), float('-inf')]
        platform_b_error_range = [float('inf'), float('-inf')]
        
        # Collect all data points and calculate ranges
        for title, truth, imputed, idx in comparisons:
            # Flatten and clean data
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat)) & (truth_flat > 0)
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            # Calculate log10 truth intensity and absolute error
            log_truth = np.log10(truth_clean)
            abs_error = np.abs(imp_clean - truth_clean)
            
            # Update ranges based on platform
            if idx < 2:  # Platform A
                platform_a_log_truth_range[0] = min(platform_a_log_truth_range[0], np.min(log_truth)) if len(log_truth) > 0 else platform_a_log_truth_range[0]
                platform_a_log_truth_range[1] = max(platform_a_log_truth_range[1], np.max(log_truth)) if len(log_truth) > 0 else platform_a_log_truth_range[1]
                platform_a_error_range[0] = min(platform_a_error_range[0], np.min(abs_error)) if len(abs_error) > 0 else platform_a_error_range[0]
                platform_a_error_range[1] = max(platform_a_error_range[1], np.max(abs_error)) if len(abs_error) > 0 else platform_a_error_range[1]
            else:  # Platform B
                platform_b_log_truth_range[0] = min(platform_b_log_truth_range[0], np.min(log_truth)) if len(log_truth) > 0 else platform_b_log_truth_range[0]
                platform_b_log_truth_range[1] = max(platform_b_log_truth_range[1], np.max(log_truth)) if len(log_truth) > 0 else platform_b_log_truth_range[1]
                platform_b_error_range[0] = min(platform_b_error_range[0], np.min(abs_error)) if len(abs_error) > 0 else platform_b_error_range[0]
                platform_b_error_range[1] = max(platform_b_error_range[1], np.max(abs_error)) if len(abs_error) > 0 else platform_b_error_range[1]
        
        # Handle edge cases where there might not be enough data
        if platform_a_log_truth_range[0] > platform_a_log_truth_range[1]:
            platform_a_log_truth_range = [0, 1]
        if platform_a_error_range[0] > platform_a_error_range[1]:
            platform_a_error_range = [0, 1]
        if platform_b_log_truth_range[0] > platform_b_log_truth_range[1]:
            platform_b_log_truth_range = [0, 1]
        if platform_b_error_range[0] > platform_b_error_range[1]:
            platform_b_error_range = [0, 1]
        
        # Second pass: create the plots with consistent scales
        axes_flat = axes.flatten()
        
        for i, (title, truth, imputed, _) in enumerate(comparisons):
            ax = axes_flat[i]
            
            # Flatten and clean data
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat)) & (truth_flat > 0)
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            # Calculate log10 truth intensity and absolute error
            log_truth = np.log10(truth_clean)
            abs_error = np.abs(imp_clean - truth_clean)
            
            # Create hexbin plot
            hb = ax.hexbin(log_truth, abs_error, gridsize=30, cmap='Blues', 
                          mincnt=1, alpha=0.8)
            
            # Add trend line
            from scipy.stats import binned_statistic
            bin_means, bin_edges, _ = binned_statistic(log_truth, abs_error, 
                                                     statistic='median', bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            valid_bins = ~np.isnan(bin_means)
            valid_centers = bin_centers[valid_bins]
            valid_means = bin_means[valid_bins]
            
            # Plot trend line
            ax.plot(valid_centers, valid_means, 'r-', linewidth=2, alpha=0.8, label='Median Error')
            
            # Calculate AUC of the trend line using trapezoidal rule, but only after median x-value
            if len(valid_centers) > 1:
                # Sort points by x-value to ensure proper integration
                sorted_indices = np.argsort(valid_centers)
                x_sorted = valid_centers[sorted_indices]
                y_sorted = valid_means[sorted_indices]
                
                # Find the median x-value
                median_x = np.median(log_truth)
                
                # Add vertical line at median x
                ax.axvline(x=median_x, color='green', linestyle='--', alpha=0.7)
                
                # Get points after the median x-value
                after_median_mask = x_sorted >= median_x
                x_after_median = x_sorted[after_median_mask]
                y_after_median = y_sorted[after_median_mask]
                
                # Calculate AUC using trapezoidal rule for points after median
                if len(x_after_median) > 1:  # Need at least 2 points for integration
                    auc_value = np.trapz(y=y_after_median, x=x_after_median)
                    
                    # Calculate normalized AUC (divide by x-range)
                    x_range = x_after_median[-1] - x_after_median[0]
                    if x_range > 0:
                        normalized_auc = auc_value / x_range
                        
                        # Add the AUC value to the legend
                        ax.plot([], [], ' ', label=f'Right-AUC = {normalized_auc:.4f}')
                        
                        # Add label for median line
                        ax.plot([], [], color='green', linestyle='--', label=f'Median log10 (Int.) = {median_x:.2f}')
            
            # Set consistent axis limits based on platform
            if i < 2:  # Platform A
                ax.set_xlim(platform_a_log_truth_range)
                ax.set_ylim(platform_a_error_range)
            else:  # Platform B
                ax.set_xlim(platform_b_log_truth_range)
                ax.set_ylim(platform_b_error_range)
            
            ax.set_xlabel('log10 (Truth Intensity)')
            ax.set_ylabel('Absolute Error')
            ax.set_title(title)
            ax.legend(loc='upper left')  # Changed from default to upper left
            ax.grid(True, alpha=0.3)
            
            # Add text explaining AUC calculation
            ax.text(0.02, 0.98, "Right-AUC: Area under curve after median intensity", 
                   transform=ax.transAxes, fontsize=8, va='top', alpha=0.7)
            
            # Add colorbar
            plt.colorbar(hb, ax=ax, label='Count')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_8_global_correlation_heatmap(self, data: AnalysisData):
        """Figure 8: Global correlation heatmap of all datasets"""
        print("Generating Figure 8: Global correlation heatmap...")
        
        # Prepare data matrices in specified order - include all available methods
        datasets = {
            'Truth_A': data.truth_a,
            f'{data.method1_name}_A': data.imp_a_m1,
            f'{data.method2_name}_A': data.imp_a_m2,
        }
        
        # Add optional methods for platform A
        if data.imp_a_m3 is not None and data.method3_name is not None:
            datasets[f'{data.method3_name}_A'] = data.imp_a_m3
        if data.imp_a_m4 is not None and data.method4_name is not None:
            datasets[f'{data.method4_name}_A'] = data.imp_a_m4
        
        # Add platform B datasets
        datasets['Truth_B'] = data.truth_b
        datasets[f'{data.method1_name}_B'] = data.imp_b_m1
        datasets[f'{data.method2_name}_B'] = data.imp_b_m2
        
        # Add optional methods for platform B
        if data.imp_b_m3 is not None and data.method3_name is not None:
            datasets[f'{data.method3_name}_B'] = data.imp_b_m3
        if data.imp_b_m4 is not None and data.method4_name is not None:
            datasets[f'{data.method4_name}_B'] = data.imp_b_m4
        
        # Calculate correlations between datasets using consistent global correlation
        # Use flattened vectors for all comparisons for consistency
        n_datasets = len(datasets)
        corr_matrix = np.full((n_datasets, n_datasets), np.nan)
        dataset_names = list(datasets.keys())
        
        for i, (name1, data1) in enumerate(datasets.items()):
            for j, (name2, data2) in enumerate(datasets.items()):
                if i <= j:  # Only compute upper triangle
                    # Determine which features to use for comparison
                    if ('_A' in name1 and '_B' in name2) or ('_B' in name1 and '_A' in name2):
                        # Cross-platform: use overlapping features only
                        common_features = list(set(data1.index) & set(data2.index))
                        print(f"    Cross-platform {name1} vs {name2}: {len(common_features)} overlapping features")
                        
                        if len(common_features) >= 5:
                            data1_subset = data1.loc[common_features]
                            data2_subset = data2.loc[common_features]
                        else:
                            # Too few overlapping features for meaningful correlation
                            corr_matrix[i, j] = np.nan
                            corr_matrix[j, i] = np.nan
                            continue
                    else:
                        # Same platform: use all features in common
                        common_features = list(set(data1.index) & set(data2.index))
                        data1_subset = data1.loc[common_features]
                        data2_subset = data2.loc[common_features]
                    
                    # Flatten data and remove NaNs
                    vals1 = data1_subset.values.flatten()
                    vals2 = data2_subset.values.flatten()
                    mask = ~(np.isnan(vals1) | np.isnan(vals2))
                    
                    if np.sum(mask) > 10:
                        corr, _ = pearsonr(vals1[mask], vals2[mask])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                    else:
                        # Insufficient valid data points - use NaN instead of fabricated zero
                        corr_matrix[i, j] = np.nan
                        corr_matrix[j, i] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use a special value for missing correlations (outside normal range)
        # This will be colored differently from actual 0 correlations
        corr_matrix_viz = corr_matrix.copy()
        missing_mask = np.isnan(corr_matrix)
        corr_matrix_viz[missing_mask] = -999  # Special value for missing
        
        # Create custom colormap that handles missing values
        import matplotlib.colors as mcolors
        from matplotlib.colors import ListedColormap
        
        # Use standard RdBu_r for -1 to 1, and gray for missing
        cmap = plt.cm.RdBu_r
        cmap.set_bad(color='lightgray')  # Color for NaN/missing values
        
        # Use the original matrix with NaNs so missing values show as gray
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(range(n_datasets))
        ax.set_yticks(range(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(dataset_names)
        
        # Add correlation values as text
        for i in range(n_datasets):
            for j in range(n_datasets):
                if not np.isnan(corr_matrix[i, j]):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
                else:
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="gray", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        ax.set_title('Global Correlation Matrix', fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_9_umap_concordance(self, data: AnalysisData):
        """Figure 9: PCA and UMAP analysis - structure preservation assessment"""
        print("Generating Figure 9: Structure preservation analysis (PCA/UMAP)...")
        
        # Fit dimensionality reduction models on truth data, then transform imputed data
        # This directly assesses how well imputation preserves the original sample structure
        
        try:
            # Prepare datasets for UMAP (samples as observations)
            sample_data_list = []
            labels = []
            platforms = []
            data_types = []
            
            # Platform A datasets (exclude Method 4 as it's a permuted baseline)
            platform_a_datasets = {
                'Truth_A': data.truth_a,
                f'{data.method1_name}_A': data.imp_a_m1,
                f'{data.method2_name}_A': data.imp_a_m2,
            }
            if data.imp_a_m3 is not None and data.method3_name is not None:
                platform_a_datasets[f'{data.method3_name}_A'] = data.imp_a_m3
            # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            
            # Platform B datasets (exclude Method 4 as it's a permuted baseline)
            platform_b_datasets = {
                'Truth_B': data.truth_b,
                f'{data.method1_name}_B': data.imp_b_m1,
                f'{data.method2_name}_B': data.imp_b_m2,
            }
            if data.imp_b_m3 is not None and data.method3_name is not None:
                platform_b_datasets[f'{data.method3_name}_B'] = data.imp_b_m3
            # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            
            # Process each platform separately
            for platform_name, datasets_dict in [('Platform_A', platform_a_datasets), 
                                                 ('Platform_B', platform_b_datasets)]:
                
                if len(datasets_dict) > 0:
                    # Get the first dataset to determine sample names
                    first_dataset = list(datasets_dict.values())[0]
                    n_samples = first_dataset.shape[1]
                    n_features = first_dataset.shape[0]
                    
                    print(f"    Processing {platform_name}: {n_samples} samples × {n_features} features")
                    
                    for name, dataset in datasets_dict.items():
                        # Transpose to get samples × features
                        sample_data = dataset.T.values  # samples × features
                        
                        # Handle NaN values by replacing with feature means
                        col_means = np.nanmean(sample_data, axis=0)
                        inds = np.where(np.isnan(sample_data))
                        sample_data[inds] = np.take(col_means, inds[1])
                        
                        # Only use a subset of features to make UMAP faster and handle high dimensionality
                        if sample_data.shape[1] > 500:
                            # Select features with highest variance
                            feature_vars = np.var(sample_data, axis=0)
                            top_features = np.argsort(feature_vars)[-500:]
                            sample_data = sample_data[:, top_features]
                        
                        sample_data_list.append(sample_data)
                        
                        # Create labels
                        n_samples_in_dataset = sample_data.shape[0]
                        labels.extend([name] * n_samples_in_dataset)
                        platforms.extend([platform_name] * n_samples_in_dataset)
                        
                        if 'Truth' in name:
                            data_types.extend(['Truth'] * n_samples_in_dataset)
                        else:
                            data_types.extend(['Imputed'] * n_samples_in_dataset)
            
            if len(sample_data_list) == 0:
                print("    No valid data for dimensionality reduction analysis")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.text(0.5, 0.5, 'No valid data for PCA/UMAP analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Dimensionality Reduction Concordance')
                return fig
            
            # Create separate plots for each platform with both PCA and UMAP
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle('Structure Preservation Assessment: PCA and UMAP\n(Models fitted on truth data, imputed data transformed)', 
                        fontsize=14, fontweight='bold')
            
            # Create platform name mapping
            platform_name_map = {
                'Platform_A': data.platform_a_name,
                'Platform_B': data.platform_b_name
            }
            
            platform_colors = {'Platform_A': NATURE_COLORS['primary'], 
                              'Platform_B': NATURE_COLORS['secondary']}
            type_markers = {'Truth': 'o', 'Imputed': '^'}
            
            for platform_idx, (platform_key, datasets_dict) in enumerate([('Platform_A', platform_a_datasets), 
                                                                          ('Platform_B', platform_b_datasets)]):
                
                # Get data for this platform only
                platform_sample_data = []
                platform_labels = []
                platform_types = []
                
                for name, dataset in datasets_dict.items():
                    sample_data = dataset.T.values
                    
                    # Handle NaN values
                    col_means = np.nanmean(sample_data, axis=0)
                    inds = np.where(np.isnan(sample_data))
                    sample_data[inds] = np.take(col_means, inds[1])
                    
                    # Dimensionality reduction for high-dimensional data
                    if sample_data.shape[1] > 500:
                        feature_vars = np.var(sample_data, axis=0)
                        top_features = np.argsort(feature_vars)[-500:]
                        sample_data = sample_data[:, top_features]
                    
                    platform_sample_data.append(sample_data)
                    platform_labels.extend([name] * sample_data.shape[0])
                    platform_types.extend(['Truth' if 'Truth' in name else 'Imputed'] * sample_data.shape[0])
                
                if len(platform_sample_data) > 0:
                    # Get truth data for this platform (first dataset in the list)
                    truth_data = platform_sample_data[0]  # Truth is always first
                    
                    # Scale features using truth data
                    scaler = StandardScaler()
                    scaled_truth = scaler.fit_transform(truth_data)
                    
                    # Fit PCA on truth data only
                    pca = PCA(n_components=2, random_state=42)
                    pca_embedding_truth = pca.fit_transform(scaled_truth)
                    
                    # Fit UMAP on truth data only - enable transform for out-of-sample data
                    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_neighbors=15, transform_seed=42)
                    umap_embedding_truth = reducer.fit_transform(scaled_truth)
                    
                    # Transform all datasets (including truth) using the fitted models
                    pca_embeddings = []
                    umap_embeddings = []
                    
                    for data_matrix in platform_sample_data:
                        # Scale using the same scaler fitted on truth
                        scaled_data = scaler.transform(data_matrix)
                        
                        # Transform using fitted PCA and UMAP
                        pca_transformed = pca.transform(scaled_data)
                        umap_transformed = reducer.transform(scaled_data)
                        
                        pca_embeddings.append(pca_transformed)
                        umap_embeddings.append(umap_transformed)
                    
                    # Combine all embeddings for plotting
                    pca_embedding = np.vstack(pca_embeddings)
                    umap_embedding = np.vstack(umap_embeddings)
                    
                    # Plot setup
                    unique_labels = list(set(platform_labels))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                    
                    # PCA plot (top row)
                    ax_pca = axes[0, platform_idx]
                    plotted_labels = set()
                    for i, (label, data_type) in enumerate(zip(platform_labels, platform_types)):
                        label_idx = unique_labels.index(label)
                        marker = type_markers[data_type]
                        
                        ax_pca.scatter(pca_embedding[i, 0], pca_embedding[i, 1],
                                     c=[colors[label_idx]], s=30, alpha=0.7, 
                                     marker=marker, label=f'{label}' if label not in plotted_labels else "")
                        plotted_labels.add(label)
                    
                    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    ax_pca.set_title(f'PCA - {platform_name_map[platform_key]}')
                    ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax_pca.grid(True, alpha=0.3)
                    
                    # UMAP plot (bottom row)
                    ax_umap = axes[1, platform_idx]
                    plotted_labels_umap = set()
                    for i, (label, data_type) in enumerate(zip(platform_labels, platform_types)):
                        label_idx = unique_labels.index(label)
                        marker = type_markers[data_type]
                        
                        ax_umap.scatter(umap_embedding[i, 0], umap_embedding[i, 1],
                                      c=[colors[label_idx]], s=30, alpha=0.7, 
                                      marker=marker, label=f'{label}' if label not in plotted_labels_umap else "")
                        plotted_labels_umap.add(label)
                    
                    ax_umap.set_xlabel('UMAP 1')
                    ax_umap.set_ylabel('UMAP 2')
                    ax_umap.set_title(f'UMAP - {platform_name_map[platform_key]}')
                    ax_umap.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax_umap.grid(True, alpha=0.3)
                else:
                    # No data for this platform
                    ax_pca = axes[0, platform_idx]
                    ax_umap = axes[1, platform_idx]
                    
                    ax_pca.text(0.5, 0.5, f'No data for {platform_name_map[platform_key]}', 
                              ha='center', va='center', transform=ax_pca.transAxes, fontsize=12)
                    ax_pca.set_title(f'PCA - {platform_name_map[platform_key]}')
                    
                    ax_umap.text(0.5, 0.5, f'No data for {platform_name_map[platform_key]}', 
                               ha='center', va='center', transform=ax_umap.transAxes, fontsize=12)
                    ax_umap.set_title(f'UMAP - {platform_name_map[platform_key]}')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in dimensionality reduction analysis: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in PCA/UMAP analysis:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Dimensionality Reduction Concordance')
            return fig
    
    def generate_figure_9b_feature_level_umap_pca(self, data: AnalysisData):
        """Figure 9b: Feature-level PCA and UMAP analysis"""
        print("  Generating Figure 9b: Feature-level PCA and UMAP analysis...")
        
        # Calculate the number of methods (2-3, excluding Method 4 as it's a permuted baseline)
        n_methods = 2
        if data.imp_a_m3 is not None:
            n_methods = 3
        # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
        
        # Create figure with 2 columns (PCA, UMAP) and n_methods*2 rows (one for each platform)
        fig, axes = plt.subplots(n_methods * 2, 2, figsize=(10, 5 * n_methods * 2))
        if n_methods * 2 == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Feature-Level Structure Analysis: PCA and UMAP', fontsize=14, fontweight='bold')
        
        # Process each platform
        row_idx = 0
        for platform_idx, (truth_data, platform_name) in enumerate([(data.truth_a, data.platform_a_name), 
                                                                     (data.truth_b, data.platform_b_name)]):
            # Get method data for this platform (exclude Method 4 - it's a permuted baseline)
            method_data_list = []
            method_names = []
            
            if platform_idx == 0:  # Platform A
                method_data_list = [
                    (data.imp_a_m1, data.method1_name),
                    (data.imp_a_m2, data.method2_name)
                ]
                if data.imp_a_m3 is not None:
                    method_data_list.append((data.imp_a_m3, data.method3_name))
                # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            else:  # Platform B
                method_data_list = [
                    (data.imp_b_m1, data.method1_name),
                    (data.imp_b_m2, data.method2_name)
                ]
                if data.imp_b_m3 is not None:
                    method_data_list.append((data.imp_b_m3, data.method3_name))
                # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            
            # Process each method for this platform
            for method_idx, (method_data, method_name) in enumerate(method_data_list):
                # Transpose to get features as rows
                truth_features = truth_data.values
                method_features = method_data.values
                
                # Check if we need to reduce dimensions first
                if truth_features.shape[1] > 50:
                    # First reduce to 50 dimensions with PCA
                    pca_reducer = PCA(n_components=min(50, truth_features.shape[1] - 1))
                    truth_features = pca_reducer.fit_transform(truth_features)
                    method_features = pca_reducer.transform(method_features)
                
                # PCA (2D)
                pca = PCA(n_components=2)
                truth_pca = pca.fit_transform(truth_features)
                method_pca = pca.transform(method_features)
                
                # UMAP
                umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, truth_features.shape[0] - 1))
                truth_umap = umap_reducer.fit_transform(truth_features)
                method_umap = umap_reducer.transform(method_features)
                
                # Determine colors
                if data.groups is not None and len(data.groups) == len(truth_data.index):
                    # Use group colors
                    unique_groups = data.groups.unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
                    group_to_color = dict(zip(unique_groups, colors))
                    feature_colors = [group_to_color[g] for g in data.groups]
                else:
                    # Use single color
                    feature_colors = [NATURE_COLORS['primary']] * len(truth_data.index)
                
                # Plot PCA
                ax_pca = axes[row_idx, 0]
                ax_pca.scatter(truth_pca[:, 0], truth_pca[:, 1], c=feature_colors, alpha=0.6, s=60, 
                              marker='o', label='Truth', edgecolors='black', linewidth=0.5)
                ax_pca.scatter(method_pca[:, 0], method_pca[:, 1], c=feature_colors, alpha=0.6, s=40, 
                              marker='^', label=method_name, edgecolors='gray', linewidth=0.5)
                
                ax_pca.set_xlabel('PC1')
                ax_pca.set_ylabel('PC2')
                ax_pca.set_title(f'{platform_name} - {method_name} (PCA)')
                ax_pca.legend(loc='upper right')
                ax_pca.grid(True, alpha=0.3)
                
                # Plot UMAP
                ax_umap = axes[row_idx, 1]
                ax_umap.scatter(truth_umap[:, 0], truth_umap[:, 1], c=feature_colors, alpha=0.6, s=60, 
                               marker='o', label='Truth', edgecolors='black', linewidth=0.5)
                ax_umap.scatter(method_umap[:, 0], method_umap[:, 1], c=feature_colors, alpha=0.6, s=40, 
                               marker='^', label=method_name, edgecolors='gray', linewidth=0.5)
                
                ax_umap.set_xlabel('UMAP1')
                ax_umap.set_ylabel('UMAP2')
                ax_umap.set_title(f'{platform_name} - {method_name} (UMAP)')
                ax_umap.legend(loc='upper right')
                ax_umap.grid(True, alpha=0.3)
                
                row_idx += 1
        
        plt.tight_layout()
        return fig
    
    def generate_figure_9c_feature_level_umap_pca_knn_clusters(self, data: AnalysisData):
        """Figure 9c: Feature-level PCA and UMAP with K-Means clustering"""
        print("  Generating Figure 9c: Feature-level PCA and UMAP with K-Means clustering...")
        
        # Calculate the number of methods (2-3, excluding Method 4 as it's a permuted baseline)
        n_methods = 2
        if data.imp_a_m3 is not None:
            n_methods = 3
        # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
        
        # Create figure with 2 columns (PCA, UMAP) and n_methods*2 rows
        fig, axes = plt.subplots(n_methods * 2, 2, figsize=(10, 5 * n_methods * 2))
        if n_methods * 2 == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Feature-Level Structure Analysis with K-Means Clustering', fontsize=14, fontweight='bold')
        
        # Process each platform
        row_idx = 0
        for platform_idx, (truth_data, platform_name) in enumerate([(data.truth_a, data.platform_a_name), 
                                                                     (data.truth_b, data.platform_b_name)]):
            # Get method data for this platform
            method_data_list = []
            
            if platform_idx == 0:  # Platform A
                method_data_list = [
                    (data.imp_a_m1, data.method1_name),
                    (data.imp_a_m2, data.method2_name)
                ]
                if data.imp_a_m3 is not None:
                    method_data_list.append((data.imp_a_m3, data.method3_name))
                # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            else:  # Platform B
                method_data_list = [
                    (data.imp_b_m1, data.method1_name),
                    (data.imp_b_m2, data.method2_name)
                ]
                if data.imp_b_m3 is not None:
                    method_data_list.append((data.imp_b_m3, data.method3_name))
                # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            
            # Transpose to get features as rows
            truth_features = truth_data.values
            
            # Check if we need to reduce dimensions first
            if truth_features.shape[1] > 50:
                # First reduce to 50 dimensions with PCA
                pca_reducer = PCA(n_components=min(50, truth_features.shape[1] - 1))
                truth_features_reduced = pca_reducer.fit_transform(truth_features)
            else:
                truth_features_reduced = truth_features
            
            # Determine optimal number of clusters (3-8)
            n_clusters = min(8, max(3, len(truth_data.index) // 20))
            
            # Perform K-Means clustering on truth data
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(truth_features_reduced)
            
            # Create color map for clusters
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            # Process each method for this platform
            for method_idx, (method_data, method_name) in enumerate(method_data_list):
                # Get method features
                method_features = method_data.values
                
                if truth_features.shape[1] > 50:
                    method_features = pca_reducer.transform(method_features)
                
                # PCA (2D)
                pca = PCA(n_components=2)
                truth_pca = pca.fit_transform(truth_features_reduced)
                method_pca = pca.transform(method_features if truth_features.shape[1] <= 50 else method_features)
                
                # UMAP
                umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, truth_features.shape[0] - 1))
                truth_umap = umap_reducer.fit_transform(truth_features_reduced)
                method_umap = umap_reducer.transform(method_features if truth_features.shape[1] <= 50 else method_features)
                
                # Plot PCA
                ax_pca = axes[row_idx, 0]
                for cluster_id in range(n_clusters):
                    mask = cluster_labels == cluster_id
                    ax_pca.scatter(truth_pca[mask, 0], truth_pca[mask, 1], c=[colors[cluster_id]], 
                                  alpha=0.6, s=60, marker='o', label=f'Cluster {cluster_id+1}' if method_idx == 0 else "", 
                                  edgecolors='black', linewidth=0.5)
                    ax_pca.scatter(method_pca[mask, 0], method_pca[mask, 1], c=[colors[cluster_id]], 
                                  alpha=0.6, s=40, marker='^', edgecolors='gray', linewidth=0.5)
                
                ax_pca.set_xlabel('PC1')
                ax_pca.set_ylabel('PC2')
                ax_pca.set_title(f'{platform_name} - {method_name} (PCA)')
                if row_idx == 0:
                    ax_pca.legend(loc='upper right', fontsize=8)
                ax_pca.grid(True, alpha=0.3)
                
                # Plot UMAP
                ax_umap = axes[row_idx, 1]
                for cluster_id in range(n_clusters):
                    mask = cluster_labels == cluster_id
                    ax_umap.scatter(truth_umap[mask, 0], truth_umap[mask, 1], c=[colors[cluster_id]], 
                                   alpha=0.6, s=60, marker='o', label=f'Cluster {cluster_id+1}' if method_idx == 0 else "", 
                                   edgecolors='black', linewidth=0.5)
                    ax_umap.scatter(method_umap[mask, 0], method_umap[mask, 1], c=[colors[cluster_id]], 
                                   alpha=0.6, s=40, marker='^', edgecolors='gray', linewidth=0.5)
                
                ax_umap.set_xlabel('UMAP1')
                ax_umap.set_ylabel('UMAP2')
                ax_umap.set_title(f'{platform_name} - {method_name} (UMAP)')
                if row_idx == 0:
                    ax_umap.legend(loc='upper right', fontsize=8)
                ax_umap.grid(True, alpha=0.3)
                
                row_idx += 1
        
        plt.tight_layout()
        return fig
    
    def generate_figure_10_radar_chart(self, data: AnalysisData):
        """Figure 10: Radar chart of summary metrics"""
        print("Generating Figure 10: Radar chart summary...")
        
        # Calculate summary metrics for each method (averaged across platforms)
        feat_metrics = data.metrics['feature_wise']
        
        if feat_metrics.empty:
            print("    ⚠️  No feature metrics available for radar chart")
            return self._create_insufficient_data_figure(
                'Summary Metrics Comparison (Radar Chart)',
                'No feature metrics available for analysis'
            )
        
        # Group by method and calculate summary statistics
        method_summaries = {}
        
        # Build list of available methods
        method_list = [('Method_1', data.method1_name), ('Method_2', data.method2_name)]
        if data.method3_name is not None:
            method_list.append(('Method_3', data.method3_name))
        if data.method4_name is not None:
            method_list.append(('Method_4', data.method4_name))
        
        for method_key, method_name in method_list:
            method_data = feat_metrics[feat_metrics['method'] == method_key]
            
            if len(method_data) > 0:
                # Calculate raw metrics
                median_r = np.median(method_data['r'].values)
                percentile_90_r = np.percentile(method_data['r'].values, 90)
                median_mae = np.median(method_data['mae'].values)
                median_rmse = np.median(method_data['rmse'].values)
                mean_abs_bias = np.abs(np.mean(method_data['bias'].values))
                
                # Use different scaling for error metrics if they're very small
                # Check if MAE/RMSE/bias are all very small (< 0.01)
                if median_mae < 0.01 and median_rmse < 0.01 and mean_abs_bias < 0.01:
                    # Use multiplicative scaling for very small errors
                    scale_factor = 100  # Scale up by 100x for visibility
                    summaries = {
                        'Median r': median_r,
                        '90th %ile r': percentile_90_r,
                        f'Low MAE (×{scale_factor})': (1 - median_mae * scale_factor) if median_mae * scale_factor < 1 else 0,
                        f'Low RMSE (×{scale_factor})': (1 - median_rmse * scale_factor) if median_rmse * scale_factor < 1 else 0,
                        f'Low |Bias| (×{scale_factor})': (1 - mean_abs_bias * scale_factor) if mean_abs_bias * scale_factor < 1 else 0
                    }
                else:
                    # Use standard inverse scaling
                    summaries = {
                        'Median r': median_r,
                        '90th %ile r': percentile_90_r,
                        'Low MAE': 1 / (1 + median_mae),  # Invert so higher is better
                        'Low RMSE': 1 / (1 + median_rmse),  # Invert so higher is better
                        'Low |Bias|': 1 / (1 + mean_abs_bias)  # Invert so higher is better
                    }
                
                method_summaries[method_name] = summaries
        
        if not method_summaries:
            print("    ⚠️  No method data available for radar chart")
            return self._create_insufficient_data_figure(
                'Summary Metrics Comparison (Radar Chart)',
                'No method data available for analysis'
            )
        
        # Create radar chart
        categories = list(next(iter(method_summaries.values())).keys())
        n_cats = len(categories)
        
        # Calculate angles for each category
        angles = [n * 2 * np.pi / n_cats for n in range(n_cats)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        fig.suptitle('Summary Metrics Comparison (Radar Chart)', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Normalize metrics to 0-1 scale for visualization using common ranges
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Define reasonable ranges for each metric type
        metric_ranges = {
            'Median r': (0, 1),           # Correlation ranges from 0 to 1
            '90th %ile r': (0, 1),        # Correlation ranges from 0 to 1
            'Low MAE': (0, 1),            # Inverted MAE (higher is better)
            'Low RMSE': (0, 1),           # Inverted RMSE (higher is better)
            'Low |Bias|': (0, 1)          # Inverted bias (higher is better)
        }
        
        for i, (method_name, summaries) in enumerate(method_summaries.items()):
            values = []
            
            # Normalize each metric using its predefined range
            for cat in categories:
                val = summaries[cat]
                min_range, max_range = metric_ranges[cat]
                
                # Normalize to 0-1 scale using predefined range
                if max_range > min_range:
                    norm_val = (val - min_range) / (max_range - min_range)
                else:
                    norm_val = 0.5  # Default middle value if range is zero
                
                # Clamp to [0,1] to handle values outside expected range
                values.append(max(0, min(1, norm_val)))
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, 
                   color=colors[i], markersize=4)  # Smaller markers
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Add note about normalization and metric interpretation
        # Check if any method used scaled metrics (look for scale factor in label)
        scaled_metrics = any('×' in cat for summaries in method_summaries.values() for cat in summaries.keys())
        
        if scaled_metrics:
            note_text = ('Note: All metrics normalized to 0-1 scale. Higher values = better performance.\n'
                        'Error metrics scaled up for visibility when very small (×100 indicates scaling factor).')
        else:
            note_text = ('Note: All metrics normalized to 0-1 scale. Higher values = better performance.\n'
                        'Error metrics (MAE, RMSE, |Bias|) inverted for consistency.')
        
        fig.text(0.5, 0.02, note_text, ha='center', fontsize=8, style='italic')
        
        plt.tight_layout()
        return fig
    
    def generate_all_figures(self, data: AnalysisData):
        """Generate all analysis figures"""
        print("\nGenerating comprehensive analysis figures...")
        
        figure_functions = [
            ("figure_1_feature_r_scatter", self.generate_figure_1_feature_r_scatter),
            ("figure_2_sample_r_scatter", self.generate_figure_2_sample_r_scatter),
            ("figure_3_r_distribution_ridge", self.generate_figure_3_r_distribution_ridge),
            ("figure_3b_vertical_violin_plots", self.generate_figure_3b_vertical_violin_plots),
            ("figure_4_bland_altman", self.generate_figure_4_bland_altman),
            ("figure_4b_bias_comparison", self.generate_figure_4b_bias_comparison),
            ("figure_4c_bland_altman_density", self.generate_figure_4c_bland_altman_density),
            ("figure_4d_variance_vs_mean", self.generate_figure_4d_variance_vs_mean),
            ("figure_5_error_ecdfs", self.generate_figure_5_error_ecdfs),
            ("figure_6_sample_error_heatmap", self.generate_figure_6_sample_error_heatmap),
            ("figure_7_hexbin_error_abundance", self.generate_figure_7_hexbin_error_abundance),
            ("figure_8_global_correlation_heatmap", self.generate_figure_8_global_correlation_heatmap),
            ("figure_9_umap_concordance", self.generate_figure_9_umap_concordance),
            ("figure_9b_feature_level_umap_pca", self.generate_figure_9b_feature_level_umap_pca),
            ("figure_9c_feature_level_umap_pca_knn_clusters", self.generate_figure_9c_feature_level_umap_pca_knn_clusters),
            ("figure_10_radar_chart", self.generate_figure_10_radar_chart),
            
            # Figures that don't require groups
            ("figure_14_volcano_plot", self.generate_figure_14_volcano_plot),
            ("figure_15_network_diagram", self.generate_figure_15_network_diagram),
            ("figure_16_sankey_diagram", self.generate_figure_16_sankey_diagram),
            
            # Comprehensive method comparison (only if 2+ methods available)
            ("figure_26_comprehensive_method_comparison", self.generate_figure_26_comprehensive_method_comparison),
            ("figure_27_comprehensive_method_comparison_spearman", self.generate_figure_27_comprehensive_method_comparison_spearman),
            
            # Additional innovative figures
            ("figure_17_performance_consistency", self.generate_figure_17_performance_consistency),
            ("figure_18_method_synergy_analysis", self.generate_figure_18_method_synergy_analysis),
            ("figure_19_cross_platform_transferability", self.generate_figure_19_cross_platform_transferability),
            ("figure_19b_cross_platform_transferability_density", self.generate_figure_19b_cross_platform_transferability_density),
            ("figure_20_feature_difficulty_profiling", self.generate_figure_20_feature_difficulty_profiling),
            ("figure_21_temporal_performance_trends", self.generate_figure_21_temporal_performance_trends),
        ]
        
        # Additional figures that require overlapping features
        shared_feature_functions = [
            ("figure_22_shared_vs_unique_performance", self.generate_figure_22_shared_vs_unique_performance),
            ("figure_23_imputation_vs_concordance", self.generate_figure_23_imputation_vs_concordance),
            ("figure_23b_imputation_vs_concordance_density", self.generate_figure_23b_imputation_vs_concordance_density),
            ("figure_23c_imputation_concordance_difference", self.generate_figure_23c_imputation_concordance_difference),
            ("figure_23c_percentage_increase", self.generate_figure_23c_percentage_increase),
            ("figure_24_shared_correlation_structure", self.generate_figure_24_shared_correlation_structure),
            ("figure_25_cross_platform_feature_correlation", self.generate_figure_25_cross_platform_feature_correlation),
        ]
        
        generated_figures = []
        
        # Generate basic figures (1-10 and 17-21)
        for fig_name, fig_func in figure_functions:
            try:
                fig = fig_func(data)
                # Save figure 4c at higher dpi for small dense regions
                if fig_name == "figure_4c_bland_altman_density":
                    self.save_figure(fig, fig_name, dpi=600)
                else:
                    self.save_figure(fig, fig_name)
                plt.close(fig)
                generated_figures.append(fig_name)
            except Exception as e:
                print(f"  Error generating {fig_name}: {str(e)}")
        
        # Generate shared feature figures if overlapping features exist
        overlapping_features = getattr(self, 'overlapping_features', [])
        if overlapping_features:
            print(f"Overlapping features detected ({len(overlapping_features)} features) - generating shared feature analyses...")
            for fig_name, fig_func in shared_feature_functions:
                try:
                    fig = fig_func(data)
                    self.save_figure(fig, fig_name)
                    plt.close(fig)
                    generated_figures.append(fig_name)
                except Exception as e:
                    print(f"  Error generating {fig_name}: {str(e)}")
        else:
            print("No overlapping features found - skipping shared feature analyses (22-24)")
        
        print(f"Generated {len(generated_figures)} basic figures saved to: {self.output_dir / 'figures'}")
        
        # Generate group-dependent figures if biological grouping is available
        if data.groups is not None:
            print("Biological grouping detected - generating additional group-based analyses...")
            group_figures = self.generate_group_dependent_figures(data)
            generated_figures.extend(group_figures)
        else:
            print("No biological grouping found - skipping group-dependent figures (11-16)")
        
        return generated_figures
    
    def run_network_analysis(self, data: AnalysisData, ppi_file: Optional[str] = None, gri_file: Optional[str] = None):
        """Run comprehensive network-based analysis"""
        if not self.network_analyzer:
            print("❌ Network analyzer not available")
            return []
        
        if not (ppi_file or gri_file):
            print("ℹ️  No network files provided - skipping network analysis")
            return []
            
        print("🔗 Running network-based analysis...")
        
        # Get feature names from both platforms
        feature_names = list(set(data.truth_a.index.tolist() + data.truth_b.index.tolist()))
        platform_a_features = data.truth_a.index.tolist()
        platform_b_features = data.truth_b.index.tolist()
        
        # Load and process network data
        network_data = self.network_analyzer.load_and_process_networks(
            ppi_file=ppi_file,
            gri_file=gri_file,
            feature_names=feature_names,
            platform_a_features=platform_a_features,
            platform_b_features=platform_b_features
        )
        
        # Check if we should skip analysis based on mapping results
        total_mapped = len(network_data.get('feature_mapping', {}))
        platform_a_mapped = len(network_data.get('platform_a_mapping', {}))
        platform_b_mapped = len(network_data.get('platform_b_mapping', {}))
        
        if total_mapped == 0:
            print("❌ Network analysis skipped - no features mapped to networks")
            return []
        
        if platform_a_mapped == 0 and platform_b_mapped == 0:
            print("❌ Network analysis skipped - no features from either platform mapped")
            return []
        
        if not network_data['network_metrics']:
            print("❌ No network metrics computed")
            return []
        
        # Prepare imputation performance data for network analysis
        imputation_df = data.metrics['feature_wise'].copy()
        
        # Generate network analysis figures
        network_figures = self.network_analyzer.generate_network_figures(imputation_df, network_data)
        
        # Generate and save network summary report
        network_report = self.network_analyzer.generate_network_report(network_data)
        report_path = self.output_dir / "logs" / f"network_analysis_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(network_report)
        
        print(f"🔗 Network analysis completed: {len(network_figures)} figures generated")
        print(f"📝 Network report saved: {report_path}")
        
        return network_figures
    
    def generate_group_dependent_figures(self, data: AnalysisData):
        """Generate figures that require biological grouping (11-13)"""
        if data.groups is None:
            return []
        
        print("  Generating group-dependent figures...")
        group_figures = []
        
        # Figure 11: Group-wise r boxplots
        try:
            fig11 = self.generate_figure_11_group_r_boxplots(data)
            self.save_figure(fig11, "figure_11_group_r_boxplots")
            plt.close(fig11)
            group_figures.append("figure_11_group_r_boxplots")
        except Exception as e:
            print(f"    Error generating figure 11: {str(e)}")
        
        # Figure 12: Group-level improvement heatmap
        try:
            fig12 = self.generate_figure_12_group_improvement_heatmap(data)
            self.save_figure(fig12, "figure_12_group_improvement_heatmap")
            plt.close(fig12)
            group_figures.append("figure_12_group_improvement_heatmap")
        except Exception as e:
            print(f"    Error generating figure 12: {str(e)}")
        
        # Figure 13: Group-stratified Bland-Altman raincloud
        try:
            fig13 = self.generate_figure_13_group_bland_altman(data)
            self.save_figure(fig13, "figure_13_group_bland_altman")
            plt.close(fig13)
            group_figures.append("figure_13_group_bland_altman")
        except Exception as e:
            print(f"    Error generating figure 13: {str(e)}")
        
        
        print(f"  Generated {len(group_figures)} group-dependent figures")
        return group_figures
    
    def generate_phenotype_dependent_figures(self, data: AnalysisData):
        """Generate figures that require phenotype data (28-29)"""
        if data.phenotype_data is None:
            return []
        
        print("  Generating phenotype-dependent figures...")
        phenotype_figures = []
        
        # Process binary phenotypes
        if data.binary_pheno_cols:
            print("  Processing binary phenotypes...")
            binary_associations = {}
            
            for pheno_col in data.binary_pheno_cols:
                try:
                    associations = self.calculate_binary_associations(data, pheno_col)
                    if associations:
                        binary_associations[pheno_col] = associations
                except Exception as e:
                    print(f"    Error calculating associations for {pheno_col}: {str(e)}")
            
            if binary_associations:
                # Figure 28: Binary phenotype forest plots
                try:
                    fig28 = self.generate_figure_28_phenotype_forest_plots_binary(data, binary_associations)
                    self.save_figure(fig28, "figure_28_binary_phenotype_forest_plots")
                    plt.close(fig28)
                    phenotype_figures.append("figure_28_binary_phenotype_forest_plots")
                except Exception as e:
                    print(f"    Error generating figure 28: {str(e)}")
        
        # Process continuous phenotypes
        if data.continuous_pheno_cols:
            print("  Processing continuous phenotypes...")
            continuous_associations = {}
            
            for pheno_col in data.continuous_pheno_cols:
                try:
                    associations = self.calculate_continuous_associations(data, pheno_col)
                    if associations:
                        continuous_associations[pheno_col] = associations
                except Exception as e:
                    print(f"    Error calculating associations for {pheno_col}: {str(e)}")
            
            if continuous_associations:
                # Figure 29: Continuous phenotype forest plots
                try:
                    fig29 = self.generate_figure_29_phenotype_forest_plots_continuous(data, continuous_associations)
                    self.save_figure(fig29, "figure_29_continuous_phenotype_forest_plots")
                    plt.close(fig29)
                    phenotype_figures.append("figure_29_continuous_phenotype_forest_plots")
                except Exception as e:
                    print(f"    Error generating figure 29: {str(e)}")
        
        print(f"  Generated {len(phenotype_figures)} phenotype-dependent figures")
        return phenotype_figures
    
    def generate_figure_12_group_improvement_heatmap(self, data: AnalysisData):
        """Figure 12: Group-level improvement heatmap"""
        print("  Generating Figure 12: Group-level improvement heatmap...")
        
        # Calculate improvement (Method 2 - Method 1) by group and platform
        feat_metrics = data.metrics['feature_wise'].copy()
        
        # Add group information - check if groups are feature-level or sample-level
        group_mapping = {}
        if data.groups is not None:
            feature_names = data.truth_a.index
            
            if len(data.groups) == len(feature_names):  # Feature-level groups
                group_mapping = dict(zip(feature_names, data.groups))
            elif len(data.groups) == len(data.truth_a.columns):  # Sample-level groups
                print("    Warning: Groups appear to be sample-level, not feature-level.")
                print("    Feature grouping analysis requires feature-level metadata.")
                print("    Skipping group improvement heatmap.")
                return self._create_insufficient_data_figure(
                    'Group-Level Performance Improvement',
                    'Feature-level groups required for this analysis'
                )
            else:
                print(f"    Warning: Groups length ({len(data.groups)}) doesn't match features ({len(feature_names)}) or samples ({len(data.truth_a.columns)})")
                return self._create_insufficient_data_figure(
                    'Group-Level Performance Improvement',
                    'Group metadata dimensions don\'t match data'
                )
        
        feat_metrics['group'] = feat_metrics['feature'].map(
            lambda x: group_mapping.get(x, 'Unknown'))
        
        # Calculate improvement matrix
        improvement_data = []
        
        for platform in ['Platform_A', 'Platform_B']:
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            
            # Pivot by feature and method
            pivot_data = platform_data.pivot(index=['feature', 'group'], 
                                            columns='method', values='r')
            
            if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
                pivot_data['improvement'] = pivot_data['Method_2'] - pivot_data['Method_1']
                pivot_data = pivot_data.reset_index()
                
                # Group by biological group and calculate median improvement
                group_improvement = pivot_data.groupby('group')['improvement'].median()
                
                for group, improvement in group_improvement.items():
                    if group != 'Unknown':
                        improvement_data.append({
                            'group': group,
                            'platform': platform,
                            'improvement': improvement
                        })
        
        if not improvement_data:
            print("    No data available for group improvement heatmap")
            return plt.figure()
        
        # Create DataFrame and pivot for heatmap
        improvement_df = pd.DataFrame(improvement_data)
        heatmap_data = improvement_df.pivot(index='group', columns='platform', values='improvement')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto',
                      vmin=-0.2, vmax=0.2)
        
        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels([col.replace('_', ' ') for col in heatmap_data.columns])
        ax.set_yticklabels(heatmap_data.index)
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not pd.isna(value):
                    ax.text(j, i, f'{value:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Median Improvement\n({data.method2_name} - {data.method1_name})', 
                      rotation=270, labelpad=20)
        
        ax.set_title('Group-Level Performance Improvement', fontweight='bold', pad=20)
        ax.set_xlabel('Platform')
        ax.set_ylabel('Biological Group')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_13_group_bland_altman(self, data: AnalysisData):
        """Figure 13: Group-stratified Bland-Altman raincloud"""
        print("  Generating Figure 13: Group-stratified Bland-Altman...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Calculate differences for each group
        feat_metrics = data.metrics['feature_wise'].copy()
        group_mapping = {}
        if data.groups is not None:
            feature_names = data.truth_a.index
            
            if len(data.groups) == len(feature_names):  # Feature-level groups
                group_mapping = dict(zip(feature_names, data.groups))
            elif len(data.groups) == len(data.truth_a.columns):  # Sample-level groups
                print("    Warning: Groups appear to be sample-level, not feature-level.")
                print("    Feature grouping analysis requires feature-level metadata.")
                print("    Skipping group Bland-Altman analysis.")
                return self._create_insufficient_data_figure(
                    'Group-Stratified Bland-Altman Analysis',
                    'Feature-level groups required for this analysis'
                )
            else:
                print(f"    Warning: Groups length ({len(data.groups)}) doesn't match features ({len(feature_names)}) or samples ({len(data.truth_a.columns)})")
                return self._create_insufficient_data_figure(
                    'Group-Stratified Bland-Altman Analysis',
                    'Group metadata dimensions don\'t match data'
                )
        
        feat_metrics['group'] = feat_metrics['feature'].map(
            lambda x: group_mapping.get(x, 'Unknown'))
        
        # Get unique groups
        groups = [g for g in feat_metrics['group'].unique() if g != 'Unknown']
        
        if len(groups) == 0:
            print("    No valid groups for Bland-Altman analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(len(groups), 2, figsize=(10, 5*len(groups)))
        if len(groups) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Group-Stratified Bland-Altman Analysis', fontsize=14, fontweight='bold')
        
        for i, group in enumerate(groups):
            group_features = feat_metrics[feat_metrics['group'] == group]['feature'].unique()
            
            for j, platform in enumerate(['Platform_A', 'Platform_B']):
                ax = axes[i, j]
                
                # Get truth and imputed data for this group
                if platform == 'Platform_A':
                    truth_data = data.truth_a.loc[group_features]
                    imp_m1 = data.imp_a_m1.loc[group_features] 
                    imp_m2 = data.imp_a_m2.loc[group_features]
                else:
                    truth_data = data.truth_b.loc[group_features]
                    imp_m1 = data.imp_b_m1.loc[group_features]
                    imp_m2 = data.imp_b_m2.loc[group_features]
                
                # Calculate BA differences for both methods
                for method_name, imp_data, color in [(data.method1_name, imp_m1, NATURE_COLORS['primary']),
                                                    (data.method2_name, imp_m2, NATURE_COLORS['secondary'])]:
                    
                    truth_flat = truth_data.values.flatten()
                    imp_flat = imp_data.values.flatten()
                    mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
                    
                    if np.sum(mask) > 10:
                        truth_clean = truth_flat[mask]
                        imp_clean = imp_flat[mask]
                        
                        mean_vals = (truth_clean + imp_clean) / 2
                        diff_vals = imp_clean - truth_clean
                        
                        # Create raincloud effect
                        # 1. KDE curve (cloud)
                        from scipy import stats
                        kde = stats.gaussian_kde(diff_vals)
                        x_range = np.linspace(np.min(diff_vals), np.max(diff_vals), 100)
                        density = kde(x_range)
                        
                        # Normalize and position the KDE
                        offset = 0.2 if method_name == data.method1_name else -0.2
                        ax.fill_betweenx(x_range, offset, offset + density/np.max(density)*0.15, 
                                       alpha=0.6, color=color, label=f'{method_name} KDE')
                        
                        # 2. Jittered points (rain)
                        y_jitter = diff_vals + np.random.normal(0, 0.01, len(diff_vals))
                        x_jitter = np.full(len(diff_vals), offset) + np.random.normal(0, 0.02, len(diff_vals))
                        ax.scatter(x_jitter, y_jitter, alpha=0.4, s=10, color=color)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Method (with KDE)')
                ax.set_ylabel('Imputed - Truth')
                ax.set_title(f'{group} - {platform_name_map[platform]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_15_network_diagram(self, data: AnalysisData):
        """Figure 15: Network diagram of well-imputed, highly-correlated features"""
        print("  Generating Figure 15: Network diagram...")
        
        try:
            import networkx as nx
        except ImportError:
            print("    NetworkX not available for network diagram")
            return plt.figure()
        
        # Get features with high cross-platform correlation and good imputation
        feat_metrics = data.metrics['feature_wise']
        cross_r2 = data.cross_platform_r2
        
        # Calculate mean performance across methods and platforms
        feature_performance = feat_metrics.groupby('feature')['r'].mean()
        
        # Filter for well-performing features
        good_features = feature_performance[feature_performance > 0.7].index
        high_r2_features = cross_r2[cross_r2 > 0.8].index
        
        # Find intersection
        network_features = list(set(good_features) & set(high_r2_features))
        
        if len(network_features) < 3:
            print("    Not enough high-quality features for network")
            return plt.figure()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with attributes
        for feature in network_features:
            G.add_node(feature, 
                      performance=feature_performance.get(feature, 0),
                      cross_r2=cross_r2.get(feature, 0))
        
        # Add edges for highly correlated features
        feature_subset = data.truth_a.loc[network_features]
        corr_matrix = feature_subset.T.corr()
        
        for i, feat1 in enumerate(network_features):
            for j, feat2 in enumerate(network_features[i+1:], i+1):
                corr = corr_matrix.loc[feat1, feat2]
                if not pd.isna(corr) and abs(corr) > 0.6:
                    G.add_edge(feat1, feat2, weight=abs(corr))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node sizes based on performance
        node_sizes = [G.nodes[node]['performance'] * 1000 for node in G.nodes()]
        
        # Node colors based on cross-platform R² quartiles
        cross_r2_values = [G.nodes[node]['cross_r2'] for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=cross_r2_values, cmap='viridis',
                              alpha=0.8, ax=ax)
        
        # Draw edges with thickness based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Network of Well-Imputed, Highly-Correlated Features', 
                    fontweight='bold', pad=20)
        ax.text(0.02, 0.98, 'Node size: Mean imputation performance\nNode color: Cross-platform R²\nEdge width: Feature correlation', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=min(cross_r2_values), 
                                                   vmax=max(cross_r2_values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cross-Platform R²', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_16_sankey_diagram(self, data: AnalysisData):
        """Figure 16: Sankey diagram showing feature performance transitions"""
        print("  Generating Figure 16: Sankey diagram...")
        
        # Calculate performance tiers for each method
        feat_metrics = data.metrics['feature_wise']
        
        # Get method 1 and method 2 performance for Platform A (can extend to both)
        platform_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        # Pivot to get both methods for comparison
        pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
        
        if 'Method_1' not in pivot_data.columns or 'Method_2' not in pivot_data.columns:
            print("    Insufficient data for Sankey diagram")
            return plt.figure()
        
        pivot_data = pivot_data.dropna()
        
        # Define performance tiers
        def get_tier(r_values):
            # Use fixed bin edges to avoid empty categories when values are clustered
            # Include right edge to handle r=1.0 properly
            return pd.cut(r_values, bins=[0, 0.33, 0.66, 1.001], labels=['Low', 'Medium', 'High'], include_lowest=True, right=True)
        
        # Add group information if available
        if data.groups is not None:
            group_mapping = data.groups.to_dict()
            pivot_data['group'] = pivot_data.index.map(
                lambda x: group_mapping.get(x, 'Ungrouped'))
        else:
            pivot_data['group'] = 'All'
        
        # Create tiers
        pivot_data['tier_m1'] = get_tier(pivot_data['Method_1'])
        pivot_data['tier_m2'] = get_tier(pivot_data['Method_2'])
        
        # Count transitions
        transition_counts = pivot_data.groupby(['group', 'tier_m1', 'tier_m2']).size().reset_index(name='count')
        
        # Create a simplified Sankey-style visualization using matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Define positions
        source_y = {'Low': 0.8, 'Medium': 0.5, 'High': 0.2}
        target_y = {'Low': 0.8, 'Medium': 0.5, 'High': 0.2}
        
        # Colors for each tier
        tier_colors = {'Low': NATURE_COLORS['neutral'], 
                      'Medium': NATURE_COLORS['secondary'], 
                      'High': NATURE_COLORS['accent']}
        
        # Draw flows
        max_count = transition_counts['count'].max()
        
        for _, row in transition_counts.iterrows():
            source_tier = row['tier_m1']
            target_tier = row['tier_m2']
            count = row['count']
            
            # Line width proportional to count
            line_width = (count / max_count) * 20
            
            # Draw curved line
            x1, y1 = 0.2, source_y[source_tier]
            x2, y2 = 0.8, target_y[target_tier]
            
            # Simple bezier-like curve
            x_mid = (x1 + x2) / 2
            
            x_curve = np.linspace(x1, x2, 100)
            y_curve = np.interp(x_curve, [x1, x_mid, x2], [y1, (y1+y2)/2, y2])
            
            ax.plot(x_curve, y_curve, linewidth=line_width, alpha=0.6,
                   color=tier_colors[source_tier])
        
        # Draw source and target labels
        for tier, y_pos in source_y.items():
            ax.text(0.15, y_pos, f'{tier}\n({data.method1_name})', 
                   ha='right', va='center', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor=tier_colors[tier], alpha=0.7))
            
            ax.text(0.85, y_pos, f'{tier}\n({data.method2_name})', 
                   ha='left', va='center', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor=tier_colors[tier], alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Feature Performance Transitions (Sankey-style)', 
                    fontweight='bold', pad=20)
        ax.text(0.5, 0.05, 'Line thickness represents number of features transitioning', 
               ha='center', fontsize=10, style='italic')
        
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_11_group_r_boxplots(self, data: AnalysisData):
        """Figure 11: Group-wise r boxplots"""
        print("  Generating Figure 11: Group-wise r boxplots...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Merge feature metrics with group information
        feat_metrics = data.metrics['feature_wise'].copy()
        
        # Add group information to metrics - check if groups are feature-level or sample-level
        group_mapping = {}
        if data.groups is not None:
            feature_names = data.truth_a.index
            
            if len(data.groups) == len(feature_names):  # Feature-level groups
                group_mapping = dict(zip(feature_names, data.groups))
            elif len(data.groups) == len(data.truth_a.columns):  # Sample-level groups
                print("    Warning: Groups appear to be sample-level, not feature-level.")
                print("    Feature grouping analysis requires feature-level metadata.")
                print("    Skipping group r boxplots.")
                return self._create_insufficient_data_figure(
                    'Performance by Biological Group',
                    'Feature-level groups required for this analysis'
                )
            else:
                print(f"    Warning: Groups length ({len(data.groups)}) doesn't match features ({len(feature_names)}) or samples ({len(data.truth_a.columns)})")
                return self._create_insufficient_data_figure(
                    'Performance by Biological Group',
                    'Group metadata dimensions don\'t match data'
                )
        
        feat_metrics['group'] = feat_metrics['feature'].map(
            lambda x: group_mapping.get(x, 'Unknown'))
        
        # Create facet grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Performance by Biological Group', fontsize=14, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        methods = ['Method_1', 'Method_2']
        
        for i, platform in enumerate(platforms):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                # Filter data
                subset = feat_metrics[(feat_metrics['platform'] == platform) & 
                                    (feat_metrics['method'] == method)]
                
                groups = []  # Initialize groups to avoid UnboundLocalError
                if len(subset) > 0:
                    # Create boxplot by group
                    groups = subset['group'].unique()
                    groups = [g for g in groups if g != 'Unknown']
                    
                    box_data = [subset[subset['group'] == group]['r'].values 
                              for group in groups]
                    
                    if len(box_data) > 0 and all(len(bd) > 0 for bd in box_data):
                        bp = ax.boxplot(box_data, tick_labels=groups, patch_artist=True)
                        
                        # Color boxes
                        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        # Add swarm plot overlay
                        for k, group in enumerate(groups):
                            group_data = subset[subset['group'] == group]['r'].values
                            if len(group_data) > 0:
                                # Add jitter
                                x_pos = np.full(len(group_data), k + 1, dtype=np.float64)
                                x_pos += np.random.normal(0, 0.05, len(group_data))
                                ax.scatter(x_pos, group_data, alpha=0.6, s=20, color='black')
                    else:
                        # If no valid data, show empty plot
                        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=12)
                
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                ax.set_title(f'{platform_name_map[platform]} - {method_name}')
                ax.set_ylabel('Correlation (r)')
                ax.set_xlabel('Biological Group')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                if len(groups) > 3:
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_14_volcano_plot(self, data: AnalysisData):
        """Figure 14: Volcano plot - cross-platform R² vs Δr"""
        print("  Generating Figure 14: Volcano plot...")
        
        # Calculate Δr (Method 2 - Method 1) for each platform
        feat_metrics = data.metrics['feature_wise']
        
        # Check if we have cross-platform R² data
        if data.cross_platform_r2 is None or len(data.cross_platform_r2) == 0:
            print("    No cross-platform R² data available - creating alternative volcano plot")
            
            # Create platform name mapping
            platform_name_map = {
                'Platform_A': data.platform_a_name,
                'Platform_B': data.platform_b_name
            }
            
            # Create alternative plot using within-platform performance difference
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Performance Difference Analysis (Within Platform)', 
                        fontsize=14, fontweight='bold')
            
            for i, platform in enumerate(['Platform_A', 'Platform_B']):
                ax = ax1 if i == 0 else ax2
                platform_data = feat_metrics[feat_metrics['platform'] == platform]
                
                # Pivot by feature and method
                pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
                
                if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
                    delta_r = pivot_data['Method_2'] - pivot_data['Method_1']
                    avg_r = (pivot_data['Method_1'] + pivot_data['Method_2']) / 2
                    
                    # Create scatter plot
                    ax.scatter(avg_r, delta_r, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
                    
                    # Add horizontal line at y=0
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    
                    ax.set_xlabel('Average Performance')
                    ax.set_ylabel(f'Δr ({data.method2_name} - {data.method1_name})')
                    ax.set_title(f'{platform_name_map[platform]}')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        # Original volcano plot code for when cross-platform R² is available
        platform_results = []
        
        for platform in ['Platform_A', 'Platform_B']:
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            
            # Pivot by feature and method
            pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
            
            if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
                delta_r = pivot_data['Method_2'] - pivot_data['Method_1']
                
                # Merge with cross-platform R²
                cross_r2 = data.cross_platform_r2
                
                # Combine data
                combined = pd.DataFrame({
                    'feature': delta_r.index,
                    'delta_r': delta_r.values,
                    'cross_platform_r2': [cross_r2.get(f, np.nan) for f in delta_r.index],
                    'platform': platform
                })
                
                platform_results.append(combined)
        
        if not platform_results:
            print("    No data available for volcano plot")
            return plt.figure()
        
        # Combine platforms
        all_results = pd.concat(platform_results, ignore_index=True)
        all_results = all_results.dropna()
        
        # Create volcano plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Volcano Plot: Cross-Platform R² vs Performance Difference', 
                    fontsize=14, fontweight='bold')
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        platforms = ['Platform_A', 'Platform_B']
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary']]
        
        for i, (platform, color) in enumerate(zip(platforms, colors)):
            ax = ax1 if i == 0 else ax2
            
            platform_data = all_results[all_results['platform'] == platform]
            
            if len(platform_data) > 0:
                # Create scatter plot
                scatter = ax.scatter(platform_data['cross_platform_r2'], 
                                   platform_data['delta_r'],
                                   alpha=0.6, s=30, c=color, edgecolors='black', linewidth=0.5)
                
                # Add quadrant lines
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Identify interesting points
                high_r2_improved = platform_data[
                    (platform_data['cross_platform_r2'] > 0.7) & 
                    (platform_data['delta_r'] > 0.1)
                ]
                
                high_r2_worsened = platform_data[
                    (platform_data['cross_platform_r2'] > 0.7) & 
                    (platform_data['delta_r'] < -0.1)
                ]
                
                # Label interesting points
                for _, row in high_r2_improved.head(3).iterrows():
                    ax.annotate(row['feature'][:10], 
                              (row['cross_platform_r2'], row['delta_r']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                for _, row in high_r2_worsened.head(3).iterrows():
                    ax.annotate(row['feature'][:10], 
                              (row['cross_platform_r2'], row['delta_r']),
                              xytext=(5, -15), textcoords='offset points',
                              fontsize=8, alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
                
                ax.set_xlabel('Cross-Platform R²')
                ax.set_ylabel(f'Δr ({data.method2_name} - {data.method1_name})')
                ax.set_title(f'{platform_name_map[platform]}')
                ax.grid(True, alpha=0.3)
                
                # Add quadrant labels
                ax.text(0.95, 0.95, 'High R²\nImproved', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(0.95, 0.05, 'High R²\nWorsened', transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def generate_figure_17_performance_consistency(self, data: AnalysisData):
        """Figure 17: Performance consistency across samples and features"""
        print("Generating Figure 17: Performance consistency analysis...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Calculate coefficient of variation (CV) for each method
        feat_metrics = data.metrics['feature_wise']
        samp_metrics = data.metrics['sample_wise']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Performance Consistency Analysis', fontsize=14, fontweight='bold')
        
        # Feature-wise consistency
        for platform in ['Platform_A', 'Platform_B']:
            ax = ax1 if platform == 'Platform_A' else ax2
            
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            
            for method, color in [('Method_1', NATURE_COLORS['primary']), 
                                 ('Method_2', NATURE_COLORS['secondary'])]:
                method_data = platform_data[platform_data['method'] == method]['r']
                
                # Calculate rolling statistics to show consistency
                sorted_data = method_data.sort_values()
                window_size = max(5, min(len(sorted_data) // 5, len(sorted_data) // 20)) if len(sorted_data) > 10 else len(sorted_data)
                if window_size >= len(sorted_data):
                    window_size = max(1, len(sorted_data) // 2)  # Ensure window is smaller than data
                rolling_std = sorted_data.rolling(window=window_size, center=True).std()
                
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                # Calculate CV safely to avoid divide by zero
                mean_val = np.mean(method_data)
                cv = np.std(method_data) / mean_val if mean_val > 1e-10 else np.inf
                ax.plot(range(len(sorted_data)), rolling_std, 
                       color=color, linewidth=2, label=f'{method_name} (CV={cv:.3f})')
            
            ax.set_xlabel('Feature Rank (by performance)')
            ax.set_ylabel('Rolling Standard Deviation')
            ax.set_title(f'{platform_name_map[platform]} - Feature Consistency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Sample-wise consistency
        for platform in ['Platform_A', 'Platform_B']:
            ax = ax3 if platform == 'Platform_A' else ax4
            
            platform_data = samp_metrics[samp_metrics['platform'] == platform]
            
            for method, color in [('Method_1', NATURE_COLORS['primary']), 
                                 ('Method_2', NATURE_COLORS['secondary'])]:
                method_data = platform_data[platform_data['method'] == method]['r']
                
                # Create histogram of performance
                ax.hist(method_data, bins=20, alpha=0.6, color=color, density=True,
                       label=f'{data.method1_name if method == "Method_1" else data.method2_name}')
            
            ax.set_xlabel('Sample-wise Correlation')
            ax.set_ylabel('Density')
            ax.set_title(f'{platform_name_map[platform]} - Sample Performance Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_18_method_synergy_analysis(self, data: AnalysisData):
        """Figure 18: Method synergy and complementarity analysis"""
        print("Generating Figure 18: Method synergy analysis...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        feat_metrics = data.metrics['feature_wise']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Method Synergy and Complementarity Analysis', fontsize=14, fontweight='bold')
        
        for i, platform in enumerate(['Platform_A', 'Platform_B']):
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
            
            if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
                m1_scores = pivot_data['Method_1'].dropna()
                m2_scores = pivot_data['Method_2'].dropna()
                
                # Synergy analysis - which features benefit from ensemble?
                ax = ax1 if i == 0 else ax2
                
                # Compute ensemble score only for positive correlations to avoid misleading results
                # For negative correlations, ensemble would typically perform worse
                ensemble_scores = np.full_like(m1_scores, np.nan)
                positive_mask = (m1_scores > 0) & (m2_scores > 0)
                
                if np.any(positive_mask):
                    # Geometric mean for positive correlations only
                    ensemble_scores[positive_mask] = np.sqrt(m1_scores[positive_mask] * m2_scores[positive_mask])
                
                max_individual = np.maximum(np.abs(m1_scores), np.abs(m2_scores))
                synergy_gain = ensemble_scores - max_individual
                
                # Scatter plot showing synergy potential
                scatter = ax.scatter(max_individual, synergy_gain, 
                                   alpha=0.6, s=30, c=m1_scores-m2_scores, 
                                   cmap='RdBu_r', edgecolors='black', linewidth=0.3)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Best Individual Method Performance')
                ax.set_ylabel('Potential Synergy Gain')
                ax.set_title(f'{platform_name_map[platform]} - Synergy Potential')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{data.method1_name} - {data.method2_name}', rotation=270, labelpad=15)
                
                # Complementarity analysis
                ax_comp = ax3 if i == 0 else ax4
                
                # Find features where methods disagree most
                disagreement = np.abs(m1_scores - m2_scores)
                agreement = np.minimum(m1_scores, m2_scores)
                
                ax_comp.scatter(agreement, disagreement, alpha=0.6, s=30,
                              c=NATURE_COLORS['primary'], edgecolors='black', linewidth=0.3)
                
                # Add quadrant labels
                ax_comp.axhline(y=np.median(disagreement), color='red', linestyle='--', alpha=0.5)
                ax_comp.axvline(x=np.median(agreement), color='red', linestyle='--', alpha=0.5)
                
                ax_comp.set_xlabel('Method Agreement (min performance)')
                ax_comp.set_ylabel('Method Disagreement (|difference|)')
                ax_comp.set_title(f'{platform_name_map[platform]} - Method Complementarity')
                
                # Add quadrant annotations
                ax_comp.text(0.95, 0.95, 'High Disagreement\nHigh Agreement', 
                           transform=ax_comp.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                ax_comp.text(0.05, 0.05, 'Low Disagreement\nLow Agreement', 
                           transform=ax_comp.transAxes, ha='left', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def generate_figure_19_cross_platform_transferability(self, data: AnalysisData):
        """Figure 19: Cross-platform transferability analysis"""
        print("Generating Figure 19: Cross-platform transferability...")
        
        # Check overlapping features for feature-wise analysis
        overlapping_features = getattr(self, 'overlapping_features', [])
        if not overlapping_features:
            print("    ⚠️  No overlapping features between platforms - feature-wise transferability will be skipped")
        else:
            print(f"    Using {len(overlapping_features)} overlapping features for feature-wise analysis")

        # Determine overlapping samples for sample-wise analysis
        samples_a = set(data.truth_a.columns)
        samples_b = set(data.truth_b.columns)
        overlapping_samples = sorted(list(samples_a.intersection(samples_b)))
        if not overlapping_samples:
            print("    ⚠️  No overlapping samples between platforms - sample-wise transferability will be skipped")
        else:
            print(f"    Using {len(overlapping_samples)} overlapping samples for sample-wise analysis")

        # Metrics
        feat_metrics = data.metrics['feature_wise']
        samp_metrics = data.metrics['sample_wise']

        # Prepare dataframes filtered to overlap
        if overlapping_features:
            feat_a = feat_metrics[(feat_metrics['platform'] == 'Platform_A') & (feat_metrics['feature'].isin(overlapping_features))]
            feat_b = feat_metrics[(feat_metrics['platform'] == 'Platform_B') & (feat_metrics['feature'].isin(overlapping_features))]
        else:
            feat_a = feat_metrics[feat_metrics['platform'] == 'Platform_A'].iloc[0:0]
            feat_b = feat_metrics[feat_metrics['platform'] == 'Platform_B'].iloc[0:0]

        if overlapping_samples:
            samp_a = samp_metrics[(samp_metrics['platform'] == 'Platform_A') & (samp_metrics['sample'].isin(overlapping_samples))]
            samp_b = samp_metrics[(samp_metrics['platform'] == 'Platform_B') & (samp_metrics['sample'].isin(overlapping_samples))]
        else:
            samp_a = samp_metrics[samp_metrics['platform'] == 'Platform_A'].iloc[0:0]
            samp_b = samp_metrics[samp_metrics['platform'] == 'Platform_B'].iloc[0:0]

        # Create a 2x4 grid: Row 1 feature-wise (scatter + rank) for two methods; Row 2 sample-wise (scatter + rank) for two methods
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Cross-Platform Performance Transferability', fontsize=14, fontweight='bold')

        method_info = [
            ('Method_1', data.method1_name, NATURE_COLORS['primary']),
            ('Method_2', data.method2_name, NATURE_COLORS['secondary'])
        ]

        # Helper to plot scatter and rank hist given series
        def _plot_pair(ax_scatter, ax_rank, a_series, b_series, title_prefix, color):
            if len(a_series) > 0 and len(b_series) > 0:
                # Align indices
                common_idx = a_series.index.intersection(b_series.index)
                if len(common_idx) == 0:
                    ax_scatter.text(0.5, 0.5, 'No overlapping items', ha='center', va='center', transform=ax_scatter.transAxes)
                    ax_rank.text(0.5, 0.5, 'No overlapping items', ha='center', va='center', transform=ax_rank.transAxes)
                    return
                a_vals = a_series.loc[common_idx]
                b_vals = b_series.loc[common_idx]
                ax_scatter.scatter(a_vals, b_vals, alpha=0.6, s=30, color=color)
                min_val = float(min(a_vals.min(), b_vals.min()))
                max_val = float(max(a_vals.max(), b_vals.max()))
                ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                try:
                    corr, _ = pearsonr(a_vals, b_vals)
                except Exception:
                    corr = np.nan
                ax_scatter.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(common_idx)}', transform=ax_scatter.transAxes,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax_scatter.set_xlabel(f"{data.platform_a_name} Performance")
                ax_scatter.set_ylabel(f"{data.platform_b_name} Performance")
                ax_scatter.set_title(f"{title_prefix} - Cross-Platform Consistency")
                ax_scatter.grid(True, alpha=0.3)

                # Rank difference
                rank_a = a_vals.rank()
                rank_b = b_vals.rank()
                rank_diff = np.abs(rank_a - rank_b)
                bins = max(5, min(20, int(len(rank_diff) // 2))) if len(rank_diff) > 1 else 5
                ax_rank.hist(rank_diff, bins=bins, alpha=0.6, density=True, color=color)
                ax_rank.set_xlabel(f'Rank Difference (|{data.platform_a_name} - {data.platform_b_name}|)')
                ax_rank.set_ylabel('Density')
                ax_rank.set_title(f'{title_prefix} - Ranking Consistency')
                ax_rank.text(0.7, 0.8, f'Mean Rank Δ: {float(rank_diff.mean()):.1f}', transform=ax_rank.transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax_rank.grid(True, alpha=0.3)
            else:
                ax_scatter.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_scatter.transAxes)
                ax_rank.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_rank.transAxes)

        # Plot for each method
        for i, (method_key, method_name, color) in enumerate(method_info):
            # Feature-wise
            ax_scatter = axes[0, i * 2]
            ax_rank = axes[0, i * 2 + 1]
            if overlapping_features:
                series_a = feat_a[feat_a['method'] == method_key].set_index('feature')['r']
                series_b = feat_b[feat_b['method'] == method_key].set_index('feature')['r']
                _plot_pair(ax_scatter, ax_rank, series_a, series_b, f'{method_name} (Feature-wise)', color)
            else:
                ax_scatter.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_scatter.transAxes)
                ax_rank.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_rank.transAxes)

            # Sample-wise
            ax_scatter_s = axes[1, i * 2]
            ax_rank_s = axes[1, i * 2 + 1]
            if overlapping_samples:
                s_series_a = samp_a[samp_a['method'] == method_key].set_index('sample')['r']
                s_series_b = samp_b[samp_b['method'] == method_key].set_index('sample')['r']
                _plot_pair(ax_scatter_s, ax_rank_s, s_series_a, s_series_b, f'{method_name} (Sample-wise)', color)
            else:
                ax_scatter_s.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_scatter_s.transAxes)
                ax_rank_s.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_rank_s.transAxes)

        plt.tight_layout()
        return fig
    
    def generate_figure_19b_cross_platform_transferability_density(self, data: AnalysisData):
        """Figure 19b: Cross-platform transferability analysis with density plots"""
        print("Generating Figure 19b: Cross-platform transferability (density plots)...")
        
        # Overlapping features for feature-wise
        overlapping_features = getattr(self, 'overlapping_features', [])
        if not overlapping_features:
            print("    ⚠️  No overlapping features between platforms - feature-wise density will be skipped")
        else:
            print(f"    Using {len(overlapping_features)} overlapping features for feature-wise density analysis")

        # Overlapping samples for sample-wise
        samples_a = set(data.truth_a.columns)
        samples_b = set(data.truth_b.columns)
        overlapping_samples = sorted(list(samples_a.intersection(samples_b)))
        if not overlapping_samples:
            print("    ⚠️  No overlapping samples between platforms - sample-wise density will be skipped")
        else:
            print(f"    Using {len(overlapping_samples)} overlapping samples for sample-wise density analysis")

        feat_metrics = data.metrics['feature_wise']
        samp_metrics = data.metrics['sample_wise']

        # Filtered views
        if overlapping_features:
            feat_a = feat_metrics[(feat_metrics['platform'] == 'Platform_A') & (feat_metrics['feature'].isin(overlapping_features))]
            feat_b = feat_metrics[(feat_metrics['platform'] == 'Platform_B') & (feat_metrics['feature'].isin(overlapping_features))]
        else:
            feat_a = feat_metrics[feat_metrics['platform'] == 'Platform_A'].iloc[0:0]
            feat_b = feat_metrics[feat_metrics['platform'] == 'Platform_B'].iloc[0:0]

        if overlapping_samples:
            samp_a = samp_metrics[(samp_metrics['platform'] == 'Platform_A') & (samp_metrics['sample'].isin(overlapping_samples))]
            samp_b = samp_metrics[(samp_metrics['platform'] == 'Platform_B') & (samp_metrics['sample'].isin(overlapping_samples))]
        else:
            samp_a = samp_metrics[samp_metrics['platform'] == 'Platform_A'].iloc[0:0]
            samp_b = samp_metrics[samp_metrics['platform'] == 'Platform_B'].iloc[0:0]

        # Create a 2x4 layout mirroring Figure 19
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        fig.suptitle('Cross-Platform Performance Transferability (Density Plots)', fontsize=14, fontweight='bold')

        method_info = [
            ('Method_1', data.method1_name, NATURE_COLORS['primary']),
            ('Method_2', data.method2_name, NATURE_COLORS['secondary'])
        ]

        # Store hexbin artists to sync color scales and add shared colorbars
        hb_feature = [None, None]
        hb_sample = [None, None]

        for i, (method_key, method_name, color) in enumerate(method_info):
            # Feature-wise hexbin
            ax_feat = axes[0, i * 2]
            if overlapping_features:
                series_a = feat_a[feat_a['method'] == method_key].set_index('feature')['r']
                series_b = feat_b[feat_b['method'] == method_key].set_index('feature')['r']
                common_idx = series_a.index.intersection(series_b.index)
                if len(common_idx) > 0:
                    a_vals = series_a.loc[common_idx]
                    b_vals = series_b.loc[common_idx]
                    # Clip negative values to 0
                    a_vals = a_vals.clip(lower=0)
                    b_vals = b_vals.clip(lower=0)
                    # Set axes limits to 0-1
                    ax_feat.set_xlim(0, 1)
                    ax_feat.set_ylim(0, 1)
                    # Plot diagonal line within 0-1 range (before hexbin so it appears underneath)
                    ax_feat.plot([0, 1], [0, 1], 'gray', alpha=0.5, linewidth=2, zorder=1)
                    try:
                        hb = ax_feat.hexbin(a_vals, b_vals, gridsize=20, cmap='Blues', alpha=0.7, zorder=2)
                        hb_feature[i] = hb
                    except Exception:
                        ax_feat.scatter(a_vals, b_vals, alpha=0.6, s=30, color=color, zorder=2)
                    try:
                        corr, _ = pearsonr(a_vals, b_vals)
                    except Exception:
                        corr = np.nan
                    ax_feat.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(common_idx)}', transform=ax_feat.transAxes,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax_feat.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_feat.transAxes)
            else:
                ax_feat.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_feat.transAxes)
            ax_feat.set_xlabel(f'{data.platform_a_name} Performance')
            ax_feat.set_ylabel(f'{data.platform_b_name} Performance')
            ax_feat.set_title(f'{method_name} (Feature-wise) - Cross-Platform Consistency')
            ax_feat.grid(True, alpha=0.3)

            # Feature-wise rank density (KDE/hist)
            ax_feat_rank = axes[0, i * 2 + 1]
            if overlapping_features:
                series_a = feat_a[feat_a['method'] == method_key].set_index('feature')['r']
                series_b = feat_b[feat_b['method'] == method_key].set_index('feature')['r']
                common_idx = series_a.index.intersection(series_b.index)
                if len(common_idx) > 0:
                    a_vals = series_a.loc[common_idx]
                    b_vals = series_b.loc[common_idx]
                    # Clip negative values to 0
                    a_vals = a_vals.clip(lower=0)
                    b_vals = b_vals.clip(lower=0)
                    rank_a = a_vals.rank()
                    rank_b = b_vals.rank()
                    rank_diff = np.abs(rank_a - rank_b)
                    try:
                        from scipy.stats import gaussian_kde
                        if len(rank_diff) > 1 and rank_diff.std() > 0:
                            kde = gaussian_kde(rank_diff)
                            x_range = np.linspace(rank_diff.min(), rank_diff.max(), 100)
                            density = kde(x_range)
                            ax_feat_rank.fill_between(x_range, density, alpha=0.6, color=color)
                            ax_feat_rank.plot(x_range, density, color=color, linewidth=2)
                        else:
                            ax_feat_rank.hist(rank_diff, bins=max(5, min(20, int(len(rank_diff)//2))), alpha=0.6, density=True, color=color)
                    except Exception:
                        ax_feat_rank.hist(rank_diff, bins=max(5, min(20, int(len(rank_diff)//2))), alpha=0.6, density=True, color=color)
                    ax_feat_rank.text(0.7, 0.8, f'Mean Rank Δ: {float(rank_diff.mean()):.1f}', transform=ax_feat_rank.transAxes,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax_feat_rank.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_feat_rank.transAxes)
            else:
                ax_feat_rank.text(0.5, 0.5, 'No overlapping features', ha='center', va='center', transform=ax_feat_rank.transAxes)
            ax_feat_rank.set_xlabel(f'Rank Difference (|{data.platform_a_name} - {data.platform_b_name}|)')
            ax_feat_rank.set_ylabel('Density')
            ax_feat_rank.set_title(f'{method_name} (Feature-wise) - Ranking Consistency')
            ax_feat_rank.grid(True, alpha=0.3)

            # Sample-wise hexbin
            ax_samp = axes[1, i * 2]
            if overlapping_samples:
                s_a = samp_a[samp_a['method'] == method_key].set_index('sample')['r']
                s_b = samp_b[samp_b['method'] == method_key].set_index('sample')['r']
                common_idx = s_a.index.intersection(s_b.index)
                if len(common_idx) > 0:
                    a_vals = s_a.loc[common_idx]
                    b_vals = s_b.loc[common_idx]
                    # Clip negative values to 0
                    a_vals = a_vals.clip(lower=0)
                    b_vals = b_vals.clip(lower=0)
                    # Set axes limits to 0-1
                    ax_samp.set_xlim(0, 1)
                    ax_samp.set_ylim(0, 1)
                    # Plot diagonal line within 0-1 range (before hexbin so it appears underneath)
                    ax_samp.plot([0, 1], [0, 1], 'gray', alpha=0.5, linewidth=2, zorder=1)
                    try:
                        hb = ax_samp.hexbin(a_vals, b_vals, gridsize=20, cmap='Blues', alpha=0.7, zorder=2)
                        hb_sample[i] = hb
                    except Exception:
                        ax_samp.scatter(a_vals, b_vals, alpha=0.6, s=30, color=color, zorder=2)
                    try:
                        corr, _ = pearsonr(a_vals, b_vals)
                    except Exception:
                        corr = np.nan
                    ax_samp.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(common_idx)}', transform=ax_samp.transAxes,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax_samp.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_samp.transAxes)
            else:
                ax_samp.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_samp.transAxes)
            ax_samp.set_xlabel(f'{data.platform_a_name} Performance')
            ax_samp.set_ylabel(f'{data.platform_b_name} Performance')
            ax_samp.set_title(f'{method_name} (Sample-wise) - Cross-Platform Consistency')
            ax_samp.grid(True, alpha=0.3)

            # Sample-wise rank density
            ax_samp_rank = axes[1, i * 2 + 1]
            if overlapping_samples:
                s_a = samp_a[samp_a['method'] == method_key].set_index('sample')['r']
                s_b = samp_b[samp_b['method'] == method_key].set_index('sample')['r']
                common_idx = s_a.index.intersection(s_b.index)
                if len(common_idx) > 0:
                    a_vals = s_a.loc[common_idx]
                    b_vals = s_b.loc[common_idx]
                    # Clip negative values to 0
                    a_vals = a_vals.clip(lower=0)
                    b_vals = b_vals.clip(lower=0)
                    rank_a = a_vals.rank()
                    rank_b = b_vals.rank()
                    rank_diff = np.abs(rank_a - rank_b)
                    try:
                        from scipy.stats import gaussian_kde
                        if len(rank_diff) > 1 and rank_diff.std() > 0:
                            kde = gaussian_kde(rank_diff)
                            x_range = np.linspace(rank_diff.min(), rank_diff.max(), 100)
                            density = kde(x_range)
                            ax_samp_rank.fill_between(x_range, density, alpha=0.6, color=color)
                            ax_samp_rank.plot(x_range, density, color=color, linewidth=2)
                        else:
                            ax_samp_rank.hist(rank_diff, bins=max(5, min(20, int(len(rank_diff)//2))), alpha=0.6, density=True, color=color)
                    except Exception:
                        ax_samp_rank.hist(rank_diff, bins=max(5, min(20, int(len(rank_diff)//2))), alpha=0.6, density=True, color=color)
                    ax_samp_rank.text(0.7, 0.8, f'Mean Rank Δ: {float(rank_diff.mean()):.1f}', transform=ax_samp_rank.transAxes,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax_samp_rank.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_samp_rank.transAxes)
            else:
                ax_samp_rank.text(0.5, 0.5, 'No overlapping samples', ha='center', va='center', transform=ax_samp_rank.transAxes)
            ax_samp_rank.set_xlabel(f'Rank Difference (|{data.platform_a_name} - {data.platform_b_name}|)')
            ax_samp_rank.set_ylabel('Density')
            ax_samp_rank.set_title(f'{method_name} (Sample-wise) - Ranking Consistency')
            ax_samp_rank.grid(True, alpha=0.3)

        # Enforce square subplots (not including colorbars)
        for ax in axes.ravel():
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass

        # Synchronize and add shared colorbars for feature-wise hexbin pair
        if any(hb_feature):
            arrays = [hb.get_array() for hb in hb_feature if hb is not None]
            vmax = max(float(arr.max()) for arr in arrays) if arrays else None
            if vmax is not None and vmax > 0:
                for hb in hb_feature:
                    if hb is not None:
                        hb.set_clim(0, vmax)
                # Shared colorbar across the two feature-wise hexbin axes
                feat_axes = [axes[0, 0], axes[0, 2]]
                cb = fig.colorbar(next(h for h in hb_feature if h is not None), ax=feat_axes, location='right', shrink=0.8)
                cb.set_label('Count')

        # Synchronize and add shared colorbars for sample-wise hexbin pair
        if any(hb_sample):
            arrays = [hb.get_array() for hb in hb_sample if hb is not None]
            vmax = max(float(arr.max()) for arr in arrays) if arrays else None
            if vmax is not None and vmax > 0:
                for hb in hb_sample:
                    if hb is not None:
                        hb.set_clim(0, vmax)
                samp_axes = [axes[1, 0], axes[1, 2]]
                cb = fig.colorbar(next(h for h in hb_sample if h is not None), ax=samp_axes, location='right', shrink=0.8)
                cb.set_label('Count')

        # With constrained_layout=True, avoid calling tight_layout to preserve square axes
        return fig
    
    def generate_figure_20_feature_difficulty_profiling(self, data: AnalysisData):
        """Figure 20: Feature difficulty profiling and characterization"""
        print("Generating Figure 20: Feature difficulty profiling...")
        
        # Characterize features by their "difficulty" to impute
        feat_metrics = data.metrics['feature_wise']
        cross_r2 = data.cross_platform_r2
        
        if feat_metrics.empty:
            print("    ⚠️  No feature metrics available for difficulty profiling")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Feature Difficulty Profiling\n\nNo feature metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax.set_title('Feature Difficulty Profiling and Characterization')
            ax.axis('off')
            return fig
        
        # Calculate difficulty metrics
        difficulty_data = []
        
        for feature in feat_metrics['feature'].unique():
            feature_data = feat_metrics[feat_metrics['feature'] == feature]
            
            if len(feature_data) >= 2:  # Need both methods or platforms
                mean_r = feature_data['r'].mean()
                std_r = feature_data['r'].std()
                cv_r = std_r / max(mean_r, 1e-10) if mean_r > 1e-10 else np.nan  # Avoid divide by zero
                
                cross_platform_r2 = cross_r2.get(feature, np.nan) if cross_r2 is not None else np.nan
                
                difficulty_data.append({
                    'feature': feature,
                    'mean_performance': mean_r,
                    'performance_variability': cv_r,
                    'cross_platform_consistency': cross_platform_r2,
                    'min_performance': feature_data['r'].min(),
                    'max_performance': feature_data['r'].max()
                })
        
        if not difficulty_data:
            print("    ⚠️  No data available for difficulty profiling")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Feature Difficulty Profiling\n\nInsufficient data for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            ax.set_title('Feature Difficulty Profiling and Characterization')
            ax.axis('off')
            return fig
        
        difficulty_df = pd.DataFrame(difficulty_data)
        print(f"    Analyzing difficulty for {len(difficulty_df)} features")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Feature Difficulty Profiling and Characterization', fontsize=14, fontweight='bold')
        
        # Difficulty scatter plot (Top left)
        # Remove NaN values for plotting
        plot_data = difficulty_df.dropna(subset=['mean_performance', 'performance_variability'])
        
        if len(plot_data) > 0:
            # Use cross-platform consistency if available, otherwise use a default color
            if 'cross_platform_consistency' in plot_data.columns and not plot_data['cross_platform_consistency'].isna().all():
                scatter = ax1.scatter(plot_data['mean_performance'], 
                                    plot_data['performance_variability'],
                                    c=plot_data['cross_platform_consistency'], 
                                    cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
                plt.colorbar(scatter, ax=ax1, label='Cross-Platform R²')
            else:
                ax1.scatter(plot_data['mean_performance'], 
                          plot_data['performance_variability'],
                          alpha=0.6, s=30, color=NATURE_COLORS['primary'], edgecolors='black', linewidth=0.3)
            
            # Add quadrant labels
            mean_perf_median = plot_data['mean_performance'].median()
            var_median = plot_data['performance_variability'].median()
            
            ax1.axhline(y=var_median, color='red', linestyle='--', alpha=0.5)
            ax1.axvline(x=mean_perf_median, color='red', linestyle='--', alpha=0.5)
            
            ax1.text(0.95, 0.95, 'Hard to\nImpute', transform=ax1.transAxes, 
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax1.text(0.95, 0.05, 'Easy &\nConsistent', transform=ax1.transAxes, 
                   ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, 'No data for difficulty landscape', 
                   ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        ax1.set_xlabel('Mean Performance Across Methods')
        ax1.set_ylabel('Performance Variability (CV)')
        ax1.set_yscale('log')  # Set y-axis to log scale
        ax1.set_title('Feature Difficulty Landscape')
        ax1.grid(True, alpha=0.3)
        
        # Performance range analysis (Top right)
        if len(difficulty_df) > 0:
            difficulty_df['performance_range'] = difficulty_df['max_performance'] - difficulty_df['min_performance']
            
            # Plot range vs cross-platform consistency if available
            range_data = difficulty_df.dropna(subset=['performance_range'])
            if len(range_data) > 0:
                if 'cross_platform_consistency' in range_data.columns and not range_data['cross_platform_consistency'].isna().all():
                    range_plot_data = range_data.dropna(subset=['cross_platform_consistency'])
                    if len(range_plot_data) > 0:
                        ax2.scatter(range_plot_data['cross_platform_consistency'], 
                                  range_plot_data['performance_range'],
                                  alpha=0.6, s=30, c=NATURE_COLORS['primary'], edgecolors='black', linewidth=0.3)
                        ax2.set_xlabel('Cross-Platform Consistency (R²)')
                        ax2.set_ylabel('Method Performance Range')
                        ax2.set_title('Consistency vs Method Sensitivity')
                    else:
                        ax2.text(0.5, 0.5, 'No cross-platform\nconsistency data', 
                               ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                        ax2.set_title('Consistency vs Method Sensitivity')
                else:
                    # Plot performance range distribution instead
                    ax2.hist(range_data['performance_range'], bins=min(20, len(range_data)//2), 
                           alpha=0.7, color=NATURE_COLORS['primary'])
                    ax2.set_xlabel('Method Performance Range')
                    ax2.set_ylabel('Count')
                    ax2.set_title('Performance Range Distribution')
            else:
                ax2.text(0.5, 0.5, 'No performance\nrange data', 
                       ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Consistency vs Method Sensitivity')
        else:
            ax2.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Consistency vs Method Sensitivity')
        
        ax2.grid(True, alpha=0.3)
        
        # Feature categorization (Bottom left)
        if len(plot_data) > 0:
            # Define difficulty categories based on median-split approach
            mean_perf_median = plot_data['mean_performance'].median()
            var_median = plot_data['performance_variability'].median()
            
            # Classification criteria:
            # EASY: High performance (>median) AND Low variability (<median) 
            # HARD: Low performance (≤median) OR High variability (≥median)
            # MEDIUM: Everything else
            difficulty_df['category'] = 'Medium'
            difficulty_df.loc[(difficulty_df['mean_performance'] > mean_perf_median) & 
                             (difficulty_df['performance_variability'] < var_median), 'category'] = 'Easy'
            difficulty_df.loc[(difficulty_df['mean_performance'] <= mean_perf_median) | 
                             (difficulty_df['performance_variability'] >= var_median), 'category'] = 'Hard'
            
            category_counts = difficulty_df['category'].value_counts()
            
            if len(category_counts) > 0:
                # Create pie chart with enhanced labels
                wedges, texts, autotexts = ax3.pie(category_counts.values, labels=None, autopct='%1.1f%%',
                                                  colors=[NATURE_COLORS['accent'], NATURE_COLORS['secondary'], NATURE_COLORS['neutral']],
                                                  startangle=90)
                
                # Add detailed legend
                legend_labels = []
                for cat in category_counts.index:
                    count = category_counts[cat]
                    if cat == 'Easy':
                        legend_labels.append(f'Easy (n={count})\nHigh perf. + Low var.')
                    elif cat == 'Hard':
                        legend_labels.append(f'Hard (n={count})\nLow perf. OR High var.')
                    else:
                        legend_labels.append(f'Medium (n={count})\nOther combinations')
                
                ax3.legend(wedges, legend_labels, title="Feature Categories", 
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
                ax3.set_title('Feature Difficulty Distribution\n(Based on Performance & Variability)')
                
                # Add explanation text
                ax3.text(0.5, -0.15, 
                        f'Criteria: Performance median = {mean_perf_median:.3f}, Variability median = {var_median:.3f}',
                        transform=ax3.transAxes, ha='center', fontsize=8, style='italic')
            else:
                ax3.text(0.5, 0.5, 'No categorization\npossible', 
                       ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Feature Difficulty Distribution')
        else:
            ax3.text(0.5, 0.5, 'No categorization\npossible', 
                   ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Feature Difficulty Distribution')
        
        # Method preference by difficulty (Bottom right)
        if len(difficulty_df) > 0 and 'category' in difficulty_df.columns:
            # Get all available methods dynamically
            available_methods = self._get_available_methods(data)
            unique_methods = list(set([(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods]))
            
            # Define colors for up to 4 methods
            method_colors = {
                'Method_1': NATURE_COLORS['primary'],
                'Method_2': NATURE_COLORS['secondary'], 
                'Method_3': NATURE_COLORS['accent'],
                'Method_4': NATURE_COLORS['alternative_1']
            }
            
            method_performance_by_category = []
            categories = ['Easy', 'Medium', 'Hard']
            
            # Calculate bar positions for multiple methods
            num_methods = len(unique_methods)
            if num_methods > 0:
                bar_width = 0.8 / num_methods  # Total width of 0.8, divided by number of methods
                
                legend_handles = []
                legend_labels = []
                
                for i, (method_key, method_name) in enumerate(unique_methods):
                    method_performances = []
                    
                    for category in categories:
                        if category in difficulty_df['category'].values:
                            cat_features = difficulty_df[difficulty_df['category'] == category]['feature']
                            if len(cat_features) > 0:
                                cat_metrics = feat_metrics[feat_metrics['feature'].isin(cat_features)]
                                method_performance = cat_metrics.groupby('method')['r'].mean()
                                performance = method_performance.get(method_key, np.nan)
                                method_performances.append(performance)
                            else:
                                method_performances.append(np.nan)
                        else:
                            method_performances.append(np.nan)
                    
                    # Create bar positions for this method
                    x_positions = [j + (i - num_methods/2 + 0.5) * bar_width for j in range(len(categories))]
                    
                    # Plot bars for this method (only non-NaN values)
                    valid_mask = ~np.isnan(method_performances)
                    if np.any(valid_mask):
                        color = method_colors.get(method_key, NATURE_COLORS['neutral'])
                        bars = ax4.bar([x_positions[j] for j in range(len(categories)) if valid_mask[j]], 
                                     [method_performances[j] for j in range(len(categories)) if valid_mask[j]], 
                                     width=bar_width, color=color, alpha=0.7, 
                                     label=method_name)
                        
                        # Store for legend
                        legend_handles.append(bars[0])
                        legend_labels.append(method_name)
                        method_performance_by_category.extend([p for p in method_performances if not np.isnan(p)])
                
                if len(method_performance_by_category) > 0:
                    ax4.set_xlabel('Feature Difficulty Category')
                    ax4.set_ylabel('Mean Performance')
                    ax4.set_title('Method Performance by Difficulty')
                    ax4.set_xticks([0, 1, 2])
                    ax4.set_xticklabels(['Easy', 'Medium', 'Hard'])
                    
                    # Add legend with proper handles and labels
                    if legend_handles:
                        ax4.legend(legend_handles, legend_labels, loc='best')
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                           ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title('Method Performance by Difficulty')
            else:
                ax4.text(0.5, 0.5, 'No methods available\nfor comparison', 
                       ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Method Performance by Difficulty')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                   ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Method Performance by Difficulty')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_21_temporal_performance_trends(self, data: AnalysisData):
        """Figure 21: Temporal performance trends (simulated based on feature ordering)"""
        print("Generating Figure 21: Temporal performance trends...")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Use feature ordering as a proxy for temporal/batch effects
        feat_metrics = data.metrics['feature_wise']
        
        if feat_metrics.empty:
            print("    ⚠️  No feature metrics available for temporal trends")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Temporal Performance Trends\n\nNo feature metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax.set_title('Performance Trends Analysis (Feature-Order Based)')
            ax.axis('off')
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Performance Trends Analysis (Feature-Order Based)', fontsize=14, fontweight='bold')
        
        for i, platform in enumerate(['Platform_A', 'Platform_B']):
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            
            if len(platform_data) == 0:
                # No data for this platform
                ax = ax1 if i == 0 else ax2
                ax_stab = ax3 if i == 0 else ax4
                
                ax.text(0.5, 0.5, f'No data available\nfor {platform_name_map[platform]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{platform_name_map[platform]} - Performance Trends')
                
                ax_stab.text(0.5, 0.5, f'No data available\nfor {platform_name_map[platform]}', 
                           ha='center', va='center', transform=ax_stab.transAxes, fontsize=12)
                ax_stab.set_title(f'{platform_name_map[platform]} - Stability Trends')
                continue
            
            # Sort features by name (as proxy for temporal order)
            platform_data = platform_data.sort_values('feature').reset_index(drop=True)
            
            # Calculate window size for this platform's data
            platform_size = len(platform_data)
            window_size = max(5, min(platform_size // 10, platform_size // 5)) if platform_size > 10 else max(1, platform_size // 2)
            if window_size >= platform_size:
                window_size = max(1, platform_size // 2)
            
            print(f"    Platform {platform}: {platform_size} data points, window size: {window_size}")
            
            ax = ax1 if i == 0 else ax2
            ax_stab = ax3 if i == 0 else ax4
            
            for method, color in [('Method_1', NATURE_COLORS['primary']), 
                                 ('Method_2', NATURE_COLORS['secondary'])]:
                method_data = platform_data[platform_data['method'] == method].copy()
                
                if len(method_data) == 0:
                    continue
                
                method_data = method_data.reset_index(drop=True)
                method_data['index'] = range(len(method_data))
                
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                
                # Only calculate rolling statistics if we have enough data
                if len(method_data) >= window_size:
                    # Calculate rolling statistics
                    method_data['rolling_mean'] = method_data['r'].rolling(window=window_size, center=True).mean()
                    method_data['rolling_std'] = method_data['r'].rolling(window=window_size, center=True).std()
                    
                    # Plot trend line (only non-NaN values)
                    valid_idx = ~method_data['rolling_mean'].isna()
                    if np.any(valid_idx):
                        ax.plot(method_data.loc[valid_idx, 'index'], method_data.loc[valid_idx, 'rolling_mean'], 
                               color=color, linewidth=2, label=f'{method_name} (trend)')
                        
                        # Add confidence band (only where we have both mean and std)
                        valid_std_idx = valid_idx & ~method_data['rolling_std'].isna()
                        if np.any(valid_std_idx):
                            ax.fill_between(method_data.loc[valid_std_idx, 'index'], 
                                          method_data.loc[valid_std_idx, 'rolling_mean'] - method_data.loc[valid_std_idx, 'rolling_std'],
                                          method_data.loc[valid_std_idx, 'rolling_mean'] + method_data.loc[valid_std_idx, 'rolling_std'],
                                          color=color, alpha=0.2)
                else:
                    # Too few data points for rolling analysis, just show mean line
                    mean_val = method_data['r'].mean()
                    ax.axhline(y=mean_val, color=color, linewidth=2, linestyle='-', 
                             label=f'{method_name} (mean={mean_val:.3f})')
                
                # Scatter plot of individual points
                ax.scatter(method_data['index'], method_data['r'], 
                         color=color, alpha=0.4, s=8)
                
                # Performance stability analysis
                if len(method_data) >= window_size:
                    # Calculate rolling variance
                    rolling_var = method_data['r'].rolling(window=window_size, center=True).var()
                    valid_var_idx = ~rolling_var.isna()
                    
                    if np.any(valid_var_idx):
                        ax_stab.plot(method_data.loc[valid_var_idx, 'index'], rolling_var[valid_var_idx], 
                                   color=color, linewidth=2, label=f'{method_name}')
                else:
                    # Show overall variance as horizontal line
                    var_val = method_data['r'].var()
                    if not np.isnan(var_val) and var_val > 0:
                        ax_stab.axhline(y=var_val, color=color, linewidth=2, linestyle='-',
                                      label=f'{method_name} (var={var_val:.4f})')
            
            ax.set_xlabel('Feature Index (Ordered)')
            ax.set_ylabel('Performance (r)')
            ax.set_title(f'{platform_name_map[platform]} - Performance Trends')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax_stab.set_xlabel('Feature Index (Ordered)')
            ax_stab.set_ylabel('Rolling Variance')
            try:
                ax_stab.set_yscale('log')  # Apply logarithmic y-axis if values are positive
            except ValueError:
                # If log scale fails (e.g., zero/negative values), use linear scale
                pass
            ax_stab.set_title(f'{platform_name_map[platform]} - Stability Trends')
            ax_stab.legend()
            ax_stab.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_22_shared_vs_unique_performance(self, data: AnalysisData):
        """Figure 22: Compare imputation performance for shared vs. platform-specific features"""
        print("Generating Figure 22: Shared vs. Platform-Specific Feature Performance...")
        
        # Check if we have overlapping features
        shared_features = getattr(self, 'overlapping_features', [])
        if not shared_features:
            return self._create_insufficient_data_figure(
                "Shared vs. Platform-Specific Feature Performance",
                "No overlapping features found between platforms"
            )
        
        print(f"    Analyzing {len(shared_features)} shared features")
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        feat_metrics = data.metrics['feature_wise'].copy()
        
        # Annotate each feature as 'Shared' or 'Platform-Specific'
        feat_metrics['feature_type'] = feat_metrics['feature'].apply(
            lambda x: 'Shared' if x in shared_features else 'Platform-Specific'
        )
        
        # Create figure with statistical tests
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Imputation Performance: Shared vs. Platform-Specific Features', 
                    fontsize=14, fontweight='bold')
        
        # Colors for feature types
        feature_type_colors = {'Shared': NATURE_COLORS['accent'], 'Platform-Specific': NATURE_COLORS['neutral']}
        
        # Statistical test function
        from scipy.stats import mannwhitneyu
        
        for i, platform in enumerate(['Platform_A', 'Platform_B']):
            platform_data = feat_metrics[feat_metrics['platform'] == platform]
            
            for j, method in enumerate(['Method_1', 'Method_2']):
                ax = [ax1, ax2, ax3, ax4][i*2 + j]
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                
                method_data = platform_data[platform_data['method'] == method]
                
                if len(method_data) > 0:
                    # Create violin plot
                    shared_vals = method_data[method_data['feature_type'] == 'Shared']['r'].values
                    specific_vals = method_data[method_data['feature_type'] == 'Platform-Specific']['r'].values
                    
                    # Only plot if we have data for both types with sufficient variance
                    if (len(shared_vals) > 2 and len(specific_vals) > 2 and 
                        np.std(shared_vals) > 1e-10 and np.std(specific_vals) > 1e-10):
                        violin_data = [shared_vals, specific_vals]
                        labels = ['Shared', 'Platform-Specific']
                        colors = [feature_type_colors['Shared'], feature_type_colors['Platform-Specific']]
                        
                        try:
                            violin_parts = ax.violinplot(violin_data, positions=[1, 2], widths=0.6)
                        except Exception as e:
                            print(f"    Warning: Violin plot failed for {platform}-{method}: {e}")
                            # Fall back to box plot
                            ax.boxplot(violin_data, positions=[1, 2], widths=0.6)
                            violin_parts = {'bodies': []}  # Empty for color loop
                        
                        # Color the violins (if they exist)
                        if 'bodies' in violin_parts and violin_parts['bodies']:
                            for pc, color in zip(violin_parts['bodies'], colors):
                                pc.set_facecolor(color)
                                pc.set_alpha(0.7)
                        
                        # Add swarm plot overlay
                        for k, (vals, label, color) in enumerate(zip(violin_data, labels, colors)):
                            x_pos = np.full(len(vals), k + 1, dtype=np.float64)
                            x_pos += np.random.normal(0, 0.05, len(vals))
                            ax.scatter(x_pos, vals, alpha=0.6, s=15, color='black', edgecolors='white', linewidth=0.5)
                        
                        # Perform statistical test
                        if len(shared_vals) >= 3 and len(specific_vals) >= 3:
                            stat, p_value = mannwhitneyu(shared_vals, specific_vals, alternative='two-sided')
                            significance = ""
                            if p_value < 0.001:
                                significance = "***"
                            elif p_value < 0.01:
                                significance = "**"
                            elif p_value < 0.05:
                                significance = "*"
                            else:
                                significance = "ns"
                            
                            # Add significance annotation
                            max_val = float(max(np.max(shared_vals), np.max(specific_vals)))
                            ax.text(1.5, max_val + 0.05, significance, ha='center', va='bottom', 
                                   fontsize=14, fontweight='bold')
                            
                            # Add p-value text
                            ax.text(0.05, 0.95, f'p = {p_value:.3f}', transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        # Add median lines
                        median_shared = float(np.median(shared_vals))
                        median_specific = float(np.median(specific_vals))
                        ax.hlines(median_shared, 0.7, 1.3, colors='red', linewidth=2)
                        ax.hlines(median_specific, 1.7, 2.3, colors='red', linewidth=2)
                        
                        ax.set_xticks([1, 2])
                        ax.set_xticklabels(labels)
                        ax.set_ylabel('Imputation Performance (r)')
                        
                        # Add summary statistics
                        ax.text(0.05, 0.85, f'Shared: n={len(shared_vals)}, μ={np.mean(shared_vals):.3f}', 
                               transform=ax.transAxes, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                        ax.text(0.05, 0.75, f'Specific: n={len(specific_vals)}, μ={np.mean(specific_vals):.3f}', 
                               transform=ax.transAxes, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    else:
                        ax.text(0.5, 0.5, f'Insufficient data\nShared: {len(shared_vals)}\nSpecific: {len(specific_vals)}', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                
                ax.set_title(f'{platform_name_map[platform]} - {method_name}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_23_imputation_vs_concordance(self, data: AnalysisData):
        """Figure 23: Imputation performance vs. cross-platform concordance"""
        print("Generating Figure 23: Imputation Performance vs. Cross-Platform Concordance...")
        
        # Check if we have cross-platform R² data
        if data.cross_platform_r2 is None or len(data.cross_platform_r2) == 0:
            return self._create_insufficient_data_figure(
                "Imputation vs. Cross-Platform Concordance",
                "No cross-platform R² metrics available (no overlapping features)"
            )
        
        print(f"    Analyzing {len(data.cross_platform_r2)} features with cross-platform data")
        
        # Get cross-platform correlation (already stored as r, not R²)
        cross_platform_r = data.cross_platform_r2.rename('cross_platform_r')
        
        # Merge with imputation performance metrics
        feat_metrics = data.metrics['feature_wise'].copy()
        merged_data = feat_metrics.merge(cross_platform_r.to_frame(), left_on='feature', right_index=True, how='inner')
        
        if merged_data.empty:
            return self._create_insufficient_data_figure(
                "Imputation vs. Cross-Platform Concordance",
                "No metrics found for overlapping features"
            )
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Imputation Performance vs. Cross-Platform Concordance\n"Are stable features easier to impute?"', 
                    fontsize=14, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        methods = ['Method_1', 'Method_2']
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary']]
        
        for i, platform in enumerate(platforms):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                # Filter data for this platform and method
                subset = merged_data[(merged_data['platform'] == platform) & 
                                   (merged_data['method'] == method)]
                
                if len(subset) > 3:  # Need at least a few points for meaningful analysis
                    x = subset['cross_platform_r']
                    y = subset['r']
                    
                    # Check for valid data ranges
                    if len(x.dropna()) < 3 or len(y.dropna()) < 3:
                        ax.text(0.5, 0.5, 'Insufficient valid data', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        continue
                    
                    # Scatter plot
                    ax.scatter(x, y, alpha=0.6, s=30, color=colors[j], 
                             edgecolors='black', linewidth=0.3)
                    
                    # Add regression line - only if we have valid range
                    if x.max() > x.min() and y.max() > y.min():
                        slope, intercept, r_val, p_val, std_err = linregress(x, y)
                        line_x = np.linspace(x.min(), x.max(), 100)
                        line_y = slope * line_x + intercept
                        ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                        
                        # Add R² from linear regression (only if regression was performed)
                        ax.text(0.05, 0.75, f'R² = {r_val**2:.3f}', transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Calculate Spearman correlation (always perform this)
                    try:
                        spearman_r, spearman_p = spearmanr(x, y)
                        
                        # Add correlation text
                        ax.text(0.05, 0.95, f'ρ = {spearman_r:.3f}', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        ax.text(0.05, 0.85, f'p = {spearman_p:.3f}', transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception:
                        # Handle correlation calculation errors
                        ax.text(0.05, 0.95, 'ρ = N/A', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Highlight extreme points
                    # High concordance, high performance
                    high_conc_high_perf = subset[(subset['cross_platform_r'] > x.quantile(0.8)) & 
                                                (subset['r'] > y.quantile(0.8))]
                    
                    # High concordance, low performance (unexpected)
                    high_conc_low_perf = subset[(subset['cross_platform_r'] > x.quantile(0.8)) & 
                                               (subset['r'] < y.quantile(0.2))]
                    
                    # Label interesting points
                    for _, row in high_conc_high_perf.head(3).iterrows():
                        ax.annotate(row['feature'][:8], 
                                  (row['cross_platform_r'], row['r']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=7, alpha=0.8,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                    
                    for _, row in high_conc_low_perf.head(2).iterrows():
                        ax.annotate(row['feature'][:8], 
                                  (row['cross_platform_r'], row['r']),
                                  xytext=(5, -15), textcoords='offset points',
                                  fontsize=7, alpha=0.8,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                else:
                    ax.text(0.5, 0.5, f'Insufficient data\n(n={len(subset)})', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                ax.set_title(f'{platform_name_map[platform]} - {method_name}')
                ax.set_xlabel('Cross-Platform Concordance (r)')
                ax.set_ylabel('Imputation Performance (r)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_23b_imputation_vs_concordance_density(self, data: AnalysisData):
        """Figure 23b: Imputation performance vs. cross-platform concordance with density plots"""
        print("Generating Figure 23b: Imputation Performance vs. Cross-Platform Concordance (density plots)...")
        
        # Check if we have cross-platform R² data
        if data.cross_platform_r2 is None or len(data.cross_platform_r2) == 0:
            return self._create_insufficient_data_figure(
                "Imputation vs. Cross-Platform Concordance (Density)",
                "No cross-platform R² metrics available (no overlapping features)"
            )
        
        print(f"    Analyzing {len(data.cross_platform_r2)} features with cross-platform data")
        
        # Get cross-platform correlation (already stored as r, not R²)
        cross_platform_r = data.cross_platform_r2.rename('cross_platform_r')
        
        # Merge with imputation performance metrics
        feat_metrics = data.metrics['feature_wise'].copy()
        merged_data = feat_metrics.merge(cross_platform_r.to_frame(), left_on='feature', right_index=True, how='inner')
        
        if merged_data.empty:
            return self._create_insufficient_data_figure(
                "Imputation vs. Cross-Platform Concordance (Density)",
                "No metrics found for overlapping features"
            )
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Imputation Performance vs. Cross-Platform Concordance (Density Plots)\n"Are stable features easier to impute?"', 
                    fontsize=14, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        methods = ['Method_1', 'Method_2']
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary']]
        
        for i, platform in enumerate(platforms):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                # Filter data for this platform and method
                subset = merged_data[(merged_data['platform'] == platform) & 
                                   (merged_data['method'] == method)]
                
                if len(subset) > 3:  # Need at least a few points for meaningful analysis
                    x = subset['cross_platform_r']
                    y = subset['r']
                    
                    # Check for valid data ranges
                    if len(x.dropna()) < 3 or len(y.dropna()) < 3:
                        ax.text(0.5, 0.5, 'Insufficient valid data', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        continue
                    
                    # Create 2D density plot
                    try:
                        # Create hexbin plot for density visualization
                        hb = ax.hexbin(x, y, gridsize=15, cmap='Blues', alpha=0.8)
                        cb = plt.colorbar(hb, ax=ax)
                        cb.set_label('Count', rotation=270, labelpad=15)
                    except:
                        # Fallback to scatter if hexbin fails
                        ax.scatter(x, y, alpha=0.6, s=30, color=colors[j], 
                                 edgecolors='black', linewidth=0.3)
                    
                    # Add regression line - only if we have valid range
                    if x.max() > x.min() and y.max() > y.min():
                        slope, intercept, r_val, p_val, std_err = linregress(x, y)
                        line_x = np.linspace(x.min(), x.max(), 100)
                        line_y = slope * line_x + intercept
                        ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                        
                        # Add R² from linear regression (only if regression was performed)
                        ax.text(0.05, 0.75, f'R² = {r_val**2:.3f}', transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Calculate Spearman correlation (always perform this)
                    try:
                        spearman_r, spearman_p = spearmanr(x, y)
                        
                        # Add correlation text
                        ax.text(0.05, 0.95, f'ρ = {spearman_r:.3f}', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        ax.text(0.05, 0.85, f'p = {spearman_p:.3f}', transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception:
                        # Handle correlation calculation errors
                        ax.text(0.05, 0.95, 'ρ = N/A', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Add marginal density plots
                    try:
                        from scipy.stats import gaussian_kde
                        
                        # Create marginal density plots on the edges
                        divider = plt.subplot_mosaic([['upper', 'upper', '.'],
                                                     ['main', 'main', 'right'],
                                                     ['main', 'main', 'right']], 
                                                    figure=fig, gridspec_kw={'width_ratios': [2, 2, 0.5], 
                                                                            'height_ratios': [0.5, 2, 2]})
                        
                        # This is a more complex approach, so let's keep it simpler for now
                        # and just create density contours on the main plot
                        
                        # Create density contours
                        if len(x) > 5 and len(y) > 5:
                            x_clean = x.dropna()
                            y_clean = y.dropna()
                            if len(x_clean) > 5 and len(y_clean) > 5:
                                # Create a grid for contour plotting
                                x_min, x_max = x_clean.min(), x_clean.max()
                                y_min, y_max = y_clean.min(), y_clean.max()
                                
                                if x_max > x_min and y_max > y_min:
                                    xx, yy = np.mgrid[x_min:x_max:.02, y_min:y_max:.02]
                                    positions = np.vstack([xx.ravel(), yy.ravel()])
                                    
                                    # Create combined data for KDE
                                    combined_data = np.vstack([x_clean, y_clean])
                                    if combined_data.shape[1] > 3:  # Need enough points for KDE
                                        kde = gaussian_kde(combined_data)
                                        density = np.reshape(kde(positions).T, xx.shape)
                                        
                                        # Add contour lines
                                        ax.contour(xx, yy, density, colors='white', alpha=0.6, linewidths=1)
                    except:
                        pass  # Skip density contours if they fail
                    
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                else:
                    ax.text(0.5, 0.5, f'Insufficient data\n(n={len(subset)})', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
                method_name = data.method1_name if method == 'Method_1' else data.method2_name
                ax.set_title(f'{platform_name_map[platform]} - {method_name}')
                ax.set_xlabel('Cross-Platform Concordance (r)')
                ax.set_ylabel('Imputation Performance (r)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_23c_imputation_concordance_difference(self, data: AnalysisData):
        """Figure 23c: Average performance difference between methods in different concordance groups"""
        print("Generating Figure 23c: Performance Difference by Concordance Groups...")
        
        # Check if we have cross-platform R² data
        if data.cross_platform_r2 is None or len(data.cross_platform_r2) == 0:
            return self._create_insufficient_data_figure(
                "Performance Difference by Concordance Groups",
                "No cross-platform R² metrics available (no overlapping features)"
            )
        
        print(f"    Analyzing {len(data.cross_platform_r2)} features with cross-platform data")
        
        # Get cross-platform correlation (already stored as r, not R²)
        cross_platform_r = data.cross_platform_r2.rename('cross_platform_r')
        
        # Merge with imputation performance metrics
        feat_metrics = data.metrics['feature_wise'].copy()
        merged_data = feat_metrics.merge(cross_platform_r.to_frame(), left_on='feature', right_index=True, how='inner')
        
        if merged_data.empty:
            return self._create_insufficient_data_figure(
                "Performance Difference by Concordance Groups",
                "No metrics found for overlapping features"
            )
        
        # Create concordance groups (Low: 0-0.33, Medium: 0.33-0.66, High: 0.66-1.0)
        merged_data['concordance_group'] = pd.cut(
            merged_data['cross_platform_r'],
            bins=[0, 0.33, 0.66, 1.001],
            labels=['Low\n(0.0-0.33)', 'Medium\n(0.33-0.66)', 'High\n(0.66-1.0)'],
            include_lowest=True,
            right=True
        )
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Calculate average performance for each method in each concordance group
        fig, axes = plt.subplots(2, 1, figsize=(6, 12))
        fig.suptitle('Average Performance Difference Between Methods by Concordance Groups\n"Do concordant features show different method preferences?"', 
                    fontsize=14, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        
        for i, platform in enumerate(platforms):
            ax = axes[i]
            
            # Filter data for this platform
            platform_data = merged_data[merged_data['platform'] == platform]
            
            if len(platform_data) == 0:
                ax.text(0.5, 0.5, f'No data for {platform_name_map[platform]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(platform_name_map[platform])
                continue
            
            # Calculate performance statistics for each concordance group
            concordance_groups = ['Low\n(0.0-0.33)', 'Medium\n(0.33-0.66)', 'High\n(0.66-1.0)']
            method1_means = []
            method1_stds = []
            method2_means = []
            method2_stds = []
            differences = []
            diff_stds = []
            group_counts = []
            
            for group in concordance_groups:
                group_data = platform_data[platform_data['concordance_group'] == group]
                
                if len(group_data) > 0:
                    # Get performance for each method
                    method1_data = group_data[group_data['method'] == 'Method_1']['r']
                    method2_data = group_data[group_data['method'] == 'Method_2']['r']
                    
                    # For difference calculation, we need paired data
                    method1_by_feature = group_data[group_data['method'] == 'Method_1'].set_index('feature')['r']
                    method2_by_feature = group_data[group_data['method'] == 'Method_2'].set_index('feature')['r']
                    
                    # Get features present in both methods
                    common_features = method1_by_feature.index.intersection(method2_by_feature.index)
                    
                    if len(common_features) > 0:
                        paired_method1 = method1_by_feature.loc[common_features]
                        paired_method2 = method2_by_feature.loc[common_features]
                        
                        # Calculate paired differences
                        paired_diff = paired_method1 - paired_method2
                        
                        method1_means.append(paired_method1.mean())
                        method1_stds.append(paired_method1.std() / np.sqrt(len(paired_method1)))
                        method2_means.append(paired_method2.mean())
                        method2_stds.append(paired_method2.std() / np.sqrt(len(paired_method2)))
                        differences.append(paired_diff.mean())
                        diff_stds.append(paired_diff.std() / np.sqrt(len(paired_diff)))
                        group_counts.append(len(common_features))
                    else:
                        method1_means.append(0)
                        method1_stds.append(0)
                        method2_means.append(0)
                        method2_stds.append(0)
                        differences.append(0)
                        diff_stds.append(0)
                        group_counts.append(0)
                else:
                    method1_means.append(0)
                    method1_stds.append(0)
                    method2_means.append(0)
                    method2_stds.append(0)
                    differences.append(0)
                    diff_stds.append(0)
                    group_counts.append(0)
            
            # Create bar plot
            x = np.arange(len(concordance_groups))
            width = 0.35
            
            # Plot bars for each method
            bars1 = ax.bar(x - width/2, method1_means, width, yerr=method1_stds,
                          label=data.method1_name, color=NATURE_COLORS['primary'], 
                          capsize=5, alpha=0.8)
            bars2 = ax.bar(x + width/2, method2_means, width, yerr=method2_stds,
                          label=data.method2_name, color=NATURE_COLORS['secondary'], 
                          capsize=5, alpha=0.8)
            
            # Store data for statistical testing
            method1_data_by_group = []
            method2_data_by_group = []
            
            for group in concordance_groups:
                group_data = platform_data[platform_data['concordance_group'] == group]
                if len(group_data) > 0:
                    method1_by_feature = group_data[group_data['method'] == 'Method_1'].set_index('feature')['r']
                    method2_by_feature = group_data[group_data['method'] == 'Method_2'].set_index('feature')['r']
                    common_features = method1_by_feature.index.intersection(method2_by_feature.index)
                    if len(common_features) > 0:
                        method1_data_by_group.append(method1_by_feature.loc[common_features].values)
                        method2_data_by_group.append(method2_by_feature.loc[common_features].values)
                    else:
                        method1_data_by_group.append(np.array([]))
                        method2_data_by_group.append(np.array([]))
                else:
                    method1_data_by_group.append(np.array([]))
                    method2_data_by_group.append(np.array([]))
            
            # Add sample counts above bars
            for j, count in enumerate(group_counts):
                if count > 0:
                    # Add count annotation
                    ax.text(j, max(method1_means[j], method2_means[j]) + 0.02, 
                           f'n={count}', ha='center', fontsize=9)
            
            # Add significance bars between methods within each concordance group
            from scipy.stats import mannwhitneyu, ttest_rel
            
            # Find the maximum y value for positioning significance bars
            max_y = max(max(method1_means), max(method2_means))
            y_increment = 0.05
            
            # Test between Method 1 and Method 2 within each concordance group
            for group_idx, group_name in enumerate(concordance_groups):
                if len(method1_data_by_group[group_idx]) > 0 and len(method2_data_by_group[group_idx]) > 0:
                    try:
                        # Since we have paired data (same features), use paired t-test
                        if len(method1_data_by_group[group_idx]) == len(method2_data_by_group[group_idx]) and len(method1_data_by_group[group_idx]) > 1:
                            _, p_value = ttest_rel(method1_data_by_group[group_idx], method2_data_by_group[group_idx])
                        else:
                            # Fall back to Mann-Whitney U test if not paired or too few samples
                            _, p_value = mannwhitneyu(method1_data_by_group[group_idx], method2_data_by_group[group_idx], alternative='two-sided')
                        
                        # Determine significance level
                        if p_value < 0.001:
                            sig_text = '***'
                        elif p_value < 0.01:
                            sig_text = '**'
                        elif p_value < 0.05:
                            sig_text = '*'
                        else:
                            sig_text = 'ns'
                        
                        # Draw significance bar between the two method bars for this group
                        y = max_y + 0.08
                        x1 = group_idx - width/2  # Method 1 position
                        x2 = group_idx + width/2  # Method 2 position
                        
                        # Draw a simple horizontal line
                        ax.plot([x1, x2], [y, y], 'k-', linewidth=0.8)
                        
                        # Add significance text
                        ax.text(group_idx, y + 0.01, sig_text, 
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
                    except Exception as e:
                        pass  # Skip if test fails
            
            # Formatting
            ax.set_xlabel('Cross-Platform Concordance Group', fontsize=11)
            ax.set_ylabel('Average Imputation Performance (r)', fontsize=11)
            ax.set_title(platform_name_map[platform], fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(concordance_groups)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust y-axis limit to accommodate significance bars
            ax.set_ylim(0, max_y + 0.15)
            
            # Add interpretation text
            if any(g > 0 for g in group_counts):
                max_diff_idx = np.argmax(np.abs(differences))
                if abs(differences[max_diff_idx]) > 0.05:
                    interpretation = f"Largest difference in {concordance_groups[max_diff_idx].replace(chr(10), ' ')} concordance group"
                    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, 
                           fontsize=9, style='italic',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def generate_figure_23c_percentage_increase(self, data: AnalysisData):
        """Figure 23c (percentage version): Percentage increase from Method 2 to Method 1 by concordance groups"""
        print("Generating Figure 23c (percentage): Percentage Increase by Concordance Groups...")
        
        # Check if we have cross-platform R² data
        if data.cross_platform_r2 is None or len(data.cross_platform_r2) == 0:
            return self._create_insufficient_data_figure(
                "Percentage Increase by Concordance Groups",
                "No cross-platform R² metrics available (no overlapping features)"
            )
        
        print(f"    Analyzing {len(data.cross_platform_r2)} features with cross-platform data")
        
        # Get cross-platform correlation (already stored as r, not R²)
        cross_platform_r = data.cross_platform_r2.rename('cross_platform_r')
        
        # Merge with imputation performance metrics
        feat_metrics = data.metrics['feature_wise'].copy()
        merged_data = feat_metrics.merge(cross_platform_r.to_frame(), left_on='feature', right_index=True, how='inner')
        
        if merged_data.empty:
            return self._create_insufficient_data_figure(
                "Percentage Increase by Concordance Groups",
                "No metrics found for overlapping features"
            )
        
        # Create concordance groups (Low: 0-0.33, Medium: 0.33-0.66, High: 0.66-1.0)
        merged_data['concordance_group'] = pd.cut(
            merged_data['cross_platform_r'],
            bins=[0, 0.33, 0.66, 1.001],
            labels=['Low\n(0.0-0.33)', 'Medium\n(0.33-0.66)', 'High\n(0.66-1.0)'],
            include_lowest=True,
            right=True
        )
        
        # Create platform name mapping
        platform_name_map = {
            'Platform_A': data.platform_a_name,
            'Platform_B': data.platform_b_name
        }
        
        # Calculate average performance for each method in each concordance group
        fig, axes = plt.subplots(2, 1, figsize=(6, 12))
        fig.suptitle('Average Percentage Increase from Method 2 to Method 1 by Concordance Groups\n"Do concordant features show different relative method preferences?"', 
                    fontsize=14, fontweight='bold')
        
        platforms = ['Platform_A', 'Platform_B']
        
        for i, platform in enumerate(platforms):
            ax = axes[i]
            
            # Filter data for this platform
            platform_data = merged_data[merged_data['platform'] == platform]
            
            if len(platform_data) == 0:
                ax.text(0.5, 0.5, f'No data for {platform_name_map[platform]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(platform_name_map[platform])
                continue
            
            # Calculate performance statistics for each concordance group
            concordance_groups = ['Low\n(0.0-0.33)', 'Medium\n(0.33-0.66)', 'High\n(0.66-1.0)']
            method1_means = []
            method1_stds = []
            method2_means = []
            method2_stds = []
            percentage_increases = []
            percentage_stds = []
            group_counts = []
            
            for group in concordance_groups:
                group_data = platform_data[platform_data['concordance_group'] == group]
                
                if len(group_data) > 0:
                    # Get performance for each method
                    method1_data = group_data[group_data['method'] == 'Method_1']['r']
                    method2_data = group_data[group_data['method'] == 'Method_2']['r']
                    
                    # For percentage calculation, we need paired data
                    method1_by_feature = group_data[group_data['method'] == 'Method_1'].set_index('feature')['r']
                    method2_by_feature = group_data[group_data['method'] == 'Method_2'].set_index('feature')['r']
                    
                    # Get features present in both methods
                    common_features = method1_by_feature.index.intersection(method2_by_feature.index)
                    
                    if len(common_features) > 0:
                        paired_method1 = method1_by_feature.loc[common_features]
                        paired_method2 = method2_by_feature.loc[common_features]
                        
                        # Calculate percentage increase from method2 to method1
                        # Handle cases where method2 might be 0 or very small
                        percentage_list = []
                        for feat in common_features:
                            m1_val = paired_method1.loc[feat]
                            m2_val = paired_method2.loc[feat]
                            if m2_val > 0.01:  # Only calculate percentage if method2 performance is meaningful
                                pct_increase = ((m1_val - m2_val) / m2_val) * 100
                                percentage_list.append(pct_increase)
                        
                        if percentage_list:
                            percentage_array = np.array(percentage_list)
                            method1_means.append(paired_method1.mean())
                            method1_stds.append(paired_method1.std() / np.sqrt(len(paired_method1)))
                            method2_means.append(paired_method2.mean())
                            method2_stds.append(paired_method2.std() / np.sqrt(len(paired_method2)))
                            percentage_increases.append(np.mean(percentage_array))
                            percentage_stds.append(np.std(percentage_array) / np.sqrt(len(percentage_array)))
                            group_counts.append(len(percentage_list))
                        else:
                            # No valid percentage calculations possible
                            method1_means.append(paired_method1.mean())
                            method1_stds.append(paired_method1.std() / np.sqrt(len(paired_method1)))
                            method2_means.append(paired_method2.mean())
                            method2_stds.append(paired_method2.std() / np.sqrt(len(paired_method2)))
                            percentage_increases.append(0)
                            percentage_stds.append(0)
                            group_counts.append(0)
                    else:
                        method1_means.append(0)
                        method1_stds.append(0)
                        method2_means.append(0)
                        method2_stds.append(0)
                        percentage_increases.append(0)
                        percentage_stds.append(0)
                        group_counts.append(0)
                else:
                    method1_means.append(0)
                    method1_stds.append(0)
                    method2_means.append(0)
                    method2_stds.append(0)
                    percentage_increases.append(0)
                    percentage_stds.append(0)
                    group_counts.append(0)
            
            # Create bar plot
            x = np.arange(len(concordance_groups))
            width = 0.35
            
            # Plot bars for each method
            bars1 = ax.bar(x - width/2, method1_means, width, yerr=method1_stds,
                          label=data.method1_name, color=NATURE_COLORS['primary'], 
                          capsize=5, alpha=0.8)
            bars2 = ax.bar(x + width/2, method2_means, width, yerr=method2_stds,
                          label=data.method2_name, color=NATURE_COLORS['secondary'], 
                          capsize=5, alpha=0.8)
            
            # Add percentage increase line plot on secondary axis
            ax2 = ax.twinx()
            line = ax2.plot(x, percentage_increases, 'ko-', linewidth=2, markersize=8, 
                           label=f'% Increase ({data.method1_name} vs {data.method2_name})')
            ax2.errorbar(x, percentage_increases, yerr=percentage_stds, fmt='none', ecolor='black', 
                        capsize=5, alpha=0.6)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel(f'Percentage Increase (%)\n({data.method1_name} relative to {data.method2_name})', fontsize=11)
            
            # Add sample counts above bars
            for j, (count, pct) in enumerate(zip(group_counts, percentage_increases)):
                if count > 0:
                    # Add count annotation
                    ax.text(j, max(method1_means[j], method2_means[j]) + 0.05, 
                           f'n={count}', ha='center', fontsize=9)
                    
                    # Add significance indicator if percentage increase is substantial
                    if abs(pct) > 10:  # Threshold for "substantial" percentage difference
                        significance = '**' if abs(pct) > 20 else '*'
                        ax2.text(j, pct + np.sign(pct) * 2, significance, 
                                ha='center', fontsize=12, fontweight='bold')
            
            # Formatting
            ax.set_xlabel('Cross-Platform Concordance Group', fontsize=11)
            ax.set_ylabel('Average Imputation Performance (r)', fontsize=11)
            ax.set_title(platform_name_map[platform], fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(concordance_groups)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add interpretation text
            if any(g > 0 for g in group_counts):
                max_pct_idx = np.argmax(np.abs(percentage_increases))
                if abs(percentage_increases[max_pct_idx]) > 10:
                    interpretation = f"Largest % increase in {concordance_groups[max_pct_idx].replace(chr(10), ' ')} concordance: {percentage_increases[max_pct_idx]:.1f}%"
                    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, 
                           fontsize=9, style='italic',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def generate_figure_24_shared_correlation_structure(self, data: AnalysisData):
        """Figure 24: Correlation structure analysis of shared features
        
        This analysis examines how well correlation patterns between features are preserved 
        across platforms. For shared features, we calculate:
        1. Feature-feature correlations within Platform A (samples as observations)
        2. Feature-feature correlations within Platform B (samples as observations)  
        3. The difference matrix showing how correlation patterns diverge
        
        Note: This is NOT cross-platform correlation of individual features, but rather
        correlation structure preservation analysis.
        """
        print("Generating Figure 24: Within-Platform Correlation Structure Preservation...")
        
        # Check if we have enough shared features
        shared_features = getattr(self, 'overlapping_features', [])
        if len(shared_features) < 5:
            return self._create_insufficient_data_figure(
                "Shared Feature Correlation Structure",
                f"Need at least 5 shared features for correlation analysis (found {len(shared_features)})"
            )
        
        print(f"    Analyzing correlation structure for {len(shared_features)} shared features")
        
        try:
            # Calculate correlation matrices for shared features
            # Features are rows, so we transpose to get samples as rows for correlation calculation
            subset_a = data.truth_a.loc[shared_features].T
            subset_b = data.truth_b.loc[shared_features].T
            
            # Check for sufficient valid data
            if subset_a.isnull().all().any() or subset_b.isnull().all().any():
                print("    Warning: Some features have all NaN values - excluding from correlation analysis")
                # Remove features that are all NaN
                valid_features_a = ~subset_a.isnull().all()
                valid_features_b = ~subset_b.isnull().all()
                valid_features = valid_features_a & valid_features_b
                
                if valid_features.sum() < 5:
                    return self._create_insufficient_data_figure(
                        "Shared Feature Correlation Structure",
                        f"Too few valid features after removing NaN columns ({valid_features.sum()} < 5)"
                    )
                
                subset_a = subset_a.loc[:, valid_features]
                subset_b = subset_b.loc[:, valid_features]
            
            corr_a = subset_a.corr(method='pearson')
            corr_b = subset_b.corr(method='pearson')
            
            # Ensure consistent ordering
            corr_b = corr_b.loc[corr_a.index, corr_a.columns]
            
            # Calculate the difference
            corr_diff = corr_a - corr_b
            
            # Use hierarchical clustering to order features based on Platform A's correlation structure
            try:
                from scipy.cluster.hierarchy import linkage, dendrogram
                from scipy.spatial.distance import pdist
            except ImportError:
                print("    Warning: scipy clustering not available - using original feature order")
                ordered_labels = corr_a.index
                corr_a_ord = corr_a
                corr_b_ord = corr_b
                corr_diff_ord = corr_diff
                clustering_performed = False
            else:
                clustering_performed = True
            
            if clustering_performed:
                # Convert correlation to distance for clustering
                distance_matrix = 1 - np.abs(corr_a)
                condensed_distances = pdist(distance_matrix.values)  # Pass .values to get ndarray
                linkage_matrix = linkage(condensed_distances, method='average')
                
                # Get the optimal leaf ordering
                dendro = dendrogram(linkage_matrix, labels=corr_a.index, no_plot=True)
                cluster_order = dendro['leaves']
                ordered_labels = corr_a.index[cluster_order]
                
                # Reorder all matrices
                corr_a_ord = corr_a.loc[ordered_labels, ordered_labels]
                corr_b_ord = corr_b.loc[ordered_labels, ordered_labels]
                corr_diff_ord = corr_diff.loc[ordered_labels, ordered_labels]
            
            # Create the figure
            fig = plt.figure(figsize=(10, 5))
            gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.5])
            
            # Heatmap 1: Platform A correlations
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(corr_a_ord.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            ax1.set_title(f'{data.platform_a_name}\nFeature Correlations', fontweight='bold')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Heatmap 2: Platform B correlations  
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(corr_b_ord.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            ax2.set_title(f'{data.platform_b_name}\nFeature Correlations', fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Heatmap 3: Difference (A - B)
            ax3 = fig.add_subplot(gs[0, 2])
            diff_max = np.abs(corr_diff_ord.values).max()
            im3 = ax3.imshow(corr_diff_ord.values, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='equal')
            ax3.set_title('Difference\n(A - B)', fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
            
            # Summary statistics panel
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.axis('off')
            
            # Calculate summary statistics
            try:
                # Remove NaN values before calculating conservation
                flat_a = corr_a.values.flatten()
                flat_b = corr_b.values.flatten()
                valid_mask = ~(np.isnan(flat_a) | np.isnan(flat_b))
                
                if np.sum(valid_mask) > 10:
                    corr_conservation = np.corrcoef(flat_a[valid_mask], flat_b[valid_mask])[0, 1]
                else:
                    corr_conservation = np.nan
                    
                mean_abs_diff = np.nanmean(np.abs(corr_diff.values))
            except Exception:
                corr_conservation = np.nan
                mean_abs_diff = np.nan
            
            # Identify most/least conserved correlations
            upper_tri_mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
            diff_upper = corr_diff.where(upper_tri_mask)
            
            # Most conserved (smallest absolute differences) - check for valid data
            abs_diff_values = np.abs(diff_upper.values)
            if not np.all(np.isnan(abs_diff_values)):
                most_conserved_idx = np.unravel_index(np.nanargmin(abs_diff_values), diff_upper.shape)
                most_conserved_pair = (diff_upper.index[most_conserved_idx[0]], diff_upper.columns[most_conserved_idx[1]])
                most_conserved_diff = diff_upper.iloc[most_conserved_idx]
            else:
                most_conserved_pair = ("N/A", "N/A")
                most_conserved_diff = np.nan
            
            # Least conserved (largest absolute differences) - check for valid data
            if not np.all(np.isnan(abs_diff_values)):
                least_conserved_idx = np.unravel_index(np.nanargmax(abs_diff_values), diff_upper.shape)
                least_conserved_pair = (diff_upper.index[least_conserved_idx[0]], diff_upper.columns[least_conserved_idx[1]])
                least_conserved_diff = diff_upper.iloc[least_conserved_idx]
            else:
                least_conserved_pair = ("N/A", "N/A")
                least_conserved_diff = np.nan
            
            # Add summary text
            summary_text = f"""
CORRELATION STRUCTURE ANALYSIS

Overall Conservation:
• Cross-platform correlation: {corr_conservation:.3f}
• Mean absolute difference: {mean_abs_diff:.3f}

Most Conserved Pair:
• {most_conserved_pair[0][:12]}
• {most_conserved_pair[1][:12]}
• Δr = {most_conserved_diff:.3f}

Least Conserved Pair:
• {least_conserved_pair[0][:12]}
• {least_conserved_pair[1][:12]}
• Δr = {least_conserved_diff:.3f}

Color Scale:
• Red: Positive correlation
• Blue: Negative correlation
• White: No correlation
            """
            
            ax4.text(0.05, 0.95, summary_text.strip(), transform=ax4.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            # Add colorbars
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Correlation (r)', rotation=270, labelpad=15)
            
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Correlation (r)', rotation=270, labelpad=15)
            
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Difference (Δr)', rotation=270, labelpad=15)
            
            fig.suptitle('Conservation of Feature Correlation Structure Across Platforms', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Add explanatory text at bottom
            fig.text(0.5, 0.02, 
                    'Note: Matrices show feature×feature correlations WITHIN each platform (not cross-platform correlations)', 
                    ha='center', fontsize=10, style='italic', weight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in correlation structure analysis: {e}")
            return self._create_insufficient_data_figure(
                "Shared Feature Correlation Structure",
                f"Error during analysis: {str(e)}"
            )
    
    def generate_figure_25_cross_platform_feature_correlation(self, data: AnalysisData):
        """Figure 25: Direct cross-platform correlation of shared features
        
        This analysis shows the correlation between Platform A and Platform B 
        measurements for each shared feature (feature by feature comparison).
        """
        print("Generating Figure 25: Cross-Platform Feature Correlations...")
        
        # Check if we have overlapping features
        shared_features = getattr(self, 'overlapping_features', [])
        if len(shared_features) < 3:
            return self._create_insufficient_data_figure(
                "Cross-Platform Feature Correlations",
                f"Need at least 3 shared features for cross-platform analysis (found {len(shared_features)})"
            )
        
        print(f"    Analyzing cross-platform correlations for {len(shared_features)} shared features")
        
        try:
            # Calculate cross-platform correlations for each shared feature
            cross_platform_corrs = []
            sample_counts = []
            feature_names = []
            
            for feature in shared_features:
                if feature in data.truth_a.index and feature in data.truth_b.index:
                    vals_a = data.truth_a.loc[feature].values
                    vals_b = data.truth_b.loc[feature].values
                    
                    # Find common samples (non-NaN in both)
                    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
                    if np.sum(mask) >= 3:  # Need at least 3 samples
                        clean_a = vals_a[mask]
                        clean_b = vals_b[mask]
                        
                        # Calculate correlation
                        r, p_val = pearsonr(clean_a, clean_b)
                        cross_platform_corrs.append(r)
                        sample_counts.append(np.sum(mask))
                        feature_names.append(feature)
            
            if len(cross_platform_corrs) == 0:
                return self._create_insufficient_data_figure(
                    "Cross-Platform Feature Correlations",
                    "No valid cross-platform correlations could be calculated"
                )
            
            # Convert to arrays for plotting
            cross_platform_corrs = np.array(cross_platform_corrs)
            sample_counts = np.array(sample_counts)
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f'Cross-Platform Feature Correlations\n{data.platform_a_name} vs {data.platform_b_name}', 
                        fontsize=14, fontweight='bold')
            
            # 1. Histogram of cross-platform correlations
            ax1.hist(cross_platform_corrs, bins=min(20, len(cross_platform_corrs)//2), 
                    alpha=0.7, color=NATURE_COLORS['primary'], edgecolor='black')
            ax1.axvline(np.median(cross_platform_corrs), color='red', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(cross_platform_corrs):.3f}')
            ax1.axvline(np.mean(cross_platform_corrs), color='orange', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(cross_platform_corrs):.3f}')
            ax1.set_xlabel('Cross-Platform Correlation (r)')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution of Cross-Platform Correlations')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Correlation vs sample count
            ax2.scatter(sample_counts, cross_platform_corrs, alpha=0.6, s=30, 
                       color=NATURE_COLORS['accent'], edgecolors='black', linewidth=0.3)
            
            # Add trend line
            if len(sample_counts) > 3:
                try:
                    slope, intercept, r_val, p_val, std_err = linregress(sample_counts, cross_platform_corrs)
                    line_x = np.linspace(sample_counts.min(), sample_counts.max(), 100)
                    line_y = slope * line_x + intercept
                    ax2.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                    ax2.text(0.05, 0.95, f'R² = {r_val**2:.3f}', transform=ax2.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
            
            ax2.set_xlabel('Number of Valid Samples')
            ax2.set_ylabel('Cross-Platform Correlation (r)')
            ax2.set_title('Correlation Quality vs Sample Size')
            ax2.grid(True, alpha=0.3)
            
            # 3. Top/bottom features ranking
            if len(feature_names) >= 6:
                # Sort by correlation
                sorted_indices = np.argsort(cross_platform_corrs)
                
                # Top and bottom features
                n_show = min(10, len(feature_names) // 2)
                top_indices = sorted_indices[-n_show:]
                bottom_indices = sorted_indices[:n_show]
                
                # Create bar plot
                y_pos = np.arange(len(top_indices) + len(bottom_indices))
                values = np.concatenate([cross_platform_corrs[bottom_indices], cross_platform_corrs[top_indices]])
                labels = [feature_names[i][:12] for i in bottom_indices] + [feature_names[i][:12] for i in top_indices]
                colors = ['lightcoral'] * len(bottom_indices) + ['lightgreen'] * len(top_indices)
                
                bars = ax3.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(labels, fontsize=8)
                ax3.set_xlabel('Cross-Platform Correlation (r)')
                ax3.set_title(f'Top/Bottom {n_show} Features')
                ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
                ax3.grid(True, alpha=0.3)
                
                # Add dividing line
                ax3.axhline(len(bottom_indices) - 0.5, color='black', linestyle='--', alpha=0.5)
            else:
                # Not enough features for ranking, show all
                y_pos = np.arange(len(feature_names))
                bars = ax3.barh(y_pos, cross_platform_corrs, 
                               color=NATURE_COLORS['primary'], alpha=0.7, edgecolor='black')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels([name[:12] for name in feature_names], fontsize=8)
                ax3.set_xlabel('Cross-Platform Correlation (r)')
                ax3.set_title('All Shared Features')
                ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
                ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics and interpretation
            ax4.axis('off')
            
            # Calculate summary stats
            high_corr = np.sum(cross_platform_corrs > 0.8)
            good_corr = np.sum(cross_platform_corrs > 0.6)
            poor_corr = np.sum(cross_platform_corrs < 0.3)
            
            summary_text = f"""
CROSS-PLATFORM CORRELATION SUMMARY

Total Shared Features: {len(cross_platform_corrs)}
Mean Correlation: {np.mean(cross_platform_corrs):.3f}
Median Correlation: {np.median(cross_platform_corrs):.3f}
Std Deviation: {np.std(cross_platform_corrs):.3f}

Performance Categories:
• High concordance (r > 0.8): {high_corr} features ({100*high_corr/len(cross_platform_corrs):.1f}%)
• Good concordance (r > 0.6): {good_corr} features ({100*good_corr/len(cross_platform_corrs):.1f}%)  
• Poor concordance (r < 0.3): {poor_corr} features ({100*poor_corr/len(cross_platform_corrs):.1f}%)

Range: [{np.min(cross_platform_corrs):.3f}, {np.max(cross_platform_corrs):.3f}]

Interpretation:
• High values indicate good cross-platform agreement
• Low values suggest platform-specific effects
• Negative values indicate systematic differences
            """
            
            ax4.text(0.05, 0.95, summary_text.strip(), transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in cross-platform correlation analysis: {e}")
            return self._create_insufficient_data_figure(
                "Cross-Platform Feature Correlations",
                f"Error during analysis: {str(e)}"
            )
    
    def _check_overlapping_features_for_analysis(self, analysis_name: str, 
                                                required_overlapping: int = 5) -> bool:
        """Check if we have sufficient overlapping features for cross-platform analysis"""
        overlapping_features = getattr(self, 'overlapping_features', [])
        
        if len(overlapping_features) < required_overlapping:
            print(f"    ⚠️  {analysis_name}: Insufficient overlapping features "
                  f"({len(overlapping_features)} < {required_overlapping}) - analysis skipped")
            return False
        
        return True
    
    def _create_insufficient_data_figure(self, title: str, message: str) -> plt.Figure:
        """Create a standardized figure for insufficient data cases"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, f'{title}\n\n{message}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    def _compute_feature_wise_metrics_spearman(self, data: AnalysisData) -> pd.DataFrame:
        """Compute feature-wise performance metrics using Spearman correlation"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        if not comparisons:
            raise ValueError("No method comparisons available for Spearman correlation computation")
        
        for platform, method, truth, imputed in comparisons:
            if truth.empty or imputed.empty:
                print(f"    Warning: Empty data for {method} on {platform}")
                continue
                
            if not truth.index.equals(imputed.index):
                raise ValueError(f"Feature indices don't match between truth and imputed data for {method} on {platform}")
            
            for feature in truth.index:
                try:
                    truth_vals = truth.loc[feature].values
                    imp_vals = imputed.loc[feature].values
                    
                    # Skip if all values are NaN
                    mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                        
                    truth_clean = truth_vals[mask]
                    imp_clean = imp_vals[mask]
                    
                    # Validate data before correlation computation
                    if len(truth_clean) != len(imp_clean):
                        raise ValueError(f"Mismatched data lengths for feature {feature}: truth={len(truth_clean)}, imputed={len(imp_clean)}")
                    
                    if np.std(truth_clean) == 0 or np.std(imp_clean) == 0:
                        print(f"    Warning: Zero variance for feature {feature} in {method} on {platform}, skipping Spearman correlation")
                        continue
                    
                    # Compute metrics - use Spearman correlation instead of Pearson
                    rho, rho_p = spearmanr(truth_clean, imp_clean)
                    
                    # Check for invalid correlation values
                    if np.isnan(rho) or np.isinf(rho):
                        print(f"    Warning: Invalid Spearman correlation for feature {feature} in {method} on {platform}: rho={rho}")
                        continue
                        
                    rmse = np.sqrt(mean_squared_error(truth_clean, imp_clean))
                    mae = mean_absolute_error(truth_clean, imp_clean)
                    bias = np.mean(imp_clean - truth_clean)
                    
                    results.append({
                        'feature': feature,
                        'platform': platform,
                        'method': method,
                        'rho': rho,  # Spearman's rho
                        'rho_pvalue': rho_p,
                        'rmse': rmse,
                        'mae': mae,
                        'bias': bias,
                        'n_samples': np.sum(mask)
                    })
                except Exception as e:
                    print(f"    Error processing feature {feature} for {method} on {platform}: {e}")
                    # Re-raise critical errors, continue for non-critical ones
                    if isinstance(e, (ValueError, KeyError)):
                        raise
                    continue
        
        if not results:
            raise ValueError("No valid Spearman correlation results computed - check input data quality")
        
        return pd.DataFrame(results)
    
    def _compute_sample_wise_metrics_spearman(self, data: AnalysisData) -> pd.DataFrame:
        """Compute sample-wise performance metrics using Spearman correlation"""
        results = []
        
        # Get all available methods dynamically
        available_methods = self._get_available_methods(data)
        comparisons = [(platform, method_key, truth, imputed) 
                      for method_key, method_name, platform, truth, imputed in available_methods]
        
        if not comparisons:
            raise ValueError("No method comparisons available for Spearman correlation computation")
        
        for platform, method, truth, imputed in comparisons:
            if truth.empty or imputed.empty:
                print(f"    Warning: Empty data for {method} on {platform}")
                continue
                
            if not truth.columns.equals(imputed.columns):
                raise ValueError(f"Sample indices don't match between truth and imputed data for {method} on {platform}")
            
            for sample in truth.columns:
                try:
                    truth_vals = truth[sample].values
                    imp_vals = imputed[sample].values
                    
                    # Skip if all values are NaN
                    mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                        
                    truth_clean = truth_vals[mask]
                    imp_clean = imp_vals[mask]
                    
                    # Validate data before correlation computation
                    if len(truth_clean) != len(imp_clean):
                        raise ValueError(f"Mismatched data lengths for sample {sample}: truth={len(truth_clean)}, imputed={len(imp_clean)}")
                    
                    if np.std(truth_clean) == 0 or np.std(imp_clean) == 0:
                        print(f"    Warning: Zero variance for sample {sample} in {method} on {platform}, skipping Spearman correlation")
                        continue
                    
                    # Compute metrics - use Spearman correlation instead of Pearson
                    rho, rho_p = spearmanr(truth_clean, imp_clean)
                    
                    # Check for invalid correlation values
                    if np.isnan(rho) or np.isinf(rho):
                        print(f"    Warning: Invalid Spearman correlation for sample {sample} in {method} on {platform}: rho={rho}")
                        continue
                        
                    rmse = np.sqrt(mean_squared_error(truth_clean, imp_clean))
                    mae = mean_absolute_error(truth_clean, imp_clean)
                    bias = np.mean(imp_clean - truth_clean)
                    
                    results.append({
                        'sample': sample,
                        'platform': platform,
                        'method': method,
                        'rho': rho,  # Spearman's rho
                        'rho_pvalue': rho_p,
                        'rmse': rmse,
                        'mae': mae,
                        'bias': bias,
                        'n_features': np.sum(mask)
                    })
                except Exception as e:
                    print(f"    Error processing sample {sample} for {method} on {platform}: {e}")
                    # Re-raise critical errors, continue for non-critical ones
                    if isinstance(e, (ValueError, KeyError)):
                        raise
                    continue
        
        if not results:
            raise ValueError("No valid Spearman correlation results computed - check input data quality")
        
        return pd.DataFrame(results)

    def _compute_cross_platform_rho(self, data: AnalysisData) -> pd.Series:
        """Compute cross-platform Spearman correlation (rho) between platforms for each overlapping feature"""
        results = {}
        
        # Only compute for overlapping features
        overlapping_features = getattr(self, 'overlapping_features', [])
        
        if not overlapping_features:
            print("    No overlapping features for cross-platform Spearman correlation calculation")
            return pd.Series(dtype=float)
        
        print(f"    Computing cross-platform Spearman correlation for {len(overlapping_features)} overlapping features")
        
        if data.truth_a.empty or data.truth_b.empty:
            raise ValueError("Truth data matrices are empty - cannot compute cross-platform Spearman correlation")
        
        for feature in overlapping_features:
            try:
                # Check if feature exists in both platforms
                if feature in data.truth_a.index and feature in data.truth_b.index:
                    vals_a = data.truth_a.loc[feature].values
                    vals_b = data.truth_b.loc[feature].values
                    
                    # Find common samples (non-NaN in both)
                    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
                    if np.sum(mask) < 3:
                        results[feature] = np.nan
                        continue
                        
                    clean_a = vals_a[mask]
                    clean_b = vals_b[mask]
                    
                    # Validate data quality
                    if len(clean_a) != len(clean_b):
                        raise ValueError(f"Mismatched data lengths for cross-platform feature {feature}")
                    
                    if np.std(clean_a) == 0 or np.std(clean_b) == 0:
                        print(f"    Warning: Zero variance for cross-platform feature {feature}, skipping")
                        results[feature] = np.nan
                        continue
                    
                    # Compute Spearman correlation rho (preserve sign)
                    rho, _ = spearmanr(clean_a, clean_b)
                    
                    # Validate result
                    if np.isnan(rho) or np.isinf(rho):
                        print(f"    Warning: Invalid cross-platform Spearman correlation for feature {feature}: rho={rho}")
                        results[feature] = np.nan
                    else:
                        results[feature] = rho
                else:
                    print(f"    Warning: Feature {feature} not found in both platforms")
                    results[feature] = np.nan
            except Exception as e:
                print(f"    Error computing cross-platform correlation for feature {feature}: {e}")
                # For cross-platform analysis, continue processing other features
                results[feature] = np.nan
        
        return pd.Series(results)
 
def main():
    """
    Main function to run comprehensive cross-platform imputation analysis.
    
    Parses command-line arguments, loads data, runs complete analysis pipeline,
    and generates publication-ready figures comparing imputation methods.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive Cross-Platform Proteomics Imputation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (2 methods):
    python compare_result.py \\
        --truth_a data/truth_platform_a.csv \\
        --truth_b data/truth_platform_b.csv \\
        --imp_a_m1 data/method1_platform_a.csv \\
        --imp_a_m2 data/method2_platform_a.csv \\
        --imp_b_m1 data/method1_platform_b.csv \\
        --imp_b_m2 data/method2_platform_b.csv \\
        --method1_name "Method 1" \\
        --method2_name "Method 2" \\
        --platform_a_name "Platform A" \\
        --platform_b_name "Platform B" \\
        --output_dir results

Example usage (4 methods):
    python compare_result.py \\
        --truth_a data/truth_platform_a.csv \\
        --truth_b data/truth_platform_b.csv \\
        --imp_a_m1 data/method1_platform_a.csv \\
        --imp_a_m2 data/method2_platform_a.csv \\
        --imp_a_m3 data/method3_platform_a.csv \\
        --imp_a_m4 data/method4_platform_a.csv \\
        --imp_b_m1 data/method1_platform_b.csv \\
        --imp_b_m2 data/method2_platform_b.csv \\
        --imp_b_m3 data/method3_platform_b.csv \\
        --imp_b_m4 data/method4_platform_b.csv \\
        --method1_name "Method 1" \\
        --method2_name "Method 2" \\
        --method3_name "Method 3" \\
        --method4_name "Method 4" \\
        --platform_a_name "Platform A" \\
        --platform_b_name "Platform B" \\
        --output_dir results \\
        --ppi_file data/ppi.txt \\
        --gri_file data/gri.txt
        
Network files format:
    PPI file: Tab-separated with columns GENE1, GENE2 (protein-protein interactions)
    GRI file: Tab-separated with columns FROM, TO (gene regulatory interactions)
        """
    )
    
    # Required arguments
    parser.add_argument('--truth_a', required=True, help='Truth file for platform A')
    parser.add_argument('--truth_b', required=True, help='Truth file for platform B')
    parser.add_argument('--imp_a_m1', required=True, help='Method 1 imputed platform A')
    parser.add_argument('--imp_a_m2', required=True, help='Method 2 imputed platform A')  
    parser.add_argument('--imp_b_m1', required=True, help='Method 1 imputed platform B')
    parser.add_argument('--imp_b_m2', required=True, help='Method 2 imputed platform B')
    parser.add_argument('--method1_name', required=True, help='Display name for method 1')
    parser.add_argument('--method2_name', required=True, help='Display name for method 2')
    parser.add_argument('--platform_a_name', required=True, help='Display name for platform A')
    parser.add_argument('--platform_b_name', required=True, help='Display name for platform B')
    
    # Optional additional methods (3 and 4)
    parser.add_argument('--imp_a_m3', help='Method 3 imputed platform A (optional)')
    parser.add_argument('--imp_a_m4', help='Method 4 imputed platform A (optional)')
    parser.add_argument('--imp_b_m3', help='Method 3 imputed platform B (optional)')
    parser.add_argument('--imp_b_m4', help='Method 4 imputed platform B (optional)')
    parser.add_argument('--method3_name', help='Display name for method 3 (optional)')
    parser.add_argument('--method4_name', help='Display name for method 4 (optional)')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='analysis_output', help='Output directory')
    parser.add_argument('--ppi_file', help='PPI network file (tab-separated, columns: GENE1, GENE2)')
    parser.add_argument('--gri_file', help='GRI network file (tab-separated, columns: FROM, TO)')
    parser.add_argument('--transpose', action='store_true', help='Transpose input files (use if rows=samples, columns=features)')
    
    # Phenotype analysis arguments
    parser.add_argument('--phenotype_file', help='Phenotype data file (CSV with sample IDs matching data files)')
    parser.add_argument('--binary_pheno', nargs='+', help='Column names for binary phenotypes')
    parser.add_argument('--continuous_pheno', nargs='+', help='Column names for continuous phenotypes')
    parser.add_argument('--pheno_sample_id_col', help='Column name for sample IDs in phenotype file (default: first column)')
    
    args = parser.parse_args()
    
    # Prepare file paths
    file_paths = {
        'truth_a': args.truth_a,
        'truth_b': args.truth_b,
        'imp_a_m1': args.imp_a_m1,
        'imp_a_m2': args.imp_a_m2,
        'imp_b_m1': args.imp_b_m1,
        'imp_b_m2': args.imp_b_m2,
    }
    
    # Add optional methods if provided
    if args.imp_a_m3:
        file_paths['imp_a_m3'] = args.imp_a_m3
    if args.imp_a_m4:
        file_paths['imp_a_m4'] = args.imp_a_m4
    if args.imp_b_m3:
        file_paths['imp_b_m3'] = args.imp_b_m3
    if args.imp_b_m4:
        file_paths['imp_b_m4'] = args.imp_b_m4
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer(args.output_dir)
    
    print("="*80)
    print("Cross-Platform Proteomics Imputation Analysis")
    print("="*80)
    print(f"Methods: {args.method1_name} vs {args.method2_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Git hash: {analyzer.git_hash}")
    print(f"Timestamp: {analyzer.timestamp}")
    print()
    
    try:
        # Load and validate data
        data = analyzer.load_and_validate_data(file_paths, args.method1_name, args.method2_name, args.platform_a_name, args.platform_b_name, args.method3_name, args.method4_name, args.transpose)
        print(f"Loaded data: Platform A: {data.truth_a.shape[0]} features, Platform B: {data.truth_b.shape[0]} features × {data.truth_a.shape[1]} samples")
        if data.groups is not None:
            print(f"Biological groups found: {data.groups.nunique()} unique groups")
        print()
        
        # Load phenotype data if provided
        if args.phenotype_file:
            print("\nLoading phenotype data...")
            data.phenotype_data = analyzer.load_phenotype_data(args.phenotype_file, args.pheno_sample_id_col)
            data.binary_pheno_cols = args.binary_pheno
            data.continuous_pheno_cols = args.continuous_pheno
            print()
        
        # Compute metrics
        data = analyzer.compute_all_metrics(data)
        print()
        
        # Generate figures
        generated_figures = analyzer.generate_all_figures(data)
        print()
        
        # Generate phenotype-dependent figures if phenotype data provided
        if data.phenotype_data is not None:
            phenotype_figures = analyzer.generate_phenotype_dependent_figures(data)
            generated_figures.extend(phenotype_figures)
            print()
        
        # Run network analysis if network files provided
        if args.ppi_file or args.gri_file:
            network_figures = analyzer.run_network_analysis(data, args.ppi_file, args.gri_file)
            generated_figures.extend(network_figures)
            print()
        
        print("Analysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")
        print(f"Generated {len(generated_figures)} figures")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 