#!/usr/bin/env python3
"""
Comprehensive Single-Platform Proteomics Imputation Analysis Script
Generates Nature-ready figures for comparing imputation methods on a single platform.
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
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.stats.multitest import multipletests
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
    Container for single-platform analysis data and metadata.
    
    Contains truth data, imputation results from multiple methods,
    computed metrics, and optional phenotype data for analysis.
    """
    truth_a: pd.DataFrame
    imp_a_m1: pd.DataFrame
    imp_a_m2: pd.DataFrame
    method1_name: str
    method2_name: str
    platform_a_name: str
    imp_a_m3: Optional[pd.DataFrame] = None
    imp_a_m4: Optional[pd.DataFrame] = None
    method3_name: Optional[str] = None
    method4_name: Optional[str] = None
    groups: Optional[pd.Series] = None
    metrics: Dict[str, pd.DataFrame] = None
    spearman_metrics: Dict[str, pd.DataFrame] = None
    phenotype_data: Optional[pd.DataFrame] = None
    binary_pheno_cols: Optional[List[str]] = None
    continuous_pheno_cols: Optional[List[str]] = None

class ComparativeAnalyzer:
    """Main class for comprehensive cross-platform imputation analysis"""
    
    def __init__(self, output_dir: str = "analysis_output", gender_col: Optional[str] = None, age_col: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.git_hash = self._get_git_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Optional covariate columns for phenotype association adjustment
        self.gender_col = gender_col
        self.age_col = age_col
        
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
                              platform_a_name: str,
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
        
        # Harmonize dimensions for Platform A methods
        print("  Harmonizing matrix dimensions for Platform A...")
        
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
        
        print(f"  Platform A features: {len(platform_a_common_features)}")
        print(f"  Platform A samples: {len(platform_a_common_samples)}")
        
        # Reorder and subset Platform A dataframes
        platform_a_features = sorted(list(platform_a_common_features))
        common_samples_sorted = sorted(list(platform_a_common_samples))
        
        print(f"  Harmonizing Platform A data to {len(platform_a_features)} features x {len(common_samples_sorted)} samples")
        for key in platform_a_keys:
            if key in data_frames:
                original_shape = data_frames[key].shape
                data_frames[key] = data_frames[key].loc[platform_a_features, common_samples_sorted]
                print(f"    {key}: {original_shape} -> {data_frames[key].shape}")
        
        # Convert to float64
        for key in data_frames:
            data_frames[key] = data_frames[key].astype(np.float64)
        
        # Validate data consistency across methods
        print("  Validating data consistency across methods...")
        self._validate_data_consistency(data_frames)
        
        # Create AnalysisData object
        analysis_data = AnalysisData(
            truth_a=data_frames['truth_a'],
            imp_a_m1=data_frames['imp_a_m1'],
            imp_a_m2=data_frames['imp_a_m2'],
            imp_a_m3=data_frames.get('imp_a_m3'),
            imp_a_m4=data_frames.get('imp_a_m4'),
            method1_name=method1_name,
            method2_name=method2_name,
            method3_name=method3_name,
            method4_name=method4_name,
            platform_a_name=platform_a_name,
            groups=groups
        )
        
        return analysis_data
    
    def _validate_data_consistency(self, data_frames: Dict[str, pd.DataFrame]):
        """Validate that all data matrices have consistent preprocessing"""
        
        # Group Platform A matrices for validation
        platform_a_matrices = {k: v for k, v in data_frames.items() if k.endswith('_a')}
        
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
    
    def _get_available_methods(self, data: AnalysisData) -> List[Tuple[str, str, str, pd.DataFrame, pd.DataFrame]]:
        """Get list of available methods and their data
        Returns: list of (method_key, method_name, platform, truth_data, imputed_data)
        """
        methods = []
        
        # Always include methods 1 and 2 for Platform A
        if data.imp_a_m1 is not None:
            methods.append(('Method_1', data.method1_name, 'Platform_A', data.truth_a, data.imp_a_m1))
        if data.imp_a_m2 is not None:
            methods.append(('Method_2', data.method2_name, 'Platform_A', data.truth_a, data.imp_a_m2))
        
        # Include methods 3 and 4 if available for Platform A
        if data.imp_a_m3 is not None and data.method3_name is not None:
            methods.append(('Method_3', data.method3_name, 'Platform_A', data.truth_a, data.imp_a_m3))
        if data.imp_a_m4 is not None and data.method4_name is not None:
            methods.append(('Method_4', data.method4_name, 'Platform_A', data.truth_a, data.imp_a_m4))
        
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
        
        # Spearman correlation metrics
        print("  Computing Spearman correlation metrics...")
        spearman_metrics = {}
        spearman_metrics['feature_wise'] = self._compute_feature_wise_metrics_spearman(data)
        spearman_metrics['sample_wise'] = self._compute_sample_wise_metrics_spearman(data)
        
        data.metrics = metrics
        data.spearman_metrics = spearman_metrics
        
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

    def _save_metrics_to_csv(self, data: AnalysisData):
        """Save computed metrics to CSV files"""
        data_dir = self.output_dir / "data"
        
        try:
            # Save Pearson metrics
            if data.metrics is None:
                raise ValueError("Pearson metrics not computed - run compute_all_metrics() first")
            
            data.metrics['feature_wise'].to_csv(data_dir / "feature_wise_metrics.csv", index=False)
            data.metrics['sample_wise'].to_csv(data_dir / "sample_wise_metrics.csv", index=False)
            
            # Save Spearman metrics if available
            if data.spearman_metrics is not None:
                if 'feature_wise' not in data.spearman_metrics or 'sample_wise' not in data.spearman_metrics:
                    raise ValueError("Incomplete Spearman metrics - missing feature_wise or sample_wise data")
                
                data.spearman_metrics['feature_wise'].to_csv(data_dir / "feature_wise_metrics_spearman.csv", index=False)
                data.spearman_metrics['sample_wise'].to_csv(data_dir / "sample_wise_metrics_spearman.csv", index=False)
            else:
                print("    Warning: Spearman metrics not available - they may not have been computed")
            
            print(f"  Metrics saved to {data_dir}")
            
        except Exception as e:
            print(f"    Error saving metrics: {e}")
            raise  # Re-raise to expose the error instead of hiding it
    
    def load_phenotype_data(self, phenotype_file: str, data: AnalysisData) -> AnalysisData:
        """Load and validate phenotype data"""
        print(f"Loading phenotype data from {phenotype_file}...")
        
        # Determine file type and load accordingly
        if phenotype_file.endswith('.csv'):
            phenotype_df = pd.read_csv(phenotype_file, index_col=0)
        else:  # Assume tab-delimited
            phenotype_df = pd.read_csv(phenotype_file, sep='\t', index_col=0)
        
        print(f"  Loaded phenotype data: {phenotype_df.shape[0]} samples × {phenotype_df.shape[1]} phenotypes")
        
        # Check for sample overlap with proteomics data
        proteomics_samples = set(data.truth_a.columns)
        phenotype_samples = set(phenotype_df.index)
        common_samples = proteomics_samples & phenotype_samples
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between proteomics and phenotype data!")
        
        print(f"  Found {len(common_samples)} common samples (out of {len(proteomics_samples)} proteomics samples)")
        
        # Filter phenotype data to common samples and align with proteomics data
        phenotype_df = phenotype_df.loc[list(common_samples)]
        phenotype_df = phenotype_df.reindex(data.truth_a.columns).dropna(how='all')
        
        # Validate specified phenotype columns exist
        all_pheno_cols = []
        if data.binary_pheno_cols:
            missing_binary = set(data.binary_pheno_cols) - set(phenotype_df.columns)
            if missing_binary:
                print(f"  WARNING: Binary phenotype columns not found: {missing_binary}")
                data.binary_pheno_cols = [col for col in data.binary_pheno_cols if col in phenotype_df.columns]
            all_pheno_cols.extend(data.binary_pheno_cols)
            print(f"  Binary phenotypes to analyze: {data.binary_pheno_cols}")
        
        if data.continuous_pheno_cols:
            missing_continuous = set(data.continuous_pheno_cols) - set(phenotype_df.columns)
            if missing_continuous:
                print(f"  WARNING: Continuous phenotype columns not found: {missing_continuous}")
                data.continuous_pheno_cols = [col for col in data.continuous_pheno_cols if col in phenotype_df.columns]
            all_pheno_cols.extend(data.continuous_pheno_cols)
            print(f"  Continuous phenotypes to analyze: {data.continuous_pheno_cols}")
        
        if not all_pheno_cols:
            print("  WARNING: No valid phenotype columns specified or found!")
            return data
        
        # If covariate columns are specified, ensure they exist and retain them for modeling
        covariate_cols = []
        if self.gender_col and self.gender_col in phenotype_df.columns:
            covariate_cols.append(self.gender_col)
        if self.age_col and self.age_col in phenotype_df.columns:
            covariate_cols.append(self.age_col)
        # Keep only specified phenotype columns (plus covariates if provided)
        keep_cols = list(dict.fromkeys(all_pheno_cols + covariate_cols)) if covariate_cols else all_pheno_cols
        phenotype_df = phenotype_df[keep_cols]
        
        # Store in data object
        data.phenotype_data = phenotype_df
        
        # Print summary statistics
        if data.binary_pheno_cols:
            print("\n  Binary phenotype summary:")
            for col in data.binary_pheno_cols:
                if col in phenotype_df.columns:
                    value_counts = phenotype_df[col].value_counts()
                    print(f"    {col}: {dict(value_counts)}")
        
        if data.continuous_pheno_cols:
            print("\n  Continuous phenotype summary:")
            for col in data.continuous_pheno_cols:
                if col in phenotype_df.columns:
                    print(f"    {col}: mean={phenotype_df[col].mean():.2f}, std={phenotype_df[col].std():.2f}")
        
        return data
    
    def calculate_binary_associations(self, data: AnalysisData) -> Dict[str, pd.DataFrame]:
        """Calculate associations between features and binary phenotypes using logistic regression"""
        if data.phenotype_data is None or not data.binary_pheno_cols:
            return {}
        
        print("Calculating binary phenotype associations...")
        results = {}
        
        # Get all available datasets (truth + imputed)
        datasets = {
            'Truth': data.truth_a,
            data.method1_name: data.imp_a_m1,
            data.method2_name: data.imp_a_m2
        }
        if data.imp_a_m3 is not None and data.method3_name is not None:
            datasets[data.method3_name] = data.imp_a_m3
        if data.imp_a_m4 is not None and data.method4_name is not None:
            datasets[data.method4_name] = data.imp_a_m4
        
        for phenotype in data.binary_pheno_cols:
            print(f"  Analyzing {phenotype}...")
            pheno_results = []
            
            # Get phenotype values aligned with proteomics samples
            y = data.phenotype_data[phenotype].dropna()
            
            for method_name, protein_data in datasets.items():
                # Align samples
                common_samples = list(set(y.index) & set(protein_data.columns))
                if len(common_samples) < 10:
                    print(f"    WARNING: Too few samples ({len(common_samples)}) for {method_name}")
                    continue
                
                y_aligned = y[common_samples]
                # Optional covariates (encode gender to 0/1 if provided)
                use_gender = self.gender_col is not None and self.gender_col in data.phenotype_data.columns and phenotype != self.gender_col
                use_age = self.age_col is not None and self.age_col in data.phenotype_data.columns and phenotype != self.age_col
                g_numeric = None
                a_numeric = None
                if use_gender:
                    g_series = data.phenotype_data.loc[common_samples, self.gender_col]
                    # Try numeric first
                    g_try = pd.to_numeric(g_series, errors='coerce')
                    if pd.unique(g_try.dropna()).size == 2:
                        g_numeric = g_try.values.astype(float)
                    else:
                        # Factorize strings into 0/1
                        cats = pd.Index(sorted(pd.unique(g_series.dropna().astype(str))))
                        if len(cats) == 2:
                            mapping = {cat: i for i, cat in enumerate(cats)}
                            g_numeric = g_series.astype(str).map(mapping).astype(float).values
                        else:
                            g_numeric = None
                if use_age:
                    a_series = data.phenotype_data.loc[common_samples, self.age_col]
                    a_numeric = pd.to_numeric(a_series, errors='coerce').values.astype(float)
                
                # Process each feature
                for feature in protein_data.index:
                    try:
                        # Get feature values
                        X = protein_data.loc[feature, common_samples].values.reshape(-1, 1)
                        
                        # Remove samples with missing values in feature/covariates
                        mask = ~np.isnan(X.flatten())
                        if use_gender and g_numeric is not None:
                            mask &= np.isfinite(g_numeric)
                        if use_age and a_numeric is not None:
                            mask &= np.isfinite(a_numeric)
                        X_clean = X[mask]
                        y_clean = y_aligned.values[mask]
                        
                        if len(np.unique(y_clean)) < 2 or len(X_clean) < 10:
                            continue
                        
                        # Build design matrix: feature + optional covariates
                        X_feature = StandardScaler().fit_transform(X_clean.reshape(-1, 1))
                        cov_cols = []
                        if use_gender and g_numeric is not None:
                            cov_cols.append(g_numeric[mask])
                        if use_age and a_numeric is not None:
                            a = a_numeric[mask]
                            a = StandardScaler().fit_transform(a.reshape(-1, 1)).flatten()
                            cov_cols.append(a)
                        if cov_cols:
                            X_design = np.column_stack([X_feature.flatten()] + cov_cols)
                        else:
                            X_design = X_feature
                        
                        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
                        lr.fit(X_design, y_clean)
                        
                        # Get coefficient and confidence interval
                        coef = lr.coef_[0][0]
                        
                        # Calculate standard error using inverse Fisher information
                        probs = lr.predict_proba(X_design)[:, 1]
                        weights = probs * (1 - probs)
                        X_aug = np.column_stack([X_design, np.ones(len(X_design))])
                        X_weighted = X_aug * np.sqrt(weights).reshape(-1, 1)
                        
                        try:
                            cov = np.linalg.inv(X_weighted.T @ X_weighted)
                            se = np.sqrt(np.diag(cov))[0]
                        except:
                            se = np.nan
                        
                        # Calculate odds ratio and CI
                        or_value = np.exp(coef)
                        ci_lower = np.exp(coef - 1.96 * se)
                        ci_upper = np.exp(coef + 1.96 * se)
                        
                        # Calculate p-value (Wald test)
                        z_score = coef / se if se > 0 else np.nan
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
                        
                        pheno_results.append({
                            'feature': feature,
                            'method': method_name,
                            'odds_ratio': or_value,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'p_value': p_value,
                            'n_samples': len(X_clean)
                        })
                        
                    except Exception as e:
                        # Skip features that cause convergence issues
                        continue
            
            if pheno_results:
                results[phenotype] = pd.DataFrame(pheno_results)
                print(f"    Completed {len(pheno_results)} associations")
        
        return results
    
    def calculate_continuous_associations(self, data: AnalysisData) -> Dict[str, pd.DataFrame]:
        """Calculate associations between features and continuous phenotypes using linear regression"""
        if data.phenotype_data is None or not data.continuous_pheno_cols:
            return {}
        
        print("Calculating continuous phenotype associations...")
        results = {}
        
        # Get all available datasets (truth + imputed)
        datasets = {
            'Truth': data.truth_a,
            data.method1_name: data.imp_a_m1,
            data.method2_name: data.imp_a_m2
        }
        if data.imp_a_m3 is not None and data.method3_name is not None:
            datasets[data.method3_name] = data.imp_a_m3
        if data.imp_a_m4 is not None and data.method4_name is not None:
            datasets[data.method4_name] = data.imp_a_m4
        
        for phenotype in data.continuous_pheno_cols:
            print(f"  Analyzing {phenotype}...")
            pheno_results = []
            
            # Get phenotype values aligned with proteomics samples
            y = data.phenotype_data[phenotype].dropna()
            
            for method_name, protein_data in datasets.items():
                # Align samples
                common_samples = list(set(y.index) & set(protein_data.columns))
                if len(common_samples) < 10:
                    print(f"    WARNING: Too few samples ({len(common_samples)}) for {method_name}")
                    continue
                
                y_aligned = y[common_samples]
                # Optional covariates (encode gender to 0/1 if provided)
                use_gender = self.gender_col is not None and self.gender_col in data.phenotype_data.columns and phenotype != self.gender_col
                use_age = self.age_col is not None and self.age_col in data.phenotype_data.columns and phenotype != self.age_col
                g_numeric = None
                a_numeric = None
                if use_gender:
                    g_series = data.phenotype_data.loc[common_samples, self.gender_col]
                    g_try = pd.to_numeric(g_series, errors='coerce')
                    if pd.unique(g_try.dropna()).size == 2:
                        g_numeric = g_try.values.astype(float)
                    else:
                        cats = pd.Index(sorted(pd.unique(g_series.dropna().astype(str))))
                        if len(cats) == 2:
                            mapping = {cat: i for i, cat in enumerate(cats)}
                            g_numeric = g_series.astype(str).map(mapping).astype(float).values
                        else:
                            g_numeric = None
                if use_age:
                    a_series = data.phenotype_data.loc[common_samples, self.age_col]
                    a_numeric = pd.to_numeric(a_series, errors='coerce').values.astype(float)
                
                # Process each feature
                for feature in protein_data.index:
                    try:
                        # Get feature values
                        X = protein_data.loc[feature, common_samples].values.reshape(-1, 1)
                        
                        # Remove samples with missing values in feature/covariates
                        mask = ~np.isnan(X.flatten())
                        if use_gender and g_numeric is not None:
                            mask &= np.isfinite(g_numeric)
                        if use_age and a_numeric is not None:
                            mask &= np.isfinite(a_numeric)
                        X_clean = X[mask]
                        y_clean = y_aligned.values[mask]
                        
                        if len(X_clean) < 10:
                            continue
                        
                        # Build design: feature + optional covariates
                        X_feature = StandardScaler().fit_transform(X_clean.reshape(-1, 1)).flatten()
                        y_scaled = StandardScaler().fit_transform(y_clean.reshape(-1, 1)).flatten()
                        cov_cols = []
                        if use_gender and g_numeric is not None:
                            cov_cols.append(g_numeric[mask])
                        if use_age and a_numeric is not None:
                            a = a_numeric[mask]
                            a = StandardScaler().fit_transform(a.reshape(-1, 1)).flatten()
                            cov_cols.append(a)
                        if cov_cols:
                            X = np.column_stack([X_feature] + cov_cols + [np.ones_like(X_feature)])
                        else:
                            X = np.column_stack([X_feature, np.ones_like(X_feature)])
                        
                        # OLS via normal equations
                        XtX = X.T @ X
                        XtX_inv = np.linalg.inv(XtX)
                        beta_vec = XtX_inv @ (X.T @ y_scaled)
                        beta = beta_vec[0]
                        y_pred = X @ beta_vec
                        residuals = y_scaled - y_pred
                        n = X.shape[0]
                        p = X.shape[1]
                        mse = (residuals @ residuals) / (n - p)
                        se = np.sqrt(np.maximum(mse * XtX_inv[0, 0], 0.0))
                        
                        # Calculate confidence interval
                        ci_lower = beta - 1.96 * se
                        ci_upper = beta + 1.96 * se
                        
                        # Calculate p-value (t-test)
                        t_stat = beta / se if se > 0 else np.nan
                        df = n - p
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if not np.isnan(t_stat) else np.nan
                        
                        # Calculate R-squared
                        ss_res = residuals @ residuals
                        ss_tot = ((y_scaled - y_scaled.mean()) @ (y_scaled - y_scaled.mean()))
                        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                        
                        pheno_results.append({
                            'feature': feature,
                            'method': method_name,
                            'beta': beta,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'p_value': p_value,
                            'r_squared': r_squared,
                            'n_samples': len(X_clean)
                        })
                        
                    except Exception as e:
                        # Skip features that cause issues
                        continue
            
            if pheno_results:
                results[phenotype] = pd.DataFrame(pheno_results)
                print(f"    Completed {len(pheno_results)} associations")
        
        return results
    
    def _calculate_association_mae_binary(self, association_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate mean absolute error of odds ratios for each method compared to Truth (only for significant Truth associations)"""
        mae_results = {}
        
        for phenotype, results_df in association_results.items():
            # Get Truth results as reference
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if len(truth_results) == 0:
                continue
            
            # Apply FDR and filter for significant Truth associations only (p_adj < 0.05)
            _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), method='fdr_bh', alpha=0.05)
            truth_results['p_adj'] = pvals_corrected
            significant_truth = truth_results[truth_results['p_adj'] < 0.05].copy()
            if len(significant_truth) == 0:
                continue
                
            significant_truth = significant_truth.set_index('feature')
            significant_features = significant_truth.index
            
            # Calculate MAE for each method
            for method in results_df['method'].unique():
                if method == 'Truth':
                    continue
                    
                method_results = results_df[results_df['method'] == method].copy()
                method_results = method_results.set_index('feature')
                
                # Find common features that are significant in Truth
                common_features = significant_features.intersection(method_results.index)
                if len(common_features) == 0:
                    continue
                
                # Get odds ratios directly (no log transformation) for significant features only
                truth_or = significant_truth.loc[common_features, 'odds_ratio']
                method_or = method_results.loc[common_features, 'odds_ratio']
                
                # Remove any infinite or NaN values
                mask = np.isfinite(truth_or) & np.isfinite(method_or) & (truth_or > 0) & (method_or > 0)
                if np.sum(mask) == 0:
                    continue
                
                # Calculate MAE
                mae = np.mean(np.abs(truth_or[mask] - method_or[mask]))
                
                if method not in mae_results:
                    mae_results[method] = []
                mae_results[method].append(mae)
        
        # Average across phenotypes
        final_mae = {}
        for method, values in mae_results.items():
            if values:
                final_mae[method] = np.mean(values)
        
        return final_mae
    
    def _calculate_association_mae_continuous(self, association_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate mean absolute error of beta coefficients for each method compared to Truth (only for significant Truth associations)"""
        mae_results = {}
        
        for phenotype, results_df in association_results.items():
            # Get Truth results as reference
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if len(truth_results) == 0:
                continue
            
            # Apply FDR and filter for significant Truth associations only (p_adj < 0.05)
            _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), method='fdr_bh', alpha=0.05)
            truth_results['p_adj'] = pvals_corrected
            significant_truth = truth_results[truth_results['p_adj'] < 0.05].copy()
            if len(significant_truth) == 0:
                continue
                
            significant_truth = significant_truth.set_index('feature')
            significant_features = significant_truth.index
            
            # Calculate MAE for each method
            for method in results_df['method'].unique():
                if method == 'Truth':
                    continue
                    
                method_results = results_df[results_df['method'] == method].copy()
                method_results = method_results.set_index('feature')
                
                # Find common features that are significant in Truth
                common_features = significant_features.intersection(method_results.index)
                if len(common_features) == 0:
                    continue
                
                # Get beta coefficients for significant features only
                truth_beta = significant_truth.loc[common_features, 'beta']
                method_beta = method_results.loc[common_features, 'beta']
                
                # Remove any NaN values
                mask = np.isfinite(truth_beta) & np.isfinite(method_beta)
                if np.sum(mask) == 0:
                    continue
                
                # Calculate MAE
                mae = np.mean(np.abs(truth_beta[mask] - method_beta[mask]))
                
                if method not in mae_results:
                    mae_results[method] = []
                mae_results[method].append(mae)
        
        # Average across phenotypes
        final_mae = {}
        for method, values in mae_results.items():
            if values:
                final_mae[method] = np.mean(values)
        
        return final_mae

    def _calculate_effect_correlation_binary(self, association_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate mean Spearman correlation of effect sizes (log OR) for each method vs Truth
        using only Truth-significant features after FDR per phenotype."""
        corr_lists: Dict[str, List[float]] = {}
        for phenotype, results_df in association_results.items():
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if truth_results.empty:
                continue
            # FDR on Truth
            _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), method='fdr_bh', alpha=0.05)
            truth_results['p_adj'] = pvals_corrected
            sig_truth = truth_results[truth_results['p_adj'] < 0.05].set_index('feature')
            if sig_truth.empty:
                continue
            # Use log(OR) as effect size
            truth_or = sig_truth['odds_ratio']
            # Filter invalid ORs
            valid_mask_truth = np.isfinite(truth_or) & (truth_or > 0)
            sig_truth = sig_truth[valid_mask_truth]
            for method in results_df['method'].unique():
                if method == 'Truth':
                    continue
                method_df = results_df[results_df['method'] == method].set_index('feature')
                common = sig_truth.index.intersection(method_df.index)
                if len(common) == 0:
                    continue
                or_truth = sig_truth.loc[common, 'odds_ratio']
                or_method = method_df.loc[common, 'odds_ratio']
                mask = np.isfinite(or_truth) & np.isfinite(or_method) & (or_truth > 0) & (or_method > 0)
                if mask.sum() == 0:
                    continue
                # Spearman on log(OR)
                r, _ = stats.spearmanr(np.log(or_truth[mask]), np.log(or_method[mask]))
                if method not in corr_lists:
                    corr_lists[method] = []
                if np.isfinite(r):
                    corr_lists[method].append(float(r))
        # Average across phenotypes
        return {m: float(np.mean(v)) for m, v in corr_lists.items() if len(v) > 0}

    def _calculate_effect_correlation_continuous(self, association_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate mean Spearman correlation of beta coefficients for each method vs Truth
        using only Truth-significant features after FDR per phenotype."""
        corr_lists: Dict[str, List[float]] = {}
        for phenotype, results_df in association_results.items():
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if truth_results.empty:
                continue
            # FDR on Truth
            _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), method='fdr_bh', alpha=0.05)
            truth_results['p_adj'] = pvals_corrected
            sig_truth = truth_results[truth_results['p_adj'] < 0.05].set_index('feature')
            if sig_truth.empty:
                continue
            beta_truth = sig_truth['beta']
            for method in results_df['method'].unique():
                if method == 'Truth':
                    continue
                method_df = results_df[results_df['method'] == method].set_index('feature')
                common = sig_truth.index.intersection(method_df.index)
                if len(common) == 0:
                    continue
                beta_m = method_df.loc[common, 'beta']
                mask = np.isfinite(beta_truth.loc[common]) & np.isfinite(beta_m)
                if mask.sum() == 0:
                    continue
                r, _ = stats.spearmanr(beta_truth.loc[common][mask], beta_m[mask])
                if method not in corr_lists:
                    corr_lists[method] = []
                if np.isfinite(r):
                    corr_lists[method].append(float(r))
        return {m: float(np.mean(v)) for m, v in corr_lists.items() if len(v) > 0}
    
    def generate_figure_26_comprehensive_method_comparison(self, data: AnalysisData):
        """Figure 26: Comprehensive comparison of all available methods (mean and median correlations)"""
        print("Generating Figure 26: Comprehensive method comparison...")
        
        # Check how many methods are available
        available_methods = self._get_available_methods(data)
        unique_methods = list(set([(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods]))
        
        if len(unique_methods) < 2:
            print("    ⚠️  Less than 2 methods available - skipping comprehensive comparison")
            return self._create_insufficient_data_figure(
                'Comprehensive Method Comparison',
                f'Only {len(unique_methods)} methods available. Need at least 2 for comparison.'
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
        fig.suptitle(f'Method Comparison: {data.platform_a_name}', fontsize=16, fontweight='bold')
        
        # Colors for up to 4 methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # 1. Feature-wise performance comparison
        if not feat_metrics.empty:
            # Only Platform A data
            platform_a_feat = feat_metrics[feat_metrics['platform'] == 'Platform_A']
            feat_summary = platform_a_feat.groupby('method')['r'].agg(['mean', 'median', 'std']).reset_index()
            
            methods = feat_summary['method'].unique()
            x = np.arange(len(methods))
            
            # Mean values
            means = feat_summary['mean'].values
            stds = feat_summary['std'].values
            medians = feat_summary['median'].values
            
            # Create bars for mean and median
            width = 0.35
            bars1 = ax1.bar(x - width/2, means, width, label='Mean', 
                           color=colors[0], alpha=0.8)
            bars2 = ax1.bar(x + width/2, medians, width, label='Median', 
                           color=colors[1], alpha=0.8)
            
            # Get method display names
            method_names = [next((name for key, name in unique_methods if key == method), method) 
                           for method in methods]
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Feature-wise Correlation (r)')
            ax1.set_title('Feature-wise Performance', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(method_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, 'No feature-wise metrics available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Feature-wise Performance')
        
        # 2. Sample-wise performance comparison
        if not sample_metrics.empty:
            # Only Platform A data
            platform_a_samp = sample_metrics[sample_metrics['platform'] == 'Platform_A']
            samp_summary = platform_a_samp.groupby('method')['r'].agg(['mean', 'median', 'std']).reset_index()
            
            methods_sample = samp_summary['method'].unique()
            x_sample = np.arange(len(methods_sample))
            
            # Mean values
            means_sample = samp_summary['mean'].values
            stds_sample = samp_summary['std'].values
            medians_sample = samp_summary['median'].values
            
            # Create bars for mean and median
            width = 0.35
            bars1_s = ax2.bar(x_sample - width/2, means_sample, width, label='Mean', 
                             color=colors[0], alpha=0.8)
            bars2_s = ax2.bar(x_sample + width/2, medians_sample, width, label='Median', 
                             color=colors[1], alpha=0.8)
            
            # Get method display names
            method_names_sample = [next((name for key, name in unique_methods if key == method), method) 
                                  for method in methods_sample]
            
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Sample-wise Correlation (r)')
            ax2.set_title('Sample-wise Performance', fontweight='bold')
            ax2.set_xticks(x_sample)
            ax2.set_xticklabels(method_names_sample, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No sample-wise metrics available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Sample-wise Performance')
        
        # Improve layout so rotated xtick labels are visible
        try:
            fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
        except Exception:
            plt.tight_layout()
        return fig
    
    def generate_figure_27_comprehensive_method_comparison_spearman(self, data: AnalysisData):
        """Figure 27: Comprehensive comparison of all available methods using Spearman correlation (mean and median correlations)"""
        print("Generating Figure 27: Comprehensive method comparison (Spearman correlation)...")
        
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
        
        if len(unique_methods) < 3:
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
        fig.suptitle(f'Method Comparison (Spearman): {data.platform_a_name}', fontsize=16, fontweight='bold')
        
        # Colors for up to 4 methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Create platform name mapping
        platform_name_map = {'Platform_A': data.platform_a_name}
        
        # 1. Feature-wise performance comparison
        if not feat_metrics.empty:
            # Only Platform A data
            platform_a_feat = feat_metrics[feat_metrics['platform'] == 'Platform_A']
            feat_summary = platform_a_feat.groupby('method')['rho'].agg(['mean', 'median', 'std']).reset_index()
            
            methods = feat_summary['method'].unique()
            x = np.arange(len(methods))
            
            # Mean values
            means = feat_summary['mean'].values
            medians = feat_summary['median'].values
            
            # Create bars for mean and median
            width = 0.35
            bars1 = ax1.bar(x - width/2, means, width, label='Mean', 
                           color=colors[0], alpha=0.8)
            bars2 = ax1.bar(x + width/2, medians, width, label='Median', 
                           color=colors[1], alpha=0.8)
            
            # Get method display names
            method_names = [next((name for key, name in unique_methods if key == method), method) 
                           for method in methods]
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Feature-wise Correlation (ρ)')
            ax1.set_title('Feature-wise Performance', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(method_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, 'No feature-wise Spearman metrics available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Feature-wise Performance')
        
        # 2. Sample-wise performance comparison
        if not sample_metrics.empty:
            # Only Platform A data
            platform_a_samp = sample_metrics[sample_metrics['platform'] == 'Platform_A']
            samp_summary = platform_a_samp.groupby('method')['rho'].agg(['mean', 'median', 'std']).reset_index()
            
            methods_sample = samp_summary['method'].unique()
            x_sample = np.arange(len(methods_sample))
            
            # Mean values
            means_sample = samp_summary['mean'].values
            medians_sample = samp_summary['median'].values
            
            # Create bars for mean and median
            width = 0.35
            bars1_s = ax2.bar(x_sample - width/2, means_sample, width, label='Mean', 
                             color=colors[0], alpha=0.8)
            bars2_s = ax2.bar(x_sample + width/2, medians_sample, width, label='Median', 
                             color=colors[1], alpha=0.8)
            
            # Get method display names
            method_names_sample = [next((name for key, name in unique_methods if key == method), method) 
                                  for method in methods_sample]
            
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Sample-wise Correlation (ρ)')
            ax2.set_title('Sample-wise Performance', fontweight='bold')
            ax2.set_xticks(x_sample)
            ax2.set_xticklabels(method_names_sample, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No sample-wise Spearman metrics available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Sample-wise Performance')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_28b_phenotype_summary_binary(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 28b: Binary phenotype association summaries (MAE and effect correlation) with square panels"""
        if not association_results:
            print("  No binary phenotype associations to summarize")
            return None
        
        mae_results = self._calculate_association_mae_binary(association_results)
        corr_results = self._calculate_effect_correlation_binary(association_results)
        
        # Method colors: Truth=black, Method1=red, Method2=blue, Method3=green, Method4=yellow/orange
        method_colors = {
            'Truth': 'black',
            data.method1_name: NATURE_COLORS['primary'],
            data.method2_name: NATURE_COLORS['secondary']
        }
        if data.method3_name:
            method_colors[data.method3_name] = NATURE_COLORS['accent']
        if data.method4_name:
            method_colors[data.method4_name] = NATURE_COLORS['highlight']
        
        fig, (ax_mae, ax_corr) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Binary Phenotype Associations: MAE and Effect Correlation', fontsize=16, fontweight='bold')
        
        # MAE panel
        if mae_results:
            methods = list(mae_results.keys())
            mae_values = [mae_results[m] for m in methods]
            ax_mae.bar(range(len(methods)), mae_values,
                       color=[method_colors.get(m, 'gray') for m in methods],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
            for i, (method, value) in enumerate(zip(methods, mae_values)):
                ax_mae.text(i, value + 0.01 * max(mae_values), f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            ax_mae.set_xlabel('Method')
            ax_mae.set_ylabel('Mean Absolute Error')
            ax_mae.set_title('MAE of Odds Ratios vs Truth (FDR<0.05)', fontweight='bold')
            ax_mae.set_xticks(range(len(methods)))
            ax_mae.set_xticklabels(methods, rotation=45, ha='right')
            ax_mae.grid(True, alpha=0.3, axis='y')
        else:
            ax_mae.text(0.5, 0.5, 'No MAE data available', transform=ax_mae.transAxes, ha='center', va='center')
            ax_mae.set_title('MAE of Odds Ratios vs Truth (FDR<0.05)')
        
        # Correlation panel
        if corr_results:
            methods_c = list(corr_results.keys())
            corr_values = [corr_results[m] for m in methods_c]
            ax_corr.bar(range(len(methods_c)), corr_values,
                        color=[method_colors.get(m, 'gray') for m in methods_c],
                        alpha=0.7, edgecolor='black', linewidth=0.5)
            for i, (method, value) in enumerate(zip(methods_c, corr_values)):
                ax_corr.text(i, value + 0.01 * max(corr_values), f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            ax_corr.set_xlabel('Method')
            ax_corr.set_ylabel('Spearman r (log OR)')
            ax_corr.set_title('Effect Correlation vs Truth (FDR<0.05)', fontweight='bold')
            ax_corr.set_xticks(range(len(methods_c)))
            ax_corr.set_xticklabels(methods_c, rotation=45, ha='right')
            ax_corr.grid(True, alpha=0.3, axis='y')
        else:
            ax_corr.text(0.5, 0.5, 'No correlation data available', transform=ax_corr.transAxes, ha='center', va='center')
            ax_corr.set_title('Effect Correlation vs Truth (FDR<0.05)')
        
        # Make panels square without distorting data aspect
        ax_mae.set_box_aspect(1)
        ax_corr.set_box_aspect(1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return fig

    def generate_figure_29b_phenotype_summary_continuous(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 29b: Continuous phenotype association summaries (MAE and effect correlation) with square panels"""
        if not association_results:
            print("  No continuous phenotype associations to summarize")
            return None
        
        mae_results = self._calculate_association_mae_continuous(association_results)
        corr_results = self._calculate_effect_correlation_continuous(association_results)
        
        # Method colors: Truth=black, Method1=red, Method2=blue, Method3=green, Method4=yellow/orange
        method_colors = {
            'Truth': 'black',
            data.method1_name: NATURE_COLORS['primary'],
            data.method2_name: NATURE_COLORS['secondary']
        }
        if data.method3_name:
            method_colors[data.method3_name] = NATURE_COLORS['accent']
        if data.method4_name:
            method_colors[data.method4_name] = NATURE_COLORS['highlight']
        
        fig, (ax_mae, ax_corr) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Continuous Phenotype Associations: MAE and Effect Correlation', fontsize=16, fontweight='bold')
        
        # MAE panel
        if mae_results:
            methods = list(mae_results.keys())
            mae_values = [mae_results[m] for m in methods]
            ax_mae.bar(range(len(methods)), mae_values,
                       color=[method_colors.get(m, 'gray') for m in methods],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
            for i, (method, value) in enumerate(zip(methods, mae_values)):
                ax_mae.text(i, value + 0.01 * max(mae_values), f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            ax_mae.set_xlabel('Method')
            ax_mae.set_ylabel('Mean Absolute Error')
            ax_mae.set_title('MAE of Beta Coefficients vs Truth (FDR<0.05)', fontweight='bold')
            ax_mae.set_xticks(range(len(methods)))
            ax_mae.set_xticklabels(methods, rotation=45, ha='right')
            ax_mae.grid(True, alpha=0.3, axis='y')
        else:
            ax_mae.text(0.5, 0.5, 'No MAE data available', transform=ax_mae.transAxes, ha='center', va='center')
            ax_mae.set_title('MAE of Beta Coefficients vs Truth (FDR<0.05)')
        
        # Correlation panel
        if corr_results:
            methods_c = list(corr_results.keys())
            corr_values = [corr_results[m] for m in methods_c]
            ax_corr.bar(range(len(methods_c)), corr_values,
                        color=[method_colors.get(m, 'gray') for m in methods_c],
                        alpha=0.7, edgecolor='black', linewidth=0.5)
            for i, (method, value) in enumerate(zip(methods_c, corr_values)):
                ax_corr.text(i, value + 0.01 * max(corr_values), f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            ax_corr.set_xlabel('Method')
            ax_corr.set_ylabel('Spearman r (beta)')
            ax_corr.set_title('Effect Correlation vs Truth (FDR<0.05)', fontweight='bold')
            ax_corr.set_xticks(range(len(methods_c)))
            ax_corr.set_xticklabels(methods_c, rotation=45, ha='right')
            ax_corr.grid(True, alpha=0.3, axis='y')
        else:
            ax_corr.text(0.5, 0.5, 'No correlation data available', transform=ax_corr.transAxes, ha='center', va='center')
            ax_corr.set_title('Effect Correlation vs Truth (FDR<0.05)')
        
        # Make panels square without distorting data aspect
        ax_mae.set_box_aspect(1)
        ax_corr.set_box_aspect(1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return fig

    def generate_figure_28_phenotype_forest_plots_binary(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 28: Forest plots for binary phenotype associations (forest plots only)"""
        if not association_results:
            print("  No binary phenotype associations to plot")
            return None
        
        print("Generating Figure 28: Binary phenotype forest plots...")
        
        # Determine subplot layout - forest plots only
        n_phenotypes = len(association_results)
        total_subplots = n_phenotypes
        n_cols = min(3, total_subplots)  # Max 3 columns for better readability
        n_rows = (total_subplots + n_cols - 1) // n_cols
        
        # Double the height per row to accommodate top 20 features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 15*n_rows))
        # Ensure axes is always a list
        if total_subplots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Binary Phenotype Associations: Forest Plots', fontsize=16, fontweight='bold')
        
        # Method colors: Truth=black, Method1=red, Method2=blue, Method3=green, Method4=yellow/orange
        method_colors = {
            'Truth': 'black',
            data.method1_name: NATURE_COLORS['primary'],
            data.method2_name: NATURE_COLORS['secondary']
        }
        if data.method3_name:
            method_colors[data.method3_name] = NATURE_COLORS['accent']
        if data.method4_name:
            method_colors[data.method4_name] = NATURE_COLORS['highlight']
        
        # Generate forest plots for each phenotype
        for idx, (phenotype, results_df) in enumerate(association_results.items()):
            ax = axes[idx]
            
            # Select top features by significance (lowest p-values from Truth)
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if len(truth_results) > 0:
                # Apply FDR correction
                _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), 
                                                        method='fdr_bh', alpha=0.05)
                truth_results['p_adj'] = pvals_corrected
                
                # Select top 20 features by adjusted p-value
                top_features = truth_results.nsmallest(20, 'p_adj')['feature'].tolist()
            else:
                # If no truth results, use results from first available method
                top_features = results_df['feature'].unique()[:20]
            
            # Filter results to top features
            plot_data = results_df[results_df['feature'].isin(top_features)].copy()
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, f'No significant associations for {phenotype}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(phenotype)
                continue
            
            # Prepare data for forest plot
            y_positions = {}
            current_y = 0
            
            for feature in top_features:
                feature_data = plot_data[plot_data['feature'] == feature]
                if len(feature_data) > 0:
                    y_positions[feature] = current_y
                    current_y += 2  # spacing per feature remains the same
            
            # Plot forest plot
            for method in ['Truth'] + [m for m in plot_data['method'].unique() if m != 'Truth']:
                method_data = plot_data[plot_data['method'] == method]
                
                for _, row in method_data.iterrows():
                    if row['feature'] in y_positions:
                        y_pos = y_positions[row['feature']]
                        
                        # Offset for multiple methods
                        method_offset = list(method_colors.keys()).index(method) * 0.3 - 0.6
                        
                        # Plot confidence interval
                        ax.plot([row['ci_lower'], row['ci_upper']], 
                               [y_pos + method_offset, y_pos + method_offset],
                               color=method_colors.get(method, 'gray'), linewidth=2, alpha=0.7)
                        
                        # Plot point estimate as circle: solid = significant, empty = non-significant
                        is_sig = (row['p_value'] < 0.05) if pd.notna(row['p_value']) else False
                        if is_sig:
                            ax.scatter(row['odds_ratio'], y_pos + method_offset,
                                       s=60, marker='o', facecolors=method_colors.get(method, 'gray'),
                                       edgecolors='black', linewidth=0.5)
                        else:
                            ax.scatter(row['odds_ratio'], y_pos + method_offset,
                                       s=60, marker='o', facecolors='none',
                                       edgecolors=method_colors.get(method, 'gray'), linewidth=1.2)
            
            # Add vertical line at OR=1
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            
            # Labels and formatting
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in y_positions.keys()], 
                              fontsize=10)
            ax.set_xlabel('Odds Ratio (95% CI)')
            ax.set_ylabel('Feature')
            ax.set_title(phenotype, fontweight='bold')
            # Linear scale for odds ratios
            ax.grid(True, alpha=0.3, axis='x')
            # Use ScalarFormatter with plain style for tick labels
            from matplotlib.ticker import ScalarFormatter
            sfmt = ScalarFormatter(useMathText=False)
            sfmt.set_scientific(False)
            ax.xaxis.set_major_formatter(sfmt)
            
            # Set y-axis limits to accommodate the increased spacing
            if y_positions:
                ax.set_ylim(-1, max(y_positions.values()) + 1)
            
            # Add legend
            legend_elements = []
            for method, color in method_colors.items():
                if method in plot_data['method'].values:
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=method))
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        # Hide unused subplots and ensure labels are not clipped
        for idx in range(total_subplots, len(axes)):
            axes[idx].axis('off')
        for ax in axes[:total_subplots]:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_29_phenotype_forest_plots_continuous(self, data: AnalysisData, association_results: Dict[str, pd.DataFrame]):
        """Figure 29: Forest plots for continuous phenotype associations (forest plots only)"""
        if not association_results:
            print("  No continuous phenotype associations to plot")
            return None
        
        print("Generating Figure 29: Continuous phenotype forest plots...")
        
        # Determine subplot layout - add one more subplot for MAE
        n_phenotypes = len(association_results)
        total_subplots = n_phenotypes
        n_cols = min(3, total_subplots)  # Max 3 columns for better readability
        n_rows = (total_subplots + n_cols - 1) // n_cols
        
        # Double the height per row to accommodate top 20 features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 15*n_rows))
        # Ensure axes is always a list
        if total_subplots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Continuous Phenotype Associations: Forest Plots', fontsize=16, fontweight='bold')
        
        # Method colors: Truth=black, Method1=red, Method2=blue, Method3=green, Method4=yellow/orange
        method_colors = {
            'Truth': 'black',
            data.method1_name: NATURE_COLORS['primary'],
            data.method2_name: NATURE_COLORS['secondary']
        }
        if data.method3_name:
            method_colors[data.method3_name] = NATURE_COLORS['accent']
        if data.method4_name:
            method_colors[data.method4_name] = NATURE_COLORS['highlight']
        
        # Generate forest plots for each phenotype
        for idx, (phenotype, results_df) in enumerate(association_results.items()):
            ax = axes[idx]
            
            # Select top features by significance (lowest p-values from Truth)
            truth_results = results_df[results_df['method'] == 'Truth'].copy()
            if len(truth_results) > 0:
                # Apply FDR correction
                _, pvals_corrected, _, _ = multipletests(truth_results['p_value'].fillna(1), 
                                                        method='fdr_bh', alpha=0.05)
                truth_results['p_adj'] = pvals_corrected
                
                # Select top 20 features by adjusted p-value
                top_features = truth_results.nsmallest(20, 'p_adj')['feature'].tolist()
            else:
                # If no truth results, use results from first available method
                top_features = results_df['feature'].unique()[:20]
            
            # Filter results to top features
            plot_data = results_df[results_df['feature'].isin(top_features)].copy()
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, f'No significant associations for {phenotype}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(phenotype)
                continue
            
            # Prepare data for forest plot
            y_positions = {}
            current_y = 0
            
            for feature in top_features:
                feature_data = plot_data[plot_data['feature'] == feature]
                if len(feature_data) > 0:
                    y_positions[feature] = current_y
                    current_y += 2  # spacing per feature remains the same
            
            # Plot forest plot
            for method in ['Truth'] + [m for m in plot_data['method'].unique() if m != 'Truth']:
                method_data = plot_data[plot_data['method'] == method]
                
                for _, row in method_data.iterrows():
                    if row['feature'] in y_positions:
                        y_pos = y_positions[row['feature']]
                        
                        # Offset for multiple methods
                        method_offset = list(method_colors.keys()).index(method) * 0.3 - 0.6
                        
                        # Plot confidence interval
                        ax.plot([row['ci_lower'], row['ci_upper']], 
                               [y_pos + method_offset, y_pos + method_offset],
                               color=method_colors.get(method, 'gray'), linewidth=2, alpha=0.7)
                        
                        # Plot point estimate as circle: solid = significant, empty = non-significant
                        is_sig = (row['p_value'] < 0.05) if pd.notna(row['p_value']) else False
                        if is_sig:
                            ax.scatter(row['beta'], y_pos + method_offset,
                                       s=60, marker='o', facecolors=method_colors.get(method, 'gray'),
                                       edgecolors='black', linewidth=0.5)
                        else:
                            ax.scatter(row['beta'], y_pos + method_offset,
                                       s=60, marker='o', facecolors='none',
                                       edgecolors=method_colors.get(method, 'gray'), linewidth=1.2)
            
            # Add vertical line at beta=0
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Labels and formatting
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in y_positions.keys()], 
                              fontsize=10)
            ax.set_xlabel('Beta Coefficient (95% CI)')
            ax.set_ylabel('Feature')
            ax.set_title(phenotype, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Set y-axis limits to accommodate the increased spacing
            if y_positions:
                ax.set_ylim(-1, max(y_positions.values()) + 1)
            
            # Add legend
            legend_elements = []
            for method, color in method_colors.items():
                if method in plot_data['method'].values:
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=method))
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
            
            # Add R² annotation for each method
            r2_text = []
            for method in ['Truth'] + [m for m in plot_data['method'].unique() if m != 'Truth']:
                method_data = plot_data[plot_data['method'] == method]
                if len(method_data) > 0:
                    mean_r2 = method_data['r_squared'].mean()
                    r2_text.append(f"{method}: R²={mean_r2:.3f}")
            
            ax.text(0.02, 0.98, '\n'.join(r2_text), transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(total_subplots, len(axes)):
            axes[idx].axis('off')
        for ax in axes[:total_subplots]:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        
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
        
        # Save PNG (for quick viewing)
        fig.savefig(png_path, format='png', dpi=png_dpi, **kwargs)
        
        print(f"  Figure saved: {fig_path}")
    
    def generate_figure_1_feature_r_scatter(self, data: AnalysisData):
        """Figure 1: Feature-wise r scatter with marginals and mean feature value coloring"""
        print("Generating Figure 1: Feature-wise r scatter...")
        
        # Get feature-wise correlations for Platform A only
        feat_metrics = data.metrics['feature_wise']
        
        # Check if we have any feature metrics
        if feat_metrics.empty:
            print("    No feature metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No feature metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Feature-wise Performance Scatter Plot')
            return fig
        
        # Platform A data only
        platform_a_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        if len(platform_a_data) == 0:
            print("    No data for Platform A - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No feature data available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Feature-wise Performance Scatter Plot')
            return fig
        
        # Get aligned method data for Platform A with mean feature values
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
        
        m1_r, m2_r, feature_means = get_aligned_data_with_means(platform_a_data, data.truth_a)
        
        # Create single figure with marginals
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(3, 3, figure=fig, 
                     left=0.1, right=0.85,
                     bottom=0.1, top=0.9,
                     hspace=0.05, wspace=0.05)
        
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_top = fig.add_subplot(gs[0, :-1])
        ax_right = fig.add_subplot(gs[1:, -1])
        
        if len(m1_r) > 0 and len(m2_r) > 0 and len(m1_r) == len(m2_r):
            # Main scatter plot with color coding by mean feature values
            valid_means = ~np.isnan(feature_means)
            
            if np.any(valid_means) and np.sum(valid_means) > 1:
                means_for_color = feature_means[valid_means]
                colorbar_label = 'Mean Feature Value'
                
                scatter = ax_main.scatter(m2_r[valid_means], m1_r[valid_means], 
                                        c=means_for_color, alpha=0.7, s=6, 
                                        cmap='viridis', edgecolors='white', linewidth=0.3)
                
                # Add colorbar
                cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
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
            ax_main.text(0.05, 0.95, f'ρ = {spearman_r:.3f}', 
                        transform=ax_main.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Top marginal (Method 2 on X-axis)
            ax_top.hist(m2_r, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], density=True)
            ax_top.set_xlim(ax_main.get_xlim())
            ax_top.set_xticks([])
            ax_top.set_ylabel('Density', fontsize=8)
            
            # Right marginal (Method 1 on Y-axis)
            ax_right.hist(m1_r, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                         orientation='horizontal', density=True)
            ax_right.set_ylim(ax_main.get_ylim())
            ax_right.set_yticks([])
            ax_right.set_xlabel('Density', fontsize=8)
            
            # Find and label extreme features
            if len(platform_a_data) > 0:
                # Get aligned data for both methods
                m1_data = platform_a_data[platform_a_data['method'] == 'Method_1'][['feature', 'r']].set_index('feature')
                m2_data = platform_a_data[platform_a_data['method'] == 'Method_2'][['feature', 'r']].set_index('feature')
                
                # Merge to ensure alignment
                merged_r = m1_data.join(m2_data, lsuffix='_m1', rsuffix='_m2').dropna()
                
                if len(merged_r) > 0:
                    # Calculate differences (m1 - m2 since Y is m1, X is m2)
                    merged_r['diff'] = merged_r['r_m1'] - merged_r['r_m2']
                    
                    # Find extreme features
                    top_features = merged_r.nlargest(2, 'diff')  # Top 2 improved
                    bottom_features = merged_r.nsmallest(2, 'diff')  # Top 2 worsened
                    
                    # Label extreme points
                    for feature_name, row in pd.concat([top_features, bottom_features]).iterrows():
                        ax_main.annotate(feature_name[:10], 
                                       (row['r_m2'], row['r_m1']),
                                       fontsize=6, alpha=0.8,
                                       xytext=(5, 5), textcoords='offset points')
        else:
            # No data available
            ax_main.text(0.5, 0.5, f'No data available\nfor {data.platform_a_name}', 
                       ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
            ax_top.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_top.transAxes)
            ax_right.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_right.transAxes)
        
        # Labels and title
        ax_main.set_xlabel(f'r ({data.method2_name})')
        ax_main.set_ylabel(f'r ({data.method1_name})')
        ax_main.set_title(f'{data.platform_a_name} - Feature-wise Performance')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_2_sample_r_scatter(self, data: AnalysisData):
        """Figure 2: Sample-wise r scatter with marginals"""
        print("Generating Figure 2: Sample-wise r scatter...")
        
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
        
        # Only process Platform A data
        platform_a_data = samp_metrics[samp_metrics['platform'] == 'Platform_A']
        
        if len(platform_a_data) == 0:
            print("    No data for Platform A - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No sample data available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Sample-wise Performance Scatter Plot')
            return fig
        
        # Get aligned method data for Platform A
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
        
        m1_r, m2_r = get_aligned_sample_data(platform_a_data)
        
        # Create figure with marginal plots
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(3, 3, figure=fig, 
                     left=0.1, right=0.9,
                     bottom=0.1, top=0.9,
                     hspace=0.05, wspace=0.05)
        
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_top = fig.add_subplot(gs[0, :-1])
        ax_right = fig.add_subplot(gs[1:, -1])
        
        if len(m1_r) > 0 and len(m2_r) > 0 and len(m1_r) == len(m2_r):
            # Main scatter plot (X=m2, Y=m1)
            ax_main.scatter(m2_r, m1_r, alpha=0.6, s=6, 
                           color=NATURE_COLORS['accent'])
            
            # Add 1:1 line
            min_r = min(np.min(m1_r), np.min(m2_r))
            max_r = max(np.max(m1_r), np.max(m2_r))
            ax_main.plot([min_r, max_r], [min_r, max_r], 
                        'k--', alpha=0.5, linewidth=1)
            
            # Add correlation coefficient
            spearman_r, spearman_p = spearmanr(m1_r, m2_r)
            ax_main.text(0.05, 0.95, f'ρ = {spearman_r:.3f}', 
                        transform=ax_main.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Top marginal (Method 2 on X-axis)
            ax_top.hist(m2_r, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], 
                       density=True)
            ax_top.set_xlim(ax_main.get_xlim())
            ax_top.set_xticks([])
            ax_top.set_ylabel('Density', fontsize=8)
            
            # Right marginal (Method 1 on Y-axis)
            ax_right.hist(m1_r, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                         orientation='horizontal', density=True)
            ax_right.set_ylim(ax_main.get_ylim())
            ax_right.set_yticks([])
            ax_right.set_xlabel('Density', fontsize=8)
        else:
            # No data available
            ax_main.text(0.5, 0.5, f'No data available\nfor {data.platform_a_name}', 
                       ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
            ax_top.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_top.transAxes)
            ax_right.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_right.transAxes)
        
        # Labels and title
        ax_main.set_xlabel(f'r ({data.method2_name})')
        ax_main.set_ylabel(f'r ({data.method1_name})')
        ax_main.set_title(f'{data.platform_a_name} - Sample-wise Performance')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_3_r_distribution_ridge(self, data: AnalysisData):
        """Figure 3: Distribution ridge/violin of feature-wise and sample-wise r with comprehensive statistics"""
        print("Generating Figure 3: Feature-wise and sample-wise r distributions...")
        
        feat_metrics = data.metrics['feature_wise']
        sample_metrics = data.metrics['sample_wise']
        
        # Check if we have any metrics
        if feat_metrics.empty and sample_metrics.empty:
            print("    No metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Distributions')
            return fig
        
        # Use 2x2 layout for comprehensive analysis (feature-wise and sample-wise)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'{data.platform_a_name} - Comprehensive Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Get both feature-wise and sample-wise data for Platform A
        platform_data_feat = feat_metrics[feat_metrics['platform'] == 'Platform_A'] if not feat_metrics.empty else pd.DataFrame()
        platform_data_samp = sample_metrics[sample_metrics['platform'] == 'Platform_A'] if not sample_metrics.empty else pd.DataFrame()
        
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Function to create violin plot with statistics
        def create_enhanced_violin_plot(ax, platform_data, metric_name, plot_title):
            methods_data = []
            labels = []
            
            if len(platform_data) > 0:
                # Collect all available methods
                method_mapping = {
                    'Method_1': data.method1_name,
                    'Method_2': data.method2_name,
                    'Method_3': data.method3_name,
                    'Method_4': data.method4_name
                }
                
                for method_key, method_name in method_mapping.items():
                    if method_name is not None:  # Only include methods that have names
                        method_subset = platform_data[platform_data['method'] == method_key]
                        if len(method_subset) > 0:
                            method_data = method_subset['r'].values
                            methods_data.append(method_data)
                            labels.append(method_name)
                
                if len(methods_data) > 0 and all(len(md) > 0 for md in methods_data):
                    # Create violin plot with enhanced features
                    positions = range(1, len(methods_data)+1)
                    violin_parts = ax.violinplot(methods_data, positions=positions, widths=0.6, 
                                                showmeans=True, showmedians=True, showextrema=True)
                    
                    # Color the violins and enhance appearance
                    for i, pc in enumerate(violin_parts['bodies']):
                        pc.set_facecolor(colors[i % len(colors)])
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
                            
                            # Add text box with statistics above each violin
                            stats_text = f'μ={mean_val:.3f}\nMd={median_val:.3f}\nQ1={q25:.3f}\nQ3={q75:.3f}\nσ={std_val:.3f}\nn={n_samples}'
                            
                            # Position text box above violin
                            y_max = np.max(method_data) if len(method_data) > 0 else 1
                            ax.text(pos, y_max + 0.05, stats_text, 
                                    ha='center', va='bottom', fontsize=7,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                             facecolor=colors[i % len(colors)], 
                                             alpha=0.3, edgecolor='black'))
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    
                    return True  # Successfully created plot
                else:
                    ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_ylabel(f'{metric_name} Correlation (r)', fontweight='bold')
            ax.set_title(plot_title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0 for correlations
            return False  # No plot created
        
        # Create all four plots
        plot1_success = create_enhanced_violin_plot(ax1, platform_data_feat, 'Feature-wise', 
                                                   f'Feature-wise Violin Plot')
        plot3_success = create_enhanced_violin_plot(ax3, platform_data_samp, 'Sample-wise', 
                                                   f'Sample-wise Violin Plot')
        
        # Function to create box plot with statistics
        def create_enhanced_box_plot(ax, platform_data, metric_name, plot_title):
            methods_data = []
            labels = []
            
            if len(platform_data) > 0:
                # Collect all available methods
                method_mapping = {
                    'Method_1': data.method1_name,
                    'Method_2': data.method2_name,
                    'Method_3': data.method3_name,
                    'Method_4': data.method4_name
                }
                
                for method_key, method_name in method_mapping.items():
                    if method_name is not None:  # Only include methods that have names
                        method_subset = platform_data[platform_data['method'] == method_key]
                        if len(method_subset) > 0:
                            method_data = method_subset['r'].values
                            methods_data.append(method_data)
                            labels.append(method_name)
                
                if len(methods_data) > 0 and all(len(md) > 0 for md in methods_data):
                    # Create box plot with statistics
                    box_parts = ax.boxplot(methods_data, positions=range(1, len(methods_data)+1), 
                                          widths=0.6, patch_artist=True, 
                                          showmeans=True, meanline=True,
                                          flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
                    
                    # Color the boxes to match violins
                    for i, patch in enumerate(box_parts['boxes']):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                    
                    # Style other box plot elements
                    for element in ['whiskers', 'caps', 'medians']:
                        if element in box_parts:
                            plt.setp(box_parts[element], color='black', linewidth=1.5)
                    
                    if 'means' in box_parts:
                        plt.setp(box_parts['means'], color='red', linewidth=2)
                    
                    # Add method names and sample sizes
                    for i, (method_data, label) in enumerate(zip(methods_data, labels)):
                        if len(method_data) > 0:
                            n_samples = len(method_data)
                            ax.text(i+1, -0.05, f'{label}\n(n={n_samples})', 
                                   ha='center', va='top', fontsize=8, 
                                   transform=ax.get_xaxis_transform(),
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', alpha=0.8))
                    
                    ax.set_xticks(range(1, len(labels)+1))
                    ax.set_xticklabels([''] * len(labels))  # Remove default labels since we added custom ones
                    
                    return True  # Successfully created plot
                else:
                    ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_ylabel(f'{metric_name} Correlation (r)', fontweight='bold')
            ax.set_title(plot_title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0 for correlations
            return False  # No plot created
        
        # Create feature-wise and sample-wise box plots
        plot2_success = create_enhanced_box_plot(ax2, platform_data_feat, 'Feature-wise', 
                                                f'Feature-wise Box Plot')
        plot4_success = create_enhanced_box_plot(ax4, platform_data_samp, 'Sample-wise', 
                                                f'Sample-wise Box Plot')
        
        # Add shared legends
        if plot1_success or plot3_success:
            legend_elements = [
                plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
                plt.Line2D([0], [0], color='blue', lw=2, label='Median'),
                plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='Q1'),
                plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Q3'),
                plt.Line2D([0], [0], color='black', lw=1.5, label='Min/Max')
            ]
            ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        if plot2_success or plot4_success:
            legend_elements = [
                plt.Line2D([0], [0], color='black', lw=1.5, label='Median'),
                plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
                plt.Line2D([0], [0], color='black', lw=1, label='Q1/Q3'),
                plt.Line2D([0], [0], marker='o', color='red', lw=0, markersize=3, label='Outliers')
            ]
            ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_3b_vertical_violin_plots(self, data: AnalysisData):
        """Figure 3b: Vertically stacked violin plots with Method 4 as horizontal baseline and statistical testing"""
        print("Generating Figure 3b: Vertical violin plots with statistical testing...")
        
        from scipy.stats import mannwhitneyu
        
        feat_metrics = data.metrics['feature_wise']
        sample_metrics = data.metrics['sample_wise']
        
        # Check if we have any metrics
        if feat_metrics.empty and sample_metrics.empty:
            print("    No metrics available - creating placeholder figure")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Distributions')
            return fig
        
        # Create 1x2 layout for feature-wise and sample-wise
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        fig.suptitle(f'{data.platform_a_name} - Vertical Distribution Analysis with Statistical Testing', 
                     fontsize=16, fontweight='bold')
        # Make subplots square
        try:
            ax1.set_box_aspect(1)
            ax2.set_box_aspect(1)
        except Exception:
            pass
        
        # Get Platform A data
        platform_data_feat = feat_metrics[feat_metrics['platform'] == 'Platform_A'] if not feat_metrics.empty else pd.DataFrame()
        platform_data_samp = sample_metrics[sample_metrics['platform'] == 'Platform_A'] if not sample_metrics.empty else pd.DataFrame()
        
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        def create_vertical_violin_plot_with_stats(ax, platform_data, metric_name, plot_title):
            """Create vertical violin plots with baseline and top significance bars"""
            methods_data = []
            labels = []
            method_keys = []
            baseline_data = None
            baseline_name = None
            
            if len(platform_data) > 0:
                # Collect all available methods
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

                    # Pairwise tests with top significance bars
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
                                    _, p_value = mannwhitneyu(methods_data_plot[i], methods_data_plot[j], alternative='two-sided')
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
                    ax.set_xticklabels(labels_plot, rotation=45, ha='right')
                    ax.set_ylabel(f'{metric_name} Correlation (r)', fontweight='bold')
                    ax.set_title(plot_title, fontweight='bold')
                    ax.set_xlim(0.5, n_methods + 0.5)
                    ax.grid(False)

                    if baseline_data is not None:
                        ax.legend(loc='upper left', fontsize=9)

                    return True
                else:
                    ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel(f'{metric_name} Correlation (r)', fontweight='bold')
            ax.set_title(plot_title, fontweight='bold')
            ax.grid(False)
            return False
        
        # Create vertical violin plots for both feature-wise and sample-wise
        create_vertical_violin_plot_with_stats(ax1, platform_data_feat, 'Feature-wise', 
                                              'Feature-wise Performance (Vertical)')
        create_vertical_violin_plot_with_stats(ax2, platform_data_samp, 'Sample-wise', 
                                              'Sample-wise Performance (Vertical)')
        
        # Add overall legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Median'),
            plt.Line2D([0], [0], color='gray', lw=1.5, label='Min/Max'),
            plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Baseline Method'),
            plt.Line2D([0], [0], color='black', lw=1, label='Statistical Comparisons')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4_bland_altman(self, data: AnalysisData):
        """Figure 4: Bland-Altman plots for bias assessment - separate subplot for each method"""
        print("Generating Figure 4: Bland-Altman plots...")
        
        # Get all available methods for Platform A only
        available_methods = self._get_available_methods(data)
        platform_a_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_A']
        
        # Exclude Method 4 (permuted baseline) from bias comparison
        platform_a_methods = [
            (method_key, method_name, truth, imputed)
            for (method_key, method_name, truth, imputed) in platform_a_methods
            if method_key != 'Method_4'
        ]
        
        n_methods = len(platform_a_methods)
        
        if n_methods == 0:
            print("No methods available for Bland-Altman analysis")
            return plt.figure()
        
        # Create subplots: 1 row, methods as columns (square plots)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        # Handle case where we have only one method
        if n_methods == 1:
            axes = [axes]
        
        # Make all plots square
        for ax in axes:
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle(f'Bland-Altman Analysis: {data.platform_a_name}', fontsize=16, fontweight='bold')
        
        # Colors for methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Calculate global limits for consistent scaling
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
        
        # Get limits for consistent scaling
        truth_limits, diff_limits = get_platform_limits(platform_a_methods)
        
        # Plot Platform A methods
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            ax = axes[i]
            
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
            ax.set_xlim(truth_limits)
            ax.set_ylim(diff_limits)
            
            ax.set_title(f'{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add statistics text
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}\n95% LoA=[{mean_diff-1.96*std_diff:.3f}, {mean_diff+1.96*std_diff:.3f}]', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4b_bias_comparison(self, data: AnalysisData):
        """Figure 4b: Bias comparison panel across all methods"""
        print("Generating Figure 4b: Bias comparison...")
        
        # Get all available methods for Platform A only
        available_methods = self._get_available_methods(data)
        platform_a_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_A']
        
        if len(platform_a_methods) == 0:
            print("No methods available for bias comparison")
            return plt.figure()
        
        # Create square figure for bias comparison - 5x5
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Colors for methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        
        # Collect bias data for comparison
        bias_data = []
        method_labels = []
        bias_colors = []
        bias_errors = []
        sample_sizes = []
        
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
                
                bias_data.append(mean_bias)
                bias_errors.append(ci_95)
                sample_sizes.append(len(diff_vals))
            else:
                bias_data.append(0)
                bias_errors.append(0)
                sample_sizes.append(0)
            
            method_labels.append(method_name)
            bias_colors.append(colors[i % len(colors)])
        
        if bias_data:
            # Create bias comparison bar chart
            x_pos = np.arange(len(method_labels))
            bars = ax.bar(x_pos, bias_data, yerr=bias_errors, capsize=5,
                         color=bias_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add horizontal reference line at bias = 0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Add value labels on bars
            if len(bias_data) > 0:
                data_range = max(bias_data) - min(bias_data) if max(bias_data) != min(bias_data) else 0.1
                for i, (bar, bias_val, n_samples) in enumerate(zip(bars, bias_data, sample_sizes)):
                    height = bar.get_height()
                    # Position label above error bar
                    label_y = height + bias_errors[i] + 0.02 * data_range
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{bias_val:.3f}\n(n={n_samples:,})',
                           ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Customize bias comparison panel
            ax.set_xlabel('Method')
            ax.set_ylabel('Bias (Imputed - Truth)')
            ax.set_title(f'Method Bias Comparison: {data.platform_a_name}', fontweight='bold', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add summary statistics text
            mean_abs_bias = np.mean([abs(b) for b in bias_data])
            ax.text(0.02, 0.98, f'Mean |Bias|: {mean_abs_bias:.3f}', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No bias data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Method Bias Comparison: {data.platform_a_name}', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_4c_bland_altman_density(self, data: AnalysisData):
        """Figure 4c: Bland-Altman density plots (high-resolution) for Platform A"""
        print("Generating Figure 4c: Bland-Altman density plots (single platform)...")

        available_methods = self._get_available_methods(data)
        platform_a_methods = [(method_key, method_name, truth, imputed)
                              for method_key, method_name, platform, truth, imputed in available_methods
                              if platform == 'Platform_A']

        n_methods = len(platform_a_methods)
        if n_methods == 0:
            print("No methods available for Bland-Altman density analysis")
            return plt.figure()

        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for ax in axes:
            ax.set_aspect('equal', adjustable='box')

        fig.suptitle(f'Bland-Altman Analysis (Density): {data.platform_a_name}', fontsize=16, fontweight='bold')

        def get_limits(methods):
            all_truth_vals, all_diff_vals = [], []
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
                tmin, tmax = np.min(all_truth_vals), np.max(all_truth_vals)
                dmin, dmax = np.min(all_diff_vals), np.max(all_diff_vals)
                tr = tmax - tmin
                dr = dmax - dmin
                return (tmin - 0.05*tr, tmax + 0.05*tr), (dmin - 0.05*dr, dmax + 0.05*dr)
            return (0, 1), (-1, 1)

        truth_limits, diff_limits = get_limits(platform_a_methods)

        def plot_density(ax, x_vals, y_vals, xlim, ylim):
            import matplotlib.colors as mcolors
            bins = (400, 400)
            x = np.clip(x_vals, xlim[0], xlim[1])
            y = np.clip(y_vals, ylim[0], ylim[1])
            H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
            H = H.T
            H_masked = np.ma.masked_where(H == 0, H)
            base = plt.cm.get_cmap('magma', 256)
            cmap = mcolors.ListedColormap(base(np.linspace(0, 1, 256)))
            cmap.set_bad('white')
            vmin = H_masked.min() if H_masked.count() > 0 else 1e-6
            vmax = H_masked.max() if H_masked.count() > 0 else 1.0
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.set_facecolor('white')
            im = ax.imshow(H_masked, extent=extent, origin='lower', aspect='auto',
                           cmap=cmap, norm=norm)
            return im

        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            ax = axes[i]
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat))
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            diff_vals = imp_clean - truth_clean

            plot_density(ax, truth_clean, diff_vals, truth_limits, diff_limits)

            mean_diff = float(np.mean(diff_vals)) if diff_vals.size else 0.0
            std_diff = float(np.std(diff_vals)) if diff_vals.size else 0.0
            ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, alpha=0.9)
            ax.axhline(mean_diff + 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(mean_diff - 1.96*std_diff, color='black', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

            ax.set_xlim(truth_limits)
            ax.set_ylim(diff_limits)
            ax.set_title(f'{method_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Truth Value')
            ax.set_ylabel('Imputed - Truth')
            ax.grid(False)
            ax.text(0.02, 0.95, f'n={len(truth_clean):,}\nBias={mean_diff:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

        plt.tight_layout()
        return fig

    def generate_figure_4d_variance_vs_mean(self, data: AnalysisData):
        """Figure 4d: Variance vs Mean plots (feature-wise and sample-wise) for Platform A"""
        print("Generating Figure 4d: Variance vs Mean plots (single platform)...")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Variance vs Mean: {data.platform_a_name}', fontsize=16, fontweight='bold')

        method_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'],
                         NATURE_COLORS['accent'], NATURE_COLORS['alternative_1']]
        truth_color = 'black'

        def plot_mv(ax, matrices, labels, title):
            eps = 1e-8
            for idx, (mat, label) in enumerate(zip(matrices, labels)):
                if mat is None:
                    continue
                means = np.nanmean(mat, axis=1)
                vars_ = np.nanvar(mat, axis=1)
                color = truth_color if label == 'Truth' else method_colors[(idx-1) % len(method_colors)] if idx > 0 else method_colors[0]
                ax.scatter(np.log10(means + eps), np.log10(vars_ + eps), s=6, alpha=0.35, c=color, edgecolors='none', rasterized=True, label=label)
            ax.set_xlabel('log10(Mean)')
            ax.set_ylabel('log10(Variance)')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Collect Platform A matrices
        platform_a_methods = []
        if data.imp_a_m1 is not None:
            platform_a_methods.append((data.imp_a_m1, data.method1_name))
        if data.imp_a_m2 is not None:
            platform_a_methods.append((data.imp_a_m2, data.method2_name))
        if getattr(data, 'imp_a_m3', None) is not None and getattr(data, 'method3_name', None) is not None:
            platform_a_methods.append((data.imp_a_m3, data.method3_name))
        if getattr(data, 'imp_a_m4', None) is not None and getattr(data, 'method4_name', None) is not None:
            platform_a_methods.append((data.imp_a_m4, data.method4_name))

        # Feature-wise (rows = features)
        a_feat_matrices = [data.truth_a.values] + [m.values for m, _ in platform_a_methods]
        a_feat_labels = ['Truth'] + [name for _, name in platform_a_methods]
        plot_mv(axes[0], a_feat_matrices, a_feat_labels, 'Feature-wise')

        # Sample-wise (rows = samples)
        a_samp_matrices = [data.truth_a.values.T] + [m.values.T for m, _ in platform_a_methods]
        a_samp_labels = ['Truth'] + [name for _, name in platform_a_methods]
        plot_mv(axes[1], a_samp_matrices, a_samp_labels, 'Sample-wise')

        plt.tight_layout()
        return fig
    def generate_figure_5_error_ecdfs(self, data: AnalysisData):
        """Figure 5: Absolute-error empirical CDFs"""
        print("Generating Figure 5: Error ECDFs...")
        
        # Support up to 4 methods with single platform layout
        available_methods = self._get_available_methods(data)
        method_count = len(available_methods)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle(f'{data.platform_a_name} - Error Distribution Analysis', fontsize=14, fontweight='bold')
        
        # Calculate absolute errors for all available methods
        errors = {}
        for method_key, method_name, platform, truth, imputed in available_methods:
            if platform == 'Platform_A':
                abs_errors = np.abs(truth.values - imputed.values).flatten()
                abs_errors = abs_errors[~np.isnan(abs_errors)]
                # Filter out exact zeros for log scale plotting
                abs_errors = abs_errors[abs_errors > 0]
                errors[method_key] = abs_errors
        
        # Define colors for up to 4 methods
        colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['accent'], NATURE_COLORS['neutral']]
        
        # Plot ECDFs for all available methods
        method_idx = 0
        for method_key, method_name, platform, truth, imputed in available_methods:
            if platform == 'Platform_A' and method_key in errors:
                sorted_errors = np.sort(errors[method_key])
                y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                ax.plot(sorted_errors, y_vals, color=colors[method_idx % len(colors)], linewidth=2, 
                       label=method_name)
                method_idx += 1
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{data.platform_a_name} - Error ECDF Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_6_sample_error_heatmap(self, data: AnalysisData):
        """Figure 6: Heatmap of sample-wise and feature-wise error landscape"""
        print("Generating Figure 6: Sample-wise and feature-wise error heatmap...")
        
        # Create 1x2 grid: sample-wise (left) and feature-wise (right) for Platform A only
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'{data.platform_a_name} - Error Landscape Analysis (MAE)', fontsize=16, fontweight='bold')
        
        # Get all available methods for Platform A
        available_methods = self._get_available_methods(data)
        method_names = [data.method1_name, data.method2_name, 
                       getattr(data, 'method3_name', 'Method 3'),
                       getattr(data, 'method4_name', 'Method 4')]
        
        platform_a_comparisons = []
        for method_key, method_name, platform, truth, imputed in available_methods:
            if platform == 'Platform_A':
                platform_a_comparisons.append((method_name, truth, imputed))
        
        # Sample-wise error map (left subplot)
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
            
            # Create heatmap for Platform A samples
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
            
            ax1.set_title('Sample-wise Error Landscape', fontweight='bold')
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Samples (Hierarchically Clustered)')
        else:
            ax1.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Sample-wise Error Landscape')
        
        # Feature-wise error map (right subplot)
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
            
            im2 = ax2.imshow(platform_a_feature_df_logged.values, cmap='RdYlBu_r', aspect='auto')
            
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
            
            ax2.set_xticks(range(len(platform_a_columns)))
            ax2.set_xticklabels(platform_a_columns, rotation=45, ha='right')
            ax2.set_yticks(ytick_positions)
            ax2.set_yticklabels(ytick_labels, fontsize=6)
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('log10(Mean Absolute Error)', rotation=270, labelpad=15)
            
            ax2.set_title('Feature-wise Error Landscape', fontweight='bold')
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Features (Hierarchically Clustered)')
        else:
            ax2.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Feature-wise Error Landscape')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_7_hexbin_error_abundance(self, data: AnalysisData):
        """Figure 7: Hexbin density plots showing error vs truth abundance"""
        print("Generating Figure 7: Hexbin error vs abundance...")
        
        # Get all available methods for Platform A only
        available_methods = self._get_available_methods(data)
        platform_a_methods = [(method_key, method_name, truth, imputed) 
                              for method_key, method_name, platform, truth, imputed in available_methods 
                              if platform == 'Platform_A']
        
        n_methods = len(platform_a_methods)
        
        if n_methods == 0:
            print("No methods available for hexbin analysis")
            return self._create_insufficient_data_figure(
                'Error vs Truth Abundance Relationship',
                'No methods available for analysis'
            )
        
        # Create layout: 1 row with n_methods columns (up to 4)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'Error vs Truth Abundance Relationship: {data.platform_a_name}', fontsize=14, fontweight='bold')
        
        comparisons = []
        for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
            comparisons.append((f'{method_name} - {data.platform_a_name}', truth, imputed, i))
        
        # First pass: collect data ranges for Platform A to determine consistent axis limits
        platform_a_log_truth_range = [float('inf'), float('-inf')]
        platform_a_error_range = [float('inf'), float('-inf')]
        
        # Collect all data points and calculate ranges
        for title, truth, imputed, idx in comparisons:
            # Flatten and clean data
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat)) & (truth_flat > 0)
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            if len(truth_clean) > 0:
                # Calculate log10 truth intensity and absolute error
                log_truth = np.log10(truth_clean)
                abs_error = np.abs(imp_clean - truth_clean)
                
                # Update ranges for Platform A
                platform_a_log_truth_range[0] = min(platform_a_log_truth_range[0], np.min(log_truth))
                platform_a_log_truth_range[1] = max(platform_a_log_truth_range[1], np.max(log_truth))
                platform_a_error_range[0] = min(platform_a_error_range[0], np.min(abs_error))
                platform_a_error_range[1] = max(platform_a_error_range[1], np.max(abs_error))
        
        # Handle edge cases where there might not be enough data
        if platform_a_log_truth_range[0] > platform_a_log_truth_range[1]:
            platform_a_log_truth_range = [0, 1]
        if platform_a_error_range[0] > platform_a_error_range[1]:
            platform_a_error_range = [0, 1]
        
        # Second pass: create the plots with consistent scales
        for i, (title, truth, imputed, _) in enumerate(comparisons):
            ax = axes[i]
            
            # Flatten and clean data
            truth_flat = truth.values.flatten()
            imp_flat = imputed.values.flatten()
            
            mask = ~(np.isnan(truth_flat) | np.isnan(imp_flat)) & (truth_flat > 0)
            truth_clean = truth_flat[mask]
            imp_clean = imp_flat[mask]
            
            if len(truth_clean) > 0:
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
                if len(valid_centers) > 0:
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
                
                # Add colorbar
                plt.colorbar(hb, ax=ax, label='Count')
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            # Set consistent axis limits for Platform A
            ax.set_xlim(platform_a_log_truth_range)
            ax.set_ylim(platform_a_error_range)
            
            ax.set_xlabel('log10 (Truth Intensity)')
            ax.set_ylabel('Absolute Error')
            ax.set_title(title)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add text explaining AUC calculation
            ax.text(0.02, 0.98, "Right-AUC: Area under curve after median intensity", 
                   transform=ax.transAxes, fontsize=8, va='top', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_8_global_correlation_heatmap(self, data: AnalysisData):
        """Figure 8: Global correlation heatmap of all datasets"""
        print("Generating Figure 8: Global correlation heatmap...")
        
        # Prepare data matrices for Platform A only - include all available methods
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
        
        # Calculate correlations between datasets using consistent global correlation
        # Use flattened vectors for all comparisons for consistency
        n_datasets = len(datasets)
        corr_matrix = np.full((n_datasets, n_datasets), np.nan)
        dataset_names = list(datasets.keys())
        
        for i, (name1, data1) in enumerate(datasets.items()):
            for j, (name2, data2) in enumerate(datasets.items()):
                if i <= j:  # Only compute upper triangle
                    # All datasets are from Platform A: use all features in common
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
        
        ax.set_title(f'Global Correlation Matrix: {data.platform_a_name}', fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    
    def generate_figure_9_umap_concordance(self, data: AnalysisData):
        """Figure 9: PCA and UMAP analysis - structure preservation assessment"""
        print("Generating Figure 9: Structure preservation analysis (PCA/UMAP)...")
        
        # Fit dimensionality reduction models on combined data, then transform each dataset
        # This shows the global structure across all datasets
        
        try:
            # Prepare datasets for UMAP (samples as observations)
            sample_data_list = []
            labels = []
            platforms = []
            data_types = []
            
            # Platform A datasets only (exclude Method 4 as it's a permuted baseline)
            platform_a_datasets = {
                'Truth_A': data.truth_a,
                f'{data.method1_name}_A': data.imp_a_m1,
                f'{data.method2_name}_A': data.imp_a_m2,
            }
            if data.imp_a_m3 is not None and data.method3_name is not None:
                platform_a_datasets[f'{data.method3_name}_A'] = data.imp_a_m3
            # Skip Method 4 - it's a permuted baseline not suitable for structure analysis
            
            # Process Platform A only
            for platform_name, datasets_dict in [('Platform_A', platform_a_datasets)]:
                
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
            
            # Create plots for Platform A with both PCA and UMAP
            fig, (ax_pca, ax_umap) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Structure Preservation Assessment: PCA and UMAP - {data.platform_a_name}\n(Models fitted on combined data)', 
                        fontsize=14, fontweight='bold')
            
            type_markers = {'Truth': 'o', 'Imputed': '^'}
            
            # Process Platform A only
            platform_key = 'Platform_A'
            datasets_dict = platform_a_datasets
            
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
                # Combine all data for fitting
                all_data = np.vstack(platform_sample_data)
                
                # Scale features using combined data
                scaler = StandardScaler()
                scaled_all_data = scaler.fit_transform(all_data)
                
                # Fit PCA on combined data
                pca = PCA(n_components=2, random_state=42)
                pca_embedding_all = pca.fit_transform(scaled_all_data)
                
                # Fit UMAP on combined data
                reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_neighbors=15)
                umap_embedding_all = reducer.fit_transform(scaled_all_data)
                
                # Transform each dataset separately for comparison
                pca_embeddings = []
                umap_embeddings = []
                
                for data_matrix in platform_sample_data:
                    # Scale using the same scaler fitted on combined data
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
                
                # PCA plot (left)
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
                ax_pca.set_title(f'PCA - {data.platform_a_name}')
                ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax_pca.grid(True, alpha=0.3)
                
                # UMAP plot (right)
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
                ax_umap.set_title(f'UMAP - {data.platform_a_name}')
                ax_umap.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax_umap.grid(True, alpha=0.3)
            else:
                # No data available
                ax_pca.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                          ha='center', va='center', transform=ax_pca.transAxes, fontsize=12)
                ax_pca.set_title(f'PCA - {data.platform_a_name}')
                
                ax_umap.text(0.5, 0.5, f'No data for {data.platform_a_name}', 
                           ha='center', va='center', transform=ax_umap.transAxes, fontsize=12)
                ax_umap.set_title(f'UMAP - {data.platform_a_name}')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in dimensionality reduction analysis (combined): {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in PCA/UMAP analysis:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Dimensionality Reduction Concordance')
            return fig
    
    def generate_figure_9b_feature_level_umap_pca(self, data: AnalysisData):
        """Figure 9b: Feature-level PCA and UMAP analysis"""
        print("Generating Figure 9b: Feature-level structure analysis (PCA/UMAP)...")
        
        try:
            # Get available methods for Platform A (exclude Method 4 - it's a permuted baseline)
            available_methods = self._get_available_methods(data)
            platform_a_methods = [(method_key, method_name, truth, imputed) 
                                  for method_key, method_name, platform, truth, imputed in available_methods 
                                  if platform == 'Platform_A' and method_key != 'Method_4']
            
            n_methods = len(platform_a_methods)
            
            if n_methods == 0:
                print("    No methods available for feature-level analysis")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.text(0.5, 0.5, 'No methods available for feature-level PCA/UMAP analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Feature-level Dimensionality Reduction')
                return fig
            
            # Create figure: n_methods rows, 2 columns (PCA, UMAP)
            fig, axes = plt.subplots(n_methods, 2, figsize=(10, 5*n_methods))
            if n_methods == 1:
                axes = axes.reshape(1, 2)
            
            fig.suptitle(f'Feature-level Structure Analysis: {data.platform_a_name}\n(Each point is a feature/protein)', 
                        fontsize=14, fontweight='bold')
            
            # Set up group mapping for coloring
            group_mapping = {}
            group_colors = None
            if data.groups is not None:
                feature_names = data.truth_a.index
                
                if len(data.groups) == len(feature_names):  # Feature-level groups
                    group_mapping = dict(zip(feature_names, data.groups))
                    unique_groups = list(set(data.groups))
                    group_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
                    group_color_map = dict(zip(unique_groups, group_colors))
                else:
                    print(f"    Warning: Groups length ({len(data.groups)}) doesn't match features ({len(feature_names)})")
            
            # Process each method
            for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
                ax_pca = axes[i, 0]
                ax_umap = axes[i, 1]
                
                # Prepare feature data (features × samples)
                truth_features = truth.values  # features × samples
                method_features = imputed.values  # features × samples
                
                # Handle NaN values by replacing with feature means
                truth_clean = truth_features.copy()
                method_clean = method_features.copy()
                
                # Replace NaNs with row means (feature means across samples)
                for j in range(truth_clean.shape[0]):
                    truth_row_mean = np.nanmean(truth_clean[j, :])
                    method_row_mean = np.nanmean(method_clean[j, :])
                    
                    if not np.isnan(truth_row_mean):
                        truth_clean[j, np.isnan(truth_clean[j, :])] = truth_row_mean
                    if not np.isnan(method_row_mean):
                        method_clean[j, np.isnan(method_clean[j, :])] = method_row_mean
                
                # Create combined data for fitting dimensionality reduction models
                # Stack truth and method features for fitting
                all_features = np.vstack([truth_clean, method_clean])
                
                # Apply dimensionality reduction if we have too many dimensions
                n_features, n_dims = all_features.shape
                if n_dims > 100:  # Reduce dimensions first if too high
                    from sklearn.decomposition import PCA as PCA_prep
                    pca_prep = PCA_prep(n_components=min(50, n_features-1), random_state=42)
                    all_features = pca_prep.fit_transform(all_features)
                
                # Fit PCA and UMAP on combined data
                pca = PCA(n_components=2, random_state=42)
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_features.shape[0]-1))
                
                # Fit on combined data
                pca.fit(all_features)
                reducer.fit(all_features)
                
                # Transform truth and method data separately
                if n_dims > 100:
                    truth_for_transform = pca_prep.transform(truth_clean)
                    method_for_transform = pca_prep.transform(method_clean)
                else:
                    truth_for_transform = truth_clean
                    method_for_transform = method_clean
                
                pca_truth = pca.transform(truth_for_transform)
                pca_method = pca.transform(method_for_transform)
                umap_truth = reducer.transform(truth_for_transform)
                umap_method = reducer.transform(method_for_transform)
                
                # Prepare colors for features
                feature_colors = []
                feature_labels = []
                
                if group_mapping:
                    for feature in truth.index:
                        group = group_mapping.get(feature, 'Unknown')
                        if group != 'Unknown' and group in group_color_map:
                            feature_colors.append(group_color_map[group])
                        else:
                            feature_colors.append('gray')
                        feature_labels.append(group)
                else:
                    feature_colors = [NATURE_COLORS['primary']] * truth_clean.shape[0]
                    feature_labels = ['No Groups'] * truth_clean.shape[0]
                
                # Plot PCA - Truth as circles, Method as triangles
                ax_pca.scatter(pca_truth[:, 0], pca_truth[:, 1],
                             c=feature_colors, alpha=0.7, s=40, marker='o',
                             edgecolors='black', linewidth=0.3, label='Truth')
                ax_pca.scatter(pca_method[:, 0], pca_method[:, 1],
                             c=feature_colors, alpha=0.7, s=40, marker='^',
                             edgecolors='black', linewidth=0.3, label=method_name)
                
                ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax_pca.set_title(f'PCA - Truth vs {method_name}')
                ax_pca.grid(True, alpha=0.3)
                ax_pca.legend()
                
                # Plot UMAP - Truth as circles, Method as triangles
                ax_umap.scatter(umap_truth[:, 0], umap_truth[:, 1],
                              c=feature_colors, alpha=0.7, s=40, marker='o',
                              edgecolors='black', linewidth=0.3, label='Truth')
                ax_umap.scatter(umap_method[:, 0], umap_method[:, 1],
                              c=feature_colors, alpha=0.7, s=40, marker='^',
                              edgecolors='black', linewidth=0.3, label=method_name)
                
                ax_umap.set_xlabel('UMAP 1')
                ax_umap.set_ylabel('UMAP 2')
                ax_umap.set_title(f'UMAP - Truth vs {method_name}')
                ax_umap.grid(True, alpha=0.3)
                ax_umap.legend()
                
                # Add legend for groups (only for the first row to avoid repetition)
                if i == 0 and group_mapping:
                    unique_groups_present = list(set(feature_labels))
                    if 'Unknown' in unique_groups_present:
                        unique_groups_present.remove('Unknown')
                    
                    legend_elements = []
                    for group in unique_groups_present:
                        if group in group_color_map:
                            legend_elements.append(Patch(facecolor=group_color_map[group], 
                                                       edgecolor='black', label=group))
                    
                    if legend_elements:
                        ax_pca.legend(handles=legend_elements, loc='best', fontsize=8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in feature-level dimensionality reduction analysis: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in feature-level PCA/UMAP analysis:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Feature-level Dimensionality Reduction')
            return fig
    
    def generate_figure_9c_feature_level_umap_pca_knn_clusters(self, data: AnalysisData):
        """Figure 9c: Feature-level PCA and UMAP analysis with K-Means clustering"""
        print("Generating Figure 9c: Feature-level structure analysis with K-Means clustering...")
        
        try:
            # Get available methods for Platform A (exclude Method 4 - it's a permuted baseline)
            available_methods = self._get_available_methods(data)
            platform_a_methods = [(method_key, method_name, truth, imputed) 
                                  for method_key, method_name, platform, truth, imputed in available_methods 
                                  if platform == 'Platform_A' and method_key != 'Method_4']
            
            n_methods = len(platform_a_methods)
            
            if n_methods == 0:
                print("    No methods available for feature-level analysis")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.text(0.5, 0.5, 'No methods available for feature-level PCA/UMAP analysis with K-Means clustering', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Feature-level Dimensionality Reduction with K-Means Clustering')
                return fig
            
            # Perform KNN clustering on the truth dataset
            truth_data = data.truth_a
            
            # Handle NaN values by replacing with feature means
            truth_features = truth_data.values  # features × samples
            truth_clean = truth_features.copy()
            
            for j in range(truth_clean.shape[0]):
                truth_row_mean = np.nanmean(truth_clean[j, :])
                if not np.isnan(truth_row_mean):
                    truth_clean[j, np.isnan(truth_clean[j, :])] = truth_row_mean
                else:
                    truth_clean[j, :] = 0  # fallback for all-NaN features
            
            # Perform K-Means clustering
            
            # Scale features for clustering
            scaler = StandardScaler()
            truth_scaled = scaler.fit_transform(truth_clean)
            
            # Determine optimal number of clusters (between 3 and 8)
            n_features = truth_scaled.shape[0]
            optimal_k = min(max(3, n_features // 20), 8)  # Heuristic: 1 cluster per 20 features, but between 3-8
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(truth_scaled)
            
            print(f"    K-Means clustering: {optimal_k} clusters for {n_features} features")
            
            # Create figure: n_methods rows, 2 columns (PCA, UMAP)
            fig, axes = plt.subplots(n_methods, 2, figsize=(10, 5*n_methods))
            if n_methods == 1:
                axes = axes.reshape(1, 2)
            
            fig.suptitle(f'Feature-level Structure Analysis with K-Means Clustering: {data.platform_a_name}\n(Each point is a feature/protein, {optimal_k} clusters)', 
                        fontsize=14, fontweight='bold')
            
            # Create cluster color mapping
            cluster_colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
            cluster_color_map = {i: cluster_colors[i] for i in range(optimal_k)}
            
            # Process each method
            for i, (method_key, method_name, truth, imputed) in enumerate(platform_a_methods):
                ax_pca = axes[i, 0]
                ax_umap = axes[i, 1]
                
                # Prepare feature data (features × samples)
                truth_features = truth.values  # features × samples
                method_features = imputed.values  # features × samples
                
                # Handle NaN values by replacing with feature means
                truth_clean = truth_features.copy()
                method_clean = method_features.copy()
                
                # Replace NaNs with row means (feature means across samples)
                for j in range(truth_clean.shape[0]):
                    truth_row_mean = np.nanmean(truth_clean[j, :])
                    method_row_mean = np.nanmean(method_clean[j, :])
                    
                    if not np.isnan(truth_row_mean):
                        truth_clean[j, np.isnan(truth_clean[j, :])] = truth_row_mean
                    else:
                        truth_clean[j, :] = 0
                    if not np.isnan(method_row_mean):
                        method_clean[j, np.isnan(method_clean[j, :])] = method_row_mean
                    else:
                        method_clean[j, :] = 0
                
                # Create combined data for fitting dimensionality reduction models
                # Stack truth and method features for fitting
                all_features = np.vstack([truth_clean, method_clean])
                
                # Apply dimensionality reduction if we have too many dimensions
                n_features_current, n_dims = all_features.shape
                if n_dims > 100:  # Reduce dimensions first if too high
                    from sklearn.decomposition import PCA as PCA_prep
                    pca_prep = PCA_prep(n_components=min(50, n_features_current-1), random_state=42)
                    all_features = pca_prep.fit_transform(all_features)
                
                # Fit PCA and UMAP on combined data
                pca = PCA(n_components=2, random_state=42)
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_features.shape[0]-1))
                
                # Fit on combined data
                pca.fit(all_features)
                reducer.fit(all_features)
                
                # Transform truth and method data separately
                if n_dims > 100:
                    truth_for_transform = pca_prep.transform(truth_clean)
                    method_for_transform = pca_prep.transform(method_clean)
                else:
                    truth_for_transform = truth_clean
                    method_for_transform = method_clean
                
                pca_truth = pca.transform(truth_for_transform)
                pca_method = pca.transform(method_for_transform)
                umap_truth = reducer.transform(truth_for_transform)
                umap_method = reducer.transform(method_for_transform)
                
                # Prepare colors for features based on cluster labels
                feature_colors = [cluster_color_map[label] for label in cluster_labels]
                
                # Plot PCA - Truth as circles, Method as triangles
                ax_pca.scatter(pca_truth[:, 0], pca_truth[:, 1],
                             c=feature_colors, alpha=0.7, s=40, marker='o',
                             edgecolors='black', linewidth=0.3, label='Truth')
                ax_pca.scatter(pca_method[:, 0], pca_method[:, 1],
                             c=feature_colors, alpha=0.7, s=40, marker='^',
                             edgecolors='black', linewidth=0.3, label=method_name)
                
                ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax_pca.set_title(f'PCA - Truth vs {method_name}')
                ax_pca.grid(True, alpha=0.3)
                ax_pca.legend()
                
                # Plot UMAP - Truth as circles, Method as triangles
                ax_umap.scatter(umap_truth[:, 0], umap_truth[:, 1],
                              c=feature_colors, alpha=0.7, s=40, marker='o',
                              edgecolors='black', linewidth=0.3, label='Truth')
                ax_umap.scatter(umap_method[:, 0], umap_method[:, 1],
                              c=feature_colors, alpha=0.7, s=40, marker='^',
                              edgecolors='black', linewidth=0.3, label=method_name)
                
                ax_umap.set_xlabel('UMAP 1')
                ax_umap.set_ylabel('UMAP 2')
                ax_umap.set_title(f'UMAP - Truth vs {method_name}')
                ax_umap.grid(True, alpha=0.3)
                ax_umap.legend()
                
                # Add legend for clusters (only for the first row to avoid repetition)
                if i == 0:
                    legend_elements = []
                    for cluster_id in range(optimal_k):
                        legend_elements.append(Patch(facecolor=cluster_color_map[cluster_id], 
                                                   edgecolor='black', label=f'Cluster {cluster_id+1}'))
                    
                    ax_pca.legend(handles=legend_elements, loc='best', fontsize=8, title='K-Means Clusters')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in feature-level dimensionality reduction analysis with K-Means clustering: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in feature-level PCA/UMAP analysis with K-Means clustering:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Feature-level Dimensionality Reduction with K-Means Clustering')
            return fig
    
    def generate_figure_9d_method_comparison_with_connections(self, data: AnalysisData):
        """Figure 9d: PCA and UMAP analysis with line connections"""
        print("Generating Figure 9d: Method comparison with connections (PCA/UMAP)...")
        
        try:
            # Check if we have at least the first two methods
            if data.imp_a_m1 is None or data.imp_a_m2 is None:
                print("    Insufficient methods available - need at least 2 methods")
                return self._create_insufficient_data_figure(
                    'Method Comparison with Connections',
                    'Need at least 2 methods for comparison analysis'
                )
            
            # Create figure: 2x2 grid (sample-wise PCA/UMAP, feature-wise PCA/UMAP)
            fig, ((ax_samp_pca, ax_samp_umap), (ax_feat_pca, ax_feat_umap)) = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Method Comparison with Connections: {data.platform_a_name}\\n(Truth vs {data.method1_name})', 
                        fontsize=14, fontweight='bold')
            
            # === SAMPLE-WISE ANALYSIS ===
            print("    Processing sample-wise analysis...")
            
            # Prepare sample data (samples × features)
            truth_samples = data.truth_a.T.values  # samples × features
            method1_samples = data.imp_a_m1.T.values
            method2_samples = data.imp_a_m2.T.values
            
            # Handle NaN values for sample-wise analysis
            def clean_sample_data(data_matrix):
                cleaned = data_matrix.copy()
                col_means = np.nanmean(cleaned, axis=0)
                inds = np.where(np.isnan(cleaned))
                cleaned[inds] = np.take(col_means, inds[1])
                return cleaned
            
            truth_samples_clean = clean_sample_data(truth_samples)
            method1_samples_clean = clean_sample_data(method1_samples)
            method2_samples_clean = clean_sample_data(method2_samples)
            
            # Dimensionality reduction for sample data if needed
            if truth_samples_clean.shape[1] > 500:
                feature_vars = np.var(truth_samples_clean, axis=0)
                top_features = np.argsort(feature_vars)[-500:]
                truth_samples_clean = truth_samples_clean[:, top_features]
                method1_samples_clean = method1_samples_clean[:, top_features]
                method2_samples_clean = method2_samples_clean[:, top_features]
            
            # Use only truth and method1 data for fitting
            all_sample_data = np.vstack([truth_samples_clean, method1_samples_clean])
            
            # Scale data
            scaler_samples = StandardScaler()
            all_sample_data_scaled = scaler_samples.fit_transform(all_sample_data)
            
            # Fit PCA and UMAP on combined data
            pca_samples = PCA(n_components=2, random_state=42)
            umap_samples = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_sample_data_scaled.shape[0]-1))
            
            pca_samples.fit(all_sample_data_scaled)
            umap_samples.fit(all_sample_data_scaled)
            
            # Transform each dataset
            truth_samples_scaled = scaler_samples.transform(truth_samples_clean)
            method1_samples_scaled = scaler_samples.transform(method1_samples_clean)
            
            pca_truth_samp = pca_samples.transform(truth_samples_scaled)
            pca_method1_samp = pca_samples.transform(method1_samples_scaled)
            
            umap_truth_samp = umap_samples.transform(truth_samples_scaled)
            umap_method1_samp = umap_samples.transform(method1_samples_scaled)
            
            # Plot sample-wise PCA - only Truth and Method1
            ax_samp_pca.scatter(pca_truth_samp[:, 0], pca_truth_samp[:, 1], 
                               c=NATURE_COLORS['primary'], alpha=0.7, s=30, marker='o', 
                               label='Truth', edgecolors='black', linewidth=0.3)
            ax_samp_pca.scatter(pca_method1_samp[:, 0], pca_method1_samp[:, 1], 
                               c=NATURE_COLORS['secondary'], alpha=0.7, s=30, marker='^', 
                               label=data.method1_name, edgecolors='black', linewidth=0.3)
            
            # Add connection lines between same samples - only Truth to Method1
            for i in range(len(pca_truth_samp)):
                # Connect truth to method1
                ax_samp_pca.plot([pca_truth_samp[i, 0], pca_method1_samp[i, 0]], 
                                [pca_truth_samp[i, 1], pca_method1_samp[i, 1]], 
                                color='gray', alpha=0.3, linewidth=0.5)
            
            ax_samp_pca.set_xlabel(f'PC1 ({pca_samples.explained_variance_ratio_[0]:.1%} variance)')
            ax_samp_pca.set_ylabel(f'PC2 ({pca_samples.explained_variance_ratio_[1]:.1%} variance)')
            ax_samp_pca.set_title('Sample-wise PCA with Connections')
            ax_samp_pca.legend()
            ax_samp_pca.grid(True, alpha=0.3)
            
            # Plot sample-wise UMAP - only Truth and Method1
            ax_samp_umap.scatter(umap_truth_samp[:, 0], umap_truth_samp[:, 1], 
                                c=NATURE_COLORS['primary'], alpha=0.7, s=30, marker='o', 
                                label='Truth', edgecolors='black', linewidth=0.3)
            ax_samp_umap.scatter(umap_method1_samp[:, 0], umap_method1_samp[:, 1], 
                                c=NATURE_COLORS['secondary'], alpha=0.7, s=30, marker='^', 
                                label=data.method1_name, edgecolors='black', linewidth=0.3)
            
            # Add connection lines - only Truth to Method1
            for i in range(len(umap_truth_samp)):
                ax_samp_umap.plot([umap_truth_samp[i, 0], umap_method1_samp[i, 0]], 
                                 [umap_truth_samp[i, 1], umap_method1_samp[i, 1]], 
                                 color='gray', alpha=0.3, linewidth=0.5)
            
            ax_samp_umap.set_xlabel('UMAP 1')
            ax_samp_umap.set_ylabel('UMAP 2')
            ax_samp_umap.set_title('Sample-wise UMAP with Connections')
            ax_samp_umap.legend()
            ax_samp_umap.grid(True, alpha=0.3)
            
            # === FEATURE-WISE ANALYSIS ===
            print("    Processing feature-wise analysis...")
            
            # Prepare feature data (features × samples) - only truth and method1
            truth_features = data.truth_a.values  # features × samples
            method1_features = data.imp_a_m1.values
            
            # Handle NaN values for feature-wise analysis
            def clean_feature_data(data_matrix):
                cleaned = data_matrix.copy()
                for j in range(cleaned.shape[0]):
                    row_mean = np.nanmean(cleaned[j, :])
                    if not np.isnan(row_mean):
                        cleaned[j, np.isnan(cleaned[j, :])] = row_mean
                    else:
                        cleaned[j, :] = 0
                return cleaned
            
            truth_features_clean = clean_feature_data(truth_features)
            method1_features_clean = clean_feature_data(method1_features)
            
            # Combine only truth and method1 feature data for fitting
            all_feature_data = np.vstack([truth_features_clean, method1_features_clean])
            
            # Apply dimensionality reduction if needed
            n_features_total, n_dims = all_feature_data.shape
            if n_dims > 100:
                from sklearn.decomposition import PCA as PCA_prep
                pca_prep = PCA_prep(n_components=min(50, n_features_total-1), random_state=42)
                all_feature_data = pca_prep.fit_transform(all_feature_data)
                truth_features_clean = pca_prep.transform(truth_features_clean)
                method1_features_clean = pca_prep.transform(method1_features_clean)
            
            # Fit PCA and UMAP on combined feature data
            pca_features = PCA(n_components=2, random_state=42)
            umap_features = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_feature_data.shape[0]-1))
            
            pca_features.fit(all_feature_data)
            umap_features.fit(all_feature_data)
            
            # Transform each dataset
            pca_truth_feat = pca_features.transform(truth_features_clean)
            pca_method1_feat = pca_features.transform(method1_features_clean)
            
            umap_truth_feat = umap_features.transform(truth_features_clean)
            umap_method1_feat = umap_features.transform(method1_features_clean)
            
            # Plot feature-wise PCA - only Truth and Method1
            ax_feat_pca.scatter(pca_truth_feat[:, 0], pca_truth_feat[:, 1], 
                               c=NATURE_COLORS['primary'], alpha=0.7, s=30, marker='o', 
                               label='Truth', edgecolors='black', linewidth=0.3)
            ax_feat_pca.scatter(pca_method1_feat[:, 0], pca_method1_feat[:, 1], 
                               c=NATURE_COLORS['secondary'], alpha=0.7, s=30, marker='^', 
                               label=data.method1_name, edgecolors='black', linewidth=0.3)
            
            # Add connection lines between same features - only Truth to Method1
            for i in range(len(pca_truth_feat)):
                ax_feat_pca.plot([pca_truth_feat[i, 0], pca_method1_feat[i, 0]], 
                                [pca_truth_feat[i, 1], pca_method1_feat[i, 1]], 
                                color='gray', alpha=0.3, linewidth=0.5)
            
            ax_feat_pca.set_xlabel(f'PC1 ({pca_features.explained_variance_ratio_[0]:.1%} variance)')
            ax_feat_pca.set_ylabel(f'PC2 ({pca_features.explained_variance_ratio_[1]:.1%} variance)')
            ax_feat_pca.set_title('Feature-wise PCA with Connections')
            ax_feat_pca.legend()
            ax_feat_pca.grid(True, alpha=0.3)
            
            # Plot feature-wise UMAP
            ax_feat_umap.scatter(umap_truth_feat[:, 0], umap_truth_feat[:, 1], 
                                c=NATURE_COLORS['primary'], alpha=0.7, s=30, marker='o', 
                                label='Truth', edgecolors='black', linewidth=0.3)
            ax_feat_umap.scatter(umap_method1_feat[:, 0], umap_method1_feat[:, 1], 
                                c=NATURE_COLORS['secondary'], alpha=0.7, s=30, marker='^', 
                                label=data.method1_name, edgecolors='black', linewidth=0.3)
            
            # Add connection lines
            for i in range(len(umap_truth_feat)):
                ax_feat_umap.plot([umap_truth_feat[i, 0], umap_method1_feat[i, 0]], 
                                 [umap_truth_feat[i, 1], umap_method1_feat[i, 1]], 
                                 color='gray', alpha=0.3, linewidth=0.5)
            
            ax_feat_umap.set_xlabel('UMAP 1')
            ax_feat_umap.set_ylabel('UMAP 2')
            ax_feat_umap.set_title('Feature-wise UMAP with Connections')
            ax_feat_umap.legend()
            ax_feat_umap.grid(True, alpha=0.3)
            
            # Add explanation text
            fig.text(0.5, 0.02, 'Gray lines: Truth-Method1 connections', 
                    ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in method comparison with connections analysis: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in method comparison analysis:\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Method Comparison with Connections')
            return fig
    
    def generate_figure_9e_euclidean_distance_analysis(self, data: AnalysisData):
        """Figure 9e: Euclidean distance analysis in original data space between methods"""
        print("Generating Figure 9e: Euclidean distance analysis...")
        
        try:
            # Check if we have at least the first two methods
            if data.imp_a_m1 is None or data.imp_a_m2 is None:
                print("    Insufficient methods available - need at least 2 methods")
                return self._create_insufficient_data_figure(
                    'Euclidean Distance Analysis',
                    'Need at least 2 methods for distance analysis'
                )
            
            # Create figure: 2x2 grid 
            fig, ((ax_feat_m1, ax_feat_m2), (ax_samp_m1, ax_samp_m2)) = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Euclidean Distance Analysis: {data.platform_a_name}\\n(Distances in Original Data Space)', 
                        fontsize=14, fontweight='bold')
            
            # === FEATURE-WISE DISTANCE ANALYSIS ===
            print("    Computing feature-wise distances...")
            
            # Calculate feature-wise distances
            feature_distances_m1 = []
            feature_distances_m2 = []
            feature_names = []
            
            for feature in data.truth_a.index:
                truth_vals = data.truth_a.loc[feature].values
                method1_vals = data.imp_a_m1.loc[feature].values
                method2_vals = data.imp_a_m2.loc[feature].values
                
                # Remove NaN values
                mask = ~(np.isnan(truth_vals) | np.isnan(method1_vals) | np.isnan(method2_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                
                truth_clean = truth_vals[mask]
                method1_clean = method1_vals[mask]
                method2_clean = method2_vals[mask]
                
                # Compute Euclidean distances
                dist_m1 = np.linalg.norm(truth_clean - method1_clean)
                dist_m2 = np.linalg.norm(truth_clean - method2_clean)
                
                feature_distances_m1.append(dist_m1)
                feature_distances_m2.append(dist_m2)
                feature_names.append(feature)
            
            # Plot feature-wise distance distributions for Method 1
            if len(feature_distances_m1) > 0:
                ax_feat_m1.hist(feature_distances_m1, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], 
                               density=True, edgecolor='black', linewidth=0.5)
                ax_feat_m1.axvline(np.mean(feature_distances_m1), color='red', linestyle='--', 
                                   linewidth=2, label=f'Mean: {np.mean(feature_distances_m1):.3f}')
                ax_feat_m1.axvline(np.median(feature_distances_m1), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Median: {np.median(feature_distances_m1):.3f}')
                ax_feat_m1.set_xlabel('Euclidean Distance')
                ax_feat_m1.set_ylabel('Density')
                ax_feat_m1.set_title(f'Feature-wise Distances: Truth vs {data.method1_name}')
                ax_feat_m1.legend()
                ax_feat_m1.grid(True, alpha=0.3)
                
                # Add summary statistics
                stats_text = f'n={len(feature_distances_m1)}\\nStd={np.std(feature_distances_m1):.3f}\\nMax={np.max(feature_distances_m1):.3f}'
                ax_feat_m1.text(0.02, 0.98, stats_text, transform=ax_feat_m1.transAxes, 
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_feat_m1.text(0.5, 0.5, 'No valid feature distances', 
                               ha='center', va='center', transform=ax_feat_m1.transAxes)
                ax_feat_m1.set_title(f'Feature-wise Distances: Truth vs {data.method1_name}')
            
            # Plot feature-wise distance distributions for Method 2
            if len(feature_distances_m2) > 0:
                ax_feat_m2.hist(feature_distances_m2, bins=30, alpha=0.7, color=NATURE_COLORS['accent'], 
                               density=True, edgecolor='black', linewidth=0.5)
                ax_feat_m2.axvline(np.mean(feature_distances_m2), color='red', linestyle='--', 
                                   linewidth=2, label=f'Mean: {np.mean(feature_distances_m2):.3f}')
                ax_feat_m2.axvline(np.median(feature_distances_m2), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Median: {np.median(feature_distances_m2):.3f}')
                ax_feat_m2.set_xlabel('Euclidean Distance')
                ax_feat_m2.set_ylabel('Density')
                ax_feat_m2.set_title(f'Feature-wise Distances: Truth vs {data.method2_name}')
                ax_feat_m2.legend()
                ax_feat_m2.grid(True, alpha=0.3)
                
                # Add summary statistics
                stats_text = f'n={len(feature_distances_m2)}\\nStd={np.std(feature_distances_m2):.3f}\\nMax={np.max(feature_distances_m2):.3f}'
                ax_feat_m2.text(0.02, 0.98, stats_text, transform=ax_feat_m2.transAxes, 
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_feat_m2.text(0.5, 0.5, 'No valid feature distances', 
                               ha='center', va='center', transform=ax_feat_m2.transAxes)
                ax_feat_m2.set_title(f'Feature-wise Distances: Truth vs {data.method2_name}')
            
            # === SAMPLE-WISE DISTANCE ANALYSIS ===
            print("    Computing sample-wise distances...")
            
            # Calculate sample-wise distances
            sample_distances_m1 = []
            sample_distances_m2 = []
            sample_names = []
            
            for sample in data.truth_a.columns:
                truth_vals = data.truth_a[sample].values
                method1_vals = data.imp_a_m1[sample].values
                method2_vals = data.imp_a_m2[sample].values
                
                # Remove NaN values
                mask = ~(np.isnan(truth_vals) | np.isnan(method1_vals) | np.isnan(method2_vals))
                if np.sum(mask) < 3:  # Need at least 3 points
                    continue
                
                truth_clean = truth_vals[mask]
                method1_clean = method1_vals[mask]
                method2_clean = method2_vals[mask]
                
                # Compute Euclidean distances
                dist_m1 = np.linalg.norm(truth_clean - method1_clean)
                dist_m2 = np.linalg.norm(truth_clean - method2_clean)
                
                sample_distances_m1.append(dist_m1)
                sample_distances_m2.append(dist_m2)
                sample_names.append(sample)
            
            # Plot sample-wise distance distributions for Method 1
            if len(sample_distances_m1) > 0:
                ax_samp_m1.hist(sample_distances_m1, bins=30, alpha=0.7, color=NATURE_COLORS['secondary'], 
                               density=True, edgecolor='black', linewidth=0.5)
                ax_samp_m1.axvline(np.mean(sample_distances_m1), color='red', linestyle='--', 
                                   linewidth=2, label=f'Mean: {np.mean(sample_distances_m1):.3f}')
                ax_samp_m1.axvline(np.median(sample_distances_m1), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Median: {np.median(sample_distances_m1):.3f}')
                ax_samp_m1.set_xlabel('Euclidean Distance')
                ax_samp_m1.set_ylabel('Density')
                ax_samp_m1.set_title(f'Sample-wise Distances: Truth vs {data.method1_name}')
                ax_samp_m1.legend()
                ax_samp_m1.grid(True, alpha=0.3)
                
                # Add summary statistics
                stats_text = f'n={len(sample_distances_m1)}\\nStd={np.std(sample_distances_m1):.3f}\\nMax={np.max(sample_distances_m1):.3f}'
                ax_samp_m1.text(0.02, 0.98, stats_text, transform=ax_samp_m1.transAxes, 
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_samp_m1.text(0.5, 0.5, 'No valid sample distances', 
                               ha='center', va='center', transform=ax_samp_m1.transAxes)
                ax_samp_m1.set_title(f'Sample-wise Distances: Truth vs {data.method1_name}')
            
            # Plot sample-wise distance distributions for Method 2
            if len(sample_distances_m2) > 0:
                ax_samp_m2.hist(sample_distances_m2, bins=30, alpha=0.7, color=NATURE_COLORS['accent'], 
                               density=True, edgecolor='black', linewidth=0.5)
                ax_samp_m2.axvline(np.mean(sample_distances_m2), color='red', linestyle='--', 
                                   linewidth=2, label=f'Mean: {np.mean(sample_distances_m2):.3f}')
                ax_samp_m2.axvline(np.median(sample_distances_m2), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Median: {np.median(sample_distances_m2):.3f}')
                ax_samp_m2.set_xlabel('Euclidean Distance')
                ax_samp_m2.set_ylabel('Density')
                ax_samp_m2.set_title(f'Sample-wise Distances: Truth vs {data.method2_name}')
                ax_samp_m2.legend()
                ax_samp_m2.grid(True, alpha=0.3)
                
                # Add summary statistics
                stats_text = f'n={len(sample_distances_m2)}\\nStd={np.std(sample_distances_m2):.3f}\\nMax={np.max(sample_distances_m2):.3f}'
                ax_samp_m2.text(0.02, 0.98, stats_text, transform=ax_samp_m2.transAxes, 
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_samp_m2.text(0.5, 0.5, 'No valid sample distances', 
                               ha='center', va='center', transform=ax_samp_m2.transAxes)
                ax_samp_m2.set_title(f'Sample-wise Distances: Truth vs {data.method2_name}')
            
            # Add explanation text
            fig.text(0.5, 0.02, 'Euclidean distances computed in original data space after removing NaN values', 
                    ha='center', fontsize=10, style='italic')
            
            # Print summary to console
            if len(feature_distances_m1) > 0 and len(feature_distances_m2) > 0:
                print(f"    Feature-wise distance summary:")
                print(f"      {data.method1_name}: mean={np.mean(feature_distances_m1):.3f}, median={np.median(feature_distances_m1):.3f}")
                print(f"      {data.method2_name}: mean={np.mean(feature_distances_m2):.3f}, median={np.median(feature_distances_m2):.3f}")
            
            if len(sample_distances_m1) > 0 and len(sample_distances_m2) > 0:
                print(f"    Sample-wise distance summary:")
                print(f"      {data.method1_name}: mean={np.mean(sample_distances_m1):.3f}, median={np.median(sample_distances_m1):.3f}")
                print(f"      {data.method2_name}: mean={np.mean(sample_distances_m2):.3f}, median={np.median(sample_distances_m2):.3f}")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"    Error in Euclidean distance analysis: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in Euclidean distance analysis:\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Euclidean Distance Analysis')
            return fig
    
    def generate_figure_9f_delta_metric_analysis(self, data: AnalysisData):
        """Figure 9f: Delta metric analysis"""
        print("Generating Figure 9f: Delta metric analysis...")
        
        try:
            # Check if we have at least the first two methods
            if data.imp_a_m1 is None or data.imp_a_m2 is None:
                print("    Insufficient methods available - need at least 2 methods")
                return self._create_insufficient_data_figure(
                    'Delta Metric Analysis',
                    'Need at least 2 methods for delta analysis'
                )
            
            # Create figure: 7x2 grid for comprehensive delta analysis including performance
            fig, axes = plt.subplots(7, 2, figsize=(10, 35))
            fig.suptitle(f'Comprehensive Delta Analysis: {data.platform_a_name}\\n(Multi-perspective PCA/UMAP on Element-wise Differences)', 
                        fontsize=16, fontweight='bold')
            
            # Define subplot aliases for clarity
            ax_samp_pca, ax_samp_umap = axes[0, 0], axes[0, 1]  # Row 1: Sample-wise
            ax_err_pca, ax_err_umap = axes[1, 0], axes[1, 1]    # Row 2: Error magnitude  
            ax_truth_pca, ax_truth_umap = axes[2, 0], axes[2, 1]  # Row 3: Truth properties
            ax_clust_pca, ax_clust_umap = axes[3, 0], axes[3, 1]  # Row 4: Clustering
            ax_stat_pca, ax_stat_umap = axes[4, 0], axes[4, 1]    # Row 5: Statistical properties
            ax_perc_pca, ax_perc_umap = axes[5, 0], axes[5, 1]    # Row 6: Percentile analysis
            ax_perf_pca, ax_perf_umap = axes[6, 0], axes[6, 1]    # Row 7: Imputation performance
            
            # === COMPUTE DELTA MATRICES ===
            print("    Computing delta matrices...")
            
            # Calculate element-wise differences - only Method1 vs Truth
            delta_m1_truth = data.imp_a_m1.values - data.truth_a.values  # Method1 - Truth
            
            # Handle NaN values in delta matrices
            def clean_delta_matrix(delta_matrix):
                cleaned = delta_matrix.copy()
                # Replace NaN with 0 (meaning no difference where data is missing)
                cleaned[np.isnan(cleaned)] = 0
                return cleaned
            
            delta_m1_truth_clean = clean_delta_matrix(delta_m1_truth)
            
            # === SAMPLE-WISE DELTA ANALYSIS ===
            print("    Processing sample-wise delta analysis...")
            
            # Transpose delta matrices for sample-wise analysis (samples × features)
            delta_m1_truth_samples = delta_m1_truth_clean.T
            
            # Use only Method1-Truth delta data for fitting
            all_delta_samples = delta_m1_truth_samples
            
            # Apply dimensionality reduction if needed
            if all_delta_samples.shape[1] > 500:
                feature_vars = np.var(all_delta_samples, axis=0)
                top_features = np.argsort(feature_vars)[-500:]
                all_delta_samples = all_delta_samples[:, top_features]
                delta_m1_truth_samples = delta_m1_truth_samples[:, top_features]
            
            # Scale delta data for PCA only
            scaler_delta_samples = StandardScaler()
            all_delta_samples_scaled = scaler_delta_samples.fit_transform(all_delta_samples)
            
            # Fit PCA and UMAP separately - PCA on scaled data, UMAP on original data
            pca_delta_samples = PCA(n_components=2, random_state=42)
            umap_delta_samples = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_delta_samples.shape[0]-1))
            
            pca_delta_samples.fit(all_delta_samples_scaled)
            umap_delta_samples.fit(all_delta_samples)  # Use original unscaled data
            
            # Transform the delta dataset
            delta_m1_truth_samples_scaled = scaler_delta_samples.transform(delta_m1_truth_samples)
            
            pca_delta_m1_truth_samp = pca_delta_samples.transform(delta_m1_truth_samples_scaled)
            
            umap_delta_m1_truth_samp = umap_delta_samples.transform(delta_m1_truth_samples)  # Use original unscaled data
            
            # === ENHANCED SAMPLE-WISE DELTA ANALYSIS ===
            print("    Computing sample coloring metrics...")
            
            # Calculate sample-wise metrics for coloring
            # Mean absolute delta per sample (across all features)
            sample_delta_magnitude = np.abs(delta_m1_truth_samples).mean(axis=1)
            
            # Sample total signal intensity (sum of truth values)
            sample_total_intensity = data.truth_a.sum(axis=0).values
            
            # Plot sample-wise delta PCA - colored by delta magnitude
            scatter_samp_pca = ax_samp_pca.scatter(pca_delta_m1_truth_samp[:, 0], pca_delta_m1_truth_samp[:, 1], 
                                                  c=sample_delta_magnitude, alpha=0.7, s=40, marker='o', 
                                                  cmap='plasma', edgecolors='black', linewidth=0.3)
            
            # Add colorbar for sample delta magnitude
            cbar_samp_pca = plt.colorbar(scatter_samp_pca, ax=ax_samp_pca, fraction=0.046, pad=0.04)
            cbar_samp_pca.set_label('Mean |\u0394| per Sample', rotation=270, labelpad=15)
            
            # Add origin line (represents no difference)
            ax_samp_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax_samp_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            ax_samp_pca.set_xlabel(f'PC1 ({pca_delta_samples.explained_variance_ratio_[0]:.1%} variance)')
            ax_samp_pca.set_ylabel(f'PC2 ({pca_delta_samples.explained_variance_ratio_[1]:.1%} variance)')
            ax_samp_pca.set_title(f'Sample-wise Delta PCA\\n(Color: Mean \u0394 Magnitude per Sample)')
            ax_samp_pca.grid(True, alpha=0.3)
            
            # Plot sample-wise delta UMAP - colored by total intensity
            scatter_samp_umap = ax_samp_umap.scatter(umap_delta_m1_truth_samp[:, 0], umap_delta_m1_truth_samp[:, 1], 
                                                    c=sample_total_intensity, alpha=0.7, s=40, marker='o', 
                                                    cmap='viridis', edgecolors='black', linewidth=0.3)
            
            # Add colorbar for sample total intensity
            cbar_samp_umap = plt.colorbar(scatter_samp_umap, ax=ax_samp_umap, fraction=0.046, pad=0.04)
            cbar_samp_umap.set_label('Total Sample Intensity', rotation=270, labelpad=15)
            
            ax_samp_umap.set_xlabel('UMAP 1')
            ax_samp_umap.set_ylabel('UMAP 2')
            ax_samp_umap.set_title(f'Sample-wise Delta UMAP\\n(Color: Total Sample Intensity)')
            ax_samp_umap.grid(True, alpha=0.3)
            
            # === FEATURE-WISE DELTA ANALYSIS ===
            print("    Processing feature-wise delta analysis...")
            
            # Use delta matrix directly for feature-wise analysis (features × samples)
            all_delta_features = delta_m1_truth_clean
            
            # Apply dimensionality reduction if needed
            n_features_total, n_dims = all_delta_features.shape
            if n_dims > 100:
                from sklearn.decomposition import PCA as PCA_prep
                pca_prep = PCA_prep(n_components=min(50, n_features_total-1), random_state=42)
                all_delta_features = pca_prep.fit_transform(all_delta_features)
                delta_m1_truth_clean_reduced = all_delta_features  # Use the reduced version
            else:
                delta_m1_truth_clean_reduced = delta_m1_truth_clean
            
            # Fit PCA and UMAP separately - PCA on reduced data (if needed), UMAP on original data
            pca_delta_features = PCA(n_components=2, random_state=42)
            umap_delta_features = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, delta_m1_truth_clean.shape[0]-1))
            
            pca_delta_features.fit(all_delta_features)  # Use reduced data for PCA
            umap_delta_features.fit(delta_m1_truth_clean)  # Use original data for UMAP
            
            # Transform the delta dataset
            pca_delta_m1_truth_feat = pca_delta_features.transform(delta_m1_truth_clean_reduced)
            
            umap_delta_m1_truth_feat = umap_delta_features.transform(delta_m1_truth_clean)  # Use original data for UMAP
            
            # === ENHANCED FEATURE-WISE DELTA ANALYSIS WITH COLORING ===
            print("    Computing feature coloring metrics...")
            
            # Calculate various metrics for coloring
            # 1. Delta magnitude (mean absolute delta per feature)
            delta_magnitude = np.abs(delta_m1_truth_clean).mean(axis=1)
            
            # 2. Delta variance per feature  
            delta_variance = np.var(delta_m1_truth_clean, axis=1)
            
            # 3. Original feature properties
            truth_mean_values = data.truth_a.mean(axis=1).values
            truth_variance = data.truth_a.var(axis=1).values
            truth_cv = data.truth_a.std(axis=1).values / (data.truth_a.mean(axis=1).values + 1e-8)  # Add small constant to avoid division by zero
            
            # === COMPREHENSIVE METRICS FOR ALL 6 ROWS ===
            print("    Computing comprehensive metrics for multi-row analysis...")
            
            # === ROW 2 METRICS: Error Magnitude Analysis ===
            # 1. Delta magnitude (mean absolute delta per feature) - already computed above
            # 2. Delta variance per feature - already computed above
            
            # === ROW 3 METRICS: Truth Properties Analysis ===
            # 1. Truth mean values - already computed above
            # 2. Truth variance - already computed above  
            # 3. Truth CV - already computed above
            # 4. Truth dynamic range
            truth_range = data.truth_a.max(axis=1).values - data.truth_a.min(axis=1).values
            
            # === ROW 4 METRICS: Clustering Analysis ===
            # 1. K-means clustering in delta space
            from sklearn.cluster import KMeans
            n_clusters_kmeans = min(6, max(3, len(delta_m1_truth_clean) // 40))  # Adaptive cluster number
            kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
            feature_clusters_kmeans = kmeans.fit_predict(delta_m1_truth_clean)
            
            # 2. Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            if len(delta_m1_truth_clean) > 3:
                linkage_matrix = linkage(pdist(delta_m1_truth_clean, metric='euclidean'), method='ward')
                n_clusters_hier = min(5, max(2, len(delta_m1_truth_clean) // 60))
                feature_clusters_hier = fcluster(linkage_matrix, n_clusters_hier, criterion='maxclust')
            else:
                feature_clusters_hier = np.ones(len(delta_m1_truth_clean))
                n_clusters_hier = 1
            
            # Create color mappings for clusters
            cluster_colors_kmeans = plt.cm.tab10(np.linspace(0, 1, n_clusters_kmeans))
            cluster_colors_hier = plt.cm.Set3(np.linspace(0, 1, n_clusters_hier))
            
            # === ROW 5 METRICS: Statistical Properties ===
            # 1. Delta skewness (systematic bias direction)
            from scipy.stats import skew
            delta_skewness = skew(delta_m1_truth_clean, axis=1)
            
            # 2. Delta range (max - min per feature)
            delta_range = np.max(delta_m1_truth_clean, axis=1) - np.min(delta_m1_truth_clean, axis=1)
            
            # 3. Delta kurtosis (tail behavior)
            from scipy.stats import kurtosis
            delta_kurtosis = kurtosis(delta_m1_truth_clean, axis=1)
            
            # === ROW 6 METRICS: Percentile Analysis ===
            # 1. 90th percentile of |Δ| per feature
            delta_p90 = np.percentile(np.abs(delta_m1_truth_clean), 90, axis=1)
            
            # 2. Percentage of outlier samples per feature (|Δ| > 2*std)
            delta_std_per_feature = np.std(delta_m1_truth_clean, axis=1)
            outlier_percentage = np.mean(np.abs(delta_m1_truth_clean) > 2 * delta_std_per_feature[:, np.newaxis], axis=1) * 100
            
            # 3. Interquartile range of deltas
            delta_iqr = np.percentile(delta_m1_truth_clean, 75, axis=1) - np.percentile(delta_m1_truth_clean, 25, axis=1)
            
            # === ROW 7 METRICS: Imputation Performance Analysis ===
            print("    Extracting imputation performance data...")
            
            # Extract feature-wise correlation data from existing metrics
            feat_metrics = data.metrics['feature_wise']
            platform_a_metrics = feat_metrics[feat_metrics['platform'] == 'Platform_A']
            method1_metrics = platform_a_metrics[platform_a_metrics['method'] == 'Method_1']
            
            # Create feature-wise correlation mapping
            feature_performance = {}
            feature_pvalues = {}
            for _, row in method1_metrics.iterrows():
                feature_performance[row['feature']] = row['r']
                feature_pvalues[row['feature']] = row.get('r_pvalue', np.nan)
            
            # Convert to arrays aligned with delta analysis features
            feature_r_values = np.array([feature_performance.get(feat, np.nan) 
                                        for feat in data.truth_a.index])
            feature_r_pvalues = np.array([feature_pvalues.get(feat, np.nan) 
                                         for feat in data.truth_a.index])
            
            # Create performance categories
            performance_categories = np.full(len(feature_r_values), 'Unknown', dtype=object)
            valid_r = ~np.isnan(feature_r_values)
            performance_categories[valid_r & (feature_r_values >= 0.8)] = 'Excellent'
            performance_categories[valid_r & (feature_r_values >= 0.6) & (feature_r_values < 0.8)] = 'Good'
            performance_categories[valid_r & (feature_r_values >= 0.4) & (feature_r_values < 0.6)] = 'Fair'
            performance_categories[valid_r & (feature_r_values < 0.4)] = 'Poor'
            
            print(f"    Computed metrics for {len(delta_m1_truth_clean)} features:")
            print(f"      - K-means clusters: {n_clusters_kmeans}")
            print(f"      - Hierarchical clusters: {n_clusters_hier}")
            print(f"      - Delta magnitude range: [{np.min(delta_magnitude):.3f}, {np.max(delta_magnitude):.3f}]")
            print(f"      - Truth CV range: [{np.min(truth_cv):.3f}, {np.max(truth_cv):.3f}]")
            print(f"      - Outlier percentage range: [{np.min(outlier_percentage):.1f}%, {np.max(outlier_percentage):.1f}%]")
            print(f"      - Imputation performance (r) range: [{np.nanmin(feature_r_values):.3f}, {np.nanmax(feature_r_values):.3f}]")
            print(f"      - Performance categories: {dict(zip(*np.unique(performance_categories, return_counts=True)))}")
            
            # ============================================================================
            # ROW 2: FEATURE-WISE DELTA - ERROR MAGNITUDE ANALYSIS
            # ============================================================================
            print("    Plotting Row 2: Error magnitude analysis...")
            
            # PCA: Color by delta magnitude
            scatter_err_pca = ax_err_pca.scatter(pca_delta_m1_truth_feat[:, 0], pca_delta_m1_truth_feat[:, 1], 
                                                c=delta_magnitude, alpha=0.7, s=35, marker='o', 
                                                cmap='viridis', edgecolors='black', linewidth=0.2)
            
            cbar_err_pca = plt.colorbar(scatter_err_pca, ax=ax_err_pca, fraction=0.046, pad=0.04)
            cbar_err_pca.set_label('Mean |Δ|', rotation=270, labelpad=12)
            
            # Add origin lines
            ax_err_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_err_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_err_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_err_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_err_pca.set_title('Row 2A: Error Magnitude (PCA)', fontweight='bold', fontsize=11)
            ax_err_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by delta variance
            scatter_err_umap = ax_err_umap.scatter(umap_delta_m1_truth_feat[:, 0], umap_delta_m1_truth_feat[:, 1], 
                                                  c=delta_variance, alpha=0.7, s=35, marker='o', 
                                                  cmap='plasma', edgecolors='black', linewidth=0.2)
            cbar_err_umap = plt.colorbar(scatter_err_umap, ax=ax_err_umap, fraction=0.046, pad=0.04)
            cbar_err_umap.set_label('Δ Variance', rotation=270, labelpad=12)
            
            ax_err_umap.set_xlabel('UMAP 1')
            ax_err_umap.set_ylabel('UMAP 2')
            ax_err_umap.set_title('Row 2B: Error Variance (UMAP)', fontweight='bold', fontsize=11)
            ax_err_umap.grid(True, alpha=0.3)
            
            # ============================================================================
            # ROW 3: FEATURE-WISE DELTA - TRUTH PROPERTIES ANALYSIS
            # ============================================================================
            print("    Plotting Row 3: Truth properties analysis...")
            
            # PCA: Color by truth mean values
            scatter_truth_pca = ax_truth_pca.scatter(pca_delta_m1_truth_feat[:, 0], pca_delta_m1_truth_feat[:, 1], 
                                                     c=truth_mean_values, alpha=0.7, s=35, marker='o', 
                                                     cmap='coolwarm', edgecolors='black', linewidth=0.2)
            cbar_truth_pca = plt.colorbar(scatter_truth_pca, ax=ax_truth_pca, fraction=0.046, pad=0.04)
            cbar_truth_pca.set_label('Truth Mean', rotation=270, labelpad=12)
            
            ax_truth_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_truth_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_truth_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_truth_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_truth_pca.set_title('Row 3A: Truth Abundance (PCA)', fontweight='bold', fontsize=11)
            ax_truth_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by truth coefficient of variation
            scatter_truth_umap = ax_truth_umap.scatter(umap_delta_m1_truth_feat[:, 0], umap_delta_m1_truth_feat[:, 1], 
                                                      c=truth_cv, alpha=0.7, s=35, marker='o', 
                                                      cmap='RdYlBu_r', edgecolors='black', linewidth=0.2)
            cbar_truth_umap = plt.colorbar(scatter_truth_umap, ax=ax_truth_umap, fraction=0.046, pad=0.04)
            cbar_truth_umap.set_label('Truth CV', rotation=270, labelpad=12)
            
            ax_truth_umap.set_xlabel('UMAP 1')
            ax_truth_umap.set_ylabel('UMAP 2')
            ax_truth_umap.set_title('Row 3B: Truth Variability (UMAP)', fontweight='bold', fontsize=11)
            ax_truth_umap.grid(True, alpha=0.3)
            
            # ============================================================================
            # ROW 4: FEATURE-WISE DELTA - CLUSTERING ANALYSIS
            # ============================================================================
            print("    Plotting Row 4: Clustering analysis...")
            
            # PCA: Color by K-means clusters
            for cluster_id in range(n_clusters_kmeans):
                cluster_mask = feature_clusters_kmeans == cluster_id
                if np.any(cluster_mask):
                    ax_clust_pca.scatter(pca_delta_m1_truth_feat[cluster_mask, 0], 
                                        pca_delta_m1_truth_feat[cluster_mask, 1], 
                                        c=[cluster_colors_kmeans[cluster_id]], alpha=0.7, s=35, marker='o', 
                                        label=f'K{cluster_id+1}', edgecolors='black', linewidth=0.2)
            
            ax_clust_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_clust_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_clust_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_clust_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_clust_pca.set_title(f'Row 4A: K-means (k={n_clusters_kmeans}) (PCA)', fontweight='bold', fontsize=11)
            ax_clust_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            ax_clust_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by hierarchical clusters
            for cluster_id in range(1, n_clusters_hier + 1):
                cluster_mask = feature_clusters_hier == cluster_id
                if np.any(cluster_mask):
                    ax_clust_umap.scatter(umap_delta_m1_truth_feat[cluster_mask, 0], 
                                         umap_delta_m1_truth_feat[cluster_mask, 1], 
                                         c=[cluster_colors_hier[cluster_id-1]], alpha=0.7, s=35, marker='o', 
                                         label=f'H{cluster_id}', edgecolors='black', linewidth=0.2)
            
            ax_clust_umap.set_xlabel('UMAP 1')
            ax_clust_umap.set_ylabel('UMAP 2')
            ax_clust_umap.set_title(f'Row 4B: Hierarchical (k={n_clusters_hier}) (UMAP)', fontweight='bold', fontsize=11)
            ax_clust_umap.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            ax_clust_umap.grid(True, alpha=0.3)
            
            # ============================================================================
            # ROW 5: FEATURE-WISE DELTA - STATISTICAL PROPERTIES
            # ============================================================================
            print("    Plotting Row 5: Statistical properties analysis...")
            
            # PCA: Color by delta skewness (systematic bias direction)
            scatter_stat_pca = ax_stat_pca.scatter(pca_delta_m1_truth_feat[:, 0], pca_delta_m1_truth_feat[:, 1], 
                                                  c=delta_skewness, alpha=0.7, s=35, marker='o', 
                                                  cmap='RdBu_r', edgecolors='black', linewidth=0.2)
            cbar_stat_pca = plt.colorbar(scatter_stat_pca, ax=ax_stat_pca, fraction=0.046, pad=0.04)
            cbar_stat_pca.set_label('Δ Skewness', rotation=270, labelpad=12)
            
            ax_stat_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_stat_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_stat_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_stat_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_stat_pca.set_title('Row 5A: Systematic Bias (PCA)', fontweight='bold', fontsize=11)
            ax_stat_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by delta range (max - min per feature)
            scatter_stat_umap = ax_stat_umap.scatter(umap_delta_m1_truth_feat[:, 0], umap_delta_m1_truth_feat[:, 1], 
                                                    c=delta_range, alpha=0.7, s=35, marker='o', 
                                                    cmap='magma', edgecolors='black', linewidth=0.2)
            cbar_stat_umap = plt.colorbar(scatter_stat_umap, ax=ax_stat_umap, fraction=0.046, pad=0.04)
            cbar_stat_umap.set_label('Δ Range', rotation=270, labelpad=12)
            
            ax_stat_umap.set_xlabel('UMAP 1')
            ax_stat_umap.set_ylabel('UMAP 2')
            ax_stat_umap.set_title('Row 5B: Error Range (UMAP)', fontweight='bold', fontsize=11)
            ax_stat_umap.grid(True, alpha=0.3)
            
            # ============================================================================
            # ROW 6: FEATURE-WISE DELTA - PERCENTILE ANALYSIS
            # ============================================================================
            print("    Plotting Row 6: Percentile analysis...")
            
            # PCA: Color by 90th percentile of |Δ| per feature
            scatter_perc_pca = ax_perc_pca.scatter(pca_delta_m1_truth_feat[:, 0], pca_delta_m1_truth_feat[:, 1], 
                                                  c=delta_p90, alpha=0.7, s=35, marker='o', 
                                                  cmap='inferno', edgecolors='black', linewidth=0.2)
            cbar_perc_pca = plt.colorbar(scatter_perc_pca, ax=ax_perc_pca, fraction=0.046, pad=0.04)
            cbar_perc_pca.set_label('90th %ile |Δ|', rotation=270, labelpad=12)
            
            ax_perc_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_perc_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_perc_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_perc_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_perc_pca.set_title('Row 6A: Extreme Errors (PCA)', fontweight='bold', fontsize=11)
            ax_perc_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by percentage of outlier samples per feature
            scatter_perc_umap = ax_perc_umap.scatter(umap_delta_m1_truth_feat[:, 0], umap_delta_m1_truth_feat[:, 1], 
                                                    c=outlier_percentage, alpha=0.7, s=35, marker='o', 
                                                    cmap='Reds', edgecolors='black', linewidth=0.2)
            cbar_perc_umap = plt.colorbar(scatter_perc_umap, ax=ax_perc_umap, fraction=0.046, pad=0.04)
            cbar_perc_umap.set_label('Outlier %', rotation=270, labelpad=12)
            
            ax_perc_umap.set_xlabel('UMAP 1')
            ax_perc_umap.set_ylabel('UMAP 2')
            ax_perc_umap.set_title('Row 6B: Outlier Frequency (UMAP)', fontweight='bold', fontsize=11)
            ax_perc_umap.grid(True, alpha=0.3)
            
            # ============================================================================
            # ROW 7: FEATURE-WISE DELTA - IMPUTATION PERFORMANCE ANALYSIS  
            # ============================================================================
            print("    Plotting Row 7: Imputation performance analysis...")
            
            # PCA: Color by feature-wise correlation (r) with Truth vs Method 1
            valid_performance = ~np.isnan(feature_r_values)
            if np.any(valid_performance):
                # Plot features with valid performance data
                scatter_perf_pca = ax_perf_pca.scatter(pca_delta_m1_truth_feat[valid_performance, 0], 
                                                      pca_delta_m1_truth_feat[valid_performance, 1], 
                                                      c=feature_r_values[valid_performance], alpha=0.7, s=35, marker='o', 
                                                      cmap='RdYlGn', edgecolors='black', linewidth=0.2,
                                                      vmin=0, vmax=1)
                cbar_perf_pca = plt.colorbar(scatter_perf_pca, ax=ax_perf_pca, fraction=0.046, pad=0.04)
                cbar_perf_pca.set_label('Correlation r', rotation=270, labelpad=12)
                
                # Plot features with missing performance data in gray
                if np.any(~valid_performance):
                    ax_perf_pca.scatter(pca_delta_m1_truth_feat[~valid_performance, 0], 
                                       pca_delta_m1_truth_feat[~valid_performance, 1], 
                                       color='lightgray', alpha=0.5, s=20, marker='x', 
                                       label='No performance data')
                    ax_perf_pca.legend(loc='upper right', fontsize=8)
            else:
                # All features have missing performance data
                ax_perf_pca.scatter(pca_delta_m1_truth_feat[:, 0], pca_delta_m1_truth_feat[:, 1], 
                                   color='lightgray', alpha=0.5, s=35, marker='o')
                ax_perf_pca.text(0.5, 0.5, 'No performance\\ndata available', 
                                transform=ax_perf_pca.transAxes, ha='center', va='center',
                                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_perf_pca.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax_perf_pca.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            
            ax_perf_pca.set_xlabel(f'PC1 ({pca_delta_features.explained_variance_ratio_[0]:.1%})')
            ax_perf_pca.set_ylabel(f'PC2 ({pca_delta_features.explained_variance_ratio_[1]:.1%})')
            ax_perf_pca.set_title('Row 7A: Imputation Quality (PCA)', fontweight='bold', fontsize=11)
            ax_perf_pca.grid(True, alpha=0.3)
            
            # UMAP: Color by performance categories
            # Create color mapping for performance categories
            perf_colors = {
                'Excellent': '#2E7D32',   # Dark Green
                'Good': '#66BB6A',        # Light Green  
                'Fair': '#FFA726',        # Orange
                'Poor': '#E53935',        # Red
                'Unknown': '#BDBDBD'      # Gray
            }
            
            for category in ['Excellent', 'Good', 'Fair', 'Poor', 'Unknown']:
                category_mask = performance_categories == category
                if np.any(category_mask):
                    ax_perf_umap.scatter(umap_delta_m1_truth_feat[category_mask, 0], 
                                        umap_delta_m1_truth_feat[category_mask, 1], 
                                        c=perf_colors[category], alpha=0.7, s=35, marker='o', 
                                        label=f'{category} (n={np.sum(category_mask)})', 
                                        edgecolors='black', linewidth=0.2)
            
            ax_perf_umap.set_xlabel('UMAP 1')
            ax_perf_umap.set_ylabel('UMAP 2')
            ax_perf_umap.set_title('Row 7B: Performance Categories (UMAP)', fontweight='bold', fontsize=11)
            ax_perf_umap.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            ax_perf_umap.grid(True, alpha=0.3)
            
            print("    Completed all 7 rows of comprehensive delta analysis.")
            
            # === CLUSTER ANALYSIS AND EXPORT ===
            print("    Performing cluster analysis...")
            
            # Export comprehensive feature analysis with all metrics
            cluster_df = pd.DataFrame({
                'feature_id': data.truth_a.index,
                # Clustering assignments
                'kmeans_cluster': feature_clusters_kmeans,
                'hierarchical_cluster': feature_clusters_hier,
                # Delta error metrics  
                'delta_magnitude': delta_magnitude,
                'delta_variance': delta_variance,
                'delta_skewness': delta_skewness,
                'delta_range': delta_range,
                'delta_kurtosis': delta_kurtosis,
                'delta_iqr': delta_iqr,
                'delta_p90': delta_p90,
                'outlier_percentage': outlier_percentage,
                # Truth properties
                'truth_mean': truth_mean_values,
                'truth_variance': truth_variance,
                'truth_cv': truth_cv,
                'truth_range': truth_range,
                # Imputation performance metrics (Row 7)
                'imputation_correlation_r': feature_r_values,
                'imputation_r_pvalue': feature_r_pvalues,
                'performance_category': performance_categories,
                # Dimensionality reduction coordinates
                'pca_coord1': pca_delta_m1_truth_feat[:, 0],
                'pca_coord2': pca_delta_m1_truth_feat[:, 1],
                'umap_coord1': umap_delta_m1_truth_feat[:, 0],
                'umap_coord2': umap_delta_m1_truth_feat[:, 1]
            })
            
            # Export sample analysis for further investigation
            sample_df = pd.DataFrame({
                'sample_id': data.truth_a.columns,
                'sample_delta_magnitude': sample_delta_magnitude,
                'sample_total_intensity': sample_total_intensity,
                'pca_coord1': pca_delta_m1_truth_samp[:, 0],
                'pca_coord2': pca_delta_m1_truth_samp[:, 1],
                'umap_coord1': umap_delta_m1_truth_samp[:, 0],
                'umap_coord2': umap_delta_m1_truth_samp[:, 1]
            })
            
            # Save comprehensive analyses with descriptive names
            cluster_output_path = self.output_dir / "data" / "comprehensive_delta_feature_analysis.csv"
            sample_output_path = self.output_dir / "data" / "comprehensive_delta_sample_analysis.csv"
            
            cluster_df.to_csv(cluster_output_path, index=False)
            sample_df.to_csv(sample_output_path, index=False)
            
            print(f"    Comprehensive feature analysis saved to: {cluster_output_path}")
            print(f"    Comprehensive sample analysis saved to: {sample_output_path}")
            
            # Also save cluster summary statistics
            cluster_summary_path = self.output_dir / "data" / "cluster_summary_statistics.csv"
            cluster_summary_data = []
            
            for cluster_id in range(n_clusters_kmeans):
                cluster_mask = feature_clusters_kmeans == cluster_id
                if np.any(cluster_mask):
                    cluster_features = data.truth_a.index[cluster_mask]
                    cluster_summary_data.append({
                        'cluster_id': cluster_id + 1,
                        'cluster_size': np.sum(cluster_mask),
                        'mean_delta_magnitude': np.mean(delta_magnitude[cluster_mask]),
                        'std_delta_magnitude': np.std(delta_magnitude[cluster_mask]),
                        'mean_truth_abundance': np.mean(truth_mean_values[cluster_mask]),
                        'std_truth_abundance': np.std(truth_mean_values[cluster_mask]),
                        'mean_truth_cv': np.mean(truth_cv[cluster_mask]),
                        'mean_outlier_percentage': np.mean(outlier_percentage[cluster_mask]),
                        'mean_imputation_r': np.nanmean(feature_r_values[cluster_mask]),
                        'std_imputation_r': np.nanstd(feature_r_values[cluster_mask]),
                        'performance_category_distribution': str(dict(zip(*np.unique(performance_categories[cluster_mask], return_counts=True)))),
                        'example_features': ', '.join(cluster_features[:5].tolist())
                    })
            
            cluster_summary_df = pd.DataFrame(cluster_summary_data)
            cluster_summary_df.to_csv(cluster_summary_path, index=False)
            print(f"    Cluster summary statistics saved to: {cluster_summary_path}")
            
            # Print detailed cluster statistics
            print(f"    \n=== COMPREHENSIVE CLUSTER ANALYSIS SUMMARY ===\n")
            print(f"    K-means clusters: {n_clusters_kmeans}")
            print(f"    Hierarchical clusters: {n_clusters_hier}")
            print(f"    Total features analyzed: {len(delta_m1_truth_clean)}")
            
            print(f"    \n--- K-MEANS CLUSTER DETAILS ---")
            for cluster_id in range(n_clusters_kmeans):
                cluster_mask = feature_clusters_kmeans == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size > 0:
                    cluster_delta_mag = delta_magnitude[cluster_mask]
                    cluster_truth_mean = truth_mean_values[cluster_mask]
                    cluster_truth_cv = truth_cv[cluster_mask]
                    cluster_outlier_pct = outlier_percentage[cluster_mask]
                    cluster_perf_r = feature_r_values[cluster_mask]
                    cluster_perf_cats = performance_categories[cluster_mask]
                    
                    print(f"    K-means Cluster {cluster_id+1}: {cluster_size} features")
                    print(f"      - Delta magnitude: {np.mean(cluster_delta_mag):.3f} ± {np.std(cluster_delta_mag):.3f}")
                    print(f"      - Truth abundance: {np.mean(cluster_truth_mean):.3f} ± {np.std(cluster_truth_mean):.3f}")
                    print(f"      - Truth variability (CV): {np.mean(cluster_truth_cv):.3f} ± {np.std(cluster_truth_cv):.3f}")
                    print(f"      - Outlier frequency: {np.mean(cluster_outlier_pct):.1f}% ± {np.std(cluster_outlier_pct):.1f}%")
                    
                    # Add performance analysis
                    valid_r_mask = ~np.isnan(cluster_perf_r)
                    if np.any(valid_r_mask):
                        valid_r_values = cluster_perf_r[valid_r_mask]
                        print(f"      - Imputation performance (r): {np.mean(valid_r_values):.3f} ± {np.std(valid_r_values):.3f} (n={np.sum(valid_r_mask)})")
                        
                        # Performance category distribution
                        cat_counts = dict(zip(*np.unique(cluster_perf_cats, return_counts=True)))
                        cat_summary = ', '.join([f"{cat}: {count}" for cat, count in cat_counts.items()])
                        print(f"      - Performance categories: {cat_summary}")
                    else:
                        print(f"      - Imputation performance: No data available")
                    
                    print(f"      - Example features: {', '.join(data.truth_a.index[cluster_mask][:3].tolist())}")
                    if cluster_size > 3:
                        print(f"        ...and {cluster_size-3} more")
                    print()
            
            # Add comprehensive explanation text
            fig.text(0.5, 0.01, 
                    f'Comprehensive Delta Analysis ({data.method1_name} - Truth): Row 1=Samples, Row 2=Error Magnitude, '
                    f'Row 3=Truth Properties, Row 4=Clustering (K={n_clusters_kmeans}, H={n_clusters_hier}), '
                    f'Row 5=Statistical Properties, Row 6=Percentiles, Row 7=Imputation Performance. '
                    f'Data exported to comprehensive_delta_*_analysis.csv', 
                    ha='center', fontsize=9, style='italic')
            
            # Print comprehensive summary statistics
            print(f"    \n=== COMPREHENSIVE DELTA STATISTICS ===\n")
            print(f"      {data.method1_name} - Truth Analysis:")
            print(f"        Mean delta: {np.mean(delta_m1_truth_clean):.3f}")
            print(f"        Std delta: {np.std(delta_m1_truth_clean):.3f}")
            print(f"        Mean |delta|: {np.mean(delta_magnitude):.3f}")
            print(f"        Delta range: [{np.min(delta_m1_truth_clean):.3f}, {np.max(delta_m1_truth_clean):.3f}]")
            print(f"        Delta skewness range: [{np.min(delta_skewness):.3f}, {np.max(delta_skewness):.3f}]")
            print(f"        90th percentile |delta| range: [{np.min(delta_p90):.3f}, {np.max(delta_p90):.3f}]")
            print(f"        Features with |delta| > 2*std: {np.sum(delta_magnitude > 2*np.std(delta_magnitude))}")
            print(f"        Features with outlier rate > 50%: {np.sum(outlier_percentage > 50)}")
            print(f"        High-error features (top 5% by magnitude): {np.sum(delta_magnitude > np.percentile(delta_magnitude, 95))}")
            print(f"        Low-error features (bottom 5% by magnitude): {np.sum(delta_magnitude < np.percentile(delta_magnitude, 5))}")
            
            # Add performance statistics
            valid_perf_mask = ~np.isnan(feature_r_values)
            if np.any(valid_perf_mask):
                valid_perf_r = feature_r_values[valid_perf_mask]
                print(f"      \n  Imputation Performance Analysis:")
                print(f"        Features with performance data: {np.sum(valid_perf_mask)}/{len(feature_r_values)}")
                print(f"        Mean correlation (r): {np.mean(valid_perf_r):.3f} ± {np.std(valid_perf_r):.3f}")
                print(f"        Performance range: [{np.min(valid_perf_r):.3f}, {np.max(valid_perf_r):.3f}]")
                
                # Performance category breakdown
                cat_counts = dict(zip(*np.unique(performance_categories, return_counts=True)))
                print(f"        Performance category distribution:")
                for cat, count in sorted(cat_counts.items()):
                    percentage = (count / len(performance_categories)) * 100
                    print(f"          {cat}: {count} features ({percentage:.1f}%)")
                
                # Performance-based feature analysis
                excellent_features = np.sum(performance_categories == 'Excellent')
                poor_features = np.sum(performance_categories == 'Poor')
                print(f"        High-performing features (Excellent, r≥0.8): {excellent_features}")
                print(f"        Low-performing features (Poor, r<0.4): {poor_features}")
            else:
                print(f"      \n  Imputation Performance Analysis: No performance data available")
            
            plt.tight_layout()
            
            # Return comprehensive analysis results for potential downstream use
            self._comprehensive_delta_analysis_results = {
                # Clustering results
                'kmeans_clusters': feature_clusters_kmeans,
                'hierarchical_clusters': feature_clusters_hier,
                'n_clusters_kmeans': n_clusters_kmeans,
                'n_clusters_hier': n_clusters_hier,
                # Comprehensive feature metrics
                'delta_magnitude': delta_magnitude,
                'delta_variance': delta_variance,
                'delta_skewness': delta_skewness,
                'delta_range': delta_range,
                'delta_kurtosis': delta_kurtosis,
                'delta_iqr': delta_iqr,
                'delta_p90': delta_p90,
                'outlier_percentage': outlier_percentage,
                # Truth properties
                'truth_mean_values': truth_mean_values,
                'truth_variance': truth_variance,
                'truth_cv': truth_cv,
                'truth_range': truth_range,
                # Imputation performance metrics (Row 7)
                'imputation_r_values': feature_r_values,
                'imputation_r_pvalues': feature_r_pvalues,
                'performance_categories': performance_categories,
                # Sample metrics
                'sample_delta_magnitude': sample_delta_magnitude,
                'sample_total_intensity': sample_total_intensity,
                # Comprehensive dataframes
                'comprehensive_feature_df': cluster_df,
                'comprehensive_sample_df': sample_df,
                'cluster_summary_df': cluster_summary_df,
                # Dimensionality reduction coordinates
                'pca_coordinates': pca_delta_m1_truth_feat,
                'umap_coordinates': umap_delta_m1_truth_feat
            }
            
            return fig
            
        except Exception as e:
            print(f"    Error in delta metric analysis: {e}")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error in delta metric analysis:\\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Delta Metric Analysis')
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
            # Figure 9 variants
            ("figure_9_umap_concordance", self.generate_figure_9_umap_concordance),
            ("figure_9b_feature_level_umap_pca", self.generate_figure_9b_feature_level_umap_pca),
            ("figure_9c_feature_level_umap_pca_knn_clusters", self.generate_figure_9c_feature_level_umap_pca_knn_clusters),
            ("figure_9d_method_comparison_with_connections", self.generate_figure_9d_method_comparison_with_connections),
            ("figure_9f_delta_metric_analysis", self.generate_figure_9f_delta_metric_analysis),
            
            # Other figure 9 variants (distance analysis doesn't use PCA/UMAP fitting)
            ("figure_9e_euclidean_distance_analysis", self.generate_figure_9e_euclidean_distance_analysis),
            ("figure_10_radar_chart", self.generate_figure_10_radar_chart),
            
            # Comprehensive method comparison (only if 3+ methods available)
            ("figure_26_comprehensive_method_comparison", self.generate_figure_26_comprehensive_method_comparison),
            ("figure_27_comprehensive_method_comparison_spearman", self.generate_figure_27_comprehensive_method_comparison_spearman),
            
            # Figures that don't require groups (moved from group-dependent)
            ("figure_14_volcano_plot", self.generate_figure_14_volcano_plot),
            ("figure_15_network_diagram", self.generate_figure_15_network_diagram),
            
            # Additional innovative figures  
            ("figure_17_performance_consistency", self.generate_figure_17_performance_consistency),
            ("figure_18_method_synergy_analysis", self.generate_figure_18_method_synergy_analysis),
            ("figure_20_feature_difficulty_profiling", self.generate_figure_20_feature_difficulty_profiling),
            ("figure_21_temporal_performance_trends", self.generate_figure_21_temporal_performance_trends),
        ]
        
        # Additional figures that require overlapping features (removed for single-platform analysis)
        shared_feature_functions = []
        
        generated_figures = []
        
        # Generate basic figures (1-10 and 17-21)
        for fig_name, fig_func in figure_functions:
            try:
                fig = fig_func(data)
                if fig_name == "figure_4c_bland_altman_density":
                    self.save_figure(fig, fig_name, dpi=600)
                else:
                    self.save_figure(fig, fig_name)
                plt.close(fig)
                generated_figures.append(fig_name)
            except Exception as e:
                print(f"  Error generating {fig_name}: {str(e)}")
        
        # Skip shared feature figures for single-platform analysis
        if shared_feature_functions:
            print("Skipping cross-platform feature analyses for single-platform analysis...")
        
        print(f"Generated {len(generated_figures)} basic figures saved to: {self.output_dir / 'figures'}")
        
        # Generate group-dependent figures if biological grouping is available
        if data.groups is not None:
            print("Biological grouping detected - generating additional group-based analyses...")
            group_figures = self.generate_group_dependent_figures(data)
            generated_figures.extend(group_figures)
        else:
            print("No biological grouping found - skipping group-dependent figures (11-13, 16)")
        
        # Generate phenotype-dependent figures if phenotype data is available
        if data.phenotype_data is not None:
            print("\nPhenotype data detected - generating phenotype association analyses...")
            phenotype_figures = self.generate_phenotype_dependent_figures(data)
            generated_figures.extend(phenotype_figures)
        else:
            print("No phenotype data provided - skipping phenotype association figures (28-29)")
        
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
        
        # Get feature names from Platform A only
        feature_names = data.truth_a.index.tolist()
        platform_a_features = data.truth_a.index.tolist()
        
        # Load and process network data
        network_data = self.network_analyzer.load_and_process_networks(
            ppi_file=ppi_file,
            gri_file=gri_file,
            feature_names=feature_names,
            platform_a_features=platform_a_features,
            platform_b_features=[]  # No Platform B features
        )
        
        # Check if we should skip analysis based on mapping results
        total_mapped = len(network_data.get('feature_mapping', {}))
        platform_a_mapped = len(network_data.get('platform_a_mapping', {}))
        
        if total_mapped == 0:
            print("❌ Network analysis skipped - no features mapped to networks")
            return []
        
        if platform_a_mapped == 0:
            print("❌ Network analysis skipped - no features from Platform A mapped")
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
        """Generate figures that require biological grouping (11-13, 16)"""
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
        
        # Figure 16: Sankey diagram (has fallback for no groups)
        try:
            fig16 = self.generate_figure_16_sankey_diagram(data)
            self.save_figure(fig16, "figure_16_sankey_diagram")
            plt.close(fig16)
            group_figures.append("figure_16_sankey_diagram")
        except Exception as e:
            print(f"    Error generating figure 16: {str(e)}")
        
        print(f"  Generated {len(group_figures)} group-dependent figures")
        return group_figures
    
    def generate_phenotype_dependent_figures(self, data: AnalysisData):
        """Generate figures that require phenotype data"""
        if data.phenotype_data is None:
            return []
        
        print("  Calculating phenotype associations...")
        phenotype_figures = []
        
        # Calculate binary phenotype associations
        binary_results = {}
        if data.binary_pheno_cols:
            binary_results = self.calculate_binary_associations(data)
            
            # Save association results
            for phenotype, results_df in binary_results.items():
                results_df.to_csv(self.output_dir / "data" / f"binary_associations_{phenotype}.csv", index=False)
        
        # Calculate continuous phenotype associations  
        continuous_results = {}
        if data.continuous_pheno_cols:
            continuous_results = self.calculate_continuous_associations(data)
            
            # Save association results
            for phenotype, results_df in continuous_results.items():
                results_df.to_csv(self.output_dir / "data" / f"continuous_associations_{phenotype}.csv", index=False)
        
        # Generate forest plots
        if binary_results:
            try:
                fig28 = self.generate_figure_28_phenotype_forest_plots_binary(data, binary_results)
                if fig28 is not None:
                    self.save_figure(fig28, "figure_28_phenotype_forest_plots_binary")
                    plt.close(fig28)
                    phenotype_figures.append("figure_28_phenotype_forest_plots_binary")
                fig28b = self.generate_figure_28b_phenotype_summary_binary(data, binary_results)
                if fig28b is not None:
                    self.save_figure(fig28b, "figure_28b_phenotype_summary_binary")
                    plt.close(fig28b)
                    phenotype_figures.append("figure_28b_phenotype_summary_binary")
            except Exception as e:
                print(f"    Error generating figure 28: {str(e)}")
        
        if continuous_results:
            try:
                fig29 = self.generate_figure_29_phenotype_forest_plots_continuous(data, continuous_results)
                if fig29 is not None:
                    self.save_figure(fig29, "figure_29_phenotype_forest_plots_continuous")
                    plt.close(fig29)
                    phenotype_figures.append("figure_29_phenotype_forest_plots_continuous")
                fig29b = self.generate_figure_29b_phenotype_summary_continuous(data, continuous_results)
                if fig29b is not None:
                    self.save_figure(fig29b, "figure_29b_phenotype_summary_continuous")
                    plt.close(fig29b)
                    phenotype_figures.append("figure_29b_phenotype_summary_continuous")
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
        
        # Calculate improvement matrix for Platform A only
        improvement_data = []
        
        platform = 'Platform_A'
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
        
        # Create DataFrame for bar chart (since we only have Platform A)
        improvement_df = pd.DataFrame(improvement_data)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        groups = improvement_df['group'].tolist()
        improvements = improvement_df['improvement'].tolist()
        
        # Create bar chart with colors based on improvement direction
        colors = ['lightgreen' if imp > 0 else 'lightcoral' for imp in improvements]
        x_positions = range(len(groups))
        bars = ax.bar(x_positions, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Biological Group')
        ax.set_ylabel(f'Median Improvement\n({data.method2_name} - {data.method1_name})')
        ax.set_title(f'Group-Level Performance Improvement: {data.platform_a_name}', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Set x-tick labels to group names
        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups)
        
        # Rotate x-axis labels if needed
        if len(groups) > 5:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_13_group_bland_altman(self, data: AnalysisData):
        """Figure 13: Group-stratified Bland-Altman raincloud"""
        print("  Generating Figure 13: Group-stratified Bland-Altman...")
        
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
        
        fig, axes = plt.subplots(len(groups), 1, figsize=(5, 5*len(groups)))
        if len(groups) == 1:
            axes = [axes]
        
        fig.suptitle(f'Group-Stratified Bland-Altman Analysis: {data.platform_a_name}', fontsize=14, fontweight='bold')
        
        for i, group in enumerate(groups):
            group_features = feat_metrics[feat_metrics['group'] == group]['feature'].unique()
            
            ax = axes[i]
            
            # Get truth and imputed data for this group (Platform A only)
            truth_data = data.truth_a.loc[group_features]
            imp_m1 = data.imp_a_m1.loc[group_features] 
            imp_m2 = data.imp_a_m2.loc[group_features]
            
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
            ax.set_title(f'{group} - {data.platform_a_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_15_network_diagram(self, data: AnalysisData):
        """Figure 15: Network diagram of well-imputed features"""
        print("  Generating Figure 15: Network diagram...")
        
        try:
            import networkx as nx
        except ImportError:
            print("    NetworkX not available for network diagram")
            return plt.figure()
        
        # Get features with good imputation performance for Platform A
        feat_metrics = data.metrics['feature_wise']
        platform_a_metrics = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        # Calculate mean performance across methods for Platform A
        feature_performance = platform_a_metrics.groupby('feature')['r'].mean()
        
        # Filter for well-performing features (use performance-based criteria only)
        good_features = feature_performance[feature_performance > 0.7].index
        
        # Use feature variance as a proxy for feature quality/stability
        feature_variances = data.truth_a.var(axis=1)  # Variance across samples
        stable_features = feature_variances[feature_variances > feature_variances.quantile(0.3)].index
        
        # Find intersection of good performance and reasonable variance
        network_features = list(set(good_features) & set(stable_features))
        
        if len(network_features) < 3:
            print("    Not enough high-quality features for network")
            return self._create_insufficient_data_figure(
                'Network of Well-Imputed Features',
                f'Only {len(network_features)} high-quality features found. Need at least 3.'
            )
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with attributes
        for feature in network_features:
            G.add_node(feature, 
                      performance=feature_performance.get(feature, 0),
                      variance=feature_variances.get(feature, 0))
        
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
        
        # Node colors based on feature variance
        variance_values = [G.nodes[node]['variance'] for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=variance_values, cmap='viridis',
                              alpha=0.8, ax=ax)
        
        # Draw edges with thickness based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'Network of Well-Imputed Features: {data.platform_a_name}', 
                    fontweight='bold', pad=20)
        ax.text(0.02, 0.98, 'Node size: Mean imputation performance\nNode color: Feature variance\nEdge width: Feature correlation', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=min(variance_values), 
                                                   vmax=max(variance_values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Feature Variance', rotation=270, labelpad=15)
        
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
            # Ensure values are numeric and handle NaN/invalid values
            numeric_values = pd.to_numeric(r_values, errors='coerce')
            # Remove any remaining NaN values
            valid_values = numeric_values.dropna()
            if len(valid_values) == 0:
                # If no valid values, return a series with NaN
                return pd.Series([np.nan] * len(r_values), index=r_values.index, dtype='category')
            # Use fixed bin edges to avoid empty categories when values are clustered
            # Include right edge to handle r=1.0 properly
            try:
                return pd.cut(numeric_values, bins=[0, 0.33, 0.66, 1.001], labels=['Low', 'Medium', 'High'], include_lowest=True, right=True)
            except Exception:
                # Fallback: create a simple categorization
                result = pd.Series(['Medium'] * len(numeric_values), index=numeric_values.index, dtype='category')
                return result
        
        # Add group information if available
        if data.groups is not None:
            try:
                group_mapping = data.groups.to_dict()
                pivot_data['group'] = pivot_data.index.map(
                    lambda x: str(group_mapping.get(x, 'Ungrouped')))
            except Exception:
                # Fallback if group mapping fails
                pivot_data['group'] = 'All'
        else:
            pivot_data['group'] = 'All'
        
        # Create tiers
        pivot_data['tier_m1'] = get_tier(pivot_data['Method_1'])
        pivot_data['tier_m2'] = get_tier(pivot_data['Method_2'])
        
        # Remove rows with NaN tiers before grouping
        valid_tiers = pivot_data.dropna(subset=['tier_m1', 'tier_m2'])
        
        if len(valid_tiers) == 0:
            print("    No valid tier data for Sankey diagram")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No valid tier data for Sankey diagram', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Feature Performance Transitions (Sankey-style)')
            ax.axis('off')
            plt.tight_layout()
            return fig
        
        # Count transitions
        transition_counts = valid_tiers.groupby(['group', 'tier_m1', 'tier_m2']).size().reset_index(name='count')
        
        # Create a simplified Sankey-style visualization using matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
            
            # Skip if tiers are NaN
            if pd.isna(source_tier) or pd.isna(target_tier):
                continue
            
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
        
        # Get available methods dynamically
        available_methods = self._get_available_methods(data)
        unique_methods = list(set([(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods if platform == 'Platform_A']))
        
        n_methods = len(unique_methods)
        if n_methods == 0:
            return self._create_insufficient_data_figure(
                'Performance by Biological Group',
                'No methods available for analysis'
            )
        
        # Create facet grid: 1 row with n_methods columns
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'Performance by Biological Group: {data.platform_a_name}', fontsize=14, fontweight='bold')
        
        # Only process Platform A data
        for j, (method_key, method_name) in enumerate(unique_methods):
            ax = axes[j]
            
            # Filter data for Platform A and this method
            subset = feat_metrics[(feat_metrics['platform'] == 'Platform_A') & 
                                (feat_metrics['method'] == method_key)]
            
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
            
            ax.set_title(f'{data.platform_a_name} - {method_name}')
            ax.set_ylabel('Correlation (r)')
            ax.set_xlabel('Biological Group')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if len(groups) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_14_volcano_plot(self, data: AnalysisData):
        """Figure 14: Volcano plot - Performance difference analysis"""
        print("  Generating Figure 14: Volcano plot...")
        
        # Calculate Δr (Method 2 - Method 1) for Platform A
        feat_metrics = data.metrics['feature_wise']
        
        # Create alternative plot using within-platform performance difference
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle(f'Performance Difference Analysis: {data.platform_a_name}', 
                    fontsize=14, fontweight='bold')
        
        platform_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        # Pivot by feature and method
        pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
        
        if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
            delta_r = pivot_data['Method_2'] - pivot_data['Method_1']
            avg_r = (pivot_data['Method_1'] + pivot_data['Method_2']) / 2
            
            # Calculate feature variance as a proxy for feature stability
            feature_vars = data.truth_a.var(axis=1)
            aligned_vars = feature_vars.reindex(delta_r.index).fillna(np.nan)
            
            # Create scatter plot colored by feature variance
            valid_mask = ~(np.isnan(delta_r) | np.isnan(avg_r) | np.isnan(aligned_vars))
            
            if np.any(valid_mask):
                scatter = ax.scatter(avg_r[valid_mask], delta_r[valid_mask], 
                                   c=aligned_vars[valid_mask], alpha=0.6, s=30, 
                                   cmap='viridis', edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Feature Variance', rotation=270, labelpad=15)
            else:
                ax.scatter(avg_r, delta_r, alpha=0.6, s=30, 
                          color=NATURE_COLORS['primary'], edgecolors='black', linewidth=0.5)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Find and label extreme points
            if len(delta_r.dropna()) > 5:
                # Most improved features
                top_improved = delta_r.nlargest(3)
                for feature, improvement in top_improved.items():
                    if not np.isnan(improvement):
                        avg_perf = avg_r.get(feature, np.nan)
                        if not np.isnan(avg_perf):
                            ax.annotate(feature[:10], (avg_perf, improvement),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, alpha=0.8,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                
                # Most degraded features
                top_degraded = delta_r.nsmallest(3)
                for feature, degradation in top_degraded.items():
                    if not np.isnan(degradation):
                        avg_perf = avg_r.get(feature, np.nan)
                        if not np.isnan(avg_perf):
                            ax.annotate(feature[:10], (avg_perf, degradation),
                                       xytext=(5, -15), textcoords='offset points',
                                       fontsize=8, alpha=0.8,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            ax.set_xlabel('Average Performance')
            ax.set_ylabel(f'Δr ({data.method2_name} - {data.method1_name})')
            ax.set_title(f'{data.platform_a_name} - Method Performance Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add quadrant labels
            ax.text(0.95, 0.95, 'High Performance\nImproved', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax.text(0.95, 0.05, 'High Performance\nWorsened', transform=ax.transAxes, 
                   ha='right', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Insufficient method data for comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{data.platform_a_name} - Method Performance Comparison')
        
        plt.tight_layout()
        return fig
    
    def generate_figure_17_performance_consistency(self, data: AnalysisData):
        """Figure 17: Performance consistency across samples and features"""
        print("Generating Figure 17: Performance consistency analysis...")
        
        # Calculate coefficient of variation (CV) for each method
        feat_metrics = data.metrics['feature_wise']
        samp_metrics = data.metrics['sample_wise']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Performance Consistency Analysis: {data.platform_a_name}', fontsize=14, fontweight='bold')
        
        # Feature-wise consistency for Platform A
        platform_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        for method, color in [('Method_1', NATURE_COLORS['primary']), 
                             ('Method_2', NATURE_COLORS['secondary'])]:
            method_data = platform_data[platform_data['method'] == method]['r']
            
            if len(method_data) > 10:  # Need enough data for rolling statistics
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
                ax1.plot(range(len(sorted_data)), rolling_std, 
                       color=color, linewidth=2, label=f'{method_name} (CV={cv:.3f})')
        
        ax1.set_xlabel('Feature Rank (by performance)')
        ax1.set_ylabel('Rolling Standard Deviation')
        ax1.set_title(f'{data.platform_a_name} - Feature Consistency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample-wise consistency for Platform A  
        platform_sample_data = samp_metrics[samp_metrics['platform'] == 'Platform_A']
        
        for method, color in [('Method_1', NATURE_COLORS['primary']), 
                             ('Method_2', NATURE_COLORS['secondary'])]:
            method_data = platform_sample_data[platform_sample_data['method'] == method]['r']
            
            if len(method_data) > 0:
                # Create histogram of performance
                ax2.hist(method_data, bins=20, alpha=0.6, color=color, density=True,
                       label=f'{data.method1_name if method == "Method_1" else data.method2_name}')
        
        ax2.set_xlabel('Sample-wise Correlation')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{data.platform_a_name} - Sample Performance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_figure_18_method_synergy_analysis(self, data: AnalysisData):
        """Figure 18: Method synergy and complementarity analysis"""
        print("Generating Figure 18: Method synergy analysis...")
        
        feat_metrics = data.metrics['feature_wise']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Method Synergy and Complementarity Analysis: {data.platform_a_name}', fontsize=14, fontweight='bold')
        
        platform_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        pivot_data = platform_data.pivot(index='feature', columns='method', values='r')
        
        if 'Method_1' in pivot_data.columns and 'Method_2' in pivot_data.columns:
            m1_scores = pivot_data['Method_1'].dropna()
            m2_scores = pivot_data['Method_2'].dropna()
            
            # Synergy analysis - which features benefit from ensemble?
            ax = ax1
            
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
            ax.set_title(f'{data.platform_a_name} - Synergy Potential')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{data.method1_name} - {data.method2_name}', rotation=270, labelpad=15)
            
            # Complementarity analysis
            ax_comp = ax2
            
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
            ax_comp.set_title(f'{data.platform_a_name} - Method Complementarity')
            
            # Add quadrant annotations
            ax_comp.text(0.95, 0.95, 'High Disagreement\nHigh Agreement', 
                       transform=ax_comp.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax_comp.text(0.05, 0.05, 'Low Disagreement\nLow Agreement', 
                       transform=ax_comp.transAxes, ha='left', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, 'Insufficient method data for synergy analysis', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title(f'{data.platform_a_name} - Synergy Potential')
            
            ax2.text(0.5, 0.5, 'Insufficient method data for complementarity analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'{data.platform_a_name} - Method Complementarity')
        
        plt.tight_layout()
        return fig
    

    
    def generate_figure_20_feature_difficulty_profiling(self, data: AnalysisData):
        """Figure 20: Feature difficulty profiling and characterization"""
        print("Generating Figure 20: Feature difficulty profiling...")
        
        # Characterize features by their "difficulty" to impute
        feat_metrics = data.metrics['feature_wise']
        cross_r2 = None  # Not available in single-platform analysis
        
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
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
        """Figure 21: Temporal performance trends (using feature ordering)"""
        print("Generating Figure 21: Temporal performance trends...")
        
        # Use feature ordering as a proxy for temporal/batch effects
        feat_metrics = data.metrics['feature_wise']
        
        # Only use Platform A data
        platform_data = feat_metrics[feat_metrics['platform'] == 'Platform_A']
        
        if len(platform_data) == 0:
            return self._create_insufficient_data_figure(
                "Temporal Performance Trends",
                "No data available for Platform A"
            )
        
        # Get available methods dynamically
        available_methods = self._get_available_methods(data)
        platform_a_methods = [(method_key, method_name) for method_key, method_name, platform, truth, imputed in available_methods if platform == 'Platform_A']
        
        n_methods = len(platform_a_methods)
        if n_methods == 0:
            return self._create_insufficient_data_figure(
                "Temporal Performance Trends",
                "No methods available for analysis"
            )
        
        # Create layout: 2 rows (trends and stability), n_methods columns
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Performance Trends Analysis: {data.platform_a_name}\\n(Feature-Order Based)', fontsize=14, fontweight='bold')
        
        # Add feature index for temporal ordering
        platform_data = platform_data.copy()
        feature_to_idx = {feat: idx for idx, feat in enumerate(data.truth_a.index)}
        platform_data['feature_order'] = platform_data['feature'].map(feature_to_idx)
        
        # Window size for rolling statistics
        window_size = max(10, len(platform_data) // 20)
        
        for j, (method_key, method_name) in enumerate(platform_a_methods):
            method_data = platform_data[platform_data['method'] == method_key].copy()
            method_data = method_data.sort_values('feature_order')
            
            if len(method_data) >= window_size:
                # Performance trends
                ax = axes[0, j]
                
                # Calculate rolling statistics
                method_data['rolling_mean'] = method_data['r'].rolling(window=window_size, center=True).mean()
                method_data['rolling_std'] = method_data['r'].rolling(window=window_size, center=True).std()
                
                # Plot raw data
                ax.scatter(method_data['feature_order'], method_data['r'], 
                          alpha=0.3, s=15, color=NATURE_COLORS['neutral'], label='Raw data')
                
                # Plot rolling mean
                ax.plot(method_data['feature_order'], method_data['rolling_mean'], 
                       color=NATURE_COLORS['primary'], linewidth=2, label=f'Rolling mean (n={window_size})')
                
                # Add confidence band
                upper = method_data['rolling_mean'] + method_data['rolling_std']
                lower = method_data['rolling_mean'] - method_data['rolling_std']
                ax.fill_between(method_data['feature_order'], lower, upper, 
                               alpha=0.2, color=NATURE_COLORS['primary'])
                
                ax.set_xlabel('Feature Order (Temporal Proxy)')
                ax.set_ylabel('Performance (r)')
                ax.set_title(f'{data.platform_a_name} - {method_name}\\nPerformance Trends')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Stability analysis
                ax_stab = axes[1, j]
                
                # Calculate local variability
                method_data['local_var'] = method_data['r'].rolling(window=window_size, center=True).var()
                
                ax_stab.plot(method_data['feature_order'], method_data['local_var'], 
                            color=NATURE_COLORS['accent'], linewidth=2)
                ax_stab.fill_between(method_data['feature_order'], 0, method_data['local_var'], 
                                    alpha=0.3, color=NATURE_COLORS['accent'])
                
                ax_stab.set_xlabel('Feature Order (Temporal Proxy)')
                ax_stab.set_ylabel('Local Variance')
                ax_stab.set_title(f'{data.platform_a_name} - {method_name}\\nStability Analysis')
                ax_stab.grid(True, alpha=0.3)
                
                # Add trend statistics
                try:
                    slope, intercept, r_val, p_val, std_err = linregress(
                        method_data['feature_order'].values, 
                        method_data['r'].values
                    )
                    
                    ax.text(0.05, 0.95, f'Trend: slope={slope:.4f}\\np={p_val:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
                
            else:
                # Insufficient data
                axes[0, j].text(0.5, 0.5, f'Insufficient data\\n(n={len(method_data)} < {window_size})', 
                              ha='center', va='center', transform=axes[0, j].transAxes, fontsize=12)
                axes[0, j].set_title(f'{data.platform_a_name} - {method_name}\\nPerformance Trends')
                
                axes[1, j].text(0.5, 0.5, f'Insufficient data\\n(n={len(method_data)} < {window_size})', 
                              ha='center', va='center', transform=axes[1, j].transAxes, fontsize=12)
                axes[1, j].set_title(f'{data.platform_a_name} - {method_name}\\nStability Analysis')
        
        plt.tight_layout()
        return fig
    

    


    

    
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


 
def main():
    """
    Main function to run comprehensive single-platform imputation analysis.
    
    Parses command-line arguments, loads data, runs complete analysis pipeline,
    and generates publication-ready figures comparing imputation methods.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive Single-Platform Proteomics Imputation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (2 methods):
    python compare_result_oneplatform.py \\
        --truth_a data/truth_platform_a.csv \\
        --imp_a_m1 data/method1_platform_a.csv \\
        --imp_a_m2 data/method2_platform_a.csv \\
        --method1_name "Method 1" \\
        --method2_name "Method 2" \\
        --platform_a_name "Platform A" \\
        --output_dir results

Example usage (4 methods):
    python compare_result_oneplatform.py \\
        --truth_a data/truth_platform_a.csv \\
        --imp_a_m1 data/method1_platform_a.csv \\
        --imp_a_m2 data/method2_platform_a.csv \\
        --imp_a_m3 data/method3_platform_a.csv \\
        --imp_a_m4 data/method4_platform_a.csv \\
        --method1_name "Method 1" \\
        --method2_name "Method 2" \\
        --method3_name "Method 3" \\
        --method4_name "Method 4" \\
        --platform_a_name "Platform A" \\
        --output_dir results \\
        --ppi_file data/ppi.txt \\
        --gri_file data/gri.txt

Example usage with phenotype analysis:
    python compare_result_oneplatform.py \\
        --truth_a data/truth_platform_a.csv \\
        --imp_a_m1 data/method1_platform_a.csv \\
        --imp_a_m2 data/method2_platform_a.csv \\
        --method1_name "Method 1" \\
        --method2_name "Method 2" \\
        --platform_a_name "Platform A" \\
        --phenotype_file data/phenotypes.csv \\
        --binary_pheno DIABETES HYPERTENSION \\
        --continuous_pheno AGE BMI \\
        --output_dir results
        
Network files format:
    PPI file: Tab-separated with columns GENE1, GENE2 (protein-protein interactions)
    GRI file: Tab-separated with columns FROM, TO (gene regulatory interactions)
    
Phenotype file format:
    CSV/TXT file with samples as rows (matching sample IDs in proteomics data)
    Columns represent different phenotypes (binary or continuous)
    Binary phenotypes: 0/1 or case/control values
    Continuous phenotypes: numeric values
        """
    )
    
    # Required arguments
    parser.add_argument('--truth_a', required=True, help='Truth file for platform A')
    parser.add_argument('--imp_a_m1', required=True, help='Method 1 imputed platform A')
    parser.add_argument('--imp_a_m2', required=True, help='Method 2 imputed platform A')
    parser.add_argument('--method1_name', required=True, help='Display name for method 1')
    parser.add_argument('--method2_name', required=True, help='Display name for method 2')
    parser.add_argument('--platform_a_name', required=True, help='Display name for platform A')
    
    # Optional additional methods (3 and 4)
    parser.add_argument('--imp_a_m3', help='Method 3 imputed platform A (optional)')
    parser.add_argument('--imp_a_m4', help='Method 4 imputed platform A (optional)')
    parser.add_argument('--method3_name', help='Display name for method 3 (optional)')
    parser.add_argument('--method4_name', help='Display name for method 4 (optional)')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='analysis_output', help='Output directory')
    parser.add_argument('--ppi_file', help='PPI network file (tab-separated, columns: GENE1, GENE2)')
    parser.add_argument('--gri_file', help='GRI network file (tab-separated, columns: FROM, TO)')
    parser.add_argument('--transpose', action='store_true', help='Transpose input files (use if rows=samples, columns=features)')
    
    # Phenotype analysis arguments (optional)
    parser.add_argument('--phenotype_file', help='Phenotype file (CSV or TXT) with sample IDs as index')
    parser.add_argument('--binary_pheno', nargs='+', help='Column names for binary phenotypes')
    parser.add_argument('--continuous_pheno', nargs='+', help='Column names for continuous phenotypes')
    parser.add_argument('--gender_col', help='Optional gender column name for covariate adjustment')
    parser.add_argument('--age_col', help='Optional age column name for covariate adjustment')
    
    args = parser.parse_args()
    
    # Prepare file paths
    file_paths = {
        'truth_a': args.truth_a,
        'imp_a_m1': args.imp_a_m1,
        'imp_a_m2': args.imp_a_m2,
    }
    
    # Add optional methods if provided
    if args.imp_a_m3:
        file_paths['imp_a_m3'] = args.imp_a_m3
    if args.imp_a_m4:
        file_paths['imp_a_m4'] = args.imp_a_m4
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer(args.output_dir, gender_col=args.gender_col, age_col=args.age_col)
    
    print("="*80)
    print("Single-Platform Proteomics Imputation Analysis")
    print("="*80)
    print(f"Methods: {args.method1_name} vs {args.method2_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Git hash: {analyzer.git_hash}")
    print(f"Timestamp: {analyzer.timestamp}")
    print()
    
    try:
        # Load and validate data
        data = analyzer.load_and_validate_data(file_paths, args.method1_name, args.method2_name, args.platform_a_name, args.method3_name, args.method4_name, args.transpose)
        print(f"Loaded data: Platform A: {data.truth_a.shape[0]} features × {data.truth_a.shape[1]} samples")
        if data.groups is not None:
            print(f"Biological groups found: {data.groups.nunique()} unique groups")
        print()
        
        # Load phenotype data if provided
        if args.phenotype_file:
            # Set phenotype column specifications
            data.binary_pheno_cols = args.binary_pheno
            data.continuous_pheno_cols = args.continuous_pheno
            
            # Load and validate phenotype data
            data = analyzer.load_phenotype_data(args.phenotype_file, data)
            print()
        
        # Compute metrics
        data = analyzer.compute_all_metrics(data)
        print()
        
        # Generate figures
        generated_figures = analyzer.generate_all_figures(data)
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