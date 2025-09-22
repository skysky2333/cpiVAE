#!/usr/bin/env python3
"""
Comprehensive cis-pQTL Analysis Script

Compares pQTL association results from multiple imputation methods against ground truth data.
Provides statistical analysis, performance metrics, and publication-ready visualizations
for evaluating cross-platform proteomics imputation methods.

"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests

try:
    import upsetplot
    UPSETPLOT_AVAILABLE = True
except ImportError:
    print("Warning: upsetplot not available. UpSet plots will be skipped.")
    UPSETPLOT_AVAILABLE = False

try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
    VENN_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib-venn not available. Venn diagrams will be skipped.")
    VENN_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
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
    'figure.figsize': (8, 6)
})

NATURE_COLORS = {
    'primary': '#e64b35',      # Red (Truth)
    'secondary': '#4dbbd5',    # Light Blue (Method 1)
    'accent': '#00a087',       # Teal (Method 2) 
    'neutral': '#3c5488',      # Dark Blue (Method 3)
    'highlight': '#f39b7f',    # Light Red (Method 4)
    'alternative_1': '#bc3c29', # Dark Red
    'alternative_2': '#0072b5', # Blue
    'alternative_3': '#e18727', # Orange
    'alternative_4': '#20854e', # Green
    'alternative_5': '#7876b1'  # Purple
}

@dataclass
class pQTLData:
    """Container for pQTL analysis data and results.
    
    Stores input data, analysis parameters, and computed results for
    comprehensive cis-pQTL analysis comparing multiple imputation methods.
    
    Attributes:
        truth: Ground truth pQTL results DataFrame
        method1-method4: Method-specific pQTL results DataFrames
        method_names: Display names for each method
        significance_threshold: P-value threshold for significance (default: 0.05)
        significant_hits: Filtered significant associations by dataset
        overlap_analysis: Overlap statistics between methods and truth
        effect_size_analysis: Effect size concordance analysis results
        performance_metrics: Composite performance scores for method ranking
    """
    truth: pd.DataFrame
    method1: pd.DataFrame
    method2: pd.DataFrame
    method3: pd.DataFrame
    method4: pd.DataFrame
    method_names: List[str]
    
    significance_threshold: float = 0.05
    
    maf_filtered: Dict[str, pd.DataFrame] = None  # Data after MAF filtering
    significant_hits: Dict[str, pd.DataFrame] = None
    overlap_analysis: Dict[str, any] = None
    effect_size_analysis: Dict[str, any] = None
    performance_metrics: Dict[str, float] = None

class CispQTLAnalyzer:
    """Comprehensive cis-pQTL analysis framework.
    
    Provides end-to-end analysis pipeline for comparing multiple imputation
    methods against ground truth data. Includes overlap analysis, effect size
    concordance, performance ranking, and publication-ready visualizations.
        
    Attributes:
        output_dir: Directory for saving results
        significance_threshold: P-value threshold for significance
        timestamp: Analysis timestamp for file naming
    """
    
    def __init__(self, output_dir: str = "cis_pqtl_analysis", significance_threshold: float = 0.05, 
                 truth_threshold: float = None, maf_threshold: float = 0.01):
        """Initialize the cis-pQTL analyzer.
        
        Args:
            output_dir: Directory for saving analysis results
            significance_threshold: P-value threshold for significance
            truth_threshold: Fixed p-value threshold for truth dataset in PR analysis (defaults to significance_threshold)
            maf_threshold: Minor Allele Frequency threshold for filtering variants
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.significance_threshold = significance_threshold
        self.truth_threshold = truth_threshold if truth_threshold is not None else significance_threshold
        self.maf_threshold = maf_threshold
        
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _deduplicate_associations(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Remove duplicate association IDs and log the process.
        
        Args:
            df: DataFrame containing pQTL associations
            dataset_name: Name of dataset for logging purposes
            
        Returns:
            DataFrame with duplicate associations removed
        """
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=['assoc_id'], keep='first')
        final_count = len(df_clean)
        
        if initial_count != final_count:
            print(f"    {dataset_name}: Removed {initial_count - final_count} duplicate associations ({final_count} remaining)")
        
        return df_clean
        
    def load_and_validate_data(self, file_paths: Dict[str, str], method_names: List[str]) -> pQTLData:
        """Load and validate all pQTL TSV files.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            method_names: List of method display names
            
        Returns:
            pQTLData object containing loaded and validated data
            
        Raises:
            FileNotFoundError: If any input file is not found
            ValueError: If required columns are missing from any file
        """
        print("Loading and validating pQTL files...")
        
        required_columns = ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se']
        
        data_frames = {}
        for key, path in file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            df = pd.read_csv(path, sep='\t')
            print(f"  Loading {key}: {len(df)} associations")
            
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns in {key}: {missing_cols}")
            
            if df.empty:
                print(f"    WARNING: {key} file is empty")
            
            critical_missing = df[required_columns].isnull().sum()
            if critical_missing.any():
                print(f"    WARNING: Missing values in {key}:")
                for col, count in critical_missing[critical_missing > 0].items():
                    print(f"      {col}: {count} missing values")
            
            df['pval_nominal'] = pd.to_numeric(df['pval_nominal'], errors='coerce')
            df['slope'] = pd.to_numeric(df['slope'], errors='coerce')
            df['slope_se'] = pd.to_numeric(df['slope_se'], errors='coerce')
            
            data_frames[key] = df
        
        data = pQTLData(
            truth=data_frames['truth'],
            method1=data_frames['method1'],
            method2=data_frames['method2'], 
            method3=data_frames['method3'],
            method4=data_frames['method4'],
            method_names=method_names,
            significance_threshold=self.significance_threshold
        )
        
        print(f"Successfully loaded all files with significance threshold p < {self.significance_threshold}")
        return data
        
    def filter_significant_hits(self, data: pQTLData) -> pQTLData:
        """Filter for significant associations in each dataset.
        
        Args:
            data: pQTLData object with loaded data
            
        Returns:
            Updated pQTLData object with significant hits filtered
        """
        print("Filtering for significant associations...")
        
        significant_hits = {}
        datasets = {
            'truth': data.truth,
            'method1': data.method1, 
            'method2': data.method2,
            'method3': data.method3,
            'method4': data.method4
        }
        
        for name, df in datasets.items():
            sig_mask = (df['pval_nominal'] < data.significance_threshold) & df['pval_nominal'].notna()
            sig_df = df[sig_mask].copy()
            
            sig_df['assoc_id'] = sig_df['phenotype_id'] + ':' + sig_df['variant_id']
            
            sig_df = self._deduplicate_associations(sig_df, name)
            
            significant_hits[name] = sig_df
            print(f"  {name}: {len(sig_df)} significant hits (p < {data.significance_threshold})")
        
        data.significant_hits = significant_hits
        return data
    
    def filter_by_maf(self, data: pQTLData) -> pQTLData:
        """Filter associations based on Minor Allele Frequency threshold.
        
        Removes variants with allele frequency below the MAF threshold or above 
        (1 - MAF threshold), effectively filtering out both rare variants and 
        near-fixed variants.
        
        Args:
            data: pQTLData object with loaded data
            
        Returns:
            Updated pQTLData object with MAF filtering applied
        """
        print(f"Filtering variants by MAF threshold: {self.maf_threshold}")
        print(f"  Keeping variants with AF in range [{self.maf_threshold}, {1 - self.maf_threshold}]")
        
        datasets = {
            'truth': data.truth,
            'method1': data.method1,
            'method2': data.method2,
            'method3': data.method3,
            'method4': data.method4
        }
        
        filtered_data = {}
        for name, df in datasets.items():
            original_count = len(df)
            
            # Check if 'af' column exists
            if 'af' not in df.columns:
                print(f"  WARNING: {name} does not have 'af' column, skipping MAF filtering")
                filtered_data[name] = df
                continue
            
            # Filter based on MAF threshold
            # Keep variants where AF is between [maf_threshold, 1-maf_threshold]
            maf_mask = (df['af'] >= self.maf_threshold) & (df['af'] <= (1 - self.maf_threshold))
            filtered_df = df[maf_mask].copy()
            
            filtered_count = len(filtered_df)
            removed_count = original_count - filtered_count
            
            print(f"  {name}: {filtered_count}/{original_count} variants kept ({removed_count} removed by MAF filter)")
            filtered_data[name] = filtered_df
        
        # Update data object with filtered dataframes
        data.truth = filtered_data['truth']
        data.method1 = filtered_data['method1']
        data.method2 = filtered_data['method2']
        data.method3 = filtered_data['method3']
        data.method4 = filtered_data['method4']
        
        # Store MAF-filtered data separately for p-value comparisons
        data.maf_filtered = {
            'truth': filtered_data['truth'].copy(),
            'method1': filtered_data['method1'].copy(),
            'method2': filtered_data['method2'].copy(),
            'method3': filtered_data['method3'].copy(),
            'method4': filtered_data['method4'].copy()
        }
        
        return data
    
    def analyze_overlaps(self, data: pQTLData) -> pQTLData:
        """Execute comprehensive overlap analysis between methods and truth.
        
        Calculates overlap statistics at phenotype, variant, and association levels,
        including sensitivity, precision, F1 score, and Jaccard index.
        
        Args:
            data: pQTLData object with significant hits
            
        Returns:
            Updated pQTLData object with overlap analysis results
        """
        print("Analyzing overlaps between methods and truth...")
        
        sig_hits = data.significant_hits
        
        phenotype_sets = {}
        for name, df in sig_hits.items():
            phenotype_sets[name] = set(df['phenotype_id'].unique())
        
        variant_sets = {}
        for name, df in sig_hits.items():
            variant_sets[name] = set(df['variant_id'].unique())
        
        assoc_sets = {}
        for name, df in sig_hits.items():
            assoc_sets[name] = set(df['assoc_id'].unique())
        
        overlap_stats = {}
        method_names = ['method1', 'method2', 'method3', 'method4']
        
        for level, sets_dict in [('phenotype', phenotype_sets), 
                                ('variant', variant_sets),
                                ('association', assoc_sets)]:
            
            overlap_stats[level] = {}
            truth_set = sets_dict['truth']
            
            for i, method in enumerate(method_names):
                method_set = sets_dict[method]
                
                intersection = truth_set & method_set
                union = truth_set | method_set
                
                overlap_count = len(intersection)
                truth_only = len(truth_set - method_set)
                method_only = len(method_set - truth_set)
                if len(truth_set) > 0:
                    sensitivity = len(intersection) / len(truth_set)  # Recall
                else:
                    sensitivity = 0
                    
                if len(method_set) > 0:
                    precision = len(intersection) / len(method_set)
                else:
                    precision = 0
                    
                if len(union) > 0:
                    jaccard = len(intersection) / len(union)
                else:
                    jaccard = 0
                
                overlap_stats[level][method] = {
                    'method_name': data.method_names[i],
                    'truth_total': len(truth_set),
                    'method_total': len(method_set),
                    'overlap_count': overlap_count,
                    'truth_only': truth_only,
                    'method_only': method_only,
                    'sensitivity': sensitivity,
                    'precision': precision,
                    'jaccard': jaccard,
                    'f1_score': 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                }
        
        data.overlap_analysis = {
            'phenotype_sets': phenotype_sets,
            'variant_sets': variant_sets, 
            'assoc_sets': assoc_sets,
            'overlap_stats': overlap_stats
        }
        
        print("  Overlap analysis completed:")
        for level in ['phenotype', 'variant', 'association']:
            print(f"    {level.capitalize()} level:")
            for method in method_names:
                stats = overlap_stats[level][method]
                print(f"      {stats['method_name']}: {stats['overlap_count']}/{stats['truth_total']} "
                      f"(sensitivity: {stats['sensitivity']:.3f}, precision: {stats['precision']:.3f})")
        
        return data
    
    def analyze_effect_sizes(self, data: pQTLData) -> pQTLData:
        """Analyze effect size concordance between methods and truth.
        
        Computes correlation metrics, error metrics, and directional concordance
        for common significant associations between methods and truth data.
        
        Args:
            data: pQTLData object with significant hits
            
        Returns:
            Updated pQTLData object with effect size analysis results
        """
        print("Analyzing effect size concordance...")
        
        sig_hits = data.significant_hits
        
        truth_raw = sig_hits['truth'].copy()
        truth_raw = self._deduplicate_associations(truth_raw, 'truth data')
        truth_df = truth_raw.set_index('assoc_id')
        
        effect_analysis = {}
        method_names = ['method1', 'method2', 'method3', 'method4']
        
        for i, method in enumerate(method_names):
            method_name = data.method_names[i]
            
            method_raw = sig_hits[method].copy()
            method_raw = self._deduplicate_associations(method_raw, f'{method_name} data')
            method_df = method_raw.set_index('assoc_id')
            
            common_assocs = truth_df.index.intersection(method_df.index)
            
            if len(common_assocs) == 0:
                print(f"    WARNING: No common significant associations between truth and {method_name}")
                continue
            
            print(f"    {method_name}: {len(common_assocs)} common associations")
            
            truth_slopes = truth_df.loc[common_assocs, 'slope'].values
            method_slopes = method_df.loc[common_assocs, 'slope'].values
            
            
            if len(truth_slopes) != len(method_slopes):
                print(f"    ERROR: Array length mismatch for {method_name}")
                continue
            
            mask = ~(np.isnan(truth_slopes) | np.isnan(method_slopes))
            truth_slopes_clean = truth_slopes[mask]
            method_slopes_clean = method_slopes[mask]
            
            if len(truth_slopes_clean) < 3:
                print(f"    WARNING: Too few clean effect size pairs for {method_name}")
                continue
            
            
            pearson_r, pearson_p = pearsonr(truth_slopes_clean, method_slopes_clean)
            spearman_r, spearman_p = spearmanr(truth_slopes_clean, method_slopes_clean)
            
            mae = mean_absolute_error(truth_slopes_clean, method_slopes_clean)
            rmse = np.sqrt(mean_squared_error(truth_slopes_clean, method_slopes_clean))
            bias = np.mean(method_slopes_clean - truth_slopes_clean)
            concordance = np.mean(np.sign(truth_slopes_clean) == np.sign(method_slopes_clean))
            
            effect_analysis[method] = {
                'method_name': method_name,
                'n_common': len(common_assocs),
                'n_clean': len(truth_slopes_clean),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'mae': mae,
                'rmse': rmse,
                'bias': bias,
                'concordance': concordance,
                'truth_slopes': truth_slopes_clean,
                'method_slopes': method_slopes_clean,
                'common_assocs': common_assocs
            }
            
            print(f"    {method_name}: r={pearson_r:.3f}, MAE={mae:.3f}, concordance={concordance:.3f} (n={len(truth_slopes_clean)})")
        
        data.effect_size_analysis = effect_analysis
        return data
    
    def calculate_performance_metrics(self, data: pQTLData) -> pQTLData:
        """Calculate comprehensive performance metrics for method ranking.
        
        Computes a weighted composite score based on overlap performance (40%),
        effect size accuracy (30%), statistical concordance (20%), and
        discovery consistency (10%).
        
        Args:
            data: pQTLData object with overlap and effect size analyses
            
        Returns:
            Updated pQTLData object with performance metrics
        """
        print("Calculating comprehensive performance metrics...")
        
        performance_metrics = {}
        method_names = ['method1', 'method2', 'method3', 'method4']
        
        for i, method in enumerate(method_names):
            method_name = data.method_names[i]
            
            scores = {}
            if data.overlap_analysis:
                overlap_scores = []
                for level in ['phenotype', 'variant', 'association']:
                    if method in data.overlap_analysis['overlap_stats'][level]:
                        stats = data.overlap_analysis['overlap_stats'][level][method]
                        # Weighted combination of sensitivity, precision, and F1
                        overlap_score = (0.4 * stats['sensitivity'] + 
                                       0.3 * stats['precision'] + 
                                       0.3 * stats['f1_score'])
                        overlap_scores.append(overlap_score)
                
                scores['overlap_score'] = np.mean(overlap_scores) if overlap_scores else 0
            else:
                scores['overlap_score'] = 0
            
            if data.effect_size_analysis and method in data.effect_size_analysis:
                effect_stats = data.effect_size_analysis[method]
                correlation_score = max(effect_stats['pearson_r'], 0)
                max_mae = max([data.effect_size_analysis[m]['mae'] 
                              for m in data.effect_size_analysis.keys()] + [1])
                mae_score = 1 - (effect_stats['mae'] / max_mae)
                concordance_score = effect_stats['concordance']
                
                scores['effect_size_score'] = (0.4 * correlation_score + 
                                             0.3 * mae_score + 
                                             0.3 * concordance_score)
            else:
                scores['effect_size_score'] = 0
            
            if data.effect_size_analysis and method in data.effect_size_analysis:
                effect_stats = data.effect_size_analysis[method]
                stat_score = max(effect_stats['spearman_r'], 0)
                scores['statistical_score'] = stat_score
            else:
                scores['statistical_score'] = 0
            
            if data.overlap_analysis:
                precisions = []
                for level in ['phenotype', 'variant', 'association']:
                    if method in data.overlap_analysis['overlap_stats'][level]:
                        precisions.append(data.overlap_analysis['overlap_stats'][level][method]['precision'])
                scores['discovery_score'] = np.mean(precisions) if precisions else 0
            else:
                scores['discovery_score'] = 0
            
            composite_score = (0.4 * scores['overlap_score'] + 
                             0.3 * scores['effect_size_score'] + 
                             0.2 * scores['statistical_score'] + 
                             0.1 * scores['discovery_score'])
            
            performance_metrics[method] = {
                'method_name': method_name,
                'composite_score': composite_score,
                **scores
            }
            
            print(f"    {method_name}: Composite score = {composite_score:.3f}")
        
        data.performance_metrics = performance_metrics
        return data
    
    def analyze_precision_recall_curves(self, data: pQTLData) -> Dict[str, any]:
        """Analyze precision-recall curves across significance thresholds.
        
        Evaluates method performance across different p-value thresholds for
        method predictions while keeping truth dataset at a fixed threshold.
        This properly evaluates how well methods recover true positives at different stringency levels.
        
        Args:
            data: pQTLData object with loaded data
            
        Returns:
            Dictionary containing PR analysis results for each method and level
        """
        print(f"Analyzing precision-recall curves (truth fixed at p < {self.truth_threshold})...")
        
        thresholds = [5e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]
        
        pr_results = {}
        method_names = ['method1', 'method2', 'method3', 'method4']
        
        datasets = {
            'truth': data.truth,
            'method1': data.method1,
            'method2': data.method2,
            'method3': data.method3,
            'method4': data.method4
        }
        
        # Filter truth dataset once with fixed threshold
        truth_sig = datasets['truth'][datasets['truth']['pval_nominal'] < self.truth_threshold].copy()
        print(f"  Truth dataset: {len(truth_sig)} associations at p < {self.truth_threshold}")
        
        # Pre-process truth sets for each level
        truth_sig['assoc_id'] = truth_sig['phenotype_id'] + ':' + truth_sig['variant_id']
        truth_sig_clean = self._deduplicate_associations(truth_sig, f'truth at p<{self.truth_threshold}')
        truth_assoc_set = set(truth_sig_clean['assoc_id'])
        truth_phenotype_set = set(truth_sig['phenotype_id'].unique())
        truth_variant_set = set(truth_sig['variant_id'].unique())
        
        truth_sets = {
            'associations': truth_assoc_set,
            'phenotypes': truth_phenotype_set,
            'variants': truth_variant_set
        }
        
        for i, method in enumerate(method_names):
            method_name = data.method_names[i]
            
            # Initialize results dictionaries for each level
            result_levels = {
                'associations': {
                    'precisions': [], 'recalls': [], 'method_overlaps': [], 'truth_overlaps': []
                },
                'phenotypes': {
                    'precisions': [], 'recalls': [], 'method_overlaps': [], 'truth_overlaps': []
                },
                'variants': {
                    'precisions': [], 'recalls': [], 'method_overlaps': [], 'truth_overlaps': []
                }
            }
            
            for threshold in thresholds:
                # Only vary the method threshold
                method_sig = datasets[method][datasets[method]['pval_nominal'] < threshold].copy()
                for level_name, level_results in result_levels.items():
                    # Get the fixed truth set for this level
                    truth_set = truth_sets[level_name]
                    
                    if len(truth_set) == 0 and len(method_sig) == 0:
                        level_results['precisions'].append(0.0)
                        level_results['recalls'].append(0.0)
                        level_results['method_overlaps'].append(0.0)
                        level_results['truth_overlaps'].append(0.0)
                        continue
                    elif len(truth_set) == 0:
                        level_results['precisions'].append(0.0)
                        level_results['recalls'].append(0.0)
                        level_results['method_overlaps'].append(0.0)
                        level_results['truth_overlaps'].append(0.0)
                        continue
                    elif len(method_sig) == 0:
                        level_results['precisions'].append(0.0)
                        level_results['recalls'].append(0.0)
                        level_results['method_overlaps'].append(0.0)
                        level_results['truth_overlaps'].append(0.0)
                        continue
                    
                    # Process method data based on level
                    if level_name == 'associations':
                        method_sig['assoc_id'] = method_sig['phenotype_id'] + ':' + method_sig['variant_id']
                        method_sig_clean = self._deduplicate_associations(method_sig, f'{method_name} at p<{threshold}')
                        method_set = set(method_sig_clean['assoc_id'])
                        
                    elif level_name == 'phenotypes':
                        method_set = set(method_sig['phenotype_id'].unique())
                        
                    elif level_name == 'variants':
                        method_set = set(method_sig['variant_id'].unique())
                    
                    intersection = truth_set & method_set
                    
                    if len(method_set) > 0:
                        precision = len(intersection) / len(method_set)
                        method_overlap = len(intersection) / len(method_set)
                    else:
                        precision = 0.0
                        method_overlap = 0.0
                    
                    if len(truth_set) > 0:
                        recall = len(intersection) / len(truth_set)
                        truth_overlap = len(intersection) / len(truth_set)
                    else:
                        recall = 0.0
                        truth_overlap = 0.0
                    
                    level_results['precisions'].append(precision)
                    level_results['recalls'].append(recall)
                    level_results['method_overlaps'].append(method_overlap)
                    level_results['truth_overlaps'].append(truth_overlap)
            
            pr_results[method] = {
                'method_name': method_name,
                'thresholds': thresholds,
                'truth_threshold': self.truth_threshold,
                **result_levels
            }
        
        pr_results['truth_threshold'] = self.truth_threshold
        return pr_results
    
    def generate_figure_0_precision_recall_analysis(self, data: pQTLData, pr_results: Dict[str, any]):
        """Generate comprehensive precision-recall analysis figure.
        
        Creates multi-panel figure showing PR curves, overlap percentages,
        and performance summaries across association, phenotype, and variant levels.
        
        Args:
            data: pQTLData object with analysis results
            pr_results: Precision-recall analysis results
            
        Returns:
            matplotlib Figure object or None if no data available
        """
        print("Generating Figure 0: Precision-Recall analysis...")
        
        if not pr_results:
            print("    No PR analysis available")
            return None
        
        fig = plt.figure(figsize=(12, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Get truth threshold from pr_results
        truth_threshold = pr_results.get('truth_threshold', self.truth_threshold)
        
        fig.suptitle(f'Precision-Recall Analysis (Truth at p < {truth_threshold:.0e})', 
                    fontsize=20, fontweight='bold')
        
        # Colors for methods
        method_colors = [NATURE_COLORS['secondary'], NATURE_COLORS['accent'], 
                        NATURE_COLORS['neutral'], NATURE_COLORS['highlight']]
        
        # Define analysis levels
        levels = ['associations', 'phenotypes', 'variants']
        level_titles = ['Association-level', 'Phenotype-level', 'Variant-level']
        markers = ['o', 's', '^']
        
        # Row 1: PR Curves for each level
        for level_idx, (level, level_title, marker) in enumerate(zip(levels, level_titles, markers)):
            ax = fig.add_subplot(gs[0, level_idx])
            
            method_idx = 0
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                if method_idx < len(method_colors) and level in results:
                    ax.plot(results[level]['recalls'], results[level]['precisions'], 
                           marker=marker, linewidth=2, markersize=4,
                           color=method_colors[method_idx], alpha=0.8,
                           label=results['method_name'])
                    method_idx += 1
            
            ax.set_xlabel(f'Recall (Fraction of Truth at p<{truth_threshold:.0e})')
            ax.set_ylabel(f'Precision (Fraction Correct)')
            ax.set_title(f'{level_title} PR Curves', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if level_idx == 0:
                ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Row 2: Method-side Overlap vs Threshold for each level
        for level_idx, (level, level_title, marker) in enumerate(zip(levels, level_titles, markers)):
            ax = fig.add_subplot(gs[1, level_idx])
            
            method_idx = 0
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                if method_idx < len(method_colors) and level in results:
                    # Sort log_thresholds to ensure x-axis goes from left to right
                    thresholds = results['thresholds']
                    log_thresholds = [-np.log10(t) for t in thresholds]
                    # Create sorted indices to ensure x-axis is properly ordered
                    sorted_indices = np.argsort(log_thresholds)
                    sorted_log_thresholds = [log_thresholds[j] for j in sorted_indices]
                    sorted_overlaps = [results[level]['method_overlaps'][j] for j in sorted_indices]
                    ax.plot(sorted_log_thresholds, sorted_overlaps, 
                           marker=marker, linewidth=2, markersize=4,
                           color=method_colors[method_idx], alpha=0.8,
                           label=results['method_name'])
                    method_idx += 1
            
            ax.set_xlabel('-log₁₀(Method p-value threshold)')
            ax.set_ylabel('Precision (%)')
            ax.set_title(f'{level_title} Precision vs Threshold', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if level_idx == 0:
                ax.legend()
            ax.set_ylim(0, 1)
        
        # Row 3: Truth-side Overlap vs Threshold for each level
        for level_idx, (level, level_title, marker) in enumerate(zip(levels, level_titles, markers)):
            ax = fig.add_subplot(gs[2, level_idx])
            
            method_idx = 0
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                if method_idx < len(method_colors) and level in results:
                    # Sort log_thresholds to ensure x-axis goes from left to right
                    thresholds = results['thresholds']
                    log_thresholds = [-np.log10(t) for t in thresholds]
                    # Create sorted indices to ensure x-axis is properly ordered
                    sorted_indices = np.argsort(log_thresholds)
                    sorted_log_thresholds = [log_thresholds[j] for j in sorted_indices]
                    sorted_overlaps = [results[level]['truth_overlaps'][j] for j in sorted_indices]
                    ax.plot(sorted_log_thresholds, sorted_overlaps, 
                           marker=marker, linewidth=2, markersize=4,
                           color=method_colors[method_idx], alpha=0.8,
                           label=results['method_name'])
                    method_idx += 1
            
            ax.set_xlabel('-log₁₀(Method p-value threshold)')
            ax.set_ylabel('Recall (%)')
            ax.set_title(f'{level_title} Recall vs Threshold', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if level_idx == 0:
                ax.legend()
            ax.set_ylim(0, 1)
        
        # Row 4: Summary analyses
        
        # Panel A: AUC Summary Table (all levels)
        ax4 = fig.add_subplot(gs[3, 0])
        auc_data = []
        for level, level_title in zip(levels, level_titles):
            auc_data.append([level_title, '', ''])  # Header row
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                if level in results:
                    # Sort by recall to ensure proper AUC calculation
                    recalls = np.array(results[level]['recalls'])
                    precisions = np.array(results[level]['precisions'])
                    idx = np.argsort(recalls)
                    auc = np.trapz(precisions[idx], recalls[idx])
                    auc_data.append(['', results['method_name'], f'{auc:.3f}'])
        
        # Create table
        table = ax4.table(cellText=auc_data,
                         colLabels=['Level', 'Method', 'PR-AUC'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        ax4.axis('off')
        ax4.set_title('PR-AUC Summary (All Levels)', fontweight='bold')
        
        # Panel B: F1 Score Heatmap (Association level)
        ax5 = fig.add_subplot(gs[3, 1])
        
        # Prepare data for heatmap (use association level)
        methods = [k for k in pr_results.keys() if k != 'truth_threshold' and isinstance(pr_results[k], dict)]
        method_names = [pr_results[m]['method_name'] for m in methods]
        thresholds = pr_results[methods[0]]['thresholds'] if methods else []
        
        # Create matrix of F1 scores for associations
        f1_matrix = []
        for method in methods:
            results = pr_results[method]
            if 'associations' in results:
                f1_scores = []
                for p, r in zip(results['associations']['precisions'], results['associations']['recalls']):
                    if p + r > 0:
                        f1 = 2 * (p * r) / (p + r)
                    else:
                        f1 = 0
                    f1_scores.append(f1)
                f1_matrix.append(f1_scores)
        
        if f1_matrix:
            f1_matrix = np.array(f1_matrix)
            
            # Create heatmap
            im = ax5.imshow(f1_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax5.set_xticks(np.arange(len(thresholds)))
            ax5.set_yticks(np.arange(len(methods)))
            ax5.set_xticklabels([f'{t:.0e}' for t in thresholds], rotation=45, ha='right')
            ax5.set_yticklabels(method_names)
            
            # Add text annotations
            for i in range(len(methods)):
                for j in range(len(thresholds)):
                    text = ax5.text(j, i, f'{f1_matrix[i, j]:.2f}',
                                  ha="center", va="center", 
                                  color="white" if f1_matrix[i, j] < 0.5 else "black",
                                  fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
            cbar.set_label('F1 Score', rotation=270, labelpad=15)
        
        ax5.set_xlabel('P-value Threshold')
        ax5.set_ylabel('Method')
        ax5.set_title('Association F1 Score Heatmap', fontweight='bold')
        
        # Panel C: Best F1 Scores Comparison
        ax6 = fig.add_subplot(gs[3, 2])
        
        # Calculate best F1 scores for each level and method
        best_f1_data = []
        for level, level_title in zip(levels, level_titles):
            level_f1s = []
            level_methods = []
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                if level in results:
                    f1_scores = []
                    for p, r in zip(results[level]['precisions'], results[level]['recalls']):
                        if p + r > 0:
                            f1 = 2 * (p * r) / (p + r)
                        else:
                            f1 = 0
                        f1_scores.append(f1)
                    level_f1s.append(max(f1_scores))
                    level_methods.append(results['method_name'])
            best_f1_data.append(level_f1s)
        
        if best_f1_data:
            x = np.arange(len(level_methods))
            width = 0.25
            
            for i, (level_f1s, level_title) in enumerate(zip(best_f1_data, level_titles)):
                ax6.bar(x + i*width, level_f1s, width, 
                       label=level_title, alpha=0.8,
                       color=[NATURE_COLORS['primary'], NATURE_COLORS['secondary'], NATURE_COLORS['accent']][i])
            
            ax6.set_xlabel('Method')
            ax6.set_ylabel('Best F1 Score')
            ax6.set_title('Best F1 Scores by Level', fontweight='bold')
            ax6.set_xticks(x + width)
            ax6.set_xticklabels(level_methods, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig

    def generate_figure_1_overlap_dashboard(self, data: pQTLData):
        """Generate comprehensive overlap analysis dashboard.
        
        Creates multi-panel dashboard showing overlap statistics from both
        method and truth perspectives, with performance heatmaps and rankings.
        
        Args:
            data: pQTLData object with overlap analysis results
            
        Returns:
            matplotlib Figure object
        """
        print("Generating Figure 1: Overlap analysis dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.25, wspace=0.25)
        
        fig.suptitle('cis-pQTL Analysis: Method vs Truth Overlap Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Colors for methods
        method_colors = [NATURE_COLORS['secondary'], NATURE_COLORS['accent'], 
                        NATURE_COLORS['neutral'], NATURE_COLORS['highlight']]
        
        overlap_stats = data.overlap_analysis['overlap_stats']
        method_names = [data.method_names[i] for i in range(4)]
        
        # Row 1: Traditional overlap bar plots
        # Panel A: Phenotype-level overlap
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_overlap_barplot(ax1, overlap_stats['phenotype'], method_names, method_colors,
                                   'Phenotype-level Overlap', 'Number of Proteins')
        
        # Panel B: Variant-level overlap  
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_overlap_barplot(ax2, overlap_stats['variant'], method_names, method_colors,
                                   'Variant-level Overlap', 'Number of SNPs')
        
        # Row 2: Bidirectional overlap percentages for Phenotypes
        # Panel C: Phenotype overlap from method perspective
        ax3 = fig.add_subplot(gs[1, 0])
        self._create_bidirectional_overlap_plot(ax3, overlap_stats['phenotype'], method_names, method_colors,
                                               'Phenotype Overlap: Method Perspective', 'method')
        
        # Panel D: Phenotype overlap from truth perspective
        ax4 = fig.add_subplot(gs[1, 1])
        self._create_bidirectional_overlap_plot(ax4, overlap_stats['phenotype'], method_names, method_colors,
                                               'Phenotype Overlap: Truth Perspective', 'truth')
        
        # Row 3: Bidirectional overlap percentages for Variants
        # Panel E: Variant overlap from method perspective
        ax5 = fig.add_subplot(gs[2, 0])
        self._create_bidirectional_overlap_plot(ax5, overlap_stats['variant'], method_names, method_colors,
                                               'Variant Overlap: Method Perspective', 'method')
        
        # Panel F: Variant overlap from truth perspective
        ax6 = fig.add_subplot(gs[2, 1])
        self._create_bidirectional_overlap_plot(ax6, overlap_stats['variant'], method_names, method_colors,
                                               'Variant Overlap: Truth Perspective', 'truth')
        
        # Row 4: Summary analyses
        # Panel G: Performance metrics heatmap
        ax7 = fig.add_subplot(gs[3, 0])
        self._create_performance_heatmap(ax7, overlap_stats, method_names)
        
        # Panel H: Method ranking
        ax8 = fig.add_subplot(gs[3, 1])
        if data.performance_metrics:
            self._create_method_ranking_plot(ax8, data.performance_metrics, method_colors)
        else:
            ax8.text(0.5, 0.5, 'Performance metrics\nnot calculated', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Method Performance Ranking')
        
        plt.tight_layout()
        return fig
    
    def _create_overlap_barplot(self, ax, overlap_data, method_names, colors, title, ylabel):
        """Create overlap bar plots showing method totals and overlaps with truth.
        
        Args:
            ax: Matplotlib axes object for plotting
            overlap_data: Dictionary containing overlap statistics for each method
            method_names: List of method display names
            colors: List of colors for each method
            title: Plot title
            ylabel: Y-axis label
        """
        methods = ['method1', 'method2', 'method3', 'method4']
        
        # Data for plotting
        overlap_counts = [overlap_data[m]['overlap_count'] for m in methods]
        truth_totals = [overlap_data[m]['truth_total'] for m in methods]
        method_totals = [overlap_data[m]['method_total'] for m in methods]
        
        x = np.arange(len(method_names))
        width = 0.25
        
        # Create grouped bars
        bars1 = ax.bar(x - width, overlap_counts, width, label='Overlap with Truth', 
                      color=colors, alpha=0.8)
        bars2 = ax.bar(x, method_totals, width, label='Method Total', 
                      color=colors, alpha=0.5)
        bars3 = ax.bar(x + width, truth_totals, width, label='Truth Total', 
                      color=NATURE_COLORS['primary'], alpha=0.7)
        
        # Add value labels on bars
        for i, (overlap, total) in enumerate(zip(overlap_counts, method_totals)):
            if total > 0:
                percentage = (overlap / total) * 100
                ax.text(i, overlap + max(overlap_counts) * 0.01, f'{percentage:.1f}%', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Method')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_bidirectional_overlap_plot(self, ax, overlap_data, method_names, colors, title, perspective):
        """Create bidirectional overlap percentage plots from method or truth perspective.
        
        Args:
            ax: Matplotlib axes object for plotting
            overlap_data: Dictionary containing overlap statistics for each method
            method_names: List of method display names
            colors: List of colors for each method
            title: Plot title
            perspective: Either 'method' or 'truth' for analysis perspective
        """
        methods = ['method1', 'method2', 'method3', 'method4']
        
        # Extract data based on perspective
        if perspective == 'method':
            # Method perspective: what percentage of method's findings overlap with truth
            overlap_percentages = []
            for method in methods:
                if overlap_data[method]['method_total'] > 0:
                    percentage = (overlap_data[method]['overlap_count'] / overlap_data[method]['method_total']) * 100
                else:
                    percentage = 0
                overlap_percentages.append(percentage)
            ylabel = 'Method Overlap %\n(% of method findings in truth)'
        else:  # truth perspective
            # Truth perspective: what percentage of truth's findings are captured by method
            overlap_percentages = []
            for method in methods:
                if overlap_data[method]['truth_total'] > 0:
                    percentage = (overlap_data[method]['overlap_count'] / overlap_data[method]['truth_total']) * 100
                else:
                    percentage = 0
                overlap_percentages.append(percentage)
            ylabel = 'Truth Overlap %\n(% of truth findings captured)'
        
        # Get method names for available methods
        available_methods = []
        available_percentages = []
        available_colors = []
        
        for i, method in enumerate(methods):
            if method in overlap_data and i < len(method_names):
                available_methods.append(method_names[i])
                available_percentages.append(overlap_percentages[i])
                available_colors.append(colors[i])
        
        if available_methods:
            x = np.arange(len(available_methods))
            bars = ax.bar(x, available_percentages, color=available_colors, alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
            
            # Add percentage labels on bars
            for i, (bar, percentage) in enumerate(zip(bars, available_percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{percentage:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # Add count information below bars
            for i, method in enumerate(methods[:len(available_methods)]):
                if method in overlap_data:
                    if perspective == 'method':
                        total_count = overlap_data[method]['method_total']
                        overlap_count = overlap_data[method]['overlap_count']
                    else:
                        total_count = overlap_data[method]['truth_total']
                        overlap_count = overlap_data[method]['overlap_count']
                    
                    ax.text(i, -5, f'{overlap_count}/{total_count}', 
                           ha='center', va='top', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax.set_xticks(x)
            ax.set_xticklabels(available_methods, rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(available_percentages) * 1.2 if available_percentages else 100)
            
            # Add reference lines
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80%')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
    
    def _create_performance_heatmap(self, ax, overlap_stats, method_names):
        """Create performance metrics heatmap showing sensitivity, precision, F1, and Jaccard.
        
        Args:
            ax: Matplotlib axes object for plotting
            overlap_stats: Dictionary containing overlap statistics across different levels
            method_names: List of method display names
        """
        metrics = ['sensitivity', 'precision', 'f1_score', 'jaccard']
        metric_labels = ['Sensitivity', 'Precision', 'F1 Score', 'Jaccard Index']
        
        # Create data matrix
        data_matrix = []
        for level in ['phenotype', 'variant', 'association']:
            level_data = []
            for method in ['method1', 'method2', 'method3', 'method4']:
                method_metrics = [overlap_stats[level][method][metric] for metric in metrics]
                level_data.extend(method_metrics)
            data_matrix.append(level_data)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics) * 4))
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(['Phenotype', 'Variant', 'Association'])
        
        # Create x-axis labels (method names repeated for each metric)
        x_labels = []
        for metric in metric_labels:
            x_labels.extend([f'{name}\n{metric}' for name in method_names])
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        
        # Add text annotations
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}', 
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Performance Metrics Heatmap', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Score', rotation=270, labelpad=15)
    
    def _create_venn_diagram(self, ax, assoc_sets, method_names):
        """Create Venn diagram showing association overlaps between truth and top methods.
        
        Args:
            ax: Matplotlib axes object for plotting
            assoc_sets: Dictionary containing association sets for each method
            method_names: List of method display names
        """
        truth_set = assoc_sets['truth']
        method_sets = [assoc_sets['method1'], assoc_sets['method2'], assoc_sets['method3']]
        
        if len(method_names) >= 3:
            # Create 3-way Venn diagram
            venn = venn3([truth_set, method_sets[0], method_sets[1]], 
                        set_labels=['Truth', method_names[0], method_names[1]], ax=ax)
            
            if venn:
                # Color the Venn diagram
                if venn.get_patch_by_id('100'):
                    venn.get_patch_by_id('100').set_facecolor(NATURE_COLORS['primary'])
                    venn.get_patch_by_id('100').set_alpha(0.7)
                if venn.get_patch_by_id('010'):
                    venn.get_patch_by_id('010').set_facecolor(NATURE_COLORS['secondary'])
                    venn.get_patch_by_id('010').set_alpha(0.7)
                if venn.get_patch_by_id('001'):
                    venn.get_patch_by_id('001').set_facecolor(NATURE_COLORS['accent'])
                    venn.get_patch_by_id('001').set_alpha(0.7)
        
        ax.set_title('Association Overlap\n(Truth vs Top 2 Methods)', fontweight='bold')
    
    def _create_method_ranking_plot(self, ax, performance_metrics, colors):
        """Create horizontal bar plot showing method performance rankings.
        
        Args:
            ax: Matplotlib axes object for plotting
            performance_metrics: Dictionary containing performance scores for each method
            colors: List of colors for each method
        """
        methods = ['method1', 'method2', 'method3', 'method4']
        
        # Sort methods by composite score
        sorted_methods = sorted(methods, key=lambda x: performance_metrics[x]['composite_score'], reverse=True)
        sorted_names = [performance_metrics[m]['method_name'] for m in sorted_methods]
        sorted_scores = [performance_metrics[m]['composite_score'] for m in sorted_methods]
        sorted_colors = [colors[methods.index(m)] for m in sorted_methods]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_names))
        bars = ax.barh(y_pos, sorted_scores, color=sorted_colors, alpha=0.8, edgecolor='black')
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Composite Performance Score')
        ax.set_title('Method Performance Ranking', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        # Add ranking numbers
        for i, name in enumerate(sorted_names):
            ax.text(-0.05, i, f'#{i+1}', ha='right', va='center', 
                   fontweight='bold', fontsize=12, color='red')
    
    def generate_figure_2_effect_size_concordance(self, data: pQTLData):
        """Generate effect size concordance analysis figure.
        
        Creates scatter plots showing correlation between method and truth
        effect sizes, with regression lines and performance statistics.
        
        Args:
            data: pQTLData object with effect size analysis results
            
        Returns:
            matplotlib Figure object or None if no data available
        """
        print("Generating Figure 2: Effect size concordance analysis...")
        
        if not data.effect_size_analysis:
            print("    No effect size analysis available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 6))
        fig.suptitle('Effect Size Concordance: Method vs Truth Comparison', 
                    fontsize=20, fontweight='bold')
        
        method_colors = [NATURE_COLORS['secondary'], NATURE_COLORS['accent'], 
                        NATURE_COLORS['neutral'], NATURE_COLORS['highlight']]
        
        methods = ['method1', 'method2', 'method3', 'method4']
        available_methods = [m for m in methods if m in data.effect_size_analysis]
        
        for i, method in enumerate(available_methods[:4]):
            row, col = divmod(i, 2)
            ax = axes[row, col]
            
            effect_data = data.effect_size_analysis[method]
            truth_slopes = effect_data['truth_slopes']
            method_slopes = effect_data['method_slopes']
            method_name = effect_data['method_name']
            
            if len(truth_slopes) != len(method_slopes):
                print(f"    WARNING: Skipping plot for {method_name} due to array length mismatch")
                ax.text(0.5, 0.5, f'Data unavailable\nfor {method_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            if len(truth_slopes) == 0:
                print(f"    WARNING: No data available for {method_name}")
                ax.text(0.5, 0.5, f'No data available\nfor {method_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Main scatter plot
            ax.scatter(truth_slopes, method_slopes, alpha=0.6, s=20, 
                      color=method_colors[i], edgecolors='white', linewidth=0.5)
            
            # Add 1:1 line
            min_slope = min(np.min(truth_slopes), np.min(method_slopes))
            max_slope = max(np.max(truth_slopes), np.max(method_slopes))
            ax.plot([min_slope, max_slope], [min_slope, max_slope], 
                   'k--', alpha=0.7, linewidth=2, label='Perfect concordance')
            
            # Add regression line
            z = np.polyfit(truth_slopes, method_slopes, 1)
            p = np.poly1d(z)
            ax.plot(truth_slopes, p(truth_slopes), color='red', linewidth=2, 
                   alpha=0.8, label=f'Regression line')
            
            # Add statistics
            pearson_r = effect_data['pearson_r']
            mae = effect_data['mae']
            concordance = effect_data['concordance']
            n_pairs = effect_data['n_clean']
            
            stats_text = (f'r = {pearson_r:.3f}\n'
                         f'MAE = {mae:.3f}\n'
                         f'Concordance = {concordance:.3f}\n'
                         f'n = {n_pairs}')
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Truth Effect Size (slope)')
            ax.set_ylabel(f'{method_name} Effect Size (slope)')
            ax.set_title(f'{method_name} vs Truth', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
        
        # Hide unused subplots
        for i in range(len(available_methods), 4):
            row, col = divmod(i, 2)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    
    def _extract_variant_position(self, variant_id: str) -> Tuple[str, int]:
        """Extract chromosome and position from variant ID.
        
        Args:
            variant_id: Variant ID in format "chr:position" (e.g., "1:1555871")
            
        Returns:
            Tuple of (chromosome, position) where chromosome is in UCSC format (e.g., "chr1")
        """
        parts = variant_id.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid variant_id format: {variant_id}")
        
        chrom = parts[0]
        # Convert to UCSC format if needed
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'
        
        try:
            position = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid position in variant_id: {variant_id}")
        
        return chrom, position
    
    def _format_browser_track_entry(self, chrom: str, start: int, end: int, 
                                  name: str, score: float) -> str:
        """Format a single BED track entry for UCSC Genome Browser.
        
        Args:
            chrom: Chromosome (e.g., "chr1")
            start: Start position (0-based)
            end: End position (1-based, typically start+1 for SNPs)
            name: Feature name (e.g., "GENE:rsID")
            score: Score value (0-1000, typically based on -log10(p-value))
            
        Returns:
            Formatted BED line string
        """
        # Ensure score is within valid range (0-1000)
        score = min(1000, max(0, int(score)))
        
        return f"{chrom}\t{start}\t{end}\t{name}\t{score}\n"
    
    def _calculate_browser_score(self, pval: float, max_score: int = 1000) -> int:
        """Calculate UCSC browser score from p-value.
        
        Score is based on -log10(p-value), scaled to 0-1000 range.
        More significant p-values get higher scores.
        
        Args:
            pval: P-value
            max_score: Maximum score value (default 1000)
            
        Returns:
            Integer score between 0 and max_score
        """
        if pval <= 0 or np.isnan(pval):
            return 0
        
        # Calculate -log10(p-value)
        neg_log_p = -np.log10(pval)
        
        # Scale to 0-1000 range
        # P-value of 1e-10 maps to score 1000
        # P-value of 0.05 maps to score ~130
        score = int(neg_log_p * 100)
        
        return min(max_score, max(0, score))
    
    def plot_pval_slope_comparison(self, data: pQTLData) -> plt.Figure:
        """Create combined p-value and slope comparison plots using density visualization.
        
        Creates a two-row figure with p-value comparisons (top) and slope comparisons (bottom)
        for each imputation method vs truth. Uses hexbin density plots to better visualize
        overlapping points. P-value plots include all MAF-filtered hits, while slope plots 
        include only hits passing both p-value and MAF cutoffs.
        
        The red dashed lines in p-value plots indicate the significance threshold (p=0.05),
        dividing the plot into quadrants for true/false positives and negatives.
        
        Args:
            data: pQTLData object containing analysis results
            
        Returns:
            matplotlib Figure object with comparison plots
        """
        print("Creating p-value and slope comparison plots...")
        
        # Check if we have the necessary data
        if not data.maf_filtered:
            print("  WARNING: No MAF-filtered data available, skipping p-value/slope comparison plots")
            return None
            
        if not data.effect_size_analysis:
            print("  WARNING: No effect size analysis available, skipping slope comparison plots")
            # Still create the figure but only with p-value plots
            
        # Prepare method data
        methods = ['method1', 'method2', 'method3', 'method4']
        method_names = data.method_names
        method_colors = [NATURE_COLORS['secondary'], NATURE_COLORS['accent'], 
                        NATURE_COLORS['neutral'], NATURE_COLORS['highlight']]
        
        # Create figure with 2 rows x 4 columns
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Prepare truth data with assoc_id (once, outside the loop)
        truth_maf = data.maf_filtered['truth'].copy()
        truth_maf = truth_maf[truth_maf['pval_nominal'] < 1]  # Remove invalid p-values
        truth_maf['assoc_id'] = truth_maf['phenotype_id'] + ':' + truth_maf['variant_id']
        
        for i, (method, method_name) in enumerate(zip(methods, method_names)):
            print(f"  Processing {method_name}...")
            
            if method not in data.maf_filtered:
                print(f"    Skipping {method_name}: no MAF-filtered data")
                continue
                
            # Get MAF-filtered data for this method
            method_maf = data.maf_filtered[method].copy()
            method_maf = method_maf[method_maf['pval_nominal'] < 1]  # Remove invalid p-values
            method_maf['assoc_id'] = method_maf['phenotype_id'] + ':' + method_maf['variant_id']
            
            # Panel 1: P-value comparison (all MAF-filtered hits)
            ax1 = fig.add_subplot(gs[0, i])
            
            # Efficient matching using merge
            merged_pval = pd.merge(
                truth_maf[['assoc_id', 'pval_nominal']],
                method_maf[['assoc_id', 'pval_nominal']],
                on='assoc_id',
                suffixes=('_truth', '_method')
            )
            
            if len(merged_pval) == 0:
                print(f"    No common associations for {method_name}")
                ax1.text(0.5, 0.5, f'No common associations\nfor {method_name}', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2 = fig.add_subplot(gs[1, i])
                ax2.text(0.5, 0.5, f'No data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                continue
            
            truth_pvals = merged_pval['pval_nominal_truth'].values
            method_pvals = merged_pval['pval_nominal_method'].values
            
            if len(truth_pvals) > 0:
                # Convert to numpy arrays and add small epsilon to avoid log(0)
                truth_pvals = np.array(truth_pvals) + 1e-300
                method_pvals = np.array(method_pvals) + 1e-300
                
                # Calculate correlation
                pearson_r, _ = pearsonr(np.log10(truth_pvals), np.log10(method_pvals))
                
                # Create hexbin density plot on log scale
                # Use hexbin for better visualization of overlapping points
                # Use log scale for bins to better show density range
                hb1 = ax1.hexbin(truth_pvals, method_pvals, 
                                 xscale='log', yscale='log',
                                 gridsize=50, cmap='Blues', mincnt=1,
                                 norm=LogNorm(), edgecolors='none', linewidths=0)
                
                # Add colorbar for density (log scale)
                cb1 = plt.colorbar(hb1, ax=ax1)
                cb1.set_label('Count (log scale)', fontsize=8)
                
                # Add significance threshold lines
                ax1.axhline(y=self.significance_threshold, color='red', linestyle='--', 
                          alpha=0.7, linewidth=1.5, label=f'p={self.significance_threshold}')
                ax1.axvline(x=self.significance_threshold, color='red', linestyle='--', 
                          alpha=0.7, linewidth=1.5)
                
                # Note: log scale already set by hexbin
                
                # Labels and title
                ax1.set_xlabel('Truth p-value')
                ax1.set_ylabel(f'{method_name} p-value')
                ax1.set_title(f'{method_name}\nP-value Comparison')
                
                # Add statistics text
                ax1.text(0.05, 0.95, f'n = {len(truth_pvals)}\nr = {pearson_r:.3f}',
                        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax1.grid(True, alpha=0.3)
            
            # Panel 2: Slope comparison (hits passing both p-value and MAF cutoffs)
            ax2 = fig.add_subplot(gs[1, i])
            
            # Use pre-computed effect size analysis if available
            if data.effect_size_analysis and method in data.effect_size_analysis:
                effect_data = data.effect_size_analysis[method]
                truth_slopes = effect_data['truth_slopes']
                method_slopes = effect_data['method_slopes']
                
                
                if len(truth_slopes) > 0 and len(method_slopes) > 0:
                    # Use the pre-computed statistics from effect_data
                    pearson_r = effect_data.get('pearson_r', 0)
                    spearman_r = effect_data.get('spearman_r', 0)
                    mae = effect_data.get('mae', 0)
                    n_clean = effect_data.get('n_clean', len(truth_slopes))
                    
                    # Create hexbin density plot for slopes with log-scale density
                    # Check if we need to handle negative values for slopes
                    min_val = min(np.min(truth_slopes), np.min(method_slopes))
                    max_val = max(np.max(truth_slopes), np.max(method_slopes))
                    
                    hb2 = ax2.hexbin(truth_slopes, method_slopes, 
                                    gridsize=40, cmap='Oranges', mincnt=1,
                                    norm=LogNorm(), edgecolors='none', linewidths=0,
                                    extent=[min_val, max_val, min_val, max_val])
                    
                    # Add colorbar for density (log scale)
                    cb2 = plt.colorbar(hb2, ax=ax2)
                    cb2.set_label('Count (log scale)', fontsize=8)
                    
                    # Add regression line
                    
                    # Add regression line
                    if len(truth_slopes) > 1:
                        z = np.polyfit(truth_slopes, method_slopes, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(np.min(truth_slopes), np.max(truth_slopes), 100)
                        ax2.plot(x_line, p(x_line), color='red', linewidth=1.5, 
                               alpha=0.8, label='Regression line')
                    
                    # Labels and title
                    ax2.set_xlabel('Truth Effect Size (slope)')
                    ax2.set_ylabel(f'{method_name} Effect Size (slope)')
                    ax2.set_title(f'{method_name}\nSlope Comparison (p<{self.significance_threshold})')
                    
                    # Add statistics text
                    stats_text = f'n = {n_clean}\n'
                    stats_text += f'Pearson r = {pearson_r:.3f}\n'
                    stats_text += f'Spearman ρ = {spearman_r:.3f}\n'
                    stats_text += f'MAE = {mae:.3f}'
                    
                    ax2.text(0.05, 0.95, stats_text,
                            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    ax2.legend(loc='lower right', fontsize=9)
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, f'No significant associations\nwith valid slopes', 
                           ha='center', va='center', transform=ax2.transAxes)
            else:
                # No effect size analysis available for this method
                ax2.text(0.5, 0.5, f'No effect size data available\nfor {method_name}', 
                       ha='center', va='center', transform=ax2.transAxes)
        
        # Add main title
        fig.suptitle('P-value and Effect Size Comparison: Imputed vs Truth', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def generate_ucsc_browser_tracks(self, data: pQTLData):
        """Generate UCSC Genome Browser custom track files for significant cis-pQTL hits.
        
        Creates BED format track files for visualization in the UCSC Genome Browser.
        Generates individual files for truth and each imputation method, plus a combined
        multi-track file. Only includes associations passing both p-value and MAF cutoffs.
        
        Args:
            data: pQTLData object with significant hits
        """
        print("Generating UCSC Genome Browser tracks...")
        
        # Create tracks directory
        tracks_dir = self.output_dir / "tracks"
        tracks_dir.mkdir(exist_ok=True)
        
        # Define colors for each dataset (RGB format)
        track_colors = {
            'truth': '255,0,0',        # Red
            'method1': '77,187,213',    # Light Blue
            'method2': '0,160,135',     # Teal
            'method3': '60,84,136',     # Dark Blue
            'method4': '243,155,127'    # Light Red
        }
        
        # Track names and descriptions
        track_info = {
            'truth': ('Truth_cis_pQTL', 'Ground truth cis-pQTL hits'),
            'method1': (f'{data.method_names[0]}_cis_pQTL', f'{data.method_names[0]} imputed cis-pQTL hits'),
            'method2': (f'{data.method_names[1]}_cis_pQTL', f'{data.method_names[1]} imputed cis-pQTL hits'),
            'method3': (f'{data.method_names[2]}_cis_pQTL', f'{data.method_names[2]} imputed cis-pQTL hits'),
            'method4': (f'{data.method_names[3]}_cis_pQTL', f'{data.method_names[3]} imputed cis-pQTL hits')
        }
        
        # Process each dataset
        all_tracks_content = []
        track_files_created = []
        
        for dataset_key in ['truth', 'method1', 'method2', 'method3', 'method4']:
            if not hasattr(data, dataset_key):
                continue
            
            df = getattr(data, dataset_key)
            
            # Filter for significant hits (p-value and MAF cutoffs already applied)
            sig_df = df[df['pval_nominal'] < self.significance_threshold].copy()
            
            if sig_df.empty:
                print(f"  No significant hits for {dataset_key}, skipping track generation")
                continue
            
            # Sort by p-value
            sig_df = sig_df.sort_values('pval_nominal')
            
            # Prepare track content
            track_name, track_desc = track_info[dataset_key]
            track_color = track_colors[dataset_key]
            
            # Individual track file
            track_file = tracks_dir / f"{dataset_key}_cis_pqtl.bed"
            track_entries = []
            
            # Add track header
            header_lines = []
            
            # Determine browser position from first few hits
            if len(sig_df) > 0:
                first_variant = sig_df.iloc[0]['variant_id']
                try:
                    first_chrom, first_pos = self._extract_variant_position(first_variant)
                    # Set browser window around first hit
                    browser_start = max(0, first_pos - 500000)
                    browser_end = first_pos + 500000
                    header_lines.append(f"browser position {first_chrom}:{browser_start}-{browser_end}\n")
                except:
                    pass
            
            header_lines.append(f'track name="{track_name}" description="{track_desc}" '
                              f'visibility=2 color={track_color} useScore=1\n')
            header_lines.append("#chrom\tchromStart\tchromEnd\tname\tscore\n")
            
            # Process each significant association
            unique_positions = set()  # To avoid duplicate entries
            
            for _, row in sig_df.iterrows():
                try:
                    # Extract chromosome and position
                    chrom, position = self._extract_variant_position(row['variant_id'])
                    
                    # Create unique identifier
                    pos_key = f"{chrom}:{position}"
                    if pos_key in unique_positions:
                        continue
                    unique_positions.add(pos_key)
                    
                    # Calculate browser score from p-value
                    score = self._calculate_browser_score(row['pval_nominal'])
                    
                    # Create feature name
                    feature_name = f"{row['phenotype_id']}:{row['variant_id']}"
                    
                    # Format BED entry (0-based start, 1-based end)
                    entry = self._format_browser_track_entry(
                        chrom, position-1, position, feature_name, score
                    )
                    track_entries.append(entry)
                    
                except Exception as e:
                    print(f"    WARNING: Could not process variant {row['variant_id']}: {e}")
                    continue
            
            # Write individual track file
            if track_entries:
                with open(track_file, 'w') as f:
                    f.writelines(header_lines)
                    f.writelines(track_entries)
                
                print(f"  Created track: {track_file} ({len(track_entries)} features)")
                track_files_created.append(str(track_file))
                
                # Add to combined tracks
                all_tracks_content.extend(header_lines[1:])  # Skip browser position for multi-track
                all_tracks_content.extend(track_entries)
                all_tracks_content.append("\n")  # Add spacing between tracks
        
        # Create combined multi-track file
        if all_tracks_content:
            combined_file = tracks_dir / "all_methods_cis_pqtl.bed"
            
            # Determine overall browser position
            all_chroms = []
            all_positions = []
            
            for dataset_key in ['truth', 'method1', 'method2', 'method3', 'method4']:
                if not hasattr(data, dataset_key):
                    continue
                df = getattr(data, dataset_key)
                sig_df = df[df['pval_nominal'] < self.significance_threshold]
                
                for variant_id in sig_df['variant_id'].head(10):  # Sample first 10
                    try:
                        chrom, pos = self._extract_variant_position(variant_id)
                        all_chroms.append(chrom)
                        all_positions.append(pos)
                    except:
                        continue
            
            # Write combined file
            with open(combined_file, 'w') as f:
                # Add browser position if we have data
                if all_positions:
                    median_pos = int(np.median(all_positions))
                    most_common_chrom = max(set(all_chroms), key=all_chroms.count)
                    f.write(f"browser position {most_common_chrom}:{median_pos-1000000}-{median_pos+1000000}\n")
                
                f.writelines(all_tracks_content)
            
            print(f"  Created combined track: {combined_file}")
            track_files_created.append(str(combined_file))
        
        # Create README with usage instructions
        readme_file = tracks_dir / "README_UCSC_tracks.txt"
        with open(readme_file, 'w') as f:
            f.write("UCSC Genome Browser Custom Track Files\n")
            f.write("=" * 50 + "\n\n")
            f.write("Generated: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write(f"Significance threshold: p < {self.significance_threshold}\n")
            f.write(f"MAF threshold: {self.maf_threshold}\n\n")
            
            f.write("Files generated:\n")
            for track_file in track_files_created:
                f.write(f"  - {Path(track_file).name}\n")
            
            f.write("\n" + "How to use these tracks:" + "\n")
            f.write("-" * 30 + "\n\n")
            
            f.write("Option 1: Upload to UCSC Genome Browser\n")
            f.write("1. Go to: https://genome.ucsc.edu/cgi-bin/hgCustom\n")
            f.write("2. Click 'Choose File' and select a .bed file\n")
            f.write("3. Click 'Submit'\n")
            f.write("4. The browser will display your custom track\n\n")
            
            f.write("Option 2: Paste track contents\n")
            f.write("1. Go to: https://genome.ucsc.edu/cgi-bin/hgCustom\n")
            f.write("2. Open a .bed file in a text editor\n")
            f.write("3. Copy all contents\n")
            f.write("4. Paste into the 'Paste URLs or data' box\n")
            f.write("5. Click 'Submit'\n\n")
            
            f.write("Track color coding:\n")
            f.write("  - Truth: Red\n")
            f.write(f"  - {data.method_names[0]}: Light Blue\n")
            f.write(f"  - {data.method_names[1]}: Teal\n")
            f.write(f"  - {data.method_names[2]}: Dark Blue\n")
            f.write(f"  - {data.method_names[3]}: Light Red\n\n")
            
            f.write("Score interpretation:\n")
            f.write("  - Scores range from 0-1000\n")
            f.write("  - Higher scores = more significant p-values\n")
            f.write("  - Score = -log10(p-value) * 100 (capped at 1000)\n")
            
        print(f"  Created README: {readme_file}")
        print(f"  Total track files created: {len(track_files_created)}")
        
        return track_files_created
    
    def save_figure(self, fig, name: str, **kwargs):
        """Save figure in both PDF and PNG formats.
        
        Args:
            fig: matplotlib Figure object
            name: Base filename (without extension)
            **kwargs: Additional arguments passed to savefig
        """
        if fig is None:
            return
            
        fig_path = self.output_dir / "figures" / f"{name}.pdf"
        png_path = self.output_dir / "figures" / f"{name}.png"
        
        # Save PDF (vector format for publication)
        fig.savefig(fig_path, format='pdf', **kwargs)
        
        # Save PNG (for quick viewing)
        fig.savefig(png_path, format='png', dpi=300, **kwargs)
        
        print(f"  Figure saved: {fig_path}")
    
    def save_analysis_results(self, data: pQTLData, pr_results: Dict[str, any] = None):
        """Save comprehensive analysis results to CSV files.
        
        Exports overlap statistics, effect size analysis, performance metrics,
        significant hits summary, and precision-recall results to CSV format.
        
        Args:
            data: pQTLData object with analysis results
            pr_results: Optional precision-recall analysis results
        """
        print("Saving analysis results...")
        
        data_dir = self.output_dir / "data"
        
        # Save overlap statistics
        if data.overlap_analysis:
            overlap_results = []
            for level in ['phenotype', 'variant', 'association']:
                for method in ['method1', 'method2', 'method3', 'method4']:
                    if method in data.overlap_analysis['overlap_stats'][level]:
                        stats = data.overlap_analysis['overlap_stats'][level][method]
                        overlap_results.append({
                            'level': level,
                            'method': method,
                            'method_name': stats['method_name'],
                            **stats
                        })
            
            overlap_df = pd.DataFrame(overlap_results)
            overlap_df.to_csv(data_dir / "overlap_analysis.csv", index=False)
            print(f"  Overlap analysis saved to {data_dir / 'overlap_analysis.csv'}")
        
        # Save effect size analysis
        if data.effect_size_analysis:
            effect_results = []
            for method, stats in data.effect_size_analysis.items():
                effect_results.append({
                    'method': method,
                    'method_name': stats['method_name'],
                    'n_common': stats['n_common'],
                    'n_clean': stats['n_clean'],
                    'pearson_r': stats['pearson_r'],
                    'pearson_p': stats['pearson_p'],
                    'spearman_r': stats['spearman_r'],
                    'spearman_p': stats['spearman_p'],
                    'mae': stats['mae'],
                    'rmse': stats['rmse'],
                    'bias': stats['bias'],
                    'concordance': stats['concordance']
                })
            
            effect_df = pd.DataFrame(effect_results)
            effect_df.to_csv(data_dir / "effect_size_analysis.csv", index=False)
            print(f"  Effect size analysis saved to {data_dir / 'effect_size_analysis.csv'}")
        
        # Save performance metrics
        if data.performance_metrics:
            perf_results = []
            for method, metrics in data.performance_metrics.items():
                perf_results.append({
                    'method': method,
                    **metrics
                })
            
            perf_df = pd.DataFrame(perf_results)
            perf_df = perf_df.sort_values('composite_score', ascending=False)
            perf_df['rank'] = range(1, len(perf_df) + 1)
            perf_df.to_csv(data_dir / "performance_metrics.csv", index=False)
            print(f"  Performance metrics saved to {data_dir / 'performance_metrics.csv'}")
        
        # Save significant hits summary
        if data.significant_hits:
            sig_summary = []
            for name, df in data.significant_hits.items():
                sig_summary.append({
                    'dataset': name,
                    'n_significant': len(df),
                    'n_unique_phenotypes': df['phenotype_id'].nunique(),
                    'n_unique_variants': df['variant_id'].nunique(),
                    'mean_pvalue': df['pval_nominal'].mean(),
                    'median_pvalue': df['pval_nominal'].median(),
                    'mean_effect_size': df['slope'].mean(),
                    'median_effect_size': df['slope'].median()
                })
            
            sig_df = pd.DataFrame(sig_summary)
            sig_df.to_csv(data_dir / "significant_hits_summary.csv", index=False)
            print(f"  Significant hits summary saved to {data_dir / 'significant_hits_summary.csv'}")
        
        # Save precision-recall results
        if pr_results:
            pr_summary = []
            levels = ['associations', 'phenotypes', 'variants']
            
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                for level in levels:
                    if level in results:
                        for i, threshold in enumerate(results['thresholds']):
                            pr_summary.append({
                                'method': method,
                                'method_name': results['method_name'],
                                'level': level,
                                'threshold': threshold,
                                'precision': results[level]['precisions'][i],
                                'recall': results[level]['recalls'][i],
                                'method_overlap': results[level]['method_overlaps'][i],
                                'truth_overlap': results[level]['truth_overlaps'][i],
                                'f1_score': 2 * (results[level]['precisions'][i] * results[level]['recalls'][i]) / 
                                           (results[level]['precisions'][i] + results[level]['recalls'][i]) 
                                           if (results[level]['precisions'][i] + results[level]['recalls'][i]) > 0 else 0
                            })
            
            pr_df = pd.DataFrame(pr_summary)
            pr_df.to_csv(data_dir / "precision_recall_analysis.csv", index=False)
            print(f"  Precision-recall analysis saved to {data_dir / 'precision_recall_analysis.csv'}")
            
            # Save AUC summary for all levels
            auc_summary = []
            for method, results in pr_results.items():
                # Skip non-method entries like 'truth_threshold'
                if method == 'truth_threshold' or not isinstance(results, dict):
                    continue
                for level in levels:
                    if level in results:
                        # Sort by recall to ensure proper AUC calculation
                        recalls = np.array(results[level]['recalls'])
                        precisions = np.array(results[level]['precisions'])
                        idx = np.argsort(recalls)
                        auc = np.trapz(precisions[idx], recalls[idx])
                        auc_summary.append({
                            'method': method,
                            'method_name': results['method_name'],
                            'level': level,
                            'pr_auc': auc
                        })
            
            auc_df = pd.DataFrame(auc_summary)
            auc_df.to_csv(data_dir / "pr_auc_summary.csv", index=False)
            print(f"  PR-AUC summary saved to {data_dir / 'pr_auc_summary.csv'}")
    
    def generate_summary_report(self, data: pQTLData, pr_results: Dict[str, any] = None):
        """Generate comprehensive summary report in text format.
        
        Creates detailed text report with significant hits summary, overlap analysis,
        effect size analysis, performance rankings, and precision-recall results.
        
        Args:
            data: pQTLData object with analysis results
            pr_results: Optional precision-recall analysis results
        """
        print("Generating summary report...")
        
        report_path = self.output_dir / "reports" / f"cis_pqtl_analysis_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CIS-PQTL ANALYSIS COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Significance threshold: p < {data.significance_threshold}\n")
            f.write(f"Truth threshold for PR analysis: p < {self.truth_threshold}\n")
            f.write(f"Method names: {', '.join(data.method_names)}\n\n")
            
            # Significant hits summary
            f.write("SIGNIFICANT HITS SUMMARY\n")
            f.write("-" * 40 + "\n")
            if data.significant_hits:
                for name, df in data.significant_hits.items():
                    dataset_name = "Truth" if name == "truth" else data.method_names[int(name[-1])-1]
                    f.write(f"{dataset_name:>12}: {len(df):>6} significant associations\n")
                    f.write(f"             {df['phenotype_id'].nunique():>6} unique proteins\n")
                    f.write(f"             {df['variant_id'].nunique():>6} unique variants\n\n")
            
            # Overlap analysis summary
            if data.overlap_analysis:
                f.write("OVERLAP ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                overlap_stats = data.overlap_analysis['overlap_stats']
                
                for level in ['phenotype', 'variant', 'association']:
                    f.write(f"\n{level.upper()} LEVEL:\n")
                    truth_total = overlap_stats[level]['method1']['truth_total']
                    f.write(f"Truth total: {truth_total}\n")
                    
                    for i, method in enumerate(['method1', 'method2', 'method3', 'method4']):
                        if method in overlap_stats[level]:
                            stats = overlap_stats[level][method]
                            f.write(f"{data.method_names[i]:>12}: "
                                   f"{stats['overlap_count']:>4}/{stats['method_total']:>4} "
                                   f"(Sens: {stats['sensitivity']:.3f}, "
                                   f"Prec: {stats['precision']:.3f}, "
                                   f"F1: {stats['f1_score']:.3f})\n")
            
            # Effect size analysis summary
            if data.effect_size_analysis:
                f.write("\nEFFECT SIZE ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                for i, (method, stats) in enumerate(data.effect_size_analysis.items()):
                    f.write(f"{data.method_names[i]:>12}: "
                           f"r={stats['pearson_r']:>6.3f}, "
                           f"MAE={stats['mae']:>6.3f}, "
                           f"Concordance={stats['concordance']:>6.3f} "
                           f"(n={stats['n_clean']})\n")
            
            # Performance ranking
            if data.performance_metrics:
                f.write("\nPERFORMANCE RANKING\n")
                f.write("-" * 40 + "\n")
                methods = ['method1', 'method2', 'method3', 'method4']
                sorted_methods = sorted([m for m in methods if m in data.performance_metrics], 
                                       key=lambda x: data.performance_metrics[x]['composite_score'], 
                                       reverse=True)
                
                for rank, method in enumerate(sorted_methods, 1):
                    metrics = data.performance_metrics[method]
                    method_idx = int(method[-1]) - 1
                    f.write(f"#{rank}. {data.method_names[method_idx]:>12}: "
                           f"Composite Score = {metrics['composite_score']:.3f}\n")
                    f.write(f"     Overlap: {metrics['overlap_score']:.3f}, "
                           f"Effect Size: {metrics['effect_size_score']:.3f}, "
                           f"Statistical: {metrics['statistical_score']:.3f}, "
                           f"Discovery: {metrics['discovery_score']:.3f}\n\n")
            
            # Precision-Recall Analysis Summary
            if pr_results:
                f.write("\nPRECISION-RECALL ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                
                levels = ['associations', 'phenotypes', 'variants']
                level_titles = ['Association-level', 'Phenotype-level', 'Variant-level']
                
                for level, level_title in zip(levels, level_titles):
                    f.write(f"\n{level_title.upper()} ANALYSIS:\n")
                    f.write("Area Under PR Curve (PR-AUC):\n")
                    
                    # Calculate and sort by AUC for this level
                    auc_results = []
                    for method, results in pr_results.items():
                        # Skip non-method entries like 'truth_threshold'
                        if method == 'truth_threshold' or not isinstance(results, dict):
                            continue
                        if level in results:
                            # Sort by recall to ensure proper AUC calculation
                            recalls = np.array(results[level]['recalls'])
                            precisions = np.array(results[level]['precisions'])
                            idx = np.argsort(recalls)
                            auc = np.trapz(precisions[idx], recalls[idx])
                            method_idx = int(method[-1]) - 1
                            auc_results.append((auc, data.method_names[method_idx]))
                    
                    auc_results.sort(reverse=True)
                    
                    for rank, (auc, method_name) in enumerate(auc_results, 1):
                        f.write(f"#{rank}. {method_name:>12}: PR-AUC = {auc:.3f}\n")
                    
                    f.write(f"\nBest F1 Performance (threshold range 5e-8 to 0.5):\n")
                    f.write(f"{'Method':>12} {'Best F1':>8} {'@Threshold':>12} {'Precision':>10} {'Recall':>8}\n")
                    f.write("-" * 55 + "\n")
                    
                    for method, results in pr_results.items():
                        # Skip non-method entries like 'truth_threshold'
                        if method == 'truth_threshold' or not isinstance(results, dict):
                            continue
                        if level in results:
                            method_idx = int(method[-1]) - 1
                            method_name = data.method_names[method_idx]
                            
                            # Find best F1 score and corresponding threshold for this level
                            f1_scores = []
                            for p, r in zip(results[level]['precisions'], results[level]['recalls']):
                                if p + r > 0:
                                    f1 = 2 * (p * r) / (p + r)
                                else:
                                    f1 = 0
                                f1_scores.append(f1)
                            
                            best_f1_idx = np.argmax(f1_scores)
                            best_f1 = f1_scores[best_f1_idx]
                            best_threshold = results['thresholds'][best_f1_idx]
                            best_precision = results[level]['precisions'][best_f1_idx]
                            best_recall = results[level]['recalls'][best_f1_idx]
                            
                            f.write(f"{method_name:>12} {best_f1:>8.3f} {best_threshold:>12.0e} "
                                   f"{best_precision:>10.3f} {best_recall:>8.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Analysis completed successfully.\n")
            f.write("See figures/ and data/ directories for detailed results.\n")
            f.write("=" * 80 + "\n")
        
        print(f"  Summary report saved to {report_path}")
    
    def run_complete_analysis(self, file_paths: Dict[str, str], method_names: List[str]):
        """Execute the complete cis-pQTL analysis pipeline.
        
        Runs the full analysis workflow including data loading, significance filtering,
        overlap analysis, effect size analysis, performance calculations, figure
        generation, and report creation.
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            method_names: List of method display names
            
        Returns:
            pQTLData object with complete analysis results
        """
        print("Starting comprehensive cis-pQTL analysis...")
        print("=" * 60)
        
        # Load and validate data
        data = self.load_and_validate_data(file_paths, method_names)
        
        # Filter by MAF threshold
        data = self.filter_by_maf(data)
        
        # Filter for significant hits
        data = self.filter_significant_hits(data)
        
        # Analyze overlaps
        data = self.analyze_overlaps(data)
        
        # Analyze effect sizes
        data = self.analyze_effect_sizes(data)
        
        # Calculate performance metrics
        data = self.calculate_performance_metrics(data)
        
        # Analyze precision-recall curves
        pr_results = self.analyze_precision_recall_curves(data)
        
        print("\nGenerating visualizations...")
        
        # Generate PR curve analysis figure
        fig0 = self.generate_figure_0_precision_recall_analysis(data, pr_results)
        if fig0:
            self.save_figure(fig0, "figure_0_precision_recall_analysis")
            plt.close(fig0)
        
        # Generate figures
        fig1 = self.generate_figure_1_overlap_dashboard(data)
        if fig1:
            self.save_figure(fig1, "figure_1_overlap_dashboard")
            plt.close(fig1)
        
        fig2 = self.generate_figure_2_effect_size_concordance(data)
        if fig2:
            self.save_figure(fig2, "figure_2_effect_size_concordance")
            plt.close(fig2)
        
        # Generate new p-value and slope comparison plot
        fig3 = self.plot_pval_slope_comparison(data)
        if fig3:
            self.save_figure(fig3, "figure_3_pval_slope_comparison")
            plt.close(fig3)
        
        # Generate UCSC Genome Browser tracks
        self.generate_ucsc_browser_tracks(data)
        
        # Save results
        self.save_analysis_results(data, pr_results)
        
        # Generate summary report
        self.generate_summary_report(data, pr_results)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return data


def main():
    """Command-line interface for comprehensive cis-pQTL analysis.
    
    Parses command-line arguments, validates input files, and executes
    the complete analysis pipeline with progress reporting.
    """
    parser = argparse.ArgumentParser(
        description='Comprehensive cis-pQTL Analysis: Compare imputation methods against truth data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python cis_pQTL_analysis.py \\
    --truth truth_pqtl.tsv \\
    --method1 vae_pqtl.tsv \\
    --method2 knn_pqtl.tsv \\
    --method3 wnn_pqtl.tsv \\
    --method4 simple_pqtl.tsv \\
    --method1_name "Joint VAE" \\
    --method2_name "KNN" \\
    --method3_name "WNN" \\
    --method4_name "Simple Imputation" \\
    --output_dir results_pqtl \\
    --significance_threshold 0.05

Output:
  - figures/: Publication-ready plots (PDF + PNG)
  - data/: Analysis results in CSV format
  - reports/: Comprehensive text summary report
        """
    )
    
    # Required arguments
    parser.add_argument('--truth', required=True, 
                       help='Path to truth pQTL results TSV file')
    parser.add_argument('--method1', required=True,
                       help='Path to method 1 pQTL results TSV file')
    parser.add_argument('--method2', required=True,
                       help='Path to method 2 pQTL results TSV file')  
    parser.add_argument('--method3', required=True,
                       help='Path to method 3 pQTL results TSV file')
    parser.add_argument('--method4', required=True,
                       help='Path to method 4 pQTL results TSV file')
    
    # Method names
    parser.add_argument('--method1_name', required=True,
                       help='Display name for method 1 (e.g., "Joint VAE")')
    parser.add_argument('--method2_name', required=True,
                       help='Display name for method 2 (e.g., "KNN")')
    parser.add_argument('--method3_name', required=True,
                       help='Display name for method 3 (e.g., "WNN")')
    parser.add_argument('--method4_name', required=True,
                       help='Display name for method 4 (e.g., "Simple Imputation")')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='output_cis_pqtl_analysis',
                       help='Output directory for results (default: output_cis_pqtl_analysis)')
    parser.add_argument('--significance_threshold', type=float, default=0.05,
                       help='P-value threshold for significance (default: 0.05)')
    parser.add_argument('--truth_threshold', type=float, default=None,
                       help='Fixed p-value threshold for truth in PR analysis (defaults to significance_threshold)')
    parser.add_argument('--maf_threshold', type=float, default=0.01,
                       help='Minor Allele Frequency threshold for filtering variants (default: 0.01). ' +
                            'Filters variants with AF < maf_threshold or AF > (1 - maf_threshold)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    file_paths = {
        'truth': args.truth,
        'method1': args.method1,
        'method2': args.method2,
        'method3': args.method3,
        'method4': args.method4
    }
    
    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    # Method names
    method_names = [args.method1_name, args.method2_name, args.method3_name, args.method4_name]
    
    print("cis-pQTL Analysis Pipeline")
    print("=" * 50)
    print(f"Truth data: {args.truth}")
    print(f"Method 1 ({args.method1_name}): {args.method1}")
    print(f"Method 2 ({args.method2_name}): {args.method2}")
    print(f"Method 3 ({args.method3_name}): {args.method3}")
    print(f"Method 4 ({args.method4_name}): {args.method4}")
    print(f"Output directory: {args.output_dir}")
    print(f"Significance threshold: p < {args.significance_threshold}")
    if args.truth_threshold:
        print(f"Truth threshold for PR analysis: p < {args.truth_threshold}")
    print(f"MAF threshold: {args.maf_threshold} (filtering AF < {args.maf_threshold} or AF > {1 - args.maf_threshold})")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CispQTLAnalyzer(
        output_dir=args.output_dir,
        significance_threshold=args.significance_threshold,
        truth_threshold=args.truth_threshold,
        maf_threshold=args.maf_threshold
    )
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(file_paths, method_names)
        
        print("\nANALYSIS SUMMARY")
        print("=" * 50)
        
        # Print key findings
        if results.performance_metrics:
            print("METHOD PERFORMANCE RANKING:")
            methods = ['method1', 'method2', 'method3', 'method4']
            sorted_methods = sorted([m for m in methods if m in results.performance_metrics], 
                                   key=lambda x: results.performance_metrics[x]['composite_score'], 
                                   reverse=True)
            
            for rank, method in enumerate(sorted_methods, 1):
                metrics = results.performance_metrics[method]
                method_idx = int(method[-1]) - 1
                print(f"  #{rank}. {method_names[method_idx]}: {metrics['composite_score']:.3f}")
        
        if results.overlap_analysis:
            print(f"\nOVERLAP WITH TRUTH (Association level):")
            overlap_stats = results.overlap_analysis['overlap_stats']['association']
            for i, method in enumerate(['method1', 'method2', 'method3', 'method4']):
                if method in overlap_stats:
                    stats = overlap_stats[method]
                    print(f"  {method_names[i]}: {stats['overlap_count']}/{stats['truth_total']} "
                          f"(F1: {stats['f1_score']:.3f})")
        
        if results.effect_size_analysis:
            print(f"\nEFFECT SIZE CONCORDANCE:")
            for i, (method, stats) in enumerate(results.effect_size_analysis.items()):
                print(f"  {method_names[i]}: r={stats['pearson_r']:.3f}, "
                      f"MAE={stats['mae']:.3f}, Concordance={stats['concordance']:.3f}")
        
        print(f"\nComplete results available in: {args.output_dir}")
        print("   - figures/: Publication-ready visualizations")
        print("   - data/: Detailed analysis results (CSV)")
        print("   - reports/: Comprehensive summary report")
        
        # Determine and highlight the best method
        if results.performance_metrics:
            best_method = max(results.performance_metrics.keys(), 
                            key=lambda x: results.performance_metrics[x]['composite_score'])
            best_method_idx = int(best_method[-1]) - 1
            best_score = results.performance_metrics[best_method]['composite_score']
            
            print(f"\nBEST PERFORMING METHOD: {method_names[best_method_idx]}")
            print(f"   Composite Score: {best_score:.3f}")
            
            # Check if VAE-related method is best (assuming method1 is VAE)
            if best_method == 'method1':
                print("   VAE method shows superior performance!")
            else:
                print(f"   Note: VAE method ({method_names[0]}) ranked #{sorted_methods.index('method1')+1}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()