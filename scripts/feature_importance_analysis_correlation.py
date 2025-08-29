#!/usr/bin/env python3
"""
Feature Correlation Network Analysis for Joint VAE Models.

This script builds networks from raw data correlations instead of importance values,
allowing comparison with importance-based networks to understand whether learned
importance relationships capture biological interactions better than simple correlations.

The analysis includes:
- Pairwise correlation computation between all features
- Network construction based on correlation thresholds
- Automatic threshold optimization to match PPI reference density
- Network topology analysis and comparison with PPI reference
- Comprehensive visualization and comparison plots

Key differences from importance-based analysis:
- Networks built from raw data correlations (Pearson r)
- Edges created when |correlation| > threshold
- Threshold optimized to match target network density

Example usage:
    python feature_importance_analysis_correlation.py \
        --truth_a truth_a.csv \
        --truth_b truth_b.csv \
        --platform_a_name "Olink" \
        --platform_b_name "SomaScan" \
        --correlation_threshold 0.3 \
        --ppi_reference ppi_reference.txt
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import seaborn as sns
import yaml
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

# Set style
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Color palette - consistent with specified scientific journal palette
COLORS = {
    'primary': '#e64b35',      # Red (230,75,53)
    'secondary': '#4dbbd5',    # Light Blue (77,187,213)
    'accent': '#00a087',       # Teal (0,160,135)
    'warning': '#e64b35',      # Red (same as primary)
    'info': '#3c5488',         # Dark Blue (60,84,136)
    'success': '#00a087',      # Teal (same as accent)
    'danger': '#e64b35',       # Red (same as primary)
    'alternative_1': '#f39b7f', # Light Red (243,155,127)
    'alternative_2': '#4dbbd5', # Light Blue (same as secondary)
    'alternative_3': '#3c5488', # Dark Blue (same as info)
    'alternative_4': '#00a087', # Teal (same as accent)
    'alternative_5': '#f39b7f'  # Light Red (same as alternative_1)
}

@dataclass
class CorrelationAnalysisData:
    """
    Container for correlation network analysis data and results.
    
    Attributes:
        importance_a_to_b: DataFrame with input features as rows, output features as columns
        importance_b_to_a: DataFrame with input features as rows, output features as columns
        platform_a_name: Human-readable name for platform A
        platform_b_name: Human-readable name for platform B
        rank_consistency_a_to_b: Results from rank consistency analysis for A→B
        rank_consistency_b_to_a: Results from rank consistency analysis for B→A
        overlapping_features: List of features present in both platforms
        truth_a/truth_b: Ground truth data for performance calculation
        imp_a_m1/imp_a_m2: Imputed data for methods 1 and 2, platform A
        imp_b_m1/imp_b_m2: Imputed data for methods 1 and 2, platform B
        feature_mapping: Dictionary mapping numeric IDs to gene names
        feature_performance_a/b: Performance metrics for each feature
        network_a_to_b/network_b_to_a: NetworkX graphs for importance networks
        ppi_reference: Reference PPI network for validation
        threshold_analyses: Results from threshold optimization analysis
    """
    # Correlation matrices (features × features)
    correlation_matrix_a: pd.DataFrame = None
    correlation_matrix_b: pd.DataFrame = None
    
    # Raw importance matrices kept for comparison (optional)
    importance_a_to_b: pd.DataFrame = None
    importance_b_to_a: pd.DataFrame = None
    
    # Metadata
    platform_a_name: str = "Platform A"
    platform_b_name: str = "Platform B"
    
    # Analysis results
    rank_consistency_a_to_b: Dict = None
    rank_consistency_b_to_a: Dict = None
    
    # Overlapping features for matched analysis
    overlapping_features: List[str] = None
    
    # Raw data for computing feature performance (from compare_result.py style analysis)
    truth_a: pd.DataFrame = None
    truth_b: pd.DataFrame = None
    imp_a_m1: pd.DataFrame = None
    imp_a_m2: pd.DataFrame = None
    imp_b_m1: pd.DataFrame = None
    imp_b_m2: pd.DataFrame = None
    
    # Feature mapping (numeric ID -> gene name)
    feature_mapping: Dict = None
    
    # Computed feature performance data (r values)
    feature_performance_a: pd.DataFrame = None  # Platform A performance metrics
    feature_performance_b: pd.DataFrame = None  # Platform B performance metrics
    
    # Self-importance vs performance correlation results
    self_importance_vs_performance: Dict = None
    
    # Cross-platform raw correlation results and analysis
    cross_platform_raw_correlation: pd.DataFrame = None  # Raw truth A vs truth B per-feature correlation
    self_importance_vs_raw_correlation: Dict = None  # Correlation of self-importance vs raw cross-platform correlation
    
    # Network analysis results
    network_a_to_b: 'nx.Graph' = None  # Network graph for A→B direction
    network_b_to_a: 'nx.Graph' = None  # Network graph for B→A direction
    network_analysis_a_to_b: Dict = None  # Network topology analysis for A→B
    network_analysis_b_to_a: Dict = None  # Network topology analysis for B→A
    network_comparison: Dict = None  # Comparison between A→B and B→A networks
    
    # PPI reference and comparison results
    ppi_reference: 'nx.Graph' = None  # Reference PPI network
    ppi_reference_stats: Dict = None  # Statistics about PPI reference network
    ppi_comparison_a_to_b: Dict = None  # Comparison of A→B network with PPI
    ppi_comparison_b_to_a: Dict = None  # Comparison of B→A network with PPI
    
    # Threshold analysis results
    threshold_analyses: Dict = None  # Results from threshold analysis for optimal network construction


class CorrelationNetworkAnalyzer:
    """
    Main class for correlation-based network analysis.
    
    This class provides methods to build and analyze networks from raw data correlations,
    including correlation computation, network topology analysis, and PPI validation.
    
    Args:
        output_dir: Directory path for saving analysis results and figures
    
    Attributes:
        output_dir: Path object for output directory
        git_hash: Current git commit hash for reproducibility
        timestamp: Analysis timestamp for file naming
    """
    
    def __init__(self, output_dir: str = "correlation_network_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.git_hash = self._get_git_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash for reproducibility.
        
        Returns:
            str: Short git commit hash or 'unknown' if unavailable
        """
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def compute_feature_correlations(self, truth_data: pd.DataFrame, 
                                    min_samples: int = 3) -> pd.DataFrame:
        """
        Compute pairwise Pearson correlations between all features.
        
        Args:
            truth_data: DataFrame with samples as rows and features as columns
            min_samples: Minimum number of non-NaN samples required for correlation
            
        Returns:
            DataFrame with correlation matrix (features × features)
        """
        print(f"Computing pairwise correlations for {len(truth_data.columns)} features...")
        
        features = truth_data.columns.tolist()
        n_features = len(features)
        
        # Initialize correlation matrix
        corr_matrix = pd.DataFrame(np.zeros((n_features, n_features)), 
                                  index=features, columns=features)
        
        # Compute pairwise correlations
        for i, feat1 in enumerate(features):
            if i % 100 == 0:
                print(f"  Processing feature {i}/{n_features}...")
            
            for j, feat2 in enumerate(features):
                if i == j:
                    corr_matrix.loc[feat1, feat2] = 1.0
                elif j < i:
                    # Use already computed value (symmetric matrix)
                    corr_matrix.loc[feat1, feat2] = corr_matrix.loc[feat2, feat1]
                else:
                    # Compute correlation
                    vals1 = truth_data[feat1].values
                    vals2 = truth_data[feat2].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(vals1) | np.isnan(vals2))
                    if np.sum(mask) < min_samples:
                        corr_matrix.loc[feat1, feat2] = 0.0
                    else:
                        r, _ = pearsonr(vals1[mask], vals2[mask])
                        corr_matrix.loc[feat1, feat2] = r
        
        print(f"  Correlation matrix computed: {corr_matrix.shape}")
        return corr_matrix
    
    def load_importance_matrices(self, file_paths: Dict[str, str], 
                                platform_a_name: str = "Platform A",
                                platform_b_name: str = "Platform B") -> CorrelationAnalysisData:
        """
        Load importance matrices from CSV files (kept for comparison).
        
        Args:
            file_paths: Dictionary with keys 'importance_a_to_b' and 'importance_b_to_a'
                       mapping to CSV file paths
            platform_a_name: Human-readable name for platform A
            platform_b_name: Human-readable name for platform B
            
        Returns:
            CorrelationAnalysisData object with loaded matrices and metadata
        """
        print("Loading importance matrices (optional for comparison)...")
        
        data = CorrelationAnalysisData(
            platform_a_name=platform_a_name,
            platform_b_name=platform_b_name
        )
        
        if 'importance_a_to_b' in file_paths and file_paths['importance_a_to_b']:
            data.importance_a_to_b = pd.read_csv(file_paths['importance_a_to_b'], index_col=0)
            print(f"  A→B importance matrix shape: {data.importance_a_to_b.shape}")
        
        if 'importance_b_to_a' in file_paths and file_paths['importance_b_to_a']:
            data.importance_b_to_a = pd.read_csv(file_paths['importance_b_to_a'], index_col=0)
            print(f"  B→A importance matrix shape: {data.importance_b_to_a.shape}")
        
        if data.importance_a_to_b is not None and data.importance_b_to_a is not None:
            input_features_a = set(data.importance_a_to_b.index)
            output_features_a = set(data.importance_a_to_b.columns)
            input_features_b = set(data.importance_b_to_a.index)
            output_features_b = set(data.importance_b_to_a.columns)
            
            overlapping_ab = input_features_a & output_features_b
            overlapping_ba = input_features_b & output_features_a
            
            data.overlapping_features = list(overlapping_ab | overlapping_ba)
            print(f"  Found {len(data.overlapping_features)} overlapping features between platforms")
        
        return data
    
    def load_raw_data_for_performance(self, data: CorrelationAnalysisData, 
                                     raw_data_file_paths: Dict[str, str]) -> CorrelationAnalysisData:
        """
        Load raw truth and imputed data files for computing feature performance.
        
        Args:
            data: CorrelationAnalysisData object to update
            raw_data_file_paths: Dictionary with keys 'truth_a', 'truth_b', 
                               'imp_a_m1', 'imp_a_m2', 'imp_b_m1', 'imp_b_m2'
                               
        Returns:
            Updated CorrelationAnalysisData object with loaded raw data
        """
        print("Loading raw data files for performance calculation...")
        
        # Load truth data
        if 'truth_a' in raw_data_file_paths and raw_data_file_paths['truth_a']:
            try:
                data.truth_a = pd.read_csv(raw_data_file_paths['truth_a'], index_col=0)
                print(f"  Truth A data shape: {data.truth_a.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading truth A data: {e}")
        
        if 'truth_b' in raw_data_file_paths and raw_data_file_paths['truth_b']:
            try:
                data.truth_b = pd.read_csv(raw_data_file_paths['truth_b'], index_col=0)
                print(f"  Truth B data shape: {data.truth_b.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading truth B data: {e}")
        
        # Load imputed data
        if 'imp_a_m1' in raw_data_file_paths and raw_data_file_paths['imp_a_m1']:
            try:
                data.imp_a_m1 = pd.read_csv(raw_data_file_paths['imp_a_m1'], index_col=0)
                print(f"  Imputed A Method 1 data shape: {data.imp_a_m1.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading imputed A method 1 data: {e}")
        
        if 'imp_a_m2' in raw_data_file_paths and raw_data_file_paths['imp_a_m2']:
            try:
                data.imp_a_m2 = pd.read_csv(raw_data_file_paths['imp_a_m2'], index_col=0)
                print(f"  Imputed A Method 2 data shape: {data.imp_a_m2.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading imputed A method 2 data: {e}")
        
        if 'imp_b_m1' in raw_data_file_paths and raw_data_file_paths['imp_b_m1']:
            try:
                data.imp_b_m1 = pd.read_csv(raw_data_file_paths['imp_b_m1'], index_col=0)
                print(f"  Imputed B Method 1 data shape: {data.imp_b_m1.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading imputed B method 1 data: {e}")
        
        if 'imp_b_m2' in raw_data_file_paths and raw_data_file_paths['imp_b_m2']:
            try:
                data.imp_b_m2 = pd.read_csv(raw_data_file_paths['imp_b_m2'], index_col=0)
                print(f"  Imputed B Method 2 data shape: {data.imp_b_m2.shape} (samples × features)")
            except Exception as e:
                print(f"  Error loading imputed B method 2 data: {e}")
        
        return data
    
    def compute_feature_performance(self, data: CorrelationAnalysisData) -> CorrelationAnalysisData:
        """
        Compute feature-wise performance metrics by comparing truth and imputed data.
        
        Calculates Pearson correlation coefficients for each feature between truth
        and imputed values. Applies feature mapping if available.
        
        Args:
            data: CorrelationAnalysisData object with truth and imputed data loaded
            
        Returns:
            Updated CorrelationAnalysisData object with performance metrics
        """
        print("Computing feature performance metrics...")
        
        # Platform A performance
        if data.truth_a is not None and (data.imp_a_m1 is not None or data.imp_a_m2 is not None):
            results_a = []
            
            # Check both methods if available
            for method_name, imputed_data in [('Method_1', data.imp_a_m1), ('Method_2', data.imp_a_m2)]:
                if imputed_data is None:
                    continue
                
                # Iterate over features (columns)
                for feature in data.truth_a.columns:
                    if feature not in imputed_data.columns:
                        continue
                        
                    truth_vals = data.truth_a[feature].values
                    imp_vals = imputed_data[feature].values
                    
                    # Skip if all values are NaN
                    mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                        
                    truth_clean = truth_vals[mask]
                    imp_clean = imp_vals[mask]
                    
                    # Compute metrics
                    r, r_p = pearsonr(truth_clean, imp_clean)
                    
                    results_a.append({
                        'feature': feature,
                        'method': method_name,
                        'r': r,
                        'r_pvalue': r_p,
                        'n_samples': np.sum(mask)
                    })
            
            if results_a:
                results_a_df = pd.DataFrame(results_a)
                
                if data.feature_mapping:
                    print(f"  Applying feature mapping to Platform A features...")
                    results_a_df['feature'] = results_a_df['feature'].map(data.feature_mapping).fillna(results_a_df['feature'])
                    mapped_count = results_a_df['feature'].apply(lambda x: isinstance(x, str)).sum()
                    print(f"  Mapped {mapped_count}/{len(results_a_df)} Platform A features to gene names")
                
                # Average across methods for each feature
                data.feature_performance_a = results_a_df.groupby('feature')['r'].mean().reset_index()
                print(f"  Platform A performance computed for {len(data.feature_performance_a)} features")
        
        # Platform B performance
        if data.truth_b is not None and (data.imp_b_m1 is not None or data.imp_b_m2 is not None):
            results_b = []
            
            # Check both methods if available
            for method_name, imputed_data in [('Method_1', data.imp_b_m1), ('Method_2', data.imp_b_m2)]:
                if imputed_data is None:
                    continue
                
                # Iterate over features (columns)
                for feature in data.truth_b.columns:
                    if feature not in imputed_data.columns:
                        continue
                        
                    truth_vals = data.truth_b[feature].values
                    imp_vals = imputed_data[feature].values
                    
                    # Skip if all values are NaN
                    mask = ~(np.isnan(truth_vals) | np.isnan(imp_vals))
                    if np.sum(mask) < 3:  # Need at least 3 points
                        continue
                        
                    truth_clean = truth_vals[mask]
                    imp_clean = imp_vals[mask]
                    
                    # Compute metrics
                    r, r_p = pearsonr(truth_clean, imp_clean)
                    
                    results_b.append({
                        'feature': feature,
                        'method': method_name,
                        'r': r,
                        'r_pvalue': r_p,
                        'n_samples': np.sum(mask)
                    })
            
            if results_b:
                results_b_df = pd.DataFrame(results_b)
                
                if data.feature_mapping:
                    print(f"  Applying feature mapping to Platform B features...")
                    results_b_df['feature'] = results_b_df['feature'].map(data.feature_mapping).fillna(results_b_df['feature'])
                    mapped_count = results_b_df['feature'].apply(lambda x: isinstance(x, str)).sum()
                    print(f"  Mapped {mapped_count}/{len(results_b_df)} Platform B features to gene names")
                
                # Average across methods for each feature
                data.feature_performance_b = results_b_df.groupby('feature')['r'].mean().reset_index()
                print(f"  Platform B performance computed for {len(data.feature_performance_b)} features")
        
        return data

    def compute_cross_platform_raw_correlation(self, data: CorrelationAnalysisData) -> CorrelationAnalysisData:
        """
        Compute raw cross-platform Pearson correlation between truth matrices for overlapping features.
        Expects `truth_a` and `truth_b` as DataFrames with shape (samples × features).

        Returns the updated `data` with `cross_platform_raw_correlation` as a DataFrame
        containing columns: ['feature', 'r'].
        """
        if data.truth_a is None or data.truth_b is None:
            return data

        print("Computing cross-platform raw correlations (truth A vs truth B)...")

        features_a = set(data.truth_a.columns)
        features_b = set(data.truth_b.columns)
        overlapping = sorted(list(features_a & features_b))

        if len(overlapping) == 0:
            print("  No overlapping features between truth_a and truth_b")
            data.cross_platform_raw_correlation = pd.DataFrame(columns=['feature', 'r'])
            return data

        results = []
        for feature in overlapping:
            vals_a = data.truth_a[feature].values
            vals_b = data.truth_b[feature].values

            mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
            if np.sum(mask) < 3:
                continue

            r, _ = pearsonr(vals_a[mask], vals_b[mask])
            results.append({
                'feature': feature,
                'r': r,
                'n_samples': int(np.sum(mask))
            })

        if not results:
            print("  Insufficient overlapping non-NaN samples to compute correlations")
            data.cross_platform_raw_correlation = pd.DataFrame(columns=['feature', 'r'])
            return data

        df = pd.DataFrame(results)

        if data.feature_mapping:
            print("  Applying feature mapping to cross-platform correlation features...")
            df['feature'] = df['feature'].map(data.feature_mapping).fillna(df['feature'])

        # Aggregate in case multiple IDs map to the same gene symbol
        data.cross_platform_raw_correlation = df.groupby('feature')['r'].mean().reset_index()
        print(f"  Cross-platform raw correlation computed for {len(data.cross_platform_raw_correlation)} features")
        return data
    
    # Removed analyze_self_feature_importance - not applicable to correlation networks
    
    # Removed analyze_rank_consistency - not applicable to correlation networks
    
    # Removed analyze_self_importance_vs_performance - not applicable to correlation networks

    # Removed analyze_self_importance_vs_raw_correlation - not applicable to correlation networks
    
    # Removed _correlate_importance_performance - not applicable to correlation networks
    
    def _interpret_correlation(self, rank_corr, rank_p, score_corr, score_p) -> str:
        """
        Provide human-readable interpretation of correlation results.
        
        Args:
            rank_corr: Spearman correlation coefficient for ranks
            rank_p: P-value for rank correlation
            score_corr: Pearson correlation coefficient for scores  
            score_p: P-value for score correlation
            
        Returns:
            String with interpretation of correlation strength and significance
        """
        alpha = 0.05
        
        interpretations = []
        if rank_p < alpha:
            if rank_corr < -0.5:
                interpretations.append("Strong negative correlation: Higher self-importance rank (worse) → Lower performance")
            elif rank_corr < -0.3:
                interpretations.append("Moderate negative correlation: Higher self-importance rank → Lower performance")
            elif rank_corr > 0.5:
                interpretations.append("Strong positive correlation: Higher self-importance rank → Higher performance")
            elif rank_corr > 0.3:
                interpretations.append("Moderate positive correlation: Higher self-importance rank → Higher performance")
            else:
                interpretations.append("Weak significant correlation between rank and performance")
        else:
            interpretations.append("No significant correlation between self-importance rank and performance")
        
        if score_p < alpha:
            if score_corr > 0.5:
                interpretations.append("Strong positive correlation: Higher self-importance score → Higher performance")
            elif score_corr > 0.3:
                interpretations.append("Moderate positive correlation: Higher self-importance score → Higher performance")
            elif score_corr < -0.5:
                interpretations.append("Strong negative correlation: Higher self-importance score → Lower performance")
            elif score_corr < -0.3:
                interpretations.append("Moderate negative correlation: Higher self-importance score → Lower performance")
            else:
                interpretations.append("Weak significant correlation between score and performance")
        else:
            interpretations.append("No significant correlation between self-importance score and performance")
        
        return "; ".join(interpretations)
    
    def build_correlation_network(self, correlation_matrix: pd.DataFrame, 
                                 threshold: float = 0.3,
                                 use_absolute: bool = True) -> 'nx.Graph':
        """
        Build a network graph based on feature correlations.
        
        Creates edges between features when their correlation exceeds the threshold.
        
        Args:
            correlation_matrix: Correlation matrix (features × features)
            threshold: Correlation threshold for creating edges
            use_absolute: If True, use absolute correlation values
            
        Returns:
            NetworkX graph object with correlation-based edges
        """
        if not NETWORKX_AVAILABLE:
            return None
        
        print(f"Building correlation network with threshold {threshold:.3f}...")
        if use_absolute:
            print(f"  Using absolute correlation values")
        
        # Always use undirected graph for correlations
        G = nx.Graph()
        
        features = list(correlation_matrix.index)
        G.add_nodes_from(features)
        
        print(f"  Added {len(features)} nodes to network")
        
        edges_added = 0
        n_features = len(features)
        
        # Iterate through upper triangle of correlation matrix
        for i in range(n_features):
            for j in range(i+1, n_features):
                feat1 = features[i]
                feat2 = features[j]
                
                corr_value = correlation_matrix.loc[feat1, feat2]
                
                # Apply threshold
                if use_absolute:
                    if abs(corr_value) > threshold:
                        G.add_edge(feat1, feat2, 
                                 weight=abs(corr_value),
                                 correlation=corr_value)
                        edges_added += 1
                else:
                    if corr_value > threshold:
                        G.add_edge(feat1, feat2, 
                                 weight=corr_value,
                                 correlation=corr_value)
                        edges_added += 1
        
        print(f"  Added {edges_added} edges to network")
        print(f"  Final network: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        G.graph['network_type'] = 'undirected'
        G.graph['threshold_method'] = 'correlation'
        G.graph['threshold_value'] = threshold
        G.graph['use_absolute'] = use_absolute
        G.graph['total_features'] = len(features)
        
        return G
    
    def analyze_network_topology(self, network: 'nx.Graph') -> Dict:
        """
        Analyze network topology and compute comprehensive metrics.
        
        Computes basic statistics, centrality measures, connectivity metrics,
        and community structure for the importance network.
        
        Args:
            network: NetworkX graph object to analyze
            
        Returns:
            Dictionary containing network topology analysis results
        """
        if not NETWORKX_AVAILABLE or network is None:
            return {}
            
        print("Analyzing network topology...")
        
        analysis = {
            'basic_stats': {},
            'centrality': {},
            'connectivity': {},
            'community': {}
        }
        
        analysis['basic_stats'] = {
            'n_nodes': len(network.nodes()),
            'n_edges': len(network.edges()),
            'density': nx.density(network),
            'is_directed': network.is_directed()
        }
        
        if len(network.nodes()) == 0:
            print("  Empty network - skipping analysis")
            return analysis
        
        degrees = dict(network.degree())
        if degrees:
            degree_values = list(degrees.values())
            analysis['basic_stats'].update({
                'mean_degree': np.mean(degree_values),
                'std_degree': np.std(degree_values),
                'max_degree': max(degree_values),
                'min_degree': min(degree_values)
            })
        
        if len(network.nodes()) <= 1000:
            try:
                print(f"  Computing centrality measures for {len(network.nodes())} nodes...")
                analysis['centrality']['degree'] = nx.degree_centrality(network)
                analysis['centrality']['betweenness'] = nx.betweenness_centrality(network, k=min(100, len(network.nodes())))
                analysis['centrality']['closeness'] = nx.closeness_centrality(network)
                
                if not network.is_directed():
                    analysis['centrality']['eigenvector'] = nx.eigenvector_centrality(network, max_iter=1000)
                    
            except Exception as e:
                print(f"  Warning: Could not compute some centrality measures: {e}")
        else:
            print(f"  Skipping centrality measures - network too large ({len(network.nodes())} > 1000 nodes)")
        
        try:
            if network.is_directed():
                analysis['connectivity']['strongly_connected_components'] = len(list(nx.strongly_connected_components(network)))
                analysis['connectivity']['weakly_connected_components'] = len(list(nx.weakly_connected_components(network)))
            else:
                analysis['connectivity']['connected_components'] = len(list(nx.connected_components(network)))
                largest_cc = max(nx.connected_components(network), key=len)
                analysis['connectivity']['largest_component_size'] = len(largest_cc)
                analysis['connectivity']['largest_component_fraction'] = len(largest_cc) / len(network.nodes())
                analysis['connectivity']['average_clustering'] = nx.average_clustering(network)
                
        except Exception as e:
            print(f"  Warning: Could not compute connectivity measures: {e}")
        
        if COMMUNITY_AVAILABLE and len(network.edges()) > 0:
            try:
                print(f"  Performing community detection...")
                undirected_network = network.to_undirected() if network.is_directed() else network.copy()
                undirected_network.remove_edges_from(nx.selfloop_edges(undirected_network))
                
                if len(undirected_network.edges()) > 0:
                    partition = community_louvain.best_partition(undirected_network)
                    modularity = community_louvain.modularity(partition, undirected_network)
                    
                    analysis['community']['partition'] = partition
                    analysis['community']['modularity'] = modularity
                    analysis['community']['n_communities'] = len(set(partition.values()))
                    
                    community_sizes = {}
                    for community_id in set(partition.values()):
                        community_sizes[community_id] = sum(1 for node_community in partition.values() 
                                                          if node_community == community_id)
                    analysis['community']['community_sizes'] = community_sizes
                    print(f"  Community detection completed: {len(set(partition.values()))} communities, modularity = {modularity:.3f}")
                else:
                    print(f"  Skipping community detection - no edges after removing self-loops")
                    
            except Exception as e:
                print(f"  Warning: Could not perform community detection: {e}")
        else:
            if not COMMUNITY_AVAILABLE:
                print(f"  Skipping community detection - python-louvain not available")
            elif len(network.edges()) == 0:
                print(f"  Skipping community detection - no edges in network")
        
        print(f"  Network analysis completed for {analysis['basic_stats']['n_nodes']} nodes, {analysis['basic_stats']['n_edges']} edges")
        
        return analysis
    
    def compare_networks(self, network_a_to_b: 'nx.Graph', network_b_to_a: 'nx.Graph', 
                        analysis_a_to_b: Dict, analysis_b_to_a: Dict) -> Dict:
        """
        Compare networks between different directions (A→B vs B→A).
        
        Computes overlap statistics, centrality correlations, and structural
        similarities between the two importance networks.
        
        Args:
            network_a_to_b: NetworkX graph for A→B direction
            network_b_to_a: NetworkX graph for B→A direction
            analysis_a_to_b: Topology analysis results for A→B
            analysis_b_to_a: Topology analysis results for B→A
            
        Returns:
            Dictionary containing network comparison results
        """
        if not NETWORKX_AVAILABLE:
            return {}
            
        print("Comparing networks between directions...")
        
        comparison = {
            'basic_comparison': {},
            'node_overlap': {},
            'edge_overlap': {},
            'centrality_correlation': {}
        }
        
        if network_a_to_b is None or network_b_to_a is None:
            print("  One or both networks are missing - skipping comparison")
            return comparison
        
        stats_a = analysis_a_to_b.get('basic_stats', {})
        stats_b = analysis_b_to_a.get('basic_stats', {})
        
        comparison['basic_comparison'] = {
            'nodes_a_to_b': stats_a.get('n_nodes', 0),
            'nodes_b_to_a': stats_b.get('n_nodes', 0),
            'edges_a_to_b': stats_a.get('n_edges', 0),
            'edges_b_to_a': stats_b.get('n_edges', 0),
            'density_a_to_b': stats_a.get('density', 0),
            'density_b_to_a': stats_b.get('density', 0)
        }
        
        nodes_a = set(network_a_to_b.nodes())
        nodes_b = set(network_b_to_a.nodes())
        edges_a = set(network_a_to_b.edges())
        edges_b = set(network_b_to_a.edges())
        
        comparison['node_overlap'] = {
            'intersection': len(nodes_a & nodes_b),
            'union': len(nodes_a | nodes_b),
            'jaccard': len(nodes_a & nodes_b) / len(nodes_a | nodes_b) if len(nodes_a | nodes_b) > 0 else 0
        }
        
        comparison['edge_overlap'] = {
            'intersection': len(edges_a & edges_b),
            'union': len(edges_a | edges_b),
            'jaccard': len(edges_a & edges_b) / len(edges_a | edges_b) if len(edges_a | edges_b) > 0 else 0
        }
        
        if ('centrality' in analysis_a_to_b and 'centrality' in analysis_b_to_a and 
            len(nodes_a & nodes_b) >= 3):
            
            overlapping_nodes = list(nodes_a & nodes_b)
            
            for centrality_type in ['degree', 'betweenness', 'closeness']:
                if (centrality_type in analysis_a_to_b['centrality'] and 
                    centrality_type in analysis_b_to_a['centrality']):
                    
                    centrality_a = analysis_a_to_b['centrality'][centrality_type]
                    centrality_b = analysis_b_to_a['centrality'][centrality_type]
                    
                    values_a = [centrality_a.get(node, 0) for node in overlapping_nodes]
                    values_b = [centrality_b.get(node, 0) for node in overlapping_nodes]
                    
                    if len(values_a) >= 3:
                        try:
                            correlation, p_value = pearsonr(values_a, values_b)
                            comparison['centrality_correlation'][centrality_type] = {
                                'correlation': correlation,
                                'p_value': p_value,
                                'n_nodes': len(overlapping_nodes)
                            }
                        except Exception as e:
                            print(f"  Warning: Could not compute {centrality_type} centrality correlation: {e}")
        
        return comparison
    
    def analyze_correlation_thresholds(self, correlation_matrix: pd.DataFrame, 
                                      target_density: float = 0.0366,
                                      use_absolute: bool = True) -> Dict:
        """
        Analyze different correlation threshold values for optimal network construction.
        
        Tests various correlation thresholds and computes resulting network properties
        to recommend optimal parameters matching target density.
        
        Args:
            correlation_matrix: Correlation matrix (features × features)
            target_density: Target network density to match (e.g., PPI density)
            use_absolute: If True, use absolute correlation values
            
        Returns:
            Dictionary with threshold analysis results and recommendations
        """
        print(f"Analyzing correlation thresholds (use_absolute={use_absolute})...")
        
        features = list(correlation_matrix.index)
        n_features = len(features)
        
        analysis = {
            'threshold_type': 'correlation',
            'use_absolute': use_absolute,
            'n_features': n_features,
            'thresholds': [],
            'edge_counts': [],
            'node_counts': [],
            'densities': [],
            'largest_component_sizes': [],
            'mean_degrees': [],
            'recommendations': {}
        }
        
        # Test correlation thresholds from 0.05 to 0.95
        thresholds_corr = np.arange(0.05, 1.0, 0.05)
        
        for threshold in thresholds_corr:
            edge_count = 0
            connected_nodes = set()
            
            # Count edges for this threshold
            for i in range(n_features):
                for j in range(i+1, n_features):
                    feat1 = features[i]
                    feat2 = features[j]
                    
                    corr_value = correlation_matrix.loc[feat1, feat2]
                    
                    if use_absolute:
                        if abs(corr_value) > threshold:
                            edge_count += 1
                            connected_nodes.add(feat1)
                            connected_nodes.add(feat2)
                    else:
                        if corr_value > threshold:
                            edge_count += 1
                            connected_nodes.add(feat1)
                            connected_nodes.add(feat2)
            
            n_nodes = len(connected_nodes)
            density = (2 * edge_count) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            mean_degree = (2 * edge_count) / n_nodes if n_nodes > 0 else 0
            
            analysis['thresholds'].append(threshold)
            analysis['edge_counts'].append(edge_count)
            analysis['node_counts'].append(n_nodes)
            analysis['densities'].append(density)
            analysis['mean_degrees'].append(mean_degree)
            analysis['largest_component_sizes'].append(n_nodes)
        
        edge_counts = np.array(analysis['edge_counts'])
        node_counts = np.array(analysis['node_counts'])
        densities = np.array(analysis['densities'])
        
        recommendations = {}
        
        good_density_idx = np.where((densities >= 0.01) & (densities <= 0.1))[0]
        if len(good_density_idx) > 0:
            best_density_idx = good_density_idx[np.argmax(node_counts[good_density_idx])]
            rec_data = {
                'threshold': analysis['thresholds'][best_density_idx],
                'edge_count': analysis['edge_counts'][best_density_idx],
                'node_count': analysis['node_counts'][best_density_idx],
                'density': analysis['densities'][best_density_idx],
                'reason': 'Good balance of density (1-10%) with maximum nodes'
            }
            # No threshold labels needed for correlation analysis
            recommendations['moderate_density'] = rec_data
        
        good_edge_idx = np.where((edge_counts >= 1000) & (edge_counts <= 10000))[0]
        if len(good_edge_idx) > 0:
            best_edge_idx = good_edge_idx[np.argmax(node_counts[good_edge_idx])]
            rec_data = {
                'threshold': analysis['thresholds'][best_edge_idx],
                'edge_count': analysis['edge_counts'][best_edge_idx],
                'node_count': analysis['node_counts'][best_edge_idx],
                'density': analysis['densities'][best_edge_idx],
                'reason': 'Target 1K-10K edges for computational efficiency'
            }
            # No threshold labels needed for correlation analysis
            recommendations['target_edges'] = rec_data
        
        if len(edge_counts) > 2:
            edge_diffs = np.diff(edge_counts)
            if len(edge_diffs) > 0:
                elbow_idx = np.argmax(np.abs(edge_diffs)) + 1
                if elbow_idx < len(analysis['thresholds']):
                    rec_data = {
                        'threshold': analysis['thresholds'][elbow_idx],
                        'edge_count': analysis['edge_counts'][elbow_idx],
                        'node_count': analysis['node_counts'][elbow_idx],
                        'density': analysis['densities'][elbow_idx],
                        'reason': 'Elbow point in edge count curve'
                    }
                    # No threshold labels needed for correlation analysis
                    recommendations['elbow_method'] = rec_data
        
        # Add edge saturation recommendation (preferred automatic method)
        if analysis['edge_counts'] and analysis['node_counts']:
            edge_saturation_idx = None
            
            # Find the threshold where edges <= nodes (meaningful network structure)
            # and edges stop increasing significantly
            for i in range(len(analysis['edge_counts'])):
                edge_count = analysis['edge_counts'][i]
                node_count = analysis['node_counts'][i]
                
                # Check if edges <= nodes (reasonable network structure)
                if edge_count <= node_count:
                    edge_saturation_idx = i
                    break
            
            # If no threshold satisfies edges <= nodes, find the one with the smallest edge/node ratio
            if edge_saturation_idx is None:
                edge_node_ratios = [e/n if n > 0 else float('inf') for e, n in zip(analysis['edge_counts'], analysis['node_counts'])]
                edge_saturation_idx = edge_node_ratios.index(min(edge_node_ratios))
            
            if edge_saturation_idx is not None:
                best_threshold = analysis['thresholds'][edge_saturation_idx]
                edge_count = analysis['edge_counts'][edge_saturation_idx]
                node_count = analysis['node_counts'][edge_saturation_idx]
                
                recommendations['edge_saturation'] = {
                    'threshold': best_threshold,
                    'edge_count': edge_count,
                    'node_count': node_count,
                    'density': analysis['densities'][edge_saturation_idx] if analysis['densities'] else 0,
                    'mean_degree': analysis['mean_degrees'][edge_saturation_idx] if analysis['mean_degrees'] else 0,
                    'reason': f'Edge saturation point: {edge_count} edges ≤ {node_count} nodes (ratio: {edge_count/node_count:.3f})'
                }
        
        # Add node plateau recommendation (first threshold reaching ~max nodes)
        if analysis['node_counts']:
            max_nodes = max(analysis['node_counts'])
            plateau_idx = None
            for i, n in enumerate(analysis['node_counts']):
                if n >= 0.99 * max_nodes:
                    plateau_idx = i
                    break
            if plateau_idx is not None:
                recommendations['node_plateau'] = {
                    'threshold': analysis['thresholds'][plateau_idx],
                    'edge_count': analysis['edge_counts'][plateau_idx],
                    'node_count': analysis['node_counts'][plateau_idx],
                    'density': analysis['densities'][plateau_idx] if analysis['densities'] else 0,
                    'mean_degree': analysis['mean_degrees'][plateau_idx] if analysis['mean_degrees'] else 0,
                    'reason': f'Node count plateau reached (~99% of max nodes: {max_nodes})'
                }
        
        # Add target density recommendation
        if analysis['densities']:
            density_diffs = [abs(d - target_density) for d in analysis['densities']]
            min_diff_idx = density_diffs.index(min(density_diffs))
            best_threshold = analysis['thresholds'][min_diff_idx]
            best_density = analysis['densities'][min_diff_idx]
            
            recommendations['target_density'] = {
                'threshold': best_threshold,
                'edge_count': analysis['edge_counts'][min_diff_idx],
                'node_count': analysis['node_counts'][min_diff_idx],
                'density': best_density,
                'mean_degree': analysis['mean_degrees'][min_diff_idx] if analysis['mean_degrees'] else 0,
                'reason': f'Closest to target density {target_density:.4f} (actual: {best_density:.4f})'
            }
        
        analysis['recommendations'] = recommendations
        
        print(f"  Analyzed {len(analysis['thresholds'])} threshold values")
        print(f"  Edge count range: {min(edge_counts)} - {max(edge_counts)}")
        print(f"  Generated {len(recommendations)} recommendations")
        
        if recommendations:
            print(f"  → For correlation thresholding, use: --correlation_threshold <threshold>")
            for rec_name, rec_data in recommendations.items():
                threshold_val = rec_data['threshold']
                if isinstance(threshold_val, (int, float)):
                    print(f"    {rec_name}: --correlation_threshold {threshold_val:.3f}")
                else:
                    print(f"    {rec_name}: --correlation_threshold {threshold_val}")
        
        # Print edge saturation recommendation separately (this is the automatic default)
        if 'edge_saturation' in recommendations:
            rec_data = recommendations['edge_saturation']
            threshold_val = rec_data['threshold']
            edge_count = rec_data['edge_count']
            node_count = rec_data['node_count']
            print(f"\n  → EDGE SATURATION RECOMMENDATION (AUTOMATIC DEFAULT):")
            print(f"    Recommended threshold: --correlation_threshold {threshold_val:.3f}")
            print(f"    {rec_data['reason']}")
            print(f"    Network density: {rec_data['density']:.4f}")
        
        # Print target density recommendation separately
        if 'target_density' in recommendations:
            rec_data = recommendations['target_density']
            threshold_val = rec_data['threshold']
            actual_density = rec_data['density']
            print(f"\n  → TARGET DENSITY RECOMMENDATION (density {target_density:.4f}):")
            print(f"    Best threshold: --correlation_threshold {threshold_val:.3f}")
            print(f"    Achieved density: {actual_density:.4f} (diff: {abs(actual_density - target_density):.4f})")
            print(f"    Network size: {rec_data['edge_count']:,} edges, {rec_data['node_count']:,} nodes")
        
        if 'node_plateau' in recommendations:
            rec_data = recommendations['node_plateau']
            threshold_val = rec_data['threshold']
            print(f"\n  → NODE PLATEAU THRESHOLD:")
            print(f"    Threshold: --correlation_threshold {threshold_val:.3f}")
            print(f"    {rec_data['reason']}")
        
        return analysis
    
    def load_ppi_reference(self, ppi_file_path: str, symbol1_col: str = "symbol1", 
                          symbol2_col: str = "symbol2", confidence_col: str = None,
                          confidence_threshold: float = 0.0) -> 'nx.Graph':
        """
        Load protein-protein interaction (PPI) reference network from file.
        
        Reads PPI interactions from a tab-delimited file and creates a NetworkX graph
        for comparison with importance networks.
        
        Args:
            ppi_file_path: Path to PPI reference file (tab-delimited)
            symbol1_col: Column name for first protein symbol
            symbol2_col: Column name for second protein symbol
            confidence_col: Optional column name for confidence scores
            confidence_threshold: Minimum confidence threshold for interactions
            
        Returns:
            NetworkX undirected graph with PPI interactions
        """
        if not NETWORKX_AVAILABLE:
            return None
            
        try:
            print(f"Loading PPI reference from: {ppi_file_path}")
            
            ppi_df = pd.read_csv(ppi_file_path, sep='\t')
            print(f"  Read {len(ppi_df)} rows from PPI file")
            
            if symbol1_col not in ppi_df.columns:
                raise ValueError(f"Column '{symbol1_col}' not found in PPI file. Available columns: {list(ppi_df.columns)}")
            if symbol2_col not in ppi_df.columns:
                raise ValueError(f"Column '{symbol2_col}' not found in PPI file. Available columns: {list(ppi_df.columns)}")
            
            if confidence_col and confidence_col in ppi_df.columns:
                initial_count = len(ppi_df)
                ppi_df = ppi_df[ppi_df[confidence_col] >= confidence_threshold]
                print(f"  Filtered by confidence >= {confidence_threshold}: {len(ppi_df)}/{initial_count} interactions retained")
            elif confidence_col:
                print(f"Warning: Confidence column '{confidence_col}' not found in PPI file")
            
            G = nx.Graph()
            
            edges_added = 0
            for _, row in ppi_df.iterrows():
                protein1 = str(row[symbol1_col]).strip()
                protein2 = str(row[symbol2_col]).strip()
                
                if pd.isna(protein1) or pd.isna(protein2) or protein1 == '' or protein2 == '':
                    continue
                if protein1 == protein2:
                    continue
                
                edge_attrs = {'source': 'ppi_reference'}
                if confidence_col and confidence_col in ppi_df.columns:
                    edge_attrs['confidence'] = row[confidence_col]
                
                for col in ppi_df.columns:
                    if col not in [symbol1_col, symbol2_col, confidence_col]:
                        edge_attrs[col] = row[col]
                
                G.add_edge(protein1, protein2, **edge_attrs)
                edges_added += 1
            
            print(f"  Created PPI reference network: {len(G.nodes())} nodes, {len(G.edges())} edges")
            
            G.graph['source_file'] = ppi_file_path
            G.graph['symbol1_col'] = symbol1_col
            G.graph['symbol2_col'] = symbol2_col
            G.graph['confidence_col'] = confidence_col
            G.graph['confidence_threshold'] = confidence_threshold
            G.graph['original_interactions'] = len(ppi_df)
            
            return G
            
        except Exception as e:
            print(f"Error loading PPI reference: {e}")
            return None
    
    def analyze_ppi_reference_stats(self, ppi_network: 'nx.Graph') -> Dict:
        """
        Analyze basic statistics of the PPI reference network.
        
        Computes network properties, degree statistics, and confidence metrics
        for the loaded PPI reference network.
        
        Args:
            ppi_network: NetworkX graph with PPI interactions
            
        Returns:
            Dictionary containing PPI network statistics
        """
        if not NETWORKX_AVAILABLE or ppi_network is None:
            return {}
            
        print("Analyzing PPI reference network statistics...")
        
        stats = {
            'n_nodes': len(ppi_network.nodes()),
            'n_edges': len(ppi_network.edges()),
            'density': nx.density(ppi_network),
            'n_connected_components': len(list(nx.connected_components(ppi_network))),
            'largest_component_size': len(max(nx.connected_components(ppi_network), key=len)) if len(ppi_network.nodes()) > 0 else 0
        }
        
        degrees = dict(ppi_network.degree())
        if degrees:
            degree_values = list(degrees.values())
            stats.update({
                'mean_degree': np.mean(degree_values),
                'std_degree': np.std(degree_values),
                'max_degree': max(degree_values),
                'min_degree': min(degree_values)
            })
            
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            stats['top_hub_proteins'] = dict(top_hubs)
        
        edge_confidences = []
        for u, v, data in ppi_network.edges(data=True):
            if 'confidence' in data:
                edge_confidences.append(data['confidence'])
        
        if edge_confidences:
            stats['confidence_stats'] = {
                'mean_confidence': np.mean(edge_confidences),
                'std_confidence': np.std(edge_confidences),
                'min_confidence': min(edge_confidences),
                'max_confidence': max(edge_confidences)
            }
        
        print(f"  PPI network stats: {stats['n_nodes']} nodes, {stats['n_edges']} edges, density = {stats['density']:.4f}")
        
        return stats
    
    def compare_network_with_ppi(self, importance_network: 'nx.Graph', ppi_network: 'nx.Graph',
                                direction_name: str) -> Dict:
        """
        Compare an importance network with PPI reference for validation.
        
        Analyzes overlap between importance-derived edges and known PPI interactions,
        identifies novel connections, and performs enrichment analysis.
        
        Args:
            importance_network: Network built from importance matrix
            ppi_network: Reference PPI network  
            direction_name: Description of the importance network direction
            
        Returns:
            Dictionary with comprehensive comparison statistics
        """
        if not NETWORKX_AVAILABLE or importance_network is None or ppi_network is None:
            return {}
            
        print(f"Comparing {direction_name} network with PPI reference...")
        
        print(f"  Importance network: {len(importance_network.nodes())} nodes, {len(importance_network.edges())} edges")
        print(f"  PPI network: {len(ppi_network.nodes())} nodes, {len(ppi_network.edges())} edges")
        
        comparison = {
            'direction': direction_name,
            'importance_stats': {
                'n_nodes': len(importance_network.nodes()),
                'n_edges': len(importance_network.edges())
            },
            'ppi_stats': {
                'n_nodes': len(ppi_network.nodes()),
                'n_edges': len(ppi_network.edges())
            },
            'overlap_analysis': {},
            'novel_connections': {},
            'ppi_validation': {},
            'enrichment_analysis': {}
        }
        
        imp_nodes = set(importance_network.nodes())
        ppi_nodes = set(ppi_network.nodes())
        
        node_intersection = imp_nodes & ppi_nodes
        node_union = imp_nodes | ppi_nodes
        
        print(f"  Node intersection: {len(node_intersection)} nodes")
        print(f"  Example common nodes: {list(node_intersection)[:10]}")
        
        comparison['overlap_analysis']['nodes'] = {
            'importance_only': len(imp_nodes - ppi_nodes),
            'ppi_only': len(ppi_nodes - imp_nodes),
            'intersection': len(node_intersection),
            'union': len(node_union),
            'jaccard_similarity': len(node_intersection) / len(node_union) if len(node_union) > 0 else 0
        }
        
        imp_edges = set()
        for u, v in importance_network.edges():
            edge = tuple(sorted([u, v]))
            imp_edges.add(edge)
        
        ppi_edges = set()
        for u, v in ppi_network.edges():
            edge = tuple(sorted([u, v]))
            ppi_edges.add(edge)
        
        edge_intersection = imp_edges & ppi_edges
        edge_union = imp_edges | ppi_edges
        
        print(f"  Edge intersection: {len(edge_intersection)} edges")
        print(f"  Example common edges: {list(edge_intersection)[:5]}")
        
        comparison['overlap_analysis']['edges'] = {
            'importance_only': len(imp_edges - ppi_edges),
            'ppi_only': len(ppi_edges - imp_edges),
            'intersection': len(edge_intersection),
            'union': len(edge_union),
            'jaccard_similarity': len(edge_intersection) / len(edge_union) if len(edge_union) > 0 else 0
        }
        
        novel_edges = imp_edges - ppi_edges
        comparison['novel_connections'] = {
            'count': len(novel_edges),
            'fraction_of_importance_edges': len(novel_edges) / len(imp_edges) if len(imp_edges) > 0 else 0,
            'examples': list(novel_edges)[:20]
        }
        
        validated_edges = edge_intersection
        comparison['ppi_validation'] = {
            'count': len(validated_edges),
            'fraction_of_ppi_edges': len(validated_edges) / len(ppi_edges) if len(ppi_edges) > 0 else 0,
            'examples': list(validated_edges)[:20]
        }
        
        if len(node_intersection) >= 3:
            n_overlapping_nodes = len(node_intersection)
            max_possible_edges = n_overlapping_nodes * (n_overlapping_nodes - 1) // 2
            
            imp_edges_in_overlap = 0
            ppi_edges_in_overlap = 0
            overlap_edges_in_both = 0
            
            for u, v in edge_intersection:
                if u in node_intersection and v in node_intersection:
                    overlap_edges_in_both += 1
            
            for u, v in imp_edges:
                if u in node_intersection and v in node_intersection:
                    imp_edges_in_overlap += 1
                    
            for u, v in ppi_edges:
                if u in node_intersection and v in node_intersection:
                    ppi_edges_in_overlap += 1
            
            if max_possible_edges > 0 and imp_edges_in_overlap > 0 and ppi_edges_in_overlap > 0:
                try:
                    from scipy.stats import hypergeom
                    p_value = hypergeom.sf(overlap_edges_in_both - 1, max_possible_edges, 
                                         ppi_edges_in_overlap, imp_edges_in_overlap)
                    
                    comparison['enrichment_analysis'] = {
                        'hypergeometric_p_value': p_value,
                        'max_possible_edges': max_possible_edges,
                        'importance_edges_in_overlap': imp_edges_in_overlap,
                        'ppi_edges_in_overlap': ppi_edges_in_overlap,
                        'observed_overlap': overlap_edges_in_both,
                        'expected_overlap': (imp_edges_in_overlap * ppi_edges_in_overlap) / max_possible_edges,
                        'enrichment_significant': p_value < 0.05
                    }
                except ImportError:
                    print("  Warning: scipy not available for hypergeometric test")
                except Exception as e:
                    print(f"  Warning: Could not perform hypergeometric test: {e}")
        
        print(f"  {direction_name} vs PPI comparison:")
        print(f"    Node overlap: {comparison['overlap_analysis']['nodes']['intersection']}/{len(node_union)} (Jaccard: {comparison['overlap_analysis']['nodes']['jaccard_similarity']:.3f})")
        print(f"    Edge overlap: {comparison['overlap_analysis']['edges']['intersection']}/{len(edge_union)} (Jaccard: {comparison['overlap_analysis']['edges']['jaccard_similarity']:.3f})")
        print(f"    Novel connections: {comparison['novel_connections']['count']} ({comparison['novel_connections']['fraction_of_importance_edges']:.1%} of importance edges)")
        print(f"    PPI validation: {comparison['ppi_validation']['count']} ({comparison['ppi_validation']['fraction_of_ppi_edges']:.1%} of PPI edges)")
        
        return comparison
    
    def save_figure(self, fig: plt.Figure, name: str, **kwargs):
        """
        Save figure with standardized naming and formats.
        
        Saves the figure in both PNG (high DPI) and PDF formats with timestamp.
        
        Args:
            fig: Matplotlib figure object to save
            name: Base name for the figure file
            **kwargs: Additional arguments passed to savefig()
            
        Returns:
            Path to the saved PNG file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        png_path = self.output_dir / "figures" / f"{name}_{timestamp}.png"
        pdf_path = self.output_dir / "figures" / f"{name}_{timestamp}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', **kwargs)
        fig.savefig(pdf_path, bbox_inches='tight', **kwargs)
        
        print(f"  Saved: {png_path}")
        return png_path
    
    def plot_rank_consistency_overview_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_rank_consistency_overview_original(self, data: CorrelationAnalysisData):
        """Plot overview of rank consistency analysis"""
        print("Generating rank consistency overview...")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot for A→B if available
        if data.importance_a_to_b is not None and data.rank_consistency_a_to_b is not None:
            rank_stats = data.rank_consistency_a_to_b['rank_stats']
            
            # Top left: Mean rank vs rank variance
            scatter = axes[0, 0].scatter(rank_stats['mean_rank'], rank_stats['rank_variance'], 
                                       alpha=0.6, s=10, color=COLORS['primary'])
            axes[0, 0].set_xlabel('Mean Rank (lower = more important)')
            axes[0, 0].set_ylabel('Rank Variance (higher = less consistent)')
            axes[0, 0].set_title(f'{data.platform_a_name} → {data.platform_b_name}: Consistency vs Importance')
            
            # Highlight interesting features
            consistent_important = rank_stats[(rank_stats['mean_rank'] < rank_stats['mean_rank'].quantile(0.25)) & 
                                            (rank_stats['rank_variance'] < rank_stats['rank_variance'].quantile(0.25))]
            if len(consistent_important) > 0:
                axes[0, 0].scatter(consistent_important['mean_rank'], consistent_important['rank_variance'], 
                                 color=COLORS['accent'], s=10, label='Consistently Important', alpha=0.8)
            
            inconsistent = rank_stats[rank_stats['rank_variance'] > rank_stats['rank_variance'].quantile(0.9)]
            if len(inconsistent) > 0:
                axes[0, 0].scatter(inconsistent['mean_rank'], inconsistent['rank_variance'], 
                                 color=COLORS['warning'], s=10, label='Highly Variable', alpha=0.8)
            
            axes[0, 0].legend()
            
            # Top right: Distribution of times being top feature
            top_counts = data.rank_consistency_a_to_b['top_feature_counts']
            axes[0, 1].hist(top_counts.values, bins=20, alpha=0.7, color=COLORS['primary'], edgecolor='black')
            axes[0, 1].set_xlabel('Times Being Top Feature')
            axes[0, 1].set_ylabel('Number of Input Features')
            axes[0, 1].set_title(f'{data.platform_a_name} → {data.platform_b_name}: Top Feature Distribution')
            axes[0, 1].axvline(top_counts.mean(), color=COLORS['warning'], linestyle='--', 
                             label=f'Mean: {top_counts.mean():.1f}')
            axes[0, 1].legend()
        
        # Plot for B→A if available
        if data.importance_b_to_a is not None and data.rank_consistency_b_to_a is not None:
            rank_stats = data.rank_consistency_b_to_a['rank_stats']
            
            # Bottom left: Mean rank vs rank variance
            scatter = axes[1, 0].scatter(rank_stats['mean_rank'], rank_stats['rank_variance'], 
                                       alpha=0.6, s=10, color=COLORS['secondary'])
            axes[1, 0].set_xlabel('Mean Rank (lower = more important)')
            axes[1, 0].set_ylabel('Rank Variance (higher = less consistent)')
            axes[1, 0].set_title(f'{data.platform_b_name} → {data.platform_a_name}: Consistency vs Importance')
            
            # Highlight interesting features
            consistent_important = rank_stats[(rank_stats['mean_rank'] < rank_stats['mean_rank'].quantile(0.25)) & 
                                            (rank_stats['rank_variance'] < rank_stats['rank_variance'].quantile(0.25))]
            if len(consistent_important) > 0:
                axes[1, 0].scatter(consistent_important['mean_rank'], consistent_important['rank_variance'], 
                                 color=COLORS['accent'], s=10, label='Consistently Important', alpha=0.8)
            
            inconsistent = rank_stats[rank_stats['rank_variance'] > rank_stats['rank_variance'].quantile(0.9)]
            if len(inconsistent) > 0:
                axes[1, 0].scatter(inconsistent['mean_rank'], inconsistent['rank_variance'], 
                                 color=COLORS['warning'], s=10, label='Highly Variable', alpha=0.8)
            
            axes[1, 0].legend()
            
            # Bottom right: Distribution of times being top feature
            top_counts = data.rank_consistency_b_to_a['top_feature_counts']
            axes[1, 1].hist(top_counts.values, bins=20, alpha=0.7, color=COLORS['secondary'], edgecolor='black')
            axes[1, 1].set_xlabel('Times Being Top Feature')
            axes[1, 1].set_ylabel('Number of Input Features')
            axes[1, 1].set_title(f'{data.platform_b_name} → {data.platform_a_name}: Top Feature Distribution')
            axes[1, 1].axvline(top_counts.mean(), color=COLORS['warning'], linestyle='--', 
                             label=f'Mean: {top_counts.mean():.1f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        return self.save_figure(fig, "rank_consistency_overview")
    
    def plot_rank_distribution_analysis_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_rank_distribution_analysis_original(self, data: CorrelationAnalysisData):
        """Plot detailed rank distribution analysis"""
        print("Generating rank distribution analysis...")
        
        if data.importance_a_to_b is None and data.importance_b_to_a is None:
            print("  No importance matrices available")
            return None
        
        # Determine subplot layout
        n_plots = sum([1 for x in [data.importance_a_to_b, data.importance_b_to_a] if x is not None])
        
        fig, axes = plt.subplots(n_plots, 2, figsize=(10, 5*n_plots))
        if n_plots == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Analyze A→B
        if data.importance_a_to_b is not None and data.rank_consistency_a_to_b is not None:
            ranks = data.rank_consistency_a_to_b['ranks']
            rank_stats = data.rank_consistency_a_to_b['rank_stats']
            
            # Left: Clustered rank distribution heatmap for ALL features
            # Matrix interpretation:
            # - Rows = input features, Columns = output features  
            # - Value at [i,j] = rank of input feature i for predicting output feature j
            # - Rank 1 = most important input for that output, Rank n = least important
            # - For each column (output feature), all input features are ranked 1 to n
            
            try:
                # Cluster rows (input features) by their rank patterns across output features
                row_distances = pdist(ranks.values, metric='euclidean')
                row_linkage = linkage(row_distances, method='ward')
                row_order = leaves_list(row_linkage)
                
                # Cluster columns (output features) by their rank patterns across input features  
                col_distances = pdist(ranks.T.values, metric='euclidean')
                col_linkage = linkage(col_distances, method='ward')
                col_order = leaves_list(col_linkage)
                
                # Reorder matrix - keep all features but cluster them
                clustered_ranks = ranks.iloc[row_order, col_order]
                
                # Limit display for readability if too many features
                if clustered_ranks.shape[0] > 100:
                    # Show top 100 most variable features for visualization
                    most_variable_indices = rank_stats.nlargest(100, 'rank_variance').index
                    clustered_ranks_subset = clustered_ranks.loc[most_variable_indices]
                    display_title = f'Clustered Rank Patterns (Top 100 Variable of {ranks.shape[0]} features)'
                else:
                    clustered_ranks_subset = clustered_ranks
                    display_title = f'Clustered Rank Patterns (All {ranks.shape[0]} features)'
                
                # Create clustered heatmap
                sns.heatmap(clustered_ranks_subset, ax=axes[plot_idx, 0], cmap='viridis_r', 
                           cbar_kws={'label': 'Rank (1=most important)'},
                           xticklabels=False, yticklabels=False)
                axes[plot_idx, 0].set_title(f'{data.platform_a_name} → {data.platform_b_name}: {display_title}')
                axes[plot_idx, 0].set_xlabel('Output Features (clustered)')
                axes[plot_idx, 0].set_ylabel('Input Features (clustered by rank pattern)')
                
            except Exception as e:
                print(f"    Clustering failed, using original order: {e}")
                # Fallback to original order with subset for readability
                if ranks.shape[0] > 50:
                    most_variable = rank_stats.nlargest(50, 'rank_variance')
                    display_ranks = ranks.loc[most_variable.index]
                    display_title = f'Rank Patterns (Top 50 Variable of {ranks.shape[0]} features)'
                else:
                    display_ranks = ranks
                    display_title = f'Rank Patterns (All {ranks.shape[0]} features)'
                    
                sns.heatmap(display_ranks, ax=axes[plot_idx, 0], cmap='viridis_r', 
                           cbar_kws={'label': 'Rank (1=most important)'},
                           xticklabels=False, yticklabels=False)
                axes[plot_idx, 0].set_title(f'{data.platform_a_name} → {data.platform_b_name}: {display_title}')
                axes[plot_idx, 0].set_xlabel('Output Features')
                axes[plot_idx, 0].set_ylabel('Input Features')
            
            # Right: Box plot of rank distributions for top features
            top_features = rank_stats.nsmallest(10, 'mean_rank').index
            rank_data_for_plot = [ranks.loc[feature].values for feature in top_features]
            
            bp = axes[plot_idx, 1].boxplot(rank_data_for_plot, labels=[f[:15] + '...' if len(f) > 15 else f 
                                                                      for f in top_features], 
                                          patch_artist=True)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['primary'])
                patch.set_alpha(0.7)
            
            axes[plot_idx, 1].set_title(f'{data.platform_a_name} → {data.platform_b_name}: '
                                       f'Rank Distributions (Top 10 Features)')
            axes[plot_idx, 1].set_xlabel('Input Features')
            axes[plot_idx, 1].set_ylabel('Rank Distribution')
            axes[plot_idx, 1].tick_params(axis='x', rotation=45)
            
            plot_idx += 1
        
        # Analyze B→A
        if data.importance_b_to_a is not None and data.rank_consistency_b_to_a is not None:
            ranks = data.rank_consistency_b_to_a['ranks']
            rank_stats = data.rank_consistency_b_to_a['rank_stats']
            
            # Left: Clustered rank distribution heatmap for ALL features
            # Matrix interpretation:
            # - Rows = input features, Columns = output features  
            # - Value at [i,j] = rank of input feature i for predicting output feature j
            # - Rank 1 = most important input for that output, Rank n = least important
            # - For each column (output feature), all input features are ranked 1 to n
            
            try:
                # Cluster rows (input features) by their rank patterns across output features
                row_distances = pdist(ranks.values, metric='euclidean')
                row_linkage = linkage(row_distances, method='ward')
                row_order = leaves_list(row_linkage)
                
                # Cluster columns (output features) by their rank patterns across input features  
                col_distances = pdist(ranks.T.values, metric='euclidean')
                col_linkage = linkage(col_distances, method='ward')
                col_order = leaves_list(col_linkage)
                
                # Reorder matrix - keep all features but cluster them
                clustered_ranks = ranks.iloc[row_order, col_order]
                
                # Limit display for readability if too many features
                if clustered_ranks.shape[0] > 100:
                    # Show top 100 most variable features for visualization
                    most_variable_indices = rank_stats.nlargest(100, 'rank_variance').index
                    clustered_ranks_subset = clustered_ranks.loc[most_variable_indices]
                    display_title = f'Clustered Rank Patterns (Top 100 Variable of {ranks.shape[0]} features)'
                else:
                    clustered_ranks_subset = clustered_ranks
                    display_title = f'Clustered Rank Patterns (All {ranks.shape[0]} features)'
                
                # Create clustered heatmap
                sns.heatmap(clustered_ranks_subset, ax=axes[plot_idx, 0], cmap='viridis_r', 
                           cbar_kws={'label': 'Rank (1=most important)'},
                           xticklabels=False, yticklabels=False)
                axes[plot_idx, 0].set_title(f'{data.platform_b_name} → {data.platform_a_name}: {display_title}')
                axes[plot_idx, 0].set_xlabel('Output Features (clustered)')
                axes[plot_idx, 0].set_ylabel('Input Features (clustered by rank pattern)')
                
            except Exception as e:
                print(f"    Clustering failed, using original order: {e}")
                # Fallback to original order with subset for readability
                if ranks.shape[0] > 50:
                    most_variable = rank_stats.nlargest(50, 'rank_variance')
                    display_ranks = ranks.loc[most_variable.index]
                    display_title = f'Rank Patterns (Top 50 Variable of {ranks.shape[0]} features)'
                else:
                    display_ranks = ranks
                    display_title = f'Rank Patterns (All {ranks.shape[0]} features)'
                    
                sns.heatmap(display_ranks, ax=axes[plot_idx, 0], cmap='viridis_r', 
                           cbar_kws={'label': 'Rank (1=most important)'},
                           xticklabels=False, yticklabels=False)
                axes[plot_idx, 0].set_title(f'{data.platform_b_name} → {data.platform_a_name}: {display_title}')
                axes[plot_idx, 0].set_xlabel('Output Features')
                axes[plot_idx, 0].set_ylabel('Input Features')
            
            # Right: Box plot of rank distributions for top features
            top_features = rank_stats.nsmallest(10, 'mean_rank').index
            rank_data_for_plot = [ranks.loc[feature].values for feature in top_features]
            
            bp = axes[plot_idx, 1].boxplot(rank_data_for_plot, labels=[f[:15] + '...' if len(f) > 15 else f 
                                                                      for f in top_features], 
                                          patch_artist=True)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['secondary'])
                patch.set_alpha(0.7)
            
            axes[plot_idx, 1].set_title(f'{data.platform_b_name} → {data.platform_a_name}: '
                                       f'Rank Distributions (Top 10 Features)')
            axes[plot_idx, 1].set_xlabel('Input Features')
            axes[plot_idx, 1].set_ylabel('Rank Distribution')
            axes[plot_idx, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return self.save_figure(fig, "rank_distribution_analysis")
    
    def plot_importance_matrix_heatmaps_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_importance_matrix_heatmaps_original(self, data: CorrelationAnalysisData):
        """Plot importance matrix heatmaps with clustering"""
        print("Generating clustered importance matrix heatmaps...")
        
        if data.importance_a_to_b is None and data.importance_b_to_a is None:
            print("  No importance matrices available")
            return None
        
        # Determine subplot layout
        n_plots = sum([1 for x in [data.importance_a_to_b, data.importance_b_to_a] if x is not None])
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        def cluster_and_plot(matrix, ax, title):
            """Cluster and plot a matrix"""
            # Take top features for visualization to reduce computation
            overall_importance = matrix.mean(axis=1)
            top_inputs = overall_importance.nlargest(100).index
            matrix_subset = matrix.loc[top_inputs]
            
            # Also limit output features if too many
            if matrix_subset.shape[1] > 200:
                output_importance = matrix_subset.mean(axis=0)
                top_outputs = output_importance.nlargest(200).index
                matrix_subset = matrix_subset[top_outputs]
            
            try:
                # Cluster rows (input features)
                row_distances = pdist(matrix_subset.values, metric='euclidean')
                row_linkage = linkage(row_distances, method='ward')
                row_order = leaves_list(row_linkage)
                
                # Cluster columns (output features) 
                col_distances = pdist(matrix_subset.T.values, metric='euclidean')
                col_linkage = linkage(col_distances, method='ward')
                col_order = leaves_list(col_linkage)
                
                # Reorder matrix
                clustered_matrix = matrix_subset.iloc[row_order, col_order]
                
                # Create heatmap
                sns.heatmap(clustered_matrix, ax=ax, cmap='viridis', 
                           cbar_kws={'label': 'Feature Importance'},
                           xticklabels=False, yticklabels=False)
                ax.set_title(f'{title}\nClustered Importance Matrix '
                           f'({matrix_subset.shape[0]} × {matrix_subset.shape[1]})')
                ax.set_xlabel('Output Features (clustered)')
                ax.set_ylabel('Input Features (clustered)')
                
            except Exception as e:
                print(f"  Clustering failed, using original order: {e}")
                # Fallback to original order
                sns.heatmap(matrix_subset, ax=ax, cmap='viridis', 
                           cbar_kws={'label': 'Feature Importance'},
                           xticklabels=False, yticklabels=False)
                ax.set_title(f'{title}\nImportance Matrix '
                           f'({matrix_subset.shape[0]} × {matrix_subset.shape[1]})')
                ax.set_xlabel('Output Features')
                ax.set_ylabel('Input Features')
        
        # A→B heatmap
        if data.importance_a_to_b is not None:
            cluster_and_plot(data.importance_a_to_b, axes[plot_idx], 
                           f'{data.platform_a_name} → {data.platform_b_name}')
            plot_idx += 1
        
        # B→A heatmap
        if data.importance_b_to_a is not None:
            cluster_and_plot(data.importance_b_to_a, axes[plot_idx], 
                           f'{data.platform_b_name} → {data.platform_a_name}')
        
        plt.tight_layout()
        return self.save_figure(fig, "importance_matrix_heatmaps_clustered")
    
    def plot_overlapping_features_analysis_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_overlapping_features_analysis_original(self, data: CorrelationAnalysisData):
        """Plot matched analysis for overlapping features"""
        print("Generating overlapping features analysis...")
        
        if not data.overlapping_features or len(data.overlapping_features) < 5:
            print("  Insufficient overlapping features for matched analysis")
            return None
        
        # Randomly select 100 features for better visualization
        import random
        if len(data.overlapping_features) > 100:
            # Set seed for reproducibility
            random.seed(42)
            overlapping_subset = random.sample(data.overlapping_features, 100)
            print(f"  Randomly selected 100 features from {len(data.overlapping_features)} overlapping features")
        else:
            overlapping_subset = data.overlapping_features
            print(f"  Using all {len(data.overlapping_features)} overlapping features")
        
        # Calculate number of plots
        n_plots = 0
        if data.importance_a_to_b is not None:
            n_plots += 1
        if data.importance_b_to_a is not None:
            n_plots += 1
        
        if n_plots == 0:
            return None
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        def plot_matched_matrix(importance_matrix, overlapping_features, ax, title):
            """Plot importance matrix with matched input/output features"""
            # Filter for overlapping features that exist in both input and output
            available_input = [f for f in overlapping_features if f in importance_matrix.index]
            available_output = [f for f in overlapping_features if f in importance_matrix.columns]
            
            if len(available_input) < 5 or len(available_output) < 5:
                ax.text(0.5, 0.5, 'Insufficient matched features', ha='center', va='center')
                ax.set_title(title)
                return
            
            # Use same ordering for both input and output
            common_features = list(set(available_input) & set(available_output))
            if len(common_features) < 5:
                # Use separate lists but try to match order - use ALL available features
                matched_input = available_input
                matched_output = available_output
            else:
                # Use common features for both dimensions - use ALL common features
                common_features_sorted = sorted(common_features)
                matched_input = common_features_sorted
                matched_output = common_features_sorted
            
            # Extract submatrix
            sub_matrix = importance_matrix.loc[matched_input, matched_output]
            
            # Create heatmap
            im = ax.imshow(sub_matrix.values, cmap='viridis', aspect='auto')
            
            # Set ticks but remove labels to avoid clutter with many features
            ax.set_xticks(range(len(matched_output)))
            ax.set_yticks(range(len(matched_input)))
            ax.set_xticklabels([])  # Remove x-axis labels
            ax.set_yticklabels([])  # Remove y-axis labels
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Feature Importance')
            
            # Highlight diagonal if input/output features match
            if matched_input == matched_output:
                # Comment out the red box indicator for self-importance (can be enabled later if needed)
                # for i in range(len(matched_input)):
                #     ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, 
                #                              fill=False, edgecolor=COLORS['primary'], lw=0.5))
                ax.set_title(f'{title}\nMatched Features Analysis ({len(matched_input)} features)')
            else:
                ax.set_title(f'{title}\nOverlapping Features Analysis\n({len(matched_input)} input × {len(matched_output)} output)')
            
            ax.set_xlabel('Output Features')
            ax.set_ylabel('Input Features')
        
        # Plot A→B
        if data.importance_a_to_b is not None:
            plot_matched_matrix(data.importance_a_to_b, overlapping_subset, axes[plot_idx],
                              f'{data.platform_a_name} → {data.platform_b_name}')
            plot_idx += 1
        
        # Plot B→A
        if data.importance_b_to_a is not None:
            plot_matched_matrix(data.importance_b_to_a, overlapping_subset, axes[plot_idx],
                              f'{data.platform_b_name} → {data.platform_a_name}')
        
        plt.tight_layout()
        return self.save_figure(fig, "overlapping_features_analysis")
    
    def plot_feature_specialization_analysis_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_feature_specialization_analysis_original(self, data: CorrelationAnalysisData):
        """Plot analysis of feature specialization vs generalization"""
        print("Generating feature specialization analysis...")
        
        if data.importance_a_to_b is None and data.importance_b_to_a is None:
            print("  No importance matrices available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        def analyze_specialization(importance_matrix, rank_consistency, ax_left, ax_right, title_prefix):
            if importance_matrix is None or rank_consistency is None:
                ax_left.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax_right.text(0.5, 0.5, 'No data available', ha='center', va='center')
                return
            
            # Calculate specialization metrics
            # Gini coefficient for each input feature (how concentrated its importance is)
            def gini_coefficient(x):
                # Sort the values
                sorted_x = np.sort(x)
                n = len(x)
                index = np.arange(1, n + 1)
                return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n
            
            gini_scores = {}
            max_importance_ratios = {}
            
            for input_feature in importance_matrix.index:
                importances = importance_matrix.loc[input_feature].values
                gini_scores[input_feature] = gini_coefficient(importances)
                max_importance_ratios[input_feature] = importances.max() / importances.mean()
            
            rank_stats = rank_consistency['rank_stats']
            
            # Left plot: Gini coefficient vs mean rank
            gini_values = [gini_scores[f] for f in rank_stats.index]
            
            scatter = ax_left.scatter(rank_stats['mean_rank'], gini_values, alpha=0.6, s=10)
            ax_left.set_xlabel('Mean Rank (lower = more important)')
            ax_left.set_ylabel('Gini Coefficient (higher = more specialized)')
            ax_left.set_title(f'{title_prefix}: Specialization vs Importance')
            
            # Highlight interesting quadrants
            median_rank = rank_stats['mean_rank'].median()
            median_gini = np.median(gini_values)
            
            # Specialized and important (low rank, high gini)
            specialized_important = rank_stats[(rank_stats['mean_rank'] < median_rank) & 
                                             (np.array(gini_values) > median_gini)]
            if len(specialized_important) > 0:
                specialist_indices = [list(rank_stats.index).index(f) for f in specialized_important.index]
                ax_left.scatter([rank_stats.loc[f, 'mean_rank'] for f in specialized_important.index],
                              [gini_values[i] for i in specialist_indices],
                              color=COLORS['accent'], s=10, label='Specialist Important', alpha=0.8)
            
            # Generalist and important (low rank, low gini)
            generalist_important = rank_stats[(rank_stats['mean_rank'] < median_rank) & 
                                             (np.array(gini_values) < median_gini)]
            if len(generalist_important) > 0:
                generalist_indices = [list(rank_stats.index).index(f) for f in generalist_important.index]
                ax_left.scatter([rank_stats.loc[f, 'mean_rank'] for f in generalist_important.index],
                              [gini_values[i] for i in generalist_indices],
                              color=COLORS['info'], s=10, label='Generalist Important', alpha=0.8)
            
            ax_left.axvline(median_rank, color=COLORS['info'], linestyle='--', alpha=0.5)
            ax_left.axhline(median_gini, color=COLORS['info'], linestyle='--', alpha=0.5)
            ax_left.legend()
            
            # Right plot: Distribution of max/mean importance ratios
            max_ratios = [max_importance_ratios[f] for f in rank_stats.index]
            
            ax_right.hist(max_ratios, bins=30, alpha=0.7, edgecolor='black')
            ax_right.set_xlabel('Max/Mean Importance Ratio')
            ax_right.set_ylabel('Number of Input Features')
            ax_right.set_title(f'{title_prefix}: Importance Concentration')
            ax_right.axvline(np.median(max_ratios), color=COLORS['warning'], linestyle='--', 
                           label=f'Median: {np.median(max_ratios):.2f}')
            ax_right.legend()
        
        # Analyze A→B
        if data.importance_a_to_b is not None:
            analyze_specialization(data.importance_a_to_b, data.rank_consistency_a_to_b,
                                 axes[0, 0], axes[0, 1], 
                                 f'{data.platform_a_name} → {data.platform_b_name}')
        
        # Analyze B→A
        if data.importance_b_to_a is not None:
            analyze_specialization(data.importance_b_to_a, data.rank_consistency_b_to_a,
                                 axes[1, 0], axes[1, 1], 
                                 f'{data.platform_b_name} → {data.platform_a_name}')
        
        plt.tight_layout()
        return self.save_figure(fig, "feature_specialization_analysis")
    
    def plot_self_feature_importance_analysis_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_self_feature_importance_analysis_original(self, data: CorrelationAnalysisData):
        """Plot analysis of self-feature importance vs performance"""
        print("Generating self-feature importance analysis...")
        
        if data.importance_a_to_b is None and data.importance_b_to_a is None:
            print("  No importance matrices available")
            return None
        
        # Calculate self-feature analyses
        self_analysis_a_to_b = None
        self_analysis_b_to_a = None
        
        if data.importance_a_to_b is not None:
            self_analysis_a_to_b = self.analyze_self_feature_importance(data.importance_a_to_b)
        
        if data.importance_b_to_a is not None:
            self_analysis_b_to_a = self.analyze_self_feature_importance(data.importance_b_to_a)
        
        # Determine subplot layout
        n_plots = sum([1 for x in [self_analysis_a_to_b, self_analysis_b_to_a] 
                      if x is not None and x['overlapping_features']])
        
        if n_plots == 0:
            print("  No overlapping features found for self-importance analysis")
            return None
        
        fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
        if n_plots == 1:
            axes = axes.reshape(-1, 1)
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        
        def plot_self_analysis(analysis, ax_top, ax_bottom, title):
            if not analysis or not analysis['overlapping_features']:
                ax_top.text(0.5, 0.5, 'No overlapping features', ha='center', va='center')
                ax_bottom.text(0.5, 0.5, 'No overlapping features', ha='center', va='center')
                return
            
            # Top plot: Distribution of self-importance scores vs cross-importance scores
            self_scores = list(analysis['self_importance_scores'].values())
            ranks = list(analysis['self_importance_ranks'].values())
            
            # Create violin plot of self-importance scores
            parts = ax_top.violinplot([self_scores], positions=[1], widths=0.6, showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(COLORS['accent'])
            parts['bodies'][0].set_alpha(0.7)
            
            # Add reference lines and statistics
            if analysis['avg_self_importance'] is not None:
                ax_top.axhline(analysis['avg_self_importance'], color=COLORS['warning'], 
                             linestyle='-', linewidth=2, label=f'Avg Self: {analysis["avg_self_importance"]:.6f}')
            if analysis['avg_cross_importance'] is not None:
                ax_top.axhline(analysis['avg_cross_importance'], color=COLORS['info'], 
                             linestyle='--', linewidth=2, label=f'Avg Cross: {analysis["avg_cross_importance"]:.6f}')
            
            # Add statistical annotations
            median_self = np.median(self_scores)
            q75_self = np.percentile(self_scores, 75)
            q25_self = np.percentile(self_scores, 25)
            
            ax_top.text(1.4, median_self, f'Median: {median_self:.4f}', ha='left', va='center')
            ax_top.text(1.4, q75_self, f'Q75: {q75_self:.4f}', ha='left', va='center', fontsize=9)
            ax_top.text(1.4, q25_self, f'Q25: {q25_self:.4f}', ha='left', va='center', fontsize=9)
            
            ax_top.set_xlim(0.5, 2.2)
            ax_top.set_xticks([1])
            ax_top.set_xticklabels(['Self-Importance\nScores'])
            ax_top.set_ylabel('Importance Score')
            ax_top.set_title(f'{title}: Self-Feature Importance Distribution')
            ax_top.legend(loc='upper right')
            
            # Bottom plot: Distribution of self-importance ranks
            parts = ax_bottom.violinplot([ranks], positions=[1], widths=0.6, showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor(COLORS['primary'])
            parts['bodies'][0].set_alpha(0.7)
            
            # Add statistical annotations
            median_rank = np.median(ranks)
            q25_rank = np.percentile(ranks, 25)
            q75_rank = np.percentile(ranks, 75)
            
            ax_bottom.text(1.4, median_rank, f'Median: {median_rank:.1f}', ha='left', va='center')
            ax_bottom.text(1.4, q25_rank, f'Q25: {q25_rank:.1f}', ha='left', va='center', fontsize=9)
            ax_bottom.text(1.4, q75_rank, f'Q75: {q75_rank:.1f}', ha='left', va='center', fontsize=9)
            
            # Add reference lines for good/poor ranks
            ax_bottom.axhline(1, color=COLORS['success'], linestyle='-', alpha=0.5, label='Best Rank')
            ax_bottom.axhline(10, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Top 10')
            ax_bottom.axhline(50, color=COLORS['danger'], linestyle=':', alpha=0.5, label='Top 50')
            
            ax_bottom.set_xlim(0.5, 2.2)
            ax_bottom.set_xticks([1])
            ax_bottom.set_xticklabels(['Self-Importance\nRanks'])
            ax_bottom.set_ylabel('Rank (lower = better)')
            ax_bottom.set_title(f'{title}: Self-Feature Importance Rank Distribution')
            ax_bottom.legend(loc='upper right')
            
            # Add summary statistics text
            n_top_10 = sum(1 for r in ranks if r <= 10)
            n_top_50 = sum(1 for r in ranks if r <= 50)
            pct_top_10 = (n_top_10 / len(ranks)) * 100
            pct_top_50 = (n_top_50 / len(ranks)) * 100
            
            stats_text = f'Features in top 10: {n_top_10} ({pct_top_10:.1f}%)\nFeatures in top 50: {n_top_50} ({pct_top_50:.1f}%)'
            ax_bottom.text(0.02, 0.98, stats_text, transform=ax_bottom.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # A→B analysis
        if self_analysis_a_to_b and self_analysis_a_to_b['overlapping_features']:
            plot_self_analysis(self_analysis_a_to_b, axes[0, plot_idx], axes[1, plot_idx],
                             f'{data.platform_a_name} → {data.platform_b_name}')
            plot_idx += 1
        
        # B→A analysis
        if self_analysis_b_to_a and self_analysis_b_to_a['overlapping_features']:
            plot_self_analysis(self_analysis_b_to_a, axes[0, plot_idx], axes[1, plot_idx],
                             f'{data.platform_b_name} → {data.platform_a_name}')
        
        plt.tight_layout()
        return self.save_figure(fig, "self_feature_importance_analysis")
    
    def plot_self_importance_vs_performance_correlation_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_self_importance_vs_performance_correlation_original(self, data: CorrelationAnalysisData):
        """Plot correlation between self-feature importance rank and feature performance"""
        print("Generating self-importance vs performance correlation plot...")
        
        # First analyze the correlations
        correlation_results = self.analyze_self_importance_vs_performance(data)
        
        if not correlation_results:
            print("  No correlation data available")
            return None
        
        # Determine subplot layout
        valid_results = {k: v for k, v in correlation_results.items() 
                        if v and v.get('n_features', 0) >= 3}
        
        if not valid_results:
            print("  No valid correlation data for plotting")
            return None
        
        n_directions = len(valid_results)
        fig, axes = plt.subplots(2, n_directions, figsize=(5*n_directions, 10))
        
        if n_directions == 1:
            axes = axes.reshape(2, 1)
        
        # Use the global color scheme (already defined at module level)
        
        col_idx = 0
        for direction_key, result in valid_results.items():
            if result['n_features'] < 3:
                continue
                
            features_data = result['features_data']
            direction_name = result['direction']
            
            # Top plot: Rank vs Performance
            ax_rank = axes[0, col_idx]
            
            # Scatter plot
            scatter = ax_rank.scatter(features_data['self_importance_rank'], 
                                    features_data['performance_r'],
                                    alpha=0.7, s=10, color=COLORS['primary'],
                                    edgecolors='white', linewidth=1)
            
            # Add trend line
            if len(features_data) > 2:
                z = np.polyfit(features_data['self_importance_rank'], features_data['performance_r'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(features_data['self_importance_rank'].min(), 
                                    features_data['self_importance_rank'].max(), 100)
                ax_rank.plot(x_trend, p(x_trend), '--', color=COLORS['warning'], alpha=0.8, linewidth=2)
            
            # Add correlation statistics
            rank_corr_info = result['correlation_rank_performance']
            if rank_corr_info:
                corr_text = f"ρ = {rank_corr_info['correlation']:.3f}"
                if rank_corr_info['p_value'] < 0.05:
                    corr_text += f"\np = {rank_corr_info['p_value']:.3f}*"
                else:
                    corr_text += f"\np = {rank_corr_info['p_value']:.3f}"
                
                ax_rank.text(0.05, 0.95, corr_text, transform=ax_rank.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top', fontsize=10)
            
            ax_rank.set_xlabel('Self-Importance Rank\n(lower = more important)')
            ax_rank.set_ylabel('Feature Performance (r)')
            ax_rank.set_title(f'{direction_name}\nRank vs Performance')
            ax_rank.grid(True, alpha=0.3)
            
            # Label extreme points
            if len(features_data) > 0:
                # Find features with extreme ranks or performance
                worst_rank = features_data.loc[features_data['self_importance_rank'].idxmax()]
                best_rank = features_data.loc[features_data['self_importance_rank'].idxmin()]
                worst_perf = features_data.loc[features_data['performance_r'].idxmin()]
                best_perf = features_data.loc[features_data['performance_r'].idxmax()]
                
                # Label a few extreme points
                for point, label in [(worst_rank, 'Worst Rank'), (best_rank, 'Best Rank'), 
                                   (worst_perf, 'Worst Perf'), (best_perf, 'Best Perf')]:
                    if not point.isna().any():
                        ax_rank.annotate(f"{point['feature'][:8]}\n({label})", 
                                       (point['self_importance_rank'], point['performance_r']),
                                       xytext=(5, 5), textcoords='offset points', 
                                       fontsize=8, alpha=0.7,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
            
            # Bottom plot: Importance Score vs Performance
            ax_score = axes[1, col_idx]
            
            # Scatter plot
            scatter = ax_score.scatter(features_data['self_importance_score'], 
                                     features_data['performance_r'],
                                     alpha=0.7, s=10, color=COLORS['accent'],
                                     edgecolors='white', linewidth=1)
            
            # Add trend line
            if len(features_data) > 2:
                z = np.polyfit(features_data['self_importance_score'], features_data['performance_r'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(features_data['self_importance_score'].min(), 
                                    features_data['self_importance_score'].max(), 100)
                ax_score.plot(x_trend, p(x_trend), '--', color=COLORS['warning'], alpha=0.8, linewidth=2)
            
            # Add correlation statistics
            score_corr_info = result['correlation_score_performance']
            if score_corr_info:
                corr_text = f"r = {score_corr_info['correlation']:.3f}"
                if score_corr_info['p_value'] < 0.05:
                    corr_text += f"\np = {score_corr_info['p_value']:.3f}*"
                else:
                    corr_text += f"\np = {score_corr_info['p_value']:.3f}"
                
                ax_score.text(0.05, 0.95, corr_text, transform=ax_score.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            verticalalignment='top', fontsize=10)
            
            ax_score.set_xlabel('Self-Importance Score\n(higher = more important)')
            ax_score.set_ylabel('Feature Performance (r)')
            ax_score.set_title(f'{direction_name}\nScore vs Performance')
            ax_score.grid(True, alpha=0.3)
            
            # Label extreme points for score plot
            if len(features_data) > 0:
                # Find features with extreme scores or performance
                worst_score = features_data.loc[features_data['self_importance_score'].idxmin()]
                best_score = features_data.loc[features_data['self_importance_score'].idxmax()]
                
                # Label a few extreme points
                for point, label in [(worst_score, 'Worst Score'), (best_score, 'Best Score')]:
                    if not point.isna().any():
                        ax_score.annotate(f"{point['feature'][:8]}\n({label})", 
                                        (point['self_importance_score'], point['performance_r']),
                                        xytext=(5, 5), textcoords='offset points', 
                                        fontsize=8, alpha=0.7,
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
            
            col_idx += 1
        
        # Add figure-wide title and interpretation
        fig.suptitle('Self-Feature Importance vs Performance Correlation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add interpretation text at the bottom
        interpretation_text = []
        for direction_key, result in valid_results.items():
            if 'interpretation' in result:
                interpretation_text.append(f"{result['direction']}: {result['interpretation']}")
        
        if interpretation_text:
            fig.text(0.5, 0.02, '\n'.join(interpretation_text), 
                    ha='center', va='bottom', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.93)  # Make room for title and interpretation
        
        return self.save_figure(fig, "self_importance_vs_performance_correlation")

    def plot_self_importance_vs_raw_correlation_removed(self, data: CorrelationAnalysisData):
        # This function is for importance matrices, not correlation networks
        return None

    def plot_self_importance_vs_raw_correlation_original(self, data: CorrelationAnalysisData):
        """Plot correlation between self-feature importance and raw cross-platform correlation"""
        print("Generating self-importance vs raw cross-platform correlation plot...")

        # Ensure raw cross-platform correlation is computed
        if data.cross_platform_raw_correlation is None:
            data = self.compute_cross_platform_raw_correlation(data)

        correlation_results = self.analyze_self_importance_vs_raw_correlation(data)
        if not correlation_results:
            print("  No correlation data available")
            return None

        valid_results = {k: v for k, v in correlation_results.items() if v and v.get('n_features', 0) >= 3}
        if not valid_results:
            print("  No valid correlation data for plotting")
            return None

        n_directions = len(valid_results)
        fig, axes = plt.subplots(1, n_directions, figsize=(6*n_directions, 5))
        if n_directions == 1:
            axes = np.array([axes])

        col_idx = 0
        for _, result in valid_results.items():
            features_data = result['features_data']
            direction_name = result['direction']

            ax = axes[col_idx]
            x = features_data['performance_r']
            y = features_data['self_importance_score']

            ax.scatter(x, y, alpha=0.7, s=14, color=COLORS['accent'], edgecolors='white', linewidth=0.8)

            if len(features_data) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_trend, p(x_trend), '--', color=COLORS['warning'], alpha=0.9, linewidth=2)

            score_corr_info = result['correlation_score_performance']
            if score_corr_info:
                corr_text = f"r = {score_corr_info['correlation']:.3f}\n"
                corr_text += f"p = {score_corr_info['p_value']:.3f}{'*' if score_corr_info['p_value'] < 0.05 else ''}"
                ax.text(0.02, 0.98, corr_text, transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                        verticalalignment='top', fontsize=10)

            # Label outliers (extremes)
            try:
                rows_to_label = [
                    features_data.loc[x.idxmax()],
                    features_data.loc[x.idxmin()],
                    features_data.loc[y.idxmax()],
                    features_data.loc[y.idxmin()],
                ]
                seen = set()
                for row in rows_to_label:
                    feat = str(row['feature'])
                    if feat not in seen and not row.isna().any():
                        seen.add(feat)
                        ax.annotate(f"{feat[:12]}", (row['performance_r'], row['self_importance_score']),
                                    xytext=(6, 6), textcoords='offset points', fontsize=8, alpha=0.85,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))
            except Exception:
                pass

            ax.set_xlabel('Cross-Platform Raw Correlation (r)')
            ax.set_ylabel('Self-Importance Score (higher = more important)')
            ax.set_title(f'{direction_name}\nRaw r vs Self-Importance')
            ax.grid(True, alpha=0.3)

            col_idx += 1

        fig.suptitle('Self-Importance vs Cross-Platform Raw Correlation', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        return self.save_figure(fig, "self_importance_vs_cross_platform_raw_correlation")
    
    def plot_importance_network(self, network: 'nx.Graph', analysis: Dict, 
                              title_prefix: str, layout_type: str = "spring") -> plt.Figure:
        """Plot network visualization with nodes colored by centrality"""
        if not NETWORKX_AVAILABLE or network is None or len(network.nodes()) == 0:
            print(f"  Skipping network plot for {title_prefix} - no network available")
            return None
            
        print(f"Generating network plot for {title_prefix}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Prepare network for visualization (limit to top 50 nodes by degree)
        degrees = dict(network.degree())
        if len(network.nodes()) > 50:
            print(f"  Network has {len(network.nodes())} nodes, showing top 50 by degree")
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:50]
            network_viz = network.subgraph(top_nodes).copy()
        else:
            network_viz = network.copy()
        
        # Choose layout
        if layout_type == "spring":
            pos = nx.spring_layout(network_viz, k=1, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(network_viz)
        elif layout_type == "kamada_kawai" and len(network_viz.nodes()) <= 100:
            pos = nx.kamada_kawai_layout(network_viz)
        else:
            pos = nx.spring_layout(network_viz, k=1, iterations=50)
        
        # Plot 1: Degree centrality
        ax = axes[0, 0]
        degrees = dict(network_viz.degree())
        if degrees:
            node_colors = [degrees[node] for node in network_viz.nodes()]
            nodes = nx.draw_networkx_nodes(network_viz, pos, node_color=node_colors, 
                                         node_size=50, cmap='viridis', ax=ax)
            nx.draw_networkx_edges(network_viz, pos, alpha=0.3, ax=ax)
            
            if len(network_viz.nodes()) <= 50:  # Only show labels for small networks
                nx.draw_networkx_labels(network_viz, pos, font_size=8, ax=ax)
            
            plt.colorbar(nodes, ax=ax, label='Degree')
        ax.set_title(f'{title_prefix}: Degree Centrality')
        ax.axis('off')
        
        # Plot 2: Betweenness centrality
        ax = axes[0, 1]
        if 'centrality' in analysis and 'betweenness' in analysis['centrality']:
            betweenness = analysis['centrality']['betweenness']
            node_colors = [betweenness.get(node, 0) for node in network_viz.nodes()]
            nodes = nx.draw_networkx_nodes(network_viz, pos, node_color=node_colors, 
                                         node_size=50, cmap='plasma', ax=ax)
            nx.draw_networkx_edges(network_viz, pos, alpha=0.3, ax=ax)
            
            if len(network_viz.nodes()) <= 50:
                nx.draw_networkx_labels(network_viz, pos, font_size=8, ax=ax)
            
            plt.colorbar(nodes, ax=ax, label='Betweenness')
        else:
            ax.text(0.5, 0.5, 'Betweenness centrality\nnot available', ha='center', va='center')
        ax.set_title(f'{title_prefix}: Betweenness Centrality')
        ax.axis('off')
        
        # Plot 3: Community structure
        ax = axes[1, 0]
        if 'community' in analysis and 'partition' in analysis['community']:
            partition = analysis['community']['partition']
            # Filter partition for visualization network
            viz_partition = {node: partition[node] for node in network_viz.nodes() if node in partition}
            
            if viz_partition:
                node_colors = [viz_partition.get(node, 0) for node in network_viz.nodes()]
                nodes = nx.draw_networkx_nodes(network_viz, pos, node_color=node_colors, 
                                             node_size=50, cmap='Set3', ax=ax)
                nx.draw_networkx_edges(network_viz, pos, alpha=0.3, ax=ax)
                
                if len(network_viz.nodes()) <= 50:
                    nx.draw_networkx_labels(network_viz, pos, font_size=8, ax=ax)
                
                ax.set_title(f'{title_prefix}: Communities\n(Modularity: {analysis["community"]["modularity"]:.3f})')
            else:
                ax.text(0.5, 0.5, 'Community data\nnot available', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Community detection\nnot available', ha='center', va='center')
        ax.axis('off')
        
        # Plot 4: Edge weights
        ax = axes[1, 1]
        if network_viz.edges():
            edge_weights = [network_viz[u][v].get('weight', 1) for u, v in network_viz.edges()]
            
            # Normalize edge weights for visualization
            if edge_weights:
                min_weight = min(edge_weights)
                max_weight = max(edge_weights)
                if max_weight > min_weight:
                    edge_alphas = [(w - min_weight) / (max_weight - min_weight) * 0.8 + 0.2 
                                 for w in edge_weights]
                else:
                    edge_alphas = [0.5] * len(edge_weights)
            else:
                edge_alphas = [0.5] * len(network_viz.edges())
            
            nx.draw_networkx_nodes(network_viz, pos, node_color=COLORS['primary'], 
                                 node_size=50, ax=ax)
            
            # Draw edges with varying alpha based on weight
            for (u, v), alpha in zip(network_viz.edges(), edge_alphas):
                nx.draw_networkx_edges(network_viz, pos, [(u, v)], alpha=alpha, 
                                     edge_color='black', ax=ax)
            
            if len(network_viz.nodes()) <= 50:
                nx.draw_networkx_labels(network_viz, pos, font_size=8, ax=ax)
        
        ax.set_title(f'{title_prefix}: Edge Weights\n(darker = higher weight)')
        ax.axis('off')
        
        plt.tight_layout()
        return self.save_figure(fig, f"importance_network_{title_prefix.lower().replace(' ', '_').replace('→', '_to_')}")
    
    def plot_network_topology_analysis(self, data: CorrelationAnalysisData) -> plt.Figure:
        """Plot network topology analysis comparing A→B and B→A networks"""
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available. Skipping network topology plots.")
            return None
            
        print("Generating network topology analysis plots...")
        
        # Determine which networks are available
        has_a_to_b = (data.network_a_to_b is not None and 
                      data.network_analysis_a_to_b is not None)
        has_b_to_a = (data.network_b_to_a is not None and 
                      data.network_analysis_b_to_a is not None)
        
        if not has_a_to_b and not has_b_to_a:
            print("  No network analysis data available")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Collect data for comparison
        network_data = {}
        if has_a_to_b:
            network_data['A→B'] = {
                'network': data.network_a_to_b,
                'analysis': data.network_analysis_a_to_b,
                'color': COLORS['primary']
            }
        if has_b_to_a:
            network_data['B→A'] = {
                'network': data.network_b_to_a,
                'analysis': data.network_analysis_b_to_a,
                'color': COLORS['secondary']
            }
        
        # Plot 1: Basic network statistics
        ax = axes[0, 0]
        directions = list(network_data.keys())
        n_nodes = [network_data[d]['analysis']['basic_stats']['n_nodes'] for d in directions]
        n_edges = [network_data[d]['analysis']['basic_stats']['n_edges'] for d in directions]
        densities = [network_data[d]['analysis']['basic_stats']['density'] for d in directions]
        
        x = np.arange(len(directions))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, n_nodes, width, label='Nodes', color=COLORS['primary'], alpha=0.7)
        bars2 = ax.bar(x + width/2, n_edges, width, label='Edges', color=COLORS['secondary'], alpha=0.7)
        line = ax2.plot(x, densities, 'o-', color=COLORS['accent'], linewidth=2, markersize=8, label='Density')
        
        ax.set_xlabel('Network Direction')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Density')
        ax.set_title('Basic Network Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(directions)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 2: Degree distribution
        ax = axes[0, 1]
        for i, (direction, data_dict) in enumerate(network_data.items()):
            network = data_dict['network']
            degrees = [d for n, d in network.degree()]
            if degrees:
                ax.hist(degrees, bins=min(30, max(5, len(set(degrees)))), alpha=0.7, 
                       label=f'{direction}', color=data_dict['color'], density=True)
        
        ax.set_xlabel('Degree')
        ax.set_ylabel('Density')
        ax.set_title('Degree Distribution')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 3: Centrality comparison
        ax = axes[0, 2]
        centrality_types = ['degree', 'betweenness', 'closeness']
        centrality_data = {direction: [] for direction in directions}
        
        for centrality_type in centrality_types:
            for direction in directions:
                analysis = network_data[direction]['analysis']
                if 'centrality' in analysis and centrality_type in analysis['centrality']:
                    centrality_values = list(analysis['centrality'][centrality_type].values())
                    centrality_data[direction].append(np.mean(centrality_values))
                else:
                    centrality_data[direction].append(0)
        
        x = np.arange(len(centrality_types))
        width = 0.35
        
        for i, direction in enumerate(directions):
            color = network_data[direction]['color']
            ax.bar(x + i * width, centrality_data[direction], width, 
                  label=f'{direction}', color=color, alpha=0.7)
        
        ax.set_xlabel('Centrality Type')
        ax.set_ylabel('Mean Centrality')
        ax.set_title('Centrality Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(centrality_types)
        ax.legend()
        
        # Plot 4: Community structure
        ax = axes[1, 0]
        community_data = []
        community_labels = []
        
        for direction, data_dict in network_data.items():
            analysis = data_dict['analysis']
            if 'community' in analysis and 'community_sizes' in analysis['community']:
                sizes = list(analysis['community']['community_sizes'].values())
                community_data.extend(sizes)
                community_labels.extend([direction] * len(sizes))
        
        if community_data:
            # Create boxplot of community sizes
            direction_data = {}
            for direction in directions:
                analysis = network_data[direction]['analysis']
                if 'community' in analysis and 'community_sizes' in analysis['community']:
                    direction_data[direction] = list(analysis['community']['community_sizes'].values())
                else:
                    direction_data[direction] = []
            
            bp_data = [direction_data[d] for d in directions if direction_data[d]]
            bp_labels = [d for d in directions if direction_data[d]]
            
            if bp_data:
                bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
                
                for patch, direction in zip(bp['boxes'], bp_labels):
                    patch.set_facecolor(network_data[direction]['color'])
                    patch.set_alpha(0.7)
        
        ax.set_xlabel('Network Direction')
        ax.set_ylabel('Community Size')
        ax.set_title('Community Size Distribution')
        
        # Add message if no community data
        if not community_data:
            ax.text(0.5, 0.5, 'No community data\navailable', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 5: Connectivity metrics
        ax = axes[1, 1]
        connectivity_metrics = []
        connectivity_values = []
        connectivity_colors = []
        
        for direction, data_dict in network_data.items():
            analysis = data_dict['analysis']
            color = data_dict['color']
            
            if 'connectivity' in analysis:
                conn = analysis['connectivity']
                
                if 'largest_component_fraction' in conn:
                    connectivity_metrics.append(f'{direction}\nLargest Component')
                    connectivity_values.append(conn['largest_component_fraction'])
                    connectivity_colors.append(color)
                
                if 'average_clustering' in conn:
                    connectivity_metrics.append(f'{direction}\nClustering')
                    connectivity_values.append(conn['average_clustering'])
                    connectivity_colors.append(color)
        
        if connectivity_metrics:
            bars = ax.bar(connectivity_metrics, connectivity_values, color=connectivity_colors, alpha=0.7)
            ax.set_ylabel('Metric Value')
            ax.set_title('Connectivity Metrics')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No connectivity data\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Connectivity Metrics')
        
        # Plot 6: Network comparison (if both networks available)
        ax = axes[1, 2]
        if has_a_to_b and has_b_to_a:
            # Compare basic statistics
            comparison_metrics = ['Nodes', 'Edges', 'Density', 'Mean Degree']
            a_to_b_stats = data.network_analysis_a_to_b['basic_stats']
            b_to_a_stats = data.network_analysis_b_to_a['basic_stats']
            
            values_a = [
                a_to_b_stats['n_nodes'],
                a_to_b_stats['n_edges'],
                a_to_b_stats['density'],
                a_to_b_stats.get('mean_degree', 0)
            ]
            
            values_b = [
                b_to_a_stats['n_nodes'],
                b_to_a_stats['n_edges'],
                b_to_a_stats['density'],
                b_to_a_stats.get('mean_degree', 0)
            ]
            
            # Normalize values for comparison
            max_vals = [max(a, b) for a, b in zip(values_a, values_b)]
            norm_a = [a/m if m > 0 else 0 for a, m in zip(values_a, max_vals)]
            norm_b = [b/m if m > 0 else 0 for b, m in zip(values_b, max_vals)]
            
            x = np.arange(len(comparison_metrics))
            width = 0.35
            
            ax.bar(x - width/2, norm_a, width, label='A→B', color=COLORS['primary'], alpha=0.7)
            ax.bar(x + width/2, norm_b, width, label='B→A', color=COLORS['secondary'], alpha=0.7)
            
            ax.set_xlabel('Metric')
            ax.set_ylabel('Normalized Value')
            ax.set_title('Network Comparison\n(Normalized)')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_metrics, rotation=45)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Both networks needed\nfor comparison', ha='center', va='center')
        
        plt.tight_layout()
        return self.save_figure(fig, "network_topology_analysis")
    
    def plot_network_hub_analysis(self, data: CorrelationAnalysisData) -> plt.Figure:
        """Plot hub analysis showing most connected proteins and their properties"""
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available. Skipping hub analysis plots.")
            return None
            
        print("Generating network hub analysis plots...")
        
        # Check what network data is available
        networks = {}
        if data.network_a_to_b is not None and data.network_analysis_a_to_b is not None:
            networks['A→B'] = {
                'network': data.network_a_to_b,
                'analysis': data.network_analysis_a_to_b,
                'color': COLORS['primary']
            }
        if data.network_b_to_a is not None and data.network_analysis_b_to_a is not None:
            networks['B→A'] = {
                'network': data.network_b_to_a,
                'analysis': data.network_analysis_b_to_a,
                'color': COLORS['secondary']
            }
        
        if not networks:
            print("  No network data available for hub analysis")
            return None
        
        n_networks = len(networks)
        fig, axes = plt.subplots(2, n_networks, figsize=(5*n_networks, 10))
        
        if n_networks == 1:
            axes = axes.reshape(2, 1)
        
        col_idx = 0
        for direction, network_data in networks.items():
            network = network_data['network']
            analysis = network_data['analysis']
            color = network_data['color']
            
            # Top plot: Hub identification
            ax = axes[0, col_idx]
            
            # Get degree centrality
            degrees = dict(network.degree())
            if degrees:
                # Get top 20 hubs
                top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
                hub_names = [name[:15] + '...' if len(name) > 15 else name for name, _ in top_hubs]
                hub_degrees = [degree for _, degree in top_hubs]
                
                bars = ax.barh(range(len(hub_names)), hub_degrees, color=color, alpha=0.7)
                ax.set_yticks(range(len(hub_names)))
                ax.set_yticklabels(hub_names, fontsize=10)
                ax.set_xlabel('Degree')
                ax.set_title(f'{direction}: Top Hub Proteins')
                
                # Add value labels on bars
                for i, (bar, degree) in enumerate(zip(bars, hub_degrees)):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(degree), va='center', fontsize=9)
            
            # Bottom plot: Centrality correlation
            ax = axes[1, col_idx]
            
            if 'centrality' in analysis:
                centrality = analysis['centrality']
                
                # Compare degree vs betweenness centrality
                if 'degree' in centrality and 'betweenness' in centrality:
                    degree_cent = centrality['degree']
                    betweenness_cent = centrality['betweenness']
                    
                    # Get overlapping nodes
                    common_nodes = set(degree_cent.keys()) & set(betweenness_cent.keys())
                    
                    if len(common_nodes) >= 3:
                        degree_vals = [degree_cent[node] for node in common_nodes]
                        betweenness_vals = [betweenness_cent[node] for node in common_nodes]
                        
                        scatter = ax.scatter(degree_vals, betweenness_vals, 
                                           alpha=0.7, s=20, color=color)
                        
                        # Add trend line
                        if len(degree_vals) > 2:
                            z = np.polyfit(degree_vals, betweenness_vals, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(min(degree_vals), max(degree_vals), 100)
                            ax.plot(x_trend, p(x_trend), '--', color=COLORS['warning'], alpha=0.8)
                        
                        # Calculate correlation
                        correlation, p_value = pearsonr(degree_vals, betweenness_vals)
                        ax.text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.3f}', 
                               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               verticalalignment='top')
                        
                        # Label top hubs
                        if len(common_nodes) > 0:
                            # Find nodes with high degree or betweenness
                            node_scores = {node: degree_cent[node] + betweenness_cent[node] 
                                         for node in common_nodes}
                            top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                            
                            for node, _ in top_nodes:
                                ax.annotate(node[:8], 
                                          (degree_cent[node], betweenness_cent[node]),
                                          xytext=(5, 5), textcoords='offset points',
                                          fontsize=8, alpha=0.7,
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
                
                        ax.set_xlabel('Degree Centrality')
                        ax.set_ylabel('Betweenness Centrality')
                        ax.set_title(f'{direction}: Centrality Correlation')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'Insufficient nodes\nfor correlation', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{direction}: Centrality Correlation')
                else:
                    ax.text(0.5, 0.5, 'Centrality data\nnot available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{direction}: Centrality Correlation')
            else:
                ax.text(0.5, 0.5, 'Centrality analysis\nnot available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{direction}: Centrality Correlation')
            
            col_idx += 1
        
        plt.tight_layout()
        return self.save_figure(fig, "network_hub_analysis")
    
    def plot_ppi_network_comparison(self, data: CorrelationAnalysisData) -> plt.Figure:
        """Plot comparison between importance networks and PPI reference"""
        if not NETWORKX_AVAILABLE or data.ppi_reference is None:
            print("PPI reference not available. Skipping PPI comparison plots.")
            return None
            
        print("Generating PPI network comparison plots...")
        
        # Check which comparisons are available
        has_a_to_b = (data.network_a_to_b is not None and 
                      data.ppi_comparison_a_to_b is not None)
        has_b_to_a = (data.network_b_to_a is not None and 
                      data.ppi_comparison_b_to_a is not None)
        
        if not has_a_to_b and not has_b_to_a:
            print("  No PPI comparison data available")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Network size comparison
        ax = axes[0, 0]
        
        network_names = ['PPI Reference']
        node_counts = [len(data.ppi_reference.nodes())]
        edge_counts = [len(data.ppi_reference.edges())]
        colors = [COLORS['info']]
        
        if has_a_to_b:
            network_names.append(f"{data.platform_a_name} → {data.platform_b_name}")
            node_counts.append(data.ppi_comparison_a_to_b['importance_stats']['n_nodes'])
            edge_counts.append(data.ppi_comparison_a_to_b['importance_stats']['n_edges'])
            colors.append(COLORS['primary'])
        
        if has_b_to_a:
            network_names.append(f"{data.platform_b_name} → {data.platform_a_name}")
            node_counts.append(data.ppi_comparison_b_to_a['importance_stats']['n_nodes'])
            edge_counts.append(data.ppi_comparison_b_to_a['importance_stats']['n_edges'])
            colors.append(COLORS['secondary'])
        
        x = np.arange(len(network_names))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, node_counts, width, label='Nodes', alpha=0.7, color=colors)
        bars2 = ax2.bar(x + width/2, edge_counts, width, label='Edges', alpha=0.7, 
                       color=[COLORS['accent']] * len(network_names))
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Number of Nodes')
        ax2.set_ylabel('Number of Edges')
        ax.set_title('Network Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Overlap analysis (Jaccard similarities)
        ax = axes[0, 1]
        
        jaccard_data = {'Node Overlap': [], 'Edge Overlap': []}
        comparison_names = []
        comparison_colors = []
        
        if has_a_to_b:
            comparison_names.append(f"{data.platform_a_name} → {data.platform_b_name}")
            jaccard_data['Node Overlap'].append(data.ppi_comparison_a_to_b['overlap_analysis']['nodes']['jaccard_similarity'])
            jaccard_data['Edge Overlap'].append(data.ppi_comparison_a_to_b['overlap_analysis']['edges']['jaccard_similarity'])
            comparison_colors.append(COLORS['primary'])
        
        if has_b_to_a:
            comparison_names.append(f"{data.platform_b_name} → {data.platform_a_name}")
            jaccard_data['Node Overlap'].append(data.ppi_comparison_b_to_a['overlap_analysis']['nodes']['jaccard_similarity'])
            jaccard_data['Edge Overlap'].append(data.ppi_comparison_b_to_a['overlap_analysis']['edges']['jaccard_similarity'])
            comparison_colors.append(COLORS['secondary'])
        
        x = np.arange(len(comparison_names))
        width = 0.35
        
        ax.bar(x - width/2, jaccard_data['Node Overlap'], width, 
               label='Node Overlap', alpha=0.7, color=COLORS['accent'])
        ax.bar(x + width/2, jaccard_data['Edge Overlap'], width, 
               label='Edge Overlap', alpha=0.7, color=COLORS['warning'])
        
        ax.set_xlabel('Network Comparison')
        ax.set_ylabel('Jaccard Similarity')
        ax.set_title('Overlap with PPI Reference')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Plot 3: Novel vs Validated connections
        ax = axes[0, 2]
        
        novel_counts = []
        validated_counts = []
        
        if has_a_to_b:
            novel_counts.append(data.ppi_comparison_a_to_b['novel_connections']['count'])
            validated_counts.append(data.ppi_comparison_a_to_b['ppi_validation']['count'])
        
        if has_b_to_a:
            novel_counts.append(data.ppi_comparison_b_to_a['novel_connections']['count'])
            validated_counts.append(data.ppi_comparison_b_to_a['ppi_validation']['count'])
        
        x = np.arange(len(comparison_names))
        width = 0.35
        
        ax.bar(x - width/2, novel_counts, width, 
               label='Novel Connections', alpha=0.7, color=COLORS['warning'])
        ax.bar(x + width/2, validated_counts, width, 
               label='PPI-Validated', alpha=0.7, color=COLORS['success'])
        
        ax.set_xlabel('Network Comparison')
        ax.set_ylabel('Number of Connections')
        ax.set_title('Novel vs PPI-Validated Connections')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax.legend()
        
        # Plot 4: Enrichment analysis
        ax = axes[1, 0]
        
        enrichment_data = []
        enrichment_significance = []
        
        # Build comparison data mapping
        comparisons_data = []
        if has_a_to_b:
            comparisons_data.append((f"{data.platform_a_name} → {data.platform_b_name}", 
                                    data.ppi_comparison_a_to_b))
        if has_b_to_a:
            comparisons_data.append((f"{data.platform_b_name} → {data.platform_a_name}", 
                                    data.ppi_comparison_b_to_a))
        
        for comp_name, comp_data in comparisons_data:
            enrichment = comp_data.get('enrichment_analysis', {}) if comp_data else {}
            
            if enrichment and 'hypergeometric_p_value' in enrichment:
                p_value = enrichment['hypergeometric_p_value']
                enrichment_data.append(-np.log10(p_value))
                enrichment_significance.append(p_value < 0.05)
            else:
                enrichment_data.append(0)
                enrichment_significance.append(False)
        
        if enrichment_data:
            bar_colors = [COLORS['success'] if sig else COLORS['info'] 
                         for sig in enrichment_significance]
            bars = ax.bar(comparison_names, enrichment_data, alpha=0.7, color=bar_colors)
            
            # Add significance threshold line
            ax.axhline(-np.log10(0.05), color=COLORS['danger'], linestyle='--', 
                      label='p = 0.05 threshold')
            
            ax.set_xlabel('Network Comparison')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title('PPI Enrichment Significance')
            ax.legend()
            
            # Add p-value labels on bars
            for bar, p_val in zip(bars, [10**(-val) if val > 0 else 1 for val in enrichment_data]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'p = {p_val:.3e}' if p_val < 0.001 else f'p = {p_val:.3f}',
                       ha='center', va='bottom', rotation=45, fontsize=9)
        
        # Plot 5 & 6: Edge overlap Venn-like visualization
        for plot_idx, (comparison_name, comparison_data) in enumerate(comparisons_data):
            if plot_idx >= 2:  # Only plot first 2 comparisons
                break
                
            ax = axes[1, plot_idx + 1]
            
            # Extract overlap data
            overlap = comparison_data['overlap_analysis']['edges']
            importance_only = overlap['importance_only']
            ppi_only = overlap['ppi_only']
            intersection = overlap['intersection']
            
            # Create Venn diagram
            venn = venn2(subsets=(importance_only, ppi_only, intersection),
                        set_labels=('Importance Network', 'PPI Network'),
                        ax=ax)
            
            # Customize colors
            if venn.get_patch_by_id('10'):  # Importance only
                venn.get_patch_by_id('10').set_color(comparison_colors[plot_idx])
                venn.get_patch_by_id('10').set_alpha(0.7)
            if venn.get_patch_by_id('01'):  # PPI only
                venn.get_patch_by_id('01').set_color(COLORS['info'])
                venn.get_patch_by_id('01').set_alpha(0.7)
            if venn.get_patch_by_id('11'):  # Intersection
                venn.get_patch_by_id('11').set_color(COLORS['success'])
                venn.get_patch_by_id('11').set_alpha(0.7)
            
            # Add edge styling to circles
            venn2_circles(subsets=(importance_only, ppi_only, intersection), ax=ax, linewidth=1.5, color='gray')
            
            # Calculate percentages for labels
            total = importance_only + ppi_only + intersection
            if total > 0:
                for subset_id, value in [('10', importance_only), ('01', ppi_only), ('11', intersection)]:
                    label = venn.get_label_by_id(subset_id)
                    if label:
                        pct = (value / total) * 100
                        label.set_text(f'{value}\n({pct:.1f}%)')
            
            ax.set_title(f'{comparison_name}\nEdge Overlap')
        
        # Hide unused subplot if only one comparison
        if len(comparisons_data) == 1:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        return self.save_figure(fig, "ppi_network_comparison")
    
    def plot_ppi_validation_networks(self, data: CorrelationAnalysisData) -> plt.Figure:
        """Plot importance networks with PPI-validated edges highlighted"""
        if not NETWORKX_AVAILABLE or data.ppi_reference is None:
            print("PPI reference not available. Skipping PPI validation network plots.")
            return None
            
        print("Generating PPI validation network plots...")
        
        # Check which networks are available
        networks_to_plot = []
        if data.network_a_to_b is not None and data.ppi_comparison_a_to_b is not None:
            networks_to_plot.append({
                'network': data.network_a_to_b,
                'comparison': data.ppi_comparison_a_to_b,
                'title': f"{data.platform_a_name} → {data.platform_b_name}",
                'color': COLORS['primary']
            })
        
        if data.network_b_to_a is not None and data.ppi_comparison_b_to_a is not None:
            networks_to_plot.append({
                'network': data.network_b_to_a,
                'comparison': data.ppi_comparison_b_to_a,
                'title': f"{data.platform_b_name} → {data.platform_a_name}",
                'color': COLORS['secondary']
            })
        
        if not networks_to_plot:
            print("  No networks available for PPI validation plotting")
            return None
        
        n_networks = len(networks_to_plot)
        fig, axes = plt.subplots(2, n_networks, figsize=(5*n_networks, 10))
        
        if n_networks == 1:
            axes = axes.reshape(2, 1)
        
        for col_idx, network_info in enumerate(networks_to_plot):
            network = network_info['network']
            comparison = network_info['comparison']
            title = network_info['title']
            base_color = network_info['color']
            
            # Limit network size for visualization (top 50 nodes)
            degrees = dict(network.degree())
            if len(network.nodes()) > 50:
                print(f"  Network has {len(network.nodes())} nodes, showing top 50 by degree")
                top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:50]
                network_viz = network.subgraph(top_nodes).copy()
            else:
                network_viz = network.copy()
            
            # Get validated edges (convert to undirected tuples for comparison)
            validated_edges = set()
            for edge_tuple in comparison['ppi_validation']['examples']:
                if len(edge_tuple) == 2:
                    validated_edges.add(edge_tuple)
            
            # Create layout
            pos = nx.spring_layout(network_viz, k=1, iterations=50)
            
            # Top plot: Full network with validated edges highlighted
            ax = axes[0, col_idx]
            
            # Draw all edges in light gray
            nx.draw_networkx_edges(network_viz, pos, alpha=0.2, edge_color='gray', ax=ax)
            
            # Highlight PPI-validated edges
            validated_edges_in_viz = []
            for u, v in network_viz.edges():
                edge_tuple = tuple(sorted([u, v]))
                if edge_tuple in validated_edges:
                    validated_edges_in_viz.append((u, v))
            
            if validated_edges_in_viz:
                nx.draw_networkx_edges(network_viz, pos, edgelist=validated_edges_in_viz,
                                     edge_color=COLORS['success'], width=2, alpha=0.8, ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(network_viz, pos, node_color=base_color, 
                                 node_size=30, alpha=0.7, ax=ax)
            
            # Add labels for small networks
            if len(network_viz.nodes()) <= 50:
                nx.draw_networkx_labels(network_viz, pos, font_size=8, ax=ax)
            
            ax.set_title(f'{title}\nPPI-Validated Edges (Green)\n'
                        f'{len(validated_edges_in_viz)}/{len(network_viz.edges())} edges validated')
            ax.axis('off')
            
            # Bottom plot: Novel connections only
            ax = axes[1, col_idx]
            
            # Get novel edges
            novel_edges_examples = comparison['novel_connections']['examples']
            novel_edges_set = set()
            for edge_tuple in novel_edges_examples:
                if len(edge_tuple) == 2:
                    novel_edges_set.add(edge_tuple)
            
            # Find novel edges in visualization network
            novel_edges_in_viz = []
            for u, v in network_viz.edges():
                edge_tuple = tuple(sorted([u, v]))
                if edge_tuple in novel_edges_set:
                    novel_edges_in_viz.append((u, v))
            
            if novel_edges_in_viz:
                # Create subgraph with only novel edges
                novel_graph = nx.Graph()
                novel_graph.add_edges_from(novel_edges_in_viz)
                
                # Use same positions but only for nodes in novel graph
                novel_pos = {node: pos[node] for node in novel_graph.nodes() if node in pos}
                
                # Draw novel network
                nx.draw_networkx_edges(novel_graph, novel_pos, 
                                     edge_color=COLORS['warning'], width=2, alpha=0.8, ax=ax)
                nx.draw_networkx_nodes(novel_graph, novel_pos, 
                                     node_color=COLORS['warning'], node_size=40, alpha=0.7, ax=ax)
                
                # Add labels for small networks
                if len(novel_graph.nodes()) <= 30:
                    nx.draw_networkx_labels(novel_graph, novel_pos, font_size=8, ax=ax)
            
            ax.set_title(f'{title}\nNovel Connections (Not in PPI)\n'
                        f'{len(novel_edges_in_viz)} novel connections shown')
            ax.axis('off')
            
            if not novel_edges_in_viz:
                ax.text(0.5, 0.5, 'No novel connections\nin visualization subset', 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return self.save_figure(fig, "ppi_validation_networks")
    
    def plot_threshold_analysis(self, threshold_analyses: Dict, target_density: float = 0.0366) -> plt.Figure:
        """Plot threshold analysis results to help determine optimal network construction parameters"""
        if not threshold_analyses:
            print("No threshold analysis data available for plotting")
            return None
            
        print("Generating threshold analysis plots...")
        
        # Determine number of analyses to plot
        analyses = list(threshold_analyses.values())
        n_analyses = len(analyses)
        
        fig, axes = plt.subplots(3, n_analyses, figsize=(5*n_analyses, 15))
        if n_analyses == 1:
            axes = axes.reshape(3, 1)
        
        for col_idx, (analysis_name, analysis) in enumerate(threshold_analyses.items()):
            thresholds = analysis['thresholds'] 
            edge_counts = analysis['edge_counts']
            node_counts = analysis['node_counts']
            densities = analysis['densities']
            mean_degrees = analysis['mean_degrees']
            
            # Plot 1: Edge count vs threshold
            ax = axes[0, col_idx]
            
            bars = ax.bar(range(len(thresholds)), edge_counts, alpha=0.7, color=COLORS['primary'])
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Number of Edges')
            ax.set_title(f'{analysis_name}\nEdge Count vs Threshold')
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels(thresholds, rotation=45, ha='right')
            ax.set_yscale('log')
            
            # Highlight recommendations
            recommendations = analysis.get('recommendations', {})
            for rec_name, rec_data in recommendations.items():
                if rec_data['threshold'] in thresholds:
                    idx = thresholds.index(rec_data['threshold'])
                    bars[idx].set_color(COLORS['success'])
                    bars[idx].set_alpha(0.9)
            
            # Add target edge count range
            ax.axhline(1000, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Target range')
            ax.axhline(10000, color=COLORS['warning'], linestyle='--', alpha=0.7)
            ax.fill_between(range(len(thresholds)), 1000, 10000, alpha=0.1, color=COLORS['warning'])
            ax.legend()
            
            # Plot 2: Network density vs threshold
            ax = axes[1, col_idx]
            
            bars = ax.bar(range(len(thresholds)), densities, alpha=0.7, color=COLORS['secondary'])
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Network Density')
            ax.set_title(f'{analysis_name}\nNetwork Density vs Threshold')
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels(thresholds, rotation=45, ha='right')
            
            # Highlight recommendations
            for rec_name, rec_data in recommendations.items():
                if rec_data['threshold'] in thresholds:
                    idx = thresholds.index(rec_data['threshold'])
                    bars[idx].set_color(COLORS['success'])
                    bars[idx].set_alpha(0.9)
            
            # Add target density line
            ax.axhline(target_density, color=COLORS['danger'], linestyle='-', alpha=0.8, 
                      linewidth=2, label=f'Target density: {target_density:.4f}')
            
            # Find threshold closest to target density
            if densities:
                density_diffs = [abs(d - target_density) for d in densities]
                min_diff_idx = density_diffs.index(min(density_diffs))
                best_threshold = thresholds[min_diff_idx]
                best_density = densities[min_diff_idx]
                
                # Highlight the best threshold
                bars[min_diff_idx].set_color(COLORS['danger'])
                bars[min_diff_idx].set_alpha(0.9)
                bars[min_diff_idx].set_edgecolor('black')
                bars[min_diff_idx].set_linewidth(2)
                
                # Add annotation
                ax.annotate(f'Best: {best_threshold:.6f}\nDensity: {best_density:.4f}', 
                           xy=(min_diff_idx, best_density), 
                           xytext=(min_diff_idx + len(thresholds)*0.1, best_density + max(densities)*0.1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.legend()
            
            # Plot 3: Nodes vs Edges (efficiency plot)
            ax = axes[2, col_idx]
            
            # Color points by density
            scatter = ax.scatter(edge_counts, node_counts, c=densities, cmap='viridis', 
                               s=50, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Highlight recommendations
            for rec_name, rec_data in recommendations.items():
                if rec_data['threshold'] in thresholds:
                    idx = thresholds.index(rec_data['threshold'])
                    ax.scatter(edge_counts[idx], node_counts[idx], 
                             color=COLORS['danger'], s=100, marker='*', 
                             edgecolors='black', linewidth=2, 
                             label=f"{rec_name.replace('_', ' ').title()}")
            
            ax.set_xlabel('Number of Edges')
            ax.set_ylabel('Number of Nodes')
            ax.set_title(f'{analysis_name}\nNodes vs Edges (Color = Density)')
            ax.set_xscale('log')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Network Density')
            
            # Add target regions
            ax.axvspan(1000, 10000, alpha=0.1, color=COLORS['warning'], label='Target edges')
            
            if recommendations:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Annotate points with threshold labels
            for i in range(0, len(thresholds), max(1, len(thresholds)//5)):  # Show every 5th label
                ax.annotate(thresholds[i], (edge_counts[i], node_counts[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return self.save_figure(fig, "threshold_analysis")
    
    def plot_threshold_recommendations(self, threshold_analyses: Dict) -> plt.Figure:
        """Plot recommendations from threshold analysis"""
        if not threshold_analyses:
            print("No threshold analysis data available for recommendations plot")
            return None
            
        print("Generating threshold recommendations plot...")
        
        # Collect all recommendations
        all_recommendations = {}
        analysis_names = []
        
        for analysis_name, analysis in threshold_analyses.items():
            analysis_names.append(analysis_name)
            recommendations = analysis.get('recommendations', {})
            for rec_name, rec_data in recommendations.items():
                if rec_name not in all_recommendations:
                    all_recommendations[rec_name] = {}
                all_recommendations[rec_name][analysis_name] = rec_data
        
        if not all_recommendations:
            print("  No recommendations available to plot")
            return None
        
        n_rec_types = len(all_recommendations)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        rec_names = list(all_recommendations.keys())
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['info']]
        
        # Plot 1: Edge counts comparison
        ax = axes[0]
        
        x_pos = np.arange(len(analysis_names))
        width = 0.8 / n_rec_types
        
        for i, rec_name in enumerate(rec_names):
            rec_data = all_recommendations[rec_name]
            edge_counts = [rec_data.get(analysis_name, {}).get('edge_count', 0) 
                          for analysis_name in analysis_names]
            
            ax.bar(x_pos + i * width, edge_counts, width, 
                  label=rec_name.replace('_', ' ').title(), 
                  alpha=0.7, color=colors[i % len(colors)])
        
        ax.set_xlabel('Analysis Type')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Recommended Edge Counts by Analysis Method')
        ax.set_xticks(x_pos + width * (n_rec_types - 1) / 2)
        ax.set_xticklabels(analysis_names)
        ax.set_yscale('log')
        ax.legend()
        
        # Plot 2: Density comparison
        ax = axes[1]
        
        for i, rec_name in enumerate(rec_names):
            rec_data = all_recommendations[rec_name]
            densities = [rec_data.get(analysis_name, {}).get('density', 0) 
                        for analysis_name in analysis_names]
            
            ax.bar(x_pos + i * width, densities, width, 
                  label=rec_name.replace('_', ' ').title(), 
                  alpha=0.7, color=colors[i % len(colors)])
        
        ax.set_xlabel('Analysis Type')
        ax.set_ylabel('Network Density')
        ax.set_title('Recommended Network Densities')
        ax.set_xticks(x_pos + width * (n_rec_types - 1) / 2)
        ax.set_xticklabels(analysis_names)
        ax.legend()
        
        # Plot 3: Node counts comparison
        ax = axes[2]
        
        for i, rec_name in enumerate(rec_names):
            rec_data = all_recommendations[rec_name]
            node_counts = [rec_data.get(analysis_name, {}).get('node_count', 0) 
                          for analysis_name in analysis_names]
            
            ax.bar(x_pos + i * width, node_counts, width, 
                  label=rec_name.replace('_', ' ').title(), 
                  alpha=0.7, color=colors[i % len(colors)])
        
        ax.set_xlabel('Analysis Type')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('Recommended Node Counts')
        ax.set_xticks(x_pos + width * (n_rec_types - 1) / 2)
        ax.set_xticklabels(analysis_names)
        ax.legend()
        
        # Plot 4: Summary table
        ax = axes[3]
        ax.axis('off')
        
        # Create recommendation summary table
        table_data = []
        headers = ['Recommendation', 'Analysis', 'Threshold', 'Edges', 'Nodes', 'Density', 'Reason']
        
        for rec_name, rec_data in all_recommendations.items():
            for analysis_name, data in rec_data.items():
                table_data.append([
                    rec_name.replace('_', ' ').title(),
                    analysis_name,
                    data.get('threshold', 'N/A'),
                    f"{data.get('edge_count', 0):,}",
                    f"{data.get('node_count', 0):,}",
                    f"{data.get('density', 0):.4f}",
                    data.get('reason', 'N/A')[:30] + '...' if len(data.get('reason', '')) > 30 else data.get('reason', 'N/A')
                ])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers, 
                           cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Color header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor(COLORS['info'])
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Threshold Recommendations Summary', pad=20, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, "threshold_recommendations")
    
    def plot_threshold_pr_curves(self, data: CorrelationAnalysisData, 
                                threshold_analyses: Dict) -> plt.Figure:
        """
        Plot Precision-Recall curves for different thresholds showing network edge validation 
        against PPI reference. 
        
        Precision: % of predicted network edges that exist in PPI reference
        Recall: % of PPI reference edges that are recovered by predicted network
        
        Args:
            data: CorrelationAnalysisData object with networks and PPI reference
            threshold_analyses: Dictionary with threshold analysis results
            
        Returns:
            Matplotlib figure with PR curves
        """
        if not NETWORKX_AVAILABLE or not threshold_analyses:
            print("NetworkX not available or no threshold analyses. Skipping PR curves.")
            return None
            
        print("Generating threshold PR curves...")
        
        # Check if we have PPI reference for validation
        if data.ppi_reference is None:
            print("  No PPI reference available for PR curves")
            return None
        
        # Determine which analyses to plot - correlation analyses are supported for PR plots
        analysis_keys = [k for k, v in threshold_analyses.items() if v.get('threshold_type') == 'correlation']
        if not analysis_keys:
            print("  No threshold analyses found for PR curves")
            return None
        
        n_analyses = len(analysis_keys)
        fig, axes = plt.subplots(1, n_analyses, figsize=(6*n_analyses, 6))
        if n_analyses == 1:
            axes = [axes]
        
        for col_idx, analysis_key in enumerate(analysis_keys):
            analysis = threshold_analyses[analysis_key]
            
            # Determine which correlation matrix to use
            if 'A' in analysis_key:
                correlation_matrix = data.correlation_matrix_a
                direction_name = f"{data.platform_a_name} Correlation Network"
            else:
                correlation_matrix = data.correlation_matrix_b  
                direction_name = f"{data.platform_b_name} Correlation Network"
                
            if correlation_matrix is None:
                continue
            
            # Get thresholds and compute PR metrics
            thresholds = analysis['thresholds']
            ax = axes[col_idx]
            
            # Compute PPI validation PR curve
            use_absolute = analysis.get('use_absolute', True)
            precision_values, recall_values = self._compute_ppi_pr_curve(
                correlation_matrix, data.ppi_reference, thresholds, use_absolute=use_absolute)
            
            if len(precision_values) > 0 and len(recall_values) > 0:
                # Convert to percentages for display
                precision_pct = [p * 100 for p in precision_values]
                recall_pct = [r * 100 for r in recall_values]
                
                # Plot the PR curve using percentage values
                ax.plot(recall_pct, precision_pct, 'o-', 
                       color=COLORS['primary'], linewidth=2, markersize=6, alpha=0.8)
                
                # Set fixed axis limits to 0-100% for consistency and comparability
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                
                # Format axes as percentages
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))
                
                # Labels and title
                ax.set_xlabel('Recall (% of PPI edges recovered)')
                ax.set_ylabel('Precision (% of predicted edges in PPI)')
                ax.set_title(f'{direction_name}\nNetwork Edge Validation vs PPI Reference')
                ax.grid(True, alpha=0.3)
                
                # Add threshold annotations for key points
                step_size = max(1, len(thresholds)//6)  # Show ~6 points
                for i in range(0, len(thresholds), step_size):
                    if i < len(recall_pct) and i < len(precision_pct):
                        ax.annotate(f'{thresholds[i]:.3f}', 
                                  (recall_pct[i], precision_pct[i]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=9, alpha=0.8,
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                
                # Add summary statistics text box
                total_ppi_edges = len(data.ppi_reference.edges())
                max_pred_edges = max([analysis['edge_counts'][i] for i in range(len(thresholds))]) if analysis['edge_counts'] else 0
                max_overlap = max([int(r * total_ppi_edges / 100) for r in recall_pct]) if recall_pct else 0
                
                stats_text = f'Max overlap: {max_overlap:,} edges\nPPI edges: {total_ppi_edges:,}\nMax predicted: {max_pred_edges:,}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, 'No PPI validation\ndata available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{direction_name}\nNetwork Edge Validation vs PPI Reference')
        
        plt.tight_layout()
        return self.save_figure(fig, "threshold_pr_curves")
    
    def _compute_ppi_pr_curve(self, correlation_matrix: pd.DataFrame, 
                             ppi_network: 'nx.Graph', 
                             thresholds: List[float],
                             use_absolute: bool = True) -> Tuple[List[float], List[float]]:
        """
        Compute precision-recall curve for PPI validation across thresholds.
        
        Args:
            correlation_matrix: Feature correlation matrix
            ppi_network: Reference PPI network
            thresholds: List of threshold values to test
            use_absolute: If True, use absolute correlation values
            
        Returns:
            Tuple of (precision_values, recall_values)
        """
        if not NETWORKX_AVAILABLE or ppi_network is None:
            return [], []
        
        # Get all PPI edges as ground truth
        ppi_edges = set()
        for u, v in ppi_network.edges():
            ppi_edges.add(tuple(sorted([u, v])))
        
        if len(ppi_edges) == 0:
            return [], []
        
        precision_values = []
        recall_values = []
        
        for threshold in thresholds:
            # Build network at this threshold
            pred_edges = set()
            
            features = list(correlation_matrix.index)
            n_features = len(features)
            
            # Build edges based on correlation threshold
            for i in range(n_features):
                for j in range(i+1, n_features):
                    feat1 = features[i]
                    feat2 = features[j]
                    
                    corr_value = correlation_matrix.loc[feat1, feat2]
                    
                    if use_absolute:
                        if abs(corr_value) > threshold:
                            pred_edges.add(tuple(sorted([feat1, feat2])))
                    else:
                        if corr_value > threshold:
                            pred_edges.add(tuple(sorted([feat1, feat2])))
            
            if len(pred_edges) == 0:
                precision = 0.0
                recall = 0.0
            else:
                # Calculate precision and recall
                true_positives = len(pred_edges & ppi_edges)
                precision = true_positives / len(pred_edges)
                recall = true_positives / len(ppi_edges)
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        return precision_values, recall_values
    
    
    def generate_summary_report(self, data: CorrelationAnalysisData):
        """Generate a summary report of the analysis"""
        print("Generating summary report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': self.timestamp,
                'git_hash': self.git_hash,
                'platform_a_name': data.platform_a_name,
                'platform_b_name': data.platform_b_name,
                'networkx_available': NETWORKX_AVAILABLE,
                'community_detection_available': COMMUNITY_AVAILABLE
            },
            'matrix_summary': {},
            'rank_consistency_summary': {},
            'self_feature_importance_summary': {},
            'network_analysis_summary': {}
        }
        
        # Matrix summaries
        if data.importance_a_to_b is not None:
            matrix = data.importance_a_to_b
            report['matrix_summary']['a_to_b'] = {
                'shape': list(matrix.shape),
                'mean_importance': float(matrix.values.mean()),
                'std_importance': float(matrix.values.std()),
                'max_importance': float(matrix.values.max()),
                'sparsity': float((matrix.values == 0).sum() / matrix.size),
                'top_input_features': matrix.mean(axis=1).nlargest(10).to_dict(),
                'most_variable_output_features': matrix.std(axis=0).nlargest(10).to_dict()
            }
        
        if data.importance_b_to_a is not None:
            matrix = data.importance_b_to_a
            report['matrix_summary']['b_to_a'] = {
                'shape': list(matrix.shape),
                'mean_importance': float(matrix.values.mean()),
                'std_importance': float(matrix.values.std()),
                'max_importance': float(matrix.values.max()),
                'sparsity': float((matrix.values == 0).sum() / matrix.size),
                'top_input_features': matrix.mean(axis=1).nlargest(10).to_dict(),
                'most_variable_output_features': matrix.std(axis=0).nlargest(10).to_dict()
            }
        
        # Rank consistency summaries
        if data.rank_consistency_a_to_b is not None:
            consistency = data.rank_consistency_a_to_b
            rank_stats = consistency['rank_stats']
            
            # Convert consistently important features to proper format
            consistently_important_dict = {}
            for feature_name in rank_stats.nsmallest(5, 'mean_rank').index:
                consistently_important_dict[feature_name] = {
                    'mean_rank': float(rank_stats.loc[feature_name, 'mean_rank']),
                    'rank_variance': float(rank_stats.loc[feature_name, 'rank_variance'])
                }
            
            report['rank_consistency_summary']['a_to_b'] = {
                'most_consistently_important': consistently_important_dict,
                'most_variable_features': rank_stats.nlargest(5, 'rank_variance')[['mean_rank', 'rank_variance']].to_dict('index'),
                'top_feature_counts': consistency['top_feature_counts'].head(10).to_dict(),
                'average_rank_variance': float(rank_stats['rank_variance'].mean()),
                'features_always_top': int(consistency['top_feature_counts'].max()),
                'features_never_top': int((consistency['top_feature_counts'] == 0).sum())
            }
        
        if data.rank_consistency_b_to_a is not None:
            consistency = data.rank_consistency_b_to_a
            rank_stats = consistency['rank_stats']
            
            # Convert consistently important features to proper format
            consistently_important_dict = {}
            for feature_name in rank_stats.nsmallest(5, 'mean_rank').index:
                consistently_important_dict[feature_name] = {
                    'mean_rank': float(rank_stats.loc[feature_name, 'mean_rank']),
                    'rank_variance': float(rank_stats.loc[feature_name, 'rank_variance'])
                }
            
            report['rank_consistency_summary']['b_to_a'] = {
                'most_consistently_important': consistently_important_dict,
                'most_variable_features': rank_stats.nlargest(5, 'rank_variance')[['mean_rank', 'rank_variance']].to_dict('index'),
                'top_feature_counts': consistency['top_feature_counts'].head(10).to_dict(),
                'average_rank_variance': float(rank_stats['rank_variance'].mean()),
                'features_always_top': int(consistency['top_feature_counts'].max()),
                'features_never_top': int((consistency['top_feature_counts'] == 0).sum())
            }
        
        # Skip self-feature importance summaries - not relevant for correlation networks
        
        # Network analysis summaries
        if NETWORKX_AVAILABLE:
            if data.network_analysis_a_to_b is not None:
                network_stats = data.network_analysis_a_to_b.get('basic_stats', {})
                centrality_stats = data.network_analysis_a_to_b.get('centrality', {})
                community_stats = data.network_analysis_a_to_b.get('community', {})
                
                report['network_analysis_summary']['a_to_b'] = {
                    'n_nodes': network_stats.get('n_nodes', 0),
                    'n_edges': network_stats.get('n_edges', 0),
                    'density': network_stats.get('density', 0),
                    'mean_degree': network_stats.get('mean_degree', 0),
                    'n_communities': community_stats.get('n_communities', 0),
                    'modularity': community_stats.get('modularity', 0),
                    'centrality_available': list(centrality_stats.keys()) if centrality_stats else []
                }
                
                # Add top hub proteins if available
                if centrality_stats.get('degree'):
                    degree_centrality = centrality_stats['degree']
                    top_hubs = dict(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10])
                    report['network_analysis_summary']['a_to_b']['top_hub_proteins'] = top_hubs
            
            if data.network_analysis_b_to_a is not None:
                network_stats = data.network_analysis_b_to_a.get('basic_stats', {})
                centrality_stats = data.network_analysis_b_to_a.get('centrality', {})
                community_stats = data.network_analysis_b_to_a.get('community', {})
                
                report['network_analysis_summary']['b_to_a'] = {
                    'n_nodes': network_stats.get('n_nodes', 0),
                    'n_edges': network_stats.get('n_edges', 0),
                    'density': network_stats.get('density', 0),
                    'mean_degree': network_stats.get('mean_degree', 0),
                    'n_communities': community_stats.get('n_communities', 0),
                    'modularity': community_stats.get('modularity', 0),
                    'centrality_available': list(centrality_stats.keys()) if centrality_stats else []
                }
                
                # Add top hub proteins if available
                if centrality_stats.get('degree'):
                    degree_centrality = centrality_stats['degree']
                    top_hubs = dict(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10])
                    report['network_analysis_summary']['b_to_a']['top_hub_proteins'] = top_hubs
            
            # Network comparison summary
            if data.network_comparison is not None:
                comparison = data.network_comparison
                report['network_analysis_summary']['network_comparison'] = {
                    'node_jaccard_similarity': comparison.get('node_overlap', {}).get('jaccard', 0),
                    'edge_jaccard_similarity': comparison.get('edge_overlap', {}).get('jaccard', 0),
                    'centrality_correlations': comparison.get('centrality_correlation', {})
                }
        
        # PPI analysis summary
        if data.ppi_reference is not None:
            report['ppi_analysis_summary'] = {
                'ppi_reference_stats': data.ppi_reference_stats or {},
                'comparisons': {}
            }
            
            # Add PPI comparison results
            if data.ppi_comparison_a_to_b is not None:
                ppi_comp = data.ppi_comparison_a_to_b
                report['ppi_analysis_summary']['comparisons']['a_to_b'] = {
                    'direction': ppi_comp['direction'],
                    'node_jaccard_similarity': ppi_comp['overlap_analysis']['nodes']['jaccard_similarity'],
                    'edge_jaccard_similarity': ppi_comp['overlap_analysis']['edges']['jaccard_similarity'],
                    'novel_connections_count': ppi_comp['novel_connections']['count'],
                    'novel_connections_fraction': ppi_comp['novel_connections']['fraction_of_importance_edges'],
                    'ppi_validation_count': ppi_comp['ppi_validation']['count'],
                    'ppi_validation_fraction': ppi_comp['ppi_validation']['fraction_of_ppi_edges'],
                    'enrichment_significant': ppi_comp.get('enrichment_analysis', {}).get('enrichment_significant', False),
                    'enrichment_p_value': ppi_comp.get('enrichment_analysis', {}).get('hypergeometric_p_value', None)
                }
            
            if data.ppi_comparison_b_to_a is not None:
                ppi_comp = data.ppi_comparison_b_to_a
                report['ppi_analysis_summary']['comparisons']['b_to_a'] = {
                    'direction': ppi_comp['direction'],
                    'node_jaccard_similarity': ppi_comp['overlap_analysis']['nodes']['jaccard_similarity'],
                    'edge_jaccard_similarity': ppi_comp['overlap_analysis']['edges']['jaccard_similarity'],
                    'novel_connections_count': ppi_comp['novel_connections']['count'],
                    'novel_connections_fraction': ppi_comp['novel_connections']['fraction_of_importance_edges'],
                    'ppi_validation_count': ppi_comp['ppi_validation']['count'],
                    'ppi_validation_fraction': ppi_comp['ppi_validation']['fraction_of_ppi_edges'],
                    'enrichment_significant': ppi_comp.get('enrichment_analysis', {}).get('enrichment_significant', False),
                    'enrichment_p_value': ppi_comp.get('enrichment_analysis', {}).get('hypergeometric_p_value', None)
                }
        
        # Threshold analysis summary
        if hasattr(data, 'threshold_analyses') and data.threshold_analyses:
            report['threshold_analysis_summary'] = {}
            
            for analysis_name, analysis in data.threshold_analyses.items():
                recommendations = analysis.get('recommendations', {})
                best_recommendation = None
                
                # Find the best recommendation (prioritize moderate_density, then target_edges)
                if 'moderate_density' in recommendations:
                    best_recommendation = recommendations['moderate_density']
                elif 'target_edges' in recommendations:
                    best_recommendation = recommendations['target_edges']
                elif 'elbow_method' in recommendations:
                    best_recommendation = recommendations['elbow_method']
                elif recommendations:
                    best_recommendation = list(recommendations.values())[0]
                
                report['threshold_analysis_summary'][analysis_name] = {
                    'edge_count_range': [min(analysis['edge_counts']), max(analysis['edge_counts'])],
                    'density_range': [min(analysis['densities']), max(analysis['densities'])],
                    'recommended_threshold': best_recommendation['threshold'] if best_recommendation else None,
                    'recommended_threshold_label': best_recommendation.get('threshold_label') if best_recommendation else None,
                    'recommended_edge_count': best_recommendation['edge_count'] if best_recommendation else None,
                    'recommended_density': best_recommendation['density'] if best_recommendation else None,
                    'recommendation_reason': best_recommendation['reason'] if best_recommendation else None
                }
        
        # Save report
        report_path = self.output_dir / f"analysis_summary_{self.timestamp}.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        print(f"Summary report saved to: {report_path}")
        return report
    
    def run_full_analysis(self, data: CorrelationAnalysisData, network_params: Dict = None):
        """
        Run the complete analysis pipeline.
        
        Args:
            data: CorrelationAnalysisData object with loaded matrices
            network_params: Dictionary with network analysis parameters:
                - threshold_method: 'self_importance_ratio' or 'absolute_importance'
                - threshold_params: Threshold parameter value
                - network_type: 'directed' or 'undirected'
                - network_layout: Layout algorithm for visualization
                - ppi_analysis_enabled: Whether PPI comparison is enabled
        """
        print("Running full importance matrix analysis...")
        
        # Set default network parameters
        if network_params is None:
            network_params = {
                'threshold_method': 'self_importance_ratio',
                'threshold_params': 10.0,
                'network_type': 'directed',
                'network_layout': 'spring'
            }
        
        # Skip importance matrix analysis - this is a correlation network script
        # Compute cross-platform raw correlations
        if data.truth_a is not None and data.truth_b is not None:
            data = self.compute_cross_platform_raw_correlation(data)
        
        # Compute correlation matrices if raw data is available
        if data.truth_a is not None:
            print("\nComputing correlation matrix for Platform A...")
            data.correlation_matrix_a = self.compute_feature_correlations(data.truth_a)
            
        if data.truth_b is not None:
            print("\nComputing correlation matrix for Platform B...")
            data.correlation_matrix_b = self.compute_feature_correlations(data.truth_b)
        
        # Network analysis
        if NETWORKX_AVAILABLE:
            print("\nRunning correlation network analysis...")
            
            # Determine threshold strategy
            provided_threshold = network_params.get('correlation_threshold')
            use_absolute = network_params.get('use_absolute_correlation', True)
            threshold_analyses: Dict[str, Dict] = {}
            ppi_density = None
            if network_params.get('ppi_analysis_enabled', False) and data.ppi_reference is not None:
                try:
                    ppi_density = nx.density(data.ppi_reference)
                    print(f"PPI reference density: {ppi_density:.6f}")
                except Exception:
                    ppi_density = None
            
            # Analyze thresholds when not provided
            thresholds_for_build = {'A': provided_threshold, 'B': provided_threshold}
            if provided_threshold is None:
                # Analyze thresholds to match target density
                target_density = ppi_density if ppi_density is not None else network_params.get('target_density', 0.0366)
                
                if data.correlation_matrix_a is not None:
                    print(f"\nAnalyzing correlation thresholds for {data.platform_a_name} (target density {target_density:.6f})...")
                    analysis_a = self.analyze_correlation_thresholds(
                        data.correlation_matrix_a, target_density=target_density, use_absolute=use_absolute)
                    threshold_analyses[f'A_correlation'] = analysis_a
                    # Select threshold by closest density to target
                    if analysis_a.get('densities'):
                        idx = int(np.argmin(np.abs(np.array(analysis_a['densities']) - target_density)))
                        thresholds_for_build['A'] = analysis_a['thresholds'][idx]
                        print(f"→ Selected threshold for {data.platform_a_name}: {thresholds_for_build['A']:.3f} (density {analysis_a['densities'][idx]:.6f})")
                        # Report recommendations
                        if 'target_density' in analysis_a.get('recommendations', {}):
                            rec = analysis_a['recommendations']['target_density']
                            print(f"   Target density match: r={rec['threshold']:.3f} ({rec['node_count']} nodes, {rec['edge_count']} edges)")
                            
                if data.correlation_matrix_b is not None:
                    print(f"\nAnalyzing correlation thresholds for {data.platform_b_name} (target density {target_density:.6f})...")
                    analysis_b = self.analyze_correlation_thresholds(
                        data.correlation_matrix_b, target_density=target_density, use_absolute=use_absolute)
                    threshold_analyses[f'B_correlation'] = analysis_b
                    if analysis_b.get('densities'):
                        idx = int(np.argmin(np.abs(np.array(analysis_b['densities']) - target_density)))
                        thresholds_for_build['B'] = analysis_b['thresholds'][idx]
                        print(f"→ Selected threshold for {data.platform_b_name}: {thresholds_for_build['B']:.3f} (density {analysis_b['densities'][idx]:.6f})")
                        if 'target_density' in analysis_b.get('recommendations', {}):
                            rec = analysis_b['recommendations']['target_density']
                            print(f"   Target density match: r={rec['threshold']:.3f} ({rec['node_count']} nodes, {rec['edge_count']} edges)")
            
            # Store threshold analyses
            data.threshold_analyses = threshold_analyses
            
            # Build correlation networks using selected thresholds
            if data.correlation_matrix_a is not None:
                print(f"\nBuilding correlation network for {data.platform_a_name}...")
                data.network_a_to_b = self.build_correlation_network(
                    data.correlation_matrix_a, 
                    threshold=thresholds_for_build['A'],
                    use_absolute=use_absolute
                )
                if data.network_a_to_b is not None:
                    data.network_analysis_a_to_b = self.analyze_network_topology(data.network_a_to_b)
            
            if data.correlation_matrix_b is not None:
                print(f"\nBuilding correlation network for {data.platform_b_name}...")
                data.network_b_to_a = self.build_correlation_network(
                    data.correlation_matrix_b, 
                    threshold=thresholds_for_build['B'],
                    use_absolute=use_absolute
                )
                if data.network_b_to_a is not None:
                    data.network_analysis_b_to_a = self.analyze_network_topology(data.network_b_to_a)
            
            # Compare networks if both are available
            if (data.network_a_to_b is not None and 
                data.network_b_to_a is not None and 
                data.network_analysis_a_to_b is not None and 
                data.network_analysis_b_to_a is not None):
                data.network_comparison = self.compare_networks(
                    data.network_a_to_b, data.network_b_to_a,
                    data.network_analysis_a_to_b, data.network_analysis_b_to_a
                )
            
            # PPI comparison analysis
            if network_params.get('ppi_analysis_enabled', False) and data.ppi_reference is not None:
                print("\nRunning PPI comparison analysis...")
                
                if data.network_a_to_b is not None:
                    data.ppi_comparison_a_to_b = self.compare_network_with_ppi(
                        data.network_a_to_b, data.ppi_reference,
                        f"{data.platform_a_name} → {data.platform_b_name}"
                    )
                
                if data.network_b_to_a is not None:
                    data.ppi_comparison_b_to_a = self.compare_network_with_ppi(
                        data.network_b_to_a, data.ppi_reference,
                        f"{data.platform_b_name} → {data.platform_a_name}"
                    )
        else:
            print("\nSkipping network analysis - NetworkX not available")
        
        # Generate plots relevant to correlation networks
        figures = {}
        
        # Skip importance matrix plots - not relevant for correlation networks
        
        # Network analysis plots
        if NETWORKX_AVAILABLE:
            try:
                if data.network_a_to_b is not None and data.network_analysis_a_to_b is not None:
                    figures['network_a_to_b'] = self.plot_importance_network(
                        data.network_a_to_b, data.network_analysis_a_to_b, 
                        f"{data.platform_a_name} → {data.platform_b_name}",
                        layout_type=network_params['network_layout']
                    )
            except Exception as e:
                print(f"Error generating A→B network plot: {e}")
                
            try:
                if data.network_b_to_a is not None and data.network_analysis_b_to_a is not None:
                    figures['network_b_to_a'] = self.plot_importance_network(
                        data.network_b_to_a, data.network_analysis_b_to_a, 
                        f"{data.platform_b_name} → {data.platform_a_name}",
                        layout_type=network_params['network_layout']
                    )
            except Exception as e:
                print(f"Error generating B→A network plot: {e}")
                
            try:
                figures['network_topology_analysis'] = self.plot_network_topology_analysis(data)
            except Exception as e:
                print(f"Error generating network topology analysis: {e}")
                
            try:
                figures['network_hub_analysis'] = self.plot_network_hub_analysis(data)
            except Exception as e:
                print(f"Error generating network hub analysis: {e}")
                
            # PPI comparison plots
            if network_params.get('ppi_analysis_enabled', False) and data.ppi_reference is not None:
                try:
                    figures['ppi_network_comparison'] = self.plot_ppi_network_comparison(data)
                except Exception as e:
                    print(f"Error generating PPI network comparison: {e}")
                    
                try:
                    figures['ppi_validation_networks'] = self.plot_ppi_validation_networks(data)
                except Exception as e:
                    print(f"Error generating PPI validation networks: {e}")
            
            # Threshold analysis plots
            try:
                if hasattr(data, 'threshold_analyses') and data.threshold_analyses:
                    figures['threshold_analysis'] = self.plot_threshold_analysis(data.threshold_analyses, 
                                                                                     network_params.get('target_density', 0.0366))
                    figures['threshold_recommendations'] = self.plot_threshold_recommendations(data.threshold_analyses)
                    figures['threshold_pr_curves'] = self.plot_threshold_pr_curves(data, data.threshold_analyses)
            except Exception as e:
                print(f"Error generating threshold analysis plots: {e}")
        
        # Generate summary report
        try:
            report = self.generate_summary_report(data)
        except Exception as e:
            print(f"Error generating summary report: {e}")
            report = None
        
        print(f"Analysis completed! Generated {len([f for f in figures.values() if f is not None])} figures")
        
        # Export networks to files
        try:
            self.export_networks(data)
        except Exception as e:
            print(f"Error exporting networks: {e}")
        
        return figures, report
    
    def load_feature_mapping(self, data: CorrelationAnalysisData, 
                           mapping_file_path: str) -> CorrelationAnalysisData:
        """Load feature mapping from numeric IDs to gene names"""
        if not mapping_file_path:
            return data
            
        try:
            print(f"Loading feature mapping from: {mapping_file_path}")
            
            # Try different possible formats
            if mapping_file_path.endswith('.csv'):
                mapping_df = pd.read_csv(mapping_file_path)
                
                # Assume first column is numeric ID, second is gene name
                if len(mapping_df.columns) >= 2:
                    data.feature_mapping = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
                    print(f"  Loaded mapping for {len(data.feature_mapping)} features")
                    print(f"  Example mappings: {dict(list(data.feature_mapping.items())[:5])}")
                else:
                    print("  Error: Mapping file should have at least 2 columns (ID, Name)")
                    
            elif mapping_file_path.endswith('.json'):
                import json
                with open(mapping_file_path, 'r') as f:
                    data.feature_mapping = json.load(f)
                print(f"  Loaded mapping for {len(data.feature_mapping)} features")
                
        except Exception as e:
            print(f"  Error loading feature mapping: {e}")
            
        return data

    def export_networks(self, data: CorrelationAnalysisData) -> None:
        """
        Export constructed networks to files in multiple formats.
        
        Args:
            data: CorrelationAnalysisData object with constructed networks
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available, skipping network export")
            return
            
        print("Exporting networks to files...")
        
        # Create networks subdirectory
        networks_dir = Path(self.output_dir) / "networks"
        networks_dir.mkdir(exist_ok=True)
        
        networks_to_export = []
        if data.network_a_to_b is not None:
            networks_to_export.append((
                data.network_a_to_b, 
                f"{data.platform_a_name}_to_{data.platform_b_name}".replace(" ", "_"),
                f"{data.platform_a_name} → {data.platform_b_name}"
            ))
        
        if data.network_b_to_a is not None:
            networks_to_export.append((
                data.network_b_to_a, 
                f"{data.platform_b_name}_to_{data.platform_a_name}".replace(" ", "_"),
                f"{data.platform_b_name} → {data.platform_a_name}"
            ))
        
        for network, file_prefix, network_name in networks_to_export:
            print(f"  Exporting {network_name} network...")
            
            # Export as edge list (TSV)
            edge_list_path = networks_dir / f"{file_prefix}_edges.tsv"
            try:
                edges_data = []
                for source, target, attrs in network.edges(data=True):
                    edges_data.append({
                        'source': source,
                        'target': target,
                        'weight': attrs.get('weight', 1.0),
                        'importance': attrs.get('importance', 0.0)
                    })
                
                edges_df = pd.DataFrame(edges_data)
                edges_df.to_csv(edge_list_path, sep='\t', index=False)
                print(f"    Edge list: {edge_list_path}")
                
            except Exception as e:
                print(f"    Error exporting edge list: {e}")
            
            # Export nodes with attributes (TSV)
            node_list_path = networks_dir / f"{file_prefix}_nodes.tsv"
            try:
                nodes_data = []
                for node, attrs in network.nodes(data=True):
                    node_data = {'node': node}
                    node_data.update(attrs)
                    nodes_data.append(node_data)
                
                nodes_df = pd.DataFrame(nodes_data)
                nodes_df.to_csv(node_list_path, sep='\t', index=False)
                print(f"    Node list: {node_list_path}")
                
            except Exception as e:
                print(f"    Error exporting node list: {e}")
            
            # Export as GraphML (if possible)
            graphml_path = networks_dir / f"{file_prefix}_network.graphml"
            try:
                nx.write_graphml(network, graphml_path)
                print(f"    GraphML: {graphml_path}")
                
            except Exception as e:
                print(f"    Error exporting GraphML: {e}")
            
            # Export simple edge list (for compatibility)
            simple_edge_path = networks_dir / f"{file_prefix}_simple_edges.txt"
            try:
                with open(simple_edge_path, 'w') as f:
                    for source, target in network.edges():
                        f.write(f"{source}\t{target}\n")
                print(f"    Simple edge list: {simple_edge_path}")
                
            except Exception as e:
                print(f"    Error exporting simple edge list: {e}")
        
        # Export network summary statistics
        summary_path = networks_dir / "network_summary.yaml"
        try:
            summary_data = {
                'export_timestamp': datetime.now().isoformat(),
                'networks': {}
            }
            
            for network, file_prefix, network_name in networks_to_export:
                summary_data['networks'][file_prefix] = {
                    'name': network_name,
                    'nodes': len(network.nodes()),
                    'edges': len(network.edges()),
                    'density': nx.density(network),
                    'is_directed': network.is_directed(),
                    'files': {
                        'edges': f"{file_prefix}_edges.tsv",
                        'nodes': f"{file_prefix}_nodes.tsv", 
                        'graphml': f"{file_prefix}_network.graphml",
                        'simple_edges': f"{file_prefix}_simple_edges.txt"
                    }
                }
            
            with open(summary_path, 'w') as f:
                yaml.dump(summary_data, f, default_flow_style=False)
            print(f"  Network summary: {summary_path}")
            
        except Exception as e:
            print(f"  Error exporting network summary: {e}")
        
        print(f"Network export completed! Files saved to: {networks_dir}")


def main():
    """
    Main function for correlation-based network analysis.
    
    Builds networks from raw data correlations, computes network
    topology metrics, and generates visualization reports.
    """
    parser = argparse.ArgumentParser(description='Build and analyze correlation networks from raw data')
    
    parser.add_argument('--importance_a_to_b', type=str, default=None,
                       help='Path to importance matrix CSV file (A → B) - optional for comparison')
    parser.add_argument('--importance_b_to_a', type=str, default=None,
                       help='Path to importance matrix CSV file (B → A) - optional for comparison')
    
    # Raw data arguments (REQUIRED for correlation networks)
    parser.add_argument('--truth_a', type=str, default=None,
                       help='Path to truth data CSV file for platform A (REQUIRED)')
    parser.add_argument('--truth_b', type=str, default=None,
                       help='Path to truth data CSV file for platform B (REQUIRED)')
    parser.add_argument('--imp_a_m1', type=str, default=None,
                       help='Path to imputed data CSV file for platform A, method 1')
    parser.add_argument('--imp_a_m2', type=str, default=None,
                       help='Path to imputed data CSV file for platform A, method 2')
    parser.add_argument('--imp_b_m1', type=str, default=None,
                       help='Path to imputed data CSV file for platform B, method 1')
    parser.add_argument('--imp_b_m2', type=str, default=None,
                       help='Path to imputed data CSV file for platform B, method 2')
    parser.add_argument('--feature_mapping', type=str, default=None,
                       help='Path to feature mapping file (CSV or JSON) to map numeric IDs to gene names')
    
    # Analysis parameters
    parser.add_argument('--platform_a_name', type=str, default='Platform A',
                       help='Name for platform A in plots')
    parser.add_argument('--platform_b_name', type=str, default='Platform B',
                       help='Name for platform B in plots')
    parser.add_argument('--output_dir', type=str, default='correlation_network_analysis',
                       help='Output directory for results')
    
    # Correlation network parameters
    parser.add_argument('--correlation_threshold', type=float, default=None,
                       help='Correlation threshold for creating edges. If omitted, auto-chosen to match target density.')
    parser.add_argument('--use_absolute_correlation', action='store_true', default=True,
                       help='Use absolute correlation values for thresholding (default: True)')
    parser.add_argument('--network_layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai'],
                       help='Network layout algorithm for visualization')
    
    # PPI reference analysis parameters
    parser.add_argument('--ppi_reference', type=str, default=None,
                       help='Path to PPI reference file (tab-delimited with symbol1, symbol2 columns)')
    parser.add_argument('--ppi_symbol1_col', type=str, default='symbol1',
                       help='Column name for first protein symbol in PPI file')
    parser.add_argument('--ppi_symbol2_col', type=str, default='symbol2',
                       help='Column name for second protein symbol in PPI file')
    parser.add_argument('--ppi_confidence_col', type=str, default=None,
                       help='Column name for confidence scores in PPI file (optional)')
    parser.add_argument('--ppi_confidence_threshold', type=float, default=0.0,
                       help='Minimum confidence threshold for PPI interactions')
    parser.add_argument('--target_density', type=float, default=0.0366,
                       help='Target network density for threshold recommendations (default: 0.0366)')
    
    args = parser.parse_args()
    
    # Check for required data files
    if not args.truth_a and not args.truth_b:
        raise ValueError("At least one truth data file (--truth_a or --truth_b) must be provided for correlation network analysis")
    
    # Create analyzer
    analyzer = CorrelationNetworkAnalyzer(output_dir=args.output_dir)
    
    # Prepare file paths
    file_paths = {
        'importance_a_to_b': args.importance_a_to_b,
        'importance_b_to_a': args.importance_b_to_a
    }
    
    # Load data
    data = analyzer.load_importance_matrices(
        file_paths=file_paths,
        platform_a_name=args.platform_a_name,
        platform_b_name=args.platform_b_name
    )
    
    # Load raw data and compute feature performance if provided
    raw_data_provided = any([args.truth_a, args.truth_b, args.imp_a_m1, 
                           args.imp_a_m2, args.imp_b_m1, args.imp_b_m2])
    
    if raw_data_provided:
        raw_data_file_paths = {
            'truth_a': args.truth_a,
            'truth_b': args.truth_b,
            'imp_a_m1': args.imp_a_m1,
            'imp_a_m2': args.imp_a_m2,
            'imp_b_m1': args.imp_b_m1,
            'imp_b_m2': args.imp_b_m2
        }
        data = analyzer.load_raw_data_for_performance(data, raw_data_file_paths)
        
        # Load feature mapping if provided
        if args.feature_mapping:
            data = analyzer.load_feature_mapping(data, args.feature_mapping)
        
        data = analyzer.compute_feature_performance(data)

        # Also compute cross-platform raw correlation for overlapping proteins
        data = analyzer.compute_cross_platform_raw_correlation(data)
    
    # Load PPI reference if provided
    if args.ppi_reference and NETWORKX_AVAILABLE:
        ppi_network = analyzer.load_ppi_reference(
            ppi_file_path=args.ppi_reference,
            symbol1_col=args.ppi_symbol1_col,
            symbol2_col=args.ppi_symbol2_col,
            confidence_col=args.ppi_confidence_col,
            confidence_threshold=args.ppi_confidence_threshold
        )
        
        if ppi_network is not None:
            data.ppi_reference = ppi_network
            data.ppi_reference_stats = analyzer.analyze_ppi_reference_stats(ppi_network)
    
    # Set network analysis parameters
    network_params = {
        'correlation_threshold': args.correlation_threshold,
        'use_absolute_correlation': args.use_absolute_correlation,
        'network_layout': args.network_layout,
        'ppi_analysis_enabled': args.ppi_reference is not None,
        'target_density': args.target_density
    }
    
    # Run full analysis
    figures, report = analyzer.run_full_analysis(data, network_params=network_params)
    
    print(f"\nCorrelation network analysis completed!")
    print(f"Results saved to: {analyzer.output_dir}")
    print(f"Git hash: {analyzer.git_hash}")
    print(f"Timestamp: {analyzer.timestamp}")
    
    # Print key findings
    if report and 'rank_consistency_summary' in report:
        print("\n=== KEY FINDINGS ===")
        
        for direction in ['a_to_b', 'b_to_a']:
            if direction in report['rank_consistency_summary']:
                direction_name = f"{data.platform_a_name} → {data.platform_b_name}" if direction == 'a_to_b' else f"{data.platform_b_name} → {data.platform_a_name}"
                summary = report['rank_consistency_summary'][direction]
                
                print(f"\n{direction_name}:")
                print(f"  Average rank variance: {summary['average_rank_variance']:.2f}")
                print(f"  Features that are sometimes top: {summary['features_always_top']}")
                print(f"  Features that are never top: {summary['features_never_top']}")
                
                print(f"  Most consistently important features:")
                most_consistent = summary['most_consistently_important']
                
                # Handle both dictionary formats
                if isinstance(most_consistent, dict) and most_consistent:
                    # Check if it's nested (feature_name -> stats) or flat (feature_name directly as key with stats as dict)
                    first_key = list(most_consistent.keys())[0]
                    if isinstance(most_consistent[first_key], dict) and 'mean_rank' in most_consistent[first_key]:
                        # New format: feature_name -> {mean_rank: x, rank_variance: y}
                        for feature_name in list(most_consistent.keys())[:3]:
                            mean_rank = most_consistent[feature_name]['mean_rank']
                            variance = most_consistent[feature_name]['rank_variance']
                            print(f"    {feature_name}: rank {mean_rank:.1f} ± {variance:.1f}")
                    else:
                        # Old format: different structure, try to extract info
                        print("    Data format not recognized for consistent features")
                else:
                    print("    No consistently important features data available")
        
        # Print self-feature importance findings
        if 'self_feature_importance_summary' in report:
            print("\n=== SELF-FEATURE IMPORTANCE FINDINGS ===")
            
            for direction in ['a_to_b', 'b_to_a']:
                if direction in report['self_feature_importance_summary']:
                    direction_name = f"{data.platform_a_name} → {data.platform_b_name}" if direction == 'a_to_b' else f"{data.platform_b_name} → {data.platform_a_name}"
                    summary = report['self_feature_importance_summary'][direction]
                    
                    print(f"\n{direction_name}:")
                    print(f"  Overlapping features: {summary['n_overlapping_features']}")
                    if summary['avg_self_importance'] is not None:
                        print(f"  Average self-importance: {summary['avg_self_importance']:.6f}")
                    if summary['avg_cross_importance'] is not None:
                        print(f"  Average cross-importance: {summary['avg_cross_importance']:.6f}")
                    if summary['self_vs_cross_ratio'] is not None:
                        print(f"  Self vs Cross ratio: {summary['self_vs_cross_ratio']:.2f}")
                        ratio_msg = "HIGHER" if summary['self_vs_cross_ratio'] > 1 else "LOWER"
                        print(f"    → Self-importance is {ratio_msg} than cross-importance")
                    
                    if summary['top_self_features']:
                        print(f"  Top self-important features:")
                        for feature, score in list(summary['top_self_features'].items())[:3]:
                            print(f"    {feature}: {score:.6f}")
    
    # Skip self-importance analysis - not relevant for correlation networks
    
    # Print network analysis findings
    if NETWORKX_AVAILABLE and report and 'network_analysis_summary' in report:
        print("\n=== NETWORK ANALYSIS FINDINGS ===")
        
        for direction in ['a_to_b', 'b_to_a']:
            if direction in report['network_analysis_summary']:
                direction_name = f"{data.platform_a_name} → {data.platform_b_name}" if direction == 'a_to_b' else f"{data.platform_b_name} → {data.platform_a_name}"
                summary = report['network_analysis_summary'][direction]
                
                print(f"\n{direction_name} Network:")
                print(f"  Nodes: {summary['n_nodes']}, Edges: {summary['n_edges']}")
                print(f"  Network density: {summary['density']:.4f}")
                print(f"  Mean degree: {summary.get('mean_degree', 0):.2f}")
                
                if summary.get('n_communities', 0) > 0:
                    print(f"  Communities detected: {summary['n_communities']}")
                    print(f"  Modularity: {summary.get('modularity', 0):.3f}")
                
                if summary.get('top_hub_proteins'):
                    print(f"  Top hub proteins:")
                    for protein, centrality in list(summary['top_hub_proteins'].items())[:3]:
                        print(f"    {protein}: {centrality:.4f}")
        
        # Network comparison
        if 'network_comparison' in report['network_analysis_summary']:
            comparison = report['network_analysis_summary']['network_comparison']
            print(f"\nNetwork Comparison:")
            print(f"  Node overlap (Jaccard): {comparison['node_jaccard_similarity']:.3f}")
            print(f"  Edge overlap (Jaccard): {comparison['edge_jaccard_similarity']:.3f}")
            
            if comparison.get('centrality_correlations'):
                print(f"  Centrality correlations:")
                for centrality_type, corr_data in comparison['centrality_correlations'].items():
                    correlation = corr_data.get('correlation', 0)
                    p_value = corr_data.get('p_value', 1)
                    significance = "*" if p_value < 0.05 else ""
                    print(f"    {centrality_type}: r = {correlation:.3f} (p = {p_value:.3f}){significance}")
    elif not NETWORKX_AVAILABLE:
        print("\n=== NETWORK ANALYSIS ===")
        print("NetworkX not available - network analysis was skipped")
        print("To enable network analysis, install: pip install networkx python-louvain")
    else:
        print("\n=== NETWORK ANALYSIS ===")
        print("Network analysis was disabled or no networks were generated")
    
    # Print PPI analysis findings
    if NETWORKX_AVAILABLE and report and 'ppi_analysis_summary' in report:
        print("\n=== PPI REFERENCE ANALYSIS FINDINGS ===")
        
        ppi_summary = report['ppi_analysis_summary']
        ppi_stats = ppi_summary.get('ppi_reference_stats', {})
        
        if ppi_stats:
            print(f"\nPPI Reference Network:")
            print(f"  Nodes: {ppi_stats.get('n_nodes', 0)}, Edges: {ppi_stats.get('n_edges', 0)}")
            print(f"  Density: {ppi_stats.get('density', 0):.4f}")
            print(f"  Connected components: {ppi_stats.get('n_connected_components', 0)}")
            
            if ppi_stats.get('confidence_stats'):
                conf_stats = ppi_stats['confidence_stats']
                print(f"  Confidence range: {conf_stats['min_confidence']:.3f} - {conf_stats['max_confidence']:.3f}")
                print(f"  Mean confidence: {conf_stats['mean_confidence']:.3f}")
        
        # PPI comparison results
        comparisons = ppi_summary.get('comparisons', {})
        for direction in ['a_to_b', 'b_to_a']:
            if direction in comparisons:
                direction_name = f"{data.platform_a_name} → {data.platform_b_name}" if direction == 'a_to_b' else f"{data.platform_b_name} → {data.platform_a_name}"
                comp = comparisons[direction]
                
                print(f"\n{direction_name} vs PPI Reference:")
                print(f"  Node overlap (Jaccard): {comp['node_jaccard_similarity']:.3f}")
                print(f"  Edge overlap (Jaccard): {comp['edge_jaccard_similarity']:.3f}")
                print(f"  Novel connections: {comp['novel_connections_count']} ({comp['novel_connections_fraction']:.1%} of importance edges)")
                print(f"  PPI-validated connections: {comp['ppi_validation_count']} ({comp['ppi_validation_fraction']:.1%} of PPI edges)")
                
                if comp.get('enrichment_p_value') is not None:
                    p_val = comp['enrichment_p_value']
                    significance = "*" if comp.get('enrichment_significant', False) else ""
                    print(f"  PPI enrichment: p = {p_val:.3e}{significance}" if p_val < 0.001 else f"  PPI enrichment: p = {p_val:.3f}{significance}")
    elif args.ppi_reference:
        print("\n=== PPI REFERENCE ANALYSIS ===")
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available - PPI analysis was skipped")
        else:
            print("PPI reference file provided but analysis failed or was disabled")
    else:
        print("\n=== PPI REFERENCE ANALYSIS ===")
        print("No PPI reference file provided - skipping PPI validation analysis")
        print("To enable PPI analysis, use: --ppi_reference path/to/ppi_file.txt")
    
    # Print threshold analysis findings
    if NETWORKX_AVAILABLE and report and 'threshold_analysis_summary' in report:
        print("\n=== THRESHOLD ANALYSIS FINDINGS ===")
        
        threshold_summary = report['threshold_analysis_summary']
        
        for analysis_name, analysis_data in threshold_summary.items():
            print(f"\n{analysis_name.replace('_', ' ').title()}:")
            
            edge_range = analysis_data['edge_count_range']
            density_range = analysis_data['density_range']
            
            print(f"  Edge count range: {edge_range[0]:,} - {edge_range[1]:,}")
            print(f"  Density range: {density_range[0]:.6f} - {density_range[1]:.6f}")
            
            if analysis_data.get('recommended_threshold'):
                threshold_val = analysis_data['recommended_threshold']
                
                # For display, show threshold label if available, otherwise the numeric value
                display_threshold = analysis_data.get('recommended_threshold_label', threshold_val)
                print(f"  RECOMMENDED THRESHOLD: {display_threshold}")
                print(f"    → Edges: {analysis_data['recommended_edge_count']:,}")
                print(f"    → Density: {analysis_data['recommended_density']:.4f}")
                print(f"    → Reason: {analysis_data['recommendation_reason']}")
                
                # Add specific usage instructions based on analysis type (always use numeric value)
                if 'absolute' in analysis_name.lower():
                    if isinstance(threshold_val, (int, float)):
                        print(f"    → USAGE: --threshold_method absolute_importance --threshold_params {threshold_val:.6f}")
                    else:
                        print(f"    → USAGE: --threshold_method absolute_importance --threshold_params {threshold_val}")
                elif 'self_importance' in analysis_name.lower():
                    if isinstance(threshold_val, (int, float)):
                        print(f"    → USAGE: --threshold_method self_importance_ratio --threshold_params {threshold_val:.1f}")
                    else:
                        print(f"    → USAGE: --threshold_method self_importance_ratio --threshold_params {threshold_val}")
        
        print(f"\n💡 GUIDANCE FOR NETWORK CONSTRUCTION:")
        print(f"   Based on the threshold analysis, use the USAGE commands shown above.")
        print(f"   Different threshold methods:")
        print(f"   • Self-importance ratio: Connect if cross-importance > threshold_params × self-importance")
        print(f"   • Absolute importance: Connect if importance > threshold_params absolute value")
        print(f"   Networks with 1K-10K edges typically work well for visualization and analysis.")
        
    elif NETWORKX_AVAILABLE and hasattr(data, 'threshold_analyses'):
        print("\n=== THRESHOLD ANALYSIS ===")
        print("Threshold analysis was performed - check the generated plots:")
        print("  - threshold_analysis_*.png: Edge counts and densities vs thresholds")
        print("  - threshold_recommendations_*.png: Recommended threshold parameters")
    else:
        print("\n=== THRESHOLD ANALYSIS ===")
        print("Threshold analysis was not performed or NetworkX not available")


if __name__ == "__main__":
    main() 