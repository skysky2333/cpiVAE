#!/usr/bin/env python3
"""
Single-Platform Latent Space Analysis for VAE Models
Analyzes what the VAE has learned in its latent representation for a single platform,
with emphasis on structure preservation through K-means clustering analysis.
"""

import argparse
import os
import sys
import warnings
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
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

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
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

NATURE_COLORS = {
    'primary': '#e64b35',      # Red (230,75,53)
    'secondary': '#4dbbd5',    # Light Blue (77,187,213)
    'accent': '#00a087',       # Teal (0,160,135)
    'neutral': '#3c5488',      # Dark Blue (60,84,136)
    'highlight': '#f39b7f',    # Light Red (243,155,127)
    'alternative_1': '#e64b35', # Red (same as primary for consistency)
    'alternative_2': '#4dbbd5', # Light Blue (same as secondary)
    'alternative_3': '#00a087', # Teal (same as accent)
    'alternative_4': '#3c5488', # Dark Blue (same as neutral)
    'alternative_5': '#f39b7f'  # Light Red (same as highlight)
}

@dataclass
class LatentAnalysisData:
    """Container for single-platform latent space analysis data.
    
    Attributes:
        latent: DataFrame with samples as rows and latent dimensions as columns
        truth: DataFrame with original data for comparison
        platform_name: Display name for the platform being analyzed
        groups: Optional series containing group labels for samples
        embeddings: Dictionary of computed 2D embeddings (PCA, UMAP, t-SNE)
        kmeans_clusters: Array of cluster assignments from K-means clustering
        optimal_k: Number of clusters determined for K-means
    """
    latent: pd.DataFrame
    truth: pd.DataFrame
    platform_name: str
    groups: Optional[pd.Series] = None
    embeddings: Dict[str, np.ndarray] = None
    kmeans_clusters: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None

class LatentSpaceAnalyzer:
    """Main class for single-platform latent space analysis.
    
    Provides comprehensive analysis of VAE latent representations including
    structure visualization, clustering analysis, dimension analysis, traversal
    analysis, and interpolation studies.
    """
    
    def __init__(self, output_dir: str = "latent_analysis_oneplatform_output"):
        """Initialize the latent space analyzer.
        
        Args:
            output_dir: Directory to save analysis results and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.git_hash = self._get_git_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _get_git_hash(self) -> str:
        """Get current git commit hash for reproducibility.
        
        Returns:
            Short git commit hash or 'unknown' if not available
        """
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def load_and_validate_data(self, file_paths: Dict[str, str], 
                              platform_name: str,
                              transpose_latent: bool = False) -> LatentAnalysisData:
        """Load and validate all input files for single platform analysis.
        
        Args:
            file_paths: Dictionary containing paths to required files (latent, truth, groups)
            platform_name: Display name for the platform
            transpose_latent: Whether to transpose the latent file
            
        Returns:
            LatentAnalysisData object with loaded and validated data
            
        Raises:
            ValueError: If no common samples found between latent and truth data
        """
        print("Loading single-platform latent space and validation data...")
        
        print(f"  Loading latent space: {file_paths['latent']}")
        latent = pd.read_csv(file_paths['latent'], index_col=0)
        
        if transpose_latent:
            latent = latent.T
            print(f"    Transposed latent space")
        
        print(f"    Latent shape: {latent.shape}")
        
        print(f"  Loading truth data: {file_paths['truth']}")
        truth = pd.read_csv(file_paths['truth'], index_col=0)
        
        groups = None
        if 'groups' in truth.index:
            print("  Found 'groups' row in truth data - extracting group information")
            groups = truth.loc['groups'].copy()
            truth = truth.drop('groups')
            print(f"    Extracted groups for {len(groups)} samples")
        elif 'groups' in file_paths and file_paths['groups']:
            print(f"  Loading groups from separate file: {file_paths['groups']}")
            groups = pd.read_csv(file_paths['groups'], index_col=0, header=None).iloc[:, 0]
        
        print(f"    Latent index (first 5): {list(latent.index[:5])}")
        print(f"    Truth index (first 5): {list(truth.index[:5])}")
        print(f"    Truth columns (first 5): {list(truth.columns[:5])}")
        
        common_samples = list(set(latent.index) & set(truth.index))
        
        print(f"  Common samples: {len(common_samples)}")
        
        if len(common_samples) == 0:
            print(f"  ERROR: No common samples found!")
            print(f"    Latent samples: {len(latent.index)}")
            print(f"    Truth samples: {len(truth.index)}")
            raise ValueError("No common samples found between latent and truth data!")
        
        common_samples = sorted(common_samples)
        latent = latent.loc[common_samples]
        truth = truth.loc[common_samples]
        
        if groups is not None:
            groups = groups.loc[common_samples]
        
        analysis_data = LatentAnalysisData(
            latent=latent,
            truth=truth,
            platform_name=platform_name,
            groups=groups
        )
        
        return analysis_data
    
    def compute_embeddings(self, data: LatentAnalysisData) -> LatentAnalysisData:
        """Compute 2D embeddings using multiple dimensionality reduction methods.
        
        Args:
            data: LatentAnalysisData object containing latent representations
            
        Returns:
            Updated LatentAnalysisData object with computed embeddings
        """
        print("Computing 2D embeddings for visualization...")
        
        embeddings = {}
        latent_data = data.latent.values
        print(f"  Latent space shape: {latent_data.shape}")
        
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(latent_data)
        embeddings['pca'] = {
            'embedding': pca_embedding,
            'explained_variance': pca.explained_variance_ratio_,
            'model': pca
        }
        
        if UMAP_AVAILABLE:
            print("  Computing UMAP...")
            try:
                umap_model = umap.UMAP(n_components=2, random_state=42, 
                                      n_neighbors=15, min_dist=0.1)
                umap_embedding = umap_model.fit_transform(latent_data)
                embeddings['umap'] = {
                    'embedding': umap_embedding,
                    'model': umap_model
                }
            except Exception as e:
                print(f"    UMAP failed: {e}")
                embeddings['umap'] = None
        else:
            print("  Skipping UMAP (not available)")
            embeddings['umap'] = None
        
        print("  Computing t-SNE...")
        try:
            tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_embedding = tsne_model.fit_transform(latent_data)
            embeddings['tsne'] = {
                'embedding': tsne_embedding,
                'model': tsne_model
            }
        except Exception as e:
            print(f"    t-SNE failed: {e}")
            embeddings['tsne'] = None
        
        data.embeddings = embeddings
        return data
    
    def perform_kmeans_clustering(self, data: LatentAnalysisData, n_clusters: int = None) -> LatentAnalysisData:
        """Perform K-means clustering on original data space.
        
        Args:
            data: LatentAnalysisData object containing truth data for clustering
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Updated LatentAnalysisData object with cluster assignments
        """
        print("Performing K-means clustering on original data space...")
        
        truth_data = data.truth.T.values
        
        col_means = np.nanmean(truth_data, axis=0)
        inds = np.where(np.isnan(truth_data))
        truth_data[inds] = np.take(col_means, inds[1])
        
        scaler = StandardScaler()
        truth_scaled = scaler.fit_transform(truth_data)
        
        if n_clusters is None:
            n_samples = truth_scaled.shape[0]
            optimal_k = min(max(3, n_samples // 20), 8)
        else:
            optimal_k = n_clusters
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(truth_scaled)
        
        print(f"  K-means clustering: {optimal_k} clusters for {truth_scaled.shape[0]} samples")
        
        data.kmeans_clusters = cluster_labels
        data.optimal_k = optimal_k
        
        return data
    
    def save_figure(self, fig: plt.Figure, name: str, **kwargs):
        """Save figure with metadata in both PDF and PNG formats.
        
        Args:
            fig: Matplotlib figure object to save
            name: Base filename for the saved figure
            **kwargs: Additional arguments passed to savefig
        """
        fig_path = self.output_dir / "figures" / f"{name}.pdf"
        png_path = self.output_dir / "figures" / f"{name}.png"
        
        metadata = {
            'Title': name,
            'Creator': 'Single-Platform Latent Space Analysis',
            'Subject': 'VAE Latent Space Exploration',
            'Keywords': 'proteomics, VAE, latent space, single-platform',
            'Git_Hash': self.git_hash,
            'Timestamp': self.timestamp
        }
        
        fig.savefig(fig_path, format='pdf', metadata=metadata, **kwargs)
        fig.savefig(png_path, format='png', dpi=300, **kwargs)
        
        print(f"  Figure saved: {fig_path}")
    
    def generate_latent_structure_analysis(self, data: LatentAnalysisData):
        """Generate latent space structure visualization.
        
        Creates a comprehensive visualization showing the latent space structure
        using different dimensionality reduction methods, colored by groups and
        K-means clusters.
        
        Args:
            data: LatentAnalysisData object with computed embeddings
            
        Returns:
            Matplotlib figure object or None if embeddings not available
        """
        print("Generating latent space structure analysis...")
        
        if data.embeddings is None:
            print("  No embeddings computed - skipping")
            return
        
        n_methods = sum(1 for emb in data.embeddings.values() if emb is not None)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Latent Space Structure Analysis: {data.platform_name}', fontsize=16, fontweight='bold')
        
        method_idx = 0
        for method_name, embedding_data in data.embeddings.items():
            if embedding_data is None:
                continue
                
            embedding = embedding_data['embedding']
            
            ax1 = axes[0, method_idx]
            
            if data.groups is not None:
                unique_groups = data.groups.unique()
                palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                                NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                                NATURE_COLORS['highlight']]
                group_colors = [palette_colors[i % len(palette_colors)] for i in range(len(unique_groups))]
                
                for i, group in enumerate(unique_groups):
                    mask = data.groups.values == group
                    ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=group_colors[i], alpha=0.7, s=10, 
                              label=group, edgecolors='black', linewidth=0.1)
                
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.set_title(f'{method_name.upper()} - Colored by Groups')
            else:
                ax1.scatter(embedding[:, 0], embedding[:, 1], 
                          c=NATURE_COLORS['primary'], alpha=0.7, s=10, 
                          edgecolors='black', linewidth=0.1)
                ax1.set_title(f'{method_name.upper()} - All Samples')
            
            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax1.set_xlabel(f'{method_name.upper()} 1')
                ax1.set_ylabel(f'{method_name.upper()} 2')
            
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1, method_idx]
            
            if data.kmeans_clusters is not None:
                cluster_colors = plt.cm.Set3(np.linspace(0, 1, data.optimal_k))
                
                for cluster_id in range(data.optimal_k):
                    mask = data.kmeans_clusters == cluster_id
                    if np.any(mask) and len(embedding) == len(data.kmeans_clusters):  # Check dimensions match
                        ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                                  c=[cluster_colors[cluster_id]], alpha=0.7, s=10, 
                                  label=f'Cluster {cluster_id+1}', edgecolors='black', linewidth=0.1)
                
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.set_title(f'{method_name.upper()} - Colored by K-means Clusters')
            else:
                ax2.scatter(embedding[:, 0], embedding[:, 1], 
                          c=np.random.rand(len(embedding)), cmap='viridis',
                          alpha=0.7, s=10, edgecolors='black', linewidth=0.1)
                ax2.set_title(f'{method_name.upper()} - Random Coloring')
            
            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax2.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax2.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax2.set_xlabel(f'{method_name.upper()} 1')
                ax2.set_ylabel(f'{method_name.upper()} 2')
            
            ax2.grid(True, alpha=0.3)
            
            method_idx += 1
        
        plt.tight_layout()
        return fig 
    
    def generate_kmeans_clustering_analysis(self, data: LatentAnalysisData):
        """Generate K-means clustering analysis showing structure preservation.
        
        Visualizes how K-means clusters from original data space are preserved
        in the latent space representations, with quality metrics.
        
        Args:
            data: LatentAnalysisData object with clustering results
            
        Returns:
            Matplotlib figure object or None if clustering not available
        """
        print("Generating K-means clustering analysis...")
        
        if data.kmeans_clusters is None:
            print("  No K-means clustering performed - skipping")
            return
        
        if data.embeddings is None:
            print("  No embeddings computed - skipping")
            return
        
        n_methods = sum(1 for emb in data.embeddings.values() if emb is not None)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'K-means Clustering Structure Preservation: {data.platform_name}\n'
                    f'({data.optimal_k} clusters from original data space)', 
                    fontsize=16, fontweight='bold')
        
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, data.optimal_k))
        cluster_color_map = {i: cluster_colors[i] for i in range(data.optimal_k)}
        
        method_idx = 0
        for method_name, embedding_data in data.embeddings.items():
            if embedding_data is None:
                continue
                
            embedding = embedding_data['embedding']
            
            ax1 = axes[0, method_idx]
            
            for cluster_id in range(data.optimal_k):
                mask = data.kmeans_clusters == cluster_id
                cluster_size = np.sum(mask)
                
                if len(embedding) == len(data.kmeans_clusters):  # Check dimensions match
                    ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                               c=[cluster_color_map[cluster_id]], alpha=0.7, s=15, 
                               label=f'Cluster {cluster_id+1} (n={cluster_size})', 
                               edgecolors='black', linewidth=0.1)
            
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax1.set_xlabel(f'{method_name.upper()} 1')
                ax1.set_ylabel(f'{method_name.upper()} 2')
            
            ax1.set_title(f'{method_name.upper()} - Latent Space with Clusters')
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1, method_idx]
            
            latent_distances = squareform(pdist(data.latent.values))
            
            within_cluster_distances = []
            between_cluster_distances = []
            cluster_centers_latent = []
            
            for cluster_id in range(data.optimal_k):
                cluster_mask = data.kmeans_clusters == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 1 and len(data.latent.values) == len(data.kmeans_clusters):
                    within_distances = latent_distances[np.ix_(cluster_indices, cluster_indices)]
                    within_cluster_distances.extend(within_distances[np.triu_indices_from(within_distances, k=1)])
                    
                    cluster_center = np.mean(data.latent.values[cluster_mask], axis=0)
                    cluster_centers_latent.append(cluster_center)
                    
                    other_indices = np.where(~cluster_mask)[0]
                    if len(other_indices) > 0:
                        between_distances = latent_distances[np.ix_(cluster_indices, other_indices)]
                        between_cluster_distances.extend(between_distances.flatten())
            
            if within_cluster_distances and between_cluster_distances:
                violin_data = [within_cluster_distances, between_cluster_distances]
                labels = ['Within Cluster', 'Between Cluster']
                
                violin_parts = ax2.violinplot(violin_data, positions=[1, 2], widths=0.6)
                
                colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary']]
                for pc, color in zip(violin_parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax2.set_xticks([1, 2])
                ax2.set_xticklabels(labels)
                ax2.set_ylabel('Distance in Latent Space')
                ax2.set_title('Cluster Separation Quality')
                ax2.grid(True, alpha=0.3)
                
                mean_within = np.mean(within_cluster_distances)
                mean_between = np.mean(between_cluster_distances)
                separation_ratio = mean_between / mean_within if mean_within > 0 else np.inf
                
                ax2.text(0.02, 0.98, f'Ratio (Between/Within): {separation_ratio:.2f}', 
                        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for cluster analysis', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Cluster Separation Quality')
            
            method_idx += 1
        
        plt.tight_layout()
        return fig
    
    def generate_latent_dimension_analysis(self, data: LatentAnalysisData):
        """Generate latent dimension disentanglement analysis.
        
        Analyzes individual latent dimensions showing their distributions
        and relationships to cluster assignments.
        
        Args:
            data: LatentAnalysisData object with latent representations
            
        Returns:
            Matplotlib figure object
        """
        print("Generating latent dimension analysis...")
        
        latent_data = data.latent.values
        n_dims = min(latent_data.shape[1], 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Latent Dimension Analysis: {data.platform_name}', fontsize=16, fontweight='bold')
        
        for dim in range(n_dims):
            row = dim // 3
            col = dim % 3
            ax = axes[row, col]
            
            dim_values = latent_data[:, dim]
            
            ax.hist(dim_values, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                   density=True, edgecolor='black', linewidth=0.5)
            
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
            
            if data.kmeans_clusters is not None and len(dim_values) == len(data.kmeans_clusters):
                unique_clusters = np.unique(data.kmeans_clusters)
                cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                
                y_base = ax.get_ylim()[0]
                for i, cluster_id in enumerate(unique_clusters):
                    cluster_mask = data.kmeans_clusters == cluster_id
                    cluster_values = dim_values[cluster_mask]
                    y_offset = y_base - (i + 1) * (abs(y_base) * 0.1)
                    
                    ax.scatter(cluster_values, np.full(len(cluster_values), y_offset), 
                              alpha=0.8, s=8, c=[cluster_colors[i]], 
                              label=f'Cluster {cluster_id+1}' if dim == 0 else "", 
                              edgecolors='black', linewidth=0.3)
                
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0] - len(unique_clusters) * abs(current_ylim[0]) * 0.15, 
                           current_ylim[1])
            
            ax.set_xlabel(f'Latent Dimension {dim+1} Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Dimension {dim+1} Distribution\n(μ={mean_val:.3f}, σ={std_val:.3f})')
            if dim == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_latent_traversal_analysis(self, data: LatentAnalysisData, 
                                          target_dim: int = 0, n_steps: int = 7):
        """Generate latent space traversal analysis.
        
        Shows how traversing a single latent dimension affects the representation
        in the embedded space.
        
        Args:
            data: LatentAnalysisData object with latent representations
            target_dim: Index of latent dimension to traverse
            n_steps: Number of steps in the traversal
            
        Returns:
            Matplotlib figure object
        """
        print(f"Generating latent traversal analysis for dimension {target_dim}...")
        
        mean_latent = data.latent.mean(axis=0).values
        std_latent = data.latent.std(axis=0).values[target_dim]
        traversal_range = np.linspace(-3*std_latent, 3*std_latent, n_steps)
        
        traversal_vectors = []
        for step_val in traversal_range:
            traversal_vec = mean_latent.copy()
            traversal_vec[target_dim] = mean_latent[target_dim] + step_val
            traversal_vectors.append(traversal_vec)
        
        traversal_vectors = np.array(traversal_vectors)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Dimension {target_dim+1} Traversal Analysis: {data.platform_name}', 
                    fontsize=16, fontweight='bold')
        
        if data.embeddings and 'pca' in data.embeddings:
            pca_model = data.embeddings['pca']['model']
            traversal_2d = pca_model.transform(traversal_vectors)
            
            embedding_a = data.embeddings['pca']['embedding']
            ax1.scatter(embedding_a[:, 0], embedding_a[:, 1], 
                       c=NATURE_COLORS['primary'], alpha=0.3, s=20, 
                       label='Original samples')
            
            ax1.plot(traversal_2d[:, 0], traversal_2d[:, 1], 
                    'ro-', linewidth=3, markersize=8, 
                    label=f'Dimension {target_dim+1} traversal')
            
            for i, (x, y) in enumerate(traversal_2d):
                ax1.annotate(f'{traversal_range[i]:.1f}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            var_ratio = data.embeddings['pca']['explained_variance']
            ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            ax1.set_title('Traversal Path in PCA Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'PCA embedding not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Traversal Path in PCA Space')
        
        ax2.plot(range(n_steps), traversal_range, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Original mean')
        ax2.set_xlabel('Traversal Step')
        ax2.set_ylabel(f'Dimension {target_dim+1} Value')
        ax2.set_title(f'Dimension {target_dim+1} Value Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.text(0.5, 0.02, 
                'Note: Full traversal analysis requires VAE decoder to generate protein profiles.\n'
                'This visualization shows the conceptual traversal path in latent space.',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        return fig 
    
    def generate_interpolation_analysis(self, data: LatentAnalysisData):
        """Generate latent space interpolation analysis.
        
        Shows interpolation paths between different groups or samples in the
        latent space and how they manifest in the embedded visualization.
        
        Args:
            data: LatentAnalysisData object with latent representations
            
        Returns:
            Matplotlib figure object
        """
        print("Generating interpolation analysis...")
        
        if data.groups is None:
            print("  No groups available for interpolation - creating general analysis")
            return self._create_general_interpolation_figure(data)
        
        unique_groups = data.groups.unique()
        if len(unique_groups) < 2:
            print("  Less than 2 groups available - creating general analysis")
            return self._create_general_interpolation_figure(data)
        
        group1, group2 = unique_groups[:2]
        print(f"  Interpolating between {group1} and {group2}")
        
        group1_latent = data.latent[data.groups == group1].mean(axis=0).values
        group2_latent = data.latent[data.groups == group2].mean(axis=0).values
        
        n_steps = 9
        interpolation_steps = np.linspace(0, 1, n_steps)
        interpolated_vectors = []
        
        for step in interpolation_steps:
            interpolated_vec = (1 - step) * group1_latent + step * group2_latent
            interpolated_vectors.append(interpolated_vec)
        
        interpolated_vectors = np.array(interpolated_vectors)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Space Interpolation: {group1} → {group2} ({data.platform_name})', 
                    fontsize=16, fontweight='bold')
        
        if data.embeddings and 'pca' in data.embeddings:
            pca_model = data.embeddings['pca']['model']
            interpolated_2d = pca_model.transform(interpolated_vectors)
            embedding = data.embeddings['pca']['embedding']
            
            palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                            NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                            NATURE_COLORS['highlight']]
            
            for i, group in enumerate(unique_groups):
                group_mask = data.groups == group
                group_embedding = embedding[group_mask]
                color = palette_colors[i % len(palette_colors)]
                ax1.scatter(group_embedding[:, 0], group_embedding[:, 1], 
                           c=color, alpha=0.6, s=10, label=group)
            
            ax1.plot(interpolated_2d[:, 0], interpolated_2d[:, 1], 
                    'ko-', linewidth=3, markersize=8, alpha=0.8,
                    label='Interpolation path')
            
            ax1.annotate(f'{group1}', interpolated_2d[0], 
                        xytext=(-10, -10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=NATURE_COLORS['primary'], alpha=0.7))
            ax1.annotate(f'{group2}', interpolated_2d[-1], 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=NATURE_COLORS['secondary'], alpha=0.7))
            
            var_ratio = data.embeddings['pca']['explained_variance']
            ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            ax1.set_title('Interpolation Path in PCA Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'PCA embedding not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Interpolation Path in PCA Space')
        
        n_dims_to_show = min(5, len(group1_latent))
        palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                        NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                        NATURE_COLORS['highlight']]
        
        for dim in range(n_dims_to_show):
            dim_values = interpolated_vectors[:, dim]
            color = palette_colors[dim % len(palette_colors)]
            ax2.plot(interpolation_steps, dim_values, 'o-', 
                    color=color, linewidth=2, markersize=6,
                    label=f'Dim {dim+1}')
        
        ax2.set_xlabel('Interpolation Step (0=Group1, 1=Group2)')
        ax2.set_ylabel('Latent Dimension Value')
        ax2.set_title('Latent Dimension Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_general_interpolation_figure(self, data: LatentAnalysisData):
        """Create a general interpolation figure when groups are not available.
        
        Args:
            data: LatentAnalysisData object with latent representations
            
        Returns:
            Matplotlib figure object
        """
        latent_data = data.latent.values
        distances = squareform(pdist(latent_data))
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        
        sample1_latent = latent_data[i]
        sample2_latent = latent_data[j]
        sample1_name = data.latent.index[i]
        sample2_name = data.latent.index[j]
        
        print(f"  Interpolating between samples {sample1_name} and {sample2_name}")
        
        n_steps = 9
        interpolation_steps = np.linspace(0, 1, n_steps)
        interpolated_vectors = []
        
        for step in interpolation_steps:
            interpolated_vec = (1 - step) * sample1_latent + step * sample2_latent
            interpolated_vectors.append(interpolated_vec)
        
        interpolated_vectors = np.array(interpolated_vectors)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Space Interpolation: {sample1_name} → {sample2_name} ({data.platform_name})', 
                    fontsize=16, fontweight='bold')
        
        if data.embeddings and 'pca' in data.embeddings:
            pca_model = data.embeddings['pca']['model']
            interpolated_2d = pca_model.transform(interpolated_vectors)
            
            embedding = data.embeddings['pca']['embedding']
            ax1.scatter(embedding[:, 0], embedding[:, 1], 
                       c=NATURE_COLORS['primary'], alpha=0.3, s=2, 
                       label='All samples')
            
            ax1.scatter(embedding[i, 0], embedding[i, 1], 
                       c=NATURE_COLORS['accent'], s=100, 
                       label=sample1_name, edgecolors='black', linewidth=2)
            ax1.scatter(embedding[j, 0], embedding[j, 1], 
                       c=NATURE_COLORS['alternative_3'], s=100, 
                       label=sample2_name, edgecolors='black', linewidth=2)
            
            ax1.plot(interpolated_2d[:, 0], interpolated_2d[:, 1], 
                    'ko-', linewidth=3, markersize=8, alpha=0.8,
                    label='Interpolation path')
            
            var_ratio = data.embeddings['pca']['explained_variance']
            ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            ax1.set_title('Interpolation Path in PCA Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'PCA embedding not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Interpolation Path in PCA Space')
        
        n_dims_to_show = min(5, len(sample1_latent))
        palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                        NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                        NATURE_COLORS['highlight']]
        
        for dim in range(n_dims_to_show):
            dim_values = interpolated_vectors[:, dim]
            color = palette_colors[dim % len(palette_colors)]
            ax2.plot(interpolation_steps, dim_values, 'o-', 
                    color=color, linewidth=2, markersize=6,
                    label=f'Dim {dim+1}')
        
        ax2.set_xlabel('Interpolation Step (0=Sample1, 1=Sample2)')
        ax2.set_ylabel('Latent Dimension Value')
        ax2.set_title('Latent Dimension Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_all_figures(self, data: LatentAnalysisData):
        """Generate all latent space analysis figures.
        
        Args:
            data: LatentAnalysisData object containing all necessary data
            
        Returns:
            List of generated figure names
        """
        print("\nGenerating comprehensive single-platform latent space analysis figures...")
        
        generated_figures = []
        
        try:
            fig1 = self.generate_latent_structure_analysis(data)
            if fig1:
                self.save_figure(fig1, "latent_01_structure_analysis")
                plt.close(fig1)
                generated_figures.append("latent_01_structure_analysis")
        except Exception as e:
            print(f"  Error generating structure analysis: {str(e)}")
        
        try:
            fig2 = self.generate_kmeans_clustering_analysis(data)
            if fig2:
                self.save_figure(fig2, "latent_02_kmeans_clustering")
                plt.close(fig2)
                generated_figures.append("latent_02_kmeans_clustering")
        except Exception as e:
            print(f"  Error generating K-means clustering analysis: {str(e)}")
        
        try:
            fig3 = self.generate_latent_dimension_analysis(data)
            if fig3:
                self.save_figure(fig3, "latent_03_dimension_analysis")
                plt.close(fig3)
                generated_figures.append("latent_03_dimension_analysis")
        except Exception as e:
            print(f"  Error generating dimension analysis: {str(e)}")
        
        for dim in range(min(3, data.latent.shape[1])):
            try:
                fig4 = self.generate_latent_traversal_analysis(data, target_dim=dim)
                if fig4:
                    self.save_figure(fig4, f"latent_04_traversal_dim{dim+1}")
                    plt.close(fig4)
                    generated_figures.append(f"latent_04_traversal_dim{dim+1}")
            except Exception as e:
                print(f"  Error generating traversal analysis for dim {dim+1}: {str(e)}")
        
        try:
            fig5 = self.generate_interpolation_analysis(data)
            if fig5:
                self.save_figure(fig5, "latent_05_interpolation_analysis")
                plt.close(fig5)
                generated_figures.append("latent_05_interpolation_analysis")
        except Exception as e:
            print(f"  Error generating interpolation analysis: {str(e)}")
        
        print(f"\nGenerated {len(generated_figures)} latent space figures")
        return generated_figures

def main():
    """Main function to run the single-platform latent space analysis.
    
    Parses command line arguments, loads data, performs analysis, and generates
    comprehensive visualization figures for VAE latent space exploration.
    """
    parser = argparse.ArgumentParser(
        description="Single-Platform Latent Space Analysis for VAE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python latent_space_analysis_oneplatform.py \\
        --latent data/latent_platform.csv \\
        --truth data/truth_platform.csv \\
        --platform_name "Platform Name" \\
        --groups data/groups.csv \\
        --output_dir latent_analysis_oneplatform_output
        
Expected file formats:
    Latent file: CSV with samples as rows, latent dimensions as columns
    Truth file: CSV with features as rows, samples as columns  
    Groups file: CSV with samples as rows, group labels as values
        """
    )
    
    parser.add_argument('--latent', required=True, help='Latent space file')
    parser.add_argument('--truth', required=True, help='Truth file')
    parser.add_argument('--platform_name', required=True, help='Display name for platform')
    parser.add_argument('--groups', help='Groups/metadata file (CSV with sample labels)')
    parser.add_argument('--output_dir', default='latent_analysis_oneplatform_output', 
                       help='Output directory')
    parser.add_argument('--transpose_latent', action='store_true', 
                       help='Transpose latent file (use if dimensions are rows)')
    parser.add_argument('--n_clusters', type=int, 
                       help='Number of K-means clusters (default: auto-determine)')
    
    args = parser.parse_args()
    
    file_paths = {
        'latent': args.latent,
        'truth': args.truth,
    }
    
    if args.groups:
        file_paths['groups'] = args.groups
    
    analyzer = LatentSpaceAnalyzer(args.output_dir)
    
    print("="*80)
    print("Single-Platform VAE Latent Space Analysis")
    print("="*80)
    print(f"Platform: {args.platform_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Git hash: {analyzer.git_hash}")
    print(f"Timestamp: {analyzer.timestamp}")
    print()
    
    try:
        data = analyzer.load_and_validate_data(file_paths, args.platform_name, 
                                              args.transpose_latent)
        print(f"Loaded latent data: {data.latent.shape}")
        print(f"Loaded truth data: {data.truth.shape}")
        if data.groups is not None:
            print(f"Groups found: {data.groups.nunique()} unique groups")
        print()
        
        data = analyzer.compute_embeddings(data)
        print()
        
        data = analyzer.perform_kmeans_clustering(data, args.n_clusters)
        print()
        
        generated_figures = analyzer.generate_all_figures(data)
        print()
        
        print("Single-platform latent space analysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")
        print(f"Generated {len(generated_figures)} figures")
        
        summary_path = analyzer.output_dir / "logs" / f"analysis_summary_{analyzer.timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Single-Platform Latent Space Analysis Summary\n")
            f.write(f"Generated on: {analyzer.timestamp}\n")
            f.write(f"Git hash: {analyzer.git_hash}\n\n")
            f.write(f"Data shapes:\n")
            f.write(f"  Latent: {data.latent.shape}\n")
            f.write(f"  Truth: {data.truth.shape}\n")
            if data.groups is not None:
                f.write(f"  Groups: {data.groups.nunique()} unique groups\n")
            if data.kmeans_clusters is not None:
                f.write(f"  K-means clusters: {data.optimal_k}\n")
            f.write(f"\nGenerated figures:\n")
            for fig_name in generated_figures:
                f.write(f"  {fig_name}\n")
        
        print(f"Analysis summary saved: {summary_path}")
        
        if data.kmeans_clusters is not None:
            cluster_path = analyzer.output_dir / "data" / f"kmeans_clusters_{analyzer.timestamp}.csv"
            if len(data.kmeans_clusters) == len(data.latent.index):
                cluster_df = pd.DataFrame({
                    'sample': data.latent.index,
                    'cluster': data.kmeans_clusters + 1,
                    'cluster_0indexed': data.kmeans_clusters
                })
                cluster_df.to_csv(cluster_path, index=False)
                print(f"K-means clustering results saved: {cluster_path}")
            else:
                print(f"Warning: Cluster labels length ({len(data.kmeans_clusters)}) doesn't match samples ({len(data.latent.index)})")
                print("Skipping cluster results export")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()