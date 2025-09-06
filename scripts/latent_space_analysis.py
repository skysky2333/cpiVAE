#!/usr/bin/env python3
"""
Latent Space Analysis for Cross-Platform Proteomics VAE Models
Analyzes what the VAE has learned in its latent representation to discover biological insights.
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

# Statistical and ML imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr, levene
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up plotting style
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

# Scientific journal color palette - consistent with specified palette
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
    """Container for latent space analysis data"""
    # Latent representations
    latent_a: pd.DataFrame  # Samples x Latent dimensions for platform A
    latent_b: pd.DataFrame  # Samples x Latent dimensions for platform B
    
    # Original data for comparison
    truth_a: pd.DataFrame
    truth_b: pd.DataFrame
    
    # Metadata
    platform_a_name: str
    platform_b_name: str
    groups: Optional[pd.Series] = None
    
    # Computed embeddings
    embeddings: Dict[str, np.ndarray] = None

class LatentSpaceAnalyzer:
    """Main class for comprehensive latent space analysis"""
    
    def __init__(self, output_dir: str = "latent_analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.git_hash = self._get_git_hash()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
                              platform_a_name: str, platform_b_name: str,
                              transpose_latent: bool = False) -> LatentAnalysisData:
        """Load and validate all input files including latent representations"""
        print("Loading latent space and validation data...")
        
        # Load latent representations
        print(f"  Loading latent space A: {file_paths['latent_a']}")
        latent_a = pd.read_csv(file_paths['latent_a'], index_col=0)
        
        print(f"  Loading latent space B: {file_paths['latent_b']}")
        latent_b = pd.read_csv(file_paths['latent_b'], index_col=0)
        
        # Transpose if needed (sometimes latent files have dimensions as rows)
        if transpose_latent:
            latent_a = latent_a.T
            latent_b = latent_b.T
            print(f"    Transposed latent spaces")
        
        print(f"    Latent A shape: {latent_a.shape}")
        print(f"    Latent B shape: {latent_b.shape}")
        
        # Load truth data for comparison
        print(f"  Loading truth A: {file_paths['truth_a']}")
        truth_a = pd.read_csv(file_paths['truth_a'], index_col=0)
        
        print(f"  Loading truth B: {file_paths['truth_b']}")
        truth_b = pd.read_csv(file_paths['truth_b'], index_col=0)
        
        # Check for groups
        groups = None
        if 'groups' in file_paths and file_paths['groups']:
            print(f"  Loading groups: {file_paths['groups']}")
            groups = pd.read_csv(file_paths['groups'], index_col=0, header=None).iloc[:, 0]
        
        # Debug: Print sample information
        print(f"    Latent A index (first 5): {list(latent_a.index[:5])}")
        print(f"    Latent B index (first 5): {list(latent_b.index[:5])}")
        print(f"    Truth A index (first 5): {list(truth_a.index[:5])}")
        print(f"    Truth B index (first 5): {list(truth_b.index[:5])}")
        print(f"    Truth A columns (first 5): {list(truth_a.columns[:5])}")
        print(f"    Truth B columns (first 5): {list(truth_b.columns[:5])}")
        
        # Align samples between latent and truth data
        # Both latent and truth files should have samples as rows (index)
        common_samples_a = list(set(latent_a.index) & set(truth_a.index))
        common_samples_b = list(set(latent_b.index) & set(truth_b.index))
        
        print(f"  Common samples A: {len(common_samples_a)}")
        print(f"  Common samples B: {len(common_samples_b)}")
        
        # Find samples common to both platforms
        all_common_samples = list(set(common_samples_a) & set(common_samples_b))
        print(f"  Samples common to both platforms: {len(all_common_samples)}")
        
        if len(all_common_samples) == 0:
            print("  ERROR: No common samples found!")
            print(f"    Latent A samples: {len(latent_a.index)}")
            print(f"    Latent B samples: {len(latent_b.index)}")
            print(f"    Truth A samples: {len(truth_a.index)}")
            print(f"    Truth B samples: {len(truth_b.index)}")
            print(f"    Latent A & Truth A overlap: {len(set(latent_a.index) & set(truth_a.index))}")
            print(f"    Latent B & Truth B overlap: {len(set(latent_b.index) & set(truth_b.index))}")
            raise ValueError("No common samples found between platforms!")
        
        # Subset to common samples
        all_common_samples = sorted(all_common_samples)
        latent_a = latent_a.loc[all_common_samples]
        latent_b = latent_b.loc[all_common_samples]
        truth_a = truth_a.loc[all_common_samples]
        truth_b = truth_b.loc[all_common_samples]
        
        if groups is not None:
            groups = groups.loc[all_common_samples]
        
        # Create analysis data object
        analysis_data = LatentAnalysisData(
            latent_a=latent_a,
            latent_b=latent_b,
            truth_a=truth_a,
            truth_b=truth_b,
            platform_a_name=platform_a_name,
            platform_b_name=platform_b_name,
            groups=groups
        )
        
        return analysis_data
    
    def compute_embeddings(self, data: LatentAnalysisData) -> LatentAnalysisData:
        """Compute 2D embeddings using multiple methods"""
        print("Computing 2D embeddings for visualization...")
        
        embeddings = {}
        
        # Combine latent representations for joint analysis
        combined_latent = pd.concat([data.latent_a, data.latent_b], 
                                   keys=['A', 'B'], names=['platform', 'sample'])
        
        print(f"  Combined latent space shape: {combined_latent.shape}")
        
        # PCA
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(combined_latent)
        embeddings['pca'] = {
            'embedding': pca_embedding,
            'explained_variance': pca.explained_variance_ratio_,
            'model': pca
        }
        
        # UMAP
        if UMAP_AVAILABLE:
            print("  Computing UMAP...")
            try:
                umap_model = umap.UMAP(n_components=2, random_state=42, 
                                      n_neighbors=15, min_dist=0.1)
                umap_embedding = umap_model.fit_transform(combined_latent)
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
        
        # t-SNE
        print("  Computing t-SNE...")
        try:
            tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_embedding = tsne_model.fit_transform(combined_latent)
            embeddings['tsne'] = {
                'embedding': tsne_embedding,
                'model': tsne_model
            }
        except Exception as e:
            print(f"    t-SNE failed: {e}")
            embeddings['tsne'] = None
        
        data.embeddings = embeddings
        return data
    
    def save_figure(self, fig: plt.Figure, name: str, **kwargs):
        """Save figure with metadata"""
        fig_path = self.output_dir / "figures" / f"{name}.pdf"
        png_path = self.output_dir / "figures" / f"{name}.png"
        
        # Add metadata
        metadata = {
            'Title': name,
            'Creator': 'Latent Space Analysis',
            'Subject': 'VAE Latent Space Exploration',
            'Keywords': 'proteomics, VAE, latent space, cross-platform',
            'Git_Hash': self.git_hash,
            'Timestamp': self.timestamp
        }
        
        # Save PDF and PNG
        fig.savefig(fig_path, format='pdf', metadata=metadata, **kwargs)
        fig.savefig(png_path, format='png', dpi=300, **kwargs)
        
        print(f"  Figure saved: {fig_path}")
    
    def generate_latent_structure_analysis(self, data: LatentAnalysisData):
        """Figure 1: Latent space structure visualization"""
        print("Generating latent space structure analysis...")
        
        if data.embeddings is None:
            print("  No embeddings computed - skipping")
            return
        
        # Create figure with multiple embedding methods
        n_methods = sum(1 for emb in data.embeddings.values() if emb is not None)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Latent Space Structure Analysis', fontsize=16, fontweight='bold')
        
        # Get platform labels
        n_samples = len(data.latent_a)
        platform_labels = ['A'] * n_samples + ['B'] * n_samples
        
        method_idx = 0
        for method_name, embedding_data in data.embeddings.items():
            if embedding_data is None:
                continue
                
            embedding = embedding_data['embedding']
            
            # Top row: Color by platform
            ax1 = axes[0, method_idx]
            colors = [NATURE_COLORS['primary'] if p == 'A' else NATURE_COLORS['secondary'] 
                     for p in platform_labels]
            
            scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                                c=colors, alpha=0.7, s=10, edgecolors='black', linewidth=0.1)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=NATURE_COLORS['primary'], markersize=8, 
                          label=data.platform_a_name),
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=NATURE_COLORS['secondary'], markersize=8, 
                          label=data.platform_b_name)
            ]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax1.set_xlabel(f'{method_name.upper()} 1')
                ax1.set_ylabel(f'{method_name.upper()} 2')
            
            ax1.set_title(f'{method_name.upper()} - Colored by Platform')
            ax1.grid(True, alpha=0.3)
            
            # Bottom row: Color by groups (if available)
            ax2 = axes[1, method_idx]
            
            if data.groups is not None:
                # Duplicate groups for both platforms
                combined_groups = pd.concat([data.groups, data.groups])
                unique_groups = combined_groups.unique()
                # Use specified color palette for groups
                palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                                NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                                NATURE_COLORS['highlight']]
                group_colors = [palette_colors[i % len(palette_colors)] for i in range(len(unique_groups))]
                
                for i, group in enumerate(unique_groups):
                    mask = combined_groups == group
                    ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=group_colors[i], alpha=0.7, s=10, 
                              label=group, edgecolors='black', linewidth=0.1)
                
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.set_title(f'{method_name.upper()} - Colored by Biological Group')
            else:
                # Color by reconstruction error if groups not available
                # For now, use random coloring
                ax2.scatter(embedding[:, 0], embedding[:, 1], 
                          c=np.random.rand(len(embedding)), cmap='viridis',
                          alpha=0.7, s=10, edgecolors='black', linewidth=0.1)
                ax2.set_title(f'{method_name.upper()} - Random Coloring')
            
            if method_name == 'pca':
                ax2.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax2.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax2.set_xlabel(f'{method_name.upper()} 1')
                ax2.set_ylabel(f'{method_name.upper()} 2')
            
            ax2.grid(True, alpha=0.3)
            
            method_idx += 1
        
        plt.tight_layout()
        return fig 
    
    def generate_platform_alignment_analysis(self, data: LatentAnalysisData):
        """Figure 2: Platform alignment analysis with quiver plots"""
        print("Generating platform alignment analysis...")
        
        if data.embeddings is None:
            print("  No embeddings computed - skipping")
            return
        
        # Create figure for alignment analysis
        n_methods = sum(1 for emb in data.embeddings.values() if emb is not None)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle('Cross-Platform Latent Space Alignment', fontsize=16, fontweight='bold')
        
        n_samples = len(data.latent_a)
        
        method_idx = 0
        for method_name, embedding_data in data.embeddings.items():
            if embedding_data is None:
                continue
                
            embedding = embedding_data['embedding']
            ax = axes[method_idx]
            
            # Split embedding back into platform A and B
            embedding_a = embedding[:n_samples]
            embedding_b = embedding[n_samples:]
            
            # Plot Platform A points
            ax.scatter(embedding_a[:, 0], embedding_a[:, 1], 
                      c=NATURE_COLORS['primary'], alpha=0.7, s=10, 
                      label=f'{data.platform_a_name}', edgecolors='black', linewidth=0.1)
            
            # Plot Platform B points
            ax.scatter(embedding_b[:, 0], embedding_b[:, 1], 
                      c=NATURE_COLORS['secondary'], alpha=0.7, s=10, 
                      label=f'{data.platform_b_name}', edgecolors='black', linewidth=0.1)
            
            # Draw arrows from A to B for each sample (sample every 5th to avoid clutter)
            step = max(1, n_samples // 50)  # Show at most 50 arrows
            for i in range(0, n_samples, step):
                dx = embedding_b[i, 0] - embedding_a[i, 0]
                dy = embedding_b[i, 1] - embedding_a[i, 1]
                
                # Only draw arrow if there's meaningful displacement
                if np.sqrt(dx**2 + dy**2) > 0.01:
                    ax.arrow(embedding_a[i, 0], embedding_a[i, 1], dx, dy,
                            head_width=0.02, head_length=0.02, 
                            fc='gray', ec='gray', alpha=0.6, linewidth=0.5)
            
            # Calculate alignment metrics
            distances = np.sqrt(np.sum((embedding_a - embedding_b)**2, axis=1))
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Add statistics text
            ax.text(0.02, 0.98, 
                   f'Mean distance: {mean_distance:.3f}\n'
                   f'Std distance: {std_distance:.3f}\n'
                   f'Max distance: {np.max(distances):.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax.set_xlabel(f'{method_name.upper()} 1')
                ax.set_ylabel(f'{method_name.upper()} 2')
            
            ax.set_title(f'{method_name.upper()} - Platform Alignment\n(Arrows: A→B)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            method_idx += 1
        
        plt.tight_layout()
        return fig
    
    def generate_latent_dimension_analysis(self, data: LatentAnalysisData):
        """Figure 3: Latent dimension disentanglement analysis"""
        print("Generating latent dimension analysis...")
        
        # For this analysis, we'll focus on Platform A latent space
        latent_data = data.latent_a.values
        n_dims = min(latent_data.shape[1], 6)  # Analyze up to 6 dimensions
        
        # Create figure for dimension analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Latent Dimension Analysis - Platform A', fontsize=16, fontweight='bold')
        
        # Calculate statistics for each dimension
        for dim in range(n_dims):
            row = dim // 3
            col = dim % 3
            ax = axes[row, col]
            
            # Get values for this dimension
            dim_values = latent_data[:, dim]
            
            # Plot histogram of dimension values
            ax.hist(dim_values, bins=30, alpha=0.7, color=NATURE_COLORS['primary'], 
                   density=True, edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
            
            # Color by groups if available
            if data.groups is not None:
                # Create small rug plot showing group separation
                unique_groups = data.groups.unique()
                group_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
                
                y_base = ax.get_ylim()[0]  # Get bottom of histogram
                for i, group in enumerate(unique_groups):
                    group_mask = data.groups == group
                    group_values = dim_values[group_mask]
                    y_offset = y_base - (i + 1) * (y_base * 0.1)  # Stack groups below histogram
                    
                    ax.scatter(group_values, np.full(len(group_values), y_offset), 
                              alpha=0.8, s=8, c=[group_colors[i]], 
                              label=group, edgecolors='black', linewidth=0.3)
                
                # Extend y-axis to show group points
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0] - len(unique_groups) * abs(current_ylim[0]) * 0.15, 
                           current_ylim[1])
            
            ax.set_xlabel(f'Latent Dimension {dim+1} Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Dimension {dim+1} Distribution\n(μ={mean_val:.3f}, σ={std_val:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_latent_traversal_analysis(self, data: LatentAnalysisData, 
                                          target_dim: int = 0, n_steps: int = 7):
        """Figure 4: Latent space traversal analysis"""
        print(f"Generating latent traversal analysis for dimension {target_dim}...")
        
        # Calculate mean latent vector
        mean_latent = data.latent_a.mean(axis=0).values
        
        # Create traversal vectors
        std_latent = data.latent_a.std(axis=0).values[target_dim]
        traversal_range = np.linspace(-3*std_latent, 3*std_latent, n_steps)
        
        traversal_vectors = []
        for step_val in traversal_range:
            traversal_vec = mean_latent.copy()
            traversal_vec[target_dim] = mean_latent[target_dim] + step_val
            traversal_vectors.append(traversal_vec)
        
        traversal_vectors = np.array(traversal_vectors)
        
        # Note: For actual implementation, you would need to pass these through
        # the VAE decoder to get generated protein profiles. Since we don't have
        # access to the model here, we'll create a conceptual visualization.
        
        # Create visualization of the traversal concept
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Dimension {target_dim+1} Traversal Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Left plot: Show traversal in 2D embedding space
        if data.embeddings and 'pca' in data.embeddings:
            # Project traversal vectors to 2D for visualization
            pca_model = data.embeddings['pca']['model']
            
            # For visualization, we'll show how the traversal would look in PCA space
            traversal_2d = pca_model.transform(traversal_vectors)
            
            # Plot original data points
            embedding_a = data.embeddings['pca']['embedding'][:len(data.latent_a)]
            ax1.scatter(embedding_a[:, 0], embedding_a[:, 1], 
                       c=NATURE_COLORS['primary'], alpha=0.3, s=20, 
                       label='Original samples')
            
            # Plot traversal path
            ax1.plot(traversal_2d[:, 0], traversal_2d[:, 1], 
                    'ro-', linewidth=3, markersize=8, 
                    label=f'Dimension {target_dim+1} traversal')
            
            # Annotate steps
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
        
        # Right plot: Show absolute dimension value changes
        abs_values = mean_latent[target_dim] + traversal_range
        ax2.plot(range(n_steps), abs_values, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=mean_latent[target_dim], color='red', linestyle='--', alpha=0.7, label='Original mean')
        ax2.set_xlabel('Traversal Step')
        ax2.set_ylabel(f'Dimension {target_dim+1} Value')
        ax2.set_title(f'Dimension {target_dim+1} Value Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add note about decoder requirement
        fig.text(0.5, 0.02, 
                'Note: Full traversal analysis requires VAE decoder to generate protein profiles.\n'
                'This visualization shows the conceptual traversal path in latent space.',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        return fig 
    
    def generate_interpolation_analysis(self, data: LatentAnalysisData):
        """Figure 5: Latent space interpolation analysis"""
        print("Generating interpolation analysis...")
        
        if data.groups is None:
            print("  No groups available for interpolation - creating general analysis")
            return self._create_general_interpolation_figure(data)
        
        # Find groups for interpolation
        unique_groups = data.groups.unique()
        if len(unique_groups) < 2:
            print("  Less than 2 groups available - creating general analysis")
            return self._create_general_interpolation_figure(data)
        
        # Select two groups for interpolation
        group1, group2 = unique_groups[:2]
        print(f"  Interpolating between {group1} and {group2}")
        
        # Get latent representations for each group
        group1_latent = data.latent_a[data.groups == group1].mean(axis=0).values
        group2_latent = data.latent_a[data.groups == group2].mean(axis=0).values
        
        n_steps = 9
        interpolation_steps = np.linspace(0, 1, n_steps)
        interpolated_vectors = []
        
        for step in interpolation_steps:
            interpolated_vec = (1 - step) * group1_latent + step * group2_latent
            interpolated_vectors.append(interpolated_vec)
        
        interpolated_vectors = np.array(interpolated_vectors)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Space Interpolation: {group1} → {group2}', 
                    fontsize=16, fontweight='bold')
        
        # Left plot: Show interpolation path in 2D embedding space
        if data.embeddings and 'pca' in data.embeddings:
            pca_model = data.embeddings['pca']['model']
            
            # Project interpolated vectors to 2D
            interpolated_2d = pca_model.transform(interpolated_vectors)
            
            # Plot original data points colored by group
            embedding_a = data.embeddings['pca']['embedding'][:len(data.latent_a)]
            
            # Use specified color palette for groups
            palette_colors = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                            NATURE_COLORS['accent'], NATURE_COLORS['neutral'], 
                            NATURE_COLORS['highlight']]
            
            for i, group in enumerate(unique_groups):
                group_mask = (data.groups == group).values
                group_embedding = embedding_a[group_mask]
                color = palette_colors[i % len(palette_colors)]
                ax1.scatter(group_embedding[:, 0], group_embedding[:, 1], 
                           c=color, alpha=0.6, s=10, label=group)
            
            # Plot interpolation path
            ax1.plot(interpolated_2d[:, 0], interpolated_2d[:, 1], 
                    'ko-', linewidth=3, markersize=8, alpha=0.8,
                    label='Interpolation path')
            
            # Annotate start and end
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
        
        # Right plot: Show latent dimension changes during interpolation
        n_dims_to_show = min(5, len(group1_latent))
        # Use specified color palette for dimensions
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
        
        # Add note about decoder requirement
        fig.text(0.5, 0.02, 
                'Note: Full interpolation analysis requires VAE decoder to generate intermediate protein profiles.\n'
                'This visualization shows the conceptual interpolation path in latent space.',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        return fig
    
    def _create_general_interpolation_figure(self, data: LatentAnalysisData):
        """Create a general interpolation figure when groups are not available"""
        
        # Find two sample points that are far apart in latent space
        latent_data = data.latent_a.values
        
        # Calculate pairwise distances and find the most distant pair
        distances = squareform(pdist(latent_data))
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        
        sample1_latent = latent_data[i]
        sample2_latent = latent_data[j]
        sample1_name = data.latent_a.index[i]
        sample2_name = data.latent_a.index[j]
        
        print(f"  Interpolating between samples {sample1_name} and {sample2_name}")
        
        n_steps = 9
        interpolation_steps = np.linspace(0, 1, n_steps)
        interpolated_vectors = []
        
        for step in interpolation_steps:
            interpolated_vec = (1 - step) * sample1_latent + step * sample2_latent
            interpolated_vectors.append(interpolated_vec)
        
        interpolated_vectors = np.array(interpolated_vectors)
        
        # Create visualization (similar to group-based interpolation)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Latent Space Interpolation: {sample1_name} → {sample2_name}', 
                    fontsize=16, fontweight='bold')
        
        # Left plot: Show interpolation path in 2D embedding space
        if data.embeddings and 'pca' in data.embeddings:
            pca_model = data.embeddings['pca']['model']
            
            # Project interpolated vectors to 2D
            interpolated_2d = pca_model.transform(interpolated_vectors)
            
            # Plot original data points
            embedding_a = data.embeddings['pca']['embedding'][:len(data.latent_a)]
            ax1.scatter(embedding_a[:, 0], embedding_a[:, 1], 
                       c=NATURE_COLORS['primary'], alpha=0.3, s=2, 
                       label='All samples')
            
            # Highlight the two samples being interpolated
            ax1.scatter(embedding_a[i, 0], embedding_a[i, 1], 
                       c=NATURE_COLORS['accent'], s=100, 
                       label=sample1_name, edgecolors='black', linewidth=2)
            ax1.scatter(embedding_a[j, 0], embedding_a[j, 1], 
                       c=NATURE_COLORS['alternative_3'], s=100, 
                       label=sample2_name, edgecolors='black', linewidth=2)
            
            # Plot interpolation path
            ax1.plot(interpolated_2d[:, 0], interpolated_2d[:, 1], 
                    'ko-', linewidth=3, markersize=8, alpha=0.8,
                    label='Interpolation path')
            
            var_ratio = data.embeddings['pca']['explained_variance']
            ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            ax1.set_title('Interpolation Path in PCA Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Right plot: Show latent dimension changes during interpolation
        n_dims_to_show = min(5, len(sample1_latent))
        # Use specified color palette for dimensions
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
    
    def generate_platform_delta_analysis(self, data: LatentAnalysisData):
        """Figure 6: PCA and UMAP of per-sample latent differences (B − A)."""
        print("Generating platform delta (B − A) analysis...")

        # Compute per-sample latent deltas
        delta_df = data.latent_b - data.latent_a

        # Compute embeddings on deltas
        delta_embeddings: Dict[str, Optional[Dict[str, Any]]] = {}

        # PCA on deltas
        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(delta_df)
        delta_embeddings['pca'] = {
            'embedding': pca_embedding,
            'explained_variance': pca.explained_variance_ratio_,
        }

        # UMAP on deltas (if available)
        if UMAP_AVAILABLE:
            try:
                umap_model = umap.UMAP(n_components=2, random_state=42,
                                       n_neighbors=15, min_dist=0.1)
                umap_embedding = umap_model.fit_transform(delta_df)
                delta_embeddings['umap'] = {
                    'embedding': umap_embedding,
                }
            except Exception as e:
                print(f"  UMAP on delta failed: {e}")
                delta_embeddings['umap'] = None
        else:
            print("  Skipping UMAP on delta (not available)")
            delta_embeddings['umap'] = None

        # Create figure with available methods (PCA always present)
        n_methods = sum(1 for emb in delta_embeddings.values() if emb is not None)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        fig.suptitle('Latent Delta (B − A): PCA and UMAP', fontsize=16, fontweight='bold')

        # Color configuration for groups if available
        groups = data.groups
        method_idx = 0
        for method_name, embedding_data in delta_embeddings.items():
            if embedding_data is None:
                continue

            embedding = embedding_data['embedding']
            ax = axes[method_idx]

            if groups is not None:
                unique_groups = groups.unique()
                palette_colors = [
                    NATURE_COLORS['primary'],
                    NATURE_COLORS['secondary'],
                    NATURE_COLORS['accent'],
                    NATURE_COLORS['neutral'],
                    NATURE_COLORS['highlight'],
                ]
                for i, group in enumerate(unique_groups):
                    mask = groups == group
                    color = palette_colors[i % len(palette_colors)]
                    ax.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        c=color,
                        alpha=0.7,
                        s=12,
                        label=str(group),
                        edgecolors='black',
                        linewidth=0.1,
                    )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=NATURE_COLORS['neutral'],
                    alpha=0.7,
                    s=12,
                    edgecolors='black',
                    linewidth=0.1,
                )

            if method_name == 'pca':
                var_ratio = embedding_data['explained_variance']
                ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
            else:
                ax.set_xlabel(f'{method_name.upper()} 1')
                ax.set_ylabel(f'{method_name.upper()} 2')

            ax.set_title(f'{method_name.upper()} of Latent Delta (B − A)')
            ax.grid(True, alpha=0.3)
            method_idx += 1

        plt.tight_layout()
        return fig

    def generate_platform_delta_vs_sample_scale(self, data: LatentAnalysisData):
        """Figure 6b: Compare overall sample spread (Fig1 PCA) vs per-sample deltas projected on the same PCA axes."""
        print("Generating delta vs sample scale comparison (projected on Fig1 PCA)...")

        if data.embeddings is None or 'pca' not in data.embeddings or data.embeddings['pca'] is None:
            raise ValueError("PCA embedding not available. Run compute_embeddings first.")

        embedding = data.embeddings['pca']['embedding']
        var_ratio = data.embeddings['pca']['explained_variance']

        n_samples = len(data.latent_a)
        embedding_a = embedding[:n_samples]
        embedding_b = embedding[n_samples:]

        # Deltas in the same PCA space as Figure 1
        delta_pc = embedding_b - embedding_a

        # Shared axis limits from the full PCA scatter
        x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
        y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
        x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Delta vs Sample Scale (Projected on Fig1 PCA)', fontsize=16, fontweight='bold')

        # Left: overall sample spread in PCA
        ax_left.scatter(embedding_a[:, 0], embedding_a[:, 1],
                        c=NATURE_COLORS['primary'], alpha=0.6, s=2,
                        label=data.platform_a_name, edgecolors='black', linewidth=0.1)
        ax_left.scatter(embedding_b[:, 0], embedding_b[:, 1],
                        c=NATURE_COLORS['secondary'], alpha=0.6, s=2,
                        label=data.platform_b_name, edgecolors='black', linewidth=0.1)
        ax_left.set_xlim(xlim)
        ax_left.set_ylim(ylim)
        ax_left.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
        ax_left.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
        ax_left.set_title('Overall Sample Spread (Fig1 PCA)')
        ax_left.legend(loc='upper right')
        ax_left.grid(True, alpha=0.3)
        ax_left.set_aspect('equal', adjustable='box')

        # Right: per-sample deltas in the same PCA axes, anchored at origin
        if data.groups is not None:
            unique_groups = data.groups.unique()
            palette_colors = [
                NATURE_COLORS['primary'],
                NATURE_COLORS['secondary'],
                NATURE_COLORS['accent'],
                NATURE_COLORS['neutral'],
                NATURE_COLORS['highlight'],
            ]
            for i, group in enumerate(unique_groups):
                mask = (data.groups == group).values
                color = palette_colors[i % len(palette_colors)]
                ax_right.scatter(delta_pc[mask, 0], delta_pc[mask, 1],
                                 c=color, alpha=0.3, s=2, label=str(group),
                                 edgecolors='black', linewidth=0.1)
            ax_right.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax_right.scatter(delta_pc[:, 0], delta_pc[:, 1],
                             c=NATURE_COLORS['neutral'], alpha=0.3, s=2,
                             edgecolors='black', linewidth=0.1)

        ax_right.set_xlim(xlim)
        ax_right.set_ylim(ylim)
        ax_right.set_xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
        ax_right.set_ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
        ax_right.set_title('Per-sample Δ (B − A) in Fig1 PCA axes')
        ax_right.grid(True, alpha=0.3)
        ax_right.set_aspect('equal', adjustable='box')

        # Summary statistics annotation
        sample_radii = np.sqrt(np.sum(embedding**2, axis=1))
        mean_sample_radius = float(np.mean(sample_radii))
        delta_norms = np.sqrt(np.sum(delta_pc**2, axis=1))
        mean_delta_norm = float(np.mean(delta_norms))
        ratio = mean_delta_norm / mean_sample_radius if mean_sample_radius > 0 else np.nan
        ax_right.text(0.02, 0.98,
                      f'Mean |Δ|: {mean_delta_norm:.3f}\n'
                      f'Mean |sample|: {mean_sample_radius:.3f}\n'
                      f'Ratio (Δ/sample): {ratio:.3f}',
                      transform=ax_right.transAxes, fontsize=10, va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def generate_variance_statistical_analysis(self, data: LatentAnalysisData):
        """Figure 6c: Statistical analysis of variance differences between platforms and deltas using top 2 PCs."""
        print("Generating variance statistical analysis (Figure 6c) using top 2 PCs...")
        
        # Use PCA embedding if available, otherwise compute it
        if data.embeddings is not None and 'pca' in data.embeddings and data.embeddings['pca'] is not None:
            # Use existing PCA
            embedding = data.embeddings['pca']['embedding']
            var_ratio = data.embeddings['pca']['explained_variance']
            n_samples = len(data.latent_a)
            embedding_a = embedding[:n_samples]
            embedding_b = embedding[n_samples:]
        else:
            # Compute PCA with 2 components
            n_samples = len(data.latent_a)
            combined_latent = pd.concat([data.latent_a, data.latent_b], 
                                       keys=['A', 'B'], names=['platform', 'sample'])
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            pca_full = pca.fit_transform(combined_latent)
            
            embedding_a = pca_full[:n_samples]
            embedding_b = pca_full[n_samples:]
            var_ratio = pca.explained_variance_ratio_
        
        # Calculate deltas in PCA space
        delta_pc = embedding_b - embedding_a
        
        # Calculate variances for PC1 and PC2
        var_a_pc1 = np.var(embedding_a[:, 0])
        var_a_pc2 = np.var(embedding_a[:, 1])
        var_b_pc1 = np.var(embedding_b[:, 0])
        var_b_pc2 = np.var(embedding_b[:, 1])
        var_delta_pc1 = np.var(delta_pc[:, 0])
        var_delta_pc2 = np.var(delta_pc[:, 1])
        
        # Total variance (sum of PC1 and PC2)
        var_a_total = var_a_pc1 + var_a_pc2
        var_b_total = var_b_pc1 + var_b_pc2
        var_delta_total = var_delta_pc1 + var_delta_pc2
        
        # Test PC1 and PC2 separately
        stat_ab_pc1, p_ab_pc1 = levene(embedding_a[:, 0], embedding_b[:, 0])
        stat_ab_pc2, p_ab_pc2 = levene(embedding_a[:, 1], embedding_b[:, 1])
        
        stat_da_pc1, p_da_pc1 = levene(delta_pc[:, 0], embedding_a[:, 0])
        stat_da_pc2, p_da_pc2 = levene(delta_pc[:, 1], embedding_a[:, 1])
        
        stat_db_pc1, p_db_pc1 = levene(delta_pc[:, 0], embedding_b[:, 0])
        stat_db_pc2, p_db_pc2 = levene(delta_pc[:, 1], embedding_b[:, 1])
        
        # Conservative approach: use the maximum p-value (most conservative)
        # For A vs B: we want to be conservative about claiming similarity
        p_ab = max(p_ab_pc1, p_ab_pc2)
        
        # For Delta comparisons: use minimum p-value (most significant)
        # We want to be confident that delta has lower variance
        p_da = min(p_da_pc1, p_da_pc2)
        p_db = min(p_db_pc1, p_db_pc2)
        
        # Helper function for significance markers
        def get_significance_marker(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return 'n.s.'
        
        # Create square figure with clean layout
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[1, 1])
        
        fig.suptitle(f'Variance Analysis in PCA Space (Figure 6c)', 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Total variance comparison with significance markers (square)
        ax1 = fig.add_subplot(gs[0, 0])
        
        categories = ['Platform A', 'Platform B', 'Delta (B-A)']
        total_vars = [var_a_total, var_b_total, var_delta_total]
        
        x = np.arange(len(categories))
        bars = ax1.bar(x, total_vars, color=[NATURE_COLORS['primary'], 
                                             NATURE_COLORS['secondary'], 
                                             NATURE_COLORS['accent']], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=11)
        
        # Add significance markers
        max_height = max(total_vars)
        y_offset = max_height * 0.1
        
        # A vs B comparison
        y_line = max(var_a_total, var_b_total) + y_offset
        ax1.plot([0, 1], [y_line, y_line], 'k-', linewidth=1)
        ax1.text(0.5, y_line + y_offset*0.1, 
                get_significance_marker(p_ab),
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Delta vs A comparison
        y_line = max(var_delta_total, var_a_total) + y_offset*2
        ax1.plot([0, 2], [y_line, y_line], 'k-', linewidth=1)
        ax1.text(1, y_line + y_offset*0.1,
                get_significance_marker(p_da),
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Delta vs B comparison
        y_line = max(var_delta_total, var_b_total) + y_offset*2.5
        ax1.plot([1, 2], [y_line, y_line], 'k-', linewidth=1)
        ax1.text(1.5, y_line + y_offset*0.1,
                get_significance_marker(p_db),
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Total Variance (PC1 + PC2)', fontsize=12)
        ax1.set_title(f'Variance in Top 2 Principal Components', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=12)
        ax1.set_ylim(0, max(total_vars) * 1.4)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add variance reduction percentages
        var_reduction_a = (1 - var_delta_total/var_a_total) * 100
        var_reduction_b = (1 - var_delta_total/var_b_total) * 100
        ax1.text(0.02, 0.98, f'Δ variance reduction:\n'
                            f'vs A: {var_reduction_a:.1f}%\n'
                            f'vs B: {var_reduction_b:.1f}%',
                transform=ax1.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel 2: Individual PC p-values (new top-right panel)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Show individual PC test results
        pc_tests = ['PC1: A vs B', 'PC2: A vs B', 
                   'PC1: Δ vs A', 'PC2: Δ vs A',
                   'PC1: Δ vs B', 'PC2: Δ vs B']
        pc_pvals = [p_ab_pc1, p_ab_pc2, p_da_pc1, p_da_pc2, p_db_pc1, p_db_pc2]
        
        y_pos = np.arange(len(pc_tests))
        colors_bars = ['#e64b35' if 'PC1' in t else '#4dbbd5' for t in pc_tests]
        
        bars = ax2.barh(y_pos, pc_pvals, color=colors_bars, alpha=0.7)
        
        # Add significance threshold line
        ax2.axvline(x=0.05, color='black', linestyle='--', linewidth=1, label='α=0.05')
        
        # Add p-value labels
        for i, (bar, pval) in enumerate(zip(bars, pc_pvals)):
            ax2.text(pval + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{pval:.3f}', va='center', fontsize=9)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pc_tests, fontsize=10)
        ax2.set_xlabel('p-value', fontsize=11)
        ax2.set_title('Individual PC Tests (before combining)', fontsize=12)
        ax2.set_xlim(0, max(0.15, max(pc_pvals) * 1.2))
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: PC-wise variance breakdown (moved to bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Create bar plot for PC1 and PC2 variances
        pc_categories = ['PC1', 'PC2']
        x_pc = np.arange(len(pc_categories))
        width = 0.25
        
        vars_a_pcs = [var_a_pc1, var_a_pc2]
        vars_b_pcs = [var_b_pc1, var_b_pc2]
        vars_delta_pcs = [var_delta_pc1, var_delta_pc2]
        
        bars1 = ax3.bar(x_pc - width, vars_a_pcs, width, label='Platform A',
                       color=NATURE_COLORS['primary'], alpha=0.8)
        bars2 = ax3.bar(x_pc, vars_b_pcs, width, label='Platform B',
                       color=NATURE_COLORS['secondary'], alpha=0.8)
        bars3 = ax3.bar(x_pc + width, vars_delta_pcs, width, label='Delta (B-A)',
                       color=NATURE_COLORS['accent'], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_ylabel('Variance', fontsize=11)
        ax3.set_title('Variance by Principal Component', fontsize=12)
        ax3.set_xticks(x_pc)
        ax3.set_xticklabels(pc_categories, fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add variance explained below x-axis
        for i, pc in enumerate(pc_categories):
            ax3.text(i, -0.05, f'({var_ratio[i]*100:.1f}% var)', 
                    ha='center', fontsize=9, color='gray',
                    transform=ax3.get_xaxis_transform())
        
        # Panel 4: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Create summary text box
        summary_stats = f"""Statistical Test Results
━━━━━━━━━━━━━━━━━━━━━
Conservative p-values:
• A vs B: p = {p_ab:.4f} {get_significance_marker(p_ab)}
  (max of PC1={p_ab_pc1:.3f}, PC2={p_ab_pc2:.3f})
• Δ vs A: p = {p_da:.4f} {get_significance_marker(p_da)}
  (min of PC1={p_da_pc1:.3f}, PC2={p_da_pc2:.3f})
• Δ vs B: p = {p_db:.4f} {get_significance_marker(p_db)}
  (min of PC1={p_db_pc1:.3f}, PC2={p_db_pc2:.3f})

Variance Reduction:
• Δ has {var_reduction_a:.1f}% less than A
• Δ has {var_reduction_b:.1f}% less than B

Variance Explained:
• PC1: {var_ratio[0]*100:.1f}%
• PC2: {var_ratio[1]*100:.1f}%
• Total: {sum(var_ratio)*100:.1f}%

Interpretation:
{'✓' if p_ab > 0.05 else '✗'} Platforms similar variance
{'✓' if p_da < 0.05 else '✗'} Delta < A variance
{'✓' if p_db < 0.05 else '✗'} Delta < B variance

Conservative approach used:
Max p-value for similarity test
Min p-value for difference tests"""
        
        ax4.text(0.1, 0.9, summary_stats, transform=ax4.transAxes,
                fontsize=9, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))
        
        plt.tight_layout()
        return fig

    def generate_all_figures(self, data: LatentAnalysisData):
        """Generate all latent space analysis figures"""
        print("\nGenerating comprehensive latent space analysis figures...")
        
        generated_figures = []
        
        # Figure 1: Latent space structure
        try:
            fig1 = self.generate_latent_structure_analysis(data)
            if fig1:
                self.save_figure(fig1, "latent_01_structure_analysis")
                plt.close(fig1)
                generated_figures.append("latent_01_structure_analysis")
        except Exception as e:
            print(f"  Error generating structure analysis: {str(e)}")
        
        # Figure 2: Platform alignment
        try:
            fig2 = self.generate_platform_alignment_analysis(data)
            if fig2:
                self.save_figure(fig2, "latent_02_platform_alignment")
                plt.close(fig2)
                generated_figures.append("latent_02_platform_alignment")
        except Exception as e:
            print(f"  Error generating platform alignment: {str(e)}")
        
        # Figure 3: Dimension analysis
        try:
            fig3 = self.generate_latent_dimension_analysis(data)
            if fig3:
                self.save_figure(fig3, "latent_03_dimension_analysis")
                plt.close(fig3)
                generated_figures.append("latent_03_dimension_analysis")
        except Exception as e:
            print(f"  Error generating dimension analysis: {str(e)}")
        
        # Figure 4: Traversal analysis (for first few dimensions)
        for dim in range(min(3, data.latent_a.shape[1])):
            try:
                fig4 = self.generate_latent_traversal_analysis(data, target_dim=dim)
                if fig4:
                    self.save_figure(fig4, f"latent_04_traversal_dim{dim+1}")
                    plt.close(fig4)
                    generated_figures.append(f"latent_04_traversal_dim{dim+1}")
            except Exception as e:
                print(f"  Error generating traversal analysis for dim {dim+1}: {str(e)}")
        
        # Figure 5: Interpolation analysis
        try:
            fig5 = self.generate_interpolation_analysis(data)
            if fig5:
                self.save_figure(fig5, "latent_05_interpolation_analysis")
                plt.close(fig5)
                generated_figures.append("latent_05_interpolation_analysis")
        except Exception as e:
            print(f"  Error generating interpolation analysis: {str(e)}")
        
        # Figure 6: Delta (B − A) analysis
        try:
            fig6 = self.generate_platform_delta_analysis(data)
            if fig6:
                self.save_figure(fig6, "latent_06_delta_platform_differences")
                plt.close(fig6)
                generated_figures.append("latent_06_delta_platform_differences")
        except Exception as e:
            print(f"  Error generating delta analysis: {str(e)}")
        
        # Figure 6b: Delta vs sample scale on Fig1 PCA axes
        try:
            fig6b = self.generate_platform_delta_vs_sample_scale(data)
            if fig6b:
                self.save_figure(fig6b, "latent_06b_delta_vs_sample_scale")
                plt.close(fig6b)
                generated_figures.append("latent_06b_delta_vs_sample_scale")
        except Exception as e:
            print(f"  Error generating delta vs sample scale (6b): {str(e)}")
        
        # Figure 6c: Variance statistical analysis
        try:
            fig6c = self.generate_variance_statistical_analysis(data)
            if fig6c:
                self.save_figure(fig6c, "latent_06c_variance_statistical_analysis")
                plt.close(fig6c)
                generated_figures.append("latent_06c_variance_statistical_analysis")
        except Exception as e:
            print(f"  Error generating variance statistical analysis (6c): {str(e)}")
        
        print(f"\nGenerated {len(generated_figures)} latent space figures")
        return generated_figures

def main():
    """
    Main function to run comprehensive latent space analysis.
    
    Analyzes latent representations from Joint VAE models, computes metrics,
    and generates visualization reports for cross-platform comparison.
    """
    parser = argparse.ArgumentParser(
        description="Latent Space Analysis for Cross-Platform Proteomics VAE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python latent_space_analysis.py \\
        --latent_a data/latent_platform_a.csv \\
        --latent_b data/latent_platform_b.csv \\
        --truth_a data/truth_platform_a.csv \\
        --truth_b data/truth_platform_b.csv \\
        --platform_a_name "Platform A" \\
        --platform_b_name "Platform B" \\
        --groups data/groups.csv \\
        --output_dir latent_results
        
Expected file formats:
    Latent files: CSV with samples as rows, latent dimensions as columns
    Truth files: CSV with features as rows, samples as columns  
    Groups file: CSV with samples as rows, group labels as values
        """
    )
    
    # Required arguments
    parser.add_argument('--latent_a', required=True, help='Latent space file for platform A')
    parser.add_argument('--latent_b', required=True, help='Latent space file for platform B')
    parser.add_argument('--truth_a', required=True, help='Truth file for platform A')
    parser.add_argument('--truth_b', required=True, help='Truth file for platform B')
    parser.add_argument('--platform_a_name', required=True, help='Display name for platform A')
    parser.add_argument('--platform_b_name', required=True, help='Display name for platform B')
    
    # Optional arguments
    parser.add_argument('--groups', help='Groups/metadata file (CSV with sample labels)')
    parser.add_argument('--output_dir', default='latent_analysis_output', help='Output directory')
    parser.add_argument('--transpose_latent', action='store_true', 
                       help='Transpose latent files (use if dimensions are rows)')
    
    args = parser.parse_args()
    
    # Prepare file paths
    file_paths = {
        'latent_a': args.latent_a,
        'latent_b': args.latent_b,
        'truth_a': args.truth_a,
        'truth_b': args.truth_b,
    }
    
    if args.groups:
        file_paths['groups'] = args.groups
    
    # Initialize analyzer
    analyzer = LatentSpaceAnalyzer(args.output_dir)
    
    print("="*80)
    print("Cross-Platform Proteomics VAE Latent Space Analysis")
    print("="*80)
    print(f"Platforms: {args.platform_a_name} vs {args.platform_b_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Git hash: {analyzer.git_hash}")
    print(f"Timestamp: {analyzer.timestamp}")
    print()
    
    try:
        # Load and validate data
        data = analyzer.load_and_validate_data(file_paths, args.platform_a_name, 
                                              args.platform_b_name, args.transpose_latent)
        print(f"Loaded latent data: Platform A: {data.latent_a.shape}, Platform B: {data.latent_b.shape}")
        if data.groups is not None:
            print(f"Biological groups found: {data.groups.nunique()} unique groups")
        print()
        
        # Compute embeddings
        data = analyzer.compute_embeddings(data)
        print()
        
        # Generate figures
        generated_figures = analyzer.generate_all_figures(data)
        print()
        
        print("Latent space analysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")
        print(f"Generated {len(generated_figures)} figures")
        
        # Save analysis summary
        summary_path = analyzer.output_dir / "logs" / f"analysis_summary_{analyzer.timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Latent Space Analysis Summary\n")
            f.write(f"Generated on: {analyzer.timestamp}\n")
            f.write(f"Git hash: {analyzer.git_hash}\n\n")
            f.write(f"Data shapes:\n")
            f.write(f"  Platform A latent: {data.latent_a.shape}\n")
            f.write(f"  Platform B latent: {data.latent_b.shape}\n")
            f.write(f"  Common samples: {len(data.latent_a)}\n")
            if data.groups is not None:
                f.write(f"  Biological groups: {data.groups.nunique()}\n")
            f.write(f"\nGenerated figures:\n")
            for fig_name in generated_figures:
                f.write(f"  {fig_name}\n")
        
        print(f"Analysis summary saved: {summary_path}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()