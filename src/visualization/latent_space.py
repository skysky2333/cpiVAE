"""Latent space visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import Dict, Optional, List
import pandas as pd


def plot_latent_alignment(
    latent_a: np.ndarray,
    latent_b: np.ndarray,
    save_path: Optional[str] = None,
    config: Dict = None
) -> plt.Figure:
    """
    Plot UMAP visualization of latent space alignment.
    
    Args:
        latent_a: Latent representations from platform A encoder
        latent_b: Latent representations from platform B encoder
        save_path: Path to save the plot
        config: Visualization configuration
        
    Returns:
        Matplotlib figure
    """
    # Default config
    if config is None:
        config = {
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'umap_random_state': 42,
            'figsize': [10, 8],
            'dpi': 300
        }
    
    # Combine latent representations
    combined_latent = np.vstack([latent_a, latent_b])
    platform_labels = ['Platform A'] * len(latent_a) + ['Platform B'] * len(latent_b)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=config['umap_n_neighbors'],
        min_dist=config['umap_min_dist'],
        random_state=config['umap_random_state']
    )
    
    embedding = reducer.fit_transform(combined_latent)
    
    # Create plot
    fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
    
    # Plot points colored by platform
    for platform in ['Platform A', 'Platform B']:
        mask = [label == platform for label in platform_labels]
        ax.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            label=platform,
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Latent Space Alignment\n(Well-aligned = mixed colors)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
    
    return fig


def plot_latent_biology(
    latent_a: np.ndarray,
    latent_b: np.ndarray,
    biological_labels: List[str],
    save_path: Optional[str] = None,
    config: Dict = None
) -> plt.Figure:
    """
    Plot UMAP visualization colored by biological labels.
    
    Args:
        latent_a: Latent representations from platform A encoder
        latent_b: Latent representations from platform B encoder
        biological_labels: Biological labels (e.g., disease status, age group)
        save_path: Path to save the plot
        config: Visualization configuration
        
    Returns:
        Matplotlib figure
    """
    # Default config
    if config is None:
        config = {
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'umap_random_state': 42,
            'figsize': [12, 5],
            'dpi': 300
        }
    
    # Combine latent representations
    combined_latent = np.vstack([latent_a, latent_b])
    platform_labels = ['Platform A'] * len(latent_a) + ['Platform B'] * len(latent_b)
    
    # Duplicate biological labels for both platforms
    combined_bio_labels = biological_labels + biological_labels
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=config['umap_n_neighbors'],
        min_dist=config['umap_min_dist'],
        random_state=config['umap_random_state']
    )
    
    embedding = reducer.fit_transform(combined_latent)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config['figsize'], dpi=config['dpi'])
    
    # Plot 1: Colored by platform
    for platform in ['Platform A', 'Platform B']:
        mask = [label == platform for label in platform_labels]
        ax1.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            label=platform,
            alpha=0.6,
            s=20
        )
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Colored by Platform')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by biology
    unique_bio_labels = list(set(combined_bio_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_bio_labels)))
    
    for i, bio_label in enumerate(unique_bio_labels):
        mask = [label == bio_label for label in combined_bio_labels]
        ax2.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            label=bio_label,
            alpha=0.6,
            s=20,
            color=colors[i]
        )
    
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title('Colored by Biology')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
    
    return fig


def compute_latent_alignment_score(
    latent_a: np.ndarray,
    latent_b: np.ndarray
) -> float:
    """
    Compute quantitative alignment score between latent representations.
    
    Args:
        latent_a: Latent representations from platform A
        latent_b: Latent representations from platform B
        
    Returns:
        Alignment score (lower is better aligned)
    """
    # Mean squared distance between corresponding latent representations
    mse = np.mean((latent_a - latent_b) ** 2)
    return mse


def analyze_latent_clusters(
    latent_representations: np.ndarray,
    biological_labels: List[str],
    platform_labels: List[str]
) -> Dict:
    """
    Analyze clustering in latent space.
    
    Args:
        latent_representations: Combined latent representations
        biological_labels: Biological labels
        platform_labels: Platform labels
        
    Returns:
        Analysis results dictionary
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    unique_bio_labels = list(set(biological_labels))
    n_bio_clusters = len(unique_bio_labels)
    
    # K-means clustering
    if n_bio_clusters > 1:
        kmeans = KMeans(n_clusters=n_bio_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_representations)
        
        # Silhouette score for biological clustering
        bio_label_numeric = [unique_bio_labels.index(label) for label in biological_labels]
        bio_silhouette = silhouette_score(latent_representations, bio_label_numeric)
        
        # Adjusted Rand Index between biological labels and clusters
        ari = adjusted_rand_score(bio_label_numeric, cluster_labels)
        
        # Platform mixing score (lower is better mixed)
        platform_silhouette = silhouette_score(
            latent_representations, 
            [0 if p == 'Platform A' else 1 for p in platform_labels]
        )
    else:
        bio_silhouette = np.nan
        ari = np.nan
        platform_silhouette = np.nan
    
    return {
        'biological_silhouette_score': bio_silhouette,
        'biological_clustering_ari': ari,
        'platform_silhouette_score': platform_silhouette,  # Lower is better
        'n_biological_groups': n_bio_clusters
    } 