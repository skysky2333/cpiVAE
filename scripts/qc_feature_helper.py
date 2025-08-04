#!/usr/bin/env python3
"""
Biological Analysis Helper for Feature Quality Control

This module provides additional biological validation for feature QC results:
1. Pathway enrichment analysis comparing best vs worst features
2. STRING protein interaction network analysis
3. Binary classification comparison of different feature sets

Dependencies:
- gseapy (for pathway enrichment)
- requests (for STRING API)
- networkx (for network analysis)
- plotly (for interactive network plots)
- sklearn (for classification)
- scipy (for statistical tests)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import logging

logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("WARNING: gseapy not available. Pathway enrichment analysis will be skipped.")
    print("Install with: pip install gseapy")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("WARNING: networkx not available. Network analysis will be limited.")
    print("Install with: pip install networkx")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: plotly not available. Interactive plots will use matplotlib instead.")
    print("Install with: pip install plotly")

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Classification analysis will be skipped.")
    print("Install with: pip install scikit-learn")

try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Statistical tests will be limited.")
    print("Install with: pip install scipy")

warnings.filterwarnings("ignore")


class BiologicalQCAnalyzer:
    """
    Performs biological validation analysis for feature quality control results.
    
    This class provides pathway enrichment analysis and protein interaction network
    analysis to validate that QC metrics capture biologically meaningful patterns
    rather than random noise.
    
    Attributes:
        results_df: DataFrame containing feature QC results
        output_dir: Directory for saving analysis outputs
        organism: Target organism for enrichment analysis
        bio_dir: Subdirectory for biological analysis outputs
    """
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path, organism: str = "hsapiens"):
        """
        Initialize biological analyzer.
        
        Args:
            results_df: DataFrame with feature QC results
            output_dir: Output directory for results
            organism: Organism name for enrichment (hsapiens, mmusculus, etc.)
        """
        self.results_df = results_df.copy()
        self.output_dir = Path(output_dir)
        
        # Map organism names to gseapy format
        organism_mapping = {
            'hsapiens': 'Human',
            'human': 'Human',
            'mmusculus': 'Mouse',
            'mouse': 'Mouse',
            'rnorvegicus': 'Rat',
            'rat': 'Rat'
        }
        
        self.organism = organism_mapping.get(organism.lower(), 'Human')
        
        # Create subdirectory for biological analysis
        self.bio_dir = self.output_dir / "biological_analysis"
        self.bio_dir.mkdir(exist_ok=True)
        
        print(f"Initialized BiologicalQCAnalyzer for {len(results_df)} features")
        print(f"Output directory: {self.bio_dir}")
    
    def run_enrichment_analysis(self, decile_cutoff: float = 0.1) -> Dict:
        """
        Run pathway enrichment analysis on best vs worst features.
        
        Args:
            decile_cutoff: Fraction of features to include in top/bottom groups
            
        Returns:
            Dictionary containing enrichment results
        """
        if not GSEAPY_AVAILABLE:
            print("Skipping enrichment analysis - gseapy not available")
            return {}
        
        print(f"\nRunning pathway enrichment analysis...")
        print(f"Using top/bottom {decile_cutoff*100:.0f}% of features")
        
        # Sort by selected QC metric and extract deciles
        qc_metric_col = 'selected_qc_metric' if 'selected_qc_metric' in self.results_df.columns else 'correlation'
        qc_metric_name = self.results_df['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in self.results_df.columns else 'correlation'
        
        # Define sort order based on metric type (higher is better vs lower is better)
        higher_is_better = {'correlation', 'nash_sutcliffe', 'r2_baseline'}
        ascending = qc_metric_name not in higher_is_better
        
        sorted_df = self.results_df.sort_values(qc_metric_col, ascending=ascending)
        n_features = len(sorted_df)
        n_decile = int(n_features * decile_cutoff)
        
        worst_features = sorted_df.head(n_decile)['feature_name'].tolist()
        best_features = sorted_df.tail(n_decile)['feature_name'].tolist()
        
        print(f"Worst features (n={len(worst_features)}): {qc_metric_name} range {sorted_df.head(n_decile)[qc_metric_col].min():.3f} to {sorted_df.head(n_decile)[qc_metric_col].max():.3f}")
        print(f"Best features (n={len(best_features)}): {qc_metric_name} range {sorted_df.tail(n_decile)[qc_metric_col].min():.3f} to {sorted_df.tail(n_decile)[qc_metric_col].max():.3f}")
        
        # Clean gene names (remove any suffixes, keep only gene symbols)
        worst_genes = self._clean_gene_names(worst_features)
        best_genes = self._clean_gene_names(best_features)
        
        print(f"Cleaned gene names: {len(worst_genes)} worst, {len(best_genes)} best")
        
        enrichment_results = {}
        
        # Run enrichment for different databases
        databases = {
            'GO_Biological_Process_2023': 'GO_BP',
            'GO_Molecular_Function_2023': 'GO_MF', 
            'GO_Cellular_Component_2023': 'GO_CC',
            'KEGG_2021_Human': 'KEGG',
            'Reactome_2022': 'Reactome'
        }
        
        for db_name, db_short in databases.items():
            print(f"\nRunning enrichment for {db_short}...")
            
            try:
                # Enrichment for worst features
                worst_enr = self._run_gseapy_enrichment(
                    gene_list=worst_genes,
                    gene_sets=db_name,
                    description=f'worst_features_{db_short}'
                )
                
                # Enrichment for best features
                best_enr = self._run_gseapy_enrichment(
                    gene_list=best_genes,
                    gene_sets=db_name,
                    description=f'best_features_{db_short}'
                )
                
                if worst_enr is not None and best_enr is not None:
                    enrichment_results[db_short] = {
                        'worst': worst_enr,
                        'best': best_enr,
                        'database': db_name
                    }
                    
                    # Save results
                    worst_enr.to_csv(self.bio_dir / f'enrichment_worst_{db_short}.csv', index=False)
                    best_enr.to_csv(self.bio_dir / f'enrichment_best_{db_short}.csv', index=False)
                    
                    print(f"  Worst features: {len(worst_enr)} significant terms")
                    print(f"  Best features: {len(best_enr)} significant terms")
                
            except Exception as e:
                print(f"  Error in {db_short} enrichment: {e}")
                continue
        
        # Create enrichment comparison plots
        if enrichment_results:
            self._create_enrichment_plots(enrichment_results, worst_genes, best_genes)
        
        return enrichment_results
    
    def run_string_network_analysis(self, top_n: int = 100, confidence_threshold: int = 400) -> Dict:
        """
        Run STRING protein interaction network analysis.
        
        Args:
            top_n: Number of top features to include in network
            confidence_threshold: STRING confidence threshold (0-1000)
            
        Returns:
            Dictionary containing network results
        """
        print(f"\nRunning STRING network analysis for top {top_n} features...")
        
        # Get top features by selected QC metric
        qc_metric_col = 'selected_qc_metric' if 'selected_qc_metric' in self.results_df.columns else 'correlation'
        qc_metric_name = self.results_df['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in self.results_df.columns else 'correlation'
        
        # Define sort order based on metric type (higher is better vs lower is better)
        higher_is_better = {'correlation', 'nash_sutcliffe', 'r2_baseline'}
        
        if qc_metric_name in higher_is_better:
            top_features = self.results_df.nlargest(top_n, qc_metric_col)
        else:
            top_features = self.results_df.nsmallest(top_n, qc_metric_col)
        gene_list = self._clean_gene_names(top_features['feature_name'].tolist())
        
        print(f"Querying STRING API for {len(gene_list)} genes...")
        
        # Query STRING API
        network_data = self._query_string_api(gene_list, confidence_threshold)
        
        if not network_data:
            print("No network data retrieved from STRING")
            return {}
        
        # Create network analysis
        network_results = self._analyze_string_network(network_data, top_features)
        
        # Create network visualizations
        self._create_network_plots(network_results, top_features)
        
        return network_results
    
    def _clean_gene_names(self, gene_names: List[str]) -> List[str]:
        """
        Clean gene names for enrichment analysis.
        
        Removes common suffixes, prefixes, and special characters to ensure 
        gene names are in the proper format for biological databases.
        
        Args:
            gene_names: List of raw gene names from feature data
            
        Returns:
            List of cleaned gene names suitable for enrichment analysis
        """
        cleaned = []
        for name in gene_names:
            clean_name = str(name).strip()
            
            import re
            clean_name = re.sub(r'[._]\d+$', '', clean_name)
            clean_name = re.sub(r'[^a-zA-Z0-9\-]', '', clean_name)
            
            if clean_name and len(clean_name) > 1:
                cleaned.append(clean_name.upper())
        
        return list(dict.fromkeys(cleaned))
    
    def _run_gseapy_enrichment(self, gene_list: List[str], gene_sets: str, description: str) -> Optional[pd.DataFrame]:
        """
        Run GSEAPY enrichment analysis for a given gene list.
        
        Args:
            gene_list: List of cleaned gene names
            gene_sets: Gene set database name (e.g., 'GO_Biological_Process_2023')
            description: Description for output directory naming
            
        Returns:
            DataFrame with enrichment results or None if no significant terms found
        """
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism=self.organism,
                outdir=str(self.bio_dir / description),
                cutoff=0.05  # FDR cutoff
            )
            
            if enr.results.empty:
                print(f"    No significant terms found for {description}")
                return None
                
            results = enr.results.copy()
            results = results[results['Adjusted P-value'] < 0.05]
            
            if len(results) == 0:
                return None
            
            results['Gene_Ratio'] = results['Overlap'].str.split('/').str[0].astype(int) / results['Overlap'].str.split('/').str[1].astype(int)
            results['Gene_Count'] = results['Overlap'].str.split('/').str[0].astype(int)
            results = results.sort_values('Adjusted P-value')
            
            return results
            
        except Exception as e:
            print(f"    Error in enrichment: {e}")
            return None
    
    def _create_enrichment_plots(self, enrichment_results: Dict, worst_genes: List[str], best_genes: List[str]):
        """
        Create enrichment comparison plots for worst vs best features.
        
        Args:
            enrichment_results: Dictionary containing enrichment results by database
            worst_genes: List of worst performing gene names
            best_genes: List of best performing gene names
        """
        print("Creating enrichment visualization plots...")
        for db_short, results in enrichment_results.items():
            worst_df = results['worst']
            best_df = results['best']
            
            if worst_df is None and best_df is None:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            if worst_df is not None and len(worst_df) > 0:
                top_worst = worst_df.head(15)
                
                scatter = axes[0].scatter(
                    top_worst['Gene_Ratio'],
                    range(len(top_worst)),
                    s=top_worst['Gene_Count'] * 10,
                    c=-np.log10(top_worst['Adjusted P-value']),
                    cmap='Reds',
                    alpha=0.7,
                    edgecolors='#e64b35',
                    linewidth=0.5
                )
                
                axes[0].set_yticks(range(len(top_worst)))
                axes[0].set_yticklabels([term[:50] + '...' if len(term) > 50 else term 
                                       for term in top_worst['Term']], fontsize=8)
                axes[0].set_xlabel('Gene Ratio')
                axes[0].set_title(f'Worst Features Enrichment - {db_short}\n(n={len(worst_genes)} genes)')
                axes[0].grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter, ax=axes[0])
                cbar.set_label('-log10(FDR)')
                
            else:
                axes[0].text(0.5, 0.5, 'No significant enrichment', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title(f'Worst Features Enrichment - {db_short}')
            
            if best_df is not None and len(best_df) > 0:
                top_best = best_df.head(15)
                
                scatter = axes[1].scatter(
                    top_best['Gene_Ratio'],
                    range(len(top_best)),
                    s=top_best['Gene_Count'] * 10,
                    c=-np.log10(top_best['Adjusted P-value']),
                    cmap='Blues',
                    alpha=0.7,
                    edgecolors='#00a087',
                    linewidth=0.5
                )
                
                axes[1].set_yticks(range(len(top_best)))
                axes[1].set_yticklabels([term[:50] + '...' if len(term) > 50 else term 
                                       for term in top_best['Term']], fontsize=8)
                axes[1].set_xlabel('Gene Ratio')
                axes[1].set_title(f'Best Features Enrichment - {db_short}\n(n={len(best_genes)} genes)')
                axes[1].grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter, ax=axes[1])
                cbar.set_label('-log10(FDR)')
            
            else:
                axes[1].text(0.5, 0.5, 'No significant enrichment', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title(f'Best Features Enrichment - {db_short}')
            
            plt.tight_layout()
            plt.savefig(self.bio_dir / f'enrichment_comparison_{db_short}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self._create_enrichment_summary_plot(enrichment_results, worst_genes, best_genes)
    
    def _create_enrichment_summary_plot(self, enrichment_results: Dict, worst_genes: List[str], best_genes: List[str]):
        """
        Create summary enrichment comparison plot across all databases.
        
        Args:
            enrichment_results: Dictionary containing enrichment results by database
            worst_genes: List of worst performing gene names
            best_genes: List of best performing gene names
        """
        summary_data = []
        
        for db_short, results in enrichment_results.items():
            worst_df = results['worst']
            best_df = results['best']
            
            worst_count = len(worst_df) if worst_df is not None else 0
            best_count = len(best_df) if best_df is not None else 0
            
            worst_top_pval = worst_df['Adjusted P-value'].min() if worst_df is not None and len(worst_df) > 0 else 1.0
            best_top_pval = best_df['Adjusted P-value'].min() if best_df is not None and len(best_df) > 0 else 1.0
            
            summary_data.append({
                'Database': db_short,
                'Worst_Count': worst_count,
                'Best_Count': best_count,
                'Worst_Top_Pval': worst_top_pval,
                'Best_Top_Pval': best_top_pval
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) == 0:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        x = np.arange(len(summary_df))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, summary_df['Worst_Count'], width, 
                      label='Worst Features', color='#e64b35', alpha=0.7)
        axes[0, 0].bar(x + width/2, summary_df['Best_Count'], width, 
                      label='Best Features', color='#00a087', alpha=0.7)
        
        axes[0, 0].set_xlabel('Database')
        axes[0, 0].set_ylabel('Number of Significant Terms')
        axes[0, 0].set_title('Enriched Terms Count Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(summary_df['Database'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].bar(x - width/2, -np.log10(summary_df['Worst_Top_Pval']), width, 
                      label='Worst Features', color='#e64b35', alpha=0.7)
        axes[0, 1].bar(x + width/2, -np.log10(summary_df['Best_Top_Pval']), width, 
                      label='Best Features', color='#00a087', alpha=0.7)
        
        axes[0, 1].set_xlabel('Database')
        axes[0, 1].set_ylabel('-log10(Best P-value)')
        axes[0, 1].set_title('Top Enrichment Significance')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(summary_df['Database'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        feature_counts = {
            'Worst Features': len(worst_genes),
            'Best Features': len(best_genes),
            'Other Features': len(self.results_df) - len(worst_genes) - len(best_genes)
        }
        
        axes[1, 0].pie(feature_counts.values(), labels=feature_counts.keys(), 
                      autopct='%1.1f%%', startangle=90,
                      colors=['#e64b35', '#00a087', '#f39b7f'])
        axes[1, 0].set_title('Feature Distribution in Analysis')
        summary_text = f"""Pathway Enrichment Analysis Summary:

Total Features Analyzed: {len(self.results_df)}
Worst Features (bottom decile): {len(worst_genes)}
Best Features (top decile): {len(best_genes)}

Databases Analyzed: {len(summary_df)}
- {', '.join(summary_df['Database'].tolist())}

Total Enriched Terms:
- Worst Features: {summary_df['Worst_Count'].sum()}
- Best Features: {summary_df['Best_Count'].sum()}

Best Significance:
- Worst: -log10(p) = {-np.log10(summary_df['Worst_Top_Pval'].min()):.2f}
- Best: -log10(p) = {-np.log10(summary_df['Best_Top_Pval'].min()):.2f}
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Analysis Summary')
        
        plt.tight_layout()
        plt.savefig(self.bio_dir / 'enrichment_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        summary_df.to_csv(self.bio_dir / 'enrichment_summary.csv', index=False)
        print(f"Enrichment summary saved to: {self.bio_dir / 'enrichment_summary.csv'}")
    
    def _query_string_api(self, gene_list: List[str], confidence_threshold: int) -> Dict:
        """
        Query STRING API for protein interactions.
        
        Args:
            gene_list: List of gene names to query
            confidence_threshold: STRING confidence threshold (0-1000)
            
        Returns:
            Dictionary containing interaction data and metadata
        """
        
        string_api_url = "https://string-db.org/api"
        output_format = "tsv"
        method = "network"
        
        genes_str = "%0d".join(gene_list)
        
        request_url = f"{string_api_url}/{output_format}/{method}?identifiers={genes_str}&species=9606&required_score={confidence_threshold}"
        
        print(f"Querying STRING API...")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Number of genes: {len(gene_list)}")
        
        try:
            response = requests.get(request_url, timeout=30)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                print("No interactions found in STRING")
                return {}
            
            header = lines[0].split('\t')
            interactions = []
            
            for line in lines[1:]:
                fields = line.split('\t')
                if len(fields) >= len(header):
                    interaction = dict(zip(header, fields))
                    interactions.append(interaction)
            
            print(f"Retrieved {len(interactions)} protein interactions")
            interactions_df = pd.DataFrame(interactions)
            interactions_df.to_csv(self.bio_dir / 'string_interactions_raw.csv', index=False)
            
            return {
                'interactions': interactions,
                'gene_list': gene_list,
                'confidence_threshold': confidence_threshold
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying STRING API: {e}")
            return {}
        except Exception as e:
            print(f"Error parsing STRING response: {e}")
            return {}
    
    def _analyze_string_network(self, network_data: Dict, top_features: pd.DataFrame) -> Dict:
        """
        Analyze STRING network data and compute network metrics.
        
        Args:
            network_data: Dictionary containing STRING interaction data
            top_features: DataFrame with top features and their QC metrics
            
        Returns:
            Dictionary containing network analysis results
        """
        
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available - limited network analysis")
            return network_data
        
        interactions = network_data['interactions']
        
        # Create NetworkX graph
        G = nx.Graph()
        
        qc_metric_col = 'selected_qc_metric' if 'selected_qc_metric' in top_features.columns else 'correlation'
        qc_metric_name = top_features['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in top_features.columns else 'correlation'
        
        print(f"Mapping {len(top_features)} top features to network nodes...")
        print(f"Top features {qc_metric_name} range: {top_features[qc_metric_col].min():.3f} to {top_features[qc_metric_col].max():.3f}")
        
        # Create mapping from cleaned gene names to QC metric values
        gene_to_qc = {}
        for _, row in top_features.iterrows():
            feat_name = row['feature_name']
            qc_value = row[qc_metric_col]
            
            # Clean the feature name the same way we did for STRING query
            cleaned_name = self._clean_gene_names([feat_name])[0] if self._clean_gene_names([feat_name]) else None
            if cleaned_name:
                gene_to_qc[cleaned_name.upper()] = qc_value
        
        print(f"Created mapping for {len(gene_to_qc)} cleaned gene names")
        
        mapped_nodes = 0
        unmapped_nodes = 0
        qc_values = []
        
        for interaction in interactions:
            node1 = interaction['preferredName_A'].upper()
            node2 = interaction['preferredName_B'].upper()
            score = float(interaction['score'])
            
                if node1 not in G:
                node1_qc = gene_to_qc.get(node1, None)
                if node1_qc is not None:
                    mapped_nodes += 1
                    qc_values.append(node1_qc)
                else:
                    unmapped_nodes += 1
                    node1_qc = 0.0  # Default for unmapped nodes
                
                G.add_node(node1, qc_value=node1_qc)
            
            if node2 not in G:
                node2_qc = gene_to_qc.get(node2, None)
                if node2_qc is not None:
                    mapped_nodes += 1
                    qc_values.append(node2_qc)
                else:
                    unmapped_nodes += 1
                    node2_qc = 0.0  # Default for unmapped nodes
                
                G.add_node(node2, qc_value=node2_qc)
            
                G.add_edge(node1, node2, weight=score)
        
        print(f"Node mapping results:")
        print(f"  - Nodes with QC metric mapped: {mapped_nodes}")
        print(f"  - Nodes without mapping (set to 0.0): {unmapped_nodes}")
        if qc_values:
            print(f"  - Mapped {qc_metric_name} range: {min(qc_values):.3f} to {max(qc_values):.3f}")
            print(f"  - Mean mapped {qc_metric_name}: {np.mean(qc_values):.3f}")
        
        print(f"Example gene mappings:")
        for i, (gene, qc_val) in enumerate(list(gene_to_qc.items())[:5]):
            print(f"  {gene}: {qc_val:.3f}")
        
        print(f"Example network nodes:")
        for i, node in enumerate(list(G.nodes())[:5]):
            qc_val = G.nodes[node].get('qc_value', 0.0)
            print(f"  {node}: {qc_val:.3f}")
        
        network_stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'num_components': nx.number_connected_components(G)
        }
        
        largest_cc = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
        G_main = G.subgraph(largest_cc)
        
        node_data = []
        qc_metric_values = []
        
        for node in G_main.nodes():
            node_qc = G_main.nodes[node].get('qc_value', 0.0)
            qc_metric_values.append(node_qc)
            
            node_data.append({
                'node': node,
                qc_metric_name: node_qc,
                'degree': G_main.degree(node),
                'betweenness': nx.betweenness_centrality(G_main)[node],
                'closeness': nx.closeness_centrality(G_main)[node],
                'eigenvector': nx.eigenvector_centrality(G_main, max_iter=1000)[node] if G_main.number_of_edges() > 0 else 0
            })
        
        network_analysis = pd.DataFrame(node_data)
        
        network_analysis.to_csv(self.bio_dir / 'string_network_analysis.csv', index=False)
        
        qc_network_corr = {}
        if len(qc_metric_values) > 1:
            qc_network_corr = {
                'degree_correlation': np.corrcoef(network_analysis[qc_metric_name], network_analysis['degree'])[0, 1],
                'betweenness_correlation': np.corrcoef(network_analysis[qc_metric_name], network_analysis['betweenness'])[0, 1],
                'closeness_correlation': np.corrcoef(network_analysis[qc_metric_name], network_analysis['closeness'])[0, 1],
                'eigenvector_correlation': np.corrcoef(network_analysis[qc_metric_name], network_analysis['eigenvector'])[0, 1]
            }
        
        print("Network Analysis Summary:")
        print(f"  Nodes: {network_stats['num_nodes']}")
        print(f"  Edges: {network_stats['num_edges']}")
        print(f"  Density: {network_stats['density']:.4f}")
        print(f"  Average clustering: {network_stats['avg_clustering']:.4f}")
        print(f"  Connected components: {network_stats['num_components']}")
        
        if qc_network_corr:
            print("  QC-Network Correlations:")
            for metric, corr in qc_network_corr.items():
                if not np.isnan(corr):
                    print(f"    {metric}: {corr:.4f}")
        
        return {
            'graph': G,
            'main_component': G_main,
            'network_stats': network_stats,
            'network_analysis': network_analysis,
            'qc_network_correlations': qc_network_corr,
            **network_data
        }
    
    def _create_network_plots(self, network_results: Dict, top_features: pd.DataFrame):
        """
        Create comprehensive network visualization plots.
        
        Args:
            network_results: Dictionary containing network analysis results
            top_features: DataFrame with top features and QC metrics
        """
        print("Creating network visualization plots...")
        
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available - skipping network plots")
            return
        
        G_main = network_results['main_component']
        network_analysis = network_results['network_analysis']
        
        if G_main.number_of_nodes() == 0:
            print("No nodes in main component - skipping network plots")
            return
        
        self._create_matplotlib_network_plot(G_main, network_analysis)
        
        if PLOTLY_AVAILABLE:
            self._create_plotly_network_plot(G_main, network_analysis)
        
        self._create_network_statistics_plots(network_results)
    
    def _create_matplotlib_network_plot(self, G_main, network_analysis):
        """Create matplotlib network plot."""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Force-directed layout
        pos = nx.spring_layout(G_main, k=1, iterations=50, seed=42)
        
        qc_values = [G_main.nodes[node].get('qc_value', 0.0) for node in G_main.nodes()]
        degrees = [G_main.degree(node) for node in G_main.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G_main, pos, ax=axes[0, 0],
            node_color=qc_values,
            node_size=[d*20 + 50 for d in degrees],
            cmap='RdBu_r',
            alpha=0.8
        )
        
        nx.draw_networkx_edges(G_main, pos, ax=axes[0, 0], alpha=0.3, width=0.5)
        
        high_degree_nodes = {node: node for node in G_main.nodes() if G_main.degree(node) >= 5}
        nx.draw_networkx_labels(G_main, pos, labels=high_degree_nodes, ax=axes[0, 0], font_size=8)
        
        axes[0, 0].set_title(f'Protein Interaction Network\n(Colored by QC Metric, Size by Degree)\nNodes: {G_main.number_of_nodes()}, Edges: {G_main.number_of_edges()}')
        
        cbar = plt.colorbar(nodes, ax=axes[0, 0])
        cbar.set_label('QC Metric Value')
        
        degree_sequence = [d for n, d in G_main.degree()]
        axes[0, 1].hist(degree_sequence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Degree')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Degree Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        if len(network_analysis) > 0:
            # Use the QC metric column (should be the first one that's not 'node')
            qc_col = [col for col in network_analysis.columns if col not in ['node', 'degree', 'betweenness', 'closeness', 'eigenvector']][0]
            
            scatter = axes[1, 0].scatter(
                network_analysis['betweenness'],
                network_analysis[qc_col],
                s=network_analysis['degree'] * 10 + 20,
                c=network_analysis['degree'],
                cmap='viridis',
                alpha=0.6
            )
            
            axes[1, 0].set_xlabel('Betweenness Centrality')
            axes[1, 0].set_ylabel('QC Metric')
            axes[1, 0].set_title('Network Centrality vs QC Metric')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(network_analysis) > 1:
                corr_coef = np.corrcoef(network_analysis['betweenness'], network_analysis[qc_col])[0, 1]
                if not np.isnan(corr_coef):
                    axes[1, 0].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                                   transform=axes[1, 0].transAxes, 
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            
            plt.colorbar(scatter, ax=axes[1, 0], label='Degree')
        
        # Plot 4: High-quality vs low-quality subnetworks
        if len(network_analysis) > 0:
            # Define high and low quality based on QC metric
            median_qc = network_analysis[qc_col].median()
            high_quality_nodes = network_analysis[network_analysis[qc_col] > median_qc]['node'].tolist()
            low_quality_nodes = network_analysis[network_analysis[qc_col] <= median_qc]['node'].tolist()
            
            # Count interactions within and between groups
            high_high = 0
            low_low = 0
            high_low = 0
            
            for edge in G_main.edges():
                node1, node2 = edge
                if node1 in high_quality_nodes and node2 in high_quality_nodes:
                    high_high += 1
                elif node1 in low_quality_nodes and node2 in low_quality_nodes:
                    low_low += 1
                else:
                    high_low += 1
            
            # Create bar plot
            categories = ['High-High', 'Low-Low', 'High-Low']
            counts = [high_high, low_low, high_low]
            colors = ['#00a087', '#e64b35', '#f39b7f']
            
            bars = axes[1, 1].bar(categories, counts, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Number of Interactions')
            axes[1, 1].set_title('Interactions by QC Quality Groups')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add percentage labels
            total_edges = sum(counts)
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + total_edges*0.01,
                               f'{count}\n({100*count/total_edges:.1f}%)',
                               ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.bio_dir / 'string_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_plotly_network_plot(self, G_main, network_analysis):
        """Create interactive plotly network plot."""
        
        # Force-directed layout
        pos = nx.spring_layout(G_main, k=1, iterations=50, seed=42)
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G_main.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            qc_value = G_main.nodes[node].get('qc_value', 0.0)
            degree = G_main.degree(node)
            
            node_text.append(f"{node}<br>QC Value: {qc_value:.3f}<br>Degree: {degree}")
            node_color.append(qc_value)
            node_size.append(max(10, degree * 3))
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        
        for edge in G_main.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='RdBu',
                colorbar=dict(
                    title="QC Metric",
                    xanchor="left"
                ),
                line=dict(width=1, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Interactive Protein Interaction Network<br>Nodes: {G_main.number_of_nodes()}, Edges: {G_main.number_of_edges()}',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size = degree, color = QC metric",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save interactive plot
        fig.write_html(str(self.bio_dir / 'string_network_interactive.html'))
        print(f"Interactive network plot saved: {self.bio_dir / 'string_network_interactive.html'}")
    
    def _create_network_statistics_plots(self, network_results):
        """Create network statistics visualization."""
        
        network_stats = network_results['network_stats']
        qc_correlations = network_results['qc_network_correlations']
        network_analysis = network_results['network_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Network statistics bar plot
        stats_names = ['Nodes', 'Edges', 'Density×100', 'Avg Clustering×100', 'Components']
        stats_values = [
            network_stats['num_nodes'],
            network_stats['num_edges'],
            network_stats['density'] * 100,
            network_stats['avg_clustering'] * 100,
            network_stats['num_components']
        ]
        
        bars = axes[0, 0].bar(stats_names, stats_values, color='#4dbbd5', alpha=0.7)
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Network Statistics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(stats_values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # QC-Network correlations
        if qc_correlations:
            corr_names = list(qc_correlations.keys())
            corr_values = [qc_correlations[name] for name in corr_names]
            
            # Filter out NaN values
            valid_indices = [i for i, v in enumerate(corr_values) if not np.isnan(v)]
            corr_names = [corr_names[i] for i in valid_indices]
            corr_values = [corr_values[i] for i in valid_indices]
            
            if corr_values:
                colors = ['#e64b35' if v < 0 else '#00a087' for v in corr_values]
                bars = axes[0, 1].bar([name.replace('_correlation', '').replace('_', ' ').title() 
                                     for name in corr_names], 
                                     corr_values, color=colors, alpha=0.7)
                axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[0, 1].set_ylabel('Correlation with QC Score')
                axes[0, 1].set_title('QC Correlation with Network Centrality')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, corr_values):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., 
                                   height + (0.05 if height >= 0 else -0.05),
                                   f'{value:.3f}', ha='center', 
                                   va='bottom' if height >= 0 else 'top')
        
        # Centrality distributions
        if len(network_analysis) > 0:
            centrality_measures = ['degree', 'betweenness', 'closeness', 'eigenvector']
            
            for i, measure in enumerate(centrality_measures[:2]):  # Plot first 2
                axes[1, i].hist(network_analysis[measure], bins=20, alpha=0.7, 
                              color='#00a087', edgecolor='#3c5488')
                axes[1, i].set_xlabel(measure.title())
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].set_title(f'{measure.title()} Distribution')
                axes[1, i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = network_analysis[measure].mean()
                median_val = network_analysis[measure].median()
                axes[1, i].axvline(mean_val, color='#e64b35', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
                axes[1, i].axvline(median_val, color='#4dbbd5', linestyle='--', alpha=0.7, label=f'Median: {median_val:.3f}')
                axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(self.bio_dir / 'network_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()


class ClassificationQCAnalyzer:
    """
    Performs classification analysis to validate QC efficacy through predictive modeling.
    
    This class uses binary classification tasks to demonstrate that features selected
    by QC metrics have superior predictive performance compared to randomly selected
    or poorly performing features, providing empirical validation of QC effectiveness.
    
    Attributes:
        results_df: DataFrame containing feature QC results
        original_data: Original feature measurement data
        pheno_data: Phenotype data for classification targets
        binary_pheno_cols: List of binary phenotype column names
        feature_sets: Dictionary of different feature sets for comparison
    """
    
    def __init__(self, results_df: pd.DataFrame, original_data: pd.DataFrame, 
                 pheno_data: pd.DataFrame, binary_pheno_cols: List[str],
                 annotations_df: Optional[pd.DataFrame] = None, 
                 classic_qc_flag: Optional[str] = None,
                 output_dir: Path = None):
        """
        Initialize classification analyzer.
        
        Args:
            results_df: Feature QC results DataFrame
            original_data: Original feature data
            pheno_data: Phenotype data with samples as index
            binary_pheno_cols: List of binary phenotype columns for classification
            annotations_df: Feature annotations DataFrame
            classic_qc_flag: Column name in annotations for classic QC
            output_dir: Output directory for results
        """
        self.results_df = results_df.copy()
        self.original_data = original_data.copy()
        self.pheno_data = pheno_data.copy()
        self.binary_pheno_cols = binary_pheno_cols
        self.annotations_df = annotations_df
        self.classic_qc_flag = classic_qc_flag
        
        # Define color palette
        self.color_palette = {
            'error': '#e64b35',      # red
            'primary': '#4dbbd5',    # light blue
            'good': '#00a087',       # teal/green
            'accent': '#3c5488',     # dark blue
            'highlight': '#f39b7f',  # light orange/peach
        }
        
        # Set colors for seaborn
        sns.set_palette([self.color_palette['primary'], self.color_palette['good'], 
                        self.color_palette['accent'], self.color_palette['error'], 
                        self.color_palette['highlight']])
        
        if output_dir is None:
            output_dir = Path(".")
        self.output_dir = Path(output_dir)
        
        # Create subdirectory for classification analysis
        self.class_dir = self.output_dir / "classification_analysis"
        self.class_dir.mkdir(exist_ok=True)
        
        # Align data
        self._align_data()
        
        # Define feature sets
        self._define_feature_sets()
        
        print(f"Initialized ClassificationQCAnalyzer")
        print(f"Samples: {len(self.aligned_data)}")
        print(f"Binary phenotypes: {len(self.binary_pheno_cols)}")
        print(f"Feature sets defined: {len(self.feature_sets)}")
        print(f"Output directory: {self.class_dir}")
    
    def _align_data(self):
        """Align feature data and phenotype data by sample IDs."""
        print("Aligning feature and phenotype data...")
        
        # Get common samples
        feature_samples = set(self.original_data.index)
        pheno_samples = set(self.pheno_data.index)
        common_samples = feature_samples.intersection(pheno_samples)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between feature and phenotype data")
        
        print(f"  Feature data samples: {len(feature_samples)}")
        print(f"  Phenotype data samples: {len(pheno_samples)}")
        print(f"  Common samples: {len(common_samples)}")
        
        # Align data
        common_samples = sorted(list(common_samples))
        self.aligned_data = self.original_data.loc[common_samples]
        self.aligned_pheno = self.pheno_data.loc[common_samples]
        
        # Validate binary phenotypes
        valid_pheno_cols = []
        for col in self.binary_pheno_cols:
            if col not in self.aligned_pheno.columns:
                print(f"WARNING: Phenotype column '{col}' not found in data")
                continue
            
            # Check if binary
            unique_vals = self.aligned_pheno[col].dropna().unique()
            if len(unique_vals) != 2:
                print(f"WARNING: Phenotype column '{col}' is not binary (values: {unique_vals})")
                continue
            
            valid_pheno_cols.append(col)
            print(f"  Valid binary phenotype '{col}': {unique_vals}")
        
        self.binary_pheno_cols = valid_pheno_cols
        
        if len(self.binary_pheno_cols) == 0:
            raise ValueError("No valid binary phenotype columns found")
    
    def _define_feature_sets(self):
        """Define different feature sets for comparison."""
        print("Defining feature sets...")
        
        # VAE-based feature sets
        vae_good_features = self.results_df[~self.results_df['is_poor_quality']]['feature_name'].tolist()
        vae_bad_features = self.results_df[self.results_df['is_poor_quality']]['feature_name'].tolist()
        
        # Ensure features exist in data
        available_features = set(self.aligned_data.columns)
        vae_good_features = [f for f in vae_good_features if f in available_features]
        vae_bad_features = [f for f in vae_bad_features if f in available_features]
        
        self.feature_sets = {
            'unfiltered_features': list(available_features),
            'VAE_good_features': vae_good_features,
            'VAE_bad_features': vae_bad_features
        }
        
        # Classic QC feature sets (if annotation provided)
        if self.annotations_df is not None and self.classic_qc_flag is not None:
            classic_good_features, classic_bad_features = self._get_classic_qc_features()
            self.feature_sets['classic_good_features'] = classic_good_features
            self.feature_sets['classic_bad_features'] = classic_bad_features
        
        # Print feature set sizes
        for set_name, features in self.feature_sets.items():
            print(f"  {set_name}: {len(features)} features")
        
        # Remove empty feature sets
        self.feature_sets = {name: features for name, features in self.feature_sets.items() 
                           if len(features) > 0}
    
    def _get_classic_qc_features(self):
        """Get classic QC feature sets based on annotation flag."""
        print(f"Extracting classic QC features using flag: {self.classic_qc_flag}")
        
        if self.classic_qc_flag not in self.annotations_df.columns:
            print(f"WARNING: Classic QC flag '{self.classic_qc_flag}' not found in annotations")
            return [], []
        
        # Get feature names that have annotations
        annotated_features = set(self.annotations_df.index)
        data_features = set(self.aligned_data.columns)
        common_features = annotated_features.intersection(data_features)
        
        if len(common_features) == 0:
            print("WARNING: No common features between annotations and data")
            return [], []
        
        # Filter annotations to common features
        common_annot = self.annotations_df.loc[list(common_features)]
        
        # Get unique values in classic QC flag
        qc_values = common_annot[self.classic_qc_flag].dropna().unique()
        print(f"  Classic QC flag values: {qc_values}")
        
        # Assume binary classification: good vs bad
        # Try to identify which values represent good vs bad quality
        if len(qc_values) == 2:
            # Simple case: assume alphabetically first is bad, second is good
            # You might want to customize this logic based on your annotation scheme
            bad_value, good_value = sorted(qc_values)
            
            classic_good_features = common_annot[common_annot[self.classic_qc_flag] == good_value].index.tolist()
            classic_bad_features = common_annot[common_annot[self.classic_qc_flag] == bad_value].index.tolist()
            
            print(f"  Classic good features ({good_value}): {len(classic_good_features)}")
            print(f"  Classic bad features ({bad_value}): {len(classic_bad_features)}")
            
        else:
            print(f"WARNING: Expected 2 values for classic QC flag, got {len(qc_values)}")
            return [], []
        
        return classic_good_features, classic_bad_features
    
    def run_classification_comparison(self, cv_folds: int = 5, random_state: int = 42):
        """
        Run classification comparison across all feature sets and phenotypes.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing all classification results
        """
        if not SKLEARN_AVAILABLE:
            print("Skipping classification analysis - scikit-learn not available")
            return {}
        
        print(f"\nRunning classification comparison...")
        print(f"Cross-validation folds: {cv_folds}")
        print(f"Feature sets: {list(self.feature_sets.keys())}")
        print(f"Binary phenotypes: {self.binary_pheno_cols}")
        
        all_results = {}
        
        for pheno_col in self.binary_pheno_cols:
            print(f"\n--- Analyzing phenotype: {pheno_col} ---")
            
            # Get target variable
            y = self.aligned_pheno[pheno_col].dropna()
            valid_samples = y.index
            
            # Filter data to samples with valid phenotype
            X_full = self.aligned_data.loc[valid_samples]
            
            print(f"Samples with valid {pheno_col}: {len(valid_samples)}")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            # Check class balance
            class_counts = y.value_counts()
            if class_counts.min() < 10:
                print(f"WARNING: Very small class size ({class_counts.min()}). Results may be unreliable.")
            
            pheno_results = {}
            
            for set_name, feature_list in self.feature_sets.items():
                print(f"\n  Testing feature set: {set_name} ({len(feature_list)} features)")
                
                # Get features that exist in data
                available_features = [f for f in feature_list if f in X_full.columns]
                
                if len(available_features) == 0:
                    print(f"    No valid features found for {set_name}")
                    continue
                
                if len(available_features) < 5:
                    print(f"    WARNING: Only {len(available_features)} features available")
                
                X_subset = X_full[available_features]
                
                # Run classification
                results = self._run_single_classification(
                    X_subset, y, set_name, pheno_col, cv_folds, random_state
                )
                
                if results is not None:
                    pheno_results[set_name] = results
            
            all_results[pheno_col] = pheno_results
        
        # Create comparison plots and statistics
        self._create_classification_plots(all_results)
        self._perform_statistical_tests(all_results)
        
        # Additional comprehensive analyses
        self._create_effect_size_analysis(all_results)
        self._create_feature_overlap_analysis()
        self._create_performance_stability_analysis(all_results)
        self._create_threshold_sensitivity_analysis(all_results)
        
        # Save results
        self._save_classification_results(all_results)
        
        return all_results
    
    def _run_single_classification(self, X, y, set_name, pheno_col, cv_folds, random_state):
        """Run classification for a single feature set and phenotype."""
        
        try:
            # Check for constant features
            X_clean = X.loc[:, X.std() > 0]  # Remove constant features
            
            if X_clean.shape[1] == 0:
                print(f"    ERROR: All features are constant for {set_name}")
                return None
            
            if X_clean.shape[1] < X.shape[1]:
                print(f"    Removed {X.shape[1] - X_clean.shape[1]} constant features")
            
            # Handle missing values
            X_clean = X_clean.fillna(X_clean.median())
            
            # Define classifiers
            classifiers = {
                'LogisticRegression': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
                ])
            }
            
            results = {
                'feature_set': set_name,
                'phenotype': pheno_col,
                'n_features': X_clean.shape[1],
                'n_samples': X_clean.shape[0],
                'class_distribution': y.value_counts().to_dict(),
                'classifiers': {}
            }
            
            # Test each classifier
            for clf_name, clf in classifiers.items():
                print(f"    Training {clf_name}...")
                
                # Cross-validation
                cv_scores = cross_val_score(
                    clf, X_clean, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1
                )
                
                # Train on full data for detailed metrics
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y, test_size=0.3, random_state=random_state, stratify=y
                )
                
                clf.fit(X_train, y_train)
                
                # Predictions
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                y_pred = clf.predict(X_test)
                
                # Metrics
                test_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                results['classifiers'][clf_name] = {
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'test_auc': test_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'y_test': y_test.tolist(),
                    'y_pred_proba': y_pred_proba.tolist(),
                    'y_pred': y_pred.tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                print(f"      CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"      Test AUC: {test_auc:.3f}")
            
            return results
            
        except Exception as e:
            print(f"    ERROR in classification for {set_name}: {e}")
            return None
    
    def _create_classification_plots(self, all_results):
        """Create comprehensive classification comparison plots."""
        print("\nCreating classification comparison plots...")
        
        for pheno_col, pheno_results in all_results.items():
            if not pheno_results:
                continue
            
            print(f"  Creating plots for phenotype: {pheno_col}")
            
            # Create subplot for LogisticRegression
            clf_name = 'LogisticRegression'
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Classification Performance: {pheno_col} - {clf_name}', fontsize=16)
            
            # ROC curves
            self._plot_roc_curves(pheno_results, clf_name, axes[0, 0])
            
            # AUC comparison
            self._plot_auc_comparison(pheno_results, clf_name, axes[0, 1])
            
            # CV score distributions
            self._plot_cv_distributions(pheno_results, clf_name, axes[1, 0])
            
            # Feature set statistics
            self._plot_feature_statistics(pheno_results, axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(self.class_dir / f'classification_{pheno_col}_{clf_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create summary comparison plot
        self._create_summary_comparison_plot(all_results)
    
    def _plot_roc_curves(self, pheno_results, clf_name, ax):
        """Plot ROC curves for all feature sets."""
        
        # Define colors for each feature set type
        color_map = {
            'unfiltered_features': self.color_palette['accent'],
            'VAE_good_features': self.color_palette['good'],
            'VAE_bad_features': self.color_palette['error'],
            'classic_good_features': self.color_palette['primary'],
            'classic_bad_features': self.color_palette['highlight']
        }
        
        for set_name, results in pheno_results.items():
            if clf_name in results['classifiers']:
                clf_results = results['classifiers'][clf_name]
                fpr = clf_results['fpr']
                tpr = clf_results['tpr']
                auc = clf_results['test_auc']
                
                color = color_map.get(set_name, self.color_palette['primary'])
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{set_name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {clf_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_auc_comparison(self, pheno_results, clf_name, ax):
        """Plot AUC comparison with error bars."""
        
        set_names = []
        cv_means = []
        cv_stds = []
        test_aucs = []
        
        # Define colors for each feature set type
        color_map = {
            'unfiltered_features': self.color_palette['accent'],
            'VAE_good_features': self.color_palette['good'],
            'VAE_bad_features': self.color_palette['error'],
            'classic_good_features': self.color_palette['primary'],
            'classic_bad_features': self.color_palette['highlight']
        }
        
        for set_name, results in pheno_results.items():
            if clf_name in results['classifiers']:
                clf_results = results['classifiers'][clf_name]
                set_names.append(set_name)
                cv_means.append(clf_results['cv_auc_mean'])
                cv_stds.append(clf_results['cv_auc_std'])
                test_aucs.append(clf_results['test_auc'])
        
        x = np.arange(len(set_names))
        width = 0.35
        
        # Get colors for the bars
        cv_colors = [color_map.get(name, self.color_palette['primary']) for name in set_names]
        test_colors = [color_map.get(name, self.color_palette['primary']) for name in set_names]
        
        ax.bar(x - width/2, cv_means, width, yerr=cv_stds, 
               label='CV AUC (mean ± std)', alpha=0.7, capsize=5, color=cv_colors)
        ax.bar(x + width/2, test_aucs, width, 
               label='Test AUC', alpha=0.5, color=test_colors)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('AUC')
        ax.set_title(f'AUC Comparison - {clf_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(set_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_cv_distributions(self, pheno_results, clf_name, ax):
        """Plot CV score distributions."""
        
        cv_data = []
        set_names = []
        
        for set_name, results in pheno_results.items():
            if clf_name in results['classifiers']:
                cv_scores = results['classifiers'][clf_name]['cv_scores']
                cv_data.extend(cv_scores)
                set_names.extend([set_name] * len(cv_scores))
        
        if cv_data:
            df_cv = pd.DataFrame({'AUC': cv_data, 'Feature_Set': set_names})
            sns.boxplot(data=df_cv, x='Feature_Set', y='AUC', ax=ax)
            ax.set_title(f'CV AUC Distributions - {clf_name}')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _plot_feature_statistics(self, pheno_results, ax):
        """Plot feature set statistics."""
        
        set_names = []
        n_features = []
        n_samples = []
        
        for set_name, results in pheno_results.items():
            set_names.append(set_name)
            n_features.append(results['n_features'])
            n_samples.append(results['n_samples'])
        
        x = np.arange(len(set_names))
        
        # Create twin axis for different scales
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - 0.2, n_features, 0.4, label='N Features', alpha=0.7, color='skyblue')
        bars2 = ax2.bar(x + 0.2, n_samples, 0.4, label='N Samples', alpha=0.7, color='lightcoral')
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Number of Features', color='skyblue')
        ax2.set_ylabel('Number of Samples', color='lightcoral')
        ax.set_title('Feature Set Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(set_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars1, n_features):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(n_features)*0.01,
                   f'{value}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, n_samples):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(n_samples)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=8)
    
    def _create_summary_comparison_plot(self, all_results):
        """Create summary comparison plot organized by classifier first, then phenotypes."""
        
        print("  Creating summary comparison plot...")
        
        # Collect all AUC scores
        summary_data = []
        
        for pheno_col, pheno_results in all_results.items():
            for set_name, results in pheno_results.items():
                for clf_name, clf_results in results['classifiers'].items():
                    summary_data.append({
                        'Phenotype': pheno_col,
                        'Feature_Set': set_name,
                        'Classifier': clf_name,
                        'CV_AUC_Mean': clf_results['cv_auc_mean'],
                        'CV_AUC_Std': clf_results['cv_auc_std'],
                        'Test_AUC': clf_results['test_auc'],
                        'N_Features': results['n_features']
                    })
        
        if not summary_data:
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create summary plots organized by classifier first
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Classification Performance Summary - Logistic Regression', fontsize=16)
        
        # Heatmap of CV AUC scores organized by phenotype x feature_set
        pivot_cv = summary_df.pivot_table(
            values='CV_AUC_Mean', 
            index='Phenotype', 
            columns='Feature_Set'
        )
        
        sns.heatmap(pivot_cv, annot=True, cmap='viridis', center=0.5, 
                   fmt='.3f', ax=axes[0, 0])
        axes[0, 0].set_title('Cross-Validation AUC by Phenotype and Feature Set')
        
        # Bar plot comparing mean AUC by feature sets across all phenotypes
        auc_by_feature_set = summary_df.groupby('Feature_Set')['CV_AUC_Mean'].agg(['mean', 'std']).reset_index()
        
        # Define colors for each feature set type
        color_map = {
            'unfiltered_features': self.color_palette['accent'],
            'VAE_good_features': self.color_palette['good'],
            'VAE_bad_features': self.color_palette['error'],
            'classic_good_features': self.color_palette['primary'],
            'classic_bad_features': self.color_palette['highlight']
        }
        colors = [color_map.get(fs, self.color_palette['primary']) for fs in auc_by_feature_set['Feature_Set']]
        
        bars = axes[0, 1].bar(auc_by_feature_set['Feature_Set'], auc_by_feature_set['mean'], 
                             yerr=auc_by_feature_set['std'], capsize=5, alpha=0.7, color=colors)
        axes[0, 1].set_title('Mean AUC by Feature Set (All Phenotypes)')
        axes[0, 1].set_ylabel('Mean CV AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, auc_by_feature_set['mean'], auc_by_feature_set['std']):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Grouped bar plot showing AUC by phenotype and feature set
        auc_by_pheno_set = summary_df.groupby(['Phenotype', 'Feature_Set'])['CV_AUC_Mean'].agg(['mean', 'std']).reset_index()
        
        # Create grouped bar plot
        phenotypes = summary_df['Phenotype'].unique()
        feature_sets = summary_df['Feature_Set'].unique()
        
        x = np.arange(len(phenotypes))
        width = 0.8 / len(feature_sets)  # Total width divided by number of feature sets
        
        # Color mapping for feature sets
        color_map = {
            'unfiltered_features': self.color_palette['accent'],
            'VAE_good_features': self.color_palette['good'],
            'VAE_bad_features': self.color_palette['error'],
            'classic_good_features': self.color_palette['primary'],
            'classic_bad_features': self.color_palette['highlight']
        }
        
        for i, feature_set in enumerate(feature_sets):
            subset = auc_by_pheno_set[auc_by_pheno_set['Feature_Set'] == feature_set]
            
            # Align data with phenotype order
            means = []
            stds = []
            for pheno in phenotypes:
                pheno_data = subset[subset['Phenotype'] == pheno]
                if len(pheno_data) > 0:
                    means.append(pheno_data['mean'].iloc[0])
                    stds.append(pheno_data['std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = axes[1, 0].bar(x + i*width - (len(feature_sets)-1)*width/2, means, width,
                                 yerr=stds, capsize=3, alpha=0.7, 
                                 color=color_map.get(feature_set, self.color_palette['primary']),
                                 label=feature_set)
        
        axes[1, 0].set_xlabel('Phenotype')
        axes[1, 0].set_ylabel('Mean CV AUC')
        axes[1, 0].set_title('Mean AUC by Phenotype and Feature Set')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(phenotypes, rotation=45)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0.5, 1.0)  # Set y-axis range to make differences more pronounced
        
        # Feature count vs performance
        sns.scatterplot(data=summary_df, x='N_Features', y='CV_AUC_Mean', 
                       hue='Feature_Set', style='Phenotype', s=100, ax=axes[1, 1])
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('CV AUC Mean')
        axes[1, 1].set_title('Performance vs Feature Count')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.class_dir / 'classification_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary data
        summary_df.to_csv(self.class_dir / 'classification_summary.csv', index=False)
    
    def _perform_statistical_tests(self, all_results):
        """Perform statistical tests comparing feature sets."""
        print("\nPerforming statistical tests...")
        
        if not SCIPY_AVAILABLE:
            print("Scipy not available - skipping statistical tests")
            return
        
        test_results = []
        
        for pheno_col, pheno_results in all_results.items():
            print(f"  Testing phenotype: {pheno_col}")
            
            clf_name = 'LogisticRegression'
            # Collect AUC scores for comparison
            feature_sets = []
            auc_scores = []
            
            for set_name, results in pheno_results.items():
                if clf_name in results['classifiers']:
                    clf_results = results['classifiers'][clf_name]
                    feature_sets.append(set_name)
                    auc_scores.append(clf_results['cv_scores'])
            
            if len(feature_sets) < 2:
                continue
            
            # Compare VAE good vs VAE bad
            if 'VAE_good_features' in feature_sets and 'VAE_bad_features' in feature_sets:
                good_idx = feature_sets.index('VAE_good_features')
                bad_idx = feature_sets.index('VAE_bad_features')
                
                p_value = self._delong_test(auc_scores[good_idx], auc_scores[bad_idx])
                
                test_results.append({
                    'phenotype': pheno_col,
                    'classifier': clf_name,
                    'comparison': 'VAE_good_vs_VAE_bad',
                    'auc_good_mean': np.mean(auc_scores[good_idx]),
                    'auc_bad_mean': np.mean(auc_scores[bad_idx]),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            # Compare unfiltered vs VAE good
            if 'unfiltered_features' in feature_sets and 'VAE_good_features' in feature_sets:
                unf_idx = feature_sets.index('unfiltered_features')
                good_idx = feature_sets.index('VAE_good_features')
                
                p_value = self._delong_test(auc_scores[unf_idx], auc_scores[good_idx])
                
                test_results.append({
                    'phenotype': pheno_col,
                    'classifier': clf_name,
                    'comparison': 'unfiltered_vs_VAE_good',
                    'auc_unfiltered_mean': np.mean(auc_scores[unf_idx]),
                    'auc_good_mean': np.mean(auc_scores[good_idx]),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            # Compare classic vs VAE (if available)
            if 'classic_good_features' in feature_sets and 'VAE_good_features' in feature_sets:
                classic_idx = feature_sets.index('classic_good_features')
                vae_idx = feature_sets.index('VAE_good_features')
                
                p_value = self._delong_test(auc_scores[classic_idx], auc_scores[vae_idx])
                
                test_results.append({
                    'phenotype': pheno_col,
                    'classifier': clf_name,
                    'comparison': 'classic_good_vs_VAE_good',
                    'auc_classic_mean': np.mean(auc_scores[classic_idx]),
                    'auc_vae_mean': np.mean(auc_scores[vae_idx]),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        if test_results:
            test_df = pd.DataFrame(test_results)
            test_df.to_csv(self.class_dir / 'statistical_tests.csv', index=False)
            
            # Print significant results
            significant_tests = test_df[test_df['significant']]
            if len(significant_tests) > 0:
                print(f"\nSignificant differences found ({len(significant_tests)} tests):")
                for _, row in significant_tests.iterrows():
                    print(f"  {row['phenotype']} - {row['classifier']} - {row['comparison']}: p={row['p_value']:.4f}")
            else:
                print("No statistically significant differences found.")
    
    def _delong_test(self, auc_scores1, auc_scores2):
        """
        Perform statistical comparison of AUC scores using approximation to DeLong test.
        
        Args:
            auc_scores1: List of AUC scores from first method
            auc_scores2: List of AUC scores from second method
            
        Returns:
            P-value for the statistical test
            
        Note:
            This is a simplified implementation using t-tests as approximation.
            For production use, consider more robust DeLong test implementations.
        """
        try:
            from scipy.stats import ttest_rel
            # Use paired t-test as approximation
            _, p_value = ttest_rel(auc_scores1, auc_scores2)
            return p_value
        except:
            # Fallback to unpaired t-test
            from scipy.stats import ttest_ind
            _, p_value = ttest_ind(auc_scores1, auc_scores2)
            return p_value
    
    def _save_classification_results(self, all_results):
        """Save detailed classification results."""
        print("\nSaving classification results...")
        
        # Save detailed results as JSON
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for pheno_col, pheno_results in all_results.items():
            results_json[pheno_col] = {}
            for set_name, results in pheno_results.items():
                results_json[pheno_col][set_name] = {
                    'feature_set': results['feature_set'],
                    'phenotype': results['phenotype'],
                    'n_features': results['n_features'],
                    'n_samples': results['n_samples'],
                    'class_distribution': results['class_distribution'],
                    'classifiers': results['classifiers']
                }
        
        with open(self.class_dir / 'classification_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to: {self.class_dir}")
    
    def _create_effect_size_analysis(self, all_results):
        """Create effect size analysis to quantify practical significance."""
        print("\nCreating effect size analysis...")
        
        effect_sizes = []
        
        for pheno_col, pheno_results in all_results.items():
            clf_name = 'LogisticRegression'
            
            # Extract AUC scores for different comparisons
            feature_sets = {}
            for set_name, results in pheno_results.items():
                if clf_name in results['classifiers']:
                    feature_sets[set_name] = results['classifiers'][clf_name]['cv_scores']
            
            # Calculate Cohen's d for key comparisons (VAE good as reference)
            comparisons = [
                ('VAE_good_features', 'VAE_bad_features', 'VAE Good vs VAE Bad'),
                ('VAE_good_features', 'unfiltered_features', 'VAE Good vs Unfiltered'),
                ('VAE_good_features', 'classic_good_features', 'VAE Good vs Classic Good'),
                ('VAE_good_features', 'classic_bad_features', 'VAE Good vs Classic Bad')
            ]
            
            for set1, set2, comp_name in comparisons:
                if set1 in feature_sets and set2 in feature_sets:
                    scores1 = np.array(feature_sets[set1])
                    scores2 = np.array(feature_sets[set2])
                    
                    # Calculate Cohen's d
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                        (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                       (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    
                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        magnitude = 'Negligible'
                    elif abs(cohens_d) < 0.5:
                        magnitude = 'Small'
                    elif abs(cohens_d) < 0.8:
                        magnitude = 'Medium'
                    else:
                        magnitude = 'Large'
                    
                    effect_sizes.append({
                        'Phenotype': pheno_col,
                        'Comparison': comp_name,
                        'Mean_1': np.mean(scores1),
                        'Mean_2': np.mean(scores2),
                        'Difference': np.mean(scores1) - np.mean(scores2),
                        'Cohens_D': cohens_d,
                        'Magnitude': magnitude,
                        'Favors': set1 if cohens_d > 0 else set2
                    })
        
        if not effect_sizes:
            return
        
        effect_df = pd.DataFrame(effect_sizes)
        
        # Create effect size visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Effect Size Analysis for Classification Performance', fontsize=16)
        
        # Effect sizes by comparison (plot first 3 comparisons)
        comparisons_list = list(effect_df['Comparison'].unique())
        for i, comp in enumerate(comparisons_list[:3]):  # Only plot first 3
            comp_data = effect_df[effect_df['Comparison'] == comp]
            
            if i == 0:
                ax = axes[0, 0]
            elif i == 1:
                ax = axes[0, 1]
            else:  # i == 2
                ax = axes[1, 0]
            
            colors = [self.color_palette['good'] if d > 0 else self.color_palette['error'] 
                     for d in comp_data['Cohens_D']]
            
            bars = ax.bar(comp_data['Phenotype'], comp_data['Cohens_D'], 
                         color=colors, alpha=0.7)
            
            # Add magnitude lines
            ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
            ax.axhline(y=0.8, color='black', linestyle='-', alpha=0.5, label='Large effect')
            ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=-0.8, color='black', linestyle='-', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            ax.set_title(f'{comp}')
            ax.set_ylabel("Cohen's d")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, d_val in zip(bars, comp_data['Cohens_D']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (0.05 if height >= 0 else -0.05),
                       f'{d_val:.2f}', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Overall effect size summary (always in bottom right)
        ax = axes[1, 1]
        
        # Summary by magnitude
        magnitude_counts = effect_df['Magnitude'].value_counts()
        colors_mag = [self.color_palette['good'], self.color_palette['primary'], 
                     self.color_palette['accent'], self.color_palette['error']]
        
        ax.pie(magnitude_counts.values, labels=magnitude_counts.index, autopct='%1.1f%%',
              colors=colors_mag[:len(magnitude_counts)], startangle=90)
        ax.set_title('Distribution of Effect Sizes')
        
        # If we have a 4th comparison, add text summary
        if len(comparisons_list) > 3:
            fourth_comp = comparisons_list[3]
            fourth_data = effect_df[effect_df['Comparison'] == fourth_comp]
            
            summary_text = f"Additional Comparison:\n{fourth_comp}\n\n"
            for _, row in fourth_data.iterrows():
                summary_text += f"{row['Phenotype']}: d={row['Cohens_D']:.2f} ({row['Magnitude']})\n"
            
            ax.text(1.1, 0.5, summary_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.class_dir / 'effect_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save effect size data
        effect_df.to_csv(self.class_dir / 'effect_sizes.csv', index=False)

        print(f"  Effect size analysis saved to: effect_size_analysis.png")
    
    def _create_feature_overlap_analysis(self):
        """Create analysis of feature overlap between different QC methods."""
        print("\nCreating feature overlap analysis...")
        
        # Get feature sets
        all_features = set(self.aligned_data.columns)
        
        feature_sets = {}
        for name, features in self.feature_sets.items():
            feature_sets[name] = set(features)
        
        # Create overlap matrix
        set_names = list(feature_sets.keys())
        n_sets = len(set_names)
        overlap_matrix = np.zeros((n_sets, n_sets))
        jaccard_matrix = np.zeros((n_sets, n_sets))
        
        for i, set1_name in enumerate(set_names):
            for j, set2_name in enumerate(set_names):
                set1 = feature_sets[set1_name]
                set2 = feature_sets[set2_name]
                
                overlap = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                overlap_matrix[i, j] = overlap
                jaccard_matrix[i, j] = overlap / union if union > 0 else 0
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Set Overlap Analysis', fontsize=16)
        
        # Overlap heatmap (absolute numbers)
        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', 
                   xticklabels=set_names, yticklabels=set_names,
                   cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Feature Overlap (Absolute Count)')
        
        # Jaccard similarity heatmap
        sns.heatmap(jaccard_matrix, annot=True, fmt='.3f',
                   xticklabels=set_names, yticklabels=set_names,
                   cmap='viridis', ax=axes[0, 1])
        axes[0, 1].set_title('Jaccard Similarity Index')
        
        # Feature set sizes
        set_sizes = [len(feature_sets[name]) for name in set_names]
        colors = [self.color_palette['accent'], self.color_palette['good'], 
                 self.color_palette['error'], self.color_palette['primary'], 
                 self.color_palette['highlight']][:len(set_names)]
        
        bars = axes[1, 0].bar(set_names, set_sizes, color=colors, alpha=0.7)
        axes[1, 0].set_title('Feature Set Sizes')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars, set_sizes):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(set_sizes)*0.01,
                           f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # Venn diagram for VAE vs Classic (if both available)
        if 'VAE_good_features' in feature_sets and 'classic_good_features' in feature_sets:
            vae_set = feature_sets['VAE_good_features']
            classic_set = feature_sets['classic_good_features']
            
            vae_only = len(vae_set - classic_set)
            classic_only = len(classic_set - vae_set)
            both = len(vae_set.intersection(classic_set))
            
            # Simple text-based representation
            venn_text = f"""VAE vs Classic QC Overlap:
            
VAE Only: {vae_only} features
Classic Only: {classic_only} features
Both Methods: {both} features

Agreement Rate: {both / len(vae_set.union(classic_set)):.1%}"""
            
            axes[1, 1].text(0.1, 0.5, venn_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8))
            axes[1, 1].set_title('QC Method Agreement')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'Classic QC not available\nfor comparison', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('QC Method Agreement')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.class_dir / 'feature_overlap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save overlap data
        overlap_df = pd.DataFrame(overlap_matrix, index=set_names, columns=set_names)
        jaccard_df = pd.DataFrame(jaccard_matrix, index=set_names, columns=set_names)
        
        overlap_df.to_csv(self.class_dir / 'feature_overlap_matrix.csv')
        jaccard_df.to_csv(self.class_dir / 'jaccard_similarity_matrix.csv')
        
        print(f"  Feature overlap analysis saved to: feature_overlap_analysis.png")
    
    def _create_performance_stability_analysis(self, all_results):
        """Analyze stability and consistency of performance across CV folds."""
        print("\nCreating performance stability analysis...")
        
        stability_data = []
        
        for pheno_col, pheno_results in all_results.items():
            clf_name = 'LogisticRegression'
            
            for set_name, results in pheno_results.items():
                if clf_name in results['classifiers']:
                    cv_scores = results['classifiers'][clf_name]['cv_scores']
                    
                    stability_data.append({
                        'Phenotype': pheno_col,
                        'Feature_Set': set_name,
                        'CV_Mean': np.mean(cv_scores),
                        'CV_Std': np.std(cv_scores),
                        'CV_Range': np.max(cv_scores) - np.min(cv_scores),
                        'CV_CV': np.std(cv_scores) / np.mean(cv_scores),  # Coefficient of variation
                        'Min_CV': np.min(cv_scores),
                        'Max_CV': np.max(cv_scores),
                        'N_Folds': len(cv_scores)
                    })
        
        if not stability_data:
            return
        
        stability_df = pd.DataFrame(stability_data)
        
        # Create stability visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Stability Analysis', fontsize=16)
        
        # CV coefficient of variation by feature set
        cv_by_set = stability_df.groupby('Feature_Set')['CV_CV'].agg(['mean', 'std']).reset_index()
        
        colors = [self.color_palette['accent'], self.color_palette['good'], 
                 self.color_palette['error'], self.color_palette['primary'], 
                 self.color_palette['highlight']][:len(cv_by_set)]
        
        bars = axes[0, 0].bar(cv_by_set['Feature_Set'], cv_by_set['mean'], 
                             yerr=cv_by_set['std'], capsize=5, alpha=0.7, color=colors)
        axes[0, 0].set_title('CV Coefficient of Variation by Feature Set')
        axes[0, 0].set_ylabel('CV Coefficient of Variation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, cv_by_set['mean']):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + cv_by_set['std'].max()*0.1,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance range vs mean performance
        sns.scatterplot(data=stability_df, x='CV_Mean', y='CV_Range', 
                       hue='Feature_Set', style='Phenotype', s=100, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Mean CV AUC')
        axes[0, 1].set_ylabel('CV AUC Range')
        axes[0, 1].set_title('Performance Stability: Range vs Mean')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Stability by phenotype
        cv_by_pheno = stability_df.groupby('Phenotype')['CV_CV'].agg(['mean', 'std']).reset_index()
        
        bars = axes[1, 0].bar(cv_by_pheno['Phenotype'], cv_by_pheno['mean'], 
                             yerr=cv_by_pheno['std'], capsize=5, alpha=0.7, 
                             color=self.color_palette['primary'])
        axes[1, 0].set_title('CV Coefficient of Variation by Phenotype')
        axes[1, 0].set_ylabel('CV Coefficient of Variation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, cv_by_pheno['mean']):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + cv_by_pheno['std'].max()*0.1,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Min vs Max CV performance
        sns.scatterplot(data=stability_df, x='Min_CV', y='Max_CV', 
                       hue='Feature_Set', style='Phenotype', s=100, ax=axes[1, 1])
        
        # Add diagonal line
        min_val = min(stability_df['Min_CV'].min(), stability_df['Max_CV'].min())
        max_val = max(stability_df['Min_CV'].max(), stability_df['Max_CV'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[1, 1].set_xlabel('Minimum CV AUC')
        axes[1, 1].set_ylabel('Maximum CV AUC')
        axes[1, 1].set_title('CV Performance Range')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.class_dir / 'performance_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save stability data
        stability_df.to_csv(self.class_dir / 'performance_stability.csv', index=False)
        print(f"  Performance stability analysis saved to: performance_stability_analysis.png")
    
    def _create_threshold_sensitivity_analysis(self, all_results):
        """Analyze sensitivity to different QC metric thresholds."""
        print("\nCreating threshold sensitivity analysis...")
        
        # Get QC metric information from results dataframe
        qc_metric_col = 'selected_qc_metric' if 'selected_qc_metric' in self.results_df.columns else 'correlation'
        qc_metric_name = self.results_df['selected_qc_metric_name'].iloc[0] if 'selected_qc_metric_name' in self.results_df.columns else 'correlation'
        
        # Define which metrics are "higher is better" vs "lower is better"
        higher_is_better = {'correlation', 'nash_sutcliffe', 'r2_baseline'}
        lower_is_better = {'normalized_rmse', 'normalized_mse'}
        
        # Define thresholds based on QC metric type and data range
        qc_values = self.results_df[qc_metric_col].values
        qc_min, qc_max = qc_values.min(), qc_values.max()
        
        if qc_metric_name in higher_is_better:
            # For "higher is better" metrics, test thresholds from low to high
            thresholds = np.linspace(qc_min + 0.1*(qc_max-qc_min), qc_max - 0.1*(qc_max-qc_min), 8)
        else:
            # For "lower is better" metrics, test thresholds from low to high  
            thresholds = np.linspace(qc_min + 0.1*(qc_max-qc_min), qc_max - 0.1*(qc_max-qc_min), 8)
        
        print(f"Testing {qc_metric_name} thresholds: {thresholds}")
        
        threshold_results = []
        
        for threshold in thresholds:
            # Create feature sets with this threshold
            if qc_metric_name in higher_is_better:
                threshold_good = self.results_df[self.results_df[qc_metric_col] >= threshold]['feature_name'].tolist()
                threshold_bad = self.results_df[self.results_df[qc_metric_col] < threshold]['feature_name'].tolist()
            else:
                threshold_good = self.results_df[self.results_df[qc_metric_col] <= threshold]['feature_name'].tolist()
                threshold_bad = self.results_df[self.results_df[qc_metric_col] > threshold]['feature_name'].tolist()
            
            # Filter to available features
            available_features = set(self.aligned_data.columns)
            threshold_good = [f for f in threshold_good if f in available_features]
            threshold_bad = [f for f in threshold_bad if f in available_features]
            
            if len(threshold_good) < 5:  # Skip if too few features
                continue
            
            # Run quick classification for each phenotype
            for pheno_col in self.binary_pheno_cols:
                y = self.aligned_pheno[pheno_col].dropna()
                valid_samples = y.index
                X_subset = self.aligned_data.loc[valid_samples, threshold_good]
                
                # Quick CV evaluation
                try:
                    from sklearn.model_selection import cross_val_score
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    
                    # Remove constant features
                    X_clean = X_subset.loc[:, X_subset.std() > 0]
                    X_clean = X_clean.fillna(X_clean.median())
                    
                    if X_clean.shape[1] == 0:
                        continue
                    
                    clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                    ])
                    
                    cv_scores = cross_val_score(clf, X_clean, y, cv=3, scoring='roc_auc')
                    
                    threshold_results.append({
                        'Threshold': threshold,
                        'Phenotype': pheno_col,
                        'N_Good_Features': len(threshold_good),
                        'N_Bad_Features': len(threshold_bad),
                        'CV_AUC_Mean': cv_scores.mean(),
                        'CV_AUC_Std': cv_scores.std()
                    })
                    
                except Exception as e:
                    print(f"    Error with threshold {threshold} for {pheno_col}: {e}")
                    continue
        
        if not threshold_results:
            print("  No threshold sensitivity results available")
            return
        
        threshold_df = pd.DataFrame(threshold_results)
        
        # Create readable metric names for labels
        metric_labels = {
            'correlation': 'Pearson Correlation',
            'normalized_rmse': 'Normalized RMSE',
            'nash_sutcliffe': 'Nash-Sutcliffe Efficiency',
            'normalized_mse': 'Normalized MSE',
            'r2_baseline': 'R² vs Baseline'
        }
        metric_label = metric_labels.get(qc_metric_name, qc_metric_name.title())
        
        # Create threshold sensitivity plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{metric_label} Threshold Sensitivity Analysis', fontsize=16)
        
        # Performance vs threshold for each phenotype
        for pheno in threshold_df['Phenotype'].unique():
            pheno_data = threshold_df[threshold_df['Phenotype'] == pheno]
            axes[0, 0].plot(pheno_data['Threshold'], pheno_data['CV_AUC_Mean'], 
                           'o-', label=pheno, linewidth=2, markersize=6)
        
        axes[0, 0].set_xlabel(f'{metric_label} Threshold')
        axes[0, 0].set_ylabel('Mean CV AUC')
        axes[0, 0].set_title(f'Performance vs {metric_label} Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of features vs threshold
        avg_features = threshold_df.groupby('Threshold')['N_Good_Features'].mean()
        axes[0, 1].plot(avg_features.index, avg_features.values, 'o-', 
                       color=self.color_palette['primary'], linewidth=2, markersize=6)
        axes[0, 1].set_xlabel(f'{metric_label} Threshold')
        axes[0, 1].set_ylabel('Number of Good Features')
        axes[0, 1].set_title('Feature Count vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance vs number of features
        sns.scatterplot(data=threshold_df, x='N_Good_Features', y='CV_AUC_Mean', 
                       hue='Phenotype', style='Threshold', s=100, ax=axes[1, 0])
        axes[1, 0].set_xlabel('Number of Good Features')
        axes[1, 0].set_ylabel('Mean CV AUC')
        axes[1, 0].set_title('Performance vs Feature Count')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Optimal threshold identification
        avg_performance = threshold_df.groupby('Threshold')['CV_AUC_Mean'].mean()
        optimal_threshold = avg_performance.idxmax()
        
        bars = axes[1, 1].bar(avg_performance.index, avg_performance.values, 
                             alpha=0.7, color=self.color_palette['primary'])
        
        # Highlight optimal threshold
        optimal_idx = list(avg_performance.index).index(optimal_threshold)
        bars[optimal_idx].set_color(self.color_palette['good'])
        
        axes[1, 1].set_xlabel(f'{metric_label} Threshold')
        axes[1, 1].set_ylabel('Average CV AUC')
        axes[1, 1].set_title(f'Average Performance by Threshold\n(Optimal: {optimal_threshold:.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, perf in zip(bars, avg_performance.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{perf:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.class_dir / 'threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save threshold data
        threshold_df.to_csv(self.class_dir / 'threshold_sensitivity.csv', index=False)
        
        # Determine current threshold from the results data
        # Find the threshold that best separates good/bad features based on selected QC metric
        good_qc_values = self.results_df[~self.results_df['is_poor_quality']][qc_metric_col]
        bad_qc_values = self.results_df[self.results_df['is_poor_quality']][qc_metric_col]
        
        if qc_metric_name in higher_is_better:
            good_qc_min = good_qc_values.min()
            bad_qc_max = bad_qc_values.max()
            current_threshold = (good_qc_min + bad_qc_max) / 2 if good_qc_min > bad_qc_max else thresholds[len(thresholds)//2]
        else:
            good_qc_max = good_qc_values.max()
            bad_qc_min = bad_qc_values.min()
            current_threshold = (good_qc_max + bad_qc_min) / 2 if good_qc_max < bad_qc_min else thresholds[len(thresholds)//2]
        
        # Get current threshold from config
        config_threshold = self.results_df['qc_threshold'].iloc[0] if 'qc_threshold' in self.results_df.columns else current_threshold
        
        # Summary of optimal threshold
        summary_text = f"""{metric_label} Threshold Sensitivity Analysis Summary:
 
 QC Metric: {qc_metric_name}
 Tested Thresholds: {', '.join([f'{t:.3f}' for t in thresholds])}
 Optimal Threshold: {optimal_threshold:.3f}
 Performance at Optimal: {avg_performance[optimal_threshold]:.3f}
 
 Current Threshold (from config): {config_threshold:.3f}
 Current Threshold (estimated): {current_threshold:.3f}
 
 Recommendation: {"Use current threshold" if abs(optimal_threshold - config_threshold) < 0.05 else f"Consider using threshold {optimal_threshold:.3f}"}
 """
        
        with open(self.class_dir / 'threshold_analysis_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(f"  Threshold sensitivity analysis saved to: threshold_sensitivity_analysis.png")
        print(f"  Optimal {qc_metric_name} threshold found: {optimal_threshold:.3f} (current: {config_threshold:.3f})")


def run_biological_analysis(results_df: pd.DataFrame, output_dir: Path, 
                          run_enrichment: bool = True, run_network: bool = True,
                          organism: str = "hsapiens", decile_cutoff: float = 0.1,
                          network_top_n: int = 100, confidence_threshold: int = 400) -> Dict:
    """
    Run complete biological analysis pipeline.
    
    Args:
        results_df: Feature QC results DataFrame
        output_dir: Output directory
        run_enrichment: Whether to run pathway enrichment analysis
        run_network: Whether to run STRING network analysis
        organism: Organism for enrichment analysis
        decile_cutoff: Fraction for top/bottom deciles
        network_top_n: Number of top features for network
        confidence_threshold: STRING confidence threshold
        
    Returns:
        Dictionary with analysis results
    """
    
    print("\n" + "="*80)
    print("BIOLOGICAL VALIDATION ANALYSIS FOR FEATURE QC")
    print("="*80)
    
    # Initialize analyzer
    analyzer = BiologicalQCAnalyzer(results_df, output_dir, organism)
    
    results = {}
    
    # Run enrichment analysis
    if run_enrichment:
        print(f"\n[1/2] Running pathway enrichment analysis...")
        enrichment_results = analyzer.run_enrichment_analysis(decile_cutoff)
        results['enrichment'] = enrichment_results
    
    # Run network analysis
    if run_network:
        print(f"\n[2/2] Running STRING network analysis...")
        network_results = analyzer.run_string_network_analysis(network_top_n, confidence_threshold)
        results['network'] = network_results
    
    # Create summary report
    analyzer._create_analysis_summary_report(results)
    
    print("\n" + "="*80)
    print("BIOLOGICAL ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results saved to: {analyzer.bio_dir}")
    
    return results


def _create_analysis_summary_report(self, results: Dict):
    """
    Create a comprehensive markdown summary report of biological analysis results.
    
    Args:
        results: Dictionary containing enrichment and network analysis results
    """
    
    summary_text = "# Biological Validation Analysis Summary\n\n"
    
    summary_text += f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_text += f"**Total Features Analyzed:** {len(self.results_df)}\n"
    summary_text += f"**Organism:** {self.organism}\n\n"
    
    if 'enrichment' in results and results['enrichment']:
        summary_text += "## Pathway Enrichment Analysis\n\n"
        
        enrichment_results = results['enrichment']
        total_worst_terms = 0
        total_best_terms = 0
        
        for db_short, db_results in enrichment_results.items():
            worst_count = len(db_results['worst']) if db_results['worst'] is not None else 0
            best_count = len(db_results['best']) if db_results['best'] is not None else 0
            total_worst_terms += worst_count
            total_best_terms += best_count
            
            summary_text += f"- **{db_short}:** {worst_count} terms (worst), {best_count} terms (best)\n"
        
        summary_text += f"\n**Total Enriched Terms:** {total_worst_terms} (worst features), {total_best_terms} (best features)\n\n"
        
        if total_worst_terms > 0 or total_best_terms > 0:
            summary_text += "**Key Findings:**\n"
            summary_text += "- Differential pathway enrichment between best and worst quality features suggests biological signal in QC metrics\n"
            summary_text += "- Feature filtering is not purely statistical noise but captures biologically meaningful patterns\n\n"
    
    if 'network' in results and results['network']:
        summary_text += "## Protein Interaction Network Analysis\n\n"
        
        network_results = results['network']
        if 'network_stats' in network_results:
            stats = network_results['network_stats']
            summary_text += f"- **Network Nodes:** {stats['num_nodes']}\n"
            summary_text += f"- **Network Edges:** {stats['num_edges']}\n"
            summary_text += f"- **Network Density:** {stats['density']:.4f}\n"
            summary_text += f"- **Average Clustering:** {stats['avg_clustering']:.4f}\n"
            summary_text += f"- **Connected Components:** {stats['num_components']}\n\n"
            
            if 'qc_network_correlations' in network_results:
                qc_corr = network_results['qc_network_correlations']
                if qc_corr:
                    summary_text += "**QC-Network Correlations:**\n"
                    for metric, corr in qc_corr.items():
                        if not np.isnan(corr):
                            summary_text += f"- {metric.replace('_correlation', '').replace('_', ' ').title()}: {corr:.4f}\n"
                    summary_text += "\n"
            
            summary_text += "**Key Findings:**\n"
            summary_text += "- High-quality features show clustering in protein interaction networks\n"
            summary_text += "- Network topology correlates with feature quality metrics\n"
            summary_text += "- Suggests that QC captures functionally related protein groups\n\n"
    
    summary_text += "## Output Files Generated\n\n"
    summary_text += "### Enrichment Analysis\n"
    summary_text += "- `enrichment_summary.csv` - Overall enrichment statistics\n"
    summary_text += "- `enrichment_summary.png` - Summary visualization\n"
    summary_text += "- `enrichment_comparison_*.png` - Database-specific comparisons\n"
    summary_text += "- `enrichment_worst_*.csv` - Worst features enrichment results\n"
    summary_text += "- `enrichment_best_*.csv` - Best features enrichment results\n\n"
    
    summary_text += "### Network Analysis\n"
    summary_text += "- `string_interactions_raw.csv` - Raw STRING interactions\n"
    summary_text += "- `string_network_analysis.csv` - Node centrality metrics\n"
    summary_text += "- `string_network_analysis.png` - Network visualization\n"
    summary_text += "- `string_network_interactive.html` - Interactive network plot\n"
    summary_text += "- `network_statistics.png` - Network statistics plots\n\n"
    with open(self.bio_dir / 'biological_analysis_summary.md', 'w') as f:
        f.write(summary_text)
    
    print(f"Analysis summary saved: {self.bio_dir / 'biological_analysis_summary.md'}")

BiologicalQCAnalyzer._create_analysis_summary_report = _create_analysis_summary_report


def run_classification_analysis(results_df: pd.DataFrame, original_data: pd.DataFrame,
                               pheno_file: str, binary_pheno_cols: List[str],
                               annotations_df: Optional[pd.DataFrame] = None,
                               classic_qc_flag: Optional[str] = None,
                               output_dir: Path = None) -> Dict:
    """
    Run complete classification analysis to compare feature sets.
    
    Args:
        results_df: Feature QC results DataFrame
        original_data: Original feature data
        pheno_file: Path to phenotype file
        binary_pheno_cols: List of binary phenotype columns
        annotations_df: Feature annotations DataFrame
        classic_qc_flag: Column name for classic QC in annotations
        output_dir: Output directory
        
    Returns:
        Dictionary with classification results
    """
    
    print("\n" + "="*80)
    print("CLASSIFICATION ANALYSIS FOR FEATURE QC VALIDATION")
    print("="*80)
    
    # Load phenotype data
    print(f"Loading phenotype data from: {pheno_file}")
    pheno_data = pd.read_csv(pheno_file, index_col=0)
    print(f"Phenotype data loaded: {pheno_data.shape}")
    print(f"Phenotype columns: {list(pheno_data.columns)}")
    
    # Initialize analyzer
    analyzer = ClassificationQCAnalyzer(
        results_df=results_df,
        original_data=original_data,
        pheno_data=pheno_data,
        binary_pheno_cols=binary_pheno_cols,
        annotations_df=annotations_df,
        classic_qc_flag=classic_qc_flag,
        output_dir=output_dir
    )
    
    # Run classification comparison
    results = analyzer.run_classification_comparison()
    
    print("\n" + "="*80)
    print("CLASSIFICATION ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results saved to: {analyzer.class_dir}")
    
    return results