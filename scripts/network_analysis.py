"""
Network Analysis Module for Proteomics Imputation Analysis
Integrates PPI and GRI networks to understand biological context of imputation quality

Research Questions Addressed:
1. Do highly connected proteins have better imputation quality?
2. Are protein complexes imputed more consistently?
3. Can network topology predict imputation difficulty?
4. Do regulatory relationships affect cross-platform transferability?
5. How do hub proteins perform in imputation tasks?
6. Can network-based clustering reveal performance patterns?

Color Palettes (Scientific Journal Standard):
Group 1 (Current): #e64b35, #4dbbd5, #00a087, #3c5488, #f39b7f
Group 2 (Alternative): #bc3c29, #0072b5, #e18727, #20854e, #7876b1  
Group 3 (Alternative): #374e55, #df8f44, #00a1d5, #b24745, #79af97
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalyzer:
    """Network-based analysis for proteomics imputation quality.
    
    This class provides comprehensive network analysis capabilities for evaluating
    proteomics imputation methods using protein-protein interaction (PPI) and
    gene regulatory interaction (GRI) networks. It generates publication-quality
    figures addressing key research questions about network topology and imputation
    performance relationships.
    
    Attributes:
        output_dir: Directory for saving outputs
        figures_dir: Directory for saving figures
        colors: Color palette for visualizations
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Scientific journal color palette (colorblind-safe)
        # Colors commonly used in Nature, Science, Cell publications
        self.colors = {
            'primary': '#e64b35',      # Red (230,75,53)
            'secondary': '#4dbbd5',    # Light Blue (77,187,213)
            'accent': '#00a087',       # Teal (0,160,135)
            'neutral': '#3c5488',      # Dark Blue (60,84,136)
            'highlight': '#f39b7f',    # Light Red (243,155,127)
            'alternative_1': '#bc3c29', # Dark Red (188,60,41)
            'alternative_2': '#0072b5', # Blue (0,114,181)
            'alternative_3': '#e18727', # Orange (225,135,39)
            'alternative_4': '#20854e', # Green (32,133,78)
            'alternative_5': '#7876b1'  # Purple (120,118,177)
        }
        
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    def load_and_process_networks(self, ppi_file: Optional[str] = None, 
                                 gri_file: Optional[str] = None,
                                 feature_names: List[str] = None,
                                 platform_a_features: List[str] = None,
                                 platform_b_features: List[str] = None) -> Dict[str, Any]:
        """Load and process network files for analysis.
        
        Args:
            ppi_file: Path to PPI network file (tab-separated)
            gri_file: Path to GRI network file (tab-separated) 
            feature_names: List of all feature names to map
            platform_a_features: List of platform A specific features
            platform_b_features: List of platform B specific features
            
        Returns:
            Dictionary containing processed network data, mappings, and metrics
        """
        print("🔗 Loading network data...")
        
        network_data = {
            'ppi_network': None,
            'gri_network': None, 
            'feature_mapping': {},
            'platform_a_mapping': {},
            'platform_b_mapping': {},
            'network_metrics': {},
            'summary': {}
        }
        
        if feature_names is None:
            feature_names = []
        
        if ppi_file and Path(ppi_file).exists():
            network_data['ppi_network'] = self._load_ppi_network(ppi_file)
            
        if gri_file and Path(gri_file).exists():
            network_data['gri_network'] = self._load_gri_network(gri_file)
        
        if feature_names and (network_data['ppi_network'] or network_data['gri_network']):
            network_data['feature_mapping'] = self._map_features_to_network(
                feature_names, network_data['ppi_network'], network_data['gri_network'])
            
            if platform_a_features:
                network_data['platform_a_mapping'] = self._map_features_to_network(
                    platform_a_features, network_data['ppi_network'], network_data['gri_network'])
                print(f"  🔗 Platform A: {len(network_data['platform_a_mapping'])}/{len(platform_a_features)} features mapped")
                
            if platform_b_features:
                network_data['platform_b_mapping'] = self._map_features_to_network(
                    platform_b_features, network_data['ppi_network'], network_data['gri_network'])
                print(f"  🔗 Platform B: {len(network_data['platform_b_mapping'])}/{len(platform_b_features)} features mapped")
            
            total_mapped = len(network_data['feature_mapping'])
            platform_a_mapped = len(network_data.get('platform_a_mapping', {}))
            platform_b_mapped = len(network_data.get('platform_b_mapping', {}))
            
            if total_mapped == 0:
                print("  ❌ No features mapped to networks - network analysis will be skipped")
            elif platform_a_mapped == 0 and platform_b_mapped == 0:
                print("  ❌ No features from either platform mapped to networks - network analysis will be skipped")
            elif platform_a_mapped == 0:
                print("  ⚠️  No Platform A features mapped to networks - limited analysis possible")
            elif platform_b_mapped == 0:
                print("  ⚠️  No Platform B features mapped to networks - limited analysis possible")
            else:
                print("  ✅ Both platforms have features mapped to networks - full analysis possible")
            
            if total_mapped > 0:
                network_data['network_metrics'] = self._calculate_network_metrics(
                    network_data['ppi_network'], network_data['gri_network'], 
                    network_data['feature_mapping'])
            
            network_data['summary'] = self._generate_summary(network_data)
        
        return network_data
    
    def _load_ppi_network(self, ppi_file: str) -> Optional[nx.Graph]:
        """Load PPI network from file"""
        try:
            print(f"  📊 Loading PPI from {ppi_file}")
            df = pd.read_csv(ppi_file, sep='\t')
            
            cols = df.columns.str.upper()
            gene1_col = gene2_col = None
            
            for col in df.columns:
                col_up = col.upper()
                if any(x in col_up for x in ['GENE1', 'PROTEIN1', 'FROM']):
                    gene1_col = col
                elif any(x in col_up for x in ['GENE2', 'PROTEIN2', 'TO']):
                    gene2_col = col
            
            if not gene1_col or not gene2_col:
                gene1_col, gene2_col = df.columns[:2]
                print(f"    Using columns: {gene1_col}, {gene2_col}")
            
            edges = []
            for _, row in df.iterrows():
                g1, g2 = str(row[gene1_col]).strip(), str(row[gene2_col]).strip()
                if g1 != g2 and g1 != 'nan' and g2 != 'nan':
                    edges.append((g1, g2))
            
            network = nx.Graph()
            network.add_edges_from(edges)
            
            print(f"    ✅ PPI: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            return network
            
        except Exception as e:
            print(f"    ❌ PPI loading failed: {e}")
            return None
    
    def _load_gri_network(self, gri_file: str) -> Optional[nx.DiGraph]:
        """Load GRI network from file"""
        try:
            print(f"  📊 Loading GRI from {gri_file}")
            df = pd.read_csv(gri_file, sep='\t')
            
            from_col = to_col = None
            
            for col in df.columns:
                col_up = col.upper()
                if any(x in col_up for x in ['FROM', 'SOURCE', 'REGULATOR']):
                    from_col = col
                elif any(x in col_up for x in ['TO', 'TARGET', 'REGULATED']):
                    to_col = col
            
            if not from_col or not to_col:
                from_col, to_col = df.columns[:2]
                print(f"    Using columns: {from_col}, {to_col}")
            
            edges = []
            for _, row in df.iterrows():
                source, target = str(row[from_col]).strip(), str(row[to_col]).strip()
                if source != target and source != 'nan' and target != 'nan':
                    edges.append((source, target))
            
            network = nx.DiGraph()
            network.add_edges_from(edges)
            
            print(f"    ✅ GRI: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            return network
            
        except Exception as e:
            print(f"    ❌ GRI loading failed: {e}")
            return None
    
    def _map_features_to_network(self, features: List[str], 
                                ppi_net: Optional[nx.Graph],
                                gri_net: Optional[nx.DiGraph]) -> Dict[str, str]:
        """Map feature names to network nodes"""
        all_nodes = set()
        if ppi_net:
            all_nodes.update(ppi_net.nodes())
        if gri_net:
            all_nodes.update(gri_net.nodes())
        
        if not all_nodes:
            print("  🚫 No network nodes available")
            return {}
        
        mapping = {}
        for feature in features:
            if feature in all_nodes:
                mapping[feature] = feature
            else:
                variations = [
                    feature.upper(), feature.lower(),
                    feature.replace('_', '-'), feature.replace('-', '_'),
                    feature.split('_')[0], feature.split('-')[0],
                    feature.split('.')[0]  # Handle cases like ENSG00000123456.1
                ]
                for var in variations:
                    if var in all_nodes:
                        mapping[feature] = var
                        break
        
        print(f"  🔗 Mapped {len(mapping)}/{len(features)} features to networks")
        
        if len(mapping) == 0:
            print("  ⚠️  No features could be mapped to network nodes")
            print("  💡 This may be due to different naming conventions between platforms and networks")
            print("     Consider mapping files or network files with compatible identifiers")
        elif len(mapping) < len(features) * 0.1:
            print(f"  ⚠️  Very low mapping rate ({len(mapping)/len(features)*100:.1f}%)")
            print("  💡 Consider checking feature naming conventions")
        
        return mapping
    
    def _calculate_network_metrics(self, ppi_net: Optional[nx.Graph], 
                                  gri_net: Optional[nx.DiGraph],
                                  mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Calculate network metrics for mapped features"""
        metrics = {}
        
        ppi_centrality = {}
        gri_centrality = {}
        
        if ppi_net and len(ppi_net) > 0:
            print(f"  📈 Calculating PPI metrics for {len(ppi_net)} nodes, {len(ppi_net.edges())} edges...")
            
            if len(ppi_net) > 10000:
                print("  ⚠️  Large PPI network detected - using simplified metrics")
                try:
                    ppi_centrality['clustering'] = nx.clustering(ppi_net)
                    print("  ✅ PPI clustering calculated")
                except Exception as e:
                    print(f"  ❌ PPI clustering failed: {e}")
                    ppi_centrality['clustering'] = {}
            else:
                try:
                    print("  🔄 Calculating betweenness centrality...")
                    ppi_centrality['betweenness'] = nx.betweenness_centrality(ppi_net, k=min(1000, len(ppi_net)))
                    print("  ✅ PPI betweenness calculated")
                except Exception as e:
                    print(f"  ❌ PPI betweenness failed: {e}")
                    ppi_centrality['betweenness'] = {}
                
                try:
                    print("  🔄 Calculating eigenvector centrality...")
                    ppi_centrality['eigenvector'] = nx.eigenvector_centrality(ppi_net, max_iter=500, tol=1e-4)
                    print("  ✅ PPI eigenvector calculated")
                except Exception as e:
                    print(f"  ❌ PPI eigenvector failed: {e}")
                    ppi_centrality['eigenvector'] = {}
                
                try:
                    print("  🔄 Calculating clustering coefficient...")
                    ppi_centrality['clustering'] = nx.clustering(ppi_net)
                    print("  ✅ PPI clustering calculated")
                except Exception as e:
                    print(f"  ❌ PPI clustering failed: {e}")
                    ppi_centrality['clustering'] = {}
                
                try:
                    print("  🔄 Detecting communities...")
                    communities = list(nx.community.greedy_modularity_communities(ppi_net))
                    ppi_centrality['community'] = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            ppi_centrality['community'][node] = {'id': i, 'size': len(comm)}
                    print(f"  ✅ Found {len(communities)} PPI communities")
                except Exception as e:
                    print(f"  ❌ PPI community detection failed: {e}")
                    ppi_centrality['community'] = {}
        
        if gri_net and len(gri_net) > 0:
            print(f"  📈 Calculating GRI metrics for {len(gri_net)} nodes, {len(gri_net.edges())} edges...")
            
            if len(gri_net) > 10000:
                print("  ⚠️  Large GRI network detected - using simplified metrics")
            else:
                try:
                    print("  🔄 Calculating PageRank...")
                    gri_centrality['pagerank'] = nx.pagerank(gri_net, max_iter=500, tol=1e-4)
                    print("  ✅ GRI PageRank calculated")
                except Exception as e:
                    print(f"  ❌ GRI PageRank failed: {e}")
                    gri_centrality['pagerank'] = {}
                
                try:
                    print("  🔄 Calculating HITS (hubs and authorities)...")
                    hub_auth = nx.hits(gri_net, max_iter=500, tol=1e-4)
                    gri_centrality['hubs'] = hub_auth[0]
                    gri_centrality['authorities'] = hub_auth[1]
                    print("  ✅ GRI HITS calculated")
                except Exception as e:
                    print(f"  ❌ GRI HITS failed: {e}")
                    gri_centrality['hubs'] = {}
                    gri_centrality['authorities'] = {}
        
        print(f"  📊 Computing metrics for {len(mapping)} mapped features...")
        
        for feature, node in mapping.items():
            feature_metrics = {}
            
            if ppi_net and node in ppi_net:
                feature_metrics.update({
                    'ppi_degree': ppi_net.degree(node),
                    'ppi_clustering': ppi_centrality.get('clustering', {}).get(node, 0),
                    'ppi_betweenness': ppi_centrality.get('betweenness', {}).get(node, 0),
                    'ppi_eigenvector': ppi_centrality.get('eigenvector', {}).get(node, 0)
                })
                
                comm_info = ppi_centrality.get('community', {}).get(node, {'id': -1, 'size': 0})
                feature_metrics.update({
                    'ppi_community': comm_info['id'],
                    'ppi_community_size': comm_info['size']
                })
            else:
                feature_metrics.update({
                    'ppi_degree': 0, 'ppi_clustering': 0, 'ppi_betweenness': 0,
                    'ppi_eigenvector': 0, 'ppi_community': -1, 'ppi_community_size': 0
                })
            
            if gri_net and node in gri_net:
                feature_metrics.update({
                    'gri_in_degree': gri_net.in_degree(node),
                    'gri_out_degree': gri_net.out_degree(node),
                    'gri_total_degree': gri_net.degree(node),
                    'gri_pagerank': gri_centrality.get('pagerank', {}).get(node, 0),
                    'gri_hub_score': gri_centrality.get('hubs', {}).get(node, 0),
                    'gri_authority_score': gri_centrality.get('authorities', {}).get(node, 0)
                })
            else:
                feature_metrics.update({
                    'gri_in_degree': 0, 'gri_out_degree': 0, 'gri_total_degree': 0,
                    'gri_pagerank': 0, 'gri_hub_score': 0, 'gri_authority_score': 0
                })
            
            metrics[feature] = feature_metrics
        
        print(f"  ✅ Network metrics calculated for {len(metrics)} features")
        return metrics
    
    def _generate_summary(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate network summary statistics"""
        summary = {}
        
        if network_data['ppi_network']:
            ppi = network_data['ppi_network']
            summary['ppi'] = {
                'nodes': ppi.number_of_nodes(),
                'edges': ppi.number_of_edges(), 
                'density': nx.density(ppi),
                'avg_clustering': nx.average_clustering(ppi)
            }
        
        if network_data['gri_network']:
            gri = network_data['gri_network']
            summary['gri'] = {
                'nodes': gri.number_of_nodes(),
                'edges': gri.number_of_edges(),
                'density': nx.density(gri)
            }
        
        summary['mapping'] = {
            'mapped_features': len(network_data['feature_mapping']),
            'total_metrics': len(network_data['network_metrics'])
        }
        
        return summary
    
    def generate_network_figures(self, imputation_df: pd.DataFrame, 
                                network_data: Dict[str, Any]) -> List[str]:
        """Generate all network analysis figures"""
        if not network_data.get('network_metrics'):
            print("⚠️  No network data - skipping network analysis")
            return []
        
        if not network_data.get('feature_mapping'):
            print("⚠️  No features mapped to networks - skipping network analysis")
            return []
        
        print("🎨 Generating network analysis figures...")
        
        combined_df = self._prepare_combined_dataframe(imputation_df, network_data)
        if combined_df.empty:
            print("⚠️  No combined data available - network features not found in imputation results")
            return []
        
        print(f"📊 Network analysis will use {len(combined_df)} feature-method combinations")
        
        print(f"📊 Combined data sample:")
        print(f"   Columns: {list(combined_df.columns)}")
        print(f"   Shape: {combined_df.shape}")
        if len(combined_df) > 0:
            print(f"   Sample row:")
            sample_row = combined_df.iloc[0]
            for col in combined_df.columns[:10]:  # Show first 10 columns
                print(f"     {col}: {sample_row[col]}")
            if len(combined_df.columns) > 10:
                print(f"     ... and {len(combined_df.columns) - 10} more columns")
        
        figures = []
        generators = [
            ('22_network_connectivity_vs_imputation_quality', self._plot_connectivity_quality),
            ('23_protein_complex_imputation_consistency', self._plot_complex_consistency),
            ('24_network_topology_imputation_difficulty', self._plot_topology_difficulty),
            ('25_regulatory_network_transferability', self._plot_regulatory_transferability),
            ('26_hub_protein_performance_analysis', self._plot_hub_analysis),
            ('27_network_based_feature_clustering', self._plot_network_clustering)
        ]
        
        for fig_name, plot_func in generators:
            try:
                plot_func(combined_df, network_data)
                figures.append(fig_name)
                print(f"  ✅ {fig_name}")
            except Exception as e:
                print(f"  ❌ {fig_name}: {e}")
        
        return figures
    
    def _prepare_combined_dataframe(self, imputation_df: pd.DataFrame, 
                                   network_data: Dict[str, Any]) -> pd.DataFrame:
        """Combine imputation metrics with network metrics (ENHANCED VERSION)"""
        if not network_data.get('network_metrics'):
            print("⚠️  No network metrics available")
            return pd.DataFrame()
        
        network_df = pd.DataFrame(network_data['network_metrics']).T
        print(f"📊 Network metrics shape: {network_df.shape}")
        print(f"📊 Network metrics columns: {list(network_df.columns)}")
        
        print(f"📊 Imputation metrics shape: {imputation_df.shape}")
        print(f"📊 Imputation metrics columns: {list(imputation_df.columns)}")
        
        if 'feature' in imputation_df.columns:
            print("🔄 Using long format merge on 'feature' column")
            combined_df = imputation_df.merge(network_df, left_on='feature', right_index=True, how='inner')
        else:
            print("🔄 Using wide format join on index")
            combined_df = imputation_df.join(network_df, how='inner')
        
        print(f"📊 Combined data shape: {combined_df.shape}")
        
        if combined_df.empty:
            print("⚠️  Combined dataframe is empty - no matching features between imputation and network data")
            return pd.DataFrame()
        
        essential_cols = ['ppi_degree', 'gri_total_degree', 'gri_pagerank']
        available_cols = [col for col in essential_cols if col in combined_df.columns]
        print(f"📊 Available essential network columns: {available_cols}")
        
        perf_cols = []
        for col in combined_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['_r$', '^r$', '_corr', '_correlation', '_performance', '_mae', '_rmse', '_bias']) or \
               any(col_lower == pattern for pattern in ['r', 'correlation', 'performance', 'mae', 'rmse', 'bias']):
                if combined_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Exclude network metrics that might contain 'r' 
                    if not col.startswith(('ppi_', 'gri_')):
                        perf_cols.append(col)
        
        print(f"📊 Found performance columns: {perf_cols}")
        
        if not perf_cols:
            print("⚠️  No performance columns found in combined data")
        
        return combined_df
    
    def _plot_connectivity_quality(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 22: Network connectivity vs imputation quality (ENHANCED VERSION)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Network Connectivity vs Imputation Quality', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        print(f"🎨 Plotting connectivity quality with {len(data)} data points")
        
        connectivity_metrics = [
            ('ppi_degree', 'PPI Degree'),
            ('gri_total_degree', 'GRI Total Degree'),
            ('ppi_betweenness', 'PPI Betweenness'),
            ('gri_pagerank', 'GRI PageRank'),
            ('ppi_clustering', 'PPI Clustering'),
            ('gri_hub_score', 'GRI Hub Score')
        ]
        
        perf_cols = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['_r$', '^r$', '_corr', '_correlation', '_performance', '_mae', '_rmse', '_bias']) or \
               any(col_lower == pattern for pattern in ['r', 'correlation', 'performance', 'mae', 'rmse', 'bias']):
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Exclude network metrics that might contain 'r' 
                    if not col.startswith(('ppi_', 'gri_')):
                        perf_cols.append(col)
        
        # If no performance columns found, look for any numeric columns that might be metrics
        if not perf_cols:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            network_cols = [col for col in numeric_cols if col.startswith(('ppi_', 'gri_'))]
            perf_cols = [col for col in numeric_cols if col not in network_cols]
        
        print(f"🎨 Found performance columns: {perf_cols}")
        
        if not perf_cols:
            # No performance data available
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'No performance metrics available\n\nAvailable columns:\n' + '\n'.join(data.columns[:10]), 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(f'Connectivity vs Performance {i+1}')
            plt.tight_layout()
            self._save_figure(fig, "22_network_connectivity_vs_imputation_quality")
            plt.close()
            return
        
        primary_perf = perf_cols[0]
        
        plot_count = 0
        for conn_metric, conn_label in connectivity_metrics:
            if plot_count >= 6:
                break
                
            ax = axes[plot_count//3, plot_count%3]
            
            if conn_metric not in data.columns:
                ax.text(0.5, 0.5, f'{conn_label}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(conn_label)
                plot_count += 1
                continue
            
            conn_data = data[conn_metric]
            if conn_data.isna().all() or conn_data.sum() == 0:
                ax.text(0.5, 0.5, f'{conn_label}\nall values zero/NaN', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(conn_label)
                plot_count += 1
                continue
            
            clean_data = data.dropna(subset=[conn_metric, primary_perf])
            
            if len(clean_data) < 3:
                ax.text(0.5, 0.5, f'{conn_label}\ninsufficient data\n(n={len(clean_data)})', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(conn_label)
                plot_count += 1
                continue
            
            scatter = ax.scatter(clean_data[conn_metric], clean_data[primary_perf], 
                              alpha=0.6, s=20, color=self.colors['primary'], edgecolors='black', linewidth=0.3)
            
            if len(clean_data) > 5 and clean_data[conn_metric].std() > 0:
                try:
                    corr, p_val = stats.spearmanr(clean_data[conn_metric], clean_data[primary_perf])
                    
                    z = np.polyfit(clean_data[conn_metric], clean_data[primary_perf], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(clean_data[conn_metric].min(), 
                                        clean_data[conn_metric].max(), 100)
                    ax.plot(x_trend, p(x_trend), "--", color=self.colors['secondary'], alpha=0.8, linewidth=2)
                    
                    ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.2e}\nn = {len(clean_data)}',
                           transform=ax.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                except Exception as e:
                    print(f"⚠️  Error calculating correlation for {conn_metric}: {e}")
            
            ax.set_xlabel(conn_label)
            ax.set_ylabel(f'{primary_perf}')
            ax.set_title(f'{conn_label} vs Performance')
            ax.grid(True, alpha=0.3)
            
            if 'degree' in conn_metric or 'betweenness' in conn_metric:
                if clean_data[conn_metric].min() > 0:
                    ax.set_xscale('log')
            
            plot_count += 1
        
        for i in range(plot_count, 6):
            ax = axes[i//3, i%3]
            ax.text(0.5, 0.5, 'Additional\nanalysis\nspace', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, "22_network_connectivity_vs_imputation_quality")
        plt.close()
    
    def _plot_complex_consistency(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 23: Protein complex imputation consistency"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Protein Complex Imputation Consistency', 
                    fontsize=16, fontweight='bold')
        
        print(f"🎨 Plotting complex consistency with {len(data)} data points")
        
        # Try PPI communities first, then fall back to GRI-based grouping
        community_col = None
        community_source = None
        
        if 'ppi_community' in data.columns and not data['ppi_community'].isna().all():
            valid_ppi_communities = data[data['ppi_community'] != -1]
            if len(valid_ppi_communities) > 0:
                community_col = 'ppi_community'
                community_source = 'PPI'
        
        # Fall back to GRI-based functional grouping
        if community_col is None:
            print("⚠️  No valid PPI communities - creating GRI-based functional groups")
            if 'gri_total_degree' in data.columns:
                # Group by GRI degree quartiles as functional groups
                data = data.copy()
                gri_degrees = data['gri_total_degree']
                if gri_degrees.sum() > 0:  # Some GRI connectivity exists
                    data['gri_functional_group'] = pd.qcut(gri_degrees, 
                                                         q=4, 
                                                         labels=['Low_Reg', 'Med_Reg', 'High_Reg', 'Hub_Reg'],
                                                         duplicates='drop')
                    community_col = 'gri_functional_group'
                    community_source = 'GRI Functional'
        
        if community_col is None:
            for i, ax in enumerate(axes.flat):
                available_community_cols = [col for col in data.columns if 'community' in col.lower() or 'group' in col.lower()]
                ax.text(0.5, 0.5, f'No grouping data available\n\nPossible reasons:\n• Community detection failed\n• Networks too sparse\n• No features mapped to networks\n\nAvailable columns: {available_community_cols}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(['Complex Size vs Consistency', 'Performance vs Consistency', 
                             'Consistency Distribution', 'Size vs Variability'][i])
            plt.tight_layout()
            self._save_figure(fig, "23_protein_complex_imputation_consistency")
            plt.close()
            return
        
        # Find performance columns
        perf_cols = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['r', 'correlation', 'performance']):
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    perf_cols.append(col)
        
        if not perf_cols:
            # No performance data
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'No performance metrics available\nfor complex analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(['Complex Size vs Consistency', 'Performance vs Consistency', 
                             'Consistency Distribution', 'Size vs Variability'][i])
            plt.tight_layout()
            self._save_figure(fig, "23_protein_complex_imputation_consistency")
            plt.close()
            return
        
        if community_source == 'PPI':
            valid_communities = data[data[community_col] != -1]
        else:
            valid_communities = data[data[community_col].notna()]
        
        if len(valid_communities) == 0:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, f'No valid {community_source.lower()} groups found\n(all features unassigned)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(['Group Size vs Consistency', 'Performance vs Consistency', 
                             'Consistency Distribution', 'Size vs Variability'][i])
            plt.tight_layout()
            self._save_figure(fig, "23_protein_complex_imputation_consistency")
            plt.close()
            return
        
        communities = valid_communities.groupby(community_col)
        community_stats = []
        
        for comm_id, group in communities:
            if len(group) >= 2:  # Need at least 2 members for correlation
                community_perf = group[perf_cols].T  # Transpose so features are columns
                
                if len(community_perf.columns) >= 2:
                    corr_matrix = community_perf.corr()
                    # Get upper triangle (excluding diagonal)
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    avg_corr = upper_tri.stack().mean()
                    
                    community_stats.append({
                        'community_id': comm_id,
                        'size': len(group),
                        'avg_within_correlation': avg_corr if not np.isnan(avg_corr) else 0,
                        'avg_performance': group[perf_cols].mean().mean(),
                        'performance_std': group[perf_cols].mean(axis=1).std()
                    })
        
        if not community_stats:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, f'Insufficient data for {community_source.lower()} analysis\n(groups too small)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(['Group Size vs Consistency', 'Performance vs Consistency', 
                             'Consistency Distribution', 'Size vs Variability'][i])
            plt.tight_layout()
            self._save_figure(fig, "23_protein_complex_imputation_consistency")
            plt.close()
            return
        
        stats_df = pd.DataFrame(community_stats)
        
        fig.suptitle(f'{community_source} Group Imputation Consistency', 
                    fontsize=16, fontweight='bold')
        
        # Size vs consistency
        ax = axes[0, 0]
        if len(stats_df) > 0:
            scatter = ax.scatter(stats_df['size'], stats_df['avg_within_correlation'], 
                               s=stats_df['avg_performance']*100, alpha=0.7, c=self.colors['primary'])
            ax.set_xlabel('Group Size')
            ax.set_ylabel('Within-Group Correlation')
            ax.set_title(f'{community_source} Group Size vs Consistency')
            ax.grid(True, alpha=0.3)
            
            ax.text(0.05, 0.95, f'n = {len(stats_df)} {community_source.lower()} groups', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Performance vs consistency
        ax = axes[0, 1]
        if len(stats_df) > 0:
            ax.scatter(stats_df['avg_performance'], stats_df['avg_within_correlation'], 
                      alpha=0.7, c=self.colors['accent'], s=50)
            ax.set_xlabel('Average Performance')
            ax.set_ylabel('Within-Group Correlation')
            ax.set_title('Performance vs Consistency')
            ax.grid(True, alpha=0.3)
        
        # Consistency distribution
        ax = axes[1, 0]
        if len(stats_df) > 0:
            ax.hist(stats_df['avg_within_correlation'], bins=min(10, len(stats_df)), alpha=0.7, color=self.colors['alternative_3'])
            mean_corr = stats_df['avg_within_correlation'].mean()
            ax.axvline(mean_corr, color=self.colors['primary'], linestyle='--', linewidth=2, label=f'Mean = {mean_corr:.3f}')
            ax.set_xlabel('Within-Group Correlation')
            ax.set_ylabel(f'Number of {community_source} Groups')
            ax.set_title('Consistency Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Size vs variability
        ax = axes[1, 1]
        if len(stats_df) > 0:
            ax.scatter(stats_df['size'], stats_df['performance_std'], alpha=0.7, c=self.colors['alternative_5'], s=50)
            ax.set_xlabel('Group Size')
            ax.set_ylabel('Performance Std Dev')
            ax.set_title('Size vs Variability')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, "23_protein_complex_imputation_consistency")
        plt.close()
    
    def _plot_topology_difficulty(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 24: Network topology vs imputation difficulty (ENHANCED VERSION)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Network Topology vs Imputation Difficulty', 
                    fontsize=16, fontweight='bold')
        
        print(f"🎨 Plotting topology difficulty with {len(data)} data points")
        
        perf_cols = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['r', 'correlation', 'performance', 'mae', 'rmse', 'bias']):
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    perf_cols.append(col)
        
        if not perf_cols:
            # Look for any numeric columns that might be performance metrics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            network_cols = [col for col in numeric_cols if col.startswith(('ppi_', 'gri_'))]
            perf_cols = [col for col in numeric_cols if col not in network_cols]
        
        print(f"🎨 Found performance columns: {perf_cols}")
        
        if not perf_cols:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'No performance metrics available\nfor topology analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(f'Topology Analysis {i+1}')
            plt.tight_layout()
            self._save_figure(fig, "24_network_topology_imputation_difficulty")
            plt.close()
            return
        
        data = data.copy()
        data['avg_performance'] = data[perf_cols].mean(axis=1)
        data['difficulty'] = 1 - data['avg_performance']
        
        # Network metrics to analyze (prioritize GRI if PPI has no variance)
        network_metrics = [
            ('ppi_degree', 'PPI Degree'),
            ('gri_total_degree', 'GRI Total Degree'),  
            ('ppi_clustering', 'PPI Clustering'),
            ('gri_pagerank', 'GRI PageRank'),
            ('ppi_betweenness', 'PPI Betweenness'),
            ('gri_hub_score', 'GRI Hub Score')
        ]
        
        plot_count = 0
        for metric, label in network_metrics:
            if plot_count >= 6:
                break
                
            ax = axes[plot_count//3, plot_count%3]
            
            # Check if metric is available
            if metric not in data.columns:
                ax.text(0.5, 0.5, f'{label}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            # Check if we have meaningful variance
            if data[metric].std() == 0 or data[metric].isna().all():
                ax.text(0.5, 0.5, f'{label}\nno variance\n(all values same/NaN)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            clean_data = data.dropna(subset=[metric, 'difficulty'])
            
            if len(clean_data) < 3:
                ax.text(0.5, 0.5, f'{label}\ninsufficient data\n(n={len(clean_data)})', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            scatter = ax.scatter(clean_data[metric], clean_data['difficulty'], 
                              alpha=0.6, color=self.colors['primary'], s=20, edgecolors='black', linewidth=0.3)
            
            # Add trend line and correlation
            try:
                # Use Spearman correlation for robustness
                corr, p_val = stats.spearmanr(clean_data[metric], clean_data['difficulty'])
                
                # Add trend line
                z = np.polyfit(clean_data[metric], clean_data['difficulty'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_data[metric].min(), clean_data[metric].max(), 100)
                ax.plot(x_trend, p(x_trend), "--", color=self.colors['secondary'], alpha=0.8, linewidth=2)
                
                # Add correlation info
                ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.2e}\nn = {len(clean_data)}',
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"⚠️  Error calculating correlation for {metric}: {e}")
            
            ax.set_xlabel(label)
            ax.set_ylabel('Imputation Difficulty (1 - Performance)')
            ax.set_title(f'{label} vs Difficulty')
            ax.grid(True, alpha=0.3)
            
            if 'degree' in metric or 'betweenness' in metric:
                if clean_data[metric].min() > 0:
                    ax.set_xscale('log')
            
            plot_count += 1
        
        for i in range(plot_count, 6):
            ax = axes[i//3, i%3]
            ax.text(0.5, 0.5, 'Additional\nanalysis\nspace', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, "24_network_topology_imputation_difficulty")
        plt.close()
    
    def _plot_regulatory_transferability(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 25: Regulatory network impact on transferability (ENHANCED VERSION)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regulatory Network Impact on Cross-Platform Transferability',
                    fontsize=16, fontweight='bold')
        
        print(f"🎨 Plotting regulatory transferability with {len(data)} data points")
        
        if 'platform' in data.columns:
            # Data is in long format - we need to calculate cross-platform differences
            print("🎨 Data is in long format - calculating cross-platform metrics")
            
            # Find performance columns
            perf_cols = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['r', 'correlation', 'performance']):
                    if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        perf_cols.append(col)
            
            if not perf_cols:
                for i, ax in enumerate(axes.flat):
                    ax.text(0.5, 0.5, 'No performance metrics available\nfor transferability analysis', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    ax.set_title(f'Regulatory Analysis {i+1}')
                plt.tight_layout()
                self._save_figure(fig, "25_regulatory_network_transferability")
                plt.close()
                return
            
            primary_perf = perf_cols[0]
            
            data_pivot = data.pivot_table(index='feature', columns='platform', values=primary_perf, aggfunc='mean')
            platforms = data_pivot.columns
            
            if len(platforms) >= 2:
                data_cross = pd.DataFrame(index=data_pivot.index)
                data_cross['cross_platform_diff'] = abs(data_pivot.iloc[:, 0] - data_pivot.iloc[:, 1])
                data_cross['avg_platform_perf'] = (data_pivot.iloc[:, 0] + data_pivot.iloc[:, 1]) / 2
                
                # Merge with network metrics (take first occurrence per feature)
                network_cols = [col for col in data.columns if col.startswith('gri_')]
                if network_cols:
                    network_data = data.groupby('feature')[network_cols].first()
                    data_cross = data_cross.join(network_data, how='inner')
                else:
                    for i, ax in enumerate(axes.flat):
                        ax.text(0.5, 0.5, 'No GRI network metrics available\nfor regulatory analysis', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                        ax.set_title(f'Regulatory Analysis {i+1}')
                    plt.tight_layout()
                    self._save_figure(fig, "25_regulatory_network_transferability")
                    plt.close()
                    return
            else:
                for i, ax in enumerate(axes.flat):
                    ax.text(0.5, 0.5, f'Insufficient platforms for transferability\n(found {len(platforms)} platforms)', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                    ax.set_title(f'Regulatory Analysis {i+1}')
                plt.tight_layout()
                self._save_figure(fig, "25_regulatory_network_transferability")
                plt.close()
                return
        else:
            # Look for platform-specific columns in wide format
            platform_cols = [col for col in data.columns if 'platform' in col.lower()]
            
            if len(platform_cols) < 2:
                for i, ax in enumerate(axes.flat):
                    ax.text(0.5, 0.5, f'Insufficient platform columns\nfor transferability analysis\n\nFound: {platform_cols}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    ax.set_title(f'Regulatory Analysis {i+1}')
                plt.tight_layout()
                self._save_figure(fig, "25_regulatory_network_transferability")
                plt.close()
                return
            
            data_cross = data.copy()
            data_cross['cross_platform_diff'] = abs(data_cross[platform_cols[0]] - data_cross[platform_cols[1]])
            data_cross['avg_platform_perf'] = (data_cross[platform_cols[0]] + data_cross[platform_cols[1]]) / 2
        
        # GRI metrics to analyze
        gri_metrics = [
            ('gri_hub_score', 'GRI Hub Score'),
            ('gri_authority_score', 'GRI Authority Score'),
            ('gri_pagerank', 'GRI PageRank'),
            ('gri_total_degree', 'GRI Total Degree')
        ]
        
        plot_count = 0
        for metric, label in gri_metrics:
            if plot_count >= 4:
                break
                
            ax = axes[plot_count//2, plot_count%2]
            
            # Check if metric is available
            if metric not in data_cross.columns:
                ax.text(0.5, 0.5, f'{label}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            # Check if we have meaningful variance
            if data_cross[metric].std() == 0 or data_cross[metric].isna().all():
                ax.text(0.5, 0.5, f'{label}\nno variance\n(all values same/NaN)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            # Choose Y metric based on what we're analyzing
            y_metric = 'cross_platform_diff' if 'hub' in metric or 'authority' in metric else 'avg_platform_perf'
            y_label = 'Cross-Platform Difference' if y_metric == 'cross_platform_diff' else 'Average Performance'
            
            clean_data = data_cross.dropna(subset=[metric, y_metric])
            
            if len(clean_data) < 3:
                ax.text(0.5, 0.5, f'{label}\ninsufficient data\n(n={len(clean_data)})', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(label)
                plot_count += 1
                continue
            
            scatter = ax.scatter(clean_data[metric], clean_data[y_metric], alpha=0.6, 
                              color=self.colors['accent'], s=20, edgecolors='black', linewidth=0.3)
            
            # Add trend line and correlation
            try:
                # Use Spearman correlation for robustness
                corr, p_val = stats.spearmanr(clean_data[metric], clean_data[y_metric])
                
                # Add trend line
                z = np.polyfit(clean_data[metric], clean_data[y_metric], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_data[metric].min(), clean_data[metric].max(), 100)
                ax.plot(x_trend, p(x_trend), "--", color=self.colors['primary'], alpha=0.8, linewidth=2)
                
                # Add correlation info
                ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.2e}\nn = {len(clean_data)}',
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"⚠️  Error calculating correlation for {metric}: {e}")
            
            ax.set_xlabel(label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{label} vs {y_label}')
            ax.grid(True, alpha=0.3)
            
            plot_count += 1
        
        for i in range(plot_count, 4):
            ax = axes[i//2, i%2]
            ax.text(0.5, 0.5, 'Additional\nanalysis\nspace', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, "25_regulatory_network_transferability")
        plt.close()
    
    def _plot_hub_analysis(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 26: Hub protein performance analysis (ENHANCED VERSION)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hub Protein Imputation Performance Analysis',
                    fontsize=16, fontweight='bold')
        
        print(f"🎨 Plotting hub analysis with {len(data)} data points")
        
        if 'ppi_degree' not in data.columns:
            # No PPI data available
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'Hub analysis requires PPI network data\n\nNo PPI degree column found\n\nAvailable columns:\n' + 
                       '\n'.join([col for col in data.columns if col.startswith('ppi_')][:5]), 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(['Hub vs Non-Hub Performance', 'Connectivity vs Stability', 
                             'Local Connectivity vs Performance', 'Degree Distribution'][i])
            plt.tight_layout()
            self._save_figure(fig, "26_hub_protein_performance_analysis")
            plt.close()
            return
        
        if data['ppi_degree'].sum() == 0 or data['ppi_degree'].isna().all():
            print("⚠️  All PPI degrees are zero/NaN - falling back to GRI-based hub analysis")
            
            # Try using GRI total degree as an alternative
            if 'gri_total_degree' in data.columns and data['gri_total_degree'].sum() > 0:
                print("🔄 Using GRI total degree for hub analysis")
                
                threshold = data['gri_total_degree'].quantile(0.9)
                if threshold == 0:
                    threshold = data[data['gri_total_degree'] > 0]['gri_total_degree'].quantile(0.5) if (data['gri_total_degree'] > 0).any() else 1
                
                data = data.copy()
                data['is_hub'] = (data['gri_total_degree'] >= threshold) & (data['gri_total_degree'] > 0)
                
                # Find performance columns
                perf_cols = []
                for col in data.columns:
                    if any(keyword in col.lower() for keyword in ['r', 'correlation', 'performance', 'mae', 'rmse', 'bias']):
                        if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            perf_cols.append(col)
                
                if not perf_cols:
                    for i, ax in enumerate(axes.flat):
                        ax.text(0.5, 0.5, 'No performance metrics found\nfor GRI-based hub analysis', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                        ax.set_title(['GRI Hub vs Non-Hub Performance', 'GRI Connectivity vs Stability', 
                                     'GRI Hub Analysis', 'GRI Degree Distribution'][i])
                    plt.tight_layout()
                    self._save_figure(fig, "26_hub_protein_performance_analysis")
                    plt.close()
                    return
                
                data['avg_performance'] = data[perf_cols].mean(axis=1)
                data['performance_variability'] = data[perf_cols].std(axis=1)
                
                # Check if we identified any hubs
                n_hubs = data['is_hub'].sum()
                n_non_hubs = (~data['is_hub']).sum()
                print(f"🎨 GRI-based hub analysis: {n_hubs} hubs, {n_non_hubs} non-hubs (threshold: {threshold:.1f})")
                
                if n_hubs == 0:
                    for i, ax in enumerate(axes.flat):
                        ax.text(0.5, 0.5, f'No GRI hubs identified\n\nThreshold: {threshold:.1f}\nMax GRI degree: {data["gri_total_degree"].max():.1f}', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                        ax.set_title(['GRI Hub vs Non-Hub Performance', 'GRI Connectivity vs Stability', 
                                     'GRI Hub Analysis', 'GRI Degree Distribution'][i])
                    plt.tight_layout()
                    self._save_figure(fig, "26_hub_protein_performance_analysis")
                    plt.close()
                    return
                
                # 1. Hub vs non-hub comparison using GRI (top-left)
                ax = axes[0, 0]
                try:
                    hub_performances = data[data['is_hub']]['avg_performance'].dropna()
                    nonhub_performances = data[~data['is_hub']]['avg_performance'].dropna()
                    
                    if len(hub_performances) > 0 and len(nonhub_performances) > 0:
                        box_plot = ax.boxplot([nonhub_performances, hub_performances],
                                             patch_artist=True, labels=['Non-Hub', 'Hub'])
                        
                        colors = [self.colors['secondary'], self.colors['primary']]
                        for patch, color in zip(box_plot['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.set_title(f'GRI Hub vs Non-Hub Performance\n(n_hub={len(hub_performances)}, n_nonhub={len(nonhub_performances)})')
                        ax.set_ylabel('Average Performance')
                    else:
                        ax.text(0.5, 0.5, 'Insufficient data\nfor GRI hub comparison', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('GRI Hub vs Non-Hub Performance')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error in GRI hub comparison:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('GRI Hub vs Non-Hub Performance')
                ax.grid(True, alpha=0.3)
                
                # 2. GRI degree vs variability (top-right)
                ax = axes[0, 1]
                try:
                    hub_subset = data[data['is_hub']].dropna(subset=['gri_total_degree', 'performance_variability'])
                    if len(hub_subset) > 0:
                        scatter = ax.scatter(hub_subset['gri_total_degree'], hub_subset['performance_variability'], 
                                           alpha=0.6, color=self.colors['primary'], s=30, edgecolors='black', linewidth=0.3)
                        ax.set_xlabel('GRI Total Degree')
                        ax.set_ylabel('Performance Variability')
                        ax.set_title(f'GRI Hub Connectivity vs Stability (n={len(hub_subset)})')
                    else:
                        ax.text(0.5, 0.5, 'No GRI hub data\nfor analysis', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('GRI Hub Connectivity vs Stability')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error in GRI stability analysis:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('GRI Hub Connectivity vs Stability')
                ax.grid(True, alpha=0.3)
                
                # 3. GRI PageRank vs performance (bottom-left)
                ax = axes[1, 0]
                try:
                    if 'gri_pagerank' in data.columns:
                        hub_subset = data[data['is_hub']].dropna(subset=['gri_pagerank', 'avg_performance'])
                        if len(hub_subset) > 0:
                            scatter = ax.scatter(hub_subset['gri_pagerank'], hub_subset['avg_performance'], 
                                               alpha=0.6, color=self.colors['accent'], s=30, edgecolors='black', linewidth=0.3)
                            ax.set_xlabel('GRI PageRank')
                            ax.set_ylabel('Average Performance')
                            ax.set_title(f'GRI Hub PageRank vs Performance (n={len(hub_subset)})')
                        else:
                            ax.text(0.5, 0.5, 'No GRI PageRank data\nfor hubs', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title('GRI Hub PageRank vs Performance')
                    else:
                        ax.text(0.5, 0.5, 'GRI PageRank\ndata not available', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('GRI Hub PageRank vs Performance')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error in GRI PageRank analysis:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('GRI Hub PageRank vs Performance')
                ax.grid(True, alpha=0.3)
                
                # 4. GRI degree distribution (bottom-right)
                ax = axes[1, 1]
                try:
                    hub_degrees = data[data['is_hub']]['gri_total_degree']
                    nonhub_degrees = data[~data['is_hub']]['gri_total_degree']
                    
                    bins = 15
                    ax.hist(nonhub_degrees, bins=bins, alpha=0.7, 
                           label=f'Non-Hubs (n={len(nonhub_degrees)})', 
                           edgecolor='black', color=self.colors['secondary'], linewidth=0.5)
                    ax.hist(hub_degrees, bins=bins, alpha=0.7,
                           label=f'Hubs (n={len(hub_degrees)})', 
                           edgecolor='black', color=self.colors['primary'], linewidth=0.5)
                    
                    ax.set_xlabel('GRI Total Degree')
                    ax.set_ylabel('Count')
                    ax.set_title(f'GRI Degree Distribution (threshold: {threshold:.1f})')
                    ax.legend()
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error in GRI degree distribution:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('GRI Degree Distribution')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                self._save_figure(fig, "26_hub_protein_performance_analysis")
                plt.close()
                return
            
            # If neither PPI nor GRI work, show informative error
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'Hub analysis requires network connectivity data\n\nBoth PPI and GRI degrees are zero/NaN\n\nCheck network data processing', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(['Hub vs Non-Hub Performance', 'Connectivity vs Stability', 
                             'Local Connectivity vs Performance', 'Degree Distribution'][i])
            plt.tight_layout()
            self._save_figure(fig, "26_hub_protein_performance_analysis")
            plt.close()
            return
        
        threshold = data['ppi_degree'].quantile(0.9)
        
        # Ensure threshold is meaningful
        if threshold == 0:
            # If 90th percentile is still 0, use any non-zero degree as hub
            threshold = data[data['ppi_degree'] > 0]['ppi_degree'].quantile(0.5) if (data['ppi_degree'] > 0).any() else 1
        
        data = data.copy()
        data['is_hub'] = (data['ppi_degree'] >= threshold) & (data['ppi_degree'] > 0)
        
        # Check if we identified any hubs
        n_hubs = data['is_hub'].sum()
        n_non_hubs = (~data['is_hub']).sum()
        
        print(f"🎨 Hub analysis: {n_hubs} hubs, {n_non_hubs} non-hubs (threshold: {threshold:.1f})")
        
        if n_hubs == 0:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, f'No hub proteins identified\n\nThreshold: {threshold:.1f}\nMax degree: {data["ppi_degree"].max():.1f}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                ax.set_title(['Hub vs Non-Hub Performance', 'Connectivity vs Stability', 
                             'Local Connectivity vs Performance', 'Degree Distribution'][i])
            plt.tight_layout()
            self._save_figure(fig, "26_hub_protein_performance_analysis")
            plt.close()
            return
        
        perf_cols = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['r', 'correlation', 'performance', 'mae', 'rmse', 'bias']):
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    perf_cols.append(col)
        
        if not perf_cols:
            # Look for any numeric columns that might be performance metrics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            network_cols = [col for col in numeric_cols if col.startswith(('ppi_', 'gri_'))]
            perf_cols = [col for col in numeric_cols if col not in network_cols]
        
        print(f"🎨 Found performance columns for hub analysis: {perf_cols}")
        
        if not perf_cols:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, f'No performance metrics found\n\nHubs identified: {n_hubs}\nBut no performance data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(['Hub vs Non-Hub Performance', 'Connectivity vs Stability', 
                             'Local Connectivity vs Performance', 'Degree Distribution'][i])
            plt.tight_layout()
            self._save_figure(fig, "26_hub_protein_performance_analysis")
            plt.close()
            return
        
        data['avg_performance'] = data[perf_cols].mean(axis=1)
        data['performance_variability'] = data[perf_cols].std(axis=1)
        
        # 1. Hub vs non-hub comparison (top-left)
        ax = axes[0, 0]
        try:
            # Prepare data for boxplot
            hub_performances = data[data['is_hub']]['avg_performance'].dropna()
            nonhub_performances = data[~data['is_hub']]['avg_performance'].dropna()
            
            if len(hub_performances) > 0 and len(nonhub_performances) > 0:
                box_plot = ax.boxplot([nonhub_performances, hub_performances],
                                     patch_artist=True, labels=['Non-Hub', 'Hub'])
                
                colors = [self.colors['secondary'], self.colors['primary']]
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                from scipy.stats import mannwhitneyu
                try:
                    stat, p_val = mannwhitneyu(hub_performances, nonhub_performances, alternative='two-sided')
                    ax.text(0.5, 0.95, f'Mann-Whitney U\np = {p_val:.2e}', 
                           transform=ax.transAxes, ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                except:
                    pass
                
                ax.set_title(f'Hub vs Non-Hub Performance\n(n_hub={len(hub_performances)}, n_nonhub={len(nonhub_performances)})')
                ax.set_ylabel('Average Performance')
            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Hub vs Non-Hub Performance')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in hub comparison:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hub vs Non-Hub Performance')
        ax.grid(True, alpha=0.3)
        
        # 2. Degree vs variability (top-right)
        ax = axes[0, 1]
        try:
            hub_subset = data[data['is_hub']].dropna(subset=['ppi_degree', 'performance_variability'])
            if len(hub_subset) > 0:
                scatter = ax.scatter(hub_subset['ppi_degree'], hub_subset['performance_variability'], 
                                   alpha=0.6, color=self.colors['primary'], s=30, edgecolors='black', linewidth=0.3)
                ax.set_xlabel('PPI Degree')
                ax.set_ylabel('Performance Variability')
                ax.set_title(f'Hub Connectivity vs Stability (n={len(hub_subset)})')
                
                # Add correlation if enough data
                if len(hub_subset) > 3:
                    try:
                        corr, p_val = stats.spearmanr(hub_subset['ppi_degree'], hub_subset['performance_variability'])
                        ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.2e}', 
                               transform=ax.transAxes, fontsize=9,
                               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                    except:
                        pass
            else:
                ax.text(0.5, 0.5, 'No hub data\nfor analysis', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Hub Connectivity vs Stability')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in stability analysis:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hub Connectivity vs Stability')
        ax.grid(True, alpha=0.3)
        
        # 3. Clustering vs performance (bottom-left)
        ax = axes[1, 0]
        try:
            if 'ppi_clustering' in data.columns:
                hub_subset = data[data['is_hub']].dropna(subset=['ppi_clustering', 'avg_performance'])
                if len(hub_subset) > 0:
                    scatter = ax.scatter(hub_subset['ppi_clustering'], hub_subset['avg_performance'], 
                                       alpha=0.6, color=self.colors['accent'], s=30, edgecolors='black', linewidth=0.3)
                    ax.set_xlabel('Clustering Coefficient')
                    ax.set_ylabel('Average Performance')
                    ax.set_title(f'Hub Local Connectivity vs Performance (n={len(hub_subset)})')
                    
                    # Add correlation if enough data
                    if len(hub_subset) > 3:
                        try:
                            corr, p_val = stats.spearmanr(hub_subset['ppi_clustering'], hub_subset['avg_performance'])
                            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.2e}', 
                                   transform=ax.transAxes, fontsize=9,
                                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                        except:
                            pass
                else:
                    ax.text(0.5, 0.5, 'No clustering data\nfor hubs', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Hub Local Connectivity vs Performance')
            else:
                ax.text(0.5, 0.5, 'PPI clustering\ndata not available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Hub Local Connectivity vs Performance')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in clustering analysis:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hub Local Connectivity vs Performance')
        ax.grid(True, alpha=0.3)
        
        # 4. Degree distribution (bottom-right)
        ax = axes[1, 1]
        try:
            hub_degrees = data[data['is_hub']]['ppi_degree']
            nonhub_degrees = data[~data['is_hub']]['ppi_degree']
            
            max_degree = data['ppi_degree'].max()
            if max_degree > 100:
                bins = np.logspace(0, np.log10(max_degree), 20)
                ax.set_xscale('log')
            else:
                bins = 15
            
            ax.hist(nonhub_degrees, bins=bins, alpha=0.7, 
                   label=f'Non-Hubs (n={len(nonhub_degrees)})', 
                   edgecolor='black', color=self.colors['secondary'], linewidth=0.5)
            ax.hist(hub_degrees, bins=bins, alpha=0.7,
                   label=f'Hubs (n={len(hub_degrees)})', 
                   edgecolor='black', color=self.colors['primary'], linewidth=0.5)
            
            ax.set_xlabel('PPI Degree')
            ax.set_ylabel('Count')
            ax.set_title(f'Degree Distribution (threshold: {threshold:.1f})')
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in degree distribution:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Degree Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, "26_hub_protein_performance_analysis")
        plt.close()
    
    def _plot_network_clustering(self, data: pd.DataFrame, network_data: Dict[str, Any]):
        """Figure 27: Network-based feature clustering"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network-Based Feature Clustering and Performance Patterns',
                    fontsize=16, fontweight='bold')
        
        # Network feature columns
        network_cols = [col for col in data.columns if 
                       col.startswith(('ppi_', 'gri_')) and 
                       col not in ['ppi_community', 'ppi_community_size']]
        
        if len(network_cols) >= 2 and len(data) >= 10:
            # Prepare for clustering
            features = data[network_cols].fillna(0)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Clustering
            n_clusters = min(5, max(2, len(data) // 20))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data = data.copy()
            data['network_cluster'] = kmeans.fit_predict(features_scaled)
            
            # PCA visualization
            ax = axes[0, 0]
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
            
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                               c=data['network_cluster'], cmap='tab10', alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Network Clusters (PCA)')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            # Performance by cluster
            ax = axes[0, 1]
            perf_cols = [col for col in data.columns if any(x in col.lower() 
                        for x in ['r_', 'correlation', 'performance'])]
            
            if perf_cols:
                data['avg_performance'] = data[perf_cols].mean(axis=1)
                
                cluster_perf = []
                for cluster_id in sorted(data['network_cluster'].unique()):
                    cluster_data = data[data['network_cluster'] == cluster_id]
                    for perf in cluster_data['avg_performance']:
                        cluster_perf.append({
                            'Cluster': f'C{cluster_id}',
                            'Performance': perf
                        })
                
                if cluster_perf:
                    cluster_df = pd.DataFrame(cluster_perf)
                    clusters = sorted(cluster_df['Cluster'].unique())
                    data_by_cluster = [cluster_df[cluster_df['Cluster'] == c]['Performance'].values 
                                     for c in clusters]
                    box_plot = ax.boxplot(data_by_cluster, patch_artist=True, labels=clusters)
                    
                    # Cycle through colors for different clusters
                    color_cycle = [self.colors['primary'], self.colors['secondary'], 
                                  self.colors['accent'], self.colors['alternative_1'], 
                                  self.colors['alternative_3']]
                    for i, patch in enumerate(box_plot['boxes']):
                        patch.set_facecolor(color_cycle[i % len(color_cycle)])
                        patch.set_alpha(0.7)
                    ax.set_title('Performance by Cluster')
                    ax.grid(True, alpha=0.3)
            
            # Cluster characteristics
            ax = axes[1, 0]
            cluster_means = data.groupby('network_cluster')[network_cols].mean()
            if not cluster_means.empty:
                sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='viridis', ax=ax)
                ax.set_title('Cluster Characteristics')
                ax.set_xlabel('Cluster')
            
            # Cluster sizes
            ax = axes[1, 1]
            cluster_sizes = data['network_cluster'].value_counts().sort_index()
            ax.bar(range(len(cluster_sizes)), cluster_sizes.values)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Features')
            ax.set_title('Cluster Sizes')
            ax.set_xticks(range(len(cluster_sizes)))
            ax.set_xticklabels([f'C{i}' for i in cluster_sizes.index])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, "27_network_based_feature_clustering")
        plt.close()
    
    def _save_figure(self, fig, name: str):
        """Save figure in publication formats"""
        # PDF (vector)
        pdf_path = self.figures_dir / f"{name}.pdf"
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        
        # PNG (raster)
        png_path = self.figures_dir / f"{name}.png"
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    def generate_network_report(self, network_data: Dict[str, Any]) -> str:
        """Generate network analysis summary report"""
        if not network_data:
            return "No network data available.\n"
        
        report = "🔗 Network Analysis Summary Report\n"
        report += "=" * 50 + "\n\n"
        
        summary = network_data.get('summary', {})
        
        # PPI Summary
        if 'ppi' in summary:
            ppi = summary['ppi']
            report += "📊 Protein-Protein Interaction (PPI) Network:\n"
            report += f"  • Nodes: {ppi['nodes']:,}\n"
            report += f"  • Edges: {ppi['edges']:,}\n"
            report += f"  • Density: {ppi['density']:.4f}\n"
            report += f"  • Average Clustering: {ppi['avg_clustering']:.4f}\n\n"
        
        # GRI Summary
        if 'gri' in summary:
            gri = summary['gri']
            report += "📊 Gene Regulatory Interaction (GRI) Network:\n"
            report += f"  • Nodes: {gri['nodes']:,}\n"
            report += f"  • Edges: {gri['edges']:,}\n"
            report += f"  • Density: {gri['density']:.4f}\n\n"
        
        # Mapping Summary
        if 'mapping' in summary:
            mapping = summary['mapping']
            report += "🔗 Feature-Network Mapping:\n"
            report += f"  • Mapped Features: {mapping['mapped_features']}\n"
            report += f"  • Total Metrics: {mapping['total_metrics']}\n\n"
        
        # Research Questions
        report += "🔬 Research Questions Addressed:\n"
        questions = [
            "1. Do highly connected proteins have better imputation quality?",
            "2. Are protein complexes imputed more consistently?", 
            "3. Can network topology predict imputation difficulty?",
            "4. Do regulatory relationships affect cross-platform transferability?",
            "5. How do hub proteins perform in imputation tasks?",
            "6. Can network-based clustering reveal performance patterns?"
        ]
        
        for q in questions:
            report += f"  {q}\n"
        
        report += "\n"
        return report 