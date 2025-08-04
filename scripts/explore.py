#!/usr/bin/env python3
"""
Data exploration script for Joint VAE datasets.

This script performs comprehensive exploratory data analysis on paired CSV files,
including distribution analysis, normalization comparison, correlation analysis,
and various visualizations.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent / 'src'))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Explore Joint VAE datasets')
    
    parser.add_argument(
        '--platform_a', 
        type=str, 
        required=True,
        help='Path to platform A CSV file'
    )
    parser.add_argument(
        '--platform_b', 
        type=str, 
        required=True,
        help='Path to platform B CSV file'
    )

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs_exploration',
        help='Output directory for plots and analysis'
    )
    parser.add_argument(
        '--id_column',
        type=str,
        default=None,
        help='Name of the ID column (defaults to first column)'
    )
    
    return parser.parse_args()


class DataExplorer:
    """Comprehensive data exploration class."""
    
    def __init__(self, output_dir: str):
        """Initialize the explorer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        sns.set_palette("husl")
        
    def load_and_analyze_data(self, file_a: str, file_b: str, id_column: str = None):
        """Load data and perform basic analysis."""
        print("=" * 60)
        print("LOADING AND BASIC ANALYSIS")
        print("=" * 60)
        
        # Load CSV files
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
        
        if id_column is None:
            id_column = df_a.columns[0]
        
        print(f"Platform A Dataset:")
        print(f"  Shape: {df_a.shape}")
        print(f"  ID column: {id_column}")
        print(f"  Feature columns: {df_a.shape[1] - 1}")
        
        print(f"\nPlatform B Dataset:")
        print(f"  Shape: {df_b.shape}")
        print(f"  ID column: {id_column}")
        print(f"  Feature columns: {df_b.shape[1] - 1}")
        
        # Check for shared samples
        shared_ids = set(df_a[id_column]) & set(df_b[id_column])
        print(f"\nShared samples: {len(shared_ids)}")
        print(f"Platform A unique samples: {len(set(df_a[id_column]) - shared_ids)}")
        print(f"Platform B unique samples: {len(set(df_b[id_column]) - shared_ids)}")
        
        return df_a, df_b, id_column, shared_ids
    
    def analyze_missing_values(self, df_a: pd.DataFrame, df_b: pd.DataFrame, id_column: str):
        """Analyze missing values in both datasets."""
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)
        
        # Get feature columns
        features_a = [col for col in df_a.columns if col != id_column]
        features_b = [col for col in df_b.columns if col != id_column]
        
        # Missing value statistics
        missing_a = df_a[features_a].isnull().sum()
        missing_b = df_b[features_b].isnull().sum()
        
        print(f"Platform A missing values:")
        print(f"  Total missing: {missing_a.sum()}")
        print(f"  Features with missing values: {(missing_a > 0).sum()}")
        print(f"  Max missing per feature: {missing_a.max()}")
        
        print(f"\nPlatform B missing values:")
        print(f"  Total missing: {missing_b.sum()}")
        print(f"  Features with missing values: {(missing_b > 0).sum()}")
        print(f"  Max missing per feature: {missing_b.max()}")
        
        # Plot missing value heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Platform A
        missing_matrix_a = df_a[features_a].isnull()
        if missing_matrix_a.any().any():
            sns.heatmap(missing_matrix_a.T, cbar=True, ax=axes[0], cmap='viridis')
            axes[0].set_title('Platform A Missing Values')
            axes[0].set_xlabel('Samples')
            axes[0].set_ylabel('Features')
        else:
            axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=axes[0].transAxes, fontsize=16)
            axes[0].set_title('Platform A Missing Values')
        
        # Platform B
        missing_matrix_b = df_b[features_b].isnull()
        if missing_matrix_b.any().any():
            sns.heatmap(missing_matrix_b.T, cbar=True, ax=axes[1], cmap='viridis')
            axes[1].set_title('Platform B Missing Values')
            axes[1].set_xlabel('Samples')
            axes[1].set_ylabel('Features')
        else:
            axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=16)
            axes[1].set_title('Platform B Missing Values')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return missing_a, missing_b
    
    def analyze_distributions(self, data_a: pd.DataFrame, data_b: pd.DataFrame):
        """Analyze feature distributions."""
        print("\n" + "=" * 60)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("Platform A Statistics:")
        print(data_a.describe())
        print("\nPlatform B Statistics:")
        print(data_b.describe())
        
        # Plot distributions for first few features
        n_features_to_plot = min(6, data_a.shape[1], data_b.shape[1])
        
        fig, axes = plt.subplots(2, n_features_to_plot, figsize=(4*n_features_to_plot, 8))
        if n_features_to_plot == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_features_to_plot):
            # Platform A
            feature_a = data_a.iloc[:, i]
            axes[0, i].hist(feature_a.dropna(), bins=30, alpha=0.7, density=True)
            axes[0, i].set_title(f'Platform A: {data_a.columns[i]}')
            axes[0, i].set_ylabel('Density')
            
            # Platform B
            feature_b = data_b.iloc[:, i]
            axes[1, i].hist(feature_b.dropna(), bins=30, alpha=0.7, density=True)
            axes[1, i].set_title(f'Platform B: {data_b.columns[i]}')
            axes[1, i].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze skewness and kurtosis
        skewness_a = data_a.skew()
        kurtosis_a = data_a.kurtosis()
        skewness_b = data_b.skew()
        kurtosis_b = data_b.kurtosis()
        
        print(f"\nDistribution Properties:")
        print(f"Platform A - Median skewness: {skewness_a.median():.3f}")
        print(f"Platform A - Median kurtosis: {kurtosis_a.median():.3f}")
        print(f"Platform B - Median skewness: {skewness_b.median():.3f}")
        print(f"Platform B - Median kurtosis: {kurtosis_b.median():.3f}")
        
        return skewness_a, kurtosis_a, skewness_b, kurtosis_b
    
    def compare_normalization_methods(self, data_a: pd.DataFrame, data_b: pd.DataFrame):
        """Compare different normalization methods."""
        print("\n" + "=" * 60)
        print("NORMALIZATION COMPARISON")
        print("=" * 60)
        
        # Define normalization methods
        scalers = {
            'StandardScaler (Z-score)': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'PowerTransformer (Yeo-Johnson)': PowerTransformer(method='yeo-johnson', standardize=True),
            'PowerTransformer (Box-Cox)': PowerTransformer(method='box-cox', standardize=True)
        }
        
        # Add log scaling (handling negative values)
        def log_transform(data):
            # Shift data to positive values if needed
            min_val = data.min().min()
            if min_val <= 0:
                shifted_data = data - min_val + 1
                print(f"  Log scaling: shifted data by {-min_val + 1:.3f} to handle negative values")
            else:
                shifted_data = data
            
            # Apply log transformation
            log_data = np.log(shifted_data)
            
            # Standardize the log-transformed data
            scaler = StandardScaler()
            return pd.DataFrame(
                scaler.fit_transform(log_data),
                columns=data.columns
            )
        
        # Create log-transformed versions
        print("\nApplying log transformations...")
        try:
            data_a_log = log_transform(data_a)
            data_b_log = log_transform(data_b)
            log_success = True
        except Exception as e:
            print(f"  Warning: Log transformation failed: {e}")
            log_success = False
        
        # Sample a few features for visualization
        sample_features_a = min(3, data_a.shape[1])
        sample_features_b = min(3, data_b.shape[1])
        
        # Pre-check which methods will actually be used
        usable_methods = []
        for method_name, scaler in scalers.items():
            if 'Box-Cox' in method_name:
                if data_a.min().min() > 0 and data_b.min().min() > 0:
                    usable_methods.append((method_name, scaler))
            else:
                usable_methods.append((method_name, scaler))
        
        # Determine number of rows for plotting
        n_methods = len(usable_methods) + (1 if log_success else 0)
        if n_methods == 0:
            print("  Warning: No normalization methods can be applied to this data")
            return
            
        fig, axes = plt.subplots(n_methods, 2, figsize=(12, 4*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        plot_row = 0
        
        for i, (method_name, scaler) in enumerate(usable_methods):
            try:
                # Normalize data (pre-check already done for Box-Cox)
                
                data_a_norm = pd.DataFrame(
                    scaler.fit_transform(data_a),
                    columns=data_a.columns
                )
                data_b_norm = pd.DataFrame(
                    scaler.fit_transform(data_b),
                    columns=data_b.columns
                )
                
                # Plot normalized distributions
                for j in range(sample_features_a):
                    axes[plot_row, 0].hist(data_a_norm.iloc[:, j], bins=30, alpha=0.5, 
                                         label=f'Feature {j+1}', density=True)
                axes[plot_row, 0].set_title(f'Platform A - {method_name}')
                axes[plot_row, 0].set_ylabel('Density')
                axes[plot_row, 0].legend()
                
                for j in range(sample_features_b):
                    axes[plot_row, 1].hist(data_b_norm.iloc[:, j], bins=30, alpha=0.5, 
                                         label=f'Feature {j+1}', density=True)
                axes[plot_row, 1].set_title(f'Platform B - {method_name}')
                axes[plot_row, 1].set_ylabel('Density')
                axes[plot_row, 1].legend()
                
                # Print statistics
                print(f"\n{method_name}:")
                print(f"  Platform A - Mean: {data_a_norm.mean().mean():.3f}, Std: {data_a_norm.std().mean():.3f}")
                print(f"  Platform B - Mean: {data_b_norm.mean().mean():.3f}, Std: {data_b_norm.std().mean():.3f}")
                
                # Calculate and print skewness reduction
                orig_skew_a = data_a.skew().abs().mean()
                orig_skew_b = data_b.skew().abs().mean()
                new_skew_a = data_a_norm.skew().abs().mean()
                new_skew_b = data_b_norm.skew().abs().mean()
                
                print(f"  Skewness reduction:")
                if orig_skew_a > 0:
                    skew_change_a = ((orig_skew_a-new_skew_a)/orig_skew_a*100)
                    print(f"    Platform A: {orig_skew_a:.3f} → {new_skew_a:.3f} ({skew_change_a:+.1f}%)")
                else:
                    print(f"    Platform A: {orig_skew_a:.3f} → {new_skew_a:.3f} (N/A - original skewness was 0)")
                
                if orig_skew_b > 0:
                    skew_change_b = ((orig_skew_b-new_skew_b)/orig_skew_b*100)
                    print(f"    Platform B: {orig_skew_b:.3f} → {new_skew_b:.3f} ({skew_change_b:+.1f}%)")
                else:
                    print(f"    Platform B: {orig_skew_b:.3f} → {new_skew_b:.3f} (N/A - original skewness was 0)")
                
                plot_row += 1
                
            except Exception as e:
                print(f"  Warning: {method_name} failed: {e}")
                continue
        
        # Add log scaling plots if successful
        if log_success:
            # Plot log-transformed distributions
            for j in range(sample_features_a):
                axes[plot_row, 0].hist(data_a_log.iloc[:, j], bins=30, alpha=0.5, 
                                     label=f'Feature {j+1}', density=True)
            axes[plot_row, 0].set_title('Platform A - Log Scaling')
            axes[plot_row, 0].set_ylabel('Density')
            axes[plot_row, 0].legend()
            
            for j in range(sample_features_b):
                axes[plot_row, 1].hist(data_b_log.iloc[:, j], bins=30, alpha=0.5, 
                                     label=f'Feature {j+1}', density=True)
            axes[plot_row, 1].set_title('Platform B - Log Scaling')
            axes[plot_row, 1].set_ylabel('Density')
            axes[plot_row, 1].legend()
            
            # Print log scaling statistics
            print(f"\nLog Scaling:")
            print(f"  Platform A - Mean: {data_a_log.mean().mean():.3f}, Std: {data_a_log.std().mean():.3f}")
            print(f"  Platform B - Mean: {data_b_log.mean().mean():.3f}, Std: {data_b_log.std().mean():.3f}")
            
            # Analyze skewness reduction
            orig_skew_a = data_a.skew().abs().mean()
            orig_skew_b = data_b.skew().abs().mean()
            log_skew_a = data_a_log.skew().abs().mean()
            log_skew_b = data_b_log.skew().abs().mean()
            
            print(f"  Skewness reduction:")
            if orig_skew_a > 0:
                skew_change_a = ((orig_skew_a-log_skew_a)/orig_skew_a*100)
                print(f"    Platform A: {orig_skew_a:.3f} → {log_skew_a:.3f} ({skew_change_a:+.1f}%)")
            else:
                print(f"    Platform A: {orig_skew_a:.3f} → {log_skew_a:.3f} (N/A - original skewness was 0)")
            
            if orig_skew_b > 0:
                skew_change_b = ((orig_skew_b-log_skew_b)/orig_skew_b*100)
                print(f"    Platform B: {orig_skew_b:.3f} → {log_skew_b:.3f} ({skew_change_b:+.1f}%)")
            else:
                print(f"    Platform B: {orig_skew_b:.3f} → {log_skew_b:.3f} (N/A - original skewness was 0)")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'normalization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Recommend best normalization method
        self._recommend_normalization_method(data_a, data_b, scalers, log_success, 
                                           data_a_log if log_success else None, 
                                           data_b_log if log_success else None)
    
    def _recommend_normalization_method(self, data_a, data_b, scalers, log_success, data_a_log, data_b_log):
        """Analyze and recommend the best normalization method."""
        print("\n" + "=" * 40)
        print("NORMALIZATION RECOMMENDATION")
        print("=" * 40)
        
        methods_results = {}
        
        # Evaluate standard scaling methods
        for method_name, scaler in scalers.items():
            try:
                # Handle Box-Cox which requires positive values
                if 'Box-Cox' in method_name:
                    if data_a.min().min() <= 0 or data_b.min().min() <= 0:
                        print(f"  Skipping {method_name} in evaluation: requires positive values")
                        continue
                
                data_a_norm = pd.DataFrame(scaler.fit_transform(data_a), columns=data_a.columns)
                data_b_norm = pd.DataFrame(scaler.fit_transform(data_b), columns=data_b.columns)
                
                # Calculate metrics
                skew_a = data_a_norm.skew().abs().mean()
                skew_b = data_b_norm.skew().abs().mean()
                kurt_a = data_a_norm.kurtosis().abs().mean()
                kurt_b = data_b_norm.kurtosis().abs().mean()
                
                methods_results[method_name] = {
                    'avg_skew': (skew_a + skew_b) / 2,
                    'avg_kurtosis': (kurt_a + kurt_b) / 2,
                    'score': 1 / (1 + (skew_a + skew_b) / 2 + (kurt_a + kurt_b) / 10)  # Combined score
                }
            except Exception as e:
                print(f"  Warning: {method_name} evaluation failed: {e}")
                continue
        
        # Evaluate log scaling if successful
        if log_success and data_a_log is not None and data_b_log is not None:
            skew_a = data_a_log.skew().abs().mean()
            skew_b = data_b_log.skew().abs().mean()
            kurt_a = data_a_log.kurtosis().abs().mean()
            kurt_b = data_b_log.kurtosis().abs().mean()
            
            methods_results['Log Scaling'] = {
                'avg_skew': (skew_a + skew_b) / 2,
                'avg_kurtosis': (kurt_a + kurt_b) / 2,
                'score': 1 / (1 + (skew_a + skew_b) / 2 + (kurt_a + kurt_b) / 10)
            }
        
        # Find best method
        best_method = max(methods_results.keys(), key=lambda k: methods_results[k]['score'])
        
        print("Normalization method comparison (lower skewness/kurtosis is better):")
        for method, results in sorted(methods_results.items(), key=lambda item: item[1]['score'], reverse=True):
            print(f"  {method}:")
            print(f"    Average Skewness: {results['avg_skew']:.3f}")
            print(f"    Average Kurtosis: {results['avg_kurtosis']:.3f}")
            print(f"    Normality Score: {results['score']:.3f}")
        
        print(f"\n🏆 RECOMMENDED: {best_method}")
        
        # Additional recommendations
        orig_skew_a = data_a.skew().abs().mean()
        orig_skew_b = data_b.skew().abs().mean()
        
        if orig_skew_a > 2 or orig_skew_b > 2:
            print("📊 Data appears highly skewed - PowerTransformer or Log scaling may be particularly beneficial")
        if log_success and 'Log Scaling' in methods_results:
            if methods_results['Log Scaling']['score'] > methods_results['StandardScaler (Z-score)']['score']:
                print("📈 Log scaling shows improvement over standard normalization")
        
        # Check PowerTransformer performance
        power_methods = [method for method in methods_results.keys() if 'PowerTransformer' in method]
        if power_methods:
            best_power = max(power_methods, key=lambda k: methods_results[k]['score'])
            if methods_results[best_power]['score'] > methods_results.get('StandardScaler (Z-score)', {}).get('score', 0):
                print(f"🚀 {best_power} shows significant improvement for skewed data")
        
        return best_method
    
    def analyze_correlations(self, data_a: pd.DataFrame, data_b: pd.DataFrame):
        """Analyze feature correlations within and between platforms."""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Check if we have enough features for correlation analysis
        if data_a.shape[1] < 2 or data_b.shape[1] < 2:
            print("Insufficient features for correlation analysis (need at least 2 features per platform)")
            return None, None
        
        # Compute correlation matrices
        corr_a = data_a.corr()
        corr_b = data_b.corr()
        
        print(f"Platform A correlation statistics:")
        print(f"  Mean correlation: {corr_a.values[np.triu_indices_from(corr_a.values, k=1)].mean():.3f}")
        print(f"  Max correlation: {corr_a.values[np.triu_indices_from(corr_a.values, k=1)].max():.3f}")
        
        print(f"Platform B correlation statistics:")
        print(f"  Mean correlation: {corr_b.values[np.triu_indices_from(corr_b.values, k=1)].mean():.3f}")
        print(f"  Max correlation: {corr_b.values[np.triu_indices_from(corr_b.values, k=1)].max():.3f}")
        
        # Plot correlation heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Limit to first 50 features for readability
        max_features = 50
        corr_a_plot = corr_a.iloc[:max_features, :max_features]
        corr_b_plot = corr_b.iloc[:max_features, :max_features]
        
        sns.heatmap(corr_a_plot, ax=axes[0], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[0].set_title('Platform A Correlation Matrix')
        
        sns.heatmap(corr_b_plot, ax=axes[1], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[1].set_title('Platform B Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_a, corr_b
    
    def perform_dimensionality_reduction(self, data_a: pd.DataFrame, data_b: pd.DataFrame):
        """Perform PCA and t-SNE analysis."""
        print("\n" + "=" * 60)
        print("DIMENSIONALITY REDUCTION ANALYSIS")
        print("=" * 60)
        
        # Check if we have enough features and samples for dimensionality reduction
        if data_a.shape[1] < 2 or data_b.shape[1] < 2:
            print("Insufficient features for dimensionality reduction (need at least 2 features per platform)")
            return None, None
        
        if data_a.shape[0] < 2 or data_b.shape[0] < 2:
            print("Insufficient samples for dimensionality reduction (need at least 2 samples per platform)")
            return None, None
        
        # Standardize data for PCA/t-SNE
        scaler_a = StandardScaler()
        scaler_b = StandardScaler()
        
        data_a_scaled = scaler_a.fit_transform(data_a.fillna(data_a.mean()))
        data_b_scaled = scaler_b.fit_transform(data_b.fillna(data_b.mean()))
        
        # PCA Analysis
        pca_a = PCA()
        pca_b = PCA()
        
        pca_a_transformed = pca_a.fit_transform(data_a_scaled)
        pca_b_transformed = pca_b.fit_transform(data_b_scaled)
        
        # Plot PCA results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance
        axes[0, 0].plot(np.cumsum(pca_a.explained_variance_ratio_)[:50], 'o-')
        axes[0, 0].set_title('Platform A - PCA Explained Variance')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(np.cumsum(pca_b.explained_variance_ratio_)[:50], 'o-')
        axes[0, 1].set_title('Platform B - PCA Explained Variance')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].grid(True)
        
        # 2D PCA plots
        axes[1, 0].scatter(pca_a_transformed[:, 0], pca_a_transformed[:, 1], alpha=0.6)
        axes[1, 0].set_title('Platform A - First 2 Principal Components')
        axes[1, 0].set_xlabel(f'PC1 ({pca_a.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca_a.explained_variance_ratio_[1]:.2%} variance)')
        
        axes[1, 1].scatter(pca_b_transformed[:, 0], pca_b_transformed[:, 1], alpha=0.6)
        axes[1, 1].set_title('Platform B - First 2 Principal Components')
        axes[1, 1].set_xlabel(f'PC1 ({pca_b.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca_b.explained_variance_ratio_[1]:.2%} variance)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print PCA statistics
        print(f"Platform A PCA:")
        print(f"  First 10 components explain {np.sum(pca_a.explained_variance_ratio_[:10]):.2%} of variance")
        print(f"  50% variance explained by {np.argmax(np.cumsum(pca_a.explained_variance_ratio_) >= 0.5) + 1} components")
        
        print(f"Platform B PCA:")
        print(f"  First 10 components explain {np.sum(pca_b.explained_variance_ratio_[:10]):.2%} of variance")
        print(f"  50% variance explained by {np.argmax(np.cumsum(pca_b.explained_variance_ratio_) >= 0.5) + 1} components")
        
        # t-SNE (on a subset of data if too large)
        max_samples_tsne = 1000
        if len(data_a_scaled) > max_samples_tsne:
            indices = np.random.choice(len(data_a_scaled), max_samples_tsne, replace=False)
            data_a_tsne = data_a_scaled[indices]
            data_b_tsne = data_b_scaled[indices]
        else:
            data_a_tsne = data_a_scaled
            data_b_tsne = data_b_scaled
        
        print(f"\nRunning t-SNE on {len(data_a_tsne)} samples...")
        
        tsne_a = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_a_tsne)//4))
        tsne_b = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_b_tsne)//4))
        
        tsne_a_result = tsne_a.fit_transform(data_a_tsne)
        tsne_b_result = tsne_b.fit_transform(data_b_tsne)
        
        # Plot t-SNE results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(tsne_a_result[:, 0], tsne_a_result[:, 1], alpha=0.6)
        axes[0].set_title('Platform A - t-SNE')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        
        axes[1].scatter(tsne_b_result[:, 0], tsne_b_result[:, 1], alpha=0.6)
        axes[1].set_title('Platform B - t-SNE')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tsne_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca_a, pca_b
    
    def generate_summary_report(self, df_a: pd.DataFrame, df_b: pd.DataFrame, 
                              shared_ids: set, missing_a: pd.Series, missing_b: pd.Series):
        """Generate a summary report of the exploration."""
        print("\n" + "=" * 60)
        print("EXPLORATION SUMMARY REPORT")
        print("=" * 60)
        
        report = []
        report.append("=" * 60)
        report.append("DATA EXPLORATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"Platform A: {df_a.shape[0]} samples × {df_a.shape[1]-1} features")
        report.append(f"Platform B: {df_b.shape[0]} samples × {df_b.shape[1]-1} features")
        report.append(f"Shared samples: {len(shared_ids)}")
        report.append("")
        
        # Missing values
        report.append("MISSING VALUES:")
        report.append(f"Platform A: {missing_a.sum()} total missing values")
        report.append(f"Platform B: {missing_b.sum()} total missing values")
        report.append("")
        
        # Data quality recommendations
        report.append("RECOMMENDATIONS:")
        
        if missing_a.sum() > 0 or missing_b.sum() > 0:
            report.append("• Consider imputation strategies for missing values")
        
        if len(shared_ids) < min(len(df_a), len(df_b)) * 0.8:
            report.append("• Low overlap between datasets - verify ID matching")
        
        report.append("• Normalization comparison includes log scaling for skewed data")
        report.append("• Check normalization comparison output for best method recommendation")
        report.append("• PCA analysis suggests dimensionality reduction may be beneficial")
        report.append("")
        
        report.append("FILES GENERATED:")
        report.append("• missing_values_heatmap.png - Missing value patterns")
        report.append("• feature_distributions.png - Feature distribution histograms")
        report.append("• normalization_comparison.png - Normalization method comparison")
        report.append("• correlation_matrices.png - Feature correlation heatmaps")
        report.append("• pca_analysis.png - PCA explained variance and projections")
        report.append("• tsne_analysis.png - t-SNE projections")
        
        # Save report
        with open(self.output_dir / 'exploration_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))


def main():
    """
    Main exploration function for Joint VAE dataset analysis.
    
    Loads paired datasets, performs comprehensive exploratory analysis,
    and generates visualization reports for data understanding.
    """
    args = parse_arguments()
    
    # Initialize explorer
    explorer = DataExplorer(args.output_dir)
    
    print(f"Starting data exploration...")
    print(f"Platform A: {args.platform_a}")
    print(f"Platform B: {args.platform_b}")
    print(f"Output directory: {args.output_dir}")
    print(f"ID column: {args.id_column or 'auto-detect (first column)'}")
    
    # Load and analyze data
    df_a, df_b, id_column, shared_ids = explorer.load_and_analyze_data(
        args.platform_a, args.platform_b, args.id_column
    )
    
    # Filter to shared samples for analysis
    df_a_shared = df_a[df_a[id_column].isin(shared_ids)].reset_index(drop=True)
    df_b_shared = df_b[df_b[id_column].isin(shared_ids)].reset_index(drop=True)
    
    # Get feature columns
    features_a = [col for col in df_a_shared.columns if col != id_column]
    features_b = [col for col in df_b_shared.columns if col != id_column]
    
    data_a = df_a_shared[features_a]
    data_b = df_b_shared[features_b]
    
    # Perform analyses
    missing_a, missing_b = explorer.analyze_missing_values(df_a_shared, df_b_shared, id_column)
    explorer.analyze_distributions(data_a, data_b)
    explorer.compare_normalization_methods(data_a, data_b)
    
    # Correlation and dimensionality reduction analysis (may be skipped if insufficient data)
    corr_a, corr_b = explorer.analyze_correlations(data_a, data_b)
    pca_a, pca_b = explorer.perform_dimensionality_reduction(data_a, data_b)
    
    # Generate summary report
    explorer.generate_summary_report(df_a_shared, df_b_shared, shared_ids, missing_a, missing_b)
    
    print(f"\nExploration completed! Results saved to: {explorer.output_dir}")


if __name__ == "__main__":
    main() 
    