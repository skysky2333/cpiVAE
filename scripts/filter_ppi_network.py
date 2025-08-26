#!/usr/bin/env python3
"""
Filter PPI Network by Dataset Protein Availability

This script filters a protein-protein interaction (PPI) network to only include
edges where both proteins are present in all provided dataset files.

Usage:
    python filter_ppi_network.py ppi_network.tsv --datasets data1.csv data2.csv --output filtered_ppi.tsv

The script will:
1. Load the PPI network from a TSV file
2. Extract protein names from each dataset's column headers
3. Find proteins common to all datasets
4. Filter PPI edges to only include those with both proteins in the common set
5. Save the filtered network with statistics
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ppi_network(ppi_path: str) -> pd.DataFrame:
    """
    Load protein-protein interaction network from TSV file.
    
    Args:
        ppi_path: Path to PPI network TSV file
        
    Returns:
        DataFrame with PPI network edges
    """
    logger.info(f"Loading PPI network from {ppi_path}")
    
    try:
        ppi_df = pd.read_csv(ppi_path, sep='\t')
        
        # Validate required columns
        required_cols = ['symbol1', 'symbol2', 'combined_score']
        missing_cols = [col for col in required_cols if col not in ppi_df.columns]
        if missing_cols:
            raise ValueError(f"PPI file missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(ppi_df)} PPI edges")
        return ppi_df
        
    except Exception as e:
        logger.error(f"Error loading PPI network: {e}")
        raise


def extract_proteins_from_dataset(dataset_path: str) -> set:
    """
    Extract protein names from dataset column headers.
    
    Args:
        dataset_path: Path to dataset CSV file
        
    Returns:
        Set of protein names found in dataset columns
    """
    logger.info(f"Extracting proteins from {dataset_path}")
    
    try:
        # Read only the header to get column names
        df_header = pd.read_csv(dataset_path, nrows=0)
        
        # Get all columns except the first one (which is typically sample ID)
        protein_cols = df_header.columns[1:].tolist()
        
        logger.info(f"Found {len(protein_cols)} proteins in {Path(dataset_path).name}")
        return set(protein_cols)
        
    except Exception as e:
        logger.error(f"Error reading dataset {dataset_path}: {e}")
        raise


def filter_ppi_by_proteins(ppi_df: pd.DataFrame, common_proteins: set) -> pd.DataFrame:
    """
    Filter PPI network to only include edges with both proteins in the common set.
    
    Args:
        ppi_df: DataFrame with PPI network
        common_proteins: Set of proteins present in all datasets
        
    Returns:
        Filtered PPI DataFrame
    """
    logger.info(f"Filtering PPI network to {len(common_proteins)} common proteins")
    
    # Create boolean mask for edges where both proteins are in common set
    mask = (ppi_df['symbol1'].isin(common_proteins) & 
            ppi_df['symbol2'].isin(common_proteins))
    
    filtered_df = ppi_df[mask].copy()
    
    # Calculate statistics
    unique_proteins_in_filtered = set(filtered_df['symbol1'].unique()) | set(filtered_df['symbol2'].unique())
    
    logger.info(f"Filtered from {len(ppi_df)} to {len(filtered_df)} edges")
    logger.info(f"Proteins in filtered network: {len(unique_proteins_in_filtered)}")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='Filter PPI network based on protein availability in datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'ppi_network',
        type=str,
        help='Path to PPI network TSV file'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Paths to dataset CSV files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for filtered PPI network'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files exist
    if not os.path.exists(args.ppi_network):
        logger.error(f"PPI network file not found: {args.ppi_network}")
        return 1
    
    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return 1
    
    try:
        # Load PPI network
        ppi_df = load_ppi_network(args.ppi_network)
        
        # Extract proteins from all datasets
        all_protein_sets = []
        for dataset_path in args.datasets:
            proteins = extract_proteins_from_dataset(dataset_path)
            all_protein_sets.append(proteins)
        
        # Find intersection of proteins across all datasets
        common_proteins = set.intersection(*all_protein_sets)
        logger.info(f"Found {len(common_proteins)} proteins common to all {len(args.datasets)} datasets")
        
        # Report proteins per dataset
        for i, (dataset_path, protein_set) in enumerate(zip(args.datasets, all_protein_sets)):
            logger.info(f"  Dataset {i+1} ({Path(dataset_path).name}): {len(protein_set)} proteins")
        
        # Filter PPI network
        filtered_ppi = filter_ppi_by_proteins(ppi_df, common_proteins)
        
        # Save filtered network
        output_dir = Path(args.output).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        filtered_ppi.to_csv(args.output, sep='\t', index=False)
        logger.info(f"Saved filtered PPI network to {args.output}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("FILTERING SUMMARY")
        print("="*50)
        print(f"Original PPI edges: {len(ppi_df):,}")
        print(f"Filtered PPI edges: {len(filtered_ppi):,}")
        print(f"Reduction: {(1 - len(filtered_ppi)/len(ppi_df))*100:.1f}%")
        print(f"Common proteins across datasets: {len(common_proteins):,}")
        
        unique_proteins_original = set(ppi_df['symbol1'].unique()) | set(ppi_df['symbol2'].unique())
        unique_proteins_filtered = set(filtered_ppi['symbol1'].unique()) | set(filtered_ppi['symbol2'].unique())
        print(f"Proteins in original network: {len(unique_proteins_original):,}")
        print(f"Proteins in filtered network: {len(unique_proteins_filtered):,}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())