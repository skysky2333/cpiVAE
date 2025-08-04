#!/usr/bin/env python3
"""
Random Data Splitter for Joint VAE Platform Files

This script randomly splits two platform CSV files into train and test sets
while maintaining sample ID alignment between platforms.

Usage:
    python split_data.py platform_a.csv platform_b.csv --test_size 0.2 --output_dir ./split_data

The script will:
1. Load both platform files
2. Merge them on the ID column to ensure alignment
3. Randomly split into train/test sets
4. Save the split files with appropriate naming
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_validate_files(file_a: str, file_b: str, id_column: str = None):
    """
    Load and validate the two platform CSV files.
    
    Args:
        file_a: Path to platform A CSV file
        file_b: Path to platform B CSV file
        id_column: Name of the ID column (defaults to first column)
        
    Returns:
        Tuple of (df_a, df_b, id_column_name)
    """
    logger.info(f"Loading platform files: {file_a}, {file_b}")
    
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {e}")
    
    if id_column is None:
        id_column = df_a.columns[0]
        logger.info(f"Using '{id_column}' as the ID column")
    
    if id_column not in df_a.columns:
        raise ValueError(f"ID column '{id_column}' not found in {file_a}")
    if id_column not in df_b.columns:
        raise ValueError(f"ID column '{id_column}' not found in {file_b}")
    
    logger.info(f"Platform A: {len(df_a)} samples, {len(df_a.columns)-1} features")
    logger.info(f"Platform B: {len(df_b)} samples, {len(df_b.columns)-1} features")
    
    return df_a, df_b, id_column


def merge_and_align_data(df_a: pd.DataFrame, df_b: pd.DataFrame, id_column: str):
    """
    Merge the two dataframes on the ID column and ensure alignment.
    
    Args:
        df_a: Platform A dataframe
        df_b: Platform B dataframe
        id_column: Name of the ID column
        
    Returns:
        Tuple of (aligned_df_a, aligned_df_b)
    """
    logger.info("Merging and aligning data...")
    
    ids_a = set(df_a[id_column])
    ids_b = set(df_b[id_column])
    common_ids = ids_a.intersection(ids_b)
    
    logger.info(f"Platform A unique IDs: {len(ids_a)}")
    logger.info(f"Platform B unique IDs: {len(ids_b)}")
    logger.info(f"Common IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common sample IDs found between the two platforms!")
    
    df_a_aligned = df_a[df_a[id_column].isin(common_ids)].sort_values(id_column).reset_index(drop=True)
    df_b_aligned = df_b[df_b[id_column].isin(common_ids)].sort_values(id_column).reset_index(drop=True)
    
    if not df_a_aligned[id_column].equals(df_b_aligned[id_column]):
        raise ValueError("Sample ID alignment failed!")
    
    logger.info(f"Data aligned successfully: {len(df_a_aligned)} samples")
    return df_a_aligned, df_b_aligned


def split_data(df_a: pd.DataFrame, df_b: pd.DataFrame, test_size: float, random_seed: int = 42):
    """
    Randomly split the aligned data into train and test sets.
    
    Args:
        df_a: Aligned platform A dataframe
        df_b: Aligned platform B dataframe
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_a, test_a, train_b, test_b)
    """
    logger.info(f"Splitting data with test_size={test_size}, random_seed={random_seed}")
    
    n_samples = len(df_a)
    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )
    
    train_a = df_a.iloc[train_indices].reset_index(drop=True)
    test_a = df_a.iloc[test_indices].reset_index(drop=True)
    train_b = df_b.iloc[train_indices].reset_index(drop=True)
    test_b = df_b.iloc[test_indices].reset_index(drop=True)
    
    logger.info(f"Train set: {len(train_a)} samples")
    logger.info(f"Test set: {len(test_a)} samples")
    
    return train_a, test_a, train_b, test_b


def save_split_files(train_a, test_a, train_b, test_b, 
                    file_a_path: str, file_b_path: str, output_dir: str):
    """
    Save the split data to CSV files.
    
    Args:
        train_a, test_a, train_b, test_b: Split dataframes
        file_a_path: Original platform A file path (for naming)
        file_b_path: Original platform B file path (for naming)
        output_dir: Directory to save split files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_a = Path(file_a_path).stem
    base_b = Path(file_b_path).stem
    
    train_a_path = os.path.join(output_dir, f"{base_a}_train.csv")
    test_a_path = os.path.join(output_dir, f"{base_a}_test.csv")
    train_b_path = os.path.join(output_dir, f"{base_b}_train.csv")
    test_b_path = os.path.join(output_dir, f"{base_b}_test.csv")
    
    logger.info("Saving split files...")
    train_a.to_csv(train_a_path, index=False)
    test_a.to_csv(test_a_path, index=False)
    train_b.to_csv(train_b_path, index=False)
    test_b.to_csv(test_b_path, index=False)
    
    logger.info(f"Files saved to {output_dir}:")
    logger.info(f"  Platform A train: {train_a_path}")
    logger.info(f"  Platform A test: {test_a_path}")
    logger.info(f"  Platform B train: {train_b_path}")
    logger.info(f"  Platform B test: {test_b_path}")


def main():
    """Main function to handle command line arguments and orchestrate the splitting process."""
    parser = argparse.ArgumentParser(
        description="Randomly split two platform CSV files into train and test sets while maintaining alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 20% test split
  python split_data.py platform_a.csv platform_b.csv
  
  # Custom test size and output directory
  python split_data.py platform_a.csv platform_b.csv --test_size 0.3 --output_dir ./my_splits
  
  # Specify ID column and random seed
  python split_data.py platform_a.csv platform_b.csv --id_column sample_id --random_seed 123
        """
    )
    
    parser.add_argument("--platform_a", help="Path to platform A CSV file")
    parser.add_argument("--platform_b", help="Path to platform B CSV file") 
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--output_dir", default="./split_data",
                       help="Directory to save split files (default: ./split_data)")
    parser.add_argument("--id_column", default=None,
                       help="Name of the ID column (default: first column)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be between 0.0 and 1.0")
    
    if not os.path.exists(args.platform_a):
        raise FileNotFoundError(f"Platform A file not found: {args.platform_a}")
    
    if not os.path.exists(args.platform_b):
        raise FileNotFoundError(f"Platform B file not found: {args.platform_b}")
    
    try:
        df_a, df_b, id_column = load_and_validate_files(
            args.platform_a, args.platform_b, args.id_column
        )
        
        df_a_aligned, df_b_aligned = merge_and_align_data(df_a, df_b, id_column)
        
        train_a, test_a, train_b, test_b = split_data(
            df_a_aligned, df_b_aligned, args.test_size, args.random_seed
        )
        
        save_split_files(
            train_a, test_a, train_b, test_b,
            args.platform_a, args.platform_b, args.output_dir
        )
        
        logger.info("Data splitting completed successfully!")
        
        print("\n" + "="*60)
        print("SPLIT SUMMARY")
        print("="*60)
        print(f"Original samples: {len(df_a_aligned)}")
        print(f"Train samples: {len(train_a)} ({len(train_a)/len(df_a_aligned)*100:.1f}%)")
        print(f"Test samples: {len(test_a)} ({len(test_a)/len(df_a_aligned)*100:.1f}%)")
        print(f"Platform A features: {len(df_a_aligned.columns)-1}")
        print(f"Platform B features: {len(df_b_aligned.columns)-1}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 