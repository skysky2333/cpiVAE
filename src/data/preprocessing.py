"""
Data preprocessing module for Joint VAE.

Handles loading, merging, cleaning, and normalizing paired CSV files
from different metabolite measurement platforms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing steps for the Joint VAE model.
    
    This includes loading CSV files, merging on participant IDs,
    handling missing values, log transformation, normalization, and train/val/test splitting.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing data processing configuration
        """
        self.config = config['data']
        self.scalers = {'platform_a': None, 'platform_b': None}
        self.feature_names = {'platform_a': None, 'platform_b': None}
        
        # Log transformation parameters
        self.log_transform_params = {
            'platform_a': {'enabled': False, 'shift_value': 0.0},
            'platform_b': {'enabled': False, 'shift_value': 0.0}
        }
        
    def load_and_merge_data(
        self, 
        file_a: str, 
        file_b: str,
        id_column: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load two CSV files and merge them on a shared ID column.
        
        Args:
            file_a: Path to platform A CSV file
            file_b: Path to platform B CSV file
            id_column: Name of the ID column (defaults to first column)
            
        Returns:
            Tuple of (platform_a_data, platform_b_data, merged_metadata)
        """
        logger.info(f"Loading data from {file_a} and {file_b}")
        
        # Load CSV files - detect separator based on file extension
        sep_a = '\t' if file_a.endswith('.txt') else ','
        sep_b = '\t' if file_b.endswith('.txt') else ','
        
        df_a = pd.read_csv(file_a, sep=sep_a)
        df_b = pd.read_csv(file_b, sep=sep_b)
        
        # Use first column as ID if not specified
        if id_column is None:
            id_column = df_a.columns[0]
        
        logger.info(f"Using '{id_column}' as the ID column")
        
        # Extract feature columns (all except ID column)
        features_a = [col for col in df_a.columns if col != id_column]
        features_b = [col for col in df_b.columns if col != id_column]
        
        # Store feature names
        self.feature_names['platform_a'] = features_a
        self.feature_names['platform_b'] = features_b
        
        logger.info(f"Platform A: {len(features_a)} features, {len(df_a)} samples")
        logger.info(f"Platform B: {len(features_b)} features, {len(df_b)} samples")
        
        # Perform inner join on ID column
        merged_ids = pd.merge(
            df_a[[id_column]], 
            df_b[[id_column]], 
            on=id_column, 
            how='inner'
        )
        
        logger.info(f"After merging: {len(merged_ids)} shared samples")
        
        if len(merged_ids) == 0:
            raise ValueError("No shared samples found between the two datasets!")
        
        # Filter both datasets to only include shared samples
        df_a_filtered = df_a[df_a[id_column].isin(merged_ids[id_column])]
        df_b_filtered = df_b[df_b[id_column].isin(merged_ids[id_column])]
        
        # Sort by ID to ensure alignment
        df_a_filtered = df_a_filtered.sort_values(id_column).reset_index(drop=True)
        df_b_filtered = df_b_filtered.sort_values(id_column).reset_index(drop=True)
        
        # Extract feature data only
        data_a = df_a_filtered[features_a]
        data_b = df_b_filtered[features_b]
        
        # Create metadata DataFrame
        metadata = pd.DataFrame({
            id_column: df_a_filtered[id_column].values
        })
        
        logger.info("Data loading and merging completed successfully")
        
        return data_a, data_b, metadata
    
    def handle_missing_values(
        self, 
        data_a: pd.DataFrame, 
        data_b: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values in both datasets.
        
        Args:
            data_a: Platform A feature data
            data_b: Platform B feature data
            
        Returns:
            Tuple of cleaned DataFrames
        """
        strategy = self.config['missing_value_strategy']
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        # Report missing value statistics
        missing_a = data_a.isnull().sum().sum()
        missing_b = data_b.isnull().sum().sum()
        logger.info(f"Platform A missing values: {missing_a}")
        logger.info(f"Platform B missing values: {missing_b}")
        
        if strategy == 'drop':
            # Drop samples with any missing values
            mask_a = ~data_a.isnull().any(axis=1)
            mask_b = ~data_b.isnull().any(axis=1)
            combined_mask = mask_a & mask_b
            
            data_a = data_a[combined_mask].reset_index(drop=True)
            data_b = data_b[combined_mask].reset_index(drop=True)
            
        elif strategy in ['mean', 'median']:
            # Simple imputation
            imputer = SimpleImputer(strategy=strategy)
            data_a = pd.DataFrame(
                imputer.fit_transform(data_a),
                columns=data_a.columns
            )
            data_b = pd.DataFrame(
                imputer.fit_transform(data_b),
                columns=data_b.columns
            )
            
        elif strategy == 'knn':
            # K-nearest neighbors imputation
            imputer = KNNImputer(n_neighbors=5)
            data_a = pd.DataFrame(
                imputer.fit_transform(data_a),
                columns=data_a.columns
            )
            data_b = pd.DataFrame(
                imputer.fit_transform(data_b),
                columns=data_b.columns
            )
        
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
        
        logger.info("Missing value handling completed")
        return data_a, data_b
    
    def log_transform_data(
        self, 
        data_a: pd.DataFrame, 
        data_b: pd.DataFrame,
        fit_transforms: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply log transformation to datasets based on configuration.
        
        Args:
            data_a: Platform A feature data
            data_b: Platform B feature data
            fit_transforms: Whether to fit new transformation parameters (True for training)
            
        Returns:
            Tuple of log-transformed DataFrames
        """
        log_config = self.config.get('log_transform', {})
        
        results = []
        for data, platform in [(data_a, 'platform_a'), (data_b, 'platform_b')]:
            platform_config = log_config.get(platform, {})
            apply_log = platform_config.get('enabled', False)
            
            if apply_log and data is not None:
                logger.info(f"Applying log transformation to {platform}")
                
                if fit_transforms:
                    # Calculate shift value to handle non-positive values
                    min_val = data.min().min()
                    if min_val <= 0:
                        # Add small epsilon to ensure all values are positive
                        shift_value = -min_val + platform_config.get('epsilon', 1e-8)
                        logger.info(f"  {platform}: Shifting data by {shift_value:.6f} to handle non-positive values")
                    else:
                        shift_value = 0.0
                    
                    # Store transformation parameters
                    self.log_transform_params[platform] = {
                        'enabled': True,
                        'shift_value': shift_value
                    }
                else:
                    # Use stored transformation parameters
                    if not self.log_transform_params[platform]['enabled']:
                        raise ValueError(f"Log transformation not fitted for {platform}")
                    shift_value = self.log_transform_params[platform]['shift_value']
                
                # Apply transformation
                shifted_data = data + shift_value
                log_data = np.log(shifted_data)
                
                # Convert back to DataFrame
                log_data = pd.DataFrame(log_data, columns=data.columns, index=data.index)
                results.append(log_data)
                
                logger.info(f"  {platform}: Log transformation completed")
            else:
                # No transformation needed
                if fit_transforms:
                    self.log_transform_params[platform] = {
                        'enabled': False,
                        'shift_value': 0.0
                    }
                results.append(data)
        
        return tuple(results)
    
    def inverse_log_transform_data(
        self, 
        data_a: Optional[np.ndarray] = None, 
        data_b: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply inverse log transformation to get data back to original scale.
        
        Args:
            data_a: Platform A data to inverse transform
            data_b: Platform B data to inverse transform
            
        Returns:
            Tuple of inverse-transformed data arrays
        """
        results = []
        
        for data, platform in [(data_a, 'platform_a'), (data_b, 'platform_b')]:
            if data is not None:
                if self.log_transform_params[platform]['enabled']:
                    #logger.info(f"Applying inverse log transformation to {platform}")
                    
                    # Apply inverse transformation
                    shift_value = self.log_transform_params[platform]['shift_value']
                    exp_data = np.exp(data)
                    original_data = exp_data - shift_value
                    
                    results.append(original_data)
                else:
                    # No inverse transformation needed
                    results.append(data)
            else:
                results.append(None)
        
        return tuple(results)
    
    def normalize_data(
        self, 
        data_a: pd.DataFrame, 
        data_b: pd.DataFrame,
        fit_scalers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize both datasets independently to prevent data leakage.
        
        Args:
            data_a: Platform A feature data (possibly log-transformed)
            data_b: Platform B feature data (possibly log-transformed)
            fit_scalers: Whether to fit new scalers (True for training)
            
        Returns:
            Tuple of normalized numpy arrays
        """
        method = self.config['normalization_method']
        logger.info(f"Normalizing data using method: {method}")
        
        # Choose scaler
        if method == 'zscore':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if fit_scalers:
            # Fit scalers on training data
            self.scalers['platform_a'] = scaler_class()
            self.scalers['platform_b'] = scaler_class()
            
            data_a_norm = self.scalers['platform_a'].fit_transform(data_a)
            data_b_norm = self.scalers['platform_b'].fit_transform(data_b)
        else:
            # Use existing scalers (for validation/test data)
            if self.scalers['platform_a'] is None or self.scalers['platform_b'] is None:
                raise ValueError("Scalers not fitted. Call with fit_scalers=True first.")
            
            data_a_norm = self.scalers['platform_a'].transform(data_a)
            data_b_norm = self.scalers['platform_b'].transform(data_b)
        
        logger.info("Data normalization completed")
        return data_a_norm, data_b_norm
    
    def split_data(
        self, 
        data_a: np.ndarray, 
        data_b: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data_a: Normalized platform A data
            data_b: Normalized platform B data
            metadata: Sample metadata
            
        Returns:
            Dictionary with train/val/test splits
        """
        train_split = self.config['train_split']
        val_split = self.config['val_split']
        test_split = self.config['test_split']
        random_seed = self.config['random_seed']
        
        logger.info(f"Splitting data: train={train_split}, val={val_split}, test={test_split}")
        
        # Validate split ratios
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        n_samples = len(data_a)
        indices = np.arange(n_samples)
        
        # First split: train + val vs test
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_split,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: train vs val
        val_ratio = val_split / (train_split + val_split)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Create splits
        splits = {
            'train': (
                data_a[train_indices],
                data_b[train_indices],
                metadata.iloc[train_indices].reset_index(drop=True)
            ),
            'val': (
                data_a[val_indices],
                data_b[val_indices],
                metadata.iloc[val_indices].reset_index(drop=True)
            ),
            'test': (
                data_a[test_indices],
                data_b[test_indices],
                metadata.iloc[test_indices].reset_index(drop=True)
            )
        }
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Val: {len(val_indices)} samples")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return splits
    
    def process_data(
        self, 
        file_a: str, 
        file_b: str,
        id_column: str = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            file_a: Path to platform A CSV file
            file_b: Path to platform B CSV file
            id_column: Name of the ID column
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Starting complete data preprocessing pipeline")
        
        # Step 1: Load and merge data
        data_a, data_b, metadata = self.load_and_merge_data(file_a, file_b, id_column)
        
        # Step 2: Handle missing values
        data_a, data_b = self.handle_missing_values(data_a, data_b)
        
        # Step 3: Apply log transformation (if configured)
        data_a, data_b = self.log_transform_data(data_a, data_b, fit_transforms=True)
        
        # Step 4: Normalize data
        data_a_norm, data_b_norm = self.normalize_data(data_a, data_b, fit_scalers=True)
        
        # Step 5: Split data
        splits = self.split_data(data_a_norm, data_b_norm, metadata)
        
        logger.info("Data preprocessing pipeline completed successfully")
        return splits
    
    def transform_new_data(
        self, 
        data_a: Optional[pd.DataFrame] = None,
        data_b: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transform new data using fitted transformations and scalers (for inference).
        
        Args:
            data_a: New platform A data
            data_b: New platform B data
            
        Returns:
            Tuple of normalized data arrays
        """
        results = []
        
        for data, platform in [(data_a, 'platform_a'), (data_b, 'platform_b')]:
            if data is not None:
                # Handle missing values
                if data.isnull().any().any():
                    logger.warning(f"Found missing values in {platform} data")
                    if self.config['missing_value_strategy'] == 'mean':
                        data = data.fillna(data.mean())
                    elif self.config['missing_value_strategy'] == 'median':
                        data = data.fillna(data.median())
                
                # Apply log transformation if needed
                if self.log_transform_params[platform]['enabled']:
                    shift_value = self.log_transform_params[platform]['shift_value']
                    shifted_data = data + shift_value
                    log_data = np.log(shifted_data)
                    log_data = pd.DataFrame(log_data, columns=data.columns, index=data.index)
                    data = log_data
                
                # Normalize using fitted scaler
                if self.scalers[platform] is None:
                    raise ValueError(f"Scaler for {platform} not fitted")
                
                normalized = self.scalers[platform].transform(data)
                results.append(normalized)
            else:
                results.append(None)
        
        return tuple(results)
    
    def get_feature_names(self) -> Dict[str, list]:
        """Get feature names for both platforms."""
        return self.feature_names.copy()
    
    def get_scalers(self) -> Dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        """Get fitted scalers."""
        return self.scalers.copy()
    
    def get_log_transform_params(self) -> Dict[str, Dict[str, Union[bool, float]]]:
        """Get log transformation parameters."""
        return self.log_transform_params


class SinglePlatformPreprocessor:
    """Simple preprocessor for single platform QC analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.feature_names = None
        self.log_transform_params = {'enabled': False, 'shift_value': 0.0}
        
    def fit_transform_single(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform single platform data."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        data_df = pd.DataFrame(data)
        
        # Convert all columns to numeric, coercing errors to NaN
        print(f"Converting data to numeric format...")
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
        # Report how many values were converted to NaN
        total_nan = data_df.isnull().sum().sum()
        if total_nan > 0:
            print(f"Warning: {total_nan} non-numeric values converted to NaN")
        
        # Handle missing values
        if self.config['missing_value_strategy'] == 'mean':
            data_df = data_df.fillna(data_df.mean())
        elif self.config['missing_value_strategy'] == 'median':
            data_df = data_df.fillna(data_df.median())
        
        # Apply log transformation if enabled
        log_config = self.config.get('log_transform', {})
        if log_config.get('enabled', False):
            min_val = data_df.min().min()
            if min_val <= 0:
                shift_value = -min_val + log_config.get('epsilon', 1e-8)
            else:
                shift_value = 0.0
            
            self.log_transform_params = {
                'enabled': True,
                'shift_value': shift_value
            }
            
            data_df = np.log(data_df + shift_value)
        
        # Normalize data
        if self.config['normalization_method'] == 'zscore':
            self.scaler = StandardScaler()
        elif self.config['normalization_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config['normalization_method'] == 'robust':
            self.scaler = RobustScaler()
        
        if self.scaler is not None:
            data_normalized = self.scaler.fit_transform(data_df.values)
        else:
            data_normalized = data_df.values
            
        return data_normalized
        
    def transform_single(self, data: np.ndarray) -> np.ndarray:
        """Transform single platform data using fitted parameters."""
        import pandas as pd
        
        data_df = pd.DataFrame(data)
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
        # Handle missing values
        if self.config['missing_value_strategy'] == 'mean':
            data_df = data_df.fillna(data_df.mean())
        elif self.config['missing_value_strategy'] == 'median':
            data_df = data_df.fillna(data_df.median())
        
        # Apply log transformation if enabled
        if self.log_transform_params['enabled']:
            data_df = np.log(data_df + self.log_transform_params['shift_value'])
        
        # Normalize data
        if self.scaler is not None:
            data_normalized = self.scaler.transform(data_df.values)
        else:
            data_normalized = data_df.values
            
        return data_normalized
        
    def inverse_log_transform_single(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse log transformation to data."""
        if self.log_transform_params['enabled']:
            return np.exp(data) - self.log_transform_params['shift_value']
        else:
            return data.copy() 