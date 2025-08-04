"""
PyTorch Lightning DataModule for Joint VAE.

Handles data loading, preprocessing, and batch creation for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Tuple, Optional
from .preprocessing import DataPreprocessor


class JointVAEDataset(Dataset):
    """Dataset class for paired platform data."""
    
    def __init__(self, data_a: np.ndarray, data_b: np.ndarray):
        """
        Initialize dataset with paired data.
        
        Args:
            data_a: Platform A data array
            data_b: Platform B data array
        """
        assert len(data_a) == len(data_b), "Data arrays must have same length"
        
        self.data_a = torch.FloatTensor(data_a)
        self.data_b = torch.FloatTensor(data_b)
    
    def __len__(self) -> int:
        return len(self.data_a)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_a[idx], self.data_b[idx]


class JointVAEDataModule(pl.LightningDataModule):
    """Lightning DataModule for Joint VAE training."""
    
    def __init__(
        self,
        config: Dict,
        file_a: str = None,
        file_b: str = None,
        preprocessed_splits: Dict = None
    ):
        """
        Initialize the DataModule.
        
        Args:
            config: Configuration dictionary
            file_a: Path to platform A CSV file
            file_b: Path to platform B CSV file
            preprocessed_splits: Pre-processed data splits (alternative to file paths)
        """
        super().__init__()
        self.config = config
        self.file_a = file_a
        self.file_b = file_b
        self.preprocessed_splits = preprocessed_splits
        
        self.preprocessor = DataPreprocessor(config)
        self.splits = None
        self.dims = None
        
        # Store datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Prepare data (download, etc.). Called only once."""
        # This is where you would download data if needed
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        
        if self.splits is None:
            if self.preprocessed_splits is not None:
                # Use pre-processed splits
                self.splits = self.preprocessed_splits
            elif self.file_a is not None and self.file_b is not None:
                # The preprocessor now handles splitting and scaling correctly
                self.splits = self.preprocessor.process_data(self.file_a, self.file_b)
            else:
                raise ValueError("Either file paths or preprocessed splits must be provided")
        
        # Store dimensions
        train_data_a, _, _ = self.splits['train']
        val_data_a, _, _ = self.splits['val']
        test_data_a, _, _ = self.splits['test']

        self.dims = {
            'input_dim_a': train_data_a.shape[1],
            'input_dim_b': self.splits['train'][1].shape[1]
        }
        
        # Create datasets
        if stage == "fit" or stage is None:
            train_data_a, train_data_b, _ = self.splits['train']
            val_data_a, val_data_b, _ = self.splits['val']
            
            self.train_dataset = JointVAEDataset(train_data_a, train_data_b)
            self.val_dataset = JointVAEDataset(val_data_a, val_data_b)
        
        if stage == "test" or stage is None:
            test_data_a, test_data_b, _ = self.splits['test']
            self.test_dataset = JointVAEDataset(test_data_a, test_data_b)
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        data_config = self.config.get('data', {})
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True)
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        data_config = self.config.get('data', {})
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True)
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        data_config = self.config.get('data', {})
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True)
        )
    
    def get_dims(self) -> Dict[str, int]:
        """Get input dimensions for both platforms."""
        if self.dims is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return self.dims
    
    def get_preprocessor(self) -> DataPreprocessor:
        """Get the data preprocessor instance."""
        return self.preprocessor


class SingleDataModule(pl.LightningDataModule):
    """Simple data module for single platform QC VAE."""
    
    def __init__(self, config: dict, data_file: str):
        super().__init__()
        self.config = config
        self.data_file = data_file
        self.preprocessor = None
        self.train_data = None
        self.test_data = None
        self.full_data = None
        self.original_data = None
        
    def setup(self, stage: str = None):
        """Load and preprocess data."""
        import pandas as pd
        
        print(f"Loading data from {self.data_file}")
        
        # Load data - detect separator based on file extension
        if self.data_file.endswith('.txt'):
            separator = '\t'  # Tab-delimited for .txt files
        else:
            separator = ','   # Comma-delimited for .csv files
        
        data = pd.read_csv(self.data_file, index_col=0, sep=separator)
        print(f"Loaded data shape: {data.shape}")
        
        # Store original data
        self.original_data = data.copy()
        
        # Initialize preprocessor
        from .preprocessing import SinglePlatformPreprocessor
        self.preprocessor = SinglePlatformPreprocessor(self.config['data'])
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        train_indices, test_indices = train_test_split(
            range(len(data)), 
            test_size=self.config['data']['test_split'],
            random_state=self.config['data']['random_seed']
        )
        
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Fit preprocessor on training data and transform
        train_processed = self.preprocessor.fit_transform_single(train_data.values)
        test_processed = self.preprocessor.transform_single(test_data.values)
        
        # Convert to tensors
        self.train_data = torch.FloatTensor(train_processed)
        self.test_data = torch.FloatTensor(test_processed)
        
        # Also store full processed data for QC analysis
        full_processed = self.preprocessor.transform_single(data.values)
        self.full_data = torch.FloatTensor(full_processed)
        
        print(f"Processed train data shape: {self.train_data.shape}")
        print(f"Processed test data shape: {self.test_data.shape}")
        print(f"Processed full data shape: {self.full_data.shape}")
        
    def get_dims(self):
        """Get data dimensions."""
        return {'input_dim': self.train_data.shape[1]}
    
    def get_preprocessor(self):
        """Get the preprocessor."""
        return self.preprocessor
    
    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        """Get validation dataloader (using test set for early stopping)."""
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        ) 