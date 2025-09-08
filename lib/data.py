import os
import pandas as pd
import numpy as np
from typing import Optional, List
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings("ignore")


class TFTDataModule(pl.LightningDataModule):
    """Lightning DataModule for Temporal Fusion Transformer with pytorch-forecasting."""
    
    def __init__(
        self,
        data_dir: str = "./data/resampled",
        max_encoder_length: int = 50,
        max_prediction_length: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        min_sequence_length: int = 100,
        target: str = "duration",
        time_idx: str = "distance",
        group_ids: List[str] = None
    ):
        """
        Initialize the TFT DataModule.
        
        Args:
            data_dir: Directory containing the CSV files
            max_encoder_length: Maximum length of encoder (input sequence)
            max_prediction_length: Maximum length of prediction horizon
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            min_sequence_length: Minimum length required for a session
            target: Target variable name
            time_idx: Time index column name
            group_ids: List of group identifier columns
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.min_sequence_length = min_sequence_length
        self.target = target
        self.time_idx = time_idx
        self.group_ids = group_ids or ["session_id"]
        
        # Data storage
        self.training = None
        self.validation = None
        self.test = None
        self.full_data = None
    
    def prepare_data(self):
        """Load and prepare the data."""
        # Load all sessions
        all_sessions = []
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        print(f"Loading {len(csv_files)} training session files...")
        
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path)
                
                # Add session identifier
                df['session_id'] = file.replace('.csv', '')
                
                all_sessions.append(df)
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Combine all sessions
        self.full_data = pd.concat(all_sessions, ignore_index=True)
        
        # Create a proper time index
        self.full_data = self.full_data.sort_values(['session_id', 'distance']).reset_index(drop=True)
        
        print(f"Loaded {len(all_sessions)} sessions with {len(self.full_data)} total data points")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets."""
        if self.full_data is None:
            self.prepare_data()
        
        # Create a combined dataset
        processed_data = []
        
        for session_id in self.full_data['session_id'].unique():
            session_data = self.full_data[self.full_data['session_id'] == session_id].copy()
            session_data = session_data.sort_values('distance').reset_index(drop=True)
            
            # Add split information based on temporal position within session
            total_length = len(session_data)
            train_split_point = int(total_length * 0.7)
            val_split_point = int(total_length * 0.85)
            
            session_data['split'] = 'train'
            session_data.loc[train_split_point:val_split_point, 'split'] = 'val'
            session_data.loc[val_split_point:, 'split'] = 'test'
            
            processed_data.append(session_data)
        
        # Combine all processed sessions
        combined_data = pd.concat(processed_data, ignore_index=True)
        
        # Split data
        train_data = combined_data[combined_data['split'] == 'train'].copy()
        val_data = combined_data[combined_data['split'].isin(['train', 'val'])].copy()  # Include train for encoder
        test_data = combined_data[combined_data['split'].isin(['train', 'test'])].copy()  # Include train for encoder
        
        # Add integer encoding for session_id
        all_sessions_sorted = sorted(combined_data['session_id'].unique())
        session_mapping = {session: idx for idx, session in enumerate(all_sessions_sorted)}
        
        train_data['session_id_encoded'] = train_data['session_id'].map(session_mapping)
        val_data['session_id_encoded'] = val_data['session_id'].map(session_mapping)
        test_data['session_id_encoded'] = test_data['session_id'].map(session_mapping)
        
        print(f"Data split - Sessions: {len(all_sessions_sorted)}")
        print(f"Train data points: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Define known future variables (these will be available at prediction time)
        time_varying_known_reals = [
            "altitude", 
            "elevation_diff", 
            "elevation_gain",
            "elevation_loss", 
            "distance_diff"
        ]
        
        # Define unknown future variables (these need to be predicted/estimated)
        time_varying_unknown_reals = [
            "heartRate",
            "temperature", 
            "cadence", 
            "speed"
        ]
        
        # Create training dataset
        self.training = TimeSeriesDataSet(
            train_data,
            time_idx="distance",
            target=self.target,
            group_ids=["session_id_encoded"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["session_id_encoded"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
            allow_missing_timesteps=True,  # Allow missing timesteps
        )
        
        # Create validation dataset
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            val_data, 
            predict=True, 
            stop_randomization=True
        )
        
        # Create test dataset
        self.test = TimeSeriesDataSet.from_dataset(
            self.training,
            test_data,
            predict=True,
            stop_randomization=True
        )
        
        print(f"Training samples: {len(self.training)}")
        print(f"Validation samples: {len(self.validation)}")
        print(f"Test samples: {len(self.test)}")
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.training.to_dataloader(
            train=True, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.validation.to_dataloader(
            train=False, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return self.test.to_dataloader(
            train=False, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
