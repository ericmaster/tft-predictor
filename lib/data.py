import os
import pandas as pd
import numpy as np
from typing import Optional, List
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
import random
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
        time_idx: str = "time_idx",
        group_ids: List[str] = None,
        random_seed: int = 42
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
            random_seed: Random seed for deterministic shuffling
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
        self.random_seed = random_seed
        
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
            df = pd.read_csv(file_path)
            # Add session identifier
            df['session_id'] = file.replace('.csv', '')
            all_sessions.append(df)
        
        # Combine all sessions
        self.full_data = pd.concat(all_sessions, ignore_index=True)
        
        # Create proper sequential "time indices" for TimeSeriesDataSet
        # Instead of using distance directly, create sequential indices per session
        session_data = []
        for session_id, group in self.full_data.groupby('session_id'):
            # Sort by distance to ensure proper order
            group = group.sort_values('distance').reset_index(drop=True)
            # Create sequential "time index" starting from 0 for each session
            group['time_idx'] = range(len(group))
            session_data.append(group)
        
        self.full_data = pd.concat(session_data, ignore_index=True)
        
        print(f"Loaded {len(all_sessions)} sessions with {len(self.full_data)} total data points")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets."""
        if self.full_data is None:
            self.prepare_data()
        
        # Extract timestamp-based integer IDs from session names
        # Session format: training-session-YYYY-MM-DD-timestamp-uuid
        def extract_session_timestamp(session_id):
            """Extract timestamp from session ID for use as integer identifier."""
            try:
                # Split by '-' and get the timestamp part (index 5)
                parts = session_id.split('-')
                if len(parts) >= 6:
                    timestamp = int(parts[5])  # The numeric timestamp
                    return timestamp
                else:
                    # Fallback: use hash of session_id if format is unexpected
                    return abs(hash(session_id)) % (10**10)
            except (ValueError, IndexError):
                # Fallback: use hash of session_id
                return abs(hash(session_id)) % (10**10)
        
        self.full_data['session_id_encoded'] = self.full_data['session_id'].apply(extract_session_timestamp)
        
        # Filter out sessions that are too short for our sequence requirements
        session_lengths = self.full_data.groupby('session_id_encoded').size()
        min_required = self.max_encoder_length + self.max_prediction_length
        valid_sessions = session_lengths[session_lengths >= min_required].index
        
        print(f"Minimum required sequence length: {min_required}")
        print(f"Valid sessions (sufficient length): {len(valid_sessions)}/{len(session_lengths)}")
        
        # Filter data to only include valid sessions
        self.full_data = self.full_data[self.full_data['session_id_encoded'].isin(valid_sessions)].copy()
        
        # Split by sessions for cold-start evaluation
        # This tests the model's ability to predict on completely new sessions
        all_session_ids = sorted(self.full_data['session_id'].unique())
        n_sessions = len(all_session_ids)
        
        train_sessions = int(n_sessions * self.train_split)
        val_sessions = int(n_sessions * self.val_split)
        
        train_session_ids = all_session_ids[:train_sessions]
        val_session_ids = all_session_ids[train_sessions:train_sessions + val_sessions]
        test_session_ids = all_session_ids[train_sessions + val_sessions:]
        
        # Create data splits by sessions
        train_data = self.full_data[self.full_data['session_id'].isin(train_session_ids)].copy()
        val_data = self.full_data[self.full_data['session_id'].isin(val_session_ids)].copy()
        test_data = self.full_data[self.full_data['session_id'].isin(test_session_ids)].copy()
        
        print(f"Session-based splits for cold-start evaluation:")
        print(f"Train sessions: {len(train_session_ids)}, Val sessions: {len(val_session_ids)}, Test sessions: {len(test_session_ids)}")
        print(f"Train data points: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Verify no overlap between splits
        overlap_train_val = set(train_data['session_id'].unique()) & set(val_data['session_id'].unique())
        overlap_train_test = set(train_data['session_id'].unique()) & set(test_data['session_id'].unique())
        print(f"Overlap between train-val: {len(overlap_train_val)}, train-test: {len(overlap_train_test)}")
        
        # With timestamp-based IDs, we don't need complex re-encoding
        # Each session already has a unique integer ID based on its timestamp
        
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
        # For cold-start evaluation with timestamp-based session IDs, we need to handle unknown sessions
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        
        self.training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
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
            allow_missing_timesteps=True,
            categorical_encoders={"session_id_encoded": NaNLabelEncoder(add_nan=True)},
        )
        
        # Create validation and test datasets directly
        # Unknown timestamp-based session IDs will be handled as NaN/unknown category
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            val_data, 
            predict=True, 
            stop_randomization=True
        )
        
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
