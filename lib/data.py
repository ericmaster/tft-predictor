import os
import pandas as pd
import numpy as np
from typing import Optional, List
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
import warnings
import random
warnings.filterwarnings("ignore")


class TFTDataModule(pl.LightningDataModule):
    """Lightning DataModule for Temporal Fusion Transformer with pytorch-forecasting."""
    
    def __init__(
        self,
        data_dir: str = "./data/resampled",
        min_encoder_length: int = 30,
        max_encoder_length: int = 150,  # Larger value tend to overfit
        max_prediction_length: int = 20,  # Larger value tend to overfit
        batch_size: int = 64,
        num_workers: int = 4,
        train_split: float = 0.75,
        val_split: float = 0.15, # 0.10 test split
        min_sequence_length: int = 100,
        # target: str = "duration",
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
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.min_sequence_length = min_sequence_length
        # self.target = target
        self.time_idx = time_idx
        self.group_ids = group_ids or ["session_id"]
        self.random_seed = random_seed
        self.target_names = ['duration', 'heartRate', 'temperature', 'cadence', 'speed']
        
        # Data storage
        # Raw data
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Datasets
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
            all_sessions.append(df)
        
        # Combine all sessions
        self.full_data = pd.concat(all_sessions, ignore_index=True)
        
        print(f"Loaded {len(all_sessions)} sessions with {len(self.full_data)} total data points")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, validation, and test datasets."""

        if stage == 'fit' and self.training is not None:
            return # Already setup

        if stage == 'val' and self.validation is not None:
            return # Already setup
        
        if stage == 'test' and self.test is not None:
            return # Already setup

        if self.full_data is None:
            self.prepare_data()
        
        # Filter out sessions that are too short for our sequence requirements
        session_lengths = self.full_data.groupby('session_id_encoded').size()
        min_required = self.max_encoder_length + self.max_prediction_length
        valid_sessions = session_lengths[session_lengths >= min_required].index
        
        print(f"Minimum required sequence length: {min_required}")
        print(f"Valid sessions: {len(valid_sessions)}/{len(session_lengths)}")
        
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
        self.train_data = self.full_data[self.full_data['session_id'].isin(train_session_ids)].copy()
        self.val_data = self.full_data[self.full_data['session_id'].isin(val_session_ids)].copy()
        self.test_data = self.full_data[self.full_data['session_id'].isin(test_session_ids)].copy()
        
        print(f"Session-based splits for cold-start evaluation:")
        print(f"Train sessions: {len(train_session_ids)}, Val sessions: {len(val_session_ids)}, Test sessions: {len(test_session_ids)}")
        print(f"Train data points: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # Verify no overlap between splits
        overlap_train_val = set(self.train_data['session_id'].unique()) & set(self.val_data['session_id'].unique())
        overlap_train_test = set(self.train_data['session_id'].unique()) & set(self.test_data['session_id'].unique())
        print(f"Overlap between train-val: {len(overlap_train_val)}, train-test: {len(overlap_train_test)}")
        
        # Define known future variables (these will be available at prediction time)
        time_varying_known_reals = [
            "altitude", 
            "elevation_diff", 
            "elevation_gain",
            "elevation_loss",
            # "distance", # Skip, redundant information since series are based on distance
        ]
        
        # Define target and unknown future variables (these need to be predicted/estimated)
        # We are performing multi-target forecasting: predict all these variables
        # Their past values are used as inputs to help predict their own and others' futures
        target = time_varying_unknown_reals = self.target_names

        # Use different normalizers for different targets based on their characteristics
        # This approach avoids the numerical issues we identified with softplus
        target_normalizer = MultiNormalizer(
            [
                # Duration: Use log1p transformation for wide range (~2 - ~20,000 seconds)
                # log1p = log(1 + x) avoids issues with zero values
                GroupNormalizer(
                    groups=["session_id_encoded"],
                    transformation="log1p"
                ),
                # Heart Rate: No transformation (standard normalization) for bounded range
                GroupNormalizer(
                    groups=["session_id_encoded"],
                    transformation=None
                ),
                # Temperature: No transformation (standard normalization)
                GroupNormalizer(
                    groups=["session_id_encoded"],
                    transformation=None
                ),
                # Cadence: No transformation (standard normalization)
                GroupNormalizer(
                    groups=["session_id_encoded"],
                    transformation=None
                ),
                # Speed: No transformation (standard normalization)
                GroupNormalizer(
                    groups=["session_id_encoded"],
                    transformation=None
                )
            ]
        )
        
        # Create training dataset
        # session_id_encoded is pre-calculated by DataResampler for cold-start evaluation
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            # target=self.target,
            target=target, # multi-target
            group_ids=["session_id_encoded"],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=True,  # Enable randomization to reduce overfitting
            allow_missing_timesteps=False, # We probably want to avoid this
            categorical_encoders={"session_id_encoded": NaNLabelEncoder(add_nan=True)},
            # Additional regularization parameters
            add_encoder_length=True,  # Explicitly add encoder length as a feature
            # scalers={col: 'standard' for col in time_varying_unknown_reals},  # Standardize targets
        )
        
        # Create validation and test datasets directly
        # Set predict=False to get proper validation sets with multiple samples per session
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            self.val_data, 
            predict=False,  # Changed to False for proper validation
            stop_randomization=True
        )
        
        self.test = TimeSeriesDataSet.from_dataset(
            self.training,
            self.test_data,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length, 
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

# Create sliding window chunks from the start of each session
def create_sliding_windows(df, session_col, encoder_length, prediction_length, step_size=200):
    """
    Create sliding window chunks from the beginning of each session.
    
    Args:
        df: DataFrame with session data
        session_col: Column name for session identifier
        encoder_length: Length of encoder sequence
        prediction_length: Length of prediction sequence  
        step_size: Step size for sliding window (default 200 = 1000m steps)
    """
    chunks = []
    
    for session_id in df[session_col].unique():
        session_data = df[df[session_col] == session_id].copy().reset_index(drop=True)
        
        if len(session_data) < encoder_length + prediction_length:
            print(f"Session {session_id} too short: {len(session_data)} < {encoder_length + prediction_length}")
            continue
            
        # Create sliding windows from start to end
        max_start = len(session_data) - (encoder_length + prediction_length)
        
        for start_idx in range(0, max_start + 1, step_size):
            end_idx = start_idx + encoder_length + prediction_length
            
            if end_idx <= len(session_data):
                chunk_data = session_data.iloc[start_idx:end_idx].copy()
                
                # Reset time_idx for this chunk (relative indexing)
                chunk_data['time_idx'] = range(len(chunk_data))
                
                chunk_info = {
                    'session_id': session_id,
                    'chunk_id': len(chunks),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'data': chunk_data,
                    'encoder_length': encoder_length,
                    'prediction_length': prediction_length,
                    'start_distance': session_data.iloc[start_idx]['distance'] if 'distance' in session_data.columns else start_idx * 2,
                    'end_distance': session_data.iloc[end_idx-1]['distance'] if 'distance' in session_data.columns else (end_idx-1) * 2
                }
                
                chunks.append(chunk_info)
    
    return chunks