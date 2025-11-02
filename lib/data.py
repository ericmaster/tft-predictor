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
        min_encoder_length: int = 1,
        max_encoder_length: int = 20,
        max_prediction_length: int = 200,
        batch_size: int = 64,
        num_workers: int = 4,
        train_split: float = 0.75,
        val_split: float = 0.15, # 0.10 test split
        min_sequence_length: int = 100,
        # target: str = "duration",
        time_idx: str = "time_idx",
        group_ids: List[str] = None,
        random_seed: int = 42,
        use_sliding_windows: bool = False
    ):
        """
        Initialize the TFT DataModule.
        
        Args:
            data_dir: Directory containing the CSV files
            min_encoder_length: Minimum length of encoder (for cold-start)
            max_encoder_length: Maximum length of encoder (input sequence)
            max_prediction_length: Maximum length of prediction horizon
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            min_sequence_length: Minimum length required for a session
            time_idx: Time index column name
            group_ids: List of group identifier columns
            random_seed: Random seed for deterministic shuffling
            use_sliding_windows: If True, creates overlapping chunks from sessions
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
        self.use_sliding_windows = use_sliding_windows
        # For prediction, we will forecast multiple targets
        # Instead of duration we are predicting duration_diff (change in duration)
        # This allows the model to learn more complex patterns
        # Cumulative duration is monotonically increasing which is hard to model.
        # Gradients become unstable over long horizons.
        self.target_names = ['duration_diff', 'heartRate', 'temperature', 'cadence']
        
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
    
    def _calculate_weighted_first_sample(self, df: pd.DataFrame, up_to_session_id: int):
        """
        Calculate weighted average of first samples from all sessions up to a given session.
        
        Uses chronological weighting: more recent sessions get higher weights based on
        days from the first session. This simulates realistic cold-start conditions where
        physical conditioning evolves over time.
        
        For time_varying_known_reals (altitude, elevation_diff, elevation_gain, elevation_loss),
        uses actual values from the current session's first sample since terrain is known at start.
        
        Args:
            df: DataFrame with all session data
            up_to_session_id: Calculate average using sessions up to and including this ID
            
        Returns:
            Series with weighted average values for all columns
        """
        # Get all sessions up to the target session (chronologically)
        session_ids = sorted(df['session_id_encoded'].unique())
        target_idx = session_ids.index(up_to_session_id)
        sessions_to_use = session_ids[:target_idx + 1]
        
        # Get the actual first sample from the current session
        current_session_data = df[df['session_id_encoded'] == up_to_session_id]
        current_first_sample = current_session_data.iloc[0].copy()
        
        # Extract first sample from each session for weighted average
        first_samples = []
        session_dates = []
        
        for sid in sessions_to_use:
            session_data = df[df['session_id_encoded'] == sid]
            first_sample = session_data.iloc[0].copy()
            first_samples.append(first_sample)
            
            # Extract date from session_id (format: training-session-YYYY-MM-DD-...)
            session_id_str = first_sample['session_id']
            if 'session' in session_id_str:
                # Parse date from session_id
                parts = session_id_str.split('-')
                if len(parts) >= 5:
                    date_str = f"{parts[2]}-{parts[3]}-{parts[4]}"
                    session_dates.append(pd.to_datetime(date_str))
                else:
                    # Fallback: use session_id_encoded as proxy
                    session_dates.append(pd.to_datetime(sid, unit='s', origin='unix'))
            else:
                session_dates.append(pd.to_datetime(sid, unit='s', origin='unix'))
        
        # Calculate weights based on days from first session
        first_date = min(session_dates)
        weights = [(date - first_date).days + 1 for date in session_dates]  # +1 to avoid zero weight
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Calculate weighted average
        first_samples_df = pd.DataFrame(first_samples)
        weighted_avg = pd.Series(0.0, index=first_samples_df.columns)
        
        # Define known future variables (terrain/GPS data - known at race start)
        known_future_vars = ['altitude', 'elevation_diff', 'elevation_gain', 'elevation_loss', 'distance']
        
        for col in first_samples_df.columns:
            if col in known_future_vars:
                # Use actual value from current session for known terrain data
                weighted_avg[col] = current_first_sample[col]
            elif first_samples_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Use weighted average for unknown variables (physiological metrics)
                weighted_avg[col] = np.average(first_samples_df[col], weights=weights)
            else:
                # For non-numeric columns, use the most recent value
                weighted_avg[col] = first_samples_df[col].iloc[-1]
        
        return weighted_avg
    
    def create_sliding_window_chunks(self, df: pd.DataFrame):
        """
        Create sliding window chunks from each session for sequential prediction.
        
        Strategy:
        - First chunk (cold-start): Synthetic encoder (weighted avg of first samples) 
          + predict steps 0-199 (200 predictions)
        - Next chunks: Use last max_encoder_length (20) steps from previous prediction 
          as encoder + predict next max_prediction_length (200) steps
        
        The synthetic encoder uses a weighted average of first samples from all sessions
        up to the current session (chronologically), with more recent sessions weighted
        higher based on days from first session. This simulates realistic cold-start
        where we don't have initial data but can estimate from training history.
        
        This allows the model to:
        1. Learn cold-start prediction with synthetic initial conditions
        2. Accumulate duration by summing predicted duration_diff values
        3. Use fatigue proxy features calculated from accumulated predictions
        4. Chain predictions across the entire session
        
        Args:
            df: DataFrame with session data containing 'session_id_encoded'
            
        Returns:
            List of chunk dictionaries with session metadata and data
        """
        session_col = 'session_id_encoded'
        chunks = []

        for session_id in sorted(df[session_col].unique()):
            session_data = df[df[session_col] == session_id].copy().reset_index(drop=True)
            session_len = len(session_data)
            
            # Minimum length check (only need prediction length for first chunk now)
            min_required = self.max_prediction_length
            if session_len < min_required:
                print(f"Session {session_id} too short: {session_len} < {min_required}")
                continue
            
            chunk_counter = 0
            start_idx = 0
            
            while start_idx < session_len:
                if chunk_counter == 0:
                    # First chunk: cold-start with synthetic encoder
                    encoder_len = self.min_encoder_length
                    
                    # Calculate weighted average of first samples up to this session
                    synthetic_encoder = self._calculate_weighted_first_sample(df, session_id)
                    
                    # Prediction starts at step 0
                    pred_start = 0
                    pred_end = min(self.max_prediction_length, session_len)
                    
                    # Create chunk with synthetic encoder + actual predictions
                    synthetic_encoder['time_idx'] = 0  # Synthetic sample at time_idx 0
                    synthetic_encoder['chunk_id'] = chunk_counter
                    
                    # Ensure session_id_encoded is integer (important for chunk_identifier)
                    synthetic_encoder['session_id_encoded'] = int(session_id)
                    
                    # Get prediction data (steps 0 to pred_end)
                    pred_data = session_data.iloc[pred_start:pred_end].copy()
                    pred_data['time_idx'] = range(1, len(pred_data) + 1)  # Start from 1 after synthetic
                    pred_data['chunk_id'] = chunk_counter
                    
                    # Combine synthetic encoder + predictions
                    chunk_data = pd.concat([synthetic_encoder.to_frame().T, pred_data], ignore_index=True)
                    
                    # Ensure consistent dtypes
                    chunk_data['session_id_encoded'] = chunk_data['session_id_encoded'].astype(int)
                    chunk_data['time_idx'] = chunk_data['time_idx'].astype(int)
                    chunk_data['chunk_id'] = chunk_data['chunk_id'].astype(int)
                    
                    chunk_info = {
                        'session_id': session_id,
                        'chunk_id': chunk_counter,
                        'start_idx': pred_start,
                        'end_idx': pred_end,
                        'data': chunk_data,
                        'encoder_length': encoder_len,
                        'prediction_length': pred_end - pred_start,
                        'start_distance': session_data.iloc[pred_start]['distance'] if 'distance' in session_data.columns else pred_start * 5,
                        'end_distance': session_data.iloc[pred_end-1]['distance'] if 'distance' in session_data.columns else (pred_end-1) * 5,
                        'is_cold_start': True
                    }
                    chunks.append(chunk_info)
                    
                    # Next chunk starts where we can extract encoder from predictions
                    # Last 20 steps of predictions (steps 180-199) become next encoder
                    start_idx = pred_end - self.max_encoder_length
                    
                else:
                    # Subsequent chunks: use last 20 predicted steps as encoder
                    encoder_len = self.max_encoder_length
                    
                    # Calculate chunk boundaries
                    end_idx = start_idx + encoder_len + self.max_prediction_length
                    
                    # Check if we have enough data
                    if end_idx > session_len:
                        # Not enough data for a full chunk, stop here
                        break
                    
                    # Extract chunk data
                    chunk_data = session_data.iloc[start_idx:end_idx].copy()
                    chunk_data['time_idx'] = range(len(chunk_data))
                    chunk_data['chunk_id'] = chunk_counter
                    
                    chunk_info = {
                        'session_id': session_id,
                        'chunk_id': chunk_counter,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'data': chunk_data,
                        'encoder_length': encoder_len,
                        'prediction_length': self.max_prediction_length,
                        'start_distance': session_data.iloc[start_idx]['distance'] if 'distance' in session_data.columns else start_idx * 5,
                        'end_distance': session_data.iloc[end_idx-1]['distance'] if 'distance' in session_data.columns else (end_idx-1) * 5,
                        'is_cold_start': False
                    }
                    chunks.append(chunk_info)
                    
                    # Next chunk starts where this chunk's prediction begins
                    # Last 20 predicted steps become the encoder for the next chunk
                    start_idx += encoder_len + self.max_prediction_length - self.max_encoder_length
                
                chunk_counter += 1

        print(f"Created {len(chunks)} chunks from {df[session_col].nunique()} sessions")
        print(f"Average chunks per session: {len(chunks) / df[session_col].nunique():.1f}")
        
        return chunks
    
    def chunks_to_dataframe(self, chunks):
        """
        Convert list of chunk dictionaries to a single DataFrame.
        
        Args:
            chunks: List of chunk dictionaries from create_sliding_window_chunks
            
        Returns:
            Combined DataFrame with all chunks
        """
        if not chunks:
            return pd.DataFrame()
        
        # Concatenate all chunk data
        all_chunk_data = [chunk['data'] for chunk in chunks]
        combined_df = pd.concat(all_chunk_data, ignore_index=True)
        
        # Create unique chunk identifiers across sessions
        # Format: {session_id}_{chunk_id}
        combined_df['chunk_identifier'] = combined_df.apply(
            lambda row: f"{row['session_id_encoded']}_{row['chunk_id']}", 
            axis=1
        )
        
        return combined_df

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
        train_data_raw = self.full_data[self.full_data['session_id'].isin(train_session_ids)].copy()
        val_data_raw = self.full_data[self.full_data['session_id'].isin(val_session_ids)].copy()
        test_data_raw = self.full_data[self.full_data['session_id'].isin(test_session_ids)].copy()
        
        # Apply sliding window chunking if enabled
        if self.use_sliding_windows:
            print(f"Creating sliding window chunks...")
            train_chunks = self.create_sliding_window_chunks(train_data_raw)
            val_chunks = self.create_sliding_window_chunks(val_data_raw)
            test_chunks = self.create_sliding_window_chunks(test_data_raw)
            
            self.train_data = self.chunks_to_dataframe(train_chunks)
            self.val_data = self.chunks_to_dataframe(val_chunks)
            self.test_data = self.chunks_to_dataframe(test_chunks)
            
            # Use chunk_identifier as group_id for normalization
            self.group_ids = ["chunk_identifier"]
        else:
            self.train_data = train_data_raw
            self.val_data = val_data_raw
            self.test_data = test_data_raw
        
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
        time_varying_unknown_reals.append("speed")
        
        # Create training dataset
        # session_id_encoded is pre-calculated by DataResampler for cold-start evaluation
        # If using sliding windows, we use chunk_identifier instead
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        
        group_id_col = self.group_ids[0] if self.use_sliding_windows else "session_id_encoded"
        
        # Update normalizers to use appropriate group column
        # if self.use_sliding_windows:
        target_normalizer = MultiNormalizer(
            [GroupNormalizer(groups=[group_id_col], transformation=None) for _ in range(len(self.target_names))]
        )
        
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            # target=self.target,
            target=target, # multi-target
            group_ids=[group_id_col],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=not self.use_sliding_windows,  # Disable if using windows (already chunked)
            allow_missing_timesteps=False, # We probably want to avoid this
            categorical_encoders={group_id_col: NaNLabelEncoder(add_nan=True)},
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
    