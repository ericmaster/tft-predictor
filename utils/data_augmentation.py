import pandas as pd
import numpy as np
from typing import List, Tuple
import random


class TimeSeriesAugmenter:
    """Data augmentation techniques for time series data to reduce overfitting."""
    
    def __init__(self, noise_std: float = 0.01, jitter_std: float = 0.03):
        """
        Initialize the augmenter.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            jitter_std: Standard deviation for jittering
        """
        self.noise_std = noise_std
        self.jitter_std = jitter_std
    
    def add_noise(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add Gaussian noise to specified columns."""
        augmented = data.copy()
        for col in columns:
            if col in data.columns:
                noise = np.random.normal(0, self.noise_std * data[col].std(), len(data))
                augmented[col] = data[col] + noise
        return augmented
    
    def jitter(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply jittering (small random perturbations) to specified columns."""
        augmented = data.copy()
        for col in columns:
            if col in data.columns:
                jitter = np.random.normal(0, self.jitter_std * data[col].std(), len(data))
                augmented[col] = data[col] + jitter
        return augmented
    
    def scaling(self, data: pd.DataFrame, columns: List[str], scale_range: Tuple[float, float] = (0.95, 1.05)) -> pd.DataFrame:
        """Apply random scaling to specified columns."""
        augmented = data.copy()
        for col in columns:
            if col in data.columns:
                scale_factor = np.random.uniform(scale_range[0], scale_range[1])
                augmented[col] = data[col] * scale_factor
        return augmented
    
    def time_warping(self, data: pd.DataFrame, warp_strength: float = 0.1) -> pd.DataFrame:
        """Apply time warping by slightly stretching/compressing time intervals."""
        augmented = data.copy()
        n_points = len(data)
        
        # Create warping function
        warp_steps = np.random.normal(1.0, warp_strength, n_points)
        warp_steps = np.clip(warp_steps, 0.5, 1.5)  # Limit warping
        warp_steps = np.cumsum(warp_steps)
        warp_steps = warp_steps / warp_steps[-1] * (n_points - 1)
        
        # Apply warping to time-varying columns
        time_varying_cols = ['duration', 'heartRate', 'temperature', 'cadence', 'speed']
        for col in time_varying_cols:
            if col in data.columns:
                augmented[col] = np.interp(np.arange(n_points), warp_steps, data[col])
        
        return augmented
    
    def subsample(self, data: pd.DataFrame, subsample_ratio: float = 0.9) -> pd.DataFrame:
        """Randomly subsample the time series."""
        n_samples = int(len(data) * subsample_ratio)
        if n_samples < len(data) * 0.5:  # Don't subsample too aggressively
            n_samples = int(len(data) * 0.5)
        
        indices = sorted(random.sample(range(len(data)), n_samples))
        return data.iloc[indices].reset_index(drop=True)
    
    def augment_session(self, data: pd.DataFrame, augmentation_factor: int = 2) -> List[pd.DataFrame]:
        """
        Apply multiple augmentation techniques to create multiple versions of a session.
        
        Args:
            data: Original session data
            augmentation_factor: Number of augmented versions to create
            
        Returns:
            List of augmented DataFrames including the original
        """
        augmented_sessions = [data.copy()]  # Include original
        
        # Columns to augment (exclude distance-based features as they're route-dependent)
        sensor_columns = ['heartRate', 'temperature', 'cadence', 'speed']
        duration_columns = ['duration']
        
        for i in range(augmentation_factor):
            augmented = data.copy()
            
            # Apply random combination of augmentations
            augmentation_techniques = random.sample([
                lambda x: self.add_noise(x, sensor_columns),
                lambda x: self.jitter(x, sensor_columns + duration_columns),
                lambda x: self.scaling(x, sensor_columns, (0.98, 1.02)),
                lambda x: self.time_warping(x, 0.05),
                lambda x: self.subsample(x, 0.95)
            ], k=random.randint(1, 3))  # Apply 1-3 techniques randomly
            
            for technique in augmentation_techniques:
                augmented = technique(augmented)
            
            # Ensure session_id uniqueness
            if 'session_id' in augmented.columns:
                augmented['session_id'] = f"{data['session_id'].iloc[0]}_aug_{i+1}"
            
            # Recalculate time_idx after modifications
            augmented['time_idx'] = range(len(augmented))
            
            augmented_sessions.append(augmented)
        
        return augmented_sessions


def augment_training_data(data_dir: str, output_dir: str, augmentation_factor: int = 1):
    """
    Augment all training data and save to output directory.
    
    Args:
        data_dir: Directory containing original training data
        output_dir: Directory to save augmented data
        augmentation_factor: Number of augmented versions per original session
    """
    import os
    
    augmenter = TimeSeriesAugmenter()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        print(f"Augmenting {file}...")
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path)
        
        # Generate augmented versions
        augmented_sessions = augmenter.augment_session(data, augmentation_factor)
        
        # Save original and augmented versions
        for idx, session in enumerate(augmented_sessions):
            if idx == 0:
                # Original file
                output_path = os.path.join(output_dir, file)
            else:
                # Augmented file
                base_name = file.replace('.csv', '')
                output_path = os.path.join(output_dir, f"{base_name}_aug_{idx}.csv")
            
            session.to_csv(output_path, index=False)
    
    print(f"Augmentation complete! Original files: {len(csv_files)}, "
          f"Total files: {len(csv_files) * (augmentation_factor + 1)}")


if __name__ == "__main__":
    # Example usage
    augment_training_data(
        data_dir="./data/resampled",
        output_dir="./data/augmented",
        augmentation_factor=1  # Create 1 additional version per session
    )