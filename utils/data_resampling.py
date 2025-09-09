import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class DataResampler:
    def __init__(self, input_dir="./data/long-tr-data", output_dir="./data/resampled"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.training_files = os.listdir(input_dir)
    
    def _extract_session_timestamp(self, session_id):
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

    def process_files(self, verbose=True):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        iterator = (
            tqdm(self.training_files, desc="Processing files")
            if verbose
            else self.training_files
        )
        for file_name in iterator:
            if verbose:
                tqdm.write(f"Processing file: {file_name}")
            file_path = os.path.join(self.input_dir, file_name)
            df = pd.read_csv(file_path)

            # Resample to convert to a distance domain time series
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Fill missing values systematically
            df["heartRate"] = (
                df["heartRate"].interpolate().ffill().fillna(df["heartRate"].mean())
            )
            df["distance"] = df["distance"].interpolate().ffill().bfill()
            df["speed"] = df["speed"].interpolate().ffill().bfill()
            df["cadence"] = df["cadence"].interpolate().ffill().bfill()
            df["altitude"] = df["altitude"].interpolate().ffill().bfill()
            df["temperature"] = (
                df["temperature"].interpolate().ffill().fillna(df["temperature"].mean())
            )
            df["duration"] = df["duration"].interpolate().ffill().bfill()

            # Resample to 2 meter intervals (distance is not a time index, so use reindex)
            # Remove duplicate distances (keep first)
            df = df.drop_duplicates(subset='distance', keep='first')
            # Define resampling grid
            min_dist = np.floor(df['distance'].min())
            max_dist = np.ceil(df['distance'].max())
            target_distances = np.arange(min_dist, max_dist + 2, 2)  # +2 to include endpoint
            # Set index, reindex, interpolate
            df = (
                df.set_index('distance')
                .reindex(target_distances)
                .interpolate(method='linear')
                .reset_index()
                .rename(columns={'index': 'distance'})
            )

            # Calculate elevation difference
            df["elevation_diff"] = df["altitude"].diff().fillna(0)

            # Calculate elevation gain
            df["elevation_gain"] = (
                df["elevation_diff"].clip(lower=0).cumsum().fillna(0)
            )

            # Calculate elevation loss
            df["elevation_loss"] = (
                df["elevation_diff"].clip(upper=0).cumsum().fillna(0)
            )

            # Final check: ensure no NaN values remain after resampling
            # Forward fill any remaining NaN values introduced by resampling
            df = df.ffill()
            # Backward fill any NaN values at the beginning
            df = df.bfill()

            # Add session identifier based on file name
            df['session_id'] = file_name.replace('.csv', '')
            
            # Create sequential time indices for TimeSeriesDataSet
            df = df.sort_values('distance').reset_index(drop=True)
            df['time_idx'] = range(len(df))
            
            # Extract timestamp-based integer ID from session name for encoding
            df['session_id_encoded'] = self._extract_session_timestamp(df['session_id'].iloc[0])

            # Calculate distance difference
            df["distance_diff"] = df["distance"].diff().fillna(0)

            # Output the resampled file
            output_path = os.path.join(self.output_dir, file_name)
            df.to_csv(output_path, index=False)
            if verbose:
                tqdm.write(f"Resampled file saved to: {output_path}")


if __name__ == "__main__":
    input_dir = "./data/long-tr-data"
    output_dir = "./data/resampled"
    resampler = DataResampler(input_dir, output_dir)
    resampler.process_files(verbose=True)
