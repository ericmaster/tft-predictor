import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class DataResampler:
    def __init__(self, input_dir="./data/long-tr-data", output_dir='./data/resampled'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.training_files = os.listdir(input_dir)

    def process_files(self, verbose=True):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        iterator = tqdm(self.training_files, desc="Processing files") if verbose else self.training_files
        for file_name in iterator:
            if verbose:
                tqdm.write(f"Processing file: {file_name}")
            file_path = os.path.join(self.input_dir, file_name)
            df = pd.read_csv(file_path)

            # Resample to convert to a distance domain time series
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Fill heart rate NaN values with the mean of the column
            df["heartRate"] = df["heartRate"].fillna(df["heartRate"].mean())
            # Fill other NaN values with the closest values of the column or 0
            df["cadence"] = df["cadence"].ffill()
            df["speed"] = df["speed"].ffill()
            df["distance"] = df["distance"].ffill()
            df["cadence"] = df["cadence"].fillna(0)
            df["speed"] = df["speed"].fillna(0)
            df["distance"] = df["distance"].fillna(0)

            # Resample to 2 meter intervals (distance is not a time index, so use reindex)
            df.set_index('distance', inplace=True)
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            # Create new index for every 2 meters
            min_dist = int(df.index.min())
            max_dist = int(df.index.max())
            new_index = np.arange(min_dist, max_dist + 1, 2)
            df = df.reindex(new_index).interpolate()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'distance'}, inplace=True)

            # Output the resampled file
            output_path = os.path.join(self.output_dir, file_name)
            df.to_csv(output_path, index=False)
            if verbose:
                tqdm.write(f"Resampled file saved to: {output_path}")
            
if __name__ == "__main__":
    input_dir = './data/long-tr-data'
    output_dir = './data/resampled'
    resampler = DataResampler(input_dir, output_dir)
    resampler.process_files(verbose=True)
