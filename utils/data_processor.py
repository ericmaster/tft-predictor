"""
Data Processor

A library class for processing trail running exercise data from JSON files
and converting them to structured CSV format with features extraction.
"""

import traceback
import pandas as pd
import os
import json
import isodate
import time
import sys
from tqdm import tqdm
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataProcessor:
    """
    A class for processing trail running exercise data from JSON files.

    This class handles:
    - Filtering exercises by sport type and duration
    - Extracting and processing sensor data (heart rate, altitude, distance, etc.)
    - Feature engineering (distance differences, elevation gain/loss, etc.)
    - Batch processing with progress tracking
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sport_type: str = "TRAIL_RUNNING",
        min_duration_hours: float = 3.0,
        max_duration_hours: float = 6.0,
        min_year: int = 2020,
        sample_features: Optional[List[str]] = None,
    ):
        """
        Initialize the DataProcessor.

        Args:
            input_dir (str): Directory containing input JSON files
            output_dir (str): Directory to save processed CSV files
            sport_type (str): Sport type to filter (default: "TRAIL_RUNNING")
            min_duration_hours (float): Minimum exercise duration in hours
            max_duration_hours (float): Maximum exercise duration in hours
            min_year (int): Minimum year to include in processing (default: 2020)
            sample_features (List[str], optional): List of features to extract
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sport_type = sport_type
        self.min_duration_hours = min_duration_hours
        self.max_duration_hours = max_duration_hours
        self.min_year = min_year

        if sample_features is None:
            self.sample_features = [
                "heartRate",
                "altitude",
                "distance",
                "temperature",
                "cadence",
                "speed",
            ]
        else:
            self.sample_features = sample_features

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract year from filename with format training-session-YYYY-MM-DD-...
        or training-target-YYYY-MM-DD-...

        Args:
            filename (str): Name of the file

        Returns:
            int: Year extracted from filename, or None if extraction fails
        """
        try:
            # Split by '-' and find the year part (should be the 3rd element: training-session-YYYY or training-target-YYYY)
            parts = filename.split("-")
            if len(parts) >= 3:
                year_str = parts[2]
                return int(year_str)
        except (ValueError, IndexError):
            pass
        return None

    def _should_process_file(self, filename: str) -> bool:
        """
        Check if a file should be processed based on filename prefix and year filtering.

        Args:
            filename (str): Name of the file to check

        Returns:
            bool: True if file should be processed
        """
        # Check if filename starts with the expected prefix
        if not filename.startswith("training-session-"):
            return False

        year = self._extract_year_from_filename(filename)
        if year is None:
            return False
        return year >= self.min_year

    def _should_process_exercise(self, exercise: dict) -> bool:
        """
        Check if an exercise meets the filtering criteria.

        Args:
            exercise (dict): Exercise data from JSON

        Returns:
            bool: True if exercise should be processed
        """
        sport = exercise.get("sport")
        duration_iso = exercise.get("duration")

        if not duration_iso:
            return False

        duration = isodate.parse_duration(duration_iso).total_seconds()
        min_duration_seconds = self.min_duration_hours * 3600
        max_duration_seconds = self.max_duration_hours * 3600

        return (
            sport == self.sport_type
            and duration >= min_duration_seconds
            and duration <= max_duration_seconds
        )

    def _extract_samples_dataframe(self, samples: dict, feature: str) -> pd.DataFrame:
        """
        Extract samples for a specific feature into a DataFrame.

        Args:
            samples (dict): Samples data from exercise
            feature (str): Feature name to extract

        Returns:
            pd.DataFrame: DataFrame with timestamp and feature data
        """
        sample_data = samples.get(feature, [])
        if not sample_data:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(sample["dateTime"]),
                    feature: sample["value"] if "value" in sample else None,
                }
                for sample in sample_data
            ]
        )

    def _process_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process distance-related features.

        Args:
            df (pd.DataFrame): DataFrame with distance data

        Returns:
            pd.DataFrame: DataFrame with distance difference calculated
        """
        df["distance_diff"] = df["distance"].diff().fillna(0)
        return df

    def _process_altitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process altitude-related features including smoothing and elevation calculations.

        Args:
            df (pd.DataFrame): DataFrame with altitude data

        Returns:
            pd.DataFrame: DataFrame with processed altitude features
        """
        # Smooth altitude data
        df["altitude"] = (
            df["altitude"].rolling(window=5, min_periods=1, center=True).mean()
        )

        # Calculate elevation difference
        # df["elevation_diff"] = df["altitude"].diff().fillna(0)

        # Calculate elevation gain
        # df["elevation_gain"] = (
        #     df["elevation_diff"].clip(lower=0).cumsum().fillna(0)
        # )

        # Calculate elevation loss
        # df["elevation_loss"] = (
        #     df["elevation_diff"].clip(upper=0).cumsum().fillna(0)
        # )

        return df

    def process_single_file(self, file_name: str, verbose: bool = True) -> bool:
        """
        Process a single JSON file and save as CSV.

        Args:
            file_name (str): Name of the file to process
            verbose (bool): Whether to print processing information

        Returns:
            bool: True if file was processed successfully
        """
        # Check if file should be processed based on year filtering
        if not self._should_process_file(file_name):
            if verbose:
                print(f"Skipping {file_name}. Not a training session or older than 2020.")
            return False

        file_path = os.path.join(self.input_dir, file_name)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            exercises = data.get("exercises", [])
            if not exercises:
                if verbose:
                    print(f"No exercises found in {file_name}")
                return False
            exercise = exercises[0]

            # Check if exercise meets filtering criteria
            if not self._should_process_exercise(exercise):
                return False

            # Skip processing if the output file already exists
            output_file_name = os.path.join(
                self.output_dir, file_name.replace(".json", ".csv")
            )
            if os.path.exists(output_file_name):
                return False

            if verbose:
                print(f"Processing file: {file_name}")

            start_time = time.time()
            samples = exercise.get("samples", {})

            # Initialize main dataframe with heart rate samples
            df = self._extract_samples_dataframe(samples, "heartRate")

            if df.empty:
                if verbose:
                    print(f"No heart rate data found in {file_name}")
                return False

            # Add date column
            # df["date"] = df["timestamp"].dt.date

            # Calculate cumulative duration
            df["duration"] = (
                df["timestamp"].diff().dt.total_seconds().cumsum().fillna(0)
            )

            # Process and merge other sample types
            for sample_feature in self.sample_features[
                1:
            ]:  # Skip heartRate (already in df)
                temp_df = self._extract_samples_dataframe(samples, sample_feature)

                if temp_df.empty:
                    continue

                # Apply feature-specific processing
                if sample_feature == "distance":
                    temp_df = self._process_distance_features(temp_df)
                elif sample_feature == "altitude":
                    temp_df = self._process_altitude_features(temp_df)

                # Forward fill within the same feature to handle duplicates
                if sample_feature in df.columns:
                    temp_df[sample_feature] = temp_df[sample_feature].fillna(
                        method="ffill"
                    )

                # Merge with main dataframe
                df = pd.merge(df, temp_df, on="timestamp", how="left")

            # Save the dataframe to a CSV file
            df.to_csv(output_file_name, index=False)

            if verbose:
                end_time = time.time()
                print(f"Saved processed data to: {output_file_name}")
                print(
                    f"Processing time for {file_name}: {end_time - start_time:.2f} seconds"
                )

            return True

        except FileNotFoundError:
            if verbose:
                print(f"Error: File not found at path: {file_name}")
        except json.JSONDecodeError:
            if verbose:
                print(f"Error: Invalid JSON format in file: {file_name}")
        except Exception as e:
            if verbose:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"An unexpected error occurred in file: {file_name}")
                print(f"With error {exc_type}: {exc_value}")
                print("Traceback:")
                traceback.print_exception(exc_type, exc_value, exc_traceback)

        return False

    def process_all_files(
        self,
        max_workers: Optional[int] = None,
        use_parallel: bool = False,
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """
        Process all JSON files in the input directory.

        Args:
            max_workers (int, optional): Maximum number of parallel workers
            use_parallel (bool): Whether to use parallel processing

        Returns:
            Tuple[int, int]: (processed_count, skipped_count)
        """
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        processed_count = 0
        skipped_count = 0

        if use_parallel and max_workers:
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_file, file_name, verbose=verbose
                    ): file_name
                    for file_name in files
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing files"
                ):
                    if future.result():
                        processed_count += 1
                    else:
                        skipped_count += 1
        else:
            # Process files sequentially
            for file_name in tqdm(files, desc="Processing files"):
                if self.process_single_file(file_name):
                    processed_count += 1
                else:
                    skipped_count += 1

        print(f"Processed {processed_count} files.")
        print(f"Skipped {skipped_count} files.")

        return processed_count, skipped_count

    def get_processed_files(self) -> List[str]:
        """
        Get list of already processed CSV files.

        Returns:
            List[str]: List of processed CSV file names
        """
        if not os.path.exists(self.output_dir):
            return []

        return [f for f in os.listdir(self.output_dir) if f.endswith(".csv")]

    def get_processing_stats(self) -> dict:
        """
        Get statistics about the processing status.

        Returns:
            dict: Dictionary with processing statistics
        """
        total_json_files = len(
            [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        )
        valid_year_files = len(
            [
                f
                for f in os.listdir(self.input_dir)
                if f.endswith(".json") and self._should_process_file(f)
            ]
        )
        processed_csv_files = len(self.get_processed_files())

        return {
            "total_input_files": total_json_files,
            "files_after_year_filter": valid_year_files,
            "processed_files": processed_csv_files,
            "remaining_files": valid_year_files - processed_csv_files,
            "input_directory": self.input_dir,
            "output_directory": self.output_dir,
            "sport_type": self.sport_type,
            "duration_range_hours": (self.min_duration_hours, self.max_duration_hours),
            "min_year": self.min_year,
            "sample_features": self.sample_features,
        }


if __name__ == "__main__":
    input_dir = "./data/full-data"
    output_dir = "./data/long-tr-data"

    data_processor = DataProcessor(input_dir, output_dir)
    processed_count, skipped_count = data_processor.process_all_files(
        use_parallel=False, max_workers=4, verbose=True
    )
    print(f"Processed {processed_count} files, skipped {skipped_count} files.")
