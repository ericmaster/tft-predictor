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
        min_avg_speed: float = 3.0,
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
            min_avg_speed (float): Minimum average speed in m/s to exclude outliers (default: 3.0)
            sample_features (List[str], optional): List of features to extract
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sport_type = sport_type
        self.min_duration_hours = min_duration_hours
        self.max_duration_hours = max_duration_hours
        self.min_year = min_year
        self.min_avg_speed = min_avg_speed

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
        
        # Metadata file path
        self.metadata_file = os.path.join(self.output_dir, "sessions_metadata.csv")

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
        # Check if file has .json extension
        if not filename.endswith(".json"):
            return False
        
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

        # Get average speed from exercise data
        avg_speed = exercise.get("speed", {}).get("avg")
        
        # Filter out sessions with average speed below minimum threshold
        # This excludes outliers caused by exceptional conditions (injuries, digestive issues, etc.)
        if avg_speed is not None and avg_speed < self.min_avg_speed:
            return False

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

    def _extract_session_metadata(self, exercise: dict, file_name: str, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Extract session metadata from exercise data.
        If altitude metadata is missing and DataFrame is provided, calculate from samples.

        Args:
            exercise (dict): Exercise data from JSON
            file_name (str): Name of the source file
            df (pd.DataFrame, optional): Processed DataFrame with sample data

        Returns:
            dict: Dictionary containing session metadata
        """
        # Parse duration
        duration_iso = exercise.get("duration")
        duration_seconds = isodate.parse_duration(duration_iso).total_seconds() if duration_iso else 0
        
        # Extract altitude metadata from exercise
        avg_altitude = exercise.get("altitude", {}).get("avg")
        min_altitude = exercise.get("altitude", {}).get("min")
        max_altitude = exercise.get("altitude", {}).get("max")
        
        # If altitude metadata is missing and DataFrame is provided, calculate from samples
        if df is not None and "altitude" in df.columns:
            altitude_data = df["altitude"].dropna()
            if len(altitude_data) > 0:
                if avg_altitude is None:
                    avg_altitude = float(altitude_data.mean())
                if min_altitude is None:
                    min_altitude = float(altitude_data.min())
                if max_altitude is None:
                    max_altitude = float(altitude_data.max())
        
        return {
            "file_name": file_name,
            "start_time": exercise.get("startTime"),
            "stop_time": exercise.get("stopTime"),
            "duration_seconds": duration_seconds,
            "distance_meters": exercise.get("distance"),
            "ascent_meters": exercise.get("ascent"),
            "descent_meters": exercise.get("descent"),
            "kilo_calories": exercise.get("kiloCalories"),
            "avg_altitude": avg_altitude,
            "min_altitude": min_altitude,
            "max_altitude": max_altitude,
            "avg_cadence": exercise.get("cadence", {}).get("avg"),
            "max_cadence": exercise.get("cadence", {}).get("max"),
            "avg_heart_rate": exercise.get("heartRate", {}).get("avg"),
            "min_heart_rate": exercise.get("heartRate", {}).get("min"),
            "max_heart_rate": exercise.get("heartRate", {}).get("max"),
            "avg_power": exercise.get("power", {}).get("avg"),
            "max_power": exercise.get("power", {}).get("max"),
            "avg_speed": exercise.get("speed", {}).get("avg"),
            "max_speed": exercise.get("speed", {}).get("max"),
            "sport": exercise.get("sport"),
            "latitude": exercise.get("latitude"),
            "longitude": exercise.get("longitude"),
        }

    def _append_metadata(self, metadata: dict) -> None:
        """
        Append session metadata to the metadata CSV file.

        Args:
            metadata (dict): Session metadata to append
        """
        metadata_df = pd.DataFrame([metadata])
        
        # Check if metadata file exists
        if os.path.exists(self.metadata_file):
            # Append to existing file
            metadata_df.to_csv(self.metadata_file, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            metadata_df.to_csv(self.metadata_file, mode='w', header=True, index=False)

    def _process_altitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process altitude-related features including smoothing and elevation calculations.

        Args:
            df (pd.DataFrame): DataFrame with altitude data

        Returns:
            pd.DataFrame: DataFrame with processed altitude features
        """

        # Copy altitude data
        df["altitude_raw"] = df["altitude"].copy()

        # Check for sudden spikes and correct them
        altitude_diff = df["altitude"].diff().fillna(0)
        spike_threshold = 5
        spike_indices = altitude_diff[altitude_diff.abs() > spike_threshold].index

        if len(spike_indices) > 0:
            print(f"  Detected {len(spike_indices)} sudden altitude spikes, correcting...")
            
            for idx in spike_indices:
                idx_pos = df.index.get_loc(idx)
                
                # Get previous and next values
                prev_val = df["altitude"].iloc[idx_pos - 1] if idx_pos > 0 else df["altitude"].iloc[idx_pos]
                next_val = df["altitude"].iloc[idx_pos + 1] if idx_pos < len(df) - 1 else df["altitude"].iloc[idx_pos]
                
                # Calculate average of neighbors
                avg_neighbors = (prev_val + next_val) / 2
                
                # Current value
                current_val = df.loc[idx, "altitude"]
                
                # Replace with the minimum between spike_threshold and the difference to average
                if abs(current_val - avg_neighbors) > spike_threshold:
                    # Determine direction of correction
                    if current_val > avg_neighbors:
                        df.loc[idx, "altitude"] = avg_neighbors + spike_threshold
                    else:
                        df.loc[idx, "altitude"] = avg_neighbors - spike_threshold
                else:
                    df.loc[idx, "altitude"] = avg_neighbors

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
            # if verbose:
            #     print(f"Skipping {file_name}. Not a training session or older than 2020.")
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

            # Process and merge other sample types
            for sample_feature in self.sample_features[
                1:
            ]:  # Skip heartRate (already in df)
                temp_df = self._extract_samples_dataframe(samples, sample_feature)

                if temp_df.empty:
                    if verbose:
                        print(f"No {sample_feature} data found in {file_name}, skipping this feature.")
                    continue

                # Apply feature-specific processing
                if sample_feature == "altitude":
                    temp_df = self._process_altitude_features(temp_df)

                # Forward fill within the same feature to handle duplicates
                if sample_feature in df.columns:
                    temp_df[sample_feature] = temp_df[sample_feature]
                    # .fillna(
                    #     method="ffill"
                    # )

                # Merge with main dataframe
                df = pd.merge(df, temp_df, on="timestamp", how="left")
                
                # Check for NA values after merge
                if verbose and sample_feature in df.columns:
                    na_count = df[sample_feature].isna().sum()
                    total_count = len(df)
                    if na_count > 0:
                        na_percentage = (na_count / total_count) * 100
                        print(f"  {sample_feature}: {na_count}/{total_count} NA values ({na_percentage:.2f}%)")

            # Calculate cumulative duration
            df["duration"] = (
                df["timestamp"].diff().dt.total_seconds().cumsum().fillna(method="bfill")
            )
            # Check if there is any NA in duration
            if verbose:
                na_count = df["duration"].isna().sum()
                total_count = len(df)
                if na_count > 0:
                    na_percentage = (na_count / total_count) * 100
                    print(f"  duration: {na_count}/{total_count} NA values ({na_percentage:.2f}%)")

            # Save the dataframe to a CSV file
            df.to_csv(output_file_name, index=False)
            
            # Extract and save session metadata (pass df for altitude calculation if needed)
            metadata = self._extract_session_metadata(exercise, file_name, df)
            self._append_metadata(metadata)

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

    def process_files(
        self,
        max_workers: Optional[int] = None,
        use_parallel: bool = False,
        verbose: bool = False,
        limit: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Process all JSON files in the input directory.

        Args:
            max_workers (int, optional): Maximum number of parallel workers
            use_parallel (bool): Whether to use parallel processing
            verbose (bool): Whether to print processing information
            limit (int, optional): Maximum number of files to process. If None, process all files

        Returns:
            Tuple[int, int]: (processed_count, skipped_count)
        """
        files = [
            f for f in os.listdir(self.input_dir) 
            if self._should_process_file(f)
        ]
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
                    # Apply limit if specified
                    if limit is not None and limit > 0 and processed_count >= limit:
                        print("Reached processing limit.")
                        break
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
                if self._should_process_file(f)
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
            "min_avg_speed": self.min_avg_speed,
            "sample_features": self.sample_features,
        }


if __name__ == "__main__":
    input_dir = "./data/full-data"
    output_dir = "./data/long-tr-data"

    limit=None  # Set to an integer to limit number of files processed
    if limit is not None:
        print(f"Limiting processing to {limit} files.")

    data_processor = DataProcessor(input_dir, output_dir)
    processed_count, skipped_count = data_processor.process_files(
        use_parallel=False, max_workers=4, verbose=True, limit=limit
    )
    print(f"Processed {processed_count} files, skipped {skipped_count} files.")
