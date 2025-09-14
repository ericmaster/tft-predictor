# from lib.data import create_sliding_windows
import importlib
# Reload TFTDataModule
import lib.data
importlib.reload(lib.data)
from lib.data import create_sliding_windows

# Create sliding window chunks from the start of data sequences for full session inference
print("Creating sliding window chunks from the beginning of sessions...")

# Access the raw test data to create sliding windows
test_data = data_module.test_data  # Raw DataFrame

print(f"Raw test data shape: {test_data.shape}")
print(f"Test sessions: {test_data['session_id'].nunique()}")

# Create sliding window chunks
print("Creating sliding window chunks...")
sliding_chunks = create_sliding_windows(
    test_data, 
    'session_id', 
    data_module.max_encoder_length, 
    data_module.max_prediction_length,
    step_size=200  # Move 200 steps (1000m) at a time
)

print(f"Created {len(sliding_chunks)} sliding window chunks")

# Display chunk information
if len(sliding_chunks) > 0:
    print(f"\nChunk details:")
    print(f"Chunks per session:")
    chunks_per_session = {}
    for chunk in sliding_chunks:
        session_id = chunk['session_id']
        chunks_per_session[session_id] = chunks_per_session.get(session_id, 0) + 1
    
    for session_id, count in chunks_per_session.items():
        print(f"  Session {session_id}: {count} chunks")
    
    # Show details of first few chunks
    print(f"\nFirst 3 chunks:")
    for i, chunk in enumerate(sliding_chunks[:3]):
        print(f"  Chunk {i}: Session {chunk['session_id']}, Distance {chunk['start_distance']:.0f}m - {chunk['end_distance']:.0f}m")

# Get the first chunk (from beginning of first session)
first_chunk = sliding_chunks[0]
chunk_data = first_chunk['data']

print(f"First chunk details:")
print(f"  Session: {first_chunk['session_id']}")
print(f"  Distance range: {first_chunk['start_distance']:.0f}m - {first_chunk['end_distance']:.0f}m") 
print(f"  Data shape: {chunk_data.shape}")
print(f"  Encoder length: {first_chunk['encoder_length']}")
print(f"  Prediction length: {first_chunk['prediction_length']}")

# Prepare data for prediction - we need to create a proper TimeSeriesDataSet sample
# Extract encoder and decoder portions
encoder_data = chunk_data.iloc[:first_chunk['encoder_length']].copy()
decoder_data = chunk_data.iloc[first_chunk['encoder_length']:].copy()

print(f"  Encoder data: {encoder_data.shape[0]} steps")
print(f"  Decoder data: {decoder_data.shape[0]} steps")

# Get the feature columns (continuous variables)
feature_cols = [col for col in chunk_data.columns if col not in ['session_id', 'session_id_encoded', 'time_idx']]
target_cols = data_module.target_names

print(f"  Feature columns: {len(feature_cols)}")
print(f"  Target columns: {target_cols}")

# For now, let's just show what data we have
print(f"\nEncoder data sample (first 5 rows):")
print(encoder_data[target_cols].head())

print(f"\nDecoder data sample (first 5 rows):")  
print(decoder_data[target_cols].head())

# Let's manually extract features and make predictions using the model's forward pass
import torch
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE

# Function to prepare data for manual prediction
def prepare_chunk_for_manual_prediction(chunk):
    """Prepare chunk data for manual model prediction"""
    chunk_data = chunk['data']
    encoder_length = chunk['encoder_length']
    
    # Get encoder data (input sequence)
    encoder_data = chunk_data.iloc[:encoder_length]
    
    # Select the target columns that the model was trained on
    target_cols = ["duration", "heartRate", "temperature", "cadence", "speed"]
    features = encoder_data[target_cols].values
    
    # Convert to tensor and add batch dimension
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Shape: (1, 200, 5)
    
    return features_tensor, encoder_data

# Test manual prediction on first chunk
chunk = sliding_chunks[0]
distance_range = f"{chunk['start_distance']:.0f}m - {chunk['end_distance']:.0f}m"
print(f"Testing manual prediction on chunk 0: {distance_range}")

features_tensor, encoder_data = prepare_chunk_for_manual_prediction(chunk)
print(f"Features tensor shape: {features_tensor.shape}")
print(f"Encoder data shape: {encoder_data.shape}")

# Get decoder data for comparison
decoder_data = chunk['data'].iloc[200:400]
actual_targets = decoder_data[["duration", "heartRate", "temperature", "cadence", "speed"]].values

print(f"Actual targets shape: {actual_targets.shape}")

# Sample of encoder data
print(f"\nEncoder data sample (first 5 steps):")
print(encoder_data[["duration", "heartRate", "temperature", "cadence", "speed"]].head())

# Prepare the chunk data for TimeSeriesDataSet
chunk_data = chunk['data'].copy()

# Ensure we have all required columns for TimeSeriesDataSet
print(f"Chunk data columns: {list(chunk_data.columns)}")
print(f"Chunk data shape: {chunk_data.shape}")

# Create TimeSeriesDataSet from the chunk
# Note: Use from_dataset to inherit all configurations from training dataset
chunk_dataset = TimeSeriesDataSet.from_dataset(
    data_module.training,  # Use training dataset as template
    chunk_data,            # New data for prediction
    predict=True,          # Set to prediction mode
    stop_randomization=True  # Ensure deterministic behavior for inference
)

print(f"Created TimeSeriesDataSet with {len(chunk_dataset)} samples")

# Create dataloader from the dataset
chunk_dataloader = chunk_dataset.to_dataloader(
    train=False, 
    batch_size=1, 
    num_workers=0
)

print(f"Created DataLoader with batch_size=1")

# Make predictions using the model's predict method
print(f"Making predictions on chunk using model.predict()...")
chunk_predictions = model.predict(chunk_dataloader, return_y=True, return_index=True)

print(f"Predictions completed!")
print(f"Prediction type: {type(chunk_predictions)}")
print(f"Prediction shape: {len(chunk_predictions.output)}")
print(f"RMSE: {RMSE()(chunk_predictions.output[0], chunk_predictions.y[0][0])}")

raw_predictions = model.predict(
    chunk_dataloader, mode="raw", return_x=True
)

model.plot_prediction(
    raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=False, plot_attention=False
)

# # Progressive inference visualization across multiple chunks
# import matplotlib.pyplot as plt
# import numpy as np

# # Process multiple chunks from the same session to show progression
# session_chunks = [chunk for chunk in sliding_chunks if chunk['session_id'] == sliding_chunks[0]['session_id']]
# print(f"Found {len(session_chunks)} chunks from session: {session_chunks[0]['session_id'][:50]}...")

# # Select chunks at different progression points (every 10th chunk to show progression)
# selected_chunks = session_chunks[::10][:6]  # Take every 10th chunk, max 6 chunks
# print(f"Selected {len(selected_chunks)} chunks for progressive analysis")

# # Analyze progression through the session
# chunk_analysis = []

# for i, chunk in enumerate(selected_chunks):
#     distance_range = f"{chunk['start_distance']:.0f}m - {chunk['end_distance']:.0f}m"
#     chunk_data = chunk['data']
    
#     # Get encoder and decoder data
#     encoder_data = chunk_data.iloc[:200]
#     decoder_data = chunk_data.iloc[200:400]
    
#     # Calculate some statistics for this chunk
#     analysis = {
#         'chunk_idx': i,
#         'distance_range': distance_range,
#         'start_distance': chunk['start_distance'],
#         'end_distance': chunk['end_distance'],
#         'encoder_stats': {
#             'avg_duration': encoder_data['duration'].mean(),
#             'avg_heartrate': encoder_data['heartRate'].mean(),
#             'avg_speed': encoder_data['speed'].mean(),
#             'avg_cadence': encoder_data['cadence'].mean(),
#         },
#         'decoder_stats': {
#             'avg_duration': decoder_data['duration'].mean(),
#             'avg_heartrate': decoder_data['heartRate'].mean(),
#             'avg_speed': decoder_data['speed'].mean(),
#             'avg_cadence': decoder_data['cadence'].mean(),
#         }
#     }
#     chunk_analysis.append(analysis)
    
#     print(f"Chunk {i}: {distance_range}")
#     print(f"  Encoder avg: Duration={analysis['encoder_stats']['avg_duration']:.1f}s, HR={analysis['encoder_stats']['avg_heartrate']:.1f}bpm, Speed={analysis['encoder_stats']['avg_speed']:.2f}m/s")
#     print(f"  Decoder avg: Duration={analysis['decoder_stats']['avg_duration']:.1f}s, HR={analysis['decoder_stats']['avg_heartrate']:.1f}bpm, Speed={analysis['decoder_stats']['avg_speed']:.2f}m/s")

# # Create visualization showing progression through the session
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('Progressive Analysis: How Trail Running Metrics Change Throughout Session', fontsize=16)

# # Extract data for plotting
# distances = [chunk['start_distance'] for chunk in chunk_analysis]
# encoder_hr = [chunk['encoder_stats']['avg_heartrate'] for chunk in chunk_analysis]
# decoder_hr = [chunk['decoder_stats']['avg_heartrate'] for chunk in chunk_analysis]
# encoder_speed = [chunk['encoder_stats']['avg_speed'] for chunk in chunk_analysis]
# decoder_speed = [chunk['decoder_stats']['avg_speed'] for chunk in chunk_analysis]
# encoder_duration = [chunk['encoder_stats']['avg_duration'] for chunk in chunk_analysis]
# decoder_duration = [chunk['decoder_stats']['avg_duration'] for chunk in chunk_analysis]
# encoder_cadence = [chunk['encoder_stats']['avg_cadence'] for chunk in chunk_analysis]
# decoder_cadence = [chunk['decoder_stats']['avg_cadence'] for chunk in chunk_analysis]

# # Plot 1: Heart Rate progression
# axes[0, 0].plot(distances, encoder_hr, 'b-o', label='Encoder (Past 200 steps)', markersize=6)
# axes[0, 0].plot(distances, decoder_hr, 'r-s', label='Decoder (Next 200 steps)', markersize=6)
# axes[0, 0].set_title('Heart Rate Progression')
# axes[0, 0].set_xlabel('Distance (m)')
# axes[0, 0].set_ylabel('Heart Rate (bpm)')
# axes[0, 0].legend()
# axes[0, 0].grid(True, alpha=0.3)

# # Plot 2: Speed progression
# axes[0, 1].plot(distances, encoder_speed, 'b-o', label='Encoder (Past 200 steps)', markersize=6)
# axes[0, 1].plot(distances, decoder_speed, 'r-s', label='Decoder (Next 200 steps)', markersize=6)
# axes[0, 1].set_title('Speed Progression')
# axes[0, 1].set_xlabel('Distance (m)')
# axes[0, 1].set_ylabel('Speed (m/s)')
# axes[0, 1].legend()
# axes[0, 1].grid(True, alpha=0.3)

# # Plot 3: Duration progression
# axes[1, 0].plot(distances, encoder_duration, 'b-o', label='Encoder (Past 200 steps)', markersize=6)
# axes[1, 0].plot(distances, decoder_duration, 'r-s', label='Decoder (Next 200 steps)', markersize=6)
# axes[1, 0].set_title('Cumulative Duration Progression')
# axes[1, 0].set_xlabel('Distance (m)')
# axes[1, 0].set_ylabel('Duration (s)')
# axes[1, 0].legend()
# axes[1, 0].grid(True, alpha=0.3)

# # Plot 4: Cadence progression
# axes[1, 1].plot(distances, encoder_cadence, 'b-o', label='Encoder (Past 200 steps)', markersize=6)
# axes[1, 1].plot(distances, decoder_cadence, 'r-s', label='Decoder (Next 200 steps)', markersize=6)
# axes[1, 1].set_title('Cadence Progression')
# axes[1, 1].set_xlabel('Distance (m)')
# axes[1, 1].set_ylabel('Cadence (steps/min)')
# axes[1, 1].legend()
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
