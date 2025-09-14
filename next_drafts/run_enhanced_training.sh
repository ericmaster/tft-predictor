#!/bin/bash

echo "🚀 Enhanced Training Pipeline for TFT Model"
echo "=========================================="

# Step 1: Create augmented dataset
echo "📊 Step 1: Creating augmented dataset..."
python -c "
import sys
sys.path.append('./utils')
from data_augmentation import augment_training_data

augment_training_data(
    data_dir='./data/resampled',
    output_dir='./data/augmented',
    augmentation_factor=1
)
print('✅ Data augmentation completed!')
"

# Step 2: Train with enhanced regularization on original data
echo ""
echo "🧠 Step 2: Training with enhanced regularization..."
python training.py

# Step 3: Optional - Train with augmented data
echo ""
echo "🔄 Step 3: Training with augmented data..."
python -c "
from training import train_tft_model

print('Training on augmented dataset...')
model, data_module, trainer = train_tft_model(
    data_dir='./data/augmented',
    max_epochs=30,
    min_encoder_length=20,
    max_encoder_length=80,
    max_prediction_length=10,
    batch_size=64,
    hidden_size=32
)
print('✅ Augmented training completed!')
"

# Step 4: Cross-validation
echo ""
echo "📊 Step 4: Running cross-validation..."
python enhanced_training.py cv

echo ""
echo "🎉 Enhanced training pipeline completed!"
echo "Check the logs and checkpoints directories for results."