import os
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.model_selection import KFold
from lib.model import TrailRunningTFT
from lib.data import TFTDataModule
import torch


def k_fold_cross_validation(
    data_dir: str = "./data/resampled",
    k_folds: int = 5,
    max_epochs: int = 30,
    min_encoder_length: int = 20,
    max_encoder_length: int = 80,
    max_prediction_length: int = 10,
    batch_size: int = 64,
    hidden_size: int = 32
):
    """
    Perform k-fold cross-validation to better estimate model performance.
    """
    
    # Load all session files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Create k-fold splits
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(csv_files)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("=" * 50)
        
        # Create train and validation file lists
        train_files = [csv_files[i] for i in train_idx]
        val_files = [csv_files[i] for i in val_idx]
        
        # Create temporary directories for this fold
        fold_train_dir = f"./temp_fold_{fold}_train"
        fold_val_dir = f"./temp_fold_{fold}_val"
        
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_val_dir, exist_ok=True)
        
        # Copy files to temporary directories
        import shutil
        for file in train_files:
            shutil.copy(os.path.join(data_dir, file), fold_train_dir)
        for file in val_files:
            shutil.copy(os.path.join(data_dir, file), fold_val_dir)
        
        try:
            # Train model for this fold
            model, trainer = train_single_fold(
                train_dir=fold_train_dir,
                val_dir=fold_val_dir,
                fold=fold,
                max_epochs=max_epochs,
                min_encoder_length=min_encoder_length,
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                batch_size=batch_size,
                hidden_size=hidden_size
            )
            
            # Get validation results
            val_result = trainer.validate(model)[0]
            fold_results.append({
                'fold': fold,
                'val_loss': val_result['val_loss'],
                'val_SMAPE': val_result.get('val_SMAPE', None),
                'val_MAE': val_result.get('val_MAE', None)
            })
            
        finally:
            # Clean up temporary directories
            shutil.rmtree(fold_train_dir, ignore_errors=True)
            shutil.rmtree(fold_val_dir, ignore_errors=True)
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    print("=" * 50)
    val_losses = [r['val_loss'] for r in fold_results]
    print(f"Mean Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    
    for result in fold_results:
        print(f"Fold {result['fold'] + 1}: Val Loss = {result['val_loss']:.4f}")
    
    return fold_results


def train_single_fold(
    train_dir: str,
    val_dir: str,
    fold: int,
    max_epochs: int = 30,
    min_encoder_length: int = 20,
    max_encoder_length: int = 80,
    max_prediction_length: int = 10,
    batch_size: int = 64,
    hidden_size: int = 32
):
    """Train a single fold of the cross-validation."""
    
    # Create custom data module for this fold
    class FoldDataModule(TFTDataModule):
        def __init__(self, train_dir, val_dir, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_dir = train_dir
            self.val_dir = val_dir
        
        def prepare_data(self):
            # Load training data
            train_sessions = []
            for file in os.listdir(self.train_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.train_dir, file)
                    df = pd.read_csv(file_path)
                    train_sessions.append(df)
            
            # Load validation data
            val_sessions = []
            for file in os.listdir(self.val_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.val_dir, file)
                    df = pd.read_csv(file_path)
                    val_sessions.append(df)
            
            self.train_data = pd.concat(train_sessions, ignore_index=True)
            self.val_data = pd.concat(val_sessions, ignore_index=True)
    
    data_module = FoldDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        data_dir=train_dir,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size
    )
    
    # Setup data
    data_module.setup(stage="fit")
    
    # Create model
    model = TrailRunningTFT.from_dataset(
        data_module.training,
        hidden_size=hidden_size,
        learning_rate=0.0005,
        output_size=[1] * 5
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints_fold_{fold}",
        filename=f"fold-{fold}-best-checkpoint-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Reduced patience for cross-validation
        verbose=False,
        mode="min"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        strategy='auto',
        logger=False,  # Disable logging for cross-validation
        devices=1,
        gradient_clip_val=0.5,
        enable_checkpointing=True,
        precision="32-true",
        callbacks=[early_stopping_callback, checkpoint_callback],
        enable_progress_bar=False  # Reduce output clutter
    )
    
    # Train model
    trainer.fit(model, datamodule=data_module)
    
    return model, trainer


def train_with_regularization_schedule():
    """
    Train with progressive regularization - start with more regularization
    and gradually reduce it.
    """
    print("Training with progressive regularization schedule...")
    
    # Stage 1: High regularization
    print("\nStage 1: High regularization")
    model1, dm1, trainer1 = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=15,
        min_encoder_length=15,
        max_encoder_length=60,
        max_prediction_length=8,
        batch_size=128,
        hidden_size=24,
        learning_rate=0.0003
    )
    
    # Stage 2: Medium regularization
    print("\nStage 2: Medium regularization")
    model2, dm2, trainer2 = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=20,
        min_encoder_length=20,
        max_encoder_length=80,
        max_prediction_length=10,
        batch_size=64,
        hidden_size=32,
        learning_rate=0.0005
    )
    
    # Stage 3: Lower regularization (fine-tuning)
    print("\nStage 3: Fine-tuning with lower regularization")
    model3, dm3, trainer3 = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=25,
        min_encoder_length=25,
        max_encoder_length=100,
        max_prediction_length=12,
        batch_size=32,
        hidden_size=48,
        learning_rate=0.0008
    )
    
    return model3, dm3, trainer3


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cv":
        # Run cross-validation
        print("Running k-fold cross-validation...")
        results = k_fold_cross_validation(k_folds=5)
    elif len(sys.argv) > 1 and sys.argv[1] == "progressive":
        # Run progressive training
        model, dm, trainer = train_with_regularization_schedule()
    else:
        # Run standard training with enhanced regularization
        from training import train_tft_model
        model, dm, trainer = train_tft_model()