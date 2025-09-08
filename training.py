import glob
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lib.model import TrailRunningTFT
from lib.data import TFTDataModule

def find_latest_checkpoint():
    checkpoints_dir = f"./checkpoints/"
    all_checkpoints = []
    for filename in glob.glob(os.path.join(checkpoints_dir, '*.ckpt')):
        all_checkpoints.append(filename)
    
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

# Training function
def train_tft_model(
    data_dir: str = "./data/resampled",
    max_epochs: int = 30,
    max_encoder_length: int = 50,
    max_prediction_length: int = 10,
    batch_size: int = 32,
    hidden_size: int = 16,
    learning_rate: float = 0.03
):
    """
    Train the TFT model.
    
    Args:
        data_dir: Directory containing the training data
        max_epochs: Maximum number of training epochs
        gpus: Number of GPUs to use
        max_encoder_length: Length of encoder sequence
        max_prediction_length: Length of prediction horizon
        batch_size: Batch size for training
        hidden_size: Hidden size of the model
        learning_rate: Learning rate
    
    Returns:
        Trained model and data module
    """
    # Create data module
    data_module = TFTDataModule(
        data_dir=data_dir,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size
    )
    
    # Setup data
    data_module.setup()
    
    # Create model
    model = TrailRunningTFT.from_dataset(
        data_module.training,
        hidden_size=hidden_size,
        learning_rate=learning_rate
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    logger = pl.loggers.CSVLogger("logs", name="tft_model")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        logger=logger,
        devices=1,  # Use only 1 device to avoid distributed training issues
        gradient_clip_val=0.1,
        # limit_train_batches=50,  # Limit for faster training during development
        enable_checkpointing=True,
        callbacks=[
            early_stopping_callback,
            learning_rate_callback,
            checkpoint_callback
        ]
    )

    ckpt_path = find_latest_checkpoint()
    if (ckpt_path):
        print(f"Cargando modelo desde checkpoint: {ckpt_path}")
    else:
        print(f"No se encontró checkpoint en {ckpt_path}, entrenando desde cero.")
    
    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    return model, data_module, trainer


if __name__ == "__main__":
    # Train the model
    model, data_module, trainer = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=100,
        max_encoder_length=250,
        max_prediction_length=50,
        batch_size=32,
        hidden_size=16
    )
    
    print("Training completed!")
    
    # Make predictions on test set
    # test_predictions = model.predict(data_module.test_dataloader())
    # print(f"Test predictions shape: {test_predictions.shape}")