import pytorch_lightning as pl
from lib.model import TrailRunningTFT
from lib.data import TFTDataModule

# Training function
def train_tft_model(
    data_dir: str = "./data/resampled",
    max_epochs: int = 50,
    gpus: int = 0,
    max_encoder_length: int = 50,
    max_prediction_length: int = 10,
    batch_size: int = 64,
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
    model = TrailRunningTFT(
        training_dataset=data_module.training,
        learning_rate=learning_rate,
        hidden_size=hidden_size
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # Limit for faster training during development
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=False),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    return model, data_module, trainer


if __name__ == "__main__":
    # Train the model
    model, data_module, trainer = train_tft_model(
        data_dir="./data/resampled",
        max_epochs=20,
        max_encoder_length=50,
        max_prediction_length=10,
        batch_size=32,
        hidden_size=16
    )
    
    print("Training completed!")
    
    # Make predictions on test set
    test_predictions = model.predict(data_module.test_dataloader())
    print(f"Test predictions shape: {test_predictions.shape}")