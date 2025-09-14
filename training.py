import glob
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lib.model import TrailRunningTFT
from lib.data import TFTDataModule

pl.seed_everything(42, workers=True)

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
    min_encoder_length: int = 2,
    max_encoder_length: int = 200,
    max_prediction_length: int = 200,
    batch_size: int = 64,
    hidden_size: int = 64,
    learning_rate: float = 0.0005
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
        learning_rate=learning_rate,
        output_size=[1] * 5, # Multi-target output 
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
        patience=30,
        verbose=True,
        mode="min",
        min_delta=0.001,  # Minimum change to qualify as improvement
        strict=True
    )

    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    
    # Add StochasticWeightAveraging for better generalization
    # from lightning.pytorch.callbacks import StochasticWeightAveraging
    # swa_callback = StochasticWeightAveraging(
    #     swa_lrs=1e-5,  # Lower learning rate for SWA
    #     swa_epoch_start=0.8,  # Start SWA at 80% of training
    #     annealing_epochs=5
    # )

    logger = pl.loggers.CSVLogger("logs", name="tft_model")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        strategy='auto',  # Let Lightning choose the best strategy
        logger=logger,
        devices=2,
        gradient_clip_val=0.1,
        # limit_train_batches=50,  # Limit for faster training during development
        enable_checkpointing=True,
        precision="32-true",  # Use full precision for better accuracy
        # precision="16-mixed",  # Mixed precision for speed and resource efficiency
        # accumulate_grad_batches=2,  # Gradient accumulation for effective larger batch size
        # val_check_interval=0.25,  # Check validation more frequently (4 times per epoch)
        # log_every_n_steps=10,  # Log more frequently
        callbacks=[
            early_stopping_callback,
            learning_rate_callback,
            checkpoint_callback,
            # swa_callback
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
        max_epochs=30,  # Likely early stop will trigger
        min_encoder_length=2, # Limited for cold-start
        max_encoder_length=200,  # Larger may overfit
        max_prediction_length=200,
        # batch_size=32,
        batch_size=64,
        hidden_size=64,
        learning_rate=0.0005
    )
    
    print("Training completed!")
    
    # Make predictions on test set
    # test_predictions = model.predict(data_module.test_dataloader())
    # print(f"Test predictions shape: {test_predictions.shape}")