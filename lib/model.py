from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE


# import warnings
# warnings.filterwarnings("ignore")


class TrailRunningTFT(pl.LightningModule):
    """Temporal Fusion Transformer for Trail Running Time Prediction."""
    
    def __init__(
        self,
        training_dataset: TimeSeriesDataSet,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 1,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        output_size: int = 7,
        loss: torch.nn.Module = None,
        logging_metrics: List = None
    ):
        """
        Initialize the TFT model.
        
        Args:
            training_dataset: Training dataset to derive model parameters
            learning_rate: Learning rate for optimization
            hidden_size: Hidden size of the TFT
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
            output_size: Number of outputs from TFT (quantile outputs)
            loss: Loss function
            logging_metrics: List of metrics to log
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Default loss and metrics
        if loss is None:
            loss = SMAPE()
        
        if logging_metrics is None:
            logging_metrics = [SMAPE(), MAE(), RMSE()]
        
        # Create TFT model
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,
            loss=loss,
            log_interval=10,
            log_val_interval=1,
            logging_metrics=logging_metrics,
            reduce_on_plateau_patience=4,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def predict(self, dataloader, mode: str = "prediction"):
        """Make predictions using the trained model."""
        return self.model.predict(dataloader, mode=mode)


