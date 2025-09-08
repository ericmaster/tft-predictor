from typing import Optional, List
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE


class TrailRunningTFT(TemporalFusionTransformer):
    """Temporal Fusion Transformer for Trail Running Time Prediction."""
    
    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """
        Instantiate TrailRunningTFT from a TimeSeriesDataSet with optimized defaults for trail running.
        """
        # Set our preferred defaults only if not already provided
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 0.03
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = 16
        if 'attention_head_size' not in kwargs:
            kwargs['attention_head_size'] = 1
        if 'dropout' not in kwargs:
            kwargs['dropout'] = 0.1
        if 'hidden_continuous_size' not in kwargs:
            kwargs['hidden_continuous_size'] = 8
        if 'output_size' not in kwargs:
            kwargs['output_size'] = 1  # Single output for regression with SMAPE
        if 'loss' not in kwargs:
            kwargs['loss'] = SMAPE()
        if 'logging_metrics' not in kwargs:
            kwargs['logging_metrics'] = [SMAPE(), MAE(), RMSE()]
        if 'reduce_on_plateau_patience' not in kwargs:
            kwargs['reduce_on_plateau_patience'] = 4
        
        # Call parent's from_dataset
        return super().from_dataset(dataset, **kwargs)
    