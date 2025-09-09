from typing import Optional, List
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import InterpretableMultiHeadAttention

class TrailRunningTFT(TemporalFusionTransformer):
    """Temporal Fusion Transformer for Trail Running Time Prediction."""

    def __init__(self, mask_bias: float = -1e4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._override_mask_bias(mask_bias)

    def _override_mask_bias(self, mask_bias_value: float):
        """Override mask_bias in all InterpretableMultiHeadAttention modules."""
        for module in self.modules():
            if isinstance(module, InterpretableMultiHeadAttention):
                # Override the mask_bias used in its internal attention
                if hasattr(module, "mask_bias"):
                    module.mask_bias = mask_bias_value
                # Also override in the nested MultiHeadAttention if accessible
                if hasattr(module, "_attention"):
                    attention_module = getattr(module, "_attention", None)
                    if attention_module is not None and hasattr(attention_module, "mask_bias"):
                        attention_module.mask_bias = mask_bias_value

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
    