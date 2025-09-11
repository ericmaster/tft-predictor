from typing import Optional, List
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, MultiLoss
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import InterpretableMultiHeadAttention


class WeightedMultiTargetSMAPE(MultiLoss):
    """
    Weighted SMAPE loss for multi-target forecasting.
    Applies different weights to different target variables.
    """
    
    def __init__(self, target_weights: List[float], target_names: List[str] = None, **kwargs):
        """
        Initialize weighted SMAPE loss.
        
        Args:
            target_weights: List of weights for each target variable (should sum to 1.0)
            target_names: Optional list of target names for debugging
        """
        # Create individual SMAPE metrics for each target with appropriate weights
        metrics = []
        self.target_names = target_names or [f"target_{i}" for i in range(len(target_weights))]
        
        for i, weight in enumerate(target_weights):
            metrics.append(SMAPE())
        
        # Initialize with the metrics and weights
        super().__init__(metrics=metrics, weights=target_weights, **kwargs)
        
        self.target_weights = target_weights
        
        # Ensure weights sum to 1.0
        if not abs(sum(target_weights) - 1.0) < 1e-6:
            print(f"Warning: Target weights sum to {sum(target_weights):.6f}, not 1.0")
        
        print(f"Initialized WeightedMultiTargetSMAPE with weights:")
        for name, weight in zip(self.target_names, target_weights):
            print(f"  {name}: {weight:.1%}")

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
            kwargs['learning_rate'] = 0.001
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = 64
        if 'attention_head_size' not in kwargs:
            kwargs['attention_head_size'] = 4
        if 'dropout' not in kwargs:
            kwargs['dropout'] = 0.2  # Larger to reduce overfitting
        if 'hidden_continuous_size' not in kwargs:
            kwargs['hidden_continuous_size'] = 32
        if 'output_size' not in kwargs:
            kwargs['output_size'] = len(dataset.target_names)  # Multi-target output
        if 'loss' not in kwargs:
            # Define weights for multi-target forecasting
            # 80% weight for duration, 5% each for the other 4 variables
            target_names = dataset.target_names
            target_weights = []
            
            for name in target_names:
                if name == "duration":
                    target_weights.append(0.8)  # 80% weight for duration
                else:
                    target_weights.append(0.05)  # 5% weight for each other variable
            
            kwargs['loss'] = WeightedMultiTargetSMAPE(
                target_weights=target_weights,
                target_names=target_names
            )
        if 'logging_metrics' not in kwargs:
            kwargs['logging_metrics'] = [SMAPE(), MAE(), RMSE()]
        if 'reduce_on_plateau_patience' not in kwargs:
            kwargs['reduce_on_plateau_patience'] = 4
        
        # Call parent's from_dataset
        return super().from_dataset(dataset, **kwargs)
    