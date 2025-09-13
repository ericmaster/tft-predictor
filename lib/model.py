from typing import Optional, List
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, MultiLoss
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import InterpretableMultiHeadAttention


class WeightedMultiTargetSMAPE(nn.Module):
    """
    Weighted SMAPE loss for multi-target forecasting.
    Applies different weights to different target variables.
    """
    
    def __init__(self, target_weights: List[float], target_names: List[str] = None):
        """
        Initialize weighted SMAPE loss.
        
        Args:
            target_weights: List of weights for each target variable (should sum to 1.0)
            target_names: Optional list of target names for debugging
        """
        super().__init__()
        
        self.target_names = target_names or [f"target_{i}" for i in range(len(target_weights))]
        self.target_weights = torch.tensor(target_weights, dtype=torch.float32)
        
        # Ensure weights sum to 1.0
        if not abs(sum(target_weights) - 1.0) < 1e-6:
            print(f"Warning: Target weights sum to {sum(target_weights):.6f}, not 1.0")
        
        print(f"Initialized WeightedMultiTargetSMAPE with weights:")
        for name, weight in zip(self.target_names, target_weights):
            print(f"  {name}: {weight:.1%}")
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted SMAPE loss.
        
        Args:
            prediction: Predicted values [batch_size, seq_len, num_targets]
            target: Target values [batch_size, seq_len, num_targets]
        
        Returns:
            Weighted SMAPE loss
        """
        # Move weights to the same device as inputs
        weights = self.target_weights.to(prediction.device)
        
        # Calculate SMAPE for each target
        # SMAPE = 2 * |prediction - target| / (|prediction| + |target|)
        smape_per_target = 2 * torch.abs(prediction - target) / (torch.abs(prediction) + torch.abs(target) + 1e-8)
        
        # Average over batch and sequence dimensions, keep target dimension
        smape_per_target = smape_per_target.mean(dim=(0, 1))  # [num_targets]
        
        # Apply weights and sum
        weighted_loss = (smape_per_target * weights).sum()
        
        return weighted_loss

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
            kwargs['dropout'] = 0.25
        if 'hidden_continuous_size' not in kwargs:
            kwargs['hidden_continuous_size'] = 32
        if 'output_size' not in kwargs:
            kwargs['output_size'] = len(dataset.target_names)  # Multi-target output
        if 'lstm_layers' not in kwargs:
            kwargs['lstm_layers'] = 1
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = 0.005  # L2 regularization
        if 'loss' not in kwargs:
            # Use a simpler approach - just use MultiLoss with SMAPE for each target
            # This avoids potential compatibility issues with pytorch-forecasting
            target_names = dataset.target_names
            
            # Create individual SMAPE losses for each target
            from pytorch_forecasting.metrics import MultiLoss, SMAPE
            
            # Define weights for multi-target forecasting
            target_weights = []
            for name in target_names:
                if name == "duration":
                    target_weights.append(0.8)  # 80% weight for duration
                else:
                    target_weights.append(0.05)  # 5% weight for each other variable
            
            kwargs['loss'] = MultiLoss(
                metrics=[SMAPE() for _ in target_names],
                weights=target_weights
            )
            
            print(f"Initialized MultiLoss with target weights:")
            for name, weight in zip(target_names, target_weights):
                print(f"  {name}: {weight:.1%}")
        if 'logging_metrics' not in kwargs:
            kwargs['logging_metrics'] = [SMAPE(), MAE(), RMSE()]
        if 'reduce_on_plateau_patience' not in kwargs:
            kwargs['reduce_on_plateau_patience'] = 4
        
        # Call parent's from_dataset
        return super().from_dataset(dataset, **kwargs)
    