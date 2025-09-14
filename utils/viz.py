import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_hparams(hparams_file):
    """Extract simple key-value parameters from hparams.yaml without loading complex objects"""
    params = {}
    
    with open(hparams_file, 'r') as f:
        content = f.read()
        
    # Extract simple key-value parameters
    for line in content.split('\n'):
        line = line.strip()
        # Skip empty lines, comments, and complex objects
        if not line or line.startswith('#') or '!!python' in line or line.startswith('-'):
            continue
        
        # Match simple key: value patterns
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', line)
        if match:
            key, value = match.groups()
            
            # Skip complex nested structures
            if value.strip() in ['{}', '[]', 'null'] or value.startswith('&') or value.startswith('*'):
                continue
            
            # Parse different value types
            value = value.strip()
            if value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            elif value.lower() == 'null':
                params[key] = None
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                params[key] = float(value) if '.' in value else int(value)
            else:
                # Keep as string, remove quotes if present
                params[key] = value.strip('"\'')
    
    # Extract nested parameters using regex
    # Look for max_prediction_length
    pred_length_match = re.search(r'max_prediction_length:\s*(\d+)', content)
    if pred_length_match:
        params['max_prediction_length'] = int(pred_length_match.group(1))
    
    # Look for min_encoder_length
    min_enc_match = re.search(r'min_encoder_length:\s*(\d+)', content)
    if min_enc_match:
        params['min_encoder_length'] = int(min_enc_match.group(1))
    
    # Look for randomize_length
    random_match = re.search(r'randomize_length:\s*(true|false|null)', content)
    if random_match:
        params['randomize_length'] = random_match.group(1)
    
    # Look for predict_mode
    predict_match = re.search(r'predict_mode:\s*(true|false)', content)
    if predict_match:
        params['predict_mode'] = predict_match.group(1) == 'true'
    
    # Look for target variables
    target_match = re.search(r'target:\s*&\w+\s*\n((?:\s*-\s*\w+\s*\n)+)', content)
    if target_match:
        targets = re.findall(r'-\s*(\w+)', target_match.group(1))
        params['target_variables'] = targets
    
    return params

def plot_metrics(
    metrics_csv_path=None,
    trainer=None,
    plot_metrics=[["train_loss", "valid_loss"], ["train_acc", "valid_acc"]],
    plot_titles=["Loss", "Accuracy"],
    save_svg_path=None,
):
    """Plot training and validation metrics from a CSV file or a PyTorch Lightning trainer."""
    if metrics_csv_path is None:
        if trainer is None:
            raise ValueError("Either metrics_csv_path or trainer must be provided.")
        else:
            metrics_csv_path = f"{trainer.logger.log_dir}/metrics.csv"
    metrics = pd.read_csv(metrics_csv_path)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    for i, metrics in enumerate(plot_metrics):
        ax = df_metrics[metrics].plot(
            grid=True,
            legend=True,
            xlabel="Epoch",
            ylabel=plot_titles[i].replace('_', ' ').title(),
            title=f"{metrics[0].replace('_', ' ').title()} vs {metrics[1].replace('_', ' ').title()}",
        )
        if save_svg_path:
            # Save each plot to SVG, appending index if multiple plots
            base, ext = save_svg_path.rsplit('.', 1) if '.' in save_svg_path else (save_svg_path, 'svg')
            svg_file = f"{base}.{ext}"
            fig = ax.get_figure()
            fig.savefig(svg_file, format="svg")
            plt.close(fig)
    if not save_svg_path:
        plt.show()

def visualize_predictions(raw_predictions, batch_id=0, target_idx=0, target_name="Duration"):
    """
    Visualize predictions vs actual values for a specific batch and target variable.
    
    Args:
        raw_predictions: Raw predictions from model.predict()
        batch_id: Batch index to visualize (default: 0)
        target_idx: Target variable index (0=duration, 1=heartRate, 2=temperature, 3=cadence, 4=speed)
        target_name: Name of the target variable for labeling
    """
    # Debug the structure
    # print(f"Output type: {type(raw_predictions.output)}")
    # print(f"Output length: {len(raw_predictions.output)}")
    
    # for i, output in enumerate(raw_predictions.output):
    #     print(f"Output {i} type: {type(output)}")
    #     if hasattr(output, 'shape'):
    #         print(f"Output {i} shape: {output.shape}")
    #     elif isinstance(output, list):
    #         print(f"Output {i} is a list with {len(output)} elements")
    #         if len(output) > 0 and hasattr(output[0], 'shape'):
    #             print(f"Output {i}[0] shape: {output[0].shape}")

    # Debug the decoder_target structure
    # print(f"\nDecoder target structure:")
    # print(f"decoder_target type: {type(raw_predictions.x['decoder_target'])}")
    # print(f"decoder_target length: {len(raw_predictions.x['decoder_target'])}")
    # for i, target in enumerate(raw_predictions.x['decoder_target']):
    #     print(f"decoder_target[{i}] shape: {target.shape}")

    # Extract predictions - handle the nested list structure
    if isinstance(raw_predictions.output[0], list):
        # If output[0] is a list, get the specified batch sample, all time steps, specified target
        predictions = raw_predictions.output[0][target_idx][batch_id, :, 0].detach().cpu().numpy()
    else:
        # If it's a tensor directly, get specified batch sample, all time steps, specified target
        predictions = raw_predictions.output[0][batch_id, :, target_idx].detach().cpu().numpy()

    # Extract actuals - select specified batch sample and target
    actuals = raw_predictions.x['decoder_target'][target_idx][batch_id, :].detach().cpu().numpy()

    # print(f"\n{target_name} predictions shape: {predictions.shape}")
    # print(f"{target_name} actuals shape: {actuals.shape}")

    # Create visualization plot
    plt.figure(figsize=(12, 6))
    time_steps = range(len(predictions))

    plt.plot(time_steps, actuals, label=f'Actual {target_name}', color='blue', linewidth=2)
    plt.plot(time_steps, predictions, label=f'Predicted {target_name}', color='orange', linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel(f'{target_name}')
    plt.title(f'{target_name} Predictions vs Actual (Batch {batch_id})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate and display error metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    print(f"\nError Metrics for {target_name} (Batch {batch_id}):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return predictions, actuals