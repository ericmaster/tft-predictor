import re
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
        df_metrics[metrics].plot(
            grid=True,
            legend=True,
            xlabel="Epoch",
            ylabel=plot_titles[i].replace('_', ' ').title(),
            title=f"{metrics[0].replace('_', ' ').title()} vs {metrics[1].replace('_', ' ').title()}",
        )
    # df_metrics[["train_loss", "valid_loss"]].plot(
    #     grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    # )
    # df_metrics[["train_acc", "valid_acc"]].plot(
    #     grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    # )

    plt.show()
