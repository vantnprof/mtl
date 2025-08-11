import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from matplotlib.patches import Patch

def read_param_data(data_path):
    """Reads and processes initial and final parameter data from the CSV."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file at {data_path} was not found.")
        return None, None

    initial_params, final_params = {}, {}
    param_cols = [col for col in df.columns if "model_params" in col and "MIN" not in col and "MAX" not in col]

    for col in param_cols:
        base_name = col.split(' - model_params')[0].strip()
        label = 'Resnet101' if base_name == 'Resnet101-CIFAR10' else base_name.replace('LR-', '').replace('-Resnet101-CIFAR10', '')
        
        if not df.empty:
            initial_params[label] = df[col].iloc[0] / 1e6
            final_params[label] = df[col].iloc[-1] / 1e6
        else:
            initial_params[label], final_params[label] = 0, 0
            
    return initial_params, final_params

def plot_parameter_chart(initial_data, final_data, save_path):
    """Plots the compact bar chart with differentiated shades of red for MTL models."""
    # **MODIFICATION: Assigned unique shades of red to each MTL model**
    name2color = {
        "Resnet101": "#000000",                 # Black
        "MTL[T=3,lambda_ce=0.70]": "#228B22",   # Forest Green
        "MTL[T=15,lambda_ce=0.70]": "#228B22",  # Forest Green
        "MTL[T=42,lambda_ce=0.70]": "#228B22",  # Forest Green
        # Unique shades of red for standard MTL models
        "MTL[T=3]": "#d62728",                  # Bright Red
        "MTL[T=4]": "#e377c2",                  # Pink/Red
        "MTL[T=15]": "#ff7f0e",                 # Orange
        "MTL[T=16]": "#ff9896",                 # Light Red
        "MTL[T=42]": "#8c564b",                 # Brown/Red
        "MTL[T=43]": "#c44e52",                 # Darker Red
    }

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    groups = [
        ["Resnet101"],
        ["MTL[T=3,lambda_ce=0.70]", "MTL[T=3]", "MTL[T=4]"],
        ["MTL[T=15,lambda_ce=0.70]", "MTL[T=15]", "MTL[T=16]"],
        ["MTL[T=42,lambda_ce=0.70]", "MTL[T=42]", "MTL[T=43]"],
    ]
    group_labels = ["Baseline", "T≈3", "T≈15", "T≈42"]
    
    bar_width, x_pos, xtick_positions = 0.8, 0, []
    legend_handles_map = {}

    for group_models in groups:
        group_start_pos = x_pos
        for model_name in group_models:
            color = name2color.get(model_name, 'gray')
            
            display_name = "ResNet101"
            if '[' in model_name:
                config = re.search(r'\[(.*?)\]', model_name).group(1).replace('lambda_ce=0.70', r'$ \lambda$=0.70')
                display_name = f"MTL [{config}] "

            if display_name not in legend_handles_map:
                legend_handles_map[display_name] = Patch(facecolor=color, label=display_name)
            
            # Apply hatches only to LR-MTL models
            if 'lambda_ce' in model_name:
                ax.bar(x_pos, initial_data.get(model_name, 0), width=bar_width, color=color, hatch='////', edgecolor='black', zorder=3)
                x_pos += 1
                ax.bar(x_pos, final_data.get(model_name, 0), width=bar_width, color=color, hatch='\\\\\\\\', edgecolor='black', zorder=3)
                x_pos += 1
            else:
                # Plot solid bars for Resnet and standard MTL
                value = initial_data.get(model_name, 0) if model_name == 'Resnet101' else final_data.get(model_name, 0)
                ax.bar(x_pos, value, width=bar_width, color=color, edgecolor='black', zorder=3)
                x_pos += 1
        
        xtick_positions.append(group_start_pos + (x_pos - group_start_pos - bar_width) / 2)
        x_pos += bar_width

    # --- Formatting and Legend --- 
    ax.set_ylabel(r"Number of parameters ($ \times 10^6 $)", fontsize=7)
    # ax.set_xlabel("Model Groups", fontsize=7)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(group_labels, fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    
    grid_color = 'grey'
    ax.grid(True, axis='y', linestyle='--', color=grid_color, alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    
    hatch_legend = [
        Patch(facecolor='gray', hatch='////', edgecolor='black', label='LR-MTL Initial'),
        Patch(facecolor='gray', hatch='\\\\\\\\', edgecolor='black', label='LR-MTL Final')
    ]
    # The legend will now correctly show a unique entry for each model
    color_legend = sorted(legend_handles_map.values(), key=lambda h: h.get_label())
    
    fig.legend(
        handles=hatch_legend + color_legend,
        loc='lower center', bbox_to_anchor=(0.5, 0.05),
        ncol=3, fontsize=5, frameon=True, handletextpad=0.5
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    
    # --- Save Plot ---
    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"✅ Plot with differentiated red shades saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot compact styled initial vs final model parameters.")
    parser.add_argument("--data_path", type=str, default="param_initial_aftertrain_CIFAR10.csv", help="Path to CSV file.")
    parser.add_argument("--save_path", type=str, default="parameters_differentiated_reds.pdf", help="Path to save output plot.")
    args = parser.parse_args()

    initial_data, final_data = read_param_data(args.data_path)
    if initial_data:
        plot_parameter_chart(initial_data, final_data, args.save_path)