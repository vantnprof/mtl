from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import argparse
import re

# --- Load configuration files (no changes here) ---
with open("./utils/nam2parameters.json", "r") as f:
    name2params = json.load(f)
with open("./utils/name2k.json", "r") as f:
    name2k = json.load(f)
with open("./utils/name2color.json", "r") as f:
    name2color = json.load(f)
with open("./utils/name2marker.json", "r") as f:
    name2marker = json.load(f)
with open("./utils/name2t.json", "r") as f:
    name2t = json.load(f)
with open("./utils/ignores.json", "r") as f:
    ignores = json.load(f)

another_ignores = {
    "LR-MTL[T=42,lambda_ce=0.70]-ResNet101-CIFAR10",
    "LR-MTL[T=15,lambda_ce=0.70]-ResNet101-CIFAR10",
    "LR-MTL[T=3,lambda_ce=0.70]-ResNet101-CIFAR10",
    "MTL[T=16]-ResNet101-CIFAR10",
    "MTL[T=4]-ResNet101-CIFAR10",
    "MTL[T=43]-ResNet101-CIFAR10",
    "LR-MTL[T=42,lambda_ce=0.70]-ResNet101-CIFAR10-25",
    "LR-MTL[T=15,lambda_ce=0.70]-ResNet101-CIFAR10-25",
    "LR-MTL[T=3,lambda_ce=0.70]-ResNet101-CIFAR10-25",
    "MTL[T=16]-ResNet101-CIFAR10-25",
    "MTL[T=4]-ResNet101-CIFAR10-25",
    "MTL[T=43]-ResNet101-CIFAR10-25",
}

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot mAP50 vs Number of Parameters for two datasets")
    parser.add_argument("--data_path_full", type=str, required=True, help="Path to the CSV for the full training dataset.")
    parser.add_argument("--data_path_2_5", type=str, required=True, help="Path to the CSV for the 2/5 training dataset.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the combined plot.")
    return parser.parse_args()

def read_data_from_file(file_path):
    """Reads and processes model performance data from a CSV file."""
    df = pd.read_csv(file_path)
    acc_param_dict = {}
    for _, row in df.iterrows():
        name = row["Name"].strip()
        print("Processing:", name)
        if name not in name2params:
            print(f"Warning: Column '{name}' not found in name2params. Skipping...")
            continue
        if name in ignores:
            print(f"Warning: {name} was ignored. Skipping.")
            continue
        # if name in another_ignores:
        #     print(f"Warning: {name} was ignored in another_ignores. Skipping.")
        #     continue
        acc_param_dict[name] = {
            "param": name2params[name],
            "acc": row["best_val_accuracy"]
        }
    return dict(sorted(acc_param_dict.items(), key=lambda item: item[0]))

def plot_combined_charts(acc_param_dict_full, acc_param_dict_2_5, y_label, saving_path):
    """Plots two charts in one figure with a common legend and separate y-limits."""
    # Note: `sharey` is set to False to allow for different y-axis limits.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    
    legend_handles_for_sorting = []
    scatter_to_name_map = {}

    plot_configs = [
        (acc_param_dict_full, '(a) Full training data'),
        (acc_param_dict_2_5, '(b) 2/5 training data')
    ]

    for i, (data_dict, title) in enumerate(plot_configs):
        ax = axes[i]
        is_first_plot = (i == 0)
        
        for name, values in data_dict.items():
            model = name.split("[")[0] if '[' in name else name.split("-")[0]

            if '[' in name:
                config_text = re.search(r'\[(.*?)\]', name).group(1)
                config_text = config_text.replace('lambda_ce', r'$\lambda$')
                config = f'[{config_text}]'
            else:
                config = ""
            
            percentage = f"({round(name2k[name]*100, 2)}%)" if "[" in name else ""
            label = f"{model} {config} {percentage}"
            
            sc = ax.scatter(values['param'] / 1e7, values['acc'],
                            marker=name2marker[name],
                            color=name2color[name],
                            label=label,
                            s=100)
            
            if is_first_plot:
                legend_handles_for_sorting.append(sc)
                scatter_to_name_map[sc] = name

        # --- Subplot Formatting ---
        ax.set_xlabel(f'Number of parameters ($x10^{7}$)', fontsize=10)
        ax.set_title(title, fontsize=10)

        # Set y-limits and ticks specifically for each plot
        if is_first_plot:
            ax.set_ylim(70, 86 + 1)
            ax.set_yticks(range(70, 86 + 1, 2))
        else:
            ax.set_ylim(70, 86 + 1)
            ax.set_yticks(range(70, 86 + 1, 2))
        
        grid_color = 'grey'
        ax.grid(True, which='both', axis='both', linestyle='--', color=grid_color, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)

    axes[0].set_ylabel(y_label, fontsize=10)

    # --- Common Legend Logic ---
    handles_std, handles_lrmtl, handles_mtl, handles_mlconv, handles_lr = [], [], [], [], []
    for sc in legend_handles_for_sorting:
        name = scatter_to_name_map[sc]
        model_name = name.split("[")[0] if '[' in name else name.split("-")[0]
        lower_model = model_name.lower()
        
        if lower_model == 'resnet101': handles_std.append(sc)
        elif 'lr-mtl' in lower_model or 'lrmtl' in lower_model: handles_lrmtl.append(sc)
        elif 'mtl' in lower_model and not ('lr-mtl' in lower_model or 'lrmtl' in lower_model): handles_mtl.append(sc)
        elif 'mlconv' in lower_model: handles_mlconv.append(sc)
        elif (lower_model.startswith('lr') or lower_model == 'lr') and not ('lr-mtl' in lower_model or 'lrmtl' in lower_model): handles_lr.append(sc)
        else:
            if 'mlconv' in lower_model: handles_mlconv.append(sc)
            elif 'mtl' in lower_model: handles_mtl.append(sc)
            elif 'lr' in lower_model: handles_lr.append(sc)
            else: handles_std.append(sc)

    def extract_percentage(sc):
        name = scatter_to_name_map.get(sc)
        try: return float(name2k[name])
        except (KeyError, ValueError, TypeError): return float('inf')

    std_sorted = sorted(handles_std, key=extract_percentage)
    lrmtl_sorted = sorted(handles_lrmtl, key=extract_percentage)
    mtl_sorted = sorted(handles_mtl, key=extract_percentage)
    mlconv_sorted = sorted(handles_mlconv, key=extract_percentage)
    lr_sorted = sorted(handles_lr, key=extract_percentage)
    
    final_handles = (mlconv_sorted + lr_sorted) + mtl_sorted + (lrmtl_sorted + std_sorted)
    import matplotlib.patches as mpatches
    ncol = 3  # The number of columns specified in your legend
    if len(final_handles) % ncol != 0:
        num_to_add = ncol - (len(final_handles) % ncol)
        for _ in range(num_to_add):
            final_handles.append(mpatches.Patch(color='none', label=''))

    # Generate labels from the potentially padded handle list
    final_labels = [h.get_label() for h in final_handles]
    # final_labels = [h.get_label() for h in final_handles]

    fig.legend(
        handles=final_handles,
        labels=final_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=10,
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    
    fig.savefig(saving_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)
    print(f"Combined plot saved to {saving_path}")

if __name__ == '__main__':
    args = parse_args()
    
    acc_param_full = read_data_from_file(args.data_path_full)
    acc_param_2_5 = read_data_from_file(args.data_path_2_5)
    
    y_axis_label = "Validation accuracy (%)"
    
    plot_combined_charts(acc_param_full, acc_param_2_5, y_label=y_axis_label, saving_path=args.save_path)