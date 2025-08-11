from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import argparse
import re
from utils import smooth


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
with open("./utils/name2linestyle.json", 'r') as f:
    linestyles = json.load(f)
    
markevery = [0, 20, 40, 60, 80, 99]  # Mark every nth point for visibility in the plot


def parse_args():
    parser = argparse.ArgumentParser(description="Plot mAP50 vs Number of Parameters")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing mAP50 and parameters data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the plot")
    parser.add_argument("--smooth_ratio", type=float, default=0, required=False, help="Used for moving average plotting")
    return parser.parse_args()


def plot_map_over_epochs(epochs, map_values_dict: dict, ylabel="", saving_path:str=None, smooth_ratio: float=0.7):
    def extract_percentage(name):
        try:
            return name2k[name]
        except Exception:
            return 0.0

    # Group keys
    groups = {
        'LR': [],
        'MLconv': [],
        'MTL': [],
        'LR-MTL': [],
        'ResNet101': []
    }
    for name, _ in map_values_dict.items():
        if name.startswith("LR-MTL"):
            groups['LR-MTL'].append(name)
        elif name.startswith("MTL"):
            groups['MTL'].append(name)
        elif name.startswith("LR"):
            groups['LR'].append(name)
        elif name.startswith("MLconv"):
            groups['MLconv'].append(name)
        elif name.startswith("ResNet101"):
            groups['ResNet101'].append(name)

    # Sort within each group by descending percentage
    for k in groups:
        groups[k] = sorted(groups[k], key=extract_percentage, reverse=True)

    # Desired order: LR → MLconv → MTL → LR-MTL → ResNet101
    ordered_names = []
    for g in ['LR', 'MLconv', 'MTL', 'LR-MTL', 'ResNet101']:
        ordered_names.extend(groups[g])

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for name in ordered_names:
        if name not in name2params.keys():
            continue
        map_values = map_values_dict[name]
        x = np.array(epochs)
        y = np.array(map_values)
        y = smooth(y, smooth_ratio)
        model = name.split("[")[0] if '[' in name else name.split("-")[0]
        config = "[" + re.search(r'\[(.*?)\]', name).group(1) + "]" if '[' in name else ""
        config = config.replace("lambda_ce", "$\lambda$")
        percentage = "(" + str(round(name2k[name]*100, 2))+ "%)" if "[" in name else ""
        label = f"{model}{config}{percentage}"
        marker = name2marker[name]
        color = name2color[name]
        linestyle = linestyles[name]
        ax.plot(x, y, label=label, marker=marker, color=color, linestyle=linestyle, markersize=5, markevery=markevery)

    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 100)
    grid_color = 'grey'
    ax.grid(True, axis='x', color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.set_xlabel("Epochs", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([1]+list(range(20, 100+1, 20)))
    # ax.set_title("", fontsize=9)
    ax.xaxis.set_tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(saving_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    

def read_data_from_file(file_path):
    """
    Read epochs and mAP values from a file.

    :param file_path: Path to the file containing epochs and mAP values.
    :return: Tuple of lists (epochs, map_values).
    """
    acc_values_dict = {}
    df = pd.read_csv(file_path)
    # epochs = [1, 5, 10, 15, 20, 25, 35, 40, 45, 50, 55, 65, 70, 75, 80, 85, 90, 95, 100]
    # epochs = [1, 20, 40, 60, 80, 100]
    epochs = df['epoch'].tolist()
    # epochs = list(range(0, 101, 1))
    epochs[0] = 1
    indices = [e-1 for e in epochs]
    epochs = df['epoch'][indices].tolist()
    for column in df.columns:
        key = column.split(" ")[0].strip()
        # key = "-".join(column.split(" ")[0].strip().split("-")[0:-1])
        if key == "epoch" or key not in name2params or key in ignores:
            print(f"Column '{column}' not found in name2params. Skipping...")
            continue
        map_values = df[column][indices].tolist()
        acc_values_dict[key] = map_values
    acc_values_dict = dict(sorted(acc_values_dict.items(), key=lambda item: item[0]))
    return epochs, acc_values_dict


def plot_multiple_acc_figures(data_path, save_path, smooth_ratio):
    epochs, acc_values_dict = read_data_from_file(data_path)
    y_label = "Validation accuracy (%)"

    fig_groups = {
        "fig1": ["ResNet101-CIFAR10", "MTL[T=3]-ResNet101-CIFAR10", "LR-MTL[T=3,lambda_ce=0.70]-ResNet101-CIFAR10", "MTL[T=4]-ResNet101-CIFAR10", "MLconv[R=400]-ResNet101-CIFAR10", "LR[K=300]-ResNet101-CIFAR10"],
        "fig2": ["ResNet101-CIFAR10", "MTL[T=15]-ResNet101-CIFAR10", "LR-MTL[T=15,lambda_ce=0.70]-ResNet101-CIFAR10", "MTL[T=16]-ResNet101-CIFAR10", "MLconv[R=345]-ResNet101-CIFAR10", "LR[K=260]-ResNet101-CIFAR10"],
        "fig3": ["ResNet101-CIFAR10", "MTL[T=42]-ResNet101-CIFAR10", "LR-MTL[T=42,lambda_ce=0.70]-ResNet101-CIFAR10", "MTL[T=43]-ResNet101-CIFAR10", "MLconv[R=300]-ResNet101-CIFAR10", "LR[K=229]-ResNet101-CIFAR10"],
        "fig4": ["ResNet101-CIFAR10", "MLconv[R=200]-ResNet101-CIFAR10", "MLconv[R=100]-ResNet101-CIFAR10", "LR[K=150]-ResNet101-CIFAR10", "LR[K=75]-ResNet101-CIFAR10"],
    }

    for suffix, names in fig_groups.items():
        filtered_map = {k: v for k, v in acc_values_dict.items() if k in names}
        save_fig_path = save_path.replace(".pdf", f"_{suffix}.pdf")
        plot_map_over_epochs(epochs, filtered_map, ylabel=y_label, saving_path=save_fig_path, smooth_ratio=smooth_ratio)
        print(f"Saved: {save_fig_path}")


if __name__ == '__main__':
    args = parse_args()
    plot_multiple_acc_figures(args.data_path, args.save_path, args.smooth_ratio)