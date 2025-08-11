from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import argparse
import re
from utils import smooth
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Markers to use for plotting
markevery = [0, 75, 150, 225, 299]
TARGET_EPOCHS = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

with open("./utils/nam2parameters.json", "r") as f:
    name2params = json.load(f)
with open("./utils/name2k.json", "r") as f:
    name2k = json.load(f)
with open("./utils/name2color.json", "r") as f:
    name2color = json.load(f)
with open("./utils/name2marker.json", "r") as f:
    name2marker = json.load(f)
with open("./utils/name2k.json", "r") as f:
    name2layer = json.load(f)
with open("./utils/ignores.json", "r") as f:
    ignores = json.load(f)
with open("./utils/name2linestyle.json", "r") as f:
    linestyles = json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot mAP50 vs Number of Parameters")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing mAP50 and parameters data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the plot")
    parser.add_argument("--smooth_ratio", type=float, default=0, required=False, help="Used for moving average plotting")
    return parser.parse_args()


def plot_map_over_epochs(epochs, map_values_dict: dict, ylabel="", saving_path:str=None, smooth_ratio: float=0.7):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for name, map_values in map_values_dict.items():
        if name not in name2params.keys():
            continue
        x = np.array(epochs)
        y = np.array(map_values)*100
        y = smooth(y, smooth_ratio)
        model = name.split("[")[0] if '[' in name else name.split("-")[0]
        config = "[" + re.search(r'\[(.*?)\]', name).group(1) + "]" if '[' in name else ""
        percentage = "(" + str(round(name2k[name]*100, 2))+ "%)" if "[" in name else ""
        label = f"{model}{config}{percentage}"
        marker = name2marker[name]
        color = name2color[name]
        linestyle = linestyles[name]
        # ax.plot(x, y, label=label, marker=marker, color=color, linestyle=linestyle, markersize=4, markevery=25, linewidth=1.5, alpha=0.9)
        ax.plot(x, y, label=label, marker=marker, color=color, linestyle=linestyle, markersize=5, markevery=1, linewidth=1.5, alpha=0.9)

    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0, 60)
    grid_color = 'grey'
    ax.grid(True, axis='x', color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.set_xlabel("Epochs", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks([1]+list(range(50, 300+1, 50)))
    ax.xaxis.set_tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(saving_path, dpi=1000, bbox_inches='tight', pad_inches=0)

# New function to plot multiple map figures for specified groups
def plot_multiple_map_figures(data_path, save_path, smooth_ratio):
    epochs, map_values_dict = read_data_from_file(data_path)
    y_label = "mAP@50 (%)"

    fig_groups = {
        "fig1": ["YOLO11x-DOTAv1", "MTL[T=17]-YOLO11x-DOTAv1", "MLconv[R=500]-YOLO11x-DOTAv1", "LR[K=300]-YOLO11x-DOTAv1"],
        "fig2": ["YOLO11x-DOTAv1", "MTL[T=24]-YOLO11x-DOTAv1", "MLconv[R=430]-YOLO11x-DOTAv1", "LR[K=265]-YOLO11x-DOTAv1"],
        "fig3": ["YOLO11x-DOTAv1", "MTL[T=60]-YOLO11x-DOTAv1", "MLconv[R=350]-YOLO11x-DOTAv1", "LR[K=220]-YOLO11x-DOTAv1"],
        "fig4": ["YOLO11x-DOTAv1", "MTL[T=61]-YOLO11x-DOTAv1", "MLconv[R=300]-YOLO11x-DOTAv1", "LR[K=180]-YOLO11x-DOTAv1"]
    }
    for suffix, names in fig_groups.items():
        filtered_map = {k: v for k, v in map_values_dict.items() if k in names}
        save_fig_path = save_path.replace(".pdf", f"_{suffix}.pdf")
        plot_map_over_epochs(epochs, filtered_map, ylabel=y_label, saving_path=save_fig_path, smooth_ratio=smooth_ratio)
        print(f"Saved: {save_fig_path}")

def read_data_from_file(file_path):
    """
    Read epochs and mAP values from a file.

    :param file_path: Path to the file containing epochs and mAP values.
    :return: Tuple of lists (epochs, map_values).
    """
    map_values_dict = {}
    df = pd.read_csv(file_path)

    filtered_df = df[df['Step'].isin(TARGET_EPOCHS)].copy()

    if filtered_df.empty:
        print(f"Warning: None of the target epochs {TARGET_EPOCHS} were found in the data file.")
        return [], {}
    
    epochs = filtered_df['Step'].tolist()
    indices = [e-1 for e in epochs]
    epochs = filtered_df['Step'][indices].tolist()
    for column in filtered_df.columns:
        key = column.split(" ")[0].strip()
        # key = "-".join(column.split(" ")[0].strip().split("-")[0:-1])
        if column == 'Step' or column not in name2params.keys() or column in ignores:
            print(f"Column '{column}' not found in name2params. Skipping...")
            continue
        map_values = filtered_df[column][indices].tolist()
        map_values_dict[key] = map_values
    map_values_dict = dict(sorted(map_values_dict.items(), key=lambda item: item[0]))
    return epochs, map_values_dict


if __name__ == '__main__':
    args = parse_args()
    plot_multiple_map_figures(args.data_path, args.save_path, args.smooth_ratio)