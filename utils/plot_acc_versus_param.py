from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import argparse
import re


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
    parser = argparse.ArgumentParser(description="Plot mAP50 vs Number of Parameters")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing mAP50 and parameters data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the plot")
    return parser.parse_args()


def plot_acc_vs_params(acc_param_dict, y_label:str, saving_path):
    fig, ax = plt.subplots(figsize=(5, 3))
    # Prepare handle groups
    handles_std = []
    handles_lrmtl = []
    handles_mtl = []
    handles_mlconv = []
    handles_lr = []
    scatter_to_name = {}
    for name, values in acc_param_dict.items():
        model = name.split("[")[0] if '[' in name else name.split("-")[0]

        # Get config and replace lambda_ce with the math symbol
        if '[' in name:
            config_text = re.search(r'\[(.*?)\]', name).group(1)
            config_text = config_text.replace('lambda_ce', r'$\lambda$')
            config = f'[{config_text}]'
        else:
            config = ""
            
        percentage = "(" + str(round(name2k[name]*100, 2))+ "%)" if "[" in name else ""
        # label = f"{model} {config} {percentage}"
        label_parts = [model, config, percentage]
        label = ' '.join(part for part in label_parts if part)
        print(f"Plotting: {label}")
        input()

        sc = ax.scatter(values['param']/(1e7), values['acc'],
                        marker=name2marker[name],
                        color=name2color[name],
                        label=label,
                        s=100,
        )
        scatter_to_name[sc] = name
        # Group handles according to legend groups
        lower_model = model.lower()
        if lower_model == 'resnet101':
            handles_std.append(sc)
        elif ('lr-mtl' in lower_model or 'lrmtl' in lower_model):
            handles_lrmtl.append(sc)
        elif 'mtl' in lower_model and not ('lr-mtl' in lower_model or 'lrmtl' in lower_model):
            handles_mtl.append(sc)
        elif 'mlconv' in lower_model:
            handles_mlconv.append(sc)
        elif (lower_model.startswith('lr') or lower_model == 'lr') and not ('lr-mtl' in lower_model or 'lrmtl' in lower_model):
            handles_lr.append(sc)
        else:
            # fallback: try to categorize as best as possible
            if 'mlconv' in lower_model:
                handles_mlconv.append(sc)
            elif 'mtl' in lower_model:
                handles_mtl.append(sc)
            elif 'lr' in lower_model:
                handles_lr.append(sc)
            else:
                handles_std.append(sc)

    ax.set_xlabel(f'Number of parameters ($x10^{7}$)', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_yticks(range(70, 82+1, 2))
    ax.set_title('(b) 2/5 training data', fontsize=10)
    ax.grid(True)

    # Helper function to extract percentage value from name2k
    def extract_percentage(name):
        try:
            return float(name2k[name])
        except (KeyError, ValueError, TypeError):
            return float('inf')

    # Maintain the fixed group order: ResNet101 → LR-MTL → MTL → MLconv → LR
    # Within each group, sort handles by ascending percentage value
    std_sorted = sorted(
        handles_std,
        key=lambda sc: extract_percentage(scatter_to_name[sc])
    )
    lrmtl_sorted = sorted(
        handles_lrmtl,
        key=lambda sc: extract_percentage(scatter_to_name[sc])
    )
    mtl_sorted = sorted(
        handles_mtl,
        key=lambda sc: extract_percentage(scatter_to_name[sc])
    )
    mlconv_sorted = sorted(
        handles_mlconv,
        key=lambda sc: extract_percentage(scatter_to_name[sc])
    )
    lr_sorted = sorted(
        handles_lr,
        key=lambda sc: extract_percentage(scatter_to_name[sc])
    )
    # Group legend handles into columns as specified
    col1 = mlconv_sorted
    col2 = lr_sorted
    col3 = mtl_sorted
    col4 = lrmtl_sorted + std_sorted
    handles = col1 + col2 + col3 + col4
    labels = [h.get_label() for h in handles]

    # --- LEGEND POSITION REVERTED TO BOTTOM ---
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3), 
        ncol=4,
        fontsize=8,
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.5,
        borderaxespad=0.3
    )
    
    # Reverted rect to make space at the bottom
    plt.tight_layout() 
    grid_color = 'grey'
    ax.grid(True, axis='x', color=grid_color)
    # ax.set_ylim(70, 85)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color(grid_color)
    plt.gca().spines['bottom'].set_color(grid_color)
    fig.savefig(saving_path, dpi=1000, bbox_inches='tight',pad_inches=0)
    

def read_data_from_file(file_path):
    """
    Read epochs and mAP values from a file.

    :param file_path: Path to the file containing epochs and mAP values.
    :return: Tuple of lists (epochs, map_values).
    """
    df = pd.read_csv(file_path)
    acc_param_dict = {}
    for _, row in df.iterrows():
        name = row["Name"].strip()
        print("Processing:", name)
        if name not in name2params.keys():
            print(f"Column '{name}' not found in name2params. Skipping...")
            continue
        if name in ignores:
            print(f"Warning: {name} was ignored. Skipping.")
            continue
        acc_param_dict[name] = {
            "param": name2params[name],
            "acc": row["best_val_accuracy"]
        }
    acc_param_dict = dict(sorted(acc_param_dict.items(), key=lambda item: item[0]))
    return acc_param_dict


if __name__ == '__main__':
    # Example usage
    args = parse_args()
    acc_param_dict = read_data_from_file(args.data_path)
    y_label = "Validation accuracy (%)"
    plot_acc_vs_params(acc_param_dict, y_label=y_label, saving_path=args.save_path)
    print(f"Plot saved to {args.save_path}")