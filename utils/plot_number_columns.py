import pandas as pd
import matplotlib.pyplot as plt
import argparse


markers = [4, "o", 2]
markevery = [0, 40, 80, 120, 160, 199]


def plot_initial_vs_final_columns(data_path, save_path):
    # Read the CSV
    df = pd.read_csv(data_path)
    print(df.head())


    # Extract relevant columns
    h_cols = [col for col in df.columns if "h_dict" in col and "MIN" not in col and "MAX" not in col]

    layers = []
    initial_values = []
    final_values = []

    for col in h_cols:
        layers.append(col.split("- h_dict.")[-1])
        initial_values.append(df[col].iloc[0] / 100)  # initial (epoch 0)
        final_values.append(df[col].iloc[-1] / 100)   # final (last epoch)

    sorted_data = sorted(zip(layers, initial_values, final_values), key=lambda x: x[0])
    layers, initial_values, final_values = zip(*sorted_data)

    indices = list(range(1, len(layers)+1))
    x_positions = indices
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(3, 2.5))
    # Increase hatch density for both bars
    ax.bar([x - bar_width/2 for x in x_positions], initial_values, width=bar_width, color='skyblue', hatch='////', label='Initial')
    ax.bar([x + bar_width/2 for x in x_positions], final_values, width=bar_width, color='steelblue', hatch='\\\\\\\\', label='Final')

    ax.set_xticks(list([len(indices)+1-x for x in x_positions]))
    ax.set_xticklabels(indices, rotation=45, fontsize=7)
    ax.set_xlabel(r"$i$-th LR-MTL Layer", fontsize=7)
    ax.set_ylabel(r"Number of Columns ($x10^2$)", fontsize=7)
    # ax.set_title("Initial vs Final Columns per Layer", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved to {save_path}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot initial vs final number of columns per layer.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output plot.")
    args = parser.parse_args()

    plot_initial_vs_final_columns(args.data_path, args.save_path)