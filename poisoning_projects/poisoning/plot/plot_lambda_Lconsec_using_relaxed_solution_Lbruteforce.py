import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from load_loss_comparison import load_loss_comparison_data
from load_loss import load_loss
from load_optimal_poison import load_optimal_poison
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, DATASET_NAMES_BRUTE_FORCE,
    BOXPLOT, ERROR_BARS
)


VERBOSE = False

# Boxplot spacing and width configuration
BOXPLOT_WIDTH = 1.0
BOXPLOT_ALPHA = 1.0
BOXPLOT_LINEWIDTH = 2
ROW_HEIGHT = 4.5
COLUMN_WIDTH = 5

XTICK_INTERVAL = 2

def plot_lambda_Lconsec_using_relaxed_solution_Lbruteforce(relaxed_solution_df, df_optimal_poison, fig_path, n=None, show_boxplot=None):
    if show_boxplot is None:
        show_boxplot = BOXPLOT

    if n is not None:
        relaxed_solution_df = relaxed_solution_df[relaxed_solution_df['n'] == n]
        df_optimal_poison = df_optimal_poison[df_optimal_poison['n'] == n]

    combined_df = pd.concat([relaxed_solution_df, df_optimal_poison], ignore_index=True)
    available_combinations = [
        (name, dtype, R) for name, dtype, R in combined_df[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES_BRUTE_FORCE
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES_BRUTE_FORCE[x][0])

    if len(distributions) == 0:
        print("No distributions to plot (no matching DATASET_NAMES_BRUTE_FORCE entries).")
        return

    plt.rcParams['figure.figsize'] = [COLUMN_WIDTH * len(distributions) + 0.5, ROW_HEIGHT * 1]
    fig, axes = plt.subplots(1, len(distributions))
    if len(distributions) == 1:
        axes = [axes]

    all_ratios = []
    y_min = float('inf')
    y_max = float('-inf')

    for j, dist in enumerate(distributions):
        dataset_name, data_type, R = dist
        print(f"dataset_name={dataset_name}, data_type={data_type}, R={R}")

        relaxed_data = relaxed_solution_df[(relaxed_solution_df['dataset_name'] == dataset_name) &
                                          (relaxed_solution_df['data_type'] == data_type) &
                                          (relaxed_solution_df['n'] == 50) &
                                          (relaxed_solution_df['lambda'] > 0) &
                                          (relaxed_solution_df['R'] == R)]

        brute_force_data = df_optimal_poison[(df_optimal_poison['dataset_name'] == dataset_name) &
                                            (df_optimal_poison['data_type'] == data_type) &
                                            (df_optimal_poison['n'] == 50) &
                                            (df_optimal_poison['R'] == R) &
                                            (df_optimal_poison['algorithm'] == 'brute_force')]

        if relaxed_data.empty or brute_force_data.empty:
            print(f"Warning: Missing data for {dataset_name}, {data_type}, n=50, R={R}")
            continue

        relaxed_data = relaxed_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
        brute_force_data = brute_force_data[['lambda', 'seed', 'loss']].sort_values(by=['lambda', 'seed'])

        relaxed_data = relaxed_data.rename(columns={LOSS_COLUMN: 'relaxed_solution_loss'})
        brute_force_data = brute_force_data.rename(columns={'loss': 'brute_force_loss'})

        merged = relaxed_data.merge(brute_force_data, on=['lambda', 'seed'], how='inner')

        if merged.empty:
            print(f"Warning: No matching data after merge for {dataset_name}, {data_type}, n=50, R={R}")
            continue

        merged['percentage'] = merged['lambda'] / 50 * 100
        merged['relaxed_divided_by_bruteforce'] = merged['relaxed_solution_loss'] / merged['brute_force_loss']

        all_ratios.extend(merged['relaxed_divided_by_bruteforce'].tolist())
        y_min = min(y_min, merged['relaxed_divided_by_bruteforce'].min())
        y_max = max(y_max, merged['relaxed_divided_by_bruteforce'].max())

        ax = axes[j]
        if show_boxplot:
            box_data = []
            box_positions = []
            for percentage in sorted(merged['percentage'].unique()):
                vals = merged[merged['percentage'] == percentage]['relaxed_divided_by_bruteforce'].values
                if len(vals) > 0:
                    box_data.append(vals)
                    box_positions.append(percentage)
            if box_data:
                bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True, widths=BOXPLOT_WIDTH)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(BOXPLOT_ALPHA)
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
                    plt.setp(bp[element], linewidth=BOXPLOT_LINEWIDTH)
                ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                if max(box_positions) - min(box_positions) <= 10:
                    ax.set_xticks(box_positions)
                    ax.set_xticklabels([f'{x:.1f}' for x in box_positions])
                else:
                    xticks_positions = [i_ * XTICK_INTERVAL for i_ in range(int(min(box_positions) // XTICK_INTERVAL), int(max(box_positions) // XTICK_INTERVAL) + 1)]
                    ax.set_xticks(xticks_positions)
                    ax.set_xticklabels([f'{x}' for x in xticks_positions])
        else:
            stats = merged.groupby('percentage')['relaxed_divided_by_bruteforce'].agg(['mean', 'min', 'max'])
            ax.plot(stats.index, stats['mean'], 's-', label='Mean', color='blue', linewidth=2)
            ax.plot(stats.index, stats['min'], 'v:', label='Min', color='blue', linewidth=2)
            ax.plot(stats.index, stats['max'], 'v-.', label='Max', color='blue', linewidth=2)

        ax.grid(True, which='both', linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{DATASET_NAMES_BRUTE_FORCE[dist][1]}', fontsize=XLABEL_SIZE)
        ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
        if j == 0:
            ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{SEG{+}E(\mathrm{Heu.})}}{\mathrm{MSE}_\mathrm{OPT}}$', fontsize=FONT_SIZE*1.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)

    for j in range(len(distributions)):
        axes[j].set_ylim(0.0, 1.1)

    if not show_boxplot:
        handles, labels = axes[len(distributions) - 1].get_legend_handles_labels()
        axes[len(distributions) - 1].legend(handles, labels,
                                           loc='lower right',
                                           frameon=True,
                                           facecolor='white',
                                           edgecolor='black',
                                           fontsize=LEGEND_SIZE)

    if all_ratios:
        overall_min = min(all_ratios)
        overall_max = max(all_ratios)
        overall_mean = sum(all_ratios) / len(all_ratios)
        print("Overall relaxed/L_opt ratio statistics:")
        print(f"  Minimum: {overall_min:.12f}")
        print(f"  Maximum: {overall_max:.12f}")
        print(f"  Mean: {overall_mean:.12f}")
        print(f"  Total samples: {len(all_ratios)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    output_dir = "../results/fig/lambda_Lconsec_using_relaxed_solution_Lbruteforce"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = ".."

    print("Loading comparison data...")
    consecutive_w_endpoints_using_relaxed_solution_df = load_loss(data_dir, approach="consecutive_w_endpoints_using_relaxed_solution")


    print("Loading optimal poison data...")
    df_optimal_poison = load_optimal_poison(data_dir)

    if consecutive_w_endpoints_using_relaxed_solution_df.empty or df_optimal_poison.empty:
        print("Error: No data loaded")
        exit(1)

    ns = [50]
    consecutive_w_endpoints_using_relaxed_solution_df_ns = consecutive_w_endpoints_using_relaxed_solution_df[consecutive_w_endpoints_using_relaxed_solution_df['n'].isin(ns)]
    df_optimal_poison_ns = df_optimal_poison[df_optimal_poison['n'].isin(ns)]

    if BOXPLOT:
        plot_lambda_Lconsec_using_relaxed_solution_Lbruteforce(consecutive_w_endpoints_using_relaxed_solution_df, df_optimal_poison_ns,
                                                               f'{output_dir}/lambda_Lconsec_using_relaxed_solution_Lbruteforce_all_boxplot.pdf',
                                                               n=None, show_boxplot='boxplot')
    else:
        plot_lambda_Lconsec_using_relaxed_solution_Lbruteforce(consecutive_w_endpoints_using_relaxed_solution_df, df_optimal_poison_ns,
                                                               f'{output_dir}/lambda_Lconsec_using_relaxed_solution_Lbruteforce_all.pdf',
                                                               n=None, show_boxplot='error_bars')
