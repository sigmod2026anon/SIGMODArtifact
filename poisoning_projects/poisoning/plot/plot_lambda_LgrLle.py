import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from load_loss import load_loss
from load_upper_bound import load_upper_bound
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE, LOSS_COLUMN, DATASET_NAMES,
    BOXPLOT, ERROR_BARS
)

VERBOSE = False

XTICK_INTERVAL = 2

def plot_lambda_LgrLle(df_results, fig_path, n=None, plot_type=None):
    """
    Plot for each n. The x-axis of each graph is lambda. The y-axis of each graph displays the value obtained by dividing the "poisoning loss" by the "legitimate loss".
    
    Args:
        df_results: Result dataframe
        fig_path: Output file path
        n: Plot only specific n values
        plot_type: Specify 'boxplot' or 'scatter'
    """
    # Determine plot_type based on settings
    if plot_type is None:
        plot_type = 'boxplot' if BOXPLOT else 'scatter'

    if n is not None:
        df_results = df_results[df_results['n'] == n]

    available_combinations = [
        (name, dtype, R) for name, dtype, R in df_results[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])
    
    # Fixed distribution order (leave blank if exponential is not present)
    fixed_distributions = [
        key for key in sorted(DATASET_NAMES.keys(), key=lambda x: DATASET_NAMES[x][0])
    ]
    distributions = fixed_distributions
    n_values = sorted(df_results['n'].unique())
    
    # Fixed n value list (include cases where data does not exist)
    target_n_values = [1000]
    n_values = target_n_values

    # Calculate y-axis range for all data
    all_ratios = []
    for dist in distributions:
        for n_val in n_values:
            dataset_name, data_type, R = dist

            # Get data for this distribution
            dist_data = df_results[
                (df_results['dataset_name'] == dataset_name) &
                (df_results['data_type'] == data_type) &
                (df_results['R'] == R) &
                (df_results['n'] == n_val)
            ]

            # Get legitimate data (lambda=0)
            legitimate_data = dist_data[dist_data['lambda'] == 0]

            # Get poisoning data (lambda>0)
            poison_data = dist_data[dist_data['lambda'] > 0]

            if len(legitimate_data) == 0 or len(poison_data) == 0:
                continue

            # Save legitimate loss for each seed as a dictionary
            legitimate_losses = {}
            for _, row in legitimate_data.iterrows():
                legitimate_losses[row['seed']] = row[LOSS_COLUMN]

            # Calculate ratio
            for _, row in poison_data.iterrows():
                seed = row['seed']
                if seed in legitimate_losses:
                    ratio = row[LOSS_COLUMN] / legitimate_losses[seed]
                    all_ratios.append(ratio)

    # Set y-axis range
    if all_ratios:
        min_ratio = min(all_ratios)
        max_ratio = max(all_ratios)
    else:
        min_ratio, max_ratio = 1, 10

    plt.rcParams['figure.figsize'] = [5 * len(distributions) + 0.5, 5 * len(n_values)]
    fig, axes = plt.subplots(len(n_values), len(distributions))

    # Handle case where axes is 1D
    if len(n_values) == 1:
        if len(distributions) == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(1, -1)
    elif len(distributions) == 1:
        axes = axes.reshape(-1, 1)

    for i, n_val in enumerate(n_values):
        for j, dist in enumerate(distributions):
            ax = axes[i][j]
            dataset_name, data_type, R = dist

            # Get data for this distribution
            dist_data = df_results[
                (df_results['dataset_name'] == dataset_name) &
                (df_results['data_type'] == data_type) &
                (df_results['R'] == R) &
                (df_results['n'] == n_val)
            ]

            # If no data, create empty plot
            if len(dist_data) == 0:
                ax.set_xlim(0, 15)  # Default x-axis range
                ax.set_ylim(min_ratio * 0.5, max_ratio * 2.0)
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', linewidth=0.8)
                ax.tick_params(axis='both', labelsize=TICK_SIZE)
                ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
                if i == 0:
                    ax.set_title(DATASET_NAMES[dist][1], fontsize=FONT_SIZE)
                if j == 0:
                    if len(n_values) == 1:
                        ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)
                    else:
                        ax.set_ylabel(rf'$\mathbf{{n={n_val}}}$' + '\n' + r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)
                continue

            # Get legitimate data (lambda=0)
            legitimate_data = dist_data[dist_data['lambda'] == 0]

            # Get poisoning data (lambda>0)
            poison_data = dist_data[dist_data['lambda'] > 0]

            if len(legitimate_data) == 0 or len(poison_data) == 0:
                print(f"[Warning] No data for distribution {dist}, n={n_val}")
                # Set title and y-axis label even if data does not exist
                ax.set_xlim(0, 15)  # Default x-axis range
                ax.set_ylim(min_ratio * 0.5, max_ratio * 2.0)
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', linewidth=0.8)
                ax.tick_params(axis='both', labelsize=TICK_SIZE)
                ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
                if i == 0:
                    ax.set_title(DATASET_NAMES[dist][1], fontsize=FONT_SIZE)
                if j == 0:
                    if len(n_values) == 1:
                        ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)
                    else:
                        ax.set_ylabel(rf'$\mathbf{{n={n_val}}}$' + '\n' + r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)
                continue

            # Save legitimate loss for each seed as a dictionary
            legitimate_losses = {}
            for _, row in legitimate_data.iterrows():
                legitimate_losses[row['seed']] = row[LOSS_COLUMN]

            # Get lambda values
            lambda_values = sorted(poison_data['lambda'].unique())

            if plot_type == 'boxplot':
                # Prepare data for boxplot
                box_data = []
                box_positions = []
                box_labels = []
                
                for lambda_val in lambda_values:
                    lambda_data = poison_data[poison_data['lambda'] == lambda_val]
                    ratios = []
                    
                    for _, row in lambda_data.iterrows():
                        seed = row['seed']
                        if seed in legitimate_losses:
                            ratio = row[LOSS_COLUMN] / legitimate_losses[seed]
                            ratios.append(ratio)
                    
                    if ratios:
                        box_data.append(ratios)
                        percentage = lambda_val / n_val * 100
                        box_positions.append(percentage)
                        box_labels.append(f'{percentage:.1f}%')
                
                if box_data:
                    # Draw boxplot (widen the width)
                    bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True, widths=0.8)
                    
                    # Set color of boxplot (white fill)
                    for patch in bp['boxes']:
                        patch.set_facecolor('white')
                        patch.set_alpha(1.0)
                    
                    # Make boxplot lines thicker
                    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
                        plt.setp(bp[element], linewidth=2)
                    
                    # Output number of samples for each boxplot
                    if VERBOSE:
                        for k, data in enumerate(box_data):
                            print(f"Boxplot - {DATASET_NAMES[dist][1]}, n={n_val}, percentage={box_positions[k]:.1f}%: {len(data)} samples")
                    
                    # ax.set_xticks(box_positions)
                    # ax.set_xticklabels(box_labels, rotation=45)
                    ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                    # Display x-axis labels that are multiples of XTICK_INTERVAL
                    xticks_positions = [i_ * XTICK_INTERVAL for i_ in range(int(min(box_positions) // XTICK_INTERVAL), int(max(box_positions) // XTICK_INTERVAL) + 1)]
                    ax.set_xticks(xticks_positions)
                    ax.set_xticklabels([f'{x}' for x in xticks_positions])
                    
            else:  # plot_type == 'scatter'
                # Prepare data for each lambda, for each seed
                seed_data = {}
                scatter_x = []
                scatter_y = []

                for lambda_val in lambda_values:
                    lambda_data = poison_data[poison_data['lambda'] == lambda_val]

                    for _, row in lambda_data.iterrows():
                        seed = row['seed']
                        if seed in legitimate_losses:
                            ratio = row[LOSS_COLUMN] / legitimate_losses[seed]
                            # Convert lambda to percentage
                            percentage = lambda_val / n_val * 100
                            
                            # Save data for each seed
                            if seed not in seed_data:
                                seed_data[seed] = {'x': [], 'y': []}
                            seed_data[seed]['x'].append(percentage)
                            seed_data[seed]['y'].append(ratio)
                            
                            # Save data for scatter plot
                            scatter_x.append(percentage)
                            scatter_y.append(ratio)

                if len(scatter_x) > 0:
                    # Connect data points for the same seed (transparent line)
                    for seed, data in seed_data.items():
                        if len(data['x']) > 1:  # Draw line only if there are at least 2 points
                            # Sort x-coordinates and draw line
                            sorted_indices = np.argsort(data['x'])
                            sorted_x = [data['x'][i] for i in sorted_indices]
                            sorted_y = [data['y'][i] for i in sorted_indices]
                            ax.plot(sorted_x, sorted_y, color='blue', alpha=0.3, linewidth=1)
                    
                    # Draw scatter plot (blue cross)
                    ax.scatter(scatter_x, scatter_y, marker='x', color='blue', s=80, alpha=0.8)
                    
                    # Output number of samples for each point
                    if VERBOSE:
                        print(f"Scatter - {DATASET_NAMES[dist][1]}, n={n_val}: {len(scatter_x)} samples")

            # Common settings
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
            if i == 0:
                ax.set_title(DATASET_NAMES[dist][1], fontsize=FONT_SIZE)

            if j == 0:
                if len(n_values) == 1:
                    ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)
                else:
                    ax.set_ylabel(rf'$\mathbf{{n={n_val}}}$' + '\n' + r'$\frac{\mathrm{MSE}_\mathrm{G}}{\mathrm{MSE}_\mathrm{L}}$', fontsize=FONT_SIZE*1.5)

            # Unify y-axis range
            ax.set_ylim(min_ratio * 0.5, max_ratio * 2.0)

    # Place legend in the lower right corner of the graph (if applicable)
    if len(axes) > 0:
        handles, labels = axes[len(n_values) - 1][len(distributions) - 1].get_legend_handles_labels()
        if handles and labels:  # Display legend only if it exists
            axes[len(n_values) - 1][len(distributions) - 1].legend(handles, labels, 
                                             loc='lower right',
                                             frameon=True,
                                             facecolor='white',
                                             edgecolor='black',
                                             fontsize=LEGEND_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")

if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/lambda_LgrLle"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    df_results = load_loss(data_dir)

    # Plot all (including multiple n values)
    ns = [1000]
    df_results_ns = df_results[df_results['n'].isin(ns)]
    
    # Plot based on settings
    plot_lambda_LgrLle(df_results_ns, f'{output_dir}/lambda_lambda_LgrLle_all.pdf', n=None, plot_type=None)
