import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from load_loss_comparison import load_loss_comparison_data
from load_optimal_poison import load_optimal_poison
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, DATASET_NAMES_BRUTE_FORCE,
    BOXPLOT, ERROR_BARS
)

VERBOSE = False

# Boxplot spacing and width configuration
BOXPLOT_WIDTH = 1.0    # Width of each boxplot
BOXPLOT_ALPHA = 1.0
BOXPLOT_LINEWIDTH = 2
ROW_HEIGHT = 4.5
COLUMN_WIDTH = 5

XTICK_INTERVAL = 2

def plot_lambda_LconsecLbruteforce(consecutive_w_endpoints_df, df_optimal_poison, fig_path, n=None, show_boxplot=None):
    """
    Plot for each n. The x-axis of each graph is lambda. The y-axis displays MSE_SEG+E/MSE_BF.
    
    Args:
        consecutive_w_endpoints_df: Consecutive with endpoints loss dataframe
        df_optimal_poison: Optimal poison dataframe (brute force results)
        fig_path: Output file path
        n: Plot only specific n values
        show_boxplot: Specify 'boxplot' or 'scatter'
    """
    # Determine show_boxplot based on settings
    if show_boxplot is None:
        show_boxplot = BOXPLOT

    if n is not None:
        consecutive_w_endpoints_df = consecutive_w_endpoints_df[consecutive_w_endpoints_df['n'] == n]
        df_optimal_poison = df_optimal_poison[df_optimal_poison['n'] == n]

    # Combine data for finding available combinations
    combined_df = pd.concat([consecutive_w_endpoints_df, df_optimal_poison], ignore_index=True)
    available_combinations = [
        (name, dtype, R) for name, dtype, R in combined_df[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES_BRUTE_FORCE
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES_BRUTE_FORCE[x][0])

    # distributions = [distributions[0]]

    plt.rcParams['figure.figsize'] = [COLUMN_WIDTH * len(distributions) + 0.5, ROW_HEIGHT * 1]
    fig, axes = plt.subplots(1, len(distributions))

    # Handle case where axes is 1D
    if len(distributions) == 1:
        axes = [axes]

    # Settings for LconsecE/LbruteForce ratio plots
    # Collect all ratios for overall statistics
    all_lconsecE_lbruteforce_ratios = []
    y_min_0 = float('inf')
    y_max_0 = float('-inf')
    
    for j, dist in enumerate(distributions):
        dataset_name, data_type, R = dist

        print(f"dataset_name={dataset_name}, data_type={data_type}, R={R}")
        
        # Get data for consecutive_w_endpoints approach
        consecutive_w_endpoints_data = consecutive_w_endpoints_df[(consecutive_w_endpoints_df['dataset_name'] == dataset_name) & 
                                                                (consecutive_w_endpoints_df['data_type'] == data_type) & 
                                                                (consecutive_w_endpoints_df['n'] == 50) & 
                                                                (consecutive_w_endpoints_df['lambda'] > 0) & 
                                                                (consecutive_w_endpoints_df['R'] == R)]
        
        # Get brute force data (duplicate not allowed)
        brute_force_data = df_optimal_poison[(df_optimal_poison['dataset_name'] == dataset_name) & 
                                           (df_optimal_poison['data_type'] == data_type) & 
                                           (df_optimal_poison['n'] == 50) & 
                                           (df_optimal_poison['R'] == R) & 
                                           (df_optimal_poison['algorithm'] == 'brute_force')]
        
        if consecutive_w_endpoints_data.empty or brute_force_data.empty:
            print(f"Warning: Missing data for {dataset_name}, {data_type}, n=50, R={R}")
            continue
        
        # Merge data on lambda and seed
        consecutive_w_endpoints_data = consecutive_w_endpoints_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
        brute_force_data = brute_force_data[['lambda', 'seed', 'loss']].sort_values(by=['lambda', 'seed'])
        
        # Rename columns for merge
        consecutive_w_endpoints_data = consecutive_w_endpoints_data.rename(columns={LOSS_COLUMN: 'consecutive_w_endpoints_loss'})
        brute_force_data = brute_force_data.rename(columns={'loss': 'brute_force_loss'})
        
        # Merge datasets
        merged_data = consecutive_w_endpoints_data.merge(brute_force_data, on=['lambda', 'seed'], how='inner')
        
        if merged_data.empty:
            print(f"Warning: No matching data after merge for {dataset_name}, {data_type}, n=50, R={R}")
            continue
        
        # Convert lambda to percentage
        merged_data['percentage'] = merged_data['lambda'] / 50 * 100
        merged_data['LconsecE_divided_by_LbruteForce'] = merged_data['consecutive_w_endpoints_loss'] / merged_data['brute_force_loss']
        
        # # Print minimum/maximum LconsecE/LbruteForce values for each lambda (percentage) and their corresponding seeds
        # print(f"\n=== Minimum LconsecE/LbruteForce values for {dataset_name}, {data_type}, R={R} ===")
        # for percentage in sorted(merged_data['percentage'].unique()):
        #     group_data = merged_data[merged_data['percentage'] == percentage]
        #     min_idx = group_data['LconsecE_divided_by_LbruteForce'].idxmin()
        #     min_row = group_data.loc[min_idx]
        #     print(f"  Lambda={min_row['lambda']} ({percentage:.1f}%): Min LconsecE/LbruteForce = {min_row['LconsecE_divided_by_LbruteForce']:.30f} (seed={min_row['seed']})")
        #     print(f"    Consecutive with endpoints loss: {min_row['consecutive_w_endpoints_loss']:.30f}, Brute force loss: {min_row['brute_force_loss']:.30f}")
        
        # print(f"\n=== Maximum LconsecE/LbruteForce values for {dataset_name}, {data_type}, R={R} ===")
        # for percentage in sorted(merged_data['percentage'].unique()):
        #     group_data = merged_data[merged_data['percentage'] == percentage]
        #     max_idx = group_data['LconsecE_divided_by_LbruteForce'].idxmax()
        #     max_row = group_data.loc[max_idx]
        #     print(f"  Lambda={max_row['lambda']} ({percentage:.1f}%): Max LconsecE/LbruteForce = {max_row['LconsecE_divided_by_LbruteForce']:.30f} (seed={max_row['seed']})")
        #     print(f"    Consecutive with endpoints loss: {max_row['consecutive_w_endpoints_loss']:.30f}, Brute force loss: {max_row['brute_force_loss']:.30f}")

        # Update global statistics
        all_lconsecE_lbruteforce_ratios.extend(merged_data['LconsecE_divided_by_LbruteForce'].tolist())
        
        # Update global y-axis limits
        y_min_0 = min(y_min_0, merged_data['LconsecE_divided_by_LbruteForce'].min())
        y_max_0 = max(y_max_0, merged_data['LconsecE_divided_by_LbruteForce'].max())

        # Plot LconsecE/LbruteForce ratio
        ax = axes[j]
        if show_boxplot:
            # Boxplot version for LconsecE/LbruteForce
            box_data_bruteforce = []
            box_positions = []
            for percentage in sorted(merged_data['percentage'].unique()):
                group_data = merged_data[merged_data['percentage'] == percentage]['LconsecE_divided_by_LbruteForce'].values
                if len(group_data) > 0:
                    box_data_bruteforce.append(group_data)
                    box_positions.append(percentage)
            
            if box_data_bruteforce:
                # Draw boxplot for LconsecE/LbruteForce
                bp = ax.boxplot(box_data_bruteforce, positions=box_positions, patch_artist=True, widths=BOXPLOT_WIDTH)
                
                # Set color of boxplot (white fill)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(BOXPLOT_ALPHA)
                
                # Make boxplot lines thicker
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
                    plt.setp(bp[element], linewidth=BOXPLOT_LINEWIDTH)
                
                # Set x-axis
                ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                # Display x-axis labels - use actual data points if range is small, otherwise use multiples of 5
                if max(box_positions) - min(box_positions) <= 10:
                    # For small ranges, show all data points
                    ax.set_xticks(box_positions)
                    ax.set_xticklabels([f'{x:.1f}' for x in box_positions])
                else:
                    # For larger ranges, show multiples of XTICK_INTERVAL
                    xticks_positions = [i_ * XTICK_INTERVAL for i_ in range(int(min(box_positions) // XTICK_INTERVAL), int(max(box_positions) // XTICK_INTERVAL) + 1)]
                    ax.set_xticks(xticks_positions)
                    ax.set_xticklabels([f'{x}' for x in xticks_positions])
        else:
            # Mean/Min/Max version for LconsecE/LbruteForce
            stats_bruteforce = merged_data.groupby('percentage')['LconsecE_divided_by_LbruteForce'].agg(['mean', 'min', 'max'])
            
            ax.plot(stats_bruteforce.index, stats_bruteforce['mean'], 's-', label='Mean', color='blue', linewidth=2)
            ax.plot(stats_bruteforce.index, stats_bruteforce['min'], 'v:', label='Min', color='blue', linewidth=2)
            ax.plot(stats_bruteforce.index, stats_bruteforce['max'], 'v-.', label='Max', color='blue', linewidth=2)
        
        # Common settings for LconsecE/LbruteForce plot
        ax.grid(True, which='both', linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{DATASET_NAMES_BRUTE_FORCE[dist][1]}', fontsize=XLABEL_SIZE)
        ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
        if j == 0:
            ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{SEG{+}E}}{\mathrm{MSE}_\mathrm{OPT}}$', fontsize=FONT_SIZE*1.5)

        # Add horizontal line at y=1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits across all subplots
    # y_margin_0 = (y_max_0 - y_min_0) * 0.1
    # for j in range(len(distributions)):
    #     axes[j].set_ylim(max(0, y_min_0 - y_margin_0), y_max_0 + y_margin_0)

    # Place legend in the lower right corner of the graph (if applicable)
    if not show_boxplot:
        handles, labels = axes[len(distributions) - 1].get_legend_handles_labels()
        axes[len(distributions) - 1].legend(handles, labels, 
                                    loc='lower right',
                                    frameon=True,
                                    facecolor='white',
                                    edgecolor='black',
                                    fontsize=LEGEND_SIZE)
    
    # Print overall statistics for LconsecE/LbruteForce ratio
    if all_lconsecE_lbruteforce_ratios:
        overall_min = min(all_lconsecE_lbruteforce_ratios)
        overall_max = max(all_lconsecE_lbruteforce_ratios)
        overall_mean = sum(all_lconsecE_lbruteforce_ratios) / len(all_lconsecE_lbruteforce_ratios)
        print(f"Overall LconsecE/LbruteForce ratio statistics:")
        print(f"  Minimum: {overall_min:.12f}")
        print(f"  Maximum: {overall_max:.12f}")
        print(f"  Mean: {overall_mean:.12f}")
        print(f"  Total samples: {len(all_lconsecE_lbruteforce_ratios)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")


if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/lambda_LconsecLbruteforce"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    
    # Load data
    # print("Loading consecutive with endpoints data...")
    consecutive_w_endpoints_df = load_loss_comparison_data(data_dir, "consecutive_w_endpoints")
    
    # print("Loading optimal poison data...")
    df_optimal_poison = load_optimal_poison(data_dir)
    
    if consecutive_w_endpoints_df.empty or df_optimal_poison.empty:
        print("Error: No data loaded")
        exit(1)
    
    # Filter for n=50 only
    ns = [50]
    consecutive_w_endpoints_df_ns = consecutive_w_endpoints_df[consecutive_w_endpoints_df['n'].isin(ns)]
    df_optimal_poison_ns = df_optimal_poison[df_optimal_poison['n'].isin(ns)]
    
    # Plot based on settings
    if BOXPLOT:
        plot_lambda_LconsecLbruteforce(consecutive_w_endpoints_df_ns, df_optimal_poison_ns, 
                                      f'{output_dir}/lambda_Lconsec_divided_by_Lbruteforce_all.pdf', 
                                      n=None, show_boxplot='boxplot')
    else:
        plot_lambda_LconsecLbruteforce(consecutive_w_endpoints_df_ns, df_optimal_poison_ns, 
                                      f'{output_dir}/lambda_Lconsec_divided_by_Lbruteforce_all.pdf', 
                                      n=None, show_boxplot='scatter')
