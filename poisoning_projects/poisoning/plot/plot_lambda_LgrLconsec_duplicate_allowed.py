import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from load_loss import load_loss
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, DATASET_NAMES, BOXPLOT, ERROR_BARS
)

VERBOSE = False
ROW_HEIGHT = 4.55
COLUMN_WIDTH = 5

XTICK_INTERVAL = 5

def plot_lambda_LgrLconsec_duplicate_allowed(original_duplicate_allowed_df, consecutive_w_endpoints_duplicate_allowed_df, fig_path, n=None, show_boxplot=None):
    """
    Plot for each n. The x-axis of each graph is lambda. The y-axis displays MSE_G/MSE_{CONSEC_E} for duplicate_allowed case.
    
    Args:
        original_duplicate_allowed_df: Original duplicate_allowed loss dataframe
        consecutive_w_endpoints_duplicate_allowed_df: Consecutive with endpoints duplicate_allowed loss dataframe
        fig_path: Output file path
        n: Plot only specific n values
        show_boxplot: Specify 'boxplot' or 'scatter'
    """
    # Determine show_boxplot based on settings
    if show_boxplot is None:
        show_boxplot = BOXPLOT

    if n is not None:
        original_duplicate_allowed_df = original_duplicate_allowed_df[original_duplicate_allowed_df['n'] == n]
        consecutive_w_endpoints_duplicate_allowed_df = consecutive_w_endpoints_duplicate_allowed_df[consecutive_w_endpoints_duplicate_allowed_df['n'] == n]

    # Combine data for finding available combinations
    combined_df = pd.concat([original_duplicate_allowed_df, consecutive_w_endpoints_duplicate_allowed_df], ignore_index=True)
    available_combinations = [
        (name, dtype, R) for name, dtype, R in combined_df[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])

    plt.rcParams['figure.figsize'] = [COLUMN_WIDTH * len(distributions) + 0.5, ROW_HEIGHT * 1]
    fig, axes = plt.subplots(1, len(distributions))

    # Handle case where axes is 1D
    if len(distributions) == 1:
        axes = axes.reshape(1, 1)

    # Settings for Lgr/LconsecE ratio plots (duplicate_allowed)
    # Collect all ratios for overall statistics
    all_lgr_lconsecE_ratios = []
    y_min_0 = float('inf')
    y_max_0 = float('-inf')
    
    for j, dist in enumerate(distributions):
        dataset_name, data_type, R = dist
        
        # Get data for original and consecutive_w_endpoints duplicate_allowed approaches
        original_duplicate_allowed_data = original_duplicate_allowed_df[(original_duplicate_allowed_df['dataset_name'] == dataset_name) & 
                                  (original_duplicate_allowed_df['data_type'] == data_type) & 
                                  (original_duplicate_allowed_df['n'] == 1000) & 
                                  (original_duplicate_allowed_df['lambda'] > 0) & 
                                  (original_duplicate_allowed_df['R'] == R)]
        
        consecutive_w_endpoints_duplicate_allowed_data = consecutive_w_endpoints_duplicate_allowed_df[(consecutive_w_endpoints_duplicate_allowed_df['dataset_name'] == dataset_name) & 
                                                                (consecutive_w_endpoints_duplicate_allowed_df['data_type'] == data_type) & 
                                                                (consecutive_w_endpoints_duplicate_allowed_df['n'] == 1000) & 
                                                                (consecutive_w_endpoints_duplicate_allowed_df['lambda'] > 0) & 
                                                                (consecutive_w_endpoints_duplicate_allowed_df['R'] == R)]
        
        if original_duplicate_allowed_data.empty or consecutive_w_endpoints_duplicate_allowed_data.empty:
            print(f"Warning: Missing data for {dataset_name}, {data_type}, n=1000, R={R} (duplicate_allowed)")
            continue
        
        # Merge data on lambda and seed
        original_duplicate_allowed_data = original_duplicate_allowed_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
        consecutive_w_endpoints_duplicate_allowed_data = consecutive_w_endpoints_duplicate_allowed_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
        
        # Rename columns for merge
        original_duplicate_allowed_data = original_duplicate_allowed_data.rename(columns={LOSS_COLUMN: 'original_duplicate_allowed_loss'})
        consecutive_w_endpoints_duplicate_allowed_data = consecutive_w_endpoints_duplicate_allowed_data.rename(columns={LOSS_COLUMN: 'consecutive_w_endpoints_duplicate_allowed_loss'})
        
        # Merge datasets
        merged_data = original_duplicate_allowed_data.merge(consecutive_w_endpoints_duplicate_allowed_data, on=['lambda', 'seed'], how='inner')
        
        if merged_data.empty:
            print(f"Warning: No matching data after merge for {dataset_name}, {data_type}, n=1000, R={R} (duplicate_allowed)")
            continue
        
        # Convert lambda to percentage
        merged_data['percentage'] = merged_data['lambda'] / 1000 * 100
        merged_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'] = merged_data['original_duplicate_allowed_loss'] / merged_data['consecutive_w_endpoints_duplicate_allowed_loss']
        
        # Print minimum/maximum Lgr_duplicate_allowed/LconsecE_duplicate_allowed values for each lambda (percentage) and their corresponding seeds
        print(f"\n=== Minimum Lgr_duplicate_allowed/LconsecE_duplicate_allowed values for {dataset_name}, {data_type}, R={R} ===")
        for percentage in sorted(merged_data['percentage'].unique()):
            group_data = merged_data[merged_data['percentage'] == percentage]
            min_idx = group_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].idxmin()
            min_row = group_data.loc[min_idx]
            print(f"  Lambda={min_row['lambda']} ({percentage:.1f}%): Min Lgr_duplicate_allowed/LconsecE_duplicate_allowed = {min_row['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed']:.30f} (seed={min_row['seed']})")
            # print(f"    Original duplicate_allowed loss: {min_row['original_duplicate_allowed_loss']:.30f}, Consecutive with endpoints duplicate_allowed loss: {min_row['consecutive_w_endpoints_duplicate_allowed_loss']:.30f}")
        
        print(f"\n=== Maximum Lgr_duplicate_allowed/LconsecE_duplicate_allowed values for {dataset_name}, {data_type}, R={R} ===")
        for percentage in sorted(merged_data['percentage'].unique()):
            group_data = merged_data[merged_data['percentage'] == percentage]
            max_idx = group_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].idxmax()
            max_row = group_data.loc[max_idx]
            print(f"  Lambda={max_row['lambda']} ({percentage:.1f}%): Max Lgr_duplicate_allowed/LconsecE_duplicate_allowed = {max_row['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed']:.30f} (seed={max_row['seed']})")
            # print(f"    Original duplicate_allowed loss: {max_row['original_duplicate_allowed_loss']:.30f}, Consecutive with endpoints duplicate_allowed loss: {max_row['consecutive_w_endpoints_duplicate_allowed_loss']:.30f}")

        # Update global statistics
        all_lgr_lconsecE_ratios.extend(merged_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].tolist())
        
        # Update global y-axis limits
        y_min_0 = min(y_min_0, merged_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].min())
        y_max_0 = max(y_max_0, merged_data['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].max())

        # Plot Lgr_duplicate_allowed/LconsecE_duplicate_allowed ratio
        ax = axes[j]
        if show_boxplot:
            # Boxplot version for Lgr_duplicate_allowed/LconsecE_duplicate_allowed
            box_data_lconsecE_duplicate_allowed = []
            box_positions = []
            for percentage in sorted(merged_data['percentage'].unique()):
                group_data = merged_data[merged_data['percentage'] == percentage]['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].values
                if len(group_data) > 0:
                    box_data_lconsecE_duplicate_allowed.append(group_data)
                    box_positions.append(percentage)
            
            if box_data_lconsecE_duplicate_allowed:
                # Draw boxplot for Lgr_duplicate_allowed/LconsecE_duplicate_allowed
                bp = ax.boxplot(box_data_lconsecE_duplicate_allowed, positions=box_positions, patch_artist=True, widths=0.8)
                
                # Set color of boxplot (white fill)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(1.0)
                
                # Make boxplot lines thicker
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
                    plt.setp(bp[element], linewidth=2)
                
                # Set x-axis
                ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                # Display x-axis labels that are multiples of XTICK_INTERVAL
                xticks_positions = [i_ * XTICK_INTERVAL for i_ in range(int(min(box_positions) // XTICK_INTERVAL), int(max(box_positions) // XTICK_INTERVAL) + 1)]
                ax.set_xticks(xticks_positions)
                ax.set_xticklabels([f'{x}' for x in xticks_positions])
        else:
            # Mean/Min/Max version for Lgr_duplicate_allowed/LconsecE_duplicate_allowed
            stats_lconsecE_duplicate_allowed = merged_data.groupby('percentage')['Lgr_duplicate_allowed_divided_by_LconsecE_duplicate_allowed'].agg(['mean', 'min', 'max'])
            
            ax.plot(stats_lconsecE_duplicate_allowed.index, stats_lconsecE_duplicate_allowed['mean'], 's-', label='Mean', color='green', linewidth=2)
            ax.plot(stats_lconsecE_duplicate_allowed.index, stats_lconsecE_duplicate_allowed['min'], 'v:', label='Min', color='green', linewidth=2)
            ax.plot(stats_lconsecE_duplicate_allowed.index, stats_lconsecE_duplicate_allowed['max'], 'v-.', label='Max', color='green', linewidth=2)
        
        # Common settings for Lgr_duplicate_allowed/LconsecE_duplicate_allowed plot
        ax.grid(True, which='both', linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{DATASET_NAMES[dist][1]}', fontsize=XLABEL_SIZE)
        ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
        if j == 0:
            ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{G(R.)}}{\mathrm{MSE}_\mathrm{SEG{+}E(R.)}}$', fontsize=FONT_SIZE*1.5)

        # Add horizontal line at y=1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits across all subplots
    # y_margin_0 = (y_max_0 - y_min_0) * 0.1
    # for j in range(len(distributions)):
    #     axes[j].set_ylim(max(0, y_min_0 - y_margin_0), y_max_0 + y_margin_0)
    for j in range(len(distributions)):
        axes[j].set_ylim(0.78, 1.02)

    # Place legend in the lower right corner of the graph (if applicable)
    if not show_boxplot:
        handles, labels = axes[len(distributions) - 1].get_legend_handles_labels()
        axes[len(distributions) - 1].legend(handles, labels, 
                                    loc='lower right',
                                    frameon=True,
                                    facecolor='white',
                                    edgecolor='black',
                                    fontsize=LEGEND_SIZE)
    
    # Print overall statistics for Lgr_duplicate_allowed/LconsecE_duplicate_allowed ratio
    if all_lgr_lconsecE_ratios:
        overall_min = min(all_lgr_lconsecE_ratios)
        overall_max = max(all_lgr_lconsecE_ratios)
        overall_mean = sum(all_lgr_lconsecE_ratios) / len(all_lgr_lconsecE_ratios)
        print(f"Overall Lgr_duplicate_allowed/LconsecE_duplicate_allowed ratio statistics:")
        print(f"  Minimum: {overall_min:.30f}")
        print(f"  Maximum: {overall_max:.30f}")
        print(f"  Mean: {overall_mean:.30f}")
        print(f"  Total samples: {len(all_lgr_lconsecE_ratios)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")



if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/lambda_LgrLconsec_duplicate_allowed"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    
    # Load data using load_loss
    # print("Loading comparison duplicate_allowed data...")
    original_duplicate_allowed_df = load_loss(data_dir, approach="duplicate_allowed")
    consecutive_w_endpoints_duplicate_allowed_df = load_loss(data_dir, "consecutive_w_endpoints_duplicate_allowed")
    
    if original_duplicate_allowed_df.empty or consecutive_w_endpoints_duplicate_allowed_df.empty:
        print("Error: No data loaded")
        exit(1)
    
    # print(f"Loaded data:")
    # print(f"  Original duplicate_allowed: {len(original_duplicate_allowed_df)} rows")
    # print(f"  Consecutive with endpoints duplicate_allowed: {len(consecutive_w_endpoints_duplicate_allowed_df)} rows")
    
    # Filter for n=1000 only
    ns = [1000]
    original_duplicate_allowed_df_ns = original_duplicate_allowed_df[original_duplicate_allowed_df['n'].isin(ns)]
    consecutive_w_endpoints_duplicate_allowed_df_ns = consecutive_w_endpoints_duplicate_allowed_df[consecutive_w_endpoints_duplicate_allowed_df['n'].isin(ns)]
    
    # Plot based on settings
    if BOXPLOT:
        plot_lambda_LgrLconsec_duplicate_allowed(original_duplicate_allowed_df_ns, consecutive_w_endpoints_duplicate_allowed_df_ns, 
                              f'{output_dir}/lambda_Lgr_divided_by_Lconsec_duplicate_allowed_all.pdf', 
                              n=None, show_boxplot='boxplot')
    else:
        plot_lambda_LgrLconsec_duplicate_allowed(original_duplicate_allowed_df_ns, consecutive_w_endpoints_duplicate_allowed_df_ns, 
                              f'{output_dir}/lambda_Lgr_divided_by_Lconsec_duplicate_allowed_all.pdf', 
                              n=None, show_boxplot='scatter')

