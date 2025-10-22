import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from load_loss import load_loss
from load_upper_bound import load_upper_bound
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES,
    BOXPLOT, ERROR_BARS
)

VERBOSE = False
ROW_HEIGHT = 5
COLUMN_WIDTH = 5

XTICK_INTERVAL = 2

def plot_lambda_LconsecLub(consecutive_w_endpoints_df, df_upper_bound, fig_path, n=None, algorithm=None, show_boxplot=None):
    """
    Plot for each n. The x-axis of each graph is lambda. The y-axis displays MSE_{SEG+E}/MSE_{UB}.
    
    Args:
        consecutive_w_endpoints_df: Consecutive with endpoints loss dataframe
        df_upper_bound: Upper bound dataframe
        fig_path: Output file path
        n: Plot only specific n values
        algorithm: Specify algorithm
        show_boxplot: Specify 'boxplot' or 'scatter'
    """
    # Determine show_boxplot based on settings
    if show_boxplot is None:
        show_boxplot = BOXPLOT

    if n is not None:
        consecutive_w_endpoints_df = consecutive_w_endpoints_df[consecutive_w_endpoints_df['n'] == n]
        df_upper_bound = df_upper_bound[df_upper_bound['n'] == n]

    if algorithm is not None:
        df_upper_bound = df_upper_bound[df_upper_bound['algorithm'] == algorithm]
        # drop algorithm column
        df_upper_bound = df_upper_bound.drop(columns=['algorithm'])

    # Combine data for finding available combinations
    combined_df = pd.concat([consecutive_w_endpoints_df, df_upper_bound], ignore_index=True)
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

    # Settings for LconsecE/Lub ratio plots
    # Collect all ratios for overall statistics
    all_lconsecE_lub_ratios = []
    y_min_0 = float('inf')
    y_max_0 = float('-inf')
    
    for j, dist in enumerate(distributions):
        dataset_name, data_type, R = dist
        
        # Get data for consecutive_w_endpoints and upper_bound approaches
        consecutive_w_endpoints_data = consecutive_w_endpoints_df[(consecutive_w_endpoints_df['dataset_name'] == dataset_name) & 
                                                                (consecutive_w_endpoints_df['data_type'] == data_type) & 
                                                                (consecutive_w_endpoints_df['n'] == 1000) & 
                                                                (consecutive_w_endpoints_df['lambda'] > 0) & 
                                                                (consecutive_w_endpoints_df['R'] == R)]
        
        upper_bound_data = df_upper_bound[(df_upper_bound['dataset_name'] == dataset_name) & 
                                        (df_upper_bound['data_type'] == data_type) & 
                                        (df_upper_bound['n'] == 1000) & 
                                        (df_upper_bound['R'] == R)]
        
        if consecutive_w_endpoints_data.empty or upper_bound_data.empty:
            print(f"Warning: Missing data for {dataset_name}, {data_type}, n=1000, R={R}")
            continue
        
        # Merge data on lambda and seed
        consecutive_w_endpoints_data = consecutive_w_endpoints_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
        upper_bound_data = upper_bound_data[['lambda', 'seed', UPPER_BOUND_COLUMN]].sort_values(by=['lambda', 'seed'])
        
        # Rename columns for merge
        consecutive_w_endpoints_data = consecutive_w_endpoints_data.rename(columns={LOSS_COLUMN: 'consecutive_w_endpoints_loss'})
        upper_bound_data = upper_bound_data.rename(columns={UPPER_BOUND_COLUMN: 'upper_bound_loss'})
        
        # Merge datasets
        merged_data = consecutive_w_endpoints_data.merge(upper_bound_data, on=['lambda', 'seed'], how='inner')
        
        if merged_data.empty:
            print(f"Warning: No matching data after merge for {dataset_name}, {data_type}, n=1000, R={R}")
            continue
        
        # Convert lambda to percentage
        merged_data['percentage'] = merged_data['lambda'] / 1000 * 100
        merged_data['LconsecE_divided_by_Lub'] = merged_data['consecutive_w_endpoints_loss'] / merged_data['upper_bound_loss']
        
        # Print minimum/maximum LconsecE/Lub values for each lambda (percentage) and their corresponding seeds
        print(f"\n=== Minimum LconsecE/Lub values for {dataset_name}, {data_type}, R={R} ===")
        for percentage in sorted(merged_data['percentage'].unique()):
            group_data = merged_data[merged_data['percentage'] == percentage]
            min_idx = group_data['LconsecE_divided_by_Lub'].idxmin()
            min_row = group_data.loc[min_idx]
            print(f"  Lambda={min_row['lambda']} ({percentage:.1f}%): Min LconsecE/Lub = {min_row['LconsecE_divided_by_Lub']:.30f} (seed={min_row['seed']})")
            print(f"    Consecutive with endpoints loss: {min_row['consecutive_w_endpoints_loss']:.30f}, Upper bound loss: {min_row['upper_bound_loss']:.30f}")
        
        print(f"\n=== Maximum LconsecE/Lub values for {dataset_name}, {data_type}, R={R} ===")
        for percentage in sorted(merged_data['percentage'].unique()):
            group_data = merged_data[merged_data['percentage'] == percentage]
            max_idx = group_data['LconsecE_divided_by_Lub'].idxmax()
            max_row = group_data.loc[max_idx]
            print(f"  Lambda={max_row['lambda']} ({percentage:.1f}%): Max LconsecE/Lub = {max_row['LconsecE_divided_by_Lub']:.30f} (seed={max_row['seed']})")
            print(f"    Consecutive with endpoints loss: {max_row['consecutive_w_endpoints_loss']:.30f}, Upper bound loss: {max_row['upper_bound_loss']:.30f}")

        # Update global statistics
        all_lconsecE_lub_ratios.extend(merged_data['LconsecE_divided_by_Lub'].tolist())
        
        # Update global y-axis limits
        y_min_0 = min(y_min_0, merged_data['LconsecE_divided_by_Lub'].min())
        y_max_0 = max(y_max_0, merged_data['LconsecE_divided_by_Lub'].max())

        # Plot LconsecE/Lub ratio
        ax = axes[j]
        if show_boxplot:
            # Boxplot version for LconsecE/Lub
            box_data_lub = []
            box_positions = []
            for percentage in sorted(merged_data['percentage'].unique()):
                group_data = merged_data[merged_data['percentage'] == percentage]['LconsecE_divided_by_Lub'].values
                if len(group_data) > 0:
                    box_data_lub.append(group_data)
                    box_positions.append(percentage)
            
            if box_data_lub:
                # Draw boxplot for LconsecE/Lub
                bp = ax.boxplot(box_data_lub, positions=box_positions, patch_artist=True, widths=0.8)
                
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
            # Mean/Min/Max version for LconsecE/Lub
            stats_lub = merged_data.groupby('percentage')['LconsecE_divided_by_Lub'].agg(['mean', 'min', 'max'])
            
            ax.plot(stats_lub.index, stats_lub['mean'], 's-', label='Mean', color='blue', linewidth=2)
            ax.plot(stats_lub.index, stats_lub['min'], 'v:', label='Min', color='blue', linewidth=2)
            ax.plot(stats_lub.index, stats_lub['max'], 'v-.', label='Max', color='blue', linewidth=2)
        
        # Common settings for LconsecE/Lub plot
        ax.grid(True, which='both', linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{DATASET_NAMES[dist][1]}', fontsize=XLABEL_SIZE)
        ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
        if j == 0:
            ax.set_ylabel(r'$\frac{\mathrm{MSE}_\mathrm{SEG{+}E}}{\mathrm{MSE}_\mathrm{UB}}$', fontsize=FONT_SIZE*1.5)
        
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
    
    # Print overall statistics for LconsecE/Lub ratio
    if all_lconsecE_lub_ratios:
        overall_min = min(all_lconsecE_lub_ratios)
        overall_max = max(all_lconsecE_lub_ratios)
        overall_mean = sum(all_lconsecE_lub_ratios) / len(all_lconsecE_lub_ratios)
        print(f"Overall LconsecE/Lub ratio statistics:")
        print(f"  Minimum: {overall_min:.30f}")
        print(f"  Maximum: {overall_max:.30f}")
        print(f"  Mean: {overall_mean:.30f}")
        print(f"  Total samples: {len(all_lconsecE_lub_ratios)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")



if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/lambda_LconsecLub"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    
    # Load data using load_loss and load_upper_bound
    # print("Loading comparison data...")
    consecutive_w_endpoints_df = load_loss(data_dir, "consecutive_w_endpoints")
    df_upper_bound = load_upper_bound(data_dir)
    
    if consecutive_w_endpoints_df.empty or df_upper_bound.empty:
        print("Error: No data loaded")
        exit(1)
    
    # print(f"Loaded data:")
    # print(f"  Consecutive with endpoints: {len(consecutive_w_endpoints_df)} rows")
    # print(f"  Upper bound: {len(df_upper_bound)} rows")
    
    # Filter for n=1000 only
    ns = [1000]
    consecutive_w_endpoints_df_ns = consecutive_w_endpoints_df[consecutive_w_endpoints_df['n'].isin(ns)]
    df_upper_bound_ns = df_upper_bound[df_upper_bound['n'].isin(ns)]
    
    # Plot based on settings
    if BOXPLOT:
        plot_lambda_LconsecLub(consecutive_w_endpoints_df_ns, df_upper_bound_ns, 
                              f'{output_dir}/lambda_Lconsec_divided_by_Lub_all.pdf', 
                              n=None, algorithm="binary_search", show_boxplot='boxplot')
    else:
        plot_lambda_LconsecLub(consecutive_w_endpoints_df_ns, df_upper_bound_ns, 
                              f'{output_dir}/lambda_Lconsec_divided_by_Lub_all.pdf', 
                              n=None, algorithm="binary_search", show_boxplot='scatter')
