import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from load_loss import load_loss
from load_loss_comparison import load_loss_comparison_data
from load_upper_bound import load_upper_bound
from plot_config import (
    TICK_SIZE, LEGEND_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES,
    BOXPLOT, ERROR_BARS
)

VERBOSE = False

def plot_maxLgrLconsec_Lub(df_results, df_consecutive_w_endpoints, df_upper_bound, fig_path, n = None, algorithm = None, show_boxplot=None):
    """
    Plot for each n. The x-axis of each graph is lambda. The y-axis of each graph displays the value obtained by dividing the "max(poisoning loss, consecutive with endpoints loss)" by the "upper bound".
    
    Args:
        df_results: Result dataframe
        df_consecutive_w_endpoints: Consecutive with endpoints loss dataframe
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
        df_results = df_results[df_results['n'] == n]
        df_consecutive_w_endpoints = df_consecutive_w_endpoints[df_consecutive_w_endpoints['n'] == n]
        df_upper_bound = df_upper_bound[df_upper_bound['n'] == n]

    if algorithm is not None:
        df_upper_bound = df_upper_bound[df_upper_bound['algorithm'] == algorithm]
        # drop algorithm column
        df_upper_bound = df_upper_bound.drop(columns=['algorithm'])

    available_combinations = [
        (name, dtype, R) for name, dtype, R in df_results[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])
    n_values = sorted(df_results['n'].unique())

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

    # Settings for max(Lgr, L_{seg+e})/Lub ratio plot
    # Collect all max(Lgr, L_{seg+e})/Lub ratios for overall statistics
    all_max_ratios = []
    
    for j, dist in enumerate(distributions):
        for i, n in enumerate(n_values):
            ax = axes[i][j]
            dataset_name, data_type, R = dist
            poison_data      = df_results[(df_results['dataset_name'] == dataset_name) & (df_results['data_type'] == data_type) & (df_results['n'] == n) & (df_results['lambda'] > 0) & (df_results['R'] == R)]
            consecutive_w_endpoints_data = df_consecutive_w_endpoints[(df_consecutive_w_endpoints['dataset_name'] == dataset_name) & (df_consecutive_w_endpoints['data_type'] == data_type) & (df_consecutive_w_endpoints['n'] == n) & (df_consecutive_w_endpoints['lambda'] > 0) & (df_consecutive_w_endpoints['R'] == R)]
            upper_bound_data = df_upper_bound[(df_upper_bound['dataset_name'] == dataset_name) & (df_upper_bound['data_type'] == data_type) & (df_upper_bound['n'] == n) & (df_upper_bound['R'] == R)]
            
            poison_data      = poison_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
            consecutive_w_endpoints_data = consecutive_w_endpoints_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
            
            # Rename columns for merge
            poison_data = poison_data.rename(columns={LOSS_COLUMN: 'poison_loss'})
            consecutive_w_endpoints_data = consecutive_w_endpoints_data.rename(columns={LOSS_COLUMN: 'consecutive_w_endpoints_loss'})
            
            # Handle different algorithms
            if 'algorithm' in upper_bound_data.columns:
                # New format with algorithm column
                algorithms = upper_bound_data['algorithm'].unique()
                colors = {'binary_search': 'red', 'golden_section': 'orange', 'strict': 'brown'}
                markers = {'binary_search': 'o', 'golden_section': '^', 'strict': 'd'}
                min_markers = {'binary_search': 'v', 'golden_section': 's', 'strict': '<'}
                max_markers = {'binary_search': '*', 'golden_section': 'D', 'strict': '>'}
                linestyles = {'binary_search': '-', 'golden_section': '--', 'strict': '-.'}
                labels = {'binary_search': 'Binary Search', 'golden_section': 'Golden Section', 'strict': 'Strict'}
                
                for alg in algorithms:
                    alg_upper_bound_data = upper_bound_data[upper_bound_data['algorithm'] == alg][['lambda', 'seed', UPPER_BOUND_COLUMN]].sort_values(by=['lambda', 'seed'])
                    
                    # Merge all three datasets
                    merged_data = poison_data.merge(consecutive_w_endpoints_data, on=['lambda', 'seed'], how='inner')
                    merged_data = merged_data.merge(alg_upper_bound_data, on=['lambda', 'seed'], how='outer')
                    
                    if merged_data['poison_loss'].isna().sum() > 0 or merged_data['consecutive_w_endpoints_loss'].isna().sum() > 0 or merged_data[UPPER_BOUND_COLUMN].isna().sum() > 0:
                        print(f"Warning: Missing data for {dataset_name}, {data_type}, n={n}, R={R}, algorithm={alg}.")
                        print("poison_data")
                        print(poison_data)
                        print("consecutive_w_endpoints_data")
                        print(consecutive_w_endpoints_data)
                        print("alg_upper_bound_data")
                        print(alg_upper_bound_data)
                        continue
                    
                    # Convert lambda to percentage
                    merged_data['percentage'] = merged_data['lambda'] / n * 100
                    merged_data = merged_data[merged_data['percentage'] <= 18]
                    merged_data['max_loss'] = merged_data[['poison_loss', 'consecutive_w_endpoints_loss']].max(axis=1)
                    merged_data['max_divided_by_Lub'] = merged_data['max_loss'] / merged_data[UPPER_BOUND_COLUMN]
                    
                    # Collect ratios for overall statistics
                    all_max_ratios.extend(merged_data['max_divided_by_Lub'].values)
                    
                    if show_boxplot:
                        # Boxplot version
                        box_data = []
                        box_positions = []
                        for percentage in sorted(merged_data['percentage'].unique()):
                            group_data = merged_data[merged_data['percentage'] == percentage]['max_divided_by_Lub'].values
                            if len(group_data) > 0:
                                box_data.append(group_data)
                                box_positions.append(percentage)
                                all_max_ratios.extend(group_data)
                        
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
                                    print(f"Boxplot ({alg}) - {DATASET_NAMES[dist][1]}, n={n}, percentage={box_positions[k]:.1f}%: {len(data)} samples")
                            
                            # Set x-axis
                            ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                            # Display x-axis labels that are multiples of 5
                            xticks_positions = [i_ * 5 for i_ in range(int(min(box_positions) // 5) + 1, int(max(box_positions) // 5) + 1)]
                            ax.set_xticks(xticks_positions)
                            ax.set_xticklabels([f'{x}' for x in xticks_positions])
                    else:
                        # Mean/Min/Max version
                        stats = merged_data.groupby('percentage')['max_divided_by_Lub'].agg(['mean', 'min', 'max'])
                        
                        color = colors.get(alg, 'blue')
                        marker = markers.get(alg, 'o')
                        min_marker = min_markers.get(alg, 'v')
                        max_marker = max_markers.get(alg, '*')
                        linestyle = linestyles.get(alg, '-')
                        label_prefix = labels.get(alg, alg)
                        
                        ax.plot(stats.index, stats['mean'], marker + linestyle, label=f'Mean ({label_prefix})', color=color, linewidth=2)
                        ax.plot(stats.index, stats['min'], min_marker + ':', label=f'Min ({label_prefix})', color=color, linewidth=2)
                        ax.plot(stats.index, stats['max'], max_marker + '-.', label=f'Max ({label_prefix})', color=color, linewidth=2)
                    
                    # Output number of samples for each point
                    if VERBOSE and not show_boxplot:
                        for percentage, group in merged_data.groupby('percentage'):
                            print(f"Max Loss ({alg}) - {DATASET_NAMES[dist][1]}, n={n}, percentage={percentage:.1f}%: {len(group)} samples")
            else:
                # Legacy format without algorithm column
                upper_bound_data = upper_bound_data[['lambda', 'seed', UPPER_BOUND_COLUMN]].sort_values(by=['lambda', 'seed'])
                
                # Merge all three datasets
                merged_data = poison_data.merge(consecutive_w_endpoints_data, on=['lambda', 'seed'], how='inner')
                merged_data = merged_data.merge(upper_bound_data, on=['lambda', 'seed'], how='outer')
                
                if merged_data['poison_loss'].isna().sum() > 0 or merged_data['consecutive_w_endpoints_loss'].isna().sum() > 0 or merged_data[UPPER_BOUND_COLUMN].isna().sum() > 0:
                    print(f"Warning: Missing data for {dataset_name}, {data_type}, n={n}, R={R}.")
                    print("poison_data")
                    print(poison_data)
                    print("consecutive_w_endpoints_data")
                    print(consecutive_w_endpoints_data)
                    print("upper_bound_data")
                    print(upper_bound_data)
                    continue
                
                # Convert lambda to percentage
                merged_data['percentage'] = merged_data['lambda'] / n * 100
                merged_data = merged_data[merged_data['percentage'] <= 18]
                merged_data['max_loss'] = merged_data[['poison_loss', 'consecutive_w_endpoints_loss']].max(axis=1)
                merged_data['max_divided_by_Lub'] = merged_data['max_loss'] / merged_data[UPPER_BOUND_COLUMN]
                
                if show_boxplot:
                    # Boxplot version
                    box_data = []
                    box_positions = []
                    for percentage in sorted(merged_data['percentage'].unique()):
                        group_data = merged_data[merged_data['percentage'] == percentage]['max_divided_by_Lub'].values
                        if len(group_data) > 0:
                            box_data.append(group_data)
                            box_positions.append(percentage)
                            all_max_ratios.extend(group_data)
                    
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
                                print(f"Boxplot - {DATASET_NAMES[dist][1]}, n={n}, percentage={box_positions[k]:.1f}%: {len(data)} samples")
                        
                        # Set x-axis
                        ax.set_xlim(min(box_positions) - 1, max(box_positions) + 1)
                        # Display x-axis labels that are multiples of 5
                        xticks_positions = [i_ * 5 for i_ in range(int(min(box_positions) // 5) + 1, int(max(box_positions) // 5) + 1)]
                        ax.set_xticks(xticks_positions)
                        ax.set_xticklabels([f'{x}' for x in xticks_positions])
                else:
                    # Mean/Min/Max version
                    stats = merged_data.groupby('percentage')['max_divided_by_Lub'].agg(['mean', 'min', 'max'])
                    
                    ax.plot(stats.index, stats['mean'], 'o-', label='Mean', color='blue', linewidth=2)
                    ax.plot(stats.index, stats['min'], 'x:', label='Min', color='blue', linewidth=2)
                    ax.plot(stats.index, stats['max'], 'x-.', label='Max', color='blue', linewidth=2)
                
                # Output number of samples for each point
                if VERBOSE and not show_boxplot:
                    for percentage, group in merged_data.groupby('percentage'):
                        print(f"Max Loss - {DATASET_NAMES[dist][1]}, n={n}, percentage={percentage:.1f}%: {len(group)} samples")

            # Common settings
            if show_boxplot:
                # For boxplot, do not use log scale for y-axis
                ax.set_ylim(0.78, 1.02)
            else:
                # For mean/min/max, use the default settings
                ax.set_ylim(0.78, 1.02)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
            if i == 0:
                ax.set_title(DATASET_NAMES[dist][1], fontsize=FONT_SIZE)
            if j == 0:
                if len(n_values) == 1:
                    ax.set_ylabel(r'$\frac{\max(\mathrm{MSE}_\mathrm{G}, \mathrm{MSE}_\mathrm{SEG{+}E})}{\mathrm{MSE}_\mathrm{UB}}$', fontsize=FONT_SIZE*1.5)
                else:
                    ax.set_ylabel(rf'$\mathbf{{n={n}}}$' + '\n' + r'$\frac{\max(\mathrm{MSE}_\mathrm{G}, \mathrm{MSE}_\mathrm{SEG{+}E})}{\mathrm{MSE}_\mathrm{UB}}$', fontsize=FONT_SIZE*1.5)
    # Place legend in the lower right corner of the graph (if applicable)
    if not show_boxplot:
        handles, labels = axes[len(n_values) - 1][len(distributions) - 1].get_legend_handles_labels()
        axes[len(n_values) - 1][len(distributions) - 1].legend(handles, labels, 
                                        loc='lower right',
                                        frameon=True,
                                        facecolor='white',
                                        edgecolor='black',
                                        fontsize=LEGEND_SIZE)
    
    # Print overall statistics for max(Lgr, L_{seg+e})/Lub ratios
    if all_max_ratios:
        overall_min = min(all_max_ratios)
        overall_mean = sum(all_max_ratios) / len(all_max_ratios)
        print(f"Overall max(Lgr, L_seg+e)/Lub ratio statistics:")
        print(f"  Minimum: {overall_min:.6f}")
        print(f"  Mean: {overall_mean:.6f}")
        print(f"  Total samples: {len(all_max_ratios)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")

if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/maxLgrLconsec_Lub"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    df_results = load_loss(data_dir)
    df_consecutive_w_endpoints = load_loss_comparison_data(data_dir, "consecutive_w_endpoints")
    df_upper_bound = load_upper_bound(data_dir)

    # Get available n values
    # available_n_values = sorted(df_results['n'].unique())
    # for n in available_n_values:
    #     plot_maxLgrLconsec_Lub(df_results, df_consecutive_w_endpoints, df_upper_bound, f'{output_dir}/maxLgrLconsec_Lub_n{n}.pdf', n = n, algorithm = "binary_search")

    ns = [1000]
    df_results_ns = df_results[df_results['n'].isin(ns)]
    df_consecutive_w_endpoints_ns = df_consecutive_w_endpoints[df_consecutive_w_endpoints['n'].isin(ns)]
    df_upper_bound_ns = df_upper_bound[df_upper_bound['n'].isin(ns)]
    
    # Plot based on settings
    if BOXPLOT:
        plot_maxLgrLconsec_Lub(df_results_ns, df_consecutive_w_endpoints_ns, df_upper_bound_ns, f'{output_dir}/maxLgrLconsec_Lub_all.pdf', n = None, algorithm = "binary_search", show_boxplot=True)
    else:
        plot_maxLgrLconsec_Lub(df_results_ns, df_consecutive_w_endpoints_ns, df_upper_bound_ns, f'{output_dir}/maxLgrLconsec_Lub_all.pdf', n = None, algorithm = "binary_search", show_boxplot=False) 
