import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from load_loss import load_loss
from load_upper_bound import load_upper_bound
from plot_config import (
    TICK_SIZE, LEGEND_SEED0_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES
)

XTICK_INTERVAL = 5
ROW_HEIGHT = 4.25
COLUMN_WIDTH = 5

def plot_lambda_Ls_seed0(df_results, df_upper_bound, fig_path, n=None):
    """
    Plot for each n. The horizontal axis of each graph is lambda. Plot both max_loss_greedy and upper_bound3 on the vertical axis. Only seed 0 is used.
    """
    df_results = df_results[df_results['seed'] == 0]
    if n is not None:
        df_results = df_results[df_results['n'] == n]

    df_upper_bound = df_upper_bound[df_upper_bound['seed'] == 0]
    if n is not None:
        df_upper_bound = df_upper_bound[df_upper_bound['n'] == n]

    available_combinations = [
        (name, dtype, R) for name, dtype, R in df_results[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])
    n_values = sorted(df_results['n'].unique())

    plt.rcParams['figure.figsize'] = [COLUMN_WIDTH * len(distributions) + 0.5, ROW_HEIGHT * len(n_values)]
    fig, axes = plt.subplots(len(n_values), len(distributions))

    # Process if axes is one-dimensional
    if len(n_values) == 1:
        if len(distributions) == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(1, -1)
    elif len(distributions) == 1:
        axes = axes.reshape(-1, 1)

    min_legitimate  = min(df_results[LOSS_COLUMN].min(), df_upper_bound[UPPER_BOUND_COLUMN].min())
    max_upper_bound = max(df_results[LOSS_COLUMN].max(), df_upper_bound[UPPER_BOUND_COLUMN].max())
    for i, n_val in enumerate(n_values):
        for j, dist in enumerate(distributions):
            ax = axes[i][j]
            dataset_name, data_type, R = dist
            poison_data      = df_results[(df_results['dataset_name'] == dataset_name) & (df_results['data_type'] == data_type) & (df_results['n'] == n_val) & (df_results['lambda'] > 0) & (df_results['R'] == R)]
            legitimate_data  = df_results[(df_results['dataset_name'] == dataset_name) & (df_results['data_type'] == data_type) & (df_results['n'] == n_val) & (df_results['lambda'] == 0) & (df_results['R'] == R)]
            upper_bound_data = df_upper_bound[(df_upper_bound['dataset_name'] == dataset_name) & (df_upper_bound['data_type'] == data_type) & (df_upper_bound['n'] == n_val) & (df_upper_bound['R'] == R)]
            poison_data      = poison_data[['lambda', LOSS_COLUMN]].sort_values(by='lambda')
            if len(legitimate_data) == 0:
                continue

            assert len(legitimate_data) == 1, f"legitimate_data: {legitimate_data}"

            # Convert lambda to percentage for poison data
            poison_data['percentage'] = poison_data['lambda'] / n_val * 100

            legitimate_loss = legitimate_data[LOSS_COLUMN].values[0]
            ax.plot(poison_data['percentage'], poison_data[LOSS_COLUMN], 'o-', label=r'$\mathrm{MSE}_\mathrm{G}$', color='blue')
            
            # # Plot upper bounds for each algorithm
            # if 'algorithm' in upper_bound_data.columns:
            #     # New format with algorithm column
            #     algorithms = upper_bound_data['algorithm'].unique()
            #     colors = {'binary_search': 'red', 'golden_section': 'orange', 'strict': 'brown', 'legacy': 'red'}
            #     markers = {'binary_search': 'x', 'golden_section': '^', 'strict': 'd', 'legacy': 'x'}
            #     linestyles = {'binary_search': '-.', 'golden_section': '--', 'strict': ':', 'legacy': '-.'}
            #     labels = {'binary_search': 'Upper Bound (Binary Search)', 'golden_section': 'Upper Bound (Golden Section)', 'strict': 'Upper Bound (Strict)', 'legacy': 'Upper Bound'}
                
            #     for alg in algorithms:
            #         alg_data = upper_bound_data[upper_bound_data['algorithm'] == alg][['lambda', UPPER_BOUND_COLUMN]].sort_values(by='lambda')
            #         alg_data['percentage'] = alg_data['lambda'] / n_val * 100
                    
            #         if len(alg_data) > 0:
            #             ax.plot(alg_data['percentage'], alg_data[UPPER_BOUND_COLUMN], 
            #                    marker=markers.get(alg, 'x'), linestyle=linestyles.get(alg, '-.'), 
            #                    label=labels.get(alg, f'Upper Bound ({alg})'), 
            #                    color=colors.get(alg, 'red'))
                        
            #             # Check data length consistency
            #             if len(poison_data) != len(alg_data):
            #                 print(f"[Warning] len(poison_data) != len({alg}_data): {len(poison_data)} != {len(alg_data)}")
            #                 print(f"poison_data:")
            #                 print(poison_data)
            #                 print(f"{alg}_data:")
            #                 print(alg_data)
            # else:

            upper_bound_data = upper_bound_data[['lambda', UPPER_BOUND_COLUMN]].sort_values(by='lambda')
            upper_bound_data['percentage'] = upper_bound_data['lambda'] / n_val * 100
            ax.plot(upper_bound_data['percentage'], upper_bound_data[UPPER_BOUND_COLUMN], 'x-.', label=r'$\mathrm{MSE}_\mathrm{UB}$', color='red')
            
            if len(poison_data) != len(upper_bound_data):
                print(f"[Warning] len(poison_data) != len(upper_bound_data): {len(poison_data)} != {len(upper_bound_data)} (dataset_name: {dataset_name}, data_type: {data_type}, n: {n_val}, R: {R})")
                print(f"poison_data lambda values: {sorted(poison_data['lambda'].tolist())}")
                print(f"upper_bound_data lambda values: {sorted(upper_bound_data['lambda'].tolist())}")
                print(f"poison_data:")
                print(poison_data[['lambda', 'loss', 'percentage']].to_string(index=False))
                print(f"upper_bound_data:")
                print(upper_bound_data[['lambda', 'upper_bound', 'percentage']].to_string(index=False))
            
            ax.axhline(y=legitimate_loss, color='green', linestyle='--', label=r'$\mathrm{MSE}_\mathrm{L}$')

            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.set_xlabel('Poisoning Percentage', fontsize=XLABEL_SIZE)
            if i == 0:
                ax.set_title(DATASET_NAMES[dist][1], fontsize=FONT_SIZE)
            if j == 0:
                if len(n_values) == 1:
                    ax.set_ylabel('MSE', fontsize=FONT_SIZE)
                else:
                    ax.set_ylabel(rf'$\mathbf{{n={n_val}}}$' + '\n' + 'MSE', fontsize=FONT_SIZE)
            ax.set_ylim(min_legitimate * 0.5, max_upper_bound * 2.0)

    # Place legend in the lower right corner of the graph (if applicable)
    LEGEND_ON_EXPONENTIAL = True
    distribution_names = [dist[0] for dist in distributions]
    if LEGEND_ON_EXPONENTIAL and "exponential" in distribution_names:
        handles, labels = axes[len(n_values) - 1][distribution_names.index("exponential")].get_legend_handles_labels()
        axes[len(n_values) - 1][distribution_names.index("exponential")].legend(handles, labels, 
                                    loc='lower right',
                                    frameon=True,
                                    facecolor='white',
                                    edgecolor='black',
                                    fontsize=LEGEND_SEED0_SIZE)
    else:
        handles, labels = axes[len(n_values) - 1][len(distributions) - 1].get_legend_handles_labels()
        axes[len(n_values) - 1][len(distributions) - 1].legend(handles, labels, 
                                        loc='lower right',
                                        frameon=True,
                                        facecolor='white',
                                        edgecolor='black',
                                        fontsize=LEGEND_SEED0_SIZE)
            
    # handles, labels = axes[len(n_values) - 1][len(distributions) - 1].get_legend_handles_labels()
    # axes[len(n_values) - 1][len(distributions) - 1].legend(handles, labels, 
    #                                   loc='lower right',
    #                                   frameon=True,
    #                                   facecolor='white',
    #                                   edgecolor='black',
    #                                   fontsize=LEGEND_SEED0_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved {fig_path}")

if __name__ == '__main__':
    # Set output directory to results/fig
    output_dir = "../results/fig/lambda_Ls_seed0"
    os.makedirs(output_dir, exist_ok=True)

    # Data directory (poisoning folder)
    data_dir = ".."
    df_results = load_loss(data_dir)

    df_upper_bound = load_upper_bound(data_dir)
    df_upper_bound = df_upper_bound[df_upper_bound['algorithm'] == 'binary_search']
    df_upper_bound = df_upper_bound.drop(columns=['algorithm'])

    # all plot (including multiple n values)
    ns = [1000]
    df_results_ns = df_results[df_results['n'].isin(ns)]
    df_upper_bound_ns = df_upper_bound[df_upper_bound['n'].isin(ns)]
    plot_lambda_Ls_seed0(df_results_ns, df_upper_bound_ns, f'{output_dir}/seed0_all.pdf', n = None)
