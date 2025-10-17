#!/usr/bin/env python3
"""
Analysis script for L_gr / L_ub
Calculates minimum, median, maximum, and count for L_gr / L_ub for each dataset and lambda setting.
"""

import numpy as np
import pandas as pd
import os
from load_loss import load_loss
from load_upper_bound import load_upper_bound
from plot_config import LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES
import statistics

def analyze_Lgr_divided_by_Lub(df_results, df_upper_bound, algorithm, n=None, poisoning_percentage=None):
    """
    Calculate simple statistics for L_gr / L_ub for each dataset and lambda setting
    
    Args:
        df_results: Results dataframe
        df_upper_bound: Upper bound dataframe
        algorithm: Specific algorithm
        n: Specific n value
        poisoning_percentage: Specific poisoning percentage value
    """
    
    df_results = df_results.copy()
    df_upper_bound = df_upper_bound.copy()
    df_results['poisoning_percentage'] = (df_results['lambda'] / df_results['n'] * 100).astype(int)
    df_upper_bound['poisoning_percentage'] = (df_upper_bound['lambda'] / df_upper_bound['n'] * 100).astype(int)
    
    if n is not None:
        df_results = df_results[df_results['n'] == n]
        df_upper_bound = df_upper_bound[df_upper_bound['n'] == n]
    if poisoning_percentage is not None:
        df_results = df_results[df_results['poisoning_percentage'] == poisoning_percentage]
        df_upper_bound = df_upper_bound[df_upper_bound['poisoning_percentage'] == poisoning_percentage]
    df_upper_bound = df_upper_bound[df_upper_bound['algorithm'] == algorithm]

    # Get available combinations
    available_combinations = [
        (name, dtype, R) for name, dtype, R in df_results[['dataset_name', 'data_type', 'R']].drop_duplicates().itertuples(index=False)
        if (name, dtype, R) in DATASET_NAMES
    ]
    distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])
    
    print(f"\n=== L_gr / L_ub Analysis for n={n} and poisoning_percentage={poisoning_percentage} ===")
    
    all_ratios_list = np.array([])

    # Analyze each distribution
    for dist in distributions:
        dataset_name, data_type, R = dist
        dataset_display_name = DATASET_NAMES[dist][1]
        
        print(f"\n--- {dataset_display_name} ({dataset_name}, {data_type}, R={R}, n={n}, poisoning_percentage={poisoning_percentage}) ---")
        
        # Get poisoning data (those with lambda > 0)
        poison_data = df_results[
            (df_results['dataset_name'] == dataset_name) & 
            (df_results['data_type'] == data_type) & 
            (True if n is None else df_results['n'] == n) & 
            (True if poisoning_percentage is None else df_results['poisoning_percentage'] == poisoning_percentage) & 
            (df_results['lambda'] > 0) & 
            (df_results['R'] == R)
        ]
        
        if poison_data.empty:
            print("  No poisoning data found")
            continue
        
        # Get upper bound data
        upper_bound_data = df_upper_bound[
            (df_upper_bound['dataset_name'] == dataset_name) & 
            (df_upper_bound['data_type'] == data_type) & 
            (True if n is None else df_upper_bound['n'] == n) & 
            (True if poisoning_percentage is None else df_upper_bound['poisoning_percentage'] == poisoning_percentage) & 
            (df_upper_bound['R'] == R) &
            (df_upper_bound['algorithm'] == algorithm)
        ]
        
        if upper_bound_data.empty:
            print("  No upper bound data found")
            continue
        
        # Merge data
        poison_data_sorted = poison_data[['poisoning_percentage', 'seed', 'n', 'lambda', LOSS_COLUMN]].sort_values(by=['poisoning_percentage', 'seed'])
        upper_bound_data_sorted = upper_bound_data[['poisoning_percentage', 'seed', 'n', 'lambda', UPPER_BOUND_COLUMN]].sort_values(by=['poisoning_percentage', 'seed'])

        # Use suffixes to avoid column name conflicts
        if poisoning_percentage is None and n is not None:
            merged_data = poison_data_sorted.merge(upper_bound_data_sorted, on=['n', 'poisoning_percentage', 'seed', 'lambda'], how='inner', suffixes=('_poison', '_upper'))
        elif poisoning_percentage is not None and n is None:
            merged_data = poison_data_sorted.merge(upper_bound_data_sorted, on=['n', 'poisoning_percentage', 'seed', 'lambda'], how='inner', suffixes=('_poison', '_upper'))
        else:
            merged_data = poison_data_sorted.merge(upper_bound_data_sorted, on=['n', 'poisoning_percentage', 'seed', 'lambda'], how='inner', suffixes=('_poison', '_upper'))
        
        # Rename lambda column if it was suffixed
        if 'lambda_poison' in merged_data.columns:
            merged_data['lambda'] = merged_data['lambda_poison']
        elif 'lambda_upper' in merged_data.columns:
            merged_data['lambda'] = merged_data['lambda_upper']
        
        if merged_data.empty:
            print("  No matching data found")
            continue
        
        # Calculate L_gr / L_ub
        merged_data['Lgr_divided_by_Lub'] = merged_data[LOSS_COLUMN] / merged_data[UPPER_BOUND_COLUMN]
        
        # Calculate statistics for each lambda value
        if poisoning_percentage is None and n is not None:
            for lambda_val in sorted(merged_data['lambda'].unique()):
                lambda_data = merged_data[merged_data['lambda'] == lambda_val]
                ratios = lambda_data['Lgr_divided_by_Lub']
                min_ratio = ratios.min()
                min_ratio_seed = lambda_data[lambda_data['Lgr_divided_by_Lub'] == min_ratio]['seed'].values[0]
                print(f"  λ={lambda_val:3d}: min={ratios.min():.4f} (seed={min_ratio_seed}), mean={ratios.mean():.4f}, median={ratios.median():.4f}, max={ratios.max():.4f}, count={len(ratios)}")
                all_ratios_list = np.concatenate([all_ratios_list, ratios])
        elif poisoning_percentage is not None and n is None:
            for n_val in sorted(merged_data['n'].unique()):
                n_data = merged_data[merged_data['n'] == n_val]
                ratios = n_data['Lgr_divided_by_Lub']
                min_ratio = ratios.min()
                min_ratio_seed = n_data[n_data['Lgr_divided_by_Lub'] == min_ratio]['seed'].values[0]
                print(f"  n={n_val:5d}: min={ratios.min():.4f} (seed={min_ratio_seed}), mean={ratios.mean():.4f}, median={ratios.median():.4f}, max={ratios.max():.4f}, count={len(ratios)}")
                all_ratios_list = np.concatenate([all_ratios_list, ratios])
        else:
            # Default case: show overall statistics
            ratios = merged_data['Lgr_divided_by_Lub']
            print(f"  Overall: min={ratios.min():.4f}, mean={ratios.mean():.4f}, median={ratios.median():.4f}, max={ratios.max():.4f}, count={len(ratios)}")
            all_ratios_list = np.concatenate([all_ratios_list, ratios])

    print()
    print(f"Overall: min={all_ratios_list.min():.4f}, mean={all_ratios_list.mean():.4f}, median={np.median(all_ratios_list):.4f}, max={all_ratios_list.max():.4f}, count={len(all_ratios_list)}")

def main():
    """
    Main function
    """
    print("Starting L_gr / L_ub analysis...")
    
    # Set path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..")
    
    # Load data
    print("Loading data...")
    df_results = load_loss(data_dir)
    df_upper_bound = load_upper_bound(data_dir)
    
    n = 1000
    df_results_filtered = df_results[df_results['n'] == n]
    df_upper_bound_filtered = df_upper_bound[df_upper_bound['n'] == n]
    analyze_Lgr_divided_by_Lub(df_results_filtered, df_upper_bound_filtered, algorithm="binary_search", n=n)

    poisoning_percentage = 10
    df_results_filtered = df_results.copy()
    df_upper_bound_filtered = df_upper_bound.copy()
    df_results_filtered['poisoning_percentage'] = (df_results_filtered['lambda'] / df_results_filtered['n'] * 100).astype(int)
    df_upper_bound_filtered['poisoning_percentage'] = (df_upper_bound_filtered['lambda'] / df_upper_bound_filtered['n'] * 100).astype(int)
    df_results_filtered = df_results_filtered[df_results_filtered['poisoning_percentage'] == poisoning_percentage]
    df_upper_bound_filtered = df_upper_bound_filtered[df_upper_bound_filtered['poisoning_percentage'] == poisoning_percentage]
    analyze_Lgr_divided_by_Lub(df_results_filtered, df_upper_bound_filtered, algorithm="binary_search", poisoning_percentage=poisoning_percentage)

    poisoning_percentage = 10
    df_results_filtered = df_results.copy()
    df_upper_bound_filtered = df_upper_bound.copy()
    df_results_filtered['poisoning_percentage'] = (df_results_filtered['lambda'] / df_results_filtered['n'] * 100).astype(int)
    df_upper_bound_filtered['poisoning_percentage'] = (df_upper_bound_filtered['lambda'] / df_upper_bound_filtered['n'] * 100).astype(int)
    df_results_filtered = df_results_filtered[df_results_filtered['poisoning_percentage'] == poisoning_percentage]
    df_upper_bound_filtered = df_upper_bound_filtered[df_upper_bound_filtered['poisoning_percentage'] == poisoning_percentage]
    # Amzn, 
    analyze_Lgr_divided_by_Lub(df_results_filtered, df_upper_bound_filtered, algorithm="binary_search", poisoning_percentage=poisoning_percentage)
    
    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
