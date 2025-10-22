import pandas as pd
import numpy as np
import os
from load_loss import load_loss
from load_upper_bound import load_upper_bound
from plot_config import LOSS_COLUMN

def analyze_algorithm_differences(df_results, df_upper_bound):
    """
    Analyze the difference between the results of 3 upper bound algorithms for each dataset
    """
    # Use only data with lambda > 0
    df_results = df_results[df_results['lambda'] > 0]
    df_upper_bound = df_upper_bound[df_upper_bound['lambda'] > 0]

    # Calculate percentage
    df_results = df_results.copy()
    df_results['percentage'] = df_results['lambda'] / df_results['n'] * 100
    df_upper_bound = df_upper_bound.copy()
    df_upper_bound['percentage'] = df_upper_bound['lambda'] / df_upper_bound['n'] * 100

    # Get available algorithms
    algorithms = df_upper_bound['algorithm'].unique()
    print(f"Available algorithms: {algorithms}")

    # Get combinations of dataset and data_type
    dataset_combinations = df_results[['dataset_name', 'data_type']].drop_duplicates()
    
    # List to store overall statistics
    all_differences = []
    
    # print("======================================")
    # print("Analysis of differences between algorithms for each dataset")
    # print("======================================")
    
    for _, (dataset_name, data_type) in dataset_combinations.iterrows():
        # print(f"\nDataset: {dataset_name} ({data_type})")
        # print("--------------------------------------")
        
        # Get poisoning results for this dataset
        dataset_results = df_results[
            (df_results['dataset_name'] == dataset_name) & 
            (df_results['data_type'] == data_type)
        ]
        
        # Get upper bound results for this dataset
        dataset_upper_bounds = df_upper_bound[
            (df_upper_bound['dataset_name'] == dataset_name) & 
            (df_upper_bound['data_type'] == data_type)
        ]
        
        if dataset_results.empty or dataset_upper_bounds.empty:
            print("  Data is missing")
            continue
        
        # Store results for each algorithm
        algorithm_results = {}
        
        for alg in algorithms:
            alg_upper_bounds = dataset_upper_bounds[dataset_upper_bounds['algorithm'] == alg]
            
            if alg_upper_bounds.empty:
                print(f"  {alg}: No data")
                continue
            
            # Merge poisoning and upper bound results
            merged_data = dataset_results.merge(
                alg_upper_bounds[['lambda', 'seed', 'n', 'R', 'upper_bound']], 
                on=['lambda', 'seed', 'n', 'R'], 
                how='inner'
            )
            
            if merged_data.empty:
                print(f"  {alg}: No matching data")
                continue
            
            # Calculate Lgr/Lub
            merged_data['Lgr_divided_by_Lub'] = merged_data[LOSS_COLUMN] / merged_data['upper_bound']
            algorithm_results[alg] = merged_data['Lgr_divided_by_Lub'].values
            
            # print(f"  {alg}: {len(merged_data)} samples")
        
        if len(algorithm_results) < 2:
            print("  Not enough algorithms to compare")
            continue
        
        # Calculate differences between algorithms
        alg_names = list(algorithm_results.keys())
        differences = []
        
        for i in range(len(alg_names)):
            for j in range(i + 1, len(alg_names)):
                alg1, alg2 = alg_names[i], alg_names[j]
                
                # Match common number of samples
                min_samples = min(len(algorithm_results[alg1]), len(algorithm_results[alg2]))
                if min_samples == 0:
                    continue
                
                # Calculate difference
                diff = algorithm_results[alg1][:min_samples] - algorithm_results[alg2][:min_samples]
                differences.extend(diff)
                
                # Calculate statistics
                mean_diff = np.mean(diff)
                min_diff = np.min(diff)
                max_diff = np.max(diff)
                std_diff = np.std(diff)
        
        if differences:
            # Statistics for this dataset
            dataset_mean = np.mean(differences)
            dataset_min = np.min(differences)
            dataset_max = np.max(differences)
            dataset_std = np.std(differences)
            
            # Add to overall statistics
            all_differences.extend(differences)
    
    # Calculate overall statistics
    if all_differences:
        print("\n" + "="*80)
        print("Overall statistics")
        print("="*80)
        
        overall_mean = np.mean(all_differences)
        overall_min = np.min(all_differences)
        overall_max = np.max(all_differences)
        overall_std = np.std(all_differences)
        
        print(f"Overall mean: {overall_mean:.20f}")
        print(f"Overall min: {overall_min:.20f}")
        print(f"Overall max: {overall_max:.20f}")
        print(f"Overall std: {overall_std:.20f}")
        print(f"Overall sample count: {len(all_differences)}")
        unique_percentages = np.unique(df_results['percentage'])
        unique_seeds = np.unique(df_results['seed'])
        unique_dataset_name_data_type_n_R = np.unique(df_results['dataset_name'] + '_' + df_results['data_type'] + '_' + df_results['n'].astype(str) + '_' + df_results['R'].astype(str))
        print(f"- Unique percentage: {len(unique_percentages)}")
        print(f"- Unique dataset_name_data_type_n_R: {len(unique_dataset_name_data_type_n_R)}")
        print(f"- Unique seed: {len(unique_seeds)}")
        
        # Calculate statistics for absolute values
        abs_differences = [abs(d) for d in all_differences]
        abs_mean = np.mean(abs_differences)
        abs_min = np.min(abs_differences)
        abs_max = np.max(abs_differences)
        abs_std = np.std(abs_differences)
        
        print(f"\nAbsolute value statistics:")
        print(f"Absolute value mean: {abs_mean:.20f}")
        print(f"Absolute value min: {abs_min:.20f}")
        print(f"Absolute value max: {abs_max:.20f}")
        print(f"Absolute value std: {abs_std:.20f}")
    else:
        print("\nNo comparable data")

def main():
    # Data directory (poisoning folder)
    data_dir = ".."
    df_results = load_loss(data_dir)
    df_upper_bound = load_upper_bound(data_dir)
    
    # print("Data loading complete")
    # print(f"Result data: {len(df_results)} rows")
    # print(f"Upper bound data: {len(df_upper_bound)} rows")
    
    # Analyze differences between algorithms
    analyze_algorithm_differences(df_results, df_upper_bound)

if __name__ == '__main__':
    main()
