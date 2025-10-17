#!/usr/bin/env python3
"""
Dataset plot script
For seed0 datasets, generate plots for each distribution with n=100, 1000, 10000.
Horizontal axis: value, vertical axis: rank
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from typing import Dict, Any, Optional
from load_loss import load_loss
from load_upper_bound import load_upper_bound

from plot_config import (
    TICK_SIZE, XLABEL_SIZE, FONT_SIZE,
    LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES
)

# Constants for styling (matching plot_poisoned_datasets.py)
MARKER_SIZE = 20
ROW_HEIGHT = 5.5
COL_WIDTH = 5

def read_binary_data(filename: str) -> np.ndarray:
    """
    Read data from binary file
    
    Args:
        filename: Path to binary file
        
    Returns:
        Data numpy array
    """
    with open(filename, 'rb') as f:
        # Read data size from first 8 bytes
        size_bytes = f.read(8)
        size = struct.unpack("Q", size_bytes)[0]
        
        # Determine data type from filename
        if "uint32" in filename:
            dtype = np.uint32
        elif "uint64" in filename:
            dtype = np.uint64
        else:
            raise ValueError(f"Unknown data type in filename: {filename}")
        
        # Read data
        data = np.fromfile(f, dtype=dtype)
        
        # Check size
        assert len(data) == size, f"Data size mismatch: expected {size}, got {len(data)}"
        
        # Check sorting
        assert np.array_equal(np.sort(data), data), f"Data is not sorted in {filename}"
        
        return data

def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse parameters from filename
    
    Args:
        filename: File name
        
    Returns:
        Dictionary of parameters
    """
    # File name pattern:
    # 1. {dataset}_{size}_n{n}_seed{seed}_{type} (e.g. books_200M_n100_seed0_uint64)
    # 2. {distribution}_n{n}_R{R}_seed{seed}_{type} (e.g. normal_n100_R1000_seed0_uint64)
    
    # Pattern 1: {dataset}_{size}_n{n}_seed{seed}_{type}
    pattern1 = r'^([a-zA-Z_]+)_(\d+M)_n(\d+)_seed(\d+)_(uint\d+)$'
    match = re.match(pattern1, filename)
    if match:
        dataset, size, n, seed, data_type = match.groups()
        # Recognize osm_cellids as osm
        if dataset == 'osm_cellids':
            dataset = 'osm'
        return {
            'distribution': dataset,
            'n': int(n),
            'R': 0,
            'seed': int(seed),
            'lambda': 0,
            'data_type': data_type,
            'filename': filename
        }
    
    # Pattern 2: {distribution}_n{n}_R{R}_seed{seed}_{type}
    pattern2 = r'^([a-zA-Z]+)_n(\d+)_R(\d+)_seed(\d+)_(uint\d+)$'
    match = re.match(pattern2, filename)
    if match:
        distribution, n, R, seed, data_type = match.groups()
        return {
            'distribution': distribution,
            'n': int(n),
            'R': int(R),
            'seed': int(seed),
            'lambda': 0,
            'data_type': data_type,
            'filename': filename
        }
    
    # Pattern 3: {dataset}_ts_{size}_n{n}_seed{seed}_{type}
    pattern3 = r'^([a-zA-Z]+)_ts_(\d+M)_n(\d+)_seed(\d+)_(uint\d+)$'
    match = re.match(pattern3, filename)
    if match:
        dataset, size, n, seed, data_type = match.groups()
        return {
            'distribution': dataset,
            'n': int(n),
            'R': 0,
            'seed': int(seed),
            'lambda': 0,
            'data_type': data_type,
            'filename': filename
        }
    
    return None

def find_seed_files(data_dir: str, target_n_values: list, seed: int = 0) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Search for files with specified seed and organize file paths by distribution, n value, and data type
    
    Args:
        data_dir: Path to data directory
        target_n_values: List of n values to target
        seed: Target seed value (default: 0)
        
    Returns:
        Dictionary of distribution name -> {n value: {data type: file path}}
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(('uint32', 'uint64')) and '_lambda' not in f]
    
    # Define distribution order (only include those in DATASET_NAMES)
    distribution_order = []
    for dataset_key in sorted(DATASET_NAMES.keys(), key=lambda x: DATASET_NAMES[x][0]):
        distribution, data_type, R = dataset_key
        if distribution not in distribution_order:
            distribution_order.append(distribution)
    
    result = {}
    
    for filename in files:
        params = parse_filename(filename)
        if params is None:
            continue
            
        # Target only specified seed
        if params['seed'] != seed:
            continue
            
        # Target only specified n values
        if params['n'] not in target_n_values:
            continue
            
        distribution = params['distribution']
        data_type = params['data_type']
        
        if distribution not in result:
            result[distribution] = {}
            
        if params['n'] not in result[distribution]:
            result[distribution][params['n']] = {}
        
        # For artificial datasets, prioritize files with R=100000
        if distribution in ['uniform', 'normal', 'exponential']:
            filepath = os.path.join(data_dir, filename)
            if 'R100000' in filename:
                # If R=100000 file is found, replace existing file
                result[distribution][params['n']][data_type] = filepath
            elif data_type not in result[distribution][params['n']]:
                # For non-R=100000 files, set only if file is not already set
                result[distribution][params['n']][data_type] = filepath
        else:
            # For real datasets, proceed as usual
            result[distribution][params['n']][data_type] = os.path.join(data_dir, filename)
    
    return result

def get_Lgr_divided_by_Lub_value(df_results, df_upper_bound, distribution: str, n: int, data_type: str, R: int, target_percentage: float = 10.0):
    """
    Get L_G/L_{UB} value when poisoning percentage = target_percentage
    
    Args:
        df_results: Result dataframe
        df_upper_bound: Upper bound dataframe
        distribution: Distribution name
        n: Data size
        data_type: Data type
        R: R value
        target_percentage: Target poisoning percentage
        
    Returns:
        L_G/L_{UB} value (None if data is not found)
    """
    # Dataset name mapping
    dataset_name_mapping = {
        'uniform': 'uniform',
        'normal': 'normal', 
        'exponential': 'exponential',
        'books': 'books',
        'fb': 'fb',
        'osm': 'osm'
    }
    
    dataset_name = dataset_name_mapping.get(distribution, distribution)
    
    # Calculate lambda value
    target_lambda = int(n * target_percentage / 100)
    
    # Get poisoning results
    poison_data = df_results[
        (df_results['dataset_name'] == dataset_name) & 
        (df_results['data_type'] == data_type) & 
        (df_results['n'] == n) & 
        (df_results['lambda'] == target_lambda) & 
        (df_results['R'] == R)
    ]
    
    if poison_data.empty:
        return None
    
    # Get upper bound results (using binary_search algorithm)
    upper_bound_data = df_upper_bound[
        (df_upper_bound['dataset_name'] == dataset_name) & 
        (df_upper_bound['data_type'] == data_type) & 
        (df_upper_bound['n'] == n) & 
        (df_upper_bound['lambda'] == target_lambda) & 
        (df_upper_bound['R'] == R) &
        (df_upper_bound['algorithm'] == 'binary_search')
    ]
    
    if upper_bound_data.empty:
        return None
    
    # Merge data
    poison_data = poison_data[['lambda', 'seed', LOSS_COLUMN]].sort_values(by=['lambda', 'seed'])
    upper_bound_data = upper_bound_data[['lambda', 'seed', UPPER_BOUND_COLUMN]].sort_values(by=['lambda', 'seed'])
    
    merged_data = poison_data.merge(upper_bound_data, on=['lambda', 'seed'], how='inner')
    
    if merged_data.empty:
        return None
    
    # Calculate L_G/L_{UB}
    merged_data['Lgr_divided_by_Lub'] = merged_data[LOSS_COLUMN] / merged_data[UPPER_BOUND_COLUMN]
    
    # Return average value
    return merged_data['Lgr_divided_by_Lub'].mean()

def create_rank_plot_subplot(ax, data: np.ndarray, distribution: str, n: int, data_type: str, 
                           df_results=None, df_upper_bound=None, R: int = 0, ylim=None):
    """
    Create rank plot as a subplot
    
    Args:
        ax: matplotlib axes object
        data: Data array
        distribution: Distribution name
        n: Data size
        data_type: Data type (uint32/uint64)
        df_results: Result dataframe (for L_G/L_{UB} calculation)
        df_upper_bound: Upper bound dataframe (for L_G/L_{UB} calculation)
        R: R value
        ylim: y-axis range (tuple of (ymin, ymax))
    """
    # For real datasets, subtract minimum value to adjust offset
    if distribution in ['books', 'fb', 'osm']:
        min_value = np.min(data)
        max_value = np.max(data)
        adjusted_data = data - min_value
        print(f"  {distribution} n={n} {data_type}: min_value = {min_value}, max_value = {max_value}")
    else:
        adjusted_data = data
        min_value = 0
    
    # Calculate ranks (0-based)
    ranks = np.arange(len(adjusted_data))
    
    # Create plot
    ax.scatter(adjusted_data, ranks, c='black', s=MARKER_SIZE, zorder=2)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Set ylim if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # # Show L_G/L_{UB} value in bottom right
    # if df_results is not None and df_upper_bound is not None:
    #     lg_lub_value = get_Lgr_divided_by_Lub_value(df_results, df_upper_bound, distribution, n, data_type, R)
    #     if lg_lub_value is not None:
    #         # Add text in bottom right
    #         ax.text(0.95, 0.05, f'L_G/L_{{UB}} = {lg_lub_value:.3f}', 
    #                transform=ax.transAxes, fontsize=16, 
    #                horizontalalignment='right', verticalalignment='bottom',
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def calculate_ylim_for_n(n: int, available_combinations_with_R: list, seed_files: dict) -> tuple:
    """
    Calculate y-axis range for a specific n value
    
    Args:
        n: Data size
        available_combinations_with_R: List of available combinations
        seed_files: Dictionary of seed files
        
    Returns:
        y-axis range (ymin, ymax)
    """
    all_ranks = []
    
    for comb in available_combinations_with_R:
        distribution, comb_n, data_type, R = comb
        if comb_n != n:
            continue
            
        if distribution not in seed_files or n not in seed_files[distribution]:
            continue
            
        if data_type not in seed_files[distribution][n]:
            continue
            
        filepath = seed_files[distribution][n][data_type]
        
        try:
            data = read_binary_data(filepath)
            
            # For real datasets, subtract minimum value to adjust offset
            if distribution in ['books', 'fb', 'osm']:
                min_value = np.min(data)
                adjusted_data = data - min_value
            else:
                adjusted_data = data
            
            # Calculate ranks (0-based)
            ranks = np.arange(len(adjusted_data))
            all_ranks.extend(ranks)
            
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            continue
    
    if not all_ranks:
        return (0, 1)  # Default value
    
    range = max(all_ranks) - min(all_ranks)
    return (min(all_ranks) - range * 0.05, max(all_ranks) + range * 0.05)

def main(seed: int = 0):
    """
    Main function
    
    Args:
        seed: Target seed value (default: 0)
    """
    # print("Starting dataset plot generation...")
    
    # Set scientific notation format
    plt.rcParams['font.size'] = TICK_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = XLABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    
    # Set path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    output_dir = os.path.join(script_dir, "..", "results", "fig", "datasets")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load data for L_G/L_{UB} calculation
    # print("Loading data for L_G/L_{UB} calculation...")
    base_dir = os.path.join(script_dir, "..")
    df_results = load_loss(base_dir)
    df_upper_bound = load_upper_bound(base_dir)
    
    # Target n values
    target_n_values = [1000]
    
    # Search for files with specified seed
    seed_files = find_seed_files(data_dir, target_n_values, seed)
    
    if not seed_files:
        print(f"seed{seed} file not found.")
        raise FileNotFoundError(f"seed{seed} file not found.")
    
    # Define distribution order (only include those in DATASET_NAMES)
    distribution_order = []
    for dataset_key in sorted(DATASET_NAMES.keys(), key=lambda x: DATASET_NAMES[x][0]):
        distribution, data_type, R = dataset_key
        if distribution not in distribution_order:
            distribution_order.append(distribution)
    
    # Collect available combinations of distribution and n values
    available_combinations = []
    for distribution in distribution_order:
        if distribution not in seed_files:
            continue
            
        for n in target_n_values:
            if n not in seed_files[distribution]:
                continue
                
            n_files = seed_files[distribution][n]
            for data_type in n_files.keys():
                # For artificial datasets (uniform, normal, exponential), target only R=100000
                if distribution in ['uniform', 'normal', 'exponential']:
                    # Target only R=100000 files
                    filepath = n_files[data_type]
                    if 'R100000' not in filepath:
                        continue
                
                # Target only datasets in DATASET_NAMES
                R = 100000 if distribution in ['uniform', 'normal', 'exponential'] else 0
                dataset_key = (distribution, data_type, R)
                if dataset_key not in DATASET_NAMES:
                    continue
                
                available_combinations.append((distribution, n, data_type))
    
    # For artificial datasets, set R=100000, for real datasets, set R=0
    available_combinations_with_R = []
    for distribution, n, data_type in available_combinations:
        if distribution in ['uniform', 'normal', 'exponential']:
            R = 100000
        else:
            R = 0
        available_combinations_with_R.append((distribution, n, data_type, R))
    
    if not available_combinations_with_R:
        print("No available datasets found.")
        raise FileNotFoundError(f"No available datasets found.")
    
    # Get list of distributions and n values (including R and data type)
    # For books, treat uint32 and uint64 as separate distributions
    distributions_with_R = []
    for comb in available_combinations_with_R:
        distribution, n, data_type, R = comb
        if distribution == 'books':
            # For books, treat data type as part of the distribution
            distributions_with_R.append((distribution, data_type, R))
        else:
            # For other distributions, proceed as usual
            distributions_with_R.append((distribution, R))
    
    # Remove duplicates and sort
    distributions_with_R = sorted(list(set(distributions_with_R)), 
                                key=lambda x: (distribution_order.index(x[0]) if x[0] in distribution_order else len(distribution_order), x[1] if len(x) == 2 else x[2]))
    n_values = sorted(list(set([comb[1] for comb in available_combinations_with_R])))
    
    # Set plot size (matching plot_poisoned_datasets.py style)
    plt.rcParams['figure.figsize'] = [COL_WIDTH * len(distributions_with_R) + 0.5, ROW_HEIGHT * len(n_values) + 0.5]
    fig, axes = plt.subplots(len(n_values), len(distributions_with_R))
    
    # Handle case where axes is 1D
    if len(n_values) == 1:
        if len(distributions_with_R) == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(1, -1)
    elif len(distributions_with_R) == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate y-axis range for each n value in advance
    ylims_by_n = {}
    for n in n_values:
        ylims_by_n[n] = calculate_ylim_for_n(n, available_combinations_with_R, seed_files)
    
    # Plot data for each subplot
    for j, dist_info in enumerate(distributions_with_R):
        for i, n in enumerate(n_values):
            ax = axes[i][j]
            
            # Analyze distribution information
            if len(dist_info) == 3:  # For books
                distribution, data_type, R = dist_info
            else:  # For other distributions
                distribution, R = dist_info
                data_type = None  # Determine later
            
            # Find data for this combination
            if data_type is not None:
                # For books, specify data type
                matching_combinations = [comb for comb in available_combinations_with_R 
                                       if comb[0] == distribution and comb[1] == n and comb[2] == data_type and comb[3] == R]
            else:
                # For other distributions, data type is determined later
                matching_combinations = [comb for comb in available_combinations_with_R 
                                       if comb[0] == distribution and comb[1] == n and comb[3] == R]
            
            if not matching_combinations:
                # If no data, plot empty
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                continue
            
            # Determine data type
            if data_type is None:
                # For other distributions, use the first found data type
                data_type = matching_combinations[0][2]
            
            filepath = seed_files[distribution][n][data_type]
            
            try:
                # Read data
                data = read_binary_data(filepath)
                
                # Plot data (including L_G/L_{UB} value)
                create_rank_plot_subplot(ax, data, distribution, n, data_type, 
                                       df_results, df_upper_bound, R, ylims_by_n[n])
                
            except Exception as e:
                print(f"  Error: {filepath} processing: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings (matching plot_poisoned_datasets.py style)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.tick_params(axis='x', labelsize=TICK_SIZE)
            # Adjust scientific notation format for x-axis (keep ×10^{6} display)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            ax.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            if i == 0:
                # Get dataset name (including R and data type)
                dataset_key = (distribution, data_type, R)
                dataset_info = DATASET_NAMES.get(dataset_key, None)
                if dataset_info is not None:
                    # If obtained from DATASET_NAMES, use the second element (label)
                    dataset_name = dataset_info[1]
                else:
                    # If not found, use default name
                    dataset_name = f"{distribution} ({data_type})"
                ax.set_title(dataset_name, fontsize=FONT_SIZE)
            if j == 0:
                if len(n_values) == 1:
                    ax.set_ylabel('Rank', fontsize=FONT_SIZE)
                else:
                    ax.set_ylabel(rf'$\mathbf{{n={n}}}$' + '\n' + 'Rank', fontsize=FONT_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save file
    output_filename = f"datasets_rank_plots_seed{seed}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"File saved: {output_path}")

if __name__ == "__main__":
    # SEED = 16
    SEED = 0
    main(SEED)
