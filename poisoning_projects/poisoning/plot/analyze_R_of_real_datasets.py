#!/usr/bin/env python3
"""
Analysis script for legitimate dataset ranges (real datasets only)
Calculates statistics (min, max, mean, median) of dataset ranges (max - min) for legitimate real datasets
(books, fb, osm) across seeds 0-99 for specified n values.
"""

import os
import struct
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from plot_config import DATASET_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

class DatasetAnalyzer:
    def __init__(self, distribution: str, n: int, data_type: str = "uint64", R: int = 0):
        self.distribution = distribution
        self.n = n
        self.data_type = data_type
        self.R = R

        assert self.distribution in [dist[0] for dist in DATASET_NAMES.keys()], f"Distribution {self.distribution} not found in DATASET_NAMES"

    def _distribution_to_str_on_file(self) -> str:
        """Get the name of the distribution"""
        if self.distribution == "uniform":
            return "uniform"
        elif self.distribution == "normal":
            return "normal"
        elif self.distribution == "exponential":
            return "exponential"
        elif self.distribution == "books":
            return "books_200M"
        elif self.distribution == "fb":
            return "fb_200M"
        elif self.distribution == "osm":
            return "osm_cellids_200M"
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _get_legitimate_file_path(self, seed: int) -> str:
        """Get the path to the legitimate data file for a specific seed"""
        if self.R == 0:
            return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_seed{seed}_{self.data_type}")
        else:
            return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_R{self.R}_seed{seed}_{self.data_type}")

    def _read_binary_data(self, file_path: str) -> np.ndarray:
        """
        Read data from binary file
        """
        with open(file_path, 'rb') as f:
            size_bytes = f.read(8)
            size = struct.unpack("Q", size_bytes)[0]
            if "uint32" in file_path:
                dtype = np.uint32
            elif "uint64" in file_path:
                dtype = np.uint64
            else:
                raise ValueError(f"Unknown data type in filename: {file_path}")
            data = np.fromfile(f, dtype=dtype)
            assert len(data) == size, f"Data size mismatch: expected {size}, got {len(data)}"
            assert np.array_equal(np.sort(data), data), f"Data is not sorted in {file_path}"
            return data

    def calculate_dataset_range(self, seed: int) -> float:
        """
        Calculate the range (max - min) of a legitimate dataset for a specific seed
        
        Args:
            seed: Seed value for the dataset
            
        Returns:
            float: Range of the dataset (max - min)
        """
        try:
            file_path = self._get_legitimate_file_path(seed)
            data = self._read_binary_data(file_path)
            return float(np.max(data) - np.min(data))
        except FileNotFoundError:
            print(f"  Warning: File not found for {self.distribution} n={self.n} seed={seed}")
            return None
        except Exception as e:
            print(f"  Error reading {self.distribution} n={self.n} seed={seed}: {e}")
            return None

    def analyze_ranges_across_seeds(self, seeds: List[int] = None) -> Dict[str, Any]:
        """
        Analyze dataset ranges across multiple seeds
        
        Args:
            seeds: List of seed values to analyze (default: 0-99)
            
        Returns:
            dict: Dictionary containing statistics about the ranges
        """
        if seeds is None:
            seeds = list(range(100))  # 0-99
        
        ranges = []
        valid_seeds = []
        
        print(f"Analyzing {self.distribution} n={self.n} across {len(seeds)} seeds...")
        
        for seed in seeds:
            range_val = self.calculate_dataset_range(seed)
            if range_val is not None:
                ranges.append(range_val)
                valid_seeds.append(seed)
        
        if not ranges:
            return None
        
        # Calculate statistics
        ranges_array = np.array(ranges)
        stats = {
            'min': float(np.min(ranges_array)),
            'max': float(np.max(ranges_array)),
            'mean': float(np.mean(ranges_array)),
            'median': float(np.median(ranges_array)),
            'std': float(np.std(ranges_array)),
            'count': len(ranges),
            'valid_seeds': valid_seeds,
            'min_seed': valid_seeds[np.argmin(ranges_array)],
            'max_seed': valid_seeds[np.argmax(ranges_array)]
        }
        
        return stats


def analyze_all_datasets(target_n_values: List[int], seeds: List[int] = None) -> Dict[int, Dict[str, Any]]:
    """
    Analyze real datasets for specified n values
    
    Args:
        target_n_values: List of n values to analyze
        seeds: List of seed values to analyze (default: 0-99)
        
    Returns:
        dict: Dictionary with n values as keys and result dictionaries as values
    """
    if seeds is None:
        seeds = list(range(100))  # 0-99
    
    all_results = {}
    
    # Define real datasets (books, fb, osm)
    real_datasets = ["books", "fb", "osm"]
    
    for n in target_n_values:
        print(f"\n=== Analyzing n={n} ===")
        results = {}
        
        # Get available combinations from DATASET_NAMES, but only real datasets
        available_combinations = [
            (name, dtype, R) for name, dtype, R in DATASET_NAMES.keys()
            if name in real_datasets
        ]
        distributions = sorted(available_combinations, key=lambda x: DATASET_NAMES[x][0])
        
        for dist in distributions:
            dataset_name, data_type, R = dist
            dataset_display_name = DATASET_NAMES[dist][1]
            
            analyzer = DatasetAnalyzer(dataset_name, n, data_type, R)
            stats = analyzer.analyze_ranges_across_seeds(seeds)
            
            if stats is not None:
                results[dataset_display_name] = stats
            else:
                results[dataset_display_name] = None
        
        all_results[n] = results
    
    return all_results


def print_table(all_results: Dict[int, Dict[str, Any]], target_n_values: List[int]):
    """
    Output results in table format
    
    Args:
        all_results: Dictionary with n values as keys and result dictionaries as values
        target_n_values: List of target n values
    """
    
    # Get list of dataset names (ordered)
    dataset_names = []
    for n in target_n_values:
        if n in all_results:
            for dataset_name in all_results[n].keys():
                if dataset_name not in dataset_names:
                    dataset_names.append(dataset_name)
    
    print(f"\nDataset Range Statistics (Max - Min)")
    print("=" * 200)
    print(f"Min~Max, Mean, Median, Std, Count")
    
    # Output header
    header = f"{'n':<8}"
    for dataset in dataset_names:
        header += f"{dataset:<40}"
    print(header)
    print("-" * 200)
    
    # Output rows for each n value
    for n in target_n_values:
        if n not in all_results:
            continue
        row = f"{n:<8}"
        for dataset in dataset_names:
            if dataset in all_results[n] and all_results[n][dataset] is not None:
                stats = all_results[n][dataset]
                cell_content = f"{stats['min']:.0f}~{stats['max']:.0f}, {stats['mean']:.0f}, {stats['median']:.0f}, {stats['std']:.0f}, {stats['count']}"
                row += f"{cell_content:<40}"
            else:
                row += f"{'N/A':<40}"
        print(row)
    print("=" * 200)


def save_results_to_file(all_results: Dict[int, Dict[str, Any]], target_n_values: List[int], output_file_path: str = None):
    """
    Save results to CSV file
    
    Args:
        all_results: Dictionary with n values as keys and result dictionaries as values
        target_n_values: List of target n values
        output_file_path: Output file path (default: poisoning_projects/poisoning/results/dataset_range_analysis.csv)
    """
    if output_file_path is None:
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dataset_range_analysis.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for n in target_n_values:
        if n not in all_results:
            continue
            
        for dataset_name, stats in all_results[n].items():
            if stats is not None:
                row = {
                    'n': n,
                    'dataset': dataset_name,
                    'min_range': stats['min'],
                    'max_range': stats['max'],
                    'mean_range': stats['mean'],
                    'median_range': stats['median'],
                    'std_range': stats['std'],
                    'count': stats['count'],
                    'min_seed': stats['min_seed'],
                    'max_seed': stats['max_seed']
                }
                csv_data.append(row)
            else:
                row = {
                    'n': n,
                    'dataset': dataset_name,
                    'min_range': None,
                    'max_range': None,
                    'mean_range': None,
                    'median_range': None,
                    'std_range': None,
                    'count': None,
                    'min_seed': None,
                    'max_seed': None
                }
                csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file_path, index=False)
    
    print(f"\nResults saved to: {output_file_path}")


def print_detailed_stats(all_results: Dict[int, Dict[str, Any]], target_n_values: List[int]):
    """
    Print detailed statistics including min/max seed information
    
    Args:
        all_results: Dictionary with n values as keys and result dictionaries as values
        target_n_values: List of target n values
    """
    print(f"\nDetailed Statistics")
    print("=" * 100)
    
    for n in target_n_values:
        if n not in all_results:
            continue
            
        print(f"\n--- n={n} ---")
        for dataset_name, stats in all_results[n].items():
            if stats is not None:
                print(f"{dataset_name}:")
                print(f"  Min: {stats['min']:.0f} (seed={stats['min_seed']})")
                print(f"  Max: {stats['max']:.0f} (seed={stats['max_seed']})")
                print(f"  Mean: {stats['mean']:.0f}")
                print(f"  Median: {stats['median']:.0f}")
                print(f"  Std: {stats['std']:.0f}")
                print(f"  Count: {stats['count']}")
                print()


def main():
    """
    Main function
    """
    print("Starting legitimate dataset range analysis for real datasets...")
    
    # Target n values
    target_n_values = [50, 1000]
    
    # Analyze real datasets only
    all_results = analyze_all_datasets(target_n_values)
    
    # Print results in table format
    print_table(all_results, target_n_values)
    
    # Print detailed statistics
    print_detailed_stats(all_results, target_n_values)
    
    # Save results to file
    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dataset_range_analysis.csv")
    save_results_to_file(all_results, target_n_values, output_file_path)


if __name__ == "__main__":
    main()