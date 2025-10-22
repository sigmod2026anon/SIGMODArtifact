#!/usr/bin/env python3
"""
Poisoned dataset plot script
Compare legitimate and poisoned datasets, plotting legitimate points in blue and poisoned points in red.
Horizontal axis: value, vertical axis: rank
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Dict, Any, Optional, List, Tuple

from plot_config import (
    TICK_SIZE, XLABEL_SIZE, FONT_SIZE,
    DATASET_NAMES,
    DATASET_NAMES_BRUTE_FORCE
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
MARKER_SIZE = 5
ROW_HEIGHT = 5.25
COL_WIDTH = 5

class PlotConfig:
    def __init__(self, distribution: str, n: int, seed: int, lambda_val: int, R: int = 0, data_type: str = "uint64", algorithm: str = "greedy"):
        self.distribution = distribution
        self.n = n
        self.seed = seed
        self.lambda_val = lambda_val
        self.R = R
        self.data_type = data_type
        self.algorithm = algorithm

        assert self.distribution in [dist[0] for dist in DATASET_NAMES.keys()], f"Distribution {self.distribution} not found in DATASET_NAMES"
        assert self.algorithm in ["greedy", "brute_force", "consecutive_w_endpoints"], f"Algorithm {self.algorithm} not found in ['greedy', 'brute_force', 'consecutive_w_endpoints']"

    def get_distribution_title(self) -> str:
        # return DATASET_NAMES_BRUTE_FORCE[(self.distribution, self.data_type, self.R)][1]
        return self.distribution

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

    def _get_poisoned_file_path(self) -> str:
        """Get the path to the poisoned data file"""
        if self.algorithm == "greedy":
            if self.R == 0:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_seed{self.seed}_lambda{self.lambda_val}_{self.data_type}")
            else:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_R{self.R}_seed{self.seed}_lambda{self.lambda_val}_{self.data_type}")
        elif self.algorithm == "brute_force":
            if self.R == 0:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_seed{self.seed}_lambda{self.lambda_val}_optimal_poison_{self.data_type}")
            else:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_R{self.R}_seed{self.seed}_lambda{self.lambda_val}_optimal_poison_{self.data_type}")
        elif self.algorithm == "consecutive_w_endpoints":
            if self.R == 0:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_seed{self.seed}_lambda{self.lambda_val}_consecutive_w_endpoints_{self.data_type}")
            else:
                return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_R{self.R}_seed{self.seed}_lambda{self.lambda_val}_consecutive_w_endpoints_{self.data_type}")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _get_legitimate_file_path(self) -> str:
        """Get the path to the legitimate data file"""
        if self.R == 0:
            return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_seed{self.seed}_{self.data_type}")
        else:
            return os.path.join(DATA_DIR, f"{self._distribution_to_str_on_file()}_n{self.n}_R{self.R}_seed{self.seed}_{self.data_type}")

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

    def _convert_to_start_from_0(self, data: np.ndarray) -> np.ndarray:
        return data - np.min(data)

    def read_poisoned_data(self) -> np.ndarray:
        poisoned_file_path = self._get_poisoned_file_path()
        return self._convert_to_start_from_0(self._read_binary_data(poisoned_file_path))

    def read_legitimate_data(self) -> np.ndarray:
        legitimate_file_path = self._get_legitimate_file_path()
        return self._convert_to_start_from_0(self._read_binary_data(legitimate_file_path))

    def get_poisoned_data_with_label(self) -> Tuple[np.ndarray, np.ndarray]:
        poisoned_data = self.read_poisoned_data()
        legitimate_data = self.read_legitimate_data()
        poisoned_data_w_id = [(p, i) for i, p in enumerate(poisoned_data)]
        for l in legitimate_data:
            for p, i in poisoned_data_w_id:
                if l == p:
                    poisoned_data_w_id.remove((p, i))
                    break
        poison_ids = [i for (_, i) in poisoned_data_w_id]
        labels = [0 if i in poison_ids else 1 for i in range(len(poisoned_data))]
        return poisoned_data, labels
    
    def calculate_linear_regression(self) -> Tuple[float, float, float]:
        """
        Calculate linear regression parameters (slope, intercept, MSE) for poisoned data
        using explicit calculation with variance and covariance
        
        Returns:
            Tuple of (slope, intercept, mse)
        """
        poisoned_data = self.read_poisoned_data()
        ranks = np.arange(len(poisoned_data))
        
        if len(poisoned_data) < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate means
        x_mean = np.mean(poisoned_data)
        y_mean = np.mean(ranks)
        
        # Calculate variance and covariance
        x_var = np.var(poisoned_data, ddof=1)  # Sample variance
        xy_cov = np.cov(poisoned_data, ranks, ddof=1)[0, 1]  # Sample covariance
        
        # Calculate slope and intercept
        if x_var > 0:
            slope = xy_cov / x_var
        else:
            slope = 0.0
        
        intercept = y_mean - slope * x_mean
        
        # Calculate MSE
        y_pred = slope * poisoned_data + intercept
        mse = np.mean((ranks - y_pred) ** 2)
        
        return slope, intercept, mse

    def calculate_linear_regression_legitimate(self) -> Tuple[float, float, float]:
        """
        Calculate linear regression parameters (slope, intercept, MSE) for legitimate data
        using explicit calculation with variance and covariance
        
        Returns:
            Tuple of (slope, intercept, mse)
        """
        legitimate_data = self.read_legitimate_data()
        ranks = np.arange(len(legitimate_data))
        
        if len(legitimate_data) < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate means
        x_mean = np.mean(legitimate_data)
        y_mean = np.mean(ranks)
        
        # Calculate variance and covariance
        x_var = np.var(legitimate_data, ddof=1)  # Sample variance
        xy_cov = np.cov(legitimate_data, ranks, ddof=1)[0, 1]  # Sample covariance
        
        # Calculate slope and intercept
        if x_var > 0:
            slope = xy_cov / x_var
        else:
            slope = 0.0
        
        intercept = y_mean - slope * x_mean
        
        # Calculate MSE
        y_pred = slope * legitimate_data + intercept
        mse = np.mean((ranks - y_pred) ** 2)
        
        return slope, intercept, mse


def create_legitimate_rank_plot_subplot(ax, config: PlotConfig, xlim=None, ylim=None):
    """
    Create legitimate rank plot as a subplot (only legitimate data)
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
        ylim: y-axis range (tuple of (ymin, ymax))
    """
    # Get legitimate data
    legitimate_data = config.read_legitimate_data()
    
    # Calculate ranks (0-based)
    ranks = np.arange(len(legitimate_data))
    
    # Create scatter plot with black for legitimate data
    ax.scatter(legitimate_data, ranks, c='black', s=MARKER_SIZE, zorder=2, alpha=0.7)

    # Linear regression for legitimate data
    if len(legitimate_data) > 1:
        # Calculate linear regression parameters
        slope, intercept, mse = config.calculate_linear_regression_legitimate()
        
        # Plot regression line
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.8, zorder=1)
        
        ax.text(0.95, 0.05, f'MSE: {mse:.1f}', transform=ax.transAxes, 
                fontsize=FONT_SIZE, ha='right', va='bottom', fontweight='bold')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Set ylim if specified
    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Adjust scientific notation format for x-axis (add ×10^{6} display)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))


def create_poisoned_rank_plot_subplot(ax, config: PlotConfig, xlim=None, ylim=None):
    """
    Create poisoned rank plot as a subplot using get_poisoned_data_with_label
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
        ylim: y-axis range (tuple of (ymin, ymax))
    """
    # Get poisoned data with labels using the method
    poisoned_data, labels = config.get_poisoned_data_with_label()
    
    # Calculate ranks (0-based)
    ranks = np.arange(len(poisoned_data))
    
    # Separate legitimate and poisoned data based on labels
    legitimate_mask = np.array(labels) == 1
    poisoned_mask = np.array(labels) == 0
    
    legitimate_data = poisoned_data[legitimate_mask]
    legitimate_ranks = ranks[legitimate_mask]
    poisoned_data = poisoned_data[poisoned_mask]
    poisoned_ranks = ranks[poisoned_mask]

    # print(f"legitimate: {legitimate_data}")
    # print(f"poisones: {poisoned_data}")
    
    # Create scatter plot with blue for legitimate and red for poisoned
    if len(legitimate_data) > 0:
        ax.scatter(legitimate_data, legitimate_ranks, c='black', s=MARKER_SIZE, zorder=2, alpha=0.7)
    if len(poisoned_data) > 0:
        ax.scatter(poisoned_data, poisoned_ranks, c='red', s=MARKER_SIZE, zorder=2, alpha=0.7)

    # Linear regression for poisoned data
    if len(poisoned_data) > 1:
        # Calculate linear regression parameters using PlotConfig method
        slope, intercept, mse = config.calculate_linear_regression()
        
        # Plot regression line
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.8, zorder=1)
        

        ax.text(0.95, 0.05, f'MSE: {mse:.1f}', transform=ax.transAxes, 
                fontsize=FONT_SIZE, ha='right', va='bottom', fontweight='bold')

        # print(f"MSE: mse: {mse:.20f}")
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Set ylim if specified
    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    # Greedy zoom
    # ax.set_xlim(15421889608453 - 10, 15421889608493 + 10)
    # ax.set_ylim(760, 840)

    # segment zoom
    # ax.set_xlim(15420441645944 - 10, 15420441645983 + 10)
    # ax.set_ylim(760, 840)
    
    # Adjust scientific notation format for x-axis (add ×10^{6} display)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))



def plot_poisoned_datasets(target_configs: List[List[PlotConfig]], 
                         column_seeds: List[int],
                         row_titles: List[str],
                         output_filename: str = "poisoned_datasets_rank_plots.pdf",
                         xlim = None,
                         ylim = None):
    """
    Create poisoned dataset plots arranged in a grid
    
    Args:
        target_configs: 2D list of PlotConfig objects
                       Each inner list represents a row of plots
        output_filename: Output filename
    """
    # Set scientific notation format
    plt.rcParams['font.size'] = TICK_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = XLABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    

    
    # Set path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "results", "fig", "datasets")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate y-axis range for all configurations
    
    # Set plot size
    num_rows = len(target_configs)
    num_cols = max(len(row) for row in target_configs)
    plt.rcParams['figure.figsize'] = [COL_WIDTH * num_cols + 0.5, ROW_HEIGHT * num_rows + 0.5]
    fig, axes = plt.subplots(num_rows, num_cols)
    
    # Handle case where axes is 1D
    if num_rows == 1:
        if num_cols == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot data for each subplot
    for i, row in enumerate(target_configs):
        for j, config in enumerate(row):
            ax = axes[i][j]

            # print()
            # print(f"config: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} R={config.R} data_type={config.data_type} algorithm={config.algorithm}")
            
            try:
                if config.lambda_val == 0:
                    create_legitimate_rank_plot_subplot(ax, config, xlim, ylim)
                else:
                    # For other rows (poisoned data), use poisoned plot function
                    create_poisoned_rank_plot_subplot(ax, config, xlim, ylim)
            
                # print()
                
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} processing: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.tick_params(axis='x', labelsize=TICK_SIZE)
            ax.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            
            # Set title
            if i == 0:
                title = f"{config.get_distribution_title()} (seed={column_seeds[j]})"
                ax.set_title(title, fontsize=FONT_SIZE)
            
            # Set y-label for leftmost plots
            if j == 0:
                ax.set_ylabel(rf'$\mathbf{{{row_titles[i]}}}$' + '\nRank', fontsize=FONT_SIZE)

    # Adjust layout
    plt.tight_layout()
    
    # Save file
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"File saved: {output_path}")


def main():
    target_configs = [
        [
            PlotConfig('books', 1000, 0, 0,  0, "uint64", "greedy"),
            PlotConfig('books', 1000, 0, 20, 0, "uint64", "greedy"),
            PlotConfig('books', 1000, 0, 20, 0, "uint64", "consecutive_w_endpoints"),
        ]
    ]
    row_titles = [""]
    column_seeds = [0, 0, 0]
    plot_poisoned_datasets(target_configs, column_seeds, row_titles, "poisoned_datasets_largest_diff_segment_greedy_error_3.pdf")


if __name__ == "__main__":
    main()
