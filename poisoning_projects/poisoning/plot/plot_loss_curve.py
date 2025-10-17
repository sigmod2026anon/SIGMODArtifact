#!/usr/bin/env python3
"""
Loss curve plot script
Plot MSE changes when adding different poison values to legitimate data.
Horizontal axis: poison value, vertical axis: MSE after adding poison
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Dict, Any, Optional, List, Tuple
from matplotlib.colors import TwoSlopeNorm

from plot_config import (
    TICK_SIZE, XLABEL_SIZE, FONT_SIZE,
    DATASET_NAMES,
    DATASET_NAMES_BRUTE_FORCE
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
MARKER_SIZE = 20
ROW_HEIGHT = 4.75
COL_WIDTH = 5.5

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
        assert self.algorithm in ["greedy", "brute_force"], f"Algorithm {self.algorithm} not found in ['greedy', 'brute_force']"

    def get_distribution_title(self) -> str:
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

    def read_legitimate_data(self) -> np.ndarray:
        legitimate_file_path = self._get_legitimate_file_path()
        return self._convert_to_start_from_0(self._read_binary_data(legitimate_file_path))

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

    def calculate_loss_after_insert(self, poison_value: float, position: int) -> float:
        """
        Calculate MSE after inserting a poison value at a specific position
        This is a simplified version of the C++ calc_loss.cpp logic
        """
        legitimate_data = self.read_legitimate_data()
        
        # Insert poison value at the specified position
        data_with_poison = np.insert(legitimate_data, position, poison_value)
        
        # Calculate ranks (1-based for loss calculation)
        ranks = np.arange(1, len(data_with_poison) + 1)
        
        if len(data_with_poison) < 2:
            return 0.0
        
        # Calculate means
        x_mean = np.mean(data_with_poison)
        y_mean = np.mean(ranks)
        
        # Calculate variance and covariance
        x_var = np.var(data_with_poison, ddof=0)  # Population variance for loss calculation
        xy_cov = np.cov(data_with_poison, ranks, ddof=0)[0, 1]  # Population covariance
        
        # Calculate loss (MSE)
        if x_var > 0:
            loss = np.var(ranks, ddof=0) - (xy_cov * xy_cov) / x_var
        else:
            loss = np.var(ranks, ddof=0)
        
        return loss

    def calculate_mse_with_poison_values(self, poison_values: np.ndarray, legitimate_data: Optional[np.ndarray] = None) -> float:
        """
        Calculate MSE after adding multiple poison values (e.g., two values) to legitimate data.
        The added poison values are merged with legitimate data and sorted to reflect proper ranks.
        """
        if legitimate_data is None:
            legitimate_data = self.read_legitimate_data()
        data_with_poison = np.concatenate([legitimate_data.astype(np.float64), np.asarray(poison_values, dtype=np.float64)])

        # Sort to ensure ranks correspond to sorted order of keys
        data_with_poison.sort()

        ranks = np.arange(1, len(data_with_poison) + 1)

        if len(data_with_poison) < 2:
            return 0.0

        x_var = np.var(data_with_poison, ddof=0)
        xy_cov = np.cov(data_with_poison, ranks, ddof=0)[0, 1]

        if x_var > 0:
            loss = np.var(ranks, ddof=0) - (xy_cov * xy_cov) / x_var
        else:
            loss = np.var(ranks, ddof=0)

        return loss

    def find_optimal_insertion_position(self, poison_value: float) -> Tuple[int, float]:
        """
        Find the optimal position to insert poison value to minimize MSE
        Returns: (position, min_mse)
        """
        legitimate_data = self.read_legitimate_data()
        n = len(legitimate_data)
        
        min_mse = float('inf')
        best_position = 0
        
        # Try inserting at each possible position
        for pos in range(n + 1):
            mse = self.calculate_loss_after_insert(poison_value, pos)
            if mse < min_mse:
                min_mse = mse
                best_position = pos
        
        return best_position, min_mse


def create_loss_curve_subplot(ax, config: PlotConfig, num_poison_values: int = 500):
    """
    Create loss curve plot as a subplot
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
        num_poison_values: number of poison values to test
    """
    legitimate_data = config.read_legitimate_data()
    
    # Calculate legitimate MSE
    legitimate_mse = config.calculate_linear_regression_legitimate()[2]
    
    # Generate poison values from min to max of legitimate data
    min_val = np.min(legitimate_data)
    max_val = np.max(legitimate_data)
    poison_values = np.linspace(min_val, max_val, num_poison_values)
    
    # Calculate MSE for each poison value
    mse_values = []
    for poison_val in poison_values:
        _, mse = config.find_optimal_insertion_position(poison_val)
        mse_values.append(mse)
    
    # Plot the loss curve
    ax.plot(poison_values, mse_values, 'r-', linewidth=2, zorder=2, label='MSE after poison insertion')
    
    # Plot legitimate MSE as horizontal line
    ax.axhline(y=legitimate_mse, color='green', linestyle='--', linewidth=2, 
               zorder=1, label=f'Legitimate MSE: {legitimate_mse:.2f}')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Add legend
    # ax.legend(fontsize=FONT_SIZE - 4)
    
    # Adjust scientific notation format for x-axis
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))


def create_derivative_subplot(ax, config: PlotConfig, num_poison_values: int = 500):
    """
    Create derivative plot as a subplot
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
        num_poison_values: number of poison values to test
    """
    legitimate_data = config.read_legitimate_data()
    
    # Generate poison values from min to max of legitimate data
    min_val = np.min(legitimate_data)
    max_val = np.max(legitimate_data)
    poison_values = np.linspace(min_val, max_val, num_poison_values)
    
    # Calculate MSE for each poison value
    mse_values = []
    for poison_val in poison_values:
        _, mse = config.find_optimal_insertion_position(poison_val)
        mse_values.append(mse)
    
    # Calculate derivative (finite difference)
    mse_values = np.array(mse_values)
    poison_values = np.array(poison_values)
    
    # Calculate differences between adjacent MSE values
    mse_diff = np.diff(mse_values)
    poison_diff = np.diff(poison_values)
    
    # Calculate derivative (mse_diff / poison_diff)
    derivative = mse_diff / poison_diff
    
    # Plot derivative (use poison_values[:-1] since we lose one point in diff)
    ax.plot(poison_values[:-1], derivative, 'b-', linewidth=2, zorder=2, label='MSE derivative')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Add legend
    # ax.legend(fontsize=FONT_SIZE - 4)
    
    # Adjust scientific notation format for x-axis
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))


def create_legitimate_rank_subplot(ax, config: PlotConfig):
    """
    Create legitimate rank plot as a subplot (only legitimate data)
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
    """
    # Get legitimate data
    legitimate_data = config.read_legitimate_data()
    
    # Calculate ranks (0-based)
    ranks = np.arange(len(legitimate_data))
    
    # Create scatter plot with black for legitimate data
    ax.scatter(legitimate_data, ranks, c='black', s=MARKER_SIZE, zorder=2)

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
                fontsize=FONT_SIZE + 2, ha='right', va='bottom', fontweight='bold')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Adjust scientific notation format for x-axis (add ×10^{6} display)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))


def create_residual_subplot(ax, config: PlotConfig):
    """
    Create residual plot as a subplot (negative of residual errors)
    
    Args:
        ax: matplotlib axes object
        config: PlotConfig object
    """
    # Get legitimate data
    legitimate_data = config.read_legitimate_data()
    
    # Calculate ranks (0-based)
    ranks = np.arange(len(legitimate_data))
    
    # Calculate linear regression parameters
    slope, intercept, mse = config.calculate_linear_regression_legitimate()
    
    # Calculate predicted values
    predicted_ranks = slope * legitimate_data + intercept
    
    # Calculate residuals (actual - predicted)
    residuals = ranks - predicted_ranks
    
    # Plot negative of residuals
    negative_residuals = -residuals
    ax.scatter(legitimate_data, negative_residuals, c='purple', s=MARKER_SIZE, zorder=2)
    
    # Connect points with thin lines
    ax.plot(legitimate_data, negative_residuals, 'purple', linewidth=0.8, alpha=0.7, zorder=1)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    
    # Adjust scientific notation format for x-axis
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))


def plot_loss_curves(target_configs: List[PlotConfig], 
                    column_seeds: List[int],
                    output_filename: str = "loss_curves.pdf",
                    vertical_layout: bool = True):
    """
    Create loss curve plots arranged in a grid with 4 rows: MSE, derivative, legitimate rank, and residuals
    
    Args:
        target_configs: List of PlotConfig objects
        column_seeds: List of seeds for column titles
        output_filename: Output filename
        vertical_layout: If True, distributions are arranged vertically (4 rows x num_cols)
                        If False, distributions are arranged horizontally (num_cols x 4 columns)
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
    
    num_configs = len(target_configs)
    
    if vertical_layout:
        # Vertical layout: 4 rows x num_configs columns
        plt.rcParams['figure.figsize'] = [COL_WIDTH * num_configs + 0.5, ROW_HEIGHT * 4 + 0.5]
        fig, axes = plt.subplots(4, num_configs)
        
        # Handle case where axes is 1D
        if num_configs == 1:
            axes = axes.reshape(4, 1)
        
        # Plot data for each subplot
        for j, config in enumerate(target_configs):
            print(f"config: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} R={config.R} data_type={config.data_type} algorithm={config.algorithm}")
            
            # First row: Rank plots
            ax_rank = axes[0][j]
            try:
                create_legitimate_rank_subplot(ax_rank, config)
                print("Legitimate rank plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} rank processing: {e}")
                ax_rank.text(0.5, 0.5, 'Error', ha='center', va='center', 
                            transform=ax_rank.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for rank plots
            ax_rank.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_rank.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_rank.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            
            # Set title for rank plots
            title = f"{config.get_distribution_title()} (seed={column_seeds[j]})"
            # ax_rank.set_title(title, fontsize=FONT_SIZE)
            
            # Set y-label for leftmost rank plot
            if j == 0:
                ax_rank.set_ylabel('Rank', fontsize=FONT_SIZE)
            
            # Second row: Residual plots
            ax_resid = axes[1][j]
            try:
                create_residual_subplot(ax_resid, config)
                print("Residual plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} residual processing: {e}")
                ax_resid.text(0.5, 0.5, 'Error', ha='center', va='center', 
                             transform=ax_resid.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for residual plots
            ax_resid.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_resid.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_resid.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            
            # Set y-label for leftmost residual plot
            if j == 0:
                ax_resid.set_ylabel('-Residual', fontsize=FONT_SIZE)
            
            # Third row: MSE plots
            ax_mse = axes[2][j]
            try:
                create_loss_curve_subplot(ax_mse, config)
                print("MSE plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} MSE processing: {e}")
                ax_mse.text(0.5, 0.5, 'Error', ha='center', va='center', 
                           transform=ax_mse.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for MSE plots
            ax_mse.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_mse.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_mse.set_xlabel('Poison Value', fontsize=XLABEL_SIZE)
            
            # Set y-label for leftmost MSE plot
            if j == 0:
                ax_mse.set_ylabel('MSE', fontsize=FONT_SIZE)
            
            # Fourth row: Derivative plots
            ax_deriv = axes[3][j]
            try:
                create_derivative_subplot(ax_deriv, config)
                print("Derivative plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} derivative processing: {e}")
                ax_deriv.text(0.5, 0.5, 'Error', ha='center', va='center', 
                             transform=ax_deriv.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for derivative plots
            ax_deriv.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_deriv.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_deriv.set_xlabel('Poison Value', fontsize=XLABEL_SIZE)
            
            # Set y-label for leftmost derivative plot
            if j == 0:
                ax_deriv.set_ylabel('MSE Derivative', fontsize=FONT_SIZE)
    
    else:
        # Horizontal layout: num_configs rows x 4 columns
        plt.rcParams['figure.figsize'] = [COL_WIDTH * 4 + 0.5, ROW_HEIGHT * num_configs + 0.5]
        fig, axes = plt.subplots(num_configs, 4)
        
        # Handle case where axes is 1D
        if num_configs == 1:
            axes = axes.reshape(1, 4)
        
        # Plot data for each subplot
        for i, config in enumerate(target_configs):
            print(f"config: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} R={config.R} data_type={config.data_type} algorithm={config.algorithm}")
            
            # First column: Rank plots
            ax_rank = axes[i][0]
            try:
                create_legitimate_rank_subplot(ax_rank, config)
                print("Legitimate rank plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} rank processing: {e}")
                ax_rank.text(0.5, 0.5, 'Error', ha='center', va='center', 
                            transform=ax_rank.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for rank plots
            ax_rank.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_rank.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_rank.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            ax_rank.set_ylabel('Rank', fontsize=FONT_SIZE)
            
            # Second column: Residual plots
            ax_resid = axes[i][1]
            try:
                create_residual_subplot(ax_resid, config)
                print("Residual plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} residual processing: {e}")
                ax_resid.text(0.5, 0.5, 'Error', ha='center', va='center', 
                             transform=ax_resid.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for residual plots
            ax_resid.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_resid.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_resid.set_xlabel('Keys', fontsize=XLABEL_SIZE)
            ax_resid.set_ylabel('-Residual', fontsize=FONT_SIZE)
            
            # Third column: MSE plots
            ax_mse = axes[i][2]
            try:
                create_loss_curve_subplot(ax_mse, config)
                print("MSE plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} MSE processing: {e}")
                ax_mse.text(0.5, 0.5, 'Error', ha='center', va='center', 
                           transform=ax_mse.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for MSE plots
            ax_mse.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_mse.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_mse.set_xlabel('Poison Value', fontsize=XLABEL_SIZE)
            ax_mse.set_ylabel('MSE', fontsize=FONT_SIZE)
            
            # Fourth column: Derivative plots
            ax_deriv = axes[i][3]
            try:
                create_derivative_subplot(ax_deriv, config)
                print("Derivative plot Success")
            except Exception as e:
                print(f"  Error: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} derivative processing: {e}")
                ax_deriv.text(0.5, 0.5, 'Error', ha='center', va='center', 
                             transform=ax_deriv.transAxes, fontsize=14, color='red')
                continue
            
            # Common settings for derivative plots
            ax_deriv.tick_params(axis='both', labelsize=TICK_SIZE)
            ax_deriv.tick_params(axis='x', labelsize=TICK_SIZE)
            ax_deriv.set_xlabel('Poison Value', fontsize=XLABEL_SIZE)
            ax_deriv.set_ylabel('MSE Derivative', fontsize=FONT_SIZE)
            
            # Add distribution name to ylabel of the leftmost plot (Rank plot)
            distribution_name = config.get_distribution_title()
            ax_rank.set_ylabel(f"{distribution_name}\nRank", fontsize=FONT_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save file
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"File saved: {output_path}")


def create_two_poison_mse_heatmap(ax,
                                 config: Optional[PlotConfig] = None,
                                 num_poison_values: int = 200,
                                 K: Optional[List[float]] = None,
                                 show_colorbar_label: bool = True):
    """
    Create heatmap of MSE(K ∪ {p2} ∪ {p1}) where x-axis is p2 and y-axis is p1.
    """
    # Determine base keys (K)
    if K is not None:
        base_keys = np.asarray(K, dtype=np.float64)
    else:
        assert config is not None, "Either config or K must be provided"
        base_keys = config.read_legitimate_data().astype(np.float64)

    # Determine poison axis ticks with uniform 100-division (or specified num)
    min_val = float(np.min(base_keys))
    max_val = float(np.max(base_keys))
    p_values = np.linspace(min_val, max_val, num_poison_values)

    mse_grid = np.zeros((num_poison_values, num_poison_values), dtype=np.float64)

    for j, p1 in enumerate(p_values):
        for i, p2 in enumerate(p_values):
            data_with_poison = np.concatenate([base_keys, np.array([p2, p1], dtype=np.float64)])
            data_with_poison.sort()
            ranks = np.arange(1, len(data_with_poison) + 1)
            x_var = np.var(data_with_poison, ddof=0)
            xy_cov = np.cov(data_with_poison, ranks, ddof=0)[0, 1]
            mse_grid[j, i] = (np.var(ranks, ddof=0) - (xy_cov * xy_cov) / x_var) if x_var > 0 else np.var(ranks, ddof=0)

    im = ax.imshow(
        mse_grid,
        origin='lower',
        extent=[p_values[0], p_values[-1], p_values[0], p_values[-1]],
        aspect='equal',
        cmap='viridis'
    )

    # Mark the (p1, p2) points that yield the maximum MSE (within a tolerance) and print them
    try:
        tol = 1e-12
        max_mse = float(np.max(mse_grid))
        near_max_mask = np.abs(mse_grid - max_mse) <= tol
        indices = np.argwhere(near_max_mask)
        if indices.size > 0:
            p1_points = p_values[indices[:, 0]]
            p2_points = p_values[indices[:, 1]]
            # Scatter all near-maximum points
            ax.scatter(p2_points, p1_points, marker='x', c='red', s=240, linewidths=5.0, zorder=3)
            # Print each point
            for jj, ii in indices:
                print(f"[MSE-argmax] p1={p_values[jj]}, p2={p_values[ii]}, MSE={mse_grid[jj, ii]}, tol={tol}")
    except Exception as e:
        print(f"[MSE-argmax] failed to compute: {e}")

    ax.set_xlabel('$p_2$')
    ax.set_ylabel('$p_1$')
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_two_poison_derivative_heatmap(ax,
                                        config: Optional[PlotConfig] = None,
                                        num_poison_values: int = 200,
                                        K: Optional[List[float]] = None,
                                        show_colorbar_label: bool = True):
    """
    Create heatmap of ∂/∂p2 MSE(K ∪ {p2} ∪ {p1}) where x-axis is p2 and y-axis is p1.
    """
    if K is not None:
        base_keys = np.asarray(K, dtype=np.float64)
    else:
        assert config is not None, "Either config or K must be provided"
        base_keys = config.read_legitimate_data().astype(np.float64)

    min_val = float(np.min(base_keys))
    max_val = float(np.max(base_keys))
    p_values = np.linspace(min_val, max_val, num_poison_values)

    mse_grid = np.zeros((num_poison_values, num_poison_values), dtype=np.float64)
    for j, p1 in enumerate(p_values):
        for i, p2 in enumerate(p_values):
            data_with_poison = np.concatenate([base_keys, np.array([p2, p1], dtype=np.float64)])
            data_with_poison.sort()
            ranks = np.arange(1, len(data_with_poison) + 1)
            x_var = np.var(data_with_poison, ddof=0)
            xy_cov = np.cov(data_with_poison, ranks, ddof=0)[0, 1]
            mse_grid[j, i] = (np.var(ranks, ddof=0) - (xy_cov * xy_cov) / x_var) if x_var > 0 else np.var(ranks, ddof=0)

    # Finite forward difference along p2 axis (x-axis)
    derivative_grid = np.zeros_like(mse_grid)
    dp = np.diff(p_values)
    for j in range(num_poison_values):
        for i in range(num_poison_values - 1):
            derivative_grid[j, i] = (mse_grid[j, i + 1] - mse_grid[j, i]) / dp[i]
        # Fill last column with the previous derivative to keep shape consistent
        derivative_grid[j, -1] = derivative_grid[j, -2]

    im = ax.imshow(
        derivative_grid,
        origin='lower',
        extent=[p_values[0], p_values[-1], p_values[0], p_values[-1]],
        aspect='equal',
        cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0.0)
    )

    ax.set_xlabel('$p_2$')
    ax.set_ylabel('$p_1$')
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_two_poison_mixed_derivative_heatmap(ax,
                                               config: Optional[PlotConfig] = None,
                                               num_poison_values: int = 200,
                                               K: Optional[List[float]] = None,
                                               show_colorbar_label: bool = True,
                                               vrange: Optional[Tuple[float, float]] = None):
    """
    Create heatmap of ∂^2 MSE/∂p1 ∂p2 for K ∪ {p1} ∪ {p2}.
    x-axis is p2, y-axis is p1. Uses central cross difference.
    """
    if K is not None:
        base_keys = np.asarray(K, dtype=np.float64)
    else:
        assert config is not None, "Either config or K must be provided"
        base_keys = config.read_legitimate_data().astype(np.float64)

    min_val = float(np.min(base_keys))
    max_val = float(np.max(base_keys))
    p_values = np.linspace(min_val, max_val, num_poison_values)

    mse_grid = np.zeros((num_poison_values, num_poison_values), dtype=np.float64)
    for j, p1 in enumerate(p_values):
        for i, p2 in enumerate(p_values):
            data_with_poison = np.concatenate([base_keys, np.array([p2, p1], dtype=np.float64)])
            data_with_poison.sort()
            ranks = np.arange(1, len(data_with_poison) + 1)
            x_var = np.var(data_with_poison, ddof=0)
            xy_cov = np.cov(data_with_poison, ranks, ddof=0)[0, 1]
            mse_grid[j, i] = (np.var(ranks, ddof=0) - (xy_cov * xy_cov) / x_var) if x_var > 0 else np.var(ranks, ddof=0)

    dp = p_values[1] - p_values[0] if num_poison_values > 1 else 1.0
    mixed_grid = np.zeros_like(mse_grid)
    # Compute mixed derivative via sequential central differences to improve stability
    # First derivative w.r.t. p2 (x-axis)
    dfdx = np.zeros_like(mse_grid)
    dfdx[:, 1:-1] = (mse_grid[:, 2:] - mse_grid[:, :-2]) / (2.0 * dp)
    dfdx[:, 0] = dfdx[:, 1]
    dfdx[:, -1] = dfdx[:, -2]
    # Then derivative of that w.r.t. p1 (y-axis)
    mixed_grid[1:-1, :] = (dfdx[2:, :] - dfdx[:-2, :]) / (2.0 * dp)
    mixed_grid[0, :] = mixed_grid[1, :]
    mixed_grid[-1, :] = mixed_grid[-2, :]

    # Fixed color scale
    # vmin, vmax = (-0.00002, 0.00002) if vrange is None else (vrange[0], vrange[1])
    vmin, vmax = (vrange[0], vrange[1])

    im = ax.imshow(
        mixed_grid,
        origin='lower',
        extent=[p_values[0], p_values[-1], p_values[0], p_values[-1]],
        aspect='equal',
        cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    )

    ax.set_xlabel('$p_2$')
    ax.set_ylabel('$p_1$')
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_two_poison_heatmaps(target_configs: List[PlotConfig],
                             column_seeds: List[int],
                             output_filename: str = "two_poison_heatmaps.pdf",
                             vertical_layout: bool = True,
                             num_poison_values: int = 500):
    """
    Plot, for each config, two rows (or columns) of heatmaps:
      - MSE(K ∪ {p2} ∪ {p1})
      - ∂/∂p2 MSE(K ∪ {p2} ∪ {p1})
    """
    plt.rcParams['font.size'] = TICK_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = XLABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "results", "fig", "datasets")
    os.makedirs(output_dir, exist_ok=True)

    num_configs = len(target_configs)

    if vertical_layout:
        # 3 rows x num_configs columns (MSE, derivative, mixed)
        plt.rcParams['figure.figsize'] = [(COL_WIDTH * num_configs + 0.5) * 1.5, ((COL_WIDTH * 3) + 0.5) * 1.5]
        fig, axes = plt.subplots(3, num_configs)
        if num_configs == 1:
            axes = axes.reshape(3, 1)

        for j, config in enumerate(target_configs):
            print(f"[TwoPoison] config: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} R={config.R} data_type={config.data_type} algorithm={config.algorithm}")

            # Row 0: MSE
            ax_mse = axes[0][j]
            try:
                create_two_poison_mse_heatmap(ax_mse, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  MSE heatmap Success")
            except Exception as e:
                print(f"  Error (MSE heatmap): {e}")
                ax_mse.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mse.transAxes, fontsize=14, color='red')

            title = f"{config.get_distribution_title()} (seed={column_seeds[j]})"
            # ax_mse.set_title(title, fontsize=FONT_SIZE)
            if j == 0:
                ax_mse.set_ylabel(r"$p_1$", fontsize=FONT_SIZE)

            # Row 1: First derivative
            ax_deriv = axes[1][j]
            try:
                create_two_poison_derivative_heatmap(ax_deriv, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  Derivative heatmap Success")
            except Exception as e:
                print(f"  Error (Derivative heatmap): {e}")
                ax_deriv.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_deriv.transAxes, fontsize=14, color='red')

            ax_deriv.set_xlabel(r"$p_2$", fontsize=XLABEL_SIZE)
            if j == 0:
                ax_deriv.set_ylabel(r"$p_1$", fontsize=FONT_SIZE)

            # Row 2: Mixed second derivative
            ax_mixed = axes[2][j]
            try:
                create_two_poison_mixed_derivative_heatmap(ax_mixed, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  Mixed derivative heatmap Success")
            except Exception as e:
                print(f"  Error (Mixed derivative heatmap): {e}")
                ax_mixed.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mixed.transAxes, fontsize=14, color='red')

            ax_mixed.set_xlabel(r"$p_2$", fontsize=XLABEL_SIZE)
            if j == 0:
                ax_mixed.set_ylabel(r"$p_1$", fontsize=FONT_SIZE)
    else:
        # num_configs rows x 3 columns (MSE, derivative, mixed)
        plt.rcParams['figure.figsize'] = [(COL_WIDTH * 3 + 0.5) * 1.5, ((COL_WIDTH * num_configs) + 0.5) * 1.5]
        fig, axes = plt.subplots(num_configs, 3)
        if num_configs == 1:
            axes = axes.reshape(1, 3)

        for i, config in enumerate(target_configs):
            print(f"[TwoPoison] config: {config.distribution} n={config.n} seed={config.seed} lambda={config.lambda_val} R={config.R} data_type={config.data_type} algorithm={config.algorithm}")

            # Col 0: MSE
            ax_mse = axes[i][0]
            try:
                create_two_poison_mse_heatmap(ax_mse, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  MSE heatmap Success")
            except Exception as e:
                print(f"  Error (MSE heatmap): {e}")
                ax_mse.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mse.transAxes, fontsize=14, color='red')
            ax_mse.set_ylabel('p1', fontsize=FONT_SIZE)

            # Col 1: First derivative
            ax_deriv = axes[i][1]
            try:
                create_two_poison_derivative_heatmap(ax_deriv, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  Derivative heatmap Success")
            except Exception as e:
                print(f"  Error (Derivative heatmap): {e}")
                ax_deriv.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_deriv.transAxes, fontsize=14, color='red')
            ax_deriv.set_ylabel(r"$p_1$", fontsize=FONT_SIZE)
            ax_deriv.set_xlabel(r"$p_2$", fontsize=XLABEL_SIZE)

            # Col 2: Mixed second derivative
            ax_mixed = axes[i][2]
            try:
                create_two_poison_mixed_derivative_heatmap(ax_mixed, config, num_poison_values=num_poison_values, show_colorbar_label=False)
                print("  Mixed derivative heatmap Success")
            except Exception as e:
                print(f"  Error (Mixed derivative heatmap): {e}")
                ax_mixed.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mixed.transAxes, fontsize=14, color='red')
            ax_mixed.set_ylabel(r"$p_1$", fontsize=FONT_SIZE)
            ax_mixed.set_xlabel(r"$p_2$", fontsize=XLABEL_SIZE)

    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"File saved: {output_path}")


def main():
    # Example configurations for loss curve plots
    # Using the same seeds as in plot_poisoned_datasets.py
    # column_seeds = [51, 79, 4, 78, 11, 50]
    column_seeds = [0, 1, 2]
    
    target_configs = [
        PlotConfig('uniform', 50, column_seeds[0], 0, 1000, "uint64", "greedy"),
        PlotConfig('normal', 50, column_seeds[1], 0, 1000, "uint64", "greedy"),
        PlotConfig('exponential', 50, column_seeds[2], 0, 1000, "uint64", "greedy"),
    ]
    
    # # Vertical layout (default): distributions arranged vertically
    # plot_loss_curves(target_configs, column_seeds, "loss_curves_vertical.pdf", vertical_layout=True)
    
    # Horizontal layout: distributions arranged horizontally
    plot_loss_curves(target_configs, column_seeds, "loss_curves_horizontal.pdf", vertical_layout=False)

    # Two-poison heatmaps (horizontal layout per distribution)
    plot_two_poison_heatmaps(target_configs, column_seeds, output_filename="two_poison_heatmaps.pdf", vertical_layout=False, num_poison_values=100)

    # Fixed dataset K
    xs = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 72]
    
    # Create fixed K heatmaps using abstracted functions
    plt.rcParams['font.size'] = TICK_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = XLABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "results", "fig", "datasets")
    os.makedirs(output_dir, exist_ok=True)

    # Create 1 row x 3 columns layout (MSE, first derivative, mixed derivative)
    plt.rcParams['figure.figsize'] = [(COL_WIDTH * 3 + 0.5) * 1.5, (COL_WIDTH + 0.5) * 1.5]
    fig, axes = plt.subplots(1, 3)

    print(f"[FixedK] Creating heatmaps for K = {xs}")

    # First column: MSE heatmap
    ax_mse = axes[0]
    try:
        create_two_poison_mse_heatmap(ax_mse, K=xs)
        print("  MSE heatmap Success")
    except Exception as e:
        print(f"  Error (MSE heatmap): {e}")
        ax_mse.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mse.transAxes, fontsize=14, color='red')

    ax_mse.set_title('$MSE(K \\cup \\{p_1\\} \\cup \\{p_2\\})$', fontsize=FONT_SIZE)
    ax_mse.set_ylabel('$p_2$', fontsize=FONT_SIZE)

    # Second column: Derivative heatmap
    ax_deriv = axes[1]
    try:
        create_two_poison_derivative_heatmap(ax_deriv, K=xs)
        print("  Derivative heatmap Success")
    except Exception as e:
        print(f"  Error (Derivative heatmap): {e}")
        ax_deriv.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_deriv.transAxes, fontsize=14, color='red')

    ax_deriv.set_title('$\\partial MSE/\\partial p_1$', fontsize=FONT_SIZE)
    ax_deriv.set_ylabel('$p_2$', fontsize=FONT_SIZE)
    ax_deriv.set_xlabel('$p_1$', fontsize=XLABEL_SIZE)

    # Third column: Mixed second derivative heatmap
    ax_mixed = axes[2]
    try:
        create_two_poison_mixed_derivative_heatmap(ax_mixed, K=xs, vrange=(-0.0015, 0.0015))
        print("  Mixed derivative heatmap Success")
    except Exception as e:
        print(f"  Error (Mixed derivative heatmap): {e}")
        ax_mixed.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_mixed.transAxes, fontsize=14, color='red')

    ax_mixed.set_title('$\\partial^2 MSE/\\partial p_1 \\partial p_2$', fontsize=FONT_SIZE)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "two_poison_heatmaps_fixed_k.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"File saved: {output_path}")

    # target_configs = [
    #     PlotConfig('uniform', 10, column_seeds[0], 0, 1000, "uint64", "greedy"),
    #     PlotConfig('normal', 10, column_seeds[1], 0, 1000, "uint64", "greedy"),
    #     PlotConfig('exponential', 10, column_seeds[2], 0, 1000, "uint64", "greedy"),
    #     PlotConfig('books', 10, column_seeds[3], 0, 0, "uint64", "greedy"),
    #     PlotConfig('fb', 10, column_seeds[4], 0, 0, "uint64", "greedy"),
    #     PlotConfig('osm', 10, column_seeds[5], 0, 0, "uint64", "greedy")
    # ]
    
    # Vertical layout (default): distributions arranged vertically
    # plot_loss_curves(target_configs, column_seeds, "loss_curves_vertical_n10.pdf", vertical_layout=True)
    
    # Horizontal layout: distributions arranged horizontally
    # plot_loss_curves(target_configs, column_seeds, "loss_curves_horizontal_n10.pdf", vertical_layout=False)

    # Counter-example configurations
    # column_seeds_counter = [0, 0, 1, 78, 11, 50]
    
    # target_configs_counter = [
    #     PlotConfig('uniform', 50, column_seeds_counter[0], 0, 950, "uint64", "greedy"),
    #     PlotConfig('normal', 50, column_seeds_counter[1], 0, 950, "uint64", "greedy"),
    #     PlotConfig('exponential', 50, column_seeds_counter[2], 0, 950, "uint64", "greedy"),
    #     PlotConfig('books', 50, column_seeds_counter[3], 0, 0, "uint64", "greedy"),
    #     PlotConfig('fb', 50, column_seeds_counter[4], 0, 0, "uint64", "greedy"),
    #     PlotConfig('osm', 50, column_seeds_counter[5], 0, 0, "uint64", "greedy")
    # ]
    
    # plot_loss_curves(target_configs_counter, column_seeds_counter, "loss_curves_counter_example.pdf")


if __name__ == "__main__":
    main() 