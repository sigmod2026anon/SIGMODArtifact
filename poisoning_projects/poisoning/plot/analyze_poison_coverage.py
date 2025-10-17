#!/usr/bin/env python3
"""
Analysis script for poison coverage verification
Checks if all integers from poison min to poison max are covered by either
original legitimate data or poison data (no empty gaps)
"""

import os
import struct
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Set
import glob
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

class PoisonCoverageAnalyzer:
    def __init__(self, data_type: str = "uint64"):
        self.data_type = data_type

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

    def _get_legitimate_file_path(self, poison_file_path: str) -> str:
        """
        Get the corresponding legitimate file path from poison file path
        """
        # Remove '_lambda5_optimal_poison' from the filename
        legitimate_path = poison_file_path.replace('_lambda5_optimal_poison', '')
        return legitimate_path

    def _is_a_segment(self, poison_file_path: str) -> bool:
        """
        Check if the poison file is a segment file
        """
        try:
            # Read poisoned data (combined legitimate + poison)
            poisoned_data = self._read_binary_data(poison_file_path)
            
            # Get legitimate file path
            legitimate_file_path = self._get_legitimate_file_path(poison_file_path)
            
            # Read legitimate data
            legitimate_data = self._read_binary_data(legitimate_file_path)
            
            # Extract actual poison data (poisoned_data - legitimate_data)
            legitimate_set = set(legitimate_data)
            poisons = []
            for value in poisoned_data:
                if value not in legitimate_set:
                    poisons.append(value)
            
            # Get poison min and max from actual poison data
            if len(poisons) > 0:
                poison_min = int(np.min(poisons))
                poison_max = int(np.max(poisons))
            else:
                poison_min = 0
                poison_max = 0
            
            # Create sets for efficient lookup
            legitimate_set = set(legitimate_data)
            poison_set = set(poisons)
            
            # Check coverage property - stop at first missing value for efficiency
            missing_value = None
            
            for value in range(poison_min, poison_max + 1):
                if value in legitimate_set or value in poison_set:
                    continue
                else:
                    missing_value = value
                    break
            
            is_segment = (missing_value is None)
            return is_segment
        except FileNotFoundError as e:
            return False
        except Exception as e:
            return False

    def _is_a_segment_w_endpoints(self, poison_file_path: str) -> bool:
        if self._is_a_segment(poison_file_path):
            return True
        
        return False

    def check_coverage_property(self, poison_file_path: str) -> Dict[str, Any]:
        """
        Check if the coverage property holds for a given poison file
        
        Property: All integers from poison min to poison max are covered by
        either original legitimate data or poison data (no empty gaps)
        
        Args:
            poison_file_path: Path to the poison file
            
        Returns:
            dict: Dictionary containing analysis results
        """
        try:
            is_segment = self._is_a_segment(poison_file_path)
            is_segment_w_endpoints = self._is_a_segment_w_endpoints(poison_file_path)
            
            result = {
                'poison_file': os.path.basename(poison_file_path),
                'is_segment': is_segment,
                'is_segment_w_endpoints': is_segment_w_endpoints,
                'error': None
            }
            
            return result
            
        except FileNotFoundError as e:
            return {
                'poison_file': os.path.basename(poison_file_path),
                'is_segment': None,
                'is_segment_w_endpoints': None,
                'error': f"File not found: {e}"
            }
        except Exception as e:
            return {
                'poison_file': os.path.basename(poison_file_path),
                'is_segment': None,
                'is_segment_w_endpoints': None,
                'error': f"Error: {e}"
            }

    def analyze_all_poison_files(self, pattern: str = "*_lambda5_optimal_poison_uint64") -> List[bool]:
        """
        Analyze all poison files matching the pattern
        
        Args:
            pattern: Glob pattern to match poison files
            
        Returns:
            list: List of boolean results (True if property holds, False otherwise)
        """
        # Find all poison files
        poison_files = glob.glob(os.path.join(DATA_DIR, pattern))
        poison_files.sort()
        
        print(f"Found {len(poison_files)} poison files to analyze")
        
        results = []
        is_segment_sum = 0
        is_segment_w_endpoints_sum = 0
        
        for i, poison_file in enumerate(poison_files):
            print(f"Analyzing {i+1}/{len(poison_files)}: {os.path.basename(poison_file)}")
            
            result = self.check_coverage_property(poison_file)
            
            if result['error'] is None:
                is_segment = result['is_segment']
                is_segment_w_endpoints = result['is_segment_w_endpoints']
                results.append(
                    {
                        "file": poison_file,
                        "is_segment": is_segment,
                        "is_segment_w_endpoints": is_segment_w_endpoints
                    }
                )
                if is_segment:
                    is_segment_sum += 1
                if is_segment_w_endpoints:
                    is_segment_w_endpoints_sum += 1
            else:
                print(f"  ERROR: {result['error']}")
        
        print(f"\nSummary:")
        print(f"Total files: {len(results)}")
        print(f"Is segment: {is_segment_sum}/{len(results)} ({is_segment_sum/len(results)*100:.2f}%)")
        print(f"Is segment with endpoints: {is_segment_w_endpoints_sum}/{len(results)} ({is_segment_w_endpoints_sum/len(results)*100:.2f}%)")

        return results


def main():
    """
    Main function
    """
    print("Starting poison coverage analysis...")
    
    # Create analyzer
    analyzer = PoisonCoverageAnalyzer()
    
    # Analyze all poison files
    results = analyzer.analyze_all_poison_files()
    
    # print(f"\nFinal result: {sum(results)}/{len(results)} files satisfy the property ({sum(results)/len(results)*100:.2f}%)")
    
    # Save failed files to text file
    print(f"Failed files:")
    failed_files = {
        "uniform": [],
        "normal": [],
        "exponential": [],
        "books": [],
        "fb": [],
        "osm": []
    }
    all_files = {
        "uniform": [],
        "normal": [],
        "exponential": [],
        "books": [],
        "fb": [],
        "osm": []
    }
    for i, result in enumerate(results):
        for key in failed_files.keys():
            if key in result["file"]:
                all_files[key].append(result["file"])
                if not result["is_segment"]:
                    failed_files[key].append(result["file"])
                break
    
    for key in failed_files.keys():
        print(f"{key}: {len(failed_files[key])}/{len(all_files[key])} files failed")
        # for file in failed_files[key]:
        #     print(file)
        # print()

if __name__ == "__main__":
    main()
