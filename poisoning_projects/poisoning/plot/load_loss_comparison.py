import pandas as pd
import os
import json
import glob

def load_loss_comparison_data(base_dir, approach=None):
    """
    Load loss data from specified approach or both approaches for comparison.
    
    Args:
        base_dir: Base directory path
        approach: Specific approach to load ("original", "consecutive_w_endpoints", 
                 "consecutive_w_endpoints_duplicate_allowed", "duplicate_allowed", or None for original)
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - dataset_name: Name of the dataset
            - n: Dataset size
            - R: Range parameter (for synthetic datasets)
            - seed: Random seed
            - lambda: Poisoning percentage
            - approach: Approach used
            - loss: Loss value
            - source_dir: Source directory name
    """
    
    # Map approach names to directory paths
    approach_dirs = {
        "original": "loss",
        "consecutive_w_endpoints": "loss_consecutive_w_endpoints",
        "consecutive_w_endpoints_duplicate_allowed": "loss_consecutive_w_endpoints_duplicate_allowed",
        "duplicate_allowed": "loss_duplicate_allowed"
    }

    if approach is None:
        approach = "original"
    
    if approach not in approach_dirs:
        raise ValueError(f"Unknown approach: {approach}. Available approaches: {list(approach_dirs.keys())}")
    
    dir_name = approach_dirs[approach]
    loss_dir = os.path.join(base_dir, "results", dir_name)
    
    if not os.path.exists(loss_dir):
        raise FileNotFoundError(f"Loss directory not found: {loss_dir}")
    
    # print(f"Loading {approach} results from: {loss_dir}")
    
    all_data = []
    
    # Load loss data from this directory
    json_files = glob.glob(os.path.join(loss_dir, "**/*.json"), recursive=True)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if required keys exist
            expected_keys = ["dataset_name", "n", "R", "seed", "data_type", "lambda", "loss"]
            if all(key in data for key in expected_keys):
                # Add approach information
                data['approach'] = approach
                data['source_dir'] = dir_name
                all_data.append(data)
            else:
                print(f"Warning: Missing keys in {json_file}")
                print(f"  Expected: {expected_keys}")
                print(f"  Found: {list(data.keys())}")
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {json_file}: {e}")
    
    if not all_data:
        print(f"No valid results found in {loss_dir}")
        print(f"  Directory: {loss_dir}")
        print(f"  JSON files found: {len(json_files)}")
        raise FileNotFoundError(f"No valid results found in {loss_dir}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort required columns (same as load_loss.py)
    expected_columns = ["dataset_name", "n", "R", "seed", "data_type", "lambda", "loss"]
    df = df[expected_columns]
    
    # Handle duplicate data (take average if same parameter set)
    duplicate_key_columns = ["dataset_name", "n", "R", "seed", "data_type", "lambda"]
    duplicates = df.duplicated(subset=duplicate_key_columns, keep=False)
    
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate entries. Taking average of duplicate values.")
        # If there are duplicates, take the average
        df = df.groupby(duplicate_key_columns, as_index=False)['loss'].mean()
        # Reorder columns
        df = df[expected_columns]
    
    return df


if __name__ == "__main__":
    base_dir = ".."
    try:
        # Load consecutive with endpoints approach only
        print("Loading consecutive with endpoints approach...")
        consecutive_w_endpoints_df = load_loss_comparison_data(base_dir, "consecutive_w_endpoints")
        consecutive_w_endpoints_df = consecutive_w_endpoints_df.sort_values(by=["dataset_name", "n", "R", "seed", "data_type", "lambda"])
        print(f"Loaded {len(consecutive_w_endpoints_df)} consecutive with endpoints data points")
        
        
        print("\nSample data:")
        print(consecutive_w_endpoints_df.head())
        
        # Save to CSV
        csv_path = "../results/loss_consecutive_w_endpoints.csv"
        consecutive_w_endpoints_df.to_csv(csv_path, index=False)
        print(f"\nData saved to: {csv_path}")


        # Load original approach only
        print("Loading original approach...")
        original_df = load_loss_comparison_data(base_dir)
        original_df = original_df.sort_values(by=["dataset_name", "n", "R", "seed", "data_type", "lambda"])
        print(f"Loaded {len(original_df)} original data points")
        
        print("\nSample data:")
        print(original_df.head())
        
        # Save to CSV
        csv_path = "../results/loss_original.csv"
        original_df.to_csv(csv_path, index=False)
        print(f"\nData saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
