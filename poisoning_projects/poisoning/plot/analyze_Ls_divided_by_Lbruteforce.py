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
from load_optimal_poison import load_optimal_poison
from plot_config import LOSS_COLUMN, UPPER_BOUND_COLUMN, DATASET_NAMES_BRUTE_FORCE
import statistics

def analyze_L_divided_by_Lbruteforce(df_loss, df_optimal):
    merged = pd.merge(
        df_loss, 
        df_optimal, 
        on=['dataset_name', 'data_type', 'R', 'n', 'lambda', 'poisoning_percentage', 'seed'],
        suffixes=('_loss', '_optimal')
    )
    merged['L_div_Lbruteforce'] = merged[LOSS_COLUMN + '_loss'] / merged[LOSS_COLUMN + '_optimal']
    mean_, min_, max_ = statistics.mean(merged['L_div_Lbruteforce']), min(merged['L_div_Lbruteforce']), max(merged['L_div_Lbruteforce'])
    return mean_, min_, max_, len(merged)

def analyze_UB_divided_by_Lbruteforce(df_upper_bound, df_optimal):
    merged = pd.merge(
        df_upper_bound, 
        df_optimal, 
        on=['dataset_name', 'data_type', 'R', 'n', 'lambda', 'poisoning_percentage', 'seed'],
        suffixes=('_ub', '_optimal')
    )
    merged['UB_div_Lbruteforce'] = merged[UPPER_BOUND_COLUMN] / merged[LOSS_COLUMN]
    mean_, min_, max_ = statistics.mean(merged['UB_div_Lbruteforce']), min(merged['UB_div_Lbruteforce']), max(merged['UB_div_Lbruteforce'])
    return mean_, min_, max_, len(merged)

def main():
    # Set path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..")

    def preprocess_df(df, n=None, poisoning_percentages=None, Rs=None, algorithm=None):
        df = df.copy()
        df['poisoning_percentage'] = (df['lambda'] / df['n'] * 100).astype(int)
        df = df[df['n'] == n]
        df = df[df['poisoning_percentage'].isin(poisoning_percentages)]
        if Rs is not None:
            df = df[df['R'].isin(Rs)]
        if algorithm is not None:
            df = df[df['algorithm'] == algorithm]
        return df
    
    # Load data
    n = 50
    poisoning_percentages = [2, 4, 6, 8, 10]
    Rs = [0, 1000]
    algorithm="binary_search"

    # print("Loading data...")
    df_greedy_loss = load_loss(data_dir)
    df_greedy_relaxed_loss = load_loss(data_dir, approach="duplicate_allowed")
    df_sege_loss = load_loss(data_dir, approach="consecutive_w_endpoints")
    df_sege_heu_loss = load_loss(data_dir, approach="consecutive_w_endpoints_using_relaxed_solution")
    df_sege_relaxed_loss = load_loss(data_dir, approach="consecutive_w_endpoints_duplicate_allowed")
    df_upper_bound = load_upper_bound(data_dir)
    df_optimal_poison = load_optimal_poison(data_dir)

    df_greedy_loss = preprocess_df(df_greedy_loss, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)
    df_greedy_relaxed_loss = preprocess_df(df_greedy_relaxed_loss, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)
    df_sege_loss = preprocess_df(df_sege_loss, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)
    df_sege_heu_loss = preprocess_df(df_sege_heu_loss, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)
    df_sege_relaxed_loss = preprocess_df(df_sege_relaxed_loss, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)
    df_upper_bound = preprocess_df(df_upper_bound, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs, algorithm=algorithm)
    df_optimal_poison = preprocess_df(df_optimal_poison, n=n, poisoning_percentages=poisoning_percentages, Rs=Rs)

    df_optimal_poinson_relaxed = df_optimal_poison[df_optimal_poison['algorithm'] == 'brute_force_duplicate_allowed']
    df_optimal_poison = df_optimal_poison[df_optimal_poison['algorithm'] != 'brute_force_duplicate_allowed']

    print(f"Greedy loss results: {len(df_greedy_loss)} entries")
    print(F"Greedy relaxed loss results: {len(df_greedy_relaxed_loss)} entries")
    print(f"SEGE loss results: {len(df_sege_loss)} entries")
    print(f"SEGE heuristic loss results: {len(df_sege_heu_loss)} entries")
    print(f"SEGE relaxed loss results: {len(df_sege_relaxed_loss)} entries")
    print(f"Upper bound results: {len(df_upper_bound)} entries")
    print(f"Optimal poison results: {len(df_optimal_poison)} entries")
    print(f"Optimal poison relaxed results: {len(df_optimal_poinson_relaxed)} entries")

    # df_greedy_loss.to_csv('df_greedy_loss.csv', index=False)
    # df_greedy_relaxed_loss.to_csv('df_greedy_relaxed_loss.csv', index=False)
    # df_sege_loss.to_csv('df_sege_loss.csv', index=False)
    # df_sege_heu_loss.to_csv('df_sege_heu_loss.csv', index=False)
    # df_sege_relaxed_loss.to_csv('df_sege_relaxed_loss.csv', index=False)
    # df_upper_bound.to_csv('df_upper_bound.csv', index=False)
    # df_optimal_poison.to_csv('df_optimal_poison.csv', index=False)
    # df_optimal_poinson_relaxed.to_csv('df_optimal_poinson_relaxed.csv', index=False)

    mean_, min_, max_, count_ = analyze_L_divided_by_Lbruteforce(df_greedy_loss, df_optimal_poison)
    print(F"L_G / L_OPT              : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_L_divided_by_Lbruteforce(df_greedy_relaxed_loss, df_optimal_poinson_relaxed)
    print(F"L_G / L_OPT (Relaxed)    : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_L_divided_by_Lbruteforce(df_sege_loss, df_optimal_poison)
    print(F"L_SEG+E / L_OPT          : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_L_divided_by_Lbruteforce(df_sege_heu_loss, df_optimal_poison)
    print(F"L_SEG+E(HEU.) / L_OPT    : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_L_divided_by_Lbruteforce(df_sege_relaxed_loss, df_optimal_poinson_relaxed)
    print(F"L_SEG / L_OPT (Relaxed)  : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_UB_divided_by_Lbruteforce(df_upper_bound, df_optimal_poison)
    print(F"UB / L_OPT:              : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")
    mean_, min_, max_, count_ = analyze_UB_divided_by_Lbruteforce(df_upper_bound, df_optimal_poinson_relaxed)
    print(F"UB / L_OPT (Relaxed).    : ({count_}) {mean_:.16f} [{min_:.16f}, {max_:.16f}]")


if __name__ == "__main__":
    main()
