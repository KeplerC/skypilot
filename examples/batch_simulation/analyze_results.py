#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(results_dir):
    """Analyze results from batch simulation runs."""
    # Find all statistics.csv files
    results = []
    for root, dirs, files in os.walk(results_dir):
        if 'statistics.csv' in files:
            # Parse directory structure to get parameters
            parts = Path(root).parts
            config_type = parts[-4]
            delta_k = int(parts[-3])
            brake_threshold = float(parts[-2])
            run_number = int(parts[-1])
            
            # Read statistics file
            df = pd.read_csv(os.path.join(root, 'statistics.csv'))
            
            # Add configuration parameters
            df['config_type'] = config_type
            df['delta_k'] = delta_k
            df['brake_threshold'] = brake_threshold
            df['run_number'] = run_number
            
            results.append(df)
    
    # Combine all results
    if not results:
        print("No results found!")
        return
    
    combined_df = pd.concat(results, ignore_index=True)
    
    # Calculate aggregate statistics
    agg_stats = combined_df.groupby(['config_type', 'delta_k', 'brake_threshold']).agg({
        'collision': ['mean', 'std', 'count'],
        'delta_k_used': ['mean', 'std']
    }).reset_index()
    
    # Rename columns for clarity
    agg_stats.columns = ['config_type', 'delta_k', 'brake_threshold', 
                        'collision_rate', 'collision_std', 'num_runs',
                        'avg_delta_k_used', 'delta_k_std']
    
    # Save aggregate statistics
    output_file = os.path.join(results_dir, 'aggregate_statistics.csv')
    agg_stats.to_csv(output_file, index=False)
    print(f"Saved aggregate statistics to {output_file}")
    
    # Create visualizations
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Collision rate vs brake threshold for each delta_k
    plt.figure(figsize=(10, 6))
    for delta_k in agg_stats['delta_k'].unique():
        data = agg_stats[agg_stats['delta_k'] == delta_k]
        plt.errorbar(data['brake_threshold'], data['collision_rate'], 
                    yerr=data['collision_std'], 
                    label=f'delta_k={delta_k}',
                    marker='o')
    
    plt.xlabel('Emergency Brake Threshold')
    plt.ylabel('Collision Rate')
    plt.title('Collision Rate vs Emergency Brake Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'collision_rate_vs_threshold.png'))
    plt.close()
    
    # Plot 2: Average delta_k used vs brake threshold
    plt.figure(figsize=(10, 6))
    for delta_k in agg_stats['delta_k'].unique():
        data = agg_stats[agg_stats['delta_k'] == delta_k]
        plt.errorbar(data['brake_threshold'], data['avg_delta_k_used'],
                    yerr=data['delta_k_std'],
                    label=f'delta_k={delta_k}',
                    marker='o')
    
    plt.xlabel('Emergency Brake Threshold')
    plt.ylabel('Average Delta K Used')
    plt.title('Average Delta K Used vs Emergency Brake Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'delta_k_vs_threshold.png'))
    plt.close()
    
    # Plot 3: Heatmap of collision rates
    if len(agg_stats['delta_k'].unique()) > 1:
        pivot_data = agg_stats.pivot(index='delta_k', 
                                   columns='brake_threshold',
                                   values='collision_rate')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Collision Rate Heatmap')
        plt.savefig(os.path.join(plot_dir, 'collision_rate_heatmap.png'))
        plt.close()
    
    print(f"Generated plots in {plot_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("------------------")
    print(agg_stats.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Analyze batch simulation results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing simulation results')
    args = parser.parse_args()
    
    analyze_results(args.results_dir)

if __name__ == '__main__':
    main() 