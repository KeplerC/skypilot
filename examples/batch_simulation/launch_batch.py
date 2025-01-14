#!/usr/bin/env python3

import argparse
import os
import sky

def main():
    parser = argparse.ArgumentParser(description='Launch batch CARLA simulations')
    parser.add_argument('--delta_k_values', type=int, nargs='+', default=[20],
                      help='List of delta_k values to test')
    parser.add_argument('--brake_threshold_values', type=float, nargs='+', 
                      default=[0.3],
                      help='List of emergency brake threshold values to test')
    parser.add_argument('--config_types', type=str, nargs='+', 
                      default=['right_turn'],
                      help='List of configuration types to test')
    parser.add_argument('--num_runs', type=int, default=100,
                      help='Number of runs per configuration')
    args = parser.parse_args()

    # Load the task template
    task = sky.Task.from_yaml('carla.yaml')

    # Launch jobs for each parameter combination
    job_idx = 1
    for config_type in args.config_types:
        for delta_k in args.delta_k_values:
            for brake_threshold in args.brake_threshold_values:
                # Update environment variables for this configuration
                task.update_envs({
                    'DELTA_K': delta_k,
                    'BRAKE_THRESHOLD': brake_threshold,
                    'CONFIG_TYPE': config_type,
                    'NUM_RUNS': args.num_runs
                })
                
                # Launch the job
                sky.jobs.launch(
                    task,
                    name=f'carla-sim-{config_type}-dk{delta_k}-bt{brake_threshold}',
                    detach_run=True,
                    retry_until_up=True,
                )
                job_idx += 1

if __name__ == '__main__':
    main() 