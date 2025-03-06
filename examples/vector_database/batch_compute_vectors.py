"""
Use skypilot to launch managed jobs that will run the embedding calculation.  

This script is responsible for:
1. Launching a monitoring service cluster
2. Splitting the input dataset among several workers
3. Launching worker clusters with unique worker IDs
"""

#!/usr/bin/env python3

import argparse
import os
import time
import uuid
import sky


def calculate_job_range(start_idx: int, end_idx: int, job_rank: int,
                        total_jobs: int) -> tuple[int, int]:
    """Calculate the range of indices this job should process.
    
    Args:
        start_idx: Global start index
        end_idx: Global end index
        job_rank: Current job's rank (0-based)
        total_jobs: Total number of jobs
        
    Returns:
        Tuple of [job_start_idx, job_end_idx)
    """
    total_range = end_idx - start_idx
    chunk_size = total_range // total_jobs
    remainder = total_range % total_jobs

    # Distribute remainder across first few jobs
    job_start = start_idx + (job_rank * chunk_size) + min(job_rank, remainder)
    if job_rank < remainder:
        chunk_size += 1
    job_end = job_start + chunk_size

    return job_start, job_end


def main():
    parser = argparse.ArgumentParser(
        description='Launch batch CLIP inference jobs')
    parser.add_argument('--start-idx',
                        type=int,
                        default=0,
                        help='Global start index in dataset')
    parser.add_argument('--end-idx',
                        type=int,
                        default= 1000, #29475453, # text
                        help='Global end index in dataset, not inclusive')
    parser.add_argument('--num-jobs',
                        type=int,
                        default=2,
                        help='Number of jobs to partition the work across')
    parser.add_argument('--env-path',
                        type=str,
                        default='~/.env',
                        help='Path to the environment file')
    parser.add_argument('--skip-monitor',
                        action='store_true',
                        help='Skip launching the monitoring service')
    parser.add_argument('--bucket-name',
                        type=str,
                        default='sky-embeddings',
                        help='Name of the bucket to store embeddings')
    args = parser.parse_args()

    # Try to get HF_TOKEN from environment first, then ~/.env file
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        env_path = os.path.expanduser(args.env_path)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        hf_token = line.strip().split('=')[1]
                        break

    if not hf_token:
        raise ValueError("HF_TOKEN not found in ~/.env or environment variable")


    # make sure every run use different bucket name
    args.bucket_name = f"{args.bucket_name}-{str(uuid.uuid4())[:4]}"
    # eric: todo; this doesn't work

    # Launch monitoring service first (unless skipped)
    print("Launching monitoring service cluster...")
    monitor_task = sky.Task.from_yaml('monitor_progress.yaml')
    monitor_task.update_envs({
        'EMBEDDINGS_BUCKET_NAME': args.bucket_name,
    })  
    monitor_job = sky.launch(monitor_task)

    # Load the worker task template
    task = sky.Task.from_yaml('compute_text_vectors.yaml')

    # Launch jobs for each partition
    for job_rank in range(args.num_jobs):
        # Calculate index range for this job
        job_start, job_end = calculate_job_range(args.start_idx, args.end_idx,
                                                 job_rank, args.num_jobs)
        
        # Create a unique worker ID
        worker_id = f"worker_{job_rank}"

        # Update environment variables for this job
        task_copy = task.update_envs({
            'START_IDX': str(job_start),  # Convert to string for env vars
            'END_IDX': str(job_end),
            'HF_TOKEN': hf_token,
            'WORKER_ID': worker_id,
            'EMBEDDINGS_BUCKET_NAME': args.bucket_name,
        })

        job_name = f'vector-compute-{job_start}-{job_end}'
        print(f"Launching {job_name} with {worker_id}...")
        
        sky.jobs.launch(
            task_copy,
            name=job_name,
        )

if __name__ == '__main__':
    main()
