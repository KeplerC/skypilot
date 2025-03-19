#!/usr/bin/env python3
import re
import sys
from datetime import timedelta

def parse_duration(duration_str):
    """Parse duration string to minutes"""
    if not duration_str or duration_str == '-':
        return 0
    
    total_minutes = 0
    # Match hours, minutes, and seconds
    hours_match = re.search(r'(\d+)h', duration_str)
    mins_match = re.search(r'(\d+)m', duration_str)
    secs_match = re.search(r'(\d+)s', duration_str)
    
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    if mins_match:
        total_minutes += int(mins_match.group(1))
    if secs_match:
        total_minutes += int(secs_match.group(1)) / 60
    
    return total_minutes

def parse_cost(cost_str):
    """Parse cost string to float"""
    if not cost_str or cost_str == '-':
        return 0.0
    
    # Remove $ and convert to float
    return float(cost_str.replace('$', '').strip())

def calculate_hourly_cost(log_file):
    """Calculate hourly cost from the spot report log"""
    total_cost = 0.0
    total_duration_minutes = 0
    current_running_cost = 0.0
    active_instances = 0
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    data_lines = [line for line in lines if re.search(r'compute-text-vectors-\d+', line)]
    
    up_instances = []
    
    for line in data_lines:
        # Skip lines with non-relevant status
        if 'TERMINATED' in line:
            continue
        
        # Extract fields
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) < 7:
            continue
        
        name = parts[0]
        duration_str = parts[2]
        status = parts[4]
        cost_per_hour_str = parts[5]
        current_cost_str = parts[6]
        
        # Only consider UP instances for current hourly cost
        if status == 'UP':
            cost_per_hour = parse_cost(cost_per_hour_str)
            current_running_cost += cost_per_hour
            active_instances += 1
            up_instances.append((name, cost_per_hour))
        
        # Calculate total cost and duration for all instances
        duration_minutes = parse_duration(duration_str)
        current_cost = parse_cost(current_cost_str)
        
        total_cost += current_cost
        total_duration_minutes += duration_minutes
    
    # Calculate hourly cost metrics
    hours_run = total_duration_minutes / 60
    average_hourly_cost = total_cost / hours_run if hours_run > 0 else 0
    
    # Print results
    print(f"Total instances analyzed: {len(data_lines)}")
    print(f"Active (UP) instances: {active_instances}")
    print(f"Current hourly burn rate: ${current_running_cost:.2f}/hour")
    print(f"Total cost accrued: ${total_cost:.2f}")
    print(f"Total run time: {hours_run:.2f} hours")
    print(f"Average cost per hour: ${average_hourly_cost:.2f}/hour")
    
    print("\nTop 5 most expensive running instances:")
    for name, cost in sorted(up_instances, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: ${cost:.2f}/hour")

if __name__ == "__main__":
    log_file = "spot_report.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    calculate_hourly_cost(log_file) 