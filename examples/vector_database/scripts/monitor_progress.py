"""
Monitoring service for CLIP vector computation workers.
Aggregates metrics from all workers and serves a dashboard.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List

import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

class MonitoringService:
    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
        self.last_update = {}
        self.worker_metrics = {}
        self.aggregate_metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'total_workers': 0,
            'active_workers': 0,
            'completed_workers': 0,
            'failed_workers': 0,
            'overall_progress': 0.0,
            'overall_speed': 0.0,
            'estimated_time_remaining': None
        }

    async def update_metrics(self):
        """Read and aggregate metrics from all worker files."""
        try:
            # Read all worker metric files
            worker_files = list(self.metrics_dir.glob('worker_*.json'))
            
            new_metrics = {}
            for file in worker_files:
                try:
                    async with aiofiles.open(file, 'r') as f:
                        content = await f.read()
                        metrics = json.loads(content)
                        worker_id = metrics['worker_id']
                        new_metrics[worker_id] = metrics
                except Exception as e:
                    print(f"Error reading metrics from {file}: {e}")
                    continue

            # Update worker metrics
            self.worker_metrics = new_metrics

            # Calculate aggregate metrics
            total_processed = 0
            total_failed = 0
            total_speed = 0
            active_workers = 0
            completed_workers = 0
            failed_workers = 0
            total_progress = 0
            total_items = 0

            for metrics in self.worker_metrics.values():
                total_processed += metrics['processed_count']
                total_failed += metrics['failed_count']
                total_speed += metrics.get('images_per_second', 0)
                
                if metrics['status'] == 'running':
                    active_workers += 1
                elif metrics['status'] == 'completed':
                    completed_workers += 1
                elif metrics['status'] == 'failed':
                    failed_workers += 1

                if metrics['end_idx']:
                    total_items += metrics['end_idx'] - metrics['start_idx']
                    total_progress += metrics['processed_count']

            # Update aggregate metrics
            self.aggregate_metrics.update({
                'total_processed': total_processed,
                'total_failed': total_failed,
                'total_workers': len(self.worker_metrics),
                'active_workers': active_workers,
                'completed_workers': completed_workers,
                'failed_workers': failed_workers,
                'overall_progress': (total_progress / total_items * 100) if total_items > 0 else 0,
                'overall_speed': total_speed,
                'estimated_time_remaining': (total_items - total_progress) / total_speed if total_speed > 0 else None
            })

        except Exception as e:
            print(f"Error updating metrics: {e}")

    def get_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        refresh_rate = 5  # seconds
        
        # Convert metrics to human-readable format
        metrics = self.aggregate_metrics.copy()
        metrics['overall_progress'] = f"{metrics['overall_progress']:.2f}%"
        metrics['overall_speed'] = f"{metrics['overall_speed']:.2f} images/sec"
        if metrics['estimated_time_remaining']:
            hours = metrics['estimated_time_remaining'] / 3600
            metrics['estimated_time_remaining'] = f"{hours:.1f} hours"
        else:
            metrics['estimated_time_remaining'] = "N/A"

        # Generate worker status table
        worker_rows = []
        for worker_id, worker in self.worker_metrics.items():
            progress = ((worker['current_idx'] - worker['start_idx']) / 
                       (worker['end_idx'] - worker['start_idx']) * 100
                       if worker['end_idx'] else 0)
            
            row = f"""
            <tr>
                <td>{worker_id}</td>
                <td>{worker['status']}</td>
                <td>{progress:.2f}%</td>
                <td>{worker['processed_count']}</td>
                <td>{worker['failed_count']}</td>
                <td>{worker.get('images_per_second', 0):.2f}</td>
                <td>{datetime.fromtimestamp(worker['last_update']).strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
            """
            worker_rows.append(row)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CLIP Vector Computation Progress</title>
            <meta http-equiv="refresh" content="{refresh_rate}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ color: #666; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>CLIP Vector Computation Progress</h1>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Overall Progress</div>
                    <div class="metric-value">{metrics['overall_progress']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Processing Speed</div>
                    <div class="metric-value">{metrics['overall_speed']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Estimated Time Remaining</div>
                    <div class="metric-value">{metrics['estimated_time_remaining']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Processed</div>
                    <div class="metric-value">{metrics['total_processed']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Failed Images</div>
                    <div class="metric-value">{metrics['total_failed']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Active Workers</div>
                    <div class="metric-value">{metrics['active_workers']} / {metrics['total_workers']}</div>
                </div>
            </div>

            <h2>Worker Status</h2>
            <table>
                <thead>
                    <tr>
                        <th>Worker ID</th>
                        <th>Status</th>
                        <th>Progress</th>
                        <th>Processed</th>
                        <th>Failed</th>
                        <th>Speed (img/s)</th>
                        <th>Last Update</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(worker_rows)}
                </tbody>
            </table>
        </body>
        </html>
        """

monitoring_service = None

@app.on_event("startup")
async def startup_event():
    global monitoring_service
    metrics_dir = "/output/metrics"  # This should match the directory in compute_vectors.py
    monitoring_service = MonitoringService(metrics_dir)
    
    # Start background task to update metrics
    asyncio.create_task(periodic_metrics_update())

async def periodic_metrics_update():
    while True:
        await monitoring_service.update_metrics()
        await asyncio.sleep(5)  # Update every 5 seconds

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Monitoring service not initialized")
    return monitoring_service.get_dashboard_html()

@app.get("/api/metrics")
async def get_metrics():
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Monitoring service not initialized")
    return {
        "aggregate_metrics": monitoring_service.aggregate_metrics,
        "worker_metrics": monitoring_service.worker_metrics
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 