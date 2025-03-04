"""
Monitoring service for CLIP vector computation workers.
Aggregates metrics from all workers and serves a dashboard.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, DefaultDict

import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

class MonitoringService:
    def __init__(self, metrics_dir: str, history_window: int = 3600):
        self.metrics_dir = Path(metrics_dir)
        self.last_update = {}
        self.worker_metrics = {}
        self.history_window = history_window  # Keep 1 hour of history by default
        
        # Track historical throughput data
        self.throughput_history: DefaultDict[str, List[Dict]] = defaultdict(list)
        self.last_processed_count: Dict[str, int] = {}
        
        # Worker history tracking
        self.worker_history: DefaultDict[str, List[Dict]] = defaultdict(list)
        self.worker_sessions: DefaultDict[str, List[str]] = defaultdict(list)
        
        self.aggregate_metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'total_workers': 0,
            'active_workers': 0,
            'completed_workers': 0,
            'failed_workers': 0,
            'overall_progress': 0.0,
            'overall_speed': 0.0,
            'estimated_time_remaining': None,
            'total_restarts': 0
        }

    def update_throughput_history(self, worker_id: str, metrics: Dict):
        """Update historical throughput data for a worker."""
        current_time = time.time()
        current_count = metrics['processed_count']
        
        # Calculate throughput since last update
        if worker_id in self.last_processed_count:
            time_diff = current_time - self.throughput_history[worker_id][-1]['timestamp']
            count_diff = current_count - self.last_processed_count[worker_id]
            if time_diff > 0:  # Avoid division by zero
                throughput = count_diff / time_diff
                
                # Add new data point
                self.throughput_history[worker_id].append({
                    'timestamp': current_time,
                    'throughput': throughput,
                    'session_id': metrics.get('session_id', 'unknown')
                })
                
                # Remove old data points outside the history window
                cutoff_time = current_time - self.history_window
                self.throughput_history[worker_id] = [
                    point for point in self.throughput_history[worker_id]
                    if point['timestamp'] > cutoff_time
                ]
        else:
            # First data point for this worker
            self.throughput_history[worker_id].append({
                'timestamp': current_time,
                'throughput': metrics.get('images_per_second', 0),
                'session_id': metrics.get('session_id', 'unknown')
            })
            
        self.last_processed_count[worker_id] = current_count

    async def read_worker_history(self, worker_id: str):
        """Read the complete history for a worker."""
        history_file = self.metrics_dir / f'worker_{worker_id}_history.json'
        try:
            if history_file.exists():
                async with aiofiles.open(history_file, 'r') as f:
                    content = await f.read()
                    history = json.loads(content)
                    
                    # Update worker history
                    self.worker_history[worker_id] = history
                    
                    # Extract unique session IDs
                    sessions = set()
                    for entry in history:
                        if 'session_id' in entry:
                            sessions.add(entry['session_id'])
                    
                    self.worker_sessions[worker_id] = sorted(list(sessions))
                    
                    return history
            return []
        except Exception as e:
            print(f"Error reading history for worker {worker_id}: {e}")
            return []

    async def update_metrics(self):
        """Read and aggregate metrics from all worker files."""
        try:
            # Read all worker metric files
            worker_files = list(self.metrics_dir.glob('worker_*.json'))
            
            # Filter out history files
            worker_files = [f for f in worker_files if '_history' not in f.name]
            
            new_metrics = {}
            total_restarts = 0
            
            for file in worker_files:
                try:
                    async with aiofiles.open(file, 'r') as f:
                        content = await f.read()
                        metrics = json.loads(content)
                        worker_id = metrics['worker_id']
                        new_metrics[worker_id] = metrics
                        
                        # Read worker history
                        await self.read_worker_history(worker_id)
                        
                        # Count number of sessions as restarts
                        total_restarts += max(0, len(self.worker_sessions[worker_id]) - 1)
                        
                        self.update_throughput_history(worker_id, metrics)
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
                total_failed += metrics.get('failed_count', 0)
                total_speed += metrics.get('images_per_second', 0)
                
                if metrics.get('status') == 'running':
                    active_workers += 1
                elif metrics.get('status') == 'completed':
                    completed_workers += 1
                elif metrics.get('status') == 'failed':
                    failed_workers += 1

                if metrics.get('end_idx'):
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
                'estimated_time_remaining': (total_items - total_progress) / total_speed if total_speed > 0 else None,
                'total_restarts': total_restarts
            })

        except Exception as e:
            print(f"Error updating metrics: {e}")

    def get_throughput_chart_data(self) -> Dict:
        """Prepare throughput history data for Chart.js."""
        # Get the earliest and latest timestamps across all workers
        all_timestamps = []
        for history in self.throughput_history.values():
            all_timestamps.extend(point['timestamp'] for point in history)
        
        if not all_timestamps:
            return {'labels': [], 'datasets': []}
            
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Generate dataset for each worker
        datasets = []
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        
        for i, (worker_id, history) in enumerate(self.throughput_history.items()):
            color = colors[i % len(colors)]
            
            # Group by session ID
            sessions = {}
            for point in history:
                session_id = point.get('session_id', 'unknown')
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(point)
            
            # Create a separate dataset for each session
            for j, (session_id, session_data) in enumerate(sessions.items()):
                # Modify color brightness for different sessions of the same worker
                session_color = self._adjust_color_brightness(color, j * 15)
                
                datasets.append({
                    'label': f"{worker_id} (session {j+1})",
                    'data': [{'x': point['timestamp'] * 1000, 'y': point['throughput']}  # Convert to milliseconds for JS
                             for point in session_data],
                    'borderColor': session_color,
                    'backgroundColor': session_color + '20',
                    'fill': False,
                    'pointRadius': 3
                })
        
        return {
            'datasets': datasets
        }
    
    def _adjust_color_brightness(self, hex_color, percent):
        """Adjust the brightness of a hex color."""
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Increase brightness
        r = min(255, r + percent)
        g = min(255, g + percent)
        b = min(255, b + percent)
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def get_session_history_data(self) -> Dict:
        """Prepare session history data for visualization."""
        session_data = []
        
        for worker_id, history in self.worker_history.items():
            # Group events by session
            sessions = {}
            for event in history:
                session_id = event.get('session_id', 'unknown')
                if session_id not in sessions:
                    sessions[session_id] = {
                        'worker_id': worker_id,
                        'session_id': session_id,
                        'start_time': None,
                        'end_time': None,
                        'duration': 0,
                        'processed': 0,
                        'failed': 0,
                        'status': 'unknown',
                        'termination_reason': None
                    }
                
                # Update session data
                timestamp = event.get('timestamp', 0)
                if event.get('event') == 'start' or (not sessions[session_id]['start_time'] and event.get('timestamp')):
                    sessions[session_id]['start_time'] = timestamp
                
                if event.get('status') in ['completed', 'failed']:
                    sessions[session_id]['end_time'] = timestamp
                    sessions[session_id]['status'] = event.get('status')
                
                # Track spot VM termination events
                if event.get('event') == 'termination' or event.get('status') == 'terminated':
                    sessions[session_id]['end_time'] = timestamp
                    sessions[session_id]['status'] = 'terminated'
                    sessions[session_id]['termination_reason'] = 'Spot VM interruption'
                
                if 'processed_count' in event:
                    sessions[session_id]['processed'] = max(sessions[session_id]['processed'], event['processed_count'])
                
                if 'failed_count' in event:
                    sessions[session_id]['failed'] = max(sessions[session_id]['failed'], event['failed_count'])
            
            # Calculate duration and add to data
            for session in sessions.values():
                if session['start_time']:
                    if session['end_time']:
                        session['duration'] = session['end_time'] - session['start_time']
                    else:
                        # Session might still be running
                        session['duration'] = time.time() - session['start_time']
                        session['status'] = 'running'
                
                session_data.append(session)
        
        return sorted(session_data, key=lambda x: x.get('start_time', 0))

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
            progress = ((worker.get('current_idx', 0) - worker.get('start_idx', 0)) / 
                       (worker.get('end_idx', 1) - worker.get('start_idx', 0)) * 100
                       if worker.get('end_idx') else 0)
            
            # Count sessions/restarts for this worker
            session_count = len(self.worker_sessions.get(worker_id, []))
            restart_count = max(0, session_count - 1)
            
            # Determine status class for styling
            status_class = ''
            if worker.get('status') == 'running':
                status_class = 'status-running'
            elif worker.get('status') == 'completed':
                status_class = 'status-completed'
            elif worker.get('status') == 'failed' or worker.get('status') == 'terminated':
                status_class = 'status-failed'
            
            row = f"""
            <tr>
                <td>{worker_id}</td>
                <td class="{status_class}">{worker.get('status', 'unknown')}</td>
                <td>{progress:.2f}%</td>
                <td>{worker.get('processed_count', 0)}</td>
                <td>{worker.get('failed_count', 0)}</td>
                <td>{worker.get('images_per_second', 0):.2f}</td>
                <td>{restart_count}</td>
                <td>{datetime.fromtimestamp(worker.get('last_update', 0)).strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
            """
            worker_rows.append(row)

        # Generate session history table
        session_data = self.get_session_history_data()
        session_rows = []
        for session in session_data:
            start_time = datetime.fromtimestamp(session.get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S') if session.get('start_time') else 'N/A'
            end_time = datetime.fromtimestamp(session.get('end_time', 0)).strftime('%Y-%m-%d %H:%M:%S') if session.get('end_time') else 'N/A'
            duration_mins = session.get('duration', 0) / 60
            
            # Determine status class for styling
            status_class = ''
            if session.get('status') == 'running':
                status_class = 'status-running'
            elif session.get('status') == 'completed':
                status_class = 'status-completed'
            elif session.get('status') in ['failed', 'terminated']:
                status_class = 'status-failed'
            
            # Show termination reason for spot VM interruptions
            status_text = session.get('status', 'unknown')
            if session.get('termination_reason'):
                status_text += f" ({session.get('termination_reason')})"
            
            row = f"""
            <tr>
                <td>{session.get('worker_id', 'unknown')}</td>
                <td>{session.get('session_id', 'unknown')[-8:]}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{start_time}</td>
                <td>{end_time if session.get('end_time') else 'Running'}</td>
                <td>{duration_mins:.1f} min</td>
                <td>{session.get('processed', 0)}</td>
                <td>{session.get('failed', 0)}</td>
            </tr>
            """
            session_rows.append(row)

        # Get chart data
        chart_data = json.dumps(self.get_throughput_chart_data())
        
        # Get termination events for chart annotations
        termination_events = []
        for worker_id, history in self.worker_history.items():
            for event in history:
                if event.get('event') == 'termination' or event.get('status') == 'terminated':
                    termination_events.append({
                        'worker_id': worker_id,
                        'session_id': event.get('session_id', 'unknown'),
                        'timestamp': event.get('timestamp', 0) * 1000  # Convert to milliseconds for JS
                    })
        
        termination_events_json = json.dumps(termination_events)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CLIP Vector Computation Progress</title>
            <meta http-equiv="refresh" content="{refresh_rate}">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/moment@2.29.1"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment@1.0.0"></script>
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
                .chart-container {{
                    width: 100%;
                    height: 400px;
                    margin: 30px 0;
                }}
                .status-running {{ color: green; }}
                .status-completed {{ color: blue; }}
                .status-failed {{ color: red; }}
                .tabs {{
                    margin-top: 30px;
                    border-bottom: 1px solid #ccc;
                    display: flex;
                }}
                .tab {{
                    padding: 10px 15px;
                    cursor: pointer;
                    background: #f5f5f5;
                    margin-right: 5px;
                    border-radius: 5px 5px 0 0;
                }}
                .tab.active {{
                    background: #ddd;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px 0;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .event-marker {{
                    position: absolute;
                    width: 2px;
                    background-color: red;
                    opacity: 0.7;
                }}
                .event-marker::after {{
                    content: "VM terminated";
                    position: absolute;
                    top: 0;
                    left: 5px;
                    background: rgba(255,0,0,0.8);
                    color: white;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-size: 10px;
                    white-space: nowrap;
                }}
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
                    <div class="metric-label">VM Restarts</div>
                    <div class="metric-value">{metrics['total_restarts']}</div>
                </div>
            </div>

            <h2>Throughput History</h2>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
                <div id="terminationMarkers"></div>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('currentStatus')">Current Status</div>
                <div class="tab" onclick="showTab('sessionHistory')">Session History</div>
            </div>
            
            <div id="currentStatus" class="tab-content active">
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
                            <th>Restarts</th>
                            <th>Last Update</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(worker_rows)}
                    </tbody>
                </table>
            </div>
            
            <div id="sessionHistory" class="tab-content">
                <h2>Session History</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Worker ID</th>
                            <th>Session ID</th>
                            <th>Status</th>
                            <th>Start Time</th>
                            <th>End Time</th>
                            <th>Duration</th>
                            <th>Processed</th>
                            <th>Failed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(session_rows)}
                    </tbody>
                </table>
            </div>

            <script>
                // Chart setup
                const ctx = document.getElementById('throughputChart');
                const chartData = {chart_data};
                const terminationEvents = {termination_events_json};
                
                const chart = new Chart(ctx, {{
                    type: 'line',
                    data: chartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{
                                type: 'time',
                                time: {{
                                    unit: 'minute'
                                }},
                                title: {{
                                    display: true,
                                    text: 'Time'
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Images/second'
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Worker Throughput Over Time (Gaps indicate VM restarts)'
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return `${{context.dataset.label}}: ${{context.parsed.y.toFixed(2)}} img/s`;
                                    }}
                                }}
                            }},
                            annotation: {{
                                annotations: terminationEvents.map(event => ({{
                                    type: 'line',
                                    xMin: event.timestamp,
                                    xMax: event.timestamp,
                                    borderColor: 'red',
                                    borderWidth: 2,
                                    label: {{
                                        display: true,
                                        content: 'VM terminated',
                                        position: 'top'
                                    }}
                                }}))
                            }}
                        }}
                    }}
                }});
                
                // Add visual markers for termination events
                function addTerminationMarkers() {{
                    const container = document.getElementById('terminationMarkers');
                    container.innerHTML = '';
                    
                    // Wait for chart to render
                    setTimeout(() => {{
                        const chartRect = ctx.getBoundingClientRect();
                        
                        terminationEvents.forEach(event => {{
                            // Convert timestamp to x position on chart
                            const xScale = chart.scales.x;
                            if (!xScale) return;
                            
                            const xPos = xScale.getPixelForValue(event.timestamp);
                            if (isNaN(xPos)) return;
                            
                            const marker = document.createElement('div');
                            marker.className = 'event-marker';
                            marker.style.height = `${{chartRect.height}}px`;
                            marker.style.left = `${{xPos}}px`;
                            marker.title = `Worker ${{event.worker_id}} terminated at ${{new Date(event.timestamp).toLocaleString()}}`;
                            
                            container.appendChild(marker);
                        }});
                    }}, 500);
                }}
                
                // Execute after chart is rendered
                chart.options.animation.onComplete = addTerminationMarkers;
                
                function showTab(tabId) {{
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(content => {{
                        content.classList.remove('active');
                    }});
                    
                    // Deactivate all tabs
                    document.querySelectorAll('.tab').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    
                    // Show the selected tab content
                    document.getElementById(tabId).classList.add('active');
                    
                    // Activate the clicked tab
                    event.currentTarget.classList.add('active');
                }}
            </script>
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
    main() 