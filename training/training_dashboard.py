#!/usr/bin/env python3
"""
Web-based Training Dashboard
Real-time visualization of MEV arbitrage training progress
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import webbrowser
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Sakana MEV Training Dashboard")

# Store for active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.training_data = {
            'generations': [],
            'agents': [],
            'opportunities': [],
            'profits': [],
            'events': []
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current state to new connection
        await websocket.send_json({
            'type': 'initial_state',
            'data': self.training_data
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Connection might be closed
                pass

    async def update_training_data(self, update_type: str, data: Any):
        """Update training data and broadcast"""
        if update_type == 'generation':
            self.training_data['generations'].append(data)
        elif update_type == 'agents':
            self.training_data['agents'] = data
        elif update_type == 'opportunity':
            self.training_data['opportunities'].append(data)
        elif update_type == 'profit':
            self.training_data['profits'].append(data)
        elif update_type == 'event':
            self.training_data['events'].append(data)
        
        # Broadcast update
        await self.broadcast({
            'type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

manager = ConnectionManager()

# HTML Dashboard
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sakana MEV Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .panel h3 {
            margin-top: 0;
            color: #4CAF50;
        }
        .chart {
            height: 300px;
        }
        .agents-table {
            width: 100%;
            border-collapse: collapse;
        }
        .agents-table th, .agents-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        .agents-table th {
            background-color: #333;
            color: #4CAF50;
        }
        .event-log {
            height: 200px;
            overflow-y: auto;
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }
        .event-log .event {
            margin-bottom: 5px;
        }
        .event-log .profit { color: #4CAF50; }
        .event-log .loss { color: #f44336; }
        .event-log .info { color: #2196F3; }
        .stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
        .stat-box {
            text-align: center;
            background-color: #333;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #666;
            cursor: not-allowed;
        }
        #connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
        }
        .connected {
            background-color: #4CAF50;
        }
        .disconnected {
            background-color: #f44336;
        }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">Disconnected</div>
    
    <div class="container">
        <div class="header">
            <h1>🌊 Sakana MEV Training Dashboard</h1>
            <p>Real-time Evolution Progress</p>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="pauseTraining()">⏸️ Pause</button>
            <button class="btn" onclick="resumeTraining()">▶️ Resume</button>
            <button class="btn" onclick="saveCheckpoint()">💾 Save Checkpoint</button>
            <button class="btn" onclick="exportResults()">📊 Export Results</button>
        </div>
        
        <div class="stats" id="stats">
            <div class="stat-box">
                <div class="stat-value" id="generation-count">0</div>
                <div class="stat-label">Generation</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="total-profit">$0</div>
                <div class="stat-label">Total Profit</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="best-fitness">0.00</div>
                <div class="stat-label">Best Fitness</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="opportunities-found">0</div>
                <div class="stat-label">Opportunities</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h3>Fitness Evolution</h3>
                <div id="fitness-chart" class="chart"></div>
            </div>
            <div class="panel">
                <h3>Profit Accumulation</h3>
                <div id="profit-chart" class="chart"></div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h3>Current Population</h3>
                <div style="overflow-x: auto;">
                    <table class="agents-table" id="agents-table">
                        <thead>
                            <tr>
                                <th>Agent ID</th>
                                <th>Fitness</th>
                                <th>Real Profit</th>
                                <th>Success Rate</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody id="agents-tbody">
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="panel">
                <h3>Strategy Distribution</h3>
                <div id="strategy-chart" class="chart"></div>
            </div>
        </div>
        
        <div class="panel">
            <h3>Event Log</h3>
            <div class="event-log" id="event-log">
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let trainingData = {
            generations: [],
            agents: [],
            opportunities: [],
            profits: [],
            events: []
        };
        
        // Connect to WebSocket
        function connect() {
            ws = new WebSocket("ws://localhost:8000/ws");
            
            ws.onopen = function() {
                document.getElementById('connection-status').className = 'connected';
                document.getElementById('connection-status').textContent = 'Connected';
                addEvent('Connected to training server', 'info');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').className = 'disconnected';
                document.getElementById('connection-status').textContent = 'Disconnected';
                addEvent('Disconnected from server', 'info');
                // Reconnect after 2 seconds
                setTimeout(connect, 2000);
            };
        }
        
        function handleMessage(message) {
            switch(message.type) {
                case 'initial_state':
                    trainingData = message.data;
                    updateAllCharts();
                    break;
                case 'generation':
                    trainingData.generations.push(message.data);
                    updateFitnessChart();
                    updateStats();
                    addEvent(`Generation ${message.data.generation} completed`, 'info');
                    break;
                case 'agents':
                    trainingData.agents = message.data;
                    updateAgentsTable();
                    updateStrategyChart();
                    break;
                case 'opportunity':
                    trainingData.opportunities.push(message.data);
                    updateStats();
                    addEvent(`Found opportunity: $${message.data.net_profit.toFixed(2)}`, 'profit');
                    break;
                case 'profit':
                    trainingData.profits.push(message.data);
                    updateProfitChart();
                    updateStats();
                    break;
                case 'event':
                    addEvent(message.data.message, message.data.type);
                    break;
            }
        }
        
        function updateFitnessChart() {
            const generations = trainingData.generations;
            if (generations.length === 0) return;
            
            const data = [{
                x: generations.map(g => g.generation),
                y: generations.map(g => g.best_fitness),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Best Fitness',
                line: { color: '#4CAF50' }
            }, {
                x: generations.map(g => g.generation),
                y: generations.map(g => g.avg_fitness),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Average Fitness',
                line: { color: '#2196F3' }
            }];
            
            const layout = {
                title: '',
                xaxis: { title: 'Generation', color: '#888' },
                yaxis: { title: 'Fitness', color: '#888' },
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: { color: '#888' }
            };
            
            Plotly.newPlot('fitness-chart', data, layout);
        }
        
        function updateProfitChart() {
            const profits = trainingData.profits;
            if (profits.length === 0) return;
            
            let cumulative = 0;
            const cumulativeProfits = profits.map(p => {
                cumulative += p.amount;
                return cumulative;
            });
            
            const data = [{
                x: profits.map((p, i) => i),
                y: cumulativeProfits,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: '#4CAF50' }
            }];
            
            const layout = {
                title: '',
                xaxis: { title: 'Trade #', color: '#888' },
                yaxis: { title: 'Cumulative Profit ($)', color: '#888' },
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: { color: '#888' }
            };
            
            Plotly.newPlot('profit-chart', data, layout);
        }
        
        function updateAgentsTable() {
            const tbody = document.getElementById('agents-tbody');
            tbody.innerHTML = '';
            
            trainingData.agents.sort((a, b) => b.fitness - a.fitness).forEach(agent => {
                const row = tbody.insertRow();
                row.insertCell(0).textContent = agent.id;
                row.insertCell(1).textContent = agent.fitness.toFixed(4);
                row.insertCell(2).textContent = `$${agent.real_profit.toFixed(2)}`;
                row.insertCell(3).textContent = `${(agent.success_rate * 100).toFixed(1)}%`;
                row.insertCell(4).textContent = `${agent.strategy.min_profit}/${agent.strategy.gas_mult}x`;
            });
        }
        
        function updateStrategyChart() {
            if (trainingData.agents.length === 0) return;
            
            // Group agents by strategy type
            const strategies = {};
            trainingData.agents.forEach(agent => {
                const key = `${agent.strategy.min_profit}_${agent.strategy.gas_mult}`;
                strategies[key] = (strategies[key] || 0) + 1;
            });
            
            const data = [{
                labels: Object.keys(strategies),
                values: Object.values(strategies),
                type: 'pie',
                marker: {
                    colors: ['#4CAF50', '#2196F3', '#FF9800', '#f44336', '#9C27B0']
                }
            }];
            
            const layout = {
                title: '',
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: { color: '#888' }
            };
            
            Plotly.newPlot('strategy-chart', data, layout);
        }
        
        function updateStats() {
            // Update generation count
            if (trainingData.generations.length > 0) {
                const lastGen = trainingData.generations[trainingData.generations.length - 1];
                document.getElementById('generation-count').textContent = lastGen.generation;
                document.getElementById('best-fitness').textContent = lastGen.best_fitness.toFixed(4);
            }
            
            // Update total profit
            const totalProfit = trainingData.profits.reduce((sum, p) => sum + p.amount, 0);
            document.getElementById('total-profit').textContent = `$${totalProfit.toFixed(2)}`;
            
            // Update opportunities
            document.getElementById('opportunities-found').textContent = trainingData.opportunities.length;
        }
        
        function updateAllCharts() {
            updateFitnessChart();
            updateProfitChart();
            updateAgentsTable();
            updateStrategyChart();
            updateStats();
        }
        
        function addEvent(message, type) {
            const eventLog = document.getElementById('event-log');
            const event = document.createElement('div');
            event.className = `event ${type}`;
            const timestamp = new Date().toLocaleTimeString();
            event.textContent = `[${timestamp}] ${message}`;
            eventLog.appendChild(event);
            eventLog.scrollTop = eventLog.scrollHeight;
        }
        
        // Control functions
        function pauseTraining() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'pause' }));
                addEvent('Training paused', 'info');
            }
        }
        
        function resumeTraining() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'resume' }));
                addEvent('Training resumed', 'info');
            }
        }
        
        function saveCheckpoint() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'save_checkpoint' }));
                addEvent('Checkpoint saved', 'info');
            }
        }
        
        function exportResults() {
            const dataStr = JSON.stringify(trainingData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `training_results_${new Date().toISOString()}.json`;
            link.click();
            addEvent('Results exported', 'info');
        }
        
        // Connect on load
        connect();
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(dashboard_html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive messages from client (control commands)
            data = await websocket.receive_json()
            
            if data.get('action') == 'pause':
                # Handle pause command
                await manager.broadcast({
                    'type': 'control',
                    'action': 'pause'
                })
            elif data.get('action') == 'resume':
                # Handle resume command
                await manager.broadcast({
                    'type': 'control',
                    'action': 'resume'
                })
            elif data.get('action') == 'save_checkpoint':
                # Handle save checkpoint
                await manager.broadcast({
                    'type': 'control',
                    'action': 'save_checkpoint'
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class TrainingDashboard:
    """Interface for training scripts to send updates to dashboard"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.server = None
        self.loop = None
        
    async def start_server(self):
        """Start the dashboard server"""
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    def start(self):
        """Start dashboard in background thread"""
        import threading
        
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.start_server())
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        # Open browser
        import time
        time.sleep(2)  # Wait for server to start
        webbrowser.open(f'http://localhost:{self.port}')
        
        logger.info(f"Dashboard started at http://localhost:{self.port}")
    
    async def update_generation(self, generation: int, best_fitness: float, avg_fitness: float):
        """Send generation update"""
        await manager.update_training_data('generation', {
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness
        })
    
    async def update_agents(self, agents: List[Dict]):
        """Send agent population update"""
        agent_data = []
        for agent in agents:
            agent_data.append({
                'id': agent.get('id'),
                'fitness': agent.get('fitness', 0),
                'real_profit': agent.get('real_profit', 0),
                'success_rate': agent.get('success_rate', 0),
                'strategy': {
                    'min_profit': agent.get('genome', {}).get('min_profit_usd', 0),
                    'gas_mult': agent.get('genome', {}).get('gas_multiplier', 1.5)
                }
            })
        await manager.update_training_data('agents', agent_data)
    
    async def add_opportunity(self, opportunity: Dict):
        """Add found opportunity"""
        await manager.update_training_data('opportunity', {
            'net_profit': opportunity.get('net_profit_usd', 0),
            'path': opportunity.get('path', []),
            'success_rate': opportunity.get('success_probability', 0)
        })
    
    async def add_profit(self, amount: float, success: bool):
        """Add profit/loss event"""
        await manager.update_training_data('profit', {
            'amount': amount,
            'success': success
        })
    
    async def log_event(self, message: str, event_type: str = 'info'):
        """Log an event"""
        await manager.update_training_data('event', {
            'message': message,
            'type': event_type
        })


if __name__ == "__main__":
    # Test the dashboard
    dashboard = TrainingDashboard()
    dashboard.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Dashboard stopped")