# 🎮 Interactive Training System

Run MEV arbitrage training with real-time visual feedback through Claude Code commands!

## 🚀 Quick Start

### 1. Web Dashboard (Recommended)
```bash
python train_interactive.py --mode dashboard
```
Opens a beautiful web dashboard at http://localhost:8000 with:
- Real-time fitness evolution graphs
- Live profit tracking
- Agent population tables
- Strategy distribution charts
- Event log with color coding

### 2. Terminal UI (No Browser)
```bash
python train_interactive.py --mode terminal
```
Rich terminal interface with:
- ASCII charts
- Colored tables
- Progress indicators
- Keyboard controls

### 3. Quick Training with Popup Results
```bash
python train_interactive.py --mode quick --generations 20 --show-results
```
Runs training and shows matplotlib popup with:
- Fitness evolution plot
- Top agent profits
- Strategy distribution pie chart
- Summary statistics

## 📊 Dashboard Features

### Real-Time Monitoring
- **Generation Progress**: Watch fitness evolve live
- **Profit Accumulation**: Track total profits as agents trade
- **Agent Population**: See all agents sorted by performance
- **Strategy Distribution**: Pie chart of different strategies
- **Event Log**: Color-coded feed of all training events

### Interactive Controls
- **Pause/Resume**: Control training flow
- **Save Checkpoint**: Save current state anytime
- **Export Results**: Download training data as JSON

### WebSocket Updates
- Updates push automatically to browser
- No refresh needed
- Multiple clients can connect

## 🖥️ Terminal UI Features

### Rich Console Display
```
🌊 Sakana MEV Training System | Status: RUNNING | Elapsed: 5m 23s
┌─────────────────────────────────────────────────────────┐
│                  Training Statistics                     │
├─────────────────────────────────────────────────────────┤
│ Generation       23                                      │
│ Best Fitness     0.8734                                  │
│ Avg Fitness      0.5612                                  │
│ Total Profit     $1,247.83                              │
│ Opportunities    156                                     │
└─────────────────────────────────────────────────────────┘
```

### Keyboard Controls
- `P` - Pause/Resume training
- `S` - Save checkpoint
- `E` - Export results
- `Q` - Quit training

## 🎯 Command Options

### Basic Usage
```bash
python train_interactive.py [OPTIONS]
```

### Options
- `--mode {dashboard,terminal,quick}` - UI mode (default: dashboard)
- `--generations N` - Number of generations to train (default: 50)
- `--population N` - Population size (default: 10)
- `--min-profit N` - Minimum profit threshold (default: 50)
- `--port N` - Dashboard port (default: 8000)
- `--show-results` - Show popup results in quick mode

### Examples

**Long training with dashboard:**
```bash
python train_interactive.py --mode dashboard --generations 100 --population 20
```

**Quick test with terminal UI:**
```bash
python train_interactive.py --mode terminal --generations 10
```

**Fast results with popup:**
```bash
python train_interactive.py --mode quick --generations 20 --show-results
```

## 📈 What You'll See

### Dashboard Mode
1. Browser opens automatically
2. Real-time charts update as training progresses
3. Agent table shows current population
4. Event log shows opportunities and trades
5. Stats update every second

### Terminal Mode
1. Full-screen terminal UI
2. ASCII charts for fitness/profit
3. Colored agent table
4. Scrolling event log
5. Keyboard shortcuts for control

### Quick Mode
1. Progress logs in console
2. Matplotlib window pops up at end
3. 4-panel summary of results
4. Can save plot as image

## 🔧 Advanced Features

### Custom Configuration
Create `training_config.json`:
```json
{
    "population_size": 20,
    "min_profit": 75,
    "mutation_rate": 0.1,
    "elite_count": 3
}
```

### Resuming Training
```bash
# Save checkpoint during training (press S)
# Resume from checkpoint:
python train_interactive.py --resume checkpoint_20240115_143022.json
```

### Multiple Dashboards
Run multiple training sessions:
```bash
# Session 1
python train_interactive.py --port 8000

# Session 2
python train_interactive.py --port 8001
```

## 🎨 Customization

### Dashboard Themes
Edit CSS in `training_dashboard.py`:
- Dark theme (default)
- Light theme
- Custom colors

### Terminal Colors
Configure Rich themes:
```python
console = Console(theme=custom_theme)
```

### Chart Settings
Modify chart update frequency, colors, and styles in the respective UI files.

## 🐛 Troubleshooting

### Dashboard won't open
- Check if port 8000 is free
- Try different port: `--port 8080`
- Check firewall settings

### Terminal UI looks broken
- Ensure terminal supports Unicode
- Try different terminal emulator
- Check Rich is installed: `pip install rich`

### Popup doesn't show
- Install matplotlib: `pip install matplotlib`
- Check display settings
- Use `--mode dashboard` instead

## 💡 Tips

1. **Start with Terminal Mode** for quick tests
2. **Use Dashboard Mode** for serious training sessions
3. **Quick Mode** is great for parameter testing
4. **Save checkpoints** regularly (press S)
5. **Export results** before closing (press E)

## 🚦 Performance

- Dashboard updates: 2 FPS (configurable)
- Terminal updates: 2 FPS
- WebSocket latency: <10ms
- Memory usage: ~100MB for 100 agents

## 🎯 Next Steps

After training:
1. Review exported results
2. Select best strategies
3. Test with `quick_start_profit.py`
4. Deploy winning agents

Happy Training! 🚀