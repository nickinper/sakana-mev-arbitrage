# Sakana-Inspired MEV Arbitrage System

A nature-inspired evolutionary system for discovering and executing MEV arbitrage strategies, based on Sakana AI's evolutionary approaches.

## 🚀 Quick Start: Generate Profits TODAY

**Want to start making money immediately?** Check out our [**Profit Now Guide**](README_PROFIT_NOW.md) to begin executing profitable trades within 30 minutes!

```bash
# Start making money in one command:
python quick_start_profit.py
```

No complex setup. No huge capital requirements. Just find opportunities, execute, and profit.

## Overview

This system uses evolutionary algorithms to discover profitable MEV arbitrage strategies without manual strategy design. The population of agents evolves through crossover and mutation, with natural selection based on actual profit performance.

## Key Features

- **Evolutionary Multi-Agent Architecture**: Population of specialized agents that evolve over generations
- **Collective Intelligence**: Multiple specialists vote on opportunities with reputation-weighted decisions
- **No Manual Strategy Design**: The system discovers profitable strategies through evolution
- **Real-Time Adaptation**: Continuously evolves with market conditions
- **Risk Distribution**: Multiple agents reduce single-point-of-failure

## Project Structure

```
sakana-mev-arbitrage/
├── core/                   # Core evolutionary system
│   ├── evolutionary/       # Evolution algorithms
│   ├── agents/            # Agent implementations
│   └── collective/        # Collective intelligence
├── data/                  # Data pipeline
│   ├── pipeline/          # Data ingestion and streaming
│   └── storage/           # Caching and databases
├── execution/             # Trade execution
├── fitness/               # Fitness evaluation
├── monitoring/            # Dashboard and alerts
├── testing/               # Fork testing framework
├── config/                # Configuration files
└── main.py               # Main entry point
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository>
cd sakana-mev-arbitrage

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - Add Ethereum RPC URLs
# - Add API keys
# - Configure initial capital
```

### 2. Ingest Historical Data

```bash
# Ingest last 1000 blocks for training data
python main.py data --start-block 18500000 --end-block 18501000
```

### 3. Run Evolution

```bash
# Evolve strategies for 100 generations
python main.py evolution --generations 100
```

### 4. Backtest Best Strategies

```bash
# Test evolved strategies on historical data
python main.py backtest
```

### 5. Paper Trading

```bash
# Run live paper trading with best strategy
python main.py live
```

## Configuration

### Evolution Parameters (`config/evolution_params.yaml`)

- **Population Size**: 20 agents (focused approach)
- **Mutation Rate**: 5% (conservative)
- **Crossover Rate**: 80% (high mixing)
- **Elite Retention**: 20% (preserve best)

### Strategy Genes (`config/strategy_genes.yaml`)

Agents evolve these parameters:
- Token pairs to trade
- Minimum profit thresholds
- Gas price limits
- Slippage tolerance
- Execution timing
- Flash loan usage

### Risk Limits (`config/risk_limits.yaml`)

- Maximum 10% position size
- 5% daily loss limit
- Circuit breakers and emergency stops

## Operating Modes

### Data Mode
Ingests historical blockchain data to identify arbitrage opportunities for training.

### Evolution Mode
Runs the evolutionary algorithm to discover profitable strategies.

### Backtest Mode
Tests evolved strategies against historical data using fork testing.

### Live Mode
Paper trades in real-time using evolved strategies (no real funds).

## Evolutionary Algorithm

### 1. Population Initialization
- 20 agents with random strategy genes
- Diverse initial strategies

### 2. Fitness Evaluation
- Test each agent on historical opportunities
- Fitness = profit × success_rate

### 3. Selection
- Tournament selection with size 4
- Elite retention of top 20%

### 4. Crossover
- 80% chance of breeding
- Multiple crossover methods (uniform, weighted, blend)

### 5. Mutation
- 5% mutation rate
- Conservative mutations prefer small changes
- Occasional diversity injection

### 6. Generation Loop
- Repeat until convergence or generation limit

## Key Innovations

### Collective Intelligence
Multiple agents vote on opportunities, similar to a school of fish making decisions.

### Species Management
Agents are organized into species based on genetic similarity, maintaining diversity.

### Adaptive Evolution
Mutation rates and strategies adapt based on market conditions.

### Teacher-Student Learning
Successful strategies learn to explain their decisions for debugging and improvement.

## Performance Metrics

### Target Performance (Month 1)
- MEV success rate: 5-10%
- Average profit per transaction: $20-100
- Daily profit target: $50-200

### Scaling Targets (Month 3)
- MEV success rate: 15-25%
- Average profit per transaction: $50-300
- Daily profit target: $500-2000

## Safety Features

- Fork testing before real deployment
- Paper trading validation
- Progressive capital allocation
- Multiple risk limits and circuit breakers
- Emergency shutdown procedures

## Development Roadmap

### Phase 1: Foundation ✅
- Data pipeline
- Evolutionary framework
- Basic arbitrage agent

### Phase 2: Evolution (Current)
- Population evolution
- Fitness evaluation
- Strategy discovery

### Phase 3: Execution
- Flash loan integration
- Gas optimization
- MEV protection

### Phase 4: Production
- Live paper trading
- Performance monitoring
- Gradual capital deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational purposes. MEV extraction involves risks including:
- Total loss of funds
- High gas costs
- Smart contract vulnerabilities
- Competitive environment

Always test thoroughly and never risk more than you can afford to lose.