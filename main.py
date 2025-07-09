#!/usr/bin/env python3
"""
Main entry point for Sakana-inspired MEV arbitrage system
"""
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
import yaml
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pipeline.block_ingestion import BlockIngestion
from data.pipeline.mempool_stream import MempoolStream
from data.pipeline.gas_oracle import GasOracle
from data.storage.opportunity_db import OpportunityDB
from core.evolutionary.trainer import ArbitrageEvolutionaryTrainer
from testing.fork_testing import ForkTester

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/sakana_mev_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class SakanaMEVSystem:
    """Main system orchestrator"""
    
    def __init__(self, mode: str = 'evolution'):
        self.mode = mode
        self.config = self._load_config()
        
        # Initialize components based on mode
        self.block_ingestion = None
        self.mempool_stream = None
        self.gas_oracle = None
        self.evolutionary_trainer = None
        self.fork_tester = None
        self.opportunity_db = None
        
    def _load_config(self) -> Dict:
        """Load configuration from files"""
        config = {}
        
        # Load evolution parameters
        with open('config/evolution_params.yaml', 'r') as f:
            config['evolution'] = yaml.safe_load(f)
        
        # Load strategy genes
        with open('config/strategy_genes.yaml', 'r') as f:
            config['genes'] = yaml.safe_load(f)
        
        # Load risk limits
        with open('config/risk_limits.yaml', 'r') as f:
            config['risk'] = yaml.safe_load(f)
        
        return config
    
    async def initialize(self):
        """Initialize system components"""
        logger.info(f"Initializing Sakana MEV system in {self.mode} mode")
        
        # Database
        db_url = os.getenv('DATABASE_URL', 'sqlite:///arbitrage_performance.db')
        self.opportunity_db = OpportunityDB(db_url)
        
        if self.mode in ['evolution', 'backtest']:
            # Initialize evolutionary trainer
            evolution_config = self.config['evolution']['evolution']
            evolution_config['population_size'] = self.config['evolution']['population']['size']
            
            self.evolutionary_trainer = ArbitrageEvolutionaryTrainer(evolution_config)
            logger.info(f"Initialized evolutionary trainer with population size {evolution_config['population_size']}")
        
        if self.mode in ['data', 'backtest']:
            # Initialize data pipeline
            rpc_url = os.getenv('ETH_RPC_URL')
            if not rpc_url:
                raise ValueError("ETH_RPC_URL not set in environment")
            
            self.block_ingestion = BlockIngestion(rpc_url, db_url)
            logger.info("Initialized block ingestion")
        
        if self.mode == 'live':
            # Initialize live components
            ws_url = os.getenv('ETH_WS_URL')
            if not ws_url:
                raise ValueError("ETH_WS_URL not set in environment")
            
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.mempool_stream = MempoolStream(ws_url, redis_url)
            
            # Gas oracle
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(os.getenv('ETH_RPC_URL')))
            self.gas_oracle = GasOracle(w3, os.getenv('ETHERSCAN_API_KEY'))
            
            logger.info("Initialized live streaming components")
        
        if self.mode in ['test', 'backtest']:
            # Initialize fork testing
            self.fork_tester = ForkTester()
            logger.info("Initialized fork testing environment")
    
    async def run_data_ingestion(self, start_block: int, end_block: int):
        """Run historical data ingestion"""
        logger.info(f"Starting data ingestion from block {start_block} to {end_block}")
        
        stats = await self.block_ingestion.ingest_historical_blocks(
            start_block, end_block, batch_size=10
        )
        
        logger.info(f"Ingestion complete: {stats}")
        
        # Get opportunity statistics
        opp_stats = await self.opportunity_db.get_opportunity_stats(hours=24)
        logger.info(f"Opportunity stats: {opp_stats}")
    
    async def run_evolution(self, generations: int = 100):
        """Run evolutionary training"""
        logger.info(f"Starting evolutionary training for {generations} generations")
        
        # Load historical opportunities for fitness evaluation
        opportunities = await self.opportunity_db.get_profitable_opportunities(
            min_profit=20, hours=24*7, limit=1000
        )
        
        logger.info(f"Loaded {len(opportunities)} historical opportunities for training")
        
        for generation in range(generations):
            # Evaluate current population
            fitness_scores = await self._evaluate_population(opportunities)
            
            # Evolve
            new_population = await self.evolutionary_trainer.evolve_generation(fitness_scores)
            
            # Check for diversity injection
            if self.evolutionary_trainer.should_inject_diversity():
                logger.info("Injecting diversity into population")
                self.evolutionary_trainer.inject_diversity()
            
            # Save checkpoint periodically
            if generation % 10 == 0:
                checkpoint_path = f"checkpoints/generation_{generation}.json"
                self.evolutionary_trainer.save_checkpoint(checkpoint_path)
            
            # Log progress
            best_agents = self.evolutionary_trainer.get_best_agents(3)
            logger.info(f"Generation {generation} - Best fitness: {best_agents[0].fitness:.2f}")
    
    async def _evaluate_population(self, opportunities: List[Dict]) -> Dict[str, float]:
        """Evaluate population fitness on historical opportunities"""
        fitness_scores = {}
        
        for agent in self.evolutionary_trainer.population:
            total_profit = 0
            successful_trades = 0
            
            for opp in opportunities[:100]:  # Limit to 100 for speed
                # Apply agent's strategy to opportunity
                if self._should_take_opportunity(agent.genome, opp):
                    total_profit += opp['net_profit_usd']
                    successful_trades += 1
            
            # Calculate fitness
            if successful_trades > 0:
                avg_profit = total_profit / successful_trades
                success_rate = successful_trades / len(opportunities[:100])
                fitness = (avg_profit * success_rate) ** 0.5  # Geometric mean
            else:
                fitness = 0
            
            fitness_scores[agent.id] = fitness
            
            # Update agent stats
            agent.total_profit = total_profit
            agent.wins = successful_trades
            agent.losses = len(opportunities[:100]) - successful_trades
        
        return fitness_scores
    
    def _should_take_opportunity(self, genome: Dict, opportunity: Dict) -> bool:
        """Determine if agent would take an opportunity based on genome"""
        # Check minimum profit threshold
        if opportunity['net_profit_usd'] < genome['min_profit_threshold']:
            return False
        
        # Check token pair
        if opportunity['token_pair'] not in genome.get('dex_pairs', []):
            return False
        
        # Simulate gas price check
        gas_price = opportunity.get('gas_price_gwei', 100)
        if gas_price > genome['max_gas_price_gwei']:
            return False
        
        # Check profit margin after slippage
        expected_profit = opportunity['net_profit_usd'] * (1 - genome['slippage_tolerance'])
        if expected_profit < genome['min_profit_threshold']:
            return False
        
        # Confidence check (simplified)
        confidence = min(opportunity['net_profit_usd'] / 100, 1.0)  # Simple confidence metric
        if confidence < genome['confidence_threshold']:
            return False
        
        return True
    
    async def run_backtest(self):
        """Run backtesting on evolved strategies"""
        logger.info("Starting backtest of evolved strategies")
        
        # Load best strategies
        best_strategy = self.evolutionary_trainer.export_best_strategy()
        
        if not best_strategy:
            logger.error("No evolved strategy found")
            return
        
        logger.info(f"Testing strategy: {best_strategy['genome']}")
        
        # Start fork
        await self.fork_tester.start_fork()
        
        try:
            # Load recent opportunities
            opportunities = await self.opportunity_db.get_profitable_opportunities(
                min_profit=10, hours=24, limit=50
            )
            
            results = []
            for opp in opportunities:
                if self._should_take_opportunity(best_strategy['genome'], opp):
                    # Simulate execution
                    result = await self.fork_tester.simulate_arbitrage({
                        'token_in': 'WETH',
                        'token_out': 'USDC',
                        'dex_1': opp['dex_1'],
                        'dex_2': opp['dex_2'],
                        'amount': 1.0  # 1 ETH
                    })
                    
                    results.append(result)
                    logger.info(f"Backtest result: {result}")
            
            # Calculate statistics
            successful = sum(1 for r in results if r.get('profitable', False))
            total_profit = sum(r.get('profit_tokens', 0) for r in results)
            
            logger.info(f"Backtest complete: {successful}/{len(results)} profitable, "
                       f"Total profit: {total_profit}")
            
        finally:
            await self.fork_tester.stop_fork()
    
    async def run_live_paper_trading(self):
        """Run live paper trading with evolved strategies"""
        logger.info("Starting live paper trading")
        
        # Load best strategy
        best_strategy = self.evolutionary_trainer.export_best_strategy()
        
        if not best_strategy:
            logger.error("No evolved strategy found")
            return
        
        # Start monitoring
        await asyncio.gather(
            self.mempool_stream.start_streaming(),
            self.gas_oracle.start_monitoring(),
            self._paper_trade_loop(best_strategy['genome'])
        )
    
    async def _paper_trade_loop(self, genome: Dict):
        """Paper trading main loop"""
        paper_balance = {
            'USD': 1000,
            'ETH': 0.5
        }
        
        trades = []
        
        while True:
            try:
                # Get pending transactions
                pending_txs = await self.mempool_stream.get_pending_transactions(50)
                
                for tx in pending_txs:
                    # Analyze for arbitrage opportunity
                    opportunity = self._analyze_transaction(tx)
                    
                    if opportunity and self._should_take_opportunity(genome, opportunity):
                        # Check gas economics
                        gas_decision = self.gas_oracle.should_execute_now(
                            opportunity['net_profit_usd']
                        )
                        
                        if gas_decision['should_execute']:
                            # Paper trade execution
                            trade = {
                                'timestamp': datetime.utcnow(),
                                'opportunity': opportunity,
                                'expected_profit': opportunity['net_profit_usd'],
                                'gas_cost': gas_decision['gas_cost_usd']
                            }
                            
                            trades.append(trade)
                            paper_balance['USD'] += opportunity['net_profit_usd']
                            
                            logger.info(f"Paper trade executed: {trade}")
                
                # Log stats every minute
                await asyncio.sleep(60)
                
                if trades:
                    total_profit = sum(t['expected_profit'] for t in trades)
                    logger.info(f"Paper trading stats: {len(trades)} trades, "
                               f"Total profit: ${total_profit:.2f}, "
                               f"Balance: ${paper_balance['USD']:.2f}")
                
            except Exception as e:
                logger.error(f"Paper trading error: {e}")
                await asyncio.sleep(5)
    
    def _analyze_transaction(self, tx: Dict) -> Optional[Dict]:
        """Analyze transaction for arbitrage opportunity"""
        # Simplified analysis - in production, decode swap parameters
        # and calculate actual arbitrage opportunity
        
        if tx.get('value', 0) > 0:
            return {
                'token_pair': 'WETH/USDC',
                'dex_1': 'uniswap_v2',
                'dex_2': 'sushiswap',
                'net_profit_usd': 50,  # Mock profit
                'gas_price_gwei': tx.get('gas_price', 0) / 1e9
            }
        
        return None


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Sakana MEV Arbitrage System')
    parser.add_argument('mode', choices=['data', 'evolution', 'backtest', 'test', 'live', 'superhuman'],
                       help='Operating mode')
    parser.add_argument('--start-block', type=int, help='Start block for data ingestion')
    parser.add_argument('--end-block', type=int, help='End block for data ingestion')
    parser.add_argument('--generations', type=int, default=100, help='Generations for evolution')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['logs', 'checkpoints']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Initialize system
    system = SakanaMEVSystem(mode=args.mode)
    await system.initialize()
    
    # Run appropriate mode
    if args.mode == 'data':
        if not args.start_block or not args.end_block:
            logger.error("Start and end blocks required for data mode")
            return
        
        await system.run_data_ingestion(args.start_block, args.end_block)
    
    elif args.mode == 'evolution':
        await system.run_evolution(args.generations)
    
    elif args.mode == 'backtest':
        await system.run_backtest()
    
    elif args.mode == 'test':
        # Run test suite
        logger.info("Running system tests...")
        # Add test execution here
    
    elif args.mode == 'live':
        await system.run_live_paper_trading()
    
    elif args.mode == 'superhuman':
        # Run superhuman mode
        logger.info("Starting SUPERHUMAN mode - Beyond human capabilities with full transparency")
        
        # Import superhuman system
        from core.superhuman.integrated_system import SuperhumanTransparentMEVSystem
        
        # Create superhuman configuration
        superhuman_config = {
            'pattern_dimensions': 50,
            'attention_heads': 64,
            'num_experts': 32,
            'strategy_dimensions': 200,
            'population_size': 100,
            'blockchain_trace': True,
            'input_dim': 1000
        }
        
        # Create and run superhuman system
        superhuman_system = SuperhumanTransparentMEVSystem(superhuman_config)
        await superhuman_system.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)