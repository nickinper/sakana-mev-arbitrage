#!/usr/bin/env python3
"""
Interactive Training Controller
Main script for running MEV arbitrage training with visual feedback
"""
import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profit_generation'))

from profit_generation.minimal_arbitrage_detector import SimulationEngine, MinimalDataPipeline
from profit_generation.profit_focused_evolution import ProfitFocusedEvolution, MarketFeedback
from training.training_dashboard import TrainingDashboard
from training.training_terminal_ui import TerminalTrainingUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveTrainingController:
    """Controls the training process with visual feedback"""
    
    def __init__(self, mode: str = 'dashboard', config: Dict = None):
        self.mode = mode
        self.config = config or {}
        
        # Initialize components
        self.evolution = ProfitFocusedEvolution(
            population_size=self.config.get('population_size', 10)
        )
        self.simulation_engine = SimulationEngine()
        self.feedback = MarketFeedback(self.evolution)
        
        # Initialize UI based on mode
        self.ui = None
        self.dashboard = None
        
        if mode == 'dashboard':
            self.dashboard = TrainingDashboard(port=self.config.get('port', 8000))
        elif mode == 'terminal':
            self.ui = TerminalTrainingUI()
        
        # Training state
        self.is_paused = False
        self.should_stop = False
        self.current_generation = 0
        self.total_profit = 0.0
        self.opportunities_found = 0
        
    async def start(self):
        """Start the training process"""
        logger.info(f"Starting interactive training in {self.mode} mode")
        
        # Initialize evolution population
        self.evolution.initialize_population()
        
        # Start UI
        if self.mode == 'dashboard':
            self.dashboard.start()
            await asyncio.sleep(3)  # Wait for dashboard to start
            await self.run_training_loop()
        elif self.mode == 'terminal':
            # Run terminal UI with training loop
            await self.ui.run(self.run_training_step)
        elif self.mode == 'quick':
            await self.run_quick_training()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    async def run_training_loop(self):
        """Main training loop for dashboard mode"""
        try:
            while not self.should_stop:
                if not self.is_paused:
                    await self.run_training_step(None)
                    await asyncio.sleep(1)  # Control training speed
                else:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            await self.save_final_results()
    
    async def run_training_step(self, ui=None):
        """Run one training step"""
        # 1. Find arbitrage opportunities
        opportunities = await self.find_opportunities()
        
        # 2. Evaluate agents on opportunities
        if opportunities:
            await self.evaluate_agents(opportunities)
        
        # 3. Simulate some executions
        executed = await self.simulate_executions(opportunities)
        
        # 4. Update evolution if enough data
        if self.should_evolve():
            await self.evolve_generation()
        
        # 5. Update UI
        await self.update_ui(ui)
    
    async def find_opportunities(self) -> List[Dict]:
        """Find arbitrage opportunities"""
        opportunities = await self.simulation_engine.data_pipeline.scan_opportunities(
            min_profit_usd=self.config.get('min_profit', 50)
        )
        
        self.opportunities_found += len(opportunities)
        
        # Update UI with opportunities
        for opp in opportunities[:3]:  # Top 3
            if self.mode == 'dashboard' and self.dashboard:
                await self.dashboard.add_opportunity(opp.to_dict())
            elif self.ui:
                self.ui.add_opportunity(opp.to_dict())
        
        return opportunities
    
    async def evaluate_agents(self, opportunities: List):
        """Evaluate agents on opportunities"""
        # Convert to format expected by evolution
        sim_opportunities = [opp.to_dict() for opp in opportunities]
        
        # Create simulation report
        agents_for_sim = [
            {'id': agent.id, 'genome': agent.genome}
            for agent in self.evolution.population
        ]
        
        report = self.simulation_engine.projector.generate_report(
            agents_for_sim, opportunities
        )
        
        # Update agents with simulation results
        self.evolution.evaluate_on_simulations(report)
    
    async def simulate_executions(self, opportunities: List) -> int:
        """Simulate some executions and update agents"""
        executed = 0
        
        # Simulate execution of top opportunities
        for i, opp in enumerate(opportunities[:2]):  # Execute top 2
            # Find best agent for this opportunity
            best_agent = max(
                self.evolution.population,
                key=lambda a: self._calculate_expected_profit(a, opp)
            )
            
            # Simulate execution
            expected_profit = self._calculate_expected_profit(best_agent, opp)
            
            # Simulate realistic execution (90% of expected on average)
            import random
            actual_profit = expected_profit * random.uniform(0.7, 1.1)
            success = actual_profit > 0
            
            # Update agent with real results
            self.evolution.update_with_real_results(
                agent_id=best_agent.id,
                opportunity_id=opp.id,
                predicted_profit=expected_profit,
                actual_profit=actual_profit,
                success=success
            )
            
            # Update total profit
            self.total_profit += actual_profit
            
            # Update UI
            if self.mode == 'dashboard' and self.dashboard:
                await self.dashboard.add_profit(actual_profit, success)
            elif self.ui:
                self.ui.add_profit(actual_profit, success)
            
            executed += 1
        
        return executed
    
    def _calculate_expected_profit(self, agent, opportunity) -> float:
        """Calculate expected profit for agent on opportunity"""
        projection = self.simulation_engine.projector.project_opportunity(
            agent.genome, opportunity
        )
        
        if projection['would_execute']:
            return projection['expected_profit']
        return 0.0
    
    def should_evolve(self) -> bool:
        """Check if we should evolve to next generation"""
        # Evolve every 10 simulated trades or 5 real trades
        total_trades = sum(
            agent.opportunities_taken 
            for agent in self.evolution.population
        )
        
        real_trades = sum(
            agent.successful_trades + (agent.opportunities_taken - agent.successful_trades)
            for agent in self.evolution.population
        )
        
        return total_trades >= 10 or real_trades >= 5
    
    async def evolve_generation(self):
        """Evolve to next generation"""
        self.current_generation += 1
        
        # Evolve
        self.evolution.evolve_generation()
        
        # Calculate stats
        best_fitness = max(a.fitness() for a in self.evolution.population)
        avg_fitness = sum(a.fitness() for a in self.evolution.population) / len(self.evolution.population)
        
        # Update UI
        if self.mode == 'dashboard' and self.dashboard:
            await self.dashboard.update_generation(
                self.current_generation, best_fitness, avg_fitness
            )
            await self.dashboard.log_event(
                f"Generation {self.current_generation} completed",
                'info'
            )
        elif self.ui:
            self.ui.update_generation(
                self.current_generation, best_fitness, avg_fitness
            )
        
        logger.info(f"Generation {self.current_generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
    
    async def update_ui(self, ui=None):
        """Update UI with current state"""
        # Prepare agent data
        agent_data = []
        for agent in self.evolution.population:
            agent_data.append({
                'id': agent.id,
                'fitness': agent.fitness(),
                'real_profit': agent.real_profit,
                'success_rate': agent.successful_trades / max(agent.opportunities_taken, 1),
                'genome': agent.genome
            })
        
        # Update based on mode
        if self.mode == 'dashboard' and self.dashboard:
            await self.dashboard.update_agents(agent_data)
        elif ui:  # Terminal UI
            ui.update_agents(agent_data)
    
    async def run_quick_training(self):
        """Run quick training with popup results"""
        logger.info("Running quick training...")
        
        generations = self.config.get('generations', 10)
        
        # Initialize population
        self.evolution.initialize_population()
        
        # Run training
        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            
            # Find opportunities
            opportunities = await self.find_opportunities()
            
            # Evaluate and execute
            if opportunities:
                await self.evaluate_agents(opportunities)
                await self.simulate_executions(opportunities)
            
            # Evolve
            self.evolution.evolve_generation()
            
            # Log progress
            best_fitness = max(a.fitness() for a in self.evolution.population)
            logger.info(f"Best fitness: {best_fitness:.4f}")
        
        # Show results
        if self.config.get('show_results', True):
            await self.show_popup_results()
    
    async def show_popup_results(self):
        """Show results in matplotlib popup"""
        try:
            import matplotlib.pyplot as plt
            
            # Prepare data
            generations = list(range(self.current_generation + 1))
            best_fitness = []
            avg_fitness = []
            
            # Calculate fitness history
            for i in range(self.current_generation + 1):
                # Simplified - in real implementation would track history
                best = max(a.fitness() for a in self.evolution.population)
                avg = sum(a.fitness() for a in self.evolution.population) / len(self.evolution.population)
                best_fitness.append(best)
                avg_fitness.append(avg)
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Sakana MEV Training Results', fontsize=16)
            
            # Fitness evolution
            ax1.plot(generations, best_fitness, 'g-', label='Best Fitness')
            ax1.plot(generations, avg_fitness, 'b-', label='Average Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Evolution')
            ax1.legend()
            ax1.grid(True)
            
            # Agent performance
            agents = sorted(self.evolution.population, key=lambda x: x.fitness(), reverse=True)
            agent_ids = [a.id[:10] for a in agents[:5]]
            agent_profits = [a.real_profit for a in agents[:5]]
            
            ax2.bar(agent_ids, agent_profits, color='green')
            ax2.set_xlabel('Agent ID')
            ax2.set_ylabel('Real Profit ($)')
            ax2.set_title('Top 5 Agents by Profit')
            ax2.tick_params(axis='x', rotation=45)
            
            # Strategy distribution
            strategies = {}
            for agent in self.evolution.population:
                key = f"${agent.genome['min_profit_usd']}"
                strategies[key] = strategies.get(key, 0) + 1
            
            ax3.pie(strategies.values(), labels=strategies.keys(), autopct='%1.1f%%')
            ax3.set_title('Strategy Distribution')
            
            # Summary text
            summary = f"""Training Summary:
            
Generations: {self.current_generation}
Total Profit: ${self.total_profit:.2f}
Opportunities Found: {self.opportunities_found}
Best Agent: {agents[0].id}
Best Fitness: {agents[0].fitness():.4f}
Best Strategy: Min ${agents[0].genome['min_profit_usd']}, Gas {agents[0].genome['gas_multiplier']:.1f}x
            """
            
            ax4.text(0.1, 0.5, summary, fontsize=12, verticalalignment='center')
            ax4.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for popup results")
            # Fallback to text output
            await self.save_final_results()
    
    async def save_final_results(self):
        """Save final training results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'generations': self.current_generation,
            'total_profit': self.total_profit,
            'opportunities_found': self.opportunities_found,
            'best_strategies': self.evolution.export_best_strategies(top_n=5),
            'config': self.config
        }
        
        filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save evolution checkpoint
        self.evolution.save_checkpoint(
            f"evolution_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive MEV Arbitrage Training with Visual Feedback'
    )
    
    parser.add_argument(
        '--mode',
        choices=['dashboard', 'terminal', 'quick'],
        default='dashboard',
        help='UI mode: dashboard (web), terminal (console), or quick (popup results)'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of generations to train'
    )
    
    parser.add_argument(
        '--population',
        type=int,
        default=10,
        help='Population size'
    )
    
    parser.add_argument(
        '--min-profit',
        type=float,
        default=50,
        help='Minimum profit threshold for opportunities'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for dashboard mode'
    )
    
    parser.add_argument(
        '--show-results',
        action='store_true',
        help='Show popup results in quick mode'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'generations': args.generations,
        'population_size': args.population,
        'min_profit': args.min_profit,
        'port': args.port,
        'show_results': args.show_results
    }
    
    # Create and start controller
    controller = InteractiveTrainingController(mode=args.mode, config=config)
    
    try:
        await controller.start()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    finally:
        await controller.save_final_results()


if __name__ == "__main__":
    asyncio.run(main())