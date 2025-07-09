#!/usr/bin/env python3
"""
Autonomous Training Script
Run self-directed agents that discover patterns independently
"""
import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
import json

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autonomous.discovery_engine import DiscoveryEngine
from autonomous.intelligent_reporter import IntelligentReporter
from training.training_dashboard import TrainingDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutonomousTrainingSystem:
    """
    Fully autonomous training system where agents self-direct their learning
    """
    
    def __init__(self, num_agents: int = 20, use_dashboard: bool = True):
        self.discovery_engine = DiscoveryEngine(num_agents=num_agents)
        self.reporter = IntelligentReporter()
        self.use_dashboard = use_dashboard
        
        if use_dashboard:
            self.dashboard = TrainingDashboard()
            self.dashboard.start()
            logger.info("Dashboard started at http://localhost:8000")
        
        self.running = True
        self.cycle_delay = 2.0  # seconds between cycles
        
    async def run(self, max_generations: int = None):
        """Run autonomous training"""
        logger.info("Starting autonomous training system")
        
        generation = 0
        try:
            while self.running and (max_generations is None or generation < max_generations):
                # Run discovery cycle
                cycle_results = await self.discovery_engine.run_discovery_cycle()
                
                # Update dashboard if enabled
                if self.use_dashboard:
                    await self._update_dashboard(cycle_results)
                
                # Generate report every 10 generations
                if generation % 10 == 0:
                    report = self.reporter.generate_report(self.discovery_engine)
                    await self._display_report(report)
                    
                    # Save report
                    self._save_report(report)
                
                # Log brief status
                self._log_cycle_status(cycle_results)
                
                generation += 1
                await asyncio.sleep(self.cycle_delay)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Generate final report
            final_report = self.reporter.generate_report(self.discovery_engine)
            await self._display_report(final_report)
            self._save_report(final_report, final=True)
    
    async def _update_dashboard(self, cycle_results: Dict):
        """Update dashboard with cycle results"""
        # Update generation
        best_profit = max(
            (p['profit'] for p in cycle_results['profitable_trades']),
            default=0
        )
        avg_profit = sum(
            p['profit'] for p in cycle_results['profitable_trades']
        ) / max(len(cycle_results['profitable_trades']), 1)
        
        await self.dashboard.update_generation(
            cycle_results['generation'],
            best_profit,
            avg_profit
        )
        
        # Update agents
        await self.dashboard.update_agents(
            cycle_results['agent_performance']
        )
        
        # Log discoveries
        for discovery in cycle_results['discoveries']:
            await self.dashboard.add_opportunity({
                'net_profit_usd': discovery['profitability'],
                'path': [discovery['agent'], 'discovered', discovery['discovery'][:50]],
                'success_probability': 0.8
            })
        
        # Log profitable trades
        for trade in cycle_results['profitable_trades']:
            await self.dashboard.add_profit(trade['profit'], True)
        
        # Log events
        for behavior in cycle_results['emergent_behaviors']:
            await self.dashboard.log_event(
                f"Emergent: {behavior['description']}", 
                'info'
            )
        
        # Log interesting questions
        for question in cycle_results['new_questions'][:3]:
            await self.dashboard.log_event(
                f"{question['agent']} asks: {question['question']}", 
                'info'
            )
    
    async def _display_report(self, report: Dict):
        """Display report to console"""
        formatted = self.reporter.format_report_for_display(report)
        print("\n" + formatted)
    
    def _save_report(self, report: Dict, final: bool = False):
        """Save report to file"""
        prefix = "final_" if final else ""
        filename = f"{prefix}autonomous_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filename}")
    
    def _log_cycle_status(self, cycle_results: Dict):
        """Log brief cycle status"""
        discoveries = len(cycle_results['discoveries'])
        profits = len(cycle_results['profitable_trades'])
        questions = len(cycle_results['new_questions'])
        emergent = len(cycle_results['emergent_behaviors'])
        
        logger.info(
            f"Generation {cycle_results['generation']}: "
            f"{discoveries} discoveries, {profits} profitable trades, "
            f"{questions} new questions, {emergent} emergent behaviors"
        )
    
    def adjust_parameters(self, **kwargs):
        """Adjust system parameters based on results"""
        if 'cycle_delay' in kwargs:
            self.cycle_delay = kwargs['cycle_delay']
            logger.info(f"Cycle delay adjusted to {self.cycle_delay}s")
        
        if 'curiosity_boost' in kwargs:
            # Increase curiosity of all agents
            boost = kwargs['curiosity_boost']
            for agent in self.discovery_engine.agents:
                agent.curiosity = min(1.0, agent.curiosity + boost)
            logger.info(f"Boosted all agent curiosity by {boost}")
        
        if 'reset_bottom_percent' in kwargs:
            # Reset bottom performing agents
            percent = kwargs['reset_bottom_percent']
            num_reset = int(len(self.discovery_engine.agents) * percent)
            
            sorted_agents = sorted(
                self.discovery_engine.agents,
                key=lambda a: a.total_profit
            )
            
            for agent in sorted_agents[:num_reset]:
                old_profit = agent.total_profit
                agent.__init__(agent.id)  # Reset
                logger.info(f"Reset agent {agent.id} (was ${old_profit:.2f})")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run autonomous MEV discovery system'
    )
    
    parser.add_argument(
        '--agents',
        type=int,
        default=20,
        help='Number of autonomous agents'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=None,
        help='Maximum generations to run (None for infinite)'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable web dashboard'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode with minimal delay'
    )
    
    args = parser.parse_args()
    
    # Create system
    system = AutonomousTrainingSystem(
        num_agents=args.agents,
        use_dashboard=not args.no_dashboard
    )
    
    if args.fast:
        system.cycle_delay = 0.5
    
    # Print startup message
    print("""
╔═══════════════════════════════════════════════════════════╗
║        🧬 AUTONOMOUS MEV DISCOVERY SYSTEM 🧬              ║
║                                                           ║
║  Agents will explore independently and discover patterns  ║
║  You'll receive reports on their findings                 ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n🤖 Starting {args.agents} autonomous agents...")
    print(f"📊 Dashboard: {'http://localhost:8000' if not args.no_dashboard else 'Disabled'}")
    print(f"📝 Reports will be generated every 10 generations\n")
    
    # Run system
    try:
        await system.run(max_generations=args.generations)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in training: {e}", exc_info=True)
    
    print("\n✅ Training complete! Check the generated reports for discoveries.")


if __name__ == "__main__":
    asyncio.run(main())