#!/usr/bin/env python3
"""
Demo Generation & Execution Interface
Generates actionable reports and tracks real execution results
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import our modules (these would be in separate files in production)
# from minimal_arbitrage_detector import SimulationEngine, MinimalDataPipeline
# from profit_focused_evolution import ProfitFocusedEvolution, MarketFeedback

logger = logging.getLogger(__name__)


class DemoGenerator:
    """Generates executable demos with clear profit projections"""
    
    def __init__(self):
        self.template = """
=== ARBITRAGE OPPORTUNITY DEMO ===
Generated: {timestamp}

OPPORTUNITY #{opp_num}
-----------------
Path: {path}
DEXes: {dexes}
Input: {input_amount} {input_token}
Expected Output: {output_amount} {output_token}

PROFIT BREAKDOWN:
- Gross Profit: ${gross_profit:.2f}
- Gas Cost: ${gas_cost:.2f}
- NET PROFIT: ${net_profit:.2f}

BEST AGENT: {agent_id}
Strategy: {strategy}
Success Probability: {success_prob:.1%}
Expected Profit (with slippage): ${expected_profit:.2f}

EXECUTION STEPS:
{execution_steps}

RISKS:
- Slippage Risk: {slippage:.1%}
- MEV Competition: {mev_risk}
- Gas Price Volatility: {gas_risk}

RECOMMENDATION: {recommendation}

MANUAL EXECUTION COMMAND:
{execution_command}
===============================
        """
    
    def generate_demo(self, opportunity: Dict, best_projection: Dict) -> str:
        """Generate human-readable demo for an opportunity"""
        opp = opportunity['opportunity']
        
        # Build execution steps
        steps = []
        for i in range(len(opp['path']) - 1):
            steps.append(f"{i+1}. Swap {opp['path'][i]} → {opp['path'][i+1]} on {opp['dexes'][i]}")
        
        # Determine risks
        slippage_risk = best_projection['slippage_estimate']
        mev_risk = "HIGH" if len(opp['path']) > 3 else "MEDIUM"
        gas_risk = "HIGH" if opp['gas_cost_usd'] > 50 else "LOW"
        
        # Generate execution command (simplified)
        execution_command = self._generate_execution_command(opp)
        
        return self.template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            opp_num=opportunity.get('id', 1),
            path=" → ".join(opp['path']),
            dexes=" → ".join(opp['dexes']),
            input_amount=opp['input_amount'],
            input_token=opp['path'][0],
            output_amount=opp['output_amount'],
            output_token=opp['path'][-1],
            gross_profit=opp['profit_usd'],
            gas_cost=opp['gas_cost_usd'],
            net_profit=opp['net_profit_usd'],
            agent_id=best_projection['agent_id'],
            strategy=json.dumps(best_projection.get('genome', {})),
            success_prob=best_projection['success_probability'],
            expected_profit=best_projection['expected_profit'],
            execution_steps="\n".join(steps),
            slippage=slippage_risk,
            mev_risk=mev_risk,
            gas_risk=gas_risk,
            recommendation=best_projection['recommended_action'],
            execution_command=execution_command
        )
    
    def _generate_execution_command(self, opp: Dict) -> str:
        """Generate execution command for manual trading"""
        # This would generate actual contract calls in production
        return f"Execute via wallet: {opp['path'][0]} → {opp['path'][-1]} (Amount: {opp['input_amount']})"
    
    def generate_summary_report(self, simulation_report: Dict, top_n: int = 5) -> str:
        """Generate executive summary of best opportunities"""
        summary = f"""
=== PROFIT OPPORTUNITY SUMMARY ===
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TOTAL OPPORTUNITIES FOUND: {simulation_report['total_opportunities']}
EXECUTABLE OPPORTUNITIES: {simulation_report['summary']['executable_opportunities']}
TOTAL EXPECTED PROFIT: ${simulation_report['summary']['total_expected_profit']:.2f}
AVERAGE SUCCESS RATE: {simulation_report['summary']['average_success_rate']:.1%}

TOP {top_n} OPPORTUNITIES:
"""
        
        # Add top opportunities
        for i, opp in enumerate(simulation_report['opportunities'][:top_n]):
            if 'best_agent' in opp:
                summary += f"""
{i+1}. {' → '.join(opp['opportunity']['path'])}
   Net Profit: ${opp['opportunity']['net_profit_usd']:.2f}
   Best Agent: {opp['best_agent']}
   Expected: ${opp['best_expected_profit']:.2f}
   Action: EXECUTE
"""
        
        return summary


class ExecutionTracker:
    """Tracks manual execution results and feeds back to evolution"""
    
    def __init__(self, evolution_system, simulation_engine):
        self.evolution = evolution_system
        self.simulation = simulation_engine
        self.pending_executions = {}
        
    def prepare_execution(self, opportunity_id: str, agent_id: str, 
                         predicted_profit: float) -> Dict:
        """Prepare an execution for tracking"""
        execution_id = f"exec_{datetime.now().timestamp()}"
        
        self.pending_executions[execution_id] = {
            'opportunity_id': opportunity_id,
            'agent_id': agent_id,
            'predicted_profit': predicted_profit,
            'prepared_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        return {
            'execution_id': execution_id,
            'instructions': 'Execute the trade and report back with actual results'
        }
    
    def report_execution_result(self, execution_id: str, 
                              actual_profit: float, 
                              success: bool,
                              notes: str = "") -> Dict:
        """Report actual execution results"""
        if execution_id not in self.pending_executions:
            return {'error': 'Execution ID not found'}
        
        execution = self.pending_executions[execution_id]
        execution.update({
            'actual_profit': actual_profit,
            'success': success,
            'executed_at': datetime.now().isoformat(),
            'status': 'completed',
            'notes': notes
        })
        
        # Feed back to evolution
        self.evolution.feedback.record_execution({
            'agent_id': execution['agent_id'],
            'opportunity_id': execution['opportunity_id'],
            'predicted_profit': execution['predicted_profit'],
            'actual_profit': actual_profit,
            'success': success
        })
        
        # Calculate accuracy
        accuracy = 1 - abs(execution['predicted_profit'] - actual_profit) / max(execution['predicted_profit'], 1)
        
        return {
            'execution_id': execution_id,
            'prediction_accuracy': f"{accuracy:.1%}",
            'profit_difference': actual_profit - execution['predicted_profit'],
            'agent_updated': True
        }
    
    def get_pending_executions(self) -> List[Dict]:
        """Get list of pending executions"""
        return [
            {**exec_data, 'execution_id': exec_id}
            for exec_id, exec_data in self.pending_executions.items()
            if exec_data['status'] == 'pending'
        ]


class ProfitSystem:
    """Main system orchestrating everything for immediate profit"""
    
    def __init__(self):
        # Initialize components (these would be imported in production)
        from minimal_arbitrage_detector import MinimalDataPipeline, SimulationEngine
        from profit_focused_evolution import ProfitFocusedEvolution, MarketFeedback
        
        self.data_pipeline = MinimalDataPipeline()
        self.simulation_engine = SimulationEngine()
        self.evolution = ProfitFocusedEvolution(population_size=10)
        self.feedback = MarketFeedback(self.evolution)
        self.demo_generator = DemoGenerator()
        self.execution_tracker = ExecutionTracker(self.evolution, self.simulation_engine)
        
        # Initialize population
        self.evolution.initialize_population()
        
    async def daily_profit_cycle(self):
        """Run one complete daily cycle"""
        logger.info("Starting daily profit cycle...")
        
        # 1. Run simulation with current agents
        agents_for_sim = [
            {'id': agent.id, 'genome': agent.genome}
            for agent in self.evolution.population
        ]
        
        simulation_report = await self.simulation_engine.run_simulation(agents_for_sim)
        
        # 2. Update agents with simulation results
        self.evolution.evaluate_on_simulations(simulation_report)
        
        # 3. Generate demos for top opportunities
        demos = []
        for i, opp in enumerate(simulation_report['opportunities'][:5]):
            if 'best_agent' in opp:
                # Find best projection
                best_proj = next(
                    p for p in opp['agent_projections'] 
                    if p['agent_id'] == opp['best_agent']
                )
                
                # Generate demo
                demo = self.demo_generator.generate_demo(
                    {'opportunity': opp['opportunity'], 'id': i+1},
                    best_proj
                )
                demos.append(demo)
                
                # Prepare for execution tracking
                exec_info = self.execution_tracker.prepare_execution(
                    opportunity_id=opp['opportunity']['id'],
                    agent_id=opp['best_agent'],
                    predicted_profit=opp['best_expected_profit']
                )
                
                print(f"\nExecution ID: {exec_info['execution_id']}")
                print(demo)
        
        # 4. Generate summary
        summary = self.demo_generator.generate_summary_report(simulation_report)
        print(summary)
        
        # 5. Save everything
        with open(f"daily_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write(summary)
            f.write("\n\nDETAILED OPPORTUNITIES:\n")
            for demo in demos:
                f.write(demo)
                f.write("\n")
        
        return {
            'opportunities_found': len(simulation_report['opportunities']),
            'executable_opportunities': simulation_report['summary']['executable_opportunities'],
            'total_expected_profit': simulation_report['summary']['total_expected_profit'],
            'demos_generated': len(demos)
        }
    
    async def process_execution_feedback(self, execution_id: str, 
                                       actual_profit: float, 
                                       success: bool):
        """Process real execution results"""
        result = self.execution_tracker.report_execution_result(
            execution_id=execution_id,
            actual_profit=actual_profit,
            success=success
        )
        
        logger.info(f"Processed execution feedback: {result}")
        
        # If we have enough real data, evolve
        real_executions = sum(
            agent.opportunities_taken 
            for agent in self.evolution.population
        )
        
        if real_executions >= 10:  # Evolve after 10 real trades
            logger.info("Evolving based on real performance...")
            self.evolution.evolve_generation()
            self.evolution.save_checkpoint(
                f"evolution_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        return result
    
    def get_performance_report(self) -> Dict:
        """Get current system performance"""
        perf = self.feedback.analyze_performance()
        
        # Add evolution stats
        perf['evolution'] = {
            'current_generation': self.evolution.generation,
            'best_fitness': self.evolution.best_agent_ever.fitness() if self.evolution.best_agent_ever else 0,
            'population_size': len(self.evolution.population)
        }
        
        # Add best strategies
        perf['top_strategies'] = self.evolution.export_best_strategies(top_n=3)
        
        return perf


# Quick demo of the system
if __name__ == "__main__":
    async def run_demo():
        # Create system
        system = ProfitSystem()
        
        # Run one cycle
        print("Running profit generation cycle...")
        cycle_results = await system.daily_profit_cycle()
        
        print(f"\nCycle Results:")
        print(f"- Opportunities found: {cycle_results['opportunities_found']}")
        print(f"- Executable: {cycle_results['executable_opportunities']}")
        print(f"- Expected profit: ${cycle_results['total_expected_profit']:.2f}")
        
        # Simulate some executions
        print("\nSimulating manual executions...")
        
        # Get pending executions
        pending = system.execution_tracker.get_pending_executions()
        
        if pending:
            # Simulate executing the first one
            exec_id = pending[0]['execution_id']
            predicted = pending[0]['predicted_profit']
            
            # Simulate 90% of predicted profit (realistic slippage)
            actual = predicted * 0.9
            
            print(f"Executing {exec_id}...")
            print(f"Predicted: ${predicted:.2f}")
            print(f"Actual: ${actual:.2f}")
            
            result = await system.process_execution_feedback(
                execution_id=exec_id,
                actual_profit=actual,
                success=True
            )
            
            print(f"Accuracy: {result['prediction_accuracy']}")
        
        # Get performance report
        perf = system.get_performance_report()
        print(f"\nSystem Performance:")
        print(json.dumps(perf, indent=2))
    
    # Run the demo
    asyncio.run(run_demo())