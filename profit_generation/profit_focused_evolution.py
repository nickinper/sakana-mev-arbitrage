#!/usr/bin/env python3
"""
Profit-Focused Evolution System
Evolves agents based on real market feedback for immediate profit
"""
import logging
import random
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfitAgent:
    """Simplified agent focused on profit generation"""
    id: str
    generation: int
    genome: Dict[str, any]
    
    # Performance metrics
    simulated_profit: float = 0.0
    real_profit: float = 0.0
    prediction_accuracy: float = 0.0
    opportunities_taken: int = 0
    successful_trades: int = 0
    
    # For evolution
    age: int = 0
    lineage: List[str] = field(default_factory=list)
    
    def fitness(self) -> float:
        """Calculate fitness based on real performance"""
        if self.opportunities_taken == 0:
            return self.simulated_profit * 0.1  # Use simulation if no real data
        
        # Weighted combination of real profit and prediction accuracy
        real_weight = 0.7
        accuracy_weight = 0.3
        
        # Normalize metrics
        profit_score = self.real_profit / max(self.opportunities_taken * 100, 1)  # Avg profit per opportunity
        accuracy_score = self.prediction_accuracy
        
        return (real_weight * profit_score + accuracy_weight * accuracy_score)


class ProfitFocusedEvolution:
    """Evolution system focused on immediate profit generation"""
    
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.generation = 0
        self.population: List[ProfitAgent] = []
        
        # Simplified gene pool for quick results
        self.gene_pool = {
            'min_profit_usd': [50, 75, 100, 150, 200],
            'gas_multiplier': [1.2, 1.5, 1.8, 2.0],
            'execution_speed': ['fast', 'normal'],
            'max_slippage': [0.02, 0.03, 0.05],  # 2%, 3%, 5%
            'risk_tolerance': ['low', 'medium', 'high'],
        }
        
        # Evolution parameters (simplified)
        self.mutation_rate = 0.1
        self.elite_count = 2
        
        # Performance tracking
        self.best_agent_ever: Optional[ProfitAgent] = None
        self.profit_history = []
        
    def initialize_population(self):
        """Create initial population with diverse strategies"""
        self.population = []
        
        # Create one conservative agent
        conservative = ProfitAgent(
            id=f"agent_0_conservative",
            generation=0,
            genome={
                'min_profit_usd': 150,
                'gas_multiplier': 2.0,
                'execution_speed': 'normal',
                'max_slippage': 0.02,
                'risk_tolerance': 'low'
            }
        )
        self.population.append(conservative)
        
        # Create one aggressive agent
        aggressive = ProfitAgent(
            id=f"agent_0_aggressive",
            generation=0,
            genome={
                'min_profit_usd': 50,
                'gas_multiplier': 1.2,
                'execution_speed': 'fast',
                'max_slippage': 0.05,
                'risk_tolerance': 'high'
            }
        )
        self.population.append(aggressive)
        
        # Fill rest with random agents
        for i in range(2, self.population_size):
            agent = ProfitAgent(
                id=f"agent_0_{i}",
                generation=0,
                genome=self._random_genome()
            )
            self.population.append(agent)
        
        logger.info(f"Initialized population with {len(self.population)} agents")
    
    def _random_genome(self) -> Dict:
        """Generate random genome"""
        return {
            key: random.choice(values)
            for key, values in self.gene_pool.items()
        }
    
    def evaluate_on_simulations(self, simulation_results: Dict):
        """Update agent metrics based on simulation results"""
        for opp_report in simulation_results['opportunities']:
            for projection in opp_report['agent_projections']:
                agent_id = projection['agent_id']
                agent = self._find_agent(agent_id)
                
                if agent and projection['would_execute']:
                    agent.simulated_profit += projection['expected_profit']
                    agent.opportunities_taken += 1
    
    def update_with_real_results(self, agent_id: str, opportunity_id: str, 
                               predicted_profit: float, actual_profit: float, success: bool):
        """Update agent with real execution results"""
        agent = self._find_agent(agent_id)
        if not agent:
            return
        
        # Update real performance metrics
        agent.real_profit += actual_profit
        if success:
            agent.successful_trades += 1
        
        # Calculate prediction accuracy
        if predicted_profit > 0:
            accuracy = 1 - abs(predicted_profit - actual_profit) / predicted_profit
            # Update running average of accuracy
            if agent.prediction_accuracy == 0:
                agent.prediction_accuracy = max(0, accuracy)
            else:
                agent.prediction_accuracy = 0.8 * agent.prediction_accuracy + 0.2 * max(0, accuracy)
        
        logger.info(f"Updated {agent_id}: real_profit=${agent.real_profit:.2f}, accuracy={agent.prediction_accuracy:.2%}")
    
    def evolve_generation(self):
        """Evolve to next generation based on performance"""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness(), reverse=True)
        
        # Track best agent
        if not self.best_agent_ever or self.population[0].fitness() > self.best_agent_ever.fitness():
            self.best_agent_ever = self.population[0]
            logger.info(f"New best agent: {self.best_agent_ever.id} with fitness {self.best_agent_ever.fitness():.4f}")
        
        # Create new population
        new_population = []
        
        # Keep elite agents
        for i in range(self.elite_count):
            elite = self.population[i]
            elite.age += 1
            new_population.append(elite)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            child_genome = self._crossover(parent1.genome, parent2.genome)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_genome = self._mutate(child_genome)
            
            # Create child
            child = ProfitAgent(
                id=f"agent_{self.generation + 1}_{len(new_population)}",
                generation=self.generation + 1,
                genome=child_genome,
                lineage=[parent1.id, parent2.id]
            )
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Log generation stats
        avg_fitness = sum(a.fitness() for a in self.population) / len(self.population)
        logger.info(f"Generation {self.generation}: avg_fitness={avg_fitness:.4f}")
    
    def _find_agent(self, agent_id: str) -> Optional[ProfitAgent]:
        """Find agent by ID"""
        for agent in self.population:
            if agent.id == agent_id:
                return agent
        return None
    
    def _tournament_select(self, tournament_size: int = 3) -> ProfitAgent:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness())
    
    def _crossover(self, genome1: Dict, genome2: Dict) -> Dict:
        """Simple uniform crossover"""
        child_genome = {}
        for key in genome1:
            if random.random() < 0.5:
                child_genome[key] = genome1[key]
            else:
                child_genome[key] = genome2[key]
        return child_genome
    
    def _mutate(self, genome: Dict) -> Dict:
        """Mutate single gene"""
        mutated = genome.copy()
        gene_to_mutate = random.choice(list(self.gene_pool.keys()))
        mutated[gene_to_mutate] = random.choice(self.gene_pool[gene_to_mutate])
        return mutated
    
    def export_best_strategies(self, top_n: int = 3) -> List[Dict]:
        """Export top performing strategies"""
        sorted_agents = sorted(self.population, key=lambda x: x.fitness(), reverse=True)
        
        strategies = []
        for agent in sorted_agents[:top_n]:
            strategy = {
                'agent_id': agent.id,
                'genome': agent.genome,
                'fitness': agent.fitness(),
                'real_profit': agent.real_profit,
                'simulated_profit': agent.simulated_profit,
                'prediction_accuracy': agent.prediction_accuracy,
                'success_rate': agent.successful_trades / max(agent.opportunities_taken, 1)
            }
            strategies.append(strategy)
        
        return strategies
    
    def save_checkpoint(self, filename: str):
        """Save current state"""
        checkpoint = {
            'generation': self.generation,
            'population': [
                {
                    'id': agent.id,
                    'genome': agent.genome,
                    'real_profit': agent.real_profit,
                    'simulated_profit': agent.simulated_profit,
                    'prediction_accuracy': agent.prediction_accuracy,
                    'opportunities_taken': agent.opportunities_taken,
                    'successful_trades': agent.successful_trades
                }
                for agent in self.population
            ],
            'best_agent_ever': {
                'id': self.best_agent_ever.id,
                'genome': self.best_agent_ever.genome,
                'fitness': self.best_agent_ever.fitness()
            } if self.best_agent_ever else None
        }
        
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {filename}")


class MarketFeedback:
    """Collects and processes real market execution feedback"""
    
    def __init__(self, evolution_system: ProfitFocusedEvolution):
        self.evolution = evolution_system
        self.execution_log = []
        
    def record_execution(self, execution_data: Dict):
        """Record real execution results"""
        # Log the execution
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            **execution_data
        })
        
        # Update agent performance
        self.evolution.update_with_real_results(
            agent_id=execution_data['agent_id'],
            opportunity_id=execution_data['opportunity_id'],
            predicted_profit=execution_data['predicted_profit'],
            actual_profit=execution_data['actual_profit'],
            success=execution_data['success']
        )
        
        # Save log
        with open(f"execution_log_{datetime.now().strftime('%Y%m%d')}.json", 'a') as f:
            json.dump(execution_data, f)
            f.write('\n')
    
    def analyze_performance(self) -> Dict:
        """Analyze real vs predicted performance"""
        if not self.execution_log:
            return {'status': 'No executions yet'}
        
        total_predicted = sum(e['predicted_profit'] for e in self.execution_log)
        total_actual = sum(e['actual_profit'] for e in self.execution_log)
        success_count = sum(1 for e in self.execution_log if e['success'])
        
        return {
            'total_executions': len(self.execution_log),
            'successful_executions': success_count,
            'success_rate': success_count / len(self.execution_log),
            'total_predicted_profit': total_predicted,
            'total_actual_profit': total_actual,
            'prediction_accuracy': 1 - abs(total_predicted - total_actual) / max(total_predicted, 1),
            'average_profit_per_trade': total_actual / len(self.execution_log)
        }


# Example usage for immediate profit generation
if __name__ == "__main__":
    # Create evolution system
    evolution = ProfitFocusedEvolution(population_size=10)
    evolution.initialize_population()
    
    # Create feedback system
    feedback = MarketFeedback(evolution)
    
    # Simulate one generation cycle
    print("Initial Population:")
    for agent in evolution.population:
        print(f"  {agent.id}: {agent.genome}")
    
    # Simulate some results
    evolution.update_with_real_results(
        agent_id="agent_0_conservative",
        opportunity_id="opp_001",
        predicted_profit=120,
        actual_profit=95,
        success=True
    )
    
    evolution.update_with_real_results(
        agent_id="agent_0_aggressive",
        opportunity_id="opp_002", 
        predicted_profit=80,
        actual_profit=110,
        success=True
    )
    
    # Evolve
    evolution.evolve_generation()
    
    # Export best strategies
    best_strategies = evolution.export_best_strategies()
    print("\nBest Strategies After Evolution:")
    for strategy in best_strategies:
        print(f"  {strategy['agent_id']}: fitness={strategy['fitness']:.4f}")
        print(f"    Genome: {strategy['genome']}")