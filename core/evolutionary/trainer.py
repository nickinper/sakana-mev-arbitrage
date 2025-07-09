"""
Core evolutionary training system inspired by Sakana AI
"""
import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Individual agent in the population"""
    id: str
    generation: int
    genome: Dict[str, any]
    fitness: float = 0.0
    age: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    lineage: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'generation': self.generation,
            'genome': self.genome,
            'fitness': self.fitness,
            'age': self.age,
            'wins': self.wins,
            'losses': self.losses,
            'total_profit': self.total_profit,
            'lineage': self.lineage
        }


class ArbitrageEvolutionaryTrainer:
    """
    Evolutionary trainer for arbitrage strategies
    Inspired by Sakana AI's nature-inspired approach
    """
    
    def __init__(self, config: Dict):
        # Evolution parameters
        self.population_size = config.get('population_size', 20)
        self.mutation_rate = config.get('mutation_rate', 0.05)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elite_retention = config.get('elite_retention', 0.2)
        self.tournament_size = config.get('tournament_size', 4)
        
        # Tracking
        self.generation = 0
        self.population: List[Agent] = []
        self.best_agent_ever: Optional[Agent] = None
        self.fitness_history = []
        self.diversity_history = []
        
        # Gene pool for arbitrage strategies
        self.gene_pool = {
            'dex_pairs': ['WETH/USDC', 'WETH/USDT', 'USDC/USDT', 'WETH/DAI', 'USDC/DAI'],
            'min_profit_threshold': [10, 20, 50, 100, 200, 500],
            'max_gas_price_gwei': [30, 50, 100, 150, 200, 300, 500],
            'slippage_tolerance': [0.001, 0.003, 0.005, 0.01, 0.02, 0.03],
            'execution_delay_blocks': [0, 1, 2, 3],
            'route_complexity': [2, 3, 4],
            'flashloan_provider': ['aave', 'dydx', 'uniswap_v3', 'none'],
            'gas_multiplier': [0.9, 1.0, 1.1, 1.2, 1.5, 2.0],
            'position_size_pct': [0.05, 0.1, 0.15, 0.2, 0.25],
            'confidence_threshold': [0.6, 0.7, 0.8, 0.9, 0.95]
        }
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Create initial population with random genomes"""
        logger.info(f"Initializing population of {self.population_size} agents")
        
        for i in range(self.population_size):
            genome = self._create_random_genome()
            agent = Agent(
                id=f"agent_{self.generation}_{i}",
                generation=self.generation,
                genome=genome,
                lineage=[f"genesis_{i}"]
            )
            self.population.append(agent)
    
    def _create_random_genome(self) -> Dict[str, any]:
        """Create a random genome from gene pool"""
        genome = {}
        for gene, values in self.gene_pool.items():
            genome[gene] = random.choice(values)
        return genome
    
    async def evolve_generation(self, fitness_scores: Dict[str, float]) -> List[Agent]:
        """
        Evolve population for one generation
        
        Args:
            fitness_scores: Dictionary mapping agent IDs to fitness scores
            
        Returns:
            New population of agents
        """
        logger.info(f"Evolving generation {self.generation}")
        
        # Update fitness scores
        for agent in self.population:
            if agent.id in fitness_scores:
                agent.fitness = fitness_scores[agent.id]
        
        # Track best agent
        best_current = max(self.population, key=lambda a: a.fitness)
        if not self.best_agent_ever or best_current.fitness > self.best_agent_ever.fitness:
            self.best_agent_ever = best_current
            logger.info(f"New best agent: {best_current.id} with fitness {best_current.fitness}")
        
        # Record statistics
        self._record_generation_stats()
        
        # Elite selection - keep top performers
        elite_count = int(self.population_size * self.elite_retention)
        elite_agents = self._select_elite(elite_count)
        
        # Create new population
        new_population = []
        
        # Add elite agents (they survive unchanged)
        for agent in elite_agents:
            agent.age += 1
            new_population.append(agent)
        
        # Fill rest of population through breeding
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Clone a parent
                parent = self._tournament_selection()
                child = self._clone_agent(parent)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        # Update generation
        self.generation += 1
        self.population = new_population
        
        return self.population
    
    def _select_elite(self, count: int) -> List[Agent]:
        """Select top performing agents"""
        sorted_agents = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        return sorted_agents[:count]
    
    def _tournament_selection(self) -> Agent:
        """Select agent using tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda a: a.fitness)
    
    def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """Create offspring through crossover of two parents"""
        child_genome = {}
        
        # Uniform crossover - each gene randomly from either parent
        for gene in self.gene_pool.keys():
            if random.random() < 0.5:
                child_genome[gene] = parent1.genome.get(gene)
            else:
                child_genome[gene] = parent2.genome.get(gene)
        
        # Create child agent
        child = Agent(
            id=f"agent_{self.generation}_{len(self.population)}",
            generation=self.generation,
            genome=child_genome,
            lineage=parent1.lineage + parent2.lineage
        )
        
        return child
    
    def _mutate(self, agent: Agent) -> Agent:
        """Mutate agent's genome"""
        # Choose random gene to mutate
        gene_to_mutate = random.choice(list(self.gene_pool.keys()))
        
        # Conservative mutation - prefer neighboring values
        current_value = agent.genome[gene_to_mutate]
        possible_values = self.gene_pool[gene_to_mutate]
        
        if isinstance(current_value, (int, float)) and len(possible_values) > 2:
            # For numeric values, prefer small changes
            current_idx = possible_values.index(current_value)
            
            if random.random() < 0.7:  # 70% chance of small change
                # Move to neighbor
                if current_idx == 0:
                    new_idx = 1
                elif current_idx == len(possible_values) - 1:
                    new_idx = len(possible_values) - 2
                else:
                    new_idx = current_idx + random.choice([-1, 1])
            else:
                # Random jump
                new_idx = random.randint(0, len(possible_values) - 1)
            
            agent.genome[gene_to_mutate] = possible_values[new_idx]
        else:
            # For categorical values, random selection
            agent.genome[gene_to_mutate] = random.choice(possible_values)
        
        logger.debug(f"Mutated {gene_to_mutate} in agent {agent.id}")
        return agent
    
    def _clone_agent(self, parent: Agent) -> Agent:
        """Create a copy of an agent"""
        return Agent(
            id=f"agent_{self.generation}_{len(self.population)}",
            generation=self.generation,
            genome=parent.genome.copy(),
            lineage=parent.lineage + [f"clone_of_{parent.id}"]
        )
    
    def _record_generation_stats(self):
        """Record statistics for current generation"""
        fitness_values = [agent.fitness for agent in self.population]
        
        stats = {
            'generation': self.generation,
            'timestamp': datetime.utcnow().isoformat(),
            'mean_fitness': np.mean(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            'std_fitness': np.std(fitness_values),
            'diversity': self._calculate_diversity()
        }
        
        self.fitness_history.append(stats)
        self.diversity_history.append(stats['diversity'])
        
        logger.info(f"Generation {self.generation} stats: "
                   f"Mean fitness: {stats['mean_fitness']:.2f}, "
                   f"Max fitness: {stats['max_fitness']:.2f}, "
                   f"Diversity: {stats['diversity']:.2f}")
    
    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity of population"""
        if len(self.population) < 2:
            return 0.0
        
        # Count unique values for each gene
        diversity_scores = []
        
        for gene in self.gene_pool.keys():
            unique_values = set(agent.genome[gene] for agent in self.population)
            diversity_score = len(unique_values) / len(self.gene_pool[gene])
            diversity_scores.append(diversity_score)
        
        return np.mean(diversity_scores)
    
    def inject_diversity(self, count: int = 3):
        """Inject random agents to increase diversity"""
        logger.info(f"Injecting {count} random agents for diversity")
        
        # Replace worst performers with random agents
        sorted_agents = sorted(self.population, key=lambda a: a.fitness)
        
        for i in range(min(count, len(sorted_agents))):
            genome = self._create_random_genome()
            new_agent = Agent(
                id=f"agent_{self.generation}_diverse_{i}",
                generation=self.generation,
                genome=genome,
                lineage=[f"diversity_injection_{self.generation}"]
            )
            
            # Replace worst agent
            idx = self.population.index(sorted_agents[i])
            self.population[idx] = new_agent
    
    def should_inject_diversity(self, stagnation_limit: int = 10) -> bool:
        """Check if diversity injection is needed"""
        if len(self.fitness_history) < stagnation_limit:
            return False
        
        # Check if fitness has stagnated
        recent_fitness = [h['max_fitness'] for h in self.fitness_history[-stagnation_limit:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        # Check diversity trend
        recent_diversity = self.diversity_history[-stagnation_limit:]
        diversity_trend = recent_diversity[-1] - recent_diversity[0]
        
        # Inject if fitness stagnated and diversity is low
        return fitness_improvement < 0.01 or (diversity_trend < -0.1 and recent_diversity[-1] < 0.3)
    
    def get_best_agents(self, count: int = 5) -> List[Agent]:
        """Get top performing agents"""
        return sorted(self.population, key=lambda a: a.fitness, reverse=True)[:count]
    
    def save_checkpoint(self, filepath: str):
        """Save trainer state to file"""
        checkpoint = {
            'generation': self.generation,
            'population': [agent.to_dict() for agent in self.population],
            'best_agent_ever': self.best_agent_ever.to_dict() if self.best_agent_ever else None,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load trainer state from file"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.generation = checkpoint['generation']
        self.fitness_history = checkpoint['fitness_history']
        self.diversity_history = checkpoint['diversity_history']
        
        # Recreate population
        self.population = []
        for agent_data in checkpoint['population']:
            agent = Agent(
                id=agent_data['id'],
                generation=agent_data['generation'],
                genome=agent_data['genome'],
                fitness=agent_data['fitness'],
                age=agent_data['age'],
                wins=agent_data['wins'],
                losses=agent_data['losses'],
                total_profit=agent_data['total_profit'],
                lineage=agent_data['lineage']
            )
            self.population.append(agent)
        
        # Recreate best agent
        if checkpoint['best_agent_ever']:
            data = checkpoint['best_agent_ever']
            self.best_agent_ever = Agent(
                id=data['id'],
                generation=data['generation'],
                genome=data['genome'],
                fitness=data['fitness'],
                age=data['age'],
                wins=data['wins'],
                losses=data['losses'],
                total_profit=data['total_profit'],
                lineage=data['lineage']
            )
        
        logger.info(f"Loaded checkpoint from {filepath}, generation {self.generation}")
    
    def export_best_strategy(self) -> Dict:
        """Export the best strategy found"""
        if not self.best_agent_ever:
            return None
        
        return {
            'agent_id': self.best_agent_ever.id,
            'generation_found': self.best_agent_ever.generation,
            'genome': self.best_agent_ever.genome,
            'fitness': self.best_agent_ever.fitness,
            'total_profit': self.best_agent_ever.total_profit,
            'win_rate': self.best_agent_ever.wins / (self.best_agent_ever.wins + self.best_agent_ever.losses)
                       if (self.best_agent_ever.wins + self.best_agent_ever.losses) > 0 else 0
        }