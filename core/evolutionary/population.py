"""
Population management for evolutionary system
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages agent population with advanced features"""
    
    def __init__(self, config: Dict):
        self.max_population = config.get('max_population', 100)
        self.species_threshold = config.get('species_threshold', 0.3)
        self.age_penalty_start = config.get('age_penalty_start', 20)
        self.age_penalty_factor = config.get('age_penalty_factor', 0.95)
        
        # Species tracking for diversity
        self.species: Dict[str, List] = {}
        self.species_representatives: Dict[str, Dict] = {}
        
        # Performance tracking
        self.hall_of_fame: List[Dict] = []
        self.hall_of_fame_size = 10
        
        # Genealogy tracking
        self.family_tree: Dict[str, List[str]] = defaultdict(list)
    
    def update_species(self, population: List) -> Dict[str, List]:
        """
        Organize population into species based on genetic similarity
        Inspired by NEAT algorithm
        """
        self.species.clear()
        
        for agent in population:
            assigned = False
            
            # Try to assign to existing species
            for species_id, representative in self.species_representatives.items():
                if self._genetic_distance(agent.genome, representative) < self.species_threshold:
                    if species_id not in self.species:
                        self.species[species_id] = []
                    self.species[species_id].append(agent)
                    assigned = True
                    break
            
            # Create new species if needed
            if not assigned:
                species_id = f"species_{len(self.species)}"
                self.species[species_id] = [agent]
                self.species_representatives[species_id] = agent.genome.copy()
        
        # Update representatives
        for species_id, members in self.species.items():
            if members:
                # Use the fittest member as representative
                best_member = max(members, key=lambda a: a.fitness)
                self.species_representatives[species_id] = best_member.genome.copy()
        
        logger.info(f"Population organized into {len(self.species)} species")
        return self.species
    
    def _genetic_distance(self, genome1: Dict, genome2: Dict) -> float:
        """Calculate genetic distance between two genomes"""
        distance = 0.0
        gene_count = 0
        
        for gene in genome1.keys():
            if gene in genome2:
                gene_count += 1
                
                # Different distance calculation for different types
                val1 = genome1[gene]
                val2 = genome2[gene]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized numeric distance
                    distance += abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                elif val1 != val2:
                    # Categorical difference
                    distance += 1.0
        
        return distance / max(gene_count, 1)
    
    def apply_age_penalty(self, population: List) -> List:
        """Apply fitness penalty to old agents to encourage turnover"""
        for agent in population:
            if agent.age > self.age_penalty_start:
                penalty_generations = agent.age - self.age_penalty_start
                penalty = self.age_penalty_factor ** penalty_generations
                agent.fitness *= penalty
        
        return population
    
    def update_hall_of_fame(self, population: List):
        """Update hall of fame with best agents ever"""
        for agent in population:
            agent_record = {
                'id': agent.id,
                'generation': agent.generation,
                'genome': agent.genome.copy(),
                'fitness': agent.fitness,
                'total_profit': agent.total_profit,
                'win_rate': agent.wins / (agent.wins + agent.losses) if (agent.wins + agent.losses) > 0 else 0
            }
            
            # Check if belongs in hall of fame
            if len(self.hall_of_fame) < self.hall_of_fame_size:
                self.hall_of_fame.append(agent_record)
            else:
                # Replace worst if better
                worst_idx = min(range(len(self.hall_of_fame)), 
                              key=lambda i: self.hall_of_fame[i]['fitness'])
                
                if agent.fitness > self.hall_of_fame[worst_idx]['fitness']:
                    self.hall_of_fame[worst_idx] = agent_record
        
        # Sort hall of fame
        self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
    
    def get_species_stats(self) -> Dict:
        """Get statistics about current species"""
        stats = {}
        
        for species_id, members in self.species.items():
            if members:
                fitness_values = [m.fitness for m in members]
                stats[species_id] = {
                    'size': len(members),
                    'mean_fitness': np.mean(fitness_values),
                    'max_fitness': np.max(fitness_values),
                    'min_fitness': np.min(fitness_values),
                    'best_agent': max(members, key=lambda a: a.fitness).id
                }
        
        return stats
    
    def calculate_species_offspring(self, total_offspring: int) -> Dict[str, int]:
        """
        Calculate how many offspring each species should produce
        Based on species fitness share
        """
        species_fitness = {}
        total_fitness = 0.0
        
        # Calculate adjusted fitness for each species
        for species_id, members in self.species.items():
            if members:
                # Species fitness is average of members
                species_mean_fitness = np.mean([m.fitness for m in members])
                
                # Boost for smaller species (diversity bonus)
                diversity_bonus = 1.0 + (0.1 / max(len(members), 1))
                
                adjusted_fitness = species_mean_fitness * diversity_bonus
                species_fitness[species_id] = adjusted_fitness
                total_fitness += adjusted_fitness
        
        # Allocate offspring proportionally
        offspring_counts = {}
        allocated = 0
        
        for species_id, fitness in species_fitness.items():
            if total_fitness > 0:
                share = fitness / total_fitness
                count = int(share * total_offspring)
                offspring_counts[species_id] = count
                allocated += count
        
        # Distribute remaining offspring to best species
        remaining = total_offspring - allocated
        if remaining > 0 and species_fitness:
            best_species = max(species_fitness.keys(), key=lambda k: species_fitness[k])
            offspring_counts[best_species] = offspring_counts.get(best_species, 0) + remaining
        
        return offspring_counts
    
    def track_genealogy(self, child_id: str, parent_ids: List[str]):
        """Track family relationships"""
        for parent_id in parent_ids:
            self.family_tree[parent_id].append(child_id)
    
    def get_lineage_fitness(self, agent_id: str, population_dict: Dict) -> float:
        """Calculate average fitness of an agent's lineage"""
        descendants = self._get_all_descendants(agent_id)
        
        fitness_values = []
        for desc_id in descendants:
            if desc_id in population_dict:
                fitness_values.append(population_dict[desc_id].fitness)
        
        return np.mean(fitness_values) if fitness_values else 0.0
    
    def _get_all_descendants(self, agent_id: str) -> List[str]:
        """Recursively get all descendants of an agent"""
        descendants = []
        to_check = [agent_id]
        
        while to_check:
            current = to_check.pop()
            children = self.family_tree.get(current, [])
            descendants.extend(children)
            to_check.extend(children)
        
        return descendants
    
    def prune_population(self, population: List, target_size: int) -> List:
        """
        Prune population to target size using multi-criteria selection
        Considers: fitness, age, species diversity
        """
        if len(population) <= target_size:
            return population
        
        # Update species
        self.update_species(population)
        
        # Calculate composite scores
        scores = []
        for agent in population:
            # Base fitness score
            fitness_score = agent.fitness
            
            # Age penalty
            if agent.age > self.age_penalty_start:
                age_penalty = 0.95 ** (agent.age - self.age_penalty_start)
                fitness_score *= age_penalty
            
            # Species diversity bonus
            species_id = self._find_agent_species(agent)
            if species_id and species_id in self.species:
                species_size = len(self.species[species_id])
                diversity_bonus = 1.0 + (0.2 / species_size)  # Smaller species get bonus
                fitness_score *= diversity_bonus
            
            scores.append((fitness_score, agent))
        
        # Sort by composite score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top agents
        return [agent for _, agent in scores[:target_size]]
    
    def _find_agent_species(self, agent) -> Optional[str]:
        """Find which species an agent belongs to"""
        for species_id, members in self.species.items():
            if agent in members:
                return species_id
        return None
    
    def get_diversity_metrics(self, population: List) -> Dict:
        """Calculate various diversity metrics"""
        # Genetic diversity
        gene_diversity = {}
        for gene in population[0].genome.keys():
            unique_values = set(agent.genome[gene] for agent in population)
            gene_diversity[gene] = len(unique_values)
        
        # Species diversity
        species_count = len(self.species)
        species_sizes = [len(members) for members in self.species.values()]
        species_entropy = self._calculate_entropy(species_sizes) if species_sizes else 0
        
        # Fitness diversity
        fitness_values = [agent.fitness for agent in population]
        fitness_std = np.std(fitness_values) if len(fitness_values) > 1 else 0
        
        return {
            'gene_diversity': gene_diversity,
            'mean_gene_diversity': np.mean(list(gene_diversity.values())),
            'species_count': species_count,
            'species_entropy': species_entropy,
            'fitness_std': fitness_std,
            'population_size': len(population)
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy"""
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def export_population_data(self, population: List, filepath: str):
        """Export population data for analysis"""
        data = {
            'population': [agent.to_dict() for agent in population],
            'species': {
                species_id: [agent.id for agent in members]
                for species_id, members in self.species.items()
            },
            'species_stats': self.get_species_stats(),
            'diversity_metrics': self.get_diversity_metrics(population),
            'hall_of_fame': self.hall_of_fame
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported population data to {filepath}")