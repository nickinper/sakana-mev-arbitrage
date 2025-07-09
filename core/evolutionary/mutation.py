"""
Mutation operators for evolutionary algorithm
"""
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MutationOperators:
    """Collection of mutation operators for strategy evolution"""
    
    def __init__(self, config: Dict, gene_pool: Dict[str, List]):
        self.gene_pool = gene_pool
        
        # Mutation parameters
        self.conservative_prob = config.get('conservative_mutation_prob', 0.7)
        self.multi_gene_prob = config.get('multi_gene_mutation_prob', 0.1)
        self.creep_rate = config.get('creep_rate', 0.1)
        self.reset_prob = config.get('reset_mutation_prob', 0.05)
        
        # Adaptive mutation rates
        self.min_mutation_rate = config.get('min_mutation_rate', 0.01)
        self.max_mutation_rate = config.get('max_mutation_rate', 0.3)
        self.current_mutation_rate = config.get('initial_mutation_rate', 0.05)
        
        # Gene relationships for correlated mutations
        self.gene_correlations = {
            'min_profit_threshold': ['slippage_tolerance', 'confidence_threshold'],
            'max_gas_price_gwei': ['gas_multiplier'],
            'route_complexity': ['execution_delay_blocks'],
            'position_size_pct': ['min_profit_threshold']
        }
    
    def mutate(self, genome: Dict, mutation_rate: Optional[float] = None) -> Dict:
        """
        Main mutation function
        
        Args:
            genome: Agent's genome to mutate
            mutation_rate: Override default mutation rate
            
        Returns:
            Mutated genome
        """
        if mutation_rate is None:
            mutation_rate = self.current_mutation_rate
        
        # Decide how many genes to mutate
        if random.random() < self.multi_gene_prob:
            # Multi-gene mutation
            num_genes = random.randint(2, min(4, len(genome)))
            genes_to_mutate = random.sample(list(genome.keys()), num_genes)
        else:
            # Single gene mutation
            genes_to_mutate = [random.choice(list(genome.keys()))]
        
        # Apply mutations
        mutated_genome = genome.copy()
        
        for gene in genes_to_mutate:
            if random.random() < mutation_rate:
                mutated_genome = self._mutate_gene(mutated_genome, gene)
        
        return mutated_genome
    
    def _mutate_gene(self, genome: Dict, gene: str) -> Dict:
        """Mutate a single gene"""
        current_value = genome[gene]
        possible_values = self.gene_pool.get(gene, [current_value])
        
        # Determine mutation type
        if isinstance(current_value, (int, float)) and len(possible_values) > 2:
            # Numeric gene - use appropriate mutation
            if random.random() < self.conservative_prob:
                # Conservative mutation - small change
                new_value = self._conservative_numeric_mutation(
                    current_value, possible_values
                )
            else:
                # Random reset
                new_value = random.choice(possible_values)
        else:
            # Categorical gene - random selection
            other_values = [v for v in possible_values if v != current_value]
            if other_values:
                new_value = random.choice(other_values)
            else:
                new_value = current_value
        
        genome[gene] = new_value
        
        # Check for correlated mutations
        if gene in self.gene_correlations and random.random() < 0.3:
            # 30% chance to also mutate correlated genes
            correlated_gene = random.choice(self.gene_correlations[gene])
            if correlated_gene in genome:
                genome = self._mutate_gene(genome, correlated_gene)
        
        return genome
    
    def _conservative_numeric_mutation(self, current_value: float, 
                                     possible_values: List[float]) -> float:
        """Conservative mutation for numeric values"""
        # Find current position in list
        try:
            current_idx = possible_values.index(current_value)
        except ValueError:
            # Current value not in list, find closest
            distances = [abs(v - current_value) for v in possible_values]
            current_idx = distances.index(min(distances))
        
        # Move to neighbor with bias towards middle values
        max_idx = len(possible_values) - 1
        
        if current_idx == 0:
            new_idx = 1
        elif current_idx == max_idx:
            new_idx = max_idx - 1
        else:
            # Bias towards middle values
            if random.random() < 0.5:
                # Move towards middle
                middle_idx = max_idx // 2
                if current_idx < middle_idx:
                    new_idx = current_idx + 1
                else:
                    new_idx = current_idx - 1
            else:
                # Random neighbor
                new_idx = current_idx + random.choice([-1, 1])
        
        return possible_values[new_idx]
    
    def creep_mutation(self, genome: Dict, gene: str) -> Dict:
        """
        Creep mutation - small random perturbation
        For continuous numeric values
        """
        current_value = genome[gene]
        
        if isinstance(current_value, (int, float)):
            # Add Gaussian noise
            noise = np.random.normal(0, self.creep_rate * abs(current_value))
            new_value = current_value + noise
            
            # Ensure within bounds if specified
            possible_values = self.gene_pool.get(gene, [])
            if possible_values:
                min_val = min(possible_values)
                max_val = max(possible_values)
                new_value = np.clip(new_value, min_val, max_val)
            
            # Maintain type
            if isinstance(current_value, int):
                new_value = int(round(new_value))
            
            genome[gene] = new_value
        
        return genome
    
    def boundary_mutation(self, genome: Dict) -> Dict:
        """
        Boundary mutation - occasionally set genes to extreme values
        Helps explore edge cases
        """
        gene = random.choice(list(genome.keys()))
        possible_values = self.gene_pool.get(gene, [])
        
        if possible_values and isinstance(possible_values[0], (int, float)):
            # Set to min or max value
            if random.random() < 0.5:
                genome[gene] = min(possible_values)
            else:
                genome[gene] = max(possible_values)
        
        return genome
    
    def swap_mutation(self, genome: Dict) -> Dict:
        """
        Swap values between two compatible genes
        """
        # Find numeric genes
        numeric_genes = [
            gene for gene in genome.keys()
            if isinstance(genome[gene], (int, float))
        ]
        
        if len(numeric_genes) >= 2:
            gene1, gene2 = random.sample(numeric_genes, 2)
            
            # Check if swap makes sense (same type of value)
            val1_range = self.gene_pool.get(gene1, [])
            val2_range = self.gene_pool.get(gene2, [])
            
            if val1_range and val2_range:
                # Swap if values are in each other's range
                if genome[gene2] in val1_range and genome[gene1] in val2_range:
                    genome[gene1], genome[gene2] = genome[gene2], genome[gene1]
        
        return genome
    
    def inversion_mutation(self, genome: Dict, segment_size: int = 3) -> Dict:
        """
        Inversion mutation - reverse order of related genes
        Useful for exploring different execution orders
        """
        # This is more applicable to sequence-based genes
        # For our use case, we'll implement a variant
        
        # Invert numeric preferences
        inversion_groups = [
            ['min_profit_threshold', 'slippage_tolerance'],
            ['gas_multiplier', 'max_gas_price_gwei']
        ]
        
        for group in inversion_groups:
            if all(gene in genome for gene in group):
                if random.random() < 0.5:
                    # Invert the relationship
                    values = [genome[gene] for gene in group]
                    
                    # Reverse map to opposite end of ranges
                    for i, gene in enumerate(group):
                        possible = self.gene_pool.get(gene, [])
                        if possible:
                            current_idx = possible.index(genome[gene])
                            new_idx = len(possible) - 1 - current_idx
                            genome[gene] = possible[new_idx]
        
        return genome
    
    def adaptive_mutation(self, genome: Dict, performance_history: List[float]) -> Dict:
        """
        Adaptive mutation that adjusts based on performance
        """
        # Adjust mutation rate based on performance trend
        if len(performance_history) >= 5:
            recent_performance = performance_history[-5:]
            
            # Check if performance is improving
            improvement = recent_performance[-1] - recent_performance[0]
            
            if improvement > 0:
                # Performance improving, reduce mutation
                self.current_mutation_rate *= 0.95
            else:
                # Performance stagnant, increase mutation
                self.current_mutation_rate *= 1.05
            
            # Keep within bounds
            self.current_mutation_rate = np.clip(
                self.current_mutation_rate,
                self.min_mutation_rate,
                self.max_mutation_rate
            )
        
        # Apply mutation with adaptive rate
        return self.mutate(genome, self.current_mutation_rate)
    
    def guided_mutation(self, genome: Dict, successful_genomes: List[Dict]) -> Dict:
        """
        Guided mutation - bias mutations towards successful patterns
        """
        if not successful_genomes:
            return self.mutate(genome)
        
        # Analyze successful patterns
        gene_success_values = {}
        
        for gene in genome.keys():
            gene_success_values[gene] = {}
            
            for success_genome in successful_genomes:
                value = success_genome.get(gene)
                if value is not None:
                    if value not in gene_success_values[gene]:
                        gene_success_values[gene][value] = 0
                    gene_success_values[gene][value] += 1
        
        # Mutate with bias towards successful values
        gene_to_mutate = random.choice(list(genome.keys()))
        
        if gene_success_values[gene_to_mutate]:
            # Weight by success frequency
            values = list(gene_success_values[gene_to_mutate].keys())
            weights = list(gene_success_values[gene_to_mutate].values())
            
            # Add some randomness
            weights = [w + 1 for w in weights]  # Avoid zero weights
            
            genome[gene_to_mutate] = random.choices(values, weights=weights)[0]
        else:
            # No success data, use regular mutation
            genome = self._mutate_gene(genome, gene_to_mutate)
        
        return genome
    
    def hypermutation(self, genome: Dict, mutation_strength: float = 0.5) -> Dict:
        """
        Hypermutation - aggressive mutation for diversity injection
        """
        num_genes = int(len(genome) * mutation_strength)
        genes_to_mutate = random.sample(list(genome.keys()), num_genes)
        
        for gene in genes_to_mutate:
            # Always pick a different value
            current = genome[gene]
            possible = self.gene_pool.get(gene, [current])
            different = [v for v in possible if v != current]
            
            if different:
                genome[gene] = random.choice(different)
        
        return genome
    
    def get_mutation_stats(self) -> Dict:
        """Get current mutation statistics"""
        return {
            'current_rate': self.current_mutation_rate,
            'min_rate': self.min_mutation_rate,
            'max_rate': self.max_mutation_rate,
            'conservative_prob': self.conservative_prob,
            'multi_gene_prob': self.multi_gene_prob
        }