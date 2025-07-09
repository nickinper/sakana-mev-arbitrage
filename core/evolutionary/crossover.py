"""
Crossover operators for evolutionary algorithm
"""
import random
import logging
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class CrossoverOperators:
    """Collection of crossover operators for strategy evolution"""
    
    def __init__(self, config: Dict):
        self.blend_alpha = config.get('blend_alpha', 0.5)
        self.multi_point_prob = config.get('multi_point_prob', 0.3)
        self.segment_crossover_prob = config.get('segment_crossover_prob', 0.2)
    
    def uniform_crossover(self, parent1_genome: Dict, parent2_genome: Dict) -> Dict:
        """
        Uniform crossover - each gene randomly selected from either parent
        Most general and widely applicable
        """
        child_genome = {}
        
        for gene in parent1_genome.keys():
            if random.random() < 0.5:
                child_genome[gene] = parent1_genome[gene]
            else:
                child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
        
        return child_genome
    
    def weighted_crossover(self, parent1_genome: Dict, parent2_genome: Dict,
                         parent1_fitness: float, parent2_fitness: float) -> Dict:
        """
        Weighted crossover - genes selected based on parent fitness
        Better parent has higher chance of passing genes
        """
        child_genome = {}
        
        # Calculate selection probability for parent1
        total_fitness = parent1_fitness + parent2_fitness
        if total_fitness > 0:
            p1_prob = parent1_fitness / total_fitness
        else:
            p1_prob = 0.5
        
        for gene in parent1_genome.keys():
            if random.random() < p1_prob:
                child_genome[gene] = parent1_genome[gene]
            else:
                child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
        
        return child_genome
    
    def blend_crossover(self, parent1_genome: Dict, parent2_genome: Dict) -> Dict:
        """
        Blend crossover for numeric genes
        Creates values between and slightly beyond parent values
        """
        child_genome = {}
        
        for gene in parent1_genome.keys():
            val1 = parent1_genome[gene]
            val2 = parent2_genome.get(gene, val1)
            
            # Check if both values are numeric
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # BLX-alpha crossover
                min_val = min(val1, val2)
                max_val = max(val1, val2)
                range_val = max_val - min_val
                
                # Extend range by alpha on both sides
                lower = min_val - self.blend_alpha * range_val
                upper = max_val + self.blend_alpha * range_val
                
                # Generate new value
                new_val = random.uniform(lower, upper)
                
                # Maintain type (int or float)
                if isinstance(val1, int) and isinstance(val2, int):
                    child_genome[gene] = int(round(new_val))
                else:
                    child_genome[gene] = new_val
            else:
                # Non-numeric, use uniform selection
                child_genome[gene] = val1 if random.random() < 0.5 else val2
        
        return child_genome
    
    def multi_point_crossover(self, parent1_genome: Dict, parent2_genome: Dict,
                            num_points: int = 2) -> Dict:
        """
        Multi-point crossover
        Genes are grouped and swapped at crossover points
        """
        child_genome = {}
        genes = list(parent1_genome.keys())
        
        if len(genes) <= num_points:
            # Too few genes, use uniform crossover
            return self.uniform_crossover(parent1_genome, parent2_genome)
        
        # Generate random crossover points
        points = sorted(random.sample(range(1, len(genes)), num_points))
        points = [0] + points + [len(genes)]
        
        # Alternate between parents at each segment
        use_parent1 = True
        
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            for j in range(start, end):
                gene = genes[j]
                if use_parent1:
                    child_genome[gene] = parent1_genome[gene]
                else:
                    child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
            
            use_parent1 = not use_parent1
        
        return child_genome
    
    def segment_crossover(self, parent1_genome: Dict, parent2_genome: Dict,
                         segment_groups: Dict[str, List[str]]) -> Dict:
        """
        Segment-based crossover
        Related genes are grouped and inherited together
        """
        child_genome = {}
        
        # Define default segments for arbitrage strategies
        if not segment_groups:
            segment_groups = {
                'execution': ['execution_delay_blocks', 'gas_multiplier', 'flashloan_provider'],
                'risk': ['min_profit_threshold', 'slippage_tolerance', 'confidence_threshold'],
                'routing': ['dex_pairs', 'route_complexity'],
                'sizing': ['position_size_pct', 'max_gas_price_gwei']
            }
        
        # For each segment, inherit from one parent
        used_genes = set()
        
        for segment_name, genes in segment_groups.items():
            # Choose parent for this segment
            use_parent1 = random.random() < 0.5
            
            for gene in genes:
                if gene in parent1_genome:
                    if use_parent1:
                        child_genome[gene] = parent1_genome[gene]
                    else:
                        child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
                    used_genes.add(gene)
        
        # Handle remaining genes with uniform crossover
        for gene in parent1_genome.keys():
            if gene not in used_genes:
                if random.random() < 0.5:
                    child_genome[gene] = parent1_genome[gene]
                else:
                    child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
        
        return child_genome
    
    def adaptive_crossover(self, parent1_genome: Dict, parent2_genome: Dict,
                         parent1_stats: Dict, parent2_stats: Dict) -> Dict:
        """
        Adaptive crossover that considers parent performance patterns
        Inherits genes based on which parent performs better in specific conditions
        """
        child_genome = {}
        
        # Analyze parent strengths
        p1_strengths = self._analyze_strengths(parent1_stats)
        p2_strengths = self._analyze_strengths(parent2_stats)
        
        for gene in parent1_genome.keys():
            # Determine which parent is likely better for this gene
            gene_category = self._categorize_gene(gene)
            
            if gene_category in p1_strengths and gene_category in p2_strengths:
                # Use parent with better performance in this category
                if p1_strengths[gene_category] > p2_strengths[gene_category]:
                    child_genome[gene] = parent1_genome[gene]
                else:
                    child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
            else:
                # No clear winner, use fitness-weighted selection
                total_fitness = parent1_stats.get('fitness', 0) + parent2_stats.get('fitness', 0)
                if total_fitness > 0:
                    p1_prob = parent1_stats.get('fitness', 0) / total_fitness
                else:
                    p1_prob = 0.5
                
                if random.random() < p1_prob:
                    child_genome[gene] = parent1_genome[gene]
                else:
                    child_genome[gene] = parent2_genome.get(gene, parent1_genome[gene])
        
        return child_genome
    
    def _analyze_strengths(self, stats: Dict) -> Dict[str, float]:
        """Analyze what a parent is good at based on performance stats"""
        strengths = {}
        
        # Gas efficiency
        if 'avg_gas_efficiency' in stats:
            strengths['gas'] = stats['avg_gas_efficiency']
        
        # Profit margins
        if 'avg_profit_margin' in stats:
            strengths['profit'] = stats['avg_profit_margin']
        
        # Risk management
        if 'loss_rate' in stats:
            strengths['risk'] = 1.0 - stats['loss_rate']
        
        # Execution success
        if 'execution_success_rate' in stats:
            strengths['execution'] = stats['execution_success_rate']
        
        return strengths
    
    def _categorize_gene(self, gene: str) -> str:
        """Categorize a gene for adaptive crossover"""
        categories = {
            'gas': ['max_gas_price_gwei', 'gas_multiplier'],
            'profit': ['min_profit_threshold', 'position_size_pct'],
            'risk': ['slippage_tolerance', 'confidence_threshold'],
            'execution': ['execution_delay_blocks', 'flashloan_provider', 'route_complexity']
        }
        
        for category, genes in categories.items():
            if gene in genes:
                return category
        
        return 'general'
    
    def select_crossover_method(self, parent1: Any, parent2: Any) -> str:
        """Select appropriate crossover method based on parent characteristics"""
        # Random selection with configured probabilities
        rand = random.random()
        
        if rand < self.multi_point_prob:
            return 'multi_point'
        elif rand < self.multi_point_prob + self.segment_crossover_prob:
            return 'segment'
        elif hasattr(parent1, 'fitness') and hasattr(parent2, 'fitness'):
            # Use weighted crossover if fitness available
            return 'weighted'
        else:
            return 'uniform'
    
    def perform_crossover(self, parent1: Any, parent2: Any, method: str = None) -> Dict:
        """
        Main crossover function that selects and applies appropriate method
        """
        if method is None:
            method = self.select_crossover_method(parent1, parent2)
        
        logger.debug(f"Performing {method} crossover")
        
        # Extract genomes
        genome1 = parent1.genome if hasattr(parent1, 'genome') else parent1
        genome2 = parent2.genome if hasattr(parent2, 'genome') else parent2
        
        # Apply selected method
        if method == 'uniform':
            return self.uniform_crossover(genome1, genome2)
        
        elif method == 'weighted' and hasattr(parent1, 'fitness'):
            return self.weighted_crossover(
                genome1, genome2,
                parent1.fitness, parent2.fitness
            )
        
        elif method == 'blend':
            return self.blend_crossover(genome1, genome2)
        
        elif method == 'multi_point':
            return self.multi_point_crossover(genome1, genome2)
        
        elif method == 'segment':
            return self.segment_crossover(genome1, genome2, {})
        
        elif method == 'adaptive' and hasattr(parent1, 'stats'):
            return self.adaptive_crossover(
                genome1, genome2,
                parent1.stats, parent2.stats
            )
        
        else:
            # Default to uniform
            return self.uniform_crossover(genome1, genome2)