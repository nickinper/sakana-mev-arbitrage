#!/usr/bin/env python3
"""
Autonomous Agent Brain
Self-directed agents that generate their own questions and explore independently
"""
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

@dataclass
class Question:
    """A question an agent wants to answer"""
    id: str
    question: str
    category: str  # 'price', 'timing', 'correlation', 'pattern', 'anomaly'
    priority: float
    created_at: datetime
    answer: Optional[Any] = None
    
@dataclass
class Hypothesis:
    """A theory the agent has developed"""
    id: str
    statement: str
    confidence: float
    evidence_for: List[Dict]
    evidence_against: List[Dict]
    profit_impact: float
    tested: bool = False

@dataclass 
class Discovery:
    """Something new the agent found"""
    id: str
    description: str
    pattern: Dict
    profitability: float
    reproducibility: float
    timestamp: datetime

class AutonomousAgent:
    """
    Self-directed agent that explores market data autonomously
    Each agent develops its own personality and approach
    """
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.birth_time = datetime.now()
        
        # Personality traits (random for each agent)
        self.curiosity = random.uniform(0.3, 0.9)  # Explore vs exploit
        self.risk_tolerance = random.uniform(0.2, 0.8)  
        self.pattern_sensitivity = random.uniform(0.4, 0.9)
        self.time_preference = random.choice(['fast', 'medium', 'slow'])
        self.focus_area = random.choice(['prices', 'timing', 'correlations', 'anomalies', 'mixed'])
        
        # Learning components
        self.memory = []  # Everything I've observed
        self.questions = []  # What I want to know
        self.hypotheses = []  # My theories
        self.discoveries = []  # Profitable patterns I've found
        self.failed_ideas = []  # What didn't work
        
        # Performance tracking
        self.total_profit = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        self.exploration_count = 0
        
        # Current state
        self.current_focus = None
        self.energy = 100.0  # Depletes with activity
        
    def generate_questions(self, market_state: Dict) -> List[Question]:
        """
        Generate questions based on personality and observations
        Each agent asks different types of questions
        """
        new_questions = []
        
        if self.focus_area in ['prices', 'mixed']:
            # Price-focused questions
            if random.random() < self.curiosity:
                new_questions.append(Question(
                    id=self._generate_id('q'),
                    question=f"What happens to {self._random_token()} price when gas exceeds {random.randint(100, 300)}?",
                    category='price',
                    priority=self.curiosity,
                    created_at=datetime.now()
                ))
            
            if len(self.memory) > 10:
                # Questions based on observations
                new_questions.append(Question(
                    id=self._generate_id('q'),
                    question=f"Why did {self._random_token()} spike when {self._random_token()} dropped?",
                    category='correlation',
                    priority=0.8,
                    created_at=datetime.now()
                ))
        
        if self.focus_area in ['timing', 'mixed']:
            # Timing questions
            new_questions.append(Question(
                id=self._generate_id('q'),
                question=f"Are profits higher in blocks ending in {random.randint(0, 9)}?",
                category='timing',
                priority=self.curiosity * 0.7,
                created_at=datetime.now()
            ))
            
        if self.focus_area in ['anomalies', 'mixed']:
            # Anomaly questions
            if random.random() < self.pattern_sensitivity:
                new_questions.append(Question(
                    id=self._generate_id('q'),
                    question="What patterns exist in failed transactions?",
                    category='anomaly',
                    priority=0.9,
                    created_at=datetime.now()
                ))
        
        # Limit questions based on energy
        self.questions.extend(new_questions[:int(self.energy / 20)])
        return new_questions
    
    def form_hypothesis(self, observations: List[Dict]) -> Optional[Hypothesis]:
        """
        Form theories based on observations and personality
        """
        if len(observations) < 5:
            return None
            
        # Different agents form different types of hypotheses
        if self.focus_area == 'prices':
            # Price pattern hypothesis
            avg_profit = np.mean([o.get('profit', 0) for o in observations])
            if avg_profit > 0:
                hypothesis = Hypothesis(
                    id=self._generate_id('h'),
                    statement=f"Trading {self._random_token()} during high volatility yields {avg_profit:.2f} average profit",
                    confidence=min(len(observations) / 20, 0.9),
                    evidence_for=observations[:5],
                    evidence_against=[],
                    profit_impact=avg_profit
                )
                self.hypotheses.append(hypothesis)
                return hypothesis
                
        elif self.focus_area == 'timing':
            # Time-based hypothesis
            profitable_times = [o for o in observations if o.get('profit', 0) > 50]
            if profitable_times:
                hypothesis = Hypothesis(
                    id=self._generate_id('h'),
                    statement=f"Block times ending in even numbers are {len(profitable_times)/len(observations):.1%} more profitable",
                    confidence=0.6,
                    evidence_for=profitable_times[:3],
                    evidence_against=[],
                    profit_impact=np.mean([p.get('profit', 0) for p in profitable_times])
                )
                self.hypotheses.append(hypothesis)
                return hypothesis
        
        return None
    
    def explore_autonomously(self, market_data: Dict) -> Dict[str, Any]:
        """
        Explore market data in own unique way
        Returns actions to take
        """
        self.exploration_count += 1
        actions = {
            'trades': [],
            'observations': [],
            'new_patterns': []
        }
        
        # Depleet energy
        self.energy = max(0, self.energy - 5)
        
        # Based on personality, explore differently
        if self.curiosity > random.random():
            # Explore new areas
            if self.focus_area == 'prices':
                actions['observations'].append(self._explore_price_patterns(market_data))
            elif self.focus_area == 'timing':
                actions['observations'].append(self._explore_time_patterns(market_data))
            elif self.focus_area == 'anomalies':
                actions['observations'].append(self._explore_anomalies(market_data))
            else:  # mixed
                actions['observations'].append(self._explore_creatively(market_data))
        else:
            # Exploit known patterns
            if self.discoveries:
                best_discovery = max(self.discoveries, key=lambda d: d.profitability)
                actions['trades'].append(self._execute_discovery(best_discovery, market_data))
        
        # Learn from exploration
        self._update_memory(actions['observations'])
        
        # Occasionally form new hypotheses
        if random.random() < 0.2 and len(self.memory) > 10:
            hypothesis = self.form_hypothesis(self.memory[-20:])
            if hypothesis:
                actions['new_patterns'].append(hypothesis)
        
        return actions
    
    def _explore_price_patterns(self, market_data: Dict) -> Dict:
        """Explore price-based patterns"""
        tokens = market_data.get('tokens', ['WETH', 'USDC', 'USDT', 'DAI'])
        token_pair = random.sample(tokens, 2)
        
        observation = {
            'type': 'price_exploration',
            'tokens': token_pair,
            'prices': {t: market_data.get('prices', {}).get(t, 0) for t in token_pair},
            'timestamp': datetime.now(),
            'gas': market_data.get('gas_price', 50),
            'discovery': None
        }
        
        # Look for interesting patterns
        if observation['prices'][token_pair[0]] > 0 and observation['prices'][token_pair[1]] > 0:
            ratio = observation['prices'][token_pair[0]] / observation['prices'][token_pair[1]]
            if ratio > 2000 or ratio < 0.0005:  # Unusual ratio
                observation['discovery'] = f"Unusual price ratio {ratio:.4f} between {token_pair[0]}/{token_pair[1]}"
                
        return observation
    
    def _explore_time_patterns(self, market_data: Dict) -> Dict:
        """Explore time-based patterns"""
        current_block = market_data.get('block_number', 0)
        
        observation = {
            'type': 'time_exploration',
            'block': current_block,
            'block_mod_10': current_block % 10,
            'timestamp': datetime.now(),
            'discovery': None
        }
        
        # Look for time patterns
        if current_block % 10 in [0, 5]:
            observation['discovery'] = f"Block {current_block} ends in {current_block % 10} - checking profitability"
            
        return observation
    
    def _explore_anomalies(self, market_data: Dict) -> Dict:
        """Look for unusual patterns"""
        observation = {
            'type': 'anomaly_exploration',
            'timestamp': datetime.now(),
            'anomalies_found': [],
            'discovery': None
        }
        
        # Check for anomalies
        gas = market_data.get('gas_price', 50)
        if gas > 200:
            observation['anomalies_found'].append('extreme_gas')
            observation['discovery'] = f"Extreme gas price {gas} detected"
            
        # Check for missing liquidity
        for token in market_data.get('tokens', []):
            if market_data.get('liquidity', {}).get(token, 1) < 0.1:
                observation['anomalies_found'].append(f'low_liquidity_{token}')
                
        return observation
    
    def _explore_creatively(self, market_data: Dict) -> Dict:
        """Creative exploration - each agent different"""
        exploration_type = random.choice(['reverse_trades', 'multi_hop', 'gas_correlation', 'random_walk'])
        
        observation = {
            'type': f'creative_{exploration_type}',
            'timestamp': datetime.now(),
            'discovery': None
        }
        
        if exploration_type == 'reverse_trades':
            # Try trading backwards
            observation['approach'] = "What if I trade USDC->WETH->USDC instead?"
            observation['discovery'] = "Testing reverse trading paths"
            
        elif exploration_type == 'multi_hop':
            # Try longer paths
            num_hops = random.randint(3, 5)
            observation['approach'] = f"Testing {num_hops}-hop paths"
            observation['discovery'] = f"Exploring {num_hops}-hop arbitrage"
            
        elif exploration_type == 'gas_correlation':
            # Correlate gas with profit
            gas = market_data.get('gas_price', 50)
            observation['approach'] = f"Does gas price {gas} predict profits?"
            
        else:  # random_walk
            # Completely random exploration
            observation['approach'] = "Random exploration for serendipity"
            
        return observation
    
    def _execute_discovery(self, discovery: Discovery, market_data: Dict) -> Dict:
        """Execute a discovered pattern"""
        trade = {
            'agent_id': self.id,
            'discovery_id': discovery.id,
            'pattern': discovery.pattern,
            'expected_profit': discovery.profitability,
            'timestamp': datetime.now()
        }
        
        # Simulate execution based on personality
        success_chance = discovery.reproducibility * (1 - self.risk_tolerance * 0.2)
        if random.random() < success_chance:
            actual_profit = discovery.profitability * random.uniform(0.8, 1.2)
            self.successful_trades += 1
        else:
            actual_profit = -market_data.get('gas_price', 50)  # Lost gas
            self.failed_trades += 1
            
        trade['actual_profit'] = actual_profit
        trade['success'] = actual_profit > 0
        self.total_profit += actual_profit
        
        return trade
    
    def _update_memory(self, observations: List[Dict]):
        """Update agent memory with observations"""
        self.memory.extend(observations)
        
        # Limit memory size based on personality
        max_memory = 1000 if self.time_preference == 'slow' else 100
        if len(self.memory) > max_memory:
            self.memory = self.memory[-max_memory:]
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        content = f"{prefix}_{self.id}_{datetime.now().isoformat()}_{random.random()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _random_token(self) -> str:
        """Get random token name"""
        return random.choice(['WETH', 'USDC', 'USDT', 'DAI', 'LINK', 'UNI'])
    
    def rest(self):
        """Restore energy"""
        self.energy = min(100, self.energy + 20)
    
    def get_personality_summary(self) -> Dict:
        """Describe agent personality"""
        return {
            'id': self.id,
            'age': (datetime.now() - self.birth_time).total_seconds() / 60,  # minutes
            'personality': {
                'curiosity': self.curiosity,
                'risk_tolerance': self.risk_tolerance,
                'pattern_sensitivity': self.pattern_sensitivity,
                'time_preference': self.time_preference,
                'focus_area': self.focus_area
            },
            'performance': {
                'total_profit': self.total_profit,
                'success_rate': self.successful_trades / max(self.successful_trades + self.failed_trades, 1),
                'discoveries': len(self.discoveries),
                'questions_asked': len(self.questions),
                'hypotheses_formed': len(self.hypotheses)
            }
        }
    
    def share_knowledge(self) -> Dict:
        """Share discoveries with other agents"""
        return {
            'agent_id': self.id,
            'top_discoveries': sorted(self.discoveries, key=lambda d: d.profitability, reverse=True)[:3],
            'proven_hypotheses': [h for h in self.hypotheses if h.confidence > 0.7],
            'personality_type': self.focus_area
        }