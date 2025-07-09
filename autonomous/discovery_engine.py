#!/usr/bin/env python3
"""
Discovery Engine for Open-Ended Exploration
Manages autonomous agents and facilitates emergent discoveries
"""
import asyncio
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from .autonomous_agent import AutonomousAgent, Discovery, Question, Hypothesis

logger = logging.getLogger(__name__)


class MarketEnvironment:
    """Dynamic market environment for agents to explore"""
    
    def __init__(self):
        self.block_number = 18000000
        self.base_prices = {
            'WETH': 2000.0,
            'USDC': 1.0,
            'USDT': 1.0,
            'DAI': 1.0,
            'LINK': 15.0,
            'UNI': 6.0
        }
        self.volatility = 0.02
        self.gas_base = 50
        self.time_factor = 0
        
    def tick(self) -> Dict:
        """Advance market state"""
        self.block_number += 1
        self.time_factor += 0.1
        
        # Dynamic gas based on "network activity"
        gas_price = self.gas_base + 50 * abs(np.sin(self.time_factor * 0.5)) + random.randint(-20, 20)
        
        # Dynamic prices with patterns
        current_prices = {}
        for token, base_price in self.base_prices.items():
            # Add various patterns for agents to discover
            trend = np.sin(self.time_factor * 0.1) * 0.01  # Slow trend
            noise = random.gauss(0, self.volatility)
            
            # Hidden patterns
            if token == 'WETH' and self.block_number % 100 < 10:
                # WETH spikes every 100 blocks for 10 blocks
                spike = 0.05
            else:
                spike = 0
                
            if token == 'LINK' and gas_price > 80:
                # LINK correlates with high gas
                correlation = 0.03
            else:
                correlation = 0
                
            current_prices[token] = base_price * (1 + trend + noise + spike + correlation)
        
        # Create arbitrage opportunities
        dex_prices = self._create_dex_prices(current_prices, gas_price)
        
        return {
            'block_number': self.block_number,
            'timestamp': datetime.now(),
            'gas_price': gas_price,
            'prices': current_prices,
            'dex_prices': dex_prices,
            'liquidity': {token: random.uniform(0.5, 2.0) for token in current_prices},
            'volume': {token: random.uniform(100, 10000) for token in current_prices},
            'tokens': list(current_prices.keys())
        }
    
    def _create_dex_prices(self, spot_prices: Dict, gas_price: float) -> Dict:
        """Create DEX prices with occasional arbitrage opportunities"""
        dexes = ['uniswap_v2', 'uniswap_v3', 'sushiswap', 'curve']
        dex_prices = {}
        
        for dex in dexes:
            dex_prices[dex] = {}
            for token1 in spot_prices:
                for token2 in spot_prices:
                    if token1 != token2:
                        base_rate = spot_prices[token1] / spot_prices[token2]
                        
                        # Add DEX-specific variations
                        if dex == 'uniswap_v2':
                            variation = random.uniform(-0.002, 0.002)
                        elif dex == 'uniswap_v3':
                            variation = random.uniform(-0.001, 0.001)  # More efficient
                        elif dex == 'sushiswap':
                            variation = random.uniform(-0.003, 0.003)
                        else:  # curve
                            # Curve is good for stablecoins
                            if token1 in ['USDC', 'USDT', 'DAI'] and token2 in ['USDC', 'USDT', 'DAI']:
                                variation = random.uniform(-0.0001, 0.0001)
                            else:
                                variation = random.uniform(-0.005, 0.005)
                        
                        # Hidden opportunity: high gas creates more arbitrage
                        if gas_price > 100 and random.random() < 0.1:
                            variation += random.choice([-0.01, 0.01])
                            
                        dex_prices[dex][(token1, token2)] = base_rate * (1 + variation)
        
        return dex_prices


class DiscoveryEngine:
    """
    Manages autonomous agents and their discoveries
    Facilitates emergent behavior and knowledge sharing
    """
    
    def __init__(self, num_agents: int = 20):
        self.agents = []
        self.market = MarketEnvironment()
        self.generation = 0
        self.all_discoveries = []
        self.shared_knowledge = {}
        self.emergent_strategies = []
        
        # Create diverse population of agents
        for i in range(num_agents):
            agent = AutonomousAgent(f"agent_{i}")
            self.agents.append(agent)
            
        logger.info(f"Created {num_agents} autonomous agents with diverse personalities")
        self._log_agent_personalities()
    
    def _log_agent_personalities(self):
        """Log agent personality distribution"""
        focus_areas = {}
        for agent in self.agents:
            focus = agent.focus_area
            focus_areas[focus] = focus_areas.get(focus, 0) + 1
            
        logger.info(f"Agent focus distribution: {focus_areas}")
    
    async def run_discovery_cycle(self) -> Dict:
        """Run one complete discovery cycle"""
        self.generation += 1
        cycle_results = {
            'generation': self.generation,
            'discoveries': [],
            'profitable_trades': [],
            'new_questions': [],
            'emergent_behaviors': [],
            'agent_performance': []
        }
        
        # Get current market state
        market_state = self.market.tick()
        
        # Each agent explores independently
        for agent in self.agents:
            # Agent generates questions
            questions = agent.generate_questions(market_state)
            cycle_results['new_questions'].extend([
                {'agent': agent.id, 'question': q.question} 
                for q in questions
            ])
            
            # Agent explores based on personality
            actions = agent.explore_autonomously(market_state)
            
            # Process discoveries
            for obs in actions['observations']:
                if obs.get('discovery'):
                    discovery = self._process_discovery(agent, obs)
                    if discovery:
                        cycle_results['discoveries'].append({
                            'agent': agent.id,
                            'discovery': discovery.description,
                            'profitability': discovery.profitability
                        })
                        agent.discoveries.append(discovery)
                        self.all_discoveries.append(discovery)
            
            # Process trades
            for trade in actions['trades']:
                if trade['success']:
                    cycle_results['profitable_trades'].append({
                        'agent': agent.id,
                        'profit': trade['actual_profit'],
                        'pattern': trade.get('pattern', {})
                    })
            
            # Check for emergent behaviors
            emergent = self._check_emergent_behavior(agent, actions)
            if emergent:
                cycle_results['emergent_behaviors'].append(emergent)
                self.emergent_strategies.append(emergent)
        
        # Knowledge sharing phase
        if self.generation % 5 == 0:
            self._share_knowledge_between_agents()
        
        # Evolution phase - unsuccessful agents learn from successful ones
        if self.generation % 10 == 0:
            self._evolutionary_learning()
        
        # Collect performance metrics
        for agent in self.agents:
            cycle_results['agent_performance'].append(agent.get_personality_summary())
        
        return cycle_results
    
    def _process_discovery(self, agent: AutonomousAgent, observation: Dict) -> Optional[Discovery]:
        """Process and validate a potential discovery"""
        if not observation.get('discovery'):
            return None
            
        # Calculate profitability based on observation type
        profitability = 0
        reproducibility = 0.5
        
        if observation['type'] == 'price_exploration':
            # Price discoveries
            if 'unusual' in observation['discovery'].lower():
                profitability = random.uniform(50, 200)
                reproducibility = 0.7
                
        elif observation['type'] == 'time_exploration':
            # Time-based discoveries  
            if 'block' in observation['discovery']:
                profitability = random.uniform(30, 150)
                reproducibility = 0.6
                
        elif observation['type'] == 'anomaly_exploration':
            # Anomaly discoveries often most profitable
            if 'extreme' in observation['discovery'].lower():
                profitability = random.uniform(100, 500)
                reproducibility = 0.4  # But less reproducible
                
        elif 'creative' in observation['type']:
            # Creative discoveries - high variance
            profitability = random.uniform(-50, 300)
            reproducibility = random.uniform(0.2, 0.8)
        
        if profitability > 0:
            return Discovery(
                id=f"disc_{agent.id}_{self.generation}",
                description=observation['discovery'],
                pattern=observation,
                profitability=profitability,
                reproducibility=reproducibility,
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_emergent_behavior(self, agent: AutonomousAgent, actions: Dict) -> Optional[Dict]:
        """Check if agent is exhibiting emergent behavior"""
        # Look for unexpected patterns
        
        # Multiple agents discovering same thing independently
        if actions.get('new_patterns'):
            pattern = actions['new_patterns'][0]
            similar_count = sum(
                1 for a in self.agents 
                if any(h.statement[:20] == pattern.statement[:20] for h in a.hypotheses)
            )
            if similar_count >= 3:
                return {
                    'type': 'convergent_discovery',
                    'description': f"Multiple agents independently discovered: {pattern.statement}",
                    'agent_count': similar_count
                }
        
        # Agent doing something completely different
        if agent.focus_area == 'mixed' and len(agent.discoveries) > 5:
            avg_profit = np.mean([d.profitability for d in agent.discoveries])
            if avg_profit > 100:
                return {
                    'type': 'novel_strategy',
                    'description': f"Agent {agent.id} found unique profitable approach",
                    'avg_profit': avg_profit
                }
        
        # Agent asking profound questions
        if len(agent.questions) > 20:
            unique_questions = len(set(q.category for q in agent.questions))
            if unique_questions >= 4:
                return {
                    'type': 'deep_exploration',
                    'description': f"Agent {agent.id} exploring {unique_questions} different dimensions",
                    'question_diversity': unique_questions
                }
        
        return None
    
    def _share_knowledge_between_agents(self):
        """Agents share their best discoveries"""
        logger.info("Knowledge sharing phase initiated")
        
        # Each agent shares top discoveries
        for agent in self.agents:
            knowledge = agent.share_knowledge()
            self.shared_knowledge[agent.id] = knowledge
        
        # Agents learn from each other based on personality
        for agent in self.agents:
            if agent.curiosity > 0.5:  # Curious agents learn more
                # Learn from top performers
                top_agents = sorted(
                    self.agents, 
                    key=lambda a: a.total_profit, 
                    reverse=True
                )[:5]
                
                for top_agent in top_agents:
                    if top_agent.id != agent.id:
                        # Copy some discoveries
                        for discovery in top_agent.discoveries[:2]:
                            if random.random() < agent.curiosity * 0.5:
                                # Agent adapts discovery to their style
                                adapted = Discovery(
                                    id=f"adapted_{discovery.id}_{agent.id}",
                                    description=f"Adapted: {discovery.description}",
                                    pattern=discovery.pattern,
                                    profitability=discovery.profitability * 0.8,  # Slightly less effective
                                    reproducibility=discovery.reproducibility * 0.9,
                                    timestamp=datetime.now()
                                )
                                agent.discoveries.append(adapted)
    
    def _evolutionary_learning(self):
        """Evolutionary pressure - weak agents learn or get replaced"""
        logger.info("Evolutionary learning phase")
        
        # Sort by performance
        sorted_agents = sorted(self.agents, key=lambda a: a.total_profit)
        
        # Bottom 20% learn from top 20%
        bottom_count = len(self.agents) // 5
        top_count = len(self.agents) // 5
        
        bottom_agents = sorted_agents[:bottom_count]
        top_agents = sorted_agents[-top_count:]
        
        for bottom_agent in bottom_agents:
            if random.random() < 0.5:  # 50% chance to learn
                # Learn from a random top agent
                mentor = random.choice(top_agents)
                
                # Adopt some traits
                bottom_agent.curiosity = (bottom_agent.curiosity + mentor.curiosity) / 2
                bottom_agent.risk_tolerance = (bottom_agent.risk_tolerance + mentor.risk_tolerance) / 2
                
                logger.info(f"Agent {bottom_agent.id} learned from {mentor.id}")
            else:
                # Reset agent with new random personality
                bottom_agent.__init__(bottom_agent.id)
                logger.info(f"Agent {bottom_agent.id} reset with new personality")
    
    def get_top_discoveries(self, n: int = 10) -> List[Dict]:
        """Get top discoveries across all agents"""
        sorted_discoveries = sorted(
            self.all_discoveries,
            key=lambda d: d.profitability * d.reproducibility,
            reverse=True
        )[:n]
        
        return [
            {
                'description': d.description,
                'profitability': d.profitability,
                'reproducibility': d.reproducibility,
                'pattern': d.pattern
            }
            for d in sorted_discoveries
        ]
    
    def get_emergent_strategies(self) -> List[Dict]:
        """Get emergent strategies discovered"""
        return self.emergent_strategies[-10:]  # Last 10
    
    def get_agent_diversity_metrics(self) -> Dict:
        """Measure diversity of agent approaches"""
        metrics = {
            'focus_distribution': {},
            'avg_curiosity': np.mean([a.curiosity for a in self.agents]),
            'avg_risk_tolerance': np.mean([a.risk_tolerance for a in self.agents]),
            'unique_discoveries': len(set(d.description for d in self.all_discoveries)),
            'total_questions': sum(len(a.questions) for a in self.agents),
            'profitable_agents': sum(1 for a in self.agents if a.total_profit > 0)
        }
        
        for agent in self.agents:
            focus = agent.focus_area
            metrics['focus_distribution'][focus] = metrics['focus_distribution'].get(focus, 0) + 1
        
        return metrics