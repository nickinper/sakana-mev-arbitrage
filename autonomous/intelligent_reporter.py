#!/usr/bin/env python3
"""
Intelligent Reporter
Translates autonomous agent discoveries into actionable human insights
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from collections import Counter


class IntelligentReporter:
    """
    Analyzes agent behavior and discoveries to provide clear, actionable insights
    """
    
    def __init__(self):
        self.report_history = []
        self.key_insights = []
        self.action_items = []
        
    def generate_report(self, discovery_engine) -> Dict:
        """
        Generate comprehensive report from discovery engine state
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'generation': discovery_engine.generation,
            'executive_summary': self._generate_executive_summary(discovery_engine),
            'surprising_discoveries': self._analyze_surprises(discovery_engine),
            'agent_personalities': self._analyze_personalities(discovery_engine),
            'profitable_patterns': self._extract_profitable_patterns(discovery_engine),
            'failed_experiments': self._analyze_failures(discovery_engine),
            'emergent_behaviors': self._summarize_emergent_behaviors(discovery_engine),
            'recommendations': self._generate_recommendations(discovery_engine),
            'technical_details': self._compile_technical_details(discovery_engine)
        }
        
        self.report_history.append(report)
        return report
    
    def _generate_executive_summary(self, engine) -> str:
        """Create high-level summary for quick understanding"""
        total_profit = sum(agent.total_profit for agent in engine.agents)
        profitable_agents = sum(1 for agent in engine.agents if agent.total_profit > 0)
        top_discovery = engine.get_top_discoveries(1)[0] if engine.all_discoveries else None
        
        summary_parts = []
        
        # Overall performance
        summary_parts.append(
            f"After {engine.generation} generations, {profitable_agents}/{len(engine.agents)} agents "
            f"are profitable with total earnings of ${total_profit:.2f}."
        )
        
        # Key discovery
        if top_discovery:
            summary_parts.append(
                f"Top discovery: '{top_discovery['description']}' yields "
                f"${top_discovery['profitability']:.2f} profit with "
                f"{top_discovery['reproducibility']:.0%} success rate."
            )
        
        # Emergent behavior
        if engine.emergent_strategies:
            summary_parts.append(
                f"{len(engine.emergent_strategies)} emergent behaviors observed, "
                f"including {engine.emergent_strategies[-1]['description']}."
            )
        
        # Diversity insight
        diversity = engine.get_agent_diversity_metrics()
        summary_parts.append(
            f"Agent population shows {diversity['unique_discoveries']} unique discoveries "
            f"with {diversity['total_questions']} questions explored."
        )
        
        return " ".join(summary_parts)
    
    def _analyze_surprises(self, engine) -> List[Dict]:
        """Identify and explain surprising discoveries"""
        surprises = []
        
        # Look for discoveries that don't fit patterns
        for discovery in engine.all_discoveries:
            # High profit + high reproducibility = surprising
            if discovery.profitability > 200 and discovery.reproducibility > 0.7:
                surprises.append({
                    'type': 'highly_profitable_stable',
                    'description': discovery.description,
                    'why_surprising': "Usually high profits come with low reproducibility, but this is both profitable and stable",
                    'profit': discovery.profitability,
                    'reproducibility': discovery.reproducibility
                })
            
            # Creative discoveries
            if 'creative' in discovery.pattern.get('type', ''):
                if discovery.profitability > 100:
                    surprises.append({
                        'type': 'creative_success',
                        'description': discovery.description,
                        'why_surprising': f"Random exploration found profitable pattern: {discovery.pattern.get('approach', 'unknown')}",
                        'profit': discovery.profitability
                    })
            
            # Reverse or unusual patterns
            if 'reverse' in discovery.description.lower() or 'backwards' in discovery.description.lower():
                surprises.append({
                    'type': 'counterintuitive',
                    'description': discovery.description,
                    'why_surprising': "Goes against conventional trading wisdom",
                    'profit': discovery.profitability
                })
        
        # Look for convergent discoveries
        for emergent in engine.emergent_strategies:
            if emergent['type'] == 'convergent_discovery':
                surprises.append({
                    'type': 'convergent',
                    'description': emergent['description'],
                    'why_surprising': f"{emergent['agent_count']} agents independently found same pattern",
                    'implications': "Strong signal this pattern is real and reproducible"
                })
        
        return surprises[:5]  # Top 5 surprises
    
    def _analyze_personalities(self, engine) -> Dict:
        """Analyze how different agent personalities perform"""
        personality_performance = {
            'by_focus_area': {},
            'by_risk_level': {'low': [], 'medium': [], 'high': []},
            'by_curiosity': {'low': [], 'medium': [], 'high': []},
            'insights': []
        }
        
        # Group agents by traits
        for agent in engine.agents:
            summary = agent.get_personality_summary()
            
            # By focus area
            focus = summary['personality']['focus_area']
            if focus not in personality_performance['by_focus_area']:
                personality_performance['by_focus_area'][focus] = []
            personality_performance['by_focus_area'][focus].append(summary['performance']['total_profit'])
            
            # By risk level
            risk = summary['personality']['risk_tolerance']
            if risk < 0.33:
                personality_performance['by_risk_level']['low'].append(summary['performance']['total_profit'])
            elif risk < 0.66:
                personality_performance['by_risk_level']['medium'].append(summary['performance']['total_profit'])
            else:
                personality_performance['by_risk_level']['high'].append(summary['performance']['total_profit'])
            
            # By curiosity
            curiosity = summary['personality']['curiosity']
            if curiosity < 0.4:
                personality_performance['by_curiosity']['low'].append(summary['performance']['total_profit'])
            elif curiosity < 0.7:
                personality_performance['by_curiosity']['medium'].append(summary['performance']['total_profit'])
            else:
                personality_performance['by_curiosity']['high'].append(summary['performance']['total_profit'])
        
        # Generate insights
        # Best performing focus area
        focus_avg = {}
        for focus, profits in personality_performance['by_focus_area'].items():
            if profits:
                focus_avg[focus] = np.mean(profits)
        
        if focus_avg:
            best_focus = max(focus_avg, key=focus_avg.get)
            personality_performance['insights'].append(
                f"Agents focusing on '{best_focus}' are most profitable with "
                f"average profit of ${focus_avg[best_focus]:.2f}"
            )
        
        # Risk analysis
        for level in ['low', 'medium', 'high']:
            profits = personality_performance['by_risk_level'][level]
            if profits:
                avg = np.mean(profits)
                personality_performance['insights'].append(
                    f"{level.capitalize()} risk agents average ${avg:.2f} profit"
                )
        
        # Curiosity correlation
        high_curiosity_avg = np.mean(personality_performance['by_curiosity']['high']) if personality_performance['by_curiosity']['high'] else 0
        low_curiosity_avg = np.mean(personality_performance['by_curiosity']['low']) if personality_performance['by_curiosity']['low'] else 0
        
        if high_curiosity_avg > low_curiosity_avg * 1.5:
            personality_performance['insights'].append(
                "High curiosity agents significantly outperform low curiosity agents"
            )
        
        return personality_performance
    
    def _extract_profitable_patterns(self, engine) -> List[Dict]:
        """Extract and explain the most profitable patterns discovered"""
        top_discoveries = engine.get_top_discoveries(10)
        
        patterns = []
        for disc in top_discoveries:
            pattern_info = {
                'description': disc['description'],
                'profit': disc['profitability'],
                'success_rate': disc['reproducibility'],
                'pattern_type': disc['pattern'].get('type', 'unknown'),
                'actionable_insight': self._make_actionable(disc)
            }
            patterns.append(pattern_info)
        
        return patterns
    
    def _make_actionable(self, discovery: Dict) -> str:
        """Convert discovery into actionable trading advice"""
        pattern = discovery.get('pattern', {})
        pattern_type = pattern.get('type', '')
        
        if 'price' in pattern_type:
            tokens = pattern.get('tokens', ['unknown'])
            return f"Monitor {tokens[0]}/{tokens[1] if len(tokens) > 1 else 'USD'} ratio for unusual spreads"
            
        elif 'time' in pattern_type:
            return f"Execute trades when block number ends in {pattern.get('block_mod_10', 'specific digits')}"
            
        elif 'anomaly' in pattern_type:
            anomalies = pattern.get('anomalies_found', [])
            if 'extreme_gas' in anomalies:
                return "Trade during gas spikes above 200 gwei for higher profits"
            else:
                return "Monitor for unusual market conditions"
                
        elif 'creative' in pattern_type:
            approach = pattern.get('approach', '')
            if 'reverse' in approach:
                return "Try reversed trading paths (e.g., USDC→WETH→USDC)"
            elif 'multi_hop' in approach:
                return "Explore longer trading paths (4-5 hops)"
            else:
                return "Experiment with unconventional strategies"
        
        return "Further investigation needed"
    
    def _analyze_failures(self, engine) -> List[Dict]:
        """Analyze what didn't work and why"""
        failures = []
        
        # Unprofitable agents
        for agent in engine.agents:
            if agent.total_profit < -100:  # Significant losses
                summary = agent.get_personality_summary()
                failures.append({
                    'type': 'unprofitable_agent',
                    'agent_id': agent.id,
                    'loss': agent.total_profit,
                    'personality': summary['personality'],
                    'lesson': self._extract_lesson(agent)
                })
        
        # Failed hypotheses
        for agent in engine.agents:
            for hypothesis in agent.hypotheses:
                if hypothesis.confidence < 0.3 and hypothesis.tested:
                    failures.append({
                        'type': 'failed_hypothesis',
                        'hypothesis': hypothesis.statement,
                        'why_failed': "Low confidence after testing",
                        'lesson': "This pattern doesn't hold in practice"
                    })
        
        return failures[:5]  # Top 5 failures
    
    def _extract_lesson(self, agent) -> str:
        """Extract lesson from failed agent"""
        if agent.risk_tolerance > 0.7:
            return "Excessive risk-taking without proper validation"
        elif agent.curiosity < 0.3:
            return "Insufficient exploration of new opportunities"
        elif agent.focus_area == 'timing' and agent.total_profit < 0:
            return "Time-based patterns may be less reliable than price patterns"
        else:
            return "Strategy needs refinement"
    
    def _summarize_emergent_behaviors(self, engine) -> List[Dict]:
        """Summarize emergent behaviors in human terms"""
        behaviors = []
        
        for emergent in engine.emergent_strategies[-5:]:  # Last 5
            human_explanation = {
                'description': emergent['description'],
                'type': emergent['type'],
                'human_interpretation': self._interpret_emergence(emergent),
                'significance': self._assess_significance(emergent)
            }
            behaviors.append(human_explanation)
        
        return behaviors
    
    def _interpret_emergence(self, emergent: Dict) -> str:
        """Interpret emergent behavior in human terms"""
        if emergent['type'] == 'convergent_discovery':
            return (
                "Multiple agents independently discovered the same profitable pattern. "
                "This strong convergence suggests a real market inefficiency."
            )
        elif emergent['type'] == 'novel_strategy':
            return (
                f"An agent developed a unique approach averaging ${emergent.get('avg_profit', 0):.2f} profit. "
                "This creativity demonstrates the value of diverse exploration."
            )
        elif emergent['type'] == 'deep_exploration':
            return (
                f"An agent is exploring {emergent.get('question_diversity', 'multiple')} different dimensions simultaneously. "
                "This comprehensive approach may uncover complex multi-factor opportunities."
            )
        else:
            return "Unexpected behavior pattern emerged from agent interactions."
    
    def _assess_significance(self, emergent: Dict) -> str:
        """Assess the significance of emergent behavior"""
        if emergent['type'] == 'convergent_discovery':
            return "HIGH - Multiple independent confirmations"
        elif emergent['type'] == 'novel_strategy' and emergent.get('avg_profit', 0) > 100:
            return "HIGH - Profitable and unique"
        elif emergent['type'] == 'deep_exploration':
            return "MEDIUM - Potential for complex discoveries"
        else:
            return "MEDIUM - Worth monitoring"
    
    def _generate_recommendations(self, engine) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on discoveries
        top_discoveries = engine.get_top_discoveries(3)
        if top_discoveries:
            recommendations.append({
                'action': 'IMPLEMENT',
                'description': f"Implement top discovery: {top_discoveries[0]['description']}",
                'expected_profit': f"${top_discoveries[0]['profitability']:.2f} per opportunity",
                'confidence': 'HIGH'
            })
        
        # Based on agent performance
        diversity_metrics = engine.get_agent_diversity_metrics()
        if diversity_metrics['profitable_agents'] < len(engine.agents) * 0.3:
            recommendations.append({
                'action': 'ADJUST',
                'description': "Increase agent diversity - too many agents are unprofitable",
                'suggestion': "Reset bottom 30% of agents with new personalities",
                'confidence': 'MEDIUM'
            })
        
        # Based on emergent behaviors
        if engine.emergent_strategies:
            latest_emergent = engine.emergent_strategies[-1]
            if latest_emergent['type'] == 'convergent_discovery':
                recommendations.append({
                    'action': 'INVESTIGATE',
                    'description': f"Deep dive into convergent discovery: {latest_emergent['description']}",
                    'rationale': "Multiple agents found this independently",
                    'confidence': 'HIGH'
                })
        
        # Based on personality analysis
        personality_perf = self._analyze_personalities(engine)
        if personality_perf['insights']:
            recommendations.append({
                'action': 'OPTIMIZE',
                'description': f"Optimize agent mix: {personality_perf['insights'][0]}",
                'suggestion': "Create more agents with successful personality traits",
                'confidence': 'MEDIUM'
            })
        
        return recommendations
    
    def _compile_technical_details(self, engine) -> Dict:
        """Compile technical details for deeper analysis"""
        return {
            'generation': engine.generation,
            'total_agents': len(engine.agents),
            'total_discoveries': len(engine.all_discoveries),
            'diversity_metrics': engine.get_agent_diversity_metrics(),
            'top_performers': [
                {
                    'id': agent.id,
                    'profit': agent.total_profit,
                    'personality': agent.get_personality_summary()['personality']
                }
                for agent in sorted(engine.agents, key=lambda a: a.total_profit, reverse=True)[:5]
            ]
        }
    
    def format_report_for_display(self, report: Dict) -> str:
        """Format report for easy reading"""
        formatted = []
        
        formatted.append("=" * 60)
        formatted.append("AUTONOMOUS MEV DISCOVERY REPORT")
        formatted.append("=" * 60)
        formatted.append(f"\nGenerated: {report['timestamp']}")
        formatted.append(f"Generation: {report['generation']}")
        
        formatted.append("\n🎯 EXECUTIVE SUMMARY")
        formatted.append("-" * 40)
        formatted.append(report['executive_summary'])
        
        formatted.append("\n🔍 SURPRISING DISCOVERIES")
        formatted.append("-" * 40)
        for i, surprise in enumerate(report['surprising_discoveries'], 1):
            formatted.append(f"\n{i}. {surprise['description']}")
            formatted.append(f"   Why surprising: {surprise['why_surprising']}")
            if 'profit' in surprise:
                formatted.append(f"   Profit: ${surprise['profit']:.2f}")
        
        formatted.append("\n💰 TOP PROFITABLE PATTERNS")
        formatted.append("-" * 40)
        for i, pattern in enumerate(report['profitable_patterns'][:5], 1):
            formatted.append(f"\n{i}. {pattern['description']}")
            formatted.append(f"   Profit: ${pattern['profit']:.2f} | Success Rate: {pattern['success_rate']:.0%}")
            formatted.append(f"   Action: {pattern['actionable_insight']}")
        
        formatted.append("\n🧬 EMERGENT BEHAVIORS")
        formatted.append("-" * 40)
        for behavior in report['emergent_behaviors']:
            formatted.append(f"\n• {behavior['description']}")
            formatted.append(f"  Interpretation: {behavior['human_interpretation']}")
            formatted.append(f"  Significance: {behavior['significance']}")
        
        formatted.append("\n📊 RECOMMENDATIONS")
        formatted.append("-" * 40)
        for rec in report['recommendations']:
            formatted.append(f"\n[{rec['action']}] {rec['description']}")
            if 'suggestion' in rec:
                formatted.append(f"  → {rec['suggestion']}")
            formatted.append(f"  Confidence: {rec['confidence']}")
        
        formatted.append("\n" + "=" * 60)
        
        return "\n".join(formatted)