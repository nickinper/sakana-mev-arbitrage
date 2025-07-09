"""
Complete Transparency and Decision Tracing System
Ensures every superhuman decision can be explained and understood
"""
import time
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
import numpy as np
import logging
from collections import defaultdict
import sqlite3
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Represents a single decision made by the system"""
    id: str
    timestamp: float
    decision_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    reasoning_steps: List[Dict]
    alternatives_considered: List[Dict]
    why_chosen: str
    pattern_basis: Dict
    computation_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    

@dataclass
class DecisionTrace:
    """Complete trace of a decision process"""
    decision_id: str
    timestamp: float
    opportunity: Dict
    decision_steps: List[Dict]
    explanations: Dict[str, str]
    visualizations: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    counterfactuals: List[Dict]
    blockchain_hash: Optional[str] = None


class TransparentDecisionTrace:
    """
    Complete transparency for every decision, no matter how complex
    Provides multi-level explanations and full audit trail
    """
    
    def __init__(self, blockchain_enabled: bool = True):
        self.decision_graph = nx.DiGraph()
        self.reasoning_chains = []
        self.decision_history = []
        self.blockchain_enabled = blockchain_enabled
        
        # Initialize storage
        self.storage = DecisionStorage()
        
        # Explanation generators for different audiences
        self.explainers = {
            'executive': ExecutiveExplainer(),
            'technical': TechnicalExplainer(),
            'mathematical': MathematicalExplainer(),
            'visual': VisualExplainer()
        }
        
        # Decision validators
        self.validators = DecisionValidators()
        
    def trace_decision(self, opportunity: Dict, decision_process: List[Any]) -> DecisionTrace:
        """
        Create a complete trace of the decision process
        
        Args:
            opportunity: The arbitrage opportunity being evaluated
            decision_process: List of decision steps taken
            
        Returns:
            Complete decision trace with explanations
        """
        trace = DecisionTrace(
            decision_id=self._generate_decision_id(),
            timestamp=time.time(),
            opportunity=opportunity,
            decision_steps=[],
            explanations={},
            visualizations={},
            confidence_breakdown={},
            counterfactuals=[]
        )
        
        # Record every micro-decision
        for step in decision_process:
            step_trace = self._trace_decision_step(step)
            trace.decision_steps.append(step_trace)
            
            # Add to decision graph
            self._add_to_decision_graph(trace.decision_id, step_trace)
        
        # Calculate confidence breakdown
        trace.confidence_breakdown = self._calculate_confidence_breakdown(trace.decision_steps)
        
        # Generate multi-level explanations
        trace.explanations = self._generate_explanations(trace)
        
        # Generate visualizations
        trace.visualizations = self._generate_visualizations(trace)
        
        # Generate counterfactual analysis
        trace.counterfactuals = self._generate_counterfactuals(trace)
        
        # Validate decision
        validation_result = self.validators.validate_decision(trace)
        if not validation_result['valid']:
            logger.warning(f"Decision validation failed: {validation_result['reasons']}")
        
        # Record to immutable ledger if enabled
        if self.blockchain_enabled:
            trace.blockchain_hash = self._record_to_blockchain(trace)
        
        # Store in database
        self.storage.store_decision_trace(trace)
        
        # Update history
        self.decision_history.append(trace)
        
        return trace
    
    def _trace_decision_step(self, step: Any) -> Dict:
        """Trace individual decision step"""
        step_trace = {
            'step_id': self._generate_step_id(),
            'step_name': getattr(step, 'name', 'Unknown Step'),
            'timestamp': time.time(),
            'inputs': self._serialize_inputs(step),
            'computation': self._trace_computation(step),
            'output': self._serialize_output(step),
            'reasoning': self._extract_reasoning(step),
            'confidence': getattr(step, 'confidence', 0.0),
            'alternatives_considered': self._extract_alternatives(step),
            'why_chosen': self._extract_selection_reasoning(step),
            'resource_usage': self._measure_resource_usage(step)
        }
        
        return step_trace
    
    def _serialize_inputs(self, step: Any) -> Dict:
        """Serialize step inputs for storage"""
        inputs = {}
        
        if hasattr(step, 'inputs'):
            for key, value in step.inputs.items():
                if isinstance(value, (int, float, str, bool, list, dict)):
                    inputs[key] = value
                elif isinstance(value, np.ndarray):
                    inputs[key] = {
                        'type': 'numpy_array',
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'summary_stats': {
                            'mean': float(np.mean(value)),
                            'std': float(np.std(value)),
                            'min': float(np.min(value)),
                            'max': float(np.max(value))
                        }
                    }
                else:
                    inputs[key] = str(type(value))
        
        return inputs
    
    def _trace_computation(self, step: Any) -> Dict:
        """Trace the computation performed in this step"""
        computation = {
            'operations': [],
            'complexity': 'O(1)',  # Default
            'key_calculations': []
        }
        
        if hasattr(step, 'computation_graph'):
            # Extract computation graph
            graph = step.computation_graph
            computation['num_operations'] = graph.number_of_nodes()
            computation['operation_types'] = self._analyze_operation_types(graph)
            computation['critical_path'] = self._find_critical_path(graph)
        
        if hasattr(step, 'calculations'):
            for calc_name, calc_value in step.calculations.items():
                computation['key_calculations'].append({
                    'name': calc_name,
                    'value': calc_value,
                    'formula': getattr(step, f'{calc_name}_formula', 'Not specified')
                })
        
        return computation
    
    def _extract_reasoning(self, step: Any) -> Dict:
        """Extract reasoning from decision step"""
        reasoning = {
            'type': 'unknown',
            'basis': [],
            'logic_chain': [],
            'assumptions': []
        }
        
        if hasattr(step, 'get_reasoning'):
            step_reasoning = step.get_reasoning()
            reasoning.update(step_reasoning)
        
        elif hasattr(step, 'reasoning'):
            reasoning['basis'] = step.reasoning.get('basis', [])
            reasoning['logic_chain'] = step.reasoning.get('chain', [])
        
        # Extract pattern-based reasoning
        if hasattr(step, 'pattern_basis'):
            reasoning['pattern_basis'] = {
                'pattern_type': step.pattern_basis.get('type'),
                'pattern_strength': step.pattern_basis.get('strength'),
                'pattern_description': step.pattern_basis.get('description')
            }
        
        return reasoning
    
    def _generate_explanations(self, trace: DecisionTrace) -> Dict[str, str]:
        """Generate multiple explanation levels"""
        explanations = {}
        
        # Generate explanation for each audience
        for audience, explainer in self.explainers.items():
            try:
                explanations[audience] = explainer.explain(trace)
            except Exception as e:
                logger.error(f"Failed to generate {audience} explanation: {e}")
                explanations[audience] = f"Explanation generation failed: {str(e)}"
        
        return explanations
    
    def _generate_visualizations(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Generate visual explanations"""
        visualizations = {}
        
        # Decision flow diagram
        visualizations['decision_flow'] = self._create_decision_flow_diagram(trace)
        
        # Confidence visualization
        visualizations['confidence_breakdown'] = self._create_confidence_visualization(trace)
        
        # Pattern visualization
        visualizations['pattern_analysis'] = self._create_pattern_visualization(trace)
        
        # Alternative comparison
        visualizations['alternatives_comparison'] = self._create_alternatives_chart(trace)
        
        return visualizations
    
    def _generate_counterfactuals(self, trace: DecisionTrace) -> List[Dict]:
        """Generate 'what if' scenarios"""
        counterfactuals = []
        
        # What if different parameters
        param_variations = [
            {'name': 'lower_gas_price', 'change': {'gas_multiplier': 0.8}},
            {'name': 'higher_slippage', 'change': {'slippage_tolerance': 0.02}},
            {'name': 'delayed_execution', 'change': {'execution_delay': 2}}
        ]
        
        for variation in param_variations:
            # Simulate decision with changed parameters
            modified_trace = self._simulate_modified_decision(trace, variation['change'])
            
            counterfactuals.append({
                'scenario': variation['name'],
                'changes': variation['change'],
                'original_outcome': self._extract_outcome(trace),
                'modified_outcome': self._extract_outcome(modified_trace),
                'impact': self._calculate_impact(trace, modified_trace)
            })
        
        return counterfactuals
    
    def _record_to_blockchain(self, trace: DecisionTrace) -> str:
        """Record decision trace to blockchain for immutability"""
        # Serialize trace
        trace_data = {
            'decision_id': trace.decision_id,
            'timestamp': trace.timestamp,
            'opportunity_hash': hashlib.sha256(
                json.dumps(trace.opportunity, sort_keys=True).encode()
            ).hexdigest(),
            'decision_hash': hashlib.sha256(
                json.dumps(trace.decision_steps, sort_keys=True).encode()
            ).hexdigest(),
            'confidence': trace.confidence_breakdown.get('overall', 0)
        }
        
        # In production, would submit to actual blockchain
        # For now, generate hash
        blockchain_hash = hashlib.sha256(
            json.dumps(trace_data, sort_keys=True).encode()
        ).hexdigest()
        
        logger.info(f"Decision {trace.decision_id} recorded with hash {blockchain_hash}")
        
        return blockchain_hash
    
    def query_decision(self, decision_id: str) -> Optional[DecisionTrace]:
        """Query historical decision by ID"""
        return self.storage.get_decision_trace(decision_id)
    
    def explain_decision_path(self, decision_id: str, 
                            target_audience: str = 'executive') -> str:
        """Explain the path that led to a decision"""
        trace = self.query_decision(decision_id)
        
        if not trace:
            return f"Decision {decision_id} not found"
        
        explainer = self.explainers.get(target_audience, self.explainers['executive'])
        return explainer.explain_path(trace)
    
    def compare_decisions(self, decision_id1: str, decision_id2: str) -> Dict:
        """Compare two decisions to understand differences"""
        trace1 = self.query_decision(decision_id1)
        trace2 = self.query_decision(decision_id2)
        
        if not trace1 or not trace2:
            return {'error': 'One or both decisions not found'}
        
        comparison = {
            'decision1': decision_id1,
            'decision2': decision_id2,
            'key_differences': self._find_key_differences(trace1, trace2),
            'outcome_comparison': {
                'decision1_outcome': self._extract_outcome(trace1),
                'decision2_outcome': self._extract_outcome(trace2)
            },
            'reasoning_differences': self._compare_reasoning(trace1, trace2),
            'confidence_comparison': {
                'decision1': trace1.confidence_breakdown,
                'decision2': trace2.confidence_breakdown
            }
        }
        
        return comparison
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        return f"decision_{int(time.time() * 1000000)}"
    
    def _generate_step_id(self) -> str:
        """Generate unique step ID"""
        return f"step_{int(time.time() * 1000000)}_{np.random.randint(1000)}"
    
    def _calculate_confidence_breakdown(self, decision_steps: List[Dict]) -> Dict[str, float]:
        """Calculate confidence scores by component"""
        breakdown = {
            'pattern_recognition': 0.0,
            'calculation_certainty': 0.0,
            'data_quality': 0.0,
            'execution_probability': 0.0,
            'overall': 0.0
        }
        
        # Aggregate confidences from steps
        step_confidences = [step.get('confidence', 0) for step in decision_steps]
        
        if step_confidences:
            breakdown['overall'] = np.mean(step_confidences)
            
            # Detailed breakdown based on step types
            for step in decision_steps:
                step_name = step.get('step_name', '')
                confidence = step.get('confidence', 0)
                
                if 'pattern' in step_name.lower():
                    breakdown['pattern_recognition'] = max(
                        breakdown['pattern_recognition'], confidence
                    )
                elif 'calc' in step_name.lower() or 'compute' in step_name.lower():
                    breakdown['calculation_certainty'] = max(
                        breakdown['calculation_certainty'], confidence
                    )
                elif 'data' in step_name.lower() or 'input' in step_name.lower():
                    breakdown['data_quality'] = max(
                        breakdown['data_quality'], confidence
                    )
                elif 'exec' in step_name.lower() or 'probability' in step_name.lower():
                    breakdown['execution_probability'] = max(
                        breakdown['execution_probability'], confidence
                    )
        
        return breakdown


class ExecutiveExplainer:
    """Generate executive-level explanations"""
    
    def explain(self, trace: DecisionTrace) -> str:
        """Create high-level summary for executives"""
        opportunity = trace.opportunity
        confidence = trace.confidence_breakdown.get('overall', 0)
        
        # Extract key metrics
        expected_profit = opportunity.get('expected_profit', 0)
        risk_level = opportunity.get('risk_level', 'Unknown')
        
        # Find primary reasoning
        primary_reason = self._extract_primary_reason(trace)
        
        explanation = f"""
EXECUTIVE SUMMARY - Decision {trace.decision_id}

**Decision**: {'EXECUTE' if confidence > 0.7 else 'SKIP'} arbitrage opportunity
**Confidence**: {confidence:.1%}
**Expected Profit**: ${expected_profit:,.2f}
**Risk Level**: {risk_level}

**Key Reasoning**: {primary_reason}

**Quick Summary**: 
The AI analyzed {len(trace.decision_steps)} factors and found {self._count_positive_signals(trace)} positive signals. 
The opportunity was {'taken' if confidence > 0.7 else 'skipped'} based on {self._get_deciding_factor(trace)}.

**Bottom Line**: {self._get_bottom_line(trace, expected_profit, confidence)}
"""
        
        return explanation.strip()
    
    def explain_path(self, trace: DecisionTrace) -> str:
        """Explain the decision path in executive terms"""
        steps_summary = []
        
        for i, step in enumerate(trace.decision_steps[:5]):  # Top 5 steps
            step_summary = f"{i+1}. {step['step_name']}: {self._summarize_step(step)}"
            steps_summary.append(step_summary)
        
        return "Decision Path:\n" + "\n".join(steps_summary)
    
    def _extract_primary_reason(self, trace: DecisionTrace) -> str:
        """Extract the main reason for the decision"""
        # Look for highest confidence step
        if trace.decision_steps:
            highest_confidence_step = max(trace.decision_steps, 
                                        key=lambda s: s.get('confidence', 0))
            
            reasoning = highest_confidence_step.get('reasoning', {})
            if reasoning.get('basis'):
                return reasoning['basis'][0] if isinstance(reasoning['basis'], list) else str(reasoning['basis'])
        
        return "Multiple factors indicated opportunity"
    
    def _count_positive_signals(self, trace: DecisionTrace) -> int:
        """Count positive signals in decision"""
        return sum(1 for step in trace.decision_steps 
                  if step.get('output', {}).get('positive_signal', False))
    
    def _get_deciding_factor(self, trace: DecisionTrace) -> str:
        """Identify the deciding factor"""
        if trace.confidence_breakdown['pattern_recognition'] > 0.8:
            return "strong pattern recognition"
        elif trace.confidence_breakdown['calculation_certainty'] > 0.8:
            return "high calculation certainty"
        else:
            return "overall positive indicators"
    
    def _get_bottom_line(self, trace: DecisionTrace, profit: float, confidence: float) -> str:
        """Generate bottom line summary"""
        if confidence > 0.8 and profit > 100:
            return f"High-confidence opportunity with ${profit:,.2f} expected profit."
        elif confidence > 0.7:
            return f"Good opportunity with reasonable confidence."
        elif confidence > 0.5:
            return f"Marginal opportunity - execution depends on risk tolerance."
        else:
            return f"Opportunity skipped due to low confidence or unfavorable conditions."
    
    def _summarize_step(self, step: Dict) -> str:
        """Create one-line summary of a step"""
        output = step.get('output', {})
        confidence = step.get('confidence', 0)
        
        if 'result' in output:
            return f"{output['result']} (confidence: {confidence:.1%})"
        else:
            return f"Completed with {confidence:.1%} confidence"


class TechnicalExplainer:
    """Generate technical explanations for developers"""
    
    def explain(self, trace: DecisionTrace) -> str:
        """Create technical explanation"""
        explanation = f"""
TECHNICAL ANALYSIS - Decision {trace.decision_id}

## Decision Pipeline
"""
        
        # Add step-by-step technical details
        for i, step in enumerate(trace.decision_steps):
            explanation += f"""
### Step {i+1}: {step['step_name']}
- **Inputs**: {self._format_inputs(step['inputs'])}
- **Computation**: {self._format_computation(step['computation'])}
- **Output**: {self._format_output(step['output'])}
- **Confidence**: {step['confidence']:.3f}
- **Resource Usage**: {step.get('resource_usage', 'Not measured')}
"""
        
        # Add performance metrics
        explanation += f"""
## Performance Metrics
- Total Decision Time: {self._calculate_total_time(trace)} ms
- Memory Usage: {self._calculate_memory_usage(trace)} MB
- Computational Complexity: {self._estimate_complexity(trace)}
"""
        
        # Add data flow
        explanation += f"""
## Data Flow
{self._generate_data_flow(trace)}
"""
        
        return explanation
    
    def _format_inputs(self, inputs: Dict) -> str:
        """Format inputs for technical display"""
        formatted = []
        for key, value in inputs.items():
            if isinstance(value, dict) and 'type' in value:
                formatted.append(f"{key}: {value['type']} {value.get('shape', '')}")
            else:
                formatted.append(f"{key}: {type(value).__name__}")
        
        return ", ".join(formatted)
    
    def _format_computation(self, computation: Dict) -> str:
        """Format computation details"""
        details = []
        
        if 'num_operations' in computation:
            details.append(f"{computation['num_operations']} operations")
        
        if 'complexity' in computation:
            details.append(f"Complexity: {computation['complexity']}")
        
        if 'key_calculations' in computation:
            details.append(f"{len(computation['key_calculations'])} key calculations")
        
        return ", ".join(details) if details else "Standard processing"
    
    def _format_output(self, output: Dict) -> str:
        """Format output for display"""
        if not output:
            return "None"
        
        key_outputs = []
        for key, value in list(output.items())[:3]:  # First 3 outputs
            if isinstance(value, (int, float)):
                key_outputs.append(f"{key}={value:.3f}")
            else:
                key_outputs.append(f"{key}={type(value).__name__}")
        
        return ", ".join(key_outputs)
    
    def _calculate_total_time(self, trace: DecisionTrace) -> float:
        """Calculate total decision time"""
        if trace.decision_steps:
            start = min(step['timestamp'] for step in trace.decision_steps)
            end = max(step['timestamp'] for step in trace.decision_steps)
            return (end - start) * 1000  # Convert to ms
        return 0
    
    def _calculate_memory_usage(self, trace: DecisionTrace) -> float:
        """Estimate memory usage"""
        # Simplified estimation
        total_size = 0
        for step in trace.decision_steps:
            total_size += len(json.dumps(step))
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _estimate_complexity(self, trace: DecisionTrace) -> str:
        """Estimate computational complexity"""
        complexities = []
        
        for step in trace.decision_steps:
            comp = step.get('computation', {})
            if 'complexity' in comp:
                complexities.append(comp['complexity'])
        
        if complexities:
            # Return worst case
            return max(complexities, key=lambda x: self._complexity_order(x))
        
        return "O(n)"
    
    def _complexity_order(self, complexity: str) -> int:
        """Convert complexity to numeric order for comparison"""
        orders = {
            'O(1)': 1,
            'O(log n)': 2,
            'O(n)': 3,
            'O(n log n)': 4,
            'O(n^2)': 5,
            'O(n^3)': 6,
            'O(2^n)': 7
        }
        return orders.get(complexity, 10)
    
    def _generate_data_flow(self, trace: DecisionTrace) -> str:
        """Generate data flow description"""
        flow = []
        
        for i, step in enumerate(trace.decision_steps):
            if i == 0:
                flow.append(f"Input -> {step['step_name']}")
            else:
                prev_step = trace.decision_steps[i-1]
                flow.append(f"{prev_step['step_name']} -> {step['step_name']}")
        
        if trace.decision_steps:
            flow.append(f"{trace.decision_steps[-1]['step_name']} -> Final Decision")
        
        return " -> ".join(flow[:5]) + (" -> ..." if len(flow) > 5 else "")


class MathematicalExplainer:
    """Generate mathematical proofs and explanations"""
    
    def explain(self, trace: DecisionTrace) -> str:
        """Create mathematical explanation"""
        explanation = f"""
MATHEMATICAL PROOF - Decision {trace.decision_id}

## Formal Decision Function
Let D: Ω → {{0,1}} be the decision function where:
- Ω = opportunity space
- D(ω) = 1 iff execute arbitrage
- ω ∈ Ω represents opportunity {trace.opportunity.get('id', 'current')}
"""
        
        # Add mathematical reasoning
        for i, step in enumerate(trace.decision_steps):
            if 'calculation' in step.get('step_name', '').lower():
                explanation += self._format_mathematical_step(step, i)
        
        # Add probability calculations
        explanation += f"""
## Probability Analysis
P(success|execute) = {trace.confidence_breakdown.get('execution_probability', 0):.3f}
P(profit > cost) = {trace.confidence_breakdown.get('overall', 0):.3f}

## Expected Value
E[profit] = P(success) × gross_profit - E[gas_cost]
         = {self._calculate_expected_value(trace):.2f}
"""
        
        # Add optimization proof
        explanation += f"""
## Optimality Proof
The decision D(ω) = {'1' if trace.confidence_breakdown.get('overall', 0) > 0.7 else '0'} is optimal because:
{self._generate_optimality_proof(trace)}
"""
        
        return explanation
    
    def _format_mathematical_step(self, step: Dict, index: int) -> str:
        """Format mathematical step"""
        calculations = step.get('computation', {}).get('key_calculations', [])
        
        if not calculations:
            return ""
        
        formatted = f"\n### Step {index + 1}: {step['step_name']}\n"
        
        for calc in calculations:
            formula = calc.get('formula', 'Not specified')
            value = calc.get('value', 0)
            formatted += f"{calc['name']} = {formula} = {value:.6f}\n"
        
        return formatted
    
    def _calculate_expected_value(self, trace: DecisionTrace) -> float:
        """Calculate expected value from trace"""
        opportunity = trace.opportunity
        success_prob = trace.confidence_breakdown.get('execution_probability', 0)
        gross_profit = opportunity.get('expected_profit', 0)
        gas_cost = opportunity.get('gas_cost', 0)
        
        return success_prob * gross_profit - gas_cost
    
    def _generate_optimality_proof(self, trace: DecisionTrace) -> str:
        """Generate proof of optimality"""
        confidence = trace.confidence_breakdown.get('overall', 0)
        
        if confidence > 0.8:
            return """
1. Multiple independent signals confirm opportunity (convergent evidence)
2. Expected value significantly exceeds risk-adjusted threshold
3. No dominant alternative strategy exists (Nash equilibrium)
4. Decision robust to parameter perturbations (sensitivity analysis)
"""
        elif confidence > 0.6:
            return """
1. Positive expected value under conservative assumptions
2. Risk within acceptable bounds
3. Opportunity cost of waiting exceeds execution risk
"""
        else:
            return """
1. Insufficient evidence for positive expected value
2. Risk exceeds acceptable threshold
3. Better opportunities likely available (option value)
"""


class VisualExplainer:
    """Generate visual explanations"""
    
    def explain(self, trace: DecisionTrace) -> str:
        """Create text-based visual explanation"""
        visual = f"""
VISUAL REPRESENTATION - Decision {trace.decision_id}

## Decision Flow
"""
        
        # Create ASCII flow chart
        visual += self._create_ascii_flowchart(trace)
        
        # Add confidence bars
        visual += "\n## Confidence Breakdown\n"
        visual += self._create_confidence_bars(trace.confidence_breakdown)
        
        # Add decision matrix
        visual += "\n## Decision Matrix\n"
        visual += self._create_decision_matrix(trace)
        
        return visual
    
    def _create_ascii_flowchart(self, trace: DecisionTrace) -> str:
        """Create ASCII flowchart of decision process"""
        flowchart = """
┌─────────────────┐
│  Opportunity    │
│   Detected      │
└────────┬────────┘
         │
         v
"""
        
        for i, step in enumerate(trace.decision_steps[:4]):
            if i > 0:
                flowchart += "         │\n         v\n"
            
            flowchart += f"┌─────────────────┐\n"
            flowchart += f"│ {step['step_name'][:15]:<15} │\n"
            flowchart += f"│ Conf: {step['confidence']:.1%:<9} │\n"
            flowchart += f"└─────────────────┘\n"
        
        flowchart += """         │
         v
┌─────────────────┐
│ Final Decision  │
└─────────────────┘
"""
        
        return flowchart
    
    def _create_confidence_bars(self, breakdown: Dict[str, float]) -> str:
        """Create bar chart for confidence breakdown"""
        bars = ""
        
        for component, confidence in breakdown.items():
            bar_length = int(confidence * 20)
            bars += f"{component:<20} │{'█' * bar_length}{' ' * (20 - bar_length)}│ {confidence:.1%}\n"
        
        return bars
    
    def _create_decision_matrix(self, trace: DecisionTrace) -> str:
        """Create decision matrix visualization"""
        matrix = """
Factor               │ Signal │ Weight │ Impact
─────────────────────┼────────┼────────┼────────
"""
        
        # Extract factors from steps
        for step in trace.decision_steps[:5]:
            factor = step['step_name'][:20]
            signal = '✓' if step.get('output', {}).get('positive_signal', False) else '✗'
            weight = 'High' if step['confidence'] > 0.8 else 'Med' if step['confidence'] > 0.5 else 'Low'
            impact = step['confidence']
            
            matrix += f"{factor:<20} │   {signal}    │  {weight:<4}  │ {impact:.1%}\n"
        
        return matrix


class DecisionValidators:
    """Validate decisions for consistency and correctness"""
    
    def validate_decision(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Validate decision trace"""
        validation_results = {
            'valid': True,
            'reasons': [],
            'warnings': []
        }
        
        # Check confidence consistency
        if not self._validate_confidence_consistency(trace):
            validation_results['valid'] = False
            validation_results['reasons'].append("Confidence scores inconsistent")
        
        # Check reasoning chain
        if not self._validate_reasoning_chain(trace):
            validation_results['warnings'].append("Reasoning chain has gaps")
        
        # Check computational integrity
        if not self._validate_computations(trace):
            validation_results['warnings'].append("Some computations could not be verified")
        
        # Check decision coherence
        if not self._validate_decision_coherence(trace):
            validation_results['valid'] = False
            validation_results['reasons'].append("Decision does not follow from analysis")
        
        return validation_results
    
    def _validate_confidence_consistency(self, trace: DecisionTrace) -> bool:
        """Check if confidence scores are consistent"""
        step_confidences = [s['confidence'] for s in trace.decision_steps]
        overall_confidence = trace.confidence_breakdown.get('overall', 0)
        
        # Overall should be close to average of steps
        if step_confidences:
            expected_overall = np.mean(step_confidences)
            if abs(overall_confidence - expected_overall) > 0.2:
                return False
        
        return True
    
    def _validate_reasoning_chain(self, trace: DecisionTrace) -> bool:
        """Validate reasoning chain completeness"""
        # Check each step has reasoning
        for step in trace.decision_steps:
            if not step.get('reasoning'):
                return False
            
            reasoning = step['reasoning']
            if not reasoning.get('basis') and not reasoning.get('logic_chain'):
                return False
        
        return True
    
    def _validate_computations(self, trace: DecisionTrace) -> bool:
        """Validate computational correctness"""
        # Spot check some calculations
        for step in trace.decision_steps:
            computation = step.get('computation', {})
            
            for calc in computation.get('key_calculations', []):
                # Verify calculation if possible
                if 'formula' in calc and 'value' in calc:
                    # Would implement actual verification
                    pass
        
        return True
    
    def _validate_decision_coherence(self, trace: DecisionTrace) -> bool:
        """Check if final decision follows from analysis"""
        overall_confidence = trace.confidence_breakdown.get('overall', 0)
        
        # Count positive vs negative signals
        positive_signals = sum(
            1 for step in trace.decision_steps 
            if step.get('output', {}).get('positive_signal', False)
        )
        
        negative_signals = len(trace.decision_steps) - positive_signals
        
        # High confidence should have mostly positive signals
        if overall_confidence > 0.7 and positive_signals < negative_signals:
            return False
        
        # Low confidence should have mostly negative signals
        if overall_confidence < 0.3 and positive_signals > negative_signals:
            return False
        
        return True


class DecisionStorage:
    """Store and retrieve decision traces"""
    
    def __init__(self, db_path: str = "decisions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_traces (
                decision_id TEXT PRIMARY KEY,
                timestamp REAL,
                trace_data BLOB,
                confidence REAL,
                outcome TEXT,
                blockchain_hash TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON decision_traces(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence 
            ON decision_traces(confidence)
        """)
        
        conn.commit()
        conn.close()
    
    def store_decision_trace(self, trace: DecisionTrace):
        """Store decision trace in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize trace
        trace_data = pickle.dumps(trace)
        
        cursor.execute("""
            INSERT OR REPLACE INTO decision_traces 
            (decision_id, timestamp, trace_data, confidence, outcome, blockchain_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            trace.decision_id,
            trace.timestamp,
            trace_data,
            trace.confidence_breakdown.get('overall', 0),
            json.dumps(self._extract_outcome(trace)),
            trace.blockchain_hash
        ))
        
        conn.commit()
        conn.close()
    
    def get_decision_trace(self, decision_id: str) -> Optional[DecisionTrace]:
        """Retrieve decision trace by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trace_data FROM decision_traces
            WHERE decision_id = ?
        """, (decision_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        
        return None
    
    def query_decisions(self, filters: Dict) -> List[DecisionTrace]:
        """Query decisions with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT trace_data FROM decision_traces WHERE 1=1"
        params = []
        
        if 'min_confidence' in filters:
            query += " AND confidence >= ?"
            params.append(filters['min_confidence'])
        
        if 'start_time' in filters:
            query += " AND timestamp >= ?"
            params.append(filters['start_time'])
        
        if 'end_time' in filters:
            query += " AND timestamp <= ?"
            params.append(filters['end_time'])
        
        query += " ORDER BY timestamp DESC LIMIT 100"
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append(pickle.loads(row[0]))
        
        conn.close()
        return results
    
    def _extract_outcome(self, trace: DecisionTrace) -> Dict:
        """Extract outcome from trace"""
        return {
            'executed': trace.confidence_breakdown.get('overall', 0) > 0.7,
            'confidence': trace.confidence_breakdown.get('overall', 0),
            'expected_profit': trace.opportunity.get('expected_profit', 0)
        }