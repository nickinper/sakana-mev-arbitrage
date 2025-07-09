#!/usr/bin/env python3
"""
Demo script to show autonomous agents in action
"""
import asyncio
from autonomous.discovery_engine import DiscoveryEngine
from autonomous.intelligent_reporter import IntelligentReporter


async def run_demo():
    """Run a quick demo of the autonomous system"""
    print("🧬 AUTONOMOUS AGENT DEMO")
    print("=" * 60)
    print("\nCreating 5 autonomous agents with different personalities...\n")
    
    # Create small discovery engine
    engine = DiscoveryEngine(num_agents=5)
    reporter = IntelligentReporter()
    
    # Show agent personalities
    print("Agent Personalities:")
    for agent in engine.agents:
        summary = agent.get_personality_summary()
        print(f"\n{agent.id}:")
        print(f"  Focus: {summary['personality']['focus_area']}")
        print(f"  Curiosity: {summary['personality']['curiosity']:.2f}")
        print(f"  Risk Tolerance: {summary['personality']['risk_tolerance']:.2f}")
    
    print("\n" + "-" * 60)
    print("Running 5 discovery cycles...\n")
    
    # Run 5 cycles
    for i in range(5):
        print(f"\n🔄 Cycle {i + 1}:")
        
        # Run discovery
        results = await engine.run_discovery_cycle()
        
        # Show discoveries
        if results['discoveries']:
            print("\n  💡 Discoveries:")
            for disc in results['discoveries']:
                print(f"    - {disc['agent']}: {disc['discovery']}")
                print(f"      Profit: ${disc['profitability']:.2f}")
        
        # Show questions
        if results['new_questions']:
            print("\n  ❓ Questions:")
            for q in results['new_questions'][:3]:
                print(f"    - {q['agent']}: {q['question']}")
        
        # Show emergent behaviors
        if results['emergent_behaviors']:
            print("\n  🌟 Emergent Behaviors:")
            for behavior in results['emergent_behaviors']:
                print(f"    - {behavior['description']}")
        
        await asyncio.sleep(1)
    
    # Generate final report
    print("\n" + "=" * 60)
    print("GENERATING FINAL REPORT...")
    print("=" * 60)
    
    report = reporter.generate_report(engine)
    formatted = reporter.format_report_for_display(report)
    print(formatted)
    
    # Show top discoveries
    print("\n🏆 TOP DISCOVERIES:")
    for i, disc in enumerate(engine.get_top_discoveries(3), 1):
        print(f"\n{i}. {disc['description']}")
        print(f"   Profit: ${disc['profitability']:.2f}")
        print(f"   Success Rate: {disc['reproducibility']:.0%}")


if __name__ == "__main__":
    asyncio.run(run_demo())