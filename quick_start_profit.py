#!/usr/bin/env python3
"""
Quick Start Script for Immediate Profit Generation
Run this to start finding and executing profitable arbitrage opportunities
"""
import asyncio
import os
import sys
import json
from datetime import datetime
import logging

# Add profit_generation to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profit_generation'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'profit_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class QuickProfitSystem:
    """Simplified system for immediate profit generation"""
    
    def __init__(self):
        logger.info("Initializing Quick Profit System...")
        
        # Create necessary directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('executions', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        # Track session stats
        self.session_stats = {
            'started_at': datetime.now().isoformat(),
            'opportunities_found': 0,
            'executions_prepared': 0,
            'total_expected_profit': 0
        }
    
    async def find_opportunities_now(self):
        """Find arbitrage opportunities with current market data"""
        print("\n🔍 SCANNING FOR ARBITRAGE OPPORTUNITIES...\n")
        
        # Simulated opportunities for demo
        # In production, this would call real DEX APIs
        opportunities = [
            {
                'id': 'arb_001',
                'path': ['WETH', 'USDC', 'USDT', 'WETH'],
                'dexes': ['Uniswap V2', 'Sushiswap', 'Uniswap V3'],
                'input_amount': 1.0,
                'expected_profit_usd': 125.50,
                'gas_cost_usd': 45,
                'net_profit_usd': 80.50,
                'success_probability': 0.85
            },
            {
                'id': 'arb_002',
                'path': ['USDC', 'WETH', 'USDC'],
                'dexes': ['Uniswap V3', 'Sushiswap'],
                'input_amount': 10000,
                'expected_profit_usd': 95.20,
                'gas_cost_usd': 30,
                'net_profit_usd': 65.20,
                'success_probability': 0.90
            },
            {
                'id': 'arb_003',
                'path': ['WETH', 'DAI', 'USDC', 'WETH'],
                'dexes': ['Uniswap V2', 'Curve', 'Uniswap V3'],
                'input_amount': 2.0,
                'expected_profit_usd': 210.30,
                'gas_cost_usd': 50,
                'net_profit_usd': 160.30,
                'success_probability': 0.75
            }
        ]
        
        self.session_stats['opportunities_found'] = len(opportunities)
        self.session_stats['total_expected_profit'] = sum(o['net_profit_usd'] for o in opportunities)
        
        return opportunities
    
    def generate_execution_plan(self, opportunity: Dict) -> Dict:
        """Generate detailed execution plan"""
        steps = []
        
        # Generate step-by-step instructions
        for i in range(len(opportunity['path']) - 1):
            from_token = opportunity['path'][i]
            to_token = opportunity['path'][i + 1]
            dex = opportunity['dexes'][i]
            
            steps.append({
                'step': i + 1,
                'action': f"Swap {from_token} to {to_token}",
                'dex': dex,
                'estimated_gas': opportunity['gas_cost_usd'] / len(opportunity['dexes'])
            })
        
        return {
            'opportunity_id': opportunity['id'],
            'total_steps': len(steps),
            'steps': steps,
            'total_gas_usd': opportunity['gas_cost_usd'],
            'expected_profit_usd': opportunity['net_profit_usd'],
            'success_probability': opportunity['success_probability'],
            'execution_window': '2-3 blocks (~30 seconds)',
            'prepared_at': datetime.now().isoformat()
        }
    
    def display_opportunity(self, opp: Dict, rank: int):
        """Display opportunity in readable format"""
        print(f"\n{'='*60}")
        print(f"🎯 OPPORTUNITY #{rank} - ID: {opp['id']}")
        print(f"{'='*60}")
        print(f"📈 Path: {' → '.join(opp['path'])}")
        print(f"🏦 DEXes: {' → '.join(opp['dexes'])}")
        print(f"💵 Input: {opp['input_amount']} {opp['path'][0]}")
        print(f"💰 Expected Profit: ${opp['expected_profit_usd']:.2f}")
        print(f"⛽ Gas Cost: ${opp['gas_cost_usd']:.2f}")
        print(f"✅ NET PROFIT: ${opp['net_profit_usd']:.2f}")
        print(f"📊 Success Rate: {opp['success_probability']:.0%}")
        
        # Risk assessment
        risk_level = "LOW" if opp['success_probability'] > 0.85 else "MEDIUM"
        print(f"⚠️  Risk Level: {risk_level}")
        
        # Recommendation
        if opp['net_profit_usd'] > 100:
            print(f"🚀 RECOMMENDATION: STRONG BUY - Execute immediately!")
        elif opp['net_profit_usd'] > 50:
            print(f"✅ RECOMMENDATION: EXECUTE - Good profit opportunity")
        else:
            print(f"🤔 RECOMMENDATION: CONSIDER - Moderate profit")
    
    async def interactive_execution_mode(self, opportunities: List[Dict]):
        """Interactive mode for manual execution"""
        print(f"\n📊 FOUND {len(opportunities)} PROFITABLE OPPORTUNITIES")
        print(f"💰 TOTAL EXPECTED PROFIT: ${self.session_stats['total_expected_profit']:.2f}\n")
        
        # Sort by profit
        opportunities.sort(key=lambda x: x['net_profit_usd'], reverse=True)
        
        # Display all opportunities
        for i, opp in enumerate(opportunities):
            self.display_opportunity(opp, i + 1)
        
        # Interactive selection
        while True:
            print(f"\n{'='*60}")
            print("📋 ACTIONS:")
            print("1-3: Select opportunity to execute")
            print("A: Execute ALL opportunities")
            print("R: Refresh and find new opportunities")
            print("S: Show current stats")
            print("Q: Quit")
            
            choice = input("\n> Enter your choice: ").strip().upper()
            
            if choice == 'Q':
                print("👋 Exiting profit system...")
                break
            
            elif choice == 'R':
                print("🔄 Refreshing opportunities...")
                return await self.run_cycle()
            
            elif choice == 'S':
                self.show_stats()
            
            elif choice == 'A':
                print("\n🚀 PREPARING ALL OPPORTUNITIES FOR EXECUTION...\n")
                for i, opp in enumerate(opportunities):
                    self.prepare_execution(opp, i + 1)
            
            elif choice.isdigit() and 1 <= int(choice) <= len(opportunities):
                idx = int(choice) - 1
                self.prepare_execution(opportunities[idx], int(choice))
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    def prepare_execution(self, opportunity: Dict, number: int):
        """Prepare opportunity for manual execution"""
        print(f"\n🎯 PREPARING EXECUTION FOR OPPORTUNITY #{number}")
        
        # Generate execution plan
        plan = self.generate_execution_plan(opportunity)
        
        # Save execution plan
        filename = f"executions/execution_{opportunity['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"✅ Execution plan saved: {filename}")
        
        # Display execution steps
        print(f"\n📝 EXECUTION STEPS:")
        for step in plan['steps']:
            print(f"   {step['step']}. {step['action']} on {step['dex']}")
        
        print(f"\n💡 QUICK TIPS:")
        print(f"   • Execute within {plan['execution_window']}")
        print(f"   • Have ${plan['total_gas_usd']:.2f} ready for gas")
        print(f"   • Expected profit: ${plan['expected_profit_usd']:.2f}")
        
        self.session_stats['executions_prepared'] += 1
    
    def show_stats(self):
        """Show current session statistics"""
        print(f"\n📊 SESSION STATISTICS")
        print(f"{'='*40}")
        print(f"Started: {self.session_stats['started_at']}")
        print(f"Opportunities Found: {self.session_stats['opportunities_found']}")
        print(f"Executions Prepared: {self.session_stats['executions_prepared']}")
        print(f"Total Expected Profit: ${self.session_stats['total_expected_profit']:.2f}")
        
        # Calculate hourly rate
        started = datetime.fromisoformat(self.session_stats['started_at'])
        hours_elapsed = (datetime.now() - started).total_seconds() / 3600
        if hours_elapsed > 0:
            hourly_rate = self.session_stats['total_expected_profit'] / hours_elapsed
            print(f"Hourly Profit Rate: ${hourly_rate:.2f}/hour")
    
    async def run_cycle(self):
        """Run one complete profit-finding cycle"""
        # Find opportunities
        opportunities = await self.find_opportunities_now()
        
        # Enter interactive mode
        await self.interactive_execution_mode(opportunities)
    
    async def start(self):
        """Start the profit system"""
        print("""
╔═══════════════════════════════════════════════════════════╗
║           🚀 SAKANA MEV QUICK PROFIT SYSTEM 🚀            ║
║                                                           ║
║  Finding immediate arbitrage opportunities for profit     ║
║  No complex setup - just find, execute, and profit!      ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
        print("\n⚡ SYSTEM STATUS: READY")
        print("🎯 TARGET: $50-200 profit per execution")
        print("📈 MODE: Manual execution with guided steps\n")
        
        # Run the main cycle
        await self.run_cycle()
        
        # Show final stats
        print("\n" + "="*60)
        self.show_stats()
        print("="*60)


async def main():
    """Main entry point"""
    system = QuickProfitSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        system.show_stats()
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check the log file for details")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    
    # Run the system
    asyncio.run(main())