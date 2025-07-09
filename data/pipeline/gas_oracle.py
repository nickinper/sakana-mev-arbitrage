"""
Gas price oracle for optimal transaction pricing
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
from web3 import Web3
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class GasOracle:
    """Real-time gas price monitoring and prediction"""
    
    def __init__(self, w3: Web3, etherscan_api_key: Optional[str] = None):
        self.w3 = w3
        self.etherscan_api_key = etherscan_api_key
        
        # Historical gas prices (last 100 blocks)
        self.gas_history = deque(maxlen=100)
        self.base_fee_history = deque(maxlen=100)
        
        # Current gas metrics
        self.current_gas_price = 0
        self.current_base_fee = 0
        self.priority_fee_stats = {
            'low': 1.0,      # 25th percentile
            'medium': 2.0,   # 50th percentile  
            'high': 3.0,     # 75th percentile
            'urgent': 5.0    # 90th percentile
        }
        
        # Competition tracking
        self.competitor_gas_prices = deque(maxlen=50)
        
    async def start_monitoring(self):
        """Start continuous gas price monitoring"""
        while True:
            try:
                await self.update_gas_prices()
                await asyncio.sleep(1)  # Update every block (~12 seconds)
                
            except Exception as e:
                logger.error(f"Gas monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def update_gas_prices(self):
        """Update current gas prices from network"""
        try:
            # Get latest block for base fee
            latest_block = self.w3.eth.get_block('latest')
            
            if 'baseFeePerGas' in latest_block:
                base_fee_gwei = latest_block['baseFeePerGas'] / 1e9
                self.current_base_fee = base_fee_gwei
                self.base_fee_history.append({
                    'block': latest_block['number'],
                    'base_fee': base_fee_gwei,
                    'timestamp': datetime.utcnow()
                })
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price / 1e9  # Convert to Gwei
            self.current_gas_price = gas_price
            self.gas_history.append({
                'block': latest_block['number'],
                'gas_price': gas_price,
                'timestamp': datetime.utcnow()
            })
            
            # Update priority fee statistics
            await self._update_priority_fees()
            
        except Exception as e:
            logger.error(f"Error updating gas prices: {e}")
    
    async def _update_priority_fees(self):
        """Analyze recent blocks for priority fee statistics"""
        try:
            # Get last 10 blocks
            latest = self.w3.eth.block_number
            
            priority_fees = []
            
            for block_num in range(max(0, latest - 10), latest + 1):
                block = self.w3.eth.get_block(block_num, full_transactions=True)
                
                if 'baseFeePerGas' in block:
                    base_fee = block['baseFeePerGas']
                    
                    for tx in block['transactions'][:50]:  # Sample first 50 txs
                        if 'gasPrice' in tx and tx['gasPrice'] > base_fee:
                            priority_fee = (tx['gasPrice'] - base_fee) / 1e9
                            priority_fees.append(priority_fee)
            
            if priority_fees:
                # Calculate percentiles
                self.priority_fee_stats = {
                    'low': np.percentile(priority_fees, 25),
                    'medium': np.percentile(priority_fees, 50),
                    'high': np.percentile(priority_fees, 75),
                    'urgent': np.percentile(priority_fees, 90)
                }
                
        except Exception as e:
            logger.error(f"Error updating priority fees: {e}")
    
    def get_recommended_gas_price(self, priority: str = 'medium') -> Dict[str, float]:
        """Get recommended gas price for given priority"""
        priority_fee = self.priority_fee_stats.get(priority, 2.0)
        
        # EIP-1559 transaction
        max_priority_fee = priority_fee
        max_fee = self.current_base_fee + (priority_fee * 2)  # 2x buffer
        
        # Legacy transaction
        legacy_gas_price = self.current_base_fee + priority_fee
        
        return {
            'max_fee_per_gas_gwei': max_fee,
            'max_priority_fee_per_gas_gwei': max_priority_fee,
            'legacy_gas_price_gwei': legacy_gas_price,
            'base_fee_gwei': self.current_base_fee,
            'estimated_cost_usd': self._estimate_cost_usd(legacy_gas_price)
        }
    
    def get_competitive_gas_price(self, competitor_gas: float, 
                                 outbid_factor: float = 1.1) -> Dict[str, float]:
        """Calculate gas price to outbid competitors"""
        # Track competitor gas prices
        self.competitor_gas_prices.append({
            'gas_price': competitor_gas,
            'timestamp': datetime.utcnow()
        })
        
        # Calculate competitive price
        competitive_priority = competitor_gas - self.current_base_fee
        our_priority = competitive_priority * outbid_factor
        
        return {
            'max_fee_per_gas_gwei': self.current_base_fee + (our_priority * 2),
            'max_priority_fee_per_gas_gwei': our_priority,
            'legacy_gas_price_gwei': self.current_base_fee + our_priority,
            'competitor_gas_gwei': competitor_gas,
            'outbid_factor': outbid_factor
        }
    
    def predict_next_base_fee(self) -> float:
        """Predict next block's base fee using EIP-1559 formula"""
        if not self.base_fee_history:
            return self.current_base_fee
        
        latest_block = self.w3.eth.get_block('latest')
        
        if 'gasUsed' not in latest_block or 'gasLimit' not in latest_block:
            return self.current_base_fee
        
        gas_used = latest_block['gasUsed']
        gas_limit = latest_block['gasLimit']
        base_fee = self.current_base_fee * 1e9  # Convert to Wei
        
        # EIP-1559 base fee calculation
        target_gas = gas_limit // 2
        
        if gas_used > target_gas:
            gas_delta = gas_used - target_gas
            base_fee_delta = max(
                base_fee * gas_delta // target_gas // 8,
                1
            )
            next_base_fee = base_fee + base_fee_delta
        else:
            gas_delta = target_gas - gas_used
            base_fee_delta = base_fee * gas_delta // target_gas // 8
            next_base_fee = base_fee - base_fee_delta
        
        return next_base_fee / 1e9  # Convert back to Gwei
    
    def analyze_gas_trends(self, minutes: int = 10) -> Dict:
        """Analyze recent gas price trends"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_gas = [
            g['gas_price'] 
            for g in self.gas_history 
            if g['timestamp'] >= cutoff_time
        ]
        
        recent_base = [
            b['base_fee'] 
            for b in self.base_fee_history 
            if b['timestamp'] >= cutoff_time
        ]
        
        if not recent_gas:
            return {
                'trend': 'unknown',
                'volatility': 0,
                'recommendation': 'Use medium priority'
            }
        
        # Calculate trends
        gas_mean = np.mean(recent_gas)
        gas_std = np.std(recent_gas)
        volatility = gas_std / gas_mean if gas_mean > 0 else 0
        
        # Determine trend
        if len(recent_gas) >= 2:
            if recent_gas[-1] > recent_gas[0] * 1.1:
                trend = 'rising'
                recommendation = 'Consider waiting or use high priority'
            elif recent_gas[-1] < recent_gas[0] * 0.9:
                trend = 'falling'
                recommendation = 'Good time to transact'
            else:
                trend = 'stable'
                recommendation = 'Use medium priority'
        else:
            trend = 'unknown'
            recommendation = 'Insufficient data'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'current_gas_gwei': self.current_gas_price,
            'mean_gas_gwei': gas_mean,
            'std_gas_gwei': gas_std,
            'recommendation': recommendation,
            'samples': len(recent_gas)
        }
    
    def should_execute_now(self, min_profit_usd: float, gas_limit: int = 300000) -> Dict[str, bool]:
        """Determine if current gas prices allow profitable execution"""
        gas_cost_usd = self._estimate_cost_usd(self.current_gas_price, gas_limit)
        
        # Check if profitable after gas
        profitable = min_profit_usd > gas_cost_usd * 1.5  # 50% margin
        
        # Check if gas is reasonable
        gas_reasonable = self.current_gas_price < self.priority_fee_stats['high'] + self.current_base_fee
        
        return {
            'should_execute': profitable and gas_reasonable,
            'profitable': profitable,
            'gas_reasonable': gas_reasonable,
            'gas_cost_usd': gas_cost_usd,
            'required_profit_usd': gas_cost_usd * 1.5,
            'current_gas_gwei': self.current_gas_price
        }
    
    def _estimate_cost_usd(self, gas_price_gwei: float, gas_limit: int = 300000) -> float:
        """Estimate transaction cost in USD"""
        # Simplified - in production, fetch real ETH price
        eth_price_usd = 2000.0  # Placeholder
        
        gas_cost_eth = (gas_price_gwei * gas_limit) / 1e9
        gas_cost_usd = gas_cost_eth * eth_price_usd
        
        return gas_cost_usd
    
    async def get_etherscan_gas_oracle(self) -> Optional[Dict]:
        """Get gas prices from Etherscan API"""
        if not self.etherscan_api_key:
            return None
        
        url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={self.etherscan_api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    if data['status'] == '1':
                        return {
                            'safe_gas_price': float(data['result']['SafeGasPrice']),
                            'propose_gas_price': float(data['result']['ProposeGasPrice']),
                            'fast_gas_price': float(data['result']['FastGasPrice']),
                            'base_fee': float(data['result'].get('suggestBaseFee', 0))
                        }
                        
        except Exception as e:
            logger.error(f"Error fetching Etherscan gas oracle: {e}")
        
        return None