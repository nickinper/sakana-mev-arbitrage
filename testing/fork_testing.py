"""
Fork testing infrastructure for safe arbitrage testing
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_account import Account
import subprocess
import time
import requests
import json

logger = logging.getLogger(__name__)


class ForkTester:
    """Test arbitrage strategies on forked mainnet"""
    
    def __init__(self, fork_url: str = "http://localhost:8545"):
        self.fork_url = fork_url
        self.w3 = None
        self.test_account = None
        self.anvil_process = None
        
        # Contract addresses
        self.contracts = {
            'weth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'usdc': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'usdt': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'dai': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap_router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
        }
        
        # ERC20 ABI for token interactions
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }
        ]
    
    async def start_fork(self, block_number: Optional[int] = None):
        """Start Anvil fork of mainnet"""
        logger.info("Starting Anvil fork...")
        
        # Build Anvil command
        cmd = [
            "anvil",
            "--fork-url", self.fork_url,
            "--port", "8545",
            "--accounts", "10",
            "--balance", "10000",
            "--gas-limit", "30000000",
            "--base-fee", "1000000000",  # 1 Gwei
        ]
        
        if block_number:
            cmd.extend(["--fork-block-number", str(block_number)])
        
        # Start Anvil process
        self.anvil_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Anvil to start
        for _ in range(30):
            try:
                response = requests.post(
                    "http://localhost:8545",
                    json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
                )
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Anvil failed to start")
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
        
        # Create test account with funds
        account = Account.create()
        self.test_account = {
            'address': account.address,
            'private_key': account.key.hex()
        }
        
        # Fund test account with ETH
        await self._fund_account(self.test_account['address'], Web3.to_wei(100, 'ether'))
        
        logger.info(f"Fork started at block {self.w3.eth.block_number}")
        logger.info(f"Test account: {self.test_account['address']}")
    
    async def stop_fork(self):
        """Stop Anvil fork"""
        if self.anvil_process:
            self.anvil_process.terminate()
            self.anvil_process.wait()
            logger.info("Anvil fork stopped")
    
    async def _fund_account(self, address: str, amount: int):
        """Fund an account with ETH using Anvil's auto-mining"""
        # Anvil provides funded accounts we can use
        funded_account = self.w3.eth.accounts[0]
        
        tx = {
            'from': funded_account,
            'to': address,
            'value': amount,
            'gas': 21000,
            'gasPrice': self.w3.eth.gas_price
        }
        
        tx_hash = self.w3.eth.send_transaction(tx)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    async def simulate_arbitrage(self, arbitrage_params: Dict) -> Dict:
        """Simulate an arbitrage transaction on the fork"""
        try:
            # Build arbitrage transaction
            tx = await self._build_arbitrage_tx(arbitrage_params)
            
            # Estimate gas
            try:
                gas_estimate = self.w3.eth.estimate_gas(tx)
                tx['gas'] = int(gas_estimate * 1.2)  # 20% buffer
            except Exception as e:
                logger.error(f"Gas estimation failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'profitable': False
                }
            
            # Get balances before
            token_in = arbitrage_params['token_in']
            token_out = arbitrage_params['token_out']
            
            balance_before = await self._get_token_balance(
                self.contracts[token_in], 
                self.test_account['address']
            )
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, 
                self.test_account['private_key']
            )
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if successful
            if receipt['status'] == 1:
                # Get balance after
                balance_after = await self._get_token_balance(
                    self.contracts[token_out], 
                    self.test_account['address']
                )
                
                # Calculate profit
                gas_used = receipt['gasUsed']
                gas_price = tx['gasPrice']
                gas_cost_wei = gas_used * gas_price
                gas_cost_eth = gas_cost_wei / 1e18
                
                # Simplified profit calculation
                profit_tokens = balance_after - balance_before
                
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'gas_used': gas_used,
                    'gas_cost_eth': gas_cost_eth,
                    'profit_tokens': profit_tokens,
                    'profitable': profit_tokens > 0,
                    'block_number': receipt['blockNumber']
                }
            else:
                # Transaction reverted
                return {
                    'success': False,
                    'error': 'Transaction reverted',
                    'profitable': False,
                    'gas_used': receipt['gasUsed']
                }
                
        except Exception as e:
            logger.error(f"Arbitrage simulation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'profitable': False
            }
    
    async def _build_arbitrage_tx(self, params: Dict) -> Dict:
        """Build arbitrage transaction"""
        # This is a simplified example - real implementation would use
        # proper router interfaces and multicall for atomic execution
        
        # For testing, we'll simulate a simple swap
        router_address = self.contracts[params['dex_1'] + '_router']
        
        # Build transaction data (simplified)
        tx = {
            'from': self.test_account['address'],
            'to': router_address,
            'value': 0,
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.test_account['address']),
            'data': '0x'  # Would contain encoded swap data
        }
        
        return tx
    
    async def _get_token_balance(self, token_address: str, account: str) -> int:
        """Get ERC20 token balance"""
        token_contract = self.w3.eth.contract(
            address=token_address,
            abi=self.erc20_abi
        )
        
        return token_contract.functions.balanceOf(account).call()
    
    async def test_flashloan_arbitrage(self, params: Dict) -> Dict:
        """Test arbitrage using flash loans"""
        # Deploy or use existing flashloan arbitrage contract
        # This is a placeholder - real implementation would deploy
        # a contract that executes the arbitrage atomically
        
        logger.info("Testing flashloan arbitrage...")
        
        # Simulate flashloan execution
        result = {
            'method': 'flashloan',
            'loan_amount': params.get('loan_amount', 0),
            'loan_fee': params.get('loan_amount', 0) * 0.0009,  # 0.09% Aave fee
            'success': True,  # Placeholder
            'profit': 0  # Would calculate actual profit
        }
        
        return result
    
    async def replay_historical_opportunity(self, block_number: int, 
                                          opportunity: Dict) -> Dict:
        """Replay a historical arbitrage opportunity"""
        # Fork at specific block
        await self.stop_fork()
        await self.start_fork(block_number - 1)
        
        # Simulate the arbitrage
        result = await self.simulate_arbitrage(opportunity)
        
        return {
            'historical_block': block_number,
            'replay_result': result,
            'would_have_worked': result.get('profitable', False)
        }
    
    async def batch_test_strategies(self, strategies: List[Dict], 
                                  opportunities: List[Dict]) -> List[Dict]:
        """Test multiple strategies against multiple opportunities"""
        results = []
        
        for strategy in strategies:
            strategy_results = {
                'strategy_id': strategy.get('id', 'unknown'),
                'strategy_genes': strategy.get('genes', {}),
                'opportunity_results': []
            }
            
            for opportunity in opportunities:
                # Apply strategy parameters to opportunity
                test_params = self._apply_strategy_to_opportunity(strategy, opportunity)
                
                # Test execution
                result = await self.simulate_arbitrage(test_params)
                
                strategy_results['opportunity_results'].append({
                    'opportunity_id': opportunity.get('id', 'unknown'),
                    'result': result
                })
            
            # Calculate strategy fitness
            successful = sum(
                1 for r in strategy_results['opportunity_results'] 
                if r['result'].get('profitable', False)
            )
            
            total = len(strategy_results['opportunity_results'])
            strategy_results['success_rate'] = successful / total if total > 0 else 0
            
            results.append(strategy_results)
        
        return results
    
    def _apply_strategy_to_opportunity(self, strategy: Dict, 
                                     opportunity: Dict) -> Dict:
        """Apply strategy genes to opportunity parameters"""
        params = opportunity.copy()
        
        # Apply strategy-specific modifications
        genes = strategy.get('genes', {})
        
        if 'min_profit_threshold' in genes:
            params['min_profit'] = genes['min_profit_threshold']
        
        if 'slippage_tolerance' in genes:
            params['slippage'] = genes['slippage_tolerance']
        
        if 'gas_multiplier' in genes:
            params['gas_price'] = self.w3.eth.gas_price * genes['gas_multiplier']
        
        return params
    
    async def test_gas_optimization(self, base_tx: Dict) -> Dict:
        """Test different gas strategies"""
        gas_results = []
        
        # Test different gas prices
        base_gas_price = self.w3.eth.gas_price
        multipliers = [0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
        
        for multiplier in multipliers:
            tx = base_tx.copy()
            tx['gasPrice'] = int(base_gas_price * multiplier)
            
            # Simulate transaction
            try:
                # This would actually simulate the transaction
                # For now, we'll estimate success based on gas price
                likely_success = multiplier >= 1.1
                
                gas_results.append({
                    'multiplier': multiplier,
                    'gas_price_gwei': tx['gasPrice'] / 1e9,
                    'likely_success': likely_success,
                    'cost_usd': (tx['gasPrice'] * tx['gas']) / 1e18 * 2000  # Assuming $2000 ETH
                })
                
            except Exception as e:
                logger.error(f"Gas test error with multiplier {multiplier}: {e}")
        
        # Find optimal gas price
        successful_results = [r for r in gas_results if r['likely_success']]
        if successful_results:
            optimal = min(successful_results, key=lambda x: x['cost_usd'])
        else:
            optimal = gas_results[-1]  # Highest gas price
        
        return {
            'results': gas_results,
            'optimal_multiplier': optimal['multiplier'],
            'optimal_gas_gwei': optimal['gas_price_gwei']
        }