"""
Real-time mempool monitoring for arbitrage opportunities
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import websockets
from web3 import Web3
from web3.types import TxData
from eth_abi import decode
import aioredis

logger = logging.getLogger(__name__)


class MempoolStream:
    """Real-time mempool monitoring and arbitrage opportunity detection"""
    
    def __init__(self, ws_url: str, redis_url: str):
        self.ws_url = ws_url
        self.redis = None
        self.redis_url = redis_url
        self.pending_txs = asyncio.Queue(maxsize=10000)
        self.seen_hashes: Set[str] = set()
        
        # DEX method signatures for filtering
        self.dex_methods = {
            # Uniswap V2
            'swapExactTokensForTokens': '0x38ed1739',
            'swapTokensForExactTokens': '0x8803dbee',
            'swapExactETHForTokens': '0x7ff36ab5',
            'swapTokensForExactETH': '0x4a25d94a',
            'swapExactTokensForETH': '0x18cbafe5',
            'swapETHForExactTokens': '0xfb3bdb41',
            
            # Uniswap V3
            'exactInputSingle': '0x414bf389',
            'exactOutputSingle': '0xdb3e2198',
            'exactInput': '0xc04b8d59',
            'exactOutput': '0xf28c0498',
            
            # SushiSwap (same as V2)
            # Curve
            'exchange': '0x3df02124',
            'exchange_underlying': '0xa6417ed6',
        }
        
        self.dex_addresses = {
            'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'curve': '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
        }
    
    async def start_streaming(self):
        """Start streaming pending transactions from mempool"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # Subscribe to pending transactions
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newPendingTransactions", True]
                    }))
                    
                    # Get subscription ID
                    response = await ws.recv()
                    data = json.loads(response)
                    subscription_id = data.get('result')
                    
                    if not subscription_id:
                        logger.error(f"Failed to subscribe: {data}")
                        await asyncio.sleep(5)
                        continue
                    
                    logger.info(f"Subscribed to mempool with ID: {subscription_id}")
                    
                    # Stream transactions
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if 'params' in data and 'result' in data['params']:
                                tx = data['params']['result']
                                
                                # Filter and queue relevant transactions
                                if self._is_dex_transaction(tx):
                                    await self._process_pending_tx(tx)
                        
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
            
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)  # Reconnect after 5 seconds
    
    def _is_dex_transaction(self, tx: Dict) -> bool:
        """Check if transaction is a DEX swap"""
        if not tx.get('to'):
            return False
        
        # Check if to address is a known DEX
        to_address = tx['to'].lower()
        if to_address not in [addr.lower() for addr in self.dex_addresses.values()]:
            return False
        
        # Check method signature
        if tx.get('input') and len(tx['input']) >= 10:
            method_sig = tx['input'][:10]
            return method_sig in self.dex_methods.values()
        
        return False
    
    async def _process_pending_tx(self, tx: Dict):
        """Process a pending DEX transaction"""
        tx_hash = tx['hash']
        
        # Dedup check
        if tx_hash in self.seen_hashes:
            return
        
        self.seen_hashes.add(tx_hash)
        
        # Clean old hashes periodically (keep last 10k)
        if len(self.seen_hashes) > 20000:
            self.seen_hashes = set(list(self.seen_hashes)[-10000:])
        
        # Parse transaction details
        parsed_tx = self._parse_dex_transaction(tx)
        
        if parsed_tx:
            # Store in Redis for analysis
            await self.redis.setex(
                f"pending_tx:{tx_hash}",
                60,  # Expire after 60 seconds
                json.dumps(parsed_tx)
            )
            
            # Queue for opportunity analysis
            try:
                await self.pending_txs.put(parsed_tx)
            except asyncio.QueueFull:
                logger.warning("Pending transaction queue full")
    
    def _parse_dex_transaction(self, tx: Dict) -> Optional[Dict]:
        """Parse DEX transaction details"""
        try:
            method_sig = tx['input'][:10]
            dex_name = self._identify_dex(tx['to'])
            
            parsed = {
                'hash': tx['hash'],
                'from': tx['from'],
                'to': tx['to'],
                'value': int(tx.get('value', '0x0'), 16),
                'gas_price': int(tx.get('gasPrice', '0x0'), 16),
                'gas': int(tx.get('gas', '0x0'), 16),
                'input': tx['input'],
                'method_sig': method_sig,
                'dex': dex_name,
                'timestamp': datetime.utcnow().isoformat(),
                'nonce': int(tx.get('nonce', '0x0'), 16)
            }
            
            # Decode swap parameters based on method
            swap_details = self._decode_swap_params(method_sig, tx['input'], dex_name)
            if swap_details:
                parsed.update(swap_details)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing transaction {tx.get('hash')}: {e}")
            return None
    
    def _identify_dex(self, address: str) -> str:
        """Identify which DEX a transaction is for"""
        address_lower = address.lower()
        for dex_name, dex_addr in self.dex_addresses.items():
            if dex_addr.lower() == address_lower:
                return dex_name
        return 'unknown'
    
    def _decode_swap_params(self, method_sig: str, input_data: str, 
                           dex: str) -> Optional[Dict]:
        """Decode swap parameters from transaction input"""
        # This is simplified - in production, use proper ABI decoding
        # For now, return mock data for testing
        
        if method_sig == '0x38ed1739':  # swapExactTokensForTokens
            return {
                'swap_type': 'exact_in',
                'token_in': 'WETH',
                'token_out': 'USDC',
                'amount_in': 1000000000000000000,  # 1 ETH
                'min_amount_out': 1800000000,  # 1800 USDC
            }
        
        return None
    
    async def get_pending_transactions(self, limit: int = 100) -> List[Dict]:
        """Get pending transactions from queue"""
        txs = []
        
        for _ in range(min(limit, self.pending_txs.qsize())):
            try:
                tx = await asyncio.wait_for(self.pending_txs.get(), timeout=0.1)
                txs.append(tx)
            except asyncio.TimeoutError:
                break
        
        return txs
    
    async def get_mempool_stats(self) -> Dict:
        """Get current mempool statistics"""
        stats = {
            'queue_size': self.pending_txs.qsize(),
            'seen_transactions': len(self.seen_hashes),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Get DEX transaction counts from Redis
        if self.redis:
            for dex in self.dex_addresses.keys():
                count = await self.redis.get(f"dex_count:{dex}")
                stats[f"{dex}_count"] = int(count) if count else 0
        
        return stats