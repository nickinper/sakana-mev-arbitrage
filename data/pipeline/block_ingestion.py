"""
Block ingestion module for historical arbitrage opportunity analysis
"""
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from web3 import Web3
from web3.types import BlockData, TxData
import aiohttp
from sqlalchemy import create_engine
from ..storage.block_cache import BlockCache
from ..storage.opportunity_db import OpportunityDB

logger = logging.getLogger(__name__)


class BlockIngestion:
    """Handles historical block data ingestion and arbitrage opportunity identification"""
    
    def __init__(self, rpc_url: str, db_url: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.block_cache = BlockCache(db_url)
        self.opportunity_db = OpportunityDB(db_url)
        
        # DEX contract addresses for filtering
        self.dex_addresses = {
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap_router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'curve_registry': '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
        }
        
        # Token addresses for common pairs
        self.tokens = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        }
    
    async def ingest_historical_blocks(self, start_block: int, end_block: int, 
                                     batch_size: int = 10) -> Dict[str, int]:
        """
        Ingest historical blocks and identify arbitrage opportunities
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            batch_size: Number of blocks to process in parallel
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting block ingestion from {start_block} to {end_block}")
        
        stats = {
            'blocks_processed': 0,
            'transactions_analyzed': 0,
            'opportunities_found': 0,
            'errors': 0
        }
        
        # Process blocks in batches
        for batch_start in range(start_block, end_block + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_block)
            
            try:
                # Fetch blocks in parallel
                tasks = [
                    self._process_block(block_num) 
                    for block_num in range(batch_start, batch_end + 1)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing block: {result}")
                        stats['errors'] += 1
                    else:
                        stats['blocks_processed'] += 1
                        stats['transactions_analyzed'] += result['tx_count']
                        stats['opportunities_found'] += result['opportunities']
                
                # Log progress
                if stats['blocks_processed'] % 100 == 0:
                    logger.info(f"Progress: {stats['blocks_processed']} blocks, "
                              f"{stats['opportunities_found']} opportunities found")
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                stats['errors'] += 1
        
        logger.info(f"Ingestion complete. Stats: {stats}")
        return stats
    
    async def _process_block(self, block_number: int) -> Dict[str, int]:
        """Process a single block for arbitrage opportunities"""
        block = self.w3.eth.get_block(block_number, full_transactions=True)
        
        result = {
            'tx_count': len(block['transactions']),
            'opportunities': 0
        }
        
        # Cache block data
        await self.block_cache.store_block(block_number, block)
        
        # Extract DEX transactions
        dex_txs = self._filter_dex_transactions(block['transactions'])
        
        # Analyze for arbitrage opportunities
        opportunities = await self._identify_arbitrage_opportunities(dex_txs, block)
        
        # Store opportunities
        for opp in opportunities:
            await self.opportunity_db.store_opportunity(opp)
            result['opportunities'] += 1
        
        return result
    
    def _filter_dex_transactions(self, transactions: List[TxData]) -> List[TxData]:
        """Filter transactions that interact with DEX contracts"""
        dex_txs = []
        
        for tx in transactions:
            if tx['to'] and tx['to'].lower() in [addr.lower() for addr in self.dex_addresses.values()]:
                dex_txs.append(tx)
        
        return dex_txs
    
    async def _identify_arbitrage_opportunities(self, dex_txs: List[TxData], 
                                              block: BlockData) -> List[Dict]:
        """Identify potential arbitrage opportunities from DEX transactions"""
        opportunities = []
        
        # Group transactions by block timestamp to find same-block opportunities
        for i, tx1 in enumerate(dex_txs):
            for tx2 in dex_txs[i+1:]:
                # Check if transactions involve same token pairs
                opp = self._check_arbitrage_opportunity(tx1, tx2, block)
                if opp:
                    opportunities.append(opp)
        
        return opportunities
    
    def _check_arbitrage_opportunity(self, tx1: TxData, tx2: TxData, 
                                    block: BlockData) -> Optional[Dict]:
        """Check if two transactions create an arbitrage opportunity"""
        # This is a simplified check - in production, decode swap events
        # and calculate actual price differences
        
        # For now, return a mock opportunity for testing
        if tx1['value'] > 0 and tx2['value'] > 0:
            return {
                'block_number': block['number'],
                'timestamp': block['timestamp'],
                'tx1_hash': tx1['hash'].hex(),
                'tx2_hash': tx2['hash'].hex(),
                'estimated_profit': 50.0,  # USD
                'gas_cost': 20.0,  # USD
                'token_pair': 'WETH/USDC',
                'dex_1': 'uniswap_v2',
                'dex_2': 'sushiswap'
            }
        
        return None
    
    def parse_dex_events(self, block: BlockData) -> List[Dict]:
        """Parse DEX swap events from block logs"""
        swap_events = []
        
        # Get logs for the block
        logs = self.w3.eth.get_logs({
            'fromBlock': block['number'],
            'toBlock': block['number']
        })
        
        # Parse swap events (simplified)
        for log in logs:
            if log['address'].lower() in [addr.lower() for addr in self.dex_addresses.values()]:
                # Decode swap event - implementation depends on DEX ABI
                swap_events.append({
                    'block': block['number'],
                    'tx_hash': log['transactionHash'].hex(),
                    'dex': self._identify_dex(log['address']),
                    'log_index': log['logIndex']
                })
        
        return swap_events
    
    def _identify_dex(self, address: str) -> str:
        """Identify which DEX a contract address belongs to"""
        address_lower = address.lower()
        for dex_name, dex_addr in self.dex_addresses.items():
            if dex_addr.lower() == address_lower:
                return dex_name
        return 'unknown'