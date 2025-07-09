"""
DEX event parsing and decoding
"""
import logging
from typing import Dict, List, Optional, Union
from web3 import Web3
from eth_abi import decode
from eth_utils import encode_hex
import json

logger = logging.getLogger(__name__)


class DexEventParser:
    """Parse and decode DEX swap events"""
    
    def __init__(self):
        # Event signatures
        self.event_signatures = {
            # Uniswap V2 / SushiSwap
            'Swap': '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',
            'Sync': '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1',
            
            # Uniswap V3
            'SwapV3': '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67',
            
            # Curve
            'TokenExchange': '0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140',
            'TokenExchangeUnderlying': '0xd013ca23e77a65003c2c659c5442c00c805371b7fc1ebd4c206c41d1536bd90b',
        }
        
        # Method signatures for input decoding
        self.method_signatures = {
            # Uniswap V2 Router
            'swapExactTokensForTokens': {
                'sig': '0x38ed1739',
                'inputs': ['uint256', 'uint256', 'address[]', 'address', 'uint256']
            },
            'swapTokensForExactTokens': {
                'sig': '0x8803dbee',
                'inputs': ['uint256', 'uint256', 'address[]', 'address', 'uint256']
            },
            'swapExactETHForTokens': {
                'sig': '0x7ff36ab5',
                'inputs': ['uint256', 'address[]', 'address', 'uint256']
            },
            'swapTokensForExactETH': {
                'sig': '0x4a25d94a',
                'inputs': ['uint256', 'uint256', 'address[]', 'address', 'uint256']
            },
            'swapExactTokensForETH': {
                'sig': '0x18cbafe5',
                'inputs': ['uint256', 'uint256', 'address[]', 'address', 'uint256']
            },
            'swapETHForExactTokens': {
                'sig': '0xfb3bdb41',
                'inputs': ['uint256', 'address[]', 'address', 'uint256']
            },
            
            # Uniswap V3 Router
            'exactInputSingle': {
                'sig': '0x414bf389',
                'inputs': ['(address,address,uint24,address,uint256,uint256,uint256,uint160)']
            },
            'exactOutputSingle': {
                'sig': '0xdb3e2198',
                'inputs': ['(address,address,uint24,address,uint256,uint256,uint256,uint160)']
            },
        }
        
        # Common token addresses
        self.tokens = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
        }
        
        self.token_decimals = {
            'WETH': 18,
            'USDC': 6,
            'USDT': 6,
            'DAI': 18,
            'WBTC': 8,
        }
    
    def parse_swap_event(self, log: Dict) -> Optional[Dict]:
        """Parse a swap event from transaction log"""
        try:
            # Check if it's a known swap event
            topic0 = log['topics'][0].hex() if log['topics'] else None
            
            if topic0 == self.event_signatures['Swap']:
                return self._parse_v2_swap(log)
            elif topic0 == self.event_signatures['SwapV3']:
                return self._parse_v3_swap(log)
            elif topic0 in [self.event_signatures['TokenExchange'], 
                           self.event_signatures['TokenExchangeUnderlying']]:
                return self._parse_curve_swap(log)
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing swap event: {e}")
            return None
    
    def _parse_v2_swap(self, log: Dict) -> Dict:
        """Parse Uniswap V2 / SushiSwap swap event"""
        # Swap event structure:
        # event Swap(
        #   address indexed sender,
        #   uint amount0In,
        #   uint amount1In,
        #   uint amount0Out,
        #   uint amount1Out,
        #   address indexed to
        # );
        
        # Decode non-indexed parameters from data
        data = bytes.fromhex(log['data'][2:])  # Remove '0x'
        decoded = decode(['uint256', 'uint256', 'uint256', 'uint256'], data)
        
        amount0_in, amount1_in, amount0_out, amount1_out = decoded
        
        # Indexed parameters from topics
        sender = '0x' + log['topics'][1].hex()[26:]  # Remove padding
        to = '0x' + log['topics'][2].hex()[26:]
        
        # Determine swap direction and amounts
        if amount0_in > 0 and amount1_out > 0:
            token_in_amount = amount0_in
            token_out_amount = amount1_out
            token_in_index = 0
            token_out_index = 1
        else:
            token_in_amount = amount1_in
            token_out_amount = amount0_out
            token_in_index = 1
            token_out_index = 0
        
        return {
            'type': 'uniswap_v2',
            'pool_address': log['address'],
            'sender': sender,
            'to': to,
            'token_in_amount': token_in_amount,
            'token_out_amount': token_out_amount,
            'token_in_index': token_in_index,
            'token_out_index': token_out_index,
            'block_number': log['blockNumber'],
            'tx_hash': log['transactionHash'].hex(),
            'log_index': log['logIndex']
        }
    
    def _parse_v3_swap(self, log: Dict) -> Dict:
        """Parse Uniswap V3 swap event"""
        # Swap event structure:
        # event Swap(
        #   address indexed sender,
        #   address indexed recipient,
        #   int256 amount0,
        #   int256 amount1,
        #   uint160 sqrtPriceX96,
        #   uint128 liquidity,
        #   int24 tick
        # );
        
        # Decode non-indexed parameters
        data = bytes.fromhex(log['data'][2:])
        decoded = decode(['int256', 'int256', 'uint160', 'uint128', 'int24'], data)
        
        amount0, amount1, sqrt_price, liquidity, tick = decoded
        
        # Indexed parameters
        sender = '0x' + log['topics'][1].hex()[26:]
        recipient = '0x' + log['topics'][2].hex()[26:]
        
        # V3 uses signed integers for amounts
        # Positive = token in, Negative = token out
        if amount0 > 0:
            token_in_amount = amount0
            token_out_amount = abs(amount1)
            token_in_index = 0
            token_out_index = 1
        else:
            token_in_amount = amount1
            token_out_amount = abs(amount0)
            token_in_index = 1
            token_out_index = 0
        
        return {
            'type': 'uniswap_v3',
            'pool_address': log['address'],
            'sender': sender,
            'recipient': recipient,
            'token_in_amount': token_in_amount,
            'token_out_amount': token_out_amount,
            'token_in_index': token_in_index,
            'token_out_index': token_out_index,
            'sqrt_price_x96': sqrt_price,
            'liquidity': liquidity,
            'tick': tick,
            'block_number': log['blockNumber'],
            'tx_hash': log['transactionHash'].hex(),
            'log_index': log['logIndex']
        }
    
    def _parse_curve_swap(self, log: Dict) -> Dict:
        """Parse Curve swap event"""
        # TokenExchange event structure:
        # event TokenExchange(
        #   address indexed buyer,
        #   int128 sold_id,
        #   uint256 tokens_sold,
        #   int128 bought_id,
        #   uint256 tokens_bought
        # );
        
        data = bytes.fromhex(log['data'][2:])
        decoded = decode(['int128', 'uint256', 'int128', 'uint256'], data)
        
        sold_id, tokens_sold, bought_id, tokens_bought = decoded
        buyer = '0x' + log['topics'][1].hex()[26:]
        
        return {
            'type': 'curve',
            'pool_address': log['address'],
            'buyer': buyer,
            'token_in_amount': tokens_sold,
            'token_out_amount': tokens_bought,
            'token_in_index': sold_id,
            'token_out_index': bought_id,
            'block_number': log['blockNumber'],
            'tx_hash': log['transactionHash'].hex(),
            'log_index': log['logIndex']
        }
    
    def decode_swap_input(self, input_data: str) -> Optional[Dict]:
        """Decode swap transaction input data"""
        if len(input_data) < 10:
            return None
        
        method_sig = input_data[:10]
        
        # Find matching method
        for method_name, method_info in self.method_signatures.items():
            if method_info['sig'] == method_sig:
                return self._decode_method_params(
                    method_name, 
                    input_data[10:], 
                    method_info['inputs']
                )
        
        return None
    
    def _decode_method_params(self, method_name: str, data: str, 
                            param_types: List[str]) -> Dict:
        """Decode method parameters"""
        try:
            # Remove '0x' if present
            if data.startswith('0x'):
                data = data[2:]
            
            # Decode based on method
            if method_name.startswith('swap') and 'V3' not in method_name:
                # V2 router methods
                decoded = decode(param_types, bytes.fromhex(data))
                
                if 'TokensForTokens' in method_name:
                    return {
                        'method': method_name,
                        'amount_in': decoded[0] if 'Exact' in method_name else decoded[1],
                        'amount_out_min': decoded[1] if 'Exact' in method_name else decoded[0],
                        'path': decoded[2],
                        'to': decoded[3],
                        'deadline': decoded[4]
                    }
                elif 'ETHForTokens' in method_name:
                    return {
                        'method': method_name,
                        'amount_out_min': decoded[0],
                        'path': decoded[1],
                        'to': decoded[2],
                        'deadline': decoded[3]
                    }
                elif 'TokensForETH' in method_name:
                    return {
                        'method': method_name,
                        'amount_in': decoded[0] if 'Exact' in method_name else decoded[1],
                        'amount_out_min': decoded[1] if 'Exact' in method_name else decoded[0],
                        'path': decoded[2],
                        'to': decoded[3],
                        'deadline': decoded[4]
                    }
            
            elif 'V3' in method_name or method_name in ['exactInputSingle', 'exactOutputSingle']:
                # V3 router methods - struct parameters
                # This is simplified - full struct decoding would be more complex
                return {
                    'method': method_name,
                    'params': 'V3 struct - requires full ABI decoding'
                }
            
            return {'method': method_name, 'raw_params': decoded}
            
        except Exception as e:
            logger.error(f"Error decoding method params: {e}")
            return None
    
    def identify_arbitrage_path(self, swap_events: List[Dict]) -> Optional[Dict]:
        """Identify potential arbitrage paths from swap events"""
        if len(swap_events) < 2:
            return None
        
        # Group swaps by same block and similar timestamp
        block_swaps = {}
        
        for swap in swap_events:
            block = swap['block_number']
            if block not in block_swaps:
                block_swaps[block] = []
            block_swaps[block].append(swap)
        
        # Look for circular paths
        for block, swaps in block_swaps.items():
            if len(swaps) >= 2:
                # Check if swaps form a cycle
                for i, swap1 in enumerate(swaps):
                    for swap2 in swaps[i+1:]:
                        if self._forms_arbitrage_cycle(swap1, swap2):
                            return {
                                'block': block,
                                'swap1': swap1,
                                'swap2': swap2,
                                'potential_arbitrage': True
                            }
        
        return None
    
    def _forms_arbitrage_cycle(self, swap1: Dict, swap2: Dict) -> bool:
        """Check if two swaps could form an arbitrage cycle"""
        # Simplified check - in production, trace actual token flows
        # Check if swaps are on different DEXes
        if swap1['type'] == swap2['type']:
            return False
        
        # Check if they involve same token pairs (simplified)
        return True
    
    def calculate_price_impact(self, token_in_amount: int, token_out_amount: int,
                              token_in_decimals: int = 18, 
                              token_out_decimals: int = 18) -> float:
        """Calculate effective price from swap amounts"""
        # Normalize amounts by decimals
        amount_in_normalized = token_in_amount / (10 ** token_in_decimals)
        amount_out_normalized = token_out_amount / (10 ** token_out_decimals)
        
        if amount_in_normalized > 0:
            return amount_out_normalized / amount_in_normalized
        
        return 0.0