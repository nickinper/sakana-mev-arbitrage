"""
Efficient block data caching system
"""
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
import aioredis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

Base = declarative_base()


class BlockData(Base):
    """SQLAlchemy model for block data"""
    __tablename__ = 'blocks'
    
    block_number = Column(Integer, primary_key=True)
    timestamp = Column(Integer)
    gas_used = Column(Integer)
    gas_limit = Column(Integer)
    base_fee = Column(Float)
    transaction_count = Column(Integer)
    dex_transaction_count = Column(Integer)
    arbitrage_opportunities = Column(Integer)
    data = Column(JSON)  # Full block data
    indexed_at = Column(DateTime, default=datetime.utcnow)


class BlockCache:
    """Efficient caching system for block data with Redis and SQLite"""
    
    def __init__(self, db_url: str, redis_url: Optional[str] = None):
        # SQLite for persistent storage
        self.engine = create_engine(
            db_url,
            connect_args={'check_same_thread': False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis for fast access
        self.redis_url = redis_url
        self.redis = None
        
        # Local memory cache for very recent blocks
        self.memory_cache: Dict[int, Dict] = {}
        self.cache_size = 1000
    
    async def initialize_redis(self):
        """Initialize Redis connection"""
        if self.redis_url:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
    
    async def store_block(self, block_number: int, block_data: Dict):
        """Store block data in cache layers"""
        # Memory cache
        self.memory_cache[block_number] = block_data
        
        # Maintain cache size
        if len(self.memory_cache) > self.cache_size:
            oldest_block = min(self.memory_cache.keys())
            del self.memory_cache[oldest_block]
        
        # Redis cache (if available)
        if self.redis:
            await self.redis.setex(
                f"block:{block_number}",
                3600,  # 1 hour TTL
                json.dumps(self._serialize_block(block_data))
            )
        
        # SQLite persistent storage
        session = self.Session()
        try:
            block_record = BlockData(
                block_number=block_number,
                timestamp=block_data.get('timestamp', 0),
                gas_used=block_data.get('gasUsed', 0),
                gas_limit=block_data.get('gasLimit', 0),
                base_fee=float(block_data.get('baseFeePerGas', 0)) / 1e9 if 'baseFeePerGas' in block_data else 0,
                transaction_count=len(block_data.get('transactions', [])),
                dex_transaction_count=self._count_dex_transactions(block_data),
                arbitrage_opportunities=0,  # Updated later by analysis
                data=self._serialize_block(block_data)
            )
            
            session.merge(block_record)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error storing block {block_number}: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def get_block(self, block_number: int) -> Optional[Dict]:
        """Retrieve block data from cache layers"""
        # Check memory cache first
        if block_number in self.memory_cache:
            return self.memory_cache[block_number]
        
        # Check Redis cache
        if self.redis:
            data = await self.redis.get(f"block:{block_number}")
            if data:
                block_data = json.loads(data)
                self.memory_cache[block_number] = block_data
                return block_data
        
        # Check SQLite
        session = self.Session()
        try:
            block_record = session.query(BlockData).filter_by(
                block_number=block_number
            ).first()
            
            if block_record and block_record.data:
                block_data = block_record.data
                
                # Restore to faster caches
                self.memory_cache[block_number] = block_data
                if self.redis:
                    await self.redis.setex(
                        f"block:{block_number}",
                        3600,
                        json.dumps(block_data)
                    )
                
                return block_data
                
        finally:
            session.close()
        
        return None
    
    async def get_blocks_range(self, start: int, end: int) -> List[Dict]:
        """Get multiple blocks efficiently"""
        blocks = []
        
        # Try to get from memory/Redis first
        missing_blocks = []
        
        for block_num in range(start, end + 1):
            if block_num in self.memory_cache:
                blocks.append(self.memory_cache[block_num])
            else:
                missing_blocks.append(block_num)
        
        # Batch fetch missing blocks from Redis
        if self.redis and missing_blocks:
            pipeline = self.redis.pipeline()
            for block_num in missing_blocks:
                pipeline.get(f"block:{block_num}")
            
            results = await pipeline.execute()
            
            for i, data in enumerate(results):
                if data:
                    block_data = json.loads(data)
                    blocks.append(block_data)
                    self.memory_cache[missing_blocks[i]] = block_data
                else:
                    # Still missing, will fetch from SQLite
                    pass
        
        # Fetch remaining from SQLite
        if len(blocks) < (end - start + 1):
            session = self.Session()
            try:
                block_records = session.query(BlockData).filter(
                    BlockData.block_number.between(start, end)
                ).all()
                
                for record in block_records:
                    if record.data:
                        blocks.append(record.data)
                        
            finally:
                session.close()
        
        return sorted(blocks, key=lambda x: x.get('number', 0))
    
    def _serialize_block(self, block_data: Dict) -> Dict:
        """Serialize block data for storage"""
        # Convert any bytes to hex strings
        serialized = {}
        
        for key, value in block_data.items():
            if isinstance(value, bytes):
                serialized[key] = value.hex()
            elif key == 'transactions' and isinstance(value, list):
                # Don't store full transaction data to save space
                serialized[key] = [tx['hash'].hex() if isinstance(tx, dict) else str(tx) 
                                 for tx in value[:100]]  # Limit to 100 txs
            else:
                serialized[key] = value
        
        return serialized
    
    def _count_dex_transactions(self, block_data: Dict) -> int:
        """Count DEX transactions in a block"""
        # Simplified counting - in production, check transaction details
        dex_count = 0
        
        dex_addresses = {
            '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Uniswap V2
            '0xE592427A0AEce92De3Edee1F18E0157C05861564',  # Uniswap V3
            '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',  # SushiSwap
        }
        
        for tx in block_data.get('transactions', [])[:100]:  # Check first 100
            if isinstance(tx, dict) and tx.get('to'):
                if tx['to'].lower() in [addr.lower() for addr in dex_addresses]:
                    dex_count += 1
        
        return dex_count
    
    async def get_stats(self, hours: int = 24) -> Dict:
        """Get cache statistics"""
        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_blocks = session.query(BlockData).filter(
                BlockData.indexed_at >= cutoff_time
            ).count()
            
            total_blocks = session.query(BlockData).count()
            
            stats = {
                'memory_cache_size': len(self.memory_cache),
                'total_blocks_cached': total_blocks,
                'recent_blocks_cached': recent_blocks,
                'cache_hit_rate': 0.0,  # Would need to track this
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if self.redis:
                stats['redis_connected'] = await self.redis.ping()
            
            return stats
            
        finally:
            session.close()
    
    async def cleanup_old_blocks(self, days: int = 7):
        """Remove old blocks from cache"""
        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            deleted = session.query(BlockData).filter(
                BlockData.indexed_at < cutoff_time
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted} old blocks")
            
        except Exception as e:
            logger.error(f"Error cleaning up blocks: {e}")
            session.rollback()
        finally:
            session.close()