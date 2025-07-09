"""
Database for storing and analyzing arbitrage opportunities
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, and_

logger = logging.getLogger(__name__)

Base = declarative_base()


class ArbitrageOpportunity(Base):
    """SQLAlchemy model for arbitrage opportunities"""
    __tablename__ = 'arbitrage_opportunities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_number = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    
    # Opportunity details
    token_pair = Column(String(50), index=True)
    dex_1 = Column(String(50))
    dex_2 = Column(String(50))
    
    # Financial metrics
    estimated_profit_usd = Column(Float)
    gas_cost_usd = Column(Float)
    net_profit_usd = Column(Float)
    profit_percentage = Column(Float)
    
    # Execution details
    tx1_hash = Column(String(66))
    tx2_hash = Column(String(66))
    amount_in = Column(Float)
    amount_out_1 = Column(Float)
    amount_out_2 = Column(Float)
    
    # Competition metrics
    was_executed = Column(Boolean, default=False)
    executor_address = Column(String(42))
    our_attempt = Column(Boolean, default=False)
    our_success = Column(Boolean, default=False)
    
    # Analysis metadata
    discovered_at = Column(DateTime, default=datetime.utcnow)
    gas_price_gwei = Column(Float)
    
    # Create composite indexes for common queries
    __table_args__ = (
        Index('idx_profitable_opportunities', 'timestamp', 'net_profit_usd'),
        Index('idx_token_pair_profits', 'token_pair', 'net_profit_usd'),
        Index('idx_dex_combinations', 'dex_1', 'dex_2'),
    )


class OpportunityDB:
    """Database interface for arbitrage opportunities"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    async def store_opportunity(self, opportunity: Dict) -> int:
        """Store a new arbitrage opportunity"""
        session = self.Session()
        try:
            opp = ArbitrageOpportunity(
                block_number=opportunity['block_number'],
                timestamp=datetime.fromtimestamp(opportunity['timestamp']),
                token_pair=opportunity['token_pair'],
                dex_1=opportunity['dex_1'],
                dex_2=opportunity['dex_2'],
                estimated_profit_usd=opportunity.get('estimated_profit', 0),
                gas_cost_usd=opportunity.get('gas_cost', 0),
                net_profit_usd=opportunity.get('estimated_profit', 0) - opportunity.get('gas_cost', 0),
                profit_percentage=(opportunity.get('estimated_profit', 0) / opportunity.get('amount_in', 1)) * 100 if opportunity.get('amount_in', 0) > 0 else 0,
                tx1_hash=opportunity.get('tx1_hash'),
                tx2_hash=opportunity.get('tx2_hash'),
                amount_in=opportunity.get('amount_in', 0),
                gas_price_gwei=opportunity.get('gas_price_gwei', 0)
            )
            
            session.add(opp)
            session.commit()
            
            return opp.id
            
        except Exception as e:
            logger.error(f"Error storing opportunity: {e}")
            session.rollback()
            return -1
        finally:
            session.close()
    
    async def get_profitable_opportunities(self, 
                                         min_profit: float = 10.0,
                                         hours: int = 24,
                                         limit: int = 100) -> List[Dict]:
        """Get recent profitable opportunities"""
        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            opportunities = session.query(ArbitrageOpportunity).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.net_profit_usd >= min_profit
                )
            ).order_by(
                ArbitrageOpportunity.net_profit_usd.desc()
            ).limit(limit).all()
            
            return [self._opportunity_to_dict(opp) for opp in opportunities]
            
        finally:
            session.close()
    
    async def get_opportunity_stats(self, hours: int = 24) -> Dict:
        """Get statistics on arbitrage opportunities"""
        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Basic stats
            total_opps = session.query(ArbitrageOpportunity).filter(
                ArbitrageOpportunity.timestamp >= cutoff_time
            ).count()
            
            profitable_opps = session.query(ArbitrageOpportunity).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.net_profit_usd > 0
                )
            ).count()
            
            # Aggregate stats
            stats = session.query(
                func.avg(ArbitrageOpportunity.net_profit_usd).label('avg_profit'),
                func.max(ArbitrageOpportunity.net_profit_usd).label('max_profit'),
                func.sum(ArbitrageOpportunity.net_profit_usd).label('total_profit'),
                func.avg(ArbitrageOpportunity.gas_cost_usd).label('avg_gas_cost')
            ).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.net_profit_usd > 0
                )
            ).first()
            
            # Token pair breakdown
            pair_stats = session.query(
                ArbitrageOpportunity.token_pair,
                func.count(ArbitrageOpportunity.id).label('count'),
                func.avg(ArbitrageOpportunity.net_profit_usd).label('avg_profit')
            ).filter(
                ArbitrageOpportunity.timestamp >= cutoff_time
            ).group_by(
                ArbitrageOpportunity.token_pair
            ).all()
            
            # DEX combination stats
            dex_stats = session.query(
                ArbitrageOpportunity.dex_1,
                ArbitrageOpportunity.dex_2,
                func.count(ArbitrageOpportunity.id).label('count'),
                func.avg(ArbitrageOpportunity.net_profit_usd).label('avg_profit')
            ).filter(
                ArbitrageOpportunity.timestamp >= cutoff_time
            ).group_by(
                ArbitrageOpportunity.dex_1,
                ArbitrageOpportunity.dex_2
            ).all()
            
            return {
                'total_opportunities': total_opps,
                'profitable_opportunities': profitable_opps,
                'profitability_rate': profitable_opps / total_opps if total_opps > 0 else 0,
                'avg_profit_usd': float(stats.avg_profit or 0),
                'max_profit_usd': float(stats.max_profit or 0),
                'total_profit_usd': float(stats.total_profit or 0),
                'avg_gas_cost_usd': float(stats.avg_gas_cost or 0),
                'token_pairs': {
                    pair.token_pair: {
                        'count': pair.count,
                        'avg_profit': float(pair.avg_profit or 0)
                    }
                    for pair in pair_stats
                },
                'dex_combinations': [
                    {
                        'dex_1': dex.dex_1,
                        'dex_2': dex.dex_2,
                        'count': dex.count,
                        'avg_profit': float(dex.avg_profit or 0)
                    }
                    for dex in dex_stats
                ],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()
    
    async def update_execution_result(self, opportunity_id: int, 
                                    executed: bool, 
                                    executor: Optional[str] = None,
                                    our_attempt: bool = False,
                                    our_success: bool = False):
        """Update opportunity with execution results"""
        session = self.Session()
        try:
            opp = session.query(ArbitrageOpportunity).filter_by(id=opportunity_id).first()
            
            if opp:
                opp.was_executed = executed
                opp.executor_address = executor
                opp.our_attempt = our_attempt
                opp.our_success = our_success
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error updating execution result: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def get_competition_analysis(self, hours: int = 24) -> Dict:
        """Analyze competition performance"""
        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Our performance
            our_attempts = session.query(ArbitrageOpportunity).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.our_attempt == True
                )
            ).count()
            
            our_successes = session.query(ArbitrageOpportunity).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.our_success == True
                )
            ).count()
            
            # Competition stats
            executed_by_others = session.query(ArbitrageOpportunity).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.was_executed == True,
                    ArbitrageOpportunity.our_success == False
                )
            ).count()
            
            # Top competitors
            competitors = session.query(
                ArbitrageOpportunity.executor_address,
                func.count(ArbitrageOpportunity.id).label('wins'),
                func.sum(ArbitrageOpportunity.net_profit_usd).label('total_profit')
            ).filter(
                and_(
                    ArbitrageOpportunity.timestamp >= cutoff_time,
                    ArbitrageOpportunity.was_executed == True,
                    ArbitrageOpportunity.executor_address != None
                )
            ).group_by(
                ArbitrageOpportunity.executor_address
            ).order_by(
                func.count(ArbitrageOpportunity.id).desc()
            ).limit(10).all()
            
            return {
                'our_attempts': our_attempts,
                'our_successes': our_successes,
                'our_success_rate': our_successes / our_attempts if our_attempts > 0 else 0,
                'executed_by_others': executed_by_others,
                'top_competitors': [
                    {
                        'address': comp.executor_address,
                        'wins': comp.wins,
                        'total_profit': float(comp.total_profit or 0)
                    }
                    for comp in competitors
                ],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()
    
    def _opportunity_to_dict(self, opp: ArbitrageOpportunity) -> Dict:
        """Convert SQLAlchemy object to dictionary"""
        return {
            'id': opp.id,
            'block_number': opp.block_number,
            'timestamp': opp.timestamp.isoformat(),
            'token_pair': opp.token_pair,
            'dex_1': opp.dex_1,
            'dex_2': opp.dex_2,
            'estimated_profit_usd': opp.estimated_profit_usd,
            'gas_cost_usd': opp.gas_cost_usd,
            'net_profit_usd': opp.net_profit_usd,
            'profit_percentage': opp.profit_percentage,
            'was_executed': opp.was_executed,
            'our_attempt': opp.our_attempt,
            'our_success': opp.our_success
        }