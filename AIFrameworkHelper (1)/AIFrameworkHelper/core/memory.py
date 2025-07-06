import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import func

from models import Conversation, ModelPerformance, EmbeddingCache
from app import db
from .types import ModelResponse, ModelProvider

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation history, performance tracking, and caching"""
    
    def __init__(self, config):
        self.config = config
        self.enable_caching = config.enable_caching
        self.enable_metrics = config.enable_metrics
    
    async def store_conversation(
        self,
        user_id: str,
        prompt: str,
        response: str,
        task_type: str,
        providers_used: List[ModelProvider],
        consensus_confidence: Optional[float],
        processing_time_ms: float,
        complexity_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store conversation in database"""
        
        try:
            conversation = Conversation(
                user_id=user_id,
                prompt=prompt,
                response=response,
                task_type=task_type,
                providers_used=json.dumps([p.value for p in providers_used]),
                consensus_confidence=consensus_confidence,
                processing_time_ms=processing_time_ms,
                complexity_score=complexity_score,
                extra_metadata=metadata or {}
            )
            
            db.session.add(conversation)
            db.session.commit()
            
            logger.info(f"Stored conversation {conversation.id} for user {user_id}")
            return conversation.id
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            db.session.rollback()
            raise
    
    async def store_model_performance(
        self,
        response: ModelResponse,
        task_type: str,
        success: bool = True,
        error_message: Optional[str] = None,
        quality_score: Optional[float] = None
    ):
        """Store model performance metrics"""
        
        if not self.enable_metrics:
            return
        
        try:
            performance = ModelPerformance(
                provider=response.provider.value,
                model_name=response.model_name,
                task_type=task_type,
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
                success=success,
                error_message=error_message,
                quality_score=quality_score
            )
            
            db.session.add(performance)
            db.session.commit()
            
            logger.debug(f"Stored performance data for {response.provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to store performance data: {e}")
            db.session.rollback()
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 50,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        
        try:
            query = db.session.query(Conversation).filter(
                Conversation.user_id == user_id
            )
            
            if task_type:
                query = query.filter(Conversation.task_type == task_type)
            
            conversations = query.order_by(
                Conversation.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': conv.id,
                    'prompt': conv.prompt,
                    'response': conv.response,
                    'task_type': conv.task_type,
                    'providers_used': json.loads(conv.providers_used),
                    'consensus_confidence': conv.consensus_confidence,
                    'processing_time_ms': conv.processing_time_ms,
                    'complexity_score': conv.complexity_score,
                    'metadata': conv.extra_metadata,
                    'created_at': conv.created_at.isoformat()
                }
                for conv in conversations
            ]
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def get_performance_metrics(
        self,
        time_window_hours: int = 24,
        provider: Optional[ModelProvider] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for the specified time window"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            query = db.session.query(ModelPerformance).filter(
                ModelPerformance.created_at >= cutoff_date
            )
            
            if provider:
                query = query.filter(ModelPerformance.provider == provider.value)
            
            if task_type:
                query = query.filter(ModelPerformance.task_type == task_type)
            
            performances = query.all()
            
            # Aggregate metrics
            metrics = {
                'total_requests': len(performances),
                'successful_requests': sum(1 for p in performances if p.success),
                'success_rate': 0.0,
                'avg_latency_ms': 0.0,
                'total_tokens': sum(p.tokens_used for p in performances),
                'by_provider': {},
                'by_task_type': {}
            }
            
            if metrics['total_requests'] > 0:
                metrics['success_rate'] = metrics['successful_requests'] / metrics['total_requests']
                metrics['avg_latency_ms'] = sum(p.latency_ms for p in performances) / len(performances)
            
            # Aggregate by provider
            provider_stats = {}
            for perf in performances:
                provider = perf.provider
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        'requests': 0,
                        'successes': 0,
                        'latencies': [],
                        'tokens': 0
                    }
                
                provider_stats[provider]['requests'] += 1
                if perf.success:
                    provider_stats[provider]['successes'] += 1
                provider_stats[provider]['latencies'].append(perf.latency_ms)
                provider_stats[provider]['tokens'] += perf.tokens_used
            
            # Calculate provider metrics
            for provider, stats in provider_stats.items():
                metrics['by_provider'][provider] = {
                    'requests': stats['requests'],
                    'success_rate': stats['successes'] / stats['requests'] if stats['requests'] > 0 else 0,
                    'avg_latency_ms': sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0,
                    'total_tokens': stats['tokens']
                }
            
            # Aggregate by task type
            task_stats = {}
            for perf in performances:
                task = perf.task_type
                if task not in task_stats:
                    task_stats[task] = {
                        'requests': 0,
                        'successes': 0,
                        'latencies': [],
                        'tokens': 0
                    }
                
                task_stats[task]['requests'] += 1
                if perf.success:
                    task_stats[task]['successes'] += 1
                task_stats[task]['latencies'].append(perf.latency_ms)
                task_stats[task]['tokens'] += perf.tokens_used
            
            # Calculate task type metrics
            for task_type, stats in task_stats.items():
                metrics['by_task_type'][task_type] = {
                    'requests': stats['requests'],
                    'success_rate': stats['successes'] / stats['requests'] if stats['requests'] > 0 else 0,
                    'avg_latency_ms': sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0,
                    'total_tokens': stats['tokens']
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old conversation and performance data"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old conversations
            old_conversations = db.session.query(Conversation).filter(
                Conversation.created_at < cutoff_date
            ).count()
            
            if old_conversations > 0:
                db.session.query(Conversation).filter(
                    Conversation.created_at < cutoff_date
                ).delete()
                
                logger.info(f"Cleaned up {old_conversations} old conversations")
            
            # Clean up old performance data
            old_performance = db.session.query(ModelPerformance).filter(
                ModelPerformance.created_at < cutoff_date
            ).count()
            
            if old_performance > 0:
                db.session.query(ModelPerformance).filter(
                    ModelPerformance.created_at < cutoff_date
                ).delete()
                
                logger.info(f"Cleaned up {old_performance} old performance records")
            
            # Clean up old embeddings
            old_embeddings = db.session.query(EmbeddingCache).filter(
                EmbeddingCache.created_at < cutoff_date
            ).count()
            
            if old_embeddings > 0:
                db.session.query(EmbeddingCache).filter(
                    EmbeddingCache.created_at < cutoff_date
                ).delete()
                
                logger.info(f"Cleaned up {old_embeddings} old embedding cache entries")
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            db.session.rollback()
