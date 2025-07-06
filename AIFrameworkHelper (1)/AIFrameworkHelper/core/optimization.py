"""
Model Optimization and Fine-tuning Module

This module handles performance-based optimization and continuous improvement
of model routing decisions based on historical performance data.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sqlalchemy import func

from .types import ModelProvider, TaskComplexity
from models import ModelPerformance, Conversation
from app import db

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimizes model routing based on performance data"""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.performance_scores = defaultdict(lambda: defaultdict(float))
        self.confidence_weights = defaultdict(lambda: defaultdict(float))
        self._initialize_scores()
    
    def _initialize_scores(self):
        """Initialize performance scores from historical data"""
        try:
            # Get recent performance data
            recent_data = db.session.query(ModelPerformance).filter(
                ModelPerformance.created_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            # Calculate initial scores
            for perf in recent_data:
                provider = perf.provider
                task_type = perf.task_type
                
                # Calculate weighted score based on success, latency, and quality
                score = self._calculate_performance_score(perf)
                self.performance_scores[provider][task_type] = score
                
        except Exception as e:
            logger.error(f"Failed to initialize performance scores: {e}")
    
    def _calculate_performance_score(self, perf: ModelPerformance) -> float:
        """Calculate a composite performance score"""
        # Base score from success/failure
        score = 1.0 if perf.success else 0.0
        
        # Adjust for latency (normalize to 0-1 range, lower is better)
        # Assume 5000ms is the worst acceptable latency
        latency_score = max(0, 1 - (perf.latency_ms / 5000))
        score *= latency_score
        
        # Include quality score if available
        if perf.quality_score is not None:
            score = (score + perf.quality_score) / 2
        
        return score
    
    async def update_performance(
        self,
        provider: ModelProvider,
        task_type: str,
        success: bool,
        latency_ms: float,
        quality_score: Optional[float] = None,
        user_feedback: Optional[float] = None
    ):
        """Update performance scores with new data"""
        
        # Calculate immediate performance score
        immediate_score = 1.0 if success else 0.0
        
        # Adjust for latency
        latency_score = max(0, 1 - (latency_ms / 5000))
        immediate_score *= latency_score
        
        # Include quality and feedback if available
        if quality_score is not None:
            immediate_score = (immediate_score + quality_score) / 2
        
        if user_feedback is not None:
            immediate_score = (immediate_score + user_feedback) / 2
        
        # Update using exponential moving average
        current_score = self.performance_scores[provider.value][task_type]
        new_score = (1 - self.learning_rate) * current_score + self.learning_rate * immediate_score
        
        self.performance_scores[provider.value][task_type] = new_score
        
        # Update confidence weight based on consistency
        self._update_confidence_weight(provider.value, task_type, immediate_score)
        
        logger.info(f"Updated {provider.value} performance for {task_type}: {new_score:.3f}")
    
    def _update_confidence_weight(self, provider: str, task_type: str, score: float):
        """Update confidence weight based on consistency"""
        current_weight = self.confidence_weights[provider][task_type]
        expected_score = self.performance_scores[provider][task_type]
        
        # Calculate variance from expected
        variance = abs(score - expected_score)
        
        # Update confidence (higher consistency = higher confidence)
        consistency_factor = 1 - variance
        new_weight = (1 - self.learning_rate) * current_weight + self.learning_rate * consistency_factor
        
        self.confidence_weights[provider][task_type] = new_weight
    
    def get_provider_ranking(
        self,
        task_type: str,
        available_providers: List[ModelProvider],
        complexity: TaskComplexity
    ) -> List[Tuple[ModelProvider, float]]:
        """Get ranked providers based on performance"""
        
        rankings = []
        
        for provider in available_providers:
            # Get base performance score
            score = self.performance_scores[provider.value].get(task_type, 0.5)
            
            # Get confidence weight
            confidence = self.confidence_weights[provider.value].get(task_type, 0.5)
            
            # Adjust score based on complexity
            if complexity == TaskComplexity.SIMPLE:
                # Prefer faster providers for simple tasks
                score *= 1.2
            elif complexity == TaskComplexity.COMPLEX:
                # Prefer accurate providers for complex tasks
                score *= confidence
            
            # Apply temporal decay to older scores
            score *= self.decay_factor
            
            rankings.append((provider, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    async def analyze_patterns(
        self,
        time_window_hours: int = 168
    ) -> Dict[str, Any]:
        """Analyze performance patterns and trends"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Get performance data
            perf_data = db.session.query(
                ModelPerformance.provider,
                ModelPerformance.task_type,
                func.count(ModelPerformance.id).label('request_count'),
                func.avg(ModelPerformance.latency_ms).label('avg_latency'),
                func.sum(ModelPerformance.success.cast(db.Integer)).label('success_count'),
                func.avg(ModelPerformance.quality_score).label('avg_quality')
            ).filter(
                ModelPerformance.created_at >= cutoff_date
            ).group_by(
                ModelPerformance.provider,
                ModelPerformance.task_type
            ).all()
            
            patterns = {
                'provider_performance': {},
                'task_type_distribution': {},
                'optimization_suggestions': []
            }
            
            for data in perf_data:
                provider_key = f"{data.provider}_{data.task_type}"
                success_rate = data.success_count / data.request_count if data.request_count > 0 else 0
                
                patterns['provider_performance'][provider_key] = {
                    'request_count': data.request_count,
                    'avg_latency': float(data.avg_latency) if data.avg_latency else 0,
                    'success_rate': float(success_rate),
                    'avg_quality': float(data.avg_quality) if data.avg_quality else None
                }
                
                # Track task type distribution
                if data.task_type not in patterns['task_type_distribution']:
                    patterns['task_type_distribution'][data.task_type] = 0
                patterns['task_type_distribution'][data.task_type] += data.request_count
            
            # Generate optimization suggestions
            patterns['optimization_suggestions'] = self._generate_suggestions(patterns['provider_performance'])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return {}
    
    def _generate_suggestions(self, performance_data: Dict) -> List[str]:
        """Generate optimization suggestions based on patterns"""
        suggestions = []
        
        # Analyze each provider-task combination
        for key, metrics in performance_data.items():
            provider, task_type = key.split('_', 1)
            
            # Check for poor performance
            if metrics['success_rate'] < 0.8:
                suggestions.append(
                    f"Consider reducing usage of {provider} for {task_type} "
                    f"(success rate: {metrics['success_rate']:.1%})"
                )
            
            # Check for high latency
            if metrics['avg_latency'] > 3000:
                suggestions.append(
                    f"{provider} has high latency for {task_type} "
                    f"({metrics['avg_latency']:.0f}ms avg)"
                )
            
            # Check for low quality
            if metrics['avg_quality'] is not None and metrics['avg_quality'] < 0.7:
                suggestions.append(
                    f"Quality issues detected with {provider} for {task_type} "
                    f"(avg quality: {metrics['avg_quality']:.2f})"
                )
        
        return suggestions


class AdaptiveRouter:
    """Adaptive routing with continuous learning"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.routing_history = defaultdict(list)
        self.exploration_rate = 0.1  # For exploration vs exploitation
    
    async def get_optimal_providers(
        self,
        task_type: str,
        complexity: TaskComplexity,
        available_providers: List[ModelProvider],
        require_consensus: bool = False
    ) -> List[ModelProvider]:
        """Get optimal providers using adaptive routing"""
        
        # Get provider rankings
        rankings = self.optimizer.get_provider_ranking(
            task_type, available_providers, complexity
        )
        
        selected_providers = []
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: randomly select providers
            np.random.shuffle(rankings)
            logger.info("Exploring alternative provider selection")
        
        # Select providers based on complexity and consensus requirement
        if complexity == TaskComplexity.SIMPLE and not require_consensus:
            # Single best provider for simple tasks
            selected_providers = [rankings[0][0]] if rankings else []
        
        elif complexity == TaskComplexity.MEDIUM:
            # Top 2 providers for medium complexity
            num_providers = 2 if require_consensus else 1
            selected_providers = [p[0] for p in rankings[:num_providers]]
        
        else:  # COMPLEX
            # Use multiple providers for complex tasks
            if require_consensus:
                # Use top 3 providers or all available if less than 3
                num_providers = min(3, len(rankings))
                selected_providers = [p[0] for p in rankings[:num_providers]]
            else:
                # Use best 2 providers
                selected_providers = [p[0] for p in rankings[:2]]
        
        # Track routing decision
        self.routing_history[task_type].append({
            'timestamp': datetime.utcnow(),
            'complexity': complexity.value,
            'selected_providers': [p.value for p in selected_providers],
            'scores': {p[0].value: p[1] for p in rankings}
        })
        
        return selected_providers
    
    def adjust_exploration_rate(self, performance_stability: float):
        """Adjust exploration rate based on performance stability"""
        # Reduce exploration as performance stabilizes
        self.exploration_rate = max(0.05, 0.2 * (1 - performance_stability))