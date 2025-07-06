import re
import logging
import asyncio
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from .types import TaskComplexity, ModelProvider
from models import RoutingRule, ModelPerformance
from app import db

logger = logging.getLogger(__name__)

class ComplexityEvaluator(ABC):
    """Abstract base class for task complexity evaluation"""
    
    @abstractmethod
    async def evaluate(self, prompt: str, task_type: str) -> TaskComplexity:
        """Evaluate the complexity of a task"""
        pass

class SimpleComplexityEvaluator(ComplexityEvaluator):
    """Basic complexity evaluator using prompt analysis"""
    
    def __init__(self):
        # Keywords that indicate complexity
        self.complex_keywords = [
            'optimize', 'refactor', 'architecture', 'system design', 'database',
            'performance', 'scalability', 'security', 'integration', 'algorithm',
            'data structure', 'machine learning', 'artificial intelligence',
            'distributed', 'microservices', 'concurrent', 'parallel'
        ]
        
        self.simple_keywords = [
            'hello', 'simple', 'basic', 'easy', 'quick', 'small',
            'straightforward', 'minimal', 'brief'
        ]
    
    async def evaluate(self, prompt: str, task_type: str) -> TaskComplexity:
        """Evaluate complexity based on prompt analysis"""
        prompt_lower = prompt.lower()
        
        # Count complexity indicators
        complex_score = 0
        simple_score = 0
        
        # Check for complex keywords
        for keyword in self.complex_keywords:
            if keyword in prompt_lower:
                complex_score += 1
        
        # Check for simple keywords
        for keyword in self.simple_keywords:
            if keyword in prompt_lower:
                simple_score += 1
        
        # Length-based scoring
        word_count = len(prompt.split())
        if word_count > 100:
            complex_score += 2
        elif word_count > 50:
            complex_score += 1
        elif word_count < 10:
            simple_score += 1
        
        # Task type specific scoring
        if task_type in ['code_generation', 'code_optimization']:
            # Look for programming complexity indicators
            if re.search(r'\b(class|function|algorithm|optimize|refactor)\b', prompt_lower):
                complex_score += 1
        
        # Determine final complexity
        if complex_score > simple_score + 1:
            return TaskComplexity.COMPLEX
        elif simple_score > complex_score:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM

class TaskRouter:
    """Handles task routing and provider selection"""
    
    def __init__(self, config):
        self.config = config
        self.complexity_evaluator = None
        self._initialize_evaluator()
        self.performance_cache = {}
    
    def _initialize_evaluator(self):
        """Initialize the complexity evaluator (ML or Simple)"""
        try:
            # Try to use ML-based evaluator first
            from core.ml_routing.ml_complexity_evaluator import MLComplexityEvaluator
            self.complexity_evaluator = MLComplexityEvaluator()
            
            # Check if model exists
            if self.complexity_evaluator.model:
                logger.info("Using ML-based complexity evaluator")
            else:
                logger.info("ML model not found, falling back to simple evaluator")
                self.complexity_evaluator = SimpleComplexityEvaluator()
        except ImportError:
            logger.info("ML routing not available, using simple evaluator")
            self.complexity_evaluator = SimpleComplexityEvaluator()
    
    async def evaluate_and_route(
        self, 
        prompt: str, 
        task_type: str,
        available_providers: List[ModelProvider],
        complexity_override: Optional[TaskComplexity] = None
    ) -> Dict:
        """Evaluate task complexity and determine routing"""
        
        # Evaluate complexity
        if complexity_override:
            complexity = complexity_override
        else:
            complexity = await self.complexity_evaluator.evaluate(prompt, task_type)
        
        # Get routing decision
        routing_decision = await self._get_routing_decision(
            task_type, complexity, available_providers
        )
        
        return {
            'complexity': complexity,
            'selected_providers': routing_decision['providers'],
            'require_consensus': routing_decision['require_consensus'],
            'metadata': {
                'routing_strategy': routing_decision['strategy'],
                'confidence': routing_decision.get('confidence', 0.8)
            }
        }
    
    async def _get_routing_decision(
        self,
        task_type: str,
        complexity: TaskComplexity,
        available_providers: List[ModelProvider]
    ) -> Dict:
        """Get routing decision based on rules and performance data"""
        
        # Check for custom routing rules
        custom_rule = await self._get_custom_routing_rule(task_type, complexity)
        if custom_rule:
            return custom_rule
        
        # Use performance-based routing
        return await self._performance_based_routing(
            task_type, complexity, available_providers
        )
    
    async def _get_custom_routing_rule(
        self, 
        task_type: str, 
        complexity: TaskComplexity
    ) -> Optional[Dict]:
        """Check for custom routing rules"""
        
        complexity_scores = {
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MEDIUM: 0.6,
            TaskComplexity.COMPLEX: 0.9
        }
        
        complexity_score = complexity_scores[complexity]
        
        # Query database for matching rules
        rules = db.session.query(RoutingRule).filter(
            RoutingRule.task_type == task_type,
            RoutingRule.complexity_min <= complexity_score,
            RoutingRule.complexity_max >= complexity_score,
            RoutingRule.active == True
        ).order_by(RoutingRule.priority.desc()).all()
        
        if rules:
            rule = rules[0]  # Take highest priority rule
            try:
                import json
                providers = [ModelProvider(p) for p in json.loads(rule.preferred_providers)]
                return {
                    'providers': providers,
                    'require_consensus': rule.require_consensus,
                    'strategy': 'custom_rule',
                    'confidence': 0.9
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Invalid providers in routing rule {rule.id}")
        
        return None
    
    async def _performance_based_routing(
        self,
        task_type: str,
        complexity: TaskComplexity,
        available_providers: List[ModelProvider]
    ) -> Dict:
        """Route based on historical performance data"""
        
        # Get recent performance data
        performance_data = await self._get_performance_data(task_type)
        
        # Score providers based on performance
        provider_scores = {}
        for provider in available_providers:
            score = await self._calculate_provider_score(
                provider, task_type, performance_data
            )
            provider_scores[provider] = score
        
        # Select providers based on complexity
        if complexity == TaskComplexity.SIMPLE:
            # Use single best provider
            best_provider = max(provider_scores.keys(), key=lambda p: provider_scores[p])
            return {
                'providers': [best_provider],
                'require_consensus': False,
                'strategy': 'performance_single',
                'confidence': provider_scores[best_provider]
            }
        
        elif complexity == TaskComplexity.MEDIUM:
            # Use top 2 providers if consensus required
            sorted_providers = sorted(
                provider_scores.keys(), 
                key=lambda p: provider_scores[p], 
                reverse=True
            )
            
            require_consensus = self.config.routing.require_consensus_for_complex
            providers = sorted_providers[:2] if require_consensus else sorted_providers[:1]
            
            return {
                'providers': providers,
                'require_consensus': require_consensus,
                'strategy': 'performance_multi',
                'confidence': sum(provider_scores[p] for p in providers) / len(providers)
            }
        
        else:  # COMPLEX
            # Use all available providers with consensus
            return {
                'providers': available_providers,
                'require_consensus': True,
                'strategy': 'performance_consensus',
                'confidence': sum(provider_scores.values()) / len(provider_scores)
            }
    
    async def _get_performance_data(self, task_type: str) -> Dict:
        """Get recent performance data for providers"""
        
        # Use cached data if available and recent
        cache_key = f"performance_{task_type}"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # Query recent performance data
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        performance_records = db.session.query(ModelPerformance).filter(
            ModelPerformance.task_type == task_type,
            ModelPerformance.created_at >= cutoff_date
        ).all()
        
        # Aggregate performance data
        performance_data = {}
        for record in performance_records:
            provider = record.provider
            if provider not in performance_data:
                performance_data[provider] = {
                    'latency_ms': [],
                    'success_rate': 0,
                    'total_requests': 0,
                    'successful_requests': 0
                }
            
            performance_data[provider]['latency_ms'].append(record.latency_ms)
            performance_data[provider]['total_requests'] += 1
            if record.success:
                performance_data[provider]['successful_requests'] += 1
        
        # Calculate success rates
        for provider_data in performance_data.values():
            if provider_data['total_requests'] > 0:
                provider_data['success_rate'] = (
                    provider_data['successful_requests'] / provider_data['total_requests']
                )
        
        # Cache results
        self.performance_cache[cache_key] = performance_data
        
        return performance_data
    
    async def _calculate_provider_score(
        self,
        provider: ModelProvider,
        task_type: str,
        performance_data: Dict
    ) -> float:
        """Calculate a score for a provider based on performance"""
        
        provider_key = provider.value
        
        if provider_key not in performance_data:
            # No historical data, return default score
            return 0.7
        
        data = performance_data[provider_key]
        
        # Calculate success rate score (0-1)
        success_score = data.get('success_rate', 0.5)
        
        # Calculate latency score (inverted and normalized)
        latencies = data.get('latency_ms', [1000])
        avg_latency = sum(latencies) / len(latencies)
        # Normalize latency: faster = better (max score at 100ms, min at 10000ms)
        latency_score = max(0, min(1, (10000 - avg_latency) / 9900))
        
        # Combine scores with weights
        final_score = (success_score * 0.7) + (latency_score * 0.3)
        
        return final_score
