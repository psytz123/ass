"""
Task Router for AI Orchestration Framework

This module handles task complexity evaluation and intelligent routing
to appropriate AI model providers based on performance and complexity.
"""

import re
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .model_connectors import ModelProvider, TaskComplexity


@dataclass
class RoutingRule:
    task_type: str
    complexity: str  # "simple", "medium", "complex", or "any"
    providers: List[str]
    require_consensus: bool
    min_providers: int = 1
    priority: int = 0  # Higher priority rules are checked first


@dataclass
class PerformanceMetric:
    provider: ModelProvider
    avg_latency: float
    success_rate: float
    avg_confidence: float
    total_requests: int
    last_updated: float


class ComplexityEvaluator(ABC):
    """Abstract base class for task complexity evaluation"""

    @abstractmethod
    async def evaluate(self, prompt: str, task_type: str = "general") -> TaskComplexity:
        """Evaluate the complexity of a given prompt"""
        pass


class KeywordComplexityEvaluator(ComplexityEvaluator):
    """Evaluates complexity based on keyword analysis"""

    def __init__(self):
        self.simple_patterns = [
            r'\b(summary|summarize|list|extract|simple|basic|quick)\b',
            r'\b(what is|who is|when is|where is)\b',
            r'\b(define|explain briefly)\b'
        ]
        
        self.medium_patterns = [
            r'\b(analyze|compare|contrast|evaluate|assess)\b',
            r'\b(how to|step by step|process|method)\b',
            r'\b(pros and cons|advantages|disadvantages)\b'
        ]
        
        self.complex_patterns = [
            r'\b(strategy|optimize|design|architect|implement)\b',
            r'\b(complex|comprehensive|detailed|thorough)\b',
            r'\b(multi-step|integration|system|framework)\b',
            r'\b(code|programming|algorithm|technical)\b'
        ]

    async def evaluate(self, prompt: str, task_type: str = "general") -> TaskComplexity:
        prompt_lower = prompt.lower()
        
        # Count pattern matches
        simple_matches = sum(1 for pattern in self.simple_patterns 
                           if re.search(pattern, prompt_lower))
        medium_matches = sum(1 for pattern in self.medium_patterns 
                           if re.search(pattern, prompt_lower))
        complex_matches = sum(1 for pattern in self.complex_patterns 
                            if re.search(pattern, prompt_lower))
        
        # Length-based complexity
        word_count = len(prompt.split())
        if word_count > 100:
            complex_matches += 1
        elif word_count > 50:
            medium_matches += 1
        else:
            simple_matches += 1
        
        # Task type specific adjustments
        if task_type in ["code_generation", "technical_analysis"]:
            complex_matches += 2
        elif task_type in ["business_automation", "document_analysis"]:
            medium_matches += 1
        
        # Determine complexity based on highest score
        if complex_matches > max(simple_matches, medium_matches):
            return TaskComplexity.COMPLEX
        elif medium_matches > simple_matches:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE


class PerformanceTracker:
    """Tracks performance metrics for model providers"""

    def __init__(self):
        self.metrics: Dict[ModelProvider, PerformanceMetric] = {}
        self.request_history: List[Dict[str, Any]] = []

    def record_request(self, provider: ModelProvider, latency: float, 
                      success: bool, confidence: float = 0.8):
        """Record a request for performance tracking"""
        import time
        
        # Update metrics
        if provider not in self.metrics:
            self.metrics[provider] = PerformanceMetric(
                provider=provider,
                avg_latency=latency,
                success_rate=1.0 if success else 0.0,
                avg_confidence=confidence,
                total_requests=1,
                last_updated=time.time()
            )
        else:
            metric = self.metrics[provider]
            total = metric.total_requests
            
            # Update running averages
            metric.avg_latency = (metric.avg_latency * total + latency) / (total + 1)
            metric.avg_confidence = (metric.avg_confidence * total + confidence) / (total + 1)
            metric.success_rate = (metric.success_rate * total + (1.0 if success else 0.0)) / (total + 1)
            metric.total_requests += 1
            metric.last_updated = time.time()
        
        # Store in history
        self.request_history.append({
            "provider": provider,
            "latency": latency,
            "success": success,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

    def get_performance_score(self, provider: ModelProvider) -> float:
        """Calculate overall performance score for a provider"""
        if provider not in self.metrics:
            return 0.5  # Default score for unknown providers
        
        metric = self.metrics[provider]
        
        # Normalize metrics (lower latency is better)
        latency_score = max(0, 1 - (metric.avg_latency / 10000))  # Normalize to 10s max
        success_score = metric.success_rate
        confidence_score = metric.avg_confidence
        
        # Weighted combination
        return (latency_score * 0.3 + success_score * 0.4 + confidence_score * 0.3)

    def get_best_providers(self, count: int = 3) -> List[ModelProvider]:
        """Get the best performing providers"""
        if not self.metrics:
            return []
        
        providers_with_scores = [
            (provider, self.get_performance_score(provider))
            for provider in self.metrics.keys()
        ]
        
        # Sort by score descending
        providers_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [provider for provider, _ in providers_with_scores[:count]]


class TaskRouter:
    """Intelligent task routing based on complexity and performance"""

    def __init__(self):
        self.complexity_evaluators: Dict[str, ComplexityEvaluator] = {}
        self.routing_rules: List[RoutingRule] = []
        self.performance_tracker = PerformanceTracker()
        self.default_evaluator = KeywordComplexityEvaluator()
        
        # Default routing rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default routing rules"""
        self.routing_rules = [
            RoutingRule(
                task_type="code_generation",
                complexity="simple",
                providers=["openai"],
                require_consensus=False,
                priority=10
            ),
            RoutingRule(
                task_type="code_generation",
                complexity="complex",
                providers=["openai", "anthropic"],
                require_consensus=True,
                min_providers=2,
                priority=10
            ),
            RoutingRule(
                task_type="business_automation",
                complexity="any",
                providers=["anthropic", "openai"],
                require_consensus=False,
                priority=5
            ),
            RoutingRule(
                task_type="document_analysis",
                complexity="any",
                providers=["anthropic", "google"],
                require_consensus=True,
                min_providers=2,
                priority=5
            ),
            RoutingRule(
                task_type="general",
                complexity="simple",
                providers=["openai", "google"],
                require_consensus=False,
                priority=1
            ),
            RoutingRule(
                task_type="general",
                complexity="medium",
                providers=["openai", "anthropic"],
                require_consensus=False,
                priority=1
            ),
            RoutingRule(
                task_type="general",
                complexity="complex",
                providers=["openai", "anthropic", "google"],
                require_consensus=True,
                min_providers=2,
                priority=1
            )
        ]

    def register_evaluator(self, task_type: str, evaluator: ComplexityEvaluator):
        """Register a complexity evaluator for a specific task type"""
        self.complexity_evaluators[task_type] = evaluator

    def add_routing_rule(self, rule: RoutingRule):
        """Add a custom routing rule"""
        self.routing_rules.append(rule)
        # Sort by priority (higher first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    async def evaluate_complexity(self, prompt: str, task_type: str = "general") -> TaskComplexity:
        """Evaluate the complexity of a prompt"""
        evaluator = self.complexity_evaluators.get(task_type, self.default_evaluator)
        return await evaluator.evaluate(prompt, task_type)

    def find_matching_rule(self, task_type: str, complexity: TaskComplexity) -> Optional[RoutingRule]:
        """Find the best matching routing rule"""
        for rule in self.routing_rules:
            if (rule.task_type == task_type or rule.task_type == "general") and \
               (rule.complexity == "any" or rule.complexity == complexity.value):
                return rule
        
        # Fallback to general rules
        for rule in self.routing_rules:
            if rule.task_type == "general" and \
               (rule.complexity == "any" or rule.complexity == complexity.value):
                return rule
        
        return None

    async def route_task(self, prompt: str, task_type: str = "general", 
                        available_providers: List[ModelProvider] = None,
                        require_consensus: Optional[bool] = None,
                        min_providers: Optional[int] = None) -> Dict[str, Any]:
        """Route a task to appropriate providers"""
        
        # Evaluate complexity
        complexity = await self.evaluate_complexity(prompt, task_type)
        
        # Find matching rule
        rule = self.find_matching_rule(task_type, complexity)
        
        if not rule:
            # Default fallback
            rule = RoutingRule(
                task_type="general",
                complexity="any",
                providers=["openai"],
                require_consensus=False
            )
        
        # Convert string providers to ModelProvider enum
        selected_providers = []
        for provider_str in rule.providers:
            try:
                provider = ModelProvider(provider_str)
                if available_providers is None or provider in available_providers:
                    selected_providers.append(provider)
            except ValueError:
                continue
        
        # Apply performance-based reordering
        if len(selected_providers) > 1:
            best_providers = self.performance_tracker.get_best_providers()
            # Reorder selected providers based on performance
            performance_ordered = [p for p in best_providers if p in selected_providers]
            remaining = [p for p in selected_providers if p not in performance_ordered]
            selected_providers = performance_ordered + remaining
        
        # Override rule settings if specified
        final_require_consensus = require_consensus if require_consensus is not None else rule.require_consensus
        final_min_providers = min_providers if min_providers is not None else rule.min_providers
        
        # Ensure we have enough providers for consensus if required
        if final_require_consensus and len(selected_providers) < 2:
            # Add more providers if available
            all_providers = [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
            for provider in all_providers:
                if provider not in selected_providers:
                    if available_providers is None or provider in available_providers:
                        selected_providers.append(provider)
                        if len(selected_providers) >= 2:
                            break
        
        return {
            "providers": selected_providers[:final_min_providers] if final_min_providers > 0 else selected_providers,
            "require_consensus": final_require_consensus,
            "complexity": complexity,
            "rule_used": rule,
            "task_type": task_type
        }

    def record_performance(self, provider: ModelProvider, latency: float, 
                          success: bool, confidence: float = 0.8):
        """Record performance metrics for a provider"""
        self.performance_tracker.record_request(provider, latency, success, confidence)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {}
        for provider, metric in self.performance_tracker.metrics.items():
            metrics[provider.value] = {
                "avg_latency": metric.avg_latency,
                "success_rate": metric.success_rate,
                "avg_confidence": metric.avg_confidence,
                "total_requests": metric.total_requests,
                "performance_score": self.performance_tracker.get_performance_score(provider)
            }
        return metrics

