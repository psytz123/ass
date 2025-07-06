"""
Task Router for AI Orchestration Framework

This module handles task complexity evaluation and intelligent routing
to appropriate AI model providers based on performance and complexity.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .model_connectors import ModelProvider, TaskComplexity
from .ml_complexity_evaluator import MLComplexityEvaluator
from .enhanced_performance_tracker import EnhancedPerformanceTracker
from .base_evaluator import ComplexityEvaluator # Import from new base_evaluator.py


@dataclass
class RoutingRule:
    task_type: str
    complexity: str  # "simple", "medium", "complex", or "any"
    providers: List[str]
    require_consensus: bool
    min_providers: int = 1
    priority: int = 0  # Higher priority rules are checked first


# Removed PerformanceMetric dataclass as it\\\"s now handled within EnhancedPerformanceTracker

# Removed ComplexityEvaluator class from here as it\\\"s now in base_evaluator.py

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


class TaskRouter:
    """Intelligent task routing based on complexity and performance"""

    def __init__(self):
        self.complexity_evaluators: Dict[str, ComplexityEvaluator] = {}
        self.routing_rules: List[RoutingRule] = []
        self.performance_tracker = EnhancedPerformanceTracker() # Use EnhancedPerformanceTracker
        self.default_evaluator = MLComplexityEvaluator() 
        
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
        
        # Apply performance-based reordering, passing complexity and task_type
        if len(selected_providers) > 1:
            best_providers = self.performance_tracker.get_best_providers(
                complexity=complexity, task_type=task_type
            )
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
        
        # Ensure min_providers is met, adding more if necessary and available
        while final_min_providers > len(selected_providers):
            all_providers_set = set([ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE])
            current_providers_set = set(selected_providers)
            
            added_one = False
            for provider in all_providers_set - current_providers_set:
                if available_providers is None or provider in available_providers:
                    selected_providers.append(provider)
                    added_one = True
                    break
            if not added_one: # No more providers to add
                break

        return {
            "providers": selected_providers,
            "require_consensus": final_require_consensus,
            "complexity": complexity,
            "rule_used": rule,
            "task_type": task_type,
            "min_providers": final_min_providers
        }

    def record_performance(self, provider: ModelProvider, latency: float, 
                          success: bool, confidence: float = 0.8,
                          complexity: TaskComplexity = TaskComplexity.SIMPLE,
                          task_type: str = "general"):
        """Record performance metrics for a provider"""
        self.performance_tracker.record_request(provider, latency, success, confidence, complexity, task_type)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_tracker.get_performance_metrics()



