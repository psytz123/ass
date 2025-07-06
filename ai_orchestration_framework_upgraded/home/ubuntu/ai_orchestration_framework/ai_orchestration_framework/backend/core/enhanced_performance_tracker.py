
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .model_connectors import ModelProvider, TaskComplexity


@dataclass
class PerformanceMetric:
    provider: ModelProvider
    avg_latency: float
    success_rate: float
    avg_confidence: float
    total_requests: int
    last_updated: float
    # Contextual metrics
    contextual_metrics: Dict[Tuple[TaskComplexity, str], Dict[str, Any]] # (complexity, task_type) -> metrics


class EnhancedPerformanceTracker:
    """Tracks performance metrics for model providers with enhanced capabilities"""

    def __init__(self):
        self.metrics: Dict[ModelProvider, PerformanceMetric] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.decay_factor = 0.95  # For Exponentially Weighted Moving Average (EWMA)

    def record_request(self, provider: ModelProvider, latency: float,
                      success: bool, confidence: float = 0.8,
                      complexity: TaskComplexity = TaskComplexity.SIMPLE,
                      task_type: str = "general"):
        """Record a request for performance tracking"""
        current_time = time.time()

        # Update global metrics
        if provider not in self.metrics:
            self.metrics[provider] = PerformanceMetric(
                provider=provider,
                avg_latency=latency,
                success_rate=1.0 if success else 0.0,
                avg_confidence=confidence,
                total_requests=1,
                last_updated=current_time,
                contextual_metrics={}
            )
        else:
            metric = self.metrics[provider]
            total = metric.total_requests

            # Apply EWMA for global metrics
            metric.avg_latency = (self.decay_factor * metric.avg_latency) + ((1 - self.decay_factor) * latency)
            metric.avg_confidence = (self.decay_factor * metric.avg_confidence) + ((1 - self.decay_factor) * confidence)
            metric.success_rate = (self.decay_factor * metric.success_rate) + ((1 - self.decay_factor) * (1.0 if success else 0.0))
            metric.total_requests += 1
            metric.last_updated = current_time

        # Update contextual metrics
        context_key = (complexity, task_type)
        if context_key not in self.metrics[provider].contextual_metrics:
            self.metrics[provider].contextual_metrics[context_key] = {
                "avg_latency": latency,
                "success_rate": 1.0 if success else 0.0,
                "avg_confidence": confidence,
                "total_requests": 1,
                "last_updated": current_time
            }
        else:
            context_metric = self.metrics[provider].contextual_metrics[context_key]
            total_context = context_metric["total_requests"]

            # Apply EWMA for contextual metrics
            context_metric["avg_latency"] = (self.decay_factor * context_metric["avg_latency"]) + ((1 - self.decay_factor) * latency)
            context_metric["avg_confidence"] = (self.decay_factor * context_metric["avg_confidence"]) + ((1 - self.decay_factor) * confidence)
            context_metric["success_rate"] = (self.decay_factor * context_metric["success_rate"]) + ((1 - self.decay_factor) * (1.0 if success else 0.0))
            context_metric["total_requests"] += 1
            context_metric["last_updated"] = current_time

        # Store in history (for potential time series analysis or debugging)
        self.request_history.append({
            "provider": provider,
            "latency": latency,
            "success": success,
            "confidence": confidence,
            "complexity": complexity,
            "task_type": task_type,
            "timestamp": current_time
        })

        # Keep only last 10000 requests to prevent excessive memory usage
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-10000:]

    def get_performance_score(self, provider: ModelProvider,
                              complexity: TaskComplexity = TaskComplexity.SIMPLE,
                              task_type: str = "general",
                              latency_weight: float = 0.3,
                              success_weight: float = 0.4,
                              confidence_weight: float = 0.3) -> float:
        """Calculate overall performance score for a provider, considering context and dynamic weights"""
        if provider not in self.metrics:
            return 0.5  # Default score for unknown providers

        context_key = (complexity, task_type)
        metric_data = self.metrics[provider].contextual_metrics.get(context_key, None)

        if metric_data is None or metric_data["total_requests"] < 5: # Fallback to global if not enough contextual data
            metric_data = self.metrics[provider]

        # Normalize metrics (lower latency is better, higher success/confidence is better)
        # Max latency for normalization, e.g., 5000 ms (5 seconds)
        max_latency_ms = 5000.0
        latency_score = max(0.0, 1.0 - (metric_data["avg_latency"] / max_latency_ms))
        success_score = metric_data["success_rate"]
        confidence_score = metric_data["avg_confidence"]

        # Weighted combination
        score = (
            latency_score * latency_weight +
            success_score * success_weight +
            confidence_score * confidence_weight
        )
        return score

    def get_best_providers(self, count: int = 3,
                           complexity: TaskComplexity = TaskComplexity.SIMPLE,
                           task_type: str = "general",
                           latency_weight: float = 0.3,
                           success_weight: float = 0.4,
                           confidence_weight: float = 0.3) -> List[ModelProvider]:
        """Get the best performing providers based on contextual scores"""
        if not self.metrics:
            return []

        providers_with_scores = [
            (provider, self.get_performance_score(
                provider, complexity, task_type, latency_weight, success_weight, confidence_weight
            ))
            for provider in self.metrics.keys()
        ]

        # Sort by score descending
        providers_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [provider for provider, _ in providers_with_scores[:count]]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics, including contextual ones"""
        metrics_output = {}
        for provider, metric in self.metrics.items():
            provider_data = {
                "avg_latency": metric.avg_latency,
                "success_rate": metric.success_rate,
                "avg_confidence": metric.avg_confidence,
                "total_requests": metric.total_requests,
                "last_updated": metric.last_updated,
                "global_performance_score": self.get_performance_score(provider, TaskComplexity.SIMPLE, "general", 0.3, 0.4, 0.3) # Default weights for global score
            }
            contextual_data = {}
            for (comp, tt), c_metric in metric.contextual_metrics.items():
                contextual_data[f"{comp.value}_{tt}"] = {
                    "avg_latency": c_metric["avg_latency"],
                    "success_rate": c_metric["success_rate"],
                    "avg_confidence": c_metric["avg_confidence"],
                    "total_requests": c_metric["total_requests"],
                    "last_updated": c_metric["last_updated"],
                    "performance_score": self.get_performance_score(provider, comp, tt) # Using default weights for contextual score
                }
            provider_data["contextual_metrics"] = contextual_data
            metrics_output[provider.value] = provider_data
        return metrics_output


# Example Usage (for testing)
async def main():
    tracker = EnhancedPerformanceTracker()

    # Simulate some requests
    tracker.record_request(ModelProvider.OPENAI, 150, True, 0.9, TaskComplexity.SIMPLE, "general")
    tracker.record_request(ModelProvider.OPENAI, 200, True, 0.85, TaskComplexity.SIMPLE, "general")
    tracker.record_request(ModelProvider.ANTHROPIC, 300, True, 0.92, TaskComplexity.MEDIUM, "document_analysis")
    tracker.record_request(ModelProvider.GOOGLE, 180, False, 0.7, TaskComplexity.SIMPLE, "general")
    tracker.record_request(ModelProvider.OPENAI, 1000, True, 0.75, TaskComplexity.COMPLEX, "code_generation")
    tracker.record_request(ModelProvider.OPENAI, 1200, False, 0.6, TaskComplexity.COMPLEX, "code_generation")

    print("\n--- Performance Metrics ---")
    print(tracker.get_performance_metrics())

    print("\n--- Best Providers for Simple General Task ---")
    print(tracker.get_best_providers(complexity=TaskComplexity.SIMPLE, task_type="general"))

    print("\n--- Best Providers for Complex Code Generation Task ---")
    print(tracker.get_best_providers(complexity=TaskComplexity.COMPLEX, task_type="code_generation"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


