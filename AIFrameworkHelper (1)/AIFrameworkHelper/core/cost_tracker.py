"""
Cost tracking module for monitoring API usage and costs across providers
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
from sqlalchemy import func, and_

from core.types import ModelProvider
from models import ModelPerformance, db

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks API usage costs across different providers"""
    
    # Pricing per 1K tokens (in USD)
    # These are approximate prices and should be updated regularly
    PROVIDER_PRICING = {
        ModelProvider.OPENAI: {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.0001, "output": 0.0003},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        },
        ModelProvider.ANTHROPIC: {
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-opus": {"input": 0.015, "output": 0.075}
        },
        ModelProvider.GOOGLE: {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
            "gemini-1.5-flash": {"input": 0.00035, "output": 0.00105}
        },
        ModelProvider.ABACUSAI: {
            "default": {"input": 0.001, "output": 0.003}  # Estimated pricing
        }
    }
    
    def __init__(self):
        self.cost_cache = {}
        self.last_cache_update = None
        self.cache_duration = timedelta(minutes=5)
    
    def calculate_request_cost(
        self,
        provider: ModelProvider,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a single request"""
        
        provider_pricing = self.PROVIDER_PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model_name)
        
        if not model_pricing:
            # Use default pricing if model not found
            model_pricing = provider_pricing.get("default", {"input": 0.001, "output": 0.003})
        
        # Calculate cost (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    async def get_cost_metrics(
        self,
        time_window_hours: int = 24,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive cost metrics"""
        
        # Check cache
        cache_key = f"{time_window_hours}_{user_id or 'all'}"
        if (self.cost_cache.get(cache_key) and 
            self.last_cache_update and 
            datetime.utcnow() - self.last_cache_update < self.cache_duration):
            return self.cost_cache[cache_key]
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Base query
            query = db.session.query(ModelPerformance).filter(
                ModelPerformance.created_at >= cutoff_date
            )
            
            if user_id:
                # Need to join with Conversation table to filter by user
                from models import Conversation
                query = query.join(
                    Conversation,
                    and_(
                        ModelPerformance.created_at >= Conversation.created_at - timedelta(seconds=5),
                        ModelPerformance.created_at <= Conversation.created_at + timedelta(seconds=5)
                    )
                ).filter(Conversation.user_id == user_id)
            
            performances = query.all()
            
            # Calculate metrics
            metrics = {
                "total_cost": 0.0,
                "total_requests": len(performances),
                "total_tokens": 0,
                "by_provider": defaultdict(lambda: {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0,
                    "models": defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0})
                }),
                "by_task_type": defaultdict(lambda: {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }),
                "cost_over_time": self._calculate_cost_over_time(performances),
                "top_expensive_requests": self._get_top_expensive_requests(performances, limit=10),
                "cost_trends": self._calculate_cost_trends(performances),
                "provider_cost_breakdown": {}
            }
            
            # Process each performance record
            for perf in performances:
                # Estimate tokens (if not available, estimate based on latency)
                tokens = perf.tokens_used if perf.tokens_used else self._estimate_tokens(perf)
                input_tokens = int(tokens * 0.3)  # Rough estimate: 30% input
                output_tokens = int(tokens * 0.7)  # 70% output
                
                # Extract model name from metadata if available
                model_name = "default"
                if perf.extra_metadata and isinstance(perf.extra_metadata, dict):
                    model_name = perf.extra_metadata.get("model", "default")
                
                # Calculate cost
                cost = self.calculate_request_cost(
                    perf.provider,
                    model_name,
                    input_tokens,
                    output_tokens
                )
                
                # Update metrics
                metrics["total_cost"] += cost
                metrics["total_tokens"] += tokens
                
                # By provider
                provider_metrics = metrics["by_provider"][perf.provider]
                provider_metrics["cost"] += cost
                provider_metrics["requests"] += 1
                provider_metrics["tokens"] += tokens
                provider_metrics["models"][model_name]["cost"] += cost
                provider_metrics["models"][model_name]["requests"] += 1
                provider_metrics["models"][model_name]["tokens"] += tokens
                
                # By task type
                task_metrics = metrics["by_task_type"][perf.task_type]
                task_metrics["cost"] += cost
                task_metrics["requests"] += 1
                task_metrics["tokens"] += tokens
            
            # Calculate provider cost breakdown percentages
            if metrics["total_cost"] > 0:
                for provider, data in metrics["by_provider"].items():
                    percentage = (data["cost"] / metrics["total_cost"]) * 100
                    metrics["provider_cost_breakdown"][provider] = {
                        "percentage": round(percentage, 2),
                        "cost": round(data["cost"], 4)
                    }
            
            # Convert defaultdicts to regular dicts
            metrics["by_provider"] = dict(metrics["by_provider"])
            metrics["by_task_type"] = dict(metrics["by_task_type"])
            
            # Round costs
            metrics["total_cost"] = round(metrics["total_cost"], 4)
            
            # Cache results
            self.cost_cache[cache_key] = metrics
            self.last_cache_update = datetime.utcnow()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate cost metrics: {e}")
            return {
                "total_cost": 0.0,
                "total_requests": 0,
                "total_tokens": 0,
                "by_provider": {},
                "by_task_type": {},
                "cost_over_time": [],
                "error": str(e)
            }
    
    def _estimate_tokens(self, performance: ModelPerformance) -> int:
        """Estimate tokens based on latency if not provided"""
        # Rough estimation: ~100 tokens per second of processing
        return int(performance.latency_ms / 10)
    
    def _calculate_cost_over_time(
        self,
        performances: List[ModelPerformance]
    ) -> List[Dict[str, Any]]:
        """Calculate cost trends over time"""
        
        if not performances:
            return []
        
        # Group by hour
        hourly_costs = defaultdict(float)
        
        for perf in performances:
            hour_key = perf.created_at.replace(minute=0, second=0, microsecond=0)
            
            # Estimate cost
            tokens = perf.tokens_used if perf.tokens_used else self._estimate_tokens(perf)
            input_tokens = int(tokens * 0.3)
            output_tokens = int(tokens * 0.7)
            
            model_name = "default"
            if perf.extra_metadata and isinstance(perf.extra_metadata, dict):
                model_name = perf.extra_metadata.get("model", "default")
            
            cost = self.calculate_request_cost(
                perf.provider,
                model_name,
                input_tokens,
                output_tokens
            )
            
            hourly_costs[hour_key] += cost
        
        # Convert to sorted list
        cost_timeline = [
            {
                "timestamp": timestamp.isoformat(),
                "cost": round(cost, 4)
            }
            for timestamp, cost in sorted(hourly_costs.items())
        ]
        
        return cost_timeline
    
    def _get_top_expensive_requests(
        self,
        performances: List[ModelPerformance],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the most expensive requests"""
        
        request_costs = []
        
        for perf in performances:
            tokens = perf.tokens_used if perf.tokens_used else self._estimate_tokens(perf)
            input_tokens = int(tokens * 0.3)
            output_tokens = int(tokens * 0.7)
            
            model_name = "default"
            if perf.extra_metadata and isinstance(perf.extra_metadata, dict):
                model_name = perf.extra_metadata.get("model", "default")
            
            cost = self.calculate_request_cost(
                perf.provider,
                model_name,
                input_tokens,
                output_tokens
            )
            
            request_costs.append({
                "provider": perf.provider,
                "model": model_name,
                "task_type": perf.task_type,
                "tokens": tokens,
                "cost": round(cost, 4),
                "timestamp": perf.created_at.isoformat(),
                "success": perf.success
            })
        
        # Sort by cost descending
        request_costs.sort(key=lambda x: x["cost"], reverse=True)
        
        return request_costs[:limit]
    
    def _calculate_cost_trends(
        self,
        performances: List[ModelPerformance]
    ) -> Dict[str, Any]:
        """Calculate cost trends and projections"""
        
        if not performances:
            return {"daily_average": 0, "projected_monthly": 0}
        
        # Calculate daily average
        days_span = max(1, (performances[-1].created_at - performances[0].created_at).days)
        total_cost = sum(
            self.calculate_request_cost(
                perf.provider,
                "default",
                int((perf.tokens_used or self._estimate_tokens(perf)) * 0.3),
                int((perf.tokens_used or self._estimate_tokens(perf)) * 0.7)
            )
            for perf in performances
        )
        
        daily_average = total_cost / days_span if days_span > 0 else total_cost
        projected_monthly = daily_average * 30
        
        return {
            "daily_average": round(daily_average, 4),
            "projected_monthly": round(projected_monthly, 2)
        }
    
    async def get_budget_alerts(
        self,
        daily_budget: float = 10.0,
        monthly_budget: float = 300.0
    ) -> Dict[str, Any]:
        """Check if costs are exceeding budget thresholds"""
        
        # Get current costs
        daily_metrics = await self.get_cost_metrics(time_window_hours=24)
        monthly_metrics = await self.get_cost_metrics(time_window_hours=720)  # 30 days
        
        alerts = {
            "daily": {
                "budget": daily_budget,
                "spent": daily_metrics["total_cost"],
                "percentage": (daily_metrics["total_cost"] / daily_budget * 100) if daily_budget > 0 else 0,
                "exceeded": daily_metrics["total_cost"] > daily_budget
            },
            "monthly": {
                "budget": monthly_budget,
                "spent": monthly_metrics["total_cost"],
                "percentage": (monthly_metrics["total_cost"] / monthly_budget * 100) if monthly_budget > 0 else 0,
                "exceeded": monthly_metrics["total_cost"] > monthly_budget
            },
            "projected_monthly": daily_metrics.get("cost_trends", {}).get("projected_monthly", 0),
            "recommendations": []
        }
        
        # Add recommendations
        if alerts["daily"]["exceeded"]:
            alerts["recommendations"].append(
                f"Daily budget exceeded! Consider reducing usage or optimizing provider selection."
            )
        
        if alerts["monthly"]["percentage"] > 80:
            alerts["recommendations"].append(
                f"Monthly budget usage at {alerts['monthly']['percentage']:.1f}%. Monitor closely."
            )
        
        if alerts["projected_monthly"] > monthly_budget:
            alerts["recommendations"].append(
                f"Projected monthly cost (${alerts['projected_monthly']:.2f}) exceeds budget."
            )
        
        return alerts