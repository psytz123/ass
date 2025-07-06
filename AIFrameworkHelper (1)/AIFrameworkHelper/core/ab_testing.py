"""
A/B Testing module for comparing different routing strategies
"""

import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum
import asyncio
from sqlalchemy import func, and_

from core.types import ModelProvider, TaskComplexity
from models import db, ModelPerformance, Conversation

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Available routing strategies for testing"""
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_FOCUSED = "quality_focused"
    RANDOM = "random"
    CONSENSUS_HEAVY = "consensus_heavy"
    SINGLE_PROVIDER = "single_provider"

class ABTestGroup:
    """Represents an A/B test group with specific configuration"""
    
    def __init__(
        self,
        name: str,
        strategy: RoutingStrategy,
        config: Dict[str, Any],
        allocation_percentage: float
    ):
        self.name = name
        self.strategy = strategy
        self.config = config
        self.allocation_percentage = allocation_percentage
        self.metrics = defaultdict(lambda: {
            "requests": 0,
            "successes": 0,
            "total_latency": 0,
            "total_cost": 0,
            "quality_scores": []
        })

class ABTestingFramework:
    """Manages A/B testing for routing strategies"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        self.user_assignments = {}  # Track which users are in which test groups
    
    def create_test(
        self,
        test_id: str,
        name: str,
        groups: List[ABTestGroup],
        duration_hours: int = 24,
        min_samples_per_group: int = 100
    ) -> Dict[str, Any]:
        """Create a new A/B test"""
        
        # Validate allocation percentages
        total_allocation = sum(group.allocation_percentage for group in groups)
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Group allocations must sum to 1.0, got {total_allocation}")
        
        test = {
            "id": test_id,
            "name": name,
            "groups": {group.name: group for group in groups},
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow() + timedelta(hours=duration_hours),
            "duration_hours": duration_hours,
            "min_samples_per_group": min_samples_per_group,
            "status": "active",
            "total_requests": 0
        }
        
        self.active_tests[test_id] = test
        logger.info(f"Created A/B test '{name}' with {len(groups)} groups")
        
        return {
            "test_id": test_id,
            "name": name,
            "groups": [g.name for g in groups],
            "duration_hours": duration_hours,
            "status": "active"
        }
    
    def assign_user_to_group(
        self,
        test_id: str,
        user_id: str
    ) -> Optional[ABTestGroup]:
        """Assign a user to a test group"""
        
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Check if test is still active
        if datetime.utcnow() > test["end_time"]:
            test["status"] = "completed"
            return None
        
        # Check if user already assigned
        user_test_key = f"{test_id}:{user_id}"
        if user_test_key in self.user_assignments:
            group_name = self.user_assignments[user_test_key]
            return test["groups"][group_name]
        
        # Assign user to group based on allocation
        rand = random.random()
        cumulative = 0.0
        
        for group in test["groups"].values():
            cumulative += group.allocation_percentage
            if rand <= cumulative:
                self.user_assignments[user_test_key] = group.name
                logger.debug(f"Assigned user {user_id} to group {group.name} in test {test_id}")
                return group
        
        # Fallback to last group
        last_group = list(test["groups"].values())[-1]
        self.user_assignments[user_test_key] = last_group.name
        return last_group
    
    def get_routing_config_for_user(
        self,
        user_id: str,
        task_type: str,
        complexity: TaskComplexity
    ) -> Dict[str, Any]:
        """Get routing configuration based on user's test group"""
        
        config = {
            "strategy": RoutingStrategy.PERFORMANCE_BASED,
            "providers": None,
            "require_consensus": None,
            "test_group": None
        }
        
        # Check all active tests
        for test_id, test in self.active_tests.items():
            if test["status"] != "active":
                continue
            
            group = self.assign_user_to_group(test_id, user_id)
            if not group:
                continue
            
            config["test_group"] = f"{test_id}:{group.name}"
            
            # Apply group-specific routing configuration
            if group.strategy == RoutingStrategy.COST_OPTIMIZED:
                # Prefer cheaper providers
                config["strategy"] = group.strategy
                config["providers"] = [ModelProvider.GOOGLE, ModelProvider.OPENAI]
                config["require_consensus"] = False
                
            elif group.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
                # Prefer fastest providers
                config["strategy"] = group.strategy
                config["providers"] = [ModelProvider.OPENAI]
                config["require_consensus"] = False
                
            elif group.strategy == RoutingStrategy.QUALITY_FOCUSED:
                # Use multiple providers with consensus
                config["strategy"] = group.strategy
                config["providers"] = [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
                config["require_consensus"] = True
                
            elif group.strategy == RoutingStrategy.CONSENSUS_HEAVY:
                # Always use consensus with at least 3 providers
                config["strategy"] = group.strategy
                config["require_consensus"] = True
                config["min_providers"] = 3
                
            elif group.strategy == RoutingStrategy.SINGLE_PROVIDER:
                # Use only one provider
                config["strategy"] = group.strategy
                config["providers"] = [group.config.get("provider", ModelProvider.OPENAI)]
                config["require_consensus"] = False
                
            elif group.strategy == RoutingStrategy.RANDOM:
                # Random provider selection
                available = [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
                config["strategy"] = group.strategy
                config["providers"] = [random.choice(available)]
                config["require_consensus"] = False
            
            # Apply any custom configuration from the group
            config.update(group.config)
            
            break  # Use first matching test
        
        return config
    
    def record_request_outcome(
        self,
        test_group: str,
        success: bool,
        latency_ms: float,
        cost: float,
        quality_score: Optional[float] = None
    ):
        """Record the outcome of a request for a test group"""
        
        if not test_group or ":" not in test_group:
            return
        
        test_id, group_name = test_group.split(":", 1)
        
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        if group_name not in test["groups"]:
            return
        
        group = test["groups"][group_name]
        metrics = group.metrics[datetime.utcnow().date()]
        
        # Update metrics
        metrics["requests"] += 1
        if success:
            metrics["successes"] += 1
        metrics["total_latency"] += latency_ms
        metrics["total_cost"] += cost
        if quality_score is not None:
            metrics["quality_scores"].append(quality_score)
        
        test["total_requests"] += 1
        
        # Check if test should be concluded
        self._check_test_completion(test_id)
    
    def _check_test_completion(self, test_id: str):
        """Check if a test should be marked as complete"""
        
        test = self.active_tests.get(test_id)
        if not test or test["status"] != "active":
            return
        
        # Check time-based completion
        if datetime.utcnow() > test["end_time"]:
            self._complete_test(test_id)
            return
        
        # Check sample-based completion
        min_samples = test["min_samples_per_group"]
        all_groups_have_enough = True
        
        for group in test["groups"].values():
            total_requests = sum(
                metrics["requests"] 
                for metrics in group.metrics.values()
            )
            if total_requests < min_samples:
                all_groups_have_enough = False
                break
        
        if all_groups_have_enough:
            self._complete_test(test_id)
    
    def _complete_test(self, test_id: str):
        """Mark a test as complete and calculate results"""
        
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        test["status"] = "completed"
        test["end_time"] = datetime.utcnow()
        
        # Calculate results for each group
        results = {
            "test_id": test_id,
            "name": test["name"],
            "duration_hours": (test["end_time"] - test["start_time"]).total_seconds() / 3600,
            "total_requests": test["total_requests"],
            "groups": {}
        }
        
        for group_name, group in test["groups"].items():
            # Aggregate all metrics
            total_requests = 0
            total_successes = 0
            total_latency = 0
            total_cost = 0
            all_quality_scores = []
            
            for daily_metrics in group.metrics.values():
                total_requests += daily_metrics["requests"]
                total_successes += daily_metrics["successes"]
                total_latency += daily_metrics["total_latency"]
                total_cost += daily_metrics["total_cost"]
                all_quality_scores.extend(daily_metrics["quality_scores"])
            
            # Calculate aggregate metrics
            group_results = {
                "strategy": group.strategy.value,
                "requests": total_requests,
                "success_rate": total_successes / total_requests if total_requests > 0 else 0,
                "avg_latency_ms": total_latency / total_requests if total_requests > 0 else 0,
                "total_cost": total_cost,
                "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
                "avg_quality_score": sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else None,
                "allocation_percentage": group.allocation_percentage
            }
            
            results["groups"][group_name] = group_results
        
        # Determine winner based on composite score
        results["winner"] = self._determine_winner(results["groups"])
        results["statistical_significance"] = self._calculate_significance(results["groups"])
        
        self.test_results[test_id] = results
        logger.info(f"Completed A/B test '{test['name']}' - Winner: {results['winner']}")
        
        return results
    
    def _determine_winner(self, group_results: Dict[str, Dict]) -> str:
        """Determine the winning group based on composite score"""
        
        best_score = -1
        winner = None
        
        for group_name, results in group_results.items():
            # Calculate composite score (weighted average)
            # Higher success rate, lower latency, lower cost, higher quality = better
            score = 0
            
            # Success rate (40% weight)
            score += results["success_rate"] * 0.4
            
            # Latency score (30% weight) - inverted and normalized
            if results["avg_latency_ms"] > 0:
                latency_score = max(0, 1 - (results["avg_latency_ms"] / 5000))  # 5s as max
                score += latency_score * 0.3
            
            # Cost score (20% weight) - inverted and normalized
            if results["avg_cost_per_request"] > 0:
                cost_score = max(0, 1 - (results["avg_cost_per_request"] / 0.1))  # $0.10 as max
                score += cost_score * 0.2
            
            # Quality score (10% weight)
            if results["avg_quality_score"] is not None:
                score += results["avg_quality_score"] * 0.1
            
            if score > best_score:
                best_score = score
                winner = group_name
        
        return winner
    
    def _calculate_significance(self, group_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate statistical significance of results"""
        
        # Simple confidence calculation based on sample size
        # In production, use proper statistical tests (chi-square, t-test, etc.)
        
        significance = {
            "confidence_level": 0.0,
            "is_significant": False,
            "min_samples_reached": True
        }
        
        # Check sample sizes
        for results in group_results.values():
            if results["requests"] < 100:
                significance["min_samples_reached"] = False
                break
        
        if significance["min_samples_reached"]:
            # Simple confidence based on difference in success rates
            success_rates = [r["success_rate"] for r in group_results.values()]
            if len(success_rates) >= 2:
                max_rate = max(success_rates)
                second_rate = sorted(success_rates, reverse=True)[1]
                difference = max_rate - second_rate
                
                # Rough confidence calculation
                if difference > 0.1:
                    significance["confidence_level"] = 0.95
                    significance["is_significant"] = True
                elif difference > 0.05:
                    significance["confidence_level"] = 0.80
                else:
                    significance["confidence_level"] = 0.50
        
        return significance
    
    async def get_test_results(
        self,
        test_id: Optional[str] = None,
        include_active: bool = True
    ) -> Dict[str, Any]:
        """Get results for completed tests"""
        
        if test_id:
            # Get specific test
            if test_id in self.test_results:
                return self.test_results[test_id]
            elif test_id in self.active_tests and include_active:
                # Calculate interim results for active test
                return self._calculate_interim_results(test_id)
            else:
                return {"error": f"Test {test_id} not found"}
        
        # Get all tests
        all_results = {
            "completed_tests": list(self.test_results.values()),
            "active_tests": []
        }
        
        if include_active:
            for test_id in self.active_tests:
                if self.active_tests[test_id]["status"] == "active":
                    interim = self._calculate_interim_results(test_id)
                    all_results["active_tests"].append(interim)
        
        return all_results
    
    def _calculate_interim_results(self, test_id: str) -> Dict[str, Any]:
        """Calculate interim results for an active test"""
        
        test = self.active_tests.get(test_id)
        if not test:
            return {}
        
        results = {
            "test_id": test_id,
            "name": test["name"],
            "status": "active",
            "duration_so_far_hours": (datetime.utcnow() - test["start_time"]).total_seconds() / 3600,
            "total_requests": test["total_requests"],
            "groups": {}
        }
        
        # Calculate current metrics for each group
        for group_name, group in test["groups"].items():
            total_requests = 0
            total_successes = 0
            total_latency = 0
            total_cost = 0
            
            for daily_metrics in group.metrics.values():
                total_requests += daily_metrics["requests"]
                total_successes += daily_metrics["successes"]
                total_latency += daily_metrics["total_latency"]
                total_cost += daily_metrics["total_cost"]
            
            results["groups"][group_name] = {
                "strategy": group.strategy.value,
                "requests": total_requests,
                "success_rate": total_successes / total_requests if total_requests > 0 else 0,
                "avg_latency_ms": total_latency / total_requests if total_requests > 0 else 0,
                "total_cost": total_cost
            }
        
        return results
    
    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """Manually stop an active test"""
        
        if test_id in self.active_tests:
            return self._complete_test(test_id)
        else:
            return {"error": f"Test {test_id} not found or already completed"}