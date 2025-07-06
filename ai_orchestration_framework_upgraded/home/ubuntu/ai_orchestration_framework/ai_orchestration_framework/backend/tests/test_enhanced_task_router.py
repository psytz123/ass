
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.task_router import TaskRouter, RoutingRule
from core.model_connectors import ModelProvider, TaskComplexity
from core.ml_complexity_evaluator import MLComplexityEvaluator
from core.enhanced_performance_tracker import EnhancedPerformanceTracker
from core.base_evaluator import ComplexityEvaluator

class TestEnhancedTaskRouter(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.task_router = TaskRouter()
        # Ensure the default evaluator is the MLComplexityEvaluator
        self.assertIsInstance(self.task_router.default_evaluator, MLComplexityEvaluator)
        # Ensure the performance tracker is the EnhancedPerformanceTracker
        self.assertIsInstance(self.task_router.performance_tracker, EnhancedPerformanceTracker)

        # Mock the evaluate method of MLComplexityEvaluator for predictable results
        self.mock_ml_evaluator = AsyncMock(spec=MLComplexityEvaluator)
        self.task_router.default_evaluator = self.mock_ml_evaluator

        # Mock the get_best_providers method of EnhancedPerformanceTracker
        self.mock_performance_tracker = MagicMock(spec=EnhancedPerformanceTracker)
        self.task_router.performance_tracker = self.mock_performance_tracker

    async def test_complexity_evaluation_integration(self):
        # Test that TaskRouter uses the MLComplexityEvaluator
        self.mock_ml_evaluator.evaluate.return_value = TaskComplexity.COMPLEX
        prompt = "Design a complex system architecture."
        complexity = await self.task_router.evaluate_complexity(prompt)
        self.assertEqual(complexity, TaskComplexity.COMPLEX)
        self.mock_ml_evaluator.evaluate.assert_awaited_once_with(prompt, "general")

    async def test_dynamic_provider_selection_integration(self):
        # Test that TaskRouter uses the EnhancedPerformanceTracker for provider selection
        self.mock_ml_evaluator.evaluate.return_value = TaskComplexity.MEDIUM
        self.mock_performance_tracker.get_best_providers.return_value = [
            ModelProvider.ANTHROPIC, ModelProvider.OPENAI
        ]

        prompt = "Analyze this document."
        routing_result = await self.task_router.route_task(prompt, task_type="document_analysis")

        # Verify that get_best_providers was called with correct context
        self.mock_performance_tracker.get_best_providers.assert_called_once()
        args, kwargs = self.mock_performance_tracker.get_best_providers.call_args
        self.assertEqual(kwargs["complexity"], TaskComplexity.MEDIUM)
        self.assertEqual(kwargs["task_type"], "document_analysis")

        # Verify the order of providers based on mocked performance
        # The default rule for document_analysis includes Anthropic and Google.
        # If mock_performance_tracker returns Anthropic, OpenAI, then the order should be Anthropic, OpenAI, Google
        expected_providers = [ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
        self.assertEqual(routing_result["providers"], expected_providers)
        self.assertEqual(routing_result["complexity"], TaskComplexity.MEDIUM)

    async def test_record_performance_integration(self):
        # Test that TaskRouter records performance with context to the EnhancedPerformanceTracker
        provider = ModelProvider.OPENAI
        latency = 150.0
        success = True
        confidence = 0.9
        complexity = TaskComplexity.SIMPLE
        task_type = "general"

        self.task_router.record_performance(provider, latency, success, confidence, complexity, task_type)

        self.mock_performance_tracker.record_request.assert_called_once_with(
            provider, latency, success, confidence, complexity, task_type
        )

    async def test_routing_with_complex_task_and_consensus(self):
        self.mock_ml_evaluator.evaluate.return_value = TaskComplexity.COMPLEX
        self.mock_performance_tracker.get_best_providers.return_value = [
            ModelProvider.ANTHROPIC, ModelProvider.OPENAI, ModelProvider.GOOGLE
        ]

        prompt = "Develop a new AI framework for complex data analysis."
        routing_result = await self.task_router.route_task(prompt, task_type="code_generation")

        self.assertEqual(routing_result["complexity"], TaskComplexity.COMPLEX)
        self.assertTrue(routing_result["require_consensus"])
        self.assertEqual(routing_result["min_providers"], 2)
        # Ensure providers are reordered and include at least min_providers for consensus
        expected_providers = [ModelProvider.ANTHROPIC, ModelProvider.OPENAI]
        self.assertEqual(routing_result["providers"], expected_providers)

    async def test_routing_with_simple_task_no_consensus(self):
        self.mock_ml_evaluator.evaluate.return_value = TaskComplexity.SIMPLE
        self.mock_performance_tracker.get_best_providers.return_value = [
            ModelProvider.GOOGLE, ModelProvider.OPENAI
        ]

        prompt = "Summarize this article."
        routing_result = await self.task_router.route_task(prompt, task_type="general")

        self.assertEqual(routing_result["complexity"], TaskComplexity.SIMPLE)
        self.assertFalse(routing_result["require_consensus"])
        self.assertEqual(routing_result["min_providers"], 1)
        expected_providers = [ModelProvider.GOOGLE, ModelProvider.OPENAI]
        self.assertEqual(routing_result["providers"], expected_providers)

if __name__ == '__main__':
    unittest.main()




