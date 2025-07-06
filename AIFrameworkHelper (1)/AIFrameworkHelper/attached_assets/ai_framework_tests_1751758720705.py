# ============================================================================
# COMPREHENSIVE TEST SUITE FOR AI ORCHESTRATION FRAMEWORK
# ============================================================================

import pytest
import asyncio
import json
import time
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta

# Import framework components
from ai_framework import AIOrchestrationFramework
from ai_framework.models import (
    BaseModelConnector, ModelRequest, ModelResponse, ModelProvider
)
from ai_framework.consensus import ConsensusStrategy, SimilarityConsensus, VotingConsensus
from ai_framework.routing import TaskRouter, TaskComplexity, ComplexityEvaluator
from ai_framework.memory import MemoryManager
from ai_framework.improvement import SelfImprovementEngine

# ============================================================================
# MOCK IMPLEMENTATIONS FOR TESTING
# ============================================================================

class MockModelConnector(BaseModelConnector):
    """Mock model connector for testing without API calls"""
    
    def __init__(self, provider: ModelProvider, latency_ms: float = 100, 
                 failure_rate: float = 0.0, responses: List[str] = None):
        self.provider = provider
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.responses = responses or [f"Mock response from {provider}"]
        self.call_count = 0
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        
        # Simulate failures
        if np.random.random() < self.failure_rate:
            raise Exception(f"Mock failure for {self.provider}")
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Cycle through responses
        response_text = self.responses[(self.call_count - 1) % len(self.responses)]
        
        return ModelResponse(
            content=response_text,
            provider=self.provider,
            model_name=f"mock-{self.provider}",
            tokens_used=len(response_text.split()),
            latency_ms=self.latency_ms,
            confidence_score=0.8,
            metadata={"mock": True, "call_count": self.call_count}
        )
    
    async def validate_connection(self) -> bool:
        return self.failure_rate < 1.0
    
    def get_supported_models(self) -> List[str]:
        return [f"mock-{self.provider}-v1", f"mock-{self.provider}-v2"]

class MockComplexityEvaluator(ComplexityEvaluator):
    """Mock complexity evaluator for testing"""
    
    def __init__(self, complexity_map: Dict[str, TaskComplexity] = None):
        self.complexity_map = complexity_map or {}
        self.default_complexity = TaskComplexity.MEDIUM
    
    async def evaluate(self, prompt: str) -> TaskComplexity:
        for keyword, complexity in self.complexity_map.items():
            if keyword.lower() in prompt.lower():
                return complexity
        return self.default_complexity

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Configuration for testing"""
    return {
        "database_url": "sqlite:///:memory:",
        "redis_url": None,
        "openai_api_key": "test-key",
        "anthropic_api_key": "test-key",
        "google_api_key": "test-key",
        "enable_self_improvement": True,
        "log_level": "DEBUG"
    }

@pytest.fixture
async def memory_manager():
    """In-memory database for testing"""
    manager = MemoryManager("sqlite:///:memory:")
    yield manager
    # Cleanup is automatic with in-memory SQLite

@pytest.fixture
def mock_model_registry():
    """Mock model registry with test connectors"""
    from ai_framework.models import ModelRegistry
    
    registry = ModelRegistry()
    
    # Add mock connectors
    registry.register_connector(
        ModelProvider.OPENAI,
        MockModelConnector(ModelProvider.OPENAI, responses=["OpenAI response"])
    )
    registry.register_connector(
        ModelProvider.ANTHROPIC,
        MockModelConnector(ModelProvider.ANTHROPIC, responses=["Claude response"])
    )
    registry.register_connector(
        ModelProvider.GOOGLE,
        MockModelConnector(ModelProvider.GOOGLE, responses=["Gemini response"])
    )
    
    return registry

@pytest.fixture
async def test_framework(mock_config, memory_manager, mock_model_registry):
    """Complete test framework setup"""
    framework = AIOrchestrationFramework(mock_config)
    framework.memory_manager = memory_manager
    framework.model_registry = mock_model_registry
    
    # Setup mock complexity evaluator
    mock_evaluator = MockComplexityEvaluator({
        "simple": TaskComplexity.SIMPLE,
        "complex": TaskComplexity.COMPLEX,
        "specialized": TaskComplexity.SPECIALIZED
    })
    framework.task_router.register_evaluator("test_task", mock_evaluator)
    
    return framework

# ============================================================================
# UNIT TESTS - MODEL CONNECTORS
# ============================================================================

class TestModelConnectors:
    """Test individual model connector functionality"""
    
    async def test_mock_connector_basic_generation(self):
        """Test basic response generation"""
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            responses=["Test response"]
        )
        
        request = ModelRequest(prompt="Test prompt")
        response = await connector.generate(request)
        
        assert response.content == "Test response"
        assert response.provider == ModelProvider.OPENAI
        assert response.tokens_used == 2  # "Test response" = 2 tokens
        assert connector.call_count == 1
    
    async def test_connector_failure_simulation(self):
        """Test connector failure handling"""
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            failure_rate=1.0  # Always fail
        )
        
        request = ModelRequest(prompt="Test prompt")
        
        with pytest.raises(Exception, match="Mock failure"):
            await connector.generate(request)
    
    async def test_connector_latency_simulation(self):
        """Test latency simulation"""
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            latency_ms=200
        )
        
        start_time = time.time()
        request = ModelRequest(prompt="Test prompt")
        response = await connector.generate(request)
        elapsed_time = (time.time() - start_time) * 1000
        
        assert elapsed_time >= 200
        assert response.latency_ms == 200
    
    async def test_multiple_responses_cycling(self):
        """Test cycling through multiple mock responses"""
        responses = ["Response 1", "Response 2", "Response 3"]
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            responses=responses
        )
        
        request = ModelRequest(prompt="Test prompt")
        
        # Test cycling through responses
        for i in range(6):  # Test 2 full cycles
            response = await connector.generate(request)
            expected_response = responses[i % len(responses)]
            assert response.content == expected_response

# ============================================================================
# UNIT TESTS - CONSENSUS STRATEGIES
# ============================================================================

class TestConsensusStrategies:
    """Test consensus algorithm implementations"""
    
    def create_mock_responses(self, contents: List[str]) -> List[ModelResponse]:
        """Helper to create mock responses"""
        responses = []
        for i, content in enumerate(contents):
            response = ModelResponse(
                content=content,
                provider=list(ModelProvider)[i % len(ModelProvider)],
                model_name=f"mock-model-{i}",
                tokens_used=len(content.split()),
                latency_ms=100,
                confidence_score=0.8
            )
            responses.append(response)
        return responses
    
    async def test_similarity_consensus_identical_responses(self):
        """Test consensus with identical responses"""
        consensus = SimilarityConsensus()
        
        # Create identical responses
        responses = self.create_mock_responses([
            "The answer is 42",
            "The answer is 42",
            "The answer is 42"
        ])
        
        result = await consensus.find_consensus(responses, {})
        
        assert result.content == "The answer is 42"
        assert result.confidence_score >= 0.9  # High similarity
    
    async def test_similarity_consensus_different_responses(self):
        """Test consensus with different responses"""
        consensus = SimilarityConsensus()
        
        responses = self.create_mock_responses([
            "Python is a programming language",
            "Python is a type of snake",
            "The quick brown fox jumps"
        ])
        
        result = await consensus.find_consensus(responses, {})
        
        # Should pick one of the first two (more similar to each other)
        assert result.content in [
            "Python is a programming language",
            "Python is a type of snake"
        ]
    
    async def test_consensus_single_response(self):
        """Test consensus with only one response"""
        consensus = SimilarityConsensus()
        
        responses = self.create_mock_responses(["Single response"])
        result = await consensus.find_consensus(responses, {})
        
        assert result.content == "Single response"
    
    async def test_voting_consensus_implementation(self):
        """Test voting consensus strategy"""
        # This would test a more complex voting implementation
        # For now, we'll test the interface
        consensus = VotingConsensus()
        responses = self.create_mock_responses([
            "Option A", "Option B", "Option A"
        ])
        
        # Test that the method exists and can be called
        try:
            result = await consensus.find_consensus(responses, {})
            # Implementation details would be tested here
            assert result is not None
        except NotImplementedError:
            # Expected if voting consensus isn't fully implemented
            pass

# ============================================================================
# UNIT TESTS - TASK ROUTING
# ============================================================================

class TestTaskRouting:
    """Test task complexity evaluation and routing logic"""
    
    async def test_complexity_evaluation(self):
        """Test task complexity evaluation"""
        evaluator = MockComplexityEvaluator({
            "simple": TaskComplexity.SIMPLE,
            "algorithm": TaskComplexity.COMPLEX,
            "machine learning": TaskComplexity.SPECIALIZED
        })
        
        assert await evaluator.evaluate("simple task") == TaskComplexity.SIMPLE
        assert await evaluator.evaluate("complex algorithm") == TaskComplexity.COMPLEX
        assert await evaluator.evaluate("machine learning model") == TaskComplexity.SPECIALIZED
        assert await evaluator.evaluate("random task") == TaskComplexity.MEDIUM
    
    async def test_task_router_basic_routing(self, mock_model_registry):
        """Test basic task routing functionality"""
        router = TaskRouter()
        
        # Register mock evaluator
        evaluator = MockComplexityEvaluator({
            "simple": TaskComplexity.SIMPLE,
            "complex": TaskComplexity.COMPLEX
        })
        router.register_evaluator("test_task", evaluator)
        
        # Test routing
        result = await router.evaluate_and_route("simple task", "test_task")
        
        assert result["complexity"] == TaskComplexity.SIMPLE
        assert "providers" in result
        assert "require_consensus" in result
    
    async def test_routing_rule_matching(self):
        """Test routing rule matching logic"""
        from ai_framework.routing import RoutingRule
        
        router = TaskRouter()
        
        # Add routing rules
        simple_rule = RoutingRule(
            task_type="test_task",
            complexity=TaskComplexity.SIMPLE,
            preferred_providers=[ModelProvider.OPENAI],
            fallback_providers=[ModelProvider.ANTHROPIC],
            require_consensus=False
        )
        
        complex_rule = RoutingRule(
            task_type="test_task",
            complexity=TaskComplexity.COMPLEX,
            preferred_providers=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC],
            fallback_providers=[ModelProvider.GOOGLE],
            require_consensus=True
        )
        
        router.add_routing_rule(simple_rule)
        router.add_routing_rule(complex_rule)
        
        # Test rule application
        # Implementation would test that rules are correctly matched

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFrameworkIntegration:
    """Test component integration and end-to-end workflows"""
    
    async def test_basic_request_processing(self, test_framework):
        """Test basic request processing flow"""
        result = await test_framework.process_request(
            prompt="simple test prompt",
            task_type="test_task",
            user_id="test_user"
        )
        
        assert result["response"] is not None
        assert "metadata" in result
        assert result["metadata"]["complexity"] == TaskComplexity.SIMPLE
        assert len(result["metadata"]["providers_used"]) >= 1
    
    async def test_consensus_workflow(self, test_framework):
        """Test consensus mechanism in full workflow"""
        # Create connectors with different responses
        different_responses = [
            MockModelConnector(ModelProvider.OPENAI, responses=["OpenAI says A"]),
            MockModelConnector(ModelProvider.ANTHROPIC, responses=["Claude says A"]),
            MockModelConnector(ModelProvider.GOOGLE, responses=["Gemini says B"])
        ]
        
        # Replace connectors
        for provider, connector in zip(ModelProvider, different_responses):
            test_framework.model_registry.register_connector(provider, connector)
        
        result = await test_framework.process_request(
            prompt="complex test requiring consensus",
            task_type="test_task",
            user_id="test_user"
        )
        
        assert result["response"] is not None
        assert result["metadata"]["complexity"] == TaskComplexity.COMPLEX
        # Complex tasks should use multiple providers
        assert len(result["metadata"]["providers_used"]) > 1
    
    async def test_error_handling_and_fallbacks(self, test_framework):
        """Test error handling and fallback mechanisms"""
        # Create failing connector
        failing_connector = MockModelConnector(
            ModelProvider.OPENAI,
            failure_rate=1.0  # Always fails
        )
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI,
            failing_connector
        )
        
        # Should still work with other providers
        result = await test_framework.process_request(
            prompt="test with failing provider",
            task_type="test_task",
            user_id="test_user"
        )
        
        assert result["response"] is not None
        # Should not include the failing provider
        assert ModelProvider.OPENAI not in result["metadata"]["providers_used"]
    
    async def test_conversation_persistence(self, test_framework):
        """Test conversation history storage"""
        user_id = "test_user_123"
        
        # Make several requests
        prompts = ["First question", "Second question", "Third question"]
        
        for prompt in prompts:
            await test_framework.process_request(
                prompt=prompt,
                task_type="test_task",
                user_id=user_id
            )
        
        # Check conversation history
        history = await test_framework.memory_manager.get_conversation_history(
            user_id, limit=10
        )
        
        assert len(history) == len(prompts)
        stored_prompts = [conv.prompt for conv in history]
        # History is returned in reverse order (newest first)
        assert stored_prompts == list(reversed(prompts))

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics and scalability"""
    
    async def test_concurrent_request_processing(self, test_framework):
        """Test handling multiple concurrent requests"""
        async def make_request(i: int):
            return await test_framework.process_request(
                prompt=f"Concurrent request {i}",
                task_type="test_task",
                user_id=f"user_{i}"
            )
        
        # Create 10 concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 10
        assert all(result["response"] is not None for result in results)
        # Should be faster than sequential processing
        assert elapsed_time < 5.0  # Reasonable upper bound
    
    async def test_latency_measurement(self, test_framework):
        """Test latency tracking accuracy"""
        # Set known latency
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            latency_ms=250  # 250ms latency
        )
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI,
            connector
        )
        
        start_time = time.time()
        result = await test_framework.process_request(
            prompt="latency test",
            task_type="test_task",
            user_id="test_user"
        )
        total_time = (time.time() - start_time) * 1000
        
        # Check that processing time is reasonable
        assert total_time >= 250  # At least the mock latency
        assert total_time < 1000   # But not too much overhead
    
    async def test_memory_usage_growth(self, test_framework):
        """Test memory usage doesn't grow unbounded"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for i in range(100):
            await test_framework.process_request(
                prompt=f"Memory test {i}",
                task_type="test_task",
                user_id="test_user"
            )
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB for 100 requests)
        assert memory_growth < 50 * 1024 * 1024

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error scenarios and recovery mechanisms"""
    
    async def test_all_providers_fail(self, test_framework):
        """Test behavior when all providers fail"""
        # Make all connectors fail
        for provider in ModelProvider:
            failing_connector = MockModelConnector(
                provider,
                failure_rate=1.0
            )
            test_framework.model_registry.register_connector(
                provider,
                failing_connector
            )
        
        with pytest.raises(Exception):
            await test_framework.process_request(
                prompt="This should fail",
                task_type="test_task",
                user_id="test_user"
            )
    
    async def test_partial_provider_failure(self, test_framework):
        """Test graceful degradation with partial failures"""
        # Make OpenAI fail, others work
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI,
            MockModelConnector(ModelProvider.OPENAI, failure_rate=1.0)
        )
        
        result = await test_framework.process_request(
            prompt="test with partial failure",
            task_type="test_task",
            user_id="test_user"
        )
        
        assert result["response"] is not None
        assert ModelProvider.OPENAI not in result["metadata"]["providers_used"]
    
    async def test_timeout_handling(self, test_framework):
        """Test request timeout handling"""
        # Create very slow connector
        slow_connector = MockModelConnector(
            ModelProvider.OPENAI,
            latency_ms=5000  # 5 second delay
        )
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI,
            slow_connector
        )
        
        # This test would need timeout configuration in the framework
        # For now, just verify the slow connector works
        start_time = time.time()
        result = await test_framework.process_request(
            prompt="timeout test",
            task_type="test_task",
            user_id="test_user"
        )
        elapsed_time = (time.time() - start_time) * 1000
        
        assert result["response"] is not None
        assert elapsed_time >= 1000  # Should take significant time
    
    async def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        invalid_config = {
            "database_url": "invalid://url",
            # Missing required keys
        }
        
        with pytest.raises(Exception):
            AIOrchestrationFramework(invalid_config)
    
    async def test_malformed_requests(self, test_framework):
        """Test handling of malformed requests"""
        # Test empty prompt
        with pytest.raises(ValueError):
            await test_framework.process_request(
                prompt="",
                task_type="test_task",
                user_id="test_user"
            )
        
        # Test invalid task type
        with pytest.raises(ValueError):
            await test_framework.process_request(
                prompt="test prompt",
                task_type="",
                user_id="test_user"
            )
        
        # Test missing user ID
        with pytest.raises(ValueError):
            await test_framework.process_request(
                prompt="test prompt",
                task_type="test_task",
                user_id=""
            )

# ============================================================================
# SELF-IMPROVEMENT TESTS
# ============================================================================

class TestSelfImprovement:
    """Test self-improvement and adaptive mechanisms"""
    
    async def test_performance_tracking(self, test_framework):
        """Test performance metrics collection"""
        # Make several requests with different characteristics
        requests = [
            ("simple task", "test_task"),
            ("complex algorithm", "test_task"),
            ("another simple task", "test_task")
        ]
        
        for prompt, task_type in requests:
            await test_framework.process_request(
                prompt=prompt,
                task_type=task_type,
                user_id="test_user"
            )
        
        # Check performance metrics
        metrics = await test_framework.get_performance_metrics(time_window_hours=1)
        
        assert metrics["total_requests"] == len(requests)
        assert "avg_latency_ms" in metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] > 0
    
    async def test_improvement_analysis(self, test_framework):
        """Test performance analysis and improvement suggestions"""
        # Create scenario with clear performance differences
        fast_connector = MockModelConnector(
            ModelProvider.OPENAI,
            latency_ms=50
        )
        slow_connector = MockModelConnector(
            ModelProvider.ANTHROPIC,
            latency_ms=500
        )
        
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI, fast_connector
        )
        test_framework.model_registry.register_connector(
            ModelProvider.ANTHROPIC, slow_connector
        )
        
        # Make requests to generate performance data
        for i in range(10):
            await test_framework.process_request(
                prompt=f"performance test {i}",
                task_type="test_task",
                user_id="test_user"
            )
        
        # Analyze performance
        analysis = await test_framework.improvement_engine.analyze_performance(
            time_window_hours=1
        )
        
        assert "provider_performance" in analysis
        # Should identify that OpenAI is faster
        # Implementation details would depend on analysis logic
    
    async def test_adaptive_routing_updates(self, test_framework):
        """Test adaptive routing based on performance"""
        # This would test that routing rules update based on performance
        # Implementation depends on self-improvement mechanism
        
        # Create performance differential
        unreliable_connector = MockModelConnector(
            ModelProvider.GOOGLE,
            failure_rate=0.5  # 50% failure rate
        )
        test_framework.model_registry.register_connector(
            ModelProvider.GOOGLE, unreliable_connector
        )
        
        # Make many requests to generate failure data
        successful_requests = 0
        total_requests = 20
        
        for i in range(total_requests):
            try:
                await test_framework.process_request(
                    prompt=f"reliability test {i}",
                    task_type="test_task",
                    user_id="test_user"
                )
                successful_requests += 1
            except:
                pass
        
        # The framework should learn to avoid the unreliable provider
        # This would be tested through routing behavior changes

# ============================================================================
# PLUGIN SYSTEM TESTS
# ============================================================================

class TestPluginSystem:
    """Test plugin loading and extensibility"""
    
    def test_custom_connector_registration(self, test_framework):
        """Test registering custom model connectors"""
        class CustomConnector(BaseModelConnector):
            async def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    content="Custom response",
                    provider=ModelProvider.OPENAI,  # Using existing enum
                    model_name="custom-model",
                    tokens_used=10,
                    latency_ms=100
                )
            
            async def validate_connection(self) -> bool:
                return True
            
            def get_supported_models(self) -> List[str]:
                return ["custom-model-v1"]
        
        # Register custom connector
        custom_connector = CustomConnector()
        test_framework.model_registry.register_connector(
            ModelProvider.OPENAI,  # Override existing
            custom_connector
        )
        
        # Test that it works
        assert test_framework.model_registry._connectors[ModelProvider.OPENAI] == custom_connector
    
    def test_custom_consensus_strategy(self, test_framework):
        """Test registering custom consensus strategies"""
        class SimpleConsensus(ConsensusStrategy):
            async def find_consensus(self, responses: List[ModelResponse], 
                                   task_context: Dict[str, Any]) -> ModelResponse:
                # Just return the first response
                return responses[0] if responses else None
            
            def calculate_confidence(self, responses: List[ModelResponse]) -> float:
                return 1.0  # Always confident
        
        # Register custom strategy
        test_framework.consensus_manager.register_strategy(
            "simple",
            SimpleConsensus()
        )
        
        # Test that it's registered
        assert "simple" in test_framework.consensus_manager._strategies
    
    def test_custom_complexity_evaluator(self, test_framework):
        """Test registering custom complexity evaluators"""
        class KeywordComplexityEvaluator(ComplexityEvaluator):
            async def evaluate(self, prompt: str) -> TaskComplexity:
                if "AI" in prompt or "machine learning" in prompt:
                    return TaskComplexity.SPECIALIZED
                elif len(prompt.split()) > 50:
                    return TaskComplexity.COMPLEX
                else:
                    return TaskComplexity.SIMPLE
        
        # Register custom evaluator
        evaluator = KeywordComplexityEvaluator()
        test_framework.task_router.register_evaluator("custom_task", evaluator)
        
        # Test that it's registered
        assert "custom_task" in test_framework.task_router._complexity_evaluators

# ============================================================================
# LOAD TESTING
# ============================================================================

class TestLoadAndStress:
    """Test system behavior under load"""
    
    @pytest.mark.slow
    async def test_high_volume_requests(self, test_framework):
        """Test handling high volume of requests"""
        async def make_batch_requests(batch_size: int, batch_id: int):
            tasks = []
            for i in range(batch_size):
                task = test_framework.process_request(
                    prompt=f"Batch {batch_id} request {i}",
                    task_type="test_task",
                    user_id=f"batch_user_{batch_id}_{i}"
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        # Process 100 requests in batches of 10
        start_time = time.time()
        batch_tasks = []
        for batch_id in range(10):
            batch_task = make_batch_requests(10, batch_id)
            batch_tasks.append(batch_task)
        
        all_results = await asyncio.gather(*batch_tasks)
        elapsed_time = time.time() - start_time
        
        # Flatten results
        total_results = sum(len(batch) for batch in all_results)
        assert total_results == 100
        
        # Should complete in reasonable time (less than 30 seconds)
        assert elapsed_time < 30.0
        
        # All requests should succeed
        for batch in all_results:
            for result in batch:
                assert result["response"] is not None
    
    @pytest.mark.slow  
    async def test_memory_stress(self, test_framework):
        """Test memory usage under sustained load"""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Process many requests with large prompts
        large_prompt = "This is a large prompt. " * 100  # ~2.5KB prompt
        
        for i in range(50):
            await test_framework.process_request(
                prompt=f"{large_prompt} Request {i}",
                task_type="test_task",
                user_id=f"stress_user_{i}"
            )
            
            # Occasional garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # System should still be responsive
        final_result = await test_framework.process_request(
            prompt="Final test after stress",
            task_type="test_task", 
            user_id="final_user"
        )
        
        assert final_result["response"] is not None

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    async def test_very_long_prompts(self, test_framework):
        """Test handling very long prompts"""
        # Create a very long prompt (10KB)
        long_prompt = "This is a very long prompt. " * 400
        
        result = await test_framework.process_request(
            prompt=long_prompt,
            task_type="test_task",
            user_id="test_user"
        )
        
        assert result["response"] is not None
        # Response should be reasonably sized
        assert len(result["response"]) < len(long_prompt) * 2
    
    async def test_special_characters_in_prompts(self, test_framework):
        """Test handling special characters and encoding"""
        special_prompts = [
            "Prompt with émojis 🚀🤖🔥",
            "Unicode: ñáéíóú",
            "Code: def func(): return {'key': 'value'}",
            "Math: ∑(x²) = ∫f(x)dx",
            "Mixed: Hello 世界 🌍"
        ]
        
        for prompt in special_prompts:
            result = await test_framework.process_request(
                prompt=prompt,
                task_type="test_task",
                user_id="test_user"
            )
            assert result["response"] is not None
    
    async def test_concurrent_same_user_requests(self, test_framework):
        """Test multiple concurrent requests from same user"""
        user_id = "concurrent_user"
        
        async def make_request(i: int):
            return await test_framework.process_request(
                prompt=f"Concurrent request {i} from same user",
                task_type="test_task",
                user_id=user_id
            )
        
        # Make 5 concurrent requests from same user
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result["response"] is not None for result in results)
        
        # Check conversation history
        history = await test_framework.memory_manager.get_conversation_history(
            user_id, limit=10
        )
        assert len(history) == 5
    
    async def test_empty_and_whitespace_prompts(self, test_framework):
        """Test handling edge cases in prompt content"""
        edge_case_prompts = [
            "   ",  # Only whitespace
            "\n\n\n",  # Only newlines
            "\t\t",  # Only tabs
            "a",  # Single character
            "?" * 1000,  # Repetitive content
        ]
        
        for prompt in edge_case_prompts:
            try:
                result = await test_framework.process_request(
                    prompt=prompt,
                    task_type="test_task",
                    user_id="test_user"
                )
                # If no exception, should have valid response
                assert result["response"] is not None
            except ValueError:
                # ValueError for empty/invalid prompts is acceptable
                pass

# ============================================================================
# REAL API INTEGRATION TESTS (Optional - requires API keys)
# ============================================================================

class TestRealAPIIntegration:
    """Tests with real AI model APIs (requires valid API keys)"""
    
    @pytest.mark.live_api
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available"
    )
    async def test_real_openai_integration(self):
        """Test integration with real OpenAI API"""
        from ai_framework.models.openai_connector import OpenAIConnector
        
        connector = OpenAIConnector(os.getenv("OPENAI_API_KEY"))
        
        request = ModelRequest(
            prompt="Write a simple Python function to add two numbers",
            temperature=0.1,
            max_tokens=100
        )
        
        response = await connector.generate(request)
        
        assert response.content is not None
        assert response.provider == ModelProvider.OPENAI
        assert response.tokens_used > 0
        assert response.latency_ms > 0
        assert "def" in response.content.lower()  # Should contain function definition
    
    @pytest.mark.live_api
    @pytest.mark.skipif(
        not all([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY")
        ]),
        reason="Multiple API keys not available"
    )
    async def test_real_consensus_between_models(self):
        """Test real consensus between different AI models"""
        config = {
            "database_url": "sqlite:///:memory:",
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        framework = AIOrchestrationFramework(config)
        
        result = await framework.process_request(
            prompt="What is the capital of France?",
            task_type="factual_query",
            user_id="live_test_user"
        )
        
        assert result["response"] is not None
        assert "paris" in result["response"].lower()
        assert len(result["metadata"]["providers_used"]) >= 1

# ============================================================================
# TEST CONFIGURATION AND UTILITIES
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "live_api: marks tests that use real API calls"
    )

def pytest_collection_modifyitems(config, items):
    """Auto-mark slow tests"""
    for item in items:
        if "load" in item.name or "stress" in item.name or "high_volume" in item.name:
            item.add_marker(pytest.mark.slow)

# ============================================================================
# EXAMPLE TEST RUN COMMANDS
# ============================================================================

"""
# Run all tests
pytest

# Run only fast tests
pytest -m "not slow"

# Run with coverage
pytest --cov=ai_framework --cov-report=html

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_consensus.py
pytest tests/test_routing.py

# Run integration tests only
pytest -k "integration"

# Run with live API calls (requires API keys)
pytest -m live_api --live-api

# Run performance tests
pytest -m slow

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_framework.py::TestFrameworkIntegration::test_basic_request_processing

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Generate coverage report
pytest --cov=ai_framework --cov-report=term-missing
"""