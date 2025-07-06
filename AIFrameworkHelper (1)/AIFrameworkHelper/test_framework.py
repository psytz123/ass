"""
Test suite for AI Orchestration Framework

This test suite validates the core functionality of the AI Orchestration Framework,
including model connectors, consensus mechanisms, and task routing.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from core.framework import AIOrchestrationFramework
from core.types import ModelProvider, ModelRequest, ModelResponse, TaskComplexity
from core.model_connectors import BaseModelConnector
from config import Config

# Mock implementations for testing
class MockModelConnector(BaseModelConnector):
    """Mock model connector for testing without API calls"""
    
    def __init__(self, provider: ModelProvider, latency_ms: float = 100, 
                 failure_rate: float = 0.0, responses: List[str] = None):
        self.provider = provider
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.responses = responses or [f"Mock response from {provider.value}"]
        self.call_count = 0
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a mock response"""
        self.call_count += 1
        
        # Simulate failures
        if self.failure_rate > 0 and (self.call_count % int(1/self.failure_rate)) == 0:
            raise Exception(f"Mock failure for {self.provider.value}")
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Cycle through responses
        response_text = self.responses[(self.call_count - 1) % len(self.responses)]
        
        return ModelResponse(
            content=response_text,
            provider=self.provider,
            model_name=f"mock-{self.provider.value}",
            tokens_used=len(response_text.split()),
            latency_ms=self.latency_ms,
            confidence_score=0.8
        )
    
    async def validate_connection(self) -> bool:
        """Validate connection (always true for mock)"""
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return [f"mock-{self.provider.value}-v1"]

# Test fixtures
@pytest.fixture
def test_config():
    """Test configuration"""
    return Config({
        'openai': {'api_key': 'test-key'},
        'anthropic': {'api_key': 'test-key'},
        'google': {'api_key': 'test-key'},
        'database': {'url': 'sqlite:///:memory:'},
        'consensus': {'default_strategy': 'similarity'},
        'routing': {'default_strategy': 'performance_based'}
    })

@pytest.fixture
async def framework(test_config):
    """Create test framework instance"""
    fw = AIOrchestrationFramework(test_config)
    
    # Replace with mock connectors
    fw.model_connectors = {
        ModelProvider.OPENAI: MockModelConnector(
            ModelProvider.OPENAI, 
            responses=["OpenAI mock response"]
        ),
        ModelProvider.ANTHROPIC: MockModelConnector(
            ModelProvider.ANTHROPIC,
            responses=["Anthropic mock response"]
        ),
        ModelProvider.GOOGLE: MockModelConnector(
            ModelProvider.GOOGLE,
            responses=["Google mock response"]
        )
    }
    
    return fw

# Basic functionality tests
class TestBasicFunctionality:
    """Test basic framework functionality"""
    
    async def test_framework_initialization(self, test_config):
        """Test framework can be initialized"""
        framework = AIOrchestrationFramework(test_config)
        assert framework is not None
        assert framework.config is not None
    
    async def test_single_provider_request(self, framework):
        """Test basic request with single provider"""
        result = await framework.process_request(
            prompt="Test prompt",
            task_type="general",
            user_id="test_user"
        )
        
        assert result is not None
        assert result.final_response is not None
        assert result.final_response.content is not None
        assert len(result.individual_responses) >= 1
    
    async def test_multiple_provider_request(self, framework):
        """Test request requiring multiple providers"""
        # Force consensus requirement
        result = await framework.process_request(
            prompt="Complex analysis task",
            task_type="analysis",
            user_id="test_user",
            require_consensus=True,
            min_providers=2
        )
        
        assert result is not None
        assert result.final_response is not None
        assert len(result.individual_responses) >= 2
    
    async def test_error_handling(self, framework):
        """Test error handling when providers fail"""
        # Replace with failing connector
        framework.model_connectors[ModelProvider.OPENAI] = MockModelConnector(
            ModelProvider.OPENAI,
            failure_rate=1.0  # Always fails
        )
        
        # Should still work with other providers
        result = await framework.process_request(
            prompt="Test with failing provider",
            task_type="general",
            user_id="test_user"
        )
        
        assert result is not None
        assert result.final_response is not None

class TestConsensusAlgorithms:
    """Test consensus mechanisms"""
    
    async def test_similarity_consensus(self, framework):
        """Test similarity-based consensus"""
        # Create similar responses
        framework.model_connectors[ModelProvider.OPENAI] = MockModelConnector(
            ModelProvider.OPENAI,
            responses=["The capital of France is Paris"]
        )
        framework.model_connectors[ModelProvider.ANTHROPIC] = MockModelConnector(
            ModelProvider.ANTHROPIC,
            responses=["Paris is the capital of France"]
        )
        
        result = await framework.process_request(
            prompt="What is the capital of France?",
            task_type="factual",
            user_id="test_user",
            require_consensus=True,
            min_providers=2
        )
        
        assert result is not None
        assert result.consensus_result is not None
        assert result.consensus_result.confidence > 0.5
    
    async def test_conflicting_responses(self, framework):
        """Test consensus with conflicting responses"""
        # Create very different responses
        framework.model_connectors[ModelProvider.OPENAI] = MockModelConnector(
            ModelProvider.OPENAI,
            responses=["Answer A is correct"]
        )
        framework.model_connectors[ModelProvider.ANTHROPIC] = MockModelConnector(
            ModelProvider.ANTHROPIC,
            responses=["Answer B is definitely right"]
        )
        
        result = await framework.process_request(
            prompt="Which answer is correct?",
            task_type="comparison",
            user_id="test_user",
            require_consensus=True,
            min_providers=2
        )
        
        assert result is not None
        assert result.final_response is not None
        # Should handle low consensus gracefully

class TestPerformanceAndScaling:
    """Test performance characteristics"""
    
    async def test_concurrent_requests(self, framework):
        """Test handling multiple concurrent requests"""
        async def make_request(i):
            return await framework.process_request(
                prompt=f"Test request {i}",
                task_type="general",
                user_id=f"user_{i}"
            )
        
        # Create 5 concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 5
        assert all(r.final_response is not None for r in results)
        assert elapsed_time < 5.0  # Should complete reasonably fast
    
    async def test_latency_tracking(self, framework):
        """Test latency measurement"""
        # Set known latency
        framework.model_connectors[ModelProvider.OPENAI] = MockModelConnector(
            ModelProvider.OPENAI,
            latency_ms=200
        )
        
        start_time = time.time()
        result = await framework.process_request(
            prompt="Latency test",
            task_type="general",
            user_id="test_user"
        )
        total_time = (time.time() - start_time) * 1000
        
        assert result is not None
        assert total_time >= 200  # Should include mock latency
        assert result.processing_time_ms > 0

class TestTaskRouting:
    """Test task routing and complexity evaluation"""
    
    async def test_complexity_evaluation(self, framework):
        """Test task complexity evaluation"""
        # Simple task
        result = await framework.process_request(
            prompt="Hello world",
            task_type="general",
            user_id="test_user"
        )
        
        assert result is not None
        assert result.complexity_score is not None
        
        # Complex task
        complex_result = await framework.process_request(
            prompt="Analyze the economic implications of artificial intelligence on global markets",
            task_type="analysis",
            user_id="test_user"
        )
        
        assert complex_result is not None
        assert complex_result.complexity_score is not None

# Integration tests
class TestIntegration:
    """Test full integration scenarios"""
    
    async def test_full_workflow(self, framework):
        """Test complete workflow from request to response"""
        result = await framework.process_request(
            prompt="Explain machine learning in simple terms",
            task_type="explanation",
            user_id="integration_user"
        )
        
        assert result is not None
        assert result.final_response is not None
        assert result.final_response.content is not None
        assert result.providers_used is not None
        assert len(result.providers_used) >= 1
    
    async def test_conversation_history(self, framework):
        """Test conversation history tracking"""
        user_id = "history_user"
        
        # Make multiple requests
        for i in range(3):
            await framework.process_request(
                prompt=f"Question {i+1}",
                task_type="general",
                user_id=user_id
            )
        
        # Check history
        history = await framework.get_conversation_history(user_id, limit=10)
        assert len(history) == 3
    
    async def test_performance_metrics(self, framework):
        """Test performance metrics collection"""
        # Make some requests
        for i in range(5):
            await framework.process_request(
                prompt=f"Metrics test {i}",
                task_type="general",
                user_id="metrics_user"
            )
        
        # Get metrics
        metrics = await framework.get_performance_metrics(time_window_hours=1)
        assert metrics is not None
        assert "total_requests" in metrics
        assert metrics["total_requests"] > 0

# Configuration for pytest
pytest_plugins = ["pytest_asyncio"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])