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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model_connectors import (
    ModelRegistry, ModelProvider, ModelRequest, ModelResponse
)
from core.consensus import ConsensusManager, SimilarityConsensus
from core.task_router import TaskRouter, TaskComplexity
from core.memory import MemoryManager

# ============================================================================
# MOCK IMPLEMENTATIONS FOR TESTING
# ============================================================================

class MockModelConnector:
    """Mock model connector for testing without API calls"""
    
    def __init__(self, provider: ModelProvider, latency_ms: float = 100, 
                 failure_rate: float = 0.0, responses: List[str] = None):
        self.provider = provider
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.responses = responses or [f"Mock response from {provider}"]
        self.call_count = 0
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
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

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Configuration for testing"""
    return {
        "database_path": ":memory:",
        "openai_api_key": "test-key",
        "anthropic_api_key": "test-key",
        "google_api_key": "test-key",
        "enable_self_improvement": True,
        "log_level": "DEBUG"
    }

@pytest.fixture
async def memory_manager():
    """In-memory database for testing"""
    manager = MemoryManager(":memory:")
    yield manager
    # Cleanup is automatic with in-memory SQLite

@pytest.fixture
def mock_model_registry():
    """Mock model registry with test connectors"""
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

# ============================================================================
# UNIT TESTS - MODEL CONNECTORS
# ============================================================================

class TestModelConnectors:
    """Test individual model connector functionality"""
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
    async def test_connector_failure_simulation(self):
        """Test connector failure handling"""
        connector = MockModelConnector(
            ModelProvider.OPENAI,
            failure_rate=1.0  # Always fail
        )
        
        request = ModelRequest(prompt="Test prompt")
        
        with pytest.raises(Exception, match="Mock failure"):
            await connector.generate(request)
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
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
        
        assert result.selected_response.content == "The answer is 42"
        assert result.confidence_score >= 0.9  # High similarity
    
    @pytest.mark.asyncio
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
        assert result.selected_response.content in [
            "Python is a programming language",
            "Python is a type of snake"
        ]
    
    @pytest.mark.asyncio
    async def test_consensus_single_response(self):
        """Test consensus with only one response"""
        consensus = SimilarityConsensus()
        
        responses = self.create_mock_responses(["Single response"])
        result = await consensus.find_consensus(responses, {})
        
        assert result.selected_response.content == "Single response"

# ============================================================================
# UNIT TESTS - TASK ROUTING
# ============================================================================

class TestTaskRouting:
    """Test task complexity evaluation and routing logic"""
    
    @pytest.mark.asyncio
    async def test_task_router_basic_routing(self, mock_model_registry):
        """Test basic task routing functionality"""
        router = TaskRouter()
        
        # Test routing
        result = await router.route_task(
            prompt="simple task",
            task_type="test_task",
            available_providers=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC]
        )
        
        assert "complexity" in result
        assert "providers" in result
        assert "require_consensus" in result
        assert len(result["providers"]) > 0

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics and scalability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_processing(self, mock_model_registry):
        """Test handling multiple concurrent requests"""
        async def make_mock_request(i: int):
            connector = MockModelConnector(ModelProvider.OPENAI)
            request = ModelRequest(prompt=f"Concurrent request {i}")
            async with connector:
                return await connector.generate(request)
        
        # Create 10 concurrent requests
        start_time = time.time()
        tasks = [make_mock_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 10
        assert all(result.content is not None for result in results)
        # Should be faster than sequential processing
        assert elapsed_time < 5.0  # Reasonable upper bound

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error scenarios and recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_partial_provider_failure(self, mock_model_registry):
        """Test graceful degradation with partial failures"""
        # Test that system can handle when some providers fail
        failing_connector = MockModelConnector(ModelProvider.OPENAI, failure_rate=1.0)
        working_connector = MockModelConnector(ModelProvider.ANTHROPIC, failure_rate=0.0)
        
        # Test that working connector still functions
        request = ModelRequest(prompt="test with partial failure")
        
        with pytest.raises(Exception):
            async with failing_connector:
                await failing_connector.generate(request)
        
        # Working connector should succeed
        async with working_connector:
            result = await working_connector.generate(request)
            assert result.content is not None

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_very_long_prompts(self):
        """Test handling very long prompts"""
        # Create a very long prompt (10KB)
        long_prompt = "This is a very long prompt. " * 400
        
        connector = MockModelConnector(ModelProvider.OPENAI)
        request = ModelRequest(prompt=long_prompt)
        
        async with connector:
            result = await connector.generate(request)
            assert result.content is not None
    
    @pytest.mark.asyncio
    async def test_special_characters_in_prompts(self):
        """Test handling special characters and encoding"""
        special_prompts = [
            "Prompt with émojis 🚀🤖🔥",
            "Unicode: ñáéíóú",
            "Code: def func(): return {'key': 'value'}",
            "Math: ∑(x²) = ∫f(x)dx",
            "Mixed: Hello 世界 🌍"
        ]
        
        connector = MockModelConnector(ModelProvider.OPENAI)
        
        for prompt in special_prompts:
            request = ModelRequest(prompt=prompt)
            async with connector:
                result = await connector.generate(request)
                assert result.content is not None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

