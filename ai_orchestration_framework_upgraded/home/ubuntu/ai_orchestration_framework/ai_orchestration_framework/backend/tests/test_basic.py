"""
Basic tests for the AI Orchestration Framework
"""

import pytest
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model_connectors import ModelRegistry, ModelProvider, ModelRequest, ModelResponse
from core.task_router import TaskRouter, KeywordComplexityEvaluator, TaskComplexity
from core.consensus import ConsensusManager, SimilarityConsensus
from core.memory import MemoryManager


class MockModelConnector:
    """Mock model connector for testing"""
    
    def __init__(self, provider_name):
        self.provider_name = provider_name
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def generate(self, request):
        return ModelResponse(
            content=f"Mock response from {self.provider_name}",
            provider=ModelProvider.OPENAI,  # Default for testing
            model_name="mock-model",
            tokens_used=50,
            latency_ms=100.0,
            confidence_score=0.8
        )
        
    async def validate_connection(self):
        return True
        
    def get_supported_models(self):
        return ["mock-model"]


def test_model_registry():
    """Test model registry functionality"""
    registry = ModelRegistry()
    
    # Test registration
    mock_connector = MockModelConnector("test")
    registry.register_connector(ModelProvider.OPENAI, mock_connector)
    
    # Test retrieval
    connector = registry.get_connector(ModelProvider.OPENAI)
    assert connector is not None
    assert connector.provider_name == "test"
    
    # Test get all providers
    providers = registry.get_all_providers()
    assert ModelProvider.OPENAI in providers


@pytest.mark.asyncio
async def test_complexity_evaluator():
    """Test complexity evaluation"""
    evaluator = KeywordComplexityEvaluator()
    
    # Test simple prompt
    simple_complexity = await evaluator.evaluate("What is the weather?")
    assert simple_complexity == TaskComplexity.SIMPLE
    
    # Test complex prompt
    complex_complexity = await evaluator.evaluate("Design a comprehensive system architecture for a distributed microservices platform")
    assert complex_complexity == TaskComplexity.COMPLEX


@pytest.mark.asyncio
async def test_task_router():
    """Test task routing functionality"""
    router = TaskRouter()
    
    # Test routing
    routing_result = await router.route_task(
        prompt="Hello world",
        task_type="general",
        available_providers=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC]
    )
    
    assert "providers" in routing_result
    assert "require_consensus" in routing_result
    assert "complexity" in routing_result
    assert len(routing_result["providers"]) > 0


@pytest.mark.asyncio
async def test_consensus_manager():
    """Test consensus functionality"""
    manager = ConsensusManager()
    
    # Create mock responses
    responses = [
        ModelResponse(
            content="Hello there!",
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            tokens_used=10,
            latency_ms=100.0,
            confidence_score=0.9
        ),
        ModelResponse(
            content="Hello there!",
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3",
            tokens_used=12,
            latency_ms=120.0,
            confidence_score=0.8
        )
    ]
    
    # Test consensus
    result = await manager.find_consensus(responses, "similarity")
    assert result.selected_response is not None
    assert result.confidence_score > 0
    assert result.consensus_method == "similarity"


def test_memory_manager():
    """Test memory manager functionality"""
    # Use in-memory database for testing
    memory = MemoryManager(":memory:")
    
    # Test that database is initialized
    assert memory.connection is not None
    
    # Test basic functionality (without async operations for simplicity)
    cursor = memory.connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['conversations', 'performance_metrics', 'embeddings_cache', 'routing_rules']
    for table in expected_tables:
        assert table in tables


def test_model_request_response():
    """Test model request and response data structures"""
    # Test ModelRequest
    request = ModelRequest(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=100
    )
    assert request.prompt == "Test prompt"
    assert request.temperature == 0.7
    assert request.max_tokens == 100
    assert request.model_specific_params == {}
    
    # Test ModelResponse
    response = ModelResponse(
        content="Test response",
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        tokens_used=50,
        latency_ms=200.0
    )
    assert response.content == "Test response"
    assert response.provider == ModelProvider.OPENAI
    assert response.model_name == "gpt-4"
    assert response.tokens_used == 50
    assert response.latency_ms == 200.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

