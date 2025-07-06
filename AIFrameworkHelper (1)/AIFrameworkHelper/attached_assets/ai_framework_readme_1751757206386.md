# AI Orchestration Framework

A powerful, extensible Python framework for building AI-powered applications with multi-model integration, dynamic routing, consensus mechanisms, and self-improvement capabilities.

## 🚀 Overview

The AI Orchestration Framework enables developers to build sophisticated AI applications that can:
- **Integrate multiple AI models** (ChatGPT, Claude, Gemini) seamlessly
- **Dynamically route tasks** to the most suitable model based on complexity and performance
- **Find consensus** across multiple model responses using pluggable algorithms
- **Self-improve** through performance analysis and adaptive routing
- **Scale efficiently** with async processing and intelligent caching

Perfect for building code generation tools, business automation agents, and complex AI workflows.

## ✨ Key Features

### 🔗 Multi-Model Integration
- **Universal API**: Single interface for OpenAI, Anthropic, and Google models
- **Async Processing**: Parallel model calls for improved performance
- **Auto-Failover**: Automatic fallback to alternative models
- **Rate Limiting**: Built-in request throttling and quota management

### 🧠 Intelligent Task Routing
- **Complexity Analysis**: Automatic task complexity evaluation
- **Performance-Based Routing**: Routes tasks based on historical model performance
- **Pluggable Evaluators**: Custom complexity evaluation algorithms
- **Dynamic Optimization**: Continuously improves routing decisions

### 🤝 Consensus Mechanisms
- **Multiple Strategies**: Similarity-based, voting, confidence-weighted consensus
- **Pluggable Architecture**: Easy to add custom consensus algorithms
- **Quality Metrics**: Confidence scoring and consensus quality assessment
- **Configurable Thresholds**: Customizable consensus requirements

### 📊 Memory & Analytics
- **Conversation History**: Persistent storage of all interactions
- **Performance Tracking**: Detailed metrics on model performance
- **Embedding Cache**: Efficient similarity search and caching
- **Data Export**: Easy access to historical data for analysis

### 🔄 Self-Improvement
- **Performance Analysis**: Automated analysis of model and routing performance
- **Adaptive Learning**: Continuously optimizes routing and consensus strategies
- **A/B Testing**: Built-in experimentation framework
- **Meta-Prompting**: Uses AI to improve its own prompts and strategies

### 🔌 Extensible Plugin System
- **Custom Models**: Easy integration of new AI model providers
- **Custom Strategies**: Pluggable consensus and routing algorithms
- **Event Hooks**: Extensible event system for custom logic
- **Auto-Discovery**: Automatic plugin loading and registration

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip or poetry for package management
- Redis (optional, for enhanced caching)
- PostgreSQL/SQLite for data persistence

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-orchestration-framework.git
cd ai-orchestration-framework

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

### Docker Installation

```bash
# Using Docker Compose (includes Redis and PostgreSQL)
docker-compose up -d

# Or build from source
docker build -t ai-framework .
docker run -p 8000:8000 ai-framework
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black . && flake8
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ai_framework
# Or for SQLite: sqlite:///ai_framework.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# AI Model API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Framework Configuration
LOG_LEVEL=INFO
ENABLE_SELF_IMPROVEMENT=true
PERFORMANCE_ANALYSIS_INTERVAL=24  # hours
```

### Configuration File

Create `config.yaml`:

```yaml
# Core Framework Settings
framework:
  name: "AI Orchestration Framework"
  version: "1.0.0"
  enable_metrics: true
  enable_caching: true

# Model Configuration
models:
  openai:
    default_model: "gpt-4"
    max_tokens: 4000
    temperature: 0.7
  anthropic:
    default_model: "claude-3-opus-20240229"
    max_tokens: 4000
  google:
    default_model: "gemini-pro"
    max_tokens: 4000

# Routing Configuration
routing:
  default_strategy: "performance_based"
  complexity_threshold: 0.7
  require_consensus_for_complex: true
  
  rules:
    - task_type: "code_generation"
      complexity: "simple"
      providers: ["openai", "anthropic"]
      require_consensus: false
    
    - task_type: "code_generation" 
      complexity: "complex"
      providers: ["openai", "anthropic", "google"]
      require_consensus: true

# Consensus Configuration
consensus:
  default_strategy: "similarity"
  similarity_threshold: 0.8
  confidence_threshold: 0.7
  
  strategies:
    similarity:
      model: "all-MiniLM-L6-v2"
      threshold: 0.8
    voting:
      require_majority: true
      weight_by_confidence: true

# Self-Improvement Configuration
improvement:
  enable: true
  analysis_interval: 24  # hours
  auto_apply_threshold: 0.9
  min_data_points: 100
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from ai_framework import AIOrchestrationFramework

async def main():
    # Initialize framework
    framework = AIOrchestrationFramework.from_config("config.yaml")
    
    # Process a simple request
    result = await framework.process_request(
        prompt="Write a Python function to calculate fibonacci numbers",
        task_type="code_generation",
        user_id="user123"
    )
    
    print("Response:", result["response"])
    print("Metadata:", result["metadata"])

# Run the example
asyncio.run(main())
```

### Advanced Usage with Custom Settings

```python
from ai_framework import AIOrchestrationFramework
from ai_framework.models import ModelRequest
from ai_framework.routing import TaskComplexity

async def advanced_example():
    framework = AIOrchestrationFramework.from_config("config.yaml")
    
    # Custom model request
    request = ModelRequest(
        prompt="Optimize this database query for better performance",
        temperature=0.3,
        max_tokens=2000,
        system_prompt="You are a database optimization expert"
    )
    
    # Process with specific requirements
    result = await framework.process_request(
        prompt=request.prompt,
        task_type="code_optimization",
        user_id="user123",
        require_consensus=True,
        min_providers=2,
        complexity_override=TaskComplexity.COMPLEX
    )
    
    # Access detailed results
    print(f"Consensus confidence: {result['metadata']['confidence']}")
    print(f"Providers used: {result['metadata']['providers_used']}")
    print(f"Processing time: {result['metadata']['processing_time_ms']}ms")

asyncio.run(advanced_example())
```

## 📖 API Reference

### Core Framework

#### `AIOrchestrationFramework`

Main framework class that orchestrates all components.

```python
class AIOrchestrationFramework:
    def __init__(self, config: Dict[str, Any])
    
    @classmethod
    def from_config(cls, config_path: str) -> 'AIOrchestrationFramework'
    
    async def process_request(
        self,
        prompt: str,
        task_type: str,
        user_id: str,
        require_consensus: Optional[bool] = None,
        min_providers: int = 1,
        complexity_override: Optional[TaskComplexity] = None
    ) -> Dict[str, Any]
    
    async def add_model_provider(
        self, 
        provider: ModelProvider, 
        connector: BaseModelConnector
    ) -> None
    
    async def get_performance_metrics(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]
```

### Model Integration

#### `BaseModelConnector`

Abstract base class for model connectors.

```python
class BaseModelConnector(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse
    
    @abstractmethod
    async def validate_connection(self) -> bool
    
    @abstractmethod
    def get_supported_models(self) -> List[str]
```

#### `ModelRequest` and `ModelResponse`

```python
class ModelRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    model_specific_params: Dict[str, Any] = {}

class ModelResponse(BaseModel):
    content: str
    provider: ModelProvider
    model_name: str
    tokens_used: int
    latency_ms: float
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
```

### Consensus Strategies

#### `ConsensusStrategy`

Base class for implementing consensus algorithms.

```python
class ConsensusStrategy(ABC):
    @abstractmethod
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ModelResponse
    
    @abstractmethod
    def calculate_confidence(self, responses: List[ModelResponse]) -> float
```

#### Built-in Consensus Strategies

- **`SimilarityConsensus`**: Uses semantic similarity to find consensus
- **`VotingConsensus`**: Implements majority voting with confidence weighting
- **`ConfidenceConsensus`**: Selects response with highest confidence score

### Task Routing

#### `TaskRouter`

Handles task complexity evaluation and model selection.

```python
class TaskRouter:
    def register_evaluator(self, task_type: str, evaluator: ComplexityEvaluator)
    
    async def evaluate_and_route(
        self, 
        prompt: str, 
        task_type: str
    ) -> Dict[str, Any]
    
    def add_routing_rule(self, rule: RoutingRule)
```

#### `ComplexityEvaluator`

Base class for task complexity evaluation.

```python
class ComplexityEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, prompt: str) -> TaskComplexity
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Code Generation │    │    Business Agents              │ │
│  │   Tools         │    │ (Automation, Extraction, etc.)  │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                 Core Orchestration Engine                   │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Task Router    │  │ Consensus Engine│  │Self-Improve │ │
│  │ (Complexity     │  │ (Pluggable      │  │   Engine    │ │
│  │  Evaluation)    │  │  Algorithms)    │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                             │                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Memory & Data Manager                       │ │
│  │  [Conversation History] [Embeddings] [Performance Data]│ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│              Multi-Model Connector Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   OpenAI    │  │   Google    │  │     Anthropic       │ │
│  │  Connector  │  │  Connector  │  │    Connector        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Request Processing Flow**:
   - Application layer submits request to framework
   - Task Router evaluates complexity and selects models
   - Model Connectors generate responses in parallel
   - Consensus Engine finds the best response
   - Memory Manager stores interaction data
   - Self-Improvement Engine analyzes performance

2. **Data Flow**:
   - All interactions stored in persistent database
   - Performance metrics tracked for continuous improvement
   - Embeddings cached for similarity calculations
   - Configuration updates propagated to all components

## 🔌 Plugin Development

### Creating a Custom Model Connector

```python
from ai_framework.models import BaseModelConnector, ModelRequest, ModelResponse
from ai_framework.models import ModelProvider

class CustomModelConnector(BaseModelConnector):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        # Implement your model's API call
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            ) as response:
                data = await response.json()
                
                return ModelResponse(
                    content=data["text"],
                    provider=ModelProvider.CUSTOM,
                    model_name="custom-model-v1",
                    tokens_used=data["tokens_used"],
                    latency_ms=data["latency_ms"]
                )
    
    async def validate_connection(self) -> bool:
        # Implement connection validation
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except:
            return False
    
    def get_supported_models(self) -> List[str]:
        return ["custom-model-v1", "custom-model-v2"]

# Register the connector
framework.model_registry.register_connector(
    ModelProvider.CUSTOM,
    CustomModelConnector("your-api-key", "https://api.custom-model.com")
)
```

### Creating a Custom Consensus Strategy

```python
from ai_framework.consensus import ConsensusStrategy
import numpy as np

class MLBasedConsensus(ConsensusStrategy):
    def __init__(self, model_path: str):
        # Load your trained ML model
        self.model = load_consensus_model(model_path)
    
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ModelResponse:
        # Extract features from responses
        features = self._extract_features(responses, task_context)
        
        # Use ML model to predict best response
        scores = self.model.predict(features)
        best_idx = np.argmax(scores)
        
        result = responses[best_idx]
        result.confidence_score = float(scores[best_idx])
        result.metadata["consensus_method"] = "ml_based"
        
        return result
    
    def _extract_features(self, responses: List[ModelResponse], task_context: Dict[str, Any]) -> np.ndarray:
        # Implement feature extraction logic
        features = []
        for response in responses:
            features.append([
                response.latency_ms,
                response.tokens_used,
                len(response.content),
                # Add more features...
            ])
        return np.array(features)

# Register the strategy
framework.consensus_manager.register_strategy(
    "ml_based",
    MLBasedConsensus("path/to/consensus_model.pkl")
)
```

### Creating a Custom Complexity Evaluator

```python
from ai_framework.routing import ComplexityEvaluator, TaskComplexity
import re

class BusinessTaskEvaluator(ComplexityEvaluator):
    def __init__(self):
        self.simple_patterns = [
            r'\b(summary|summarize|list|extract)\b',
            r'\b(simple|basic|quick)\b'
        ]
        self.complex_patterns = [
            r'\b(analysis|analyze|strategy|optimize)\b',
            r'\b(complex|detailed|comprehensive)\b'
        ]
    
    async def evaluate(self, prompt: str) -> TaskComplexity:
        prompt_lower = prompt.lower()
        
        simple_matches = sum(1 for pattern in self.simple_patterns 
                           if re.search(pattern, prompt_lower))
        complex_matches = sum(1 for pattern in self.complex_patterns 
                            if re.search(pattern, prompt_lower))
        
        if complex_matches > simple_matches:
            return TaskComplexity.COMPLEX
        elif simple_matches > 0:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM

# Register the evaluator
framework.task_router.register_evaluator(
    "business_automation",
    BusinessTaskEvaluator()
)
```

## 🛠️ Advanced Usage Examples

### Building a Code Generation Tool

```python
from ai_framework import AIOrchestrationFramework
from ai_framework.applications import CodeGenerationApp

class SmartCodeGenerator:
    def __init__(self, config_path: str):
        self.framework = AIOrchestrationFramework.from_config(config_path)
        self.setup_code_specific_config()
    
    def setup_code_specific_config(self):
        # Add specialized routing for code tasks
        self.framework.task_router.add_routing_rule({
            "task_type": "code_generation",
            "complexity": "simple",
            "providers": ["openai"],
            "require_consensus": False
        })
        
        self.framework.task_router.add_routing_rule({
            "task_type": "code_review", 
            "complexity": "any",
            "providers": ["openai", "anthropic"],
            "require_consensus": True
        })
    
    async def generate_code(
        self, 
        description: str, 
        language: str = "python",
        include_tests: bool = True
    ) -> Dict[str, Any]:
        # Generate main code
        code_prompt = f"""
        Generate {language} code for: {description}
        
        Requirements:
        - Clean, readable code
        - Proper error handling
        - Documentation/comments
        - Follow best practices
        """
        
        code_result = await self.framework.process_request(
            prompt=code_prompt,
            task_type="code_generation",
            user_id="code_generator"
        )
        
        # Generate tests if requested
        if include_tests:
            test_prompt = f"""
            Generate comprehensive unit tests for this {language} code:
            
            {code_result['response']}
            
            Requirements:
            - Test edge cases
            - Use appropriate testing framework
            - Good test coverage
            """
            
            test_result = await self.framework.process_request(
                prompt=test_prompt,
                task_type="test_generation", 
                user_id="code_generator"
            )
            
            return {
                "code": code_result['response'],
                "tests": test_result['response'],
                "metadata": {
                    "code_metadata": code_result['metadata'],
                    "test_metadata": test_result['metadata']
                }
            }
        
        return {
            "code": code_result['response'],
            "metadata": code_result['metadata']
        }

# Usage
async def main():
    generator = SmartCodeGenerator("config.yaml")
    
    result = await generator.generate_code(
        description="A function to parse CSV files with error handling",
        language="python",
        include_tests=True
    )
    
    print("Generated Code:")
    print(result["code"])
    print("\nGenerated Tests:")
    print(result["tests"])

asyncio.run(main())
```

### Building a Business Automation Agent

```python
class BusinessAutomationAgent:
    def __init__(self, config_path: str):
        self.framework = AIOrchestrationFramework.from_config(config_path)
        self.setup_business_workflows()
    
    def setup_business_workflows(self):
        # Configure for business tasks
        self.framework.task_router.register_evaluator(
            "document_analysis",
            BusinessDocumentEvaluator()
        )
    
    async def analyze_document(self, document_text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze business documents with multi-model consensus"""
        
        prompts = {
            "summary": f"Provide a concise summary of this document:\n\n{document_text}",
            "key_points": f"Extract key points and action items from:\n\n{document_text}",
            "sentiment": f"Analyze the sentiment and tone of:\n\n{document_text}",
            "compliance": f"Review for compliance issues:\n\n{document_text}"
        }
        
        if analysis_type not in prompts:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        result = await self.framework.process_request(
            prompt=prompts[analysis_type],
            task_type="document_analysis",
            user_id="business_agent",
            require_consensus=True
        )
        
        return result
    
    async def generate_report(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate business reports from structured data"""
        
        prompt = f"""
        Generate a {report_type} report based on this data:
        
        {json.dumps(data, indent=2)}
        
        Requirements:
        - Professional formatting
        - Clear insights and recommendations
        - Executive summary
        - Data-driven conclusions
        """
        
        result = await self.framework.process_request(
            prompt=prompt,
            task_type="report_generation",
            user_id="business_agent"
        )
        
        return result["response"]

# Usage
async def business_example():
    agent = BusinessAutomationAgent("config.yaml")
    
    # Analyze a contract
    contract_text = "... contract content ..."
    analysis = await agent.analyze_document(contract_text, "compliance")
    
    # Generate summary report
    report_data = {
        "analysis_results": analysis,
        "date": "2024-01-15",
        "reviewer": "AI Agent"
    }
    
    report = await agent.generate_report(report_data, "compliance_review")
    print(report)

asyncio.run(business_example())
```

## 📊 Performance Monitoring

### Built-in Metrics

The framework automatically tracks:

- **Response Latency**: Time to generate responses
- **Token Usage**: Token consumption per provider
- **Success Rates**: Percentage of successful requests
- **Consensus Quality**: Agreement levels between models
- **User Satisfaction**: Optional feedback integration
- **Cost Tracking**: API usage costs per provider

### Accessing Metrics

```python
# Get performance overview
metrics = await framework.get_performance_metrics(time_window_hours=24)

print(f"Average latency: {metrics['avg_latency_ms']}ms")
print(f"Total requests: {metrics['total_requests']}")
print(f"Success rate: {metrics['success_rate']}%")
print(f"Top performing provider: {metrics['best_provider']}")

# Get detailed provider comparison
provider_stats = await framework.get_provider_comparison()
for provider, stats in provider_stats.items():
    print(f"{provider}: {stats['avg_latency']}ms, {stats['success_rate']}%")

# Export data for analysis
await framework.export_metrics("metrics_export.json", days=7)
```

### Custom Metrics

```python
from ai_framework.monitoring import MetricsCollector

class CustomMetricsCollector(MetricsCollector):
    async def collect_custom_metric(self, metric_name: str, value: float, tags: Dict[str, str]):
        # Implement custom metric collection
        # Could integrate with Prometheus, DataDog, etc.
        pass

# Register custom collector
framework.add_metrics_collector(CustomMetricsCollector())
```

## 🔧 Troubleshooting

### Common Issues

#### API Key Configuration
```bash
# Verify API keys are set
python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10] + '...')"

# Test individual connections
python -m ai_framework.tools.test_connections
```

#### Database Connection Issues
```python
# Test database connectivity
from ai_framework.memory import MemoryManager

async def test_db():
    try:
        manager = MemoryManager("your_database_url")
        await manager.test_connection()
        print("Database connection successful")
    except Exception as e:
        print(f"Database error: {e}")

asyncio.run(test_db())
```

#### Performance Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor resource usage
from ai_framework.monitoring import ResourceMonitor

monitor = ResourceMonitor()
await monitor.start_monitoring()
```

### Debug Mode

```python
# Enable comprehensive debugging
framework = AIOrchestrationFramework.from_config(
    "config.yaml",
    debug=True,
    log_level="DEBUG",
    enable_profiling=True
)

# Access debug information
debug_info = await framework.get_debug_info()
print(debug_info["last_request_trace"])
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModelProviderNotFound` | API key not configured | Check environment variables |
| `ConsensusTimeout` | Models taking too long | Increase timeout or reduce providers |
| `DatabaseConnectionError` | DB unavailable | Check database URL and permissions |
| `InsufficientResponses` | All models failed | Check API quotas and connectivity |

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_framework

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run tests with live API calls (requires API keys)
pytest tests/live/ --live-api
```

### Writing Tests

```python
import pytest
from ai_framework.testing import MockModelConnector, TestFramework

@pytest.fixture
async def test_framework():
    framework = TestFramework()
    
    # Add mock connectors for testing
    framework.add_mock_connector("openai", MockModelConnector("openai"))
    framework.add_mock_connector("anthropic", MockModelConnector("anthropic"))
    
    return framework

async def test_basic_request_processing(test_framework):
    result = await test_framework.process_request(
        prompt="Test prompt",
        task_type="test",
        user_id="test_user"
    )
    
    assert result["response"] is not None
    assert result["metadata"]["providers_used"] == ["openai"]

async def test_consensus_mechanism(test_framework):
    # Configure for consensus testing
    result = await test_framework.process_request(
        prompt="Test consensus",
        task_type="test",
        user_id="test_user",
        require_consensus=True,
        min_providers=2
    )
    
    assert len(result["metadata"]["providers_used"]) >= 2
    assert result["metadata"]["consensus_applied"] is True
```

## 📈 Performance Optimization

### Best Practices

1. **Async Usage**: Always use async/await for model calls
2. **Connection Pooling**: Configure appropriate connection limits
3. **Caching**: Enable Redis for embedding and response caching
4. **Batch Processing**: Process multiple requests in parallel
5. **Resource Limits**: Set appropriate timeout and token limits

### Configuration Tuning

```yaml
# High-performance configuration
performance:
  max_concurrent_requests: 50
  request_timeout: 30
  connection_pool_size: 20
  enable_response_caching: true
  cache_ttl: 3600  # 1 hour
  
  # Model-specific optimizations
  models:
    openai:
      max_concurrent: 20
      timeout: 15
    anthropic:
      max_concurrent: 15
      timeout: 20
    google:
      max_concurrent: 25
      timeout: 10
```

### Monitoring Performance

```python
from ai_framework.performance import PerformanceProfiler

async def optimized_processing():
    with PerformanceProfiler() as profiler:
        result = await framework.process_request(...)
        
    print(f"Processing time: {profiler.elapsed_time}ms")
    print(f"Memory usage: {profiler.memory_usage}MB")
    print(f"API calls made: {profiler.api_calls}")
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/ai-orchestration-framework.git
cd ai-orchestration-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Standards

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing
- **Google-style** docstrings

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Write tests for your changes
3. Ensure all tests pass: `pytest`
4. Update documentation as needed
5. Submit a pull request with a clear description

### Areas for Contribution

- 🔗 **New Model Connectors**: Add support for additional AI providers
- 🧠 **Consensus Algorithms**: Implement new consensus strategies
- 📊 **Monitoring Tools**: Enhance observability and metrics
- 🔌 **Plugins**: Build domain-specific extensions
- 📚 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Expand test coverage and scenarios

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, and Google for their excellent AI APIs
- The Python asyncio community for async best practices
- Contributors and beta testers who helped shape this framework

## 📞 Support

- **Documentation**: [https://ai-framework.readthedocs.io](https://ai-framework.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/ai-orchestration-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ai-orchestration-framework/discussions)
- **Email**: support@ai-framework.dev

## 🗺️ Roadmap

### Version 1.1 (Next Release)
- [ ] GraphQL API support
- [ ] Enhanced plugin system
- [ ] Real-time streaming responses
- [ ] Multi-user support

### Version 1.2 (Future)
- [ ] Distributed processing
- [ ] Advanced ML-based routing
- [ ] Integration with vector databases
- [ ] Cost optimization algorithms

### Version 2.0 (Long-term)
- [ ] Agent-to-agent communication
- [ ] Advanced reasoning chains
- [ ] Custom model fine-tuning integration
- [ ] Enterprise security features

---

**Built with ❤️ by the AI Orchestration Framework Team**