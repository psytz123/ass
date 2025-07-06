# AI Orchestration Framework

## Overview

The AI Orchestration Framework is a sophisticated Python-based system that provides intelligent routing and consensus mechanisms for multiple AI model providers. The system enables seamless integration with OpenAI, Anthropic, and Google AI models, automatically routing tasks based on complexity analysis and finding consensus across multiple model responses.

## System Architecture

### Backend Architecture
- **Flask Web Framework**: RESTful API server with SQLAlchemy ORM for database operations
- **Modular Core**: Plugin-based architecture with separate modules for model connectors, task routing, consensus algorithms, and memory management
- **Async Processing**: Asynchronous model calls for improved performance and parallel processing
- **Configuration Management**: YAML-based configuration with environment variable overrides

### Frontend Architecture
- **Bootstrap-based UI**: Dark-themed responsive web interface with Feather icons
- **Dashboard Interface**: Real-time performance monitoring and provider status
- **Test Framework**: Interactive testing interface for experimenting with different prompts and configurations

## Key Components

### 1. Model Connectors (`core/model_connectors.py`)
- **BaseModelConnector**: Abstract interface for AI model providers
- **OpenAIConnector**: Integration with OpenAI GPT models
- **AnthropicConnector**: Integration with Claude models  
- **GoogleConnector**: Integration with Gemini models
- **AbacusAIConnector**: Integration with custom AbacusAI deployed models
- **Auto-failover**: Automatic fallback to alternative models on failures

### 2. Task Router (`core/task_router.py`)
- **ComplexityEvaluator**: Analyzes prompt complexity using keyword analysis
- **Performance-based Routing**: Routes tasks based on historical model performance
- **Custom Routing Rules**: Database-stored rules for specific routing preferences
- **Dynamic Optimization**: Continuously improves routing decisions based on outcomes

### 3. Consensus Engine (`core/consensus.py`)
- **SimilarityConsensus**: Uses sentence transformers for semantic similarity analysis
- **Multiple Strategies**: Voting, confidence-weighted, and similarity-based consensus
- **Pluggable Architecture**: Easy integration of custom consensus algorithms
- **Quality Metrics**: Confidence scoring and consensus quality assessment

### 4. Memory Manager (`core/memory.py`)
- **Conversation Storage**: Persistent storage of all user interactions
- **Performance Tracking**: Detailed metrics collection for each model call
- **Embedding Cache**: Efficient caching of embeddings for similarity calculations
- **Analytics Support**: Historical data aggregation for performance analysis

## Data Flow

1. **Request Processing**: User requests are received through Flask API endpoints
2. **Complexity Analysis**: Task complexity is evaluated using keyword analysis and prompt structure
3. **Model Selection**: Tasks are routed to appropriate models based on complexity and performance history
4. **Parallel Execution**: Multiple models may be called simultaneously for consensus
5. **Consensus Calculation**: Responses are analyzed for agreement using semantic similarity
6. **Result Storage**: All interactions and performance metrics are stored in the database
7. **Response Delivery**: Final consensus response is returned to the user

## External Dependencies

### AI Model Providers
- **OpenAI API**: GPT-4o and other OpenAI models
- **Anthropic API**: Claude Sonnet and other Anthropic models
- **Google AI API**: Gemini models
- **AbacusAI API**: Custom deployed models and ML solutions

### Machine Learning Libraries
- **sentence-transformers**: For semantic similarity analysis
- **scikit-learn**: For cosine similarity calculations
- **numpy**: For numerical computations

### Web Framework
- **Flask**: Web application framework
- **SQLAlchemy**: Database ORM with support for SQLite/PostgreSQL

## Deployment Strategy

### Development Environment
- **SQLite Database**: Local file-based database for development
- **Environment Variables**: API keys and configuration stored in environment
- **Debug Mode**: Flask debug mode enabled for development

### Production Considerations
- **PostgreSQL Database**: Recommended for production deployments
- **Environment Configuration**: Secure API key management
- **Session Management**: Configurable session secrets
- **Proxy Support**: Built-in proxy fix middleware for deployment behind reverse proxies

## User Preferences

Preferred communication style: Simple, everyday language.

## New Features (July 06, 2025)

### Real-time Processing with WebSocket Support
- **WebSocket Integration**: Added Flask-SocketIO for bi-directional real-time communication
- **Live Updates**: Clients receive processing stage updates in real-time (complexity evaluation, provider selection, consensus calculation)
- **Event Handlers**: Comprehensive WebSocket event handling for connection management, request processing, and error handling
- **Periodic Metrics**: Automatic metrics broadcasting to subscribed clients every 5 seconds

### Batch Processing Capabilities
- **REST API Endpoint**: New `/api/batch` endpoint for processing multiple requests simultaneously
- **WebSocket Batch Support**: Real-time batch processing with progress tracking via WebSocket
- **Concurrent Processing**: Requests processed in chunks of 5 for optimal performance
- **Progress Tracking**: Live progress updates with per-item completion notifications
- **Error Handling**: Individual error tracking for each batch item without affecting others

### User Interface Enhancements
- **Real-time Demo Page**: Interactive demonstration of WebSocket and batch processing features
- **Live Metrics Dashboard**: Real-time performance metrics with subscription-based updates
- **Visual Progress Indicators**: Progress bars and stage indicators for processing feedback
- **Interactive Testing**: Forms for testing both single and batch request processing

## Advanced Features (July 06, 2025)

### Cost Tracking Module
- **Real-time Cost Monitoring**: Track API usage costs across all providers (OpenAI, Anthropic, Google AI)
- **Budget Management**: Set daily and monthly budget limits with automatic alerts
- **Provider Cost Breakdown**: Detailed cost analysis by provider, model, and task type
- **Cost Visualization**: Interactive charts showing cost trends over time and provider distribution
- **API Endpoint**: `/api/costs` for programmatic access to cost metrics

### A/B Testing Framework
- **Routing Strategy Comparison**: Test different routing strategies (performance-based, cost-optimized, latency-optimized, quality-focused)
- **Statistical Analysis**: Automatic calculation of success rates, latency, cost, and quality metrics
- **Winner Determination**: Statistical significance testing with confidence intervals
- **Visual Dashboard**: Interactive interface for creating, monitoring, and analyzing A/B tests
- **API Endpoints**: `/api/ab-tests` for test management and results retrieval

### Usage Analytics
- **Comprehensive Insights**: Track request patterns, user behavior, and task distribution
- **Temporal Analysis**: Identify peak usage hours and weekly patterns
- **User Segmentation**: Automatic classification of users (power, regular, casual)
- **Content Analysis**: Popular topics extraction and prompt length analysis
- **Recommendations Engine**: Data-driven suggestions for system optimization
- **API Endpoint**: `/api/usage-analytics` for accessing analytics data

### REST API Documentation
- **OpenAPI/Swagger Integration**: Complete REST API documentation using Flask-RESTX
- **Interactive API Explorer**: Swagger UI interface for testing endpoints at `/api/v1/docs`
- **API Versioning**: RESTful API endpoints organized under `/api/v1/` namespace
- **Comprehensive Documentation**: Detailed parameter descriptions, response schemas, and example payloads
- **Test Interface**: Built-in request testing directly from documentation

### Webhook System
- **Event-Driven Notifications**: Comprehensive webhook management for external system integration
- **Supported Events**: REQUEST_COMPLETED, REQUEST_FAILED, CONSENSUS_ACHIEVED, PERFORMANCE_THRESHOLD, BUDGET_ALERT
- **Secure Delivery**: HMAC signature validation for webhook security
- **Retry Logic**: Automatic retry with exponential backoff for failed deliveries
- **Management UI**: Visual interface for creating, testing, and managing webhooks at `/api/webhooks`
- **Real-time Triggers**: Webhook events fired automatically during request processing

## Changelog

Changelog:
- July 05, 2025. Initial setup
- July 05, 2025. Comprehensive test suite implemented and validated - all core tests passing, framework operational
- July 06, 2025. Added AbacusAI integration as fourth AI provider with custom model deployment capabilities
- July 06, 2025. Implemented WebSocket support for real-time updates and batch processing capabilities
- July 06, 2025. Implemented advanced monitoring and analytics features:
  - Cost Tracking module for API usage monitoring and budget management
  - A/B Testing framework for comparing routing strategies
  - Usage Analytics system for comprehensive user behavior analysis
- July 06, 2025. Added REST API Documentation and Webhook System:
  - Flask-RESTX integration for OpenAPI/Swagger documentation
  - Complete webhook management system for external notifications
  - Interactive API testing interface at `/api/v1/docs`
- July 06, 2025. Fixed configuration issues - achieved 100% component validation:
  - Fixed Config object access to use proper dictionary structure
  - Added model connector properties to AIOrchestrationFramework for backward compatibility
  - Fixed ModelRequest task_type issues in context memory module
  - Added missing uuid import to framework
  - All 5 validation tests now passing with active OpenAI, Anthropic, and Google connectors
- July 06, 2025. UI Improvements and Knowledge Integration Testing:
  - Fixed invalid Feather icons (replaced 'rocket' with 'star' and 'route' with 'share-2')
  - Integrated comprehensive knowledge validation test suite with 10 domains
  - Added test runner script for easy knowledge assessment execution
  - Knowledge tests cover programming, mathematics, science, history, business, reasoning, creativity, and more