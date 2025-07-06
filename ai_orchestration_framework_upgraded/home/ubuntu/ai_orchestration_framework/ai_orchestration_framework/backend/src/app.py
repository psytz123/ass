"""
Flask Application for AI Orchestration Framework

This is the main Flask application that provides REST API endpoints
for the AI orchestration framework.
"""

import os
import asyncio
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import Dict, Any, List, Optional
import json

from core.model_connectors import (
    ModelRegistry, OpenAIConnector, AnthropicConnector, 
    GoogleConnector, AbacusAIConnector, ModelRequest, ModelProvider
)
from core.task_router import TaskRouter
from core.consensus import ConsensusManager
from core.memory import MemoryManager


class AIOrchestrationFramework:
    """Main framework class that orchestrates all components"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_registry = ModelRegistry()
        self.task_router = TaskRouter()
        self.consensus_manager = ConsensusManager()
        self.memory_manager = MemoryManager(
            self.config.get('database_path', 'ai_framework.db')
        )
        
        # Initialize model connectors
        self._initialize_connectors()

    def _initialize_connectors(self):
        """Initialize AI model connectors"""
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.model_registry.register_connector(
                ModelProvider.OPENAI, 
                OpenAIConnector(openai_key)
            )

        # Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.model_registry.register_connector(
                ModelProvider.ANTHROPIC, 
                AnthropicConnector(anthropic_key)
            )

        # Google
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            self.model_registry.register_connector(
                ModelProvider.GOOGLE, 
                GoogleConnector(google_key)
            )

        # AbacusAI
        abacus_key = os.getenv('ABACUS_AI_API_KEY')
        abacus_deployment = os.getenv('ABACUS_AI_DEPLOYMENT_ID')
        if abacus_key and abacus_deployment:
            self.model_registry.register_connector(
                ModelProvider.ABACUS_AI, 
                AbacusAIConnector(abacus_key, abacus_deployment)
            )

    async def process_request(self, prompt: str, task_type: str = "general",
                            user_id: str = "default", require_consensus: Optional[bool] = None,
                            min_providers: Optional[int] = None,
                            consensus_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Process a request through the AI orchestration framework"""
        start_time = time.time()

        try:
            # Get available providers
            available_providers = self.model_registry.get_all_providers()
            if not available_providers:
                raise Exception("No AI model providers configured")

            # Route the task
            routing_result = await self.task_router.route_task(
                prompt=prompt,
                task_type=task_type,
                available_providers=available_providers,
                require_consensus=require_consensus,
                min_providers=min_providers
            )

            selected_providers = routing_result['providers']
            need_consensus = routing_result['require_consensus']

            # Generate responses from selected providers
            responses = []
            for provider in selected_providers:
                connector = self.model_registry.get_connector(provider)
                if connector:
                    try:
                        async with connector:
                            model_request = ModelRequest(prompt=prompt)
                            response = await connector.generate(model_request)
                            responses.append(response)
                            
                            # Record performance
                            await self.memory_manager.store_performance_metric(
                                response, task_type, success=True
                            )
                            self.task_router.record_performance(
                                provider, response.latency_ms, True, response.confidence_score or 0.8
                            )
                    except Exception as e:
                        print(f"Error with provider {provider}: {e}")
                        # Record failure
                        self.task_router.record_performance(provider, 5000, False, 0.0)

            if not responses:
                raise Exception("No successful responses from any provider")

            # Find consensus if needed
            if need_consensus and len(responses) > 1:
                consensus_result = await self.consensus_manager.find_consensus(
                    responses, consensus_strategy
                )
            else:
                # Use the best single response
                best_response = max(responses, key=lambda r: r.confidence_score or 0.8)
                from core.consensus import ConsensusResult
                consensus_result = ConsensusResult(
                    selected_response=best_response,
                    confidence_score=best_response.confidence_score or 0.8,
                    consensus_method="single_response",
                    all_responses=responses
                )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Store conversation
            await self.memory_manager.store_conversation(
                user_id=user_id,
                prompt=prompt,
                task_type=task_type,
                complexity=routing_result['complexity'].value,
                providers_used=[p.value for p in selected_providers],
                consensus_result=consensus_result,
                processing_time_ms=processing_time_ms
            )

            return {
                "response": consensus_result.selected_response.content,
                "metadata": {
                    "providers_used": [p.value for p in selected_providers],
                    "consensus_method": consensus_result.consensus_method,
                    "confidence_score": consensus_result.confidence_score,
                    "processing_time_ms": processing_time_ms,
                    "task_complexity": routing_result['complexity'].value,
                    "consensus_applied": need_consensus and len(responses) > 1,
                    "total_responses": len(responses)
                }
            }

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return {
                "error": str(e),
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "success": False
                }
            }

    async def get_performance_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics"""
        return await self.memory_manager.get_aggregated_performance(time_window_hours)

    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        for provider in self.model_registry.get_all_providers():
            connector = self.model_registry.get_connector(provider)
            if connector:
                async with connector:
                    is_available = await connector.validate_connection()
                    status[provider.value] = {
                        "available": is_available,
                        "models": connector.get_supported_models()
                    }
        return status


# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize framework
framework = AIOrchestrationFramework()


@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_request():
    """Process an AI request"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt in request"}), 400
    
    prompt = data['prompt']
    task_type = data.get('task_type', 'general')
    user_id = data.get('user_id', 'default')
    require_consensus = data.get('require_consensus')
    min_providers = data.get('min_providers')
    consensus_strategy = data.get('consensus_strategy')
    
    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            framework.process_request(
                prompt=prompt,
                task_type=task_type,
                user_id=user_id,
                require_consensus=require_consensus,
                min_providers=min_providers,
                consensus_strategy=consensus_strategy
            )
        )
        return jsonify(result)
    finally:
        loop.close()


@app.route('/api/providers/status', methods=['GET'])
def get_provider_status():
    """Get status of all AI providers"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        status = loop.run_until_complete(framework.get_provider_status())
        return jsonify(status)
    finally:
        loop.close()


@app.route('/api/metrics/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics"""
    time_window = request.args.get('time_window_hours', 24, type=int)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        metrics = loop.run_until_complete(
            framework.get_performance_metrics(time_window)
        )
        return jsonify(metrics)
    finally:
        loop.close()


@app.route('/api/metrics/routing', methods=['GET'])
def get_routing_metrics():
    """Get task routing performance metrics"""
    metrics = framework.task_router.get_performance_metrics()
    return jsonify(metrics)


@app.route('/api/conversations/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history for a user"""
    user_id = request.args.get('user_id', 'default')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        conversations = loop.run_until_complete(
            framework.memory_manager.get_conversation_history(user_id, limit, offset)
        )
        # Convert to dictionaries for JSON serialization
        conversation_dicts = [conv.to_dict() for conv in conversations]
        return jsonify(conversation_dicts)
    finally:
        loop.close()


@app.route('/api/usage/statistics', methods=['GET'])
def get_usage_statistics():
    """Get usage statistics"""
    time_window = request.args.get('time_window_hours', 24, type=int)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        stats = loop.run_until_complete(
            framework.memory_manager.get_usage_statistics(time_window)
        )
        return jsonify(stats)
    finally:
        loop.close()


@app.route('/api/consensus/strategies', methods=['GET'])
def get_consensus_strategies():
    """Get available consensus strategies"""
    strategies = framework.consensus_manager.get_available_strategies()
    return jsonify({"strategies": strategies})


@app.route('/api/test', methods=['POST'])
def test_framework():
    """Test endpoint for framework functionality"""
    data = request.get_json()
    test_prompt = data.get('prompt', 'Hello, this is a test prompt.')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            framework.process_request(
                prompt=test_prompt,
                task_type='test',
                user_id='test_user'
            )
        )
        return jsonify(result)
    finally:
        loop.close()


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

