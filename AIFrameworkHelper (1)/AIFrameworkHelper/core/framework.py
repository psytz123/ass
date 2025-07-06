import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from .model_connectors import create_connector, BaseModelConnector
from .task_router import TaskRouter
from .consensus import ConsensusEngine
from .memory import MemoryManager
from .optimization import PerformanceOptimizer, AdaptiveRouter
from .context_memory import ContextMemory, ContextAwareRouter
from .cost_tracker import CostTracker
from .ab_testing import ABTestingFramework
from .usage_analytics import UsageAnalytics
from .types import ModelProvider, TaskComplexity, ModelRequest, ModelResponse, FrameworkResult
from config import Config

logger = logging.getLogger(__name__)

class AIOrchestrationFramework:
    """Main AI Orchestration Framework class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connectors: Dict[ModelProvider, BaseModelConnector] = {}
        self.task_router = TaskRouter(config)
        self.consensus_engine = ConsensusEngine(config)
        self.memory_manager = MemoryManager(config)
        
        # Initialize optimization components
        self.performance_optimizer = PerformanceOptimizer()
        self.adaptive_router = AdaptiveRouter(self.performance_optimizer)
        
        # Initialize context memory
        self.context_memory = ContextMemory()
        self.context_aware_router = ContextAwareRouter(self.context_memory)
        
        # Initialize analytics and optimization components
        self.cost_tracker = CostTracker()
        self.ab_testing = ABTestingFramework()
        self.usage_analytics = UsageAnalytics()
        
        # Initialize model connectors
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize model connectors for available providers"""
        
        for provider_name, model_config in self.config.models.items():
            if not model_config.api_key:
                logger.warning(f"No API key found for {provider_name}, skipping")
                continue
            
            try:
                provider = ModelProvider(provider_name)
                connector = create_connector(
                    provider=provider,
                    api_key=model_config.api_key,
                    model_name=model_config.default_model
                )
                self.connectors[provider] = connector
                logger.info(f"Initialized {provider_name} connector")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} connector: {e}")
    
    @property
    def model_connectors(self) -> Dict[ModelProvider, BaseModelConnector]:
        """Get available model connectors"""
        return self.connectors
    
    @property
    def openai_connector(self) -> Optional[BaseModelConnector]:
        """Get OpenAI connector if available"""
        return self.connectors.get(ModelProvider.OPENAI)
    
    @property
    def anthropic_connector(self) -> Optional[BaseModelConnector]:
        """Get Anthropic connector if available"""
        return self.connectors.get(ModelProvider.ANTHROPIC)
    
    @property
    def google_connector(self) -> Optional[BaseModelConnector]:
        """Get Google connector if available"""
        return self.connectors.get(ModelProvider.GOOGLE)
    
    @property
    def abacusai_connector(self) -> Optional[BaseModelConnector]:
        """Get AbacusAI connector if available"""
        return self.connectors.get(ModelProvider.ABACUSAI)
    
    @classmethod
    def from_config(cls, config_path: str) -> 'AIOrchestrationFramework':
        """Create framework instance from config file"""
        config = Config.from_file(config_path)
        return cls(config)
    
    async def process_request(
        self,
        prompt: str,
        task_type: str,
        user_id: str,
        require_consensus: Optional[bool] = None,
        min_providers: int = 1,
        complexity_override: Optional[TaskComplexity] = None,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None
    ) -> FrameworkResult:
        """Process a request through the AI orchestration framework"""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")
            
            if not self.connectors:
                raise RuntimeError("No model connectors available")
            
            # Get available providers
            available_providers = list(self.connectors.keys())
            
            # Route the task
            routing_result = await self.task_router.evaluate_and_route(
                prompt=prompt,
                task_type=task_type,
                available_providers=available_providers,
                complexity_override=complexity_override
            )
            
            # Use adaptive router for optimized provider selection
            optimized_providers = await self.adaptive_router.get_optimal_providers(
                task_type=task_type,
                complexity=routing_result['complexity'],
                available_providers=available_providers,
                require_consensus=routing_result['require_consensus']
            )
            
            # Apply A/B testing configuration if applicable
            ab_config = self.ab_testing.get_routing_config_for_user(
                user_id=user_id,
                task_type=task_type,
                complexity=routing_result['complexity']
            )
            
            # Apply context-aware routing adjustments
            routing_with_context = await self.context_aware_router.adjust_routing_for_context(
                user_id=user_id,
                base_routing={
                    'selected_providers': ab_config.get('providers') or optimized_providers,
                    'require_consensus': ab_config.get('require_consensus', routing_result['require_consensus'])
                },
                task_type=task_type
            )
            
            selected_providers = routing_with_context['selected_providers']
            need_consensus = require_consensus if require_consensus is not None else routing_with_context['require_consensus']
            test_group = ab_config.get('test_group')
            
            # Ensure minimum providers
            if len(selected_providers) < min_providers:
                selected_providers = available_providers[:min_providers]
            
            # Create model request
            model_request = ModelRequest(
                prompt=prompt,
                temperature=temperature or self.config.models['openai'].temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            
            # Enhance request with context memory
            enhanced_request = await self.context_memory.enhance_request_with_context(
                request=model_request,
                user_id=user_id,
                task_type=task_type
            )
            
            # Generate responses from selected providers using enhanced request
            logger.info(f"Generating responses from {len(selected_providers)} providers")
            responses = await self._generate_responses(selected_providers, enhanced_request, task_type)
            
            # Handle consensus if needed
            consensus_result = None
            final_response = None
            
            if need_consensus and len(responses) > 1:
                # Calculate provider reliability for consensus
                provider_reliability = {}
                for provider in selected_providers:
                    perf_data = await self.memory_manager.get_performance_metrics(
                        time_window_hours=168,  # Last week
                        provider=provider,
                        task_type=task_type
                    )
                    provider_reliability[provider.value] = perf_data.get('success_rate', 0.8)
                
                # Use enhanced consensus with context
                consensus_result = await self.consensus_engine.find_consensus(
                    responses=responses,
                    task_context={
                        'task_type': task_type,
                        'user_id': user_id,
                        'prompt': prompt,
                        'provider_reliability': provider_reliability,
                        'provider_performance': {
                            p.value: self.performance_optimizer.performance_scores[p.value].get(task_type, 0.5)
                            for p in selected_providers
                        }
                    },
                    strategy_override='confidence' if routing_result['complexity'] == TaskComplexity.COMPLEX else None
                )
                final_response = consensus_result.final_response.content
            else:
                # Use the best single response
                final_response = responses[0].content if responses else "No response generated"
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Calculate and track costs
            total_cost = 0.0
            for response in responses:
                # Extract model name from response metadata if available
                model_name = response.metadata.get('model', 'default') if response.metadata else 'default'
                
                # Estimate tokens if not provided
                tokens = response.tokens_used
                input_tokens = int(tokens * 0.3)  # Rough estimate
                output_tokens = int(tokens * 0.7)
                
                cost = self.cost_tracker.calculate_request_cost(
                    provider=response.provider,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                total_cost += cost
            
            # Record A/B test outcome if applicable
            if test_group:
                quality_score = sum(r.confidence_score for r in responses) / len(responses) if responses else 0
                self.ab_testing.record_request_outcome(
                    test_group=test_group,
                    success=True,
                    latency_ms=processing_time_ms,
                    cost=total_cost,
                    quality_score=quality_score
                )
            
            # Store conversation and performance data
            await self._store_results(
                user_id=user_id,
                prompt=prompt,
                response=final_response,
                task_type=task_type,
                providers_used=selected_providers,
                consensus_confidence=consensus_result.confidence if consensus_result else None,
                processing_time_ms=processing_time_ms,
                complexity_score=routing_result.get('complexity_score'),
                responses=responses
            )
            
            # Update performance optimizer with results
            for i, response in enumerate(responses):
                await self.performance_optimizer.update_performance(
                    provider=response.provider,
                    task_type=task_type,
                    success=True,  # If we got here, the request was successful
                    latency_ms=response.latency_ms,
                    quality_score=response.confidence_score
                )
            
            # Prepare result metadata
            metadata = {
                'providers_used': [p.value for p in selected_providers],
                'processing_time_ms': processing_time_ms,
                'task_complexity': routing_result['complexity'].value,
                'consensus_used': need_consensus,
                'consensus_confidence': consensus_result.confidence if consensus_result else None,
                'routing_strategy': routing_result['metadata'].get('routing_strategy'),
                'total_tokens': sum(r.tokens_used for r in responses),
                'total_cost': round(total_cost, 4),
                'ab_test_group': test_group
            }
            
            logger.info(f"Request processed successfully in {processing_time_ms:.2f}ms")
            
            # Trigger webhook for successful request
            from core.webhooks import webhook_manager, WebhookEvent
            webhook_data = {
                'user_id': user_id,
                'task_type': task_type.value if hasattr(task_type, 'value') else task_type,
                'providers_used': [p.value for p in selected_providers],
                'processing_time_ms': processing_time_ms,
                'consensus_confidence': consensus_result.confidence if consensus_result else None,
                'cost': round(total_cost, 4),
                'response_preview': final_response[:200] + '...' if len(final_response) > 200 else final_response
            }
            
            # Fire and forget webhook
            asyncio.create_task(
                webhook_manager.trigger_event(
                    WebhookEvent.REQUEST_COMPLETED,
                    webhook_data,
                    metadata={'request_id': str(uuid.uuid4())}
                )
            )
            
            # Check for consensus achievement
            if consensus_result and consensus_result.confidence > 0.8:
                asyncio.create_task(
                    webhook_manager.trigger_event(
                        WebhookEvent.CONSENSUS_ACHIEVED,
                        {
                            'confidence': consensus_result.confidence,
                            'providers': [p.value for p in selected_providers],
                            'user_id': user_id
                        }
                    )
                )
            
            return FrameworkResult(
                response=final_response,
                metadata=metadata,
                consensus_result=consensus_result
            )
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Trigger webhook for failed request
            from core.webhooks import webhook_manager, WebhookEvent
            webhook_data = {
                'user_id': user_id,
                'task_type': task_type.value if hasattr(task_type, 'value') else task_type,
                'error_message': str(e),
                'processing_time_ms': processing_time_ms,
                'prompt_preview': prompt[:200] + '...' if len(prompt) > 200 else prompt
            }
            
            # Fire and forget webhook
            asyncio.create_task(
                webhook_manager.trigger_event(
                    WebhookEvent.REQUEST_FAILED,
                    webhook_data,
                    metadata={'request_id': str(uuid.uuid4())}
                )
            )
            
            # Return error response
            return FrameworkResult(
                response=f"Error processing request: {str(e)}",
                metadata={
                    'error': True,
                    'error_message': str(e),
                    'processing_time_ms': processing_time_ms
                }
            )
    
    async def _generate_responses(
        self,
        providers: List[ModelProvider],
        request: ModelRequest,
        task_type: str
    ) -> List[ModelResponse]:
        """Generate responses from multiple providers concurrently"""
        
        tasks = []
        for provider in providers:
            if provider in self.connectors:
                task = self._generate_single_response(
                    self.connectors[provider], request, task_type
                )
                tasks.append(task)
        
        if not tasks:
            raise RuntimeError("No valid connectors found for selected providers")
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Provider {providers[i].value} failed: {response}")
            else:
                valid_responses.append(response)
        
        if not valid_responses:
            raise RuntimeError("All providers failed to generate responses")
        
        return valid_responses
    
    async def _generate_single_response(
        self,
        connector: BaseModelConnector,
        request: ModelRequest,
        task_type: str
    ) -> ModelResponse:
        """Generate response from a single provider"""
        
        try:
            response = await connector.generate(request)
            
            # Store performance metrics
            await self.memory_manager.store_model_performance(
                response=response,
                task_type=task_type,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Store error metrics
            error_response = ModelResponse(
                content="",
                provider=connector.provider,
                model_name=connector.model_name,
                tokens_used=0,
                latency_ms=0.0
            )
            
            await self.memory_manager.store_model_performance(
                response=error_response,
                task_type=task_type,
                success=False,
                error_message=str(e)
            )
            
            raise
    
    async def _store_results(
        self,
        user_id: str,
        prompt: str,
        response: str,
        task_type: str,
        providers_used: List[ModelProvider],
        consensus_confidence: Optional[float],
        processing_time_ms: float,
        complexity_score: Optional[float],
        responses: List[ModelResponse]
    ):
        """Store conversation and performance results"""
        
        try:
            # Store conversation
            await self.memory_manager.store_conversation(
                user_id=user_id,
                prompt=prompt,
                response=response,
                task_type=task_type,
                providers_used=providers_used,
                consensus_confidence=consensus_confidence,
                processing_time_ms=processing_time_ms,
                complexity_score=complexity_score,
                metadata={
                    'response_count': len(responses),
                    'individual_latencies': [r.latency_ms for r in responses]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
    
    async def add_model_provider(
        self,
        provider: ModelProvider,
        connector: BaseModelConnector
    ):
        """Add a new model provider"""
        
        # Validate connection
        if not await connector.validate_connection():
            raise RuntimeError(f"Failed to validate connection for {provider.value}")
        
        self.connectors[provider] = connector
        logger.info(f"Added model provider: {provider.value}")
    
    async def get_performance_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance metrics for the framework"""
        
        return await self.memory_manager.get_performance_metrics(
            time_window_hours=time_window_hours
        )
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 50,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        
        return await self.memory_manager.get_conversation_history(
            user_id=user_id,
            limit=limit,
            task_type=task_type
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available model providers"""
        
        return [provider.value for provider in self.connectors.keys()]
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types"""
        
        return [
            "code_generation",
            "code_optimization", 
            "text_analysis",
            "question_answering",
            "summarization",
            "creative_writing",
            "data_analysis",
            "general"
        ]
