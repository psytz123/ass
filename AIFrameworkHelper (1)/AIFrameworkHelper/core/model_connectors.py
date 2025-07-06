import asyncio
import logging
import time
import os
from abc import ABC, abstractmethod
from typing import Optional

# Import AI SDKs
import openai
from anthropic import Anthropic
from google import genai
from abacusai import ApiClient

from .types import ModelProvider, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

class BaseModelConnector(ABC):
    """Abstract base class for model connectors"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = None
    
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the connection to the model works"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get list of supported models for this provider"""
        pass

class OpenAIConnector(BaseModelConnector):
    """OpenAI model connector"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        super().__init__(api_key, model_name)
        self.provider = ModelProvider.OPENAI
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI"""
        start_time = time.time()
        
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    **request.model_specific_params
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=response.choices[0].message.content,
                provider=self.provider,
                model_name=self.model_name,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms,
                confidence_score=None,  # OpenAI doesn't provide confidence scores
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump() if response.usage else {}
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate OpenAI connection"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection validation failed: {e}")
            return False
    
    def get_supported_models(self) -> list[str]:
        """Get supported OpenAI models"""
        return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]

class AnthropicConnector(BaseModelConnector):
    """Anthropic model connector"""
    
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model_name)
        self.provider = ModelProvider.ANTHROPIC
        self.client = Anthropic(api_key=api_key)
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Anthropic"""
        start_time = time.time()
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=request.max_tokens or 4000,
                    temperature=request.temperature,
                    system=request.system_prompt or "",
                    messages=[{"role": "user", "content": request.prompt}],
                    **request.model_specific_params
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=response.content[0].text,
                provider=self.provider,
                model_name=self.model_name,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency_ms,
                confidence_score=None,  # Anthropic doesn't provide confidence scores
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate Anthropic connection"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic connection validation failed: {e}")
            return False
    
    def get_supported_models(self) -> list[str]:
        """Get supported Anthropic models"""
        return ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"]

class GoogleConnector(BaseModelConnector):
    """Google AI model connector"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        super().__init__(api_key, model_name)
        self.provider = ModelProvider.GOOGLE
        self.client = genai.Client(api_key=api_key)
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Google AI"""
        start_time = time.time()
        
        try:
            # Prepare content
            contents = []
            if request.system_prompt:
                # Google uses system instructions differently
                contents.append(request.system_prompt + "\n\n" + request.prompt)
            else:
                contents.append(request.prompt)
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        temperature=request.temperature,
                        max_output_tokens=request.max_tokens,
                        **request.model_specific_params
                    )
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=response.text or "",
                provider=self.provider,
                model_name=self.model_name,
                tokens_used=0,  # Google doesn't always provide token counts
                latency_ms=latency_ms,
                confidence_score=None,  # Google doesn't provide confidence scores
                metadata={
                    "finish_reason": getattr(response.candidates[0] if response.candidates else None, 'finish_reason', None),
                    "safety_ratings": getattr(response.candidates[0] if response.candidates else None, 'safety_ratings', [])
                }
            )
            
        except Exception as e:
            logger.error(f"Google AI generation error: {e}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate Google AI connection"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents="test"
                )
            )
            return True
        except Exception as e:
            logger.error(f"Google AI connection validation failed: {e}")
            return False
    
    def get_supported_models(self) -> list[str]:
        """Get supported Google models"""
        return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]

class AbacusAIConnector(BaseModelConnector):
    """AbacusAI model connector"""
    
    def __init__(self, api_key: str, model_name: str = "custom-model"):
        super().__init__(api_key, model_name)
        self.provider = ModelProvider.ABACUSAI
        self.client = ApiClient(api_key)
        self.deployment_id = None  # Will be set when deploying models
        self.deployment_token = None
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using AbacusAI deployed model"""
        start_time = time.time()
        
        try:
            if not self.deployment_id:
                raise ValueError("No deployed model available. Deploy a model first using deploy_model()")
            
            # Prepare query data from prompt
            # For custom models, this would need to be adapted based on your model's input schema
            query_data = {
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens or 4000
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                query_data["system_prompt"] = request.system_prompt
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Get deployment and make prediction
            deployment = await loop.run_in_executor(
                None,
                lambda: self.client.describe_deployment(self.deployment_id)
            )
            
            prediction = await loop.run_in_executor(
                None,
                lambda: deployment.predict(query_data=query_data)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response content (this may need adjustment based on your model's output format)
            content = prediction.get('response', '') if isinstance(prediction, dict) else str(prediction)
            
            return ModelResponse(
                content=content,
                provider=self.provider,
                model_name=self.model_name,
                tokens_used=0,  # AbacusAI doesn't always provide token counts
                latency_ms=latency_ms,
                confidence_score=prediction.get('confidence') if isinstance(prediction, dict) else None,
                metadata={
                    "deployment_id": self.deployment_id,
                    "prediction_result": prediction
                }
            )
            
        except Exception as e:
            logger.error(f"AbacusAI generation error: {e}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate AbacusAI connection"""
        try:
            loop = asyncio.get_event_loop()
            # Try to list projects to validate API key
            await loop.run_in_executor(
                None,
                lambda: self.client.list_projects()
            )
            return True
        except Exception as e:
            logger.error(f"AbacusAI connection validation failed: {e}")
            return False
    
    def get_supported_models(self) -> list[str]:
        """Get supported AbacusAI models"""
        return ["custom-model", "python-model", "forecasting-model", "recommendation-model"]
    
    async def set_deployment(self, deployment_id: str, deployment_token: str = None):
        """Set the deployment ID for making predictions"""
        self.deployment_id = deployment_id
        self.deployment_token = deployment_token
        logger.info(f"AbacusAI deployment set to: {deployment_id}")
    
    async def list_deployments(self):
        """List available deployments"""
        try:
            loop = asyncio.get_event_loop()
            projects = await loop.run_in_executor(
                None,
                lambda: self.client.list_projects()
            )
            
            deployments = []
            for project in projects:
                project_deployments = await loop.run_in_executor(
                    None,
                    lambda: project.list_deployments()
                )
                deployments.extend(project_deployments)
            
            return deployments
        except Exception as e:
            logger.error(f"Failed to list AbacusAI deployments: {e}")
            return []

def create_connector(provider: ModelProvider, api_key: str, model_name: str = None) -> BaseModelConnector:
    """Factory function to create model connectors"""
    if provider == ModelProvider.OPENAI:
        return OpenAIConnector(api_key, model_name or "gpt-4o")
    elif provider == ModelProvider.ANTHROPIC:
        return AnthropicConnector(api_key, model_name or "claude-sonnet-4-20250514")
    elif provider == ModelProvider.GOOGLE:
        return GoogleConnector(api_key, model_name or "gemini-2.5-flash")
    elif provider == ModelProvider.ABACUSAI:
        return AbacusAIConnector(api_key, model_name or "custom-model")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
