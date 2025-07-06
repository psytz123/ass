"""
Model Connectors for AI Orchestration Framework

This module provides connectors for various AI model providers including
OpenAI, Anthropic, Google, and AbacusAI.
"""

import asyncio
import aiohttp
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    ABACUS_AI = "abacus_ai"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ModelRequest:
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    model_specific_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_specific_params is None:
            self.model_specific_params = {}


@dataclass
class ModelResponse:
    content: str
    provider: ModelProvider
    model_name: str
    tokens_used: int
    latency_ms: float
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModelConnector(ABC):
    """Abstract base class for AI model connectors"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from the model"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the model provider"""
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this provider"""
        pass

    def _calculate_confidence(self, response_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on response metadata"""
        # Default implementation - can be overridden by specific connectors
        return 0.8


class OpenAIConnector(BaseModelConnector):
    """Connector for OpenAI models"""

    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.default_model = "gpt-4o"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": request.model_specific_params.get("model", self.default_model),
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4000
        }

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"OpenAI API error: {response.status}")
            
            data = await response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=data["choices"][0]["message"]["content"],
                provider=ModelProvider.OPENAI,
                model_name=payload["model"],
                tokens_used=data["usage"]["total_tokens"],
                latency_ms=latency_ms,
                confidence_score=self._calculate_confidence(data),
                metadata={"finish_reason": data["choices"][0]["finish_reason"]}
            )

    async def validate_connection(self) -> bool:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(
                f"{self.base_url}/models",
                headers=headers
            ) as response:
                return response.status == 200
        except:
            return False

    def get_supported_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]


class AnthropicConnector(BaseModelConnector):
    """Connector for Anthropic Claude models"""

    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.default_model = "claude-3-5-sonnet-20241022"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start_time = time.time()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": request.model_specific_params.get("model", self.default_model),
            "max_tokens": request.max_tokens or 4000,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}]
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        async with self.session.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"Anthropic API error: {response.status}")
            
            data = await response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=data["content"][0]["text"],
                provider=ModelProvider.ANTHROPIC,
                model_name=payload["model"],
                tokens_used=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                latency_ms=latency_ms,
                confidence_score=self._calculate_confidence(data),
                metadata={"stop_reason": data["stop_reason"]}
            )

    async def validate_connection(self) -> bool:
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # Test with a minimal request
            payload = {
                "model": self.default_model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                return response.status == 200
        except:
            return False

    def get_supported_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]


class GoogleConnector(BaseModelConnector):
    """Connector for Google Gemini models"""

    def __init__(self, api_key: str):
        super().__init__(api_key, "https://generativelanguage.googleapis.com/v1beta")
        self.default_model = "gemini-pro"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start_time = time.time()
        
        model_name = request.model_specific_params.get("model", self.default_model)
        url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
        
        contents = []
        if request.system_prompt:
            contents.append({"parts": [{"text": request.system_prompt}]})
        contents.append({"parts": [{"text": request.prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens or 4000
            }
        }

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Google API error: {response.status}")
            
            data = await response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            if "candidates" not in data or not data["candidates"]:
                raise Exception("No response from Google API")
            
            candidate = data["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]
            
            return ModelResponse(
                content=content,
                provider=ModelProvider.GOOGLE,
                model_name=model_name,
                tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
                latency_ms=latency_ms,
                confidence_score=self._calculate_confidence(data),
                metadata={"finish_reason": candidate.get("finishReason", "STOP")}
            )

    async def validate_connection(self) -> bool:
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            async with self.session.get(url) as response:
                return response.status == 200
        except:
            return False

    def get_supported_models(self) -> List[str]:
        return ["gemini-pro", "gemini-pro-vision"]


class AbacusAIConnector(BaseModelConnector):
    """Connector for AbacusAI custom deployed models"""

    def __init__(self, api_key: str, deployment_id: str, base_url: Optional[str] = None):
        super().__init__(api_key, base_url or "https://api.abacus.ai")
        self.deployment_id = deployment_id
        self.default_model = "custom-model"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "deployment_id": self.deployment_id,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4000
        }

        if request.system_prompt:
            payload["system_prompt"] = request.system_prompt

        async with self.session.post(
            f"{self.base_url}/v1/deployments/{self.deployment_id}/predict",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"AbacusAI API error: {response.status}")
            
            data = await response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=data["prediction"]["text"],
                provider=ModelProvider.ABACUS_AI,
                model_name=self.default_model,
                tokens_used=data.get("tokens_used", 0),
                latency_ms=latency_ms,
                confidence_score=self._calculate_confidence(data),
                metadata={"deployment_id": self.deployment_id}
            )

    async def validate_connection(self) -> bool:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(
                f"{self.base_url}/v1/deployments/{self.deployment_id}",
                headers=headers
            ) as response:
                return response.status == 200
        except:
            return False

    def get_supported_models(self) -> List[str]:
        return ["custom-model"]


class ModelRegistry:
    """Registry for managing model connectors"""

    def __init__(self):
        self.connectors: Dict[ModelProvider, BaseModelConnector] = {}

    def register_connector(self, provider: ModelProvider, connector: BaseModelConnector):
        """Register a model connector"""
        self.connectors[provider] = connector

    def get_connector(self, provider: ModelProvider) -> Optional[BaseModelConnector]:
        """Get a model connector by provider"""
        return self.connectors.get(provider)

    def get_all_providers(self) -> List[ModelProvider]:
        """Get all registered providers"""
        return list(self.connectors.keys())

    async def validate_all_connections(self) -> Dict[ModelProvider, bool]:
        """Validate connections for all registered connectors"""
        results = {}
        for provider, connector in self.connectors.items():
            async with connector:
                results[provider] = await connector.validate_connection()
        return results

