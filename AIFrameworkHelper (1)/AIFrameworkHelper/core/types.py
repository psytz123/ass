from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    ABACUSAI = "abacusai"

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
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class ConsensusResult:
    final_response: ModelResponse
    confidence: float
    individual_responses: List[ModelResponse]
    consensus_method: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FrameworkResult:
    response: str
    metadata: Dict[str, Any]
    consensus_result: Optional[ConsensusResult] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
