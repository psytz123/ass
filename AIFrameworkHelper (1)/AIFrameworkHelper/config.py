import os
import yaml
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    default_model: str
    max_tokens: int = 4000
    temperature: float = 0.7
    api_key: str = None

@dataclass
class ConsensusConfig:
    default_strategy: str = "similarity"
    similarity_threshold: float = 0.8
    confidence_threshold: float = 0.7

@dataclass
class RoutingConfig:
    default_strategy: str = "performance_based"
    complexity_threshold: float = 0.7
    require_consensus_for_complex: bool = True

class Config:
    def __init__(self, config_path: str = None):
        self.config_data = {}
        
        # Load from YAML file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}
        
        # Initialize model configurations
        self.models = {
            'openai': ModelConfig(
                default_model=self.get('models.openai.default_model', 'gpt-4o'),
                max_tokens=self.get('models.openai.max_tokens', 4000),
                temperature=self.get('models.openai.temperature', 0.7),
                api_key=os.environ.get('OPENAI_API_KEY')
            ),
            'anthropic': ModelConfig(
                default_model=self.get('models.anthropic.default_model', 'claude-sonnet-4-20250514'),
                max_tokens=self.get('models.anthropic.max_tokens', 4000),
                temperature=self.get('models.anthropic.temperature', 0.7),
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            ),
            'google': ModelConfig(
                default_model=self.get('models.google.default_model', 'gemini-2.5-flash'),
                max_tokens=self.get('models.google.max_tokens', 4000),
                temperature=self.get('models.google.temperature', 0.7),
                api_key=os.environ.get('GEMINI_API_KEY')
            ),
            'abacusai': ModelConfig(
                default_model=self.get('models.abacusai.default_model', 'custom-model'),
                max_tokens=self.get('models.abacusai.max_tokens', 4000),
                temperature=self.get('models.abacusai.temperature', 0.7),
                api_key=os.environ.get('ABACUSAI_API_KEY')
            )
        }
        
        # Initialize consensus configuration
        self.consensus = ConsensusConfig(
            default_strategy=self.get('consensus.default_strategy', 'similarity'),
            similarity_threshold=self.get('consensus.similarity_threshold', 0.8),
            confidence_threshold=self.get('consensus.confidence_threshold', 0.7)
        )
        
        # Initialize routing configuration
        self.routing = RoutingConfig(
            default_strategy=self.get('routing.default_strategy', 'performance_based'),
            complexity_threshold=self.get('routing.complexity_threshold', 0.7),
            require_consensus_for_complex=self.get('routing.require_consensus_for_complex', True)
        )
        
        # Framework settings
        self.enable_caching = self.get('framework.enable_caching', True)
        self.enable_metrics = self.get('framework.enable_metrics', True)
        self.log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @classmethod
    def from_file(cls, config_path: str):
        """Create config instance from YAML file"""
        return cls(config_path)
