"""
Machine Learning based Complexity Evaluator using LightGBM
Replaces the simple keyword-based evaluator with ML model
"""

import pickle
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from core.task_router import ComplexityEvaluator
from core.types import TaskComplexity

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    logger.warning(f"LightGBM not available: {e}. ML-based routing will not be available.")

class MLComplexityEvaluator(ComplexityEvaluator):
    """Machine Learning based complexity evaluator using LightGBM"""
    
    def __init__(self, model_path: str = "models/routing_classifier.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.load_model()
    
    def load_model(self):
        """Load the trained LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available")
            return
        
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded ML routing model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning(f"Model file not found at {self.model_path}")
    
    async def evaluate(self, prompt: str, task_type: str) -> TaskComplexity:
        """Evaluate complexity using ML model"""
        if not self.model or not LIGHTGBM_AVAILABLE:
            # Fallback to simple evaluation
            return await self._simple_fallback(prompt, task_type)
        
        try:
            # Extract features
            features = self.feature_extractor.extract(prompt, task_type)
            
            # Convert to numpy array in correct order
            feature_vector = self._features_to_vector(features)
            
            # Predict complexity and probability
            prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
            
            # Get confidence score
            confidence = max(probabilities)
            
            # Log prediction details
            logger.debug(f"ML Prediction: {prediction}, Confidence: {confidence:.2f}")
            
            # Map prediction to TaskComplexity
            if prediction == 0:
                return TaskComplexity.SIMPLE
            elif prediction == 1:
                return TaskComplexity.MEDIUM
            else:
                return TaskComplexity.COMPLEX
                
        except Exception as e:
            logger.error(f"Error in ML evaluation: {e}")
            return await self._simple_fallback(prompt, task_type)
    
    async def evaluate_with_confidence(self, prompt: str, task_type: str) -> Tuple[TaskComplexity, float]:
        """Evaluate complexity and return confidence score"""
        if not self.model or not LIGHTGBM_AVAILABLE:
            complexity = await self._simple_fallback(prompt, task_type)
            return complexity, 0.5  # Low confidence for fallback
        
        try:
            features = self.feature_extractor.extract(prompt, task_type)
            feature_vector = self._features_to_vector(features)
            
            prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
            confidence = max(probabilities)
            
            # Map to TaskComplexity
            if prediction == 0:
                complexity = TaskComplexity.SIMPLE
            elif prediction == 1:
                complexity = TaskComplexity.MEDIUM
            else:
                complexity = TaskComplexity.COMPLEX
            
            return complexity, confidence
            
        except Exception as e:
            logger.error(f"Error in ML evaluation: {e}")
            complexity = await self._simple_fallback(prompt, task_type)
            return complexity, 0.5
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order"""
        # Define feature order (must match training)
        feature_order = [
            'length', 'word_count', 'avg_word_length', 'sentence_count',
            'has_code_keywords', 'has_system_keywords', 'has_ml_keywords',
            'has_data_keywords', 'question_marks', 'exclamation_marks',
            'code_blocks', 'urls', 'business_terms', 'technical_terms',
            # Task type one-hot encoding
            'is_code_generation', 'is_code_optimization', 'is_text_generation',
            'is_analysis', 'is_question_answering', 'is_other'
        ]
        
        # Create vector
        vector = []
        for feature in feature_order:
            if feature in features:
                vector.append(float(features[feature]))
            else:
                # Handle task type features
                if feature.startswith('is_'):
                    task_type = feature[3:]  # Remove 'is_'
                    vector.append(1.0 if features.get('task_type') == task_type else 0.0)
                else:
                    vector.append(0.0)
        
        return np.array(vector)
    
    async def _simple_fallback(self, prompt: str, task_type: str) -> TaskComplexity:
        """Simple keyword-based fallback when ML model unavailable"""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        # Simple heuristics
        complex_indicators = ['optimize', 'architecture', 'system design', 'algorithm']
        simple_indicators = ['hello', 'simple', 'basic', 'quick']
        
        complex_score = sum(1 for ind in complex_indicators if ind in prompt_lower)
        simple_score = sum(1 for ind in simple_indicators if ind in prompt_lower)
        
        if word_count > 100 or complex_score > 2:
            return TaskComplexity.COMPLEX
        elif word_count < 20 and simple_score > 0:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM


class FeatureExtractor:
    """Extract features from prompts for ML model"""
    
    def extract(self, prompt: str, task_type: str) -> Dict:
        """Extract all features from a prompt"""
        prompt_lower = prompt.lower()
        words = prompt.split()
        
        features = {
            # Basic features
            'length': len(prompt),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'sentence_count': prompt.count('.') + prompt.count('?') + prompt.count('!'),
            
            # Complexity indicators
            'has_code_keywords': int(any(kw in prompt_lower for kw in [
                'function', 'class', 'algorithm', 'optimize', 'refactor', 'implement',
                'debug', 'fix', 'error', 'bug'
            ])),
            'has_system_keywords': int(any(kw in prompt_lower for kw in [
                'architecture', 'system', 'design', 'scalability', 'performance',
                'distributed', 'microservice', 'infrastructure'
            ])),
            'has_ml_keywords': int(any(kw in prompt_lower for kw in [
                'machine learning', 'neural', 'model', 'train', 'predict',
                'ai', 'deep learning', 'classifier'
            ])),
            'has_data_keywords': int(any(kw in prompt_lower for kw in [
                'database', 'query', 'data', 'sql', 'nosql', 'schema',
                'table', 'index', 'migration'
            ])),
            
            # Linguistic features
            'question_marks': prompt.count('?'),
            'exclamation_marks': prompt.count('!'),
            'code_blocks': prompt.count('```'),
            'urls': prompt.lower().count('http'),
            
            # Domain indicators
            'business_terms': sum(1 for term in [
                'revenue', 'cost', 'budget', 'profit', 'roi', 'kpi',
                'customer', 'sales', 'marketing'
            ] if term in prompt_lower),
            'technical_terms': sum(1 for term in [
                'api', 'backend', 'frontend', 'server', 'client', 'protocol',
                'endpoint', 'request', 'response'
            ] if term in prompt_lower),
            
            # Task type
            'task_type': task_type
        }
        
        return features