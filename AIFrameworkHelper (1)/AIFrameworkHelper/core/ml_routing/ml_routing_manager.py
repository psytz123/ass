"""
ML Routing Manager - Coordinates ML-based routing functionality
Integrates data collection, model training, and deployment
"""

import logging
import asyncio
from typing import Dict, Optional
from pathlib import Path

from .data_collector import PromptDataCollector
from .ml_complexity_evaluator import MLComplexityEvaluator
from .training import RoutingModelTrainer
from core.types import TaskComplexity

logger = logging.getLogger(__name__)

class MLRoutingManager:
    """Manages ML-based routing system"""
    
    def __init__(self, data_path: str = "datasets/prompt_training_data.jsonl",
                 model_path: str = "models/routing_classifier.pkl"):
        self.data_collector = PromptDataCollector(data_path)
        self.trainer = RoutingModelTrainer(data_path)
        self.evaluator = MLComplexityEvaluator(model_path)
        self.model_exists = Path(model_path).exists()
    
    async def initialize(self) -> bool:
        """Initialize ML routing system"""
        if not self.model_exists:
            logger.info("ML model not found. Generating sample data and training...")
            await self.bootstrap_training()
        return True
    
    async def bootstrap_training(self):
        """Bootstrap the ML system with sample data"""
        try:
            # Generate sample training data
            logger.info("Generating sample training data...")
            stats = self.trainer.generate_sample_data(n_samples=500)
            logger.info(f"Generated training data: {stats}")
            
            # Train initial model
            logger.info("Training initial ML routing model...")
            results = self.trainer.train()
            
            logger.info(f"Initial model trained successfully:")
            logger.info(f"- Test F1 Score: {results['test_f1_weighted']:.3f}")
            logger.info(f"- CV F1 Score: {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
            
            # Reload evaluator with new model
            self.evaluator.load_model()
            self.model_exists = True
            
        except Exception as e:
            logger.error(f"Error during bootstrap training: {e}")
            raise
    
    def collect_prompt(self, prompt: str, task_type: str, 
                      complexity: TaskComplexity, processing_time_ms: float,
                      providers_used: list, consensus_confidence: Optional[float] = None):
        """Collect a new prompt for training data"""
        try:
            # Add to collector with automatic feature extraction
            self.data_collector.add_manual_prompt(
                prompt=prompt,
                task_type=task_type,
                complexity=complexity
            )
        except Exception as e:
            logger.error(f"Error collecting prompt: {e}")
    
    async def retrain_model(self) -> Dict:
        """Retrain the model with collected data"""
        try:
            # First, collect any new data from database
            new_prompts = self.data_collector.collect_from_database(limit=1000)
            logger.info(f"Collected {new_prompts} new prompts from database")
            
            # Get dataset statistics
            stats = self.data_collector.get_dataset_stats()
            
            if stats['total'] < 100:
                logger.warning(f"Insufficient training data: {stats['total']} samples")
                return {"error": "Insufficient training data", "stats": stats}
            
            # Train model
            results = self.trainer.train()
            
            # Reload evaluator with new model
            self.evaluator.load_model()
            
            return {
                "success": True,
                "stats": stats,
                "training_results": results
            }
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """Get ML routing statistics"""
        dataset_stats = self.data_collector.get_dataset_stats()
        
        return {
            "model_exists": self.model_exists,
            "dataset_stats": dataset_stats,
            "model_path": str(Path("models/routing_classifier.pkl").absolute()),
            "data_path": str(Path("datasets/prompt_training_data.jsonl").absolute())
        }