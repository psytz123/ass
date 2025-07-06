"""
Training module for LightGBM routing classifier
Trains the ML model for intelligent task routing
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    logger.warning(f"LightGBM not available: {e}. Training will not be available.")

class RoutingModelTrainer:
    """Trains LightGBM model for task complexity routing"""
    
    def __init__(self, data_path: str = "datasets/prompt_training_data.jsonl"):
        self.data_path = Path(data_path)
        self.model_path = Path("models/routing_classifier.pkl")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.data_path}")
        
        # Load JSONL data
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if not data:
            raise ValueError("No training data found")
        
        # Extract features and labels
        features_list = []
        labels = []
        
        for item in data:
            features = item.get('features', {})
            label = item.get('complexity_label', 'medium')
            
            # Create feature vector
            feature_vector = self._create_feature_vector(features, item.get('task_type'))
            features_list.append(feature_vector)
            labels.append(label)
        
        # Create DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(labels)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Loaded {len(X)} training samples")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _create_feature_vector(self, features: Dict, task_type: str) -> Dict:
        """Create feature vector from features dict"""
        # Basic features
        vector = {
            'length': features.get('length', 0),
            'word_count': features.get('word_count', 0),
            'avg_word_length': features.get('avg_word_length', 0),
            'sentence_count': features.get('sentence_count', 0),
            'has_code_keywords': features.get('has_code_keywords', 0),
            'has_system_keywords': features.get('has_system_keywords', 0),
            'has_ml_keywords': features.get('has_ml_keywords', 0),
            'has_data_keywords': features.get('has_data_keywords', 0),
            'question_marks': features.get('question_marks', 0),
            'exclamation_marks': features.get('exclamation_marks', 0),
            'code_blocks': features.get('code_blocks', 0),
            'urls': features.get('urls', 0),
            'business_terms': features.get('business_terms', 0),
            'technical_terms': features.get('technical_terms', 0)
        }
        
        # One-hot encode task type
        task_types = ['code_generation', 'code_optimization', 'text_generation',
                     'analysis', 'question_answering', 'other']
        
        for tt in task_types:
            vector[f'is_{tt}'] = 1 if task_type == tt else 0
        
        return vector
    
    def train(self, test_size: float = 0.2, random_state: int = 42, 
             cv_folds: int = 5) -> Dict:
        """Train the LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        # Load data
        X, y = self.load_training_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )
        
        # Define LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(self.label_encoder.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        logger.info("Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
        )
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Get classification report
        report = classification_report(
            y_test, y_pred_labels, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Calculate cross-validation score
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_model = lgb.LGBMClassifier(**params, n_estimators=100)
        cv_scores = cross_val_score(cv_model, X, y_encoded, cv=cv_folds, scoring='f1_weighted')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Save model
        self._save_model()
        
        results = {
            'test_accuracy': report['accuracy'],
            'test_f1_weighted': report['weighted avg']['f1-score'],
            'cv_f1_scores': cv_scores.tolist(),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'classification_report': report,
            'feature_importance': feature_importance.to_dict('records')[:10],  # Top 10
            'confusion_matrix': confusion_matrix(y_test, y_pred_labels).tolist(),
            'label_classes': self.label_encoder.classes_.tolist()
        }
        
        logger.info(f"Training completed. Test F1: {results['test_f1_weighted']:.3f}, "
                   f"CV F1: {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
        
        return results
    
    def _save_model(self):
        """Save trained model and encoder"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def generate_sample_data(self, n_samples: int = 500):
        """Generate sample training data for demonstration"""
        from core.ml_routing.data_collector import PromptDataCollector
        from core.types import TaskComplexity
        
        collector = PromptDataCollector()
        
        # Sample prompts for each complexity level
        sample_prompts = {
            TaskComplexity.SIMPLE: [
                "Hello, how are you?",
                "What's the weather today?",
                "Tell me a joke",
                "What is 2 + 2?",
                "Translate 'hello' to Spanish"
            ],
            TaskComplexity.MEDIUM: [
                "Write a Python function to sort a list",
                "Explain how recursion works",
                "Create a simple REST API endpoint",
                "Debug this code snippet",
                "Write a SQL query to join two tables"
            ],
            TaskComplexity.COMPLEX: [
                "Design a microservices architecture for an e-commerce platform",
                "Optimize this algorithm for better time complexity",
                "Implement a distributed caching system with Redis",
                "Create a machine learning pipeline for fraud detection",
                "Build a real-time data processing system using Apache Kafka"
            ]
        }
        
        task_types = ['code_generation', 'code_optimization', 'text_generation',
                     'analysis', 'question_answering']
        
        # Generate samples
        samples_per_category = n_samples // (len(sample_prompts) * len(task_types))
        
        for complexity, prompts in sample_prompts.items():
            for task_type in task_types:
                for i in range(samples_per_category):
                    # Create variations of prompts
                    base_prompt = prompts[i % len(prompts)]
                    variation = f"{base_prompt} - variation {i}"
                    
                    collector.add_manual_prompt(
                        prompt=variation,
                        task_type=task_type,
                        complexity=complexity
                    )
        
        stats = collector.get_dataset_stats()
        logger.info(f"Generated {stats['total']} sample training prompts")
        return stats