"""
Data Collector for ML-based routing training
Collects prompts and metadata for training LightGBM classifier
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from app import db
from models import Conversation, ModelPerformance
from core.types import TaskComplexity

logger = logging.getLogger(__name__)

class PromptDataCollector:
    """Collects and stores prompts for ML training"""
    
    def __init__(self, data_path: str = "datasets/prompt_training_data.jsonl"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.collected_prompts = []
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing training data if available"""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.collected_prompts.append(json.loads(line))
                logger.info(f"Loaded {len(self.collected_prompts)} existing prompts")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
    
    def collect_from_database(self, limit: Optional[int] = None) -> int:
        """Collect prompts from conversation history in database"""
        query = db.session.query(
            Conversation.prompt,
            Conversation.task_type,
            Conversation.complexity_score,
            Conversation.providers_used,
            Conversation.processing_time_ms,
            Conversation.consensus_confidence
        )
        
        if limit:
            query = query.limit(limit)
        
        conversations = query.all()
        new_prompts_count = 0
        
        for conv in conversations:
            # Skip if already collected
            if self._is_duplicate(conv.prompt):
                continue
            
            # Determine complexity label based on score
            complexity_label = self._score_to_complexity(conv.complexity_score)
            
            # Parse providers used
            try:
                providers = json.loads(conv.providers_used)
            except:
                providers = []
            
            prompt_data = {
                "prompt": conv.prompt,
                "task_type": conv.task_type,
                "complexity_score": conv.complexity_score or 0.5,
                "complexity_label": complexity_label,
                "providers_used": providers,
                "processing_time_ms": conv.processing_time_ms,
                "consensus_confidence": conv.consensus_confidence,
                "features": self._extract_features(conv.prompt, conv.task_type),
                "collected_at": datetime.utcnow().isoformat()
            }
            
            self.collected_prompts.append(prompt_data)
            new_prompts_count += 1
        
        # Save updated dataset
        self._save_data()
        logger.info(f"Collected {new_prompts_count} new prompts from database")
        return new_prompts_count
    
    def add_manual_prompt(self, prompt: str, task_type: str, 
                         complexity: TaskComplexity, domain: Optional[str] = None) -> bool:
        """Manually add a labeled prompt for training"""
        if self._is_duplicate(prompt):
            return False
        
        prompt_data = {
            "prompt": prompt,
            "task_type": task_type,
            "complexity_label": complexity.value,
            "complexity_score": self._complexity_to_score(complexity),
            "domain": domain,
            "features": self._extract_features(prompt, task_type),
            "manual_label": True,
            "collected_at": datetime.utcnow().isoformat()
        }
        
        self.collected_prompts.append(prompt_data)
        self._save_data()
        return True
    
    def _extract_features(self, prompt: str, task_type: str) -> Dict:
        """Extract features from prompt for ML training"""
        prompt_lower = prompt.lower()
        words = prompt.split()
        
        features = {
            # Basic features
            "length": len(prompt),
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "sentence_count": prompt.count('.') + prompt.count('?') + prompt.count('!'),
            
            # Complexity indicators
            "has_code_keywords": any(kw in prompt_lower for kw in [
                'function', 'class', 'algorithm', 'optimize', 'refactor', 'implement'
            ]),
            "has_system_keywords": any(kw in prompt_lower for kw in [
                'architecture', 'system', 'design', 'scalability', 'performance'
            ]),
            "has_ml_keywords": any(kw in prompt_lower for kw in [
                'machine learning', 'neural', 'model', 'train', 'predict'
            ]),
            "has_data_keywords": any(kw in prompt_lower for kw in [
                'database', 'query', 'data', 'sql', 'nosql', 'schema'
            ]),
            
            # Task type indicators
            "is_" + task_type: 1,
            
            # Linguistic features
            "question_marks": prompt.count('?'),
            "exclamation_marks": prompt.count('!'),
            "code_blocks": prompt.count('```'),
            "urls": prompt.lower().count('http'),
            
            # Domain indicators
            "business_terms": sum(1 for term in [
                'revenue', 'cost', 'budget', 'profit', 'roi', 'kpi'
            ] if term in prompt_lower),
            "technical_terms": sum(1 for term in [
                'api', 'backend', 'frontend', 'server', 'client', 'protocol'
            ] if term in prompt_lower),
        }
        
        return features
    
    def _is_duplicate(self, prompt: str) -> bool:
        """Check if prompt already exists in dataset"""
        prompt_normalized = prompt.strip().lower()
        for existing in self.collected_prompts:
            if existing['prompt'].strip().lower() == prompt_normalized:
                return True
        return False
    
    def _score_to_complexity(self, score: Optional[float]) -> str:
        """Convert numerical score to complexity label"""
        if score is None:
            return TaskComplexity.MEDIUM.value
        if score < 0.4:
            return TaskComplexity.SIMPLE.value
        elif score < 0.7:
            return TaskComplexity.MEDIUM.value
        else:
            return TaskComplexity.COMPLEX.value
    
    def _complexity_to_score(self, complexity: TaskComplexity) -> float:
        """Convert complexity enum to numerical score"""
        mapping = {
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MEDIUM: 0.6,
            TaskComplexity.COMPLEX: 0.9
        }
        return mapping.get(complexity, 0.5)
    
    def _save_data(self):
        """Save collected data to JSONL file"""
        try:
            with open(self.data_path, 'w') as f:
                for prompt_data in self.collected_prompts:
                    f.write(json.dumps(prompt_data) + '\n')
            logger.info(f"Saved {len(self.collected_prompts)} prompts to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the collected dataset"""
        if not self.collected_prompts:
            return {"total": 0}
        
        stats = {
            "total": len(self.collected_prompts),
            "by_task_type": {},
            "by_complexity": {},
            "manual_labels": sum(1 for p in self.collected_prompts if p.get('manual_label')),
            "avg_prompt_length": sum(p['features']['length'] for p in self.collected_prompts) / len(self.collected_prompts)
        }
        
        # Count by task type
        for prompt in self.collected_prompts:
            task_type = prompt.get('task_type', 'unknown')
            stats['by_task_type'][task_type] = stats['by_task_type'].get(task_type, 0) + 1
            
            complexity = prompt.get('complexity_label', 'unknown')
            stats['by_complexity'][complexity] = stats['by_complexity'].get(complexity, 0) + 1
        
        return stats
    
    def export_for_training(self) -> List[Dict]:
        """Export dataset in format ready for ML training"""
        training_data = []
        
        for prompt in self.collected_prompts:
            # Flatten features for training
            features = prompt['features'].copy()
            features['task_type'] = prompt['task_type']
            
            training_item = {
                'features': features,
                'label': prompt['complexity_label'],
                'score': prompt['complexity_score']
            }
            
            training_data.append(training_item)
        
        return training_data