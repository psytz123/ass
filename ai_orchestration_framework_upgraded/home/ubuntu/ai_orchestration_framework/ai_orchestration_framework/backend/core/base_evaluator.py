
from abc import ABC, abstractmethod
from .model_connectors import TaskComplexity

class ComplexityEvaluator(ABC):
    """Abstract base class for task complexity evaluation"""

    @abstractmethod
    async def evaluate(self, prompt: str, task_type: str = "general") -> TaskComplexity:
        """Evaluate the complexity of a given prompt"""
        pass


