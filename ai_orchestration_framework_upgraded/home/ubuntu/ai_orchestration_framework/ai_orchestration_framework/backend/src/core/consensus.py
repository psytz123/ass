"""
Consensus Engine for AI Orchestration Framework

This module provides various consensus mechanisms to find the best
response among multiple AI model outputs.
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .model_connectors import ModelResponse, ModelProvider


@dataclass
class ConsensusResult:
    selected_response: ModelResponse
    confidence_score: float
    consensus_method: str
    all_responses: List[ModelResponse]
    similarity_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConsensusStrategy(ABC):
    """Abstract base class for consensus strategies"""

    @abstractmethod
    async def find_consensus(self, responses: List[ModelResponse], 
                           task_context: Dict[str, Any] = None) -> ConsensusResult:
        """Find consensus among multiple model responses"""
        pass

    @abstractmethod
    def calculate_confidence(self, responses: List[ModelResponse]) -> float:
        """Calculate overall confidence in the consensus"""
        pass


class SimilarityConsensus(ConsensusStrategy):
    """Consensus based on semantic similarity between responses"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.8):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.sentence_model = None
        self._model_loaded = False

    async def _load_model(self):
        """Load the sentence transformer model"""
        if not self._model_loaded:
            # Load in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.sentence_model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            self._model_loaded = True

    async def find_consensus(self, responses: List[ModelResponse], 
                           task_context: Dict[str, Any] = None) -> ConsensusResult:
        if len(responses) == 1:
            return ConsensusResult(
                selected_response=responses[0],
                confidence_score=responses[0].confidence_score or 0.8,
                consensus_method="single_response",
                all_responses=responses
            )

        await self._load_model()

        # Extract response texts
        texts = [response.content for response in responses]

        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.sentence_model.encode(texts)
        )

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find the response with highest average similarity to others
        avg_similarities = np.mean(similarity_matrix, axis=1)
        best_idx = np.argmax(avg_similarities)
        best_response = responses[best_idx]

        # Calculate consensus confidence
        max_similarity = avg_similarities[best_idx]
        confidence = min(max_similarity, 1.0)

        # Check if consensus meets threshold
        if max_similarity < self.similarity_threshold:
            # If no strong consensus, select based on individual confidence
            confidence_scores = [r.confidence_score or 0.8 for r in responses]
            best_idx = np.argmax(confidence_scores)
            best_response = responses[best_idx]
            confidence = confidence_scores[best_idx] * 0.8  # Reduce confidence due to low consensus

        return ConsensusResult(
            selected_response=best_response,
            confidence_score=confidence,
            consensus_method="similarity",
            all_responses=responses,
            similarity_matrix=similarity_matrix,
            metadata={
                "avg_similarity": float(max_similarity),
                "similarity_threshold": self.similarity_threshold,
                "threshold_met": max_similarity >= self.similarity_threshold
            }
        )

    def calculate_confidence(self, responses: List[ModelResponse]) -> float:
        """Calculate confidence based on response similarities"""
        if len(responses) <= 1:
            return responses[0].confidence_score or 0.8 if responses else 0.0

        # This is a simplified version - in practice, you'd use the full similarity calculation
        individual_confidences = [r.confidence_score or 0.8 for r in responses]
        return np.mean(individual_confidences)


class VotingConsensus(ConsensusStrategy):
    """Consensus based on voting with confidence weighting"""

    def __init__(self, require_majority: bool = True, weight_by_confidence: bool = True):
        self.require_majority = require_majority
        self.weight_by_confidence = weight_by_confidence

    async def find_consensus(self, responses: List[ModelResponse], 
                           task_context: Dict[str, Any] = None) -> ConsensusResult:
        if len(responses) == 1:
            return ConsensusResult(
                selected_response=responses[0],
                confidence_score=responses[0].confidence_score or 0.8,
                consensus_method="single_response",
                all_responses=responses
            )

        # Group similar responses (simplified - could use more sophisticated clustering)
        response_groups = self._group_similar_responses(responses)

        # Calculate votes for each group
        group_scores = {}
        for group_id, group_responses in response_groups.items():
            if self.weight_by_confidence:
                score = sum(r.confidence_score or 0.8 for r in group_responses)
            else:
                score = len(group_responses)
            group_scores[group_id] = score

        # Find winning group
        winning_group_id = max(group_scores.keys(), key=lambda k: group_scores[k])
        winning_responses = response_groups[winning_group_id]

        # Select best response from winning group (highest confidence)
        best_response = max(winning_responses, key=lambda r: r.confidence_score or 0.8)

        # Calculate consensus confidence
        total_votes = sum(group_scores.values())
        winning_votes = group_scores[winning_group_id]
        vote_ratio = winning_votes / total_votes if total_votes > 0 else 0

        # Check majority requirement
        majority_met = vote_ratio > 0.5 if self.require_majority else True
        confidence = vote_ratio if majority_met else vote_ratio * 0.7

        return ConsensusResult(
            selected_response=best_response,
            confidence_score=confidence,
            consensus_method="voting",
            all_responses=responses,
            metadata={
                "vote_ratio": vote_ratio,
                "majority_required": self.require_majority,
                "majority_met": majority_met,
                "group_scores": group_scores
            }
        )

    def _group_similar_responses(self, responses: List[ModelResponse]) -> Dict[int, List[ModelResponse]]:
        """Group similar responses together (simplified implementation)"""
        # This is a simplified grouping based on response length and first few words
        # In practice, you might use more sophisticated similarity measures
        groups = {}
        group_id = 0

        for response in responses:
            content = response.content.strip().lower()
            words = content.split()[:5]  # First 5 words
            length_bucket = len(content) // 100  # Group by length buckets

            # Create a simple signature
            signature = (tuple(words), length_bucket)

            # Find existing group or create new one
            found_group = None
            for gid, group_responses in groups.items():
                if len(group_responses) > 0:
                    existing_content = group_responses[0].content.strip().lower()
                    existing_words = existing_content.split()[:5]
                    existing_length_bucket = len(existing_content) // 100

                    if (tuple(existing_words), existing_length_bucket) == signature:
                        found_group = gid
                        break

            if found_group is not None:
                groups[found_group].append(response)
            else:
                groups[group_id] = [response]
                group_id += 1

        return groups

    def calculate_confidence(self, responses: List[ModelResponse]) -> float:
        """Calculate confidence based on voting consensus"""
        if len(responses) <= 1:
            return responses[0].confidence_score or 0.8 if responses else 0.0

        # Simplified confidence calculation
        individual_confidences = [r.confidence_score or 0.8 for r in responses]
        return np.mean(individual_confidences)


class ConfidenceConsensus(ConsensusStrategy):
    """Consensus based on individual response confidence scores"""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    async def find_consensus(self, responses: List[ModelResponse], 
                           task_context: Dict[str, Any] = None) -> ConsensusResult:
        if len(responses) == 1:
            return ConsensusResult(
                selected_response=responses[0],
                confidence_score=responses[0].confidence_score or 0.8,
                consensus_method="single_response",
                all_responses=responses
            )

        # Sort responses by confidence score
        sorted_responses = sorted(responses, 
                                key=lambda r: r.confidence_score or 0.8, 
                                reverse=True)

        best_response = sorted_responses[0]
        best_confidence = best_response.confidence_score or 0.8

        # Check if best response meets threshold
        threshold_met = best_confidence >= self.confidence_threshold

        return ConsensusResult(
            selected_response=best_response,
            confidence_score=best_confidence if threshold_met else best_confidence * 0.8,
            consensus_method="confidence",
            all_responses=responses,
            metadata={
                "confidence_threshold": self.confidence_threshold,
                "threshold_met": threshold_met,
                "confidence_scores": [r.confidence_score or 0.8 for r in responses]
            }
        )

    def calculate_confidence(self, responses: List[ModelResponse]) -> float:
        """Calculate confidence as the maximum individual confidence"""
        if not responses:
            return 0.0
        return max(r.confidence_score or 0.8 for r in responses)


class ConsensusManager:
    """Manager for different consensus strategies"""

    def __init__(self):
        self.strategies: Dict[str, ConsensusStrategy] = {}
        self.default_strategy = "similarity"
        
        # Register default strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default consensus strategies"""
        self.strategies["similarity"] = SimilarityConsensus()
        self.strategies["voting"] = VotingConsensus()
        self.strategies["confidence"] = ConfidenceConsensus()

    def register_strategy(self, name: str, strategy: ConsensusStrategy):
        """Register a custom consensus strategy"""
        self.strategies[name] = strategy

    def get_strategy(self, name: str) -> Optional[ConsensusStrategy]:
        """Get a consensus strategy by name"""
        return self.strategies.get(name)

    async def find_consensus(self, responses: List[ModelResponse], 
                           strategy_name: Optional[str] = None,
                           task_context: Dict[str, Any] = None) -> ConsensusResult:
        """Find consensus using specified or default strategy"""
        strategy_name = strategy_name or self.default_strategy
        strategy = self.get_strategy(strategy_name)
        
        if not strategy:
            raise ValueError(f"Unknown consensus strategy: {strategy_name}")
        
        return await strategy.find_consensus(responses, task_context)

    def set_default_strategy(self, strategy_name: str):
        """Set the default consensus strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown consensus strategy: {strategy_name}")
        self.default_strategy = strategy_name

    def get_available_strategies(self) -> List[str]:
        """Get list of available consensus strategies"""
        return list(self.strategies.keys())

