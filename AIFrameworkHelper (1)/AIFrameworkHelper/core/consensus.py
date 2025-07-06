import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import hashlib

from .types import ModelResponse, ConsensusResult
from models import EmbeddingCache
from app import db

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConsensusStrategy(ABC):
    """Abstract base class for consensus algorithms"""
    
    @abstractmethod
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ConsensusResult:
        """Find consensus among multiple responses"""
        pass

class SimilarityConsensus(ConsensusStrategy):
    """Consensus based on semantic similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.8):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        if not SIMILARITY_AVAILABLE:
            logger.warning("Sentence transformers not available, using simple similarity")
            self.embedding_model = None
            return
            
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ConsensusResult:
        """Find consensus using semantic similarity"""
        
        if len(responses) == 1:
            return ConsensusResult(
                final_response=responses[0],
                confidence=1.0,
                individual_responses=responses,
                consensus_method="single_response"
            )
        
        if not SIMILARITY_AVAILABLE or self.embedding_model is None:
            # Fall back to simple string similarity
            return await self._simple_similarity_consensus(responses, task_context)
        
        # Get embeddings for all responses
        embeddings = await self._get_embeddings([r.content for r in responses])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find the response with highest average similarity to others
        avg_similarities = []
        for i in range(len(responses)):
            # Average similarity to all other responses
            similarities = similarity_matrix[i]
            avg_sim = np.mean([similarities[j] for j in range(len(similarities)) if i != j])
            avg_similarities.append(avg_sim)
        
        # Select the response with highest average similarity
        best_idx = np.argmax(avg_similarities)
        best_response = responses[best_idx]
        confidence = float(avg_similarities[best_idx])
        
        # Check if consensus meets threshold
        if confidence < self.similarity_threshold:
            # If no clear consensus, use weighted combination or best individual response
            logger.warning(f"Low consensus confidence: {confidence}")
        
        return ConsensusResult(
            final_response=best_response,
            confidence=confidence,
            individual_responses=responses,
            consensus_method="similarity",
            metadata={
                "similarity_matrix": similarity_matrix.tolist(),
                "avg_similarities": avg_similarities,
                "threshold": self.similarity_threshold
            }
        )
    
    async def _simple_similarity_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ConsensusResult:
        """Simple string-based similarity when advanced libraries aren't available"""
        
        # Use simple string-based metrics for consensus
        response_texts = [r.content for r in responses]
        similarities = []
        
        for i, text1 in enumerate(response_texts):
            total_similarity = 0
            for j, text2 in enumerate(response_texts):
                if i != j:
                    # Simple Jaccard similarity on words
                    words1 = set(text1.lower().split())
                    words2 = set(text2.lower().split())
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    if len(union) > 0:
                        similarity = len(intersection) / len(union)
                    else:
                        similarity = 0
                    total_similarity += similarity
            
            avg_similarity = total_similarity / (len(responses) - 1) if len(responses) > 1 else 0
            similarities.append(avg_similarity)
        
        # Select the response with highest average similarity
        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        best_response = responses[best_idx]
        confidence = similarities[best_idx]
        
        logger.info(f"Simple similarity consensus: confidence={confidence}")
        
        return ConsensusResult(
            final_response=best_response,
            confidence=confidence,
            individual_responses=responses,
            consensus_method="simple_similarity",
            metadata={
                "similarities": similarities,
                "threshold": self.similarity_threshold,
                "method": "jaccard_words"
            }
        )
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts, using cache when possible"""
        if self.embedding_model is None:
            # Return dummy embeddings if model not available
            return np.random.rand(len(texts), 384)
            
        embeddings = []
        
        for text in texts:
            # Check cache first
            content_hash = hashlib.md5(text.encode()).hexdigest()
            cached_embedding = db.session.query(EmbeddingCache).filter(
                EmbeddingCache.content_hash == content_hash,
                EmbeddingCache.model_name == self.model_name
            ).first()
            
            if cached_embedding:
                embedding = json.loads(cached_embedding.embedding)
                embeddings.append(embedding)
            else:
                # Generate new embedding
                embedding = self.embedding_model.encode(text)
                embeddings.append(embedding.tolist())
                
                # Cache the embedding
                try:
                    cache_entry = EmbeddingCache(
                        content_hash=content_hash,
                        content=text[:1000],  # Truncate for storage
                        embedding=json.dumps(embedding.tolist()),
                        model_name=self.model_name
                    )
                    db.session.add(cache_entry)
                except Exception as e:
                    logger.warning(f"Failed to create cache entry: {e}")
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
            db.session.rollback()
        
        return np.array(embeddings)

class VotingConsensus(ConsensusStrategy):
    """Consensus based on voting with confidence weighting"""
    
    def __init__(self, require_majority: bool = True, weight_by_confidence: bool = True,
                 similarity_threshold: float = 0.85):
        self.require_majority = require_majority
        self.weight_by_confidence = weight_by_confidence
        self.similarity_threshold = similarity_threshold
    
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ConsensusResult:
        """Find consensus using weighted voting mechanism"""
        
        if len(responses) == 1:
            return ConsensusResult(
                final_response=responses[0],
                confidence=1.0,
                individual_responses=responses,
                consensus_method="single_response"
            )
        
        # Group similar responses together
        response_groups = await self._group_similar_responses(responses)
        
        # Calculate weighted votes for each group
        group_votes = {}
        for group_id, group_responses in response_groups.items():
            total_weight = 0.0
            
            for response in group_responses:
                # Base weight of 1
                weight = 1.0
                
                # Weight by confidence if enabled
                if self.weight_by_confidence and response.confidence_score:
                    weight *= response.confidence_score
                
                # Weight by provider reliability (if available in context)
                if 'provider_reliability' in task_context:
                    provider_reliability = task_context['provider_reliability'].get(
                        response.provider.value, 1.0
                    )
                    weight *= provider_reliability
                
                total_weight += weight
            
            group_votes[group_id] = {
                'weight': total_weight,
                'responses': group_responses,
                'representative': group_responses[0]  # First response as representative
            }
        
        # Find winning group
        winning_group = max(group_votes.values(), key=lambda x: x['weight'])
        total_votes = sum(g['weight'] for g in group_votes.values())
        
        # Calculate confidence based on vote distribution
        vote_percentage = winning_group['weight'] / total_votes if total_votes > 0 else 0
        
        # Check if majority requirement is met
        if self.require_majority and vote_percentage < 0.5:
            logger.warning(f"No majority consensus achieved (highest vote: {vote_percentage:.1%})")
        
        # Select best response from winning group
        best_response = self._select_best_from_group(winning_group['responses'])
        
        return ConsensusResult(
            final_response=best_response,
            confidence=vote_percentage,
            individual_responses=responses,
            consensus_method="weighted_voting",
            metadata={
                'vote_distribution': {
                    f"group_{k}": {
                        'votes': v['weight'],
                        'percentage': v['weight'] / total_votes if total_votes > 0 else 0,
                        'response_count': len(v['responses'])
                    }
                    for k, v in group_votes.items()
                },
                'total_groups': len(group_votes),
                'winning_vote_percentage': vote_percentage
            }
        )
    
    async def _group_similar_responses(self, responses: List[ModelResponse]) -> Dict[int, List[ModelResponse]]:
        """Group similar responses together"""
        groups = {}
        group_id = 0
        
        for response in responses:
            assigned = False
            
            # Check similarity with existing groups
            for gid, group_responses in groups.items():
                # Compare with first response in group
                similarity = self._calculate_similarity(response.content, group_responses[0].content)
                
                if similarity >= self.similarity_threshold:
                    groups[gid].append(response)
                    assigned = True
                    break
            
            # Create new group if not assigned
            if not assigned:
                groups[group_id] = [response]
                group_id += 1
        
        return groups
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _select_best_from_group(self, responses: List[ModelResponse]) -> ModelResponse:
        """Select best response from a group based on confidence and quality"""
        best_response = responses[0]
        best_score = 0.0
        
        for response in responses:
            score = 1.0
            
            # Consider confidence score
            if response.confidence_score:
                score *= response.confidence_score
            
            # Consider response length (prefer moderate length)
            length_score = min(1.0, len(response.content) / 500)  # Normalize to ~500 chars
            score *= (0.5 + 0.5 * length_score)
            
            if score > best_score:
                best_score = score
                best_response = response
        
        return best_response

class ConfidenceConsensus(ConsensusStrategy):
    """Advanced consensus based on confidence scores with quality weighting"""
    
    def __init__(self, min_confidence_threshold: float = 0.6, 
                 quality_weight: float = 0.3,
                 consistency_weight: float = 0.2):
        self.min_confidence_threshold = min_confidence_threshold
        self.quality_weight = quality_weight
        self.consistency_weight = consistency_weight
    
    async def find_consensus(
        self, 
        responses: List[ModelResponse], 
        task_context: Dict[str, Any]
    ) -> ConsensusResult:
        """Find consensus using advanced confidence scoring"""
        
        if len(responses) == 1:
            return ConsensusResult(
                final_response=responses[0],
                confidence=1.0,
                individual_responses=responses,
                consensus_method="single_response"
            )
        
        # Calculate composite scores for each response
        scored_responses = []
        
        for response in responses:
            # Base confidence score
            base_confidence = response.confidence_score or 0.5
            
            # Quality factors
            quality_score = self._calculate_quality_score(response, task_context)
            
            # Consistency with other responses
            consistency_score = await self._calculate_consistency_score(response, responses)
            
            # Composite score
            composite_score = (
                base_confidence * (1 - self.quality_weight - self.consistency_weight) +
                quality_score * self.quality_weight +
                consistency_score * self.consistency_weight
            )
            
            # Provider-specific adjustments
            if 'provider_performance' in task_context:
                provider_perf = task_context['provider_performance'].get(
                    response.provider.value, 1.0
                )
                composite_score *= provider_perf
            
            scored_responses.append({
                'response': response,
                'composite_score': composite_score,
                'base_confidence': base_confidence,
                'quality_score': quality_score,
                'consistency_score': consistency_score
            })
        
        # Sort by composite score
        scored_responses.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Select best response
        best_scored = scored_responses[0]
        best_response = best_scored['response']
        final_confidence = best_scored['composite_score']
        
        # Check if confidence meets threshold
        if final_confidence < self.min_confidence_threshold:
            logger.warning(f"Low consensus confidence: {final_confidence:.2f}")
            
            # Try to combine insights from multiple responses
            combined_response = await self._combine_top_responses(scored_responses[:3])
            if combined_response:
                best_response = combined_response
                final_confidence = min(0.85, final_confidence * 1.2)  # Boost confidence for combined
        
        return ConsensusResult(
            final_response=best_response,
            confidence=final_confidence,
            individual_responses=responses,
            consensus_method="advanced_confidence",
            metadata={
                'scoring_details': [
                    {
                        'provider': sr['response'].provider.value,
                        'composite_score': sr['composite_score'],
                        'base_confidence': sr['base_confidence'],
                        'quality_score': sr['quality_score'],
                        'consistency_score': sr['consistency_score']
                    }
                    for sr in scored_responses
                ],
                'confidence_threshold': self.min_confidence_threshold,
                'weights': {
                    'quality': self.quality_weight,
                    'consistency': self.consistency_weight
                }
            }
        )
    
    def _calculate_quality_score(self, response: ModelResponse, task_context: Dict[str, Any]) -> float:
        """Calculate quality score based on response characteristics"""
        quality_score = 0.5  # Base score
        
        # Response length (prefer moderate length)
        content_length = len(response.content)
        if 100 <= content_length <= 1000:
            quality_score += 0.2
        elif content_length < 50:
            quality_score -= 0.2
        elif content_length > 5000:
            quality_score -= 0.1
        
        # Check for structured response (lists, code blocks, etc.)
        if any(marker in response.content for marker in ['1.', '```', '- ', '* ']):
            quality_score += 0.1
        
        # Task-specific quality checks
        task_type = task_context.get('task_type', '')
        if task_type == 'code_generation' and '```' in response.content:
            quality_score += 0.2
        elif task_type == 'analysis' and len(response.content.split('\n')) > 3:
            quality_score += 0.15
        
        # Cap between 0 and 1
        return max(0.0, min(1.0, quality_score))
    
    async def _calculate_consistency_score(self, response: ModelResponse, all_responses: List[ModelResponse]) -> float:
        """Calculate how consistent this response is with others"""
        if len(all_responses) <= 1:
            return 1.0
        
        similarities = []
        for other in all_responses:
            if other != response:
                similarity = self._simple_similarity(response.content, other.content)
                similarities.append(similarity)
        
        # Average similarity to other responses
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _combine_top_responses(self, scored_responses: List[Dict]) -> Optional[ModelResponse]:
        """Attempt to combine insights from top responses"""
        if len(scored_responses) < 2:
            return None
        
        # Extract key points from each response
        key_points = []
        for sr in scored_responses:
            response = sr['response']
            # Simple extraction: split by sentences and take unique ones
            sentences = [s.strip() for s in response.content.split('.') if s.strip()]
            key_points.extend(sentences[:3])  # Take up to 3 key sentences
        
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in key_points:
            if point.lower() not in seen:
                seen.add(point.lower())
                unique_points.append(point)
        
        if not unique_points:
            return None
        
        # Create combined response
        combined_content = '. '.join(unique_points[:5]) + '.'
        
        # Use the best response as template
        best_response = scored_responses[0]['response']
        return ModelResponse(
            content=combined_content,
            provider=best_response.provider,
            model_name=best_response.model_name + "_combined",
            tokens_used=best_response.tokens_used,
            latency_ms=best_response.latency_ms,
            confidence_score=0.75  # Moderate confidence for combined response
        )

class ConsensusEngine:
    """Main consensus engine that manages different strategies"""
    
    def __init__(self, config):
        self.config = config
        self.strategies = {
            "similarity": SimilarityConsensus(
                similarity_threshold=config.consensus.similarity_threshold
            ),
            "voting": VotingConsensus(),
            "confidence": ConfidenceConsensus()
        }
        self.default_strategy = config.consensus.default_strategy
    
    async def find_consensus(
        self,
        responses: List[ModelResponse],
        task_context: Dict[str, Any],
        strategy_override: Optional[str] = None
    ) -> ConsensusResult:
        """Find consensus using specified or default strategy"""
        
        strategy_name = strategy_override or self.default_strategy
        
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown consensus strategy: {strategy_name}, using similarity")
            strategy_name = "similarity"
        
        strategy = self.strategies[strategy_name]
        
        try:
            result = await strategy.find_consensus(responses, task_context)
            logger.info(f"Consensus found using {strategy_name} with confidence {result.confidence}")
            return result
        except Exception as e:
            logger.error(f"Consensus strategy {strategy_name} failed: {e}")
            # Fall back to similarity strategy
            if strategy_name != "similarity":
                return await self.strategies["similarity"].find_consensus(responses, task_context)
            raise
