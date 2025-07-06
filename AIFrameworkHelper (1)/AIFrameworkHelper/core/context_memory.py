"""
Context Memory Module

This module provides conversation context awareness to improve response quality
by leveraging historical conversations and user preferences.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

from .types import ModelRequest, ModelProvider
from models import Conversation
from app import db

logger = logging.getLogger(__name__)


class ContextMemory:
    """Manages conversation context and provides context-aware enhancements"""
    
    def __init__(self, max_context_length: int = 10, context_window_hours: int = 24):
        self.max_context_length = max_context_length
        self.context_window_hours = context_window_hours
        self.user_preferences = defaultdict(dict)
        self.context_cache = {}
        
    async def get_relevant_context(
        self,
        user_id: str,
        current_prompt: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Get relevant context for the current request"""
        
        # Get recent conversation history
        recent_conversations = await self._get_recent_conversations(user_id, task_type)
        
        # Extract context patterns
        context_patterns = self._extract_patterns(recent_conversations)
        
        # Get user preferences
        user_prefs = self.user_preferences.get(user_id, {})
        
        # Find similar past conversations
        similar_conversations = await self._find_similar_conversations(
            user_id, current_prompt, recent_conversations
        )
        
        # Build context summary
        context = {
            'conversation_history': self._format_conversation_history(similar_conversations),
            'user_preferences': user_prefs,
            'context_patterns': context_patterns,
            'relevant_past_responses': self._extract_relevant_responses(similar_conversations),
            'conversation_count': len(recent_conversations),
            'last_interaction': recent_conversations[0]['created_at'] if recent_conversations else None
        }
        
        return context
    
    async def enhance_request_with_context(
        self,
        request: ModelRequest,
        user_id: str,
        task_type: str = "general"
    ) -> ModelRequest:
        """Enhance a request with contextual information"""
        
        # Get relevant context
        context = await self.get_relevant_context(
            user_id, request.prompt, task_type
        )
        
        # If there's relevant history, enhance the prompt
        if context['conversation_history']:
            enhanced_prompt = self._build_enhanced_prompt(
                request.prompt, context
            )
            
            # Create enhanced request
            enhanced_request = ModelRequest(
                prompt=enhanced_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model_specific_params={
                    **request.model_specific_params,
                    'context_enhanced': True,
                    'context_conversations': len(context['conversation_history'])
                }
            )
            
            logger.info(f"Enhanced request with {len(context['conversation_history'])} context conversations")
            return enhanced_request
        
        return request
    
    def _build_enhanced_prompt(self, original_prompt: str, context: Dict[str, Any]) -> str:
        """Build an enhanced prompt with context"""
        
        enhanced_parts = []
        
        # Add conversation context if available
        if context['conversation_history']:
            enhanced_parts.append("Based on our previous conversations:")
            enhanced_parts.append(context['conversation_history'])
            enhanced_parts.append("")
        
        # Add user preferences if known
        if context['user_preferences']:
            pref_summary = self._summarize_preferences(context['user_preferences'])
            if pref_summary:
                enhanced_parts.append(f"User preferences: {pref_summary}")
                enhanced_parts.append("")
        
        # Add the current prompt
        enhanced_parts.append("Current request:")
        enhanced_parts.append(original_prompt)
        
        # Add relevant past responses if similar questions were asked
        if context['relevant_past_responses']:
            enhanced_parts.append("")
            enhanced_parts.append("For context, here are relevant past discussions:")
            enhanced_parts.append(context['relevant_past_responses'])
        
        return "\n".join(enhanced_parts)
    
    async def _get_recent_conversations(
        self,
        user_id: str,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent conversations for a user"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=self.context_window_hours)
            
            query = db.session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= cutoff_date
            )
            
            if task_type:
                query = query.filter(Conversation.task_type == task_type)
            
            conversations = query.order_by(
                Conversation.created_at.desc()
            ).limit(self.max_context_length).all()
            
            return [
                {
                    'id': conv.id,
                    'prompt': conv.prompt,
                    'response': conv.response,
                    'task_type': conv.task_type,
                    'created_at': conv.created_at.isoformat(),
                    'confidence': conv.consensus_confidence
                }
                for conv in conversations
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    def _extract_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from conversation history"""
        
        patterns = {
            'common_topics': defaultdict(int),
            'task_distribution': defaultdict(int),
            'average_confidence': 0.0,
            'interaction_frequency': 'low'
        }
        
        if not conversations:
            return patterns
        
        # Analyze topics and tasks
        for conv in conversations:
            patterns['task_distribution'][conv['task_type']] += 1
            
            # Extract common words/topics (simple approach)
            words = conv['prompt'].lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    patterns['common_topics'][word] += 1
        
        # Calculate average confidence
        confidences = [c['confidence'] for c in conversations if c['confidence']]
        if confidences:
            patterns['average_confidence'] = sum(confidences) / len(confidences)
        
        # Determine interaction frequency
        if len(conversations) >= 10:
            patterns['interaction_frequency'] = 'high'
        elif len(conversations) >= 5:
            patterns['interaction_frequency'] = 'medium'
        
        # Get top topics
        patterns['common_topics'] = dict(
            sorted(patterns['common_topics'].items(), 
                   key=lambda x: x[1], reverse=True)[:5]
        )
        
        return dict(patterns)
    
    async def _find_similar_conversations(
        self,
        user_id: str,
        current_prompt: str,
        recent_conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find conversations similar to the current prompt"""
        
        if not recent_conversations:
            return []
        
        # Calculate similarity scores
        similarities = []
        current_words = set(current_prompt.lower().split())
        
        for conv in recent_conversations:
            past_words = set(conv['prompt'].lower().split())
            
            # Jaccard similarity
            intersection = current_words.intersection(past_words)
            union = current_words.union(past_words)
            similarity = len(intersection) / len(union) if union else 0
            
            if similarity > 0.3:  # Threshold for relevance
                similarities.append((conv, similarity))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [conv for conv, _ in similarities[:3]]
    
    def _format_conversation_history(self, conversations: List[Dict[str, Any]]) -> str:
        """Format conversation history for context"""
        
        if not conversations:
            return ""
        
        formatted = []
        for i, conv in enumerate(conversations[:3], 1):
            formatted.append(f"{i}. Previous: {conv['prompt'][:100]}...")
            formatted.append(f"   Response: {conv['response'][:150]}...")
        
        return "\n".join(formatted)
    
    def _extract_relevant_responses(self, conversations: List[Dict[str, Any]]) -> str:
        """Extract relevant responses from similar conversations"""
        
        if not conversations:
            return ""
        
        relevant_parts = []
        for conv in conversations[:2]:
            # Extract key sentences from responses
            sentences = conv['response'].split('.')[:3]
            relevant_parts.extend([s.strip() for s in sentences if s.strip()])
        
        return ". ".join(relevant_parts[:5]) if relevant_parts else ""
    
    def _summarize_preferences(self, preferences: Dict[str, Any]) -> str:
        """Summarize user preferences"""
        
        if not preferences:
            return ""
        
        pref_parts = []
        for key, value in preferences.items():
            if isinstance(value, bool):
                if value:
                    pref_parts.append(key.replace('_', ' '))
            else:
                pref_parts.append(f"{key.replace('_', ' ')}: {value}")
        
        return ", ".join(pref_parts[:3])  # Limit to top 3 preferences
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ):
        """Update user preferences based on interactions"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
        
        logger.info(f"Updated preferences for user {user_id}: {preferences}")
    
    async def learn_from_feedback(
        self,
        user_id: str,
        conversation_id: int,
        feedback: Dict[str, Any]
    ):
        """Learn from user feedback to improve future responses"""
        
        try:
            # Get the conversation
            conversation = db.session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return
            
            # Extract learning points
            if feedback.get('quality_score'):
                # Update preferences based on quality
                if feedback['quality_score'] >= 0.8:
                    # Good response - learn what worked
                    self._extract_positive_patterns(user_id, conversation)
                elif feedback['quality_score'] < 0.5:
                    # Poor response - learn what to avoid
                    self._extract_negative_patterns(user_id, conversation)
            
            # Store feedback in conversation metadata
            if conversation.extra_metadata is None:
                conversation.extra_metadata = {}
            
            conversation.extra_metadata['user_feedback'] = feedback
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
            db.session.rollback()
    
    def _extract_positive_patterns(self, user_id: str, conversation: Conversation):
        """Extract patterns from successful interactions"""
        
        # Simple pattern extraction - could be enhanced with NLP
        if 'detailed' in conversation.response.lower():
            self.user_preferences[user_id]['prefers_detailed_responses'] = True
        
        if len(conversation.response.split('\n')) > 5:
            self.user_preferences[user_id]['prefers_structured_responses'] = True
        
        # Track successful providers
        try:
            providers = json.loads(conversation.providers_used)
            for provider in providers:
                pref_key = f'successful_{provider}_count'
                current = self.user_preferences[user_id].get(pref_key, 0)
                self.user_preferences[user_id][pref_key] = current + 1
        except:
            pass
    
    def _extract_negative_patterns(self, user_id: str, conversation: Conversation):
        """Extract patterns from unsuccessful interactions"""
        
        # Learn from failures
        if len(conversation.response) < 50:
            self.user_preferences[user_id]['avoid_brief_responses'] = True
        
        # Track unsuccessful providers
        try:
            providers = json.loads(conversation.providers_used)
            for provider in providers:
                pref_key = f'failed_{provider}_count'
                current = self.user_preferences[user_id].get(pref_key, 0)
                self.user_preferences[user_id][pref_key] = current + 1
        except:
            pass


class ContextAwareRouter:
    """Router that considers conversation context"""
    
    def __init__(self, context_memory: ContextMemory):
        self.context_memory = context_memory
    
    async def adjust_routing_for_context(
        self,
        user_id: str,
        base_routing: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Adjust routing decisions based on user context"""
        
        # Get user preferences
        user_prefs = self.context_memory.user_preferences.get(user_id, {})
        
        # Adjust provider selection based on past success
        adjusted_providers = []
        for provider in base_routing['selected_providers']:
            provider_key = provider.value
            
            # Check success/failure rates
            success_count = user_prefs.get(f'successful_{provider_key}_count', 0)
            failure_count = user_prefs.get(f'failed_{provider_key}_count', 0)
            
            # Calculate provider score
            total_interactions = success_count + failure_count
            if total_interactions > 0:
                success_rate = success_count / total_interactions
                
                # Boost or penalize provider based on history
                if success_rate > 0.8:
                    # Prioritize this provider
                    adjusted_providers.insert(0, provider)
                elif success_rate < 0.3 and len(base_routing['selected_providers']) > 1:
                    # Skip this provider if we have alternatives
                    continue
                else:
                    adjusted_providers.append(provider)
            else:
                adjusted_providers.append(provider)
        
        # Ensure we have at least one provider
        if not adjusted_providers:
            adjusted_providers = base_routing['selected_providers']
        
        return {
            **base_routing,
            'selected_providers': adjusted_providers[:3],  # Limit to top 3
            'context_adjusted': True,
            'user_preference_applied': len(user_prefs) > 0
        }