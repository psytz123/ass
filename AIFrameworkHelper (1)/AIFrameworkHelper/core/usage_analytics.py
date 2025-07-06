"""
Usage Analytics module for tracking popular task types and patterns
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import logging
from sqlalchemy import func, desc, and_
import re

from core.types import ModelProvider
from models import db, Conversation, ModelPerformance

logger = logging.getLogger(__name__)

class UsageAnalytics:
    """Analyzes usage patterns and provides insights"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=10)
        self.last_cache_update = {}
    
    async def get_usage_insights(
        self,
        time_window_hours: int = 168,  # 1 week default
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive usage analytics and insights"""
        
        cache_key = f"{time_window_hours}_{user_id or 'all'}"
        
        # Check cache
        if (cache_key in self.cache and 
            cache_key in self.last_cache_update and
            datetime.utcnow() - self.last_cache_update[cache_key] < self.cache_duration):
            return self.cache[cache_key]
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Build base query
            query = db.session.query(Conversation).filter(
                Conversation.created_at >= cutoff_date
            )
            
            if user_id:
                query = query.filter(Conversation.user_id == user_id)
            
            conversations = query.all()
            
            # Calculate comprehensive analytics
            insights = {
                "summary": self._calculate_summary_stats(conversations),
                "task_distribution": self._analyze_task_distribution(conversations),
                "temporal_patterns": self._analyze_temporal_patterns(conversations),
                "complexity_trends": self._analyze_complexity_trends(conversations),
                "user_behavior": self._analyze_user_behavior(conversations),
                "content_analysis": self._analyze_content_patterns(conversations),
                "performance_insights": await self._get_performance_insights(cutoff_date, user_id),
                "recommendations": self._generate_recommendations(conversations)
            }
            
            # Cache results
            self.cache[cache_key] = insights
            self.last_cache_update[cache_key] = datetime.utcnow()
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to calculate usage insights: {e}")
            return {"error": str(e)}
    
    def _calculate_summary_stats(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        if not conversations:
            return {
                "total_conversations": 0,
                "unique_users": 0,
                "avg_daily_conversations": 0,
                "growth_rate": 0
            }
        
        # Get unique users
        unique_users = len(set(conv.user_id for conv in conversations))
        
        # Calculate daily average
        date_range = (conversations[-1].created_at - conversations[0].created_at).days + 1
        avg_daily = len(conversations) / max(date_range, 1)
        
        # Calculate growth rate (compare first half to second half)
        mid_point = len(conversations) // 2
        if mid_point > 0:
            first_half = mid_point
            second_half = len(conversations) - mid_point
            growth_rate = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        else:
            growth_rate = 0
        
        return {
            "total_conversations": len(conversations),
            "unique_users": unique_users,
            "avg_daily_conversations": round(avg_daily, 2),
            "growth_rate": round(growth_rate, 2),
            "date_range_days": date_range
        }
    
    def _analyze_task_distribution(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze distribution of task types"""
        
        task_counts = Counter(conv.task_type for conv in conversations)
        total = sum(task_counts.values())
        
        # Calculate percentages and trends
        task_stats = {}
        for task_type, count in task_counts.most_common():
            task_stats[task_type] = {
                "count": count,
                "percentage": round((count / total * 100), 2) if total > 0 else 0
            }
            
            # Calculate trend (compare recent vs older)
            recent_count = sum(
                1 for conv in conversations 
                if conv.task_type == task_type and 
                (datetime.utcnow() - conv.created_at).days <= 1
            )
            older_count = sum(
                1 for conv in conversations 
                if conv.task_type == task_type and 
                (datetime.utcnow() - conv.created_at).days > 1
            )
            
            if older_count > 0:
                trend = ((recent_count - (older_count / 6)) / (older_count / 6)) * 100
                task_stats[task_type]["trend"] = round(trend, 2)
            else:
                task_stats[task_type]["trend"] = 100 if recent_count > 0 else 0
        
        return {
            "by_task_type": task_stats,
            "most_popular": task_counts.most_common(1)[0] if task_counts else None,
            "least_popular": task_counts.most_common()[-1] if task_counts else None,
            "diversity_score": len(task_counts) / 10  # Normalized diversity
        }
    
    def _analyze_temporal_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze temporal usage patterns"""
        
        if not conversations:
            return {
                "hourly_distribution": {},
                "daily_distribution": {},
                "peak_hours": [],
                "quiet_hours": []
            }
        
        # Hourly distribution
        hourly_counts = Counter(conv.created_at.hour for conv in conversations)
        hourly_dist = {
            hour: count for hour, count in 
            sorted(hourly_counts.items(), key=lambda x: x[0])
        }
        
        # Daily distribution
        daily_counts = Counter(conv.created_at.weekday() for conv in conversations)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_dist = {
            days[day]: count for day, count in 
            sorted(daily_counts.items(), key=lambda x: x[0])
        }
        
        # Find peak and quiet hours
        if hourly_counts:
            avg_hourly = sum(hourly_counts.values()) / 24
            peak_hours = [hour for hour, count in hourly_counts.items() if count > avg_hourly * 1.5]
            quiet_hours = [hour for hour in range(24) if hourly_counts.get(hour, 0) < avg_hourly * 0.5]
        else:
            peak_hours = []
            quiet_hours = []
        
        return {
            "hourly_distribution": hourly_dist,
            "daily_distribution": daily_dist,
            "peak_hours": sorted(peak_hours),
            "quiet_hours": sorted(quiet_hours),
            "busiest_day": max(daily_dist.items(), key=lambda x: x[1])[0] if daily_dist else None,
            "quietest_day": min(daily_dist.items(), key=lambda x: x[1])[0] if daily_dist else None
        }
    
    def _analyze_complexity_trends(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze task complexity trends"""
        
        complexity_scores = [
            conv.complexity_score for conv in conversations 
            if conv.complexity_score is not None
        ]
        
        if not complexity_scores:
            return {
                "avg_complexity": 0,
                "complexity_trend": "stable",
                "distribution": {}
            }
        
        # Calculate average and trend
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        # Compare recent vs older complexity
        recent_scores = [
            conv.complexity_score for conv in conversations 
            if conv.complexity_score is not None and
            (datetime.utcnow() - conv.created_at).days <= 1
        ]
        older_scores = [
            conv.complexity_score for conv in conversations 
            if conv.complexity_score is not None and
            (datetime.utcnow() - conv.created_at).days > 1
        ]
        
        if recent_scores and older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Distribution
        distribution = {
            "simple": sum(1 for s in complexity_scores if s < 0.3),
            "medium": sum(1 for s in complexity_scores if 0.3 <= s < 0.7),
            "complex": sum(1 for s in complexity_scores if s >= 0.7)
        }
        
        return {
            "avg_complexity": round(avg_complexity, 3),
            "complexity_trend": trend,
            "distribution": distribution,
            "recent_avg": round(sum(recent_scores) / len(recent_scores), 3) if recent_scores else 0,
            "historical_avg": round(sum(older_scores) / len(older_scores), 3) if older_scores else 0
        }
    
    def _analyze_user_behavior(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        
        if not conversations:
            return {
                "avg_session_length": 0,
                "return_rate": 0,
                "power_users": [],
                "engagement_levels": {}
            }
        
        # Group conversations by user
        user_conversations = defaultdict(list)
        for conv in conversations:
            user_conversations[conv.user_id].append(conv)
        
        # Calculate metrics
        session_lengths = []
        returning_users = 0
        
        for user_id, user_convs in user_conversations.items():
            # Sort by time
            user_convs.sort(key=lambda x: x.created_at)
            
            # Session analysis (gap > 30 minutes = new session)
            sessions = []
            current_session = [user_convs[0]]
            
            for i in range(1, len(user_convs)):
                time_gap = (user_convs[i].created_at - user_convs[i-1].created_at).total_seconds() / 60
                
                if time_gap > 30:  # New session
                    sessions.append(current_session)
                    current_session = [user_convs[i]]
                else:
                    current_session.append(user_convs[i])
            
            sessions.append(current_session)
            
            # Calculate session lengths
            for session in sessions:
                if len(session) > 1:
                    session_length = (session[-1].created_at - session[0].created_at).total_seconds() / 60
                    session_lengths.append(session_length)
            
            # Check if returning user
            if len(sessions) > 1:
                returning_users += 1
        
        # Identify power users (top 10%)
        user_activity = [(user_id, len(convs)) for user_id, convs in user_conversations.items()]
        user_activity.sort(key=lambda x: x[1], reverse=True)
        
        power_user_threshold = max(1, len(user_activity) // 10)
        power_users = [
            {"user_id": user_id, "conversations": count}
            for user_id, count in user_activity[:power_user_threshold]
        ]
        
        # Engagement levels
        engagement_levels = {
            "high": sum(1 for _, count in user_activity if count >= 10),
            "medium": sum(1 for _, count in user_activity if 3 <= count < 10),
            "low": sum(1 for _, count in user_activity if count < 3)
        }
        
        return {
            "avg_session_length": round(sum(session_lengths) / len(session_lengths), 2) if session_lengths else 0,
            "return_rate": round((returning_users / len(user_conversations) * 100), 2) if user_conversations else 0,
            "power_users": power_users[:5],  # Top 5
            "engagement_levels": engagement_levels,
            "unique_users": len(user_conversations),
            "avg_conversations_per_user": round(len(conversations) / len(user_conversations), 2) if user_conversations else 0
        }
    
    def _analyze_content_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze content patterns in prompts"""
        
        if not conversations:
            return {
                "common_keywords": [],
                "avg_prompt_length": 0,
                "question_types": {},
                "topics": []
            }
        
        # Extract all prompts
        prompts = [conv.prompt for conv in conversations]
        
        # Calculate average length
        avg_length = sum(len(prompt.split()) for prompt in prompts) / len(prompts)
        
        # Extract keywords (simple approach - could use NLP libraries for better results)
        all_words = []
        question_types = Counter()
        
        for prompt in prompts:
            # Clean and tokenize
            words = re.findall(r'\b\w{4,}\b', prompt.lower())
            all_words.extend(words)
            
            # Classify question type
            prompt_lower = prompt.lower()
            if prompt_lower.startswith(('what', 'which')):
                question_types['what/which'] += 1
            elif prompt_lower.startswith('how'):
                question_types['how'] += 1
            elif prompt_lower.startswith('why'):
                question_types['why'] += 1
            elif prompt_lower.startswith(('can', 'could', 'would', 'should')):
                question_types['modal'] += 1
            elif '?' in prompt:
                question_types['other_question'] += 1
            else:
                question_types['statement'] += 1
        
        # Get common keywords (exclude common words)
        stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'what', 'which', 'when', 'where', 'how', 'why'}
        word_counts = Counter(word for word in all_words if word not in stop_words)
        common_keywords = word_counts.most_common(20)
        
        # Extract topics (simple clustering based on keywords)
        topics = self._extract_topics(word_counts)
        
        return {
            "common_keywords": [{"word": word, "count": count} for word, count in common_keywords],
            "avg_prompt_length": round(avg_length, 2),
            "question_types": dict(question_types),
            "topics": topics
        }
    
    def _extract_topics(self, word_counts: Counter) -> List[Dict[str, Any]]:
        """Extract topics from word counts"""
        
        # Simple topic extraction based on keyword categories
        topic_keywords = {
            "programming": ["code", "python", "javascript", "function", "error", "debug", "api", "database"],
            "data_analysis": ["data", "analysis", "statistics", "chart", "graph", "dataset", "visualization"],
            "machine_learning": ["model", "training", "neural", "learning", "algorithm", "prediction", "classification"],
            "web_development": ["website", "html", "css", "frontend", "backend", "server", "deploy"],
            "business": ["business", "strategy", "marketing", "sales", "customer", "revenue", "growth"],
            "education": ["learn", "explain", "understand", "concept", "theory", "example", "tutorial"]
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(word_counts.get(kw, 0) for kw in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # Sort and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"topic": topic, "relevance_score": score}
            for topic, score in sorted_topics[:5]
        ]
    
    async def _get_performance_insights(
        self,
        cutoff_date: datetime,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get performance-related insights"""
        
        try:
            # Query performance data
            query = db.session.query(
                ModelPerformance.task_type,
                ModelPerformance.provider,
                func.avg(ModelPerformance.latency_ms).label('avg_latency'),
                func.count(ModelPerformance.id).label('count'),
                func.sum(ModelPerformance.success.cast(db.Integer)).label('successes')
            ).filter(
                ModelPerformance.created_at >= cutoff_date
            ).group_by(
                ModelPerformance.task_type,
                ModelPerformance.provider
            )
            
            results = query.all()
            
            # Analyze results
            task_performance = defaultdict(lambda: {
                "best_provider": None,
                "worst_provider": None,
                "avg_latency": 0,
                "providers": {}
            })
            
            for result in results:
                task_type = result.task_type
                provider = result.provider
                success_rate = result.successes / result.count if result.count > 0 else 0
                
                task_performance[task_type]["providers"][provider] = {
                    "avg_latency": round(result.avg_latency, 2),
                    "success_rate": round(success_rate, 3),
                    "requests": result.count
                }
            
            # Determine best/worst providers per task
            for task_type, data in task_performance.items():
                if data["providers"]:
                    # Best provider (highest success rate)
                    best = max(
                        data["providers"].items(),
                        key=lambda x: x[1]["success_rate"]
                    )
                    data["best_provider"] = best[0]
                    
                    # Worst provider (lowest success rate)
                    worst = min(
                        data["providers"].items(),
                        key=lambda x: x[1]["success_rate"]
                    )
                    data["worst_provider"] = worst[0]
                    
                    # Average latency across providers
                    total_latency = sum(p["avg_latency"] * p["requests"] for p in data["providers"].values())
                    total_requests = sum(p["requests"] for p in data["providers"].values())
                    data["avg_latency"] = round(total_latency / total_requests, 2) if total_requests > 0 else 0
            
            return dict(task_performance)
            
        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {}
    
    def _generate_recommendations(self, conversations: List[Conversation]) -> List[str]:
        """Generate recommendations based on usage patterns"""
        
        recommendations = []
        
        if not conversations:
            recommendations.append("Start using the AI framework to get personalized recommendations")
            return recommendations
        
        # Analyze patterns
        task_counts = Counter(conv.task_type for conv in conversations)
        complexity_scores = [conv.complexity_score for conv in conversations if conv.complexity_score]
        
        # Task diversity recommendation
        if len(task_counts) < 3:
            recommendations.append(
                "Try exploring different task types to leverage the full capabilities of the framework"
            )
        
        # Complexity recommendation
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            if avg_complexity < 0.3:
                recommendations.append(
                    "Consider trying more complex tasks to benefit from multi-model consensus"
                )
            elif avg_complexity > 0.8:
                recommendations.append(
                    "High complexity tasks detected - ensure you're using consensus mode for better accuracy"
                )
        
        # Usage pattern recommendation
        hourly_counts = Counter(conv.created_at.hour for conv in conversations)
        if hourly_counts:
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
            if 0 <= peak_hour <= 6:
                recommendations.append(
                    "You're often active during off-peak hours - response times should be optimal"
                )
        
        # Frequency recommendation
        days_active = len(set(conv.created_at.date() for conv in conversations))
        if days_active < 3:
            recommendations.append(
                "Increase usage frequency to build better context and get more personalized responses"
            )
        
        return recommendations[:5]  # Return top 5 recommendations