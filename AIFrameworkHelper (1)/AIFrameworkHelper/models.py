from datetime import datetime
from app import db
from sqlalchemy import Text, JSON, Float, Integer, String, DateTime, Boolean, Column

class Conversation(db.Model):
    """Store conversation history and metadata"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(128), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    task_type = Column(String(64), nullable=False)
    providers_used = Column(String(256), nullable=False)  # JSON array as string
    consensus_confidence = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=False)
    complexity_score = Column(Float, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPerformance(db.Model):
    """Track performance metrics for each model provider"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(64), nullable=False)
    model_name = Column(String(128), nullable=False)
    task_type = Column(String(64), nullable=False)
    latency_ms = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class EmbeddingCache(db.Model):
    """Cache embeddings for similarity calculations"""
    __tablename__ = 'embedding_cache'
    
    id = Column(Integer, primary_key=True)
    content_hash = Column(String(64), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON array as string
    model_name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class RoutingRule(db.Model):
    """Store custom routing rules"""
    __tablename__ = 'routing_rules'
    
    id = Column(Integer, primary_key=True)
    task_type = Column(String(64), nullable=False)
    complexity_min = Column(Float, nullable=False, default=0.0)
    complexity_max = Column(Float, nullable=False, default=1.0)
    preferred_providers = Column(String(256), nullable=False)  # JSON array as string
    require_consensus = Column(Boolean, nullable=False, default=False)
    priority = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Webhook(db.Model):
    """Store webhook configurations"""
    __tablename__ = 'webhooks'
    
    id = Column(Integer, primary_key=True)
    webhook_id = Column(String(128), nullable=False, unique=True)
    url = Column(String(512), nullable=False)
    events = Column(Text, nullable=False)  # JSON array of event types
    secret = Column(String(256), nullable=True)  # For signature validation
    description = Column(Text, nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime, nullable=True)
    trigger_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
