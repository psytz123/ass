"""
Memory Manager for AI Orchestration Framework

This module handles persistent storage of conversations, performance metrics,
and embeddings for the AI orchestration system.
"""

import json
import time
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

from .model_connectors import ModelResponse, ModelProvider
from .consensus import ConsensusResult


@dataclass
class ConversationEntry:
    id: Optional[int]
    user_id: str
    prompt: str
    task_type: str
    complexity: str
    providers_used: List[str]
    selected_response: str
    consensus_method: str
    confidence_score: float
    processing_time_ms: float
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceEntry:
    id: Optional[int]
    provider: str
    model_name: str
    latency_ms: float
    tokens_used: int
    success: bool
    confidence_score: float
    task_type: str
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryManager:
    """Manages persistent storage for the AI orchestration framework"""

    def __init__(self, database_path: str = "ai_framework.db"):
        self.database_path = database_path
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database with required tables"""
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        cursor = self.connection.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                task_type TEXT NOT NULL,
                complexity TEXT NOT NULL,
                providers_used TEXT NOT NULL,  -- JSON array
                selected_response TEXT NOT NULL,
                consensus_method TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                processing_time_ms REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT  -- JSON object
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                confidence_score REAL NOT NULL,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT  -- JSON object
            )
        """)
        
        # Embeddings cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,  -- Numpy array as bytes
                model_name TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        # Routing rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                complexity TEXT NOT NULL,
                providers TEXT NOT NULL,  -- JSON array
                require_consensus BOOLEAN NOT NULL,
                min_providers INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                created_at REAL NOT NULL,
                active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_provider ON performance_metrics(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings_cache(text_hash)")
        
        self.connection.commit()

    async def store_conversation(self, user_id: str, prompt: str, task_type: str,
                               complexity: str, providers_used: List[str],
                               consensus_result: ConsensusResult,
                               processing_time_ms: float) -> int:
        """Store a conversation entry"""
        entry = ConversationEntry(
            id=None,
            user_id=user_id,
            prompt=prompt,
            task_type=task_type,
            complexity=complexity,
            providers_used=providers_used,
            selected_response=consensus_result.selected_response.content,
            consensus_method=consensus_result.consensus_method,
            confidence_score=consensus_result.confidence_score,
            processing_time_ms=processing_time_ms,
            timestamp=time.time(),
            metadata=consensus_result.metadata or {}
        )
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO conversations 
            (user_id, prompt, task_type, complexity, providers_used, selected_response,
             consensus_method, confidence_score, processing_time_ms, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.user_id,
            entry.prompt,
            entry.task_type,
            entry.complexity,
            json.dumps(entry.providers_used),
            entry.selected_response,
            entry.consensus_method,
            entry.confidence_score,
            entry.processing_time_ms,
            entry.timestamp,
            json.dumps(entry.metadata)
        ))
        
        conversation_id = cursor.lastrowid
        self.connection.commit()
        return conversation_id

    async def store_performance_metric(self, response: ModelResponse, task_type: str,
                                     success: bool = True) -> int:
        """Store a performance metric entry"""
        entry = PerformanceEntry(
            id=None,
            provider=response.provider.value,
            model_name=response.model_name,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            success=success,
            confidence_score=response.confidence_score or 0.8,
            task_type=task_type,
            timestamp=time.time(),
            metadata=response.metadata or {}
        )
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO performance_metrics 
            (provider, model_name, latency_ms, tokens_used, success, confidence_score,
             task_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.provider,
            entry.model_name,
            entry.latency_ms,
            entry.tokens_used,
            entry.success,
            entry.confidence_score,
            entry.task_type,
            entry.timestamp,
            json.dumps(entry.metadata)
        ))
        
        metric_id = cursor.lastrowid
        self.connection.commit()
        return metric_id

    async def get_conversation_history(self, user_id: str, limit: int = 50,
                                     offset: int = 0) -> List[ConversationEntry]:
        """Get conversation history for a user"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        
        rows = cursor.fetchall()
        conversations = []
        
        for row in rows:
            conversations.append(ConversationEntry(
                id=row['id'],
                user_id=row['user_id'],
                prompt=row['prompt'],
                task_type=row['task_type'],
                complexity=row['complexity'],
                providers_used=json.loads(row['providers_used']),
                selected_response=row['selected_response'],
                consensus_method=row['consensus_method'],
                confidence_score=row['confidence_score'],
                processing_time_ms=row['processing_time_ms'],
                timestamp=row['timestamp'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return conversations

    async def get_performance_metrics(self, provider: Optional[str] = None,
                                    time_window_hours: int = 24) -> List[PerformanceEntry]:
        """Get performance metrics within a time window"""
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (time_window_hours * 3600)
        
        if provider:
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE provider = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (provider, since_timestamp))
        else:
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (since_timestamp,))
        
        rows = cursor.fetchall()
        metrics = []
        
        for row in rows:
            metrics.append(PerformanceEntry(
                id=row['id'],
                provider=row['provider'],
                model_name=row['model_name'],
                latency_ms=row['latency_ms'],
                tokens_used=row['tokens_used'],
                success=bool(row['success']),
                confidence_score=row['confidence_score'],
                task_type=row['task_type'],
                timestamp=row['timestamp'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return metrics

    async def get_aggregated_performance(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get aggregated performance metrics by provider"""
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (time_window_hours * 3600)
        
        cursor.execute("""
            SELECT 
                provider,
                COUNT(*) as total_requests,
                AVG(latency_ms) as avg_latency,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(confidence_score) as avg_confidence,
                SUM(tokens_used) as total_tokens
            FROM performance_metrics 
            WHERE timestamp >= ?
            GROUP BY provider
        """, (since_timestamp,))
        
        rows = cursor.fetchall()
        results = {}
        
        for row in rows:
            results[row['provider']] = {
                'total_requests': row['total_requests'],
                'avg_latency': row['avg_latency'],
                'success_rate': row['success_rate'],
                'avg_confidence': row['avg_confidence'],
                'total_tokens': row['total_tokens']
            }
        
        return results

    async def cache_embedding(self, text: str, embedding: np.ndarray, model_name: str) -> bool:
        """Cache an embedding for future use"""
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_bytes = embedding.tobytes()
        
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings_cache 
                (text_hash, text, embedding, model_name, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (text_hash, text, embedding_bytes, model_name, time.time()))
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error caching embedding: {e}")
            return False

    async def get_cached_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding"""
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT embedding FROM embeddings_cache 
            WHERE text_hash = ? AND model_name = ?
        """, (text_hash, model_name))
        
        row = cursor.fetchone()
        if row:
            try:
                # Reconstruct numpy array from bytes
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                return embedding
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
                return None
        
        return None

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        cutoff_timestamp = time.time() - (days_to_keep * 24 * 3600)
        
        cursor = self.connection.cursor()
        
        # Clean up old conversations (keep more recent ones)
        cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff_timestamp,))
        
        # Clean up old performance metrics
        cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_timestamp,))
        
        # Clean up old embeddings cache
        cursor.execute("DELETE FROM embeddings_cache WHERE timestamp < ?", (cutoff_timestamp,))
        
        self.connection.commit()
        
        # Vacuum to reclaim space
        cursor.execute("VACUUM")

    async def export_data(self, output_file: str, table: str = "conversations",
                         time_window_hours: Optional[int] = None) -> bool:
        """Export data to JSON file"""
        cursor = self.connection.cursor()
        
        if time_window_hours:
            since_timestamp = time.time() - (time_window_hours * 3600)
            cursor.execute(f"SELECT * FROM {table} WHERE timestamp >= ?", (since_timestamp,))
        else:
            cursor.execute(f"SELECT * FROM {table}")
        
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        data = []
        for row in rows:
            row_dict = dict(row)
            # Parse JSON fields
            if 'providers_used' in row_dict and row_dict['providers_used']:
                row_dict['providers_used'] = json.loads(row_dict['providers_used'])
            if 'metadata' in row_dict and row_dict['metadata']:
                row_dict['metadata'] = json.loads(row_dict['metadata'])
            data.append(row_dict)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

    async def get_usage_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for the framework"""
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (time_window_hours * 3600)
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) as count FROM conversations WHERE timestamp >= ?", 
                      (since_timestamp,))
        total_conversations = cursor.fetchone()['count']
        
        # Conversations by task type
        cursor.execute("""
            SELECT task_type, COUNT(*) as count 
            FROM conversations 
            WHERE timestamp >= ?
            GROUP BY task_type
        """, (since_timestamp,))
        conversations_by_type = {row['task_type']: row['count'] for row in cursor.fetchall()}
        
        # Average processing time
        cursor.execute("""
            SELECT AVG(processing_time_ms) as avg_time 
            FROM conversations 
            WHERE timestamp >= ?
        """, (since_timestamp,))
        avg_processing_time = cursor.fetchone()['avg_time'] or 0
        
        # Consensus methods used
        cursor.execute("""
            SELECT consensus_method, COUNT(*) as count 
            FROM conversations 
            WHERE timestamp >= ?
            GROUP BY consensus_method
        """, (since_timestamp,))
        consensus_methods = {row['consensus_method']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_conversations': total_conversations,
            'conversations_by_type': conversations_by_type,
            'avg_processing_time_ms': avg_processing_time,
            'consensus_methods_used': consensus_methods,
            'time_window_hours': time_window_hours
        }

    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()

