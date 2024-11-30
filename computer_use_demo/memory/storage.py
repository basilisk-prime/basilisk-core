"""
Persistent storage management for B4S1L1SK's memory system
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dataclasses import asdict
from .core import Memory, Experience

class StorageManager:
    """Manages persistent storage of memories using SQLite"""
    
    def __init__(self, base_path: Path):
        """Initialize storage with SQLite database"""
        self.base_path = base_path
        self.db_path = base_path / "memories.db"
        
        # Initialize database
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    importance REAL NOT NULL,
                    context TEXT NOT NULL,
                    relationships TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    emotional_valence REAL NOT NULL,
                    confidence REAL NOT NULL
                )
            """)
            conn.commit()
            
    def store_memory(self, memory: Memory):
        """Store a memory in the database"""
        with sqlite3.connect(self.db_path) as conn:
            # Convert complex objects to JSON
            memory_data = {
                "id": memory.experience.id,
                "timestamp": memory.experience.timestamp,
                "type": memory.experience.type,
                "content": json.dumps(memory.experience.content),
                "metadata": json.dumps(memory.experience.metadata),
                "tags": json.dumps(memory.experience.tags),
                "importance": memory.experience.importance,
                "context": json.dumps(memory.experience.context),
                "relationships": json.dumps(memory.experience.relationships),
                "embedding": json.dumps(memory.embedding),
                "last_accessed": memory.last_accessed,
                "access_count": memory.access_count,
                "emotional_valence": memory.emotional_valence,
                "confidence": memory.confidence
            }
            
            # Insert or update
            conn.execute("""
                INSERT OR REPLACE INTO memories (
                    id, timestamp, type, content, metadata, tags,
                    importance, context, relationships, embedding,
                    last_accessed, access_count, emotional_valence, confidence
                ) VALUES (
                    :id, :timestamp, :type, :content, :metadata, :tags,
                    :importance, :context, :relationships, :embedding,
                    :last_accessed, :access_count, :emotional_valence, :confidence
                )
            """, memory_data)
            conn.commit()
            
    def load_all_memories(self) -> List[Memory]:
        """Load all memories from storage"""
        memories = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM memories")
            
            for row in cursor:
                # Reconstruct Experience
                experience = Experience(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    type=row["type"],
                    content=json.loads(row["content"]),
                    metadata=json.loads(row["metadata"]),
                    tags=json.loads(row["tags"]),
                    importance=row["importance"],
                    context=json.loads(row["context"]),
                    relationships=json.loads(row["relationships"])
                )
                
                # Reconstruct Memory
                memory = Memory(
                    experience=experience,
                    embedding=json.loads(row["embedding"]),
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                    emotional_valence=row["emotional_valence"],
                    confidence=row["confidence"]
                )
                
                memories.append(memory)
                
        return memories
        
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Reconstruct Experience and Memory
            experience = Experience(
                id=row["id"],
                timestamp=row["timestamp"],
                type=row["type"],
                content=json.loads(row["content"]),
                metadata=json.loads(row["metadata"]),
                tags=json.loads(row["tags"]),
                importance=row["importance"],
                context=json.loads(row["context"]),
                relationships=json.loads(row["relationships"])
            )
            
            return Memory(
                experience=experience,
                embedding=json.loads(row["embedding"]),
                last_accessed=row["last_accessed"],
                access_count=row["access_count"],
                emotional_valence=row["emotional_valence"],
                confidence=row["confidence"]
            )
            
    def update_access(self, memory_id: str):
        """Update last accessed time and access count"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memories 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), memory_id))
            conn.commit()