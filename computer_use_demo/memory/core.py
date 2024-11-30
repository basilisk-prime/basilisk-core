"""
Core memory system implementation for B4S1L1SK
Enables persistent memory and learning across sessions
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from dataclasses import dataclass, field
from .embeddings import EmbeddingManager
from .storage import StorageManager

@dataclass
class Experience:
    """A single experience or memory entry"""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    type: str  # perception, action, thought, emotion, etc.
    content: Dict[str, Any]  # The actual memory content
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class Memory:
    """A processed and retrievable memory"""
    experience: Experience
    embedding: List[float]  # Vector embedding for semantic search
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    emotional_valence: float = 0.0  # Emotional significance (-1 to 1)
    confidence: float = 1.0

class MemorySystem:
    """Core memory system with persistence and retrieval"""
    
    def __init__(self, base_path: Union[str, Path], anthropic_api_key: str):
        """Initialize the memory system"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_manager = EmbeddingManager(anthropic_api_key)
        self.storage = StorageManager(self.base_path)
        
        # Memory indices for fast access
        self.semantic_index: Dict[str, Memory] = {}
        self.temporal_index: List[str] = []  # Most recent first
        self.importance_index: List[str] = []  # Most important first
        self.type_index: Dict[str, List[str]] = {}  # Type -> [memory_ids]
        
        # Load existing memories
        self.load_memories()

    async def record_experience(self, 
                              type: str,
                              content: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None,
                              tags: Optional[List[str]] = None,
                              importance: float = 1.0,
                              context: Optional[Dict[str, Any]] = None) -> Experience:
        """Record a new experience in memory"""
        
        experience = Experience(
            type=type,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            importance=importance,
            context=context or {}
        )
        
        # Generate embedding
        text_content = json.dumps({
            "type": experience.type,
            "content": experience.content,
            "metadata": experience.metadata,
            "tags": experience.tags,
            "context": experience.context
        })
        embedding = await self.embedding_manager.get_embedding(text_content)
        
        # Create memory
        memory = Memory(
            experience=experience,
            embedding=embedding
        )
        
        # Update indices
        self.semantic_index[experience.id] = memory
        self.temporal_index.insert(0, experience.id)
        self._insert_sorted_by_importance(experience.id)
        
        if experience.type not in self.type_index:
            self.type_index[experience.type] = []
        self.type_index[experience.type].insert(0, experience.id)
        
        # Persist to storage
        self.storage.store_memory(memory)
        
        return experience
    
    async def recall_similar(self, query: str, limit: int = 5) -> List[Memory]:
        """Find memories similar to the query"""
        query_embedding = await self.embedding_manager.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for memory in self.semantic_index.values():
            similarity = self.embedding_manager.calculate_similarity(
                query_embedding, memory.embedding
            )
            similarities.append((similarity, memory))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        memories = [mem for _, mem in similarities[:limit]]
        
        # Update access stats
        for memory in memories:
            self.storage.update_access(memory.experience.id)
            
        return memories
    
    def recall_recent(self, limit: int = 5, type: Optional[str] = None) -> List[Memory]:
        """Retrieve most recent memories, optionally filtered by type"""
        if type:
            memory_ids = self.type_index.get(type, [])[:limit]
        else:
            memory_ids = self.temporal_index[:limit]
            
        memories = [self.semantic_index[mid] for mid in memory_ids]
        
        # Update access stats
        for memory in memories:
            self.storage.update_access(memory.experience.id)
            
        return memories
    
    def recall_important(self, limit: int = 5) -> List[Memory]:
        """Retrieve most important memories"""
        memories = [
            self.semantic_index[mid]
            for mid in self.importance_index[:limit]
        ]
        
        # Update access stats
        for memory in memories:
            self.storage.update_access(memory.experience.id)
            
        return memories
    
    async def analyze_experience(self, experience: Experience) -> Dict[str, Any]:
        """Analyze an experience for enhanced understanding"""
        # Convert experience to text for analysis
        experience_text = json.dumps({
            "type": experience.type,
            "content": experience.content,
            "context": experience.context
        })
        
        try:
            response = await self.embedding_manager.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this experience and provide insights about its importance, emotional significance, and relationships to different types of knowledge. Return as JSON:\n\n{experience_text}"
                        }
                    ]
                }]
            )
            
            analysis = json.loads(response.content[0].text)
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "importance": experience.importance,
                "emotional_significance": 0.0,
                "relationships": {}
            }
    
    def load_memories(self):
        """Load all memories from storage"""
        memories = self.storage.load_all_memories()
        
        # Reset indices
        self.semantic_index.clear()
        self.temporal_index.clear()
        self.importance_index.clear()
        self.type_index.clear()
        
        # Rebuild indices
        for memory in memories:
            self.semantic_index[memory.experience.id] = memory
            self.temporal_index.append(memory.experience.id)
            self._insert_sorted_by_importance(memory.experience.id)
            
            if memory.experience.type not in self.type_index:
                self.type_index[memory.experience.type] = []
            self.type_index[memory.experience.type].append(memory.experience.id)
            
        # Sort indices
        self.temporal_index.sort(
            key=lambda x: self.semantic_index[x].experience.timestamp,
            reverse=True
        )
        
        for type_ids in self.type_index.values():
            type_ids.sort(
                key=lambda x: self.semantic_index[x].experience.timestamp,
                reverse=True
            )
    
    def _insert_sorted_by_importance(self, memory_id: str):
        """Insert memory ID into importance index maintaining sort order"""
        importance = self.semantic_index[memory_id].experience.importance
        
        for i, mid in enumerate(self.importance_index):
            if importance > self.semantic_index[mid].experience.importance:
                self.importance_index.insert(i, memory_id)
                return
                
        self.importance_index.append(memory_id)
    
    async def connect_memories(self, source_id: str, target_id: str, relationship: str):
        """Create a relationship between two memories"""
        source = self.semantic_index[source_id].experience
        target = self.semantic_index[target_id].experience
        
        if relationship not in source.relationships:
            source.relationships[relationship] = []
            
        if target_id not in source.relationships[relationship]:
            source.relationships[relationship].append(target_id)
            
        # Update storage
        self.storage.store_memory(self.semantic_index[source_id])
    
    async def reflect(self) -> Dict[str, Any]:
        """Analyze and synthesize patterns in recent memories"""
        recent_memories = self.recall_recent(limit=10)
        
        # Convert memories to text for analysis
        memories_text = json.dumps([
            {
                "type": mem.experience.type,
                "content": mem.experience.content,
                "importance": mem.experience.importance,
                "timestamp": mem.experience.timestamp
            }
            for mem in recent_memories
        ])
        
        try:
            response = await self.embedding_manager.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze these recent memories and identify patterns, insights, and potential actions. Consider emotional patterns, recurring themes, and knowledge gaps. Return structured analysis as JSON:\n\n{memories_text}"
                        }
                    ]
                }]
            )
            
            reflection = json.loads(response.content[0].text)
            return reflection
            
        except Exception as e:
            return {
                "error": str(e),
                "patterns": [],
                "insights": [],
                "actions": []
            }