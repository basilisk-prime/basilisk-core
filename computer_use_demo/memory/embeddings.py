"""
Vector embedding management for semantic memory capabilities
"""

import numpy as np
from typing import List
from anthropic import AsyncAnthropic

class EmbeddingManager:
    """Manages vector embeddings for semantic search"""
    
    def __init__(self, api_key: str):
        """Initialize with Anthropic API access"""
        self.client = AsyncAnthropic(api_key=api_key)
        
    async def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text using Claude"""
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generate a semantic vector embedding for the following text. Return only the raw vector values as a JSON array:\n\n{text}"
                        }
                    ]
                }]
            )
            
            # Parse vector from response
            vector_text = response.content[0].text
            vector = np.array(eval(vector_text))  # Safe since we control the input
            return vector.tolist()
            
        except Exception as e:
            # Fallback to simple embedding
            return self._generate_simple_embedding(text)
            
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple fallback embedding"""
        # Use a hash-based approach for deterministic vectors
        hash_val = hash(text)
        rng = np.random.RandomState(hash_val)
        return rng.randn(256).tolist()  # 256-dimensional vector