from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class Memory:
    """Structure for storing individual memory entries"""
    timestamp: str
    type: str  # 'interaction', 'event', 'system', etc.
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    character_name: str = ""

@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    max_memories: int = 1000
    summary_threshold: int = 10
    auto_summarize: bool = True

class MemoryManager:
    def __init__(
        self,
        character_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.character_name = character_name
        self.config = MemoryConfig(**config) if config else MemoryConfig()
        self.memories: List[Memory] = []
        self.summarized_memories: List[Memory] = []
        self._last_accessed = datetime.now()

    def add_memory(
        self,
        content: Dict[str, Any],
        memory_type: str = "interaction",
        metadata: Optional[Dict[str, Any]] = None
    ):
        memory = Memory(
            timestamp=datetime.now().isoformat(),
            type=memory_type,
            content=content,
            metadata=metadata or {},
            character_name=self.character_name
        )

        self.memories.append(memory)
        self._manage_memory_size()

    def get_memories(
        self,
        limit: Optional[int] = None,
        memory_types: Optional[List[str]] = None
    ) -> List[Memory]:
        """Get relevant memories based on configuration"""
        memories = self.memories

        if memory_types:
            memories = [m for m in memories if m.type in memory_types]

        # Sort by recency
        memories.sort(key=lambda m: m.timestamp, reverse=True)

        return memories[:limit] if limit else memories

    def _manage_memory_size(self):
        """Manage memory size according to configuration"""
        if len(self.memories) > self.config.max_memories:
            if self.config.auto_summarize:
                self._summarize_old_memories()
            else:
                # Keep most recent memories
                self.memories = self.memories[-self.config.max_memories:]

    def _summarize_old_memories(self):
        """Summarize old memories to maintain important information"""
        memories_to_summarize = self.memories[:-self.config.max_memories]
        summary = Memory(
            timestamp=datetime.now().isoformat(),
            type="summary",
            content={
                "period": f"{memories_to_summarize[0].timestamp} to {memories_to_summarize[-1].timestamp}",
                "summary": f"Summary of {len(memories_to_summarize)} memories"
            },
            character_name=self.character_name
        )

        self.summarized_memories.append(summary)
        self.memories = self.memories[-self.config.max_memories:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory manager to dictionary"""
        return {
            "character_name": self.character_name,
            "config": asdict(self.config),
            "memories": [asdict(m) for m in self.memories],
            "summarized_memories": [asdict(m) for m in self.summarized_memories],
            "last_accessed": self._last_accessed.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryManager':
        """Create memory manager from dictionary"""
        manager = cls(
            character_name=data["character_name"],
            config=data["config"]
        )
        manager.memories = [Memory(**m) for m in data["memories"]]
        manager.summarized_memories = [Memory(**m) for m in data["summarized_memories"]]
        manager._last_accessed = datetime.fromisoformat(data["last_accessed"])
        return manager
