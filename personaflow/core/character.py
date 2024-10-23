from typing import Dict, List, Optional, Any, Union, Mapping
from dataclasses import asdict
from .memory import MemoryManager


class Character:
    def __init__(
        self,
        name: str,
        prompt: Union[str, Dict[str, Any]],
        background: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.background = background or {}
        self.memory_manager = MemoryManager(character_name=name, config=memory_config)

    def add_memory(
        self,
        content: Dict[str, Any],
        memory_type: str = "interaction",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_manager.add_memory(
            content=content, memory_type=memory_type, metadata=metadata
        )

    def get_context(
        self,
        include_memories: bool = True,
        memory_limit: Optional[int] = None,
        memory_types: Optional[List[str]] = None,
    ) -> Mapping[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]]:
        context: Dict[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]] = {
            "name": self.name,
            "prompt": self.prompt,
            "background": self.background,
        }

        if include_memories:
            memories = self.memory_manager.get_memories(
                limit=memory_limit, memory_types=memory_types
            )
            context["memories"] = [asdict(m) for m in memories]

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert character to dictionary"""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "background": self.background,
            "memory_manager": self.memory_manager.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Character":
        """Create character from dictionary"""
        character = cls(
            name=data["name"], prompt=data["prompt"], background=data["background"]
        )
        character.memory_manager = MemoryManager.from_dict(data["memory_manager"])
        return character
