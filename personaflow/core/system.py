from typing import Dict, Optional, Any
from .character import Character


class PersonaSystem:
    def __init__(self) -> None:
        self.characters: Dict[str, Character] = {}
        self.active_character: Optional[str] = None

    def create_character(
        self,
        name: str,  # Changed from id to name to match Character class
        prompt: str,
        background: Optional[Dict] = None,
        memory_config: Optional[Dict] = None,
    ) -> Character:
        """Create and register a new character"""
        character = Character(
            name=name,  # Changed from id to name
            prompt=prompt,
            background=background,
            memory_config=memory_config,
        )
        self.characters[name] = character
        return character

    def get_character(self, name: str) -> Character:
        """Get character by name"""
        return self.characters[name]

    def switch_active_character(self, name: str):
        """Switch the active character"""
        if name in self.characters:
            self.active_character = name
        else:
            raise KeyError(f"Character {name} not found")

    def add_interaction(
        self,
        character_name: str,  # Changed from character_id
        content: Dict,  # Changed from interaction
        memory_type: str = "interaction",
        metadata: Optional[Dict] = None,
    ):
        """Add interaction to character's memory"""
        character = self.characters[character_name]
        character.add_memory(
            content=content, memory_type=memory_type, metadata=metadata
        )

    def broadcast_interaction(
        self,
        content: Dict,  # Changed from interaction
        memory_type: str = "broadcast",
        metadata: Optional[Dict] = None,
    ):
        """Broadcast interaction to all characters"""
        for character in self.characters.values():
            character.add_memory(
                content=content, memory_type=memory_type, metadata=metadata
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert system to dictionary"""
        return {
            "characters": {
                name: char.to_dict() for name, char in self.characters.items()
            },
            "active_character": self.active_character,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaSystem":
        """Create system from dictionary"""
        system = cls()
        system.characters = {
            name: Character.from_dict(char_data)
            for name, char_data in data["characters"].items()
        }
        system.active_character = data["active_character"]
        return system
