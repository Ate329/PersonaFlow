import pytest
from personaflow.core.system import PersonaSystem

class TestPersonaSystem:
    @pytest.fixture
    def system(self):
        return PersonaSystem()

    def test_create_character(self, system):
        character = system.create_character(
            name="test_char",
            prompt="Test prompt",
            background={"role": "test"}
        )

        assert "test_char" in system.characters
        assert system.characters["test_char"].name == "test_char"

    def test_get_character(self, system):
        system.create_character(name="test_char", prompt="Test prompt")
        character = system.get_character("test_char")
        assert character.name == "test_char"

        with pytest.raises(KeyError):
            system.get_character("nonexistent")

    def test_switch_active_character(self, system):
        system.create_character(name="char1", prompt="Prompt 1")
        system.create_character(name="char2", prompt="Prompt 2")

        system.switch_active_character("char1")
        assert system.active_character == "char1"

        system.switch_active_character("char2")
        assert system.active_character == "char2"

        with pytest.raises(KeyError):
            system.switch_active_character("nonexistent")

    def test_add_interaction(self, system):
        system.create_character(name="test_char", prompt="Test prompt")

        system.add_interaction(
            character_name="test_char",
            content={"user": "Hello", "response": "Hi"}
        )

        character = system.get_character("test_char")
        context = character.get_context()
        assert len(context["memories"]) == 1

    def test_broadcast_interaction(self, system):
        system.create_character(name="char1", prompt="Prompt 1")
        system.create_character(name="char2", prompt="Prompt 2")

        system.broadcast_interaction(
            content={"event": "Global event"}
        )

        for char_name in ["char1", "char2"]:
            character = system.get_character(char_name)
            context = character.get_context()
            assert len(context["memories"]) == 1
            assert context["memories"][0]["content"]["event"] == "Global event"

    def test_serialization(self, system):
        system.create_character(name="test_char", prompt="Test prompt")
        system.add_interaction(
            character_name="test_char",
            content={"user": "Hello", "response": "Hi"}
        )

        data = system.to_dict()
        new_system = PersonaSystem.from_dict(data)

        assert "test_char" in new_system.characters
        character = new_system.get_character("test_char")
        context = character.get_context()
        assert len(context["memories"]) == 1
