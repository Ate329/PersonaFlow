import pytest
from personaflow.core.character import Character

class TestCharacter:
    @pytest.fixture
    def character(self):
        return Character(
            name="test_char",
            prompt="You are a test character",
            background={"role": "test"}
        )

    def test_character_creation(self, character):
        assert character.name == "test_char"
        assert character.prompt == "You are a test character"
        assert character.background["role"] == "test"
        assert character.memory_manager is not None

    def test_add_memory(self, character):
        content = {"user": "Hello", "response": "Hi"}
        character.add_memory(content)

        context = character.get_context()
        assert len(context["memories"]) == 1
        assert context["memories"][0]["content"] == content

    def test_get_context(self, character):
        # Test without memories
        context = character.get_context(include_memories=False)
        assert "memories" not in context
        assert context["name"] == character.name
        assert context["prompt"] == character.prompt
        assert context["background"] == character.background

        # Test with memories
        character.add_memory({"user": "Hello", "response": "Hi"})
        context = character.get_context(include_memories=True)
        assert "memories" in context
        assert len(context["memories"]) == 1

    def test_serialization(self, character):
        character.add_memory({"user": "Hello", "response": "Hi"})

        data = character.to_dict()
        new_character = Character.from_dict(data)

        assert new_character.name == character.name
        assert new_character.prompt == character.prompt
        assert new_character.background == character.background

        new_context = new_character.get_context()
        assert len(new_context["memories"]) == 1
