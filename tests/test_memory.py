import pytest
from datetime import datetime
from personaflow.core.memory import Memory, MemoryConfig, MemoryManager

def test_memory_creation():
    memory = Memory(
        timestamp=datetime.now().isoformat(),
        type="interaction",
        content={"user": "Hello", "response": "Hi"},
        character_name="test_char"
    )
    assert memory.type == "interaction"
    assert memory.content["user"] == "Hello"
    assert memory.character_name == "test_char"

def test_memory_config():
    config = MemoryConfig()
    assert config.max_memories == 1000
    assert config.summary_threshold == 10
    assert config.auto_summarize == True

    custom_config = MemoryConfig(
        max_memories=500,
        summary_threshold=5,
        auto_summarize=False
    )
    assert custom_config.max_memories == 500
    assert custom_config.summary_threshold == 5
    assert custom_config.auto_summarize == False

class TestMemoryManager:
    @pytest.fixture
    def memory_manager(self):
        return MemoryManager("test_char")

    def test_add_memory(self, memory_manager):
        content = {"user": "Hello", "response": "Hi"}
        memory_manager.add_memory(content)
        assert len(memory_manager.memories) == 1
        assert memory_manager.memories[0].content == content

    def test_get_memories(self, memory_manager):
        # Add multiple memories
        for i in range(5):
            content = {"user": f"Message {i}", "response": f"Reply {i}"}
            memory_manager.add_memory(content)

        # Test limit
        memories = memory_manager.get_memories(limit=3)
        assert len(memories) == 3

        # Test memory types
        memory_manager.add_memory(
            {"event": "test"},
            memory_type="event"
        )
        event_memories = memory_manager.get_memories(memory_types=["event"])
        assert len(event_memories) == 1
        assert event_memories[0].type == "event"

    def test_memory_size_management(self, memory_manager):
        # Override config for testing
        memory_manager.config.max_memories = 5

        # Add more memories than max
        for i in range(10):
            content = {"user": f"Message {i}", "response": f"Reply {i}"}
            memory_manager.add_memory(content)

        assert len(memory_manager.memories) <= 5

    def test_serialization(self, memory_manager):
        content = {"user": "Hello", "response": "Hi"}
        memory_manager.add_memory(content)

        data = memory_manager.to_dict()
        new_manager = MemoryManager.from_dict(data)

        assert new_manager.character_name == memory_manager.character_name
        assert len(new_manager.memories) == len(memory_manager.memories)
        assert new_manager.memories[0].content == memory_manager.memories[0].content
