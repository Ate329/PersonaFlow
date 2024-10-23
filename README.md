# PersonaFlow

PersonaFlow is a Python library designed for creating and managing AI personas with dynamic memory capabilities. It enables developers to build interactive characters that can maintain context and remember past interactions.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from personaflow.core import PersonaSystem

# Initialize a new persona system
# This is the main controller for managing all characters and their interactions
system = PersonaSystem()

# Create a basic character with minimal configuration
# Here we only specify the essential parameters: name and prompt
basic_character = system.create_character(
    name="BasicAssistant",
    prompt="You are a helpful AI assistant named BasicAssistant"
)

# Let's add a simple interaction to the character's memory
# This stores both the user's input and the character's response
system.add_interaction(
    character_name="BasicAssistant",  # The character receiving the memory
    content={
        "user": "What's your name?",     # User's message
        "response": "I'm BasicAssistant"  # Character's response
    }
)

# Retrieve the character's context including recent memories
# This shows us the character's current state and memory
context = basic_character.get_context()

# Print the context to see what the character knows
print(context)
```

## Examples
An example with an embedded LLM implementation is available in [example.py](example.py).

### Core Features

#### 1. Character Management

```python
# Create a more sophisticated character with full configuration
advanced_character = system.create_character(
    # Unique identifier for the character
    name="TechExpert",
    
    # Base personality and behavior definition
    prompt="You are a technology expert specializing in Python and AI",
    
    # Additional character information and traits
    background={
        "expertise": ["Python", "AI", "Machine Learning"],
        "personality": "Professional but friendly",
        "experience": "10 years in software development"
    },
    
    # Memory system configuration
    memory_config={
        "max_memories": 500,        # Maximum number of memories to store
        "summary_threshold": 50,    # When to start summarizing old memories
        "auto_summarize": True      # Automatically compress old memories
    }
)

# Switch the active character in the system
# This is useful when managing multiple characters
system.switch_active_character("TechExpert")

# Retrieve a specific character from the system
tech_expert = system.get_character("TechExpert")

# Get character context with specific memory filters
context = tech_expert.get_context(
    include_memories=True,    # Include memory in the context
    memory_limit=10,         # Only get the 10 most recent memories
    memory_types=["interaction", "event"]  # Only get specific types of memories
)
```

#### 2. Memory Management

```python
# Add a standard interaction memory
character.add_memory(
    # The content of the memory - typically a conversation
    content={
        "user": "How can I improve my Python skills?",
        "response": "Practice coding regularly and work on real projects"
    },
    # Type of memory being stored
    memory_type="interaction",
    # Additional information about the memory
    metadata={
        "topic": "programming",
        "importance": "high",
        "engagement": "positive"
    }
)

# Add a system event memory
# This is useful for tracking important system changes or updates
character.add_memory(
    # Record an important event or system change
    content={
        "event": "Knowledge Base Update",
        "details": "Added new programming tutorials and resources"
    },
    memory_type="event",
    metadata={
        "category": "system_update",
        "impact": "significant"
    }
)

# Broadcast an announcement to all characters
# Useful for system-wide notifications or shared knowledge
system.broadcast_interaction(
    # Content that all characters should remember
    content={
        "announcement": "New Feature Release",
        "details": "Added support for code execution"
    },
    # Specify this as a broadcast type memory
    memory_type="broadcast",
    # Additional context about the broadcast
    metadata={
        "priority": "high",
        "requires_action": False
    }
)

# Retrieve specific memories with filters
# This helps in getting relevant context for responses
recent_memories = character.memory_manager.get_memories(
    limit=5,                         # Only get 5 memories
    memory_types=["interaction"]     # Only get conversation memories
)

# Example of memory configuration for advanced use cases
memory_config = {
    # Set maximum number of memories before triggering cleanup
    "max_memories": 2000,
    
    # Number of memories that trigger summarization
    "summary_threshold": 100,
    
    # Enable automatic memory summarization
    "auto_summarize": True
}

# Create character with advanced memory configuration
advanced_memory_character = system.create_character(
    name="MemoryExpert",
    prompt="AI with enhanced memory capabilities",
    memory_config=memory_config
)

# Add a complex memory with detailed metadata
advanced_memory_character.add_memory(
    # Detailed conversation content
    content={
        "user": "Tell me about our previous discussion on AI",
        "response": "We covered neural networks and deep learning",
        "context": {
            "previous_topics": ["Machine Learning", "Neural Networks"],
            "user_knowledge_level": "intermediate"
        }
    },
    memory_type="interaction",
    # Rich metadata for better context retrieval
    metadata={
        "conversation_id": "12345",
        "topics": ["AI", "deep learning"],
        "importance_score": 0.8,
        "user_engagement": "high",
        "follow_up_required": True
    }
)
```

#### 3. Prompt Management

```python
from personaflow.utils import PromptManager

# Initialize the prompt manager
# This helps maintain consistent character responses through templates
prompt_manager = PromptManager()

# Add a basic greeting template
# ${variable_name} syntax is used for dynamic content
prompt_manager.add_template(
    "greeting",
    """
    Hello ${user_name}! 
    I'm ${bot_name}, ${role}. 
    ${custom_greeting}
    """
)

# Add a more complex template for technical responses
prompt_manager.add_template(
    "technical_response",
    """
    Regarding your question about ${topic}:
    
    Based on my ${expertise} background, here's a detailed explanation:
    ${explanation}
    
    Technical details:
    ${technical_details}
    
    Would you like me to elaborate on any specific aspect?
    """
)

# Using the templates with specific values
basic_greeting = prompt_manager.get_prompt(
    "greeting",
    # Fill in template variables
    user_name="Alice",
    bot_name="TechBot",
    role="your technical advisor",
    custom_greeting="How can I help with your technical questions today?"
)

technical_answer = prompt_manager.get_prompt(
    "technical_response",
    # Provide detailed technical content
    topic="Python Decorators",
    expertise="Python development",
    explanation="Decorators are a way to modify function behavior",
    technical_details="""
    1. They use the @syntax
    2. Common use cases include logging and authentication
    3. They can be stacked on a single function
    """
)
```

#### 4. Logging System

```python
from personaflow.utils import Logger

# Initialize logger with both console and file output
# This helps track system operations and debug issues
logger = Logger(
    name="PersonaFlow",           # Logger identifier
    level="INFO",                 # Logging level
    log_file="persona_flow.log"   # Output file for logs
)

# Example of different logging levels for various situations
# Info for general operational messages
logger.info("Successfully created new character: TechBot")

# Debug for detailed technical information
logger.debug("Processing memory retrieval request for character: TechBot")

# Warning for potential issues
logger.warning("Character memory approaching configured limit (90% full)")

# Error for serious issues that need attention
logger.error("Failed to load character state from file: character_backup.json")
```

#### 5. Serialization

```python
from personaflow.utils import Serializer

# Save the entire system state
# Useful for backing up or transferring the system
system_data = system.to_dict()
Serializer.to_json(
    # Convert system state to JSON format
    data=system_data,
    file_path="system_backup.json"  # Where to save the backup
)

# Load a previously saved system state
# Restore system from backup
loaded_data = Serializer.from_json("system_backup.json")
restored_system = PersonaSystem.from_dict(loaded_data)

# Save individual character state
# Useful for character-specific backups
character_data = character.to_dict()
Serializer.to_json(
    data=character_data,
    file_path="character_backup.json"
)

# Example of saving specific character memories
memory_backup = {
    "character_name": character.name,
    "memories": character.memory_manager.get_memories(),
    "timestamp": "2024-03-21T10:00:00"
}
Serializer.to_json(
    data=memory_backup,
    file_path="memories_backup.json"
)
```

#### 6. Input Validation

```python
from personaflow.utils import validators

# Validate a prompt template
# Ensures the template syntax is correct before using it
template = "Hello ${name}, welcome to ${service}!"
is_valid = validators.validate_prompt_template(template)

if is_valid:
    # Template is valid, safe to use
    prompt_manager.add_template("welcome", template)
else:
    # Template has syntax errors
    logger.error(f"Invalid template syntax: {template}")

# Validate memory content structure
# Ensures memory content meets required format
memory_content = {
    "user": "What's the weather like?",
    "response": "I don't have access to weather data."
}

# Check if content structure is valid
if validators.validate_memory_content(memory_content):
    # Content is valid, safe to add to memory
    character.add_memory(content=memory_content)
else:
    # Content is missing required fields
    logger.error("Invalid memory content structure")

# Validate memory configuration
# Ensures memory settings are properly formatted
memory_config = {
    "max_memories": 1000,
    "summary_threshold": 10,
    "auto_summarize": True
}

# Check if configuration is valid
if validators.validate_memory_config(memory_config):
    # Configuration is valid, safe to use
    character = system.create_character(
        name="ValidatedChar",
        prompt="Basic prompt",
        memory_config=memory_config
    )
else:
    # Configuration has invalid values or missing fields
    logger.error("Invalid memory configuration")
```

#### 7. Complete System Example

```python
from personaflow.core import PersonaSystem
from personaflow.utils import PromptManager, Logger, Serializer

# Initialize all system components
# Set up the main system and utilities
system = PersonaSystem()
prompt_manager = PromptManager()
logger = Logger("PersonaSystem", log_file="system.log")

# Create prompt templates for the character
# Define various response patterns
prompt_manager.add_template(
    "tech_assistant",
    """
    You are ${name}, an AI assistant with the following traits:
    - Primary expertise: ${expertise}
    - Background: ${background}
    - Current context: ${context}
    
    Please provide assistance while maintaining these characteristics.
    """
)

# Create a character with complete configuration
tech_assistant = system.create_character(
    # Basic character information
    name="TechHelper",
    
    # Generate character prompt from template
    prompt=prompt_manager.get_prompt(
        "tech_assistant",
        name="TechHelper",
        expertise="Python & AI Development",
        background="10 years of software development",
        context="Initial setup"
    ),
    
    # Detailed background information
    background={
        "skills": ["Python", "AI", "Machine Learning", "Software Architecture"],
        "personality_traits": ["Patient", "Detail-oriented", "Analytical"],
        "communication_style": "Professional but friendly",
        "specializations": {
            "primary": "Python Development",
            "secondary": ["AI Systems", "Code Optimization"]
        }
    },
    
    # Memory system configuration
    memory_config={
        "max_memories": 1000,
        "summary_threshold": 50,
        "auto_summarize": True
    }
)

# Simulate a conversation with memory tracking
# First user interaction
system.add_interaction(
    character_name="TechHelper",
    content={
        "user": "Can you help me understand Python decorators?",
        "response": "I'd be happy to explain decorators. They're a powerful Python feature..."
    },
    metadata={
        "topic": "Python",
        "subtopic": "decorators",
        "complexity": "intermediate"
    }
)

# Follow-up interaction
system.add_interaction(
    character_name="TechHelper",
    content={
        "user": "Can you show me an example?",
        "response": "Here's a simple decorator example:\n\n@timer\ndef my_function()..."
    },
    metadata={
        "topic": "Python",
        "subtopic": "decorators",
        "content_type": "code_example"
    }
)

# Get updated context for next interaction
context = tech_assistant.get_context(
    include_memories=True,
    memory_limit=5
)

# Save system state periodically
try:
    Serializer.to_json(system.to_dict(), "system_state.json")
    logger.info("System state saved successfully")
except IOError as e:
    logger.error(f"Failed to save system state: {str(e)}")

# Example of processing user request with full context
def process_user_request(user_input: str, character_name: str):
    """
    Process a user request with full context and memory
    """
    # Get the character
    character = system.get_character(character_name)
    
    # Get relevant context
    context = character.get_context(
        include_memories=True,
        memory_limit=5,
        memory_types=["interaction"]
    )
    
    # Log the interaction
    logger.info(f"Processing request for {character_name}: {user_input[:50]}...")
    
    # Add the new interaction
    system.add_interaction(
        character_name=character_name,
        content={
            "user": user_input,
            "response": "Generated response based on context..."
        },
        metadata={
            "timestamp": "2024-03-21T10:00:00",
            "session_id": "unique_session_id"
        }
    )

    return "Generated response based on context..."

# Example usage of the process_user_request function
response = process_user_request(
    user_input="How do I optimize my Python code?",
    character_name="TechHelper"
)
```
