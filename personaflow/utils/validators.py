from typing import Dict, Any, Optional
from string import Template
import json

def validate_prompt_template(template: str) -> bool:
    """Validate prompt template syntax"""
    try:
        Template(template)
        return True
    except ValueError:
        return False

def validate_memory_content(content: Dict[str, Any]) -> bool:
    """Validate memory content structure and types"""
    if not isinstance(content, dict):
        return False
    
    # Memory content just needs to be a non-empty dictionary
    # Different memory types can have different structures
    return len(content) > 0

def validate_memory_config(config: Dict[str, Any]) -> bool:
    """Validate memory configuration structure and types"""
    if not isinstance(config, dict):
        return False

    # Define valid fields and their types
    valid_fields = {
        'max_memories': int,
        'summary_threshold': int,
        'auto_summarize': bool
    }

    # Check that all provided fields are valid and have correct types
    for field, value in config.items():
        if field not in valid_fields:
            return False
        if not isinstance(value, valid_fields[field]):
            return False
    
    return True
