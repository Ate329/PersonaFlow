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

    required_fields = {
        'user': str,
        'response': str
    }

    return all(
        field in content and isinstance(content[field], field_type)
        for field, field_type in required_fields.items()
    )

def validate_memory_config(config: Dict[str, Any]) -> bool:
    """Validate memory configuration structure and types"""
    if not isinstance(config, dict):
        return False

    required_fields = {
        'max_memories': int,
        'summary_threshold': int,
        'auto_summarize': bool
    }

    return all(
        field in config and isinstance(config[field], field_type)
        for field, field_type in required_fields.items()
    )
