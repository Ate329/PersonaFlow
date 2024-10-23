from typing import Dict, Any
from string import Template
from .validators import validate_prompt_template


class PromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = Template(template)

    def format(self, **kwargs: Any) -> str:
        """Format the template with given variables"""
        return self.template.safe_substitute(**kwargs)


class PromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def add_template(self, name: str, template: str):
        """Add a new template"""
        if not validate_prompt_template(template):
            raise ValueError(f"Invalid template syntax: {template}")
        self.templates[name] = PromptTemplate(template)

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get formatted prompt from template"""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
        return self.templates[template_name].format(**kwargs)

    def to_dict(self) -> Dict[str, str]:
        """Convert templates to dictionary"""
        return {
            name: template.template.template
            for name, template in self.templates.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "PromptManager":
        """Create PromptManager from dictionary"""
        manager = cls()
        for name, template in data.items():
            manager.add_template(name, template)
        return manager
