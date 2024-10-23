import json
from typing import Dict, Any
from datetime import datetime

class Serializer:
    @staticmethod
    def to_json(data: Dict[str, Any], file_path: str):
        """Serialize data to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to serialize data to {file_path}: {str(e)}")

    @staticmethod
    def from_json(file_path: str) -> Dict[str, Any]:
        """Deserialize data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to deserialize data from {file_path}: {str(e)}")
