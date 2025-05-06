"""
Data models for TTS API.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Voice:
    """Represents a TTS voice."""
    id: str
    name: str
    language: str
    gender: Optional[str] = None
    model: str = ""
    path: str = ""


@dataclass
class Language:
    """Represents a supported language."""
    code: str
    name: str
    models: List[str]
    default_model: str
    
    @classmethod
    def from_dict(cls, code: str, data: Dict[str, Any]):
        """Create a Language object from a dictionary."""
        return cls(
            code=code,
            name=get_language_name(code),
            models=[data["default"]] + data.get("alternatives", []),
            default_model=data["default"]
        )


@dataclass
class Model:
    """Represents a TTS model."""
    id: str
    name: str
    language: str
    path: str
    description: Optional[str] = None


# Helper function to map language codes to readable names
def get_language_name(code: str) -> str:
    """Get a readable language name from a language code."""
    language_names = {
       "aka_as": "Asante Twi"
    }
    return language_names.get(code, code)
