"""
API schemas for request and response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"  # Support for future formats


class TTSRequest(BaseModel):
    """Schema for a text-to-speech request."""
    text: str = Field(..., 
                     title="Text", 
                     description="The text to synthesize to speech", 
                     min_length=1, 
                     max_length=5000)
    voice: Optional[str] = Field(None, 
                                title="Voice", 
                                description="Voice ID to use for synthesis")
    language: str = Field("en", 
                         title="Language Code", 
                         description="Language code (e.g., 'en', 'es', 'fr')")
    model: Optional[str] = Field(None, 
                                title="Model", 
                                description="TTS model to use")
    speed: Optional[float] = Field(1.0, 
                                  title="Speed", 
                                  description="Speech rate (0.5 to 2.0)", 
                                  ge=0.5, le=2.0)
    format: AudioFormat = Field(AudioFormat.WAV, 
                               title="Audio Format", 
                               description="Output audio format")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v


class VoiceInfo(BaseModel):
    """Information about a voice."""
    id: str = Field(..., title="Voice ID")
    name: str = Field(..., title="Voice Name")
    language: str = Field(..., title="Language Code")
    gender: Optional[str] = Field(None, title="Gender")
    model: str = Field(..., title="Associated Model")


class LanguageInfo(BaseModel):
    """Information about a supported language."""
    code: str = Field(..., title="Language Code")
    name: str = Field(..., title="Language Name")
    models: List[str] = Field(..., title="Available Models")
    default_model: str = Field(..., title="Default Model")


class ModelInfo(BaseModel):
    """Information about a TTS model."""
    id: str = Field(..., title="Model ID")
    name: str = Field(..., title="Model Name")
    language: str = Field(..., title="Language Code")
    description: Optional[str] = Field(None, title="Description")


class EngineInfo(BaseModel):
    """Information about a TTS engine."""
    name: str = Field(..., title="Engine Name")
    supported_languages: List[str] = Field(..., title="Supported Languages")
    default_language: str = Field(..., title="Default Language")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    status: str = Field("error", title="Status")
    message: str = Field(..., title="Error Message")
    detail: Optional[str] = Field(None, title="Detailed Error Information")
    code: Optional[int] = Field(None, title="Error Code")


class SuccessResponse(BaseModel):
    """Schema for success responses."""
    status: str = Field("success", title="Status")
    message: str = Field(..., title="Success Message")
    data: Optional[dict] = Field(None, title="Result Data")
