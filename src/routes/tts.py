"""
API routes for the TTS server.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from schemas import (
    TTSRequest, 
    VoiceInfo, 
    LanguageInfo, 
    EngineInfo,
    ErrorResponse, 
    SuccessResponse
)
from services import TTSService

# Service instance
tts_service = TTSService()


# Dependency to get the TTS service
def get_tts_service():
    """Get the TTS service instance."""
    return tts_service

tts_router = APIRouter()

@tts_router.get(
    "/engines", 
    response_model=List[str],
    summary="Get available TTS engines",
    description="Returns a list of available text-to-speech engines",
    responses={
        200: {"description": "List of available engines"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def get_engines(tts: TTSService = Depends(get_tts_service)):
    """Get available TTS engines."""
    try:
        return tts.get_available_engines()
    except Exception as e:
        logging.error(f"Error getting engines: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={"status": "error", "message": "Failed to get engines", "detail": str(e)}
        )


@tts_router.get(
    "/engines/{engine_name}",
    response_model=EngineInfo,
    summary="Get engine information",
    description="Returns information about a specific TTS engine",
    responses={
        200: {"description": "Engine information"},
        404: {"model": ErrorResponse, "description": "Engine not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def get_engine_info(
    engine_name: str,
    tts: TTSService = Depends(get_tts_service)
):
    """Get information about a specific engine."""
    try:
        if engine_name not in tts.engines:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Engine '{engine_name}' not found"}
            )
        
        languages = tts.get_available_languages(engine_name)
        return {
            "name": engine_name,
            "supported_languages": [lang.code for lang in languages],
            "default_language": "en"  # Default language is usually English
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting engine info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to get engine info", "detail": str(e)}
        )


@tts_router.post(
    "/engines/set",
    response_model=SuccessResponse,
    summary="Set active TTS engine",
    description="Sets the active text-to-speech engine",
    responses={
        200: {"description": "Engine set successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Engine not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def set_engine(
    engine_name: str = Query(..., description="Name of the engine to use"),
    tts: TTSService = Depends(get_tts_service)
):
    """Set the active TTS engine."""
    try:
        if engine_name not in tts.engines:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Engine '{engine_name}' not found"}
            )
        
        success = tts.set_engine(engine_name)
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"status": "error", "message": "Failed to set engine"}
            )
        
        return {
            "status": "success",
            "message": f"Switched to {engine_name} engine"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error setting engine: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to set engine", "detail": str(e)}
        )


@tts_router.get(
    "/voices",
    response_model=List[VoiceInfo],
    summary="Get available voices",
    description="Returns a list of available voices for the current engine or specified engine",
    responses={
        200: {"description": "List of available voices"},
        404: {"model": ErrorResponse, "description": "Engine not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def get_voices(
    engine_name: Optional[str] = Query(None, description="Name of the engine (optional)"),
    tts: TTSService = Depends(get_tts_service)
):
    """Get available voices for the current engine or specified engine."""
    try:
        if engine_name and engine_name not in tts.engines:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Engine '{engine_name}' not found"}
            )
        
        voices = tts.get_available_voices(engine_name)
        
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "language": voice.language,
                "gender": voice.gender,
                "model": voice.model
            }
            for voice in voices
        ]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to get voices", "detail": str(e)}
        )


@tts_router.post(
    "/voices/set",
    response_model=SuccessResponse,
    summary="Set active voice",
    description="Sets the active voice for the current engine",
    responses={
        200: {"description": "Voice set successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Voice not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def set_voice(
    voice_name: str = Query(..., description="Name of the voice to use"),
    tts: TTSService = Depends(get_tts_service)
):
    """Set the active voice for the current engine."""
    try:
        if not tts.current_engine:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "No engine is currently selected"}
            )
        
        success = tts.set_voice(voice_name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Voice '{voice_name}' not found or not compatible"}
            )
        
        return {
            "status": "success",
            "message": f"Voice set to {voice_name} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error setting voice: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to set voice", "detail": str(e)}
        )


@tts_router.get(
    "/languages",
    response_model=List[LanguageInfo],
    summary="Get available languages",
    description="Returns a list of available languages for the current engine or specified engine",
    responses={
        200: {"description": "List of available languages"},
        404: {"model": ErrorResponse, "description": "Engine not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def get_languages(
    engine_name: Optional[str] = Query(None, description="Name of the engine (optional)"),
    tts: TTSService = Depends(get_tts_service)
):
    """Get available languages for the current engine or specified engine."""
    try:
        if engine_name and engine_name not in tts.engines:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Engine '{engine_name}' not found"}
            )
        
        languages = tts.get_available_languages(engine_name)
        
        return [
            {
                "code": lang.code,
                "name": lang.name,
                "models": lang.models,
                "default_model": lang.default_model
            }
            for lang in languages
        ]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting languages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to get languages", "detail": str(e)}
        )


@tts_router.get(
    "/models",
    response_model=List[str],
    summary="Get available models",
    description="Returns a list of available models for the specified language and engine",
    responses={
        200: {"description": "List of available models"},
        404: {"model": ErrorResponse, "description": "Language or engine not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def get_models(
    language: str = Query(..., description="Language code"),
    engine_name: Optional[str] = Query(None, description="Name of the engine (optional)"),
    tts: TTSService = Depends(get_tts_service)
):
    """Get available models for the specified language and engine."""
    try:
        if engine_name and engine_name not in tts.engines:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Engine '{engine_name}' not found"}
            )
        
        models = tts.get_language_models(language, engine_name)
        
        if not models:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"No models found for language '{language}'"}
            )
        
        return models
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to get models", "detail": str(e)}
        )


@tts_router.post(
    "/models/set",
    response_model=SuccessResponse,
    summary="Set active model",
    description="Sets the active model for the current engine",
    responses={
        200: {"description": "Model set successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def set_model(
    model_id: str = Query(..., description="ID of the model to use"),
    language: Optional[str] = Query(None, description="Language code for the model (optional)"),
    tts: TTSService = Depends(get_tts_service)
):
    """Set the active model for the current engine."""
    try:
        if not tts.current_engine:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "No engine is currently selected"}
            )
        
        success = tts.set_model(model_id, language)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": f"Model '{model_id}' not found or not compatible"}
            )
        
        return {
            "status": "success",
            "message": f"Model set to {model_id} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error setting model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to set model", "detail": str(e)}
        )


@tts_router.post(
    "/synthesize/aka_as",
    summary="Synthesize text to speech",
    description="Synthesizes text to speech and returns audio data",
    responses={
        200: {"description": "Audio data", "content": {"audio/wav": {}}},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    }
)
def synthesize_text(
    request: TTSRequest,
    tts: TTSService = Depends(get_tts_service)
):
    """Synthesize text to speech and return audio data."""
    try:
        if not tts.current_engine:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "No TTS engine available"}
            )
        
        if tts.is_speaking() and not tts.tts_semaphore.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error", 
                    "message": "Service unavailable, currently processing another request",
                    "detail": "Please try again shortly"
                },
                headers={"Retry-After": "5"}
            )
            
        # Release the semaphore immediately as we'll acquire it again in synthesize_text
        if tts.tts_semaphore._value == 0:
            tts.tts_semaphore.release()
        
        # Stream the audio response
        return StreamingResponse(
            tts.synthesize_text(
                request.text,
                language=request.language,
                model=request.model,
                voice=request.voice,
                speed=request.speed
            ),
            media_type="audio/wav" if request.format == "wav" else "audio/mpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error synthesizing text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to synthesize text", "detail": str(e)}
        )


@tts_router.get(
    "/health",
    response_model=SuccessResponse,
    summary="Health check",
    description="Check if the TTS service is healthy and ready to process requests",
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorResponse, "description": "Service unhealthy"}
    }
)
def health_check(tts: TTSService = Depends(get_tts_service)):
    """Check if the TTS service is healthy."""
    try:
        # Check if at least one engine is initialized
        if not tts.engines:
            raise HTTPException(
                status_code=503,
                detail={"status": "error", "message": "No TTS engines available"}
            )
        
        return {
            "status": "success",
            "message": "TTS service is healthy",
            "data": {
                "engines": tts.get_available_engines(),
                "current_engine": tts.current_engine_name,
                "speaking": tts.is_speaking()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "message": "Service unhealthy", "detail": str(e)}
        )
