"""
API routes for the TTS server.
"""
from fastapi import APIRouter

from routes.tts import tts_router

# Create API router
router = APIRouter()
router.include_router(tts_router, prefix="/tts")
