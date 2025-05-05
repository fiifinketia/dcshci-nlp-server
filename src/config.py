"""
Configuration settings for the TTS server.
"""
import os
import logging

# Server settings
HOST = "0.0.0.0"
PORT = int(os.environ.get("TTS_SERVER_PORT", 8000))
DEBUG = os.environ.get("TTS_DEBUG", "False").lower() == "true"
FRONTEND_PORT = int(os.environ.get("TTS_FRONTEND_PORT", 5000))

# Configure logging
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# API Settings
API_PREFIX = "/api/v1"
API_TITLE = "TTS API"
API_DESCRIPTION = "Text-to-Speech API server supporting multiple languages and models"
API_VERSION = "1.0.0"

# CORS Settings
CORS_ORIGINS = [
    f"http://localhost:{FRONTEND_PORT}",
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://0.0.0.0:{FRONTEND_PORT}",
    "https://localhost",
    "https://127.0.0.1",
]

# TTS Engine Settings
DEFAULT_ENGINE = "coqui"
SUPPORTED_ENGINES = ["coqui"]

# Audio Configuration
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_CHANNELS = 1

# Coqui Engine Settings
COQUI_MODELS = {
    "aka_as": {
        "default": "/workspaces/dcshci-nlp-server/assets/asante-twi",
        "voices": [
            ["Kwame","/workspaces/dcshci-nlp-server/assets/voices/Kwame.wav"],
            ["Kweku", "/workspaces/dcshci-nlp-server/assets/voices/Kweku.wav"]
        ]
    }
}


# Default language and model
DEFAULT_LANGUAGE = "aka_as"
DEFAULT_MODEL = COQUI_MODELS[DEFAULT_LANGUAGE]["default"]

# Advanced TTS settings
DEFAULT_TOKENIZER = "nltk"
BUFFER_THRESHOLD_SECONDS = 0.0
MINIMUM_SENTENCE_LENGTH = 10
