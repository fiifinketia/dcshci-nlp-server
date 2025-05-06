"""
Text-to-Speech service that handles the TTS engines and processing.
"""
import io
import wave
import logging
import queue
import threading
from typing import Dict, List, Optional, Any, Tuple, Generator

import pyaudio
try:
    import pyaudio._portaudio as pa
except ImportError:
    logging.error("Could not import the PyAudio C module 'pyaudio._portaudio'.")
    raise
from services.tts.engine import CoquiEngine
from services.tts.t2s import TextToAudioStream

from models import Voice, Language, Model
import config

class TTSService:
    """Service for handling text-to-speech operations."""

    def __init__(self):
        """Initialize the TTS service."""
        self.models: Dict[str, Any] = {}
        self.voices: Dict[str, List[Voice]] = {}
        self.languages: Dict[str, Dict[str, Language]] = {}
        self.current_model_name: Optional[str] = None
        self.current_model: Optional[Any] = None
        self.stream: Optional[TextToAudioStream] = None

        # Threading resources
        self.tts_semaphore = threading.Semaphore(1)
        self.speaking_lock = threading.Lock()
        self.current_speaking: Dict[str, bool] = {}

        # Initialize Models - Using a safer approach
        self._init_models()

    def _init_models(self):
        """Initialize supported TTS models."""
        for lang_code, model_info in config.SUPPORTED_MODELS.items():
            try:
                # Call directly instead of using a thread
                self._init_coqui_models(lang_code, model_info)
            except Exception as e:
                logging.error(
                    f"Failed to initialize Coqui model for {lang_code}: {str(e)}"
                )

        # Set default model if available
        if self.models and config.DEFAULT_MODEL in self.models:
            self.set_model(config.DEFAULT_MODEL)
        elif self.models:
            # Use the first available model
            first_model = next(iter(self.models.keys()))
            self.set_model(first_model)

    def _init_coqui_models(self, lang_code, model_info):
        """Initialize the Coqui TTS model."""
        logging.info(f"Initializing Coqui TTS model for {lang_code}")
        languages = {}
        languages[lang_code] = model_info["language"]

        # Initialize the model without starting a new process
        # Let CoquiEngine handle its own process creation appropriately
        model = CoquiEngine(local_models_path=model_info["path"])
        self.models[lang_code] = model

        # Get available voices
        voices = []
        for voice in model_info["voices"]:
            voices.append(
                Voice(
                    id=voice[0],
                    name=voice[0],
                    language=model_info["language"],
                    gender=None,
                    path=voice[0],
                    model=lang_code,
                )
            )
        self.voices[lang_code] = voices

        self.languages[lang_code] = languages

    def set_model(self, model_name: str) -> bool:
        """
        Set the active TTS Model.

        Args:
            model_name: Name of the Model to use

        Returns:
            bool: True if the Model was set successfully, False otherwise
        """
        if model_name not in self.models:
            logging.error(f"Model '{model_name}' not supported")
            return False

        try:
            self.current_model_name = model_name
            self.current_model = self.models[model_name]

            if self.stream is None:
                self.stream = TextToAudioStream(self.current_model, muted=True)
            else:
                self.stream.load_engine(self.current_model)

            # Set default voice if available
            if self.voices.get(model_name):
                self.current_model.set_voice(self.voices[model_name][0].name)

            logging.info(f"Switched to {model_name} model")
            return True
        except Exception as e:
            logging.error(f"Error switching model: {str(e)}")
            return False

    def set_voice(self, voice_name: str) -> bool:
        """
        Set the active voice for the current model.

        Args:
            voice_name: Name of the voice to use

        Returns:
            bool: True if the voice was set successfully, False otherwise
        """
        if not self.current_model:
            logging.error("No Model is currently selected")
            return False

        try:
            self.current_model.set_voice(voice_name)
            logging.info(f"Voice set to {voice_name}")
            return True
        except Exception as e:
            logging.error(f"Error setting voice: {str(e)}")
            return False

    def get_stream_format(self) -> Tuple[int, int, int]:
        """
        Get the audio stream format from the current model.

        Returns:
            Tuple[int, int, int]: Format, channels, sample rate
        """
        if not self.current_model:
            return pyaudio.paInt16, 1, config.DEFAULT_SAMPLE_RATE
        return self.current_model.get_stream_info()

    def synthesize_text(
        self, 
        text: str, 
        *,
        language: str = None,
        model: str = None,
        voice: str = None,
        speed: float = 1.0
    ) -> Generator[bytes, None, None]:
        """
        Synthesize text to speech and yield audio chunks.
        
        Args:
            text: Text to synthesize
            language: Language code
            model: Model ID
            voice: Voice name
            speed: Speech rate (0.5 to 2.0)
            
        Yields:
            Audio data chunks
        """
        if not self.current_model or not self.stream:
            logging.error("No TTS model available")
            return

        # Check if we can acquire the semaphore
        if not self.tts_semaphore.acquire(blocking=False):
            logging.warning("TTS service is busy, request queued")
            self.tts_semaphore.acquire()  # Wait for the semaphore

        try:
            # Set speaking status
            self._set_speaking(text, True)

            # Set the requested voice if provided
            if voice:
                self.set_voice(voice)

            # Set the requested model if provided
            if model and model != self.current_model_name:
                self.set_model(model)

            # Create a queue for audio chunks
            audio_queue = queue.Queue()

            # Callback for audio chunks
            def on_audio_chunk(chunk):
                audio_queue.put(chunk)

            # Start text-to-speech synthesis in a separate thread
            def synthesize_thread():
                try:
                    self.stream.feed(text)
                    logging.info(f"Synthesizing: {text[:50]}...")
                    self.stream.play(
                        on_audio_chunk=on_audio_chunk, 
                        muted=True,
                        buffer_threshold_seconds=config.BUFFER_THRESHOLD_SECONDS,
                        minimum_sentence_length=config.MINIMUM_SENTENCE_LENGTH
                    )
                    # Signal end of stream
                    audio_queue.put(None)
                except Exception as e:
                    logging.error(f"Error during synthesis: {str(e)}")
                    audio_queue.put(None)
                finally:
                    self._set_speaking(text, False)
                    self.tts_semaphore.release()

            # Start synthesis thread
            synthesis_thread = threading.Thread(target=synthesize_thread, daemon=True)
            synthesis_thread.start()

            # Generate wave header for proper audio playback in browsers
            header_sent = False

            # Yield audio chunks as they become available
            while True:
                try:
                    chunk = audio_queue.get(timeout=10.0)  # 10-second timeout

                    if chunk is None:
                        logging.debug("End of audio stream")
                        break

                    if not header_sent:
                        # Send wave header for browsers
                        yield self._create_wave_header()
                        header_sent = True

                    yield chunk

                except queue.Empty:
                    logging.error("Timeout waiting for audio chunks")
                    break
                except Exception as e:
                    logging.error(f"Error in audio streaming: {str(e)}")
                    break

        except Exception as e:
            logging.error(f"Error in synthesize_text: {str(e)}")
            # Make sure to release the semaphore in case of error
            self._set_speaking(text, False)
            self.tts_semaphore.release()

    def _create_wave_header(self) -> bytes:
        """
        Create a WAV header for the current audio format.
        
        Returns:
            bytes: WAV header
        """
        _, channels, sample_rate = self.get_stream_format()

        # Create a WAV header in memory
        wav_header = io.BytesIO()
        with wave.open(wav_header, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)

        wav_header.seek(0)
        header_bytes = wav_header.read()
        wav_header.close()

        return header_bytes

    def _set_speaking(self, text: str, status: bool):
        """
        Set the speaking status for a text.
        
        Args:
            text: The text being spoken
            status: True if speaking, False if not
        """
        with self.speaking_lock:
            self.current_speaking[text] = status

    def is_speaking(self, text: str = None) -> bool:
        """
        Check if the service is currently speaking.
        
        Args:
            text: Specific text to check (optional)
            
        Returns:
            bool: True if speaking, False if not
        """
        with self.speaking_lock:
            if text:
                return self.current_speaking.get(text, False)
            return any(self.current_speaking.values())

    def get_available_engines(self) -> List[str]:
        """Get a list of available TTS Model names."""
        return list(self.models.keys())

    def get_available_voices(self, model_name: str = None) -> List[Voice]:
        """
        Get available voices for an Model.

        Args:
            model_name: Model name, or None for current Model

        Returns:
            List[Voice]: List of available voices
        """
        if model_name is None:
            model_name = self.current_model_name

        if model_name and model_name in self.voices:
            return self.voices[model_name]
        return []

    def get_available_languages(self, model_name: str = None) -> List[Language]:
        """
        Get available languages for an Model.

        Args:
            model_name: Model name, or None for current Model

        Returns:
            List[Language]: List of available languages
        """
        if model_name is None:
            model_name = self.current_model_name

        if model_name and model_name in self.languages:
            return list(self.languages[model_name].values())
        return []

    def get_language_models(
        self, language_code: str, model_name: str = None
    ) -> List[str]:
        """
        Get available models for a language.

        Args:
            language_code: Language code
            model_name: Model name, or None for current Model

        Returns:
            List[str]: List of model IDs
        """
        if model_name is None:
            model_name = self.current_model_name

        if (
            model_name
            and model_name in self.languages
            and language_code in self.languages[model_name]
        ):
            return self.languages[model_name][language_code].models
        return []
