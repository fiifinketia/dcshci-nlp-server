"""
Text-to-Speech service that handles the TTS engines and processing.
"""
import io
from multiprocessing import Process
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
        self.engines: Dict[str, Any] = {}
        self.voices: Dict[str, List[Voice]] = {}
        self.languages: Dict[str, Dict[str, Language]] = {}
        self.current_engine_name: Optional[str] = None
        self.current_engine: Optional[Any] = None
        self.stream: Optional[TextToAudioStream] = None
        
        # Threading resources
        self.tts_semaphore = threading.Semaphore(1)
        self.speaking_lock = threading.Lock()
        self.current_speaking: Dict[str, bool] = {}
        
        # Initialize engines
        self._init_engines()
        
    def _init_engines(self):
        """Initialize supported TTS engines."""
        for engine_name in config.SUPPORTED_ENGINES:
            if engine_name == "coqui":
                try:
                    p = Process(target=self._init_coqui_engine)
                    p.start()
                    p.join()
                    # self._init_coqui_engine()
                except Exception as e:
                    logging.error(f"Failed to initialize Coqui engine: {str(e)}")
        
        # Set default engine if available
        if self.engines and config.DEFAULT_ENGINE in self.engines:
            self.set_engine(config.DEFAULT_ENGINE)
        elif self.engines:
            # Use the first available engine
            first_engine = next(iter(self.engines.keys()))
            self.set_engine(first_engine)
    
    def _init_coqui_engine(self):
        """Initialize the Coqui TTS engine."""
        logging.info("Initializing Coqui TTS engine")
        
        # Initialize the engine
        coqui_engine = CoquiEngine()
        self.engines["coqui"] = coqui_engine
        
        # Get available voices
        voices = []
        for voice in coqui_engine.get_voices():
            voices.append(Voice(
                id=voice.id,
                name=voice.name,
                language=voice.language,
                gender=None,
                model=coqui_engine.model_name
            ))
        self.voices["coqui"] = voices
        
        # Initialize languages and models
        languages = {}
        models = []
        
        for lang_code, model_info in config.COQUI_MODELS.items():
            # Create Language object
            language = Language.from_dict(lang_code, model_info)
            languages[lang_code] = language
            
            # Create Model objects for each model in the language
            default_model = Model(
                id=model_info["default"],
                name=model_info["default"].split("/")[-1],
                language=lang_code,
                description=f"Default {language.name} model"
            )
            models.append(default_model)
            
            for alt_model in model_info.get("alternatives", []):
                models.append(Model(
                    id=alt_model,
                    name=alt_model.split("/")[-1],
                    language=lang_code,
                    description=f"Alternative {language.name} model"
                ))
        
        self.languages["coqui"] = languages
    
    def set_engine(self, engine_name: str) -> bool:
        """
        Set the active TTS engine.
        
        Args:
            engine_name: Name of the engine to use
            
        Returns:
            bool: True if the engine was set successfully, False otherwise
        """
        if engine_name not in self.engines:
            logging.error(f"Engine '{engine_name}' not supported")
            return False
        
        try:
            self.current_engine_name = engine_name
            self.current_engine = self.engines[engine_name]
            
            if self.stream is None:
                self.stream = TextToAudioStream(self.current_engine, muted=True)
            else:
                self.stream.load_engine(self.current_engine)
            
            # Set default voice if available
            if self.voices.get(engine_name):
                self.current_engine.set_voice(self.voices[engine_name][0].name)
            
            logging.info(f"Switched to {engine_name} engine")
            return True
        except Exception as e:
            logging.error(f"Error switching engine: {str(e)}")
            return False
    
    def set_voice(self, voice_name: str) -> bool:
        """
        Set the active voice for the current engine.
        
        Args:
            voice_name: Name of the voice to use
            
        Returns:
            bool: True if the voice was set successfully, False otherwise
        """
        if not self.current_engine:
            logging.error("No engine is currently selected")
            return False
        
        try:
            self.current_engine.set_voice(voice_name)
            logging.info(f"Voice set to {voice_name}")
            return True
        except Exception as e:
            logging.error(f"Error setting voice: {str(e)}")
            return False
    
    def set_model(self, model_id: str, language: str = None) -> bool:
        """
        Set the active model for the current engine.
        
        Args:
            model_id: ID of the model to use
            language: Language code for the model
            
        Returns:
            bool: True if the model was set successfully, False otherwise
        """
        if not self.current_engine:
            logging.error("No engine is currently selected")
            return False
        
        try:
            if self.current_engine_name == "coqui":
                self.current_engine.load_model(model_id)
                logging.info(f"Model set to {model_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error setting model: {str(e)}")
            return False
    
    def get_stream_format(self) -> Tuple[int, int, int]:
        """
        Get the audio stream format from the current engine.
        
        Returns:
            Tuple[int, int, int]: Format, channels, sample rate
        """
        if not self.current_engine:
            return pyaudio.paInt16, 1, config.DEFAULT_SAMPLE_RATE
        return self.current_engine.get_stream_info()
    
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
        if not self.current_engine or not self.stream:
            logging.error("No TTS engine available")
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
            if model and model != self.current_engine.model_name:
                self.set_model(model, language)
            
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
        """Get a list of available TTS engine names."""
        return list(self.engines.keys())
    
    def get_available_voices(self, engine_name: str = None) -> List[Voice]:
        """
        Get available voices for an engine.
        
        Args:
            engine_name: Engine name, or None for current engine
            
        Returns:
            List[Voice]: List of available voices
        """
        if engine_name is None:
            engine_name = self.current_engine_name
        
        if engine_name and engine_name in self.voices:
            return self.voices[engine_name]
        return []
    
    def get_available_languages(self, engine_name: str = None) -> List[Language]:
        """
        Get available languages for an engine.
        
        Args:
            engine_name: Engine name, or None for current engine
            
        Returns:
            List[Language]: List of available languages
        """
        if engine_name is None:
            engine_name = self.current_engine_name
        
        if engine_name and engine_name in self.languages:
            return list(self.languages[engine_name].values())
        return []
    
    def get_language_models(self, language_code: str, engine_name: str = None) -> List[str]:
        """
        Get available models for a language.
        
        Args:
            language_code: Language code
            engine_name: Engine name, or None for current engine
            
        Returns:
            List[str]: List of model IDs
        """
        if engine_name is None:
            engine_name = self.current_engine_name
        
        if (engine_name and engine_name in self.languages and 
                language_code in self.languages[engine_name]):
            return self.languages[engine_name][language_code].models
        return []
