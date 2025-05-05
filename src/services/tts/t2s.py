from services.tts.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator
from typing import Union, Iterator, List
from services.tts.engine import BaseEngine
try:
    import pyaudio._portaudio as pa
except ImportError:
    print("Could not import the PyAudio C module 'pyaudio._portaudio'.")
    raise
import stream2sentence as s2s
import numpy as np
import threading
import traceback
import logging
import pyaudio
import queue
import time
import wave


from pydub import AudioSegment
import subprocess
import resampy
import shutil
import io


class AudioConfiguration:
    """
    Defines the configuration for an audio stream.
    """

    def __init__(
        self,
        format: int = pyaudio.paInt16,
        channels: int = 1,
        rate: int = 16000,
        output_device_index=None,
        muted: bool = False,
        frames_per_buffer: int = pa.paFramesPerBufferUnspecified,
        playout_chunk_size: int = -1,
    ):
        """
        Args:
            format (int): Audio format, typically one of PyAudio's predefined constants, e.g., pyaudio.paInt16 (default).
            channels (int): Number of audio channels, e.g., 1 for mono or 2 for stereo. Defaults to 1 (mono).
            rate (int): Sample rate of the audio stream in Hz. Defaults to 16000.
            output_device_index (int): Index of the audio output device. If None, the default output device is used.
            muted (bool): If True, audio playback is muted. Defaults to False.
            frames_per_buffer (int): Number of frames per buffer for PyAudio. Defaults to pa.paFramesPerBufferUnspecified, letting PyAudio choose.
            playout_chunk_size (int): Size of audio chunks (in bytes) to be played out. Defaults to -1, which determines the chunk size based on frames_per_buffer or a default value.

        """
        self.format = format
        self.channels = channels
        self.rate = rate
        self.output_device_index = output_device_index
        self.muted = muted
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size


class AudioStream:
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    def __init__(self, config: AudioConfiguration):
        """
        Args:
            config (AudioConfiguration): Object containing audio settings.
        """
        self.config = config
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.actual_sample_rate = 0
        self.mpv_process = None

    def get_supported_sample_rates(self, device_index):
        """
        Test which standard sample rates are supported by the specified device.
        
        Args:
            device_index (int): The index of the audio device to test
            
        Returns:
            list: List of supported sample rates
        """
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxOutputChannels')

        # Test each standard sample rate
        for rate in standard_rates:
            try:
                if self.pyaudio_instance.is_format_supported(
                    rate,
                    output_device=device_index,
                    output_channels=max_channels,
                    output_format=self.config.format,
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, device_index, desired_rate):
        """
        Determines the best available sample rate for the device.
        
        Args:
            device_index: Index of the audio device
            desired_rate: Preferred sample rate
            
        Returns:
            int: Best available sample rate
        """
        try:
            # First determine the actual device index to use
            actual_device_index = (device_index if device_index is not None 
                                else self.pyaudio_instance.get_default_output_device_info()['index'])

            # Now use the actual_device_index for getting device info and supported rates
            device_info = self.pyaudio_instance.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)

            # Check if desired rate is supported
            if desired_rate in supported_rates:
                return desired_rate

            # Find the highest supported rate that's lower than desired_rate
            lower_rates = [r for r in supported_rates if r <= desired_rate]
            if lower_rates:
                return max(lower_rates)

            # If no lower rates, get the lowest higher rate
            higher_rates = [r for r in supported_rates if r > desired_rate]
            if higher_rates:
                return min(higher_rates)

            # If nothing else works, return device's default rate
            return int(device_info.get('defaultSampleRate', 44100))

        except Exception as e:
            logging.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def is_installed(self, lib_name: str) -> bool:
        """
        Check if the given library or software is installed and accessible.

        This method uses shutil.which to determine if the given library or software is
        installed and available in the system's PATH.

        Args:
            lib_name (str): Name of the library or software to check.

        Returns:
            bool: True if the library is installed, otherwise False.
        """
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True

    def open_stream(self):
        """Opens an audio stream."""

        # check for mpeg format
        pyChannels = self.config.channels
        desired_rate = self.config.rate
        pyOutput_device_index = self.config.output_device_index

        if self.config.muted:
            logging.debug("Muted mode, no opening stream")

        else:
            if self.config.format == pyaudio.paCustomFormat and pyChannels == -1 and desired_rate == -1:
                logging.debug("Opening mpv stream for mpeg audio chunks, no need to determine sample rate")
                if not self.is_installed("mpv"):
                    message = (
                        "mpv not found, necessary to stream audio. "
                        "On mac you can install it with 'brew install mpv'. "
                        "On linux and windows you can install it from https://mpv.io/"
                    )
                    raise ValueError(message)

                mpv_command = [
                    "mpv",
                    "--no-terminal",
                    "--stream-buffer-size=4096",
                    "--demuxer-max-bytes=4096",
                    "--demuxer-max-back-bytes=4096",
                    "--ad-queue-max-bytes=4096",
                    "--cache=no",
                    "--cache-secs=0",
                    "--",
                    "fd://0"
                ]

                self.mpv_process = subprocess.Popen(
                    mpv_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            # Determine the best sample rate
            best_rate = self._get_best_sample_rate(pyOutput_device_index, desired_rate)
            self.actual_sample_rate = best_rate

            if self.config.format == pyaudio.paCustomFormat:
                pyFormat = self.pyaudio_instance.get_format_from_width(2)
                logging.debug(
                    "Opening stream for mpeg audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            else:
                pyFormat = self.config.format
                logging.debug(
                    "Opening stream for wave audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            try:
                self.stream = self.pyaudio_instance.open(
                    format=pyFormat,
                    channels=pyChannels,
                    rate=best_rate,
                    output_device_index=pyOutput_device_index,
                    frames_per_buffer=self.config.frames_per_buffer,
                    output=True,
                )
            except Exception as e:
                print(
                    "Error opening stream with parameters:"
                    f" format={pyFormat}, channels={pyChannels}, rate={best_rate}, output_device_index={pyOutput_device_index}"
                    f"Error message: {e}")

                # Get the number of available audio devices
                device_count = self.pyaudio_instance.get_device_count()

                print("Available Audio Devices:\n")

                # Iterate through all devices and print their details
                for i in range(device_count):
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    print(f"Device Index: {i}")
                    print(f"  Name: {device_info['name']}")
                    print(f"  Sample Rate (Default): {device_info['defaultSampleRate']} Hz")
                    print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                    print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
                    print(f"  Host API: {self.pyaudio_instance.get_host_api_info_by_index(device_info['hostApi'])['name']}")
                    print("\n")

                exit(0)

    def start_stream(self):
        """Starts the audio stream."""
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def close_stream(self):
        """Closes the audio stream."""
        if self.stream:
            self.stop_stream()
            self.stream.close()
            self.stream = None
        elif self.mpv_process:
            if self.mpv_process.stdin:
                self.mpv_process.stdin.close()
            self.mpv_process.wait()
            self.mpv_process.terminate()

    def is_stream_active(self) -> bool:
        """
        Checks if the audio stream is active.

        Returns:
            bool: True if the stream is active, False otherwise.
        """
        return self.stream and self.stream.is_active()


class AudioBufferManager:
    """
    Manages an audio buffer, allowing addition and retrieval of audio data.
    """

    def __init__(self, audio_buffer: queue.Queue, config: AudioConfiguration):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
        """
        self.config = config
        self.audio_buffer = audio_buffer
        self.total_samples = 0

    def add_to_buffer(self, audio_data):
        """
        Adds audio data to the buffer.

        Args:
            audio_data: Audio data to be added.
        """
        self.audio_buffer.put(audio_data)
        self.total_samples += len(audio_data) // 2

    def clear_buffer(self):
        """Clears all audio data from the buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                continue
        self.total_samples = 0

    def get_from_buffer(self, timeout: float = 0.05):
        """
        Retrieves audio data from the buffer.

        Args:
            timeout (float): Time (in seconds) to wait
              before raising a queue.Empty exception.

        Returns:
            The audio data chunk or None if the buffer is empty.
        """
        try:
            chunk = self.audio_buffer.get(timeout=timeout)
            # Map PyAudio format to bytes per sample
            format_bytes = {
                pyaudio.paCustomFormat: 4,
                pyaudio.paFloat32: 4,
                pyaudio.paInt32: 4,
                pyaudio.paInt24: 3,
                pyaudio.paInt16: 2,
                pyaudio.paInt8: 1,
                pyaudio.paUInt8: 1
            }

            # Get format and channels from config
            audio_format = self.config.format
            channels = self.config.channels

            # Log if format is unknown
            if audio_format not in format_bytes:
                print(f"Warning: Unknown audio format {audio_format} (0x{audio_format:x})")
                print(f"Available formats: {[hex(k) for k in format_bytes.keys()]}")
                format_bytes[audio_format] = 4  # Default to 4 bytes

            # Calculate bytes per frame
            bytes_per_frame = format_bytes[audio_format] * channels

            # Update total samples counter
            self.total_samples -= len(chunk) // bytes_per_frame
            return chunk
        except queue.Empty:
            return None

    def get_buffered_seconds(self, rate: int) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Args:
            rate (int): Sample rate of the audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        return self.total_samples / rate


class StreamPlayer:
    """
    Manages audio playback operations such as start, stop, pause, and resume.
    """

    def __init__(
        self,
        audio_buffer: queue.Queue,
        config: AudioConfiguration,
        on_playback_start=None,
        on_playback_stop=None,
        on_audio_chunk=None,
        muted=False,
    ):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
            config (AudioConfiguration): Object containing audio settings.
            on_playback_start (Callable, optional): Callback function to be
              called at the start of playback. Defaults to None.
            on_playback_stop (Callable, optional): Callback function to be
              called at the stop of playback. Defaults to None.
        """
        self.buffer_manager = AudioBufferManager(audio_buffer, config)
        self.audio_stream = AudioStream(config)
        self.playback_active = False
        self.immediate_stop = threading.Event()
        self.pause_event = threading.Event()
        self.playback_thread = None
        self.on_playback_start = on_playback_start
        self.on_playback_stop = on_playback_stop
        self.on_audio_chunk = on_audio_chunk
        self.first_chunk_played = False
        self.muted = muted

    def _play_chunk(self, chunk):
        """
        Plays a chunk of audio data.

        Args:
            chunk: Chunk of audio data to be played.
        """

        # handle mpeg
        if self.audio_stream.config.format == pyaudio.paCustomFormat and self.audio_stream.config.channels == -1 and self.audio_stream.config.rate == -1:
            try:
                # Pause playback if the event is set
                if not self.first_chunk_played and self.on_playback_start:
                    self.on_playback_start()
                    self.first_chunk_played = True

                if not self.muted:
                    if self.audio_stream.mpv_process and self.audio_stream.mpv_process.stdin:
                        self.audio_stream.mpv_process.stdin.write(chunk)
                        self.audio_stream.mpv_process.stdin.flush()

                if self.on_audio_chunk:
                    self.on_audio_chunk(chunk)

                import time
                while self.pause_event.is_set():
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error sending audio data to mpv: {e}")
            return

        if self.audio_stream.config.format == pyaudio.paCustomFormat:
            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            chunk = segment.raw_data
            sample_width = segment.sample_width
            channels = segment.channels
        else:
            sample_width = self.audio_stream.pyaudio_instance.get_sample_size(self.audio_stream.config.format)
            channels = self.audio_stream.config.channels

        if self.audio_stream.config.rate != self.audio_stream.actual_sample_rate and self.audio_stream.actual_sample_rate > 0:
            if self.audio_stream.config.format == pyaudio.paFloat32:
                audio_data = np.frombuffer(chunk, dtype=np.float32)
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = resampled_data.astype(np.float32).tobytes()
            else:
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = (resampled_data * 32768.0).astype(np.int16).tobytes()

        if self.audio_stream.config.playout_chunk_size > 0:
            sub_chunk_size = self.audio_stream.config.playout_chunk_size
        else:
            if self.audio_stream.config.frames_per_buffer == pa.paFramesPerBufferUnspecified:
                sub_chunk_size = 512
            else:
                sub_chunk_size = self.audio_stream.config.frames_per_buffer * sample_width * channels
        
        for i in range(0, len(chunk), sub_chunk_size):
            sub_chunk = chunk[i : i + sub_chunk_size]

            if not self.first_chunk_played and self.on_playback_start:
                self.on_playback_start()
                self.first_chunk_played = True

            if not self.muted:
                try:
                    import time

                    # Define the timeout duration in seconds
                    timeout = 0.1

                    # Record the start time
                    start_time = time.time()

                    frames_in_sub_chunk = len(sub_chunk) // (sample_width * channels)

                    # Wait until there's space in the buffer or the timeout is reached
                    while self.audio_stream.stream.get_write_available() < frames_in_sub_chunk:
                        if time.time() - start_time > timeout:
                            print(f"Wait aborted: Timeout of {timeout}s exceeded. "
                                f"Buffer availability: {self.audio_stream.stream.get_write_available()}, "
                                f"Frames in sub-chunk: {frames_in_sub_chunk}")
                            break
                        time.sleep(0.001)  # Small sleep to let the stream process audio


                    self.audio_stream.stream.write(sub_chunk)
                except Exception as e:
                    print(f"RealtimeTTS error sending audio data: {e}")

            if self.on_audio_chunk:
                self.on_audio_chunk(sub_chunk)

            # Pause playback if the event is set
            while self.pause_event.is_set():
                time.sleep(0.01)

            if self.immediate_stop.is_set():
                break

    def _process_buffer(self):
        """
        Processes and plays audio data from the buffer
        until it's empty or playback is stopped.
        """
        while self.playback_active or not self.buffer_manager.audio_buffer.empty():
            chunk = self.buffer_manager.get_from_buffer()
            if chunk:
                self._play_chunk(chunk)

            if self.immediate_stop.is_set():
                logging.info("Immediate stop requested, aborting playback")
                break

        if self.on_playback_stop:
            self.on_playback_stop()

    def get_buffered_seconds(self) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        if self.audio_stream.config.rate > 0:
            return self.buffer_manager.get_buffered_seconds(self.audio_stream.config.rate)
        else: # mpeg
            return self.buffer_manager.get_buffered_seconds(16000)
        # total_samples = sum(
        #     len(chunk) // 2 for chunk in list(self.buffer_manager.audio_buffer.queue)
        # )
        # return total_samples / self.audio_stream.config.rate

    def start(self):
        """Starts audio playback."""
        self.first_chunk_played = False
        self.playback_active = True
        if not self.audio_stream.stream:
            self.audio_stream.open_stream()

        self.audio_stream.start_stream()

        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._process_buffer)
            self.playback_thread.start()

    def stop(self, immediate: bool = False):
        """
        Stops audio playback.

        Args:
            immediate (bool): If True, stops playback immediately
              without waiting for buffer to empty.
        """
        if not self.playback_thread:
            logging.warn("No playback thread found, cannot stop playback")
            return

        if immediate:
            self.immediate_stop.set()
            while self.playback_active:
                time.sleep(0.1)
            return

        self.playback_active = False

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

        time.sleep(0.1)

        self.audio_stream.close_stream()
        self.immediate_stop.clear()
        self.buffer_manager.clear_buffer()
        self.playback_thread = None

    def pause(self):
        """Pauses audio playback."""
        self.pause_event.set()

    def resume(self):
        """Resumes paused audio playback."""
        self.pause_event.clear()

    def mute(self, muted: bool = True):
        """Mutes audio playback."""
        self.muted = muted


class TextToAudioStream:
    def __init__(
        self,
        engine: Union[BaseEngine, List[BaseEngine]],
        log_characters: bool = False,
        on_text_stream_start=None,
        on_text_stream_stop=None,
        on_audio_stream_start=None,
        on_audio_stream_stop=None,
        on_character=None,
        output_device_index=None,
        tokenizer: str = "nltk",
        language: str = "en",
        muted: bool = False,
        frames_per_buffer: int = pa.paFramesPerBufferUnspecified,
        playout_chunk_size: int = -1,
        level=logging.WARNING,
    ):
        """
        Initializes the TextToAudioStream.

        Args:
            engine (Union[BaseEngine, List[BaseEngine]]):
                The engine or list of engines used for text-to-audio synthesis.
                `BaseEngine` is the interface or base class defining how
                synthesis is performed. Multiple engines can be provided for
                fallbacks.
                
            log_characters (bool, optional):
                Whether to log each character being processed for synthesis.
                Useful for debugging or monitoring character-level processing.
                Defaults to False.
                
            on_text_stream_start (callable, optional):
                A callback function triggered when the text stream begins.
                This can be used to perform setup actions or display status
                updates.

            on_text_stream_stop (callable, optional):
                A callback function triggered when the text stream ends.
                This can be used to clean up resources or indicate that the
                text-to-speech process has completed.

            on_audio_stream_start (callable, optional):
                A callback function triggered when the audio playback begins.
                Useful for tracking the start of the audio output.

            on_audio_stream_stop (callable, optional):
                A callback function triggered when the audio playback stops.
                Useful for cleaning up or updating the UI once playback ends.

            on_character (callable, optional):
                A callback function triggered for every individual character
                processed during synthesis. This can be useful for real-time
                updates, such as visualizing which character is being processed
                or sent for synthesis.

            output_device_index (int, optional):
                The index of the audio output device to use for playback.
                If None, the system's default audio output device will be used.
                This index corresponds to the device indices returned by the
                PyAudio interface.

            tokenizer (str, optional):
                Specifies the tokenizer used to split input text into sentences
                or smaller chunks for synthesis. Supported options are:
                - "nltk": Uses the Natural Language Toolkit (NLTK) tokenizer.
                - "stanza": Uses the Stanza library for advanced sentence
                  splitting.
                Defaults to "nltk".
                
            language (str, optional):
                Language code (e.g., "en" for English, "de" for German) used for
                sentence splitting and processing. Ensure the tokenizer
                supports the specified language. Defaults to "en".

            muted (bool, optional):
                If True, disables audio playback on local speakers, allowing
                audio data to be processed or saved to a file without being
                played. This is useful for silent synthesis scenarios or batch
                processing. Defaults to False.

            frames_per_buffer (int, optional):
                Determines how many audio frames PyAudio processes in each
                buffer. If set to `pa.paFramesPerBufferUnspecified`, PyAudio
                chooses an appropriate default value. Lower values may reduce
                latency but increase CPU usage. Higher values may reduce CPU
                load but increase latency. Defaults to PyAudio’s unspecified
                setting.

            playout_chunk_size (int, optional):
                The size (in bytes) of audio chunks played to the output stream
                at a time. If set to -1, the chunk size is determined based on
                `frames_per_buffer` or a default internal value. Smaller chunks
                allow for lower latency but require more frequent processing,
                while larger chunks may introduce latency but reduce overhead.
                Defaults to -1.

            level (int, optional):
                The logging level to use for internal logging. Accepts standard
                Python logging levels, such as `logging.DEBUG`, `logging.INFO`,
                `logging.WARNING`, etc. Defaults to `logging.WARNING`.
        """
        self.log_characters = log_characters
        self.on_text_stream_start = on_text_stream_start
        self.on_text_stream_stop = on_text_stream_stop
        self.on_audio_stream_start = on_audio_stream_start
        self.on_audio_stream_stop = on_audio_stream_stop
        self.output_device_index = output_device_index
        self.output_wavfile = None
        self.chunk_callback = None
        self.wf = None
        self.abort_events = []
        self.tokenizer = tokenizer
        self.language = language
        self.global_muted = muted
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size
        self.player = None
        self.play_lock = threading.Lock()
        self.is_playing_flag = False

        self._create_iterators()

        logging.info(f"Initializing tokenizer {tokenizer} " f"for language {language}")
        s2s.init_tokenizer(tokenizer, language)

        # Initialize the play_thread attribute
        # (used for playing audio in a separate thread)
        self.play_thread = None

        # Initialize an attribute to store generated text
        self.generated_text = ""

        # A flag to indicate if the audio stream is currently running or not
        self.stream_running = False

        self.on_character = on_character

        self.engine_index = 0
        if isinstance(engine, list):
            # Handle the case where engine is a list of BaseEngine instances
            self.engines = engine
        else:
            # Handle the case where engine is a single BaseEngine instance
            self.engines = [engine]

        self.load_engine(self.engines[self.engine_index])

    def load_engine(self, engine: BaseEngine):
        """
        Loads the synthesis engine and prepares the audio player for stream playback.
        This method sets up the engine that will be used for text-to-audio conversion, extracts the necessary stream information like format, channels, and rate from the engine, and initializes the StreamPlayer if the engine does not support consuming generators directly.

        Args:
            engine (BaseEngine): The synthesis engine to be used for converting text to audio.
        """

        # Store the engine instance (responsible for text-to-audio conversion)
        self.engine = engine

        # Extract stream information (format, channels, rate) from the engine
        format, channels, rate = self.engine.get_stream_info()

        # Check if the engine doesn't support consuming generators directly
        config = AudioConfiguration(
            format,
            channels,
            rate,
            self.output_device_index,
            muted=self.global_muted,
            frames_per_buffer=self.frames_per_buffer,
            playout_chunk_size=self.playout_chunk_size,
        )

        self.player = StreamPlayer(
            self.engine.queue, config, on_playback_start=self._on_audio_stream_start
        )

        logging.info(f"loaded engine {self.engine.engine_name}")

    def feed(self, text_or_iterator: Union[str, Iterator[str]]):
        """
        Feeds text or an iterator to the stream.

        Args:
            text_or_iterator: Text or iterator to be fed.

        Returns:
            Self instance.
        """
        self.char_iter.add(text_or_iterator)
        return self

    def play_async(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = True,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "",
        tokenize_sentences=None,
        language: str = "",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。",
        force_first_fragment_after_words=30,
        debug=False,
    ):
        """
        Async handling of text to audio synthesis, see play() method.
        """
        if not self.is_playing_flag:
            self.is_playing_flag = True
            args = (
                fast_sentence_fragment,
                fast_sentence_fragment_allsentences,
                fast_sentence_fragment_allsentences_multiple,
                buffer_threshold_seconds,
                minimum_sentence_length,
                minimum_first_fragment_length,
                log_synthesized_text,
                reset_generated_text,
                output_wavfile,
                on_sentence_synthesized,
                before_sentence_synthesized,
                on_audio_chunk,
                tokenizer,
                tokenize_sentences,
                language,
                context_size,
                context_size_look_overhead,
                muted,
                sentence_fragment_delimiters,
                force_first_fragment_after_words,
                True,
                debug,
            )
            self.play_thread = threading.Thread(target=self.play, args=args)
            self.play_thread.start()
        else:
            logging.warning("play_async() called while already playing audio, skipping")

    def play(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = False,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "nltk",
        tokenize_sentences=None,
        language: str = "en",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。",
        force_first_fragment_after_words=30,
        is_external_call=True,
        debug=False,
    ):
        """
        Handles the synthesis of text to audio.
        Plays the audio stream and waits until it is finished playing.
        If the engine can't consume generators, it utilizes a player.

        Args:
        - fast_sentence_fragment: Determines if sentence fragments should be quickly yielded. Useful when a faster response is desired even if a sentence isn't complete.
        - fast_sentence_fragment_allsentences: Fast_sentence_fragment only works on the first sentence. Set this to True if you want to work it on every sentence.
        - fast_sentence_fragment_allsentences_multiple: Can yield multiple sentence fragments, not only a single one.
        - buffer_threshold_seconds (float): Time in seconds for the buffering threshold, influencing the flow and continuity of audio playback. Set to 0 to deactivate. Default is 0.
          - How it Works: The system verifies whether there is more audio content in the buffer than the duration defined by buffer_threshold_seconds. If so, it proceeds to synthesize the next sentence, capitalizing on the remaining audio to maintain smooth delivery. A higher value means more audio is pre-buffered, which minimizes pauses during playback. Adjust this upwards if you encounter interruptions.
          - Helps to decide when to generate more audio based on buffered content.
        - minimum_sentence_length (int): The minimum number of characters a sentence must have. If a sentence is shorter, it will be concatenated with the following one, improving the overall readability. This parameter does not apply to the first sentence fragment, which is governed by `minimum_first_fragment_length`. Default is 10 characters.
        - minimum_first_fragment_length (int): The minimum number of characters required for the first sentence fragment before yielding. Default is 10 characters.
        - log_synthesized_text: If True, logs the synthesized text chunks.
        - reset_generated_text: If True, resets the generated text.
        - output_wavfile: If set, saves the audio to the specified WAV file.
        - on_sentence_synthesized: Callback function that gets called after hen a single sentence fragment was synthesized.
        - before_sentence_synthesized: Callback function that gets called before a single sentence fragment gets synthesized.
        - on_audio_chunk: Callback function that gets called when a single audio chunk is ready.
        - tokenizer: Tokenizer to use for sentence splitting (currently "nltk" and "stanza" are supported).
        - tokenize_sentences (Callable): A function that tokenizes sentences from the input text. You can write your own lightweight tokenizer here if you are unhappy with nltk and stanza. Defaults to None. Takes text as string and should return splitted sentences as list of strings.
        - language: Language to use for sentence splitting.
        - context_size: The number of characters used to establish context for sentence boundary detection. A larger context improves the accuracy of detecting sentence boundaries. Default is 12 characters.
        - muted: If True, disables audio playback via local speakers (in case you want to synthesize to file or process audio chunks). Default is False.
        - sentence_fragment_delimiters (str): A string of characters that are
            considered sentence delimiters. Default is ".?!;:,\n…)]}。-".
        - force_first_fragment_after_words (int): The number of words after
            which the first sentence fragment is forced to be yielded.
            Default is 30 words.
        - is_external_call: If True, the method is called from an external source.
        - debug: If True, enables debug mode.
        """
        if self.global_muted:
            muted = True

        if is_external_call:
            if not self.play_lock.acquire(blocking=False):
                logging.warning("play() called while already playing audio, skipping")
                return

        self.is_playing_flag = True

        # Log the start of the stream
        logging.info("stream start")

        tokenizer = tokenizer if tokenizer else self.tokenizer
        language = language if language else self.language

        # Set the stream_running flag to indicate the stream is active
        self.stream_start_time = time.time()
        self.stream_running = True
        abort_event = threading.Event()
        self.abort_events.append(abort_event)

        if self.player:
            self.player.mute(muted)
        elif hasattr(self.engine, "set_muted"):
            self.engine.set_muted(muted)

        self.output_wavfile = output_wavfile
        self.chunk_callback = on_audio_chunk

        if output_wavfile:
            if self._is_engine_mpeg():
                self.wf = open(output_wavfile, "wb")
            else:
                self.wf = wave.open(output_wavfile, "wb")
                _, channels, rate = self.engine.get_stream_info()
                self.wf.setnchannels(channels)
                self.wf.setsampwidth(2)
                self.wf.setframerate(rate)

        # Initialize the generated_text variable
        if reset_generated_text:
            self.generated_text = ""

        # Check if the engine can handle generators directly
        if self.engine.can_consume_generators:
            try:
                # Start the audio player to handle playback
                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                # Directly synthesize audio using the character iterator
                self.char_iter.log_characters = self.log_characters

                self.engine.synthesize(self.char_iter)

            finally:

                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

                if is_external_call:
                    if self.on_audio_stream_stop:
                        self.on_audio_stream_stop()

                # Once done, set the stream running flag to False and log the stream stop
                logging.info("stream stop")

                # Accumulate the generated text and reset the character iterators
                self.generated_text += self.char_iter.iterated_text

                self._create_iterators()

                if is_external_call:
                    self.is_playing_flag = False
                    self.play_lock.release()
        else:
            try:
                # Start the audio player to handle playback

                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                # Generate sentences from the characters
                generate_sentences = s2s.generate_sentences(
                    self.thread_safe_char_iter,
                    context_size=context_size,
                    context_size_look_overhead=context_size_look_overhead,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    quick_yield_single_sentence_fragment=fast_sentence_fragment,
                    quick_yield_for_all_sentences=fast_sentence_fragment_allsentences,
                    quick_yield_every_fragment=fast_sentence_fragment_allsentences_multiple,
                    cleanup_text_links=True,
                    cleanup_text_emojis=True,
                    tokenize_sentences=tokenize_sentences,
                    tokenizer=tokenizer,
                    language=language,
                    log_characters=self.log_characters,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    debug=debug,
                )

                # Create the synthesis chunk generator with the given sentences
                chunk_generator = self._synthesis_chunk_generator(
                    generate_sentences, buffer_threshold_seconds, log_synthesized_text
                )

                sentence_queue = queue.Queue()

                def synthesize_worker():
                    while not abort_event.is_set():
                        sentence = sentence_queue.get()
                        if sentence is None:  # Sentinel value to stop the worker
                            break

                        synthesis_successful = False
                        if log_synthesized_text:
                            print(f"\033[96m\033[1m⚡ synthesizing\033[0m \033[37m→ \033[2m'\033[22m{sentence}\033[2m'\033[0m")

                        while not synthesis_successful:
                            try:
                                if abort_event.is_set():
                                    break

                                if before_sentence_synthesized:
                                    before_sentence_synthesized(sentence)
                                success = self.engine.synthesize(sentence)
                                if success:
                                    if on_sentence_synthesized:
                                        on_sentence_synthesized(sentence)
                                    synthesis_successful = True
                                else:
                                    logging.warning(
                                        f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}", unknown error'
                                    )

                            except Exception as e:
                                logging.warning(
                                    f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}" with error: {e}'
                                )
                                tb_str = traceback.format_exc()
                                print(f"Traceback: {tb_str}")
                                print(f"Error: {e}")

                            if not synthesis_successful:
                                if len(self.engines) == 1:
                                    time.sleep(0.2)
                                    logging.warning(
                                        f"engine {self.engine.engine_name} is the only engine available, can't switch to another engine"
                                    )
                                    break
                                else:
                                    logging.warning(
                                        "fallback engine(s) available, switching to next engine"
                                    )
                                    self.engine_index = (self.engine_index + 1) % len(
                                        self.engines
                                    )

                                    self.player.stop()
                                    self.load_engine(self.engines[self.engine_index])
                                    self.player.start()
                                    self.player.on_audio_chunk = self._on_audio_chunk

                        sentence_queue.task_done()

                worker_thread = threading.Thread(target=synthesize_worker)
                worker_thread.start()

                # Iterate through the synthesized chunks and feed them to the engine for audio synthesis
                for sentence in chunk_generator:
                    if abort_event.is_set():
                        break
                    sentence = sentence.strip()
                    if sentence:
                        sentence_queue.put(sentence)
                    else:
                        continue  # Skip empty sentences

                # Signal to the worker to stop
                sentence_queue.put(None)
                worker_thread.join()

            except Exception as e:
                logging.warning(
                    f"error in play() with engine {self.engine.engine_name}: {e}"
                )
                tb_str = traceback.format_exc()
                print(f"Traceback: {tb_str}")
                print(f"Error: {e}")

            finally:
                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

            if (len(self.char_iter.items) > 0
                and self.char_iter.iterated_text == ""
                and not self.char_iter.immediate_stop.is_set()):

                # new text was feeded while playing audio but after the last character was processed
                # we need to start another play() call (!recursively!)
                self.play(
                    fast_sentence_fragment=fast_sentence_fragment,
                    buffer_threshold_seconds=buffer_threshold_seconds,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    log_synthesized_text=log_synthesized_text,
                    reset_generated_text=False,
                    output_wavfile=output_wavfile,
                    on_sentence_synthesized=on_sentence_synthesized,
                    on_audio_chunk=on_audio_chunk,
                    tokenizer=tokenizer,
                    language=language,
                    context_size=context_size,
                    muted=muted,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    is_external_call=False,
                    debug=debug,
                )

            if is_external_call:
                if self.on_audio_stream_stop:
                    self.on_audio_stream_stop()

                self.is_playing_flag = False
                self.play_lock.release()

    def pause(self):
        """
        Pauses playback of the synthesized audio stream (won't work properly with elevenlabs).
        """
        if self.is_playing():
            logging.info("stream pause")
            self.player.pause()

    def resume(self):
        """
        Resumes a previously paused playback of the synthesized audio stream
        - won't work properly with elevenlabs
        """
        if self.is_playing():
            logging.info("stream resume")
            self.player.resume()

    def stop(self):
        """
        Stops the playback of the synthesized audio stream immediately.
        """

        for abort_event in self.abort_events:
            abort_event.set()

        if self.is_playing():
            self.char_iter.stop()
            self.player.resume()
            self.player.stop(immediate=True)
            self.stream_running = False

        if self.play_thread is not None:
            if self.play_thread.is_alive():
                self.play_thread.join()
            self.play_thread = None

        self._create_iterators()

    def text(self):
        """
        Retrieves the text that has been fed into the stream.

        Returns:
            The accumulated text.
        """
        if self.generated_text:
            return self.generated_text
        return self.thread_safe_char_iter.accumulated_text()

    def is_playing(self):
        """
        Checks if the stream is currently playing.

        Returns:
            Boolean indicating if the stream is playing.
        """
        return self.stream_running

    def _on_audio_stream_start(self):
        """
        Handles the start of the audio stream.

        This method is called when the audio stream starts. It calculates and logs the latency from the stream's start time to the time when the first chunk of audio is received. If a callback for handling the start of the audio stream is set (on_audio_stream_start), it is executed.

        No parameters or returns.
        """
        latency = time.time() - self.stream_start_time
        logging.info(f"Audio stream start, latency to first chunk: {latency:.2f}s")

        if self.on_audio_stream_start:
            self.on_audio_stream_start()

    def _on_audio_chunk(self, chunk):
        """
        Postprocessing of single chunks of audio data.
        This method is called for each chunk of audio data processed. It first determines the audio stream format.
        If the format is `pyaudio.paFloat32`, we convert to paInt16.

        Args:
            chunk (bytes): The audio data chunk to be processed.
        """
        format, channels, sample_rate = self.engine.get_stream_info()

        if format == pyaudio.paFloat32:
            audio_data = np.frombuffer(chunk, dtype=np.float32)
            audio_data = np.int16(audio_data * 32767)
            chunk = audio_data.tobytes()

        if self.output_wavfile and self.wf:
            if self._is_engine_mpeg():
                self.wf.write(chunk)
            else:
                self.wf.writeframes(chunk)

        if self.chunk_callback:
            self.chunk_callback(chunk)

    def _on_last_character(self):
        """
        This method is invoked when the last character of the text stream has been processed.
        It logs information and triggers a callback, if defined.
        """

        # If an on_text_stream_stop callback is defined, invoke it to signal the end of the text stream
        if self.on_text_stream_stop:
            self.on_text_stream_stop()

        # If log_characters flag is True, print a new line for better log readability
        if self.log_characters:
            print()

        self._create_iterators()

    def _create_iterators(self):
        """
        Creates iterators required for text-to-audio streaming.

        This method initializes two types of iterators:

        1. `CharIterator`: Responsible for managing individual characters during the streaming process.
        - It takes callbacks for events like when a character is processed (`on_character`), when the first text chunk is encountered (`on_first_text_chunk`), and when the last text chunk is encountered (`on_last_text_chunk`).

        2. `AccumulatingThreadSafeGenerator`: A thread-safe wrapper around `CharIterator`.
        - Ensures that the character iterator can be safely accessed from multiple threads.
        """

        # Create a CharIterator instance for managing individual characters
        self.char_iter = CharIterator(
            on_character=self._on_character,
            on_first_text_chunk=self.on_text_stream_start,
            on_last_text_chunk=self._on_last_character,
        )

        # Create a thread-safe version of the char iterator
        self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)

    def _on_character(self, char: str):
        """
        This method is called for each character that is processed in the text stream.
        It accumulates the characters and invokes a callback.

        Args:
            char (str): The character currently being processed.
        """
        # If an on_character callback is defined, invoke it for the current character
        if self.on_character:
            self.on_character(char)

        self.generated_text += char

    def _is_engine_mpeg(self):
        """
        Checks if the engine is an MPEG engine.

        Returns:
            Boolean indicating if the engine is an MPEG engine.
        """
        format, channel, rate = self.engine.get_stream_info()
        return format == pyaudio.paCustomFormat and channel == -1 and rate == -1

    def _synthesis_chunk_generator(
        self,
        generator: Iterator[str],
        buffer_threshold_seconds: float = 2.0,
        log_synthesis_chunks: bool = False,
    ) -> Iterator[str]:
        """
        Generates synthesis chunks based on buffered audio length.

        The function buffers chunks of synthesis until the buffered audio seconds fall below the provided threshold.
        Once the threshold is crossed, the buffered synthesis chunk is yielded.

        Args:
            generator: Input iterator that provides chunks for synthesis.
            buffer_threshold_seconds: Time in seconds to specify how long audio data should be buffered before yielding the synthesis chunk.
            log_synthesis_chunks: Boolean flag that, if set to True, logs the synthesis chunks to the logging system.

        Returns:
            Iterator of synthesis chunks.
        """

        # Initializes an empty string to accumulate chunks of synthesis
        synthesis_chunk = ""

        # Iterates over each chunk from the provided generator
        for chunk in generator:
            # Fetch the total seconds of buffered audio
            if self.player:
                buffered_audio_seconds = self.player.get_buffered_seconds()
            else:
                buffered_audio_seconds = 0

            # Append the current chunk (and a space) to the accumulated synthesis_chunk
            synthesis_chunk += chunk + " "

            # Check if the buffered audio is below the specified threshold
            if (
                buffered_audio_seconds < buffer_threshold_seconds
                or buffer_threshold_seconds <= 0
            ):
                # If the log_synthesis_chunks flag is True, log the current synthesis_chunk
                if log_synthesis_chunks:
                    logging.info(
                        f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:1f}s'
                    )

                # Yield the current synthesis_chunk and reset it for the next set of accumulations
                yield synthesis_chunk
                synthesis_chunk = ""

            else:
                logging.info(
                    f"summing up chunks because buffer {buffered_audio_seconds:.1f} > threshold ({buffer_threshold_seconds:.1f}s)"
                )

        # After iterating over all chunks, check if there's any remaining data in synthesis_chunk
        if synthesis_chunk:
            # If the log_synthesis_chunks flag is True, log the remaining synthesis_chunk
            if log_synthesis_chunks:
                logging.info(
                    f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:.1f}s'
                )

            # Yield the remaining synthesis_chunk
            yield synthesis_chunk