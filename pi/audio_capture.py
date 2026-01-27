"""
Audio capture module with Voice Activity Detection (VAD).
Captures audio from microphone and detects speech segments.
"""

import asyncio
import collections
import logging
import struct
import time
from typing import AsyncGenerator, Optional, Callable

import numpy as np
import pyaudio
import webrtcvad

from . import config

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Captures audio with VAD for efficient speech detection.
    Uses webrtcvad for low-latency voice activity detection.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        chunk_size: int = config.CHUNK_SIZE,
        vad_aggressiveness: int = config.VAD_AGGRESSIVENESS,
        vad_frame_ms: int = config.VAD_FRAME_MS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.vad_frame_ms = vad_frame_ms

        # Calculate frame size in samples
        self.frame_size = int(sample_rate * vad_frame_ms / 1000)

        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # PyAudio instance
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

        # State
        self._running = False
        self._recording = False

    def _init_audio(self) -> None:
        """Initialize PyAudio."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()

    def _find_input_device(self) -> int:
        """Find the best input device."""
        self._init_audio()

        default_device = self.audio.get_default_input_device_info()
        logger.info(f"Default input device: {default_device['name']}")
        return default_device["index"]

    def start(self) -> None:
        """Start audio capture."""
        self._init_audio()

        device_index = self._find_input_device()

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.frame_size,
        )

        self._running = True
        logger.info("Audio capture started")

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None

        logger.info("Audio capture stopped")

    def read_frame(self) -> Optional[bytes]:
        """Read a single audio frame."""
        if not self.stream or not self._running:
            return None

        try:
            return self.stream.read(self.frame_size, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Error reading audio frame: {e}")
            return None

    def is_speech(self, frame: bytes) -> bool:
        """Check if audio frame contains speech."""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    async def capture_speech(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        min_speech_ms: int = config.MIN_SPEECH_MS,
        max_speech_ms: int = config.MAX_SPEECH_MS,
        silence_timeout_ms: int = config.SILENCE_TIMEOUT_MS,
        speech_pad_ms: int = config.SPEECH_PAD_MS,
    ) -> AsyncGenerator[bytes, None]:
        """
        Capture speech segments using VAD.

        Yields audio chunks when speech is detected, stopping after silence.

        Args:
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends
            min_speech_ms: Minimum speech duration to yield
            max_speech_ms: Maximum recording duration
            silence_timeout_ms: Silence duration to end recording
            speech_pad_ms: Padding before/after speech

        Yields:
            Audio data chunks (bytes)
        """
        if not self._running:
            self.start()

        # Calculate frame counts
        frames_per_ms = self.sample_rate / 1000
        pad_frames = int(speech_pad_ms / self.vad_frame_ms)
        silence_frames = int(silence_timeout_ms / self.vad_frame_ms)
        min_speech_frames = int(min_speech_ms / self.vad_frame_ms)
        max_speech_frames = int(max_speech_ms / self.vad_frame_ms)

        # Ring buffer for padding before speech
        ring_buffer = collections.deque(maxlen=pad_frames)

        # State
        triggered = False
        voiced_frames = []
        num_voiced = 0
        num_unvoiced = 0
        speech_frame_count = 0

        logger.debug("Listening for speech...")

        while self._running:
            frame = self.read_frame()
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            is_speech = self.is_speech(frame)

            if not triggered:
                # Not yet triggered - look for speech
                ring_buffer.append((frame, is_speech))
                num_voiced = sum(1 for _, speech in ring_buffer if speech)

                # Trigger if enough voiced frames
                if num_voiced > config.VAD_TRIGGER_RATIO * ring_buffer.maxlen:
                    triggered = True
                    speech_frame_count = 0

                    if on_speech_start:
                        on_speech_start()

                    logger.debug("Speech detected, recording...")

                    # Yield buffered frames (padding before speech)
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                        yield f
                        speech_frame_count += 1

                    ring_buffer.clear()
                    num_unvoiced = 0

            else:
                # Currently recording speech
                voiced_frames.append(frame)
                yield frame
                speech_frame_count += 1

                if is_speech:
                    num_unvoiced = 0
                else:
                    num_unvoiced += 1

                # Check for end of speech (silence timeout)
                if num_unvoiced > silence_frames:
                    if speech_frame_count >= min_speech_frames:
                        if on_speech_end:
                            on_speech_end()
                        logger.debug(
                            f"Speech ended after {speech_frame_count * self.vad_frame_ms}ms"
                        )
                        return
                    else:
                        # Too short, reset and continue listening
                        logger.debug("Speech too short, continuing...")
                        triggered = False
                        voiced_frames.clear()
                        num_unvoiced = 0

                # Check max duration
                if speech_frame_count >= max_speech_frames:
                    if on_speech_end:
                        on_speech_end()
                    logger.debug("Max speech duration reached")
                    return

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.001)


class ContinuousAudioCapture:
    """
    Continuous audio capture that yields complete speech segments.
    """

    def __init__(self, audio_capture: Optional[AudioCapture] = None):
        self.audio = audio_capture or AudioCapture()
        self._running = False

    async def run(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Continuously capture and yield complete speech segments.

        Args:
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends

        Yields:
            Complete speech segments as bytes
        """
        self._running = True
        self.audio.start()

        try:
            while self._running:
                # Collect a complete speech segment
                speech_chunks = []

                async for chunk in self.audio.capture_speech(
                    on_speech_start=on_speech_start,
                    on_speech_end=on_speech_end,
                ):
                    speech_chunks.append(chunk)

                if speech_chunks:
                    # Yield complete segment
                    yield b"".join(speech_chunks)

                # Brief pause before listening for next segment
                await asyncio.sleep(0.1)

        finally:
            self.audio.stop()

    def stop(self) -> None:
        """Stop continuous capture."""
        self._running = False
        self.audio.stop()
