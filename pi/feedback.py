"""
Audio feedback module for voice assistant.
Provides audio cues for user interaction.
"""

import asyncio
import logging
import math
import struct
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

# Try to import audio playback library
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available, audio feedback disabled")


class AudioFeedback:
    """
    Provides audio feedback sounds for the voice assistant.
    Generates tones programmatically to avoid file dependencies.
    """

    def __init__(
        self,
        sample_rate: int = config.FEEDBACK_SAMPLE_RATE,
        volume: float = config.FEEDBACK_VOLUME,
    ):
        self.sample_rate = sample_rate
        self.volume = volume
        self._audio: Optional["pyaudio.PyAudio"] = None
        self._enabled = config.ENABLE_AUDIO_FEEDBACK and PYAUDIO_AVAILABLE

    def _init_audio(self) -> None:
        """Initialize PyAudio for playback."""
        if not self._enabled:
            return

        if self._audio is None:
            import pyaudio
            self._audio = pyaudio.PyAudio()

    def _generate_tone(
        self,
        frequency: float,
        duration: float,
        fade_ms: float = 10,
    ) -> bytes:
        """
        Generate a sine wave tone.

        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            fade_ms: Fade in/out duration in milliseconds

        Returns:
            Raw audio bytes (16-bit mono)
        """
        num_samples = int(self.sample_rate * duration)
        fade_samples = int(self.sample_rate * fade_ms / 1000)

        samples = []
        for i in range(num_samples):
            # Generate sine wave
            t = i / self.sample_rate
            value = math.sin(2 * math.pi * frequency * t)

            # Apply fade in/out
            if i < fade_samples:
                value *= i / fade_samples
            elif i > num_samples - fade_samples:
                value *= (num_samples - i) / fade_samples

            # Apply volume and convert to 16-bit
            value = int(value * self.volume * 32767)
            samples.append(struct.pack("<h", value))

        return b"".join(samples)

    def _generate_chime(
        self,
        frequencies: list,
        duration: float = 0.1,
        gap: float = 0.05,
    ) -> bytes:
        """
        Generate a multi-tone chime.

        Args:
            frequencies: List of frequencies to play in sequence
            duration: Duration of each tone
            gap: Gap between tones

        Returns:
            Raw audio bytes
        """
        audio_data = []
        gap_samples = int(self.sample_rate * gap)
        gap_bytes = b"\x00\x00" * gap_samples

        for freq in frequencies:
            audio_data.append(self._generate_tone(freq, duration))
            audio_data.append(gap_bytes)

        return b"".join(audio_data)

    def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data."""
        if not self._enabled:
            return

        self._init_audio()

        import pyaudio

        stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
        )

        try:
            stream.write(audio_data)
        finally:
            stream.stop_stream()
            stream.close()

    async def play_listening_start(self) -> None:
        """Play sound when starting to listen."""
        if not self._enabled:
            return

        # Rising two-tone chime
        audio = self._generate_chime([440, 880], duration=0.08, gap=0.02)
        await asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio
        )

    async def play_listening_end(self) -> None:
        """Play sound when done listening."""
        if not self._enabled:
            return

        # Single low tone
        audio = self._generate_tone(440, 0.1)
        await asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio
        )

    async def play_success(self) -> None:
        """Play success sound."""
        if not self._enabled:
            return

        # Rising chime
        audio = self._generate_chime([523, 659, 784], duration=0.1, gap=0.03)
        await asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio
        )

    async def play_error(self) -> None:
        """Play error sound."""
        if not self._enabled:
            return

        # Descending two tones
        audio = self._generate_chime([440, 330], duration=0.15, gap=0.05)
        await asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio
        )

    async def play_unknown(self) -> None:
        """Play sound for unknown command."""
        if not self._enabled:
            return

        # Question-like rising tone
        audio = self._generate_chime([330, 392], duration=0.12, gap=0.04)
        await asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio
        )

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self._audio:
            self._audio.terminate()
            self._audio = None
