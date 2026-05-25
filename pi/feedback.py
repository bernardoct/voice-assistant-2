"""Simple synthesized audio chimes for user feedback."""

import asyncio
import logging
import math
import struct
from typing import List, Optional

from . import config

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Plays short tones through PyAudio. No external sound files needed."""

    def __init__(
        self,
        enabled: bool = config.ENABLE_FEEDBACK,
        sample_rate: int = config.FEEDBACK_SAMPLE_RATE,
        volume: float = config.FEEDBACK_VOLUME,
    ):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.volume = volume
        self._pa = None

        if self.enabled:
            try:
                import pyaudio  # noqa: F401
            except ImportError:
                logger.warning("PyAudio not installed; audio feedback disabled")
                self.enabled = False

    def cleanup(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _tone(self, freq: float, duration_s: float, fade_ms: float = 10.0) -> bytes:
        n = int(self.sample_rate * duration_s)
        fade = max(1, int(self.sample_rate * fade_ms / 1000))
        out = bytearray()
        for i in range(n):
            t = i / self.sample_rate
            value = math.sin(2 * math.pi * freq * t)
            if i < fade:
                value *= i / fade
            elif i > n - fade:
                value *= (n - i) / fade
            out += struct.pack("<h", int(value * self.volume * 32767))
        return bytes(out)

    def _chime(self, freqs: List[float], duration: float = 0.10, gap: float = 0.03) -> bytes:
        gap_bytes = b"\x00\x00" * int(self.sample_rate * gap)
        chunks: List[bytes] = []
        for f in freqs:
            chunks.append(self._tone(f, duration))
            chunks.append(gap_bytes)
        return b"".join(chunks)

    def _play(self, audio: bytes) -> None:
        if not self.enabled:
            return
        import pyaudio
        try:
            if self._pa is None:
                self._pa = pyaudio.PyAudio()
            stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True,
            )
        except Exception as e:
            # No output device (common under cron / headless): disable
            # further attempts so we don't spam logs on every chime.
            logger.warning("Audio feedback unavailable, disabling: %s", e)
            self.enabled = False
            return
        try:
            stream.write(audio)
        except Exception as e:
            logger.warning("Audio feedback playback failed: %s", e)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

    async def _play_async(self, audio: bytes) -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._play, audio)
        except Exception as e:
            logger.warning("Audio feedback thread failed: %s", e)

    async def listening(self) -> None:
        await self._play_async(self._chime([523, 784], duration=0.08, gap=0.02))

    async def success(self) -> None:
        await self._play_async(self._chime([523, 659, 784], duration=0.09, gap=0.02))

    async def error(self) -> None:
        await self._play_async(self._chime([392, 294], duration=0.14, gap=0.04))

    async def unknown(self) -> None:
        await self._play_async(self._chime([330, 392], duration=0.12, gap=0.04))
