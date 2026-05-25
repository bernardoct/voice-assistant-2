"""Microphone capture and post-wake VAD utterance segmentation."""

import asyncio
import logging
from typing import AsyncGenerator, Optional

import pyaudio
import webrtcvad

from . import config

logger = logging.getLogger(__name__)


class AudioStream:
    """Thin async wrapper around a blocking PyAudio input stream.

    Reads fixed-size int16 PCM frames suitable for both openwakeword (80 ms
    chunks) and webrtcvad (10/20/30 ms sub-frames carved out of each chunk).
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        frame_samples: int = config.FRAME_SAMPLES,
        device_index: Optional[int] = config.INPUT_DEVICE_INDEX,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_samples = frame_samples
        self.device_index = device_index

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        if self.device_index is None:
            info = self._pa.get_default_input_device_info()
            logger.info("Using default input device: %s", info["name"])
        else:
            info = self._pa.get_device_info_by_index(self.device_index)
            logger.info("Using input device #%d: %s", self.device_index, info["name"])

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frame_samples,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    async def read_frame(self) -> bytes:
        """Read one frame_samples-sized chunk of int16 PCM as raw bytes."""
        if self._stream is None:
            raise RuntimeError("AudioStream.start() not called")
        # PyAudio's read is blocking; offload so we don't stall the event loop.
        return await asyncio.to_thread(
            self._stream.read, self.frame_samples, exception_on_overflow=False
        )


class UtteranceRecorder:
    """Collects audio after wake-up, using VAD to find the trailing silence.

    Designed to be driven externally: the main loop hands it the same 80 ms
    audio frames it's already reading. The recorder splits each frame into
    20 ms sub-frames for webrtcvad, tracks voiced/unvoiced state, and reports
    when the utterance is complete.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        aggressiveness: int = config.VAD_AGGRESSIVENESS,
        subframe_ms: int = config.VAD_SUBFRAME_MS,
        min_command_ms: int = config.MIN_COMMAND_MS,
        max_command_ms: int = config.MAX_COMMAND_MS,
        silence_end_ms: int = config.SILENCE_END_MS,
        pre_speech_timeout_ms: int = config.PRE_SPEECH_TIMEOUT_MS,
    ):
        self.sample_rate = sample_rate
        self.subframe_bytes = int(sample_rate * subframe_ms / 1000) * 2  # int16
        self.subframe_ms = subframe_ms
        self.min_command_ms = min_command_ms
        self.max_command_ms = max_command_ms
        self.silence_end_ms = silence_end_ms
        self.pre_speech_timeout_ms = pre_speech_timeout_ms

        self._vad = webrtcvad.Vad(aggressiveness)
        self.reset()

    def reset(self) -> None:
        self._buffer = bytearray()
        self._elapsed_ms = 0
        self._silence_ms = 0
        self._voiced_ms = 0
        self._got_speech = False
        self._done = False
        self._timed_out = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def timed_out(self) -> bool:
        """True if the utterance ended without ever detecting speech."""
        return self._timed_out

    def get_audio(self) -> bytes:
        return bytes(self._buffer)

    def feed(self, frame: bytes) -> None:
        """Feed one audio frame and update state. Frame is raw int16 PCM bytes."""
        if self._done:
            return

        self._buffer.extend(frame)
        frame_ms = (len(frame) // 2) * 1000 // self.sample_rate
        self._elapsed_ms += frame_ms

        # Slice into VAD-sized sub-frames; drop any trailing remainder.
        for offset in range(0, len(frame) - self.subframe_bytes + 1, self.subframe_bytes):
            sub = bytes(frame[offset:offset + self.subframe_bytes])
            try:
                voiced = self._vad.is_speech(sub, self.sample_rate)
            except Exception:
                voiced = False
            if voiced:
                self._voiced_ms += self.subframe_ms
                self._silence_ms = 0
                self._got_speech = True
            else:
                self._silence_ms += self.subframe_ms

        # End conditions.
        if not self._got_speech:
            if self._elapsed_ms >= self.pre_speech_timeout_ms:
                logger.info("No speech detected within %d ms of wake word", self.pre_speech_timeout_ms)
                self._timed_out = True
                self._done = True
            return

        if (
            self._silence_ms >= self.silence_end_ms
            and self._voiced_ms >= self.min_command_ms
        ):
            logger.debug(
                "Utterance complete: %d ms total, %d ms voiced", self._elapsed_ms, self._voiced_ms
            )
            self._done = True
            return

        if self._elapsed_ms >= self.max_command_ms:
            logger.info("Max utterance length (%d ms) reached", self.max_command_ms)
            self._done = True
