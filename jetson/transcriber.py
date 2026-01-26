"""
Faster-whisper transcription module for low-latency speech-to-text.
"""

import json
import logging
import numpy as np
from typing import List, Optional, Tuple
from faster_whisper import WhisperModel

from . import config

logger = logging.getLogger(__name__)


class Transcriber:
    """Wrapper for faster-whisper model with optimizations for low latency."""

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        hotwords: Optional[List[str]] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self._loaded = False
        self._hotwords = hotwords or []
        self._initial_prompt = ""

    def set_hotwords(self, hotwords: List[str]) -> None:
        """
        Set hotwords to bias transcription toward specific terms.

        Args:
            hotwords: List of words/phrases to bias toward (e.g., device names)
        """
        self._hotwords = hotwords
        # Build initial prompt from hotwords
        # Whisper uses the initial_prompt to bias toward these words
        if hotwords:
            self._initial_prompt = "Smart home voice commands: " + ", ".join(hotwords) + "."
            logger.info(f"Set {len(hotwords)} hotwords for transcription bias")

    def load_hotwords_from_entities(self, entities_path: str) -> None:
        """
        Load hotwords from ha_entities.json file.
        Extracts unusual device names that Whisper might not recognize.

        Args:
            entities_path: Path to ha_entities.json file
        """
        try:
            with open(entities_path, "r") as f:
                data = json.load(f)

            hotwords = set()

            # Extract friendly names
            if "index" in data and "by_friendly_norm" in data["index"]:
                for friendly_name in data["index"]["by_friendly_norm"].keys():
                    # Add each word that might be unusual
                    words = friendly_name.split()
                    for word in words:
                        # Skip common words, keep proper nouns and unusual names
                        if len(word) > 3 and word not in {
                            "light", "lamp", "room", "floor", "living",
                            "bedroom", "kitchen", "office", "bathroom",
                            "the", "and", "for", "with"
                        }:
                            # Capitalize to hint it's a proper noun
                            hotwords.add(word.title())
                    # Also add the full name
                    hotwords.add(friendly_name.title())

            # Extract room names
            if "areas" in data:
                for area in data["areas"]:
                    name = area.get("name_norm", area.get("name", ""))
                    if name:
                        hotwords.add(name.title())

            self.set_hotwords(list(hotwords))
            logger.info(f"Loaded hotwords from entities: {sorted(hotwords)}")

        except Exception as e:
            logger.error(f"Failed to load hotwords from {entities_path}: {e}")

    def load(self) -> None:
        """Load the Whisper model into memory."""
        if self._loaded:
            return

        logger.info(
            f"Loading Whisper model: {self.model_size} on {self.device} "
            f"with {self.compute_type}"
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._loaded = True
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
    ) -> Tuple[str, float]:
        """
        Transcribe audio data to text.

        Args:
            audio: NumPy array of audio samples (float32, 16kHz, mono)
            language: Language code for transcription

        Returns:
            Tuple of (transcribed text, confidence score)
        """
        if not self._loaded:
            self.load()

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed (should be in range [-1, 1])
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Run transcription with optimizations for speed
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,  # Greedy decoding for speed
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt if self._initial_prompt else None,
            vad_filter=True,  # Use VAD to skip silence
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        # Collect all segments
        text_parts = []
        total_confidence = 0.0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text.strip())
            # Average log probability as confidence proxy
            if segment.avg_logprob:
                total_confidence += np.exp(segment.avg_logprob)
                segment_count += 1

        full_text = " ".join(text_parts).strip()
        avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

        logger.debug(f"Transcribed: '{full_text}' (confidence: {avg_confidence:.2f})")
        return full_text, avg_confidence

    def transcribe_stream(
        self,
        audio_chunks: list,
        language: str = "en",
    ) -> Tuple[str, float]:
        """
        Transcribe a stream of audio chunks.

        Args:
            audio_chunks: List of audio byte chunks
            language: Language code

        Returns:
            Tuple of (transcribed text, confidence score)
        """
        # Combine all chunks into single array
        combined = b"".join(audio_chunks)

        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

        return self.transcribe(audio, language)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
