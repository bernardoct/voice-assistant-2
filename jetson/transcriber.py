"""
Faster-whisper transcription module for low-latency speech-to-text.
"""

import json
import logging
import re
import numpy as np
from typing import List, Optional, Tuple
from faster_whisper import WhisperModel

from . import config

logger = logging.getLogger(__name__)

# Minimum audio RMS energy to consider as speech (filters silence/noise)
MIN_AUDIO_ENERGY = 0.005

# Minimum confidence threshold for transcription
MIN_CONFIDENCE = 0.3

# Maximum repetition ratio (if a phrase repeats more than this, it's hallucination)
MAX_REPETITION_RATIO = 0.4


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

    def _check_audio_energy(self, audio: np.ndarray) -> bool:
        """
        Check if audio has sufficient energy to be speech.

        Args:
            audio: Audio samples (float32, normalized)

        Returns:
            True if audio likely contains speech
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < MIN_AUDIO_ENERGY:
            logger.debug(f"Audio energy too low: {rms:.4f} < {MIN_AUDIO_ENERGY}")
            return False
        return True

    def _is_hallucination(self, text: str) -> bool:
        """
        Detect if transcription is likely a Whisper hallucination.

        Hallucinations typically have:
        - Highly repetitive phrases
        - Generic filler text
        - Very long repetitive sequences

        Args:
            text: Transcribed text

        Returns:
            True if likely hallucination
        """
        if not text:
            return False

        text_lower = text.lower().strip()

        # Check for common hallucination patterns
        hallucination_patterns = [
            r"^(no[,.]?\s*)+$",  # "no, no, no, no..."
            r"^(yes[,.]?\s*)+$",  # "yes, yes, yes..."
            r"^(i'm going to .{5,30}\.\s*)+$",  # "I'm going to X. I'm going to X..."
            r"^(it's .{3,20}\.\s*)+$",  # "It's X. It's X..."
            r"^(i .{3,15}\.\s*)+$",  # "I X. I X..."
            r"^(and .{3,20}\.\s*)+$",  # "and X. and X..."
            r"^(the .{3,20}\.\s*)+$",  # "the X. the X..."
            r"^(.{5,30}[,.]\s*)\1{2,}",  # Any phrase repeated 3+ times
        ]

        for pattern in hallucination_patterns:
            if re.match(pattern, text_lower):
                logger.debug(f"Detected hallucination pattern: {text[:50]}...")
                return True

        # Check for high repetition ratio
        words = text_lower.split()
        if len(words) > 5:
            # Count unique words
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))

            if repetition_ratio > MAX_REPETITION_RATIO:
                logger.debug(
                    f"High repetition ratio ({repetition_ratio:.2f}): {text[:50]}..."
                )
                return True

        # Check for very long text with no command keywords
        if len(words) > 15:
            command_keywords = {
                "turn", "switch", "on", "off", "dim", "set", "toggle",
                "light", "lights", "lamp", "room", "kitchen", "bedroom",
                "living", "office", "bathroom", "brightness"
            }
            if not any(w in command_keywords for w in words):
                logger.debug(f"Long text with no command keywords: {text[:50]}...")
                return True

        return False

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

        # Check audio energy - reject silent/quiet audio
        if not self._check_audio_energy(audio):
            return "", 0.0

        # Run transcription with optimizations for speed and quality
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,  # Greedy decoding for speed
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt if self._initial_prompt else None,
            no_speech_threshold=0.6,  # Higher = more aggressive filtering of non-speech
            log_prob_threshold=-1.0,  # Filter low-confidence segments
            vad_filter=True,  # Use VAD to skip silence
            vad_parameters=dict(
                threshold=0.5,  # Higher = more aggressive VAD
                min_silence_duration_ms=500,  # Longer silence needed to split
                speech_pad_ms=200,
                min_speech_duration_ms=100,  # Minimum speech duration
            ),
        )

        # Collect segments with confidence filtering
        text_parts = []
        total_confidence = 0.0
        segment_count = 0

        for segment in segments:
            # Skip low-confidence segments
            if segment.no_speech_prob > 0.5:
                logger.debug(f"Skipping high no_speech_prob segment: {segment.text}")
                continue

            if segment.avg_logprob < -1.5:
                logger.debug(f"Skipping low confidence segment: {segment.text}")
                continue

            text_parts.append(segment.text.strip())
            if segment.avg_logprob:
                total_confidence += np.exp(segment.avg_logprob)
                segment_count += 1

        full_text = " ".join(text_parts).strip()
        avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

        # Check for hallucination
        if self._is_hallucination(full_text):
            logger.info(f"Filtered hallucination: {full_text[:80]}...")
            return "", 0.0

        # Final confidence check
        if avg_confidence < MIN_CONFIDENCE and full_text:
            logger.debug(f"Low confidence transcription filtered: {full_text}")
            return "", avg_confidence

        if full_text:
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
