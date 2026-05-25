"""Wake-word detection wrapper around openwakeword."""

import logging
import time
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects a single named wake word in a stream of int16 audio frames.

    Frames must be 16 kHz mono int16. Any frame length is accepted; openwakeword
    buffers internally and produces a score after each call.
    """

    def __init__(
        self,
        model_name: str = config.WAKE_MODEL,
        threshold: float = config.WAKE_THRESHOLD,
        cooldown_ms: int = config.WAKE_COOLDOWN_MS,
        inference_framework: str = config.WAKE_INFERENCE_FRAMEWORK,
    ):
        from openwakeword.model import Model

        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_s = cooldown_ms / 1000.0

        logger.info(
            "Loading wake-word model '%s' (framework=%s, threshold=%.2f)",
            model_name, inference_framework, threshold,
        )
        # Load only the requested model to reduce memory and CPU.
        self._model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
        )
        self._available = list(self._model.models.keys())
        if model_name not in self._available:
            raise ValueError(
                f"Wake model '{model_name}' not found. Available: {self._available}"
            )

        self._last_fire = 0.0
        # Diagnostic: track peak score seen in a rolling window so the user
        # can confirm the model is actually processing audio.
        self._peak_score = 0.0
        self._peak_score_count = 0
        self._diag_every = 50  # ~50 * 80ms = 4s between diagnostic logs

    def reset(self) -> None:
        """Reset internal state -- call after capturing a command."""
        self._model.reset()
        self._last_fire = time.monotonic()

    def detect(self, frame: bytes) -> Optional[float]:
        """Feed an int16 PCM frame; return the score if the wake word fired, else None."""
        audio = np.frombuffer(frame, dtype=np.int16)

        # Cheap mic-activity check: max abs amplitude in this frame.
        amp = int(np.abs(audio).max()) if audio.size else 0

        # Honor cooldown so a long trigger doesn't fire repeatedly.
        if time.monotonic() - self._last_fire < self.cooldown_s:
            # Still feed the model so its internal buffer stays current.
            self._model.predict(audio)
            return None

        scores = self._model.predict(audio)
        score = scores.get(self.model_name, 0.0)

        # Diagnostic: periodically log the peak score + mic amplitude so the
        # user can tell whether the model is producing meaningful scores.
        if score > self._peak_score:
            self._peak_score = score
        self._peak_score_count += 1
        if self._peak_score_count >= self._diag_every:
            logger.info(
                "wake diag: peak_score=%.3f (threshold=%.2f) mic_peak=%d",
                self._peak_score, self.threshold, amp,
            )
            self._peak_score = 0.0
            self._peak_score_count = 0

        if score >= self.threshold:
            self._last_fire = time.monotonic()
            logger.info("Wake word '%s' detected (score=%.3f)", self.model_name, score)
            return float(score)
        return None
