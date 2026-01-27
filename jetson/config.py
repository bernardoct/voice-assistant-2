"""
Configuration for Jetson server.
"""

import os

# =============================================================================
# Server Settings
# =============================================================================
HOST = os.getenv("JETSON_HOST", "0.0.0.0")
PORT = int(os.getenv("JETSON_PORT", "8765"))

# WebSocket settings
WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
WS_PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
WS_MAX_SIZE = int(os.getenv("WS_MAX_SIZE", str(10 * 1024 * 1024)))  # 10MB default

# Minimum audio bytes to process (skip very short clips)
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "1000"))

# =============================================================================
# Whisper Model Settings
# =============================================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # Options: tiny.en, base.en, small.en
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # cuda or cpu
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8

# =============================================================================
# Audio Settings
# =============================================================================
SAMPLE_RATE = 16000
CHANNELS = 1

# =============================================================================
# Transcription Quality Settings
# =============================================================================
# Minimum RMS audio energy to consider as speech (filters silence/noise)
# Higher = more aggressive filtering, may miss quiet speech
MIN_AUDIO_ENERGY = float(os.getenv("MIN_AUDIO_ENERGY", "0.005"))

# Minimum confidence threshold for accepting transcription
# Range: 0.0-1.0, higher = stricter
MIN_TRANSCRIPTION_CONFIDENCE = float(os.getenv("MIN_TRANSCRIPTION_CONFIDENCE", "0.3"))

# =============================================================================
# Whisper Inference Settings
# =============================================================================
# No-speech probability threshold (higher = more aggressive non-speech filtering)
# Range: 0.0-1.0
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))

# Log probability threshold for filtering low-confidence segments
# More negative = more permissive
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0"))

# Per-segment no_speech_prob threshold (skip segments above this)
WHISPER_SEGMENT_NO_SPEECH_PROB = float(os.getenv("WHISPER_SEGMENT_NO_SPEECH_PROB", "0.5"))

# Per-segment avg_logprob threshold (skip segments below this)
WHISPER_SEGMENT_AVG_LOGPROB = float(os.getenv("WHISPER_SEGMENT_AVG_LOGPROB", "-1.5"))

# =============================================================================
# Whisper VAD Settings (Voice Activity Detection)
# =============================================================================
# VAD threshold (higher = more aggressive, may miss soft speech)
WHISPER_VAD_THRESHOLD = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.5"))

# Minimum silence duration to split segments (ms)
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "500"))

# Padding around detected speech (ms)
WHISPER_VAD_SPEECH_PAD_MS = int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "200"))

# Minimum speech duration to process (ms)
WHISPER_VAD_MIN_SPEECH_MS = int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "100"))

# =============================================================================
# Hallucination Detection Settings
# =============================================================================
# Maximum word repetition ratio before considering it a hallucination
# Range: 0.0-1.0 (e.g., 0.4 = 40% repeated words triggers filter)
MAX_REPETITION_RATIO = float(os.getenv("MAX_REPETITION_RATIO", "0.4"))

# Minimum words before checking repetition ratio
MIN_WORDS_FOR_REPETITION_CHECK = int(os.getenv("MIN_WORDS_FOR_REPETITION_CHECK", "5"))

# Maximum words without command keywords before filtering
MAX_WORDS_WITHOUT_KEYWORDS = int(os.getenv("MAX_WORDS_WITHOUT_KEYWORDS", "15"))

# =============================================================================
# LLM Settings (for complex intent parsing - optional)
# =============================================================================
LLM_MODEL = os.getenv("LLM_MODEL", "")  # Path to local LLM model
USE_LLM_FALLBACK = os.getenv("USE_LLM_FALLBACK", "false").lower() == "true"

