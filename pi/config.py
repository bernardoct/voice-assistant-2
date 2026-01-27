"""
Configuration for Raspberry Pi voice assistant.
"""

import os

# =============================================================================
# Jetson Server Settings
# =============================================================================
JETSON_HOST = os.getenv("JETSON_HOST", "192.168.1.117")
JETSON_PORT = int(os.getenv("JETSON_PORT", "8765"))
JETSON_URL = f"ws://{JETSON_HOST}:{JETSON_PORT}"

# =============================================================================
# Home Assistant Settings
# =============================================================================
HA_URL = os.getenv("HA_URL", "http://192.168.1.203:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")  # Long-lived access token
HA_ENTITIES_PATH = os.getenv("HA_ENTITIES_PATH", "/home/bernardoct/.cache/ha_entities.json")

# =============================================================================
# Audio Settings
# =============================================================================
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))  # Samples per chunk (32ms at 16kHz)
AUDIO_FORMAT = "int16"

# =============================================================================
# VAD (Voice Activity Detection) Settings
# =============================================================================
# VAD aggressiveness: 0-3, higher = more aggressive filtering (3 = most aggressive)
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3"))
# Frame duration in ms (valid values: 10, 20, or 30)
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))
# Padding around detected speech (ms)
SPEECH_PAD_MS = int(os.getenv("SPEECH_PAD_MS", "200"))
# Minimum speech duration to process (ms)
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "300"))
# Maximum recording duration (ms) - prevents hallucinations from long recordings
MAX_SPEECH_MS = int(os.getenv("MAX_SPEECH_MS", "5000"))
# Silence duration to end recording (ms)
SILENCE_TIMEOUT_MS = int(os.getenv("SILENCE_TIMEOUT_MS", "800"))
# VAD trigger threshold ratio (0.0-1.0) - fraction of ring buffer that must be voiced
VAD_TRIGGER_RATIO = float(os.getenv("VAD_TRIGGER_RATIO", "0.8"))

# =============================================================================
# WebSocket Client Settings
# =============================================================================
# Initial reconnect delay (seconds)
WS_RECONNECT_DELAY = float(os.getenv("WS_RECONNECT_DELAY", "1.0"))
# Maximum reconnect delay (seconds)
WS_MAX_RECONNECT_DELAY = float(os.getenv("WS_MAX_RECONNECT_DELAY", "30.0"))
# WebSocket receive timeout (seconds)
WS_RECEIVE_TIMEOUT = float(os.getenv("WS_RECEIVE_TIMEOUT", "30.0"))
# Audio send chunk size (bytes) - ~1 second at 16kHz
WS_AUDIO_CHUNK_SIZE = int(os.getenv("WS_AUDIO_CHUNK_SIZE", "32000"))

# =============================================================================
# Wake Word Settings (Optional)
# =============================================================================
WAKE_WORD_ENABLED = os.getenv("WAKE_WORD_ENABLED", "false").lower() == "true"
WAKE_WORDS = ["hey assistant", "ok assistant", "assistant"]

# =============================================================================
# Audio Feedback Settings
# =============================================================================
ENABLE_AUDIO_FEEDBACK = os.getenv("ENABLE_AUDIO_FEEDBACK", "true").lower() == "true"
FEEDBACK_VOLUME = float(os.getenv("FEEDBACK_VOLUME", "0.5"))
# Feedback audio sample rate (for generated tones)
FEEDBACK_SAMPLE_RATE = int(os.getenv("FEEDBACK_SAMPLE_RATE", "44100"))
