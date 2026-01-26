"""
Configuration for Raspberry Pi voice assistant.
"""

import os

# Jetson server settings
JETSON_HOST = os.getenv("JETSON_HOST", "192.168.1.117")
JETSON_PORT = int(os.getenv("JETSON_PORT", "8765"))
JETSON_URL = f"ws://{JETSON_HOST}:{JETSON_PORT}"

# Home Assistant settings
HA_URL = os.getenv("HA_URL", "http://192.168.1.203:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")  # Long-lived access token
HA_ENTITIES_PATH = os.getenv("HA_ENTITIES_PATH", "/home/bernardoct/.cache/ha_entities.json")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512  # Samples per chunk (32ms at 16kHz)
AUDIO_FORMAT = "int16"

# VAD settings
VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive filtering (3 = most aggressive)
VAD_FRAME_MS = 30  # Frame duration in ms (10, 20, or 30)
SPEECH_PAD_MS = 200  # Padding around speech
MIN_SPEECH_MS = 300  # Minimum speech duration to process
MAX_SPEECH_MS = 5000  # Maximum recording duration (5 seconds) - prevents hallucinations
SILENCE_TIMEOUT_MS = 800  # Silence duration to end recording

# Wake word settings (optional)
WAKE_WORD_ENABLED = os.getenv("WAKE_WORD_ENABLED", "false").lower() == "true"
WAKE_WORDS = ["hey assistant", "ok assistant", "assistant"]

# Feedback settings
ENABLE_AUDIO_FEEDBACK = os.getenv("ENABLE_AUDIO_FEEDBACK", "true").lower() == "true"
FEEDBACK_VOLUME = float(os.getenv("FEEDBACK_VOLUME", "0.5"))
