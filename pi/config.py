"""Configuration for the Raspberry Pi voice assistant.

Loads values from environment variables, with a .env file in the project root
auto-loaded when present so the same code works under systemd and during
interactive development.
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Populate os.environ from a .env in the project root if not already set."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


# Jetson WebSocket server
JETSON_HOST = os.getenv("JETSON_HOST", "192.168.1.117")
JETSON_PORT = int(os.getenv("JETSON_PORT", "8765"))
JETSON_URL = f"ws://{JETSON_HOST}:{JETSON_PORT}"

# Home Assistant
HA_URL = os.getenv("HA_URL", "http://192.168.1.203:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
HA_ENTITIES_PATH = os.getenv(
    "HA_ENTITIES_PATH",
    str(Path(__file__).resolve().parent.parent / "ha_entities.json"),
)

# Audio capture
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1
# 80 ms frames -- the chunk size openwakeword is tuned for.
FRAME_SAMPLES = int(os.getenv("FRAME_SAMPLES", "1280"))
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX")  # None -> system default
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX else None

# Wake word (openwakeword)
WAKE_MODEL = os.getenv("WAKE_MODEL", "hey_jarvis")  # alexa, hey_jarvis, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
# After firing, ignore further wake events for this many ms (avoid double-trigger).
WAKE_COOLDOWN_MS = int(os.getenv("WAKE_COOLDOWN_MS", "1500"))
# openwakeword inference backend: onnx works without tflite_runtime.
WAKE_INFERENCE_FRAMEWORK = os.getenv("WAKE_INFERENCE_FRAMEWORK", "onnx")

# Voice activity detection during command capture
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0-3
VAD_SUBFRAME_MS = int(os.getenv("VAD_SUBFRAME_MS", "20"))  # 10, 20, 30
MIN_COMMAND_MS = int(os.getenv("MIN_COMMAND_MS", "300"))
MAX_COMMAND_MS = int(os.getenv("MAX_COMMAND_MS", "6000"))
SILENCE_END_MS = int(os.getenv("SILENCE_END_MS", "700"))
# Maximum time to wait for the first speech frame after the wake word.
PRE_SPEECH_TIMEOUT_MS = int(os.getenv("PRE_SPEECH_TIMEOUT_MS", "2500"))

# WebSocket client
WS_RECONNECT_DELAY = float(os.getenv("WS_RECONNECT_DELAY", "1.0"))
WS_MAX_RECONNECT_DELAY = float(os.getenv("WS_MAX_RECONNECT_DELAY", "30.0"))
WS_RECEIVE_TIMEOUT = float(os.getenv("WS_RECEIVE_TIMEOUT", "15.0"))

# Audio feedback
ENABLE_FEEDBACK = os.getenv("ENABLE_FEEDBACK", "true").lower() == "true"
FEEDBACK_VOLUME = float(os.getenv("FEEDBACK_VOLUME", "0.4"))
FEEDBACK_SAMPLE_RATE = int(os.getenv("FEEDBACK_SAMPLE_RATE", "44100"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
