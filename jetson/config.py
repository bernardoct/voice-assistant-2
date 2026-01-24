"""
Configuration for Jetson server.
"""

import os

# Server settings
HOST = os.getenv("JETSON_HOST", "0.0.0.0")
PORT = int(os.getenv("JETSON_PORT", "8765"))

# Whisper settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # Options: tiny.en, base.en, small.en
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # cuda or cpu
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# LLM settings (for complex intent parsing)
LLM_MODEL = os.getenv("LLM_MODEL", "")  # Path to local LLM model (optional)
USE_LLM_FALLBACK = os.getenv("USE_LLM_FALLBACK", "false").lower() == "true"
