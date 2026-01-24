"""
Shared protocol definitions for voice assistant communication.
Uses msgpack for efficient binary serialization over WebSocket.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import json


class MessageType(str, Enum):
    # Client -> Server
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_END = "audio_end"
    CANCEL = "cancel"

    # Server -> Client
    TRANSCRIPTION = "transcription"
    PARTIAL_TRANSCRIPTION = "partial_transcription"
    INTENT = "intent"
    ERROR = "error"
    READY = "ready"


class IntentType(str, Enum):
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    SET_BRIGHTNESS = "set_brightness"
    TOGGLE = "toggle"
    UNKNOWN = "unknown"


@dataclass
class AudioChunkMessage:
    """Audio data chunk sent from Pi to Jetson."""
    type: str = MessageType.AUDIO_CHUNK
    data: bytes = b""  # Raw PCM audio data (16kHz, 16-bit, mono)
    sample_rate: int = 16000

    def to_bytes(self) -> bytes:
        """Serialize for WebSocket binary transmission."""
        # Simple format: 1 byte type + 4 bytes sample_rate + audio data
        return b'\x01' + self.sample_rate.to_bytes(4, 'little') + self.data


@dataclass
class AudioEndMessage:
    """Signal end of audio stream."""
    type: str = MessageType.AUDIO_END

    def to_json(self) -> str:
        return json.dumps({"type": self.type})


@dataclass
class TranscriptionMessage:
    """Transcription result from Jetson."""
    type: str
    text: str
    is_final: bool
    confidence: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "TranscriptionMessage":
        d = json.loads(data)
        return cls(**d)


@dataclass
class IntentMessage:
    """Parsed intent from transcription."""
    type: str = MessageType.INTENT
    intent: str = IntentType.UNKNOWN
    targets: list = None  # List of entity_ids or room names
    target_type: str = "entity"  # "entity" or "room"
    brightness: Optional[int] = None  # 0-100 for brightness commands
    original_text: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        if self.targets is None:
            self.targets = []

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "IntentMessage":
        d = json.loads(data)
        return cls(**d)


@dataclass
class ErrorMessage:
    """Error message."""
    type: str = MessageType.ERROR
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ReadyMessage:
    """Server ready signal."""
    type: str = MessageType.READY
    model_loaded: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def parse_message(data: str) -> dict:
    """Parse a JSON message from WebSocket."""
    return json.loads(data)
