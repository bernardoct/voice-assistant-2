"""
Intent parser for voice commands.
Uses pattern matching for common commands with optional LLM fallback.
"""

import re
import json
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from shared.protocol import IntentType, IntentMessage

logger = logging.getLogger(__name__)


class IntentParser:
    """
    Parse voice commands into structured intents.
    Uses regex pattern matching for speed, with optional LLM fallback.
    """

    # Common patterns for light/switch control
    TURN_ON_PATTERNS = [
        r"(?:turn|switch)\s+on\s+(?:the\s+)?(.+)",
        r"(?:turn|switch)\s+(?:the\s+)?(.+)\s+on",
        r"(?:lights?\s+on\s+(?:in\s+)?(?:the\s+)?(.+))",
        r"(?:enable|activate)\s+(?:the\s+)?(.+)",
    ]

    TURN_OFF_PATTERNS = [
        r"(?:turn|switch)\s+off\s+(?:the\s+)?(.+)",
        r"(?:turn|switch)\s+(?:the\s+)?(.+)\s+off",
        r"(?:lights?\s+off\s+(?:in\s+)?(?:the\s+)?(.+))",
        r"(?:disable|deactivate)\s+(?:the\s+)?(.+)",
    ]

    BRIGHTNESS_PATTERNS = [
        r"(?:set|dim|change)\s+(?:the\s+)?(.+?)\s+(?:to\s+)?(\d+)\s*(?:percent|%)?",
        r"(?:set|change)\s+(?:the\s+)?(.+?)\s+brightness\s+(?:to\s+)?(\d+)\s*(?:percent|%)?",
        r"dim\s+(?:the\s+)?(.+?)\s+(?:to\s+)?(\d+)\s*(?:percent|%)?",
        r"(?:make\s+)?(?:the\s+)?(.+?)\s+(?:brighter|dimmer|lighter|darker)",
    ]

    TOGGLE_PATTERNS = [
        r"toggle\s+(?:the\s+)?(.+)",
    ]

    # Room aliases and variations
    ROOM_ALIASES = {
        "living room": ["living room", "livingroom", "lounge", "front room", "main room"],
        "bedroom": ["bedroom", "bed room", "master bedroom"],
        "kitchen": ["kitchen", "the kitchen"],
        "office": ["office", "study", "home office", "work room"],
        "bathroom": ["bathroom", "bath room", "restroom"],
        "apartment": ["apartment", "whole house", "everywhere", "all lights", "all rooms", "house"],
    }

    def __init__(self, entities_path: Optional[str] = None):
        """
        Initialize the intent parser.

        Args:
            entities_path: Path to ha_entities.json file
        """
        self.entities: Dict[str, Any] = {}
        self.entity_names: List[str] = []
        self.room_names: List[str] = []
        self.friendly_to_entity: Dict[str, List[str]] = {}
        self.room_to_entities: Dict[str, List[str]] = {}

        if entities_path:
            self.load_entities(entities_path)

    def load_entities(self, path: str) -> None:
        """Load entity definitions from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.entities = data

            # Build friendly name to entity mapping
            if "index" in data and "by_friendly_norm" in data["index"]:
                self.friendly_to_entity = data["index"]["by_friendly_norm"]
                self.entity_names = list(self.friendly_to_entity.keys())

            # Build room to entities mapping
            if "areas" in data:
                for area in data["areas"]:
                    room_name = area.get("name_norm", area.get("name", ""))
                    entities = area.get("light_entities", [])
                    if room_name:
                        self.room_to_entities[room_name] = entities
                        self.room_names.append(room_name)

            logger.info(
                f"Loaded {len(self.entity_names)} entities and "
                f"{len(self.room_names)} rooms"
            )

        except Exception as e:
            logger.error(f"Failed to load entities from {path}: {e}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Lowercase and strip
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common filler words at start
        text = re.sub(r"^(please|can you|could you|hey|ok|okay)\s+", "", text)
        return text

    def _find_room(self, text: str) -> Optional[str]:
        """Find room name in text."""
        text_lower = text.lower()

        # Check direct room names first
        for room in self.room_names:
            if room in text_lower:
                return room

        # Check aliases
        for room, aliases in self.ROOM_ALIASES.items():
            for alias in aliases:
                if alias in text_lower:
                    # Map alias to actual room name if it exists
                    if room in self.room_names:
                        return room
                    # Check if alias matches any actual room
                    for actual_room in self.room_names:
                        if actual_room in room or room in actual_room:
                            return actual_room

        return None

    def _find_entity(self, text: str) -> Optional[List[str]]:
        """Find entity in text by friendly name matching."""
        text_lower = text.lower()

        # Try exact match first
        if text_lower in self.friendly_to_entity:
            return self.friendly_to_entity[text_lower]

        # Try partial matching
        best_match = None
        best_score = 0

        for friendly_name, entity_ids in self.friendly_to_entity.items():
            # Check if friendly name is contained in text
            if friendly_name in text_lower:
                score = len(friendly_name)
                if score > best_score:
                    best_score = score
                    best_match = entity_ids
            # Check if text is contained in friendly name
            elif text_lower in friendly_name:
                score = len(text_lower)
                if score > best_score:
                    best_score = score
                    best_match = entity_ids

        return best_match

    def _extract_target(self, target_text: str) -> Tuple[List[str], str]:
        """
        Extract target entities from text.

        Returns:
            Tuple of (entity_ids, target_type)
            target_type is either "entity" or "room"
        """
        target_text = self._normalize_text(target_text)

        # Check for room first
        room = self._find_room(target_text)
        if room and room in self.room_to_entities:
            entities = self.room_to_entities[room]
            if entities:
                return entities, "room"

        # Check for specific entity
        entities = self._find_entity(target_text)
        if entities:
            # Filter to only light entities for light commands
            light_entities = [e for e in entities if e.startswith("light.")]
            if light_entities:
                return light_entities, "entity"
            return entities, "entity"

        # Return empty if no match found
        return [], "unknown"

    def parse(self, text: str) -> IntentMessage:
        """
        Parse a voice command into an intent.

        Args:
            text: The transcribed voice command

        Returns:
            IntentMessage with parsed intent
        """
        original_text = text
        text = self._normalize_text(text)

        # Try brightness patterns first (more specific)
        for pattern in self.BRIGHTNESS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    target_text = groups[0]
                    brightness = int(groups[1])
                    targets, target_type = self._extract_target(target_text)

                    return IntentMessage(
                        intent=IntentType.SET_BRIGHTNESS,
                        targets=targets,
                        target_type=target_type,
                        brightness=min(100, max(0, brightness)),
                        original_text=original_text,
                        confidence=0.9 if targets else 0.5,
                    )
                elif len(groups) == 1:
                    # Relative brightness (brighter/dimmer)
                    target_text = groups[0]
                    targets, target_type = self._extract_target(target_text)
                    # Default adjustment
                    brightness = 50  # Will need context for relative adjustments

                    return IntentMessage(
                        intent=IntentType.SET_BRIGHTNESS,
                        targets=targets,
                        target_type=target_type,
                        brightness=brightness,
                        original_text=original_text,
                        confidence=0.7 if targets else 0.4,
                    )

        # Try turn on patterns
        for pattern in self.TURN_ON_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target_text = match.group(1)
                targets, target_type = self._extract_target(target_text)

                return IntentMessage(
                    intent=IntentType.TURN_ON,
                    targets=targets,
                    target_type=target_type,
                    original_text=original_text,
                    confidence=0.9 if targets else 0.5,
                )

        # Try turn off patterns
        for pattern in self.TURN_OFF_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target_text = match.group(1)
                targets, target_type = self._extract_target(target_text)

                return IntentMessage(
                    intent=IntentType.TURN_OFF,
                    targets=targets,
                    target_type=target_type,
                    original_text=original_text,
                    confidence=0.9 if targets else 0.5,
                )

        # Try toggle patterns
        for pattern in self.TOGGLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target_text = match.group(1)
                targets, target_type = self._extract_target(target_text)

                return IntentMessage(
                    intent=IntentType.TOGGLE,
                    targets=targets,
                    target_type=target_type,
                    original_text=original_text,
                    confidence=0.9 if targets else 0.5,
                )

        # No pattern matched
        return IntentMessage(
            intent=IntentType.UNKNOWN,
            targets=[],
            target_type="unknown",
            original_text=original_text,
            confidence=0.0,
        )


class LLMIntentParser:
    """
    LLM-based intent parser for complex commands.
    Uses a local LLM model for fallback parsing.
    """

    SYSTEM_PROMPT = """You are a home automation intent parser. Parse the user's voice command into a structured intent.

Available intents:
- turn_on: Turn on a light or switch
- turn_off: Turn off a light or switch
- set_brightness: Set brightness level (0-100)
- toggle: Toggle a light or switch
- unknown: Cannot determine intent

Available rooms: {rooms}
Available devices: {devices}

Respond with JSON only:
{{"intent": "<intent_type>", "targets": ["<entity_or_room>"], "target_type": "<entity|room>", "brightness": <null or 0-100>}}
"""

    def __init__(self, model_path: str, entities_path: Optional[str] = None):
        """
        Initialize LLM intent parser.

        Args:
            model_path: Path to local LLM model
            entities_path: Path to ha_entities.json
        """
        self.model_path = model_path
        self.model = None
        self.rooms = []
        self.devices = []

        if entities_path:
            self._load_entity_info(entities_path)

    def _load_entity_info(self, path: str) -> None:
        """Load entity info for prompt."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "areas" in data:
                self.rooms = [a.get("name_norm", a.get("name", "")) for a in data["areas"]]

            if "index" in data and "by_friendly_norm" in data["index"]:
                self.devices = list(data["index"]["by_friendly_norm"].keys())

        except Exception as e:
            logger.error(f"Failed to load entity info: {e}")

    def load(self) -> None:
        """Load the LLM model."""
        # Placeholder for LLM loading (e.g., llama-cpp-python)
        # This would be implemented based on the chosen LLM
        logger.info("LLM intent parser not implemented - using pattern matching only")

    def parse(self, text: str) -> Optional[IntentMessage]:
        """
        Parse command using LLM.

        Args:
            text: Voice command text

        Returns:
            IntentMessage or None if parsing fails
        """
        # Placeholder - would implement actual LLM inference
        return None
