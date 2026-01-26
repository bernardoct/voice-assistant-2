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
        Extract target entities from text. Supports multiple targets with "and".

        Returns:
            Tuple of (entity_ids, target_type)
            target_type is either "entity", "room", or "mixed"
        """
        target_text = self._normalize_text(target_text)

        # Check for multiple targets with "and"
        # e.g., "kitchen and living room" or "bedside lamp and noguchi"
        if " and " in target_text:
            return self._extract_multiple_targets(target_text)

        # Clean up common suffixes that indicate room context
        text_for_room = re.sub(r"\s*lights?\s*$", "", target_text)

        # Check for EXACT room match first (e.g., "living room" should match the room)
        # This prevents "living room" from matching "artemide tolomeo mega living room floor lamp"
        if text_for_room in self.room_names:
            entities = self.room_to_entities.get(text_for_room, [])
            if entities:
                return entities, "room"

        # Check room aliases for exact match
        for room, aliases in self.ROOM_ALIASES.items():
            if text_for_room in aliases:
                if room in self.room_names:
                    entities = self.room_to_entities.get(room, [])
                    if entities:
                        return entities, "room"

        # Check for specific entity match (exact or partial)
        # This ensures "office floor lamp" matches the device, not "office" room
        entities = self._find_entity(target_text)
        if entities:
            # Filter to only light entities for light commands
            light_entities = [e for e in entities if e.startswith("light.")]
            if light_entities:
                return light_entities, "entity"
            return entities, "entity"

        # Fall back to partial room match
        room = self._find_room(target_text)
        if room and room in self.room_to_entities:
            entities = self.room_to_entities[room]
            if entities:
                return entities, "room"

        # Return empty if no match found
        return [], "unknown"

    def _extract_multiple_targets(self, target_text: str) -> Tuple[List[str], str]:
        """
        Extract multiple targets separated by 'and'.

        Args:
            target_text: Text containing multiple targets (e.g., "kitchen and living room")

        Returns:
            Tuple of (combined entity_ids, target_type)
        """
        # Split on " and " but be careful with "and the"
        target_text = re.sub(r"\s+and\s+the\s+", " and ", target_text)
        parts = [p.strip() for p in target_text.split(" and ")]

        all_entities = []
        has_room = False
        has_entity = False

        for part in parts:
            # Clean up common suffixes
            part_clean = re.sub(r"\s*lights?\s*$", "", part)

            # Check for EXACT room match first
            if part_clean in self.room_names:
                all_entities.extend(self.room_to_entities.get(part_clean, []))
                has_room = True
                continue

            # Check room aliases for exact match
            room_matched = False
            for room, aliases in self.ROOM_ALIASES.items():
                if part_clean in aliases and room in self.room_names:
                    all_entities.extend(self.room_to_entities.get(room, []))
                    has_room = True
                    room_matched = True
                    break

            if room_matched:
                continue

            # Try to find entity
            entities = self._find_entity(part)
            if entities:
                light_entities = [e for e in entities if e.startswith("light.")]
                if light_entities:
                    all_entities.extend(light_entities)
                else:
                    all_entities.extend(entities)
                has_entity = True
            else:
                # Fall back to partial room match
                room = self._find_room(part)
                if room and room in self.room_to_entities:
                    all_entities.extend(self.room_to_entities[room])
                    has_room = True

        # Determine target type
        if has_room and has_entity:
            target_type = "mixed"
        elif has_room:
            target_type = "room"
        elif has_entity:
            target_type = "entity"
        else:
            target_type = "unknown"

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in all_entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)

        return unique_entities, target_type

    def _split_chained_commands(self, text: str) -> List[str]:
        """
        Split text into multiple commands if chained.

        Handles patterns like:
        - "turn on X and dim it to Y"
        - "turn on X and turn off Y"
        - "turn on X then set it to 50%"

        Returns:
            List of command strings
        """
        # Patterns that indicate a new command after "and" or "then"
        chain_patterns = [
            r"\s+and\s+(turn|switch|set|dim|toggle|make)\s+",
            r"\s+then\s+(turn|switch|set|dim|toggle|make)\s+",
            r"\s+and\s+(it\s+to\s+\d+)",  # "and it to 50%"
        ]

        # Check if this looks like a chained command
        for pattern in chain_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Split on "and <action>" or "then <action>"
                parts = re.split(r"\s+(?:and|then)\s+(?=turn|switch|set|dim|toggle|make|it\s+to)", text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    return [p.strip() for p in parts if p.strip()]

        return [text]

    def parse_all(self, text: str) -> List[IntentMessage]:
        """
        Parse a voice command that may contain multiple chained commands.

        Args:
            text: The transcribed voice command

        Returns:
            List of IntentMessage objects
        """
        original_text = text
        normalized = self._normalize_text(text)

        # Split into potential multiple commands
        command_parts = self._split_chained_commands(normalized)

        if len(command_parts) == 1:
            # Single command, use normal parsing
            return [self.parse(text)]

        # Multiple commands - parse each one
        intents = []
        previous_targets = []

        for i, part in enumerate(command_parts):
            # Handle "it" pronoun reference to previous targets
            if previous_targets and re.search(r"\bit\b", part, re.IGNORECASE):
                # Replace "it" with a placeholder and use previous targets
                part_modified = re.sub(r"\bit\s+to\s+", "PREV_TARGET to ", part)
                part_modified = re.sub(r"\bit\b", "PREV_TARGET", part_modified)

                # Try to parse as brightness command with previous target
                brightness_match = re.search(r"(?:to\s+)?(\d+)\s*(?:percent|%)?", part)
                if brightness_match and previous_targets:
                    brightness = int(brightness_match.group(1))
                    intent = IntentMessage(
                        intent=IntentType.SET_BRIGHTNESS,
                        targets=previous_targets.copy(),
                        target_type="entity",
                        brightness=min(100, max(0, brightness)),
                        original_text=original_text,
                        confidence=0.85,
                    )
                    intents.append(intent)
                    continue

            # Parse this part normally
            intent = self.parse(part)
            intents.append(intent)

            # Remember targets for pronoun resolution
            if intent.targets:
                previous_targets = intent.targets.copy()

        return intents

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
