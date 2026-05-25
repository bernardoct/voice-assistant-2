"""
Unit tests for the voice assistant intent parsing and Home Assistant control.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.protocol import IntentType, IntentMessage
from jetson.intent_parser import IntentParser


# Sample entities data matching the user's ha_entities.json structure
SAMPLE_ENTITIES = {
    "areas": [
        {
            "area_id": None,
            "light_entities": [
                "light.artemide_tolomeo_mega_living_room_floor_lamp",
                "light.dresser_lamp"
            ],
            "name": "living_room",
            "name_norm": "living room"
        },
        {
            "area_id": None,
            "light_entities": ["light.office_floor_lamp"],
            "name": "office",
            "name_norm": "office"
        },
        {
            "area_id": None,
            "light_entities": ["light.countertop_light"],
            "name": "kitchen",
            "name_norm": "kitchen"
        },
        {
            "area_id": None,
            "light_entities": ["light.bedside_lamp"],
            "name": "bedroom",
            "name_norm": "bedroom"
        },
    ],
    "entities": [
        {
            "domain": "light",
            "entity_id": "light.artemide_tolomeo_mega_living_room_floor_lamp",
            "friendly_name": "Artemide Tolomeo Mega (Living Room Floor Lamp)",
            "friendly_norm": "artemide tolomeo mega living room floor lamp",
            "supported_color_modes": ["onoff"]
        },
        {
            "domain": "light",
            "entity_id": "light.countertop_light",
            "friendly_name": "Countertop light",
            "friendly_norm": "countertop light",
            "supported_color_modes": ["onoff"]
        },
        {
            "domain": "light",
            "entity_id": "light.bedside_lamp",
            "friendly_name": "Bedside Lamp",
            "friendly_norm": "bedside lamp",
            "supported_color_modes": ["brightness"]
        },
        {
            "domain": "light",
            "entity_id": "light.dresser_lamp",
            "friendly_name": "Noguchi",
            "friendly_norm": "noguchi",
            "supported_color_modes": ["brightness", "color_temp_kelvin"]
        },
        {
            "domain": "light",
            "entity_id": "light.office_floor_lamp",
            "friendly_name": "Office floor lamp",
            "friendly_norm": "office floor lamp",
            "supported_color_modes": ["brightness", "color_temp_kelvin"]
        },
    ],
    "index": {
        "by_friendly_norm": {
            "artemide tolomeo mega living room floor lamp": [
                "light.artemide_tolomeo_mega_living_room_floor_lamp"
            ],
            "countertop light": ["light.countertop_light"],
            "bedside lamp": ["light.bedside_lamp"],
            "noguchi": ["light.dresser_lamp"],
            "office floor lamp": ["light.office_floor_lamp"],
        }
    }
}


@pytest.fixture
def entities_file(tmp_path):
    """Create a temporary entities file for testing."""
    entities_path = tmp_path / "ha_entities.json"
    entities_path.write_text(json.dumps(SAMPLE_ENTITIES))
    return str(entities_path)


@pytest.fixture
def intent_parser(entities_file):
    """Create an IntentParser with sample entities."""
    return IntentParser(entities_file)


class TestIntentParserTurnOn:
    """Tests for turn on commands."""

    def test_turn_on_specific_device_by_friendly_name(self, intent_parser):
        """Test: 'Turn on the Noguchi' -> turns on light.dresser_lamp"""
        intent = intent_parser.parse("Turn on the Noguchi")

        assert intent.intent == IntentType.TURN_ON
        assert "light.dresser_lamp" in intent.targets
        assert intent.target_type == "entity"
        assert intent.confidence >= 0.9

    def test_turn_on_room_lights(self, intent_parser):
        """Test: 'Turn on the living room' -> turns on all living room lights"""
        intent = intent_parser.parse("Turn on the living room")

        assert intent.intent == IntentType.TURN_ON
        assert intent.target_type == "room"
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets
        assert "light.dresser_lamp" in intent.targets
        assert len(intent.targets) == 2

    def test_turn_on_room_with_trailing_punctuation(self, intent_parser):
        """Whisper often appends '.' or '?'; that must not flip a room into
        a fuzzy entity match (which used to grab the tolomeo)."""
        for text in (
            "Turn on the living room.",
            "Turn on the living room?",
            "Turn on the living room,",
        ):
            intent = intent_parser.parse(text)
            assert intent.intent == IntentType.TURN_ON, text
            assert intent.target_type == "room", f"{text!r} -> {intent.target_type}"
            assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets
            assert "light.dresser_lamp" in intent.targets

    def test_turn_on_kitchen(self, intent_parser):
        """Test: 'Turn on the kitchen' -> turns on kitchen light"""
        intent = intent_parser.parse("Turn on the kitchen")

        assert intent.intent == IntentType.TURN_ON
        assert intent.target_type == "room"
        assert "light.countertop_light" in intent.targets

    def test_turn_on_with_lights_keyword(self, intent_parser):
        """Test: 'Turn on the bedroom lights' -> turns on bedroom lights"""
        intent = intent_parser.parse("Turn on the bedroom lights")

        assert intent.intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intent.targets

    def test_turn_on_office_floor_lamp(self, intent_parser):
        """Test: 'Turn on the office floor lamp' -> turns on specific lamp"""
        intent = intent_parser.parse("Turn on the office floor lamp")

        assert intent.intent == IntentType.TURN_ON
        assert "light.office_floor_lamp" in intent.targets

    def test_switch_on_variant(self, intent_parser):
        """Test: 'Switch on the kitchen' -> same as turn on"""
        intent = intent_parser.parse("Switch on the kitchen")

        assert intent.intent == IntentType.TURN_ON
        assert "light.countertop_light" in intent.targets

    def test_turn_on_with_please(self, intent_parser):
        """Test: 'Please turn on the bedroom' -> handles polite prefix"""
        intent = intent_parser.parse("Please turn on the bedroom")

        assert intent.intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intent.targets


class TestIntentParserTurnOff:
    """Tests for turn off commands."""

    def test_turn_off_lights_in_living_room(self, intent_parser):
        """Test: 'Turn off the lights in the living room' -> turns off all living room lights"""
        intent = intent_parser.parse("Turn off the lights in the living room")

        assert intent.intent == IntentType.TURN_OFF
        assert intent.target_type == "room"
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets
        assert "light.dresser_lamp" in intent.targets

    def test_turn_off_specific_device(self, intent_parser):
        """Test: 'Turn off the bedside lamp' -> turns off specific lamp"""
        intent = intent_parser.parse("Turn off the bedside lamp")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.bedside_lamp" in intent.targets
        assert intent.target_type == "entity"

    def test_turn_off_room(self, intent_parser):
        """Test: 'Turn off the office' -> turns off office lights"""
        intent = intent_parser.parse("Turn off the office")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.office_floor_lamp" in intent.targets

    def test_switch_off_variant(self, intent_parser):
        """Test: 'Switch off the kitchen' -> same as turn off"""
        intent = intent_parser.parse("Switch off the kitchen")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.countertop_light" in intent.targets

    def test_turn_kitchen_off_word_order(self, intent_parser):
        """Test: 'Turn the kitchen off' -> alternate word order"""
        intent = intent_parser.parse("Turn the kitchen off")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.countertop_light" in intent.targets


class TestIntentParserBrightness:
    """Tests for brightness adjustment commands."""

    def test_set_brightness_percentage(self, intent_parser):
        """Test: 'Set the bedside lamp to 50 percent' -> sets brightness"""
        intent = intent_parser.parse("Set the bedside lamp to 50 percent")

        assert intent.intent == IntentType.SET_BRIGHTNESS
        assert "light.bedside_lamp" in intent.targets
        assert intent.brightness == 50

    def test_dim_to_value(self, intent_parser):
        """Test: 'Dim the office floor lamp to 30' -> dims to 30%"""
        intent = intent_parser.parse("Dim the office floor lamp to 30")

        assert intent.intent == IntentType.SET_BRIGHTNESS
        assert "light.office_floor_lamp" in intent.targets
        assert intent.brightness == 30

    def test_set_noguchi_brightness(self, intent_parser):
        """Test: 'Set the Noguchi to 75%' -> sets Noguchi brightness"""
        intent = intent_parser.parse("Set the Noguchi to 75%")

        assert intent.intent == IntentType.SET_BRIGHTNESS
        assert "light.dresser_lamp" in intent.targets
        assert intent.brightness == 75

    def test_brightness_clamped_to_100(self, intent_parser):
        """Test: Brightness over 100 is clamped"""
        intent = intent_parser.parse("Set the bedroom to 150 percent")

        assert intent.intent == IntentType.SET_BRIGHTNESS
        assert intent.brightness == 100

    def test_brightness_clamped_to_0(self, intent_parser):
        """Test: Brightness 0 is valid"""
        intent = intent_parser.parse("Set the bedroom to 0 percent")

        assert intent.intent == IntentType.SET_BRIGHTNESS
        assert intent.brightness == 0


class TestIntentParserToggle:
    """Tests for toggle commands."""

    def test_toggle_specific_light(self, intent_parser):
        """Test: 'Toggle the kitchen light' -> toggles kitchen"""
        intent = intent_parser.parse("Toggle the kitchen light")

        assert intent.intent == IntentType.TOGGLE
        assert "light.countertop_light" in intent.targets

    def test_toggle_room(self, intent_parser):
        """Test: 'Toggle the bedroom' -> toggles bedroom lights"""
        intent = intent_parser.parse("Toggle the bedroom")

        assert intent.intent == IntentType.TOGGLE
        assert "light.bedside_lamp" in intent.targets


class TestIntentParserEdgeCases:
    """Tests for edge cases and unknown commands."""

    def test_unknown_command(self, intent_parser):
        """Test: Unrecognized command returns unknown intent"""
        intent = intent_parser.parse("What's the weather like?")

        assert intent.intent == IntentType.UNKNOWN
        assert intent.confidence == 0.0

    def test_unknown_device(self, intent_parser):
        """Test: Unknown device name has low confidence"""
        intent = intent_parser.parse("Turn on the garage")

        assert intent.intent == IntentType.TURN_ON
        assert intent.targets == []
        assert intent.confidence < 0.9

    def test_case_insensitive(self, intent_parser):
        """Test: Commands are case insensitive"""
        intent = intent_parser.parse("TURN ON THE KITCHEN")

        assert intent.intent == IntentType.TURN_ON
        assert "light.countertop_light" in intent.targets

    def test_extra_whitespace(self, intent_parser):
        """Test: Extra whitespace is handled"""
        intent = intent_parser.parse("  Turn   on   the   bedroom  ")

        assert intent.intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intent.targets

    def test_original_text_preserved(self, intent_parser):
        """Test: Original text is preserved in intent"""
        original = "Please turn on the Noguchi"
        intent = intent_parser.parse(original)

        assert intent.original_text == original


class TestIntentParserRoomAliases:
    """Tests for room alias handling."""

    def test_lounge_alias_for_living_room(self, intent_parser):
        """Test: 'lounge' maps to living room"""
        intent = intent_parser.parse("Turn on the lounge")

        assert intent.intent == IntentType.TURN_ON
        assert intent.target_type == "room"
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets

    def test_study_alias_for_office(self, intent_parser):
        """Test: 'study' maps to office"""
        intent = intent_parser.parse("Turn off the study")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.office_floor_lamp" in intent.targets


class TestMultipleTargets:
    """Tests for commands with multiple targets."""

    def test_two_rooms_with_and(self, intent_parser):
        """Test: 'Turn off the kitchen and living room' -> both rooms"""
        intent = intent_parser.parse("Turn off the kitchen and living room")

        assert intent.intent == IntentType.TURN_OFF
        # Should have lights from both rooms
        assert "light.countertop_light" in intent.targets
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets
        assert "light.dresser_lamp" in intent.targets

    def test_two_devices_with_and(self, intent_parser):
        """Test: 'Turn on the bedside lamp and noguchi' -> both devices"""
        intent = intent_parser.parse("Turn on the bedside lamp and noguchi")

        assert intent.intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intent.targets
        assert "light.dresser_lamp" in intent.targets

    def test_room_and_device_mixed(self, intent_parser):
        """Test: 'Turn on the kitchen and bedside lamp' -> mixed"""
        intent = intent_parser.parse("Turn on the kitchen and bedside lamp")

        assert intent.intent == IntentType.TURN_ON
        assert "light.countertop_light" in intent.targets
        assert "light.bedside_lamp" in intent.targets

    def test_and_the_variant(self, intent_parser):
        """Test: 'Turn off the kitchen and the bedroom' -> handles 'and the'"""
        intent = intent_parser.parse("Turn off the kitchen and the bedroom")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.countertop_light" in intent.targets
        assert "light.bedside_lamp" in intent.targets

    def test_multiple_with_lights_suffix(self, intent_parser):
        """Test: 'Turn off the kitchen and living room lights' -> removes suffix"""
        intent = intent_parser.parse("Turn off the kitchen and living room lights")

        assert intent.intent == IntentType.TURN_OFF
        assert "light.countertop_light" in intent.targets
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets


class TestChainedCommands:
    """Tests for chained commands."""

    def test_turn_on_and_dim(self, intent_parser):
        """Test: 'Turn on the bedside lamp and dim it to 30%' -> two intents"""
        intents = intent_parser.parse_all("Turn on the bedside lamp and dim it to 30%")

        assert len(intents) == 2

        # First intent: turn on
        assert intents[0].intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intents[0].targets

        # Second intent: set brightness with same target
        assert intents[1].intent == IntentType.SET_BRIGHTNESS
        assert "light.bedside_lamp" in intents[1].targets
        assert intents[1].brightness == 30

    def test_turn_on_and_set_brightness(self, intent_parser):
        """Test: 'Turn on the bedroom and set it to 50 percent' -> two intents"""
        intents = intent_parser.parse_all("Turn on the bedroom and set it to 50 percent")

        assert len(intents) == 2

        # First intent: turn on room
        assert intents[0].intent == IntentType.TURN_ON
        assert "light.bedside_lamp" in intents[0].targets

        # Second intent: set brightness
        assert intents[1].intent == IntentType.SET_BRIGHTNESS
        assert intents[1].brightness == 50

    def test_two_separate_commands(self, intent_parser):
        """Test: 'Turn on the kitchen and turn off the bedroom' -> two intents"""
        intents = intent_parser.parse_all("Turn on the kitchen and turn off the bedroom")

        assert len(intents) == 2

        assert intents[0].intent == IntentType.TURN_ON
        assert "light.countertop_light" in intents[0].targets

        assert intents[1].intent == IntentType.TURN_OFF
        assert "light.bedside_lamp" in intents[1].targets

    def test_single_command_not_chained(self, intent_parser):
        """Test: Regular command returns single intent"""
        intents = intent_parser.parse_all("Turn on the kitchen")

        assert len(intents) == 1
        assert intents[0].intent == IntentType.TURN_ON

    def test_multiple_targets_not_confused_with_chained(self, intent_parser):
        """Test: 'Turn off kitchen and bedroom' is NOT a chained command"""
        intents = intent_parser.parse_all("Turn off the kitchen and bedroom")

        # Should be one intent with multiple targets, not two intents
        assert len(intents) == 1
        assert intents[0].intent == IntentType.TURN_OFF
        assert "light.countertop_light" in intents[0].targets
        assert "light.bedside_lamp" in intents[0].targets
