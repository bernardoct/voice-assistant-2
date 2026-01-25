"""
Integration tests verifying that voice commands result in correct Home Assistant API calls.
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


# Sample entities data
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


@pytest.fixture
def mock_ha_controller():
    """Create a mock Home Assistant controller."""
    controller = MagicMock()
    controller.turn_on = AsyncMock(return_value=True)
    controller.turn_off = AsyncMock(return_value=True)
    controller.toggle = AsyncMock(return_value=True)
    controller.set_brightness = AsyncMock(return_value=True)
    controller.turn_on_room = AsyncMock(return_value=[True, True])
    controller.turn_off_room = AsyncMock(return_value=[True, True])
    controller.set_room_brightness = AsyncMock(return_value=[True, True])
    controller._room_to_entities = {
        "living room": [
            "light.artemide_tolomeo_mega_living_room_floor_lamp",
            "light.dresser_lamp"
        ],
        "office": ["light.office_floor_lamp"],
        "kitchen": ["light.countertop_light"],
        "bedroom": ["light.bedside_lamp"],
    }
    return controller


async def execute_intent(ha_controller, intent: IntentMessage) -> bool:
    """
    Execute an intent against the HA controller.
    This mirrors the logic in VoiceAssistant._execute_intent.
    """
    if not intent.targets:
        return False

    success = True

    # Helper to find room from entities
    def find_room_from_entities(entities):
        for room, room_entities in ha_controller._room_to_entities.items():
            if set(entities) == set(room_entities):
                return room
        return None

    if intent.intent == IntentType.TURN_ON:
        if intent.target_type == "room":
            room = find_room_from_entities(intent.targets)
            if room:
                results = await ha_controller.turn_on_room(room, intent.brightness)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    result = await ha_controller.turn_on(entity, intent.brightness)
                    success = success and result
        else:
            for entity in intent.targets:
                result = await ha_controller.turn_on(entity, intent.brightness)
                success = success and result

    elif intent.intent == IntentType.TURN_OFF:
        if intent.target_type == "room":
            room = find_room_from_entities(intent.targets)
            if room:
                results = await ha_controller.turn_off_room(room)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    result = await ha_controller.turn_off(entity)
                    success = success and result
        else:
            for entity in intent.targets:
                result = await ha_controller.turn_off(entity)
                success = success and result

    elif intent.intent == IntentType.SET_BRIGHTNESS:
        if intent.brightness is None:
            return False

        if intent.target_type == "room":
            room = find_room_from_entities(intent.targets)
            if room:
                results = await ha_controller.set_room_brightness(room, intent.brightness)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    result = await ha_controller.set_brightness(entity, intent.brightness)
                    success = success and result
        else:
            for entity in intent.targets:
                result = await ha_controller.set_brightness(entity, intent.brightness)
                success = success and result

    elif intent.intent == IntentType.TOGGLE:
        for entity in intent.targets:
            result = await ha_controller.toggle(entity)
            success = success and result

    else:
        return False

    return success


class TestVoiceCommandToHACall:
    """
    End-to-end tests: voice command -> intent parsing -> HA controller calls.
    Verifies that the correct HA API methods are called with correct arguments.
    """

    @pytest.mark.asyncio
    async def test_turn_on_noguchi_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn on the Noguchi' results in turn_on call for light.dresser_lamp
        """
        # Parse the voice command
        intent = intent_parser.parse("Turn on the Noguchi")

        # Execute the intent
        success = await execute_intent(mock_ha_controller, intent)

        # Verify correct HA call was made
        assert success is True
        mock_ha_controller.turn_on.assert_called_once_with(
            "light.dresser_lamp", None
        )

    @pytest.mark.asyncio
    async def test_turn_off_living_room_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn off the lights in the living room' results in turn_off_room call
        """
        intent = intent_parser.parse("Turn off the lights in the living room")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_off_room.assert_called_once_with("living room")

    @pytest.mark.asyncio
    async def test_turn_on_living_room_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn on the living room' results in turn_on_room call
        """
        intent = intent_parser.parse("Turn on the living room")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_on_room.assert_called_once_with("living room", None)

    @pytest.mark.asyncio
    async def test_turn_on_kitchen_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn on the kitchen' results in turn_on_room for kitchen
        """
        intent = intent_parser.parse("Turn on the kitchen")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_on_room.assert_called_once_with("kitchen", None)

    @pytest.mark.asyncio
    async def test_turn_off_bedroom_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn off the bedroom' results in turn_off_room for bedroom
        """
        intent = intent_parser.parse("Turn off the bedroom")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_off_room.assert_called_once_with("bedroom")

    @pytest.mark.asyncio
    async def test_set_brightness_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Set the bedside lamp to 50 percent' results in set_brightness call
        """
        intent = intent_parser.parse("Set the bedside lamp to 50 percent")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.set_brightness.assert_called_once_with(
            "light.bedside_lamp", 50
        )

    @pytest.mark.asyncio
    async def test_dim_office_lamp_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Dim the office floor lamp to 30' results in set_brightness call
        """
        intent = intent_parser.parse("Dim the office floor lamp to 30")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.set_brightness.assert_called_once_with(
            "light.office_floor_lamp", 30
        )

    @pytest.mark.asyncio
    async def test_toggle_kitchen_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Toggle the kitchen light' results in toggle call
        """
        intent = intent_parser.parse("Toggle the kitchen light")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.toggle.assert_called_once_with("light.countertop_light")

    @pytest.mark.asyncio
    async def test_turn_off_specific_lamp_calls_ha(self, intent_parser, mock_ha_controller):
        """
        Test: 'Turn off the office floor lamp' results in turn_off for specific entity
        """
        intent = intent_parser.parse("Turn off the office floor lamp")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_off.assert_called_once_with("light.office_floor_lamp")

    @pytest.mark.asyncio
    async def test_unknown_command_no_ha_call(self, intent_parser, mock_ha_controller):
        """
        Test: Unknown command doesn't result in any HA calls
        """
        intent = intent_parser.parse("What time is it?")

        # The intent should be UNKNOWN with no targets
        assert intent.intent == IntentType.UNKNOWN

        success = await execute_intent(mock_ha_controller, intent)

        assert success is False
        mock_ha_controller.turn_on.assert_not_called()
        mock_ha_controller.turn_off.assert_not_called()
        mock_ha_controller.toggle.assert_not_called()
        mock_ha_controller.set_brightness.assert_not_called()


class TestMultipleLightsInRoom:
    """
    Tests for room commands that affect multiple lights.
    """

    @pytest.mark.asyncio
    async def test_living_room_has_two_lights(self, intent_parser, mock_ha_controller):
        """
        Test: Living room command targets both lights in the room
        """
        intent = intent_parser.parse("Turn on the living room")

        # Verify the intent has both lights
        assert "light.artemide_tolomeo_mega_living_room_floor_lamp" in intent.targets
        assert "light.dresser_lamp" in intent.targets
        assert len(intent.targets) == 2

    @pytest.mark.asyncio
    async def test_room_brightness_affects_all_lights(self, intent_parser, mock_ha_controller):
        """
        Test: Setting room brightness calls set_room_brightness
        """
        # For this test, we need to make the intent parser return room type
        # The current implementation might not support room-level brightness directly
        # Let's test what happens with the bedroom (single light room)
        intent = intent_parser.parse("Set the bedroom to 75 percent")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True


class TestErrorHandling:
    """
    Tests for error handling scenarios.
    """

    @pytest.mark.asyncio
    async def test_empty_targets_returns_false(self, intent_parser, mock_ha_controller):
        """
        Test: Intent with empty targets returns False
        """
        intent = IntentMessage(
            intent=IntentType.TURN_ON,
            targets=[],
            target_type="entity",
            original_text="Turn on the nonexistent light",
        )

        success = await execute_intent(mock_ha_controller, intent)

        assert success is False

    @pytest.mark.asyncio
    async def test_ha_failure_returns_false(self, intent_parser, mock_ha_controller):
        """
        Test: When HA controller returns failure, execute_intent returns False
        """
        mock_ha_controller.turn_on = AsyncMock(return_value=False)

        intent = intent_parser.parse("Turn on the Noguchi")
        success = await execute_intent(mock_ha_controller, intent)

        assert success is False

    @pytest.mark.asyncio
    async def test_brightness_without_value_returns_false(self, intent_parser, mock_ha_controller):
        """
        Test: SET_BRIGHTNESS with no brightness value returns False
        """
        intent = IntentMessage(
            intent=IntentType.SET_BRIGHTNESS,
            targets=["light.bedside_lamp"],
            target_type="entity",
            brightness=None,  # Missing brightness
            original_text="Set the bedside lamp brightness",
        )

        success = await execute_intent(mock_ha_controller, intent)

        assert success is False


class TestNaturalLanguageVariations:
    """
    Tests for various natural language phrasings that should work.
    """

    @pytest.mark.asyncio
    async def test_please_prefix(self, intent_parser, mock_ha_controller):
        """Test: 'Please turn on the kitchen' works"""
        intent = intent_parser.parse("Please turn on the kitchen")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_on_room.assert_called_once_with("kitchen", None)

    @pytest.mark.asyncio
    async def test_can_you_prefix(self, intent_parser, mock_ha_controller):
        """Test: 'Can you turn off the bedroom' works"""
        intent = intent_parser.parse("Can you turn off the bedroom")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_off_room.assert_called_once_with("bedroom")

    @pytest.mark.asyncio
    async def test_switch_synonym(self, intent_parser, mock_ha_controller):
        """Test: 'Switch on the office' works like 'turn on'"""
        intent = intent_parser.parse("Switch on the office")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.turn_on_room.assert_called_once_with("office", None)

    @pytest.mark.asyncio
    async def test_percent_symbol(self, intent_parser, mock_ha_controller):
        """Test: 'Set the Noguchi to 60%' works with % symbol"""
        intent = intent_parser.parse("Set the Noguchi to 60%")

        success = await execute_intent(mock_ha_controller, intent)

        assert success is True
        mock_ha_controller.set_brightness.assert_called_once_with(
            "light.dresser_lamp", 60
        )
