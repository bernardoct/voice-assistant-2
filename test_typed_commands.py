#!/usr/bin/env python3
"""
Test script for voice assistant - accepts typed commands instead of voice.
Run this on the Raspberry Pi to test Home Assistant control.

Usage:
    python test_typed_commands.py --ha-token "YOUR_TOKEN"

Then type commands like:
    > Turn on the kitchen
    > Turn off the living room
    > Set the bedside lamp to 50 percent
"""

import asyncio
import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.protocol import IntentType, IntentMessage
from jetson.intent_parser import IntentParser
from pi.ha_controller import HomeAssistantController
from pi import config


async def execute_intent(ha_controller: HomeAssistantController, intent: IntentMessage) -> bool:
    """Execute a parsed intent against Home Assistant."""

    print(f"  Intent: {intent.intent}")
    print(f"  Targets: {intent.targets}")
    print(f"  Target type: {intent.target_type}")
    if intent.brightness is not None:
        print(f"  Brightness: {intent.brightness}%")
    print(f"  Confidence: {intent.confidence:.2f}")

    if not intent.targets:
        print("  [!] No targets found")
        return False

    if intent.intent == IntentType.UNKNOWN:
        print("  [!] Unknown command")
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
                print(f"  -> Calling turn_on_room('{room}')")
                results = await ha_controller.turn_on_room(room, intent.brightness)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    print(f"  -> Calling turn_on('{entity}')")
                    result = await ha_controller.turn_on(entity, intent.brightness)
                    success = success and result
        else:
            for entity in intent.targets:
                print(f"  -> Calling turn_on('{entity}')")
                result = await ha_controller.turn_on(entity, intent.brightness)
                success = success and result

    elif intent.intent == IntentType.TURN_OFF:
        if intent.target_type == "room":
            room = find_room_from_entities(intent.targets)
            if room:
                print(f"  -> Calling turn_off_room('{room}')")
                results = await ha_controller.turn_off_room(room)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    print(f"  -> Calling turn_off('{entity}')")
                    result = await ha_controller.turn_off(entity)
                    success = success and result
        else:
            for entity in intent.targets:
                print(f"  -> Calling turn_off('{entity}')")
                result = await ha_controller.turn_off(entity)
                success = success and result

    elif intent.intent == IntentType.SET_BRIGHTNESS:
        if intent.brightness is None:
            print("  [!] No brightness value specified")
            return False

        if intent.target_type == "room":
            room = find_room_from_entities(intent.targets)
            if room:
                print(f"  -> Calling set_room_brightness('{room}', {intent.brightness})")
                results = await ha_controller.set_room_brightness(room, intent.brightness)
                success = all(results) if results else False
            else:
                for entity in intent.targets:
                    print(f"  -> Calling set_brightness('{entity}', {intent.brightness})")
                    result = await ha_controller.set_brightness(entity, intent.brightness)
                    success = success and result
        else:
            for entity in intent.targets:
                print(f"  -> Calling set_brightness('{entity}', {intent.brightness})")
                result = await ha_controller.set_brightness(entity, intent.brightness)
                success = success and result

    elif intent.intent == IntentType.TOGGLE:
        for entity in intent.targets:
            print(f"  -> Calling toggle('{entity}')")
            result = await ha_controller.toggle(entity)
            success = success and result

    return success


async def process_command(
    command: str,
    intent_parser: IntentParser,
    ha_controller: HomeAssistantController,
) -> bool:
    """Process a single typed command (may contain chained commands)."""
    print(f"\nProcessing: '{command}'")
    print("-" * 50)

    # Parse the command - may return multiple intents for chained commands
    intents = intent_parser.parse_all(command)

    all_success = True
    for i, intent in enumerate(intents):
        if len(intents) > 1:
            print(f"\n  Command {i + 1}/{len(intents)}:")

        # Execute the intent
        success = await execute_intent(ha_controller, intent)
        all_success = all_success and success

        if success:
            print("  [OK] Command executed successfully")
        else:
            print("  [FAILED] Command execution failed")

    return all_success


async def interactive_mode(
    intent_parser: IntentParser,
    ha_controller: HomeAssistantController,
):
    """Run in interactive mode, accepting commands from stdin."""
    print("\n" + "=" * 60)
    print("Voice Assistant Test Mode (Typed Commands)")
    print("=" * 60)
    print("\nType commands like:")
    print("  - Turn on the kitchen")
    print("  - Turn off the living room")
    print("  - Set the bedside lamp to 50 percent")
    print("  - Toggle the office")
    print("\nMultiple targets:")
    print("  - Turn off the kitchen and living room")
    print("  - Turn on the bedside lamp and noguchi")
    print("\nChained commands:")
    print("  - Turn on the bedside lamp and dim it to 30%")
    print("  - Turn on the bedroom and set it to 50 percent")
    print("\nType 'quit' or 'exit' to stop.")
    print("Type 'rooms' to list available rooms.")
    print("Type 'devices' to list available devices.")
    print("=" * 60 + "\n")

    while True:
        try:
            command = input("> ").strip()

            if not command:
                continue

            if command.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if command.lower() == "rooms":
                print("\nAvailable rooms:")
                for room in ha_controller.get_rooms():
                    entities = ha_controller.get_room_entities(room)
                    print(f"  - {room}: {entities}")
                print()
                continue

            if command.lower() == "devices":
                print("\nAvailable devices:")
                for name in intent_parser.entity_names:
                    entities = intent_parser.friendly_to_entity.get(name, [])
                    print(f"  - {name}: {entities}")
                print()
                continue

            await process_command(command, intent_parser, ha_controller)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


async def main():
    parser = argparse.ArgumentParser(
        description="Test voice assistant with typed commands"
    )
    parser.add_argument(
        "--ha-url",
        default=config.HA_URL,
        help="Home Assistant URL (default: %(default)s)",
    )
    parser.add_argument(
        "--ha-token",
        default=config.HA_TOKEN,
        help="Home Assistant long-lived access token",
    )
    parser.add_argument(
        "--entities",
        default=config.HA_ENTITIES_PATH,
        help="Path to ha_entities.json (default: %(default)s)",
    )
    parser.add_argument(
        "--command", "-c",
        help="Execute a single command and exit",
    )

    args = parser.parse_args()

    if not args.ha_token:
        print("Error: Home Assistant token required.")
        print("Set HA_TOKEN environment variable or use --ha-token")
        sys.exit(1)

    # Initialize components
    print(f"Loading entities from: {args.entities}")
    intent_parser = IntentParser(args.entities)

    print(f"Connecting to Home Assistant at: {args.ha_url}")
    ha_controller = HomeAssistantController(
        url=args.ha_url,
        token=args.ha_token,
        entities_path=args.entities,
    )

    await ha_controller.connect()

    try:
        if args.command:
            # Single command mode
            await process_command(args.command, intent_parser, ha_controller)
        else:
            # Interactive mode
            await interactive_mode(intent_parser, ha_controller)
    finally:
        await ha_controller.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
