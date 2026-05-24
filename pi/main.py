#!/usr/bin/env python3
"""
Main voice assistant application for Raspberry Pi.
Captures voice commands and controls Home Assistant entities.
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import time
import wave
from typing import Optional

from . import config
from .audio_capture import AudioCapture, ContinuousAudioCapture
from .ha_controller import HomeAssistantController
from .jetson_client import JetsonClient
from .feedback import AudioFeedback
from shared.protocol import IntentType, IntentMessage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Main voice assistant application.
    Coordinates audio capture, transcription, and Home Assistant control.
    """

    def __init__(
        self,
        jetson_url: str = config.JETSON_URL,
        ha_url: str = config.HA_URL,
        ha_token: str = config.HA_TOKEN,
        entities_path: str = config.HA_ENTITIES_PATH,
        enable_feedback: bool = config.ENABLE_AUDIO_FEEDBACK,
    ):
        self.audio_capture = AudioCapture()
        self.continuous_capture = ContinuousAudioCapture(self.audio_capture)
        self.jetson_client = JetsonClient(jetson_url)
        self.ha_controller = HomeAssistantController(ha_url, ha_token, entities_path)
        self.feedback = AudioFeedback() if enable_feedback else None

        self._running = False
        self._processing = False

    async def start(self) -> None:
        """Start the voice assistant."""
        logger.info("Starting voice assistant...")

        # Connect to services
        await self.ha_controller.connect()

        if not await self.jetson_client.connect():
            logger.error("Failed to connect to Jetson server")
            # Continue anyway, will retry on each request

        self._running = True
        logger.info("Voice assistant started")

    async def stop(self) -> None:
        """Stop the voice assistant."""
        logger.info("Stopping voice assistant...")
        self._running = False

        self.continuous_capture.stop()
        await self.jetson_client.disconnect()
        await self.ha_controller.disconnect()

        if self.feedback:
            self.feedback.cleanup()

        logger.info("Voice assistant stopped")

    async def _execute_intent(self, intent: IntentMessage) -> bool:
        """
        Execute a parsed intent.

        Args:
            intent: Parsed intent message

        Returns:
            True if executed successfully
        """
        logger.info(
            f"Executing intent: {intent.intent} -> {intent.targets} "
            f"(type: {intent.target_type}, brightness: {intent.brightness})"
        )

        if not intent.targets:
            logger.warning("No targets specified in intent")
            return False

        success = True

        try:
            if intent.intent == IntentType.TURN_ON:
                if intent.target_type == "room":
                    # Turn on room
                    room = self._find_room_from_entities(intent.targets)
                    if room:
                        results = await self.ha_controller.turn_on_room(
                            room, intent.brightness
                        )
                        success = all(results) if results else False
                    else:
                        # Turn on individual entities
                        for entity in intent.targets:
                            result = await self.ha_controller.turn_on(
                                entity, intent.brightness
                            )
                            success = success and result
                else:
                    # Turn on individual entities
                    for entity in intent.targets:
                        result = await self.ha_controller.turn_on(
                            entity, intent.brightness
                        )
                        success = success and result

            elif intent.intent == IntentType.TURN_OFF:
                if intent.target_type == "room":
                    room = self._find_room_from_entities(intent.targets)
                    if room:
                        results = await self.ha_controller.turn_off_room(room)
                        success = all(results) if results else False
                    else:
                        for entity in intent.targets:
                            result = await self.ha_controller.turn_off(entity)
                            success = success and result
                else:
                    for entity in intent.targets:
                        result = await self.ha_controller.turn_off(entity)
                        success = success and result

            elif intent.intent == IntentType.SET_BRIGHTNESS:
                if intent.brightness is None:
                    logger.warning("No brightness value specified")
                    return False

                if intent.target_type == "room":
                    room = self._find_room_from_entities(intent.targets)
                    if room:
                        results = await self.ha_controller.set_room_brightness(
                            room, intent.brightness
                        )
                        success = all(results) if results else False
                    else:
                        for entity in intent.targets:
                            result = await self.ha_controller.set_brightness(
                                entity, intent.brightness
                            )
                            success = success and result
                else:
                    for entity in intent.targets:
                        result = await self.ha_controller.set_brightness(
                            entity, intent.brightness
                        )
                        success = success and result

            elif intent.intent == IntentType.TOGGLE:
                for entity in intent.targets:
                    result = await self.ha_controller.toggle(entity)
                    success = success and result

            else:
                logger.warning(f"Unknown intent type: {intent.intent}")
                return False

        except Exception as e:
            logger.error(f"Error executing intent: {e}")
            return False

        return success

    def _find_room_from_entities(self, entities: list) -> Optional[str]:
        """Find room name from entity list."""
        for room, room_entities in self.ha_controller._room_to_entities.items():
            if set(entities) == set(room_entities):
                return room
            # Also check if entities are a subset of room entities
            if set(entities).issubset(set(room_entities)) and len(entities) > 0:
                return room
        return None

    async def _on_speech_start(self) -> None:
        """Called when speech is detected."""
        logger.debug("Speech detected")
        if self.feedback:
            await self.feedback.play_listening_start()

    async def _on_speech_end(self) -> None:
        """Called when speech ends."""
        logger.debug("Speech ended")
        if self.feedback:
            await self.feedback.play_listening_end()

    async def _process_speech(self, audio_data: bytes) -> None:
        """
        Process captured speech.

        Args:
            audio_data: Raw audio bytes
        """
        if self._processing:
            logger.debug("Already processing, skipping")
            return

        self._processing = True

        # # Save audio to WAV file for debugging before sending to Jetson
        # try:
        #     debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug_audio")
        #     os.makedirs(debug_dir, exist_ok=True)
        #     wav_path = os.path.join(debug_dir, f"sent_{time.strftime('%Y%m%d_%H%M%S')}.wav")
        #     with wave.open(wav_path, "wb") as wf:
        #         wf.setnchannels(config.CHANNELS)
        #         wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        #         wf.setframerate(config.SAMPLE_RATE)
        #         wf.writeframes(audio_data)
        #     logger.info(f"Saved audio to {wav_path} ({len(audio_data)} bytes)")
        # except Exception as e:
        #     logger.warning(f"Failed to save debug audio: {e}")

        try:
            # Send to Jetson for transcription and intent parsing
            # May return multiple intents for chained commands
            transcription, intents = await self.jetson_client.process_audio(
                audio_data, config.SAMPLE_RATE
            )

            if transcription:
                logger.info(f"Transcription: {transcription}")

            if not intents:
                logger.info("No intents parsed")
                if self.feedback:
                    await self.feedback.play_unknown()
                return

            # Execute all intents
            all_success = True
            has_valid_intent = False

            for intent in intents:
                if intent.intent != IntentType.UNKNOWN:
                    has_valid_intent = True
                    success = await self._execute_intent(intent)
                    all_success = all_success and success
                else:
                    logger.info(f"Unknown command: {intent.original_text}")

            # Play feedback based on overall result
            if self.feedback:
                if has_valid_intent:
                    if all_success:
                        await self.feedback.play_success()
                    else:
                        await self.feedback.play_error()
                else:
                    await self.feedback.play_unknown()

        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            if self.feedback:
                await self.feedback.play_error()

        finally:
            self._processing = False

    async def run(self) -> None:
        """Main run loop."""
        await self.start()

        logger.info("Listening for voice commands... (Press Ctrl+C to stop)")

        try:
            async for audio_segment in self.continuous_capture.run(
                on_speech_start=lambda: asyncio.create_task(self._on_speech_start()),
                on_speech_end=lambda: asyncio.create_task(self._on_speech_end()),
            ):
                if not self._running:
                    break

                # Process the speech segment
                await self._process_speech(audio_segment)

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voice Assistant for Home Assistant")
    parser.add_argument(
        "--jetson-host",
        default=config.JETSON_HOST,
        help="Jetson server hostname/IP",
    )
    parser.add_argument(
        "--jetson-port",
        type=int,
        default=config.JETSON_PORT,
        help="Jetson server port",
    )
    parser.add_argument(
        "--ha-url",
        default=config.HA_URL,
        help="Home Assistant URL",
    )
    parser.add_argument(
        "--ha-token",
        default=config.HA_TOKEN,
        help="Home Assistant long-lived access token",
    )
    parser.add_argument(
        "--entities",
        default=config.HA_ENTITIES_PATH,
        help="Path to ha_entities.json",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable audio feedback",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    jetson_url = f"ws://{args.jetson_host}:{args.jetson_port}"

    # Create and run assistant
    assistant = VoiceAssistant(
        jetson_url=jetson_url,
        ha_url=args.ha_url,
        ha_token=args.ha_token,
        entities_path=args.entities,
        enable_feedback=not args.no_feedback,
    )

    # Handle shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        loop.create_task(assistant.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
