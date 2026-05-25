#!/usr/bin/env python3
"""Raspberry Pi voice assistant entry point.

Pipeline:
  1. Continuously read 80 ms audio frames from the mic.
  2. Feed each frame to the wake-word detector.
  3. On wake: play a chime, switch to VAD-driven utterance capture.
  4. Send the captured utterance to the Jetson, await intents.
  5. Execute each intent via the Home Assistant REST API.
  6. Play success/error/unknown chime, reset, return to step 2.
"""

import argparse
import asyncio
import logging
import signal
import sys

from . import config
from .audio_capture import AudioStream, UtteranceRecorder
from .feedback import AudioFeedback
from .ha_controller import HomeAssistantController
from .jetson_client import JetsonClient
from .wake_word import WakeWordDetector
from shared.protocol import IntentType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pi.main")


class VoiceAssistant:
    def __init__(self, args: argparse.Namespace):
        self.audio = AudioStream(device_index=args.input_device)
        self.wake = WakeWordDetector(
            model_name=args.wake_model,
            threshold=args.wake_threshold,
        )
        self.recorder = UtteranceRecorder()
        self.jetson = JetsonClient(url=f"ws://{args.jetson_host}:{args.jetson_port}")
        self.ha = HomeAssistantController(
            url=args.ha_url,
            token=args.ha_token,
            entities_path=args.entities,
        )
        self.feedback = AudioFeedback(enabled=not args.no_feedback)
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def start(self) -> None:
        await self.ha.connect()
        # Best-effort Jetson connect now so the first command isn't slow.
        await self.jetson.connect()
        self.audio.start()

    async def shutdown(self) -> None:
        self.audio.stop()
        await self.jetson.disconnect()
        await self.ha.disconnect()
        self.feedback.cleanup()

    async def _capture_utterance(self) -> bytes:
        """Block until the user finishes speaking or we time out. Returns PCM bytes."""
        self.recorder.reset()
        while not self.recorder.done and not self._stop.is_set():
            frame = await self.audio.read_frame()
            self.recorder.feed(frame)
        return self.recorder.get_audio()

    async def _handle_command(self) -> None:
        await self.feedback.listening()
        logger.info("Wake word detected -- capturing command...")
        audio = await self._capture_utterance()

        if self.recorder.timed_out or not audio:
            logger.info("No speech captured after wake; ignoring.")
            await self.feedback.unknown()
            return

        transcription, intents = await self.jetson.process_audio(
            audio, sample_rate=self.audio.sample_rate
        )
        if transcription is not None:
            logger.info("Transcription: %r", transcription)

        if not intents:
            logger.info("Jetson returned no intents")
            await self.feedback.unknown()
            return

        any_executed = False
        all_ok = True
        for intent in intents:
            if intent.intent == IntentType.UNKNOWN:
                logger.info("Unknown intent: %r", intent.original_text)
                continue
            any_executed = True
            ok = await self.ha.execute_intent(intent)
            all_ok = all_ok and ok

        if not any_executed:
            await self.feedback.unknown()
        elif all_ok:
            await self.feedback.success()
        else:
            await self.feedback.error()

    async def run(self) -> None:
        await self.start()
        logger.info(
            "Listening for wake word '%s' (Ctrl+C to stop)...",
            self.wake.model_name,
        )
        try:
            while not self._stop.is_set():
                frame = await self.audio.read_frame()
                if self.wake.detect(frame) is None:
                    continue
                try:
                    await self._handle_command()
                finally:
                    # Always clear the wake detector's internal state so the
                    # tail of the user's speech doesn't bleed into the next
                    # detection window.
                    self.wake.reset()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raspberry Pi voice assistant")
    p.add_argument("--jetson-host", default=config.JETSON_HOST)
    p.add_argument("--jetson-port", type=int, default=config.JETSON_PORT)
    p.add_argument("--ha-url", default=config.HA_URL)
    p.add_argument("--ha-token", default=config.HA_TOKEN)
    p.add_argument("--entities", default=config.HA_ENTITIES_PATH)
    p.add_argument("--wake-model", default=config.WAKE_MODEL,
                   help="openwakeword model (alexa, hey_jarvis, hey_mycroft, hey_rhasspy)")
    p.add_argument("--wake-threshold", type=float, default=config.WAKE_THRESHOLD)
    p.add_argument("--input-device", type=int, default=config.INPUT_DEVICE_INDEX,
                   help="PyAudio input device index (default: system default)")
    p.add_argument("--no-feedback", action="store_true",
                   help="Disable audio chimes")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # openwakeword and pyaudio are chatty at DEBUG; tame them.
    logging.getLogger("openwakeword").setLevel(logging.INFO)

    if not args.ha_token:
        logger.error("HA_TOKEN is required (set in .env or via --ha-token)")
        sys.exit(1)

    assistant = VoiceAssistant(args)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, assistant.request_stop)
        except NotImplementedError:
            pass  # Windows or limited environments.

    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
