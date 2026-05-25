#!/usr/bin/env python3
"""
Jetson WebSocket server for voice assistant.
Handles audio streaming, transcription, and intent parsing.
"""

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from . import config
from .transcriber import Transcriber
from .intent_parser import IntentParser
from shared.protocol import (
    MessageType,
    IntentMessage,
    TranscriptionMessage,
    ErrorMessage,
    ReadyMessage,
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VoiceAssistantServer:
    """WebSocket server for voice assistant processing."""

    def __init__(
        self,
        host: str = config.HOST,
        port: int = config.PORT,
        entities_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.entities_path = entities_path
        self.transcriber = Transcriber()
        self.intent_parser = IntentParser(entities_path)
        self._running = False
        self._server = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        # Pre-load models
        logger.info("Loading models...")
        self.transcriber.load()

        # Load hotwords from entities for transcription bias
        if self.entities_path:
            self.transcriber.load_hotwords_from_entities(self.entities_path)

        logger.info("Models loaded, starting server...")

        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=config.WS_PING_INTERVAL,
            ping_timeout=config.WS_PING_TIMEOUT,
            max_size=config.WS_MAX_SIZE,
        )

        logger.info(f"Server listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        # Send ready signal
        await websocket.send(ReadyMessage().to_json())

        audio_buffer = []
        sample_rate = config.SAMPLE_RATE
        _chunk_count = 0  # debug counter, reset each utterance

        try:
            async for message in websocket:
                if not self._running:
                    break

                # Handle binary audio data
                if isinstance(message, bytes):
                    # Parse binary message: 1 byte type + 4 bytes sample_rate + audio
                    if len(message) > 5 and message[0] == 0x01:
                        sample_rate = int.from_bytes(message[1:5], "little")
                        audio_data = message[5:]
                        audio_buffer.append(audio_data)
                    else:
                        # Raw audio data
                        audio_data = message

                        audio_buffer.append(audio_data)

                    _chunk_count += 1
                    _total_bytes = sum(len(c) for c in audio_buffer)
                    _duration_ms = (_total_bytes / 2) / sample_rate * 1000  # int16 = 2 bytes/sample
                    logger.debug(
                        f"[STREAM] chunk #{_chunk_count}: {len(audio_data)} B  |  "
                        f"buffer total: {_total_bytes} B  ({_duration_ms:.0f} ms @ {sample_rate} Hz)"
                    )

                # Handle JSON messages
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == MessageType.AUDIO_END:
                            _total_bytes = sum(len(c) for c in audio_buffer)
                            _duration_ms = (_total_bytes / 2) / sample_rate * 1000
                            logger.info(
                                f"[STREAM] AUDIO_END received — "
                                f"{_chunk_count} chunks, {_total_bytes} B, "
                                f"{_duration_ms:.0f} ms @ {sample_rate} Hz"
                            )
                            _chunk_count = 0  # reset for next utterance

                            # Process accumulated audio
                            if audio_buffer:
                                await self._process_audio(
                                    websocket, audio_buffer, sample_rate
                                )
                                audio_buffer = []

                        elif msg_type == MessageType.CANCEL:
                            logger.info(
                                f"[STREAM] CANCEL received after {_chunk_count} chunks "
                                f"({sum(len(c) for c in audio_buffer)} B buffered)"
                            )
                            audio_buffer = []
                            _chunk_count = 0
                            logger.debug("Audio processing cancelled")

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON message: {e}")
                        await websocket.send(
                            ErrorMessage(error=f"Invalid JSON: {e}").to_json()
                        )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.exception(f"Error handling client {client_addr}: {e}")
        finally:
            logger.info(f"Connection closed: {client_addr}")

    async def _process_audio(
        self,
        websocket: WebSocketServerProtocol,
        audio_chunks: list,
        sample_rate: int,
    ) -> None:
        """Process audio buffer and send results."""
        try:
            # Pick up entity-cache updates pushed by the per-minute HA sync
            # cron without restarting the server. Hotwords get refreshed too
            # so newly-added device names bias Whisper correctly.
            if self.intent_parser.maybe_reload():
                logger.info(
                    "ha_entities.json changed -- refreshed intent parser (%d entities, %d rooms)",
                    len(self.intent_parser.entity_names),
                    len(self.intent_parser.room_names),
                )
                if self.entities_path:
                    try:
                        self.transcriber.load_hotwords_from_entities(self.entities_path)
                    except Exception as e:
                        logger.warning("Failed to refresh hotwords: %s", e)

            # Combine audio chunks
            combined = b"".join(audio_chunks)

            if len(combined) < config.MIN_AUDIO_BYTES:
                logger.debug("Audio too short, skipping")
                return

            # Convert to numpy array
            audio = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

            # # Debug: save raw audio to WAV file so you can listen to it
            # import wave, time as _time
            # _debug_path = f"/tmp/debug_audio_{int(_time.time())}.wav"
            # with wave.open(_debug_path, "wb") as _wf:
            #     _wf.setnchannels(1)          # mono
            #     _wf.setsampwidth(2)           # 16-bit = 2 bytes
            #     _wf.setframerate(sample_rate)
            #     _wf.writeframes(combined)     # write original int16 bytes
            # logger.info(f"Debug audio saved to {_debug_path}")

            # Resample if necessary
            if sample_rate != config.SAMPLE_RATE:
                # Simple resampling (for production, use proper resampling)
                ratio = config.SAMPLE_RATE / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio,
                )

            # Transcribe
            logger.debug(f"Transcribing {len(audio)/config.SAMPLE_RATE:.2f}s of audio...")
            text, confidence = await asyncio.get_event_loop().run_in_executor(
                None, self.transcriber.transcribe, audio
            )
            logger.debug(f"Text: {text}")

            if not text:
                logger.debug("No transcription result")
                return

            # Send transcription
            transcription_msg = TranscriptionMessage(
                type=MessageType.TRANSCRIPTION,
                text=text,
                is_final=True,
                confidence=confidence,
            )
            await websocket.send(transcription_msg.to_json())

            # Parse intent(s) - may return multiple for chained commands
            intents = self.intent_parser.parse_all(text)

            for intent in intents:
                intent.confidence = min(intent.confidence, confidence)

                logger.info(
                    f"Parsed intent: {intent.intent} -> {intent.targets} "
                    f"(confidence: {intent.confidence:.2f})"
                )

                # Send intent
                await websocket.send(intent.to_json())

        except Exception as e:
            logger.exception(f"Error processing audio: {e}")
            await websocket.send(ErrorMessage(error=str(e)).to_json())


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Assistant Jetson Server")
    parser.add_argument(
        "--host", default=config.HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=config.PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--entities", type=str, help="Path to ha_entities.json file"
    )
    parser.add_argument(
        "--model", type=str, default=config.WHISPER_MODEL,
        help="Whisper model size (tiny.en, base.en, small.en)"
    )
    args = parser.parse_args()

    # Update config
    if args.model:
        config.WHISPER_MODEL = args.model

    # Create and start server
    server = VoiceAssistantServer(
        host=args.host,
        port=args.port,
        entities_path=args.entities,
    )

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    await server.start()

    # Wait for shutdown signal
    await stop_event.wait()
    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
