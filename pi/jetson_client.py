"""
WebSocket client for communicating with Jetson server.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Callable, Optional, Tuple

import websockets
from websockets.client import WebSocketClientProtocol

from . import config
from shared.protocol import (
    MessageType,
    IntentMessage,
    TranscriptionMessage,
    parse_message,
)

logger = logging.getLogger(__name__)


class JetsonClient:
    """
    WebSocket client for Jetson voice processing server.
    Handles streaming audio and receiving transcriptions/intents.
    """

    def __init__(
        self,
        url: str = config.JETSON_URL,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._reconnecting = False
        self._current_delay = reconnect_delay

    async def connect(self) -> bool:
        """
        Connect to Jetson server.

        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to Jetson server at {self.url}")
            self._ws = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )

            # Wait for ready message
            msg = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
            data = json.loads(msg)

            if data.get("type") == MessageType.READY:
                self._connected = True
                self._current_delay = self.reconnect_delay
                logger.info("Connected to Jetson server")
                return True
            else:
                logger.error(f"Unexpected message: {data}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def reconnect(self) -> bool:
        """Reconnect with exponential backoff."""
        if self._reconnecting:
            return False

        self._reconnecting = True

        while not self._connected:
            logger.info(f"Reconnecting in {self._current_delay}s...")
            await asyncio.sleep(self._current_delay)

            if await self.connect():
                self._reconnecting = False
                return True

            # Exponential backoff
            self._current_delay = min(
                self._current_delay * 2,
                self.max_reconnect_delay,
            )

        self._reconnecting = False
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """
        Send audio data to server.

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        # Binary format: 1 byte type + 4 bytes sample_rate + audio
        message = b'\x01' + sample_rate.to_bytes(4, 'little') + audio_data
        await self._ws.send(message)

    async def send_audio_end(self) -> None:
        """Signal end of audio stream."""
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        await self._ws.send(json.dumps({"type": MessageType.AUDIO_END}))

    async def send_cancel(self) -> None:
        """Cancel current processing."""
        if not self.is_connected:
            return

        await self._ws.send(json.dumps({"type": MessageType.CANCEL}))

    async def receive(self) -> Optional[dict]:
        """
        Receive a message from server.

        Returns:
            Parsed message dict or None
        """
        if not self.is_connected:
            return None

        try:
            msg = await self._ws.recv()
            return json.loads(msg)
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> Tuple[Optional[str], Optional[IntentMessage]]:
        """
        Send audio and wait for transcription and intent.

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate

        Returns:
            Tuple of (transcription text, intent message)
        """
        if not self.is_connected:
            if not await self.reconnect():
                return None, None

        transcription = None
        intent = None

        try:
            # Send audio in chunks for efficiency
            chunk_size = 32000  # ~1 second at 16kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.send_audio(chunk, sample_rate)
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.01)

            # Signal end of audio
            await self.send_audio_end()

            # Wait for responses
            while True:
                msg = await asyncio.wait_for(self.receive(), timeout=10.0)
                if msg is None:
                    break

                msg_type = msg.get("type")

                if msg_type == MessageType.TRANSCRIPTION:
                    transcription = msg.get("text", "")
                    logger.debug(f"Transcription: {transcription}")

                elif msg_type == MessageType.INTENT:
                    intent = IntentMessage.from_json(json.dumps(msg))
                    logger.debug(f"Intent: {intent.intent} -> {intent.targets}")
                    break  # Intent is the final response

                elif msg_type == MessageType.ERROR:
                    logger.error(f"Server error: {msg.get('error')}")
                    break

        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for response")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed during processing")
            self._connected = False
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

        return transcription, intent

    async def stream_audio(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        sample_rate: int = 16000,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_intent: Optional[Callable[[IntentMessage], None]] = None,
    ) -> Optional[IntentMessage]:
        """
        Stream audio to server and handle responses.

        Args:
            audio_generator: Async generator yielding audio chunks
            sample_rate: Audio sample rate
            on_transcription: Callback for transcription updates
            on_intent: Callback for intent results

        Returns:
            Final intent message
        """
        if not self.is_connected:
            if not await self.reconnect():
                return None

        # Task to send audio
        async def send_task():
            async for chunk in audio_generator:
                if not self.is_connected:
                    break
                await self.send_audio(chunk, sample_rate)
            await self.send_audio_end()

        # Task to receive responses
        intent_result = None

        async def receive_task():
            nonlocal intent_result

            while self.is_connected:
                try:
                    msg = await asyncio.wait_for(self.receive(), timeout=10.0)
                    if msg is None:
                        break

                    msg_type = msg.get("type")

                    if msg_type == MessageType.TRANSCRIPTION:
                        if on_transcription:
                            on_transcription(msg.get("text", ""))

                    elif msg_type == MessageType.PARTIAL_TRANSCRIPTION:
                        if on_transcription:
                            on_transcription(msg.get("text", ""))

                    elif msg_type == MessageType.INTENT:
                        intent_result = IntentMessage.from_json(json.dumps(msg))
                        if on_intent:
                            on_intent(intent_result)
                        break

                    elif msg_type == MessageType.ERROR:
                        logger.error(f"Server error: {msg.get('error')}")
                        break

                except asyncio.TimeoutError:
                    break

        # Run both tasks concurrently
        await asyncio.gather(send_task(), receive_task())

        return intent_result
