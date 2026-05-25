"""WebSocket client for the Jetson transcription/intent server."""

import asyncio
import json
import logging
from typing import List, Optional, Tuple

import websockets
from websockets.protocol import State

from . import config
from shared.protocol import IntentMessage, MessageType

logger = logging.getLogger(__name__)


class JetsonClient:
    """Connects to the Jetson server, streams audio, reads intents back.

    The connection is persistent across utterances. On failure we reconnect
    with exponential backoff before the next request.
    """

    def __init__(
        self,
        url: str = config.JETSON_URL,
        reconnect_delay: float = config.WS_RECONNECT_DELAY,
        max_reconnect_delay: float = config.WS_MAX_RECONNECT_DELAY,
    ):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self._ws = None
        self._current_delay = reconnect_delay

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.state is State.OPEN

    async def connect(self) -> bool:
        try:
            logger.info("Connecting to Jetson at %s", self.url)
            self._ws = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,
            )
            # The server sends a READY message on connect.
            raw = await asyncio.wait_for(self._ws.recv(), timeout=config.WS_RECEIVE_TIMEOUT)
            data = json.loads(raw)
            if data.get("type") != MessageType.READY:
                logger.warning("Unexpected handshake message: %s", data)
            logger.info("Connected to Jetson")
            self._current_delay = self.reconnect_delay
            return True
        except Exception as e:
            logger.warning("Jetson connect failed: %s", e)
            self._ws = None
            return False

    async def disconnect(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _ensure_connected(self) -> bool:
        if self.is_connected:
            return True
        if await self.connect():
            return True
        # One backoff attempt; main loop drives further retries.
        await asyncio.sleep(self._current_delay)
        self._current_delay = min(self._current_delay * 2, self.max_reconnect_delay)
        return await self.connect()

    async def process_audio(
        self,
        audio: bytes,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> Tuple[Optional[str], List[IntentMessage]]:
        """Send a full utterance and collect transcription + intent(s)."""
        if not await self._ensure_connected():
            return None, []

        ws = self._ws
        assert ws is not None
        try:
            # Send the audio as a single binary frame: 0x01 + sr(4 LE) + pcm.
            header = b"\x01" + sample_rate.to_bytes(4, "little")
            await ws.send(header + audio)
            await ws.send(json.dumps({"type": MessageType.AUDIO_END}))

            transcription: Optional[str] = None
            intents: List[IntentMessage] = []

            # The server emits TRANSCRIPTION then 1+ INTENT messages; drain
            # with a short idle timeout to allow chained-command intents.
            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=8.0 if not intents else 1.5,
                    )
                except asyncio.TimeoutError:
                    break

                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == MessageType.TRANSCRIPTION:
                    transcription = msg.get("text", "")
                elif mtype == MessageType.INTENT:
                    intents.append(IntentMessage.from_json(json.dumps(msg)))
                elif mtype == MessageType.ERROR:
                    logger.error("Jetson reported error: %s", msg.get("error"))
                    break

            return transcription, intents

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("Jetson connection closed mid-request: %s", e)
            self._ws = None
            return None, []
        except Exception as e:
            logger.exception("Jetson request failed: %s", e)
            return None, []
