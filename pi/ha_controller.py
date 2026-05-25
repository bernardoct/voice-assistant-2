"""Home Assistant REST API client for executing parsed intents."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from . import config

logger = logging.getLogger(__name__)


class HomeAssistantController:
    """Calls Home Assistant services and resolves entities/rooms from the cache."""

    def __init__(
        self,
        url: str = config.HA_URL,
        token: str = config.HA_TOKEN,
        entities_path: str = config.HA_ENTITIES_PATH,
    ):
        if not token:
            raise ValueError("HA_TOKEN is required (set in .env or environment)")
        self.url = url.rstrip("/")
        self.token = token
        self.entities_path = entities_path

        self._session: Optional[aiohttp.ClientSession] = None
        self._entity_info: Dict[str, Dict[str, Any]] = {}
        self._room_to_entities: Dict[str, List[str]] = {}
        self._entities_mtime: float = 0.0

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        )
        self._reload_entities()
        logger.info("Home Assistant client ready at %s", self.url)

    async def disconnect(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _reload_entities(self) -> None:
        """(Re)load the entities JSON if its mtime has changed."""
        try:
            mtime = os.path.getmtime(self.entities_path)
        except OSError as e:
            logger.warning("Cannot stat entities file %s: %s", self.entities_path, e)
            return

        if mtime == self._entities_mtime and self._entity_info:
            return

        try:
            with open(self.entities_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to read %s: %s", self.entities_path, e)
            return

        info: Dict[str, Dict[str, Any]] = {}
        for ent in data.get("entities", []):
            eid = ent.get("entity_id")
            if eid:
                info[eid] = ent

        rooms: Dict[str, List[str]] = {}
        for area in data.get("areas", []):
            name = (area.get("name_norm") or area.get("name") or "").lower()
            ents = area.get("light_entities", [])
            if name:
                rooms[name] = ents

        self._entity_info = info
        self._room_to_entities = rooms
        self._entities_mtime = mtime
        logger.info(
            "Loaded %d entities and %d rooms from %s",
            len(info), len(rooms), self.entities_path,
        )

    def _supports_brightness(self, entity_id: str) -> bool:
        modes = self._entity_info.get(entity_id, {}).get("supported_color_modes") or []
        return "brightness" in modes or "color_temp_kelvin" in modes

    @staticmethod
    def _domain(entity_id: str) -> str:
        return entity_id.split(".", 1)[0]

    async def _call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._session is None:
            logger.error("HA session not initialized")
            return False
        payload: Dict[str, Any] = {"entity_id": entity_id}
        if extra:
            payload.update(extra)
        url = f"{self.url}/api/services/{domain}/{service}"
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.debug("HA %s.%s -> %s OK", domain, service, entity_id)
                    return True
                body = await resp.text()
                logger.error("HA %s.%s -> %s failed (%d): %s",
                             domain, service, entity_id, resp.status, body[:200])
                return False
        except Exception as e:
            logger.error("HA service call error: %s", e)
            return False

    async def turn_on(self, entity_id: str, brightness_pct: Optional[int] = None) -> bool:
        extra: Dict[str, Any] = {}
        if brightness_pct is not None and self._supports_brightness(entity_id):
            extra["brightness_pct"] = max(0, min(100, brightness_pct))
        return await self._call_service(self._domain(entity_id), "turn_on", entity_id, extra)

    async def turn_off(self, entity_id: str) -> bool:
        return await self._call_service(self._domain(entity_id), "turn_off", entity_id)

    async def toggle(self, entity_id: str) -> bool:
        return await self._call_service(self._domain(entity_id), "toggle", entity_id)

    async def set_brightness(self, entity_id: str, brightness_pct: int) -> bool:
        brightness_pct = max(0, min(100, brightness_pct))
        if not self._supports_brightness(entity_id):
            # Plain on/off device: emulate by turning on/off.
            return await (
                self.turn_on(entity_id) if brightness_pct > 0 else self.turn_off(entity_id)
            )
        return await self.turn_on(entity_id, brightness_pct)

    async def execute_intent(self, intent) -> bool:
        """Dispatch an IntentMessage to the right HA service call(s).

        Returns True if every targeted entity reported success.
        """
        # Pick up entity file changes (the cron updates it).
        self._reload_entities()

        if not intent.targets:
            logger.warning("Intent has no targets: %s", intent)
            return False

        logger.info(
            "Executing intent=%s targets=%s brightness=%s",
            intent.intent, intent.targets, intent.brightness,
        )

        if intent.intent == "turn_on":
            ops = [self.turn_on(e, intent.brightness) for e in intent.targets]
        elif intent.intent == "turn_off":
            ops = [self.turn_off(e) for e in intent.targets]
        elif intent.intent == "toggle":
            ops = [self.toggle(e) for e in intent.targets]
        elif intent.intent == "set_brightness":
            if intent.brightness is None:
                logger.warning("set_brightness intent missing brightness")
                return False
            ops = [self.set_brightness(e, intent.brightness) for e in intent.targets]
        else:
            logger.warning("Unhandled intent type: %s", intent.intent)
            return False

        results = await asyncio.gather(*ops, return_exceptions=True)
        ok = all(r is True for r in results)
        if not ok:
            logger.warning("Some HA calls failed: %s", results)
        return ok
