"""
Home Assistant controller for light and switch control.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from . import config

logger = logging.getLogger(__name__)


class HomeAssistantController:
    """
    Controller for Home Assistant light and switch entities.
    """

    def __init__(
        self,
        url: str = config.HA_URL,
        token: str = config.HA_TOKEN,
        entities_path: str = config.HA_ENTITIES_PATH,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.entities_path = entities_path

        self._session: Optional[aiohttp.ClientSession] = None
        self._entities: Dict[str, Any] = {}
        self._entity_info: Dict[str, Dict] = {}
        self._room_to_entities: Dict[str, List[str]] = {}

    async def connect(self) -> None:
        """Initialize connection and load entities."""
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )

        # Load entity definitions
        await self._load_entities()

        logger.info(f"Connected to Home Assistant at {self.url}")

    async def disconnect(self) -> None:
        """Close connection."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _load_entities(self) -> None:
        """Load entity definitions from cache file."""
        try:
            with open(self.entities_path, "r") as f:
                self._entities = json.load(f)

            # Build entity info lookup
            for entity in self._entities.get("entities", []):
                entity_id = entity.get("entity_id")
                if entity_id:
                    self._entity_info[entity_id] = entity

            # Build room to entities mapping
            for area in self._entities.get("areas", []):
                room_name = area.get("name_norm", area.get("name", ""))
                entities = area.get("light_entities", [])
                if room_name:
                    self._room_to_entities[room_name] = entities

            logger.info(
                f"Loaded {len(self._entity_info)} entities, "
                f"{len(self._room_to_entities)} rooms"
            )

        except Exception as e:
            logger.error(f"Failed to load entities: {e}")

    async def _call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[Dict] = None,
    ) -> bool:
        """
        Call a Home Assistant service.

        Args:
            domain: Service domain (light, switch)
            service: Service name (turn_on, turn_off, toggle)
            entity_id: Entity to control
            data: Additional service data

        Returns:
            True if successful
        """
        if not self._session:
            logger.error("Not connected to Home Assistant")
            return False

        url = f"{self.url}/api/services/{domain}/{service}"
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.debug(f"Service call successful: {domain}.{service} -> {entity_id}")
                    return True
                else:
                    text = await resp.text()
                    logger.error(
                        f"Service call failed: {resp.status} - {text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error calling service: {e}")
            return False

    def _get_domain(self, entity_id: str) -> str:
        """Get domain from entity ID."""
        return entity_id.split(".")[0]

    def supports_brightness(self, entity_id: str) -> bool:
        """Check if entity supports brightness control."""
        info = self._entity_info.get(entity_id, {})
        modes = info.get("supported_color_modes") or []
        return "brightness" in modes or "color_temp_kelvin" in modes

    async def turn_on(
        self,
        entity_id: str,
        brightness_pct: Optional[int] = None,
    ) -> bool:
        """
        Turn on an entity.

        Args:
            entity_id: Entity to turn on
            brightness_pct: Optional brightness (0-100)

        Returns:
            True if successful
        """
        domain = self._get_domain(entity_id)
        data = {}

        if brightness_pct is not None and self.supports_brightness(entity_id):
            # Convert percentage to 0-255 range
            data["brightness"] = int(brightness_pct * 255 / 100)

        return await self._call_service(domain, "turn_on", entity_id, data)

    async def turn_off(self, entity_id: str) -> bool:
        """Turn off an entity."""
        domain = self._get_domain(entity_id)
        return await self._call_service(domain, "turn_off", entity_id)

    async def toggle(self, entity_id: str) -> bool:
        """Toggle an entity."""
        domain = self._get_domain(entity_id)
        return await self._call_service(domain, "toggle", entity_id)

    async def set_brightness(self, entity_id: str, brightness_pct: int) -> bool:
        """
        Set brightness of a light.

        Args:
            entity_id: Light entity
            brightness_pct: Brightness (0-100)

        Returns:
            True if successful
        """
        if not self.supports_brightness(entity_id):
            logger.warning(f"Entity {entity_id} doesn't support brightness")
            # Just turn on/off based on brightness value
            if brightness_pct > 0:
                return await self.turn_on(entity_id)
            else:
                return await self.turn_off(entity_id)

        return await self.turn_on(entity_id, brightness_pct)

    async def turn_on_room(
        self,
        room: str,
        brightness_pct: Optional[int] = None,
    ) -> List[bool]:
        """
        Turn on all lights in a room.

        Args:
            room: Room name
            brightness_pct: Optional brightness

        Returns:
            List of success results for each entity
        """
        entities = self._room_to_entities.get(room, [])
        if not entities:
            logger.warning(f"No entities found for room: {room}")
            return []

        logger.info(f"Turning on {len(entities)} lights in {room}")
        tasks = [self.turn_on(e, brightness_pct) for e in entities]
        return await asyncio.gather(*tasks)

    async def turn_off_room(self, room: str) -> List[bool]:
        """Turn off all lights in a room."""
        entities = self._room_to_entities.get(room, [])
        if not entities:
            logger.warning(f"No entities found for room: {room}")
            return []

        logger.info(f"Turning off {len(entities)} lights in {room}")
        tasks = [self.turn_off(e) for e in entities]
        return await asyncio.gather(*tasks)

    async def set_room_brightness(
        self,
        room: str,
        brightness_pct: int,
    ) -> List[bool]:
        """Set brightness for all lights in a room."""
        entities = self._room_to_entities.get(room, [])
        if not entities:
            logger.warning(f"No entities found for room: {room}")
            return []

        logger.info(f"Setting brightness to {brightness_pct}% for {len(entities)} lights in {room}")
        tasks = [self.set_brightness(e, brightness_pct) for e in entities]
        return await asyncio.gather(*tasks)

    async def get_state(self, entity_id: str) -> Optional[Dict]:
        """Get current state of an entity."""
        if not self._session:
            return None

        url = f"{self.url}/api/states/{entity_id}"

        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None

    def get_room_entities(self, room: str) -> List[str]:
        """Get list of entities in a room."""
        return self._room_to_entities.get(room, [])

    def get_rooms(self) -> List[str]:
        """Get list of available rooms."""
        return list(self._room_to_entities.keys())

    def find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by friendly name."""
        name_lower = name.lower()

        # Check friendly name index
        index = self._entities.get("index", {}).get("by_friendly_norm", {})
        if name_lower in index:
            entities = index[name_lower]
            # Prefer light entities
            for e in entities:
                if e.startswith("light."):
                    return e
            return entities[0] if entities else None

        # Partial match
        for friendly, entities in index.items():
            if name_lower in friendly or friendly in name_lower:
                for e in entities:
                    if e.startswith("light."):
                        return e
                return entities[0] if entities else None

        return None
