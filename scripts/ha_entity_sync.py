#!/usr/bin/env python3
"""Snapshot Home Assistant lights/switches + areas into a JSON cache file.

The Pi voice assistant and Jetson intent parser read this file to resolve
"living room", "office floor lamp", etc. to real entity IDs. Run periodically
so renames/added devices are picked up without redeploying code.

Configuration: env vars (also loads .env in the repo root).
    HA_URL              Required. e.g. http://192.168.1.203:8123
    HA_TOKEN            Required. Long-lived access token.
    HA_ENTITIES_PATH    Output path. Default: ~/.cache/ha_entities.json

Recommended cron entry (every minute):
    * * * * * /home/bernardoct/voice-assistant-2/.venv/bin/python \\
        /home/bernardoct/voice-assistant-2/scripts/ha_entity_sync.py \\
        >> /tmp/ha_entity_sync.log 2>&1
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _load_dotenv() -> None:
    """Populate os.environ from the project root .env if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def norm(s: str) -> str:
    """Normalize a name for fuzzy matching (lower, alphanumeric, single spaces)."""
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ha_get(ha_url: str, ha_token: str, path: str, timeout: int = 10) -> Any:
    r = requests.get(
        f"{ha_url}{path}",
        headers={"Authorization": f"Bearer {ha_token}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def ha_post(ha_url: str, ha_token: str, path: str, payload: dict, timeout: int = 10) -> str:
    r = requests.post(
        f"{ha_url}{path}",
        headers={"Authorization": f"Bearer {ha_token}"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def ha_template(ha_url: str, ha_token: str, template: str) -> Any:
    """Render a Jinja template via HA's /api/template endpoint.

    HA returns Python-repr for lists/dicts, not JSON, so we try both.
    """
    raw = ha_post(ha_url, ha_token, "/api/template", {"template": template}).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw


def _load_areas(ha_url: str, ha_token: str) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Try the documented REST endpoints first, fall back to template rendering."""
    areas: List[Dict[str, Any]] = []
    area_names: List[str] = []
    errors: List[str] = []

    for path in ("/api/areas", "/api/config/area_registry"):
        try:
            areas = ha_get(ha_url, ha_token, path)
            return areas, area_names, errors
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    try:
        area_names = ha_template(ha_url, ha_token, "{{ areas() }}") or []
    except Exception as exc:
        errors.append(f"template areas(): {exc}")

    return areas, area_names, errors


def _build_entities(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in states:
        entity_id = st.get("entity_id", "")
        if not (entity_id.startswith("light.") or entity_id.startswith("switch.")):
            continue

        attrs = st.get("attributes", {}) or {}
        friendly = attrs.get("friendly_name") or entity_id

        record = {
            "entity_id": entity_id,
            "domain": entity_id.split(".", 1)[0],
            "friendly_name": friendly,
            "friendly_norm": norm(str(friendly)),
            "entity_norm": norm(entity_id),
            "device_class": attrs.get("device_class"),
            "supported_color_modes": attrs.get("supported_color_modes"),
        }

        # HA exposes "brightness" via attrs even when supported_color_modes
        # is empty; keep our schema honest about what we can dim.
        if "brightness" in attrs:
            supported = record["supported_color_modes"]
            if not supported:
                record["supported_color_modes"] = ["brightness"]
            elif "brightness" not in supported:
                supported.append("brightness")

        # Normalize the deprecated "color_temp" mode name.
        scm = record["supported_color_modes"] or []
        if "color_temp" in scm:
            scm.remove("color_temp")
            scm.append("color_temp_kelvin")

        out.append(record)
    return out


def main() -> int:
    _load_dotenv()

    ha_url = os.environ.get("HA_URL", "http://192.168.1.203:8123").rstrip("/")
    ha_token = os.environ.get("HA_TOKEN")
    out_path = Path(
        os.environ.get("HA_ENTITIES_PATH")
        or str(Path.home() / ".cache" / "ha_entities.json")
    )

    if not ha_token:
        print("[ha_entity_sync] ERROR: HA_TOKEN not set", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    states = ha_get(ha_url, ha_token, "/api/states")
    entities = _build_entities(states)

    areas, area_names, area_errors = _load_areas(ha_url, ha_token)
    if not areas and not area_names and area_errors:
        print(
            "[ha_entity_sync] WARN: failed to load areas: " + "; ".join(area_errors),
            file=sys.stderr,
        )

    by_friendly: Dict[str, List[str]] = {}
    by_entity: Dict[str, str] = {}
    for e in entities:
        by_entity[e["entity_norm"]] = e["entity_id"]
        by_friendly.setdefault(e["friendly_norm"], []).append(e["entity_id"])

    payload: Dict[str, Any] = {
        "generated_at": time.time(),
        "ha_url": ha_url,
        "counts": {
            "lights": sum(1 for e in entities if e["domain"] == "light"),
            "switches": sum(1 for e in entities if e["domain"] == "switch"),
            "total": len(entities),
        },
        "areas": [],
        "entities": entities,
        "index": {
            "by_friendly_norm": by_friendly,
            "by_entity_norm": by_entity,
        },
    }

    if areas:
        # Build name -> entity_ids via template, since the area registry
        # itself doesn't include the entity assignment.
        valid_entities = {e["entity_id"]: e for e in entities}
        for area in areas:
            name = area.get("name")
            if not name:
                continue
            try:
                entity_ids = ha_template(
                    ha_url, ha_token, f"{{{{ area_entities({name!r}) }}}}"
                ) or []
            except Exception as exc:
                print(
                    f"[ha_entity_sync] WARN: failed to load entities for area '{name}': {exc}",
                    file=sys.stderr,
                )
                entity_ids = []
            light_entities = [
                eid for eid in entity_ids
                if (valid_entities.get(eid) or {}).get("domain") == "light"
            ]
            payload["areas"].append({
                "area_id": area.get("area_id"),
                "name": name,
                "name_norm": norm(str(name)),
                "light_entities": light_entities,
            })
    elif area_names:
        valid_entities = {e["entity_id"]: e for e in entities}
        for name in area_names:
            if not name:
                continue
            try:
                entity_ids = ha_template(
                    ha_url, ha_token, f"{{{{ area_entities({name!r}) }}}}"
                ) or []
            except Exception as exc:
                print(
                    f"[ha_entity_sync] WARN: failed to load entities for area '{name}': {exc}",
                    file=sys.stderr,
                )
                entity_ids = []
            light_entities = [
                eid for eid in entity_ids
                if (valid_entities.get(eid) or {}).get("domain") == "light"
            ]
            payload["areas"].append({
                "area_id": None,
                "name": name,
                "name_norm": norm(str(name)),
                "light_entities": light_entities,
            })

    # Atomic write so concurrent readers never see a half-written file.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ha_entity_sync] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
