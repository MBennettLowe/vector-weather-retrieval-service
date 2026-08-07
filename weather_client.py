"""
weather_client.py

Client for the National Weather Service API (api.weather.gov). Mirrors
the shape of `massive_client.py` from the reference app: given a list of
locations, resolve them, fetch unstructured text (active alerts and
forecast narratives), and normalize everything into document records
ready to be upserted into `weather_documents`.

api.weather.gov is free and requires no API key, but its usage policy
asks for a descriptive User-Agent header identifying the application and
a contact method.
"""

import hashlib
import os
from datetime import datetime, timezone

import requests

NWS_BASE_URL = "https://api.weather.gov"
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "vector-weather-retrieval-service (contact: set NWS_USER_AGENT)"
)

REQUEST_TIMEOUT_SECONDS = 15


class WeatherClientError(Exception):
    """Raised when the NWS API (or geocoding step) cannot fulfill a request."""


def _headers():
    return {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/geo+json"}


def _stable_id(*parts):
    """Build a stable dedup key from arbitrary string parts."""
    raw = "|".join(str(p) for p in parts if p is not None)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def geocode_location(location):
    """Resolve a free-text location (e.g. "Chicago, IL") to (lat, lon).

    If `location` already looks like a "lat,lon" pair, parse it directly
    instead of calling the geocoder.
    """
    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                return lat, lon
            except ValueError:
                pass  # fall through to geocoding, e.g. "Chicago, IL"

    resp = requests.get(
        GEOCODE_URL,
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise WeatherClientError(f"Could not geocode location: {location!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def resolve_grid_point(lat, lon):
    """Resolve (lat, lon) to an NWS grid office + grid x/y via GET /points/{lat},{lon}."""
    url = f"{NWS_BASE_URL}/points/{lat:.4f},{lon:.4f}"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    props = resp.json()["properties"]
    return {
        "office": props["gridId"],
        "grid_x": props["gridX"],
        "grid_y": props["gridY"],
        "forecast_url": props["forecast"],
        "forecast_hourly_url": props["forecastHourly"],
        "state": props.get("relativeLocation", {})
        .get("properties", {})
        .get("state"),
    }


def fetch_active_alerts(state=None, point=None):
    """Fetch active alerts, either for a whole state (e.g. "IL") or a
    specific lat,lon point. Returns the raw list of alert features.
    """
    params = {}
    if state:
        params["area"] = state
    elif point:
        params["point"] = f"{point[0]:.4f},{point[1]:.4f}"
    else:
        raise WeatherClientError("fetch_active_alerts requires state or point")

    resp = requests.get(
        f"{NWS_BASE_URL}/alerts/active",
        params=params,
        headers=_headers(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json().get("features", [])


def fetch_forecast(forecast_url):
    """Fetch the multi-day narrative forecast for a resolved grid point."""
    resp = requests.get(forecast_url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json().get("properties", {}).get("periods", [])


def normalize_alert(feature, location):
    """Normalize a single NWS alert GeoJSON feature into a document record."""
    props = feature.get("properties", {})
    description = (props.get("description") or "").strip()
    instruction = (props.get("instruction") or "").strip()
    narrative = "\n\n".join(part for part in (description, instruction) if part)

    return {
        "id": props.get("id") or feature.get("id") or _stable_id(location, props.get("headline")),
        "location": location,
        "source_type": "alert",
        "headline": props.get("event") or props.get("headline"),
        "narrative_text": narrative,
        "issued_at": props.get("sent"),
        "effective_at": props.get("effective"),
        "payload": feature,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_forecast_period(period, location):
    """Normalize a single forecast period into a document record."""
    narrative = (period.get("detailedForecast") or "").strip()
    dedup_key = _stable_id(location, period.get("name"), period.get("startTime"))

    return {
        "id": dedup_key,
        "location": location,
        "source_type": "forecast",
        "headline": period.get("name"),
        "narrative_text": narrative,
        "issued_at": period.get("startTime"),
        "effective_at": period.get("startTime"),
        "payload": period,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def harvest_location(location, limit=50):
    """Harvest alerts + forecast periods for a single location and return
    a list of normalized document records (deduplicated by id, capped at
    `limit`).
    """
    lat, lon = geocode_location(location)
    grid = resolve_grid_point(lat, lon)

    documents = []

    alerts = fetch_active_alerts(point=(lat, lon))
    for feature in alerts:
        doc = normalize_alert(feature, location)
        if doc["narrative_text"]:
            documents.append(doc)

    periods = fetch_forecast(grid["forecast_url"])
    for period in periods:
        doc = normalize_forecast_period(period, location)
        if doc["narrative_text"]:
            documents.append(doc)

    # Dedup by id, preserving first occurrence, then cap to `limit`.
    seen = set()
    deduped = []
    for doc in documents:
        if doc["id"] in seen:
            continue
        seen.add(doc["id"])
        deduped.append(doc)

    return deduped[:limit]


def harvest_locations(locations, limit=50):
    """Harvest documents for multiple locations. Returns a flat list of
    normalized document records across all locations.
    """
    all_documents = []
    for location in locations:
        try:
            all_documents.extend(harvest_location(location, limit=limit))
        except (WeatherClientError, requests.RequestException) as exc:
            # One bad location shouldn't fail the whole sync; the caller
            # (the /weather/sync endpoint) surfaces a per-location error
            # count in its response.
            all_documents.append(
                {
                    "id": _stable_id(location, "error"),
                    "location": location,
                    "source_type": "error",
                    "headline": "harvest_failed",
                    "narrative_text": str(exc),
                    "issued_at": None,
                    "effective_at": None,
                    "payload": {"error": str(exc)},
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return all_documents
