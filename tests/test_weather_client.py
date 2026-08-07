import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather_client import geocode_location, normalize_alert, normalize_forecast_period


def test_geocode_location_parses_lat_lon_pair_without_network_call():
    lat, lon = geocode_location("41.8781,-87.6298")
    assert lat == 41.8781
    assert lon == -87.6298


def test_normalize_alert_combines_description_and_instruction():
    feature = {
        "id": "urn:oid:2.49.0.1.840.0.alert123",
        "properties": {
            "id": "urn:oid:2.49.0.1.840.0.alert123",
            "event": "Flash Flood Warning",
            "headline": "Flash Flood Warning issued",
            "description": "Heavy rain will cause flash flooding.",
            "instruction": "Move to higher ground immediately.",
            "sent": "2026-08-07T10:00:00-05:00",
            "effective": "2026-08-07T10:05:00-05:00",
        },
    }

    doc = normalize_alert(feature, "Chicago, IL")

    assert doc["id"] == "urn:oid:2.49.0.1.840.0.alert123"
    assert doc["source_type"] == "alert"
    assert doc["headline"] == "Flash Flood Warning"
    assert "Heavy rain will cause flash flooding." in doc["narrative_text"]
    assert "Move to higher ground immediately." in doc["narrative_text"]
    assert doc["location"] == "Chicago, IL"


def test_normalize_alert_omits_missing_instruction():
    feature = {
        "properties": {
            "id": "alert-no-instruction",
            "event": "Heat Advisory",
            "description": "Hot temperatures expected.",
            "instruction": None,
        }
    }

    doc = normalize_alert(feature, "Austin, TX")
    assert doc["narrative_text"] == "Hot temperatures expected."


def test_normalize_forecast_period_uses_detailed_forecast():
    period = {
        "name": "Tonight",
        "startTime": "2026-08-07T18:00:00-05:00",
        "detailedForecast": "Clear, with a low around 68. South wind around 5 mph.",
    }

    doc = normalize_forecast_period(period, "Austin, TX")

    assert doc["source_type"] == "forecast"
    assert doc["headline"] == "Tonight"
    assert doc["narrative_text"] == "Clear, with a low around 68. South wind around 5 mph."
    # stable dedup id derived from location + name + startTime
    doc2 = normalize_forecast_period(period, "Austin, TX")
    assert doc["id"] == doc2["id"]
