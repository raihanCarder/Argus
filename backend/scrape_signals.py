"""Signal discovery pipeline: Serper (search) → Jina (read) → Gemini (extract).

Fallback chain:
  1. Serper + Jina + Gemini  (real search → real content → structured extraction)
  2. Gemini direct generation (LLM knowledge, no URL verification)
  3. Sources.json + Jina     (known gov pages → read → extract)
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

DATA_DIR = Path(__file__).parent / "data"
SOURCES_PATH = DATA_DIR / "sources.json"
SIGNALS_PATH = DATA_DIR / "signals.json"

# ── API keys ──────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_KEY", "") or os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_KEY", "") or os.getenv("JINA_API_KEY", "")

# ── Gemini config ─────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_SCRAPE_TEMPERATURE = float(os.getenv("GEMINI_SCRAPE_TEMPERATURE", "0.2"))
GEMINI_SCRAPE_MAX_TOKENS = int(os.getenv("GEMINI_SCRAPE_MAX_TOKENS", "8192"))

# ── Pipeline limits ───────────────────────────────────────────────
SCRAPE_MAX_QUERIES = int(os.getenv("SCRAPE_MAX_QUERIES", "3"))
SERPER_MAX_RESULTS_PER_QUERY = int(os.getenv("SERPER_MAX_RESULTS_PER_QUERY", "10"))
SCRAPE_MAX_URLS_TO_READ = int(os.getenv("SCRAPE_MAX_URLS_TO_READ", "6"))
JINA_MAX_CONTENT_CHARS = int(os.getenv("JINA_MAX_CONTENT_CHARS", "12000"))
SCRAPE_MAX_SIGNALS = int(os.getenv("SCRAPE_MAX_SIGNALS", "10"))
SCRAPE_TIME_BUDGET_SECONDS = float(os.getenv("SCRAPE_TIME_BUDGET_SECONDS", "55"))
GEMINI_DIRECT_SIGNALS = int(os.getenv("GEMINI_DIRECT_SIGNALS", "8"))

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ── Diagnostics ───────────────────────────────────────────────────
LAST_COLLECT_META: dict[str, Any] = {"mode": "unknown", "error": None}

BLOCKED_HOSTS = frozenset({
    "facebook.com", "twitter.com", "x.com", "linkedin.com",
    "instagram.com", "youtube.com", "tiktok.com", "reddit.com",
    "pinterest.com", "amazon.com", "ebay.com", "wikipedia.org",
})


def get_last_collect_meta() -> dict[str, Any]:
    return dict(LAST_COLLECT_META)


# ── Utilities ─────────────────────────────────────────────────────

def _is_blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS)


def _extract_terms(text: str, limit: int = 8) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "from", "into", "your", "this",
        "have", "build", "building", "including", "we", "our", "are", "to",
        "of", "in", "on", "a", "an", "also", "can", "help", "use", "using",
        "platforms", "platform", "solutions", "solution", "based", "provide",
        "services", "service", "real", "time", "powered",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
    terms: list[str] = []
    for t in tokens:
        if t in stopwords or len(t) < 3:
            continue
        if t not in terms:
            terms.append(t)
        if len(terms) >= limit:
            break
    return terms


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned, count=1).strip()
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("signals", "matches", "items", "results"):
                if isinstance(data.get(key), list):
                    return [item for item in data[key] if isinstance(item, dict)]
            return []
    except json.JSONDecodeError as e:
        logger.debug("_parse_json_array: initial parse failed: %s (text ends: %r)", e, cleaned[-80:] if len(cleaned) > 80 else cleaned)
        # Try to find a JSON array in the text
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                # Try to fix truncated JSON by closing brackets
                fragment = cleaned[start : end + 1]
                # Count open/close braces
                open_braces = fragment.count("{") - fragment.count("}")
                open_brackets = fragment.count("[") - fragment.count("]")
                fragment += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
                try:
                    data = json.loads(fragment)
                    if isinstance(data, list):
                        return [item for item in data if isinstance(item, dict)]
                except Exception:
                    pass
    except Exception as e:
        logger.debug("_parse_json_array: unexpected error: %s", e)
    return []


def _normalize_signal(raw: dict[str, Any], startup_hint: str) -> dict[str, Any] | None:
    title = str(raw.get("title") or "").strip()
    description = str(raw.get("description") or "").strip()
    source_url = str(raw.get("source_url") or "").strip()

    if not title or not description:
        return None
    if source_url and _is_blocked(source_url):
        return None

    keywords = raw.get("keywords")
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]

    # Enrich with startup terms so keyword matching in main.py works
    for term in _extract_terms(startup_hint):
        if term not in keywords and len(keywords) < 12:
            keywords.append(term)

    stakeholders = raw.get("stakeholders")
    if not isinstance(stakeholders, list):
        stakeholders = []
    stakeholders = [str(s).strip() for s in stakeholders if str(s).strip()]

    budget = raw.get("budget")
    if not isinstance(budget, (int, float)):
        budget = None

    lat, lng = raw.get("lat"), raw.get("lng")
    try:
        lat, lng = float(lat), float(lng)
    except Exception:
        lat = lng = None

    return {
        "title": title[:200],
        "description": description[:500],
        "category": str(raw.get("category") or "Government Initiative")[:100],
        "budget": budget,
        "timeline": str(raw.get("timeline") or "")[:150],
        "keywords": keywords[:12],
        "stakeholders": stakeholders[:6],
        "source_url": source_url,
        "source_type": str(raw.get("source_type") or "discovered"),
        "source_title": str(raw.get("source_title") or "")[:200],
        "city": str(raw.get("city") or "")[:100],
        "region": str(raw.get("region") or "")[:150],
        "country": str(raw.get("country") or "Unknown"),
        "lat": lat,
        "lng": lng,
        "_fallback": False,
        "_source_verified": True,
    }


# ── Geocoding ─────────────────────────────────────────────────────
# Lookup table for major government/capital cities.  Used to resolve
# city+country → lat/lng so every signal can appear on the 3D globe.

_CITY_COORDS: dict[str, tuple[float, float]] = {
    # United States
    "washington": (38.9072, -77.0369),
    "washington dc": (38.9072, -77.0369),
    "washington, dc": (38.9072, -77.0369),
    "new york": (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "austin": (30.2672, -97.7431),
    "denver": (39.7392, -104.9903),
    "boston": (42.3601, -71.0589),
    "atlanta": (33.7490, -84.3880),
    "dallas": (32.7767, -96.7970),
    "phoenix": (33.4484, -112.0740),
    "san diego": (32.7157, -117.1611),
    "portland": (45.5152, -122.6784),
    "miami": (25.7617, -80.1918),
    "philadelphia": (39.9526, -75.1652),
    "san jose": (37.3382, -121.8863),
    "baltimore": (39.2904, -76.6122),
    "sacramento": (38.5816, -121.4944),
    "raleigh": (35.7796, -78.6382),
    "columbus": (39.9612, -82.9988),
    "indianapolis": (39.7684, -86.1581),
    "nashville": (36.1627, -86.7816),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572),
    "pittsburgh": (40.4406, -79.9959),
    "las vegas": (36.1699, -115.1398),
    "honolulu": (21.3069, -157.8583),
    "anchorage": (61.2181, -149.9003),
    # Canada
    "ottawa": (45.4215, -75.6972),
    "toronto": (43.6532, -79.3832),
    "vancouver": (49.2827, -123.1207),
    "montreal": (45.5017, -73.5673),
    "calgary": (51.0447, -114.0719),
    "edmonton": (53.5461, -113.4938),
    "winnipeg": (49.8951, -97.1384),
    "quebec city": (46.8139, -71.2080),
    "halifax": (44.6488, -63.5752),
    "victoria": (48.4284, -123.3656),
    # United Kingdom
    "london": (51.5074, -0.1278),
    "edinburgh": (55.9533, -3.1883),
    "manchester": (53.4808, -2.2426),
    "birmingham": (52.4862, -1.8904),
    "bristol": (51.4545, -2.5879),
    "leeds": (53.8008, -1.5491),
    "glasgow": (55.8642, -4.2518),
    "cardiff": (51.4816, -3.1791),
    "belfast": (54.5973, -5.9301),
    # Europe
    "brussels": (50.8503, 4.3517),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "amsterdam": (52.3676, 4.9041),
    "rome": (41.9028, 12.4964),
    "madrid": (40.4168, -3.7038),
    "lisbon": (38.7223, -9.1393),
    "vienna": (48.2082, 16.3738),
    "copenhagen": (55.6761, 12.5683),
    "stockholm": (59.3293, 18.0686),
    "oslo": (59.9139, 10.7522),
    "helsinki": (60.1699, 24.9384),
    "dublin": (53.3498, -6.2603),
    "zurich": (47.3769, 8.5417),
    "geneva": (46.2044, 6.1432),
    "prague": (50.0755, 14.4378),
    "warsaw": (52.2297, 21.0122),
    "bucharest": (44.4268, 26.1025),
    "athens": (37.9838, 23.7275),
    # Australia / New Zealand
    "canberra": (-35.2809, 149.1300),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "wellington": (-41.2865, 174.7762),
    "auckland": (-36.8485, 174.7633),
    # Asia
    "tokyo": (35.6762, 139.6503),
    "singapore": (1.3521, 103.8198),
    "seoul": (37.5665, 126.9780),
    "new delhi": (28.6139, 77.2090),
    "beijing": (39.9042, 116.4074),
    # Country-level fallback centres
    "united states": (39.8283, -98.5795),
    "usa": (39.8283, -98.5795),
    "canada": (56.1304, -106.3468),
    "united kingdom": (55.3781, -3.4360),
    "uk": (55.3781, -3.4360),
    "france": (46.6034, 1.8883),
    "germany": (51.1657, 10.4515),
    "netherlands": (52.1326, 5.2913),
    "australia": (-25.2744, 133.7751),
    "european union": (50.8503, 4.3517),
    "eu": (50.8503, 4.3517),
}


# US state → capital city mapping for region-level geocoding
_STATE_CAPITALS: dict[str, str] = {
    "alabama": "montgomery", "alaska": "anchorage", "arizona": "phoenix",
    "arkansas": "little rock", "california": "sacramento", "colorado": "denver",
    "connecticut": "hartford", "delaware": "dover", "florida": "miami",
    "georgia": "atlanta", "hawaii": "honolulu", "idaho": "boise",
    "illinois": "chicago", "indiana": "indianapolis", "iowa": "des moines",
    "kansas": "topeka", "kentucky": "louisville", "louisiana": "new orleans",
    "maine": "portland", "maryland": "baltimore", "massachusetts": "boston",
    "michigan": "detroit", "minnesota": "minneapolis", "mississippi": "jackson",
    "missouri": "kansas city", "montana": "billings", "nebraska": "omaha",
    "nevada": "las vegas", "new hampshire": "concord", "new jersey": "newark",
    "new mexico": "albuquerque", "new york": "new york", "north carolina": "raleigh",
    "north dakota": "fargo", "ohio": "columbus", "oklahoma": "oklahoma city",
    "oregon": "portland", "pennsylvania": "philadelphia", "rhode island": "providence",
    "south carolina": "charleston", "south dakota": "sioux falls", "tennessee": "nashville",
    "texas": "austin", "utah": "salt lake city", "vermont": "burlington",
    "virginia": "richmond", "washington": "seattle", "west virginia": "charleston",
    "wisconsin": "milwaukee", "wyoming": "cheyenne", "district of columbia": "washington dc",
    "dc": "washington dc",
}
# Add common abbreviations
_STATE_CAPITALS.update({
    "al": "montgomery", "ak": "anchorage", "az": "phoenix", "ar": "little rock",
    "ca": "sacramento", "co": "denver", "ct": "hartford", "de": "dover",
    "fl": "miami", "ga": "atlanta", "hi": "honolulu", "id": "boise",
    "il": "chicago", "in": "indianapolis", "ia": "des moines", "ks": "topeka",
    "ky": "louisville", "la": "new orleans", "me": "portland", "md": "baltimore",
    "ma": "boston", "mi": "detroit", "mn": "minneapolis", "ms": "jackson",
    "mo": "kansas city", "mt": "billings", "ne": "omaha", "nv": "las vegas",
    "nh": "concord", "nj": "newark", "nm": "albuquerque", "ny": "new york",
    "nc": "raleigh", "nd": "fargo", "oh": "columbus", "ok": "oklahoma city",
    "or": "portland", "pa": "philadelphia", "ri": "providence", "sc": "charleston",
    "sd": "sioux falls", "tn": "nashville", "tx": "austin", "ut": "salt lake city",
    "vt": "burlington", "va": "richmond", "wa": "seattle", "wv": "charleston",
    "wi": "milwaukee", "wy": "cheyenne",
})


def _geocode(city: str, region: str, country: str) -> tuple[float, float] | None:
    """Resolve city / region / country to (lat, lng) via lookup table."""
    for candidate in [
        f"{city}, {region}".lower().strip(", "),
        city.lower().strip(),
        region.lower().strip(),
        country.lower().strip(),
    ]:
        if candidate and candidate in _CITY_COORDS:
            return _CITY_COORDS[candidate]
    # Try resolving US state → major city
    region_lower = region.lower().strip()
    if region_lower in _STATE_CAPITALS:
        capital = _STATE_CAPITALS[region_lower]
        if capital in _CITY_COORDS:
            return _CITY_COORDS[capital]
    return None


_DEMONYM_TO_COUNTRY: dict[str, str] = {
    "american": "united states", "canadian": "canada", "british": "united kingdom",
    "australian": "australia", "french": "france", "german": "germany",
    "dutch": "netherlands", "european": "european union", "italian": "italy",
    "spanish": "spain", "portuguese": "portugal", "swiss": "switzerland",
    "swedish": "sweden", "norwegian": "norway", "danish": "denmark",
    "finnish": "finland", "irish": "ireland", "japanese": "japan",
    "singaporean": "singapore", "korean": "south korea", "indian": "india",
    "chinese": "china", "scottish": "united kingdom", "welsh": "united kingdom",
    "new zealand": "new zealand", "kiwi": "new zealand",
}


def _extract_location_from_text(title: str, description: str) -> dict[str, str]:
    """Try to infer city/region/country from signal title and description text."""
    text = f"{title} {description}".lower()
    found: dict[str, str] = {}

    # Check for city names in text (only cities with 4+ char names to avoid false positives)
    for city_name in _CITY_COORDS:
        if len(city_name) >= 4 and f" {city_name}" in f" {text}":
            # Skip country-level entries
            if city_name in ("united states", "usa", "canada", "united kingdom", "uk",
                             "france", "germany", "netherlands", "australia",
                             "european union", "eu"):
                continue
            found["city"] = city_name.title()
            return found

    # Check for US state names
    for state_name, capital in _STATE_CAPITALS.items():
        if len(state_name) >= 4 and f" {state_name}" in f" {text}":
            found["region"] = state_name.title()
            found["country"] = "United States"
            if capital in _CITY_COORDS:
                found["city"] = capital.title()
            return found

    # Check for country names (full names)
    country_names = ["united states", "united kingdom", "canada", "australia",
                     "france", "germany", "netherlands", "european union",
                     "new zealand", "singapore", "japan", "south korea",
                     "india", "china", "italy", "spain", "portugal",
                     "switzerland", "sweden", "norway", "denmark",
                     "finland", "ireland"]
    for cname in country_names:
        if cname in text:
            found["country"] = cname.title()
            return found

    # Check for demonyms / adjective forms (e.g. "Canadian" → Canada)
    for demonym, country in _DEMONYM_TO_COUNTRY.items():
        if demonym in text:
            found["country"] = country.title()
            return found

    return found


def _geocode_with_gemini(locations: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """Batch-geocode a list of {city, region, country} dicts using Gemini."""
    if not GOOGLE_API_KEY or not locations:
        return {}

    location_lines = "\n".join(
        f'{i + 1}. {loc["city"]}, {loc["region"]}, {loc["country"]}'
        for i, loc in enumerate(locations)
    )
    prompt = (
        "Return lat/lng coordinates for each location below as a JSON object "
        "mapping the line number (as string) to [lat, lng].\n\n"
        f"{location_lines}\n\n"
        'Example: {{"1": [51.5074, -0.1278], "2": [40.7128, -74.006]}'
    )
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text, count=1).strip()
            text = re.sub(r"\n?```\s*$", "", text).strip()
        data = json.loads(text)
        result: dict[str, tuple[float, float]] = {}
        for key, coords in data.items():
            if isinstance(coords, list) and len(coords) == 2:
                result[key] = (float(coords[0]), float(coords[1]))
        return result
    except Exception as exc:
        logger.warning("_geocode_with_gemini failed: %s", exc)
        return {}


def _ensure_coordinates(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-process signals to ensure every signal has valid lat/lng.

    1. If city/region/country is empty, try to infer from title + description.
    2. Try the built-in lookup table.
    3. Batch any remaining unknowns to Gemini.
    4. As a last resort, assign a default based on country.
    """
    needs_gemini: list[tuple[int, dict[str, str]]] = []

    for idx, sig in enumerate(signals):
        lat, lng = sig.get("lat"), sig.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)) and lat != 0 and lng != 0:
            continue  # already geocoded

        city = str(sig.get("city") or "")
        region = str(sig.get("region") or "")
        country = str(sig.get("country") or "")

        # If location fields are weak, try to extract from text
        if not city and not region and country in ("", "Unknown"):
            inferred = _extract_location_from_text(
                str(sig.get("title") or ""),
                str(sig.get("description") or ""),
            )
            if inferred:
                if inferred.get("city") and not city:
                    sig["city"] = city = inferred["city"]
                if inferred.get("region") and not region:
                    sig["region"] = region = inferred["region"]
                if inferred.get("country") and country in ("", "Unknown"):
                    sig["country"] = country = inferred["country"]
                logger.info("_ensure_coordinates: inferred location %s for '%s'", inferred, sig.get("title", "")[:60])

        coords = _geocode(city, region, country)
        if coords:
            sig["lat"], sig["lng"] = coords
            logger.info("_ensure_coordinates: geocoded '%s' → %s", sig.get("title", "")[:60], coords)
        else:
            needs_gemini.append((idx, {"city": city, "region": region, "country": country}))

    if needs_gemini:
        locs = [loc for _, loc in needs_gemini]
        gemini_coords = _geocode_with_gemini(locs)
        for i, (idx, _) in enumerate(needs_gemini):
            coords = gemini_coords.get(str(i + 1))
            if coords:
                signals[idx]["lat"], signals[idx]["lng"] = coords
                logger.info("_ensure_coordinates: Gemini geocoded signal %d → %s", idx, coords)
            else:
                # Ultimate fallback: try country centre, else US centre
                country_lower = str(signals[idx].get("country") or "").lower()
                fallback_coords = _CITY_COORDS.get(country_lower, (39.83, -98.58))
                signals[idx]["lat"] = fallback_coords[0]
                signals[idx]["lng"] = fallback_coords[1]
                logger.info("_ensure_coordinates: fallback for signal %d (%s) → %s", idx, country_lower, fallback_coords)

    return signals


def _dedup_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for sig in signals:
        key = f"{sig.get('title', '').lower()}|{sig.get('source_url', '')}"
        if key not in seen:
            seen.add(key)
            out.append(sig)
    return out


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Search with Serper
# ═══════════════════════════════════════════════════════════════════

def build_search_queries(startup_hint: str) -> list[str]:
    """Build targeted queries from startup description + gov procurement terms."""
    terms = _extract_terms(startup_hint, limit=6)
    core = " ".join(terms[:4])

    queries = [
        f"site:gov OR site:gov.uk OR site:gc.ca {core} initiative OR modernization OR procurement",
        f"government {core} RFP OR solicitation OR contract award 2025 OR 2026",
        f"city OR state OR federal {core} budget OR funding OR initiative",
    ]
    return queries[:SCRAPE_MAX_QUERIES]


def search_serper(query: str) -> list[dict[str, str]]:
    """Search with Serper and return organic results."""
    if not SERPER_API_KEY:
        logger.warning("search_serper: no SERPER_KEY configured")
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": SERPER_MAX_RESULTS_PER_QUERY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[dict[str, str]] = []
        for item in data.get("organic", []):
            link = item.get("link", "")
            if link and not _is_blocked(link):
                results.append({
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": item.get("snippet", ""),
                })
        logger.info("search_serper: query=%r → %d results", query[:60], len(results))
        return results
    except Exception as exc:
        logger.error("search_serper failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Read with Jina
# ═══════════════════════════════════════════════════════════════════

def _clean_markdown(raw: str) -> str:
    """Strip navigation, boilerplate, and image refs from Jina markdown."""
    lines = raw.split("\n")
    cleaned: list[str] = []

    # Boilerplate phrases to skip entirely
    skip_phrases = [
        "skip to", ".gov website", "official government", "secure .gov",
        "https:// means", "share sensitive", "an official website",
        "here's how you know", "cookie", "accept all", "javascript",
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        # Skip boilerplate lines
        if any(phrase in low for phrase in skip_phrases):
            continue
        # Skip image refs
        if stripped.startswith("!["):
            continue
        # Skip short link-only lines (navigation)
        if stripped.startswith("[") and stripped.endswith(")") and len(stripped) < 100:
            continue
        if stripped.startswith("*   [") and len(stripped) < 120:
            continue
        # Skip lines that are just URLs
        if stripped.startswith("http://") or stripped.startswith("https://"):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def read_jina(url: str) -> dict[str, str] | None:
    """Read a URL with Jina Reader API and return clean markdown content."""
    if not JINA_API_KEY:
        logger.warning("read_jina: no JINA_KEY configured")
        return None
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Accept": "application/json",
                "X-Return-Format": "markdown",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_content = data.get("data", {}).get("content", "")
        title = data.get("data", {}).get("title", "")
        content = _clean_markdown(raw_content)
        if not content or len(content.strip()) < 50:
            logger.info("read_jina: %s → content too short (%d chars after clean)", url[:60], len(content))
            return None
        logger.info("read_jina: %s → %d chars (raw %d), title=%r", url[:60], len(content), len(raw_content), title[:60])
        return {
            "title": title,
            "content": content[:JINA_MAX_CONTENT_CHARS],
            "url": url,
        }
    except Exception as exc:
        logger.error("read_jina failed for %s: %s", url[:60], exc)
        return None


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Extract signals with Gemini
# ═══════════════════════════════════════════════════════════════════

def _build_extraction_prompt(doc: dict[str, str], startup_hint: str) -> str:
    url = doc["url"]
    title = doc["title"]
    content = doc["content"]
    parts = [
        f"Extract 1-3 government procurement signals from this page relevant to a startup that builds: {startup_hint}",
        "",
        f"Page title: {title}",
        f"Page URL: {url}",
        f"Page content:\n{content}",
        "",
        "Return a JSON array. Each object must have: title, description (2-3 sentences about the opportunity), "
        "category, budget (number or null), timeline, keywords (array of technical terms from the page), "
        f'stakeholders (array), source_url (use "{url}"), source_title, source_type, city, region, country, lat, lng.',
        "",
        "LOCATION IS CRITICAL — every signal will be plotted on a 3D globe.",
        "Carefully extract the city, region/state, and country from the URL domain, page content, or title.",
        "For example: a .gov.uk URL → country='United Kingdom'; a URL containing austin.gov → city='Austin', country='United States'.",
        "Set lat/lng to the numeric coordinates of that city. If you cannot determine exact coordinates, "
        "set lat/lng to the approximate centre of the country.",
        "NEVER leave city AND country both empty. At minimum, infer the country from the URL TLD.",
        "",
        "Be generous - include any government initiative, procurement, RFP, or modernization effort "
        "even if only tangentially related. Return [] only if truly no government technology content.",
    ]
    return "\n".join(parts)


def extract_signals_gemini(doc: dict[str, str], startup_hint: str) -> list[dict[str, Any]]:
    """Use Gemini to extract procurement signals from actual document content."""
    if not GOOGLE_API_KEY:
        return []

    prompt = _build_extraction_prompt(doc, startup_hint)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_SCRAPE_TEMPERATURE,
                max_output_tokens=GEMINI_SCRAPE_MAX_TOKENS,
            ),
        )
        raw_text = response.text or ""
        items = _parse_json_array(raw_text)
        logger.info(
            "extract_signals_gemini: %s → %d raw signals (response %d chars, preview: %s)",
            doc["url"][:60], len(items), len(raw_text), raw_text[:200].replace("\n", " "),
        )
        return items
    except Exception as exc:
        logger.error("extract_signals_gemini failed for %s: %s", doc["url"][:60], exc)
        return []


# ═══════════════════════════════════════════════════════════════════
# FALLBACK — Gemini direct generation (no real documents)
# ═══════════════════════════════════════════════════════════════════

def generate_direct_signals_gemini(startup_hint: str) -> list[dict[str, Any]]:
    """Ask Gemini to generate signals from its own knowledge (no live data)."""
    if not GOOGLE_API_KEY or not startup_hint.strip():
        return []

    prompt = f"""You are an OSINT analyst for GovTech sales intelligence.

Startup profile:
{startup_hint}

Task:
Based on your knowledge of real government procurement trends and initiatives,
generate up to {GEMINI_DIRECT_SIGNALS} realistic government procurement signals
relevant to this startup.

For each signal, provide a source_url pointing to a real, known government website
(e.g., .gov, .gov.uk, .gc.ca, .europa.eu). Use only URLs you are confident exist.

Return ONLY a JSON array:
[
  {{
    "title": "Signal title",
    "description": "2-3 sentence description",
    "category": "Category",
    "budget": null,
    "timeline": "",
    "keywords": ["term1", "term2"],
    "stakeholders": ["Name (Role)"],
    "source_url": "https://...",
    "source_title": "Source name",
    "source_type": "inferred",
    "city": "",
    "region": "",
    "country": "",
    "lat": 0.0,
    "lng": 0.0
  }}
]

RULES:
- Focus on signals genuinely relevant to the startup.
- Include signals from US, Canada, UK, EU.
- source_url must be a .gov or official government domain.
- Keep descriptions under 60 words.
- Keywords must be specific technical terms.

LOCATION IS CRITICAL — every signal is plotted on a 3D globe.
- ALWAYS set city, region/state, and country for every signal.
- lat/lng MUST be numeric coordinates of the city (NOT 0.0 / 0.0).
- Spread signals across different real cities. Do NOT cluster everything in one spot.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_SCRAPE_TEMPERATURE,
                max_output_tokens=GEMINI_SCRAPE_MAX_TOKENS,
            ),
        )
        raw_text = response.text or ""
        items = _parse_json_array(raw_text)
        logger.info("generate_direct_signals_gemini: %d raw (response %d chars, preview: %s)", len(items), len(raw_text), raw_text[:200].replace("\n", " "))
        return items
    except Exception as exc:
        logger.error("generate_direct_signals_gemini failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════
# FALLBACK — Sources.json + Jina
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Source:
    id: str
    title: str
    type: str
    url: str
    location: dict[str, Any]
    seed_links: list[str] | None = None
    queries: list[str] | None = None


def load_sources() -> list[Source]:
    if not SOURCES_PATH.exists():
        return []
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Source(**item) for item in raw]


def source_jina_fallback(startup_hint: str) -> list[dict[str, Any]]:
    """Read known government sources with Jina, extract signals from content."""
    sources = load_sources()
    signals: list[dict[str, Any]] = []

    for source in sources[:SCRAPE_MAX_URLS_TO_READ]:
        doc = read_jina(source.url)
        if not doc:
            continue
        extracted = extract_signals_gemini(doc, startup_hint)
        for raw_sig in extracted:
            # Patch location from the source definition
            if not raw_sig.get("city"):
                raw_sig["city"] = source.location.get("city", "")
            if not raw_sig.get("region"):
                raw_sig["region"] = source.location.get("region", "")
            if not raw_sig.get("country"):
                raw_sig["country"] = source.location.get("country", "")
            if not raw_sig.get("lat"):
                raw_sig["lat"] = source.location.get("lat")
            if not raw_sig.get("lng"):
                raw_sig["lng"] = source.location.get("lng")
            norm = _normalize_signal(raw_sig, startup_hint)
            if norm:
                signals.append(norm)
        if len(signals) >= SCRAPE_MAX_SIGNALS:
            break

    return signals


# ═══════════════════════════════════════════════════════════════════
# BATCH: Serper results → Gemini (no Jina needed)
# ═══════════════════════════════════════════════════════════════════

def extract_signals_from_search_results(
    results: list[dict[str, str]], startup_hint: str
) -> list[dict[str, Any]]:
    """Send Serper search results directly to Gemini for signal extraction.

    This avoids the slow/unreliable Jina read step. Serper snippets already
    contain enough context for Gemini to identify relevant signals.
    """
    if not GOOGLE_API_KEY or not results:
        return []

    # Build a compact summary of all search results
    result_text = ""
    for i, r in enumerate(results[:20], 1):
        result_text += f"\n{i}. Title: {r['title']}\n   URL: {r['link']}\n   Snippet: {r['snippet']}\n"

    prompt = (
        f"You are a GovTech sales intelligence analyst. A startup builds: {startup_hint}\n\n"
        f"Below are web search results about government procurement and technology initiatives.\n"
        f"Extract up to {SCRAPE_MAX_SIGNALS} government procurement signals relevant to this startup.\n\n"
        f"SEARCH RESULTS:{result_text}\n\n"
        "For each signal, return a JSON array. Each object must have:\n"
        "title, description (2-3 sentences about the opportunity), category, "
        "budget (number or null), timeline, keywords (array of specific technical terms), "
        "stakeholders (array), source_url (use the actual URL from the search result), "
        "source_title, source_type, city, region, country, lat (number), lng (number).\n\n"
        "RULES:\n"
        "- Each signal must be based on a REAL search result above — use its actual URL.\n"
        "- Be generous: include initiatives even if only tangentially related.\n"
        "- Include budget/timeline/stakeholders if mentioned in the snippet.\n"
        "- Keywords should include terms from BOTH the search result AND the startup profile.\n"
        "\n"
        "LOCATION IS CRITICAL — every signal will be plotted on a 3D globe.\n"
        "- Infer city, region/state, and country from the URL domain AND snippet text.\n"
        "  Examples: .gov.uk → country='United Kingdom'; austin.gov → city='Austin', region='Texas', country='United States'.\n"
        "- Set lat (latitude) and lng (longitude) to numeric coordinates of the city.\n"
        "- If you only know the country, use its approximate geographic centre.\n"
        "- NEVER leave both city and country empty. At minimum, infer country from the URL TLD.\n"
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_SCRAPE_TEMPERATURE,
                max_output_tokens=GEMINI_SCRAPE_MAX_TOKENS,
            ),
        )
        raw_text = response.text or ""
        items = _parse_json_array(raw_text)
        logger.info(
            "extract_signals_from_search_results: %d raw from %d results (response %d chars, preview: %s)",
            len(items), len(results), len(raw_text), raw_text[:300].replace("\n", " "),
        )
        return items
    except Exception as exc:
        logger.error("extract_signals_from_search_results failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def collect_signals(startup_hint: str = "") -> list[dict[str, Any]]:
    """Discover government procurement signals relevant to the startup.

    Pipeline priority:
      1. Serper search → Gemini batch extract  (fast, uses real search data)
      2. Serper + Jina + Gemini per-page       (slower, richer content)
      3. Gemini direct generation              (no live data, LLM knowledge)
      4. Sources.json + Jina                   (known gov pages)
    """
    global LAST_COLLECT_META
    start = time.time()
    signals: list[dict[str, Any]] = []
    logger.info(
        "collect_signals: serper=%s jina=%s gemini=%s hint=%r",
        bool(SERPER_API_KEY), bool(JINA_API_KEY), bool(GOOGLE_API_KEY),
        startup_hint[:80] if startup_hint else "",
    )

    all_results: list[dict[str, str]] = []

    def _budget_left() -> bool:
        return (time.time() - start) < SCRAPE_TIME_BUDGET_SECONDS

    # ── Primary: Serper search → Gemini batch extraction ──────────
    if SERPER_API_KEY and GOOGLE_API_KEY and startup_hint.strip():
        queries = build_search_queries(startup_hint)
        all_results: list[dict[str, str]] = []
        seen_links: set[str] = set()

        for query in queries:
            if not _budget_left():
                break
            for result in search_serper(query):
                if result["link"] not in seen_links:
                    seen_links.add(result["link"])
                    all_results.append(result)

        logger.info("collect_signals: %d unique search results", len(all_results))

        if all_results:
            raw_items = extract_signals_from_search_results(all_results, startup_hint)
            for raw_sig in raw_items:
                norm = _normalize_signal(raw_sig, startup_hint)
                if norm:
                    signals.append(norm)
            signals = _dedup_signals(signals)

        if signals:
            signals = _ensure_coordinates(signals[:SCRAPE_MAX_SIGNALS])
            LAST_COLLECT_META = {
                "mode": "serper_gemini_batch",
                "signal_count": len(signals),
                "queries_used": len(queries),
                "search_results": len(all_results),
                "error": None,
            }
            return signals

    # ── Fallback 1: Serper + Jina + Gemini per page ───────────────
    if SERPER_API_KEY and JINA_API_KEY and startup_hint.strip() and _budget_left():
        # Reuse search results if available, or search again
        if not all_results:
            queries = build_search_queries(startup_hint)
            all_results = []
            seen_links = set()
            for query in queries:
                for result in search_serper(query):
                    if result["link"] not in seen_links:
                        seen_links.add(result["link"])
                        all_results.append(result)

        urls_read = 0
        for idx, result in enumerate(all_results):
            if urls_read >= SCRAPE_MAX_URLS_TO_READ or not _budget_left():
                break
            doc = read_jina(result["link"])
            if not doc:
                continue
            urls_read += 1
            extracted = extract_signals_gemini(doc, startup_hint)
            for raw_sig in extracted:
                norm = _normalize_signal(raw_sig, startup_hint)
                if norm:
                    signals.append(norm)
            if len(signals) >= SCRAPE_MAX_SIGNALS:
                break

        signals = _dedup_signals(signals)
        if signals:
            signals = _ensure_coordinates(signals[:SCRAPE_MAX_SIGNALS])
            LAST_COLLECT_META = {
                "mode": "serper_jina_gemini",
                "signal_count": len(signals),
                "error": None,
            }
            return signals

    # ── Fallback 2: Gemini direct ─────────────────────────────────
    if GOOGLE_API_KEY and startup_hint.strip() and _budget_left():
        raw_items = generate_direct_signals_gemini(startup_hint)
        for raw_sig in raw_items:
            norm = _normalize_signal(raw_sig, startup_hint)
            if norm:
                signals.append(norm)
        signals = _dedup_signals(signals)
        if signals:
            signals = _ensure_coordinates(signals[:SCRAPE_MAX_SIGNALS])
            LAST_COLLECT_META = {
                "mode": "gemini_direct",
                "signal_count": len(signals),
                "error": None,
            }
            return signals

    # ── Fallback 3: Sources + Jina ────────────────────────────────
    if JINA_API_KEY and _budget_left():
        source_signals = source_jina_fallback(startup_hint)
        source_signals = _dedup_signals(source_signals)
        if source_signals:
            source_signals = _ensure_coordinates(source_signals[:SCRAPE_MAX_SIGNALS])
            LAST_COLLECT_META = {
                "mode": "source_jina_fallback",
                "signal_count": len(source_signals),
                "error": None,
            }
            return source_signals

    LAST_COLLECT_META = {
        "mode": "none",
        "signal_count": 0,
        "error": "All signal collection methods exhausted or returned no results.",
    }
    return []


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def run(output_path: Path, append: bool, startup_hint: str = "") -> None:
    signals = collect_signals(startup_hint=startup_hint)
    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            signals = existing + signals

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)

    meta = get_last_collect_meta()
    print(f"Wrote {len(signals)} signals to {output_path}  (mode={meta.get('mode')})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover signals into signals.json")
    parser.add_argument("--output", default=str(SIGNALS_PATH))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--startup-hint", default="")
    args = parser.parse_args()
    run(Path(args.output), args.append, startup_hint=args.startup_hint)


if __name__ == "__main__":
    main()
