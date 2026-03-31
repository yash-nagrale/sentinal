"""
recommender.py
--------------
Nearest Doctor / Specialist Recommender for NeuroGuard.

Usage from other modules:
    from recommender import get_specialists_for_diagnosis, search_nearby_doctors

Run standalone test:
    py -3.11 recommender.py
"""

import requests
import os
import math

# ── Paste your Google Places API key here OR set as environment variable ──────
GOOGLE_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "YOUR_API_KEY_HERE")

# ── Specialist mapping: diagnosis → list of specialist types ──────────────────
# Each entry: (display_name, google_places_type, google_search_keyword)
SPECIALIST_MAP = {
    # PS2 — Vital sign deterioration
    "ps2_high": [
        ("Intensivist / Critical Care",   "hospital",       "critical care ICU specialist"),
        ("General Physician",             "doctor",         "general physician"),
        ("Cardiologist",                  "doctor",         "cardiologist heart specialist"),
    ],
    "ps2_moderate": [
        ("General Physician",             "doctor",         "general physician"),
        ("Internal Medicine Specialist",  "doctor",         "internal medicine specialist"),
        ("Cardiologist",                  "doctor",         "cardiologist"),
    ],
    "ps2_low": [
        ("General Physician",             "doctor",         "general physician"),
    ],

    # PS1 — Foot wound grading
    "ps1_grade1": [
        ("General Physician",             "doctor",         "general physician"),
        ("Podiatrist",                    "doctor",         "podiatrist foot doctor"),
    ],
    "ps1_grade2": [
        ("Podiatrist",                    "doctor",         "podiatrist foot doctor"),
        ("Diabetologist",                 "doctor",         "diabetologist diabetes specialist"),
    ],
    "ps1_grade3": [
        ("Podiatrist",                    "doctor",         "podiatrist foot doctor"),
        ("Vascular Surgeon",              "doctor",         "vascular surgeon"),
        ("Diabetologist",                 "doctor",         "diabetologist diabetes specialist"),
    ],
    "ps1_grade4": [
        ("Vascular Surgeon",              "doctor",         "vascular surgeon"),
        ("Orthopedic Surgeon",            "doctor",         "orthopedic surgeon"),
        ("Podiatrist",                    "doctor",         "podiatrist foot doctor"),
    ],

    # PS5 — CT Stroke detection
    "ps5_stroke": [
        ("Neurologist",                   "doctor",         "neurologist brain specialist"),
        ("Neurosurgeon",                  "doctor",         "neurosurgeon"),
        ("Emergency Physician",           "hospital",       "emergency hospital stroke centre"),
    ],
    "ps5_normal": [
        ("Neurologist",                   "doctor",         "neurologist"),
        ("General Physician",             "doctor",         "general physician"),
    ],
}

# ── Urgency messages ──────────────────────────────────────────────────────────
URGENCY_MAP = {
    "ps2_high":     ("🔴 URGENT",    "Visit emergency or call an ambulance immediately."),
    "ps2_moderate": ("🟠 Soon",      "Book an appointment within 24 hours."),
    "ps2_low":      ("🟢 Routine",   "Schedule a routine check-up when convenient."),
    "ps1_grade1":   ("🟢 Routine",   "Routine podiatry appointment recommended."),
    "ps1_grade2":   ("🟠 Soon",      "Book a diabetology/podiatry appointment within a week."),
    "ps1_grade3":   ("🔴 URGENT",    "Seek urgent surgical/podiatry review within 24 hours."),
    "ps1_grade4":   ("🔴 EMERGENCY", "Go to emergency immediately — gangrene risk."),
    "ps5_stroke":   ("🔴 EMERGENCY", "Call emergency services NOW. Every minute matters."),
    "ps5_normal":   ("🟢 Routine",   "Follow up with a neurologist if symptoms persist."),
}


def get_specialists_for_diagnosis(diagnosis_key: str) -> dict:
    """
    Returns specialist list + urgency for a given diagnosis key.
    Works completely offline — no API key needed.

    diagnosis_key options:
        ps2_high, ps2_moderate, ps2_low
        ps1_grade1, ps1_grade2, ps1_grade3, ps1_grade4
        ps5_stroke, ps5_normal
    """
    specialists = SPECIALIST_MAP.get(diagnosis_key, SPECIALIST_MAP["ps2_low"])
    urgency_level, urgency_msg = URGENCY_MAP.get(diagnosis_key, ("🟢 Routine", "Consult a doctor."))
    return {
        "diagnosis_key":  diagnosis_key,
        "urgency_level":  urgency_level,
        "urgency_message": urgency_msg,
        "specialists":    specialists,  # list of (display_name, places_type, keyword)
    }


def geocode_location(location_text: str) -> tuple[float, float] | None:
    """
    Converts a text location (city name, pincode, address) to (lat, lng).
    Uses Google Geocoding API.
    Returns None if geocoding fails.
    """
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": location_text, "key": GOOGLE_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data["status"] == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
        return None
    except Exception:
        return None


def search_nearby_doctors(
    lat: float,
    lng: float,
    keyword: str,
    radius_m: int = 5000,
    max_results: int = 5,
) -> list[dict]:
    """
    Searches Google Places API for nearby doctors/clinics.
    Returns a list of dicts with name, address, rating, distance, phone, maps_url.
    """
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        return _mock_results(keyword, lat, lng)

    try:
        # Nearby Search
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius":   radius_m,
            "keyword":  keyword,
            "type":     "doctor",
            "key":      GOOGLE_API_KEY,
        }
        r     = requests.get(url, params=params, timeout=10)
        data  = r.json()
        results = []

        for place in data.get("results", [])[:max_results]:
            place_lat = place["geometry"]["location"]["lat"]
            place_lng = place["geometry"]["location"]["lng"]
            dist_km   = _haversine(lat, lng, place_lat, place_lng)

            result = {
                "name":       place.get("name", "Unknown"),
                "address":    place.get("vicinity", "Address not available"),
                "rating":     place.get("rating", None),
                "user_ratings_total": place.get("user_ratings_total", 0),
                "distance_km": round(dist_km, 2),
                "open_now":   place.get("opening_hours", {}).get("open_now", None),
                "place_id":   place.get("place_id", ""),
                "maps_url":   f"https://maps.google.com/?q={place_lat},{place_lng}",
                "phone":      None,  # fetched separately if needed
            }
            results.append(result)

        # Sort by distance
        results.sort(key=lambda x: x["distance_km"])
        return results

    except Exception as e:
        return _mock_results(keyword, lat, lng)


def get_place_phone(place_id: str) -> str | None:
    """Fetches phone number for a place using Place Details API."""
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or not place_id:
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields":   "formatted_phone_number",
            "key":      GOOGLE_API_KEY,
        }
        r    = requests.get(url, params=params, timeout=8)
        data = r.json()
        return data.get("result", {}).get("formatted_phone_number")
    except Exception:
        return None


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in km between two lat/lng points."""
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _mock_results(keyword: str, lat: float, lng: float) -> list[dict]:
    """
    Returns demo results when API key is not set.
    Used during development and for hackathon demo.
    """
    return [
        {
            "name":       f"City Medical Centre — {keyword.title()}",
            "address":    "Near Shivajinagar, Pune, Maharashtra",
            "rating":     4.6,
            "user_ratings_total": 312,
            "distance_km": 1.2,
            "open_now":   True,
            "place_id":   "demo_1",
            "maps_url":   f"https://maps.google.com/?q={lat+0.01},{lng+0.01}",
            "phone":      "+91 20 2553 1234",
        },
        {
            "name":       f"Ruby Hall Clinic — {keyword.title()} Dept.",
            "address":    "40 Sassoon Road, Pune, Maharashtra 411001",
            "rating":     4.8,
            "user_ratings_total": 1847,
            "distance_km": 2.4,
            "open_now":   True,
            "place_id":   "demo_2",
            "maps_url":   f"https://maps.google.com/?q={lat+0.02},{lng+0.02}",
            "phone":      "+91 20 6645 5555",
        },
        {
            "name":       f"Sahyadri Hospitals — {keyword.title()}",
            "address":    "30 Karve Road, Pune, Maharashtra 411004",
            "rating":     4.5,
            "user_ratings_total": 924,
            "distance_km": 3.1,
            "open_now":   False,
            "place_id":   "demo_3",
            "maps_url":   f"https://maps.google.com/?q={lat+0.03},{lng+0.03}",
            "phone":      "+91 20 6721 3333",
        },
        {
            "name":       f"KEM Hospital — {keyword.title()}",
            "address":    "489 Rasta Peth, Pune, Maharashtra 411011",
            "rating":     4.3,
            "user_ratings_total": 2103,
            "distance_km": 4.7,
            "open_now":   True,
            "place_id":   "demo_4",
            "maps_url":   f"https://maps.google.com/?q={lat+0.04},{lng+0.04}",
            "phone":      "+91 20 6123 4567",
        },
    ]


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== NeuroGuard Specialist Recommender — Self Test ===\n")

    for key in ["ps2_high", "ps1_grade3", "ps5_stroke"]:
        info = get_specialists_for_diagnosis(key)
        print(f"Diagnosis: {key}")
        print(f"  Urgency : {info['urgency_level']} — {info['urgency_message']}")
        print(f"  Specialists:")
        for name, ptype, kw in info["specialists"]:
            print(f"    • {name} (search: '{kw}')")

    print("\n=== Mock location search (Pune) ===")
    results = search_nearby_doctors(18.5204, 73.8567, "neurologist brain specialist")
    for r in results:
        stars = f"⭐ {r['rating']}" if r['rating'] else "No rating"
        open_status = "Open" if r['open_now'] else ("Closed" if r['open_now'] is False else "Unknown")
        print(f"\n  {r['name']}")
        print(f"    {r['address']}")
        print(f"    {stars} ({r['user_ratings_total']} reviews) · {r['distance_km']} km · {open_status}")
        print(f"    📞 {r['phone'] or 'N/A'}")
        print(f"    🗺  {r['maps_url']}")

    print("\nRecommender module ready!")
