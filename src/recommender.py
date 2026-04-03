"""
recommender.py
--------------
Nearest Doctor / Specialist Recommender for SentinAl.

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

# Readable by app.py to show API errors in the UI
last_api_error = ""

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

# ── Major Indian cities — pre-known coordinates (no geocoding API needed) ─────
MAJOR_CITIES = {
    "Pune, Maharashtra":                (18.5204, 73.8567),
    "Mumbai, Maharashtra":              (19.0760, 72.8777),
    "Delhi":                            (28.6139, 77.2090),
    "Bangalore, Karnataka":             (12.9716, 77.5946),
    "Hyderabad, Telangana":             (17.3850, 78.4867),
    "Chennai, Tamil Nadu":              (13.0827, 80.2707),
    "Kolkata, West Bengal":             (22.5726, 88.3639),
    "Ahmedabad, Gujarat":              (23.0225, 72.5714),
    "Jaipur, Rajasthan":               (26.9124, 75.7873),
    "Lucknow, Uttar Pradesh":          (26.8467, 80.9462),
    "Chandigarh":                       (30.7333, 76.7794),
    "Nagpur, Maharashtra":              (21.1458, 79.0882),
    "Indore, Madhya Pradesh":           (22.7196, 75.8577),
    "Bhopal, Madhya Pradesh":           (23.2599, 77.4126),
    "Patna, Bihar":                     (25.6093, 85.1376),
    "Coimbatore, Tamil Nadu":           (11.0168, 76.9558),
    "Kochi, Kerala":                    (9.9312, 76.2673),
    "Thiruvananthapuram, Kerala":       (8.5241, 76.9366),
    "Visakhapatnam, Andhra Pradesh":    (17.6868, 83.2185),
    "Guwahati, Assam":                  (26.1445, 91.7362),
    "Surat, Gujarat":                   (21.1702, 72.8311),
    "Vadodara, Gujarat":                (22.3072, 73.1812),
    "Nashik, Maharashtra":              (19.9975, 73.7898),
    "Aurangabad, Maharashtra":          (19.8762, 75.3433),
    "Mysore, Karnataka":                (12.2958, 76.6394),
    "Mangalore, Karnataka":             (12.9141, 74.8560),
    "Dehradun, Uttarakhand":            (30.3165, 78.0322),
    "Ranchi, Jharkhand":                (23.3441, 85.3096),
    "Raipur, Chhattisgarh":            (21.2514, 81.6296),
    "Bhubaneswar, Odisha":              (20.2961, 85.8245),
    "Goa":                              (15.2993, 74.1240),
}


def detect_location():
    """
    Detect user's approximate location via IP geolocation (ip-api.com).
    Free, no API key needed, ~45 requests/min limit.
    Returns {"city", "region", "lat", "lon"} or None on failure.
    """
    try:
        r = requests.get("http://ip-api.com/json/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return {
                    "city":   data.get("city", ""),
                    "region": data.get("regionName", ""),
                    "lat":    data.get("lat"),
                    "lon":    data.get("lon"),
                }
    except Exception:
        pass
    return None


def get_specialists_for_diagnosis(diagnosis_key):
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


def geocode_location(location_text):
    """
    Converts a text location (city name, pincode, address) to (lat, lng).
    Tries three sources in order:
      1. MAJOR_CITIES dict (instant, no API)
      2. Nominatim / OpenStreetMap (free, no API key, any location)
      3. Google Geocoding API (if API key set)
    Returns None if all fail.
    """
    query = location_text.lower().strip()
    if not query:
        return None

    # 1. Check pre-known cities (instant, no network)
    for city, coords in MAJOR_CITIES.items():
        if query in city.lower():
            return coords

    # 2. Nominatim / OpenStreetMap — free, no API key, works worldwide
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_text, "format": "json", "limit": 1},
            headers={"User-Agent": "SentinAl/1.0 (medical-ai-app)"},
            timeout=10,
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    # 3. Google Geocoding API (if key set)
    if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": location_text, "key": GOOGLE_API_KEY},
                timeout=10,
            )
            data = r.json()
            if data.get("status") == "OK":
                loc = data["results"][0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            pass

    return None


def search_nearby_doctors(lat, lng, keyword, place_type="doctor",
                          radius_m=15000, max_results=5):
    """
    Searches Google Places API for nearby doctors/clinics/hospitals.
    Returns a list of dicts with name, address, rating, distance, phone, maps_url.
    Sets `last_api_error` on failure so the caller can display it.
    """
    global last_api_error
    last_api_error = ""

    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        return _mock_results(keyword, lat, lng)

    try:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius":   radius_m,
            "keyword":  keyword,
            "type":     place_type,
            "key":      GOOGLE_API_KEY,
        }
        r     = requests.get(url, params=params, timeout=10)
        data  = r.json()

        status = data.get("status", "UNKNOWN")
        if status not in ("OK", "ZERO_RESULTS"):
            err_msg = data.get("error_message", status)
            last_api_error = (
                f"Google Places API error: **{status}** — {err_msg}\n\n"
                "Make sure the **Places API** is enabled in your "
                "[Google Cloud Console](https://console.cloud.google.com/apis/library/places-backend.googleapis.com) "
                "and the API key has no IP/referrer restrictions blocking server-side calls."
            )
            return _mock_results(keyword, lat, lng)

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

        results.sort(key=lambda x: x["distance_km"])
        return results

    except requests.exceptions.ConnectionError:
        last_api_error = "Could not connect to Google Places API. Check your internet connection."
        return _mock_results(keyword, lat, lng)
    except requests.exceptions.Timeout:
        last_api_error = "Google Places API request timed out. Try again."
        return _mock_results(keyword, lat, lng)
    except Exception as e:
        last_api_error = f"Unexpected error calling Google Places API: {e}"
        return _mock_results(keyword, lat, lng)


def get_place_phone(place_id):
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


def _haversine(lat1, lon1, lat2, lon2):
    """Returns distance in km between two lat/lng points."""
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _mock_results(keyword, lat, lng):
    """
    Returns demo results when API key is not set.
    Used during development and for hackathon demo.
    """
    # Find the closest known city for realistic labels
    city_label = "your area"
    best_dist = float("inf")
    for name, (clat, clng) in MAJOR_CITIES.items():
        d = _haversine(lat, lng, clat, clng)
        if d < best_dist:
            best_dist = d
            city_label = name.split(",")[0]  # "Pune" from "Pune, Maharashtra"

    return [
        {
            "name":       f"City Medical Centre — {keyword.title()}",
            "address":    f"Near Central Area, {city_label}",
            "rating":     4.6,
            "user_ratings_total": 312,
            "distance_km": 1.2,
            "open_now":   True,
            "place_id":   "demo_1",
            "maps_url":   f"https://maps.google.com/?q={lat+0.01},{lng+0.01}",
            "phone":      "+91 20 2553 1234",
        },
        {
            "name":       f"General Hospital — {keyword.title()} Dept.",
            "address":    f"Main Road, {city_label}",
            "rating":     4.8,
            "user_ratings_total": 1847,
            "distance_km": 2.4,
            "open_now":   True,
            "place_id":   "demo_2",
            "maps_url":   f"https://maps.google.com/?q={lat+0.02},{lng+0.02}",
            "phone":      "+91 20 6645 5555",
        },
        {
            "name":       f"Apollo Clinic — {keyword.title()}",
            "address":    f"Station Road, {city_label}",
            "rating":     4.5,
            "user_ratings_total": 924,
            "distance_km": 3.1,
            "open_now":   False,
            "place_id":   "demo_3",
            "maps_url":   f"https://maps.google.com/?q={lat+0.03},{lng+0.03}",
            "phone":      "+91 20 6721 3333",
        },
        {
            "name":       f"District Hospital — {keyword.title()}",
            "address":    f"Civil Lines, {city_label}",
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
    print("=== SentinAl Specialist Recommender — Self Test ===\n")

    for key in ["ps2_high", "ps1_grade3", "ps5_stroke"]:
        info = get_specialists_for_diagnosis(key)
        print(f"Diagnosis: {key}")
        print(f"  Urgency : {info['urgency_level']} — {info['urgency_message']}")
        print(f"  Specialists:")
        for name, ptype, kw in info["specialists"]:
            print(f"    • {name} (search: '{kw}')")

    print(f"\n=== Known cities: {len(MAJOR_CITIES)} ===")
    for city in list(MAJOR_CITIES.keys())[:5]:
        print(f"  {city}: {MAJOR_CITIES[city]}")
    print("  ...")

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
