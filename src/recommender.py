"""
recommender.py
--------------
Nearest Doctor / Specialist Recommender for SentinAl.

Uses 100% free APIs — no API keys required:
  - Nominatim (OpenStreetMap) for geocoding any location
  - Overpass API (OpenStreetMap) for finding nearby hospitals & clinics
  - ip-api.com for IP-based location detection

Usage from other modules:
    from recommender import get_specialists_for_diagnosis, search_nearby_facilities
"""

import requests
import math

# ── Specialist mapping: diagnosis → list of recommended specialist types ──────
SPECIALIST_MAP = {
    # PS2 — Vital sign deterioration
    "ps2_high": [
        "Intensivist / Critical Care",
        "General Physician",
        "Cardiologist",
    ],
    "ps2_moderate": [
        "General Physician",
        "Internal Medicine Specialist",
        "Cardiologist",
    ],
    "ps2_low": [
        "General Physician",
    ],

    # PS1 — Foot wound grading
    "ps1_grade1": ["General Physician", "Podiatrist"],
    "ps1_grade2": ["Podiatrist", "Diabetologist"],
    "ps1_grade3": ["Podiatrist", "Vascular Surgeon", "Diabetologist"],
    "ps1_grade4": ["Vascular Surgeon", "Orthopedic Surgeon", "Podiatrist"],

    # PS5 — CT Stroke detection
    "ps5_stroke": ["Neurologist", "Neurosurgeon", "Emergency Physician"],
    "ps5_normal": ["Neurologist", "General Physician"],
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

# ── Major Indian cities — pre-known coordinates (instant lookup, no network) ──
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

_OSM_HEADERS = {"User-Agent": "SentinAl/1.0 (medical-ai-app)"}


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
    """
    specialists = SPECIALIST_MAP.get(diagnosis_key, SPECIALIST_MAP["ps2_low"])
    urgency_level, urgency_msg = URGENCY_MAP.get(
        diagnosis_key, ("🟢 Routine", "Consult a doctor."))
    return {
        "diagnosis_key":   diagnosis_key,
        "urgency_level":   urgency_level,
        "urgency_message": urgency_msg,
        "specialists":     specialists,
    }


def geocode_location(location_text):
    """
    Converts a text location (city, pincode, address) to (lat, lng).
    Uses:
      1. MAJOR_CITIES dict (instant, no network)
      2. Nominatim / OpenStreetMap (free, no API key, any location worldwide)
    Returns None if both fail.
    """
    query = location_text.lower().strip()
    if not query:
        return None

    # 1. Check pre-known cities (instant)
    for city, coords in MAJOR_CITIES.items():
        if query in city.lower():
            return coords

    # 2. Nominatim — free, no API key, any location
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_text, "format": "json", "limit": 1},
            headers=_OSM_HEADERS, timeout=10,
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    return None


def search_nearby_facilities(lat, lng, radius_m=15000, max_results=10):
    """
    Search for hospitals, clinics, and doctor offices near a location
    using the Overpass API (OpenStreetMap). Free, no API key.

    Returns a list of dicts sorted by distance:
        name, facility_type, address, distance_km, phone, website,
        opening_hours, maps_url, lat, lon
    """
    query = (
        f'[out:json][timeout:20];'
        f'('
        f'  node["amenity"~"hospital|clinic|doctors"](around:{radius_m},{lat},{lng});'
        f'  way["amenity"~"hospital|clinic|doctors"](around:{radius_m},{lat},{lng});'
        f');'
        f'out center;'
    )

    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers=_OSM_HEADERS,
            timeout=25,
        )
        if r.status_code != 200:
            return []

        elements = r.json().get("elements", [])
        results = []
        seen = set()

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            # Coordinates — nodes have lat/lon directly, ways have center
            el_lat = el.get("lat") or (el.get("center", {}).get("lat"))
            el_lng = el.get("lon") or (el.get("center", {}).get("lon"))
            if not el_lat or not el_lng:
                continue

            dist = _haversine(lat, lng, el_lat, el_lng)

            # Build address from addr:* tags
            addr_parts = []
            for key in ("addr:street", "addr:city", "addr:postcode"):
                val = tags.get(key)
                if val:
                    addr_parts.append(val)
            address = ", ".join(addr_parts) if addr_parts else "Address not listed"

            amenity = tags.get("amenity", "hospital")
            type_label = {
                "hospital": "Hospital",
                "clinic":   "Clinic",
                "doctors":  "Doctor",
            }.get(amenity, "Medical")

            results.append({
                "name":          name,
                "facility_type": type_label,
                "address":       address,
                "distance_km":   round(dist, 2),
                "phone":         tags.get("phone") or tags.get("contact:phone"),
                "website":       tags.get("website") or tags.get("contact:website"),
                "opening_hours": tags.get("opening_hours"),
                "maps_url":      f"https://maps.google.com/?q={el_lat},{el_lng}",
                "lat":           el_lat,
                "lon":           el_lng,
            })

        results.sort(key=lambda x: x["distance_km"])
        return results[:max_results]

    except Exception:
        return []


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


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SentinAl Specialist Recommender — Self Test ===\n")

    for key in ["ps2_high", "ps1_grade3", "ps5_stroke"]:
        info = get_specialists_for_diagnosis(key)
        print(f"Diagnosis: {key}")
        print(f"  Urgency : {info['urgency_level']} — {info['urgency_message']}")
        print(f"  Specialists: {', '.join(info['specialists'])}")

    print(f"\n=== Geocode 'Wardha' ===")
    coords = geocode_location("Wardha")
    print(f"  Coordinates: {coords}")

    if coords:
        print(f"\n=== Search hospitals near Wardha ===")
        results = search_nearby_facilities(coords[0], coords[1], radius_m=15000)
        for r in results[:5]:
            print(f"\n  {r['name']} ({r['facility_type']})")
            print(f"    {r['address']} · {r['distance_km']} km")
            print(f"    Phone: {r['phone'] or 'N/A'}")
            print(f"    Maps:  {r['maps_url']}")

    print("\nRecommender module ready!")
