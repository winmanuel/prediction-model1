import re
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, asin, sqrt

import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

RAW_PATH = "data/raw/rentals_raw.xlsx"
OUT_PATH = "data/processed/rentals_geocoded.csv"
CACHE_PATH = "data/cache/geocode_cache.csv"
METRO_CACHE_PATH = "data/cache/metro_stations.csv"

EUR_TO_HUF = 390
CURRENT_YEAR = datetime.now().year

# Budapest center (Deák Ferenc tér)
CENTER_LAT = 47.4979
CENTER_LON = 19.0556

# District centroid fallback (guarantee coverage)
DISTRICT_CENTROIDS = {
    1: (47.4972, 19.0399), 2: (47.5420, 19.0350), 3: (47.5670, 19.0400),
    4: (47.5600, 19.0900), 5: (47.4980, 19.0550), 6: (47.5060, 19.0650),
    7: (47.5000, 19.0720), 8: (47.4920, 19.0800), 9: (47.4800, 19.0900),
    10: (47.4800, 19.1500), 11: (47.4700, 19.0350), 12: (47.5000, 19.0200),
    13: (47.5350, 19.0700), 14: (47.5250, 19.1050), 15: (47.5600, 19.1100),
    16: (47.5200, 19.1800), 17: (47.5000, 19.1100), 18: (47.4400, 19.1700),
    19: (47.4300, 19.1400), 20: (47.4300, 19.1100), 21: (47.4300, 19.0700),
    22: (47.4300, 19.0300), 23: (47.4000, 19.1100),
}

ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def roman_to_int(roman):
    roman = str(roman).upper().replace(".", "").strip()
    if not re.fullmatch(r"[IVXLCDM]+", roman):
        return None
    total, prev = 0, 0
    for ch in reversed(roman):
        val = ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if total > 0 else None


def normalize_missing_token(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("").str.strip().str.lower()
    return s.replace({"no": "unknown", "not provided": "unknown", "nan": "unknown", "": "unknown"})


def load_data():
    df = pd.read_excel(RAW_PATH, engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def load_cache() -> pd.DataFrame:
    try:
        cache = pd.read_csv(CACHE_PATH)
        return cache
    except FileNotFoundError:
        return pd.DataFrame(columns=["query", "lat", "lon"])


def save_cache(cache: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache.drop_duplicates(subset=["query"], keep="last").to_csv(CACHE_PATH, index=False)


# -----------------------------
# Cleaning (same as your pipeline)
# -----------------------------
def clean_price(df):
    s = df["price_per_month"].astype(str).str.strip()
    is_eur = s.str.contains(r"(?:€|eur|â‚¬)", case=False, regex=True)

    numeric = s.str.replace(r"[^\d,\.]", "", regex=True)
    numeric = numeric.str.replace(",", "", regex=False)
    price_num = pd.to_numeric(numeric, errors="coerce")

    df["price_huf"] = price_num
    df.loc[is_eur, "price_huf"] = price_num[is_eur] * EUR_TO_HUF
    return df


def clean_floor_area(df):
    df["floor_area_m2"] = df["floor_area"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    df["floor_area_m2"] = pd.to_numeric(df["floor_area_m2"], errors="coerce")
    return df


def clean_rooms(df):
    s = df["no__rooms"].astype(str).str.replace(" ", "", regex=False).str.lower()
    s = s.str.replace(r"^(\d+)\+0\.5$", r"\1.5", regex=True)
    df["rooms"] = pd.to_numeric(s, errors="coerce")
    return df


def parse_district(text: str):
    t = str(text).strip()

    m = re.search(r"district\s+([ivxlcdm]+|\d+)", t, flags=re.IGNORECASE)
    if m:
        token = m.group(1)
        val = int(token) if token.isdigit() else roman_to_int(token)
        if val and 1 <= val <= 23:
            return val

    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s+district\b", t, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 23:
            return val

    m = re.search(r"\b([ivxlcdm]+)\.\s*(kerület|district)\b", t, flags=re.IGNORECASE)
    if m:
        val = roman_to_int(m.group(1))
        if val and 1 <= val <= 23:
            return val

    return np.nan


def extract_district(df):
    df["district"] = df["name_district"].astype(str).apply(parse_district)
    df["district"] = pd.to_numeric(df["district"], errors="coerce")
    return df


def clean_balcony(df):
    s = df["balcony_size"].astype(str).str.lower().str.strip()
    size = s.str.extract(r"(\d+\.?\d*)")[0]
    df["balcony_m2"] = pd.to_numeric(size, errors="coerce").fillna(0)
    return df


def clean_min_rent(df):
    s = df["min_rental_time"].astype(str).str.lower().str.strip()
    num = pd.to_numeric(s.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    is_year = s.str.contains(r"\byear\b|\byears\b", regex=True, na=False)
    num[is_year] = num[is_year] * 12
    df["min_rent_months"] = num
    return df


def clean_internal_height(df):
    s = normalize_missing_token(df["internal_height"])
    ord_vals = pd.Series(np.nan, index=df.index)
    ord_vals[s.str.contains("lower", na=False)] = 0
    ord_vals[s.str.contains("higher", na=False)] = 1
    df["internal_height_ord"] = ord_vals
    return df


def clean_property_condition(df):
    s = normalize_missing_token(df["property_condition"])
    s = s.replace({"medium": "medium condition", "good": "in good condition"})
    df["property_condition_clean"] = s
    return df


def clean_year(df):
    s = normalize_missing_token(df["year_of_constr"])

    def estimate(v: str):
        v = str(v).lower().strip()
        if v == "unknown":
            return np.nan
        if re.fullmatch(r"\d{4}", v):
            return float(v)
        if "before" in v and "1950" in v:
            return 1940.0
        m = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", v)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2
        return np.nan

    df["year_of_constr_est"] = s.apply(estimate)
    df["year_missing_flag"] = df["year_of_constr_est"].isna().astype(int)
    df["building_age"] = np.where(df["year_of_constr_est"].isna(), np.nan, CURRENT_YEAR - df["year_of_constr_est"])
    return df


def clean_levels(df):
    df["floor_num"] = pd.to_numeric(df["floor"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["building_level_num"] = pd.to_numeric(df["building_level"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df


def encode_binaries(df):
    df["aircon_bin"] = df["aircon"].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})
    df["elevator_bin"] = df["elevator"].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})
    return df


# -----------------------------
# Geocoding (multi-try + cache)
# -----------------------------
def normalize_listing_queries(name_district: str, district: float) -> list[str]:
    raw = str(name_district).strip()
    raw = re.sub(r"\s*-\s*\d{1,2}\s*$", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    street_part = parts[-1] if parts else raw
    street_part = re.sub(r"\b(district|kerület)\b.*", "", street_part, flags=re.IGNORECASE).strip()
    street_part = re.sub(r"\s+", " ", street_part).strip()

    queries = []
    if "budapest" in raw.lower():
        queries.append(f"{raw}, Hungary")
    else:
        queries.append(f"Budapest, {raw}, Hungary")

    if street_part:
        queries.append(f"{street_part}, Budapest, Hungary")

    if pd.notna(district):
        d = int(district)
        queries.append(f"Budapest {d}. kerület, Hungary")

    # unique but ordered
    return list(dict.fromkeys(queries))


def geocode_with_cache(df: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    cache_map = {row["query"]: (row["lat"], row["lon"]) for _, row in cache.iterrows()}

    geolocator = Nominatim(user_agent="rental-price-prediction-geocoder")
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        max_retries=3,
        swallow_exceptions=True,
        error_wait_seconds=2.0
    )

    lats, lons, used_queries = [], [], []
    new_cache_rows = []

    print("\nGeocoding (multi-try + cached). New lookups rate-limited (1 req/sec).")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = row.get("name_district", "")
        dist = row.get("district", np.nan)

        candidates = normalize_listing_queries(name, dist)

        lat, lon, picked = (np.nan, np.nan, None)

        for q in candidates:
            if q in cache_map and pd.notna(cache_map[q][0]) and pd.notna(cache_map[q][1]):
                lat, lon = cache_map[q]
                picked = q
                break

            loc = geocode(q, timeout=10)
            if loc is not None:
                lat, lon = (loc.latitude, loc.longitude)
                picked = q
                new_cache_rows.append({"query": q, "lat": lat, "lon": lon})
                break
            else:
                new_cache_rows.append({"query": q, "lat": np.nan, "lon": np.nan})

        if (pd.isna(lat) or pd.isna(lon)) and pd.notna(dist):
            d = int(dist)
            lat, lon = DISTRICT_CENTROIDS.get(d, (np.nan, np.nan))
            picked = picked or f"FALLBACK_CENTROID_{d}"

        lats.append(lat)
        lons.append(lon)
        used_queries.append(picked or "NONE")

    if new_cache_rows:
        cache = pd.concat([cache, pd.DataFrame(new_cache_rows)], ignore_index=True)
        save_cache(cache)

    df["lat"] = pd.to_numeric(pd.Series(lats), errors="coerce")
    df["lon"] = pd.to_numeric(pd.Series(lons), errors="coerce")
    df["geocode_used_query"] = used_queries

    df["distance_to_center_km_real"] = df.apply(
        lambda r: haversine_km(r["lat"], r["lon"], CENTER_LAT, CENTER_LON),
        axis=1
    )
    return df


# -----------------------------
# Metro stations from OSM Overpass
# -----------------------------
def fetch_metro_stations_budapest() -> pd.DataFrame:
    """
    Fetch Budapest metro/subway stations using Overpass API, cache locally.
    """
    os.makedirs(os.path.dirname(METRO_CACHE_PATH), exist_ok=True)

    # Use cache if exists
    if os.path.exists(METRO_CACHE_PATH):
        metro = pd.read_csv(METRO_CACHE_PATH)
        if {"name", "lat", "lon"}.issubset(set(metro.columns)) and len(metro) > 0:
            return metro

    # Overpass query: subway/metro stations in Budapest administrative area
    query = """
    [out:json][timeout:60];
    area["name"="Budapest"]["boundary"="administrative"]->.a;
    (
      node(area.a)["railway"="station"]["station"="subway"];
      node(area.a)["railway"="station"]["subway"="yes"];
      node(area.a)["public_transport"="station"]["subway"="yes"];
      node(area.a)["railway"="station"]["network"~"M[1-4]"];
    );
    out center;
    """

    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(1, 4):
        try:
            r = requests.post(url, data=query, timeout=90)
            r.raise_for_status()
            data = r.json()
            elements = data.get("elements", [])

            rows = []
            for el in elements:
                lat = el.get("lat")
                lon = el.get("lon")
                name = (el.get("tags") or {}).get("name", "unknown")
                if lat is not None and lon is not None:
                    rows.append({"name": name, "lat": float(lat), "lon": float(lon)})

            metro = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
            metro.to_csv(METRO_CACHE_PATH, index=False)
            return metro

        except Exception as e:
            print(f"Overpass attempt {attempt}/3 failed: {e}")
            time.sleep(3 * attempt)

    # If Overpass fails, return empty (we handle later)
    return pd.DataFrame(columns=["name", "lat", "lon"])




def add_distance_to_metro(df: pd.DataFrame) -> pd.DataFrame:

    # ✅ Guard INSIDE function
    if "lat" not in df.columns or "lon" not in df.columns:
        raise KeyError("lat/lon not found. Run geocode_with_cache(df) before add_distance_to_metro(df).")

    metro = fetch_metro_stations_budapest()

    if len(metro) == 0:
        print("WARNING: No metro stations fetched.")
        df["dist_to_metro_km"] = np.nan
        return df

    metro_lats = metro["lat"].to_numpy()
    metro_lons = metro["lon"].to_numpy()

    def nearest_metro_km(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        dists = [haversine_km(lat, lon, ml, mn) for ml, mn in zip(metro_lats, metro_lons)]
        return float(np.min(dists))

    print(f"\nComputing dist_to_metro_km using {len(metro)} metro station points...")
    df["dist_to_metro_km"] = df.apply(
        lambda r: nearest_metro_km(r["lat"], r["lon"]),
        axis=1
    )

    return df

# Approximate Danube path through Budapest
DANUBE_POINTS = [
    (47.5200, 19.0400),
    (47.5100, 19.0450),
    (47.5000, 19.0500),
    (47.4900, 19.0550),
    (47.4800, 19.0600),
    (47.4700, 19.0650),
]

def add_distance_to_danube(df):
    def nearest_danube_km(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        dists = [haversine_km(lat, lon, dl, dn) for dl, dn in DANUBE_POINTS]
        return float(np.min(dists))

    print("\nComputing dist_to_danube_km...")
    df["dist_to_danube_km"] = df.apply(
        lambda r: nearest_danube_km(r["lat"], r["lon"]),
        axis=1
    )
    return df

def finalize(df):
    keep = [
        "price_huf",
        "floor_area_m2",
        "rooms",
        "district",
        "balcony_m2",
        "min_rent_months",
        "internal_height_ord",
        "property_condition_clean",
        "floor_num",
        "building_level_num",
        "year_missing_flag",
        "building_age",
        "aircon_bin",
        "elevator_bin",
        "lat",
        "lon",
        "distance_to_center_km_real",
        "dist_to_metro_km", 
        "dist_to_danube_km",
        "is_inner_ring",
    ]
    return df[keep].copy()


def main():
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = load_data()

    # clean core columns
    df = clean_price(df)
    df = clean_floor_area(df)
    df = clean_rooms(df)
    df = extract_district(df)
    df = clean_balcony(df)
    df = clean_min_rent(df)
    df = clean_internal_height(df)
    df = clean_property_condition(df)
    df = clean_year(df)
    df = clean_levels(df)
    df = encode_binaries(df)

    # ✅ MUST geocode before metro/danube
    df = geocode_with_cache(df)

    # ✅ now lat/lon exist
    df = add_distance_to_metro(df)
    df = add_distance_to_danube(df)

    # ✅ optional engineered binary
    df["is_inner_ring"] = df["district"].isin([5, 6, 7, 8, 9]).astype(int)

    cleaned = finalize(df).dropna(subset=["price_huf"])
    cleaned.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Rows:", len(cleaned))
    print("Columns:", len(cleaned.columns))
    print("\nCoverage check:")
    print("lat missing:", cleaned["lat"].isna().sum(), "/", len(cleaned))
    print("dist_to_metro_km missing:", cleaned["dist_to_metro_km"].isna().sum(), "/", len(cleaned))


if __name__ == "__main__":
    main()