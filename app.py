import os
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from math import radians, sin, cos, asin, sqrt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

MODEL_PATH = "models/rent_model.pkl"
GEO_CACHE = "data/cache/app_geocode_cache.csv"
METRO_CACHE = "data/cache/metro_stations.csv"

CENTER_LAT = 47.4979
CENTER_LON = 19.0556

# Same Danube approximation you used in preprocessing
DANUBE_POINTS = [
    (47.5200, 19.0400),
    (47.5100, 19.0450),
    (47.5000, 19.0500),
    (47.4900, 19.0550),
    (47.4800, 19.0600),
    (47.4700, 19.0650),
]


def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(np.sqrt(a))
    return R * c


@st.cache_data
def load_metro():
    if os.path.exists(METRO_CACHE):
        m = pd.read_csv(METRO_CACHE)
        if {"lat", "lon"}.issubset(m.columns) and len(m) > 0:
            return m
    return pd.DataFrame(columns=["lat", "lon"])


def load_geo_cache():
    os.makedirs(os.path.dirname(GEO_CACHE), exist_ok=True)
    if os.path.exists(GEO_CACHE):
        return pd.read_csv(GEO_CACHE)
    return pd.DataFrame(columns=["query", "lat", "lon"])


def save_geo_cache(df):
    df.drop_duplicates(subset=["query"], keep="last").to_csv(GEO_CACHE, index=False)


def geocode_address(query: str):
    cache = load_geo_cache()
    hit = cache[cache["query"] == query]
    if len(hit) == 1 and pd.notna(hit.iloc[0]["lat"]) and pd.notna(hit.iloc[0]["lon"]):
        return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"])

    geolocator = Nominatim(user_agent="rental-price-prediction-app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

    loc = geocode(query, timeout=10)
    if loc is None:
        lat, lon = (np.nan, np.nan)
    else:
        lat, lon = (loc.latitude, loc.longitude)

    cache = pd.concat([cache, pd.DataFrame([{"query": query, "lat": lat, "lon": lon}])], ignore_index=True)
    save_geo_cache(cache)
    return lat, lon


def dist_to_danube(lat, lon):
    dists = [haversine_km(lat, lon, dl, dn) for dl, dn in DANUBE_POINTS]
    return float(np.nanmin(dists)) if len(dists) else np.nan


def dist_to_metro(lat, lon, metro_df):
    if len(metro_df) == 0:
        return np.nan
    dists = [haversine_km(lat, lon, r.lat, r.lon) for r in metro_df.itertuples(index=False)]
    return float(np.nanmin(dists))


def normalize_street_input(street: str) -> str:
    """
    User should enter only: 'Street name + house number'
    We sanitize common extra text and then append 'Budapest, Hungary' for geocoding.
    """
    s = (street or "").strip()

    # Remove common unwanted tokens if user types them anyway
    # e.g. "Budapest, Deák Ferenc tér, Hungary" -> "Deák Ferenc tér"
    s = re.sub(r"\bBudapest\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bHungary\b", "", s, flags=re.IGNORECASE)

    # Remove repeated commas and extra spaces
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(,\s*){2,}", ", ", s)
    s = s.strip(" ,")

    # Build final query (force city & country)
    # If user adds a district like "V. ker" it still works fine.
    return f"{s}, Budapest, Hungary"


def main():
    st.set_page_config(page_title="Budapest Rent Predictor", layout="centered")
    st.title("Budapest Rent Predictor")

    payload = joblib.load(MODEL_PATH)
    model = payload["pipeline"]
    feats = payload["features"]
    intervals = payload.get("intervals", {})

    metro = load_metro()

    st.subheader("Property inputs")

    # ✅ Street-only input (no Budapest / Hungary shown)
    street_only = st.text_input(
        "Street name & house number",
        value="Deák Ferenc tér 1",
        help="Enter only the street name and house number. Example: 'Deák Ferenc tér 1'.",
        placeholder="e.g., Rákóczi út 12"
    )

    district = st.selectbox("District", list(range(1, 24)), index=4)

    col1, col2 = st.columns(2)
    with col1:
        floor_area_m2 = st.number_input("Floor area (m²)", min_value=10.0, max_value=300.0, value=50.0, step=1.0)
        rooms = st.number_input("Rooms", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        balcony_m2 = st.number_input("Balcony (m²)", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
        floor_num = st.number_input("Floor number", min_value=0, max_value=50, value=2, step=1)
    with col2:
        building_level_num = st.number_input("Building levels", min_value=0, max_value=60, value=5, step=1)
        building_age = st.number_input("Building age (years)", min_value=0, max_value=200, value=40, step=1)
        year_missing_flag = st.selectbox("Year missing flag", [0, 1], index=0)
        min_rent_months = st.number_input("Min rental months", min_value=0, max_value=60, value=12, step=1)

    internal_height_ord = st.selectbox("Internal height", ["unknown", "lower_than_3m", "higher_than_3m"], index=0)
    internal_height_map = {"unknown": np.nan, "lower_than_3m": 0, "higher_than_3m": 1}

    property_condition_clean = st.selectbox(
        "Property condition",
        ["unknown", "novel", "renovated", "in good condition", "medium condition"],
        index=0
    )

    aircon_bin = st.selectbox("Aircon", [0, 1], index=0)
    elevator_bin = st.selectbox("Elevator", [0, 1], index=0)

    if st.button("Predict rent (HUF)"):
        if not street_only or not street_only.strip():
            st.error("Please enter a street name and house number (e.g., 'Deák Ferenc tér 1').")
            st.stop()

        # ✅ Always geocode as Budapest, Hungary, regardless of what user typed
        query = normalize_street_input(street_only)

        lat, lon = geocode_address(query)

        if pd.isna(lat) or pd.isna(lon):
            st.error("Could not geocode that address. Try adding a house number or a clearer street name.")
            st.stop()

        distance_to_center_km_real = haversine_km(lat, lon, CENTER_LAT, CENTER_LON)
        dist_to_metro_km = dist_to_metro(lat, lon, metro)
        dist_to_danube_km = dist_to_danube(lat, lon)
        is_inner_ring = 1 if int(district) in [5, 6, 7, 8, 9] else 0

        row = {
            "floor_area_m2": float(floor_area_m2),
            "rooms": float(rooms),
            "balcony_m2": float(balcony_m2),
            "min_rent_months": float(min_rent_months),
            "internal_height_ord": internal_height_map[internal_height_ord],
            "year_missing_flag": int(year_missing_flag),
            "building_age": float(building_age),
            "floor_num": int(floor_num),
            "building_level_num": int(building_level_num),
            "aircon_bin": int(aircon_bin),
            "elevator_bin": int(elevator_bin),
            "lat": float(lat),
            "lon": float(lon),
            "distance_to_center_km_real": float(distance_to_center_km_real),
            "dist_to_metro_km": float(dist_to_metro_km),
            "dist_to_danube_km": float(dist_to_danube_km),
            "is_inner_ring": int(is_inner_ring),
            "district": str(int(district)),
            "property_condition_clean": str(property_condition_clean),
        }

        # Build X in correct order
        numeric = feats["numeric"]
        categorical = feats["categorical"]
        X = pd.DataFrame([{k: row.get(k, np.nan) for k in (numeric + categorical)}])

        pred_log = model.predict(X)[0]
        pred_huf = float(np.expm1(pred_log))
        pred_huf = max(0.0, pred_huf)

        st.success(f"Predicted rent: {pred_huf:,.0f} HUF / month")

        # Intervals
        if intervals:
            p80 = intervals.get("p80_abs_error")
            p90 = intervals.get("p90_abs_error")
            p95 = intervals.get("p95_abs_error")

            def band(p):
                if p is None:
                    return None
                lo = max(0.0, pred_huf - p)
                hi = pred_huf + p
                return (lo, hi)

            b80 = band(p80)
            b90 = band(p90)
            b95 = band(p95)

            st.subheader("Uncertainty bands (based on CV errors)")
            if b80:
                st.write(f"80% band: {b80[0]:,.0f} – {b80[1]:,.0f} HUF")
            if b90:
                st.write(f"90% band: {b90[0]:,.0f} – {b90[1]:,.0f} HUF")
            if b95:
                st.write(f"95% band: {b95[0]:,.0f} – {b95[1]:,.0f} HUF")

        st.caption(
            f"Computed features: center={distance_to_center_km_real:.2f} km, "
            f"metro={dist_to_metro_km:.2f} km, danube={dist_to_danube_km:.2f} km, "
            f"lat={lat:.5f}, lon={lon:.5f}"
        )


if __name__ == "__main__":
    main()