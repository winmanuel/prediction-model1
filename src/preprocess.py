import re
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, asin, sqrt

RAW_PATH = "data/raw/rentals_raw.xlsx"
PROCESSED_PATH = "data/processed/rentals_cleaned.csv"

EUR_TO_HUF = 390
CURRENT_YEAR = datetime.now().year

# Budapest center (Deák Ferenc tér approx)
CENTER_LAT = 47.4979
CENTER_LON = 19.0556

# Approx district centroids (Option 1)
DISTRICT_CENTROIDS = {
    1: (47.4972, 19.0399),
    2: (47.5420, 19.0350),
    3: (47.5670, 19.0400),
    4: (47.5600, 19.0900),
    5: (47.4980, 19.0550),
    6: (47.5060, 19.0650),
    7: (47.5000, 19.0720),
    8: (47.4920, 19.0800),
    9: (47.4800, 19.0900),
    10: (47.4800, 19.1500),
    11: (47.4700, 19.0350),
    12: (47.5000, 19.0200),
    13: (47.5350, 19.0700),
    14: (47.5250, 19.1050),
    15: (47.5600, 19.1100),
    16: (47.5200, 19.1800),
    17: (47.5000, 19.1100),
    18: (47.4400, 19.1700),
    19: (47.4300, 19.1400),
    20: (47.4300, 19.1100),
    21: (47.4300, 19.0700),
    22: (47.4300, 19.0300),
    23: (47.4000, 19.1100),
}

ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


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


def clean_price(df):
    s = df["price_per_month"].astype(str).str.strip()

    # no capturing groups -> no warning
    is_eur = s.str.contains(r"(?:€|eur|â‚¬)", case=False, regex=True)

    # keep digits and separators, then remove thousand commas
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

    # District XIV / District 14
    m = re.search(r"district\s+([ivxlcdm]+|\d+)", t, flags=re.IGNORECASE)
    if m:
        token = m.group(1)
        val = int(token) if token.isdigit() else roman_to_int(token)
        if val and 1 <= val <= 23:
            return val

    # 9th district / 13th district
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s+district\b", t, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 23:
            return val

    # II. kerület / VI. district
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


def add_distance_to_center(df):
    def dist(d):
        if pd.isna(d):
            return np.nan
        lat, lon = DISTRICT_CENTROIDS.get(int(d), (np.nan, np.nan))
        return haversine_km(lat, lon, CENTER_LAT, CENTER_LON)

    df["distance_to_center_km"] = df["district"].apply(dist)
    return df


def clean_balcony(df):
    s = df["balcony_size"].astype(str).str.lower().str.strip()
    size = s.str.extract(r"(\d+\.?\d*)")[0]
    df["balcony_m2"] = pd.to_numeric(size, errors="coerce").fillna(0)  # 0 means no balcony
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


def finalize(df):
    keep = [
        "price_huf",
        "floor_area_m2",
        "rooms",
        "district",
        "distance_to_center_km",
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
    ]
    return df[keep].copy()


def main():
    df = load_data()
    df = clean_price(df)
    df = clean_floor_area(df)
    df = clean_rooms(df)
    df = extract_district(df)
    df = add_distance_to_center(df)
    df = clean_balcony(df)
    df = clean_min_rent(df)
    df = clean_internal_height(df)
    df = clean_property_condition(df)
    df = clean_year(df)
    df = clean_levels(df)
    df = encode_binaries(df)

    cleaned = finalize(df).dropna(subset=["price_huf"])
    cleaned.to_csv(PROCESSED_PATH, index=False)

    print("Saved:", PROCESSED_PATH)
    print("Rows:", len(cleaned))
    print("Columns:", len(cleaned.columns))
    print("\nSanity check: price_huf describe()")
    print(cleaned["price_huf"].describe())


if __name__ == "__main__":
    main()