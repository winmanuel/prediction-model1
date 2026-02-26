
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/rentals_geocoded.csv"
MODEL_PATH = "models/rent_model.pkl"


BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_iter": 800,
    "min_samples_leaf": 15,
    "l2_regularization": 1.0,
}


def build_model(numeric_features, categorical_features, params):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ])

    reg = HistGradientBoostingRegressor(random_state=42, **params)
    return Pipeline([("preprocessor", pre), ("regressor", reg)])


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["price_huf"]).copy()

    # 1% tail trimming
    lower = df["price_huf"].quantile(0.01)
    upper = df["price_huf"].quantile(0.99)
    df = df[(df["price_huf"] >= lower) & (df["price_huf"] <= upper)].copy()

    # categoricals
    df["district"] = df["district"].astype("Int64").astype(str)
    df["property_condition_clean"] = df["property_condition_clean"].astype(str)

    numeric_features = [
        "floor_area_m2",
        "rooms",
        "balcony_m2",
        "min_rent_months",
        "internal_height_ord",
        "year_missing_flag",
        "building_age",
        "floor_num",
        "building_level_num",
        "aircon_bin",
        "elevator_bin",
        "lat",
        "lon",
        "distance_to_center_km_real",
        "dist_to_metro_km",
        "dist_to_danube_km",
        "is_inner_ring",
    ]
    categorical_features = ["district", "property_condition_clean"]

    X = df[numeric_features + categorical_features].copy()
    y_real = df["price_huf"].astype(float)
    y_log = np.log1p(y_real)

    # OOF predictions for intervals
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_real = np.zeros(len(df), dtype=float)

    for tr_idx, te_idx in cv.split(X):
        model = build_model(numeric_features, categorical_features, BEST_PARAMS)
        model.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
        pred_log = model.predict(X.iloc[te_idx])
        oof_pred_real[te_idx] = np.clip(np.expm1(pred_log), 0, None)

    abs_err = np.abs(oof_pred_real - y_real.to_numpy())
    cv_mae = float(np.mean(abs_err))

    intervals = {
        "cv_mae": cv_mae,
        "p50_abs_error": float(np.quantile(abs_err, 0.50)),
        "p80_abs_error": float(np.quantile(abs_err, 0.80)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "trim_lower": float(lower),
        "trim_upper": float(upper),
        "n_rows_used": int(len(df)),
    }

    print("Intervals (based on OOF abs errors in HUF):")
    for k, v in intervals.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    # Fit final model on ALL trimmed data
    final_model = build_model(numeric_features, categorical_features, BEST_PARAMS)
    final_model.fit(X, y_log)

    payload = {
        "pipeline": final_model,
        "target_transform": "log1p",
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "best_params": BEST_PARAMS,
        "intervals": intervals
    }

    joblib.dump(payload, MODEL_PATH)
    print("\nSaved final model with intervals to:", MODEL_PATH)


if __name__ == "__main__":
    main()