import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/rentals_geocoded.csv"
MODEL_PATH = "models/rent_model.pkl"


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

    reg = HistGradientBoostingRegressor(
        random_state=42,
        **params
    )

    return Pipeline([("preprocessor", pre), ("regressor", reg)])


def cv_mae_real(model, X, y_log, y_real):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for tr_idx, te_idx in cv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_log = y_log.iloc[tr_idx]
        y_te_real = y_real.iloc[te_idx]

        model.fit(X_tr, y_tr_log)
        pred_log = model.predict(X_te)
        pred_real = np.expm1(pred_log)
        pred_real = np.clip(pred_real, 0, None)

        maes.append(mean_absolute_error(y_te_real, pred_real))

    return float(np.mean(maes)), float(np.std(maes))


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["price_huf"]).copy()

    # -----------------------------------
    # OUTLIER TRIMMING (1% each side)
    # -----------------------------------
    lower = df["price_huf"].quantile(0.01)
    upper = df["price_huf"].quantile(0.99)

    original_rows = len(df)

    df = df[(df["price_huf"] >= lower) & (df["price_huf"] <= upper)].copy()

    print("\nOutlier trimming applied (1% tails)")
    print("-----------------------------------")
    print(f"Original rows: {original_rows}")
    print(f"Remaining rows: {len(df)}")
    print(f"Removed rows: {original_rows - len(df)}")

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
    "dist_to_danube_km",   # NEW
    "is_inner_ring",       # NEW
    ]
    categorical_features = [
        "district",
        "property_condition_clean",
    ]

    features = numeric_features + categorical_features
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    X = df[features].copy()
    y_real = df["price_huf"].astype(float)
    y_log = np.log1p(y_real)

    # Small tuning grid (fast but meaningful)
    grid = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 800, "min_samples_leaf": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.07, "max_depth": 3, "max_iter": 800, "min_samples_leaf": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.07, "max_depth": 4, "max_iter": 700, "min_samples_leaf": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_depth": 4, "max_iter": 900, "min_samples_leaf": 10, "l2_regularization": 0.5},
        {"learning_rate": 0.03, "max_depth": 4, "max_iter": 1200, "min_samples_leaf": 10, "l2_regularization": 0.5},
        {"learning_rate": 0.07, "max_depth": 4, "max_iter": 900, "min_samples_leaf": 20, "l2_regularization": 1.0},
    ]

    best = None

    print("Running 5-fold CV tuning (log target, scored in real HUF MAE)...\n")
    for i, params in enumerate(grid, start=1):
        model = build_model(numeric_features, categorical_features, params)
        mean_mae, std_mae = cv_mae_real(model, X, y_log, y_real)
        print(f"[{i}/{len(grid)}] MAE: {mean_mae:,.2f} ± {std_mae:,.2f} | params={params}")

        if best is None or mean_mae < best["mae"]:
            best = {"mae": mean_mae, "std": std_mae, "params": params}

    print("\nBest CV result")
    print("-------------")
    print(f"MAE: {best['mae']:,.2f} ± {best['std']:,.2f}")
    print(f"Params: {best['params']}")

    # Fit final model on holdout just to show sanity metrics
    X_train, X_test, y_train_log, y_test_log, y_train_real, y_test_real = train_test_split(
        X, y_log, y_real, test_size=0.2, random_state=42
    )

    final_model = build_model(numeric_features, categorical_features, best["params"])
    final_model.fit(X_train, y_train_log)

    pred_log = final_model.predict(X_test)
    pred_real = np.expm1(pred_log)
    pred_real = np.clip(pred_real, 0, None)

    mae = mean_absolute_error(y_test_real, pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
    r2 = r2_score(y_test_real, pred_real)

    print("\nHoldout Evaluation (sanity check)")
    print("-------------------------------")
    print(f"MAE:  {mae:,.2f} HUF")
    print(f"RMSE: {rmse:,.2f} HUF")
    print(f"R2:   {r2:.4f}")

    payload = {
        "pipeline": final_model,
        "target_transform": "log1p",
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "best_params": best["params"],
    }

    joblib.dump(payload, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)
    print("NOTE: Saved model predicts log1p(price_huf). Use expm1() to return to HUF.")


if __name__ == "__main__":
    main()