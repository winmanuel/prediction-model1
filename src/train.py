import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/rentals_cleaned.csv"   # your stable dataset
MODEL_PATH = "models/rent_model.pkl"


def build_pipeline(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_depth=4,
        max_iter=700,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])


def evaluate(model, X_train, X_test, y_train_log, y_test_real):
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred_real = np.expm1(pred_log)
    pred_real = np.clip(pred_real, 0, None)

    mae = mean_absolute_error(y_test_real, pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
    r2 = r2_score(y_test_real, pred_real)
    return mae, rmse, r2


def cv_mae_real(model, X, y_real, y_log):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_log = y_log.iloc[train_idx]
        y_te_real = y_real.iloc[test_idx]

        model.fit(X_tr, y_tr_log)
        pred_log = model.predict(X_te)
        pred_real = np.expm1(pred_log)
        pred_real = np.clip(pred_real, 0, None)

        maes.append(mean_absolute_error(y_te_real, pred_real))

    return float(np.mean(maes)), float(np.std(maes))


def run_experiment(df, feature_set_name, numeric_features, categorical_features):
    # force categoricals
    for c in categorical_features:
        df[c] = df[c].astype(str)

    X = df[numeric_features + categorical_features].copy()

    y_real = df["price_huf"].astype(float)
    y_log = np.log1p(y_real)

    model = build_pipeline(numeric_features, categorical_features)

    # IMPORTANT: train_test_split returns 6 outputs because we pass 3 arrays (X, y_log, y_real)
    X_train, X_test, y_train_log, y_test_log, y_train_real, y_test_real = train_test_split(
        X, y_log, y_real, test_size=0.2, random_state=42
    )

    mae, rmse, r2 = evaluate(model, X_train, X_test, y_train_log, y_test_real)
    cv_mean, cv_std = cv_mae_real(model, X, y_real, y_log)

    print(f"\n=== {feature_set_name} ===")
    print("Holdout (real HUF)")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.4f}")
    print("5-Fold CV MAE (real HUF)")
    print(f"{cv_mean:,.2f} ± {cv_std:,.2f}")

    return model, X, y_log, (mae, rmse, r2, cv_mean, cv_std)


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["price_huf"]).copy()

    # Feature Set A: FULL
    numeric_full = [
        "floor_area_m2",
        "rooms",
        "distance_to_center_km",
        "balcony_m2",
        "min_rent_months",
        "internal_height_ord",
        "year_missing_flag",
        "building_age",
        "floor_num",
        "building_level_num",
        "aircon_bin",
        "elevator_bin",
    ]
    categorical_full = ["district", "property_condition_clean"]

    # Feature Set B: PRUNED
    numeric_pruned = [
        "floor_area_m2",
        "rooms",
        "distance_to_center_km",
        "balcony_m2",
        "building_age",
        "floor_num",
        "building_level_num",
        "aircon_bin",
        "elevator_bin",
    ]
    categorical_pruned = ["district", "property_condition_clean"]

    model_full, X_full, y_log_full, metrics_full = run_experiment(
        df, "FULL FEATURES (log target)", numeric_full, categorical_full
    )

    model_pruned, X_pruned, y_log_pruned, metrics_pruned = run_experiment(
        df, "PRUNED FEATURES (log target)", numeric_pruned, categorical_pruned
    )

    # Choose best by CV MAE
    best_is_pruned = metrics_pruned[3] < metrics_full[3]
    best_name = "PRUNED" if best_is_pruned else "FULL"

    print("\n===============================")
    print(f"Best model by CV MAE: {best_name}")
    print("===============================")

    if best_is_pruned:
        best_model, best_X, best_y_log = model_pruned, X_pruned, y_log_pruned
        best_features = {"numeric": numeric_pruned, "categorical": categorical_pruned}
    else:
        best_model, best_X, best_y_log = model_full, X_full, y_log_full
        best_features = {"numeric": numeric_full, "categorical": categorical_full}

    best_model.fit(best_X, best_y_log)

    payload = {
        "pipeline": best_model,
        "target_transform": "log1p",
        "features": best_features,
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved best model to: {MODEL_PATH}")
    print("NOTE: Model predicts log1p(price_huf). Use expm1() to get HUF.")


if __name__ == "__main__":
    main()