import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

DATA_PATH = "data/processed/rentals_geocoded.csv"
MODEL_PATH = "models/rent_model.pkl"
OUT_DIR = "reports"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    payload = joblib.load(MODEL_PATH)
    model = payload["pipeline"]
    feats = payload["features"]

    numeric_features = feats["numeric"]
    categorical_features = feats["categorical"]
    raw_features = numeric_features + categorical_features

    df = pd.read_csv(DATA_PATH).dropna(subset=["price_huf"]).copy()

    # Match training trimming (1% tails) if intervals training did it
    lower = df["price_huf"].quantile(0.01)
    upper = df["price_huf"].quantile(0.99)
    df = df[(df["price_huf"] >= lower) & (df["price_huf"] <= upper)].copy()

    # Ensure categoricals match training
    for c in categorical_features:
        df[c] = df[c].astype(str)

    X = df[raw_features].copy()
    y_log = np.log1p(df["price_huf"].astype(float))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # Scoring in log space (model predicts log1p(price))
    def neg_mae_log(estimator, X_, y_):
        pred = estimator.predict(X_)
        return -float(np.mean(np.abs(pred - y_)))

    print("Computing permutation importance on RAW features...")
    r = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=neg_mae_log,
        n_repeats=25,
        random_state=42,
        n_jobs=-1,
    )

    imp = pd.DataFrame({
        "feature": raw_features,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    csv_path = os.path.join(OUT_DIR, "permutation_importance_raw.csv")
    imp.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # Plot top 20
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (Δ log-MAE; higher = more important)")
    plt.title("Top Raw Feature Importances")
    plot_path = os.path.join(OUT_DIR, "permutation_importance_raw_top20.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print("Saved:", plot_path)

    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()