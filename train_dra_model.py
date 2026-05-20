"""
DRA Model Training Script — Pipeline Optima™
Train a per-pipeline GradientBoostingRegressor for %Drag Reduction prediction.

Input CSV columns (required):
    Flow_Rate_m3h   — volumetric flow rate (m³/hr)
    Viscosity_cSt   — kinematic viscosity (cSt)
    PPM             — DRA injection rate (ppm by volume)
    DR_percent      — measured % drag reduction (0–100)

Usage:
    python train_dra_model.py --pipeline BKPL --data BKPL_data.csv
    python train_dra_model.py --pipeline PHBPL --data phbpl_training.csv --test-size 0.25

Output:
    dra_models/{PIPELINE_CODE}_model.pkl

After training, restart the Pipeline Optima app.
The app will automatically detect the model and display:
    "Pipeline-specific ML model loaded for {CODE}"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


FEATURES = ["Flow_Rate_m3h", "Viscosity_cSt", "PPM"]
TARGET   = "DR_percent"

MODELS_DIR = Path(__file__).parent / "dra_models"


def validate_data(df: pd.DataFrame, pipeline_code: str) -> pd.DataFrame:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[FEATURES + [TARGET]].copy()
    df = df.dropna()

    before = len(df)
    df = df[
        (df["Flow_Rate_m3h"] > 0) &
        (df["Viscosity_cSt"] > 0) &
        (df["PPM"] >= 0) &
        (df["DR_percent"] >= 0) &
        (df["DR_percent"] <= 100)
    ]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with invalid values.")

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} valid rows after cleaning — need at least 20 data points."
        )

    print(f"  {len(df)} valid training samples for {pipeline_code}.")
    print(f"  Flow range:       {df['Flow_Rate_m3h'].min():.1f} – {df['Flow_Rate_m3h'].max():.1f} m³/hr")
    print(f"  Viscosity range:  {df['Viscosity_cSt'].min():.1f} – {df['Viscosity_cSt'].max():.1f} cSt")
    print(f"  PPM range:        {df['PPM'].min():.0f} – {df['PPM'].max():.0f}")
    print(f"  DR% range:        {df['DR_percent'].min():.1f} – {df['DR_percent'].max():.1f}%")
    return df


def build_and_train(df: pd.DataFrame, test_size: float, pipeline_code: str):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Hyperparameter grid — small but effective for tabular DRA data
    param_grid = {
        "gbr__n_estimators":  [100, 200, 300],
        "gbr__max_depth":     [2, 3, 4],
        "gbr__learning_rate": [0.05, 0.1, 0.15],
        "gbr__subsample":     [0.8, 1.0],
    }

    # StandardScaler + GBR pipeline (scaler helps CV stability)
    model_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            random_state=42,
            min_samples_leaf=3,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )),
    ])

    print(f"\n  Running grid search (3-fold CV, {len(X_train)} training samples)...")
    cv_folds = min(3, len(X_train) // 10) or 2
    gs = GridSearchCV(
        model_pipe,
        param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    best_params = {k.replace("gbr__", ""): v for k, v in gs.best_params_.items()}
    print(f"  Best hyperparameters: {best_params}")

    # Evaluate on held-out test set
    y_pred = best.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 100.0)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ── Test Set Evaluation ({int(len(X_test))} samples) ──────────────")
    print(f"  RMSE : {rmse:.3f} % DR")
    print(f"  MAE  : {mae:.3f} % DR")
    print(f"  R²   : {r2:.4f}")

    if r2 < 0.80:
        print(f"\n  WARNING: R² = {r2:.4f} is below 0.80.")
        print(f"  Consider collecting more data or checking for outliers.")
    elif r2 >= 0.95:
        print(f"\n  Excellent fit (R² ≥ 0.95).")

    # Feature importance
    gbr_model = best.named_steps["gbr"]
    importances = gbr_model.feature_importances_
    print(f"\n  Feature importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"    {feat:<20s}: {imp*100:.1f}%")

    return best, {"rmse": rmse, "mae": mae, "r2": r2, "best_params": best_params}


def save_model(model, pipeline_code: str) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"{pipeline_code}_model.pkl"
    joblib.dump(model, out_path, compress=3)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved: {out_path}  ({size_kb:.1f} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Train pipeline-specific DRA model for Pipeline Optima™"
    )
    parser.add_argument(
        "--pipeline", required=True,
        help="Pipeline code (e.g. BKPL, PHBPL, SMPL). Must match pipeline_registry.py."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to training CSV with columns: Flow_Rate_m3h, Viscosity_cSt, PPM, DR_percent"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data for test evaluation (default: 0.20)"
    )
    args = parser.parse_args()

    pipeline_code = args.pipeline.strip().upper()
    data_path = Path(args.data)

    print(f"\nPipeline Optima™ — DRA Model Training")
    print(f"Pipeline : {pipeline_code}")
    print(f"Data file: {data_path}")
    print("-" * 50)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    # Validate pipeline code against registry
    try:
        from pipeline_registry import PIPELINE_REGISTRY
        if pipeline_code not in PIPELINE_REGISTRY:
            print(f"WARNING: '{pipeline_code}' not found in pipeline_registry.py.")
            print(f"  Available codes: {list(PIPELINE_REGISTRY.keys())}")
            ans = input("  Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                sys.exit(0)
        else:
            info = PIPELINE_REGISTRY[pipeline_code]
            print(f"Pipeline: {info['name']} ({info['category']})")
    except ImportError:
        print("  (pipeline_registry.py not found — skipping code validation)")

    print(f"\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    df = validate_data(df, pipeline_code)

    model, metrics = build_and_train(df, args.test_size, pipeline_code)
    out_path = save_model(model, pipeline_code)

    print(f"\nDone. Restart Pipeline Optima to activate the model for {pipeline_code}.")
    print(f"The app will show: \"Pipeline-specific ML model loaded for {pipeline_code}\"")


if __name__ == "__main__":
    main()
