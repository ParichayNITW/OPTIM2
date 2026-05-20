"""
DRA Dispatcher — Pipeline Optima™
Routes DRA %drag-reduction queries to a pipeline-specific ML model when one exists,
or falls back to the existing CSV-based lookup (PHBPL empirical data) otherwise.

Model files live in:  dra_models/{PIPELINE_CODE}_model.pkl
Training script:       train_dra_model.py
"""

import os
import functools
from pathlib import Path
from typing import Optional

_MODELS_DIR = Path(__file__).parent / "dra_models"

_ACTIVE_PIPELINE_CODE: Optional[str] = None


def set_active_pipeline(code: Optional[str]) -> None:
    global _ACTIVE_PIPELINE_CODE
    _ACTIVE_PIPELINE_CODE = code


def get_active_pipeline() -> Optional[str]:
    return _ACTIVE_PIPELINE_CODE


@functools.lru_cache(maxsize=64)
def _load_model(code: str):
    """
    Load a trained GradientBoostingRegressor from dra_models/{code}_model.pkl.
    Returns None if the file does not exist. Result is cached per pipeline code.
    """
    model_path = _MODELS_DIR / f"{code}_model.pkl"
    if not model_path.exists():
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        return None


def get_dr_for_pipeline(pipeline_code: str,
                         flow_m3h: float,
                         viscosity_cst: float,
                         ppm: float) -> float:
    """
    Return %Drag Reduction for the given operating conditions.

    If a trained model exists for pipeline_code:
        Uses ML prediction with 3 features: [flow_m3h, viscosity_cst, ppm]
    Otherwise:
        Falls back to CSV-based viscosity+ppm lookup (current PHBPL behaviour).

    Args:
        pipeline_code:  Pipeline registry code e.g. "BKPL", "PHBPL"
        flow_m3h:       Volumetric flow rate in m³/hr
        viscosity_cst:  Kinematic viscosity in cSt
        ppm:            DRA injection rate in ppm (parts per million by volume)

    Returns:
        %Drag Reduction as a float in [0.0, 100.0]
    """
    model = _load_model(pipeline_code)
    if model is not None:
        import numpy as np
        X = np.array([[flow_m3h, viscosity_cst, ppm]])
        return float(np.clip(model.predict(X)[0], 0.0, 100.0))
    else:
        from dra_utils import get_dr_for_ppm
        return get_dr_for_ppm(viscosity_cst, ppm)


def get_ppm_for_dr_pipeline(pipeline_code: str,
                              flow_m3h: float,
                              viscosity_cst: float,
                              target_dr: float) -> float:
    """
    Return the PPM required to achieve target_dr% drag reduction.

    If a trained model exists: binary-searches PPM space [0, 500] using the ML model.
    Otherwise: falls back to CSV-based ppm lookup.

    Args:
        pipeline_code:  Pipeline registry code
        flow_m3h:       Volumetric flow rate in m³/hr
        viscosity_cst:  Kinematic viscosity in cSt
        target_dr:      Target %drag reduction

    Returns:
        PPM value (rounded to 1 decimal) required to achieve target_dr
    """
    model = _load_model(pipeline_code)
    if model is not None:
        import numpy as np
        lo, hi = 0.0, 500.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            dr = float(np.clip(model.predict([[flow_m3h, viscosity_cst, mid]])[0], 0.0, 100.0))
            if dr < target_dr:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 1)
    else:
        from dra_utils import get_ppm_for_dr
        return get_ppm_for_dr(viscosity_cst, target_dr)


def has_model(pipeline_code: str) -> bool:
    """Return True if a trained model file exists for the given pipeline code."""
    return (_MODELS_DIR / f"{pipeline_code}_model.pkl").exists()
