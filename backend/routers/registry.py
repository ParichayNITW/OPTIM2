"""GET /api/pipeline-registry — expose pipeline categories and codes to the frontend."""

from __future__ import annotations
import sys, os

SOLVER_PATH = os.environ.get("SOLVER_PATH", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if SOLVER_PATH not in sys.path:
    sys.path.insert(0, SOLVER_PATH)

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["registry"])


@router.get("/pipeline-registry")
def get_pipeline_registry():
    from pipeline_registry import PIPELINE_CATEGORIES, PIPELINE_REGISTRY
    return {"categories": PIPELINE_CATEGORIES, "registry": PIPELINE_REGISTRY}
