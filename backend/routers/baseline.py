"""POST /api/baseline-dra — compute minimum DRA lacing requirement."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["baseline"])


class BaselineRequest(BaseModel):
    stations: list[dict[str, Any]]
    terminal: dict[str, Any]
    max_flow_m3h: float
    max_visc_cst: float
    kv_list: list[float] | None = None
    rho_list: list[float] | None = None
    fluid_density: float | None = None
    mop_kgcm2: float | None = None
    dra_upper_bound: float = 70.0
    min_suction_head: float = 0.0


@router.post("/baseline-dra")
def compute_baseline_dra(body: BaselineRequest) -> dict[str, Any]:
    try:
        from pipeline_model import compute_minimum_lacing_requirement
        result = compute_minimum_lacing_requirement(
            body.stations,
            body.terminal,
            max_flow_m3h=body.max_flow_m3h,
            max_visc_cst=body.max_visc_cst,
            kv_list=body.kv_list,
            rho_list=body.rho_list,
            fluid_density=body.fluid_density,
            mop_kgcm2=body.mop_kgcm2,
            dra_upper_bound=body.dra_upper_bound,
            min_suction_head=body.min_suction_head,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
