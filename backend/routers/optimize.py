"""POST /api/optimize — enqueue optimization job."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db import OptimizationJob, User
from backend.deps import db_session, optional_user
from backend.models import JobStatus, OptimizeRequest, OptimizeResponse

router = APIRouter(prefix="/api", tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_optimization(
    body: OptimizeRequest,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job_id = str(uuid.uuid4())
    payload = body.model_dump()

    job = OptimizationJob(
        id=job_id,
        user_id=user.id if user else None,
        status="pending",
        payload=payload,
    )
    session.add(job)
    session.commit()

    # Import here to avoid circular import at module load
    from backend.worker import run_optimization
    run_optimization.delay(job_id, payload)

    return OptimizeResponse(job_id=job_id, status=JobStatus.PENDING)
