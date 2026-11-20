"""GET /api/results/{job_id} — poll or retrieve optimization result."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db import OptimizationJob, User
from backend.deps import db_session, optional_user
from backend.models import JobStatus, ResultResponse

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results/{job_id}", response_model=ResultResponse)
def get_result(
    job_id: str,
    session: Session = Depends(db_session),
    user: User | None = Depends(optional_user),
):
    job = session.get(OptimizationJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Authenticated users can only see their own jobs
    if user and job.user_id and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return ResultResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        result=job.result,
        error=job.error_message,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs", tags=["results"])
def list_jobs(
    session: Session = Depends(db_session),
    user: User = Depends(optional_user),
    limit: int = 20,
    offset: int = 0,
):
    """List recent jobs for the authenticated user."""
    if not user:
        return {"jobs": [], "total": 0}
    jobs = (
        session.query(OptimizationJob)
        .filter(OptimizationJob.user_id == user.id)
        .order_by(OptimizationJob.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    total = session.query(OptimizationJob).filter(OptimizationJob.user_id == user.id).count()
    return {
        "jobs": [
            {
                "job_id": j.id,
                "status": j.status,
                "created_at": j.created_at,
                "finished_at": j.finished_at,
            }
            for j in jobs
        ],
        "total": total,
    }
