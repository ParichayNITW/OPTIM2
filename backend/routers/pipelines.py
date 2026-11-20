"""CRUD for saved pipeline configurations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db import Pipeline, User
from backend.deps import current_user, db_session
from backend.models import PipelineCreate, PipelineOut

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.get("", response_model=list[PipelineOut])
def list_pipelines(
    session: Session = Depends(db_session),
    user: User = Depends(current_user),
):
    return (
        session.query(Pipeline)
        .filter(Pipeline.user_id == user.id)
        .order_by(Pipeline.updated_at.desc())
        .all()
    )


@router.post("", response_model=PipelineOut, status_code=status.HTTP_201_CREATED)
def create_pipeline(
    body: PipelineCreate,
    session: Session = Depends(db_session),
    user: User = Depends(current_user),
):
    p = Pipeline(user_id=user.id, name=body.name, description=body.description, config=body.config)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@router.get("/{pipeline_id}", response_model=PipelineOut)
def get_pipeline(
    pipeline_id: int,
    session: Session = Depends(db_session),
    user: User = Depends(current_user),
):
    p = session.get(Pipeline, pipeline_id)
    if not p or p.user_id != user.id:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return p


@router.put("/{pipeline_id}", response_model=PipelineOut)
def update_pipeline(
    pipeline_id: int,
    body: PipelineCreate,
    session: Session = Depends(db_session),
    user: User = Depends(current_user),
):
    p = session.get(Pipeline, pipeline_id)
    if not p or p.user_id != user.id:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    p.name = body.name
    p.description = body.description
    p.config = body.config
    session.commit()
    session.refresh(p)
    return p


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline(
    pipeline_id: int,
    session: Session = Depends(db_session),
    user: User = Depends(current_user),
):
    p = session.get(Pipeline, pipeline_id)
    if not p or p.user_id != user.id:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    session.delete(p)
    session.commit()
