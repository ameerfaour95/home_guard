from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import LoginRequest, RefreshRequest, StaffOut, TokenPair

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=TokenPair)
def login(body: LoginRequest):
    raise HTTPException(status_code=501)


@router.post("/auth/refresh", response_model=TokenPair)
def refresh(body: RefreshRequest):
    raise HTTPException(status_code=501)


@router.get("/me", response_model=StaffOut, dependencies=[Depends(bearer)])
def me():
    raise HTTPException(status_code=501)
