from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import (
    CollectionIn,
    CollectionItems,
    CollectionOut,
    ExportOut,
    ExportRequest,
    SavedFilter,
)

router = APIRouter(prefix="/studio", tags=["studio"], dependencies=[Depends(bearer)])


@router.get("/filters", response_model=list[SavedFilter])
def list_filters():
    raise HTTPException(status_code=501)


@router.get("/collections", response_model=list[CollectionOut])
def list_collections():
    raise HTTPException(status_code=501)


@router.post("/collections", response_model=CollectionOut)
def create_collection(body: CollectionIn):
    raise HTTPException(status_code=501)


@router.post("/collections/{collection_id}/items", response_model=CollectionOut)
def add_collection_items(collection_id: int, body: CollectionItems):
    raise HTTPException(status_code=501)


@router.delete("/collections/{collection_id}/items", response_model=CollectionOut)
def remove_collection_items(collection_id: int, body: CollectionItems):
    raise HTTPException(status_code=501)


@router.get("/exports", response_model=list[ExportOut])
def list_exports():
    raise HTTPException(status_code=501)


@router.post("/exports", response_model=ExportOut)
def create_export(body: ExportRequest):
    raise HTTPException(status_code=501)


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: int):
    raise HTTPException(status_code=501)
