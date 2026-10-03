from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import CustomerIn, CustomerOut

router = APIRouter(tags=["customers"], dependencies=[Depends(bearer)])


@router.get("/customers", response_model=list[CustomerOut])
def list_customers():
    raise HTTPException(status_code=501)


@router.post("/customers", response_model=CustomerOut)
def create_customer(body: CustomerIn):
    raise HTTPException(status_code=501)


@router.get("/customers/{customer_id}", response_model=CustomerOut)
def get_customer(customer_id: int):
    raise HTTPException(status_code=501)


@router.patch("/customers/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: int, body: CustomerIn):
    raise HTTPException(status_code=501)
