"""Stub auth dependency: declares Bearer security in OpenAPI. Real auth lands in Task 7."""
from fastapi.security import HTTPBearer

bearer = HTTPBearer(auto_error=False)
