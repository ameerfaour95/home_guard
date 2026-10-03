from __future__ import annotations

import os
from dataclasses import dataclass

MIN_SECRET_BYTES = 32


@dataclass
class Settings:
    db_url: str
    jwt_secret: str
    bucket: str = "security-camera-project-v1"
    region: str = "us-east-1"
    access_ttl: int = 900
    refresh_ttl: int = 43200

    @classmethod
    def from_env(cls) -> "Settings":
        secret = os.environ["HG_CLOUD_JWT_SECRET"]
        if len(secret.encode()) < MIN_SECRET_BYTES:
            raise ValueError(f"HG_CLOUD_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes long")
        return cls(
            db_url=os.environ["HG_CLOUD_DB_URL"],
            jwt_secret=secret,
            bucket=os.environ.get("HG_CLOUD_BUCKET", "security-camera-project-v1"),
            region=os.environ.get("HG_CLOUD_REGION", "us-east-1"),
            access_ttl=int(os.environ.get("HG_CLOUD_ACCESS_TTL", "900")),
            refresh_ttl=int(os.environ.get("HG_CLOUD_REFRESH_TTL", "43200")),
        )

    @classmethod
    def for_tests(cls, db_url: str) -> "Settings":
        return cls(db_url=db_url, jwt_secret="test-secret-not-for-production-0123456789")
