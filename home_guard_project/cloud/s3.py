"""Thin S3 wrapper around an injected boto3 client (tests inject a moto client)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional

# Phase 1 is read-only toward boxes: the cloud may only write under these prefixes.
WRITABLE_PREFIXES = ("fleet/", "admin_cache/", "training_exports/")


class ETagMismatch(Exception):
    """The object was replaced after it was listed: the body read is not the listed revision."""


@dataclass(frozen=True)
class ObjInfo:
    key: str
    etag: str
    size: int
    last_modified: datetime


class S3:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def list(self, prefix: str) -> Iterator[ObjInfo]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield ObjInfo(obj["Key"], obj.get("ETag", "").strip('"'), int(obj.get("Size", 0)),
                              obj["LastModified"])

    def get_bytes(self, key: str, if_match: Optional[str] = None) -> bytes:
        """The object's body; with `if_match` (a listed ETag) only that revision, else ETagMismatch."""
        if if_match is None:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        from botocore.exceptions import ClientError

        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key, IfMatch=f'"{if_match}"')
        except ClientError as e:
            error = e.response.get("Error", {})
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error.get("Code") in ("PreconditionFailed", "412") or status == 412:
                raise ETagMismatch(key) from e
            raise
        if resp.get("ETag", "").strip('"') != if_match:  # a store that ignores If-Match
            raise ETagMismatch(key)
        return resp["Body"].read()

    def get_text(self, key: str, if_match: Optional[str] = None) -> str:
        return self.get_bytes(key, if_match).decode("utf-8", errors="replace")

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_text(key))

    def put_json(self, key: str, body: Any) -> None:
        if not key.startswith(WRITABLE_PREFIXES):
            raise ValueError(f"refusing to write outside {WRITABLE_PREFIXES}: {key}")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(body).encode("utf-8"),
                               ContentType="application/json")

    def presign(self, key: str, ttl: int = 300) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
