"""Thin S3 wrapper around an injected boto3 client (tests inject a moto client)."""
from __future__ import annotations

import base64
import hashlib
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
    def __init__(self, client, bucket: str, writable: tuple = WRITABLE_PREFIXES):
        self.client = client
        self.bucket = bucket
        self.writable = tuple(writable)

    def scoped(self, prefix: str) -> "S3":
        """The same bucket, writable ONLY under `prefix` (one folder, never a top-level area)."""
        if not prefix.endswith("/") or prefix.count("/") < 2 or ".." in prefix.split("/"):
            raise ValueError(f"refusing to scope writes to {prefix!r}")
        return S3(self.client, self.bucket, writable=(prefix,))

    def _check_write(self, key: str) -> None:
        if not key.startswith(self.writable) or key in self.writable:
            raise ValueError(f"refusing to write outside {self.writable}: {key}")

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

    def download_to(self, key: str, path, if_match: Optional[str] = None, chunk: int = 1 << 20) -> None:
        """Stream the object into the local file `path`; with `if_match` only that revision, else ETagMismatch."""
        from botocore.exceptions import ClientError

        params = dict(Bucket=self.bucket, Key=key)
        if if_match is not None:
            params["IfMatch"] = f'"{if_match}"'
        try:
            resp = self.client.get_object(**params)
        except ClientError as e:
            error = e.response.get("Error", {})
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error.get("Code") in ("PreconditionFailed", "412") or status == 412:
                raise ETagMismatch(key) from e
            raise
        body = resp["Body"]
        try:
            if if_match is not None and resp.get("ETag", "").strip('"') != if_match:
                raise ETagMismatch(key)
            with open(path, "wb") as out:
                for part in body.iter_chunks(chunk):
                    out.write(part)
        finally:
            body.close()

    def get_text(self, key: str, if_match: Optional[str] = None) -> str:
        return self.get_bytes(key, if_match).decode("utf-8", errors="replace")

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_text(key))

    def put_json(self, key: str, body: Any) -> None:
        self._check_write(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(body).encode("utf-8"),
                               ContentType="application/json")

    def copy(self, source_key: str, dest_key: str, content_type: Optional[str] = None,
             sha256: bool = False, if_match: Optional[str] = None) -> Optional[str]:
        """Server-side copy into a writable prefix. Metadata is replaced, not copied, so no stored
        Content-Disposition, filename or user metadata of the source travels with the copy.

        With `if_match` (an ETag), only that revision of the source is copied (`CopySourceIfMatch`); a source
        replaced since raises ETagMismatch. With `sha256`, S3 computes the copy's SHA-256 and its hex digest is
        returned (None if S3 gave none); the object's bytes never pass through this process."""
        self._check_write(dest_key)
        params = dict(Bucket=self.bucket, Key=dest_key, CopySource={"Bucket": self.bucket, "Key": source_key},
                      MetadataDirective="REPLACE")
        if content_type:
            params["ContentType"] = content_type
        if sha256:
            params["ChecksumAlgorithm"] = "SHA256"
        if if_match is not None:
            params["CopySourceIfMatch"] = f'"{if_match}"'
        from botocore.exceptions import ClientError

        try:
            resp = self.client.copy_object(**params)
        except ClientError as e:
            error = e.response.get("Error", {})
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error.get("Code") in ("PreconditionFailed", "412") or status == 412:
                raise ETagMismatch(source_key) from e
            raise
        checksum = (resp.get("CopyObjectResult") or {}).get("ChecksumSHA256")
        if not sha256 or not checksum or "-" in checksum:  # "-N": a per-part checksum, not the object's
            return None
        return base64.b64decode(checksum).hex()

    def put_bytes(self, key: str, body: bytes, content_type: str) -> None:
        self._check_write(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def list_dirs(self, prefix: str) -> list[str]:
        """The immediate "sub-directories" (common prefixes) under `prefix`."""
        out: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        return out

    def any_under(self, prefix: str) -> bool:
        return bool(self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1).get("Contents"))

    def presign(self, key: str, ttl: int = 300) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl)

    def download_file(self, key: str, path) -> None:
        self.client.download_file(self.bucket, key, str(path))

    def upload_file(self, path, key: str, content_type: str) -> None:
        self._check_write(key)
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})

    def head(self, key: str) -> Optional[ObjInfo]:
        """The object's current ETag, size and time, or None when there is no such object."""
        from botocore.exceptions import ClientError

        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return ObjInfo(key, resp.get("ETag", "").strip('"'), int(resp.get("ContentLength", 0)),
                       resp.get("LastModified"))

    def sha256(self, key: str, chunk: int = 1 << 20) -> str:
        """SHA-256 (hex) of the object's bytes, streamed once in chunks (never held in memory whole)."""
        digest = hashlib.sha256()
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            for part in body.iter_chunks(chunk):
                digest.update(part)
        finally:
            body.close()
        return digest.hexdigest()

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a writable `prefix`; returns how many."""
        if not prefix.startswith(self.writable) or prefix in self.writable:
            raise ValueError(f"refusing to delete outside a folder of {self.writable}: {prefix}")
        keys = [o.key for o in self.list(prefix)]
        for start in range(0, len(keys), 1000):
            self.client.delete_objects(Bucket=self.bucket, Delete={
                "Objects": [{"Key": k} for k in keys[start:start + 1000]], "Quiet": True})
        return len(keys)

    def delete_keys(self, keys) -> dict[str, str]:
        """Delete these objects (each under a writable prefix). DeleteObjects answers HTTP 200 even when single
        keys fail, so its per-key `Errors` are read: returns {key: error code} for every key that was not
        deleted (empty when all were). A request that fails as a whole raises."""
        keys = sorted(set(keys))
        for key in keys:
            self._check_write(key)
        failed: dict[str, str] = {}
        for start in range(0, len(keys), 1000):
            batch = keys[start:start + 1000]
            resp = self.client.delete_objects(Bucket=self.bucket, Delete={
                "Objects": [{"Key": k} for k in batch], "Quiet": True})
            asked = set(batch)
            for err in (resp or {}).get("Errors") or []:
                key = err.get("Key")
                if key in asked:
                    failed[key] = str(err.get("Code") or "Error")
        return failed

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
