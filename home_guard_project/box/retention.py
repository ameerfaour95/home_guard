"""
Make the bucket delete production clips after two weeks (run once on the laptop, with your own AWS login).

Usage:
    python -m home_guard_project.box.retention          # add or update the rule
    python -m home_guard_project.box.retention --show   # print the bucket's rules, change nothing

A box in inference mode uploads its alert clips to ``production_<site>/``.
This adds one lifecycle rule to the bucket: every object whose key starts with
``production_`` is deleted PRODUCTION_RETENTION_DAYS after it was uploaded.
Nothing else in the bucket is touched: the tagging data and the
``dataset_<site>/`` folders have no such rule and are kept.

S3 does the deleting, so a box needs no delete permission. S3 runs these rules
once a day, so an object can outlive its two weeks by up to a day or so.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

from .boxconfig import PRODUCTION_PREFIX_ROOT, PRODUCTION_RETENTION_DAYS

log = logging.getLogger("box.retention")

RULE_ID = "homeguard-production-expiry"


def lifecycle_rule(days: int = PRODUCTION_RETENTION_DAYS) -> Dict[str, Any]:
    """The rule that expires everything under ``production_``."""
    return {
        "ID": RULE_ID,
        "Status": "Enabled",
        "Filter": {"Prefix": PRODUCTION_PREFIX_ROOT},
        "Expiration": {"Days": days},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
    }


def merged_rules(existing: List[Dict[str, Any]], rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """*existing* with *rule* added, replacing an earlier rule of the same ID.

    Writing a lifecycle configuration replaces all of a bucket's rules, so the
    other rules must be sent back unchanged.
    """
    return [r for r in existing if r.get("ID") != rule["ID"]] + [rule]


def current_rules(s3_client: Any, bucket: str) -> List[Dict[str, Any]]:
    from botocore.exceptions import ClientError

    try:
        return s3_client.get_bucket_lifecycle_configuration(Bucket=bucket)["Rules"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
            return []
        raise


def ensure_rule(s3_client: Any, bucket: str, days: int = PRODUCTION_RETENTION_DAYS) -> List[Dict[str, Any]]:
    """Add or update the rule on *bucket*. Returns the rules now in force."""
    rules = merged_rules(current_rules(s3_client, bucket), lifecycle_rule(days))
    s3_client.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete production clips from the bucket after two weeks.")
    parser.add_argument("--show", action="store_true", help="Print the bucket's lifecycle rules and change nothing.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    import boto3

    from home_guard_project.s3_upload.config import load_config

    bucket = load_config().bucket
    s3 = boto3.client("s3")
    if args.show:
        print(json.dumps(current_rules(s3, bucket), indent=2, default=str))
        return

    rules = ensure_rule(s3, bucket)
    log.info("s3://%s: objects under %s* are deleted %d days after upload. The bucket has %d rule(s).",
             bucket, PRODUCTION_PREFIX_ROOT, PRODUCTION_RETENTION_DAYS, len(rules))


if __name__ == "__main__":
    main()
