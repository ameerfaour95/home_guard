"""
Create the AWS key for one collector box (run on the laptop, with your own AWS login).

Usage:
    python -m home_guard_project.box.make_box_key house2

Creates the IAM user ``homeguard-box-<site>`` if it is missing, limits it to
listing and writing under ``s3://<bucket>/dataset_<site>/``, and writes a new
access key to ``~/.homeguard/keys/<site>/`` as ``credentials`` and ``config``.
Copy those two files to ``C:\\Users\\<user>\\.aws\\`` on the box, then delete them
from the laptop. The secret is never printed.

A box can only upload to its own site folder: it cannot read other data, and
cannot overwrite the public-dataset pools or another site's clips. If a box
moves to a new site, run this again with the new site name and replace its key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict

from .boxconfig import s3_prefix

log = logging.getLogger("box.key")

KEY_ROOT = os.path.join(os.path.expanduser("~"), ".homeguard", "keys")
_SITE_RE = re.compile(r"^[a-z0-9_]+$")


def user_name(site: str) -> str:
    """IAM user name for a site. IAM names may not contain underscores in some tools, so use dashes."""
    return f"homeguard-box-{site.replace('_', '-')}"


def policy_for(site: str, bucket: str) -> Dict[str, Any]:
    """Least privilege for a box: list and write under its own dataset prefix only."""
    prefix = s3_prefix(site)
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": f"{prefix}/*"}},
            },
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the AWS key for one collector box.")
    parser.add_argument("site", help="Site name, e.g. house2 (lowercase letters, digits, underscores).")
    parser.add_argument("--out", default=None, help=f"Output folder (default {KEY_ROOT}/<site>).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    if not _SITE_RE.match(args.site):
        log.error("Site must be lowercase letters, digits or underscores.")
        sys.exit(2)

    import boto3
    from botocore.exceptions import ClientError

    from home_guard_project.s3_upload.config import load_config

    s3_cfg = load_config()
    name = user_name(args.site)
    iam = boto3.client("iam")

    try:
        iam.create_user(UserName=name, Tags=[{"Key": "purpose", "Value": "home-guard collector box"}])
        log.info("Created IAM user %s", name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        log.info("IAM user %s already exists", name)

    iam.put_user_policy(
        UserName=name, PolicyName="box-upload",
        PolicyDocument=json.dumps(policy_for(args.site, s3_cfg.bucket)),
    )
    log.info("Policy set: list + write under s3://%s/%s/ only", s3_cfg.bucket, s3_prefix(args.site))

    if iam.list_access_keys(UserName=name)["AccessKeyMetadata"]:
        log.error("%s already has an access key. Delete it in IAM first if you want a new one.", name)
        sys.exit(3)

    key = iam.create_access_key(UserName=name)["AccessKey"]
    out_dir = args.out or os.path.join(KEY_ROOT, args.site)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "credentials"), "w", encoding="ascii", newline="\n") as f:
        f.write(
            "[default]\n"
            f"aws_access_key_id = {key['AccessKeyId']}\n"
            f"aws_secret_access_key = {key['SecretAccessKey']}\n"
        )
    with open(os.path.join(out_dir, "config"), "w", encoding="ascii", newline="\n") as f:
        f.write(f"[default]\nregion = {s3_cfg.region}\n")

    log.info("Key written to %s (credentials, config). Copy both to the box's .aws folder, then delete them here.", out_dir)


if __name__ == "__main__":
    main()
