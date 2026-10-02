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

from .boxconfig import PRODUCTION_PREFIX_ROOT, s3_prefix

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


# Shared pools a box must never write into, even with an any-site key.
PROTECTED_PREFIXES = ("dataset_uca", "dataset_smarthome", "dataset_multi")


def policy_any_site(bucket: str) -> Dict[str, Any]:
    """For a box whose site is chosen by the installer: write under any ``dataset_<site>/``
    folder, and under any ``production_<site>/`` folder (inference-mode clips, which expire).

    The installer names the house in the setup program, so the folder is not
    known when the key is made. The shared dataset pools are denied outright.
    A box with this key can add or overwrite files in another site's folder,
    but cannot read or delete anything.
    """
    protected_objects = [f"arn:aws:s3:::{bucket}/{p}/*" for p in PROTECTED_PREFIXES]
    roots = ("dataset_", PRODUCTION_PREFIX_ROOT)
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": [f"{root}*" for root in roots]}},
            },
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": [f"arn:aws:s3:::{bucket}/{root}*/*" for root in roots],
            },
            {"Effect": "Deny", "Action": "s3:*", "Resource": protected_objects},
            {
                "Effect": "Deny",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": [f"{p}/*" for p in PROTECTED_PREFIXES]}},
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the AWS key for one collector box.")
    parser.add_argument("site", help="Site name, e.g. house2 (with --any-site: a name for the box itself).")
    parser.add_argument("--any-site", action="store_true",
                        help="Let the box upload to any dataset_<site>/ folder, for boxes whose site is set by the installer.")
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

    if args.any_site:
        policy = policy_any_site(s3_cfg.bucket)
        scope = (f"any s3://{s3_cfg.bucket}/dataset_<site>/ or production_<site>/ folder "
                 f"except {', '.join(PROTECTED_PREFIXES)}")
    else:
        policy = policy_for(args.site, s3_cfg.bucket)
        scope = f"s3://{s3_cfg.bucket}/{s3_prefix(args.site)}/ only"
    iam.put_user_policy(UserName=name, PolicyName="box-upload", PolicyDocument=json.dumps(policy))
    log.info("Policy set: list + write under %s", scope)

    if iam.list_access_keys(UserName=name)["AccessKeyMetadata"]:
        log.info("%s already has an access key, so none was created. Delete it in IAM first if you want a new one.", name)
        return

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
