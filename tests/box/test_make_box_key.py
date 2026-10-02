from __future__ import annotations

import json
import unittest

from home_guard_project.box.make_box_key import policy_any_site, policy_for, user_name

BUCKET = "my-bucket"


class MakeBoxKeyTest(unittest.TestCase):
    def test_user_name_uses_dashes(self) -> None:
        self.assertEqual(user_name("ameer_house"), "homeguard-box-ameer-house")
        self.assertEqual(user_name("house2"), "homeguard-box-house2")

    def test_policy_allows_only_the_sites_own_prefix(self) -> None:
        policy = policy_for("house2", BUCKET)
        list_stmt, write_stmt = policy["Statement"]

        self.assertEqual(list_stmt["Action"], "s3:ListBucket")
        self.assertEqual(list_stmt["Resource"], f"arn:aws:s3:::{BUCKET}")
        self.assertEqual(list_stmt["Condition"], {"StringLike": {"s3:prefix": "dataset_house2/*"}})

        self.assertEqual(write_stmt["Action"], ["s3:PutObject", "s3:AbortMultipartUpload"])
        self.assertEqual(write_stmt["Resource"], f"arn:aws:s3:::{BUCKET}/dataset_house2/*")

    def test_policy_grants_no_read_delete_or_other_prefix(self) -> None:
        text = json.dumps(policy_for("house2", BUCKET))
        for forbidden in ("GetObject", "DeleteObject", "s3:*", "dataset_uca", "tagging"):
            self.assertNotIn(forbidden, text)
        self.assertEqual({s["Effect"] for s in policy_for("house2", BUCKET)["Statement"]}, {"Allow"})

    def test_any_site_policy_allows_every_site_folder_but_no_reads(self) -> None:
        policy = policy_any_site(BUCKET)
        allows = [s for s in policy["Statement"] if s["Effect"] == "Allow"]
        self.assertEqual(
            sorted(str(s["Action"]) for s in allows),
            sorted(["s3:ListBucket", str(["s3:PutObject", "s3:AbortMultipartUpload"])]),
        )
        write = next(s for s in allows if s["Action"] != "s3:ListBucket")
        self.assertEqual(write["Resource"], f"arn:aws:s3:::{BUCKET}/dataset_*/*")
        allowed_text = json.dumps(allows)
        for forbidden in ("GetObject", "DeleteObject", '"s3:*"', "tagging"):
            self.assertNotIn(forbidden, allowed_text)

    def test_any_site_policy_denies_the_shared_pools(self) -> None:
        policy = policy_any_site(BUCKET)
        denies = [s for s in policy["Statement"] if s["Effect"] == "Deny"]
        object_deny = next(s for s in denies if s["Action"] == "s3:*")
        self.assertEqual(
            object_deny["Resource"],
            [f"arn:aws:s3:::{BUCKET}/{p}/*" for p in ("dataset_uca", "dataset_smarthome", "dataset_multi")],
        )
        list_deny = next(s for s in denies if s["Action"] == "s3:ListBucket")
        self.assertEqual(
            list_deny["Condition"]["StringLike"]["s3:prefix"],
            ["dataset_uca/*", "dataset_smarthome/*", "dataset_multi/*"],
        )


if __name__ == "__main__":
    unittest.main()
