from __future__ import annotations

import unittest
from typing import Any, Dict, List

from home_guard_project.box.retention import RULE_ID, ensure_rule, lifecycle_rule, merged_rules

OTHER = {"ID": "someone-elses-rule", "Status": "Enabled", "Filter": {"Prefix": "tmp/"}, "Expiration": {"Days": 1}}


class FakeS3:
    def __init__(self, rules: List[Dict[str, Any]]) -> None:
        self.rules = rules
        self.puts: List[Dict[str, Any]] = []

    def get_bucket_lifecycle_configuration(self, Bucket: str) -> Dict[str, Any]:  # noqa: N803
        return {"Rules": list(self.rules)}

    def put_bucket_lifecycle_configuration(self, **kwargs: Any) -> None:
        self.puts.append(kwargs)


class RetentionTest(unittest.TestCase):
    def test_rule_expires_only_the_production_folders_after_two_weeks(self) -> None:
        rule = lifecycle_rule()
        self.assertEqual(rule["Filter"], {"Prefix": "production_"})
        self.assertEqual(rule["Expiration"], {"Days": 14})
        self.assertEqual(rule["Status"], "Enabled")

    def test_other_rules_are_kept_and_ours_is_replaced_not_duplicated(self) -> None:
        old_ours = dict(lifecycle_rule(days=30))
        rules = merged_rules([OTHER, old_ours], lifecycle_rule())
        self.assertEqual(rules, [OTHER, lifecycle_rule()])

    def test_ensure_rule_writes_every_rule_back(self) -> None:
        s3 = FakeS3([OTHER])
        rules = ensure_rule(s3, "my-bucket")
        (put,) = s3.puts
        self.assertEqual(put["Bucket"], "my-bucket")
        self.assertEqual(put["LifecycleConfiguration"], {"Rules": rules})
        self.assertEqual([r["ID"] for r in rules], ["someone-elses-rule", RULE_ID])


if __name__ == "__main__":
    unittest.main()
