"""Reusable test builders: a moto bucket seeded with the sanitised fixtures under their real keys.

Tasks 8 and 10-13 share these, so keep the key constants stable.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fleet_contract" / "fixtures"
BUCKET = "security-camera-project-v1"
SITE = "test"

STEM = "front_side_1791020177_alert"
_DAY = "front_side/2026-10-03"
PROD_META = f"production_test/meta/{_DAY}/{STEM}.meta.json"
TRAIN_META = f"dataset_test/meta/{_DAY}/{STEM}.meta.json"
PROD_CLIP = f"production_test/clips/{_DAY}/{STEM}.mp4"
TRAIN_CLIP = f"dataset_test/clips/{_DAY}/{STEM}.mp4"
RAW_ANSWER = f"dataset_test/responses/{_DAY}/{STEM}.model_raw.txt"
CROPS = [f"dataset_test/vlm_crops/{_DAY}/{STEM}_f{i}.jpg" for i in range(3)]
FEEDBACK = f"production_test/feedback/{_DAY}/{STEM}_1791020300000.feedback.json"
ORPHAN_FEEDBACK = "production_test/feedback/test_ch6/2026-10-03/test_ch6_1790979739_alert_1790979837338.feedback.json"
GENERAL_FEEDBACK = "production_test/feedback/_general/2026-10-03/general_1791013449968.feedback.json"
HEARTBEAT = "dataset_test/_status/heartbeat.json"

TRAVERSAL_STEM = "front_side_1791020999_alert"
TRAVERSAL_META = f"production_test/meta/{_DAY}/{TRAVERSAL_STEM}.meta.json"

FP_STEM = "back_door_1791013299_fp"
FP_META = f"dataset_test/meta/back_door/2026-10-03/{FP_STEM}.meta.json"
FP_CLIP = f"dataset_test/clips/back_door/2026-10-03/{FP_STEM}.mp4"

FALLBACK_STEM = "back_door_1791013421_fp"
FALLBACK_META = f"dataset_test/meta/back_door/2026-10-03/{FALLBACK_STEM}.meta.json"

PAUSED_STEM = "left_side_1_1791014883_paused"
PAUSED_META = f"dataset_test/meta/left_side_1/2026-10-03/{PAUSED_STEM}.meta.json"
PAUSED_CLIP = f"dataset_test/clips/left_side_1/2026-10-03/{PAUSED_STEM}.mp4"

COLLECT_STEM = "back_door_1790944263_trigger"
COLLECT_META = f"dataset_test/meta/back_door/2026-10-02/{COLLECT_STEM}.meta.json"
COLLECT_CLIP = f"dataset_test/clips/back_door/2026-10-02/{COLLECT_STEM}.mp4"
YOLO_IMAGES = [f"dataset_test/yolo/images/back_door/2026-10-02/{COLLECT_STEM}_f{i:04d}.jpg" for i in (0, 2)]
YOLO_LABELS = [f"dataset_test/yolo/labels/back_door/2026-10-02/{COLLECT_STEM}_f{i:04d}.txt" for i in (0, 2)]

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
FAKE_JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64 + b"\xff\xd9"
RAW_TEXT = ('{"summary": "A person appears in the frame walking across the area, wearing light-colored '
            'clothing.", "label": "normal", "people": 1, "vehicle_moving": false}')
LABEL_TEXT = "0 0.5 0.5 0.2 0.4\n"


def fixture_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def feedback_body(alert_id: str = STEM, verdict: str = "true_alert",
                  time_utc: str = "2026-10-03T09:40:00Z") -> dict:
    body = fixture_json("feedback.json")
    body["alert"]["alert_id"] = alert_id
    body["alert"]["camera"] = alert_id.rsplit("_", 2)[0]
    body["verdict"] = verdict
    body["time_utc"] = time_utc
    return body


def seed_objects() -> dict[str, bytes]:
    """Every seeded key -> body bytes."""
    general = copy.deepcopy(fixture_json("feedback.json"))
    general["alert"] = None
    j = lambda body: json.dumps(body).encode("utf-8")  # noqa: E731
    objects = {
        PROD_META: (FIXTURES / "prod_alert.meta.json").read_bytes(),
        TRAIN_META: (FIXTURES / "train_alert.meta.json").read_bytes(),
        PROD_CLIP: FAKE_MP4,
        TRAIN_CLIP: FAKE_MP4,
        RAW_ANSWER: RAW_TEXT.encode("utf-8"),
        FEEDBACK: j(feedback_body()),
        ORPHAN_FEEDBACK: (FIXTURES / "feedback.json").read_bytes(),
        GENERAL_FEEDBACK: j(general),
        HEARTBEAT: (FIXTURES / "heartbeat.json").read_bytes(),
        TRAVERSAL_META: (FIXTURES / "traversal.meta.json").read_bytes(),
        FP_META: (FIXTURES / "train_fp.meta.json").read_bytes(),
        FP_CLIP: FAKE_MP4,
        FALLBACK_META: (FIXTURES / "fallback.meta.json").read_bytes(),
        PAUSED_META: (FIXTURES / "paused.meta.json").read_bytes(),
        PAUSED_CLIP: FAKE_MP4,
        COLLECT_META: (FIXTURES / "collect.meta.json").read_bytes(),
        COLLECT_CLIP: FAKE_MP4,
    }
    for key in CROPS + YOLO_IMAGES:
        objects[key] = FAKE_JPG
    for key in YOLO_LABELS:
        objects[key] = LABEL_TEXT.encode("utf-8")
    return objects


def seed_bucket(s3client, bucket: str = BUCKET, exclude=()) -> dict[str, bytes]:
    """Create the bucket (if needed) and upload the fixtures under their real keys, minus `exclude`."""
    existing = {b["Name"] for b in s3client.list_buckets().get("Buckets", [])}
    if bucket not in existing:
        s3client.create_bucket(Bucket=bucket)
    objects = {k: v for k, v in seed_objects().items() if k not in set(exclude)}
    for key, body in objects.items():
        put(s3client, key, body, bucket)
    return objects


def put(s3client, key: str, body, bucket: str = BUCKET) -> None:
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
    elif isinstance(body, str):
        body = body.encode("utf-8")
    s3client.put_object(Bucket=bucket, Key=key, Body=body)


def enroll(session, site: str = SITE, customer_name: str = "Acme"):
    """Create a customer + enrolled device for `site`; returns the Device (flushed, not committed)."""
    from home_guard_project.cloud.models import Customer, Device

    cust = Customer(name=customer_name)
    session.add(cust)
    session.flush()
    dev = Device(device_id=str(uuid.uuid4()), site=site, customer_id=cust.id,
                 enrolled_at=datetime.now(timezone.utc))
    session.add(dev)
    session.flush()
    return dev
