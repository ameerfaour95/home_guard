"""Demo data for the HomeGuardAdmin desktop app, produced by the real Admin Center routes.

    python -m home_guard_project.cloud.export_demo [out_dir]        # default: docs/admin/demo

Builds an embedded Postgres (pgserver) and a moto S3 bucket, seeds the bucket with synthetic box output (meta,
clips, feedback, heartbeats under the real keys), runs the real indexer and media worker, adds the human state
(staff, review marks, collections, audit rows) and then calls every GET route as admin and as labeler through
FastAPI's TestClient. The responses are written verbatim to `<route_name>[.labeler].json`; thumbnails and
filmstrips made by the media worker go to `media/thumb_<id>.jpg` and `media/filmstrip_<id>.jpg`; `index.json`
lists what each file is. Nothing here is hand-written response JSON, and everything is deterministic: the clock
is fixed at 2026-10-03T12:00:00Z and every random choice is seeded.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "admin" / "demo"
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
SEED = 15
BUCKET = "security-camera-project-v1"
UTC_PLUS = timedelta(hours=3)  # Asia/Jerusalem in early October
JWT_SECRET = "demo-secret-not-for-production-0123456789"
CLIP_W, CLIP_H, CLIP_FPS, CLIP_SEC = 320, 180, 5, 4

COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "bird": 14, "cat": 15, "dog": 16}


# ---------------------------------------------------------------- the households

@dataclass(frozen=True)
class CustomerSpec:
    name: str
    live: bool
    recordings: bool
    training: bool
    notes: str


@dataclass(frozen=True)
class DeviceSpec:
    customer: int  # index into CUSTOMERS
    site: str
    host: str
    cameras: tuple  # (name, display name, events)
    verdict: str
    events_until_h: float  # hours before NOW the last event may be at
    heartbeat_age_min: float
    disk_gb: float
    collector_running: bool = True


CUSTOMERS = [
    CustomerSpec("Dana Cohen", True, True, True, "Detached house in Ramat Gan. Installed in July, three cameras."),
    CustomerSpec("Yossi Levi", True, True, True, "Ground-floor flat in Haifa with a shared building entrance."),
    CustomerSpec("Tamar Mizrahi", False, True, False,
                 "Two properties: the family house in Beer Sheva and a holiday flat in Eilat. No training consent."),
]
DEVICES = [
    DeviceSpec(0, "cohen_ramatgan", "hg-box-ramatgan",
               (("front_door", "Front door", 22), ("driveway", "Driveway", 16), ("garden", "Garden", 12)),
               "healthy", 0.1, 4, 412.6),
    DeviceSpec(1, "levi_haifa", "hg-box-haifa",
               (("main_entrance", "Main entrance", 22), ("side_gate", "Side gate", 14)),
               "warning", 0.1, 9, 38.2),
    DeviceSpec(2, "mizrahi_beersheva", "hg-box-beersheva",
               (("garage", "Garage", 24),), "critical", 7.0, 12, 118.4, collector_running=False),
    DeviceSpec(2, "mizrahi_eilat", "hg-box-eilat",
               (("front_door", "Front door", 10),), "offline", 33.0, 33 * 60 + 30, 211.0),
]

# (summary as the VLM writes it, YOLO classes seen)
NORMAL = {
    "front_door": [
        ("A courier in a yellow vest places a parcel on the doorstep, rings the bell and walks back toward the street.", ["person"]),
        ("A man in a grey hoodie approaches the front door, rings the bell, waits about ten seconds and then walks away.", ["person"]),
        ("A woman carrying two grocery bags walks up the path and unlocks the front door with a key.", ["person"]),
        ("A teenager in a school uniform steps out of the front door, locks it and walks toward the gate.", ["person"]),
        ("Two people in dark jackets stand at the front door talking while one of them knocks twice.", ["person"]),
    ],
    "driveway": [
        ("A white hatchback pulls into the driveway and a woman steps out and opens the trunk.", ["car", "person"]),
        ("A man walks down the driveway toward the street carrying a gym bag.", ["person"]),
        ("A delivery van stops at the end of the driveway and the driver walks up with a parcel.", ["truck", "person"]),
        ("A dark SUV reverses out of the driveway and drives off down the street.", ["car"]),
    ],
    "garden": [
        ("A man waters the plants along the garden fence with a hose.", ["person"]),
        ("A woman walks across the lawn with a laundry basket toward the clothes line.", ["person"]),
        ("A child kicks a ball across the lawn while an adult watches from the patio.", ["person"]),
    ],
    "main_entrance": [
        ("A resident in a blue jacket swipes a key fob and enters the building lobby.", ["person"]),
        ("A man pushing a stroller waits at the entrance door until a neighbour holds it open.", ["person"]),
        ("A food courier parks a scooter outside the entrance and walks in carrying a delivery bag.", ["person", "motorcycle"]),
        ("A woman with a small dog leaves the building and turns left along the pavement.", ["person", "dog"]),
    ],
    "side_gate": [
        ("A man in work clothes opens the side gate with a key and carries a ladder inside.", ["person"]),
        ("A woman wheels a bin through the side gate and leaves it by the wall.", ["person"]),
    ],
    "garage": [
        ("A grey sedan pulls up to the garage door and the driver gets out and lifts the door by hand.", ["car", "person"]),
        ("A man carries cardboard boxes from the garage to a parked van.", ["person", "truck"]),
        ("A woman parks a small car in front of the garage and walks toward the house.", ["car", "person"]),
    ],
}
SUSPICIOUS = {
    "front_door": [
        ("A person in a dark hooded jacket and gloves tries the front door handle twice and looks up at the camera before walking away.",
         "Unknown person tried the door handle during the night."),
        ("A person with a face covering crouches at the front door lock for several seconds.",
         "Person tampering with the door lock."),
    ],
    "driveway": [
        ("A person walks slowly along the driveway looking into the parked car windows and tries a door handle.",
         "Unknown person checking parked cars."),
    ],
    "garden": [
        ("A person climbs over the back fence into the garden and crouches near the shed.",
         "Unknown person entered the garden over the fence."),
    ],
    "main_entrance": [
        ("A person follows a resident through the door without a key and lingers in the doorway checking the mailboxes.",
         "Unknown person entered behind a resident and loitered."),
    ],
    "side_gate": [
        ("A person in a dark cap shakes the side gate lock and looks over the wall.",
         "Unknown person tried to open the side gate."),
    ],
    "garage": [
        ("A person forces the garage side door with a tool while a second person watches the street.",
         "Forced entry attempt at the garage."),
    ],
}
FALSE_POSITIVES = [
    ("A cat walks along the wall; no people are present.", ["cat"]),
    ("Tree branches sway in the wind and cast moving shadows; nothing else is visible.", ["person"]),
    ("A car passes on the street in the background and does not stop.", ["car"]),
    ("Nothing notable is happening.", ["person"]),
    ("Headlights sweep across the wall as a vehicle turns on the road; no people are visible.", ["car"]),
    ("A bird lands on the railing and flies away.", ["bird"]),
]
PROMPT_HEAD = ("You are the eyes of a home security system. These are sequential frames (one short clip of a few\n"
               "seconds) from the homeowner's own camera \"{camera}\", local time {local}.\n\nWrite \"summary\": what happ")
FEEDBACK_NOTES = {"true_alert": "", "false_alarm": "Just the neighbour.", "expected": "That is the cleaner.",
                  "real_but_wrong": "It was a person but not at the door."}


# ---------------------------------------------------------------- the synthetic events

@dataclass
class Plan:
    site: str
    camera: str
    ts: int
    kind: str  # alert | fp | paused | trigger | random
    ai: str  # real | fallback | failed | none
    summary: str = ""
    classes: list = field(default_factory=list)
    suspicious: bool = False
    reason: str = ""
    copies: tuple = ("training",)
    dispatch_failed: bool = False
    boxes: bool = False
    feedback: Optional[str] = None
    seed: int = 0

    @property
    def stem(self) -> str:
        return f"{self.camera}_{self.ts}_{self.kind}"

    @property
    def local(self) -> datetime:
        return datetime.fromtimestamp(self.ts, timezone.utc) + UTC_PLUS

    @property
    def day(self) -> str:
        return f"{self.local:%Y-%m-%d}"


def _hour_weight(hour: int) -> float:
    return [0.5, 0.4, 0.3, 0.3, 0.4, 0.8, 1.2, 1.8, 2.2, 1.8, 1.4, 1.4, 1.6, 1.6, 1.4, 1.6, 2.0, 2.6, 3.0, 2.8,
            2.4, 1.8, 1.2, 0.8][hour]


def _sample_ts(rng: random.Random, spec: DeviceSpec) -> int:
    lo = (NOW - timedelta(hours=46.8)).timestamp()
    hi = (NOW - timedelta(hours=spec.events_until_h, minutes=6)).timestamp()
    while True:
        ts = rng.uniform(lo, hi)
        hour = (datetime.fromtimestamp(ts, timezone.utc) + UTC_PLUS).hour
        if rng.random() * 3.0 < _hour_weight(hour):
            return int(ts)


def build_plans(rng: random.Random) -> list:
    plans: list[Plan] = []
    used: set = set()
    for spec in DEVICES:
        for camera, _, count in spec.cameras:
            times = []
            while len(times) < count:
                ts = _sample_ts(rng, spec)
                if (camera, ts) not in used and all(abs(ts - t) > 40 for t in times):
                    used.add((camera, ts))
                    times.append(ts)
            if spec.verdict in ("healthy", "warning"):  # a live camera has something in the last 8 hours
                times[0] = int((NOW - timedelta(hours=rng.uniform(0.3, 7.5))).timestamp())
            for ts in times:
                plans.append(_plan_one(rng, spec, camera, ts))
    plans.sort(key=lambda p: (p.ts, p.site, p.camera))
    for i, plan in enumerate(plans):
        plan.seed = SEED * 1000 + i
    return plans


def _plan_one(rng: random.Random, spec: DeviceSpec, camera: str, ts: int) -> Plan:
    hour = (datetime.fromtimestamp(ts, timezone.utc) + UTC_PLUS).hour
    night = hour < 6 or hour >= 21
    kind = rng.choices(["alert", "fp", "trigger", "random", "paused"], [34, 22, 22, 10, 12])[0]
    plan = Plan(spec.site, camera, ts, kind, "none")
    if kind == "alert":
        plan.suspicious = rng.random() < (0.35 if night else 0.1)
        if plan.suspicious:
            plan.summary, plan.reason = rng.choice(SUSPICIOUS[camera])
            plan.classes = ["person"]
        else:
            plan.summary, plan.classes = rng.choice(NORMAL[camera])
        plan.ai = rng.choices(["real", "fallback", "failed", "none"], [74, 10, 7, 9])[0]
        plan.copies = ("production",) if plan.ai == "none" else (
            ("production", "training") if rng.random() < 0.7 else ("training",))
        plan.dispatch_failed = "production" in plan.copies and rng.random() < 0.05
        if "production" in plan.copies and rng.random() < 0.75:
            plan.feedback = rng.choices(["true_alert", "false_alarm", "expected", "real_but_wrong"],
                                        [46, 30, 16, 8])[0]
            if plan.suspicious and plan.feedback == "false_alarm":
                plan.feedback = "true_alert"
    elif kind == "fp":
        plan.summary, plan.classes = rng.choice(FALSE_POSITIVES)
        plan.ai = rng.choices(["real", "fallback"], [80, 20])[0]
        plan.copies = ("production", "training") if rng.random() < 0.6 else ("training",)
        if "production" in plan.copies and rng.random() < 0.65:
            plan.feedback = rng.choices(["false_alarm", "true_alert"], [85, 15])[0]
    elif kind == "paused":
        plan.classes = [rng.choice(["person", "car"])]
    elif kind == "trigger":
        plan.classes = rng.choice([["person"], ["person"], ["car"], ["dog"], ["cat"], ["person", "car"]])
    else:
        plan.classes = rng.choice([[], ["person"], ["car"]])
    plan.boxes = ("training" in plan.copies and kind in ("alert", "trigger", "fp") and rng.random() < 0.55)
    return plan


# ---------------------------------------------------------------- box output (meta, clips, feedback, heartbeat)

def _box_path(plan: Plan, u: float, cls: str) -> tuple:
    """Normalised (xc, yc, w, h) of the subject at progress u in [0, 1]; the clip renderer uses it too."""
    r = random.Random(plan.seed)
    x0, x1 = r.uniform(0.15, 0.4), r.uniform(0.55, 0.85)
    if r.random() < 0.5:
        x0, x1 = x1, x0
    x = x0 + (x1 - x0) * u
    if cls in ("car", "truck", "bus"):
        return x, 0.66, 0.34, 0.22
    if cls in ("cat", "dog", "bird"):
        return x, 0.78, 0.12, 0.1
    return x, 0.62, 0.11, 0.42


def _vlm_answer(plan: Plan, people: int) -> dict:
    label = "suspicious" if plan.suspicious else "normal"
    return {"summary": plan.summary, "label": label, "people": people,
            "vehicle_moving": any(c in plan.classes for c in ("car", "truck", "motorcycle"))}


def _meta(plan: Plan, root: str, rng: random.Random) -> dict:
    n_frames = CLIP_FPS * CLIP_SEC
    start = plan.ts - 2.5 + rng.uniform(-0.2, 0.2)
    counts = {c: (2 if (c == "person" and rng.random() < 0.15) else 1) for c in dict.fromkeys(plan.classes)}
    conf = {c: round(rng.uniform(0.42, 0.93), 3) for c in counts}
    meta: dict[str, Any] = {
        "camera_name": plan.camera,
        "kind": "false_positive" if plan.kind == "fp" else plan.kind,
        "clip_path": f"clips\\{plan.camera}\\{plan.day}\\{plan.stem}.mp4",
        "clip_start_ts": start,
        "clip_end_ts": start + CLIP_SEC,
        "clip_start_local": f"{datetime.fromtimestamp(start, timezone.utc) + UTC_PLUS:%Y-%m-%d %H:%M:%S}",
        "clip_end_local": f"{datetime.fromtimestamp(start + CLIP_SEC, timezone.utc) + UTC_PLUS:%Y-%m-%d %H:%M:%S}",
        "duration_sec": float(CLIP_SEC),
        "frames_written": n_frames,
        "fps_estimated": float(CLIP_FPS),
        "codec": "h264",
        "buffer": {"store_fps": float(CLIP_FPS), "store_size": [CLIP_W, CLIP_H]},
        "yolo": {"class_counts": counts, "class_max_conf": conf, "trigger_classes": sorted(counts),
                 "trigger_detected": bool(counts)},
    }
    if plan.kind == "paused":
        meta["alert"] = {"summary": "", "alert_command": "[none]", "alert_reason": "alerts paused by the owner",
                         "labels": sorted(counts), "muted": True, "paused": True}
        return meta
    if plan.kind in ("alert", "fp"):
        people = counts.get("person", 0)
        command = "[none]" if plan.kind == "fp" else ("[call_owner]" if plan.suspicious else "[send_message]")
        alert: dict[str, Any] = {
            "summary": plan.summary, "label": "suspicious" if plan.suspicious else "normal",
            "alert_command": command, "alert_reason": plan.reason, "labels": sorted(counts), "muted": False}
        if plan.ai == "fallback":
            alert.update(summary=f"A {plan.classes[0] if plan.classes else 'person'} was detected near the camera.",
                         label="normal", alert_reason="")
        elif plan.ai == "failed":
            alert.update(summary="", alert_reason="AI unavailable; sent as a precaution" if command != "[none]" else "")
        if plan.kind == "fp":
            alert["false_positive"] = True
            alert["vlm"] = {"people": 0, "vehicle_moving": False}
        if root == "production" and plan.kind == "alert":
            chat = [{"chat_id": "-100100", "ok": not plan.dispatch_failed, "message_id": 1 + plan.ts % 900}]
            alert["dispatch"] = {"channel": "telegram", "telegram": {"command": command, "telegram": {
                "sent": not plan.dispatch_failed, "results": chat}}}
        meta["alert"] = alert
        if root == "dataset" and plan.ai != "none":
            local = f"{plan.local:%H:%M:%S}"
            frames = [f"vlm_crops\\{plan.camera}\\{plan.day}\\{plan.stem}_f{i}.jpg" for i in range(5)]
            raw = f"responses\\{plan.camera}\\{plan.day}\\{plan.stem}.model_raw.txt"
            meta["teacher"] = {"model": "gpt-4o", "prompt_version": "2026-10-03.tagged-rules-label",
                               "prompt": PROMPT_HEAD.format(camera=plan.camera, local=local),
                               "input_frames": frames, "raw_path": raw if plan.ai == "real" else None,
                               "temperature": 0}
            if plan.ai == "real":
                meta["model_response"] = _vlm_answer(plan, people)
            elif plan.ai == "fallback":
                meta["model_response"] = {"summary": ""}
    if root == "dataset" and plan.boxes:
        frames = []
        for index in range(0, n_frames, 3):
            frames.append({
                "frame_index": index, "approx_time_offset_sec": round(index / CLIP_FPS, 4),
                "image_path": f"yolo\\images\\{plan.camera}\\{plan.day}\\{plan.stem}_f{index:04d}.jpg",
                "label_path": f"yolo\\labels\\{plan.camera}\\{plan.day}\\{plan.stem}_f{index:04d}.txt",
                "num_boxes": len(counts)})
        meta["yolo_export"] = {"enabled": True, "export_fps": 1.7, "conf": 0.25, "images_root": "yolo\\images",
                               "labels_root": "yolo\\labels", "exported_frames": frames}
    return meta


def _label_text(plan: Plan, index: int, rng: random.Random) -> Optional[str]:
    """YOLO weak label lines for sampled frame `index` (COCO ids); None leaves the frame "not run"."""
    if rng.random() < 0.06:
        return None
    n_frames = CLIP_FPS * CLIP_SEC
    lines = []
    for cls in dict.fromkeys(plan.classes):
        if rng.random() < 0.08:
            continue  # the detector missed it on this frame
        xc, yc, w, h = _box_path(plan, index / (n_frames - 1), cls)
        lines.append(f"{COCO_IDS[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _scene_frame(plan: Plan, u: float):
    """One synthetic camera frame: night or day yard, with the subjects drawn where the labels say they are."""
    import cv2
    import numpy as np

    hour = plan.local.hour
    dark = 0.38 if (hour < 6 or hour >= 20) else 1.0
    hue = (sum(map(ord, plan.camera)) * 7) % 40
    img = np.zeros((CLIP_H, CLIP_W, 3), np.uint8)
    for y in range(CLIP_H):  # sky to ground gradient
        t = y / CLIP_H
        img[y, :] = (int((150 + hue - 60 * t) * dark), int((165 - 50 * t) * dark), int((170 - 90 * t) * dark))
    cv2.rectangle(img, (0, int(CLIP_H * 0.74)), (CLIP_W, CLIP_H), (int(70 * dark), int(95 * dark), int(80 * dark)), -1)
    cv2.rectangle(img, (int(CLIP_W * 0.04), int(CLIP_H * 0.2)), (int(CLIP_W * 0.3), int(CLIP_H * 0.74)),
                  (int(150 * dark), int(150 * dark), int(165 * dark)), -1)  # a wall
    cv2.rectangle(img, (int(CLIP_W * 0.1), int(CLIP_H * 0.4)), (int(CLIP_W * 0.2), int(CLIP_H * 0.74)),
                  (int(60 * dark), int(80 * dark), int(110 * dark)), -1)  # a door
    for cls in dict.fromkeys(plan.classes):
        xc, yc, w, h = _box_path(plan, u, cls)
        x0, y0 = int((xc - w / 2) * CLIP_W), int((yc - h / 2) * CLIP_H)
        x1, y1 = int((xc + w / 2) * CLIP_W), int((yc + h / 2) * CLIP_H)
        if cls == "person":
            cv2.rectangle(img, (x0, y0 + (y1 - y0) // 4), (x1, y1), (50, 60, 120), -1)
            cv2.circle(img, ((x0 + x1) // 2, y0 + (y1 - y0) // 7), max(3, (x1 - x0) // 3), (150, 180, 220), -1)
        elif cls in ("car", "truck", "bus", "motorcycle"):
            cv2.rectangle(img, (x0, y0 + (y1 - y0) // 3), (x1, y1), (170, 170, 175), -1)
            cv2.rectangle(img, (x0 + (x1 - x0) // 5, y0), (x1 - (x1 - x0) // 5, y0 + (y1 - y0) // 2), (130, 130, 140), -1)
            cv2.circle(img, (x0 + (x1 - x0) // 4, y1), 5, (25, 25, 25), -1)
            cv2.circle(img, (x1 - (x1 - x0) // 4, y1), 5, (25, 25, 25), -1)
        else:
            cv2.ellipse(img, ((x0 + x1) // 2, (y0 + y1) // 2), (max(4, (x1 - x0) // 2), max(3, (y1 - y0) // 2)),
                        0, 0, 360, (90, 110, 150), -1)
    return img


def _render_clip(plan: Plan) -> bytes:
    """A small real H.264 mp4 (moov first), so the media worker and ffprobe treat it like a box clip."""
    n = CLIP_FPS * CLIP_SEC
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{CLIP_W}x{CLIP_H}",
           "-r", str(CLIP_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "32", "-movflags", "+faststart", "-an", "-f", "mp4", "out.mp4"]
    # +faststart rewrites the file, so the output goes to a temp file rather than a pipe
    with tempfile.TemporaryDirectory(prefix="hgdemo_clip_") as tmp:
        out = Path(tmp) / "clip.mp4"
        cmd[-1] = str(out)
        raw = b"".join(_scene_frame(plan, i / (n - 1)).tobytes() for i in range(n))
        subprocess.run(cmd, input=raw, check=True, capture_output=True, timeout=60)
        return out.read_bytes()


def _tiny_jpeg() -> bytes:
    import cv2
    import numpy as np

    img = np.full((36, 64, 3), (90, 110, 120), np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _heartbeat(spec: DeviceSpec, plans: list, template: dict) -> dict:
    hb = dict(template)
    mine = [p for p in plans if p.site == spec.site]
    hb.update(site=spec.site, host=spec.host, local_ip=f"192.168.1.{20 + DEVICES.index(spec)}",
              time_utc=f"{NOW - timedelta(minutes=spec.heartbeat_age_min):%Y-%m-%dT%H:%M:%SZ}",
              collector_running=spec.collector_running, disk_free_gb=spec.disk_gb, stopped=False,
              clips_live=2 + len(mine) % 5, clips_outbox=3 + len(mine) % 11)
    cameras = {}
    for camera, _, _ in spec.cameras:
        times = [p.ts for p in mine if p.camera == camera]
        newest = f"{datetime.fromtimestamp(max(times), timezone.utc):%Y-%m-%dT%H:%M:%SZ}" if times else None
        cameras[camera] = {"clips_waiting": len([t for t in times if t > (NOW - timedelta(hours=1)).timestamp()]),
                           "newest_clip_utc": newest}
    hb["cameras"] = cameras
    hb["newest_clip_utc"] = max((c["newest_clip_utc"] for c in cameras.values() if c["newest_clip_utc"]), default=None)
    return hb


def _builders():
    """tests/cloud/builders.py (shared fixture builders: keys, feedback bodies, heartbeat template), by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("hg_demo_builders", REPO / "tests" / "cloud" / "builders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_bucket(s3client, plans: list, rng: random.Random) -> dict:
    """Upload everything the boxes would have written; returns {stem: feedback body} for later reference."""
    b = _builders()

    s3client.create_bucket(Bucket=BUCKET)
    jpeg = _tiny_jpeg()
    heartbeat_template = b.fixture_json("heartbeat.json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        clips = list(pool.map(_render_clip, plans))
    for plan, clip in zip(plans, clips):
        prod_site, ds_site = f"production_{plan.site}", f"dataset_{plan.site}"
        for copy in plan.copies:
            root = "production" if copy == "production" else "dataset"
            top = f"{root}_{plan.site}"
            meta = _meta(plan, root, rng)
            b.put(s3client, f"{top}/meta/{plan.camera}/{plan.day}/{plan.stem}.meta.json", meta)
            if rng.random() > 0.03 or root == "production":  # a few training clips never made it up
                b.put(s3client, f"{top}/clips/{plan.camera}/{plan.day}/{plan.stem}.mp4", clip)
            if root == "dataset" and plan.ai != "none":
                for i in range(5):
                    b.put(s3client, f"{top}/vlm_crops/{plan.camera}/{plan.day}/{plan.stem}_f{i}.jpg", jpeg)
                if plan.ai == "real":
                    b.put(s3client, f"{top}/responses/{plan.camera}/{plan.day}/{plan.stem}.model_raw.txt",
                          json.dumps(meta["model_response"]))
            if root == "dataset" and plan.boxes:
                for frame in meta["yolo_export"]["exported_frames"]:
                    index = frame["frame_index"]
                    base = f"{top}/yolo/%s/{plan.camera}/{plan.day}/{plan.stem}_f{index:04d}"
                    b.put(s3client, (base % "images") + ".jpg", jpeg)
                    text = _label_text(plan, index, rng)
                    if text is not None:
                        b.put(s3client, (base % "labels") + ".txt", text)
        if plan.feedback:
            when = datetime.fromtimestamp(plan.ts + rng.randint(120, 1500), timezone.utc)
            body = b.feedback_body(plan.stem, plan.feedback, f"{when:%Y-%m-%dT%H:%M:%SZ}")
            body["alert"].update(camera=plan.camera, summary=plan.summary[:80], ts=float(plan.ts))
            body["note"] = FEEDBACK_NOTES[plan.feedback]
            body["raw_text"] = FEEDBACK_NOTES[plan.feedback] or "👍"
            body["action"] = "none"
            ms = int(when.timestamp() * 1000)
            b.put(s3client, f"{prod_site}/feedback/{plan.camera}/{plan.day}/{plan.stem}_{ms}.feedback.json", body)
    for spec in DEVICES:
        b.put(s3client, f"dataset_{spec.site}/_status/heartbeat.json", _heartbeat(spec, plans, heartbeat_template))
    # one general owner message (not about an alert) and one reply to an alert that was cleaned up already
    general = b.fixture_json("feedback.json")
    general.update(alert=None, time_utc=f"{NOW - timedelta(hours=9):%Y-%m-%dT%H:%M:%SZ}", raw_text="mute the garden tonight")
    b.put(s3client, f"production_{DEVICES[0].site}/feedback/_general/2026-10-03/general_1790999999000.feedback.json", general)
    # two objects the indexer cannot use: they show up as index problems
    b.put(s3client, f"dataset_{DEVICES[0].site}/meta/front_door/2026-10-02/front_door_1790935000_trigger.meta.json",
          "{ this is not json")
    b.put(s3client, f"production_{DEVICES[1].site}/meta/side_gate/2026-10-02/side_gate_1790936000_alert.meta.json",
          '{"camera_name": "side_gate", "kind": "al')  # an upload cut short
    return {}


# ---------------------------------------------------------------- human state

def seed_state(engine, plans: list, rng: random.Random) -> dict:
    """Customers, devices, staff, cameras (before indexing); returns lookup ids."""
    from . import auth
    from .db import session_scope
    from .models import Camera, Customer, Device, Staff

    ids: dict[str, Any] = {"customers": [], "devices": {}}
    with session_scope(engine) as s:
        for spec in CUSTOMERS:
            c = Customer(name=spec.name, timezone="Asia/Jerusalem", consent_live=spec.live,
                         consent_recordings=spec.recordings, consent_training=spec.training, notes=spec.notes)
            s.add(c)
            s.flush()
            ids["customers"].append(c.id)
        for i, spec in enumerate(DEVICES):
            d = Device(device_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"homeguard-demo-{spec.site}")), site=spec.site,
                       tailscale_host=spec.host, ssh_user="ameer", customer_id=ids["customers"][spec.customer],
                       enrolled_at=NOW - timedelta(days=90 - 17 * i))
            s.add(d)
            s.flush()
            ids["devices"][spec.site] = (d.id, d.device_id)
            for camera, shown, _ in spec.cameras:
                s.add(Camera(device_pk=d.id, name=camera, display_name=shown))
        pw = auth.hash_password("demo-password-not-a-secret")
        ids["staff"] = {}
        for role, name, email in (("admin", "Ameer Faour", "ameer@homeguard.example"),
                                  ("support", "Noa Shapira", "noa@homeguard.example"),
                                  ("labeler", "Eden Peretz", "eden@homeguard.example")):
            st = Staff(email=email, name=name, role=role, password_hash=pw, totp_secret="JBSWY3DPEHPK3PXP")
            s.add(st)
            s.flush()
            ids["staff"][role] = st.id
    return ids


def seed_review_and_collections(engine, ids: dict, rng: random.Random) -> dict:
    from sqlalchemy import select

    from .db import session_scope
    from .models import Collection, CollectionItem, Customer, Device, Event, ReviewState

    out: dict[str, Any] = {}
    admin, labeler = ids["staff"]["admin"], ids["staff"]["labeler"]
    with session_scope(engine) as s:
        rows = s.execute(select(Event, Customer.consent_training).join(Device, Device.id == Event.device_pk)
                         .join(Customer, Customer.id == Device.customer_id).order_by(Event.start_ts)).all()
        for ev, _ in rows:
            old = ev.start_ts < (NOW - timedelta(hours=24)).timestamp()
            flagged = ev.kind == "alert" and (ev.label == "suspicious") and rng.random() < 0.5
            reviewed = (not flagged) and ev.kind in ("alert", "false_positive") and rng.random() < (0.8 if old else 0.35)
            if reviewed or flagged:
                s.add(ReviewState(event_id=ev.id, reviewed=reviewed, flagged=flagged, by=admin,
                                  at=datetime.fromtimestamp(ev.start_ts, timezone.utc) + timedelta(hours=rng.uniform(1, 5))))
        night, couriers = [], []
        for ev, consent in rows:
            comp = ev.completeness or {}
            if not consent:
                continue
            if ev.kind not in ("alert", "false_positive") or comp.get("ai") != "real":
                continue
            local_hour = (datetime.fromtimestamp(ev.start_ts, timezone.utc) + UTC_PLUS).hour
            if ev.label == "suspicious" or local_hour < 7 or local_hour >= 19:
                night.append(ev.id)
            if any(word in (ev.summary or "") for word in ("parcel", "delivery", "courier")):
                couriers.append(ev.id)
        mizrahi = [ev.id for ev, consent in rows if not consent and ev.kind == "alert"][:2]
        quiet = [ev.id for ev, consent in rows if consent and ev.kind == "alert" and (ev.completeness or {}).get("ai") == "real"
                 and ev.id not in night and ev.id not in couriers][:6]
        specs = [("Night-time visitors", "Events after 19:00 or before 07:00, or labelled suspicious, with a real AI description.",
                  admin, NOW - timedelta(days=2, hours=3), night[:14] + mizrahi),
                 ("Deliveries and couriers", "Couriers and parcel drop-offs, plus a few ordinary arrivals as negatives.",
                  labeler, NOW - timedelta(days=1, hours=5), couriers[:10] + quiet)]
        for name, desc, by, when, members in specs:
            col = Collection(name=name, description=desc, created_by=by, created_at=when)
            s.add(col)
            s.flush()
            out[name] = col.id
            for event_id in dict.fromkeys(members):
                s.add(CollectionItem(collection_id=col.id, event_id=event_id, added_by=by, added_at=when))
    return out


def seed_audit(engine, ids: dict, rng: random.Random, count: int) -> None:
    """Earlier staff activity, so the audit page has history. Written through the real audit.record."""
    from . import audit
    from .db import session_scope

    admin, support, labeler = ids["staff"]["admin"], ids["staff"]["support"], ids["staff"]["labeler"]
    cust = ids["customers"]
    dev = {site: pair for site, pair in ids["devices"].items()}
    script: list[tuple] = []
    base = NOW - timedelta(days=6, hours=2)
    for i, c in enumerate(CUSTOMERS):
        script.append((admin, "customer_create", f"customer/{cust[i]}", "", cust[i], None, {"name": c.name}))
    for i, spec in enumerate(DEVICES):
        pk, device_id = dev[spec.site]
        script.append((admin, "device_enroll", spec.site, "", cust[spec.customer], device_id, None))
    script.append((admin, "customer_update", f"customer/{cust[2]}", "Consent form received by email.", cust[2], None,
                   {"consent_recordings": True}))
    script.append((admin, "customer_update", f"customer/{cust[0]}", "Owner agreed to training use.", cust[0], None,
                   {"consent_training": True}))
    for who, times in ((support, 3), (admin, 2), (labeler, 2)):
        for _ in range(times):
            script.append((who, "login", "", "", None, None, None))
    sites = list(dev)
    for n in range(6):
        site = sites[n % 2]
        script.append((support if n % 2 else admin, "event_view", f"event/{20 + n * 7}", "",
                       cust[DEVICES[n % 2].customer], dev[site][1], None))
    script.append((support, "media_view", "artifact/412", "Owner called about the 04:10 alert.", cust[0], dev[sites[0]][1],
                   {"purpose": "support"}))
    script.append((support, "media_denied", "artifact/1190", "No recording consent for live view.", cust[2], dev[sites[2]][1],
                   {"reason": "No recording consent", "purpose": "support"}))
    script.append((admin, "review", "event/31", "", cust[0], dev[sites[0]][1], {"flagged": True}))
    script.append((admin, "collection_create", "collection/1", "", None, None, {"name": "Night-time visitors"}))
    script.append((labeler, "collection_create", "collection/2", "", None, None, {"name": "Deliveries and couriers"}))
    script.append((admin, "collection_add", "collection/1", "", None, None, {"event_ids": [12, 31, 44]}))
    script.append((labeler, "collection_add", "collection/2", "", None, None, {"event_ids": [8, 19, 27]}))
    script = script[:count]
    assert len(script) == count, (len(script), count)
    span = (NOW - timedelta(hours=26) - base).total_seconds()
    with session_scope(engine) as s:
        for i, (staff, action, target, reason, customer, device_id, detail) in enumerate(script):
            ts = base + timedelta(seconds=span * i / max(1, count - 1) + rng.randint(0, 600))
            audit.record(s, staff, action, target=target, reason=reason, customer_id=customer, device_id=device_id,
                         detail=detail, ts=ts)


# ---------------------------------------------------------------- calling the routes

def _clean(value: Any) -> Any:
    """Presigned URLs carry a signature and an expiry from the wall clock: keep scheme, host and path only."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, str) and value.startswith("http") and "?" in value:
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return value


class Recorder:
    def __init__(self, client, out: Path):
        self.client, self.out = client, out
        self.entries: list[dict] = []

    def call(self, name: str, role: str, headers: dict, method: str, path: str, body: Any = None,
             params: Optional[dict] = None, optional: bool = False) -> Any:
        resp = self.client.request(method, "/v1" + path, headers=headers, json=body, params=params)
        if resp.status_code != 200:
            if optional:
                return None
            raise RuntimeError(f"{method} {path} as {role}: {resp.status_code} {resp.text[:300]}")
        data = _clean(resp.json())
        file = f"{name}{'.labeler' if role == 'labeler' else ''}.json"
        (self.out / file).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        query = "" if not params else "?" + "&".join(f"{k}={v}" for k, v in params.items())
        self.entries.append({"file": file, "role": role, "method": method, "path": "/v1" + path + query})
        return data


def record_routes(client, out: Path, tokens: dict, ids: dict, collections: dict, event_ids: dict) -> list:
    rec = Recorder(client, out)
    admin = {"Authorization": f"Bearer {tokens['admin']}"}
    labeler = {"Authorization": f"Bearer {tokens['labeler']}"}
    horizon = {"from_utc": "2026-10-01T13:00:00Z", "to_utc": "2026-10-03T13:00:00Z", "bucket": "hour"}

    # audit first: it then shows the staff activity before this run, not the views the run itself makes
    rec.call("audit", "admin", admin, "GET", "/audit")
    rec.call("index_problems", "admin", admin, "GET", "/index/problems")
    rec.call("me", "admin", admin, "GET", "/me")
    rec.call("customers", "admin", admin, "GET", "/customers")
    for cid in ids["customers"]:
        rec.call(f"customer_{cid}", "admin", admin, "GET", f"/customers/{cid}")
    rec.call("fleet", "admin", admin, "GET", "/fleet")
    rec.call("fleet_activity", "admin", admin, "GET", "/fleet/activity")
    rec.call("fleet_activity_48h", "admin", admin, "GET", "/fleet/activity", params={"hours": 48})
    page = rec.call("events", "admin", admin, "GET", "/events", params={"limit": 500, "with_total": "true"})
    assert page["next_cursor"] is None
    for cid in ids["customers"]:
        rec.call(f"events_customer_{cid}", "admin", admin, "GET", "/events", params={"customer_id": cid, "limit": 500})
        rec.call(f"events_density_customer_{cid}", "admin", admin, "GET", "/events/density",
                 params={**horizon, "customer_id": cid})
    rec.call("events_density", "admin", admin, "GET", "/events/density", params=horizon)
    rec.call("events_review_count", "admin", admin, "GET", "/events/review-count")
    for item in page["items"]:
        rec.call(f"event_{item['id']}", "admin", admin, "GET", f"/events/{item['id']}")
        rec.call(f"detections_{item['id']}", "admin", admin, "GET", f"/events/{item['id']}/detections")
    rec.call("studio_filters", "admin", admin, "GET", "/studio/filters")
    rec.call("collections", "admin", admin, "GET", "/studio/collections")
    for cid in collections.values():
        rec.call(f"collection_{cid}_items", "admin", admin, "GET", f"/studio/collections/{cid}/items",
                 params={"limit": 500})
    exports = rec.call("exports", "admin", admin, "GET", "/studio/exports")
    for exp in exports:
        rec.call(f"export_{exp['id']}", "admin", admin, "GET", f"/studio/exports/{exp['id']}")
    preview = {"collection_id": collections["Night-time visitors"], "name": "night-visitors-v2",
               "formats": ["yolo", "vlm_jsonl", "clips"], "split": {"train": 0.8, "val": 0.1, "test": 0.1},
               "include_fallback_ai": False}
    rec.call("export_preview", "admin", admin, "POST", "/studio/exports/preview", body=preview)

    # the labeler's view of the same routes (roles that the route refuses have no file)
    rec.call("me", "labeler", labeler, "GET", "/me")
    lpage = rec.call("events", "labeler", labeler, "GET", "/events", params={"limit": 500, "with_total": "true"})
    rec.call("events_density", "labeler", labeler, "GET", "/events/density", params=horizon)
    rec.call("events_review_count", "labeler", labeler, "GET", "/events/review-count")
    for item in lpage["items"]:
        rec.call(f"event_{item['id']}", "labeler", labeler, "GET", f"/events/{item['id']}")
        rec.call(f"detections_{item['id']}", "labeler", labeler, "GET", f"/events/{item['id']}/detections")
    rec.call("studio_filters", "labeler", labeler, "GET", "/studio/filters")
    rec.call("collections", "labeler", labeler, "GET", "/studio/collections")
    for cid in collections.values():
        rec.call(f"collection_{cid}_items", "labeler", labeler, "GET", f"/studio/collections/{cid}/items",
                 params={"limit": 500})
    rec.call("exports", "labeler", labeler, "GET", "/studio/exports")
    rec.call("export_preview", "labeler", labeler, "POST", "/studio/exports/preview",
             body={**preview, "collection_id": collections["Deliveries and couriers"], "name": "deliveries-v1"})
    return rec.entries


def save_media(engine, s3, out: Path) -> int:
    from sqlalchemy import select

    from .db import session_scope
    from .models import Artifact

    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    n = 0
    with session_scope(engine) as s:
        for art in s.scalars(select(Artifact).where(Artifact.role.in_(["thumbnail", "filmstrip"]),
                                                    Artifact.available.is_(True)).order_by(Artifact.id)):
            name = ("thumb_" if art.role == "thumbnail" else "filmstrip_") + f"{art.event_id}.jpg"
            (media / name).write_bytes(s3.get_bytes(art.s3_key))
            n += 1
    return n


def main(out_dir=None) -> Path:
    """Generate the demo data into `out_dir` (default docs/admin/demo) and return it."""
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.json")) + list((out / "media").glob("*.jpg") if (out / "media").exists() else []):
        old.unlink()
    saved_env = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                                                  "AWS_DEFAULT_REGION", "AWS_PROFILE")}
    os.environ.update(AWS_ACCESS_KEY_ID="testing", AWS_SECRET_ACCESS_KEY="testing", AWS_SESSION_TOKEN="testing",
                      AWS_DEFAULT_REGION="us-east-1")
    os.environ.pop("AWS_PROFILE", None)
    try:
        _generate(out)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _generate(out: Path) -> None:
    import boto3
    import pgserver
    from fastapi.testclient import TestClient
    from moto import mock_aws

    from . import audit, auth, indexer, media
    from .app import create_app
    from .manage import run_migrations
    from .models import Device, Staff
    from .s3 import S3
    from .settings import Settings
    from .db import session_scope

    rng = random.Random(SEED)
    plans = build_plans(rng)
    clock = [NOW]
    real_record = audit.record

    def record(session, staff_id, action, *args, **kwargs):  # routes stamp audit rows with the app clock
        if kwargs.get("ts") is None and len(args) < 6:
            kwargs["ts"] = clock[0]
        return real_record(session, staff_id, action, *args, **kwargs)

    root = tempfile.mkdtemp(prefix="hgdemo_pg_")
    server = None
    audit.record = record
    try:
        server = pgserver.get_server(root, cleanup_mode="stop")
        url = server.get_uri()
        run_migrations(url)
        with mock_aws():
            client_s3 = boto3.client("s3", region_name="us-east-1")
            s3 = S3(client_s3, BUCKET)
            seed_bucket(client_s3, plans, rng)
            settings = Settings(db_url=url, jwt_secret=JWT_SECRET)
            app = create_app(settings, s3=s3, init_db=False)
            app.state.clock = lambda: clock[0]
            app.state.export_runner = lambda job: job()  # exports build synchronously
            engine = app.state.engine
            ids = seed_state(engine, plans, rng)
            with session_scope(engine) as session:
                from sqlalchemy import select

                for dev in session.scalars(select(Device).order_by(Device.id)):
                    indexer.index_device(session, s3, dev, full_scan=True, now=NOW)
                    session.commit()
                while media.process_pending(session, s3, limit=50):
                    pass
            collections = seed_review_and_collections(engine, ids, rng)
            with TestClient(app) as client:
                with session_scope(engine) as session:
                    tokens = {role: auth.make_access_token(session.get(Staff, sid), settings)
                              for role, sid in ids["staff"].items()}
                admin = {"Authorization": f"Bearer {tokens['admin']}"}
                clock[0] = NOW - timedelta(hours=3, minutes=5)  # the export was requested this morning
                resp = client.post("/v1/studio/exports", headers=admin, json={
                    "collection_id": collections["Night-time visitors"], "name": "night-visitors-v1",
                    "formats": ["yolo", "vlm_jsonl"], "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "include_fallback_ai": False})
                if resp.status_code != 200 or resp.json()["state"] != "ready":
                    raise RuntimeError(f"demo export did not become ready: {resp.status_code} {resp.text[:400]}")
                clock[0] = NOW
                seed_audit(engine, ids, rng, count=29)
                entries = record_routes(client, out, tokens, ids, collections, {})
            n_media = save_media(engine, s3, out)
            app.state.engine.dispose()
        index = {"now_utc": f"{NOW:%Y-%m-%dT%H:%M:%SZ}",
                 "note": "Generated by home_guard_project.cloud.export_demo from the real routes; do not edit.",
                 "media": {"thumbnail": "media/thumb_{event_id}.jpg", "filmstrip": "media/filmstrip_{event_id}.jpg",
                           "count": n_media},
                 "files": entries}
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    finally:
        audit.record = real_record
        if server is not None:
            server.cleanup()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    target = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"demo data written to {target}")
