"""Stage-3 tag benchmark: do P1/P2/CAR1 tags drawn on the frames help the vision model say who did what,
and does it really READ them?

Clips: the human-tagged dataset (annotations/clips.jsonl + the Label Studio exports' videorectangle tracks, one
track = one person or vehicle, parsed by analysis.analyze.parse_export and placed on the clip's own frames with
its frame map and ls_convert.box_at, the 870a1dc / 62e9826 numbering). A clip qualifies with at least two
tracked people, or a tracked person and a tracked vehicle, both visible in the frames the box would send.

Frames: the box's own render_model_input on the clip's whole frames at 1 fps (the box's whole_frame_fallback
input), so every human box lands on the sent frame in exact pixels. Arms:

  A  the frames as the box sends them, the box's legacy prompt (inference.build_prompt) unchanged
  B  the same frames with the tags drawn (tag_overlay), the legacy prompt + TAGS_RULE (per_entity)
  S  B's frames with the strings of one pair swapped (P1<->P2, else CAR1<->CAR2): a model that reads the
     tags must swap its per-entity answers
  M  B's frames (tags drawn) with A's prompt unchanged: is any harm from the marks or from the extra task?

Commands (python -m home_guard_project.analysis.tag_bench <cmd>): prepare | run | judge | score | sheets. Every step
resumes: answers already saved are not asked again.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import re
import ssl
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..box import inference as inf
from ..box import providers
from ..data_collection import model_input as mi
from . import tag_overlay as to
from .analyze import MIN_BOX_SIDE, ParsedTask, _frame_map, parse_export
from .config import AnalysisConfig
from .utils.ls_convert import box_at

DATASET = r"C:/Users/ameer/Ameer/home_guard_data/dataset"
OUT = r"C:/Users/ameer/Ameer/home_guard_data/eval/tag_bench"
KEY_FILE = os.path.expanduser("~/.homeguard/api_key.env.bak-20261006-2338")
JUDGE_MODEL = "openai/gpt-4o-mini"
SAMPLE_FPS = 1.0
ARMS = ("A", "B", "S", "M")
GENERIC_CAMERA = "camera_1"      # uca/smarthome camera fields are crime categories ("Abuse"): never shown

TAGS_RULE = (
    "The frames carry drawn tags: each tracked person or vehicle has a thin outlined box with a solid label at "
    "its top-left corner, and the same label marks the same person or vehicle in every frame. Tags in these "
    "frames: {ids}.\n"
    'Also add "per_entity" to the JSON object: [{{"id": "<a tag from the list>", "action": "<what this one does '
    'across the frames, one short clause in English>", "object_or_target": "<what it holds, uses or acts on: an '
    'object, another tag such as CAR1, or a place; empty string if nothing>"}}], one item for every tag in the '
    "list. Read each tag from the frames.")

SCHEMA_TAGS: Dict[str, Any] = {
    **inf.VLM_SCHEMA,
    "properties": {**inf.VLM_SCHEMA["properties"], "per_entity": {
        "type": "array",
        "items": {"type": "object",
                  "properties": {"id": {"type": "string"}, "action": {"type": "string"},
                                 "object_or_target": {"type": "string"}},
                  "required": ["id", "action", "object_or_target"], "additionalProperties": False}}},
    "required": list(inf.VLM_SCHEMA["required"]) + ["per_entity"],
}
FORMAT_TAGS = {"type": "json_schema", "json_schema": {"name": "camera_report", "strict": True, "schema": SCHEMA_TAGS}}

# Label reading is judged BLIND: the judge never hears of a swap. It is shown the plain answer's two actions as
# options in a hashed order and one action from the other answer, and says which option it is about. (A first
# design told the judge the tags were swapped and asked "followed or ignored?": on two plain runs, where nothing
# was swapped, it still said "followed" 85 times in 179, so it was dropped.)
MATCH_JUDGE = """Two descriptions say what two different people or vehicles in one video clip do:
  Option 1: "{o1}"
  Option 2: "{o2}"
Another description, written separately, is about one of the two:
  "{q}"
Which option is about the same one? Judge by the action, the objects and the places, not by the wording.
Reply with JSON only: {{"match": "1" | "2" | "unclear", "reason": "<short>"}}
("unclear": both options fit equally well, or neither fits)."""

ATTR_JUDGE = """You check a vision model's per-tag answers against a human annotator's ground truth for one
security-camera clip. Every person and vehicle was tagged (P1, P2, CAR1, ...) on the frames from the human's own
boxes, so the positions below are true.

Human description of the clip (true; it may not mention everyone): "{desc}"
Where each tag's box is in the frames (true):
{positions}

The model said, per tag:
{per_entity}

For EACH tag in the model's list, judge whether its action (and object/target) is right for THAT tag:
- "correct": consistent with the human description AND fits that tag's position and movement (for example, the
  one who walks to the car is the tag whose box moves to the car; a parked car "is parked").
- "wrong": contradicts the description or belongs to a different tag (swapped roles, says it runs when it
  stands, says a parked car drives off, gives a person the action of another).
- "cant_tell": the description and the positions are not enough to decide.
Reply with JSON only: {{"verdicts": [{{"id": "<tag>", "verdict": "correct" | "wrong" | "cant_tell", "reason": "<short>"}}]}}"""

HARM_JUDGE = """Compare a vision model's summary of a security-camera clip with the human annotator's description,
which is the truth.

Human: "{desc}"
Model: "{summary}"

List:
- "invented": things the model states that are not there: people, objects or actions that contradict the human
  description or that a careful annotator would not have left out. Do not count wording, clothing detail, or
  extra detail that fits the description.
- "missed": actions in the human description that matter for safety or for who did what, which the model does not
  mention or contradicts.
Reply with JSON only: {{"invented": ["<short phrase>", ...], "missed": ["<short phrase>", ...]}}"""


# ----------------------------------------------------------------------------------------------------------
# clips and tracks
# ----------------------------------------------------------------------------------------------------------
def _cfg() -> AnalysisConfig:
    return AnalysisConfig(s3_bucket="", s3_prefix="", s3_region="", coco_labels={}, output_dir="")


def load_rows(dataset: str = DATASET) -> List[Dict[str, Any]]:
    with open(os.path.join(dataset, "annotations", "clips.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks(dataset: str = DATASET) -> Dict[Tuple[str, int], ParsedTask]:
    """(batch, LS task id) -> the parsed task, from the vehicle-fixed Label Studio exports."""
    out: Dict[Tuple[str, int], ParsedTask] = {}
    for path in sorted(glob.glob(os.path.join(dataset, "annotations", "label_studio", "*.vehicle_fixed.json"))):
        batch = os.path.basename(path).split(".")[0]
        for pt in parse_export(path, _cfg()):
            out[(batch, int(pt.task_id))] = pt
    return out


def kind_of(label: str) -> Optional[str]:
    if label in to.PERSON_KINDS:
        return "person"
    if label in to.VEHICLE_KINDS:
        return label
    return None


def entities_for(pt: ParsedTask, native_fps: float, size: Tuple[int, int]) -> List[to.Entity]:
    """The task's person and vehicle tracks as overlay entities on the clip's own frames (pixels of *size*)."""
    pt.native_fps = pt.native_fps or native_fps
    if pt.frame_space == "native":
        ls_count = pt.frames_count or max([t.frames_count for t in pt.tracks] + [0])
    else:
        ls_count = max([t.frames_count for t in pt.tracks] + [pt.frames_count, 0])
        if ls_count <= 0:
            ls_count = int(round(pt.duration_sec * pt.fps))
    fmap, kf_fps = _frame_map(pt, ls_count, None)
    ls_of = dict(fmap)
    w, h = size
    out = []
    for n, trk in enumerate(pt.tracks):
        kind = kind_of(trk.label)
        if kind is None:
            continue

        def at(i: int, seq=trk.sequence) -> Optional[to.Box]:
            ls = ls_of.get(i)
            b = box_at(seq, ls, kf_fps) if ls is not None else None
            if b is None or b["width"] / 100.0 < MIN_BOX_SIDE or b["height"] / 100.0 < MIN_BOX_SIDE:
                return None
            x, y = b["x"] / 100.0 * w, b["y"] / 100.0 * h
            return x, y, x + b["width"] / 100.0 * w, y + b["height"] / 100.0 * h

        out.append(to.Entity(f"t{n}", kind, at))
    return out


def read_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def local_time_of(meta: Dict[str, Any]) -> str:
    m = re.search(r"(\d{2}:\d{2}:\d{2})$", str(meta.get("clip_start_local") or ""))
    return m.group(1) if m else "12:00:00"


def swap_pair(ids: Sequence[str]) -> Optional[Tuple[str, str]]:
    if "P1" in ids and "P2" in ids:
        return "P1", "P2"
    if "CAR1" in ids and "CAR2" in ids:
        return "CAR1", "CAR2"
    return None


def id_order(ids: Sequence[str]) -> List[str]:
    return sorted(ids, key=lambda s: (0 if s.startswith("P") else 1, int(s.lstrip("PCAR"))))


def _zone(box: Tuple[int, int, int, int], w: int, h: int) -> str:
    cx, cy = (box[0] + box[2]) / 2.0 / w, (box[1] + box[3]) / 2.0 / h
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return f"{row}-{col}"


def _overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def positions(tagged: to.Tagged, kinds: Dict[str, str], size: Tuple[int, int]) -> Dict[str, str]:
    """id -> one line on where its true box is across the sent frames (for the judge)."""
    w, h = size
    out = {}
    n = len(tagged.frames)
    for eid in id_order(list(tagged.boxes)):
        seen = [(k, b) for k, b in enumerate(tagged.boxes[eid]) if b is not None]
        if not seen:
            continue
        (k0, b0), (k1, b1) = seen[0], seen[-1]
        c0 = ((b0[0] + b0[2]) / 2.0 / w, (b0[1] + b0[3]) / 2.0 / h)
        c1 = ((b1[0] + b1[2]) / 2.0 / w, (b1[1] + b1[3]) / 2.0 / h)
        moved = abs(c1[0] - c0[0]) > 0.1 or abs(c1[1] - c0[1]) > 0.1
        tall = 100.0 * float(np.median([(b[3] - b[1]) / h for _, b in seen]))
        line = (f"{eid} ({kinds.get(eid, '?')}): visible in frames {k0 + 1}-{k1 + 1} of {n}; "
                + (f"moves from {_zone(b0, w, h)} to {_zone(b1, w, h)}" if moved else f"stays at {_zone(b0, w, h)}")
                + f"; box height {tall:.0f}% of the frame")
        touching = []
        for other in id_order(list(tagged.boxes)):
            if other == eid:
                continue
            ks = [k + 1 for k, b in seen if tagged.boxes[other][k] is not None and _overlap(b, tagged.boxes[other][k])]
            if ks:
                touching.append(f"{other} in frames {ks[0]}-{ks[-1]}")
        if touching:
            line += "; its box overlaps " + ", ".join(touching)
        out[eid] = line
    return out


@dataclass
class Clip:
    row: Dict[str, Any]
    task: ParsedTask
    meta: Dict[str, Any]


def candidates(dataset: str = DATASET) -> Tuple[List[Clip], Dict[str, Counter]]:
    """Clips with >= 2 tracked people or a tracked person and a tracked vehicle (before the frames are read)."""
    tasks = load_tasks(dataset)
    counts: Dict[str, Counter] = {"rows": Counter(), "with_tracks": Counter(), "qualify_tracks": Counter()}
    out = []
    for row in load_rows(dataset):
        counts["rows"][row["source"]] += 1
        pt = tasks.get((row.get("batch"), int(row.get("ls_task_id") or -1)))
        if pt is None:
            continue
        counts["with_tracks"][row["source"]] += 1
        people = [t for t in pt.tracks if kind_of(t.label) == "person"]
        vehicles = [t for t in pt.tracks if kind_of(t.label) not in (None, "person")]
        if len(people) >= 2 or (people and vehicles):
            counts["qualify_tracks"][row["source"]] += 1
            meta_path = os.path.join(dataset, row["meta"]) if row.get("meta") else ""
            meta = json.load(open(meta_path, encoding="utf-8")) if meta_path and os.path.isfile(meta_path) else {}
            out.append(Clip(row, pt, meta))
    return out, counts


@dataclass
class Rendered:
    clip_id: str
    sent: mi.ModelInput
    tagged: to.Tagged
    swapped: Optional[to.Tagged]
    pair: Optional[Tuple[str, str]]
    kinds: Dict[str, str]
    positions: Dict[str, str]


def render(clip: Clip, dataset: str = DATASET) -> Optional[Rendered]:
    """The box's sent frames for *clip* plus their tagged and swapped copies; None when the clip is unreadable."""
    frames = read_frames(os.path.join(dataset, clip.row["clip"]))
    if not frames:
        return None
    fps = float(clip.meta.get("fps_estimated") or clip.task.native_fps or clip.task.fps)
    sent = mi.render_model_input(frames, {"fps": fps}, mi.ModelInputConfig(SAMPLE_FPS))
    h, w = frames[0].shape[:2]
    ents = entities_for(clip.task, fps, (w, h))
    tagged = to.tag_model_input(sent, ents)
    kinds = {tagged.ids[e.key]: e.kind for e in ents if e.key in tagged.ids}
    pair = swap_pair(list(tagged.ids.values()))
    swapped = to.tag_model_input(sent, ents, relabel={pair[0]: pair[1], pair[1]: pair[0]}) if pair else None
    return Rendered(clip.row["clip_id"], sent, tagged, swapped, pair, kinds,
                    positions(tagged, kinds, (sent.size or (w, h))))


def qualifies(r: Rendered) -> bool:
    kinds = list(r.kinds.values())
    people = kinds.count("person")
    return people >= 2 or (people >= 1 and len(kinds) > people)


# ----------------------------------------------------------------------------------------------------------
# asking
# ----------------------------------------------------------------------------------------------------------
def read_key(name: str, path: str = KEY_FILE) -> str:
    with open(path, encoding="utf-8") as f:
        for line in f:
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == name:
                return v.strip().strip('"').strip("'")
    raise KeyError(f"{name} not in {path}")


def client() -> Any:
    import httpx  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    return OpenAI(api_key=read_key("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", timeout=90.0,
                  http_client=httpx.Client(verify=ssl.create_default_context(), timeout=90.0))


def key_usage(c: Any) -> Optional[float]:
    """Dollars this key has spent so far (OpenRouter /key), or None."""
    try:
        import httpx  # noqa: PLC0415

        r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {c.api_key}"},
                      verify=ssl.create_default_context(), timeout=30)
        return float(r.json()["data"]["usage"])
    except Exception:  # noqa: BLE001
        return None


def camera_of(row: Dict[str, Any]) -> str:
    return row.get("camera") if row.get("source") in ("house", "external") else GENERIC_CAMERA


def prompt_of(arm: str, clip: Clip, ids: Sequence[str]) -> Tuple[str, Dict[str, Any]]:
    """The arm's prompt and response format. A is the box's legacy prompt byte for byte."""
    base = inf.build_prompt(camera_of(clip.row), 0, local_time_of(clip.meta), 0, 0, owner_language="en")
    if arm in ("A", "M"):
        return base, inf.VLM_RESPONSE_FORMAT
    return base + "\n\n" + TAGS_RULE.format(ids=", ".join(id_order(ids))), FORMAT_TAGS


def frames_of(arm: str, r: Rendered) -> List[np.ndarray]:
    if arm == "A":
        return r.sent.frames
    return r.swapped.frames if arm == "S" else r.tagged.frames  # type: ignore[union-attr]


def _complete(c: Any, model: str, content: List[Dict[str, Any]], fmt: Optional[Dict[str, Any]],
              extra: Optional[Dict[str, Any]]) -> Any:
    kwargs: Dict[str, Any] = dict(model=model, messages=[{"role": "user", "content": content}], temperature=0)
    if fmt:
        kwargs["response_format"] = fmt
    body = dict(extra or {})
    body["usage"] = {"include": True}
    kwargs["extra_body"] = body
    last: Optional[Exception] = None
    for attempt in range(5):
        try:
            return c.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + 4 * attempt)
    raise last  # type: ignore[misc]


def _cost(resp: Any, model: str) -> float:
    u = getattr(resp, "usage", None)
    cost = getattr(u, "cost", None) if u is not None else None
    if cost is None and u is not None:
        cost = (getattr(u, "model_extra", None) or {}).get("cost")
    if cost is not None:
        return float(cost)
    use = inf.usage_of(resp)
    return float(providers.cost_usd(model, use["prompt_tokens"], use["completion_tokens"]) or 0.0)


def ask_vlm(c: Any, model: str, prompt: str, fmt: Dict[str, Any], frames: Sequence[np.ndarray]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for f in frames:
        data = mi.encode_jpeg(f)
        if data:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}})
    _, _, extra = providers.resolve("openrouter", {"OPENROUTER_API_KEY": "x"}, model)
    t0 = time.time()
    try:
        resp = _complete(c, model, content, fmt, extra)
    except Exception as exc:  # noqa: BLE001
        if "response_format" not in str(exc):
            return {"error": str(exc)[:300], "raw": "", "parsed": None, "cost": 0.0}
        resp = _complete(c, model, content, {"type": "json_object"}, extra)
    raw = resp.choices[0].message.content or ""
    return {"raw": raw, "parsed": inf.parse_vlm_json(raw), "usage": inf.usage_of(resp), "cost": _cost(resp, model),
            "latency_s": round(time.time() - t0, 2), "model_answered": getattr(resp, "model", None), "error": ""}


def ask_judge(c: Any, prompt: str) -> Dict[str, Any]:
    resp = _complete(c, JUDGE_MODEL, [{"type": "text", "text": prompt}], {"type": "json_object"}, None)
    raw = resp.choices[0].message.content or ""
    return {"raw": raw, "parsed": inf.parse_vlm_json(raw), "cost": _cost(resp, JUDGE_MODEL)}


# ----------------------------------------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------------------------------------
def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_LOCK = threading.Lock()


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with _LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answers_path(out: str, model: str, arm: str, run: int) -> str:
    return os.path.join(out, "answers", slug(model), f"{arm}_r{run}.jsonl")


def answers(out: str, model: str, arm: str, run: int) -> Dict[str, Dict[str, Any]]:
    """clip_id -> the last good answer saved for this arm and run."""
    got: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(answers_path(out, model, arm, run)):
        if not row.get("error") and row.get("parsed") is not None:
            got[row["clip_id"]] = row
    return got


# ----------------------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    clips, counts = candidates(args.dataset)
    manifest = []
    kept: Counter = Counter()
    sheets_for: Dict[str, int] = defaultdict(int)
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    for clip in clips:
        r = render(clip, args.dataset)
        if r is None or not qualifies(r):
            continue
        src = clip.row["source"]
        kept[src] += 1
        manifest.append({"clip_id": r.clip_id, "source": src, "batch": clip.row.get("batch"),
                         "camera": camera_of(clip.row), "local_time": local_time_of(clip.meta),
                         "alert": bool(clip.row.get("alert")), "description": clip.row.get("description") or "",
                         "ids": id_order(list(r.tagged.ids.values())), "kinds": r.kinds,
                         "pair": list(r.pair) if r.pair else None, "positions": r.positions,
                         "frames_sent": len(r.sent.frames), "size": list(r.sent.size or ()),
                         "frame_indices": r.sent.frame_indices, "vlm_input": r.sent.vlm_input})
        if sheets_for[src] < (3 if src == "house" else 1) and r.pair:
            sheets_for[src] += 1
            for arm in ARMS:
                sheet = to.contact_sheet(frames_of(arm, r))
                cv2.imwrite(os.path.join(args.out, "sheets", f"{r.clip_id}_{arm}.jpg"), sheet,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    sel = {k: dict(v) for k, v in counts.items()}
    sel["qualify_in_sent_frames"] = dict(kept)
    sel["with_swap_pair"] = dict(Counter(m["source"] for m in manifest if m["pair"]))
    sel["swap_pair_kind"] = dict(Counter(m["pair"][0] for m in manifest if m["pair"]))
    with open(os.path.join(args.out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    print(json.dumps(sel, indent=1))


def _manifest(out: str) -> Dict[str, Dict[str, Any]]:
    return {m["clip_id"]: m for m in read_jsonl(os.path.join(out, "manifest.jsonl"))}


def cmd_run(args: argparse.Namespace) -> None:
    man = _manifest(args.out)
    clips = {c.row["clip_id"]: c for c in candidates(args.dataset)[0] if c.row["clip_id"] in man}
    order = sorted(man)
    if args.limit:
        order = order[:args.limit]
    arms = args.arms.split(",")
    runs = list(range(1, args.runs + 1))
    have = {(a, r): answers(args.out, args.model, a, r) for a in arms for r in runs}
    c = client()
    before = key_usage(c)
    spent = [0.0]

    def work(cid: str) -> int:
        todo = [(a, r) for a in arms for r in runs if cid not in have[(a, r)] and (a != "S" or man[cid]["pair"])]
        if not todo:
            return 0
        rend = render(clips[cid], args.dataset)
        if rend is None:
            return 0
        n = 0
        for arm, run in todo:
            if spent[0] > args.cap:
                return n
            prompt, fmt = prompt_of(arm, clips[cid], man[cid]["ids"])
            res = ask_vlm(c, args.model, prompt, fmt, frames_of(arm, rend))
            spent[0] += res.get("cost") or 0.0
            append_jsonl(answers_path(args.out, args.model, arm, run),
                         {"clip_id": cid, "arm": arm, "run": run, "model": args.model, **res})
            n += 1
        return n

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, cid) for cid in order]
        for i, fut in enumerate(as_completed(futs), 1):
            done += fut.result()
            if i % 20 == 0:
                print(f"{i}/{len(order)} clips, {done} calls, ${spent[0]:.4f}", flush=True)
    after = key_usage(c)
    print(f"calls {done}, summed cost ${spent[0]:.4f}, key usage {before} -> {after}")
    append_jsonl(os.path.join(args.out, "spend.jsonl"), {"step": f"run {args.model} {args.arms} x{args.runs}",
                                                          "calls": done, "summed_cost": spent[0],
                                                          "key_usage_before": before, "key_usage_after": after})


def _pe(parsed: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """id -> 'action; object_or_target' from an answer's per_entity (the first item per id)."""
    out: Dict[str, str] = {}
    for item in (parsed or {}).get("per_entity") or []:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or "").strip().upper()
        if eid and eid not in out:
            obj = str(item.get("object_or_target") or "").strip()
            out[eid] = str(item.get("action") or "").strip() + (f" (object/target: {obj})" if obj else "")
    return out


_TAG_RE = re.compile(r"\b(P|CAR)\d+\b", re.I)


def masked(text: str) -> str:
    """*text* with every tag replaced by a neutral word, so tag strings cannot decide a match."""
    return _TAG_RE.sub(lambda m: "a vehicle" if m.group(1).upper() == "CAR" else "another person", text)


def gate(pair: Sequence[str], first: Dict[str, str], second: Dict[str, str]) -> Optional[str]:
    """Why a pair cannot be judged ("missing" or "same_actions"), or None."""
    x, y = pair
    if not all(k in d and d[k] for d in (first, second) for k in (x, y)):
        return "missing"
    a, b = masked(first[x]), masked(first[y])
    if a.strip().lower() == b.strip().lower() or _sim(a, b) >= 0.75:
        return "same_actions"
    return None


def match_prompt(clip_id: str, eid: str, pair: Sequence[str], first: Dict[str, str],
                 second: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """The blind question for *eid*'s action in *second*; and which option number is which id of *first*."""
    x, y = pair
    flip = int(hashlib.sha256(f"{clip_id}|{eid}".encode()).hexdigest(), 16) % 2 == 1
    o1, o2 = (y, x) if flip else (x, y)
    prompt = MATCH_JUDGE.format(o1=masked(first[o1]).replace('"', "'"), o2=masked(first[o2]).replace('"', "'"),
                                q=masked(second[eid]).replace('"', "'"))
    return prompt, {"1": o1, "2": o2}


def reading_verdict(pair: Sequence[str], matched: Dict[str, Optional[str]]) -> str:
    """followed / ignored / indistinct from what each of the second answer's two actions matched in the first."""
    x, y = pair
    if matched.get(x) == y and matched.get(y) == x:
        return "followed"
    if matched.get(x) == x and matched.get(y) == y:
        return "ignored"
    return "indistinct"


def attr_prompt(m: Dict[str, Any], pe: Dict[str, str]) -> str:
    lines = "\n".join(f"  {v}" for v in m["positions"].values())
    said = "\n".join(f"  {k}: {v}" for k, v in pe.items()) or "  (nothing)"
    return ATTR_JUDGE.format(desc=m["description"].replace('"', "'"), positions=lines, per_entity=said)


def harm_prompt(m: Dict[str, Any], parsed: Dict[str, Any]) -> str:
    return HARM_JUDGE.format(desc=m["description"].replace('"', "'"),
                             summary=str(parsed.get("summary") or "").replace('"', "'"))


def match_jobs(cid: str, tag: str, pair: Sequence[str], first: Dict[str, str],
               second: Dict[str, str]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    if gate(pair, first, second):
        return []
    out = []
    for eid in pair:
        prompt, options = match_prompt(cid, eid, pair, first, second)
        out.append(("match", f"{cid}|{tag}|{eid}", prompt, {"clip_id": cid, "options": options}))
    return out


def cmd_judge(args: argparse.Namespace) -> None:
    man = _manifest(args.out)
    runs = list(range(1, args.runs + 1))
    jdir = os.path.join(args.out, "judge", slug(args.model))
    done = {(j["kind"], j["key"]) for j in read_jsonl(os.path.join(jdir, "judgements.jsonl"))}
    ans = {(a, r): answers(args.out, args.model, a, r) for a in ARMS for r in runs}
    jobs: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for cid, m in man.items():
        for r in runs:
            b = ans[("B", r)].get(cid)
            s = ans[("S", r)].get(cid)
            if m["pair"] and b and s:
                jobs += match_jobs(cid, f"B{r}-S{r}", m["pair"], _pe(b["parsed"]), _pe(s["parsed"]))
            if b:
                jobs.append(("attr", f"{cid}|B{r}", attr_prompt(m, _pe(b["parsed"])), {"clip_id": cid, "run": r}))
            for arm in ("A", "B", "M"):
                a = ans[(arm, r)].get(cid)
                if a:
                    jobs.append(("harm", f"{cid}|{arm}{r}", harm_prompt(m, a["parsed"]),
                                 {"clip_id": cid, "arm": arm, "run": r}))
        b1, b2 = ans[("B", 1)].get(cid), ans[("B", 2)].get(cid) if 2 in runs else None
        if m["pair"] and b1 and b2:   # the noise floor: two plain runs, nothing swapped (expect "ignored")
            jobs += match_jobs(cid, "B1-B2", m["pair"], _pe(b1["parsed"]), _pe(b2["parsed"]))
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} judgements to ask")
    c = client()
    spent = [0.0]

    def work(job):
        kind, key, prompt, extra = job
        res = ask_judge(c, prompt)
        spent[0] += res["cost"]
        append_jsonl(os.path.join(jdir, "judgements.jsonl"),
                     {"kind": kind, "key": key, **extra, "prompt": prompt, "raw": res["raw"], "parsed": res["parsed"]})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(work, j) for j in jobs]), 1):
            fut.result()
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} ${spent[0]:.4f}", flush=True)
    append_jsonl(os.path.join(args.out, "spend.jsonl"), {"step": f"judge {args.model}", "calls": len(jobs),
                                                          "summed_cost": spent[0]})
    print(f"judge cost ${spent[0]:.4f}")


_STOP = {"a", "an", "the", "and", "of", "to", "in", "on", "at", "is", "while", "with", "near", "object", "target",
          "it", "its", "their", "his", "her", "from", "into", "by", "then"}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _sim(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def lexical_swap(pair: Sequence[str], first: Dict[str, str], second: Dict[str, str], margin: float = 0.1) -> str:
    """followed / ignored / indistinct by word overlap alone (no judge): do the actions move with the tag texts?"""
    x, y = pair
    if not all(k in d for d in (first, second) for k in (x, y)):
        return "indistinct"
    moved = _sim(second[y], first[x]) + _sim(second[x], first[y])
    stayed = _sim(second[x], first[x]) + _sim(second[y], first[y])
    if moved > stayed + margin:
        return "followed"
    if stayed > moved + margin:
        return "ignored"
    return "indistinct"


def _alert(parsed: Optional[Dict[str, Any]]) -> bool:
    return inf.label_of(parsed) in ("suspicious", "escalation")


def score(out: str, model: str, runs: int = 2) -> Dict[str, Any]:
    man = _manifest(out)
    rs = list(range(1, runs + 1))
    ans = {(a, r): answers(out, model, a, r) for a in ARMS for r in rs}
    judg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for j in read_jsonl(os.path.join(out, "judge", slug(model), "judgements.jsonl")):
        judg[(j["kind"], j["key"])] = j
    res: Dict[str, Any] = {"model": model, "clips": len(man),
                           "clips_by_source": dict(Counter(m["source"] for m in man.values())),
                           "answers": {f"{a}{r}": len(ans[(a, r)]) for a in ARMS for r in rs}}

    # 1. label reading (blind match judge; gate first)
    def reading_of(first_arm: Tuple[str, int], second_arm: Tuple[str, int], tag: str) -> Dict[str, Any]:
        v: Counter = Counter()
        lex: Counter = Counter()
        per_kind: Dict[str, Counter] = defaultdict(Counter)
        for cid, m in man.items():
            f, s2 = ans[first_arm].get(cid), ans[second_arm].get(cid)
            if not (m["pair"] and f and s2):
                continue
            first, second = _pe(f["parsed"]), _pe(s2["parsed"])
            why = gate(m["pair"], first, second)
            if why:
                verdict = why
            else:
                matched = {}
                for eid in m["pair"]:
                    j = judg.get(("match", f"{cid}|{tag}|{eid}"))
                    got = str(((j or {}).get("parsed") or {}).get("match") or "")
                    matched[eid] = (j or {}).get("options", {}).get(got)
                verdict = reading_verdict(m["pair"], matched)
                lex[lexical_swap(m["pair"], first, second)] += 1
            v[verdict] += 1
            per_kind[m["pair"][0]][verdict] += 1
        decided = v["followed"] + v["ignored"]
        return {**v, "pairs": sum(v.values()), "decided": decided,
                "rate": round(v["followed"] / decided, 3) if decided else None,
                "lexical_on_judged": dict(lex), "by_pair": {k: dict(c) for k, c in per_kind.items()}}

    reading = {f"run{r}": reading_of(("B", r), ("S", r), f"B{r}-S{r}") for r in rs}
    if len(rs) > 1:
        reading["noise_B1_vs_B2"] = reading_of(("B", 1), ("B", 2), "B1-B2")
    res["label_reading"] = reading

    # 2. attribution
    attr = {}
    for r in rs:
        v: Counter = Counter()
        by_kind: Dict[str, Counter] = defaultdict(Counter)
        for cid, m in man.items():
            b = ans[("B", r)].get(cid)
            if not b:
                continue
            pe = _pe(b["parsed"])
            got = {str(x.get("id", "")).upper(): x.get("verdict")
                   for x in ((judg.get(("attr", f"{cid}|B{r}")) or {}).get("parsed") or {}).get("verdicts") or []
                   if isinstance(x, dict)}
            for eid in m["ids"]:
                verdict = "missing" if eid not in pe else got.get(eid, "unjudged")
                v[verdict] += 1
                by_kind["person" if eid.startswith("P") else "vehicle"][verdict] += 1
            v["extra_ids"] += len([k for k in pe if k not in m["ids"]])
        dec = v["correct"] + v["wrong"]
        attr[f"run{r}"] = {**v, "accuracy": round(v["correct"] / dec, 3) if dec else None,
                           "by_kind": {k: dict(x) for k, x in by_kind.items()}}
    res["attribution"] = attr

    # 3. harm
    harm: Dict[str, Any] = {}
    for arm in ("A", "B", "M"):
        for r in rs:
            got = ans[(arm, r)]
            if not got:
                continue
            tp = fn = fp = tn = 0
            inv = mis = 0
            for cid, m in man.items():
                a = got.get(cid)
                if not a:
                    continue
                said = _alert(a["parsed"])
                if m["alert"]:
                    tp += said
                    fn += not said
                else:
                    fp += said
                    tn += not said
                j = (judg.get(("harm", f"{cid}|{arm}{r}")) or {}).get("parsed") or {}
                inv += len(j.get("invented") or [])
                mis += len(j.get("missed") or [])
            harm[f"{arm}{r}"] = {"answers": len(got), "caught": tp, "missed_alerts": fn, "false_alarms": fp,
                                 "quiet_ok": tn, "invented": inv, "missed_actions": mis,
                                 "people_count": sum(int(_safe_int((got.get(cid) or {}).get("parsed", {}), "people"))
                                                     for cid in man if cid in got)}
    for arm in ("B", "M"):
        if not any(ans[(arm, r)] for r in rs):
            continue
        red, gained = [], []
        for cid, m in man.items():
            if not m["alert"]:
                continue
            a_runs = [ans[("A", r)].get(cid) for r in rs]
            x_runs = [ans[(arm, r)].get(cid) for r in rs]
            a_caught = all(a is not None and _alert(a["parsed"]) for a in a_runs)
            missed = [r for r, x in zip(rs, x_runs) if x is not None and not _alert(x["parsed"])]
            if a_caught and missed:
                red.append({"clip_id": cid, f"{arm}_runs_missed": missed})
            if all(x is not None and _alert(x["parsed"]) for x in x_runs) and \
                    not any(a is not None and _alert(a["parsed"]) for a in a_runs):
                gained.append(cid)
        harm[f"red_flags_both_A_caught_{arm}_missed"] = red
        harm[f"both_{arm}_caught_no_A_caught"] = gained
    harm["human_alerts"] = sum(1 for m in man.values() if m["alert"])
    res["harm"] = harm
    res["parse_failures"] = {f"{a}{r}": sum(1 for row in read_jsonl(answers_path(out, model, a, r))
                                            if row.get("error") or row.get("parsed") is None)
                             for a in ARMS for r in rs}
    return res


def _safe_int(d: Dict[str, Any], k: str) -> int:
    try:
        return int(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0


def cmd_sheets(args: argparse.Namespace) -> None:
    """Contact sheets of every arm for the clips named in --clips (comma-separated), for looking by eye."""
    wanted = set(filter(None, args.clips.split(",")))
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    for clip in candidates(args.dataset)[0]:
        if clip.row["clip_id"] not in wanted:
            continue
        r = render(clip, args.dataset)
        for arm in ARMS:
            if r is not None and (arm != "S" or r.swapped is not None):
                cv2.imwrite(os.path.join(args.out, "sheets", f"{r.clip_id}_{arm}.jpg"),
                            to.contact_sheet(frames_of(arm, r)), [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def cmd_score(args: argparse.Namespace) -> None:
    path = os.path.join(args.out, "scores.json")
    allres = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
    allres[args.model] = score(args.out, args.model, args.runs)
    spend = read_jsonl(os.path.join(args.out, "spend.jsonl"))
    allres["spend_summed_usd"] = round(sum(s.get("summed_cost") or 0 for s in spend), 4)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(allres, f, indent=1)
    print(json.dumps(allres[args.model], indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tag_bench", description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["prepare", "run", "judge", "score", "sheets"])
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--out", default=OUT)
    p.add_argument("--model", default="qwen/qwen3.5-9b")
    p.add_argument("--arms", default="A,B,S")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--clips", default="", help="sheets: comma-separated clip ids")
    p.add_argument("--cap", type=float, default=1.2, help="stop asking past this many dollars in one run")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    {"prepare": cmd_prepare, "run": cmd_run, "judge": cmd_judge, "score": cmd_score, "sheets": cmd_sheets}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
