"""A teacher model's suggestion per clip: what a strong model thinks the clip is.

Two implementations of one small interface (``suggest(item) -> Opinion | None``):

- :class:`EvalResultsTeacher` reads the eval results (``<eval>/results/*.jsonl`` + ``.summary.json``, written by
  ``box/eval_prompt.py``) and, per clip, takes the answer of the strongest model that answered it. No model call.
- :class:`OpenAICompatibleTeacher` asks a model over the OpenAI ``chat.completions`` API (planned: a self-hosted
  Qwen3.6-27B on vLLM). It is asked only when a person presses "Ask teacher", keeps every answer in a JSONL
  cache, and refuses Gemini: its terms forbid training a competing model on its outputs.

A teacher's suggestion never reopens a clip we tagged (its ``at`` is empty): we already saw it.
"""
from __future__ import annotations

import base64
import glob
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from ...fleet_contract import taxonomy
from . import evalfmt
from .items import EMPTY, TEACHER, ClipItem, Opinion, now_iso

log = logging.getLogger(__name__)

MAX_ERROR_SHARE = 0.2   # a results file with more failed answers than this is not used
FORBIDDEN_TEACHERS = ("gemini", "generativelanguage.googleapis.com", "aiplatform.googleapis.com")


def parse_raw(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def opinion_from_answer(parsed: Dict[str, Any], model: str, source: str, summary: str = "",
                        label: str = "") -> Optional[Opinion]:
    """A teacher opinion from a model's JSON (today's label-only answers, or the Eye's observation)."""
    label = str(label or parsed.get("raw_label") or parsed.get("label") or "").strip().lower()
    category = str(parsed.get("category") or "").strip()
    category = taxonomy.normalize_id(category) if category else ""
    if label not in taxonomy.LABELS and not category:
        return None
    detail: Dict[str, Any] = {"model": model, "source": source}
    for name in ("other_text", "zone", "movement", "flags", "visibility", "evidence_frame", "why", "people"):
        if parsed.get(name) not in (None, "", []):
            detail[name] = parsed[name]
    if category == "N10" and label in ("", "normal"):
        label = EMPTY
    return Opinion(TEACHER, label=label if label in taxonomy.LABELS + (EMPTY,) else "", category=category,
                   text=str(summary or parsed.get("summary") or ""), detail=detail,
                   knows_empty=bool(category) or label == EMPTY)


class EvalResultsTeacher:
    """Per clip, the answer of the strongest model in the eval results (alerts caught minus normals flagged)."""

    name = "eval"

    def __init__(self, results_dirs, prefer: Sequence[str] = ()):
        dirs = [results_dirs] if isinstance(results_dirs, (str, os.PathLike)) else list(results_dirs)
        self.results_dirs = [str(d) for d in dirs]
        self.prefer = tuple(p for p in prefer if p)
        self._stamp: Optional[tuple] = None
        self._ranking: List[Dict[str, Any]] = []
        self._answers: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _files(self) -> List[str]:
        return sorted(p for d in self.results_dirs for p in glob.glob(os.path.join(glob.escape(d), "*.jsonl"))
                      if not os.path.basename(p).startswith("fake"))

    def _signature(self) -> tuple:
        out = []
        for p in self._files():
            for path in (p, p[:-len(".jsonl")] + ".summary.json"):
                try:
                    out.append((path, os.path.getmtime(path)))
                except OSError:
                    pass
        return tuple(out)

    def _load(self) -> None:
        sig = self._signature()
        if sig == self._stamp:
            return
        ranking, answers = [], {}
        for path in self._files():
            # eval_set and eval_set_v2 may hold a results file of the same name: the folder keeps them apart
            tag = os.path.basename(path)[:-len(".jsonl")]
            if len(self.results_dirs) > 1:
                tag = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{tag}"
            summary: Dict[str, Any] = {}
            try:
                with open(path[:-len(".jsonl")] + ".summary.json", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, ValueError):
                pass
            rows, _ = evalfmt.read_jsonl(path)
            latest: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                if not r.get("error") and str(r.get("ai_label") or "") in taxonomy.LABELS:
                    latest[str(r.get("clip_id") or "")] = r
            model = str(summary.get("model") or (rows[-1].get("model") if rows else "") or tag)
            n = int(summary.get("rows") or len(latest) or 0)
            errors = int(summary.get("errors") or 0)
            if model == "fake" or (n and errors / n > MAX_ERROR_SHARE) or not latest:
                continue
            caught, flagged = summary.get("alerts_caught_ratio"), summary.get("normal_flagged_ratio")
            score = (caught if caught is not None else 0.0) - (flagged if flagged is not None else 1.0)
            preferred = next((i for i, p in enumerate(self.prefer) if p in tag or p in model), len(self.prefer))
            ranking.append({"tag": tag, "model": model, "score": round(score, 4), "alerts_caught_ratio": caught,
                            "normal_flagged_ratio": flagged, "answers": len(latest), "preferred": preferred,
                            "has_summary": bool(summary)})
            answers[tag] = latest
        ranking.sort(key=lambda r: (r["preferred"], not r["has_summary"], -r["score"], r["tag"]))
        self._ranking, self._answers, self._stamp = ranking, answers, sig

    def ranking(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(r) for r in self._ranking]

    def suggest(self, item: ClipItem) -> Optional[Opinion]:
        with self._lock:
            self._load()
            for rank, r in enumerate(self._ranking):
                answer = self._answers[r["tag"]].get(item.clip_id)
                if answer is None:
                    continue
                op = opinion_from_answer(parse_raw(answer.get("raw")), r["model"], f"eval:{r['tag']}",
                                         summary=str(answer.get("ai_summary") or ""),
                                         label=str(answer.get("ai_label") or ""))
                if op is not None:
                    op.detail.update(rank=rank + 1, score=r["score"])
                    return op
            return None

    def status(self) -> Dict[str, Any]:
        ranking = self.ranking()
        return {"name": self.name, "results_dirs": self.results_dirs, "models": len(ranking),
                "best": ranking[0]["model"] if ranking else "", "ranking": ranking[:5]}


def teacher_prompt() -> str:
    """The question for the teacher: the Eye's observation fields, context-free (no time, no house state)."""
    return (
        "You are labelling a short security-camera clip for a training set. The pictures are frames of the clip "
        "in order. Describe only what is visible; do not guess the time of day or who lives there.\n\n"
        "Pick the one category that fits best:\n" + taxonomy.prompt_list() + "\n\n"
        "Answer with JSON only:\n"
        "{\"summary\": \"<what happens, 1-3 short sentences>\", \"category\": \"<id>\", \"other_text\": \"\", "
        f"\"zone\": \"<{' | '.join(taxonomy.ZONES)}>\", \"movement\": \"<{' | '.join(taxonomy.MOVEMENTS)}>\", "
        f"\"flags\": [<any of {', '.join(taxonomy.FLAGS)}>], \"visibility\": \"<clear | partial>\", "
        "\"evidence_frame\": <index of the frame that shows it best>, "
        "\"raw_label\": \"<normal | suspicious | escalation>\"}"
    )


class TeacherRefused(ValueError):
    """The configured teacher may not be used."""


class OpenAICompatibleTeacher:
    """A self-hosted teacher over ``chat.completions``. Asked only through :meth:`ask`; answers are cached."""

    name = "openai"

    def __init__(self, base_url: str, model: str, cache_path: str, api_key: str = "", client: Any = None,
                 frames_for: Optional[Callable[[str], List[Any]]] = None):
        if any(word in f"{base_url} {model}".lower() for word in FORBIDDEN_TEACHERS):
            raise TeacherRefused("Gemini cannot be the teacher: its terms forbid training a competing model on it")
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.cache_path = str(cache_path)
        self._client = client
        self._frames_for = frames_for or evalfmt.sample_frames
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self._cache is None:
            rows, _ = evalfmt.read_jsonl(self.cache_path)
            self._cache = {str(r.get("clip_id")): r for r in rows
                           if r.get("model") == self.model and isinstance(r.get("parsed"), dict)}
        return self._cache

    def suggest(self, item: ClipItem) -> Optional[Opinion]:
        with self._lock:
            row = self._load_cache().get(item.clip_id)
        return opinion_from_answer(row["parsed"], self.model, f"openai:{self.model}") if row else None

    def ask(self, item: ClipItem, video_path: str) -> Optional[Opinion]:
        """Ask the model about *item* (frames from *video_path*) and cache the answer."""
        import cv2  # noqa: PLC0415

        frames = self._frames_for(video_path)
        if not frames:
            raise ValueError("no frames could be read from the clip")
        content: List[Dict[str, Any]] = [{"type": "text", "text": teacher_prompt()}]
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")}})
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(api_key=self.api_key or "none", base_url=self.base_url)
        response = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], temperature=0,
            response_format={"type": "json_object"})
        raw = response.choices[0].message.content or ""
        parsed = parse_raw(raw)
        row = {"clip_id": item.clip_id, "model": self.model, "at": now_iso(), "raw": raw, "parsed": parsed}
        with self._lock:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if parsed:
                self._load_cache()[item.clip_id] = row
        return opinion_from_answer(parsed, self.model, f"openai:{self.model}") if parsed else None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._load_cache())
        return {"name": self.name, "model": self.model, "cached": n}


def first_suggestion(teachers: Sequence[Any], item: ClipItem) -> Optional[Opinion]:
    for t in teachers:
        try:
            op = t.suggest(item)
        except Exception as exc:  # noqa: BLE001 - a broken teacher must not hide the clip
            log.warning("teacher %s failed on %s: %s", t.name, item.clip_id, type(exc).__name__)
            continue
        if op is not None:
            return op
    return None
