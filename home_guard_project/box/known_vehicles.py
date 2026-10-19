"""The cars the owner named by their plate: "זה הרכב של החשמלאי מנסור, תזכור את המספר שלו" (owner, 2026-10-15/19).

On 2026-10-15 the owner asked the box to remember the electrician's car by its number; the assistant answered "אני
אזכור" but only kept his sentence as a camera fact, and four days later the same car was "suspicious" three times.
This store is what "remember its number" means now: one record per plate, in the owner's words.

``<events dir>/known_vehicles.json`` (beside known.json and camera_profiles.json), a JSON list of::

    {"id", "plate" (digits only, 7 or 8), "label" (his words: "הרכב של החשמלאי מנסור"), "description" ("לבן"),
     "camera" (where it was seen when named), "added_at", "added_by", "source" (the alert id or the media file it was
     read from, or "typed"), "removed_at" (0: live), "replaces" (the id this one corrected)}

- The full plate stays in this file on the box only. Everything written to receipts, logs, the clip's meta or a
  model's context uses :func:`plates.mask` ("****875"); what the owner reads uses :func:`owner_mask` ("••••875":
  Telegram Markdown and the assistant's sanitize eat asterisks, which left a bare "875").
- A plate the owner names again (within one digit, plates.match) is relabelled: the old record is kept as removed and
  the new one points at it, so Undo puts the old name back.
- Writes are atomic (temp file, fsync, replace) under a lock, like events.known.json. A broken file reads as empty and
  is never overwritten silently: it is kept as ``.broken`` first.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional

from . import plates

log = logging.getLogger("box.known_vehicles")

FILE_NAME = "known_vehicles.json"
LABEL_CHARS = 80
DESCRIPTION_CHARS = 40


class KnownVehicles:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    # ---------- files ----------
    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                rows = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            log.warning("known vehicles unreadable (%s); kept as .broken", exc)
            try:
                os.replace(self.path, self.path + ".broken")
            except OSError:
                pass
            return []
        return [r for r in rows if isinstance(r, dict) and plates.valid_plate(str(r.get("plate") or ""))] \
            if isinstance(rows, list) else []

    def _write(self, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ---------- reading ----------
    def all(self, include_removed: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._read()
        return rows if include_removed else [r for r in rows if not r.get("removed_at")]

    def get(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.all(include_removed=True) if r.get("id") == vehicle_id), None)

    def match(self, digits: Any) -> Optional[Dict[str, Any]]:
        """The live record whose plate is *digits* or one digit off (plates.match: ambiguity is no match)."""
        found = plates.match(digits, self.all())
        return dict(found) if found else None

    # ---------- writing ----------
    def add(self, plate: Any, label: str, description: str = "", camera: str = "", by: str = "",
            source: str = "", now: Optional[float] = None) -> Dict[str, Any]:
        """Name the car with this plate. Returns the saved record with ``already`` (the same plate, the same words:
        nothing changed) and ``replaced`` (the record it relabelled, now removed). ValueError on a bad plate or no
        words."""
        digits = plates.digits_only(plate)
        label = " ".join(str(label or "").split())[:LABEL_CHARS]
        if not plates.valid_plate(digits):
            raise ValueError("not a plate (7 or 8 digits)")
        if not label:
            raise ValueError("no name for the car")
        now = time.time() if now is None else float(now)
        with self._lock:
            rows = self._read()
            live = [r for r in rows if not r.get("removed_at")]
            old = plates.match(digits, live)
            if old is not None and old.get("label") == label and old.get("plate") == digits:
                return dict(old, already=True, replaced=None)
            record = {"id": "V" + uuid.uuid4().hex[:8], "plate": digits, "label": label,
                      "description": " ".join(str(description or "").split())[:DESCRIPTION_CHARS],
                      "camera": str(camera or ""), "added_at": now, "added_by": str(by or ""),
                      "source": str(source or ""), "removed_at": 0.0, "replaces": ""}
            replaced = None
            if old is not None:
                for r in rows:
                    if r.get("id") == old.get("id"):
                        r["removed_at"] = now
                        replaced = dict(r)
                record["replaces"] = str(old.get("id") or "")
                if not record["description"]:
                    record["description"] = str(old.get("description") or "")
            rows.append(record)
            self._write(rows)
        return dict(record, already=False, replaced=replaced)

    def remove(self, vehicle_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Forget one car (kept in the file as removed, so Undo can bring it back). None when it is not live."""
        now = time.time() if now is None else float(now)
        with self._lock:
            rows = self._read()
            for r in rows:
                if r.get("id") == vehicle_id and not r.get("removed_at"):
                    r["removed_at"] = now
                    self._write(rows)
                    return dict(r)
        return None

    def restore(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Bring a removed record back, unless another live record now has its plate. None when it cannot."""
        with self._lock:
            rows = self._read()
            target = next((r for r in rows if r.get("id") == vehicle_id), None)
            if target is None or not target.get("removed_at"):
                return None
            live = [r for r in rows if not r.get("removed_at")]
            if plates.match(target.get("plate"), live) is not None:
                return None
            target["removed_at"] = 0.0
            self._write(rows)
            return dict(target)


def owner_mask(plate: Any) -> str:
    """A plate as the owner reads it: "••••875" (a masked "****875" stays masked)."""
    text = str(plate or "")
    return (text if "*" in text else plates.mask(plates.digits_only(text))).replace("*", "•")


COLOURS = frozenset("לבן שחור כסוף אפור אדום כחול ירוק צהוב חום כתום זהב".split())


def car_words(record: Mapping[str, Any]) -> str:
    """The car as a sentence names it: "הרכב הלבן" for a bare colour, "הרכב (טסלה לבנה)" for anything longer
    (2026-10-19: "טסלה לבנה" read "הרכב הטסלה לבנה")."""
    d = " ".join(str(record.get("description") or "").split())
    if not d:
        return "הרכב"
    return f"הרכב ה{d}" if d in COLOURS else f"הרכב ({d})"


def masked_line(record: Mapping[str, Any]) -> str:
    """"הרכב הלבן ••••875 = הרכב של החשמלאי מנסור" - what the owner and the models may read of a record."""
    return f"{car_words(record)} {owner_mask(record.get('plate'))} = {record.get('label') or ''}".strip()


def store_in(directory: str) -> KnownVehicles:
    return KnownVehicles(os.path.join(directory, FILE_NAME))


def for_services(services: Any) -> Optional[KnownVehicles]:
    """The store beside the assistant's event book (``<events>/known_vehicles.json``), or None without one."""
    book = getattr(services, "events", None)
    directory = getattr(book, "directory", "") if book is not None else ""
    return store_in(directory) if directory else None


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m home_guard_project.box.known_vehicles list | add <plate> <words> [--description ..] [--camera ..]
    [--source ..] [--dir <events dir>]``: what the box knows (masked), or one car named at deploy time."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="known_vehicles")
    parser.add_argument("--dir", default="", help="the events folder (default: the box's state folder)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("plate")
    add.add_argument("label")
    add.add_argument("--description", default="")
    add.add_argument("--camera", default="")
    add.add_argument("--source", default="")
    add.add_argument("--by", default="install")
    args = parser.parse_args(argv)
    directory = args.dir
    if not directory:
        from . import paths  # noqa: PLC0415

        directory = os.path.join(paths.state_dir(), "events")
    store = store_in(directory)
    if args.cmd == "add":
        rec = store.add(args.plate, args.label, args.description, args.camera, by=args.by, source=args.source)
        print(("already: " if rec.get("already") else "saved: ") + masked_line(rec) + f" ({rec['id']})")
    for rec in store.all():
        print(f"{rec['id']}  {masked_line(rec)}  camera={rec.get('camera') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
