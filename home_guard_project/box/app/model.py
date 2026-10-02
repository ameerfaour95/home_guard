"""Pure state and bounded, credential-safe log presentation."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from .strings import tr


@dataclass
class State:
    site: str = ""
    collecting: bool = False
    mode: str = "data_collection"
    waiting: int = 0
    disk: float = 0
    upload: str = ""
    cameras: list = field(default_factory=list)
    error: bool = False

    @classmethod
    def from_heartbeat(cls, payload, **extra):
        return cls(
            site=str(payload.get("site", "")),
            collecting=bool(payload.get("collector_running")),
            mode=payload.get("mode") or "data_collection",
            waiting=int(payload.get("clips_live", 0))
            + int(payload.get("clips_outbox", 0)),
            disk=float(payload.get("disk_free_gb", 0)),
            **extra,
        )


@dataclass
class Activity:
    text: str
    detail: str
    upload: bool = False

    @property
    def time(self):
        match = re.search(r"\b(\d{2}:\d{2})(?::\d{2})?\b", self.detail)
        return match[1] if match else tr("time_unavailable")


def redact(line):
    line = re.sub(r"(?:rtsp|https?|s3)://[^\s]+", tr("hidden_address"), line)
    return re.sub(
        r"(?i)(password|token|secret|key)\s*[=:]\s*\S+",
        lambda m: m[1] + "=" + tr("hidden"),
        line,
    )[:500]


def parse_activity(line, source=""):
    detail = redact(line)
    match = re.search(r"\[([^\]]+)\] (?:trigger|random) saved:", line)
    if match:
        return Activity(tr("saved", camera=match[1].replace("_", " ")), detail)
    match = re.search(r"Done\. Uploaded: (\d+).*Failed: (\d+)", line)
    if match:
        return Activity(
            tr("upload_error") if int(match[2]) else tr("sent", count=match[1]),
            detail,
            not int(match[2]),
        )
    match = re.search(r"Moved (\d+) finished clip", line)
    if match:
        return Activity(tr("moved", count=match[1]), detail)
    if (
        "Starting data_collection" in line
        or "Starting inference" in line
        or "Starting collector" in line
    ):
        return Activity(tr("restart"), detail)
    if "ERROR" in line:
        return Activity(
            tr("upload_error" if source.startswith("upload-") else "warning"), detail
        )
    if "WARNING" in line:
        return Activity(tr("warning"), detail)
    if "reconnect" in line.lower():
        return Activity(tr("reconnect"), detail)
    return None


def tail(path, limit=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read(limit)
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[1:] if size > limit else lines
    except OSError:
        return []


class ActivityFeed:
    """Cache bounded log tails; untouched logs are not reread on every poll."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}
        self.last_upload = ""
        self.upload_cache = {}

    def _events(self, path):
        try:
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return []
        cached = self.cache.get(path)
        if cached and cached[0] == fingerprint:
            return cached[1]
        events = []
        for line in tail(path):
            event = parse_activity(line, path.name)
            if event is None:
                continue
            stamp = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", line)
            timestamp = stamp.group() if stamp else ""
            if not timestamp:
                date = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
                clock = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", line)
                if date and clock:
                    timestamp = date[1] + " " + clock[1]
            timestamp = timestamp.replace("T", " ")
            events.append((timestamp, event))
        self.cache[path] = (fingerprint, events)
        return events

    def _last_success(self, path):
        try:
            stat = path.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            cached = self.upload_cache.get(path)
            if cached and cached[0] == fingerprint:
                return cached[1]
            previous = cached[1] if cached else ""
            # After the first scan, search only newly appended bytes. Include
            # a small overlap for a log line split across the old file end.
            floor = (
                max(0, cached[0][0] - 4096)
                if cached
                and stat.st_size > cached[0][0]
                and stat.st_ino == cached[0][2]
                else 0
            )
            found = ""
            with path.open("rb") as stream:
                end = stat.st_size
                carry = b""
                while end > floor and not found:
                    start = max(floor, end - 65536)
                    stream.seek(start)
                    data = stream.read(end - start) + carry
                    lines = data.splitlines()
                    if start and lines:
                        carry = lines[0][-4096:]
                        lines = lines[1:]
                    else:
                        carry = b""
                    for raw in reversed(lines):
                        line = raw.decode("utf-8", errors="replace")
                        if "Done. Uploaded:" not in line:
                            continue
                        event = parse_activity(line, path.name)
                        stamp = re.search(
                            r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", line
                        )
                        if event and event.upload and stamp:
                            found = stamp.group().replace("T", " ")
                            break
                    end = start
            result = found or previous
            self.upload_cache[path] = (fingerprint, result)
            return result
        except OSError:
            return ""

    def read(self):
        uploads = sorted(self.directory.glob("upload-*.log"))
        paths = [self.directory / "runner.log"]
        paths.extend(sorted(self.directory.glob("collector-*.log"))[-2:])
        paths.extend(uploads[-2:])
        events = [item for path in paths for item in self._events(path)]
        upload = ""
        # Last upload remains useful even after several offline days.
        for path in reversed(uploads):
            upload = self._last_success(path)
            if upload:
                break
        events.sort(key=lambda item: item[0], reverse=True)
        self.last_upload = max(self.last_upload, upload)
        return [event for _, event in events[:12]], self.last_upload


def read_activity(directory):
    return ActivityFeed(directory).read()
