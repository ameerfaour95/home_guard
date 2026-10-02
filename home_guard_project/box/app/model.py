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


def redact(line):
    line = re.sub(r"(?:rtsp|https?|s3)://[^\s]+", tr("hidden_address"), line)
    return re.sub(
        r"(?i)(password|token|secret|key)\s*[=:]\s*\S+",
        lambda m: m[1] + "=" + tr("hidden"),
        line,
    )[:500]


def parse_activity(line):
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
        return Activity(tr("upload_error"), detail)
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
            event = parse_activity(line)
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

    def read(self):
        uploads = sorted(self.directory.glob("upload-*.log"))
        paths = [self.directory / "runner.log"]
        paths.extend(sorted(self.directory.glob("collector-*.log"))[-2:])
        paths.extend(uploads[-2:])
        events = [item for path in paths for item in self._events(path)]
        upload = ""
        # Last upload remains useful even after several offline days.
        for path in reversed(uploads):
            times = [stamp for stamp, event in self._events(path) if event.upload]
            if times:
                upload = max(times)
                break
        events.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in events[:12]], upload


def read_activity(directory):
    return ActivityFeed(directory).read()
