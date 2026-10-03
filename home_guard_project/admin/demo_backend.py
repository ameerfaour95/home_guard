"""File-backed fixtures, replaceable without changing the desktop client."""
import json
import base64
import mimetypes
from datetime import datetime, timezone, timedelta
from threading import RLock
from urllib.parse import urlsplit, unquote
from pathlib import Path
from .backend import ForbiddenError, ServerError
from .models import decode, TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail, EventSummary, DetectionsOut, MediaAccess


class DemoBackend:
    def __init__(self, data_dir=None, role='admin'):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'demo_data'
        self.role = role
        self._reviews, self._lock = {}, RLock()
        self.now = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)

    def _load(self, filename, model):
        try:
            return decode(model, json.loads((self.data_dir / filename).read_text(encoding='utf-8')))
        except (OSError, ValueError, TypeError, AttributeError):
            raise ServerError() from None

    def _identity_access(self):
        if self.role == 'labeler':
            raise ForbiddenError()

    def login(self, email, password, totp):
        return TokenPair('demo-access', 'demo-refresh', 900, self.me())

    def me(self):
        return StaffOut(1, 'maya@homeguard.example', 'Maya Cohen', self.role)

    def fleet(self):
        self._identity_access()
        return self._load('fleet.json', FleetResponse)

    def customers(self):
        self._identity_access()
        return self._load('customers.json', list[CustomerOut])

    def customer(self, id):
        self._identity_access()
        return self._load(f'customer_{int(id)}.json', CustomerOut)

    def events(self, **filters):
        page = self._load('events.json', EventPage)
        items = page.items
        for event in items:
            self._apply_review(event)
            self._redact(event)
        for key in ('site', 'customer_id', 'camera', 'kind', 'reviewed', 'flagged'):
            if filters.get(key) is not None:
                items = [e for e in items if getattr(e, key) == filters[key]]
        if filters.get('q'):
            items = [e for e in items if filters['q'].casefold() in e.summary.casefold()]
        if filters.get('ai'):
            items = [e for e in items if e.completeness.ai == filters['ai']]
        if filters.get('verdict'):
            items = [e for e in items if filters['verdict'] in e.owner_verdicts]
        for key, lower in [('from_utc', True), ('to_utc', False)]:
            if filters.get(key):
                boundary = datetime.fromisoformat(str(filters[key]).replace('Z', '+00:00'))
                items = [e for e in items if (e.start_utc >= boundary if lower else e.start_utc < boundary)]
        items.sort(key=lambda e: (e.start_utc, e.id), reverse=True)
        if filters.get('cursor'):
            try:
                stamp, eid = json.loads(base64.urlsafe_b64decode(filters['cursor']))
                key = (datetime.fromisoformat(stamp), eid)
                items = [e for e in items if (e.start_utc, e.id) < key]
            except (ValueError, TypeError):
                raise ServerError() from None
        limit = max(1, min(500, int(filters.get('limit', 100))))
        selected = items[:limit]
        cursor = base64.urlsafe_b64encode(json.dumps([selected[-1].start_utc.isoformat(), selected[-1].id]).encode()).decode() if len(items) > limit else None
        return EventPage(selected, cursor)

    def _apply_review(self, event):
        with self._lock:
            for key, value in self._reviews.get(event.id, {}).items():
                setattr(event, key, value)

    def _redact(self, event):
        if self.role == 'labeler':
            import hashlib
            event.customer_name = f'customer-{event.customer_id:06}'
            event.site = event.customer_name
            event.camera = 'cam-'+hashlib.sha256(event.camera.encode()).hexdigest()[:6]
            if isinstance(event, EventDetail):
                event.raw_meta = {}
                event.dispatch = None
                for feedback in event.feedback:
                    feedback.raw_text = ''
                    feedback.note = ''

    def event(self, id):
        result = self._load(f'event_{int(id)}.json', EventDetail)
        self._apply_review(result)
        self._redact(result)
        return result

    def detections(self, id):
        return self._load(f'detections_{int(id)}.json', DetectionsOut)

    def review(self, id, **changes):
        with self._lock:
            self._reviews.setdefault(int(id), {}).update({k: v for k, v in changes.items() if k in ('reviewed', 'flagged') and type(v) is bool})
        event = self.event(id)
        from dataclasses import fields
        return EventSummary(**{f.name: getattr(event, f.name) for f in fields(EventSummary)})

    def artifact_access(self, id, purpose='review'):
        for path in self.data_dir.glob('event_*.json'):
            event = self._load(path.name, EventDetail)
            artifact = next((a for a in event.artifacts if a.id == id), None)
            if artifact:
                customer = self._load(f'customer_{event.customer_id}.json', CustomerOut)
                if (self.role == 'labeler' and not customer.consent_training) or (purpose == 'support' and not customer.consent_recordings):
                    raise ForbiddenError()
                if not artifact.available:
                    raise ServerError()
                local = (self.data_dir/'media'/Path(artifact.s3_key).name).resolve()
                return MediaAccess(local.as_uri(), datetime.now(timezone.utc)+timedelta(minutes=5), mimetypes.guess_type(local)[0] or 'application/octet-stream')
        raise ServerError()

    def media_bytes(self, url):
        if url.startswith('file:'):
            from urllib.request import url2pathname
            path = Path(url2pathname(unquote(urlsplit(url).path))).resolve()
        else:
            path = (self.data_dir/url).resolve()
        if not path.is_relative_to((self.data_dir/'media').resolve()):
            raise ForbiddenError()
        try:
            return path.read_bytes()
        except OSError:
            raise ServerError() from None
