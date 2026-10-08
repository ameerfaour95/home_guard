"""File-backed fixtures, replaceable without changing the desktop client."""
import json
import base64
import mimetypes
from datetime import datetime, timezone, timedelta
from threading import RLock
from urllib.parse import urlsplit, unquote
from pathlib import Path
from .backend import ForbiddenError, ServerError, ValidationError
from .models import decode, TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail, EventSummary, DetectionsOut, MediaAccess


from .demo_studio import DemoStudio
from .demo_annotations import DemoAnnotations
from .tagging_client import DemoTagging


class DemoBackend(DemoTagging, DemoAnnotations, DemoStudio):
    def __init__(self, data_dir=None, role='admin'):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'demo_data'
        self.role = role
        self._reviews, self._lock = {}, RLock()
        self._annotations, self._annotation_versions, self._publishes = {}, {}, {}
        self._collections, self._members, self._exports = None, {}, None
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
        customer = self._load(f'customer_{int(id)}.json', CustomerOut)
        for key, value in getattr(self, '_customer_changes', {}).get(int(id), {}).items():
            setattr(customer, key, value)
        return customer

    def cameras(self, customer_id=None):
        """The catalog's cameras with their newest event; an entry may say ``"current": false`` (a retired id)."""
        self._identity_access()
        from .models import CameraOut
        catalog = json.loads((self.data_dir/'camera_catalog.json').read_text(encoding='utf-8'))
        events = self._load('events.json', EventPage).items
        devices = {d.customer_id: d for d in self.fleet().devices}
        out = []
        for entry in catalog:
            if customer_id not in (None, entry['customer_id']):
                continue
            times = [e.start_utc for e in events if e.customer_id == entry['customer_id'] and e.camera == entry['camera']]
            device = devices.get(entry['customer_id'])
            out.append(CameraOut(entry['customer_id'], device.device_id if device else '', device.site if device else '',
                                 entry['camera'], entry.get('name') or entry['camera'], bool(entry.get('name')),
                                 entry.get('current', True), max(times) if times else None))
        return sorted(out, key=lambda c: (c.customer_id, not c.current, c.camera))

    def update_customer(self, customer):
        self._identity_access()
        changes = {k: getattr(customer, k) for k in ('name', 'timezone', 'consent_live', 'consent_recordings',
                                                       'consent_training', 'notes')}
        changes['consent_source'] = 'contract' if all(changes[f'consent_{k}'] for k in ('live', 'recordings', 'training')) \
            else 'withdrawn'
        self.__dict__.setdefault('_customer_changes', {})[int(customer.id)] = changes
        return self.customer(customer.id)

    def events(self, **filters):
        if filters.get('filter'):
            saved = next((f for f in self.saved_filters() if f.key == filters['filter']), None)
            if saved:
                filters = {**saved.query, **{k: v for k, v in filters.items() if v is not None}}
        page = self._load('events.json', EventPage)
        items = page.items
        for event in items:
            self._apply_review(event)
            self._redact(event)
        if self.role == 'labeler':
            items = [e for e in items if self._load(f'customer_{self._original_customer(e.id)}.json', CustomerOut).consent_training]
        annotation_filter = filters.get('filter')
        if annotation_filter == 'needs_labeling':
            items = [e for e in items if e.annotation_status not in ('submitted', 'reviewed')]
        elif annotation_filter == 'to_review':
            items = [e for e in items if e.annotation_status == 'submitted' or (e.id in self._annotations and self._annotations[e.id].needs_review)]
        elif annotation_filter == 'rejected':
            items = [e for e in items if e.annotation_status == 'rejected']
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
        total = len(items) if filters.get('with_total') else None
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
        return EventPage(selected, cursor, total, False)

    def _apply_review(self, event):
        with self._lock:
            if event.id in self._annotations:
                event.annotation_status = self._annotations[event.id].status
            for key, value in self._reviews.get(event.id, {}).items():
                setattr(event, key, value)

    def _redact(self, event):
        if self.role == 'labeler':
            import hashlib
            event.customer_name = f'customer-{event.customer_id:06}'
            event.site = event.customer_name
            event.camera = 'cam-'+hashlib.sha256(event.camera.encode()).hexdigest()[:6]
            event.customer_id = 0
            if isinstance(event, EventDetail):
                event.raw_meta = {}
                event.dispatch = None
                for run in event.ai_runs:
                    run.prompt = None
                    run.prompt_version = None
                    run.raw_text_artifact_id = None
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

    def index_problems(self):
        from .models import IndexProblem
        return [IndexProblem('synthetic/meta/recording.meta.json', 'Invalid metadata: missing start time', datetime.now(timezone.utc))]
