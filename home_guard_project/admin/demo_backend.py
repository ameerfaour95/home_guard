"""File-backed fixtures, replaceable without changing the desktop client."""
import json
from pathlib import Path
from .backend import ForbiddenError, ServerError
from .models import decode, TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail


class DemoBackend:
    def __init__(self, data_dir=None, role='admin'):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'demo_data'
        self.role = role

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
        for key in ('site', 'customer_id', 'camera', 'kind', 'reviewed', 'flagged'):
            if filters.get(key) is not None:
                items = [e for e in items if getattr(e, key) == filters[key]]
        if filters.get('q'):
            items = [e for e in items if filters['q'].casefold() in e.summary.casefold()]
        for e in items:
            self._redact(e)
        return EventPage(items[:int(filters.get('limit', 100))], page.next_cursor)

    def _redact(self, event):
        if self.role == 'labeler':
            event.customer_name = f'Household {event.customer_id:03}'
            event.site = f'site_{event.customer_id:03}'
            event.camera = 'Camera 01'
            event.summary = 'Activity detected in the camera view.'
            if isinstance(event, EventDetail):
                event.raw_meta = {}
                event.dispatch = None
                event.feedback = []
                event.artifacts = []
                event.ai_runs = []
                event.alert_reason = ''

    def event(self, id):
        result = self._load(f'event_{int(id)}.json', EventDetail)
        self._redact(result)
        return result
