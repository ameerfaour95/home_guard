"""Session-local implementation of the same Studio protocol as Cloud."""
import json
from dataclasses import replace
from datetime import timedelta
from .models import (SavedFilter, CollectionOut, ExportOut, AuditPage, DensityOut,
                     DensityRow, ReviewCount, CustomerOut)
from .backend import ForbiddenError, ServerError


class DemoStudio:
    def saved_filters(self):
        return self._load('studio_filters.json', list[SavedFilter])

    def collections(self):
        with self._lock:
            if self._collections is None:
                self._collections = self._load('studio_collections.json', list[CollectionOut])
                self._members = json.loads((self.data_dir/'studio_members.json').read_text(encoding='utf-8'))
            return [replace(c) for c in self._collections]

    def create_collection(self, name, description=''):
        with self._lock:
            collections = self.collections()
            result = CollectionOut(name, max((c.id for c in collections), default=0)+1, 0, self.me().name, self.now, description)
            self._collections.append(result)
            self._members[str(result.id)] = []
            return replace(result)

    def demo_collection_events(self, id):
        self.collections()
        return [self.event(eid) for eid in self._members[str(id)]]

    def collection_events(self, id, *, cursor=None, limit=100):
        from .models import EventPage
        events = sorted(self.demo_collection_events(id), key=lambda e: (e.start_utc, e.id), reverse=True)
        start, limit = int(cursor or 0), min(500, max(1, limit))
        return EventPage(events[start:start+limit], str(start+limit) if start+limit < len(events) else None)

    def export_preview(self, **request):
        import hashlib
        from collections import Counter
        from .models import ExportPreview, ExportExclusion
        events = self.demo_collection_events(request['collection_id'])
        consent = self.demo_training_consent(events)
        included, excluded, groups = [], [], {}
        splits = request.get('split', {'train': .8, 'val': .1, 'test': .1})
        counts = Counter({k: 0 for k in splits})
        for event in events:
            reason = ('no_training_consent' if not consent.get(event.id) else
                      'expired' if event.completeness.expired else
                      'video_unavailable' if not event.completeness.video else
                      'no_real_ai' if request.get('formats') == ['vlm_jsonl'] and not request.get('include_fallback_ai') and event.completeness.ai != 'real' else None)
            if reason:
                excluded.append(ExportExclusion(event.id, reason))
                continue
            included.append(event.id)
            group = f'{event.site}|{event.start_utc.date()}'
            if group not in groups:
                fraction = int.from_bytes(hashlib.sha256((request.get('name', '')+group).encode()).digest()[:8], 'big') / 2**64
                edge = 0
                for key, value in splits.items():
                    edge += value
                    if fraction < edge:
                        groups[group] = key
                        break
            counts[groups[group]] += 1
        warnings = ['Fallback and failed AI are omitted only from vlm.jsonl; clips and YOLO labels remain.'] if not request.get('include_fallback_ai') else []
        return ExportPreview(included, excluded, dict(counts), len(groups), warnings)

    def demo_training_consent(self, events):
        # This fixture-only helper never exposes customer identities to labelers.
        customers = self._load('customers.json', list[CustomerOut])
        consent = {c.id: c.consent_training for c in customers}
        return {e.id: consent.get(e.customer_id, False) for e in events}

    def add_collection_items(self, id, event_ids):
        with self._lock:
            self.collections()
            for eid in event_ids:
                self.event(eid)
            members = self._members[str(id)]
            members.extend(eid for eid in event_ids if eid not in members)
            return self._count_collection(id)

    def remove_collection_items(self, id, event_ids):
        with self._lock:
            self.collections()
            self._members[str(id)] = [eid for eid in self._members[str(id)] if eid not in event_ids]
            return self._count_collection(id)

    def _count_collection(self, id):
        collection = next(c for c in self._collections if c.id == id)
        collection.event_count = len(self._members[str(id)])
        return replace(collection)

    def exports(self):
        if self.role == 'support':
            raise ForbiddenError()
        with self._lock:
            if self._exports is None:
                self._exports = self._load('studio_exports.json', list[ExportOut])
            return [replace(e) for e in self._exports]

    def export(self, id):
        return next(e for e in self.exports() if e.id == id)

    def create_export(self, **request):
        from .export_logic import validate_export
        if validate_export(request['name'], request['formats'], request['split']):
            raise ServerError()
        with self._lock:
            exports = self.exports()
            summary = self.export_preview(**request)
            version = max((e.version for e in exports if e.name == request['name']), default=0)+1
            result = ExportOut(max((e.id for e in exports), default=0)+1, request['name'], version,
                               'queued', len(summary.included_ids), '', None, None, self.now, self.me().name)
            self._exports.insert(0, result)
            return replace(result)

    def audit(self, **filters):
        if self.role != 'admin':
            raise ForbiddenError()
        items = self._load('audit.json', AuditPage).items
        for key in ('staff', 'customer_id', 'action'):
            if filters.get(key) is not None:
                items = [e for e in items if getattr(e, key) == filters[key]]
        start = int(filters.get('cursor') or 0)
        return AuditPage(items[start:start+15], str(start+15) if start+15 < len(items) else None)

    def activity(self, hours=24):
        self._identity_access()
        return self._load('fleet_activity.json', DensityOut)

    def review_count(self):
        events = self.events(limit=500).items
        return ReviewCount(sum(not e.reviewed and self.now-timedelta(hours=24) <= e.start_utc < self.now for e in events),
                           sum(e.flagged and not e.reviewed for e in events))

    def density(self, **filters):
        from datetime import datetime
        start, end = [datetime.fromisoformat(filters[k]) for k in ('from_utc', 'to_utc')]
        bucket = filters.get('bucket', 'hour')
        hour = start.replace(minute=0, second=0, microsecond=0)
        if bucket == 'day':
            hour = hour.replace(hour=0)
        step = timedelta(hours=1) if bucket == 'hour' else timedelta(days=1)
        hours = []
        while hour < end:
            hours.append(hour); hour += step
        events = self.events(**{k: v for k, v in filters.items() if k != 'bucket'}, limit=500).items
        catalog = json.loads((self.data_dir/'camera_catalog.json').read_text(encoding='utf-8'))
        cameras = {e.camera for e in events}
        for entry in catalog:
            if filters.get('customer_id') not in (None, entry['customer_id']):
                continue
            camera = entry['camera']
            if self.role == 'labeler':
                import hashlib
                camera = 'cam-'+hashlib.sha256(camera.encode()).hexdigest()[:6]
            if filters.get('camera') in (None, camera):
                cameras.add(camera)
        rows = []
        for camera in sorted(cameras):
            counts = [[0]*len(hours) for _ in range(3)]
            for event in events:
                if event.camera != camera:
                    continue
                i = int((event.start_utc-hours[0])/step)
                if 0 <= i < len(hours):
                    counts[0][i] += 1; counts[1][i] += event.kind == 'alert'; counts[2][i] += 'false_alarm' in event.owner_verdicts
            rows.append(DensityRow(camera, *counts))
        return DensityOut(bucket, hours, events[0].timezone if events else 'Asia/Jerusalem', rows)
