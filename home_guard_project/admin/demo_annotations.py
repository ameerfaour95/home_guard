"""In-memory contract 2f implementation. Publishing never contacts storage."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import re

from home_guard_project.fleet_contract.tracks import tracks_from_weak_labels, validate_tracks
from .backend import ConflictError, ForbiddenError, ValidationError
from .models import (AnnotationOut, AnnotationVersion, Track, PublishOut, PublishMissing, decode, CustomerOut)


class DemoAnnotations:
    def annotation(self, event_id):
        with self._lock:
            if event_id in self._annotations:
                return deepcopy(self._annotations[event_id])
            event = self.event(event_id)
            if self.role == 'labeler' and not self._load(f'customer_{self._original_customer(event_id)}.json', CustomerOut).consent_training:
                raise ForbiddenError()
            frames = self.detections(event_id).frames
            tracks = tracks_from_weak_labels([(f.frame_index, f.t_sec,
                      [(b.label, b.xyxy) for b in f.boxes]) for f in frames if f.status != 'not_run'])
            for n, tr in enumerate(tracks, start=1):  # the YOLO boxes open as editable tracks
                tr.track_id, tr.source = f't-{n}', 'yolo'
            run = next(iter(event.ai_runs), None)
            return AnnotationOut(event_id, 0, 'new', [decode(Track, asdict(t)) for t in tracks],
                event.summary, event.summary, event.completeness.ai,
                run.model if run else None, run.prompt_version if run and self.role == 'admin' else None,
                False, False, None, None, event.fps, round((event.fps or 12)*(event.duration_sec or 6)),
                event.frame_size, False)

    def _original_customer(self, event_id):
        from .models import EventDetail
        return self._load(f'event_{event_id}.json', EventDetail).customer_id

    def save_annotation(self, event_id, annotation):
        if self.role not in ('admin', 'labeler'):
            raise ForbiddenError()
        with self._lock:
            old = self.annotation(event_id)
            if annotation.base_version != old.version:
                raise ConflictError()
            errors = validate_tracks(annotation.tracks, self.event(event_id).duration_sec or 0)
            if errors:
                raise ValidationError('\n'.join(errors))
            result = deepcopy(old)
            for key in ('tracks', 'description', 'drop_clip', 'needs_review', 'status'):
                setattr(result, key, deepcopy(getattr(annotation, key)))
            result.suggestions_used = any(t.source in ('yolo', 'suggestion') for t in old.tracks)
            self._record_annotation(old, result)
            return deepcopy(result)

    def _record_annotation(self, old, result):
        result.version = old.version + 1
        result.updated_utc = datetime.now(timezone.utc)
        result.author = 'labeler-001' if self.role == 'labeler' else self.me().name
        self._annotations[result.event_id] = result
        self._annotation_versions.setdefault(result.event_id, []).insert(0, AnnotationVersion(
            result.version, result.status, result.author, result.updated_utc, len(result.tracks),
            result.description != old.description))

    def review_annotation(self, event_id, decision):
        if self.role != 'admin':
            raise ForbiddenError()
        with self._lock:
            old = self.annotation(event_id)
            if decision.version is not None and decision.version != old.version:
                raise ConflictError()
            if old.status != 'submitted':
                raise ValidationError('Submit this clip before reviewing it.')
            result = deepcopy(old)
            result.status = 'reviewed' if decision.decision == 'accept' else 'rejected'
            result.review_note, result.review_frame = decision.note, decision.frame
            self._record_annotation(old, result)
            return deepcopy(result)

    def annotation_history(self, event_id):
        self.annotation(event_id)
        with self._lock:
            return deepcopy(self._annotation_versions.get(event_id, []))

    def publish_collection(self, collection_id, batch_name):
        if self.role != 'admin':
            raise ForbiddenError()
        if not re.fullmatch(r'[a-z0-9_]+', batch_name):
            raise ValidationError('Use lowercase letters, numbers and underscores.')
        events = self.demo_collection_events(collection_id)
        annotations = [self.annotation(e.id) for e in events]
        labeled = [a for a in annotations if a.status in ('submitted', 'reviewed')]
        missing = [PublishMissing(a.event_id, 'not labeled') for a in annotations if a not in labeled]
        active = [a for a in labeled if not a.drop_clip]
        result = PublishOut(batch_name, f's3://security-camera-project-v1/tagging/{batch_name}/',
            'partial' if missing else 'ready', len(labeled), sum(a.frame_count or 0 for a in active),
            len(active), missing, datetime.now(timezone.utc), self.me().name)
        with self._lock:
            self._publishes[batch_name] = result
        return deepcopy(result)

    def publishes(self):
        if self.role != 'admin':
            raise ForbiddenError()
        with self._lock:
            return deepcopy(list(self._publishes.values()))
