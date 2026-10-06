"""Backend calls of the tagging studio (cloud routes /v1/tagging/*), for the HTTP and the demo backend.

The studio's wire objects stay plain JSON dicts: their vocabulary (categories, zones, flags) is
fleet_contract/taxonomy.py, which both sides import, not a desktop model.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import RLock

import httpx

from home_guard_project.cloud.tagstudio import queue as work_queue
from home_guard_project.cloud.tagstudio.config import StudioPaths
from home_guard_project.cloud.tagstudio.fields import TagError, clean_fields, fold
from home_guard_project.cloud.tagstudio.service import StudioError, TagStudio, require_category
from .backend import AuthError, ConflictError, ForbiddenError, ServerError, UnsupportedError, ValidationError
from .models import AnnotationOut, decode


class ConsentError(ForbiddenError):
    def __init__(self, message):
        Exception.__init__(self, message)

    @property
    def message(self):
        return str(self)


class HttpTagging:
    def _tag_json(self, method, path, **kwargs):
        response = self._response(method, path, **kwargs)
        if response.status_code < 300:
            try:
                return response.json()
            except ValueError:
                raise ServerError() from None
        try:
            detail = response.json().get('detail')
        except (ValueError, AttributeError):
            detail = None
        detail = detail if isinstance(detail, str) else None
        if response.status_code == 401:
            raise AuthError()
        if response.status_code == 403:
            raise ConsentError(detail or ForbiddenError.message)
        if response.status_code == 501:
            raise UnsupportedError()
        if response.status_code in (400, 404, 409, 410, 422) and detail:
            raise ValidationError(detail)
        raise ServerError()

    def tagging_state(self):
        return self._tag_json('GET', 'tagging/state')

    def tagging_queue(self, tier='open', origin='', q=''):
        return self._tag_json('GET', 'tagging/queue', params=dict(tier=tier, origin=origin, q=q))

    def tagging_clip(self, key):
        return self._tag_json('GET', 'tagging/clip', params=dict(key=key))

    def tagging_save(self, key, fields):
        return self._tag_json('POST', 'tagging/tag', json=dict(key=key, fields=fields))

    def tagging_media(self, key, kind):
        return self._tag_json('POST', 'tagging/media', json=dict(key=key, kind=kind))

    def tagging_teach(self, key):
        return self._tag_json('POST', 'tagging/teach', json=dict(key=key))

    def tagging_export(self, include_needs_check=False):
        return self._tag_json('POST', 'tagging/export', json=dict(include_needs_check=include_needs_check))

    def tagging_suggest(self, key, refresh=False):
        # the model watches the clip first: up to a minute or two, not the usual 15 seconds
        return self._tag_json('POST', 'tagging/suggest', json=dict(key=key, refresh=refresh),
                              timeout=httpx.Timeout(180, connect=5))

    def clip_boxes(self, key):
        """The Label view's annotation of a dataset clip (not an indexed event)."""
        return decode(AnnotationOut, self._tag_json('GET', 'tagging/boxes', params=dict(key=key)))

    def save_clip_boxes(self, key, annotation):
        return decode(AnnotationOut, self._tag_json('PUT', 'tagging/boxes', json=dict(asdict(annotation), key=key)))


class DemoTagging:
    """The same studio logic over the bundled demo dataset, with tags kept in memory."""

    def _tag_studio(self):
        if getattr(self, '_tagstudio', None) is None:
            root = Path(__file__).parent / 'demo_data' / 'tagging'
            paths = StudioPaths(dataset=root / 'dataset', eval_results=(root / 'eval' / 'results',),
                                exports=Path.home() / 'AppData' / 'Local' / 'HomeGuardAdmin' / 'demo_exports')
            self._tagstudio = _MemoryStudio(paths)
        return self._tagstudio

    def tagging_state(self):
        return self._tag_studio().state(None)

    def tagging_queue(self, tier='open', origin='', q=''):
        return self._tag_studio().queue(None, tier=tier, origin=origin, text=q)

    def tagging_clip(self, key):
        return _demo_call(lambda: self._tag_studio().detail(None, key))

    def tagging_save(self, key, fields):
        staff = self.me()
        return _demo_call(lambda: self._tag_studio().save(None, staff, key, fields, datetime.now(timezone.utc)))

    def tagging_media(self, key, kind):
        studio = self._tag_studio()
        item, _ = _demo_call(lambda: studio._item(None, key))
        path = studio.local_media(item, kind)
        if path is None:
            raise ValidationError(f'No {kind} video for this clip on this computer')
        return dict(url=Path(path).resolve().as_uri(), mime='video/mp4',
                    expires_utc=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())

    def tagging_teach(self, key):
        raise ValidationError('No teacher model is configured in the demo')

    def tagging_suggest(self, key, refresh=False):
        """The demo answers from a canned model (no network): what "Suggest tag" would fill in."""
        studio = self._tag_studio()
        if getattr(studio, 'suggest_client', None) is None:
            studio.suggest_client = DemoSuggestClient()
        return _demo_call(lambda: studio.suggest(None, key, refresh))

    def clip_boxes(self, key):
        studio = self._tag_studio()
        with studio._lock:
            saved = studio._boxes.get(key)
        if saved is not None:
            return deepcopy(saved)
        return decode(AnnotationOut, _demo_call(lambda: studio.clip_boxes(None, key)))

    def save_clip_boxes(self, key, annotation):
        studio, old = self._tag_studio(), self.clip_boxes(key)
        if annotation.base_version != old.version:
            raise ConflictError()
        result = deepcopy(old)
        for name in ('tracks', 'description', 'drop_clip', 'needs_review', 'status'):
            setattr(result, name, deepcopy(getattr(annotation, name)))
        result.version, result.updated_utc, result.author = old.version + 1, datetime.now(timezone.utc), self.me().name
        with studio._lock:
            studio._boxes[key] = result
        return deepcopy(result)

    def tagging_export(self, include_needs_check=False):
        return self._tag_studio().export(None, include_needs_check=include_needs_check)


def _demo_call(operation):
    try:
        return operation()
    except StudioError as e:
        raise ValidationError(str(e)) from None


class _MemoryStudio(TagStudio):
    """TagStudio with its tag log in memory instead of the database (demo and offline tests)."""

    def __init__(self, paths):
        super().__init__(paths)
        self._events, self._lock, self._boxes = [], RLock(), {}

    def clip_boxes(self, session, key):
        from home_guard_project.cloud.tagstudio.boxes import dataset_tracks
        from home_guard_project.cloud.tagstudio.service import _track_dict

        item, _ = self._item(session, key)
        fps = item.fps or 7.0
        frames = round(float(item.duration_sec or 0) * fps) or None
        tracks = [_track_dict(t) for t in dataset_tracks(str(self.paths.dataset), item.clip_id, fps)]
        return dict(event_id=0, version=0, status='new', tracks=tracks, description='', ai_description='',
                    ai_status='none', ai_model=None, ai_prompt_version=None, drop_clip=False, needs_review=False,
                    author=None, updated_utc=None, fps=fps, frame_count=frames, frame_size=None,
                    suggestions_used=bool(tracks))

    def tags(self, session):
        with self._lock:
            return fold(list(self._events))

    def history(self, session, key):
        with self._lock:
            return [dict(at=e['at'], by=e['by'], fields=e['fields']) for e in reversed(self._events) if e['key'] == key][:20]

    def save(self, session, staff, key, fields, now):
        item, items = self._item(session, key)
        try:
            clean = clean_fields(fields)
        except TagError as e:
            raise StudioError(str(e), 422) from None
        require_category(self.tags(session).get(key), clean)
        with self._lock:
            self._events.append(dict(key=key, at=now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), by=staff.name, fields=clean))
        tags = self.tags(session)
        rows = work_queue.build(items.values(), tags)
        next_key = next((i.key for i, a in rows if a.tier != work_queue.DONE and i.key != key), '')
        return dict(tag=tags[key].as_dict(), assessment=work_queue.assess(item, tags[key]).as_dict(), next_key=next_key)


class DemoSuggestClient:
    """An OpenAI-shaped client that answers "Suggest tag" without the network (demo, screenshots, tests)."""

    model = 'demo/qwen3-vl-32b-instruct (offline)'

    def __init__(self, answer=None):
        import json
        self.calls = 0
        self._answer = json.dumps(answer or dict(
            summary='A man in a dark jacket walks up the driveway, looks at the front door and walks back out to the '
                    'street.', category='N1', other_text='', zone='entrance', movement='approaching',
            flags=[], people=1, vehicles=0, vehicle_moving=False, animals=0, visibility='clear',
            appearance=['dark jacket', 'grey trousers'], evidence_frame=3, raw_label='normal'))
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        from types import SimpleNamespace
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._answer))])
