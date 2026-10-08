"""Synchronous Cloud client; UI callers always dispatch through workers.py."""
from threading import RLock
import os
import ssl
import sys
from urllib.parse import urlsplit
import certifi
import httpx
from .backend import (AuthError, LoginError, ForbiddenError, OfflineError, ServerError,
                      RateLimitError, TlsError, ConfigurationError, UnsupportedError, BackendError)
from .models import ExportPreview, IndexProblem, CameraOut, ChatDay
from .backend import ValidationError
from .backend import ConflictError
from .models import AnnotationOut, AnnotationVersion, PublishOut
from dataclasses import asdict
from .models import SavedFilter, CollectionOut, ExportOut, AuditPage, DensityOut, ReviewCount
from .models import TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail, EventSummary, DetectionsOut, MediaAccess, decode
from .tagging_client import HttpTagging, ConsentError

CONSENT_REFUSALS = ('This customer withdrew consent for training use', 'This customer withdrew consent for recordings access')


def tls_context():
    """Verified TLS without implicit SSLKEYLOGFILE side effects.

    Some uv Windows runtimes abort in create_default_context's keylog setup:
    https://github.com/astral-sh/python-build-standalone/pull/1132
    PROTOCOL_TLS_CLIENT retains certificate and hostname verification.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        context.load_verify_locations(cafile=certifi.where())
        if sys.platform == 'win32':
            context.load_default_certs()
        if os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR'):
            context.load_verify_locations(cafile=os.environ.get('SSL_CERT_FILE'),
                                          capath=os.environ.get('SSL_CERT_DIR'))
    except (OSError, ssl.SSLError):
        raise TlsError() from None
    return context


class HttpBackend(HttpTagging):
    def __init__(self, base_url: str, *, transport=None):
        origin = urlsplit(base_url)
        if (origin.scheme != 'https' and not (origin.scheme == 'http' and
                origin.hostname in ('localhost', '127.0.0.1'))) or not origin.hostname or origin.username or origin.password:
            raise ConfigurationError()
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(base_url=self.base_url + '/v1/', transport=transport,
                                   timeout=httpx.Timeout(15, connect=5), follow_redirects=False, verify=tls_context())
        self.tokens = None
        self._auth_lock = RLock()
        self._epoch = 0

    def clear_session(self):
        with self._auth_lock:
            self._epoch += 1
            self.tokens = None

    def close(self):
        self.client.close()

    def _send(self, method, path, **kwargs):
        try:
            return self.client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            cause = exc
            while cause is not None:
                if isinstance(cause, ssl.SSLCertVerificationError):
                    raise TlsError() from None
                cause = cause.__cause__ or cause.__context__
            raise OfflineError() from None

    @staticmethod
    def _parse(response, model):
        if response.status_code == 409:
            if model is PublishOut:
                raise ValidationError('That batch name cannot be published. Choose a new batch name.')
            raise ConflictError()
        if response.status_code in (400, 422):
            try:
                detail = response.json().get('detail')
            except ValueError:
                detail = None
            safe = {'Name must not identify a household', 'Search is not available for this role'}
            raise ValidationError(detail if isinstance(detail, str) and detail in safe else
                                  'Check the entered values. Names must be 120 characters or fewer.')
        if response.status_code == 403:
            try:
                detail = response.json().get('detail')
            except (ValueError, AttributeError):
                detail = None
            if detail in CONSENT_REFUSALS:   # the household's consent, not the staff member's role, is missing
                raise ConsentError(f"{detail}. If the customer agrees again, an admin switches it back on on the "
                                   "customer's page.")
        if response.status_code >= 300:
            error = {401: AuthError, 403: ForbiddenError, 429: RateLimitError, 501: UnsupportedError}.get(response.status_code, ServerError)
            raise error()
        try:
            return decode(model, response.json())
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ServerError() from None

    def login(self, email, password, totp):
        if os.environ.get('HG_ADMIN_IT') == '1':
            from .logging_setup import setup_logging
            setup_logging().info('Integration sign-in submitted pid=%s', os.getpid())
        with self._auth_lock:
            self._epoch += 1
            self.tokens = None
            try:
                self.tokens = self._parse(self._send('POST', 'auth/login', json=dict(
                    email=email, password=password, totp=totp)), TokenPair)
            except AuthError:
                raise LoginError() from None
            return self.tokens

    def local_login(self):
        """Token-less sign-in for a local service; 404 means it is not allowed."""
        with self._auth_lock:
            self._epoch += 1
            self.tokens = None
            try:
                self.tokens = self._parse(self._send('POST', 'auth/local'), TokenPair)
            except AuthError:
                raise LoginError() from None
            return self.tokens

    def _response(self, method, path, **kwargs):
        with self._auth_lock:
            epoch = self._epoch
            token = self.tokens.access_token if self.tokens else ''
        response = self._send(method, path, **kwargs, headers={'Authorization': f'Bearer {token}'})
        if epoch != self._epoch:
            raise AuthError()
        if response.status_code == 401 and self.tokens:
            # Concurrent requests share one refresh; each original request retries once.
            with self._auth_lock:
                if epoch != self._epoch:
                    raise AuthError()
                if self.tokens and self.tokens.access_token == token:
                    try:
                        self.tokens = self._parse(self._send('POST', 'auth/refresh', json={
                            'refresh_token': self.tokens.refresh_token}), TokenPair)
                    except BackendError:
                        self.tokens = None
                        raise AuthError() from None
                if not self.tokens:
                    raise AuthError()
                token = self.tokens.access_token
            response = self._send(method, path, **kwargs, headers={'Authorization': f'Bearer {token}'})
        if epoch != self._epoch:
            raise AuthError()
        if response.status_code == 401:
            self.tokens = None
        return response

    def _request(self, method, path, model, **kwargs):
        return self._parse(self._response(method, path, **kwargs), model)

    def _get(self, path, model, **params):
        return self._request('GET', path, model, params=params)

    def me(self):
        return self._get('me', StaffOut)

    def annotation(self, event_id):
        return self._get(f'events/{int(event_id)}/annotation', AnnotationOut)

    def save_annotation(self, event_id, annotation):
        return self._request('PUT', f'events/{int(event_id)}/annotation', AnnotationOut, json=asdict(annotation))

    def review_annotation(self, event_id, decision):
        return self._request('POST', f'events/{int(event_id)}/annotation/review', AnnotationOut, json=asdict(decision))

    def annotation_history(self, event_id):
        return self._get(f'events/{int(event_id)}/annotation/history', list[AnnotationVersion])

    def publish_collection(self, collection_id, batch_name):
        return self._request('POST', f'studio/collections/{int(collection_id)}/publish', PublishOut, json={'batch_name': batch_name})

    def publishes(self):
        return self._get('studio/publishes', list[PublishOut])

    def fleet(self):
        return self._get('fleet', FleetResponse)

    def customers(self):
        return self._get('customers', list[CustomerOut])

    def customer(self, id):
        return self._get(f'customers/{int(id)}', CustomerOut)

    def chat(self, customer_id, day=None, q=None):
        return self._get(f'customers/{int(customer_id)}/chat', ChatDay, **{k: v for k, v in dict(day=day, q=q).items() if v})

    def chat_image(self, customer_id, site, image):
        return self._request('POST', f'customers/{int(customer_id)}/chat/images/access', MediaAccess,
                             json={'site': site, 'image': image})

    def cameras(self, customer_id=None):
        return self._get('cameras', list[CameraOut], **({'customer_id': int(customer_id)} if customer_id is not None else {}))

    def events(self, **filters):
        if self.tokens and self.tokens.staff.role == 'labeler':
            filters.pop('customer_id', None); filters.pop('q', None)
        return self._get('events', EventPage, **{k: v for k, v in filters.items() if v is not None})

    def event(self, id):
        return self._get(f'events/{int(id)}', EventDetail)

    def detections(self, id):
        return self._get(f'events/{int(id)}/detections', DetectionsOut)

    def review(self, id, **changes):
        return self._request('PATCH', f'events/{int(id)}/review', EventSummary,
                             json={k: v for k, v in changes.items() if k in ('reviewed', 'flagged')})

    def artifact_access(self, id, purpose='review'):
        return self._request('POST', f'artifacts/{int(id)}/access', MediaAccess, json={'purpose': purpose})

    def media_bytes(self, url):
        # Authenticated thumbnail endpoint may redirect to a presigned URL. Never
        # forward Bearer to the artifact host; media itself uses URL credentials.
        from urllib.parse import urljoin, urlsplit
        absolute = urljoin(self.base_url+'/', url)
        origin, api = urlsplit(absolute), urlsplit(self.base_url)
        is_api = (origin.scheme, origin.netloc) == (api.scheme, api.netloc)
        try:
            response = self._response('GET', absolute) if is_api else self.client.get(absolute)
            if response.status_code == 401:
                raise AuthError()
            for _ in range(4):
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                response = self.client.get(urljoin(str(response.url), response.headers['location']))
            if response.status_code == 501:
                raise UnsupportedError()
            if response.status_code != 200:
                raise ServerError()
            return response.content
        except httpx.TransportError:
            raise OfflineError() from None

    def density(self, **filters):
        if self.tokens and self.tokens.staff.role == 'labeler':
            filters.pop('customer_id', None); filters.pop('q', None)
        return self._get('events/density', DensityOut, **{k: v for k, v in filters.items() if v is not None})

    def review_count(self):
        return self._get('events/review-count', ReviewCount)

    def activity(self, hours=24):
        return self._get('fleet/activity', DensityOut, hours=hours)

    def saved_filters(self):
        return self._get('studio/filters', list[SavedFilter])

    def collections(self):
        return self._get('studio/collections', list[CollectionOut])

    def collection_events(self, id, *, cursor=None, limit=100):
        return self._get(f'studio/collections/{int(id)}/items', EventPage,
                         **dict(limit=limit, **({'cursor': cursor} if cursor else {})))

    def export_preview(self, **request):
        return self._request('POST', 'studio/exports/preview', ExportPreview, json=request)

    def create_collection(self, name, description=''):
        return self._request('POST', 'studio/collections', CollectionOut, json=dict(name=name, description=description))

    def add_collection_items(self, id, event_ids):
        return self._request('POST', f'studio/collections/{int(id)}/items', CollectionOut, json=dict(event_ids=event_ids))

    def remove_collection_items(self, id, event_ids):
        return self._request('DELETE', f'studio/collections/{int(id)}/items', CollectionOut, json=dict(event_ids=event_ids))

    def exports(self):
        return self._get('studio/exports', list[ExportOut])

    def export(self, id):
        return self._get(f'studio/exports/{int(id)}', ExportOut)

    def create_export(self, **request):
        return self._request('POST', 'studio/exports', ExportOut, json=request)

    def audit(self, **filters):
        return self._get('audit', AuditPage, **{k: v for k, v in filters.items() if v is not None})

    def index_problems(self):
        return self._get('index/problems', list[IndexProblem])

    def update_customer(self, customer):
        body = {key: getattr(customer, key) for key in
                ('name', 'timezone', 'consent_live', 'consent_recordings', 'consent_training', 'notes')}
        return self._request('PATCH', f'customers/{int(customer.id)}', CustomerOut, json=body)
