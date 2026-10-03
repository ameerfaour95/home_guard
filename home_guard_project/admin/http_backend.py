"""Synchronous Cloud client; UI callers always dispatch through workers.py."""
from threading import RLock
import os
import ssl
import certifi
import httpx
from .backend import AuthError, ForbiddenError, OfflineError, ServerError, RateLimitError
from .models import TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail, EventSummary, DetectionsOut, MediaAccess, decode


def tls_context():
    """Verified TLS without implicit SSLKEYLOGFILE side effects.

    Some uv Windows runtimes abort in create_default_context's keylog setup:
    https://github.com/astral-sh/python-build-standalone/pull/1132
    PROTOCOL_TLS_CLIENT retains certificate and hostname verification.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=os.environ.get('SSL_CERT_FILE') or certifi.where(),
                                  capath=os.environ.get('SSL_CERT_DIR'))
    return context


class HttpBackend:
    def __init__(self, base_url: str, *, transport=None):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(base_url=self.base_url + '/v1/', transport=transport,
                                   timeout=15.0, follow_redirects=False, verify=tls_context())
        self.tokens = None
        self._auth_lock = RLock()

    def close(self):
        self.client.close()

    def _send(self, method, path, **kwargs):
        try:
            return self.client.request(method, path, **kwargs)
        except httpx.TransportError:
            raise OfflineError() from None

    @staticmethod
    def _parse(response, model):
        if response.status_code >= 300:
            error = {401: AuthError, 403: ForbiddenError, 429: RateLimitError}.get(response.status_code, ServerError)
            raise error()
        try:
            return decode(model, response.json())
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ServerError() from None

    def login(self, email, password, totp):
        with self._auth_lock:
            self.tokens = None
            self.tokens = self._parse(self._send('POST', 'auth/login', json=dict(
                email=email, password=password, totp=totp)), TokenPair)
            return self.tokens

    def _response(self, method, path, **kwargs):
        token = self.tokens.access_token if self.tokens else ''
        response = self._send(method, path, **kwargs, headers={'Authorization': f'Bearer {token}'})
        if response.status_code == 401 and self.tokens:
            # Concurrent requests share one refresh; each original request retries once.
            with self._auth_lock:
                if self.tokens and self.tokens.access_token == token:
                    try:
                        self.tokens = self._parse(self._send('POST', 'auth/refresh', json={
                            'refresh_token': self.tokens.refresh_token}), TokenPair)
                    except AuthError:
                        self.tokens = None
                        raise
                if not self.tokens:
                    raise AuthError()
                token = self.tokens.access_token
            response = self._send(method, path, **kwargs, headers={'Authorization': f'Bearer {token}'})
        if response.status_code == 401:
            self.tokens = None
        return response

    def _request(self, method, path, model, **kwargs):
        return self._parse(self._response(method, path, **kwargs), model)

    def _get(self, path, model, **params):
        return self._request('GET', path, model, params=params)

    def me(self):
        return self._get('me', StaffOut)

    def fleet(self):
        return self._get('fleet', FleetResponse)

    def customers(self):
        return self._get('customers', list[CustomerOut])

    def customer(self, id):
        return self._get(f'customers/{int(id)}', CustomerOut)

    def events(self, **filters):
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
            if response.status_code != 200:
                raise ServerError()
            return response.content
        except httpx.TransportError:
            raise OfflineError() from None
