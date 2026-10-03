"""Backend contract and safe, user-facing failures."""
from typing import Protocol
from .models import SavedFilter, CollectionOut, ExportOut, ExportPreview, AuditPage, DensityOut, ReviewCount
from .models import TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail, EventSummary, DetectionsOut, MediaAccess


class BackendError(Exception):
    message = 'Something went wrong. Please try again.'
    def __init__(self):
        super().__init__(self.message)


class LoginError(BackendError):
    message = 'Email, password or code is wrong'


class AuthError(BackendError):
    message = 'Your session needs a new sign-in'


class TlsError(BackendError):
    message = 'Certificate not trusted — check antivirus HTTPS scanning and certificate settings.'


class ConfigurationError(BackendError):
    message = 'Use an HTTPS server address. HTTP is allowed only for localhost or 127.0.0.1 development.'


class UnsupportedError(BackendError):
    message = 'Not available yet on this server.'


class UnavailableBackend:
    """Keep the sign-in window usable when client configuration is invalid."""
    def __init__(self, base_url, error):
        self.base_url, self.error = base_url, error

    def login(self, email, password, totp):
        raise self.error


class ForbiddenError(BackendError):
    message = 'Your role does not have access to this page.'


class OfflineError(BackendError):
    message = "Can't reach Home Guard Cloud"


class ServerError(BackendError):
    message = 'Home Guard Cloud could not complete this request. Please try again.'


class InternalError(BackendError):
    message = 'This view could not be loaded. Please try again. Details were saved to the local admin log.'


class RateLimitError(BackendError):
    message = 'Too many requests. Please wait and try again.'


class AdminBackend(Protocol):
    def login(self, email: str, password: str, totp: str) -> TokenPair: ...
    def me(self) -> StaffOut: ...
    def fleet(self) -> FleetResponse: ...
    def customers(self) -> list[CustomerOut]: ...
    def customer(self, id: int) -> CustomerOut: ...
    def events(self, **filters) -> EventPage: ...
    def event(self, id: int) -> EventDetail: ...
    def detections(self, id: int) -> DetectionsOut: ...
    def review(self, id: int, **changes) -> EventSummary: ...
    def artifact_access(self, id: int, purpose: str) -> MediaAccess: ...
    def media_bytes(self, url: str) -> bytes: ...
    def density(self, **filters) -> DensityOut: ...
    def activity(self, hours=24) -> DensityOut: ...
    def review_count(self) -> ReviewCount: ...
    def saved_filters(self) -> list[SavedFilter]: ...
    def collections(self) -> list[CollectionOut]: ...
    def collection_events(self, id, *, cursor=None, limit=100) -> EventPage: ...
    def export_preview(self, **request) -> ExportPreview: ...
    def create_collection(self, name, description='') -> CollectionOut: ...
    def add_collection_items(self, id, event_ids) -> CollectionOut: ...
    def remove_collection_items(self, id, event_ids) -> CollectionOut: ...
    def exports(self) -> list[ExportOut]: ...
    def export(self, id) -> ExportOut: ...
    def create_export(self, **request) -> ExportOut: ...
    def audit(self, **filters) -> AuditPage: ...



def all_events(backend, **filters):
    """Consume every cursor for complete density counts, always called in a worker."""
    items, cursor, seen = [], None, set()
    while True:
        page = backend.events(**filters, cursor=cursor, limit=500)
        items.extend(page.items)
        cursor = page.next_cursor
        if not cursor:
            return items
        if cursor in seen:
            raise ServerError()
        seen.add(cursor)
