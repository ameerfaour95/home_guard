"""Backend contract and safe, user-facing failures."""
from typing import Protocol
from .models import TokenPair, StaffOut, FleetResponse, CustomerOut, EventPage, EventDetail


class BackendError(Exception):
    message = 'Something went wrong. Please try again.'
    def __init__(self):
        super().__init__(self.message)


class AuthError(BackendError):
    message = 'Email, password or code is wrong'


class ForbiddenError(BackendError):
    message = 'Your role does not have access to this page.'


class OfflineError(BackendError):
    message = "Can't reach Home Guard Cloud"


class ServerError(BackendError):
    message = 'Home Guard Cloud could not complete this request. Please try again.'


class RateLimitError(BackendError):
    message = 'Too many tries — wait 15 minutes'


class AdminBackend(Protocol):
    def login(self, email: str, password: str, totp: str) -> TokenPair: ...
    def me(self) -> StaffOut: ...
    def fleet(self) -> FleetResponse: ...
    def customers(self) -> list[CustomerOut]: ...
    def customer(self, id: int) -> CustomerOut: ...
    def events(self, **filters) -> EventPage: ...
    def event(self, id: int) -> EventDetail: ...
