import json
from dataclasses import asdict, fields
from pathlib import Path
import httpx
import pytest
from home_guard_project.admin.models import *
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.backend import AuthError, ForbiddenError, OfflineError, ServerError, RateLimitError

DATA = Path(__file__).parents[2] / 'home_guard_project/admin/demo_data'
TOKEN = dict(access_token='access-1', refresh_token='refresh-1', expires_in=900,
             staff=dict(id=1, email='staff@example.com', name='Test Staff', role='admin'))


@pytest.mark.parametrize('path', sorted(p for p in DATA.glob('*.json') if p.name not in ('camera_catalog.json','studio_members.json')), ids=lambda p: p.name)
def test_every_fixture_parses_and_keeps_contract_fields(path):
    name = path.name
    model = (FleetResponse if name == 'fleet.json' else list[CustomerOut] if name == 'customers.json'
             else DetectionsOut if name.startswith('detections_') else CustomerOut if name.startswith('customer_') else EventPage if name == 'events.json' else EventDetail)
    model = {'audit.json': AuditPage, 'fleet_activity.json': DensityOut, 'studio_collections.json': list[CollectionOut],
             'studio_filters.json': list[SavedFilter], 'studio_exports.json': list[ExportOut],
             'review_count.json': ReviewCount, 'events_density.json': DensityOut,
             'inbox.json': list[InboxItem]}.get(name,model)
    source = json.loads(path.read_text(encoding='utf-8'))
    parsed = decode(model, source)
    def shape(wire, actual):
        if isinstance(wire, list):
            assert len(wire) == len(actual)
            for a, b in zip(wire, actual):
                shape(a, b)
        elif isinstance(wire, dict):
            assert wire.keys() <= actual.keys()  # additive contract 2c defaults
            for key in wire:
                shape(wire[key], actual[key])
    shape(source, [asdict(v) for v in parsed] if isinstance(parsed, list) else asdict(parsed))


def test_demo_routes_and_integrity():
    backend = DemoBackend()
    fleet = backend.fleet()
    assert {d.verdict for d in fleet.devices} == {'offline', 'critical', 'warning', 'healthy'}
    assert len(backend.customers()) == 3
    for customer in backend.customers():
        assert customer == backend.customer(customer.id)
        assert customer.devices == [d for d in fleet.devices if d.customer_id == customer.id]
    for event in backend.events().items:
        detail = backend.event(event.id)
        assert all(getattr(detail, f.name) == getattr(event, f.name) for f in fields(EventSummary))
    assert backend.events(customer_id=99).items == []
    assert backend.login('', '', '').staff == backend.me()


def test_demo_labeler_redaction():
    backend = DemoBackend(role='labeler')
    for operation in [backend.fleet, backend.customers, lambda: backend.customer(1)]:
        with pytest.raises(ForbiddenError):
            operation()
    event = backend.event(101)
    assert event.customer_name == 'customer-000001'
    assert event.raw_meta == {} and event.dispatch is None
    assert 'Daniel' not in repr(event)


def test_http_login_refresh_once_and_bearer_rotation():
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path == '/v1/auth/login':
            assert json.loads(request.content) == dict(email='s@example.com', password='secret', totp='123456')
            return httpx.Response(200, json=TOKEN)
        if request.url.path == '/v1/auth/refresh':
            assert json.loads(request.content) == {'refresh_token': 'refresh-1'}
            return httpx.Response(200, json=dict(TOKEN, access_token='access-2', refresh_token='refresh-2'))
        if request.headers['Authorization'] == 'Bearer access-1':
            return httpx.Response(401)
        assert request.headers['Authorization'] == 'Bearer access-2'
        return httpx.Response(200, json=TOKEN['staff'])
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(handler))
    backend.login('s@example.com', 'secret', '123456')
    assert backend.me().name == 'Test Staff'
    assert len(calls) == 4
    backend.close()


@pytest.mark.parametrize('refresh_status', [200, 401])
def test_no_infinite_refresh(refresh_status):
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith('/refresh'):
            return httpx.Response(refresh_status, json=dict(TOKEN, access_token='access-2'))
        return httpx.Response(401)
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(handler))
    backend.tokens = decode(TokenPair, TOKEN)
    with pytest.raises(AuthError):
        backend.fleet()
    assert paths.count('/v1/auth/refresh') == 1
    assert len(paths) == (3 if refresh_status == 200 else 2)
    assert backend.tokens is None
    backend.close()


@pytest.mark.parametrize('status,error', [(401, AuthError), (403, ForbiddenError), (429, RateLimitError), (500, ServerError), (502, ServerError), (404, ServerError)])
def test_error_mapping(status, error):
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(lambda r: httpx.Response(status, text='sensitive server traceback')))
    with pytest.raises(error) as caught:
        backend.me()
    assert 'traceback' not in str(caught.value)
    backend.close()


def test_offline_and_malformed_payload():
    def offline(request):
        raise httpx.ConnectError('unreachable', request=request)
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(offline))
    with pytest.raises(OfflineError):
        backend.me()
    backend.close()
    backend = HttpBackend('https://cloud.example', transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'devices': 'bad'})))
    with pytest.raises(ServerError):
        backend.fleet()
    backend.close()


def test_all_get_routes_and_event_query():
    paths = []
    def handler(request):
        path = request.url.path
        paths.append(path)
        if path == '/v1/me':
            data = TOKEN['staff']
        else:
            name = path.removeprefix('/v1/').replace('customers/', 'customer_').replace('events/', 'event_')
            data = json.loads((DATA / (name + '.json')).read_text())
        if path == '/v1/events':
            assert request.url.params['q'] == 'front door'
            assert request.url.params['reviewed'] == 'false'
            assert 'cursor' not in request.url.params
        return httpx.Response(200, json=data)
    backend = HttpBackend('https://cloud.example/', transport=httpx.MockTransport(handler))
    assert backend.me().role == 'admin'
    assert len(backend.fleet().devices) == 4
    assert len(backend.customers()) == 3
    assert backend.customer(1).id == 1
    assert backend.events(q='front door', reviewed=False, cursor=None).items
    assert backend.event(101).id == 101
    assert len(paths) == 6
    backend.close()


@pytest.mark.parametrize('mutation', [{'events_24h': '4'}, {'last_seen_utc': 'not a date'}])
def test_malformed_models_rejected(mutation):
    wire = json.loads((DATA/'fleet.json').read_text())['devices'][0]
    with pytest.raises(ValueError):
        decode(DeviceSummary, dict(wire, **mutation))


def test_tls_context_keeps_certificate_verification_without_debug_keylog():
    import ssl
    from home_guard_project.admin.http_backend import tls_context
    context = tls_context()
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()['x509_ca'] > 0
    assert context.keylog_filename is None


def test_nullable_timestamps_and_customer_defaults():
    wire = json.loads((DATA/'fleet.json').read_text())['devices'][0]
    parsed = decode(DeviceSummary, dict(wire, last_seen_utc=None, newest_clip_utc=None))
    assert parsed.last_seen_utc is None and parsed.newest_clip_utc is None
    customer = decode(CustomerOut, {'id': 9, 'name': 'New household'})
    assert customer.devices == [] and not customer.consent_live
    assert customer.timezone == 'Asia/Jerusalem'
