"""Monitor side of the Admin Center: the customer page layout, camera names, warnings and filters."""
import pytest
from datetime import timedelta
from dataclasses import replace
from PySide6.QtCore import QPoint
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.prefs import Preferences
from home_guard_project.admin.shell import AdminWindow


@pytest.fixture(autouse=True)
def forget_owner_names():
    """Owner names the screens learn are process-wide; no test may leak them into another."""
    from home_guard_project.admin import formatting
    yield
    formatting.KNOWN_NAMES.clear()


def bottom_of(widget, ancestor):
    return widget.mapTo(ancestor, QPoint(0, widget.height())).y()


def top_of(widget, ancestor):
    return widget.mapTo(ancestor, QPoint(0, 0)).y()


@pytest.mark.parametrize('size', [(1366, 768), (1200, 720)])
def test_customer_page_scrolls_and_the_footer_sits_under_the_table(size, widgets, wait, tmp_path):
    window = AdminWindow(DemoBackend(), demo=True, prefs=Preferences(tmp_path/'prefs.json')); widgets.append(window)
    window.resize(*size); window.show()
    shell = window.shell
    shell.open_customer(1)
    page = shell.customer_page; timeline = page.timeline
    wait(lambda: bool(timeline.model.rows) and not timeline.density_runner.busy)
    assert page.tabs.currentWidget() is page.overview  # a customer opens on the Overview
    tabs_page = page.page
    page.tabs.setCurrentWidget(timeline)
    wait(lambda: timeline.table.height() >= page.TABLE_MIN_HEIGHT - 2
         and top_of(timeline.count, tabs_page) >= bottom_of(timeline.table, tabs_page))
    assert timeline.table.height() >= page.TABLE_MIN_HEIGHT - 2
    assert top_of(timeline.count, tabs_page) >= bottom_of(timeline.table, tabs_page)
    assert top_of(timeline.older, tabs_page) >= bottom_of(timeline.table, tabs_page)
    assert top_of(timeline.table, tabs_page) >= bottom_of(timeline.filter_bar, tabs_page)
    assert timeline.density_scroll.height() <= 24+22*timeline.density_rows+4
    # the footer is reachable: the page scrolls rather than squeezing it into the table
    bar = page.scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    footer = timeline.older.mapTo(page.scroll.viewport(), QPoint(0, timeline.older.height())).y()
    assert footer <= page.scroll.viewport().height()
    assert window.height() == size[1]
    # nothing is cut off at the right: the page is never wider than the window leaves it
    assert page.page.minimumSizeHint().width() <= page.scroll.viewport().width()


def test_camera_names_follow_the_box_rule_never_the_raw_id():
    from home_guard_project.admin import formatting
    from home_guard_project.admin.formatting import camera_name, remember_names
    assert camera_name('ameer_tes2_ch6') == 'Camera 6'
    assert camera_name('ameer_week_0_1/ameer_week_0_1_ch3') == 'Ameer week 0 1 / Camera 3'
    assert camera_name('front_door') == 'Front door' and camera_name('cam-abcdef') == 'cam-abcdef'
    try:
        remember_names({'ameer_week_0_1_ch1': 'כניסה ראשית', 'ameer_week_0_1_ch2': ''})
        assert camera_name('ameer_week_0_1_ch1') == '\u2068כניסה ראשית\u2069'  # isolated: never flips a line
        assert camera_name('ameer_week_0_1_ch2') == 'Camera 2'
        # exact ids only: another house's (or an old site's) ch1 never borrows this name
        assert camera_name('other_house_ch1') == 'Camera 1'
    finally:
        formatting.KNOWN_NAMES.clear()


class RenamedBackend(DemoBackend):
    """Demo customer 1 after a rename: 'Garden' is an old id the box still lists; 'Front door' has an owner name."""
    def cameras(self, customer_id=None):
        from dataclasses import replace
        out = []
        for c in super().cameras(customer_id):
            if c.customer_id == 1 and c.camera == 'Garden':
                c = replace(c, current=False)
            if c.customer_id == 1 and c.camera == 'Front door':
                c = replace(c, name='כניסה ראשית', owner_named=True)
            out.append(c)
        return out


def test_customer_page_lists_current_cameras_and_hides_retired_behind_a_toggle(widgets, wait):
    from home_guard_project.admin import formatting
    from home_guard_project.admin.customer import CustomerScreen
    screen = CustomerScreen(RenamedBackend()); widgets.append(screen)
    screen.resize(1182, 688); screen.show(); screen.open(1)
    timeline = screen.timeline
    try:
        wait(lambda: screen.camera_list is not None and timeline.density_result is not None)
        screen.tabs.setCurrentWidget(timeline)
        combo = timeline.filters['camera']
        assert list(timeline.density.rows) == ['Driveway', 'Front door']
        assert [combo.itemText(i).strip('\u2068\u2069') for i in range(combo.count())] == ['All cameras', 'Driveway', 'כניסה ראשית']
        assert timeline.retired_toggle.isVisibleTo(screen)
        timeline.retired_toggle.setChecked(True)
        assert list(timeline.density.rows) == ['Driveway', 'Front door', 'Garden']
        assert combo.itemText(combo.count()-1) == 'Garden (old)'
    finally:
        formatting.KNOWN_NAMES.clear()


def test_owner_answer_filter_speaks_the_server_vocabulary(widgets, wait):
    from home_guard_project.cloud.redact import VERDICTS as SERVER
    from home_guard_project.admin.event_logic import VERDICTS
    from home_guard_project.admin.timeline import TimelineScreen
    assert set(VERDICTS) <= SERVER and 'real' not in VERDICTS and VERDICTS['expected'] == 'Normal (expected)'
    queries = []

    class Recording(DemoBackend):
        def events(self, **filters):
            queries.append(filters); return super().events(**filters)
    screen = TimelineScreen(Recording()); widgets.append(screen); screen.show(); screen.open(1, 'Asia/Jerusalem')
    wait(lambda: not screen.runner.busy and bool(queries))
    combo = screen.filters['verdict']
    assert [combo.itemData(i) for i in range(1, combo.count())] == list(VERDICTS)
    combo.setCurrentIndex(combo.findData('expected')); wait(lambda: not screen.runner.busy)
    assert queries[-1]['verdict'] == 'expected'
    combo.setCurrentIndex(combo.findData('true_alert')); wait(lambda: not screen.runner.busy)
    assert queries[-1]['verdict'] == 'true_alert' and 101 in [e.id for e in screen.model.rows]


def test_date_range_picks_whole_days_in_the_house_zone(widgets, wait):
    from datetime import date
    from home_guard_project.admin.timeline import TimelineScreen
    queries = []

    class Recording(DemoBackend):
        def events(self, **filters):
            queries.append(filters); return super().events(**filters)
    screen = TimelineScreen(Recording()); widgets.append(screen); screen.show(); screen.open(1, 'Asia/Jerusalem')
    wait(lambda: not screen.runner.busy and bool(queries))
    assert screen.date_to.date().toPython() == date(2026, 10, 3)
    screen.set_days(date(2026, 9, 20), date(2026, 10, 2)); wait(lambda: not screen.runner.busy)
    assert queries[-1]['from_utc'] == '2026-09-19T21:00:00+00:00' and queries[-1]['to_utc'] == '2026-10-02T21:00:00+00:00'
    assert not any(chip.isChecked() for chip in screen.range_chips.values())
    assert screen.date_from.date().toPython() == date(2026, 9, 20) and screen.date_to.date().toPython() == date(2026, 10, 2)
    count = len(queries)
    screen.set_days(date(2026, 8, 1), date(2026, 10, 2))
    assert screen.banner.isVisibleTo(screen) and '31 days' in screen.banner.text() and len(queries) == count
    screen.range_chips['30 d'].click(); wait(lambda: not screen.runner.busy)
    assert screen.range_chips['30 d'].isChecked() and screen.end - screen.start == timedelta(days=30)


def test_activity_has_a_date_range_and_a_camera_filter(widgets, wait):
    from home_guard_project.admin.review import ReviewScreen
    screen = ReviewScreen(RenamedBackend()); widgets.append(screen); screen.resize(1182, 663); screen.show()
    timeline = screen.timeline
    wait(lambda: timeline.cameras is not None and timeline.density_result is not None)
    assert timeline.range_bar.isVisibleTo(screen) and timeline.date_from.isVisibleTo(screen)
    assert timeline.filters['camera'].isVisibleTo(screen) and timeline.filters['camera'].count() > 1
    assert timeline.retired_toggle.isVisibleTo(screen)
    labeler = ReviewScreen(DemoBackend(role='labeler'), role='labeler'); widgets.append(labeler); labeler.show()
    assert not labeler.timeline.filters['camera'].isVisibleTo(labeler)


def test_overview_says_how_the_house_is_its_cameras_and_retired_ids(widgets, wait):
    from home_guard_project.admin.customer import CustomerScreen
    screen = CustomerScreen(RenamedBackend()); widgets.append(screen)
    screen.resize(1182, 688); screen.show(); screen.open(1)
    wait(lambda: screen.camera_list is not None)
    overview = screen.overview
    assert screen.tabs.currentWidget() is overview
    assert [screen.tabs.tabText(i) for i in range(screen.tabs.count())][:4] == ['Overview', 'Events', 'Event', 'Chat']
    # demo customer 1: Cedar Guest House is offline with 'Last heard 3 h ago'; the next step is appended
    assert overview.status.text() == 'Cedar Guest House: Last heard 3 h ago. Check the box has power and internet.'
    assert any('Last heard 3 h ago  —  Check the box has power and internet.' in w for w in overview.warnings)
    assert [name for name, _ in overview.camera_rows] == ['Driveway', 'כניסה ראשית']
    assert overview.retired_rows and overview.retired_rows[0].startswith('Garden')


def test_overview_status_and_old_site_names():
    from datetime import datetime, timezone
    from home_guard_project.admin.models import CameraOut, DeviceSummary, HealthReason
    from home_guard_project.admin.overview import status_sentence, old_site
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    cam = CameraOut(1, 'd', 'ameer_week_0_1', 'ameer_week_0_1_ch6', 'Camera 6', False, True, now-timedelta(minutes=5))
    device = DeviceSummary('d', 'ameer_week_0_1', 1, 'Ameer', 'healthy', [], now, 'inference', 'box', 1, 0, 300., True,
                           False, now, 3, 1, 0)
    assert status_sentence([device], [cam], now) == 'All 1 cameras reporting normally; newest clip 5 min ago.'
    warned = replace(device, verdict='warning', reasons=[HealthReason('camera_quiet', 'No clip for 30 h from Camera 2 — check it has power and network', 'warning')])
    assert status_sentence([warned], [cam], now) == 'Ameer Week 0 1: No clip for 30 h from Camera 2 — check it has power and network'
    assert status_sentence([], None, now) == 'No box is enrolled for this customer yet.'
    assert old_site('ameer_tes2_ch6') == 'Ameer tes2' and old_site('front_door') == ''


def test_chat_tab_loads_only_when_opened_and_shows_the_conversation(widgets, wait):
    from PySide6.QtWidgets import QLabel
    from home_guard_project.admin.customer import CustomerScreen
    calls = []

    class Counting(DemoBackend):
        def chat(self, customer_id, day=None, q=None):
            calls.append((customer_id, day, q)); return super().chat(customer_id, day, q)
    screen = CustomerScreen(Counting()); widgets.append(screen)
    screen.resize(1182, 688); screen.show(); screen.open(1)
    wait(lambda: screen.customer is not None)
    assert calls == []  # opening a customer never reads (or audits) the chat
    screen.tabs.setCurrentWidget(screen.chat)
    chat = screen.chat
    wait(lambda: bool(chat.bubbles) and not chat.image_runner.busy and not chat.pending_images)
    assert calls == [(1, None, None)] and chat.day.currentText() == chat.result.day
    lines = [b.line for b in chat.bubbles]
    assert [line.kind for line in lines] == ['alert', 'button', 'answer', 'alert']
    texts = [w.text() for w in chat.list.findChildren(QLabel)]
    assert 'Pressed: It was expected' in texts and any(t.startswith('Not delivered to the owner') for t in texts)
    assert any(w.pixmap() and not w.pixmap().isNull() for w in chat.list.findChildren(QLabel))  # the alert pictures
    chat.search.setText('gate'); chat.search.returnPressed.emit()
    wait(lambda: not chat.runner.busy and calls[-1] == (1, None, 'gate'))
    wait(lambda: [b.line.who for b in chat.bubbles] == ['owner'])
    assert chat.clear_search.isVisibleTo(chat) and 'match' in chat.status.text()
    screen.tabs.setCurrentIndex(0); screen.tabs.setCurrentWidget(chat)
    assert len(calls) == 2  # coming back to the tab does not read it again


def test_launcher_starts_the_app_with_a_windowless_interpreter(tmp_path):
    """uv's .venv\Scripts\pythonw.exe is a console launcher (a terminal opened behind the app); the launcher runs the
    base interpreter's real pythonw.exe on admin/windowless.py, which must start the app with the venv's packages."""
    import os
    import struct
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    script = (repo / 'home_guard_project' / 'cloud' / 'launch_admin.ps1').read_text(encoding='utf-8')
    assert 'windowless.py' in script and '"-I"' in script and '.venv\Scripts\pythonw.exe"' not in script
    if sys.platform != 'win32':
        pytest.skip('Windows launcher')
    home = Path(sys._base_executable).parent
    data = (home / 'pythonw.exe').read_bytes()
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    assert struct.unpack_from('<H', data, pe + 0x5C)[0] == 2  # IMAGE_SUBSYSTEM_WINDOWS_GUI: no console
    env = {k: v for k, v in os.environ.items() if k not in ('SSLKEYLOGFILE', 'PYTHONSTARTUP', 'PYTHONPATH')}
    env['HG_ADMIN_VENV'] = sys.prefix
    result = subprocess.run([str(home / 'python.exe'), '-I', str(repo / 'home_guard_project' / 'admin' / 'windowless.py'),
                             '--demo', '--smoke-test'], cwd=tmp_path, env=env, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-2000:]


def test_a_camera_the_owner_switched_off_reads_off_by_the_owner(widgets, wait):
    from home_guard_project.admin.customer import CustomerScreen
    from home_guard_project.admin.formatting import camera_name

    class OffBackend(DemoBackend):
        def cameras(self, customer_id=None):
            return [replace(c, current=False, enabled=False, name='Pool') if c.camera == 'Garden' else c
                    for c in super().cameras(customer_id)]
    screen = CustomerScreen(OffBackend()); widgets.append(screen); screen.show(); screen.open(1)
    wait(lambda: screen.camera_list is not None)
    assert ('Pool', 'switched off') in screen.overview.camera_rows
    assert not any(row.startswith('Pool') for row in screen.overview.retired_rows)
    assert camera_name('Garden') == 'Pool (off by the owner)'


class RenamedSiteBackend(DemoBackend):
    """Demo customer 1's 'cedar_guest_house' box is an old site name of 'cedar_house' (one box, renamed)."""
    def _mark(self, device):
        if device.site == 'cedar_guest_house':
            device.replaced_by, device.replaced_by_site = 'dev-cedar', 'cedar_house'
        if device.site == 'cedar_house':
            device.old_sites = ['cedar_guest_house']
        return device

    def fleet(self):
        result = super().fleet(); result.devices = [self._mark(d) for d in result.devices]; return result

    def customer(self, id):
        result = super().customer(id); result.devices = [self._mark(d) for d in result.devices]; return result


def test_fleet_shows_one_row_per_box_with_its_old_site_names(widgets, wait):
    from home_guard_project.admin.fleet import FleetScreen
    from PySide6.QtCore import Qt
    screen = FleetScreen(RenamedSiteBackend()); widgets.append(screen); screen.show()
    wait(lambda: screen.snapshot is not None)
    sites = [d.site for d in screen.model.rows]
    assert 'cedar_guest_house' not in sites and len(sites) == len(screen.snapshot.devices) - 1
    assert screen.summary.text().startswith(f'{len(sites)} boxes')
    row = sites.index('cedar_house')
    assert screen.model.index(row, 0).data().endswith('·  was Cedar Guest House')
    screen.filter(query='guest house')
    assert [d.site for d in screen.model.rows] == ['cedar_house']


def test_customer_page_never_warns_about_an_old_site_name(widgets, wait):
    from home_guard_project.admin.customer import CustomerScreen
    screen = CustomerScreen(RenamedSiteBackend()); widgets.append(screen); screen.show(); screen.open(1)
    wait(lambda: screen.camera_list is not None)
    assert 'Cedar Guest House' not in screen.health.text() and screen.health.text().startswith('Cedar House')
    assert not any('Cedar Guest House' in w for w in screen.overview.warnings)
    assert not screen.overview.status.text().startswith('Cedar Guest House')


class SessionBackend(DemoBackend):
    """Demo 'Front door' clips are one box session; the newest was sent, the rest kept in the event."""
    def events(self, **filters):
        page = super().events(**{k: v for k, v in filters.items() if k != 'would_raise'})
        door = sorted((e for e in page.items if e.camera == 'Front door'), key=lambda e: e.start_utc, reverse=True)
        for i, e in enumerate(door):
            e.session_id = 'door-session'
            e.outcome_code, e.outcome = ('sent', 'Sent') if i == 0 else ('held', 'Kept in the event, not sent (normal)')
            e.would_raise = i == 1
        if filters.get('would_raise'):
            page.items = [e for e in page.items if e.would_raise]
        return page

    def event_sessions(self, session_ids):
        from home_guard_project.admin.models import EventSession
        door = [e for e in DemoBackend.events(self, limit=500).items if e.camera == 'Front door']
        return [EventSession('door-session', 'cedar_house', 'Front door', min(e.start_utc for e in door),
                             max(e.start_utc for e in door), 32, 1)]


def test_clips_of_one_event_are_one_expandable_row(widgets, wait):
    from home_guard_project.admin.timeline import TimelineScreen
    from home_guard_project.admin.timeline_model import HEADERS, MEMBER
    screen = TimelineScreen(SessionBackend()); widgets.append(screen); screen.resize(1120, 600); screen.show()
    screen.open(1, 'Asia/Jerusalem')
    wait(lambda: bool(screen.model.rows) and ('cedar_house', 'door-session') in screen.model.sessions)
    model, table = screen.model, screen.table
    rows = [i for i, e in enumerate(model.rows) if e.session_id]
    lead, members = rows[0], rows[1:]
    assert model.lead(lead) and all(table.isRowHidden(i) for i in members) and not table.isRowHidden(lead)
    text = model.index(lead, 1).data().split('\n')[0]
    assert text.startswith('+ 32 clips  ·  Front door') and '1 message sent' in text
    assert model.index(lead, HEADERS.index('AI decision')).data() == 'Sent'
    # j / k skip the hidden clips of a collapsed event
    table.setCurrentIndex(model.index(lead, 0))
    nxt = screen.next_row(lead, 1)
    assert nxt not in members and not table.isRowHidden(nxt)
    screen.toggle_group(lead)
    assert not any(table.isRowHidden(i) for i in members) and model.index(lead, 1).data().startswith('−')
    assert model.index(members[0], 1).data().startswith(MEMBER)
    header = table.verticalHeader()  # an open event's clips sit right under it, newest first
    assert [header.visualIndex(i) for i in members] == [header.visualIndex(lead) + 1 + n for n in range(len(members))]
    assert screen.next_row(lead, 1) == members[0] and screen.next_row(members[0], -1) == lead
    assert model.index(members[0], HEADERS.index('AI decision')).data() == 'Kept in the event, not sent (normal)'
    screen.group_toggle.setChecked(False)
    assert not any(table.isRowHidden(i) for i in range(len(model.rows)))
    assert all(header.visualIndex(i) == i for i in range(len(model.rows)))  # back to time order
    screen.group_toggle.setChecked(True); screen.toggle_group(lead, False)
    screen.reveal(members[-1])
    assert not table.isRowHidden(members[-1])
    combo = screen.filters['would_raise']
    combo.setCurrentIndex(combo.findData(True)); wait(lambda: not screen.runner.busy)
    assert [e.would_raise for e in model.rows] == [True]
