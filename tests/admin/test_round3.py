import json
import threading
from dataclasses import asdict, replace
from datetime import timedelta
import httpx
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtMultimedia import QMediaPlayer
from home_guard_project.admin.backend import ServerError, ForbiddenError
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.models import ArtifactOut, EventSummary, decode
from home_guard_project.admin.review import ReviewScreen
from home_guard_project.admin.shell import Shell
from home_guard_project.admin.export_wizard import ExportWizard, validate_export
from home_guard_project.admin.collections import CollectionPicker, CollectionGrid
from home_guard_project.admin.audit import AuditScreen, action_words
from home_guard_project.admin.studio import StudioScreen
from home_guard_project.admin.timeline_model import TimelineModel
from home_guard_project.admin.widgets.palette import CommandPalette
from home_guard_project.admin.widgets.activity import DensityStrip, bar_metrics, activity_tooltip


def make_review(widgets,wait,backend=None):
    backend = backend or DemoBackend()
    screen = ReviewScreen(backend,role=backend.role); widgets.append(screen); screen.resize(1500,850); screen.show()
    wait(lambda:bool(screen.timeline.model.rows) and screen.event_view.recording is not None and not screen.metadata.busy)
    return screen


def test_review_keyboard_autoplay_and_success(widgets,wait):
    screen = make_review(widgets,wait)
    table = screen.timeline.table; table.setFocus()
    first = screen.active_id
    QTest.keyClick(table,Qt.Key.Key_J)
    wait(lambda:screen.event_view.recording is not None and screen.event_view.recording.id != first)
    assert screen.timeline.table.currentIndex().row() == 1
    QTest.keyClick(table,Qt.Key.Key_K)
    wait(lambda:screen.event_view.recording is not None and screen.event_view.recording.id == first and not screen.event_view.evidence_runner.busy)
    player = screen.event_view.player
    wait(lambda:player.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)
    assert player.audio.isMuted() and player.player.playbackRate() == 1
    QTest.keyClick(table,Qt.Key.Key_R)
    wait(lambda:not screen.mutation.busy)
    assert screen.backend.event(first).reviewed
    assert all(e.id != first for e in screen.timeline.model.rows)
    assert screen.done == 1 and screen.active_id != first
    flagged = screen.active_event(); before = flagged.flagged
    QTest.keyClick(table,Qt.Key.Key_F); wait(lambda:not screen.mutation.busy)
    assert screen.backend.event(flagged.id).flagged is not before
    QTest.keyClick(table,Qt.Key.Key_C); wait(lambda:not screen.picker.runner.busy)
    assert screen.picker.event_id == screen.active_id
    screen.picker.add(); wait(lambda:not screen.picker.writer.busy)
    assert screen.last_collection == 1


def test_optimistic_rollback_keeps_next_selection(widgets,wait):
    gate = threading.Event()
    class Failing(DemoBackend):
        def review(self,id,**changes):
            gate.wait(3); raise ServerError()
    screen = make_review(widgets,wait,Failing()); before = replace(screen.active_event())
    try:
        screen.mutate('reviewed')
        assert next(e for e in screen.timeline.model.rows if e.id == before.id).reviewed
        assert screen.active_id != before.id
    finally: gate.set()
    wait(lambda:not screen.mutation.busy)
    restored = next(e for e in screen.timeline.model.rows if e.id == before.id)
    assert restored.reviewed == before.reviewed and screen.done == 0
    assert str(before.id) in screen.toast.text() and 'not saved' in screen.toast.text()


def test_review_search_does_not_trigger_shortcuts(widgets,wait):
    screen = make_review(widgets,wait); screen.timeline.search.setFocus()
    QTest.keyClicks(screen.timeline.search,'jkrfc')
    assert screen.timeline.search.text() == 'jkrfc'
    assert not screen.mutation.busy and not hasattr(screen,'picker')


def test_event_jump_mutates_displayed_event_not_old_row(widgets,wait):
    screen = make_review(widgets,wait)
    screen.open_event(196)
    wait(lambda:screen.event_view.recording is not None and screen.event_view.recording.id == 196)
    old = screen.backend.event(196).flagged
    screen.mutate('flagged'); wait(lambda:not screen.mutation.busy)
    assert screen.backend.event(196).flagged != old


def test_review_paging_keyboard_at_end(widgets,wait):
    screen = make_review(widgets,wait); count = len(screen.timeline.model.rows)
    screen.timeline.table.setCurrentIndex(screen.timeline.model.index(count-1,0))
    screen.move(1)
    wait(lambda:len(screen.timeline.model.rows)>count and screen.timeline.table.currentIndex().row() == count)


def test_palette_ranking_commands_filters_camera_and_event(widgets):
    b = DemoBackend(); palette = CommandPalette(None,b.fleet().devices,b.customers(),'admin',b.saved_filters(),[('Front door',1)])
    widgets.append(palette); results = []; palette.execute.connect(lambda *args:results.append(args))
    for query,kind,expected in [('Go to Batches','command','Go to Batches'),('Filter: Owner said false alarm','filter','owner_false_alarm'),
                                ('#1234','event',1234),('Front door','camera',(1,'Front door'))]:
        palette.filter(query); palette.activate(palette.model.index(0))
        assert results[-1] == (kind,expected)
    palette.filter('>'); assert all(e[2].startswith('command:') for e in palette.matches)
    assert any(e[0] == 'Export collection…' for e in palette.matches)


def test_palette_executes_navigation_export_and_signout(widgets,wait):
    b = DemoBackend(); shell = Shell(b,b.me()); widgets.append(shell); shell.show()
    shell.open_palette(); palette = shell.palette_dialog; palette.search.setText('Export collection…')
    palette.activate(palette.model.index(0))
    studio = shell.screens['Studio']; wait(lambda:hasattr(studio,'wizard'))
    assert studio.wizard.isVisible() and shell.pages.currentWidget() is studio
    studio.wizard.reject(); signed = []; shell.signed_out.connect(lambda:signed.append(True))
    shell.open_palette(); shell.palette_dialog.filter('Sign out'); shell.palette_dialog.activate(shell.palette_dialog.model.index(0))
    assert signed == [True]


@pytest.mark.parametrize('slug',['','Upper','contains space','a/b','../name','night.1','é'])
def test_export_invalid_slug(slug):
    assert validate_export(slug,['clips'],dict(train=.8,val=.1,test=.1))


@pytest.mark.parametrize('split',[dict(train=.7,val=.1,test=.1),dict(train=1.1,val=-.1,test=0),dict(train=1),dict(train=float('nan'),val=0,test=0)])
def test_export_invalid_split(split):
    assert validate_export('valid_name-1',['yolo'],split)


def test_export_formats_and_consent_are_enforced():
    assert validate_export('valid',[],dict(train=.8,val=.1,test=.1))
    b = DemoBackend()
    request = dict(collection_id=1, name='entrance_october', formats=['clips'], split=dict(train=.8,val=.1,test=.1), include_fallback_ai=False)
    preview = b.export_preview(**request)
    assert len(preview.included_ids) == 10
    assert {e.event_id for e in preview.excluded if e.reason == 'no_training_consent'} == {106,112}
    vlm = b.export_preview(**dict(request, formats=['vlm_jsonl']))
    assert len(vlm.included_ids) < len(preview.included_ids)
    result = b.create_export(**request)
    assert result.state == 'queued' and result.item_count == 10 and result.version == 4


def test_wizard_validation_balancing_and_submission(widgets,wait):
    b = DemoBackend(); wizard = ExportWizard(b,b.collections()); widgets.append(wizard); wizard.show()
    wizard.advance(); assert wizard.step == 0 and wizard.error.text()
    wizard.name.setText('review_set'); wizard.advance(); assert wizard.step == 1
    for key in ('train','val','test'):
        wizard.sliders[key].setValue(91)
        assert sum(s.value() for s in wizard.sliders.values()) == 100
    assert not wizard.fallback.isChecked()
    wizard.advance(); wait(lambda:wizard.preview is not None)
    assert not wizard.next.isEnabled(); wizard.check.setChecked(True); assert wizard.next.isEnabled()
    exports = []; wizard.exported.connect(exports.append); wizard.advance(); wait(lambda:bool(exports))
    assert exports[0].item_count == 10 and b.exports()[0].name == 'review_set'


def test_live_wizard_cannot_confirm_unknown_consent(widgets,wait):
    b = DemoBackend()
    class MissingPreview:
        def export_preview(self, **request):
            from home_guard_project.admin.backend import UnsupportedError
            raise UnsupportedError()
        def create_export(self,**request): raise AssertionError('Must not export unverified events')
    wizard = ExportWizard(MissingPreview(),b.collections()); widgets.append(wizard)
    wizard.name.setText('sample'); wizard.set_step(2); wait(lambda:not wizard.runner.busy)
    assert 'Not available yet' in wizard.summary.text() and not wizard.next.isEnabled()
    wizard.advance(); assert not wizard.writer.busy


def test_collection_picker_remembers_and_grid_removes(widgets,wait):
    class Calls(DemoBackend):
        calls = []
        def add_collection_items(self,id,event_ids):
            self.calls.append(('POST',id,event_ids)); return super().add_collection_items(id,event_ids)
        def remove_collection_items(self,id,event_ids):
            self.calls.append(('DELETE',id,event_ids)); return super().remove_collection_items(id,event_ids)
    b = Calls(); picker = CollectionPicker(b,140,2); widgets.append(picker); picker.show()
    wait(lambda:not picker.runner.busy); assert picker.choices.currentData() == 2
    picker.add(); wait(lambda:not picker.writer.busy); assert b.calls[-1] == ('POST',2,[140])
    grid = CollectionGrid(b); widgets.append(grid); grid.show(); grid.open(b.collections()[1])
    wait(lambda:len(grid.model.items) == 7)
    row = next(i for i,e in enumerate(grid.model.items) if e.id == 140)
    grid.grid.setCurrentIndex(grid.model.index(row,0)); grid.remove_selected()
    wait(lambda:not grid.writer.busy and not grid.runner.busy)
    assert b.calls[-1] == ('DELETE',2,[140]) and len(grid.model.items) == 6


def test_audit_paging_filters_and_drawer(widgets,wait):
    b = DemoBackend(); screen = AuditScreen(b); widgets.append(screen); screen.show()
    wait(lambda:screen.loaded_once); assert len(screen.model.items) == 15
    first = [e.id for e in screen.model.items]; screen.load_more(); wait(lambda:len(screen.model.items) == 30)
    assert [e.id for e in screen.model.items[:15]] == first
    assert len({e.id for e in screen.model.items}) == 30
    screen.table.setCurrentIndex(screen.model.index(0,0)); assert screen.drawer.isVisible()
    assert json.loads(screen.json.toPlainText())['id'] == first[0]
    screen.filters['staff'].setText('Maya Cohen'); screen.filters['action'].setText('recording.access'); screen.reload()
    wait(lambda:not screen.runner.busy)
    assert all(e.staff == 'Maya Cohen' and e.action == 'recording.access' for e in screen.model.items)
    assert action_words('recording.access') == 'Viewed a recording'


def test_density_math_zero_camera_and_timezone(widgets):
    b = DemoBackend(); result = b.density(customer_id=2,from_utc=(b.now-timedelta(hours=24)).isoformat(),to_utc=b.now.isoformat())
    assert next(r for r in result.rows if r.camera == 'Front side').events == [0]*24
    strip = DensityStrip(fleet=True); widgets.append(strip); strip.resize(1000,78); strip.set_density(b.activity())
    assert len(strip.hours) == 24 and len(strip.rows['Fleet activity']) == 24
    assert bar_metrics(18,2,36) == (14,14*2/18)
    assert bar_metrics(0,0,0) == (0,0)
    assert bar_metrics(3,10,6) == (14,14)
    hour = b.now.replace(hour=11)
    assert activity_tooltip(hour,(18,2,1),'Asia/Jerusalem') == '14:00–15:00 · 18 events · 2 alerts · 1 false alarm'
    assert strip.hit(strip.cell_rect(0,0).center())[1] == strip.hours[0]
    assert len({bar_metrics(v[0],v[1],48)[0] for v in strip.rows['Fleet activity']}) > 10


def test_labeler_lands_in_label_and_uses_customer_timezone(widgets,wait):
    b = DemoBackend(role='labeler'); shell = Shell(b,b.me()); widgets.append(shell); shell.show()
    assert shell.pages.currentWidget() is shell.screens['Label'] and 'Audit' not in shell.navigation
    shell.navigate('Review'); wait(lambda:shell.review_page.event_view.recording is not None)
    e = shell.review_page.event_view.recording
    assert e.timezone == 'Asia/Jerusalem' and e.customer_name.startswith('customer-')
    assert shell.review_page.event_view.zone == e.timezone
    model = TimelineModel(lambda:b.now); model.set_page([b.events(limit=1).items[0]])
    assert '14:55' in model.data(model.index(0,1))
    assert shell.review_page.event_view.player.zone == e.timezone


@pytest.mark.parametrize('method,path,fixture,args,verb,body',[
    ('density','/v1/events/density','events_density.json',{'from_utc':'2026-10-02T12:00:00Z','to_utc':'2026-10-03T12:00:00Z'},'GET',None),
    ('activity','/v1/fleet/activity','fleet_activity.json',{'hours':24},'GET',None),
    ('review_count','/v1/events/review-count','review_count.json',{},'GET',None),
    ('saved_filters','/v1/studio/filters','studio_filters.json',{},'GET',None),
    ('collections','/v1/studio/collections','studio_collections.json',{},'GET',None),
    ('exports','/v1/studio/exports','studio_exports.json',{},'GET',None),
    ('audit','/v1/audit','audit.json',{'cursor':'opaque','staff':'Maya Cohen'},'GET',None),
    ('create_collection','/v1/studio/collections','studio_collections.json',{'name':'Set','description':'Review'},'POST',{'name':'Set','description':'Review'}),
    ('add_collection_items','/v1/studio/collections/1/items','studio_collections.json',{'id':1,'event_ids':[101]},'POST',{'event_ids':[101]}),
    ('remove_collection_items','/v1/studio/collections/1/items','studio_collections.json',{'id':1,'event_ids':[101]},'DELETE',{'event_ids':[101]}),
    ('export','/v1/studio/exports/1','studio_exports.json',{'id':1},'GET',None),
    ('create_export','/v1/studio/exports','studio_exports.json',{'collection_id':1,'name':'sample','formats':['yolo'],'split':{'train':.8,'val':.1,'test':.1},'include_fallback_ai':False},'POST',{'collection_id':1,'name':'sample','formats':['yolo'],'split':{'train':.8,'val':.1,'test':.1},'include_fallback_ai':False}),
])
def test_round3_http_contract(method,path,fixture,args,verb,body):
    payload = json.loads((DemoBackend().data_dir/fixture).read_text(encoding='utf-8'))
    if method in ('create_collection','add_collection_items','remove_collection_items','export','create_export'): payload = payload[0]
    def handle(request):
        assert request.url.path == path and request.method == verb
        if body is not None: assert json.loads(request.content) == body
        if verb == 'GET':
            for key,value in args.items():
                if key != 'id': assert request.url.params[key] == str(value)
        return httpx.Response(200,json=payload)
    b = HttpBackend('https://cloud.example',transport=httpx.MockTransport(handle))
    assert getattr(b,method)(**args) is not None
    b.close()


def test_support_cannot_export_and_audit_is_admin_only(widgets,wait):
    b = DemoBackend(role='support'); studio = StudioScreen(b,'support'); widgets.append(studio); studio.show()
    wait(lambda:studio.loaded_once); assert not studio.export_button.isVisible() and not studio.tabs.isTabVisible(2)
    with pytest.raises(ForbiddenError): b.exports()
    with pytest.raises(ForbiddenError): b.audit()


def test_filmstrip_metadata_is_nullable_and_demo_sprite_present():
    assert decode(ArtifactOut,dict(id=1,role='clip',s3_key='a',bytes=None,available=True,provenance='captured',detail=None)).detail is None
    b = DemoBackend(); a = next(a for a in b.event(101).artifacts if a.role == 'filmstrip')
    assert a.detail == dict(fps=1,tile_w=160,tile_h=90,count=6)
    assert b.media_bytes(b.artifact_access(a.id).url)


def test_export_history_actions_use_returned_urls(widgets,wait,app,monkeypatch):
    from PySide6.QtGui import QDesktopServices
    b = DemoBackend(); screen = StudioScreen(b); widgets.append(screen); screen.show()
    wait(lambda:screen.loaded_once)
    ready = replace(screen.exports[0],manifest_url='https://media.example/manifest.json?signature=demo')
    screen.export_model.replace([ready]); screen.export_table.setCurrentIndex(screen.export_model.index(0,0))
    assert screen.copy.isEnabled() and screen.manifest.isEnabled()
    screen.copy_path(); assert app.clipboard().text() == ready.s3_prefix
    opened = []
    monkeypatch.setattr(b, 'export', lambda eid: ready)
    monkeypatch.setattr(b, 'media_bytes', lambda url: opened.append(url) or b'{"schema_version":2,"counts":{"clips":{"train":3}},"warnings":["Small dataset"]}')
    screen.open_manifest(); wait(lambda: screen.manifest_data is not None)
    assert opened == [ready.manifest_url]
    assert 'Training: 3' in screen.manifest_detail.text() and 'Small dataset' in screen.manifest_detail.text()


def test_aggregate_ui_does_not_walk_event_cursors(widgets,wait):
    from home_guard_project.admin.fleet import FleetScreen
    class AggregateOnly(DemoBackend):
        def events(self,**filters): raise AssertionError('Fleet must use the aggregate route')
    screen = FleetScreen(AggregateOnly()); widgets.append(screen); screen.show()
    wait(lambda:bool(screen.activity.hours)); assert len(screen.activity.hours) == 24


def test_audit_discards_late_filter_results(widgets,wait):
    gate = threading.Event()
    class Slow(DemoBackend):
        calls = []
        def audit(self,**filters):
            self.calls.append(filters.copy())
            if len(self.calls) == 1: gate.wait(3)
            return super().audit(**filters)
    b = Slow(); screen = AuditScreen(b); widgets.append(screen); screen.show()
    try:
        wait(lambda:len(b.calls) == 1)
        screen.filters['staff'].setText('Dana Shalev'); screen.reload()
    finally: gate.set()
    wait(lambda:screen.loaded_once and not screen.runner.busy)
    assert len(b.calls) == 2 and all(e.staff == 'Dana Shalev' for e in screen.model.items)


def test_studio_and_audit_failures_are_retryable(widgets,wait):
    from home_guard_project.admin.backend import OfflineError
    class Offline(DemoBackend):
        def saved_filters(self): raise OfflineError()
        def audit(self,**filters): raise OfflineError()
    b = Offline(); studio = StudioScreen(b); audit = AuditScreen(b); widgets.extend([studio,audit])
    studio.show(); audit.show()
    wait(lambda:not studio.runner.busy and not audit.runner.busy)
    assert studio.content_stack.currentWidget() is studio.failure and studio.failure.action.isVisible()
    assert audit.stack.currentWidget() is audit.failure and audit.failure.action.isVisible()
    assert "Can't reach" in studio.message.text() and "Can't reach" in audit.message.text()
