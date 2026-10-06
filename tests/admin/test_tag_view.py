"""The tagging workspace over the demo backend: queue order, the form, keyboard, saving, drafts, consent messages,
and the customer page's consent confirmation."""
import httpx
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton

from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.tagging_client import ConsentError
from home_guard_project.admin.backend import ForbiddenError, ValidationError
from home_guard_project.admin.tag_view import TagView
from home_guard_project.admin.customer import CustomerScreen
from home_guard_project.admin.shell import Shell


def tag_view(widgets, wait, backend=None):
    b = backend or DemoBackend()
    v = TagView(b, 'admin'); widgets.append(v); v.resize(1366, 700); v.show(); v.open()
    wait(lambda: v.detail is not None and not v.clip_runner.busy, 10)
    return v, b


def test_queue_opens_the_first_contradiction(widgets, wait):
    v, _ = tag_view(widgets, wait)
    assert [r['tier_name'] for r in v.model.rows][:3] == ['contradiction'] * 3
    assert v.key == v.model.rows[0]['key']
    assert v.stats['contradiction'].text() == '3' and v.stats['done'].text() == '4'
    assert v.cards['owner'].text.text() == '(no words)' and v.cards['owner'].property('conflict') == 'yes'
    assert len(v.category_buttons) == 28 and v.category_buttons['N3'].category['he']


def test_keyboard_category_raw_label_and_save(widgets, wait):
    v, b = tag_view(widgets, wait)
    first = v.key
    v.setFocus()
    QTest.keyClick(v, Qt.Key.Key_S); QTest.keyClick(v, Qt.Key.Key_3)
    assert v.form['category'] == 'S3' and v.form['raw_label'] == 'suspicious'
    QTest.keyClick(v, Qt.Key.Key_N); QTest.keyClick(v, Qt.Key.Key_0)
    assert v.form['category'] == 'N10' and v.form['raw_label'] == 'normal'
    QTest.keyClick(v, Qt.Key.Key_E, Qt.KeyboardModifier.ShiftModifier)
    assert v.form['raw_label'] == 'escalation' and v.raw_manual
    v.set_category('N3')
    assert v.form['raw_label'] == 'escalation'         # a raw label set by hand stays
    v.chips['zone'].buttons['entrance'].click(); v.chips['flags'].buttons['uniform_or_helmet'].click()
    assert v.form['zone'] == 'entrance' and v.form['flags'] == ['uniform_or_helmet']
    v.description.setPlainText('A courier leaves a parcel.')
    assert v.dirty() and v.save_state.text() == 'Unsaved changes'
    QTest.keyClick(v, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    wait(lambda: v.key != first and not v.queue_runner.busy and not v.clip_runner.busy, 10)
    saved = b.tagging_clip(first)
    assert saved['tag']['fields']['category'] == 'N3' and saved['tag']['fields']['description'] == 'A courier leaves a parcel.'
    assert saved['assessment']['tier_name'] == 'done' and first not in [r['key'] for r in v.model.rows]


def test_save_needs_a_category_or_a_flag(widgets, wait):
    v, b = tag_view(widgets, wait)
    v.form['category'] = ''; v.save()
    assert v.banner.isVisible() and 'category' in v.banner_text.text() and not v.save_runner.busy
    v.needs_check.setChecked(True); v.save()
    wait(lambda: not v.save_runner.busy, 5)


def test_moving_keeps_an_unsaved_draft(widgets, wait):
    v, _ = tag_view(widgets, wait)
    first = v.key
    v.set_category('S2')
    v.move(1); wait(lambda: v.key != first and not v.clip_runner.busy, 5)
    assert v.form['category'] != 'S2'
    v.move(-1); wait(lambda: v.key == first and not v.clip_runner.busy, 5)
    assert v.form['category'] == 'S2' and v.dirty()


def test_teacher_suggestion_and_evidence(widgets, wait):
    v, _ = tag_view(widgets, wait)
    target = next(r['key'] for r in v.model.rows if 'teacher' in r['labels'])
    v.open_key(target); wait(lambda: v.key == target and not v.clip_runner.busy, 5)
    teacher = v.detail['opinions']['teacher']
    v.accept_teacher()
    assert v.form['raw_label'] == teacher['label'] and v.form['description']
    v.player.setPosition(0); v.mark_evidence()
    assert v.form['evidence_sec'] == 0 and v.form['evidence_frame'] == 0 and 'frame 0' in v.evidence.text()


def test_consent_refusal_shows_how_to_fix_it(widgets, wait):
    class Refusing(DemoBackend):
        def tagging_media(self, key, kind):
            raise ConsentError("This customer has not agreed to training use. An admin confirms consent on the customer's page.")
    v, _ = tag_view(widgets, wait, Refusing())
    wait(lambda: not v.media_runner.busy and v.banner.isVisible(), 5)
    assert 'not agreed' in v.canvas.message and 'customer page' in v.banner_text.text()


def test_http_backend_reports_consent_and_validation_details():
    def handler(request):
        if request.url.path.endswith('/tagging/media'):
            return httpx.Response(403, json={'detail': 'This customer has not agreed to training use'})
        if request.url.path.endswith('/tagging/tag'):
            return httpx.Response(422, json={'detail': "category: 'Q1' is not one of N1, ..."})
        if request.url.path.endswith('/customers/7/consent/confirm'):
            return httpx.Response(409, json={'detail': "No consent was recorded at this customer's setup: consent can only come from the customer"})
        if request.url.path.endswith('/artifacts/5/access'):
            return httpx.Response(403, json={'detail': 'This customer has not agreed to recordings access'})
        return httpx.Response(200, json={'items': [], 'count': 0})
    backend = HttpBackend('http://127.0.0.1:8610', transport=httpx.MockTransport(handler))
    with pytest.raises(ConsentError, match='not agreed to training use'):
        backend.tagging_media('ev:1', 'clip')
    with pytest.raises(ValidationError, match='category'):
        backend.tagging_save('ev:1', {'category': 'Q1'})
    with pytest.raises(ForbiddenError, match="customer's page"):
        backend.artifact_access(5)
    with pytest.raises(ValidationError, match='can only come from the customer'):
        backend.confirm_consent(7, '2026-10-02T10:00:00Z')
    assert backend.tagging_queue()['count'] == 0


def test_shell_has_a_tag_screen_and_studio_links_to_it(widgets, wait):
    b = DemoBackend(); shell = Shell(b, b.me()); widgets.append(shell); shell.resize(1366, 768); shell.show()
    shell.screens['Studio'].tagging_requested.emit()
    wait(lambda: shell.pages.currentWidget() is shell.tag_page and shell.tag_page.detail is not None, 10)
    assert shell.navigation['Tag'].isChecked()


def test_consent_can_only_confirm_what_the_box_recorded(widgets, wait):
    b = DemoBackend()
    screen = CustomerScreen(b, 'dark', 'admin'); widgets.append(screen); screen.resize(1366, 768); screen.show()
    screen.open(3); wait(lambda: screen.customer is not None and not screen.runner.busy, 5)
    assert screen.consent_button.isVisible() and screen.consent_button.text() == 'Confirm consent given at setup…'
    assert screen.consent_note.isVisible() and 'nobody has confirmed it yet' in screen.consent_note.text()
    dialog = screen.review_consent()
    texts = [w.text() for w in dialog.findChildren(QLabel)]
    assert any('2026-10-02 09:30' in t and 'Maya' in t for t in texts)
    assert not dialog.findChildren(QCheckBox)                  # nothing to tick: only the customer's own answers
    dialog.accept()
    wait(lambda: not screen.consent_runner.busy and not screen.runner.busy and screen.customer is not None
         and screen.customer.consent_training, 5)
    assert b.customer(3).consent_proposed is None and not screen.consent_note.isVisible()
    with pytest.raises(ValidationError, match='can only come from the customer'):
        b.confirm_consent(3, '2026-10-02T09:30:00Z')            # nothing left to confirm


def test_no_proposal_means_nothing_to_confirm(widgets, wait):
    b = DemoBackend()
    screen = CustomerScreen(b, 'dark', 'admin'); widgets.append(screen); screen.resize(1366, 768); screen.show()
    screen.open(2); wait(lambda: screen.customer is not None and not screen.runner.busy, 5)
    assert screen.consent_button.text() == 'Consent…'
    dialog = screen.review_consent()
    assert any('nothing to confirm' in w.text() for w in dialog.findChildren(QLabel))
    assert [x.text() for x in dialog.findChildren(QPushButton)] == ['Close']
    from dataclasses import replace
    with pytest.raises(ValidationError, match='gave at setup'):
        b.update_customer(replace(b.customer(2), consent_live=True))
