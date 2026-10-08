"""Studio · Inbox over the demo backend: owner answers from Telegram with their chip, filters, Accept as tag (Tag · AI
opens with the owner's answer as a draft, nothing saved for the admin), Fix, Not a label, consent, and the HTTP
client's calls."""
import json

import httpx
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.http_backend import HttpBackend
from home_guard_project.admin.inbox import InboxScreen, camera_title, owner_prefill
from home_guard_project.admin.models import TokenPair, StaffOut
from home_guard_project.admin.tag_view import TagView
from home_guard_project.admin.tag_widgets import ProvenanceChip


def inbox(widgets, wait, backend=None):
    b = backend or DemoBackend()
    s = InboxScreen(b, 'admin', 'dark'); widgets.append(s); s.resize(1366, 700); s.show(); s.open()  # the shell's signature
    wait(lambda: not s.loader.busy and s.table.rowCount() > 0)
    return s, b


def row_of(screen, feedback_id):
    return next(r for r, i in enumerate(screen.items) if i.feedback_id == feedback_id)


def test_waiting_answers_with_their_chip_and_filters(widgets, wait):
    s, _ = inbox(widgets, wait)
    assert [i.feedback_id for i in s.items] == [905, 904, 903, 902]            # 901 is handled already
    assert all(isinstance(s.table.cellWidget(r, 3), ProvenanceChip) and s.table.cellWidget(r, 3).text() == 'Owner · Telegram'
               for r in range(s.table.rowCount()))
    assert s.table.item(row_of(s, 904), 2).text() == 'דלת אחורית'                 # the owner's camera name
    s.table.selectRow(row_of(s, 904))
    assert s.owner_said.text() == 'זה הגנן שלנו, הוא מגיע כל יום שלישי' and 'voice answer' in s.owner_how.text()
    assert s.model_chip.text() == 'Model · gpt-4o' and s.owner_pill.text() == 'other'
    s.owner_label.setCurrentIndex(s.owner_label.findData('other')); s.load()
    wait(lambda: not s.loader.busy and len(s.items) == 1)
    assert s.items[0].feedback_id == 904
    s.owner_label.setCurrentIndex(s.owner_label.findData('-')); s.load()
    wait(lambda: not s.loader.busy and len(s.items) == 1)
    assert s.items[0].feedback_id == 902                                         # words only, no tag
    s.owner_label.setCurrentIndex(0); s.handled.setCurrentIndex(s.handled.findData('handled')); s.load()
    wait(lambda: not s.loader.busy and [i.feedback_id for i in s.items] == [901])
    assert s.decision_chip.text() == 'Admin · Maya Cohen' and s.reopen_button.isVisible()


def test_accept_opens_tag_ai_with_the_owner_answer_as_a_draft(widgets, wait):
    s, b = inbox(widgets, wait)
    asked = []
    s.tag_requested.connect(lambda key, fields: asked.append((key, fields)))
    s.table.selectRow(row_of(s, 904)); s.activateWindow(); wait(s.isActiveWindow); s.setFocus(); QTest.keyClick(s, Qt.Key.Key_A)
    wait(lambda: asked and not s.loader.busy)
    key, fields = asked[0]
    assert key == 'of:production_demo/back_1791031000_alert'
    assert fields['category'] == 'other' and fields['other_text'].startswith('זה הגנן')
    assert fields['notes'].startswith('Owner · Telegram (other): ')
    assert 904 not in [i.feedback_id for i in s.items]                           # handled: out of the waiting list
    assert b.inbox(handled='handled')[0].decision == 'accepted'
    v = TagView(b, 'admin'); widgets.append(v); v.resize(1366, 700); v.show()
    v.open_from_owner(key, fields)
    wait(lambda: v.key == key and not v.clip_runner.busy and v.form and v.form.get('category') == 'other', 10)
    assert v.form['other_text'].startswith('זה הגנן') and 'Owner · Telegram' in v.form['notes']
    assert v.dirty() and v.started_from.text() == 'from Owner · Telegram'      # a draft: the admin saves it
    assert not v.detail.get('tag')


def test_fix_not_a_label_and_consent(widgets, wait):
    s, b = inbox(widgets, wait)
    asked = []
    s.tag_requested.connect(lambda key, fields: asked.append((key, fields)))
    s.table.selectRow(row_of(s, 902))
    assert not s.accept_button.isEnabled() and s.consent.isVisible()             # no training consent
    assert s.table.item(row_of(s, 902), 7).text() == 'Waiting · probably not a label'
    assert 'probably not a label' in s.owner_how.text()
    assert s.model_chip.toolTip() == 'Prompt version: 2026-10-03.tagged-rules-label-animals-why-owner-facts'
    s.accept(); assert not s.writer.busy and not asked
    s.note.setText('asks why there are so many alerts'); s.not_label()
    wait(lambda: not s.writer.busy and not s.loader.busy and 902 not in [i.feedback_id for i in s.items])
    handled = {i.feedback_id: i for i in b.inbox(handled='handled')}
    assert handled[902].decision == 'not_label' and handled[902].decision_note == 'asks why there are so many alerts'
    assert not asked
    s.table.selectRow(row_of(s, 903)); s.fix()
    wait(lambda: asked and not s.loader.busy)
    assert asked == [('ds:yard_1791000007_trigger', None)]
    s.handled.setCurrentIndex(s.handled.findData('all')); s.load()
    wait(lambda: not s.loader.busy and len(s.items) == 5)
    s.table.selectRow(row_of(s, 902)); s.reopen()
    wait(lambda: not s.writer.busy and not s.loader.busy and s.items[row_of(s, 902)].decision is None)


def test_owner_prefill_and_camera_titles():
    b = DemoBackend()
    item = b.inbox(handled='all')[0]
    item.owner_label, item.owner_text, item.transcript, item.raw_text = 'empty', '', '', ''
    assert owner_prefill(item) == {'category': 'N10', 'raw_label': 'normal', 'notes': 'Owner · Telegram: empty'}
    for tag, raw in (('normal', 'normal'), ('rule_mismatch', 'normal'), ('suspicious', 'suspicious'),
                     ('escalation', 'escalation')):
        item.owner_label = tag
        assert owner_prefill(item)['raw_label'] == raw and 'category' not in owner_prefill(item)
    item.owner_label, item.raw_text = '', 'why so many alerts?'
    assert owner_prefill(item) == {'notes': 'Owner · Telegram: why so many alerts?'}
    item.camera_name, item.camera = None, 'ameer_week_0_1_ch6'
    assert camera_title(item) == 'Camera 6'
    item.camera_name = 'מצלמה 6'
    assert camera_title(item) == 'מצלמה 6'


def test_http_client_sends_filters_and_decisions():
    seen = []
    one = json.loads(open(__import__('pathlib').Path(__file__).resolve().parents[2] / 'home_guard_project' / 'admin' /
                          'demo_data' / 'inbox.json', encoding='utf-8').read())[0]

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params),
                     json.loads(request.content) if request.content else None))
        if request.method == 'GET':
            return httpx.Response(200, json=[one])
        if request.url.path.endswith('/decision') and request.method == 'POST' and seen[-1][3]['decision'] == 'accepted':
            return httpx.Response(409, json={'detail': 'Dana has withdrawn consent to training use'})
        return httpx.Response(200, json=dict(one, decision='not_label' if request.method == 'POST' else None))
    b = HttpBackend('http://localhost:8000', transport=httpx.MockTransport(handler))
    b.tokens = TokenPair('a', 'r', 900, StaffOut(1, 'a@b.c', 'A', 'admin'))
    items = b.inbox(owner_label='', handled='unhandled', camera=None, customer_id=3)
    assert items[0].feedback_id == 905 and seen[0][2] == {'owner_label': '', 'handled': 'unhandled', 'customer_id': '3'}
    assert b.inbox_decide(905, 'not_label', 'a question').decision == 'not_label'
    assert seen[1][1] == '/v1/inbox/905/decision' and seen[1][3] == {'decision': 'not_label', 'note': 'a question'}
    try:
        b.inbox_decide(905, 'accepted')
        assert False, 'a refused accept must raise'
    except Exception as e:  # noqa: BLE001
        assert 'consent' in str(e)
    assert b.inbox_reopen(905).decision is None and seen[-1][0] == 'DELETE'
