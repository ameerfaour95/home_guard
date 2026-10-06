"""The tagging workspace: work queue on the left, the clip and what everyone said about it in the middle, the tag on
the right. Laid out like CVAT / Label Studio / Encord / V7 video tools: the picture dominates, the class panel is on
the right with one-key hotkeys, and next/previous walks a slim queue.

Keys (also in the help): Ctrl+Enter save and next · n/s/e then a digit picks a category (s3 = S3, n0 = N10) · o other
· Shift+N/S/E raw label · Space play · , . one frame · ← → one second · v crop/full · f evidence frame · a use the
teacher · c needs check · x delete · j/k next/previous clip · d description · ? help.
"""
from copy import deepcopy
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QImage
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QListView, QComboBox,
                               QLineEdit, QPlainTextEdit, QCheckBox, QScrollArea, QFrame, QSizePolicy, QDialog,
                               QAbstractItemView)
from .backend import AuthError
from .player import VideoCanvas
from .tag_widgets import (QueueModel, QueueDelegate, CategoryButton, ChipGroup, OpinionCard, Pill, TIER_TOKENS,
                          TIER_TITLES)
from .workers import TaskRunner
from .widgets.common import label, button

TIERS = [('open', 'To do'), ('contradiction', 'Contradictions'), ('check', 'To check'), ('untagged', 'Untagged'),
         ('done', 'Done'), ('all', 'All clips')]
ORIGINS = [('', 'All sources'), ('dataset', 'Old tags (dataset)'), ('customer', 'Customers (live)'),
           ('owner_feedback', 'Customers (local copies)')]
HELP = [('Ctrl + Enter', 'Save and open the next clip'), ('n / s / e, then 1–9 or 0', 'Category (s 3 = S3, n 0 = N10)'),
        ('o', 'Category: other'), ('Shift + N / S / E', 'Raw label: normal / suspicious / escalation'),
        ('Space', 'Play / pause'), (', / .', 'One frame back / forward'), ('← / →', 'One second back / forward'),
        ('v', 'Crop / full frame'), ('f', 'Evidence frame = this moment'), ('a', "Use the teacher's suggestion"),
        ('c / x', 'Needs check / delete'), ('j / k', 'Next / previous clip'), ('d', 'Write the description'),
        ('Esc', 'Leave a text field'), ('?', 'This help')]
EMPTY_TEXT = {'clip': 'No full-frame video for this clip', 'crop': 'No crop for this clip'}


class TagView(QWidget):
    session_expired = Signal()
    customer_requested = Signal(int)
    label_requested = Signal(int)

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__()
        self.backend, self.role, self.theme = backend, role, theme
        self.state, self.detail, self.key, self.wanted = None, None, None, None
        self.form, self.saved_form, self.drafts = None, None, {}
        self.view, self.raw_manual, self.chord = 'crop', False, None
        self.categories = {}
        self.state_runner, self.queue_runner, self.clip_runner, self.media_runner, self.save_runner, self.side_runner = \
            [TaskRunner(self) for _ in range(6)]
        self.state_runner.finished.connect(self.state_loaded); self.queue_runner.finished.connect(self.queue_loaded)
        self.clip_runner.finished.connect(self.clip_loaded); self.media_runner.finished.connect(self.media_loaded)
        self.save_runner.finished.connect(self.saved); self.side_runner.finished.connect(self.side_done)
        self.pending_queue, self.pending_media, self.select_after = None, None, None
        self.chord_timer = QTimer(self); self.chord_timer.setSingleShot(True); self.chord_timer.setInterval(1500)
        self.chord_timer.timeout.connect(lambda: self.set_chord(None))
        self.player = QMediaPlayer(self); self.sink = QVideoSink(self); self.player.setVideoSink(self.sink)
        self._build()
        self.sink.videoFrameChanged.connect(self.canvas.frame_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(lambda d: self.position_changed(self.player.position()))
        self.player.playbackStateChanged.connect(
            lambda s: self.play.setText('Pause' if s == QMediaPlayer.PlaybackState.PlayingState else 'Play'))
        self.player.errorOccurred.connect(lambda *_: self.show_video_message('This video could not be played.'))
        self._shortcuts()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(24, 16, 24, 16); root.setSpacing(12)
        head = QHBoxLayout(); head.setSpacing(16)
        titles = QVBoxLayout(); titles.setSpacing(2)
        titles.addWidget(label('TAGGING STUDIO', 'eyebrow'))
        self.title = label('Loading clips…', 'clipTitle'); self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        titles.addWidget(self.title)
        head.addLayout(titles, 1)
        self.stats = {}
        for tier in ('contradiction', 'check', 'untagged', 'done'):
            box = QVBoxLayout(); box.setSpacing(0)
            number = label('—', 'statNumber'); number.setAlignment(Qt.AlignmentFlag.AlignRight)
            caption = label(TIER_TITLES[tier], 'muted'); caption.setAlignment(Qt.AlignmentFlag.AlignRight)
            box.addWidget(number); box.addWidget(caption); head.addLayout(box); self.stats[tier] = number
        head.addSpacing(8)
        self.export_button = button('Export training set…', self.export, 'compact'); head.addWidget(self.export_button)
        head.addWidget(button('?', self.help, 'compact'))
        root.addLayout(head)
        self.banner = QFrame(); self.banner.setObjectName('banner')
        banner = QHBoxLayout(self.banner); banner.setContentsMargins(14, 8, 10, 8)
        self.banner_text = label('', '', True); banner.addWidget(self.banner_text, 1)
        self.banner_action = button('', None, 'link'); banner.addWidget(self.banner_action); self.banner_callback = None
        banner.addWidget(button('Dismiss', self.banner.hide, 'link'))
        self.banner.hide(); root.addWidget(self.banner)

        split = QSplitter(Qt.Orientation.Horizontal); split.setHandleWidth(12); split.setChildrenCollapsible(False)
        split.setStyleSheet('QSplitter::handle { background: transparent; }')
        root.addWidget(split, 1)
        split.addWidget(self._queue_panel()); split.addWidget(self._center()); split.addWidget(self._form_panel())
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setStretchFactor(2, 0)
        split.setSizes([240, 600, 440])
        self.split, self.split_moved = split, False
        split.splitterMoved.connect(lambda *_: setattr(self, 'split_moved', True))

    def _queue_panel(self):
        panel = QFrame(); panel.setObjectName('panel'); panel.setMinimumWidth(210); panel.setMaximumWidth(340)
        col = QVBoxLayout(panel); col.setContentsMargins(10, 10, 10, 0); col.setSpacing(8)
        self.tier = QComboBox(); self.origin = QComboBox()
        for value, text in TIERS: self.tier.addItem(text, value)
        for value, text in ORIGINS: self.origin.addItem(text, value)
        self.tier.setAccessibleName('Which clips'); self.origin.setAccessibleName('Source')
        self.tier.currentIndexChanged.connect(lambda: self.load_queue())
        self.origin.currentIndexChanged.connect(lambda: self.load_queue())
        col.addWidget(self.tier); col.addWidget(self.origin)
        self.search = QLineEdit(); self.search.setPlaceholderText('Filter: clip, camera, house')
        self.search.setClearButtonEnabled(True)
        self.search_timer = QTimer(self); self.search_timer.setSingleShot(True); self.search_timer.setInterval(300)
        self.search_timer.timeout.connect(lambda: self.load_queue())
        self.search.textChanged.connect(lambda: self.search_timer.start()); col.addWidget(self.search)
        self.queue_count = label('', 'muted'); col.addWidget(self.queue_count)
        self.model = QueueModel()
        self.list = QListView(); self.list.setObjectName('tagQueue'); self.list.setModel(self.model)
        self.list.setItemDelegate(QueueDelegate(self.theme, self.list)); self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.clicked.connect(lambda index: self.open_key(self.model.rows[index.row()]['key']))
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        col.addWidget(self.list, 1)
        return panel

    def _center(self):
        center = QWidget(); col = QVBoxLayout(center); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(10)
        reasons = QHBoxLayout(); reasons.setSpacing(8)
        self.tier_pill = Pill(self.theme); reasons.addWidget(self.tier_pill)
        self.reasons = label('', 'muted'); self.reasons.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        reasons.addWidget(self.reasons, 1)
        self.open_label = button('Edit boxes', lambda: self.detail and self.label_requested.emit(self.detail['item']['event_id']), 'link')
        self.open_label.hide(); reasons.addWidget(self.open_label)
        col.addLayout(reasons)
        self.canvas = VideoCanvas(self.theme); self.canvas.overlay.hide(); self.canvas.message = 'Select a clip'
        self.canvas.setMinimumHeight(220)
        col.addWidget(self.canvas, 3)
        bar = QHBoxLayout(); bar.setSpacing(0)
        self.segments = {}
        for kind, text in (('crop', 'Crop'), ('clip', 'Full frame')):
            b = button(text, lambda checked=False, k=kind: self.switch_view(k), 'segment'); b.setCheckable(True)
            b.setToolTip('Crop: what the AI sees · Full frame: the whole camera  (V)')
            self.segments[kind] = b; bar.addWidget(b)
        bar.addSpacing(12)
        self.play = button('Play', self.toggle_play, 'compact'); bar.addWidget(self.play); bar.addSpacing(4)
        back = button('‹', lambda: self.step(-1), 'compact'); back.setToolTip('One frame back  ,'); bar.addWidget(back)
        bar.addSpacing(4)
        fwd = button('›', lambda: self.step(1), 'compact'); fwd.setToolTip('One frame forward  .'); bar.addWidget(fwd)
        bar.addSpacing(10)
        self.clock = label('0.00 s', 'muted'); self.clock.setMinimumWidth(84); bar.addWidget(self.clock)
        bar.addStretch()
        self.speed = QComboBox()
        for text, rate in (('0.5×', .5), ('1×', 1.), ('2×', 2.)): self.speed.addItem(text, rate)
        self.speed.setCurrentIndex(1); self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        bar.addWidget(self.speed)
        col.addLayout(bar)
        self.cards = {who: OpinionCard(who, self.theme) for who in ('old', 'owner', 'ai', 'teacher')}
        self.card_grid = QGridLayout(); self.card_grid.setSpacing(10)
        col.addLayout(self.card_grid)
        self.card_columns = 0; self.layout_cards()
        return center

    def _form_panel(self):
        outer = QFrame(); outer.setObjectName('panel'); outer.setMinimumWidth(420); outer.setMaximumWidth(600)
        col = QVBoxLayout(outer); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(); body.setObjectName('tagForm'); scroll.setWidget(body)
        form = QVBoxLayout(body); form.setContentsMargins(16, 14, 16, 14); form.setSpacing(8)
        head = QHBoxLayout(); head.addWidget(label('CATEGORY', 'eyebrow')); head.addStretch()
        self.started_from = label('', 'muted'); head.addWidget(self.started_from)
        self.chord_label = Pill(self.theme); self.chord_label.hide(); head.addWidget(self.chord_label)
        form.addLayout(head)
        self.category_grid = QGridLayout(); self.category_grid.setHorizontalSpacing(6); self.category_grid.setVerticalSpacing(5)
        form.addLayout(self.category_grid)
        self.category_buttons = {}
        self.other_text = QLineEdit(); self.other_text.setPlaceholderText('What is it? (category other)')
        self.other_text.textChanged.connect(lambda t: self.set_field('other_text', t)); self.other_text.hide()
        form.addWidget(self.other_text)
        self.fields_box = QVBoxLayout(); self.fields_box.setSpacing(8); form.addLayout(self.fields_box)
        form.addWidget(label('DESCRIPTION', 'eyebrow'))
        self.description = QPlainTextEdit(); self.description.setPlaceholderText('What happens in the clip, in English.')
        self.description.setFixedHeight(84); self.description.textChanged.connect(
            lambda: self.set_field('description', self.description.toPlainText()))
        form.addWidget(self.description)
        self.notes = QPlainTextEdit(); self.notes.setPlaceholderText('Notes for us (not used for training)')
        self.notes.setFixedHeight(52); self.notes.textChanged.connect(lambda: self.set_field('notes', self.notes.toPlainText()))
        form.addWidget(self.notes)
        checks = QHBoxLayout()
        self.needs_check = QCheckBox('Needs check  (C)'); self.needs_check.toggled.connect(lambda v: self.set_field('needs_check', v))
        self.delete = QCheckBox('Delete  (X)'); self.delete.setToolTip('Leave this clip out of training and the eval')
        self.delete.toggled.connect(lambda v: self.set_field('delete', v))
        checks.addWidget(self.needs_check); checks.addStretch(); checks.addWidget(self.delete)
        form.addLayout(checks)
        self.history = label('', 'muted', True); form.addWidget(self.history)
        form.addStretch()
        col.addWidget(scroll, 1)
        actions = QFrame(); actions.setObjectName('banner')
        row = QHBoxLayout(actions); row.setContentsMargins(12, 10, 12, 10); row.setSpacing(8)
        self.accept_button = button('Use teacher  A', self.accept_teacher, 'compact'); row.addWidget(self.accept_button)
        self.teach_button = button('Ask teacher', self.teach, 'compact'); self.teach_button.hide(); row.addWidget(self.teach_button)
        row.addStretch()
        self.save_state = label('', 'muted'); row.addWidget(self.save_state)
        self.save_button = button('Save && next', self.save, 'primary'); self.save_button.setToolTip('Ctrl + Enter')
        row.addWidget(self.save_button)
        col.addWidget(actions)
        return outer

    def build_form(self, taxonomy):
        """The taxonomy arrives with the first state: categories in three columns, then the observation fields."""
        if self.category_buttons:
            return
        for c, group in enumerate(taxonomy['groups']):
            title = QHBoxLayout(); title.addWidget(label(group['name'].upper(), 'eyebrowMuted')); title.addStretch()
            he = label(group['he'], 'muted'); title.addWidget(he)
            self.category_grid.addLayout(title, 0, c)
            for r, cat in enumerate(group['categories']):
                b = CategoryButton(cat, self.theme); b.clicked.connect(lambda checked=False, i=cat['id']: self.set_category(i))
                self.category_grid.addWidget(b, r + 1, c); self.category_buttons[cat['id']] = b
                self.categories[cat['id']] = cat['name']
        other = next(c for c in taxonomy['categories'] if c['id'] == 'other')
        b = CategoryButton({**other, 'group': '', 'name': 'none of the above: say what it is'}, self.theme); b.clicked.connect(lambda: self.set_category('other'))
        self.category_grid.addWidget(b, 11, 0, 1, 3); self.category_buttons['other'] = b
        self.chips = {}
        for name, title, values, multi in (('raw_label', 'RAW LABEL · how serious is the scene itself', taxonomy['labels'], False),
                                           ('zone', 'ZONE', taxonomy['zones'], False),
                                           ('movement', 'MOVEMENT', taxonomy['movements'], False),
                                           ('flags', 'FLAGS', taxonomy['flags'], True),
                                           ('visibility', 'VISIBILITY', taxonomy['visibility'], False)):
            self.fields_box.addWidget(label(title, 'eyebrow'))
            chips = ChipGroup(values, multi); chips.changed.connect(lambda n=name: self.chip_changed(n))
            self.fields_box.addWidget(chips); self.chips[name] = chips
        evidence = QHBoxLayout(); evidence.addWidget(label('EVIDENCE FRAME', 'eyebrow')); evidence.addSpacing(8)
        self.evidence = label('none', 'muted'); evidence.addWidget(self.evidence); evidence.addStretch()
        mark = button('Mark  F', self.mark_evidence, 'link'); mark.setToolTip('This moment is the frame that shows best what happens (F)')
        evidence.addWidget(mark)
        evidence.addWidget(button('Go to', self.go_to_evidence, 'link'))
        evidence.addWidget(button('Clear', lambda: (self.set_field('evidence_sec', None), self.set_field('evidence_frame', None),
                                                   self.render_evidence()), 'link'))
        self.fields_box.addLayout(evidence)

    def layout_cards(self, width=None):
        columns = 4 if (width or self.canvas.width()) > 860 else 2
        if columns == self.card_columns:
            return
        self.card_columns = columns
        for i, who in enumerate(('old', 'owner', 'ai', 'teacher')):
            self.card_grid.addWidget(self.cards[who], i // columns, i % columns)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        total = self.split.width() or self.width() - 48
        if not self.split_moved and total > 0:
            queue = max(220, min(320, int(total * .19)))
            form = max(420, min(600, int(total * .32)))
            self.split.setSizes([queue, max(300, total - queue - form - 24), form])
            self.layout_cards(total - queue - form - 24)
        else:
            self.layout_cards()

    def _shortcuts(self):
        keys = {'Ctrl+Return': self.save, 'Ctrl+Enter': self.save, 'Ctrl+S': self.save, 'Space': self.toggle_play,
                ',': lambda: self.step(-1), '.': lambda: self.step(1), 'Left': lambda: self.seek_by(-1000),
                'Right': lambda: self.seek_by(1000), 'V': lambda: self.switch_view('clip' if self.view == 'crop' else 'crop'),
                'F': self.mark_evidence, 'A': self.accept_teacher, 'C': lambda: self.needs_check.toggle(),
                'X': lambda: self.delete.toggle(), 'J': lambda: self.move(1), 'K': lambda: self.move(-1),
                'D': lambda: self.description.setFocus(), 'O': lambda: self.set_category('other'), '?': self.help,
                'N': lambda: self.set_chord('N'), 'S': lambda: self.set_chord('S'), 'E': lambda: self.set_chord('E'),
                'Shift+N': lambda: self.set_raw('normal'), 'Shift+S': lambda: self.set_raw('suspicious'),
                'Shift+E': lambda: self.set_raw('escalation'), 'Escape': self.escape}
        for digit in range(10):
            keys[str(digit)] = lambda d=digit: self.chord_digit(d)
        self.shortcuts = []
        for key, slot in keys.items():
            sc = QShortcut(QKeySequence(key), self); sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot); self.shortcuts.append(sc)

    # ------------------------------------------------------------------ loading
    def open(self, key=None):
        """Show the studio; with *key*, open that clip."""
        if key:
            self.select_after = key
        if self.state is None:
            self.load_state()
        elif key:
            self.open_key(key)

    def load_state(self):
        self.state_runner.start(self.backend.tagging_state)

    def state_loaded(self, state, error):
        if self.handle_error(error, 'The tagging studio could not load'):
            return
        self.state = state
        self.build_form(state['taxonomy'])
        for tier, number in self.stats.items():
            number.setText(str(state['counts'].get(tier, 0)))
        self.teach_button.setVisible(bool(state.get('can_ask_teacher')))
        if not self.model.rows:
            self.load_queue()

    def load_queue(self, select=None):
        if select:
            self.select_after = select
        request = (self.tier.currentData(), self.origin.currentData(), self.search.text())
        if not self.queue_runner.start(lambda: self.backend.tagging_queue(*request)):
            self.pending_queue = True

    def queue_loaded(self, queue, error):
        if self.pending_queue:
            self.pending_queue = None; self.load_queue(); return
        if self.handle_error(error, 'The work queue could not load'):
            return
        self.model.set_rows(queue['items'])
        total = queue['count']
        self.queue_count.setText(f'{total} clip{"s" if total != 1 else ""}' + ('' if total <= len(queue['items']) else f' · first {len(queue["items"])} shown'))
        target = self.select_after or self.key
        self.select_after = None
        if target and self.model.row_of(target) >= 0:
            self.highlight(target)
            if target != self.key:
                self.open_key(target)
        elif target and target != self.key:
            self.open_key(target)
        elif not self.key and self.model.rows:
            self.open_key(self.model.rows[0]['key'])
        elif not self.model.rows and not self.key:
            self.title.setText('Nothing to do here'); self.canvas.message = 'No clips match this filter.'; self.canvas.update()

    def highlight(self, key):
        row = self.model.row_of(key)
        if row >= 0:
            index = self.model.index(row, 0)
            self.list.setCurrentIndex(index); self.list.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)

    def open_key(self, key):
        if self.form is not None and self.key and self.dirty():
            self.drafts[self.key] = deepcopy(self.form)      # never lose an edit: it waits for this clip's return
        self.wanted = key
        self.highlight(key)
        self.clip_runner.start(lambda: (key, self.backend.tagging_clip(key)))

    def clip_loaded(self, result, error):
        if self.handle_error(error, 'This clip could not load'):
            return
        key, detail = result
        if key != self.wanted:
            self.open_key(self.wanted); return
        self.key, self.detail = key, detail
        self.saved_form = deepcopy(detail['form'])
        self.form = deepcopy(self.drafts.pop(key, None) or detail['form'])
        self.raw_manual = bool(detail['tag'] and detail['tag']['fields'].get('raw_label'))
        item = detail['item']
        bits = [item.get('camera'), item.get('source'), item.get('date'), item.get('local_time'),
                f"{item['duration_sec']:.1f} s" if item.get('duration_sec') else '']
        self.title.setText(item['clip_id'] + '    ' + '  ·  '.join(b for b in bits if b))
        a = detail['assessment']
        self.tier_pill.show_label(TIER_TITLES.get(a['tier_name'], a['tier_name']), TIER_TOKENS.get(a['tier_name'], 'muted'))
        needs = (item.get('info') or {}).get('needs_check')
        self.reasons.setText('; '.join(a['reasons']) + (f'   ·   dataset: {needs}' if needs else ''))
        self.reasons.setToolTip(self.reasons.text())
        self.open_label.setVisible(bool(item.get('event_id')) and self.role == 'admin')
        conflicted = {c['a'] for c in a['conflicts']} | {c['b'] for c in a['conflicts']}
        for who, card in self.cards.items():
            card.show_opinion(detail['opinions'].get(who), who in conflicted, self.categories)
        self.accept_button.setEnabled('teacher' in detail['opinions'])
        self.started_from.setText({'old': 'started from the old tag', 'studio': ''}.get(detail['prefilled_from'], 'new'))
        tag = detail.get('tag')
        self.history.setText(f"Saved {len(detail['history'])} time(s) · last by {tag['by']} at {tag['at'][:16].replace('T', ' ')}"
                             if tag else '')
        self.render_form()
        self.update_save_state()
        self.banner.hide()
        if self.view not in [k for k, ok in detail['media'].items() if ok]:
            self.view = 'crop' if detail['media'].get('crop') else 'clip'
        for kind, b in self.segments.items():
            b.setEnabled(bool(detail['media'].get(kind))); b.setChecked(kind == self.view)
        self.load_media()

    def load_media(self):
        self.player.stop(); self.player.setSource(QUrl())
        self.canvas.image = QImage(); self.canvas.message = 'Loading video…'; self.canvas.update()
        if not self.detail or not self.detail['media'].get(self.view):
            self.show_video_message(EMPTY_TEXT[self.view]); return
        request = (self.key, self.view)
        if not self.media_runner.start(lambda: (request, self.backend.tagging_media(*request))):
            self.pending_media = True

    def media_loaded(self, result, error):
        if self.pending_media:
            self.pending_media = None; self.load_media(); return
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit(); return
            message = str(error)
            self.show_video_message(message)
            customer = (self.detail or {}).get('item', {}).get('info', {}).get('customer_id')
            if 'consent' in message.lower() or 'agreed' in message.lower():
                self.show_banner(f'{message}. An admin confirms consent on the customer page.',
                                 'Open customer', (lambda: self.customer_requested.emit(customer)) if customer else None)
            return
        (key, kind), access = result
        if key != self.key or kind != self.view:
            self.load_media(); return
        self.player.setSource(QUrl(access['url'])); self.player.setPlaybackRate(self.speed.currentData())
        self.player.play(); self.player.pause()

    def show_video_message(self, message):
        self.canvas.image = QImage(); self.canvas.message = message; self.canvas.update()

    def show_banner(self, text, action=None, callback=None):
        self.banner_text.setText(text)
        if self.banner_callback is not None:
            self.banner_action.clicked.disconnect(self.banner_callback)
        self.banner_callback = callback
        self.banner_action.setVisible(bool(action and callback))
        self.banner_text.setToolTip(text)
        if action and callback:
            self.banner_action.setText(action); self.banner_action.clicked.connect(callback)
        self.banner.show()

    def handle_error(self, error, title):
        if not error:
            return False
        if isinstance(error, AuthError):
            self.session_expired.emit()
        else:
            self.show_banner(f'{title}: {error}', 'Try again', lambda: (self.banner.hide(), self.load_state(), self.load_queue()))
        return True

    # ------------------------------------------------------------------ the form
    def set_field(self, name, value):
        if self.form is None or getattr(self, '_rendering', False):
            return
        self.form[name] = value
        self.update_save_state()

    def render_form(self):
        self._rendering = True
        try:
            f = self.form
            teacher = (self.detail['opinions'].get('teacher') or {}).get('category', '')
            for cid, b in self.category_buttons.items():
                b.setChecked(cid == f.get('category')); b.hint = cid == teacher; b.update()
            self.other_text.setVisible(f.get('category') == 'other')
            if self.other_text.text() != (f.get('other_text') or ''):
                self.other_text.setText(f.get('other_text') or '')
            for name, chips in self.chips.items():
                chips.set_value(f.get(name) or ([] if name == 'flags' else ''))
            for widget, name in ((self.description, 'description'), (self.notes, 'notes')):
                if widget.toPlainText() != (f.get(name) or ''):
                    widget.setPlainText(f.get(name) or '')
            self.needs_check.setChecked(bool(f.get('needs_check'))); self.delete.setChecked(bool(f.get('delete')))
            self.render_evidence()
        finally:
            self._rendering = False

    def render_evidence(self):
        sec, frame = (self.form or {}).get('evidence_sec'), (self.form or {}).get('evidence_frame')
        self.evidence.setText('none' if sec is None else f'{sec:.2f} s' + ('' if frame is None else f'  ·  frame {frame}'))

    def chip_changed(self, name):
        if self.form is None:
            return
        self.form[name] = self.chips[name].value()
        if name == 'raw_label':
            self.raw_manual = bool(self.form[name])
        self.update_save_state()

    def group_label(self, category):
        for group in (self.state or {}).get('taxonomy', {}).get('groups', []):
            if any(c['id'] == category for c in group['categories']):
                return group['label']
        return ''

    def set_category(self, category):
        if self.form is None:
            return
        self.form['category'] = '' if self.form.get('category') == category else category
        if not self.raw_manual:
            self.form['raw_label'] = self.group_label(self.form['category'])
        self.render_form(); self.update_save_state()
        if self.form['category'] == 'other':
            self.other_text.setFocus()

    def set_raw(self, value):
        if self.form is None:
            return
        self.form['raw_label'], self.raw_manual = value, True
        self.render_form(); self.update_save_state()

    def set_chord(self, group):
        self.chord = group
        if group:
            self.chord_label.show_label(f'{group} …', 'action'); self.chord_timer.start()
        else:
            self.chord_label.hide()

    def chord_digit(self, digit):
        group = self.chord
        self.set_chord(None)
        if not group:
            return
        cid = f'{group}{10 if digit == 0 else digit}'
        if cid in self.category_buttons:
            self.set_category(cid)
        else:
            self.show_banner(f'There is no category {cid}.')

    def escape(self):
        self.set_chord(None)
        focused = self.focusWidget()
        if isinstance(focused, (QLineEdit, QPlainTextEdit)):
            self.setFocus()

    def accept_teacher(self):
        t = (self.detail or {}).get('opinions', {}).get('teacher')
        if not t or self.form is None:
            return
        d = t.get('detail') or {}
        if t.get('category'):
            self.form['category'] = t['category']
        if t.get('label') in ('normal', 'suspicious', 'escalation'):
            self.form['raw_label'], self.raw_manual = t['label'], True
        elif t.get('label') == 'empty':
            self.form['category'] = self.form.get('category') or 'N10'; self.form['raw_label'] = 'normal'
        elif self.form.get('category') and not self.raw_manual:
            self.form['raw_label'] = self.group_label(self.form['category'])
        for name in ('zone', 'movement', 'visibility', 'other_text'):
            if d.get(name):
                self.form[name] = d[name]
        if isinstance(d.get('flags'), list):
            self.form['flags'] = [f for f in d['flags'] if f in self.chips['flags'].buttons]
        if not self.form.get('description') and t.get('text'):
            self.form['description'] = t['text']
        self.render_form(); self.update_save_state()
        self.save_state.setText("Teacher's suggestion applied: check it")

    def dirty(self):
        return self.form is not None and self.form != self.saved_form

    def update_save_state(self):
        if self.form is None:
            self.save_state.setText(''); return
        self.save_state.setText('Unsaved changes' if self.dirty() else ('Saved' if self.detail and self.detail.get('tag') else ''))

    # ------------------------------------------------------------------ video
    def switch_view(self, kind):
        if not self.detail or not self.detail['media'].get(kind) or kind == self.view:
            return
        self.view = kind
        for k, b in self.segments.items():
            b.setChecked(k == kind)
        position = self.player.position()
        self.load_media()
        self._resume_at = position

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif not self.player.source().isEmpty():
            self.player.play()

    def fps(self):
        return float((self.detail or {}).get('fps') or 7.0)

    def step(self, direction):
        if self.player.source().isEmpty():
            return
        self.player.pause()
        frame = round(self.player.position() * self.fps() / 1000) + direction
        self.player.setPosition(max(0, min(self.player.duration(), round(frame * 1000 / self.fps()))))

    def seek_by(self, ms):
        if not self.player.source().isEmpty():
            self.player.setPosition(max(0, min(self.player.duration(), self.player.position() + ms)))

    def position_changed(self, position):
        resume = getattr(self, '_resume_at', None)
        if resume and self.player.duration() > 0:
            self._resume_at = None; self.player.setPosition(min(resume, self.player.duration()))
        self.clock.setText(f'{position / 1000:.2f} s / {self.player.duration() / 1000:.1f}')

    def mark_evidence(self):
        if self.form is None:
            return
        sec = round(self.player.position() / 1000, 3)
        self.form['evidence_sec'], self.form['evidence_frame'] = sec, round(sec * self.fps())
        self.render_evidence(); self.update_save_state()

    def go_to_evidence(self):
        if self.form and self.form.get('evidence_sec') is not None:
            self.player.pause(); self.player.setPosition(int(self.form['evidence_sec'] * 1000))

    # ------------------------------------------------------------------ saving and moving
    def save(self):
        if self.form is None or not self.key:
            return
        if not self.form.get('category') and not self.form.get('delete') and not self.form.get('needs_check'):
            self.show_banner('Choose a category first (or mark the clip Needs check or Delete).'); return
        key, fields = self.key, deepcopy(self.form)
        self.save_button.setEnabled(False); self.save_state.setText('Saving…')
        if not self.save_runner.start(lambda: (key, fields, self.backend.tagging_save(key, fields))):
            self.save_button.setEnabled(True)

    def saved(self, result, error):
        self.save_button.setEnabled(True)
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit(); return
            self.save_state.setText('Not saved'); self.show_banner(f'Not saved: {error}'); return
        key, fields, answer = result
        self.drafts.pop(key, None)
        if key == self.key:
            self.saved_form = deepcopy(fields)
        self.save_state.setText('Saved')
        rows = self.model.rows
        index = self.model.row_of(key)
        following = [r['key'] for r in rows[index + 1:] if r['key'] != key] if index >= 0 else []
        target = following[0] if following else answer.get('next_key') or key
        self.load_state()
        self.load_queue(select=target)

    def move(self, delta):
        rows = self.model.rows
        if not rows:
            return
        index = self.model.row_of(self.key)
        target = max(0, min(len(rows) - 1, (index if index >= 0 else -delta) + delta))
        self.open_key(rows[target]['key'])

    def teach(self):
        if self.key:
            key = self.key
            self.save_state.setText('Asking the teacher…')
            self.side_runner.start(lambda: ('teach', key, self.backend.tagging_teach(key)))

    def export(self):
        self.export_button.setEnabled(False)
        self.side_runner.start(lambda: ('export', None, self.backend.tagging_export()))

    def side_done(self, result, error):
        self.export_button.setEnabled(True)
        if error:
            if isinstance(error, AuthError):
                self.session_expired.emit(); return
            self.show_banner(str(error)); self.save_state.setText(''); return
        kind, key, answer = result
        if kind == 'export':
            c = answer['counts']
            self.show_banner(f"Exported {c['training']} training rows and {c['eval']} eval rows "
                             f"({c['studio']} tagged here, {c['migrated']} old tags; {c['contradicted']} contradicted "
                             f"and {c['untagged']} untagged left out) to {answer['training_path'].rsplit(chr(92), 2)[0]}")
        elif kind == 'teach' and key == self.key:
            self.save_state.setText(''); self.open_key(key)

    def help(self):
        dialog = QDialog(self); dialog.setWindowTitle('Tagging keys'); dialog.setObjectName('commandPalette')
        grid = QGridLayout(dialog); grid.setContentsMargins(24, 20, 24, 20); grid.setHorizontalSpacing(24)
        grid.addWidget(label('Keyboard', 'section'), 0, 0, 1, 2)
        for row, (keys, text) in enumerate(HELP, 1):
            grid.addWidget(label(keys, 'eyebrow'), row, 0); grid.addWidget(label(text), row, 1)
        grid.addWidget(label('Order: contradictions first (alerts and suspicious-vs-escalation on top), then clips to '
                             'check, then untagged. Old tags count as done unless something contradicts them.', 'muted', True),
                       len(HELP) + 1, 0, 1, 2)
        dialog.show()
        return dialog

    def hideEvent(self, event):
        self.player.pause(); super().hideEvent(event)
