import json
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QPlainTextEdit, QApplication
from .event_logic import decision, ai_status, VERDICTS
from .formatting import local_time
from .widgets.common import label, button


class TextDisclosure(QWidget):
    def __init__(self, title, text=''):
        super().__init__()
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(6)
        self.toggle = button('›  '+title); self.toggle.setCheckable(True)
        self.toggle.toggled.connect(lambda checked: self.content.setVisible(checked)); layout.addWidget(self.toggle)
        self.content = QWidget(); box = QVBoxLayout(self.content); box.setContentsMargins(0, 0, 0, 0)
        self.text = QPlainTextEdit(); self.text.setReadOnly(True); self.text.setPlainText(text)
        self.text.setStyleSheet('font-family: Consolas; font-size: 9pt;'); self.text.setFixedHeight(144)
        box.addWidget(self.text); box.addWidget(button('Copy', lambda: QApplication.clipboard().setText(self.text.toPlainText()), 'link'), alignment=Qt.AlignmentFlag.AlignRight)
        self.content.hide(); layout.addWidget(self.content)


class AiRecord(QScrollArea):
    def __init__(self, role):
        super().__init__()
        self.role = role
        self.setWidgetResizable(True); self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(320)
        self.body = QWidget(); self.body.setObjectName('detailBody'); self.setWidget(self.body)
        self.layout = QVBoxLayout(self.body); self.layout.setContentsMargins(20, 18, 20, 20); self.layout.setSpacing(12)
        self.frames, self.raw_answers = {}, {}
        self.dispatch_label = None

    def set_event(self, event, zone):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.frames, self.raw_answers = {}, {}
        self.dispatch_label = None
        add = self.layout.addWidget
        add(label('AI RECORD', 'eyebrow'))
        add(label(decision(event.alert_command), 'section', True))
        add(label(event.alert_reason or 'No alert reason was saved.', 'muted', True))
        add(label('Summary', 'eyebrow'))
        add(label(event.summary or 'No summary was saved.', '', True))
        if not event.ai_runs:
            add(label(ai_status(event.completeness.ai), 'muted', True))
        for run in event.ai_runs:
            add(label(ai_status(run.status, run.model), 'badge' if run.status == 'real' else 'error' if run.status == 'failed' else 'muted', True))
            add(label(f'Model  {run.model or "Not run"}  ·  Prompt  {run.prompt_version or "Not recorded"}', 'muted', True))
            if run.input_frame_artifact_ids:
                add(label('Input frames', 'eyebrow'))
                row_widget = QWidget(); row = QHBoxLayout(row_widget); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(8)
                for i, aid in enumerate(run.input_frame_artifact_ids):
                    tile = label(f'Frame {i+1}\nLoading…', 'muted'); tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    tile.setFixedSize(96, 60); tile.setToolTip(f'Input frame {i+1}'); row.addWidget(tile); self.frames[aid] = tile
                row.addStretch()
                frame_scroll = QScrollArea(); frame_scroll.setWidgetResizable(True)
                frame_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                frame_scroll.setWidget(row_widget); frame_scroll.setFixedHeight(78); add(frame_scroll)
            if run.parsed:
                add(label('Parsed answer', 'eyebrow'))
                for key, value in run.parsed.items():
                    if key == 'alert_command':
                        value = decision(value)
                    elif isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    add(label(f'{key.replace("_", " ").capitalize()}\n{value}', '', True))
            elif run.status in ('failed', 'fallback', 'none'):
                add(label('No parsed model answer is available for this event.', 'muted', True))
            if run.raw_text_artifact_id:
                raw = TextDisclosure('Raw answer', 'Loading saved answer…'); self.raw_answers[run.raw_text_artifact_id] = raw; add(raw)
            add(TextDisclosure('Full prompt', run.prompt or 'No prompt was saved.'))
        if self.role in ('admin', 'support'):
            dispatch = event.dispatch
            sent = 'Sent' if dispatch and dispatch.sent is True else 'Not sent' if dispatch and dispatch.sent is False else 'Not recorded'
            self.dispatch_label = label(f'Delivery  ·  {sent}'+(f' via {dispatch.channel}' if dispatch and dispatch.channel else ''), 'muted', True)
            add(self.dispatch_label)
        add(label('OWNER FEEDBACK', 'eyebrow'))
        if not event.feedback:
            add(label('No owner feedback yet.', 'muted'))
        for feedback in sorted(event.feedback, key=lambda f: f.received_utc):
            add(label(VERDICTS.get(feedback.verdict, feedback.verdict.replace('_', ' ').capitalize()), '', True))
            add(label(local_time(feedback.received_utc, zone), 'muted'))
            if feedback.note:
                add(label(feedback.note, '', True))
            if self.role != 'labeler' and feedback.raw_text:
                add(TextDisclosure('Owner raw text', feedback.raw_text))
        self.layout.addStretch()
        self.verticalScrollBar().setValue(0)

    def set_assets(self, images, answers):
        for aid, tile in self.frames.items():
            pix = QPixmap(); pix.loadFromData(images.get(aid, b''))
            if pix.isNull():
                tile.setText('Frame unavailable')
            else:
                tile.setPixmap(pix.scaled(tile.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        for aid, raw in self.raw_answers.items():
            raw.text.setPlainText(answers.get(aid, 'Saved answer could not be loaded. Reopen the event to retry.'))
