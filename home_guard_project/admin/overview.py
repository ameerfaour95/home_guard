"""The customer page's first tab: is this house OK, which cameras does it have, and what to do about a warning."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout
from home_guard_project.fleet_contract.camera_names import channel_of
from .fleet_model import SEVERITY
from .formatting import site_name, age, mode_name, humanise
from .theme import verdict_color
from .widgets.common import label

# What staff do about each warning. Camera warnings carry their own next step (fleet_contract.health).
NEXT_STEP = {'no_heartbeat': 'The box has never reported: check it was set up and is online.',
             'heartbeat_old': 'Check the box has power and internet.',
             'engine_down': 'Restart the engine from the box app.',
             'stopped_by_owner': 'The owner paused it: nothing to do unless they ask.',
             'disk_low': 'Free disk space on the box (clips older than 14 days can go).',
             'upload_backlog': "Check the box's internet: clips are waiting to upload."}


def status_sentence(devices, cameras, now):
    """One plain sentence for the header of the Overview."""
    if not devices:
        return 'No box is enrolled for this customer yet.'
    worst = min(devices, key=lambda d: SEVERITY[d.verdict])
    if worst.reasons:
        reason = worst.reasons[0]
        step = '' if '—' in reason.message else NEXT_STEP.get(reason.code, '')
        return f'{site_name(worst.site)}: {reason.message}' + (f'. {step}' if step else '')
    current = [c for c in cameras or () if c.current]
    newest = max((c.newest_clip_utc for c in current if c.newest_clip_utc), default=None)
    count = f'All {len(current)} cameras' if current else 'The box'
    return f'{count} reporting normally' + (f'; newest clip {age(newest, now).lower()}.' if newest else '.')


def old_site(camera):
    """The site part of a retired id ('ameer_tes2_ch6' -> 'Ameer tes2'), so two 'Camera 6' rows can be told apart."""
    channel = channel_of(camera)
    return humanise(camera[:-(len(channel)+3)]) if channel and len(camera) > len(channel)+3 else ''


class CustomerOverview(QWidget):
    def __init__(self, theme='dark'):
        super().__init__()
        self.theme = theme
        self.content = QVBoxLayout(self); self.content.setContentsMargins(0, 12, 0, 12); self.content.setSpacing(10)

    def clear(self):
        def drop(layout):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().hide(); item.widget().deleteLater()
                elif item.layout():
                    drop(item.layout())
        drop(self.content)

    def card(self, title):
        frame = QFrame(); frame.setObjectName('panel')
        box = QVBoxLayout(frame); box.setContentsMargins(16, 12, 16, 12); box.setSpacing(6)
        box.addWidget(label(title, 'eyebrow'))
        self.content.addWidget(frame)
        return box

    def show_customer(self, customer, cameras, now, failed=False):
        """``cameras`` is the house's list from /v1/cameras, or None while it loads (or on an older server)."""
        self.clear()
        self.status = label(status_sentence(customer.devices, cameras, now), 'section', True)
        self.content.addWidget(self.status)
        self.warnings = []
        box = self.card('WARNINGS')
        for device in sorted(customer.devices, key=lambda d: SEVERITY[d.verdict]):
            for reason in (r for r in device.reasons if r.severity != 'healthy'):
                step = '' if '—' in reason.message else NEXT_STEP.get(reason.code, '')
                text = f'{site_name(device.site)}  ·  {reason.message}' + (f'  —  {step}' if step else '')
                row = label(text, '', True); row.setStyleSheet(f'color: {verdict_color(reason.severity, self.theme)};')
                box.addWidget(row); self.warnings.append(text)
        if not self.warnings:
            box.addWidget(label('No warnings. Every box and current camera is reporting.', 'muted', True))
        box = self.card('BOXES')
        for device in customer.devices:
            line = (f'{site_name(device.site)}  ·  {device.verdict.title()}  ·  last heard {age(device.last_seen_utc, now).lower()}'
                    f'  ·  {mode_name(device.mode)}')
            row = label(line, '', True); row.setStyleSheet(f'color: {verdict_color(device.verdict, self.theme)};')
            box.addWidget(row)
        if not customer.devices:
            box.addWidget(label('No boxes enrolled', 'muted'))
        self.camera_rows, self.retired_rows = [], []
        box = self.card('CAMERAS')
        if cameras is None:
            box.addWidget(label('The camera list could not be loaded from this server.' if failed else 'Loading the camera list…', 'muted'))
        else:
            grid = QGridLayout(); grid.setHorizontalSpacing(24); grid.setVerticalSpacing(4); box.addLayout(grid)
            current = [c for c in cameras if c.current]
            for i, camera in enumerate(current):
                stale = camera.newest_clip_utc is None or (now-camera.newest_clip_utc).total_seconds() >= 86400
                name = label(camera.name); grid.addWidget(name, i, 0)
                clip = label('No clip yet' if camera.newest_clip_utc is None else f'last clip {age(camera.newest_clip_utc, now).lower()}',
                             'error' if stale else 'muted')
                grid.addWidget(clip, i, 1)
                grid.addWidget(label(site_name(camera.site), 'muted'), i, 2)
                self.camera_rows.append((camera.name, clip.text()))
            grid.setColumnStretch(3, 1)
            if not current:
                box.addWidget(label('No current cameras known.', 'muted'))
            for camera in (c for c in cameras if not c.enabled):
                row = label(f'{camera.name}  ·  switched off by the owner', 'muted'); box.addWidget(row)
                self.camera_rows.append((camera.name, 'switched off'))
            retired = [c for c in cameras if not c.current and c.enabled]
            if retired:
                box = self.card('RETIRED CAMERAS')
                box.addWidget(label('Ids the box still lists from its 14-day archive (an old site name, a removed camera). '
                                    'They are never warned about.', 'muted', True))
                for camera in retired:
                    where = old_site(camera.camera)
                    text = (f'{camera.name}' + (f'  ·  old site {where}' if where else '') +
                            (f'  ·  last clip {age(camera.newest_clip_utc, now).lower()}' if camera.newest_clip_utc else ''))
                    box.addWidget(label(text, 'muted')); self.retired_rows.append(text)
        self.content.addStretch()
