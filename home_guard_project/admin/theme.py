"""Company palette copied from box/app/theme.py; no dependency on the box app."""
import re
import os
from pathlib import Path
from PySide6.QtGui import QFont, QPalette, QColor, QFontDatabase

PALETTES = {
    'dark': dict(bg='#0c1218', surface='#121c24', raised='#192731', border='#273743', text='#edf4f6', secondary='#bec8ce', muted='#a0adb8', action='#42d6c3', ok='#7edcb0', error='#f17e86', warning='#e2ba76', bubble='#143330'),
    'light': dict(bg='#edf3f4', surface='#ffffff', raised='#e4eef0', border='#ccdadd', text='#182d37', secondary='#405f6e', muted='#496775', action='#087c73', ok='#16784c', error='#b83748', warning='#966917', bubble='#d7ece7'),
}


def apply_theme(app, theme='dark'):
    t = PALETTES[theme]
    # Windows' offscreen platform does not discover installed fonts reliably.
    fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('segoeui.ttf', 'segoeuib.ttf', 'seguisb.ttf'):
        if (fonts / name).is_file():
            QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont('Segoe UI', 10))
    palette = QPalette()
    for role, token in [(QPalette.Window, 'bg'), (QPalette.Base, 'surface'),
                        (QPalette.Text, 'text'), (QPalette.WindowText, 'text'),
                        (QPalette.Highlight, 'raised'), (QPalette.HighlightedText, 'text')]:
        palette.setColor(role, QColor(t[token]))
    app.setPalette(palette)
    sheet = '''
QWidget { background: @bg; color: @text; font-family: 'Segoe UI'; font-size: 14px; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 30px; font-weight: 600; }
QLabel#hero { font-size: 40px; font-weight: 600; }
QLabel#section { font-size: 20px; font-weight: 600; }
QLabel#muted { color: @muted; }
QLabel#eyebrow { color: @action; font-size: 12px; font-weight: 600; }
QLabel#error { color: @error; }
QLabel#badge { color: @action; background: @bubble; padding: 5px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; }
QLabel#countBadge { color: @action; background: @bubble; border-radius: 4px; font-size: 11px; }
QFrame#rail, QFrame#topbar, QFrame#card, QFrame#detail { background: @surface; border: 1px solid @border; }
QFrame#card { border-radius: 12px; }
QFrame#rail QWidget, QFrame#topbar QWidget, QFrame#detail QLabel, QFrame#card QLabel { background: transparent; }
QFrame#rail QLabel#countBadge { background: @bubble; }
QLineEdit { background: @surface; border: 1px solid @border; border-radius: 6px; padding: 10px 12px; min-height: 20px; selection-background-color: @bubble; }
QLineEdit:focus { border-color: @action; }
QPushButton { background: @raised; color: @text; border: 1px solid @border; border-radius: 6px; padding: 8px 16px; min-height: 20px; }
QPushButton:hover { background: @border; }
QPushButton:focus { border-color: @action; }
QPushButton:checked { background: @bubble; color: @action; border-color: @action; }
QPushButton:disabled { color: @muted; }
QPushButton#primary { background: @action; color: @bg; border-color: @action; font-weight: 600; }
QPushButton#link { background: transparent; color: @action; border: none; padding: 8px; }
QPushButton#nav { text-align: left; padding: 12px 16px; border-color: transparent; background: transparent; }
QPushButton#nav:checked { background: @bubble; color: @action; }
QPushButton#nav:hover { background: @raised; }
QTableView, QListView { background: @surface; border: 1px solid @border; outline: none; selection-background-color: @raised; }
QTableView::item, QListView::item { border: none; padding: 8px; }
QTableView::item:selected, QListView::item:selected { background: @raised; color: @text; }
QHeaderView::section { background: @surface; color: @muted; border: none; border-bottom: 1px solid @border; padding: 12px 8px; font-size: 12px; }
QScrollBar:vertical { background: @surface; width: 8px; }
QScrollBar:horizontal { background: @surface; height: 8px; }
QScrollBar::handle { background: @border; border-radius: 4px; min-height: 24px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QToolTip { background: @raised; color: @text; border: 1px solid @border; padding: 8px; }
QComboBox { background: @surface; border: 1px solid @border; border-radius: 5px; padding: 7px 8px; min-height: 20px; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView { background: @surface; selection-background-color: @raised; }
QCheckBox { spacing: 9px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid @muted; border-radius: 3px; background: @surface; }
QCheckBox::indicator:checked { background: @action; border: 1px solid @action; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: @muted; padding: 10px 16px; margin-bottom: 8px; border-bottom: 2px solid @border; }
QTabBar::tab:selected { color: @action; border-bottom: 2px solid @action; }
QTabBar::tab:disabled { color: @muted; }
QPlainTextEdit { background: @surface; border: 1px solid @border; padding: 8px; }
QSlider::groove:horizontal { height: 4px; background: @border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @action; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; background: @action; border-radius: 6px; }
QSplitter::handle { background: @border; width: 1px; }
QProgressBar { background: @raised; border: none; height: 4px; }
QProgressBar::chunk { background: @action; }
QScrollArea { border: none; background: transparent; }
QWidget#detailBody { background: @surface; }
QDialog#commandPalette { background: @surface; border: 1px solid @border; border-radius: 12px; }
'''
    for key, value in t.items():
        sheet = sheet.replace('@' + key, value)
    app.setStyleSheet(re.sub(r'font-size: (\d+)px', lambda m: f'font-size: {int(m[1])*.75:g}pt', sheet))


def verdict_color(verdict, theme='dark'):
    return PALETTES[theme][dict(healthy='ok', warning='warning', critical='error', offline='muted', unknown='muted')[verdict]]
