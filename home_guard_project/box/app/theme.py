"""All visual tokens and application control styles."""
from pathlib import Path
import re
PALETTES={
 'dark':dict(bg='#0c1218',surface='#121c24',raised='#192731',border='#273743',text='#edf4f6',secondary='#b5c8d1',muted='#7e98a6',action='#42d6c3',ok='#7edcb0',error='#f17e86',warning='#e2ba76',bubble='#143330'),
 'light':dict(bg='#edf3f4',surface='#ffffff',raised='#e4eef0',border='#ccdadd',text='#182d37',secondary='#405f6e',muted='#5c7987',action='#087c73',ok='#16784c',error='#b83748',warning='#966917',bubble='#d7ece7')}
ACTION=PALETTES['dark']['action'];OK=PALETTES['dark']['ok'];ERROR=PALETTES['dark']['error'];WARNING=PALETTES['dark']['warning'];MUTED=PALETTES['dark']['muted']
DETECTOR_PERSON=ACTION;DETECTOR_VEHICLE=WARNING

def stylesheet(theme='dark'):
    t=PALETTES[theme]
    check=(Path(__file__).parent/'assets/icons/check-control.svg').as_posix()
    sheet="""
QWidget { background: @bg; color: @text; font-family: 'Segoe UI Variable', 'Segoe UI', 'Inter'; font-size: 16px; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 30px; font-weight: 600; }
QLabel#headline { font-size: 34px; font-weight: 600; }
QLabel#section { font-size: 21px; font-weight: 600; }
QLabel#cameraCaption { color: #edf4f6; padding-left: 16px; font-size: 16px; }
QLabel#cameraDetection { color: #a5bdc8; font-size: 14px; }
QFrame#deliveryBanner { background: @surface; border: 1px solid @error; border-radius: 16px; }
QFrame#deliveryBanner QLabel { color: @error; }
QLabel#house { font-size: 16px; color: @secondary; }
QLabel#muted { color: @muted; font-size: 14px; }
QLabel#accent { color: @action; }
QLabel#ok { color: @ok; }
QLabel#warning { color: @warning; }
QLabel#error { color: @error; }
QFrame#card, QWidget#assistantColumn, QFrame#alertCard { background: @surface; border: 1px solid @border; border-radius: 16px; }
QFrame#assistantBubble { background: @raised; border: 1px solid @border; border-radius: 16px; }
QFrame#familyBubble { background: @bubble; border: 1px solid @border; border-radius: 16px; }
QFrame#alertCard[urgent="true"] { border-left: 3px solid @error; }
QPushButton { background: @action; color: @bg; border: 1px solid @action; border-radius: 8px; padding: 12px 24px; font-weight: 600; min-height: 20px; }
QPushButton#secondary { background: @raised; color: @text; border-color: @border; }
QPushButton#iconButton { background: transparent; color: @text; border-color: @border; padding: 10px; }
QPushButton#stopAction { background: @raised; color: @error; border-color: @error; }
QPushButton:hover { border-color: @text; }
QPushButton:disabled { background: @raised; color: @muted; border-color: @border; }
QPushButton#stopAction:disabled { background: @raised; color: @muted; border-color: @border; }
QPushButton:focus, QCheckBox:focus { border: 2px solid @action; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: @surface; color: @text; border: 1px solid @border; border-radius: 8px; padding: 12px; selection-background-color: @action; min-height: 22px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: @action; }
QComboBox QAbstractItemView { background: @surface; color: @text; border: 1px solid @border; padding: 8px; selection-background-color: @raised; }
QComboBox::down-arrow { image: url(@downArrow); width: 12px; height: 12px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@upArrow); width: 12px; height: 12px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@downArrow); width: 12px; height: 12px; }
QComboBox::drop-down { border: none; width: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { border: none; background: @raised; width: 24px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { border: none; background: @raised; width: 24px; }
QCheckBox { spacing: 12px; padding: 8px 0; background: transparent; }
QCheckBox::indicator { width: 22px; height: 22px; border: 1px solid @border; border-radius: 6px; background: @surface; }
QCheckBox::indicator:checked { background: @action; image: url(@check); border-color: @action; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 6px; background: transparent; margin: 0; }
QScrollBar:horizontal { height: 6px; background: transparent; margin: 0; }
QScrollBar::handle { background: @border; border-radius: 3px; min-height: 24px; min-width: 24px; }
QScrollBar::add-line,QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page,QScrollBar::sub-page { background: transparent; }
QProgressBar { background: @raised; border: none; border-radius: 4px; height: 8px; text-align: center; }
QProgressBar::chunk { background: @action; border-radius: 4px; }
QToolTip { background: @raised; color: @text; border: 1px solid @border; padding: 8px; }
"""
    for name,value in dict(t,check=check,upArrow=(Path(__file__).parent/'assets/icons/chevron-up.svg').as_posix(),downArrow=(Path(__file__).parent/'assets/icons/chevron-down.svg').as_posix()).items(): sheet=sheet.replace('@'+name,value)
    # A pixel-sized QFont has pointSize() == -1. Qt's native/rich-text paths
    # sometimes copy that sentinel into setPointSize(), producing a warning.
    # Keep the same 96-DPI type scale, with positive point sizes throughout.
    return re.sub(r'font-size:\s*(\d+(?:\.\d+)?)px',lambda m:'font-size: '+format(float(m[1])* .75,'g')+'pt',sheet)
