"""The camera check shows each box photo on its card (not "photo failed")."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel

from home_guard_project.box.app.camera_ui import CameraPage
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.strings import tr
from home_guard_project.box.app.zone_picture import ZonePicture

NAMES = ('test_ch1', 'test_ch2')


def box_output():
    # Exactly how find_cameras --json snapshots prints on the box (Windows line ends).
    rows = [{'name': n, 'file': 'C:\\Users\\ameer\\hg_snapshots\\' + n + '.jpg', 'ok': True} for n in NAMES]
    return json.dumps({'snapshots': rows}, indent=2).replace('\n', '\r\n') + '\r\n'


class FakeRunner:
    def __init__(self, pictures):
        self.pictures = pictures

    def run(self, args):
        if args[0] == 'scp.exe':
            for picture in self.pictures.glob('*.jpg'):
                (Path(args[-1]) / picture.name).write_bytes(picture.read_bytes())
            return SimpleNamespace(returncode=0, stdout='')
        if 'snapshots' in args[-1]:
            return SimpleNamespace(returncode=0, stdout=box_output())
        if 'zones' in args[-1]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({'cameras': []}))
        return SimpleNamespace(returncode=1, stdout=json.dumps({'error': 'unsupported'}))


class CameraCheckPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        pictures = Path(self.tmp.name)
        for name in NAMES:
            image = QImage(640, 360, QImage.Format.Format_RGB32); image.fill(Qt.GlobalColor.darkGreen)
            self.assertTrue(image.save(str(pictures / (name + '.jpg')), 'JPG'))
        self.controls = RemoteCameras('ameer@100.121.29.9', NAMES, runner=FakeRunner(pictures))
        self.page = CameraPage(self.controls, lambda: None, wizard=True)

    def tearDown(self):
        self.page.widget.close(); self.page.widget.deleteLater()
        self.controls.cleanup(); self.tmp.cleanup()
        self.app.processEvents()

    def test_each_camera_card_shows_its_box_photo(self):
        records = self.page.load_photos()
        self.assertTrue(all(camera.ok for camera in records))
        self.page.render(records)
        content = self.page.scroll.widget()
        failed = [w for w in content.findChildren(QLabel) if w.text() == tr('camera_snapshot_failed')]
        self.assertEqual(failed, [])
        photos = content.findChildren(ZonePicture)
        self.assertEqual(len(photos), len(NAMES))
        self.assertTrue(all(not photo.pix.isNull() for photo in photos))
        self.assertTrue(all(self.page.zone_widgets[name][0].isEnabled() for name in NAMES))


if __name__ == '__main__':
    unittest.main()
