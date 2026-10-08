"""The camera map editor on screen: answers, hand-drawn areas, lines, saving, errors, the setup step."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QLineEdit, QStackedWidget, QWidget

from home_guard_project.box import scene_map as sm
from home_guard_project.box.app import scene_editor as se
from home_guard_project.box.app.scene_backend import DemoSceneBackend, SceneError
from home_guard_project.box.app.scene_strings import st

CAMERA = 'ameer_week_0_1_ch2'          # its owner-facing name is "מצלמה 2" / "Camera 2", never the id


def wait_until(condition, timeout_ms=15000):
    deadline = time.monotonic() + timeout_ms / 1000
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(10)
    return condition()


def texts(widget):
    """Every text a person can read in *widget*: labels, buttons, fields and placeholders."""
    out = [widget.windowTitle()] if widget.isWindow() else []
    for child in widget.findChildren(QLabel):
        out.append(child.text())
    for child in widget.findChildren(QAbstractButton):
        out += [child.text(), child.toolTip(), child.accessibleName()]
    for child in widget.findChildren(QLineEdit):
        out += [child.text(), child.placeholderText()]
    return [t for t in out if t]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.aliases = mock.patch('home_guard_project.box.camera_names._load', return_value={})
        cls.aliases.start()

    @classmethod
    def tearDownClass(cls):
        cls.aliases.stop()

    def editor(self, backend=None, lang='he', **kwargs):
        editor = se.SceneMapEditor(backend or DemoSceneBackend(), CAMERA, lang, **kwargs)
        editor.resize(1300, 720); editor.show()
        self.addCleanup(lambda: (editor.close_jobs(), editor.close(), editor.deleteLater()))
        return editor

    def loaded(self, **kwargs):
        editor = self.editor(**kwargs)
        editor.start()
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        return editor


class EditorTest(Base):
    def test_loading_then_the_numbered_places(self) -> None:
        editor = self.editor()
        editor.start()
        self.assertEqual(editor.page, se.LOADING)
        self.assertFalse(editor.primary.isEnabled())
        self.assertEqual(editor.stage.loading[0], st('loading_title', 'he'))
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        self.assertEqual(sorted(editor.region_rows), [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(editor.stage.labels[n] for n in editor.region_rows))
        self.assertTrue(editor.grid_note.isHidden())
        self.assertEqual(editor.layoutDirection(), Qt.LayoutDirection.RightToLeft)
        self.assertEqual(editor.stage.layoutDirection(), Qt.LayoutDirection.LeftToRight)   # the picture never mirrors

    def test_the_camera_id_is_never_shown(self) -> None:
        for lang, name in (('he', 'מצלמה 2'), ('en', 'Camera 2')):
            editor = self.loaded(lang=lang)
            self.assertEqual(editor.title.text(), name)
            editor.failed(SceneError('box_refused', f"could not get a picture from {CAMERA} right now (is it online?)"))
            shown = texts(editor)
            self.assertTrue(any(name in t for t in shown))
            self.assertFalse([t for t in shown if CAMERA in t])
        dialog = se.SceneMapDialog(DemoSceneBackend(), CAMERA, 'he', start=False)
        self.addCleanup(dialog.deleteLater)
        self.assertNotIn(CAMERA, dialog.windowTitle())
        self.assertIn('מצלמה 2', dialog.windowTitle())

    def test_a_click_on_the_picture_selects_the_place_and_answers_colour_it(self) -> None:
        editor = self.loaded()
        lit = lambda: [n for n, row in editor.region_rows.items() if row.lit]  # noqa: E731
        x, y = editor.stage.labels[4]
        QTest.mouseClick(editor.stage, Qt.MouseButton.LeftButton, pos=editor.stage.to_stage((x, y)).toPoint())
        self.assertEqual(lit(), [4])
        self.assertEqual(editor.stage.selected_region, 4)
        row = editor.region_rows[4]
        row.name.setText('האוטו של אבא'); row.name.textEdited.emit('האוטו של אבא')
        row.choices.buttons['mine'].click()
        self.assertEqual(editor.answers[4], 'mine')
        self.assertTrue(row.choices.buttons['mine'].isChecked())
        self.assertEqual(row.tag.text(), st('choice_mine', 'he'))
        self.assertTrue(wait_until(lambda: lit() == [5]))              # on to the next unanswered place
        editor.region_rows[5].choices.buttons['hide'].click()
        editor.region_rows[6].skip.click()
        scene = editor.build_map()
        self.assertEqual([(a['name'], a['kind'], a['zone']) for a in scene['areas']],
                         [('האוטו של אבא', 'mine', 'car'), ('area 5', 'black', 'other')])
        self.assertIn(st('count_mine', 'he', count=1), editor.count_line.text())
        row.choices.buttons['mine'].click()                             # the same answer again takes it back
        self.assertNotIn(4, editor.answers)

    def test_drawing_an_area_by_hand(self) -> None:
        editor = self.loaded()
        editor.new_area.click()
        self.assertEqual(editor.tabs.index, se.DRAW)
        stage = editor.stage
        for corner in ((.1, .1), (.4, .1), (.4, .4)):
            QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage(corner).toPoint())
        self.assertEqual(len(stage.draft), 3)
        self.assertEqual(editor.draw_status.text(), st('draw_corners', 'he', count=3))
        QTest.mouseClick(stage, Qt.MouseButton.RightButton, pos=stage.to_stage((.4, .4)).toPoint())   # undo
        self.assertEqual(len(stage.draft), 2)
        last = stage.to_stage((.1, .4)).toPoint()
        QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=last)
        QTest.mouseDClick(stage, Qt.MouseButton.LeftButton, pos=last)                    # closes on that corner
        self.assertIsNone(stage.draft)
        self.assertEqual(len(editor.hands), 1)
        self.assertEqual(len(editor.hands[0].points), 3)
        self.assertTrue(editor.hand_rows[0].lit)
        # Drag a corner of the selected area.
        corner = stage.to_stage(editor.hands[0].points[0]).toPoint()
        QTest.mousePress(stage, Qt.MouseButton.LeftButton, pos=corner)
        QTest.mouseMove(stage, stage.to_stage((.05, .05)).toPoint())
        QTest.mouseRelease(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage((.05, .05)).toPoint())
        self.assertAlmostEqual(editor.hands[0].points[0][0], .05, delta=.01)
        self.assertEqual(editor.build_map()['areas'], [])               # no answer yet: not saved
        editor.hand_rows[0].choices.buttons['public'].click()
        area = editor.build_map()['areas'][0]
        self.assertEqual((area['kind'], area['owner']), ('watch_no_alert', 'public'))
        editor.hand_rows[0].delete.click()
        self.assertEqual(editor.hands, [])

    def test_a_boundary_line_toward_our_side(self) -> None:
        editor = self.loaded()
        editor.region_rows[1].choices.buttons['mine'].click()          # the lawn, the lower part of the picture
        editor.tabs.select(se.LINES, emit=True)
        editor.new_line.click()
        stage = editor.stage
        for point in ((.2, .3), (.8, .3)):
            QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage(point).toPoint())
        self.assertEqual(editor.line_status.text(), st('line_step_side', 'he'))
        self.assertTrue(editor.from_areas.isEnabled())
        QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage((.5, .8)).toPoint())    # our side
        self.assertEqual(len(editor.lines), 1)
        line = editor.lines[0]
        self.assertEqual(sm.Line('x', line.a, line.b, line.inward).crossing((.5, .1), (.5, .9)), 'in')
        self.assertGreater(se.inward_vector(line)[1], 0)                # the arrow points down, to the lawn
        editor.line_rows[0].flip.click()
        self.assertEqual(sm.Line('x', line.a, line.b, editor.lines[0].inward).crossing((.5, .1), (.5, .9)), 'out')
        # Our side from the areas: no area of ours next to this line's middle, so it says to tap instead.
        editor.new_line.click()
        for point in ((.2, .3), (.8, .3)):
            QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage(point).toPoint())
        editor.from_areas.click()
        self.assertEqual((len(editor.lines), editor.line_status.text()), (1, st('line_no_areas', 'he')))
        editor.cancel_line.click()
        # Along the lawn's edge, our side comes from the lawn: the same side as the tap.
        editor.new_line.click()
        for point in ((.2, .44), (.8, .44)):
            QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage(point).toPoint())
        editor.from_areas.click()
        self.assertEqual(editor.lines[1].inward, line.inward)
        editor.line_rows[1].name.setText('המעקה'); editor.line_rows[1].name.textEdited.emit('המעקה')
        self.assertEqual(editor.build_map()['lines'][1]['name'], 'המעקה')

    def test_summary_then_save_on_the_box(self) -> None:
        backend = DemoSceneBackend()
        editor = self.loaded(backend=backend)
        editor.region_rows[1].choices.buttons['mine'].click()
        editor.region_rows[2].choices.buttons['neighbour'].click()
        editor.region_rows[5].choices.buttons['hide'].click()
        finished = []
        editor.finished.connect(lambda how, value: finished.append((how, value)))
        editor.primary.click()
        self.assertEqual(editor.page, se.SUMMARY)
        self.assertEqual([editor.tiles[k].number.text() for k in ('mine', 'neighbour', 'public', 'hide', 'lines')],
                         ['1', '1', '0', '1', '0'])
        self.assertFalse(editor.restart_label.isHidden())               # a hidden part: the cameras restart
        editor.back_button.click()
        self.assertEqual(editor.page, se.EDIT)
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertEqual(backend.calls, ['previous', 'propose', 'confirm'])
        self.assertFalse(editor.saved_restart.isHidden())
        self.assertEqual([a['kind'] for a in editor.saved.map['areas']], ['mine', 'watch_no_alert', 'black'])
        editor.primary.click()
        self.assertEqual(finished[0][0], 'saved')

    def test_errors_show_plain_words_and_try_again(self) -> None:
        backend = DemoSceneBackend(fail='propose')
        editor = self.editor(backend=backend)
        editor.start()
        self.assertTrue(wait_until(lambda: editor.page == se.ERROR))
        self.assertEqual(editor.error_label.text(), st('error_box_refused', 'he'))
        self.assertIn('מצלמה 2', editor.error_detail.text())
        self.assertEqual(editor.primary.text(), st('try_again', 'he'))
        editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        backend.fail = 'confirm'
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.ERROR))
        editor.primary.click()                                          # try the save again
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))

    def test_a_boundary_place_asks_which_side_is_ours_when_nothing_says(self) -> None:
        editor = self.loaded()
        editor.select_region(6)
        row = editor.region_rows[6]                                     # the right hedge: tall and thin
        row.name.setText('הגדר'); row.name.textEdited.emit('הגדר')
        row.boundary.click()
        self.assertEqual(editor.answers[6], 'boundary')
        self.assertFalse(row.side.isHidden())
        self.assertEqual(row.side.question.text(), st('side_question', 'he', name='הגדר'))
        self.assertEqual({b.text() for b in row.side.buttons.values()}, {'הצד הימני', 'הצד השמאלי'})
        self.assertTrue(row.lit)                                         # stays on the question
        scene = editor.build_map()
        self.assertEqual([(a['name'], a['kind']) for a in scene['areas']], [('הגדר', 'mine')])
        self.assertEqual(scene['lines'], [])                             # no line until the side is said
        self.assertEqual([sure for _line, sure in editor.stage.walls], [False])
        editor.primary.click()
        self.assertFalse(editor.walls_label.isHidden())
        self.assertIn('הגדר', editor.walls_label.text())
        editor.back_button.click()
        row.side.buttons['left'].click()
        self.assertEqual([(ln['name'], ln['inward']) for ln in editor.build_map()['lines']], [('הגדר', 'left')])
        self.assertTrue(row.side.buttons['left'].isChecked())
        self.assertEqual([sure for _line, sure in editor.stage.walls], [True])
        row.choices.buttons['mine'].click()                              # another answer: no line any more
        self.assertTrue(row.side.isHidden())
        self.assertEqual(editor.build_map()['lines'], [])

    def test_a_boundary_place_takes_our_side_from_the_areas_around_it(self) -> None:
        editor = self.loaded()
        editor.region_rows[2].choices.buttons['mine'].click()            # the house, left of the right hedge
        editor.region_rows[6].boundary.click()
        self.assertTrue(editor.region_rows[6].side.isHidden())          # nothing to ask
        lines = editor.build_map()['lines']
        self.assertEqual(len(lines), 1)
        line = sm.Line('x', lines[0]['a'], lines[0]['b'], lines[0]['inward'])
        self.assertEqual(line.crossing((.95, .5), (.8, .5)), 'in')       # towards the house is inward

    def test_restore_the_previous_map(self) -> None:
        backend = DemoSceneBackend()
        editor = self.loaded(backend=backend)
        self.assertFalse(editor.restore_row.isHidden())
        self.assertFalse(editor.restore_button.isEnabled())             # nothing saved yet: nothing to restore
        self.assertEqual(editor.restore_button.toolTip(), st('restore_none', 'he'))
        editor.region_rows[5].choices.buttons['hide'].click()
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertTrue(editor.restore_row.isHidden())                   # only while editing
        again = self.loaded(backend=backend)                            # the box keeps what the save replaced
        self.assertTrue(again.restore_button.isEnabled())
        self.assertTrue(again.restore_when.text().startswith('מ-'))
        again.restore_button.click()
        self.assertTrue(wait_until(lambda: again.page == se.SAVED))
        self.assertEqual(again.saved_title.text(), st('restored_title', 'he'))
        self.assertEqual(again.saved_hint.text(), 'החזרתי את המפה הקודמת של מצלמה 2. היא פועלת עכשיו.')
        self.assertTrue(again.saved.restored)
        self.assertTrue(again.saved_restart.isVisibleTo(again))          # the hidden area went away
        self.assertEqual(backend.calls[-1], 'restore')

    def test_the_grid_is_explained(self) -> None:
        editor = self.editor(backend=DemoSceneBackend(method='grid'))
        editor.start()
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        self.assertFalse(editor.grid_note.isHidden())
        self.assertEqual(len(editor.region_rows), 12)

    def test_the_cameras_map_now_is_shown_to_edit(self) -> None:
        current = {'camera': CAMERA, 'watched': [[0, 0], [.5, 0], [.5, 1]],
                   'areas': [{'name': 'gate', 'kind': 'black', 'zone': 'gate', 'points': [[.6, .6], [.9, .6], [.9, .9]]}],
                   'lines': [], 'rest': '', 'rest_owner': ''}
        editor = self.loaded(backend=DemoSceneBackend(current=current))
        self.assertEqual([(h.choice, h.name) for h in editor.hands], [('hide', 'gate'), ('mine', sm.WATCHED_NAME)])
        self.assertEqual(editor.hand_rows[1].title.text(), st('watched_name', 'he'))
        editor.primary.click()
        self.assertEqual(editor.rest_label.text(), st('rest_neighbour', 'he'))   # the zone's outside: the neighbour's
        self.assertFalse(editor.restart_label.isHidden())


class SetupStepTest(Base):
    def window(self, cameras):
        from PySide6.QtWidgets import QPushButton
        from home_guard_project.box.app.camera_controls import Camera
        records = [Camera(name, enabled, ok=False) for name, enabled in cameras]
        window = SimpleNamespace(args=SimpleNamespace(demo=True), pages=QStackedWidget(),
                                 wizard_cameras=SimpleNamespace(controls=SimpleNamespace(records=records)),
                                 validation=QLabel(), back=QPushButton(), next=QPushButton(), pages_set=[],
                                 update_step_bar=lambda index: None)
        window.set_page = window.pages_set.append
        window.pages.resize(1300, 720); window.pages.show()
        self.addCleanup(window.pages.deleteLater)
        return window

    def test_skipping_never_blocks_finishing_setup(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SKIPPED, open_scene_step
        from home_guard_project.box.app.setup_pages import Page
        window = self.window([('front_ch1', True), ('back_ch2', True), ('off_ch3', False)])
        step = open_scene_step(window)
        self.assertEqual(step.cameras, ['front_ch1', 'back_ch2'])          # switched-off cameras are not asked
        self.assertTrue(window.next.isHidden())
        self.assertEqual(step.progress.text(), st('setup_eyebrow', 'he', number=1, total=2))
        self.assertTrue(step.editor.back_button.isHidden())               # nothing before the first camera
        step.editor.skip_button.click()                                    # skip camera 1, even while it loads
        self.assertEqual(step.index, 1)
        self.assertEqual(step.editor.title.text(), 'מצלמה 2')
        step.skip_all.click()                                              # finish without maps
        self.assertEqual(window.pages_set, [Page.SUMMARY])
        self.assertEqual(window.scene_states, {'front_ch1': SKIPPED, 'back_ch2': SKIPPED})

    def test_save_and_next_then_back(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SAVED, SceneSetupStep
        step = SceneSetupStep(DemoSceneBackend(), ['front_ch1', 'back_ch2'])
        step.resize(1300, 720); step.show()
        self.addCleanup(lambda: (step.editor.close_jobs(), step.deleteLater()))
        states = []
        step.finished.connect(states.append)
        editor = step.editor
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        self.assertEqual(editor.primary.text(), st('save_next', 'he'))
        editor.region_rows[1].choices.buttons['mine'].click()
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertEqual(editor.primary.text(), st('next_camera', 'he'))
        editor.primary.click()
        self.assertEqual(step.index, 1)
        self.assertEqual(step.dots.states, [SAVED, None])
        self.assertFalse(step.editor.back_button.isHidden())
        step.editor.back_button.click()
        self.assertEqual(step.index, 0)
        step.editor.skip_button.click(); step.editor.skip_button.click()
        self.assertEqual(states, [{'front_ch1': SAVED, 'back_ch2': 'skipped'}])

    def test_no_cameras_go_straight_to_the_summary(self) -> None:
        from home_guard_project.box.app.scene_setup_step import open_scene_step
        from home_guard_project.box.app.setup_pages import Page
        window = self.window([('off_ch3', False)])
        self.assertIsNone(open_scene_step(window))
        self.assertEqual(window.pages_set, [Page.SUMMARY])


class CameraCardTest(Base):
    def test_the_card_button_is_the_map(self) -> None:
        from home_guard_project.box.app.box_controls import BoxControls
        from home_guard_project.box.app.camera_controls import CameraControls
        from home_guard_project.box.app.camera_ui import CameraPage
        page = CameraPage(CameraControls(BoxControls(demo=True), [CAMERA]), lambda: None)
        page.widget.resize(1200, 650); page.widget.show()
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        page.render(page.load_photos())
        button = page.zone_widgets[CAMERA][0]
        self.assertEqual(button.text(), st('map_button'))
        button.click()
        dialog = page.zone_dialog
        self.assertIsInstance(dialog, se.SceneMapDialog)
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.EDIT))
        saved = []
        page.zone_saved = lambda name, points: saved.append((name, points))
        dialog.editor.primary.click(); dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.SAVED))
        dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: page.zone_dialog is None))
        self.assertEqual(saved, [(CAMERA, [])])                           # confirm removed today's zone


if __name__ == '__main__':
    unittest.main()
