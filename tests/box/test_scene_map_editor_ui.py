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

from home_guard_project.box import scene_interview as si   # imported here: the box's side, never first in a worker
from home_guard_project.box import scene_map as sm
from home_guard_project.box.app import scene_editor as se
from home_guard_project.box.app.scene_backend import DemoSceneBackend, SceneError
from home_guard_project.box.app.scene_strings import st

CAMERA = 'ameer_week_0_1_ch2'          # an id: for box commands only, never on the screen
NAMED = {CAMERA: {'he': 'הכניסה', 'en': 'Front door'}}          # the box's name for it


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
        # The box is the only source of camera names: the app never asks camera_names itself.
        refuse = AssertionError('the app must not name cameras itself')
        cls.guards = [mock.patch(f'home_guard_project.box.camera_names.{f}', side_effect=refuse)
                      for f in ('display_name', 'replace_ids', 'family_names')]
        for guard in cls.guards:
            guard.start()

    @classmethod
    def tearDownClass(cls):
        for guard in cls.guards:
            guard.stop()

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
        # A box that gives no name: "מצלמה 2 מתוך 5" by the camera's place in the list, never the id.
        for lang, name in (('he', 'מצלמה 2 מתוך 5'), ('en', 'Camera 2 of 5')):
            editor = self.loaded(lang=lang, position=(2, 5))
            self.assertEqual(editor.title.text(), name)
            editor.failed(SceneError('box_refused', f"could not get a picture from {CAMERA} right now (is it online?)"))
            shown = texts(editor)
            self.assertTrue(any(name in t for t in shown))
            self.assertFalse([t for t in shown if CAMERA in t])
        self.assertEqual(self.editor().title.text(), 'מצלמה')                # no place in a list either
        # A box that names it: its name, from its own answer.
        editor = self.loaded(backend=DemoSceneBackend(names=NAMED), position=(2, 5))
        self.assertEqual(editor.title.text(), 'הכניסה')
        dialog = se.SceneMapDialog(DemoSceneBackend(names=NAMED), CAMERA, 'en', start=False, position=(1, 3))
        self.addCleanup(lambda: (dialog.editor.close_jobs(), dialog.deleteLater()))
        self.assertEqual(dialog.windowTitle(), st('window_title', 'en', camera='Camera 1 of 3'))
        dialog.editor.start()
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.EDIT))
        self.assertEqual(dialog.windowTitle(), st('window_title', 'en', camera='Front door'))
        dialog.editor.primary.click(); dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.SAVED))
        self.assertFalse([t for t in texts(dialog) if CAMERA in t])

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
        self.assertEqual(editor.error_detail.text(), 'could not get a picture from מצלמה right now (is it online?)')
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
        backend = DemoSceneBackend(names=NAMED)
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
        self.assertEqual(again.saved_hint.text(), 'החזרתי את המפה הקודמת של הכניסה. היא פועלת עכשיו.')
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
        self.assertTrue(editor.kept_view)
        self.assertEqual([(h.choice, h.name) for h in editor.kept], [('hide', 'gate'), ('mine', sm.WATCHED_NAME)])
        self.assertEqual(editor.hands, [])
        self.assertEqual(editor.kept_rows[0].title.text(), st('kept_title', 'he', number=1))   # "gate": the box's word
        self.assertEqual(editor.kept_rows[1].title.text(), st('watched_name', 'he'))
        editor.primary.click()
        self.assertEqual(editor.rest_label.text(), st('rest_neighbour', 'he'))   # the zone's outside: the neighbour's
        self.assertFalse(editor.restart_label.isHidden())

    def test_a_camera_with_only_todays_zone_still_opens_on_the_numbered_places(self) -> None:
        editor = self.loaded(backend=DemoSceneBackend(current={'camera': CAMERA, 'watched': [[0, 0], [.5, 0], [.5, 1]],
                                                               'areas': [], 'lines': []}))
        self.assertFalse(editor.kept_view)
        self.assertEqual(editor.tab_pages.currentIndex(), se.REGIONS)
        self.assertEqual(len(editor.stage.regions), 6)
        self.assertEqual([h.name for h in editor.hands], [sm.WATCHED_NAME])
        self.assertFalse(editor.saved_when.isVisibleTo(editor))


class SavedMapTest(Base):
    """Reopening a camera that has a map shows that map, editable; never fresh numbers piled on top of it."""
    def save_a_map(self, backend):
        editor = self.loaded(backend=backend)
        row = editor.region_rows[1]
        row.name.setText('הדשא'); row.name.textEdited.emit('הדשא')
        row.choices.buttons['mine'].click()
        editor.region_rows[2].choices.buttons['neighbour'].click()
        editor.region_rows[5].choices.buttons['hide'].click()
        editor.add_line(se.boundary_toward((.16, .47), (.86, .47), (.5, .75), 'המעקה'))
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        return editor.saved.map

    def shown(self, editor):
        return [(a.choice, a.name) for a in editor.kept]

    def test_save_then_reopen_shows_the_same_map_and_when_it_was_saved(self) -> None:
        from home_guard_project.box.app.scene_model import choice_of
        backend = DemoSceneBackend(names=NAMED)
        saved = self.save_a_map(backend)
        again = self.loaded(backend=backend)
        self.assertTrue(again.kept_view)
        self.assertEqual(self.shown(again), [(choice_of(a), a['name']) for a in saved['areas']])
        self.assertEqual(self.shown(again), [('mine', 'הדשא'), ('neighbour', 'area 2'), ('hide', 'area 5')])
        self.assertEqual([(ln.name, ln.inward) for ln in again.lines],
                         [(ln['name'], ln['inward']) for ln in saved['lines']])
        self.assertEqual(len(again.line_rows), 1)
        self.assertEqual([r.title.text() for r in again.kept_rows], ['הדשא', 'אזור 2', 'אזור 3'])
        self.assertEqual([r.tag.text() for r in again.kept_rows],
                         [st('choice_mine', 'he'), st('choice_neighbour', 'he'), st('choice_hide', 'he')])
        self.assertEqual(again.stage.kept_titles, ['הדשא', 'אזור 2', 'אזור 3'])        # named on the picture
        self.assertEqual(again.stage.mode, 'kept')
        # No numbered places: not on the picture, not in the open tab.
        self.assertEqual(again.stage.regions, ())
        self.assertEqual(again.answers, {})
        self.assertEqual(again.tab_pages.currentIndex(), se.KEPT_PAGE)
        self.assertTrue(again.tabs.buttons[se.REGIONS].text().startswith(st('tab_saved', 'he')))
        self.assertFalse(any(row.isVisibleTo(again) for row in again.region_rows.values()))
        from datetime import datetime
        when = datetime.fromtimestamp(saved['confirmed']).strftime('%d.%m %H:%M')
        self.assertEqual(again.saved_when.text(), st('saved_on', 'he', when=when))
        self.assertTrue(again.saved_when.isVisibleTo(again))
        self.assertTrue(again.saved_when.text().startswith('נשמרה ב-⁨'))           # the time isolated
        self.assertTrue(again.merged_note.isHidden()); self.assertTrue(again.moved_note.isHidden())
        # Clicking an area on the picture opens its row.
        x, y = again.stage.kept_labels[1]
        QTest.mouseClick(again.stage, Qt.MouseButton.LeftButton, pos=again.stage.to_stage((x, y)).toPoint())
        self.assertEqual([i for i, r in enumerate(again.kept_rows) if r.lit], [1])
        self.assertFalse([t for t in texts(again) if CAMERA in t])

    def test_reopen_change_a_kind_and_rename_then_save_has_no_duplicates(self) -> None:
        backend = DemoSceneBackend(names=NAMED)
        saved = self.save_a_map(backend)
        again = self.loaded(backend=backend)
        again.select_kept(1)
        row = again.kept_rows[1]
        row.choices.buttons['public'].click()
        row.name.setText('הרחוב'); row.name.textEdited.emit('הרחוב')
        self.assertEqual(again.stage.kept_titles[1], 'הרחוב')
        again.primary.click()
        self.assertEqual([again.tiles[k].number.text() for k in ('mine', 'neighbour', 'public', 'hide', 'lines')],
                         ['1', '0', '1', '1', '1'])
        again.primary.click()
        self.assertTrue(wait_until(lambda: again.page == se.SAVED))
        box = backend.maps[CAMERA]
        self.assertEqual(len(box['areas']), len(saved['areas']))
        self.assertEqual([(a['kind'], a.get('owner'), a['name']) for a in box['areas']],
                         [('mine', None, 'הדשא'), ('watch_no_alert', 'public', 'הרחוב'), ('black', None, 'area 5')])
        self.assertEqual([a['points'] for a in box['areas']], [a['points'] for a in saved['areas']])
        self.assertEqual(len(box['lines']), 1)
        # Delete one and draw one by hand: exactly those.
        third = self.loaded(backend=backend)
        third.select_kept(2); third.kept_rows[2].delete.click()
        third.draft_closed([[.8, .1], [.95, .1], [.95, .3]])
        third.hand_rows[0].choices.buttons['hide'].click()
        third.primary.click(); third.primary.click()
        self.assertTrue(wait_until(lambda: third.page == se.SAVED))
        self.assertEqual([a['name'] for a in backend.maps[CAMERA]['areas']], ['הדשא', 'הרחוב', 'drawn area 3'])

    def test_start_over_asks_first_then_shows_the_numbered_places(self) -> None:
        backend = DemoSceneBackend(names=NAMED)
        self.save_a_map(backend)
        again = self.loaded(backend=backend)
        calls = list(backend.calls)
        self.assertTrue(again.start_over_card.isHidden())
        again.start_over_button.click()
        self.assertFalse(again.start_over_card.isHidden())
        self.assertEqual(again.start_over_question.text(),
                         'למחוק את המפה של הכניסה ולהתחיל מחדש? המפה השמורה תישאר עד שתשמרו.')
        again.start_over_no.click()                                   # keep it: nothing changed
        self.assertTrue(again.start_over_card.isHidden())
        self.assertTrue(again.kept_view)
        self.assertEqual(len(again.kept), 3)
        self.assertEqual(again.stage.regions, ())
        again.start_over_button.click(); again.start_over_yes.click()
        self.assertFalse(again.kept_view)
        self.assertEqual((again.kept, again.lines, again.hands), ([], [], []))
        self.assertEqual(len(again.stage.regions), 6)                  # the numbered places, from the same answer
        self.assertEqual(again.tab_pages.currentIndex(), se.REGIONS)
        self.assertEqual(again.stage.mode, 'regions')
        self.assertTrue(again.region_rows[1].lit)
        self.assertEqual(again.tabs.buttons[se.REGIONS].text(), st('tab_regions', 'he') + '  ·  0/6')
        self.assertEqual(backend.calls, calls)                         # nothing on the box until Save
        self.assertEqual(len(backend.maps[CAMERA]['areas']), 3)
        self.assertFalse(again.restore_row.isHidden())                 # the previous map can still come back
        self.assertTrue(again.restore_button.isEnabled())
        again.region_rows[4].choices.buttons['mine'].click()
        again.primary.click(); again.primary.click()
        self.assertTrue(wait_until(lambda: again.page == se.SAVED))
        self.assertEqual([a['name'] for a in backend.maps[CAMERA]['areas']], ['area 4'])
        self.assertEqual(backend.maps[CAMERA]['lines'], [])
        last = self.loaded(backend=backend)                            # and the one before can still come back
        last.restore_button.click()
        self.assertTrue(wait_until(lambda: last.page == se.SAVED))
        self.assertEqual(len(backend.maps[CAMERA]['areas']), 3)

    def test_duplicates_on_the_box_are_merged_and_noted(self) -> None:
        def area(i, kind='mine', name=''):
            return {'name': name or f'area {i}', 'kind': kind, 'zone': 'other',
                    'points': [[i / 10, .1], [i / 10 + .05, .1], [i / 10 + .05, .3]]}
        areas = [area(i) for i in range(8)] + [area(i, 'black') for i in range(8)] + [area(3, 'mine', 'the gate')]
        backend = DemoSceneBackend(current={'camera': CAMERA, 'areas': areas, 'lines': [], 'confirmed': 1791640000.})
        editor = self.loaded(backend=backend)
        self.assertEqual(len(editor.kept), 8)
        self.assertEqual(editor.merged, 9)
        self.assertFalse(editor.merged_note.isHidden())
        self.assertEqual(editor.merged_note.text(), st('merged_note', 'he', count=9))
        self.assertEqual(editor.tabs.buttons[se.REGIONS].text(), st('tab_saved', 'he') + '  ·  8')
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertEqual(len(backend.maps[CAMERA]['areas']), 8)          # the next save cleans the box file
        self.assertEqual(sum(a['name'] == 'the gate' for a in backend.maps[CAMERA]['areas']), 1)
        one = self.loaded(backend=DemoSceneBackend(current={'camera': CAMERA, 'areas': areas[:9], 'lines': []}))
        self.assertEqual(one.merged_note.text(), st('merged_note_one', 'he'))

    def test_a_map_saved_under_the_cameras_old_name_is_shown_and_noted(self) -> None:
        backend = DemoSceneBackend(names=NAMED)
        saved = self.save_a_map(backend)
        moved = DemoSceneBackend(names=NAMED, current=dict(saved), from_old_name=True)
        editor = self.loaded(backend=moved)
        self.assertTrue(editor.kept_view)
        self.assertEqual(len(editor.kept), 3)
        self.assertFalse(editor.moved_note.isHidden())
        self.assertEqual(editor.moved_note.text(), 'המפה הועברה מהשם הקודם של המצלמה. שמרו כדי לקבע אותה בשם החדש.')
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertEqual(len(moved.maps[CAMERA]['areas']), 3)
        again = self.loaded(backend=moved)
        self.assertTrue(again.moved_note.isHidden())                  # its own map now
        # The box's answer carries the flag only with a map.
        import base64
        from home_guard_project.box.app import scene_backend as sb
        data = {'camera': CAMERA, 'regions': [], 'picture_b64': base64.b64encode(b'jpg').decode(), 'from_old_name': True}
        self.assertFalse(sb.proposal_from(CAMERA, data).from_old_name)
        self.assertTrue(sb.proposal_from(CAMERA, dict(data, current_map=saved)).from_old_name)

    def test_in_setup_the_saved_map_says_when_it_was_saved(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SceneSetupStep
        backend = DemoSceneBackend(names=NAMED)
        self.save_a_map(backend)
        with mock.patch('home_guard_project.box.app.strings.LANG', 'he'):
            step = SceneSetupStep(backend, [CAMERA])
        step.resize(1300, 720); step.show()
        self.addCleanup(lambda: (step.editor.close_jobs(), step.deleteLater()))
        self.assertTrue(wait_until(lambda: step.editor.page == se.EDIT))
        self.assertTrue(step.editor.kept_view)
        self.assertTrue(step.editor.saved_on_text())
        self.assertEqual(step.editor.kept_when.text(), step.editor.saved_on_text())
        self.assertTrue(step.editor.kept_when.isVisibleTo(step))           # the step's header is the camera's
        self.assertFalse([t for t in texts(step) if CAMERA in t])


class FailedSaveTest(Base):
    """A save the box did not make never shows as saved."""
    def box_replies(self, stdout, returncode):
        from home_guard_project.box.app.scene_backend import SceneBackend
        backend = DemoSceneBackend(names=NAMED)
        box = SceneBackend(runner=lambda line, **kw: SimpleNamespace(stdout=stdout, returncode=returncode), python='py')
        backend.confirm = lambda camera, scene, restart=True: box.confirm(camera, scene, restart)
        return backend

    def test_an_error_or_an_unreadable_reply_shows_the_error_page(self) -> None:
        cases = (('{"error": "the disk is full"}\n', 1, 'error_box_refused'),
                 ('{"error": "the map could not be read"}\n', 0, 'error_box_refused'),
                 ('Traceback (most recent call last):\n  boom\n', 0, 'error_no_answer'),
                 ('Traceback (most recent call last):\n  boom\n', 1, 'error_no_answer'),
                 ('{"camera": "%s", "restart_needed": false}\n' % CAMERA, 0, 'error_bad_answer'))
        for stdout, code, words_key in cases:
            editor = self.loaded(backend=self.box_replies(stdout, code))
            editor.region_rows[1].choices.buttons['mine'].click()
            editor.primary.click(); editor.primary.click()
            self.assertTrue(wait_until(lambda: editor.page in (se.ERROR, se.SAVED)), stdout)
            self.assertEqual(editor.page, se.ERROR, stdout)
            self.assertIsNone(editor.saved)
            self.assertEqual(editor.error_label.text(), st(words_key, 'he'), stdout)
            self.assertEqual(editor.primary.text(), st('try_again', 'he'))

    def test_a_confirm_that_returns_no_map_is_not_saved(self) -> None:
        backend = DemoSceneBackend(names=NAMED)
        backend.confirm = lambda camera, scene, restart=True: None
        editor = self.loaded(backend=backend)
        editor.region_rows[1].choices.buttons['mine'].click()
        finished = []
        editor.finished.connect(lambda how, value: finished.append(how))
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page in (se.ERROR, se.SAVED)))
        self.assertEqual(editor.page, se.ERROR)
        self.assertEqual(editor.error_label.text(), st('error_bad_answer', 'he'))
        del backend.confirm                                             # the box answers this time
        editor.primary.click()                                          # try again
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        self.assertEqual(finished, [])
        self.assertEqual(len(backend.maps[CAMERA]['areas']), 1)


class HebrewApp(Base):
    """The app speaking Hebrew (strings.LANG): the map step follows it, right to left."""
    def setUp(self):
        patcher = mock.patch('home_guard_project.box.app.strings.LANG', 'he')
        patcher.start(); self.addCleanup(patcher.stop)


class SetupStepTest(HebrewApp):
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
        self.assertEqual(step.cameras, ['front_ch1', 'back_ch2', 'off_ch3'])   # switched off: last, greyed
        self.assertTrue(window.next.isHidden())
        self.assertEqual(step.progress.text(), st('setup_eyebrow', 'he', number=1, total=3))
        self.assertTrue(step.editor.back_button.isHidden())               # nothing before the first camera
        step.editor.skip_button.click()                                    # skip camera 1, even while it loads
        self.assertEqual(step.index, 1)
        self.assertEqual(step.editor.title.text(), 'מצלמה 2 מתוך 3')          # the box named neither
        self.assertFalse([t for t in texts(step) if 'front_ch1' in t or 'back_ch2' in t])
        step.skip_all.click()                                              # finish without maps
        self.assertEqual(window.pages_set, [Page.SUMMARY])
        self.assertEqual(window.scene_states, {'front_ch1': SKIPPED, 'back_ch2': SKIPPED, 'off_ch3': SKIPPED})

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

    def test_the_step_lists_cameras_by_the_boxs_names(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SceneSetupStep
        ids = ['ameer_week_0_1_ch3', CAMERA, 'ameer_week_0_1_ch5']
        backend = DemoSceneBackend(names={'ameer_week_0_1_ch3': {'he': 'הגינה', 'en': 'Garden'}, **NAMED})
        step = SceneSetupStep(backend, ids)
        step.resize(1300, 720); step.show()
        self.addCleanup(lambda: (step.editor.close_jobs(), step.deleteLater()))
        self.assertTrue(wait_until(lambda: step.names and step.editor.page == se.EDIT))
        self.assertEqual(step.editor.title.text(), 'הגינה')
        self.assertIn('names', backend.calls)
        step.editor.skip_button.click()
        self.assertEqual(step.editor.title.text(), 'הכניסה')               # named before the box answers propose
        step.editor.skip_button.click()
        self.assertEqual(step.editor.title.text(), 'מצלמה 3 מתוך 3')         # the box has no name for it
        step.editor.failed(SceneError('box_refused', "unknown camera: 'ameer_week_0_1_ch5'"))
        shown = texts(step)
        self.assertFalse([t for t in shown if any(i in t for i in ids)], shown)

    def test_no_cameras_go_straight_to_the_summary(self) -> None:
        from home_guard_project.box.app.scene_setup_step import open_scene_step
        from home_guard_project.box.app.setup_pages import Page
        window = self.window([('off_ch3', False)])
        self.assertIsNone(open_scene_step(window))
        self.assertEqual(window.pages_set, [Page.SUMMARY])


class LanguageAndRoomTest(Base):
    def test_the_map_step_speaks_the_apps_language(self) -> None:
        from home_guard_project.box.app import strings
        from home_guard_project.box.app.scene_setup_step import SceneSetupStep
        self.assertEqual(strings.LANG, 'en')                               # setup and every screen are English
        step = SceneSetupStep(DemoSceneBackend(), ['front_door'])
        self.addCleanup(lambda: (step.editor.close_jobs(), step.deleteLater()))
        self.assertEqual((step.lang, step.editor.lang), ('en', 'en'))
        self.assertEqual(step.layoutDirection(), Qt.LayoutDirection.LeftToRight)
        self.assertEqual(step.editor.primary.text(), 'Save & next')
        with mock.patch.object(strings, 'LANG', 'he'):
            hebrew = SceneSetupStep(DemoSceneBackend(), ['front_door'])
            self.addCleanup(lambda: (hebrew.editor.close_jobs(), hebrew.deleteLater()))
            self.assertEqual(hebrew.layoutDirection(), Qt.LayoutDirection.RightToLeft)
            self.assertEqual(hebrew.editor.primary.text(), 'שמירה והמשך')

    def test_the_map_has_its_own_step_in_setups_bar(self) -> None:
        from home_guard_project.box.app import ui
        from home_guard_project.box.app.setup_pages import Page
        from home_guard_project.box.app.strings import TEXT
        names = TEXT['step_names']
        self.assertEqual(names[ui.MAP_STEP], 'Map')
        self.assertEqual(names[-1], 'Ready')
        fake = SimpleNamespace(step_labels=[QLabel() for _ in names], step_icons=[QLabel() for _ in names])
        self.addCleanup(lambda: [w.deleteLater() for w in fake.step_labels + fake.step_icons])

        def bar(index):
            ui.Window.update_step_bar(fake, index)
            return [(label.text(), not icon.isHidden()) for label, icon in zip(fake.step_labels, fake.step_icons)]
        on_map = bar('map')
        self.assertEqual(on_map[ui.MAP_STEP], ('6. Map', False))           # the current step
        self.assertTrue(all(done for _text, done in on_map[:ui.MAP_STEP]))
        self.assertEqual(on_map[-1], ('7. Ready', False))
        self.assertEqual(bar(Page.CAMERA_CHECK)[Page.CAMERAS], ('5. Cameras', False))
        self.assertTrue(all(done for _text, done in bar(Page.SUMMARY)))

    def test_at_1366x768_the_open_card_fits_without_scrolling(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SceneSetupStep
        for lang in ('en', 'he'):
            with mock.patch('home_guard_project.box.app.strings.LANG', lang):
                step = SceneSetupStep(DemoSceneBackend(), ['front_door', 'garden'])
            from home_guard_project.box.app.theme import stylesheet
            holder = QWidget(); holder.setStyleSheet(stylesheet()); step.setParent(holder)   # the app's look
            holder.resize(1318, 561); step.resize(1318, 561); holder.show()    # the setup card at 1366x768
            self.addCleanup(holder.deleteLater)
            self.addCleanup(lambda step=step: (step.editor.close_jobs(), step.deleteLater()))
            step.show_camera(0, start=False)                                 # as the demo does: no box job
            editor = step.editor
            se.drive(editor, 'wall')
            self.assertTrue(wait_until(lambda: editor.compact), lang)       # laid out: a short screen
            row = editor.region_rows[6]
            self.assertTrue(wait_until(lambda: not row.side.isHidden()))
            QTest.qWait(400)                                                 # the card settles, then scrolls into view
            view = editor.region_scroll.viewport()
            top = row.mapTo(view, row.rect().topLeft()).y()
            bottom = row.side.mapTo(view, row.side.rect().bottomLeft()).y()
            self.assertGreaterEqual(top, 0, lang)                           # the whole open card, header to the
            self.assertLessEqual(bottom, view.height(), lang)               # side question, is in view


class CameraCardTest(HebrewApp):
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
        self.assertEqual(dialog.editor.title.text(), 'מצלמה 1 מתוך 1')        # the card knows only the id
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.EDIT))
        self.assertNotIn(CAMERA, dialog.windowTitle())
        saved = []
        page.zone_saved = lambda name, points: saved.append((name, points))
        dialog.editor.primary.click(); dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.SAVED))
        dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: page.zone_dialog is None))
        self.assertEqual(saved, [(CAMERA, [])])                           # confirm removed today's zone


class BoxConfirm:
    """The real box command line for confirm (``SceneBackend.confirm``: ``confirm --map-b64 ...``), its map
    read back from the line exactly as the box reads it, and the box's answer shaped by the demo backend."""
    def __init__(self, backend):
        from home_guard_project.box.app.scene_backend import SceneBackend
        self.lines, self.maps = [], []
        self.demo = backend

        def runner(line, **kw):
            import json
            self.lines.append(line)
            scene = si.decode_map_b64(line[line.index('--map-b64') + 1])
            self.maps.append(scene)
            saved = DemoSceneBackend.confirm(self.demo, CAMERA, scene)
            return SimpleNamespace(stdout=json.dumps({'camera': CAMERA, 'map': saved.map, 'restart_needed': False}) + '\n',
                                   returncode=0)
        box = SceneBackend(runner=runner, python='py')
        backend.confirm = lambda camera, scene, restart=True: box.confirm(camera, scene, restart)


def ch2_like():
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'scene_map_ch2_duplicates.json')
    with open(path, encoding='utf-8') as f:
        return dict(json.load(f), camera=CAMERA)


class ReopenedMapDetailsTest(Base):
    """The ch2 bugs: duplicates cleaned on the next save, each line's side, what lies beyond the boundary."""
    def test_a_map_with_near_duplicates_saves_only_its_places(self) -> None:
        from home_guard_project.box.app import scene_model as model
        backend = DemoSceneBackend(current=ch2_like())
        box = BoxConfirm(backend)
        editor = self.loaded(backend=backend)
        self.assertEqual(len(editor.kept), 9)
        self.assertEqual(editor.merged_note.text(), st('merged_note', 'he', count=8))
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED), (editor.page, editor.error_detail.text()))
        sent = box.maps[-1]
        self.assertEqual(len(sent['areas']), 9)                          # 17 on the box: 9 places go back
        masks = [model.area_mask(a['points']) for a in sent['areas']]
        self.assertLess(max(model.overlap(a, b) for i, a in enumerate(masks) for b in masks[i + 1:]), model.SAME_PLACE)
        self.assertEqual([a['name'] for a in sent['areas']], [a['name'] for a in ch2_like()['areas'][8:]])
        self.assertEqual(len(sent['lines']), 3)
        self.assertEqual((sent['rest'], sent['rest_owner']), ('watch_no_alert', 'neighbour'))   # kept as it was
        self.assertIn('--map-b64', box.lines[-1])                        # the one path that keeps a backup on the box
        self.assertEqual(len(backend.maps[CAMERA]['areas']), 9)

    def test_flip_side_on_a_reopened_map_saves_the_other_side(self) -> None:
        backend = DemoSceneBackend(current=ch2_like())
        box = BoxConfirm(backend)
        editor = self.loaded(backend=backend)
        self.assertEqual(len(editor.kept_flips), 3)                      # each line, on the saved map's tab
        self.assertFalse(editor.kept_lines.isHidden())
        self.assertEqual(editor.kept_flips[0].text(), 'להפוך צד')
        self.assertEqual(editor.stage.mode, 'kept')
        before = [ln.inward for ln in editor.lines]
        editor.kept_flips[0].click()
        editor.line_rows[2].flip.click()                                # the same button in the Lines tab
        editor.tabs.select(se.REGIONS); editor.tab_changed(se.REGIONS)
        editor.primary.click(); editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        other = {'left': 'right', 'right': 'left'}
        self.assertEqual([ln['inward'] for ln in box.maps[-1]['lines']], [other[before[0]], before[1], other[before[2]]])
        self.assertEqual(box.maps[-1]['lines'][0]['a'], [0.557, 1.0])

    def test_beyond_the_boundary_is_the_neighbours_saves_rest_and_owner(self) -> None:
        backend = DemoSceneBackend(names=NAMED)
        box = BoxConfirm(backend)
        editor = self.loaded(backend=backend)
        editor.region_rows[1].choices.buttons['mine'].click()
        editor.add_line(se.boundary_toward((.16, .47), (.86, .47), (.5, .75), 'המעקה'))
        editor.primary.click()
        self.assertFalse(editor.beyond_card.isHidden())
        self.assertEqual([b.text() for b in editor.beyond_buttons.values()], ['של השכן', 'רחוב', 'לא יודע'])
        self.assertFalse(any(b.isChecked() for b in editor.beyond_buttons.values()))    # nothing said yet
        self.assertNotIn('rest', editor.pending_map)
        editor.beyond_buttons['neighbour'].click()
        self.assertTrue(editor.beyond_buttons['neighbour'].isChecked())
        self.assertEqual(editor.rest_label.text(), st('rest_neighbour', 'he'))
        editor.primary.click()
        self.assertTrue(wait_until(lambda: editor.page == se.SAVED))
        sent = box.maps[-1]
        self.assertEqual({k: sent[k] for k in ('rest', 'rest_owner')}, {'rest': 'watch_no_alert', 'rest_owner': 'neighbour'})
        again = sm.SceneMap.from_dict(CAMERA, sent).to_dict()           # the fields the box already has
        self.assertEqual((again['rest'], again['rest_owner']), ('watch_no_alert', 'neighbour'))
        self.assertEqual((backend.maps[CAMERA]['rest'], backend.maps[CAMERA]['rest_owner']), ('watch_no_alert', 'neighbour'))
        # Reopened: the question shows the camera's answer, and keeps it unless the owner says otherwise.
        reopened = self.loaded(backend=backend)
        reopened.primary.click()
        self.assertTrue(reopened.beyond_buttons['neighbour'].isChecked())
        self.assertEqual((reopened.pending_map['rest'], reopened.pending_map['rest_owner']), ('watch_no_alert', 'neighbour'))
        reopened.beyond_buttons['public'].click()
        self.assertEqual((reopened.pending_map['rest'], reopened.pending_map['rest_owner']), ('watch_no_alert', 'public'))
        reopened.beyond_buttons['unknown'].click()
        self.assertNotIn('rest', reopened.pending_map)                  # not known: watched as today
        self.assertEqual(reopened.rest_label.text(), st('rest_unmapped', 'he'))

    def test_no_boundary_no_question_and_the_rest_stays(self) -> None:
        current = {'camera': CAMERA, 'areas': [{'name': 'x', 'kind': 'mine', 'zone': 'yard',
                                                'points': [[0, 0], [.5, 0], [.5, .5]]}],
                   'lines': [], 'rest': 'watch_no_alert', 'rest_owner': 'public'}
        editor = self.loaded(backend=DemoSceneBackend(current=current))
        editor.primary.click()
        self.assertTrue(editor.beyond_card.isHidden())
        self.assertEqual((editor.pending_map['rest'], editor.pending_map['rest_owner']), ('watch_no_alert', 'public'))
        self.assertTrue(editor.kept_lines.isHidden())


class CameraOffTest(HebrewApp):
    """A camera switched off is greyed, says so, and is mapped only on "map anyway"; it never blocks setup."""
    def test_setup_lists_it_last_greyed_and_never_waits_for_it(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SKIPPED, SceneSetupStep
        backend = DemoSceneBackend(names=NAMED)
        step = SceneSetupStep(backend, ['front_ch1', CAMERA], off={CAMERA})
        step.resize(1300, 720); step.show()
        self.addCleanup(lambda: (step.editor.close_jobs(), step.deleteLater()))
        self.assertTrue(wait_until(lambda: step.editor.page == se.EDIT))
        self.assertEqual(step.dots.off, [False, True])
        step.editor.skip_button.click()
        editor = step.editor
        self.assertEqual(editor.page, se.OFF)
        self.assertEqual(editor.stage.off_text, 'המצלמה כבויה')
        self.assertEqual(editor.primary.text(), 'למפות בכל זאת')
        self.assertFalse(editor.skip_button.isHidden())
        self.assertEqual(backend.calls.count('propose'), 1)              # the box is not asked about it
        self.assertFalse([t for t in texts(step) if CAMERA in t or 'front_ch1' in t])
        editor.primary.click()                                           # map anyway
        self.assertTrue(wait_until(lambda: editor.page == se.EDIT))
        self.assertEqual(editor.stage.off_text, '')
        self.assertEqual(backend.calls.count('propose'), 2)
        states = []
        step.finished.connect(states.append)
        step.skip_all.click()
        self.assertEqual(states, [{'front_ch1': SKIPPED, CAMERA: SKIPPED}])

    def test_the_map_step_takes_switched_off_cameras_too(self) -> None:
        from home_guard_project.box.app.scene_setup_step import SKIPPED, open_scene_step
        from home_guard_project.box.app.setup_pages import Page
        window = SetupStepTest.window(self, [('front_ch1', True), ('off_ch3', False), ('back_ch2', True)])
        step = open_scene_step(window)
        self.assertEqual(step.cameras, ['front_ch1', 'back_ch2', 'off_ch3'])   # the switched-off camera last
        self.assertEqual(step.off, {'off_ch3'})
        step.editor.skip_button.click(); step.editor.skip_button.click()
        self.assertEqual(step.editor.page, se.OFF)
        step.editor.skip_button.click()                                   # skipping it finishes the step
        self.assertEqual(window.pages_set, [Page.SUMMARY])
        self.assertEqual(window.scene_states, {'front_ch1': SKIPPED, 'back_ch2': SKIPPED, 'off_ch3': SKIPPED})

    def test_from_the_camera_card(self) -> None:
        from dataclasses import replace
        from home_guard_project.box.app.box_controls import BoxControls
        from home_guard_project.box.app.camera_controls import CameraControls
        from home_guard_project.box.app.camera_ui import CameraPage
        page = CameraPage(CameraControls(BoxControls(demo=True), [CAMERA]), lambda: None)
        page.widget.resize(1200, 650); page.widget.show()
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        page.render(page.load_photos())
        page.controls.records = [replace(c, enabled=False) for c in page.controls.records]
        page.zone_widgets[CAMERA][0].click()
        dialog = page.zone_dialog
        self.assertEqual(dialog.editor.page, se.OFF)
        self.assertEqual(dialog.editor.primary.text(), 'למפות בכל זאת')
        dialog.editor.primary.click()
        self.assertTrue(wait_until(lambda: dialog.editor.page == se.EDIT))


if __name__ == '__main__':
    unittest.main()
