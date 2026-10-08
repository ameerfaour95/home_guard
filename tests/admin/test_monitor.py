"""Monitor side of the Admin Center: the customer page layout, camera names, warnings and filters."""
import pytest
from PySide6.QtCore import QPoint
from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.prefs import Preferences
from home_guard_project.admin.shell import AdminWindow


def bottom_of(widget, ancestor):
    return widget.mapTo(ancestor, QPoint(0, widget.height())).y()


def top_of(widget, ancestor):
    return widget.mapTo(ancestor, QPoint(0, 0)).y()


@pytest.mark.parametrize('size', [(1366, 768), (1200, 720)])
def test_customer_page_scrolls_and_the_footer_sits_under_the_table(size, widgets, wait, tmp_path):
    window = AdminWindow(DemoBackend(), demo=True, prefs=Preferences(tmp_path/'prefs.json')); widgets.append(window)
    window.resize(*size); window.show()
    shell = window.shell
    shell.open_customer(1)
    page = shell.customer_page; timeline = page.timeline
    wait(lambda: bool(timeline.model.rows) and not timeline.density_runner.busy)
    tabs_page = page.page
    assert timeline.table.height() >= page.TABLE_MIN_HEIGHT - 2
    assert top_of(timeline.count, tabs_page) >= bottom_of(timeline.table, tabs_page)
    assert top_of(timeline.older, tabs_page) >= bottom_of(timeline.table, tabs_page)
    assert top_of(timeline.table, tabs_page) >= bottom_of(timeline.filter_bar, tabs_page)
    assert timeline.density_scroll.height() <= 24+22*timeline.density_rows+4
    # the footer is reachable: the page scrolls rather than squeezing it into the table
    bar = page.scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    footer = timeline.older.mapTo(page.scroll.viewport(), QPoint(0, timeline.older.height())).y()
    assert footer <= page.scroll.viewport().height()
    assert window.height() == size[1]


def test_camera_names_follow_the_box_rule_never_the_raw_id():
    from home_guard_project.admin import formatting
    from home_guard_project.admin.formatting import camera_name, remember_names
    assert camera_name('ameer_tes2_ch6') == 'Camera 6'
    assert camera_name('ameer_week_0_1/ameer_week_0_1_ch3') == 'Ameer week 0 1 / Camera 3'
    assert camera_name('front_door') == 'Front door' and camera_name('cam-abcdef') == 'cam-abcdef'
    try:
        remember_names({'ameer_week_0_1_ch1': 'כניסה ראשית', 'ameer_week_0_1_ch2': ''})
        assert camera_name('ameer_week_0_1_ch1') == 'כניסה ראשית'
        assert camera_name('ameer_week_0_1_ch2') == 'Camera 2'
        # exact ids only: another house's (or an old site's) ch1 never borrows this name
        assert camera_name('other_house_ch1') == 'Camera 1'
    finally:
        formatting.KNOWN_NAMES.clear()


class RenamedBackend(DemoBackend):
    """Demo customer 1 after a rename: 'Garden' is an old id the box still lists; 'Front door' has an owner name."""
    def cameras(self, customer_id=None):
        from dataclasses import replace
        out = []
        for c in super().cameras(customer_id):
            if c.customer_id == 1 and c.camera == 'Garden':
                c = replace(c, current=False)
            if c.customer_id == 1 and c.camera == 'Front door':
                c = replace(c, name='כניסה ראשית', owner_named=True)
            out.append(c)
        return out


def test_customer_page_lists_current_cameras_and_hides_retired_behind_a_toggle(widgets, wait):
    from home_guard_project.admin import formatting
    from home_guard_project.admin.customer import CustomerScreen
    screen = CustomerScreen(RenamedBackend()); widgets.append(screen)
    screen.resize(1182, 688); screen.show(); screen.open(1)
    timeline = screen.timeline
    try:
        wait(lambda: screen.camera_list is not None and timeline.density_result is not None)
        combo = timeline.filters['camera']
        assert list(timeline.density.rows) == ['Driveway', 'Front door']
        assert [combo.itemText(i) for i in range(combo.count())] == ['All cameras', 'Driveway', 'כניסה ראשית']
        assert timeline.retired_toggle.isVisibleTo(screen)
        timeline.retired_toggle.setChecked(True)
        assert list(timeline.density.rows) == ['Driveway', 'Front door', 'Garden']
        assert combo.itemText(combo.count()-1) == 'Garden  (retired)'
    finally:
        formatting.KNOWN_NAMES.clear()
