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
