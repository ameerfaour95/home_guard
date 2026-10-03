"""Deterministic Round 3 review captures with decoded video evidence."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import time
from pathlib import Path
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication
from .theme import apply_theme
from .demo_backend import DemoBackend
from .shell import Shell


def main():
    app = QApplication.instance() or QApplication([]); apply_theme(app)
    output = Path('docs/admin/screenshots'); output.mkdir(parents=True,exist_ok=True)
    def settle(predicate=lambda: True,timeout=12):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            app.processEvents()
            if predicate():
                for _ in range(12): app.processEvents()
                return
            time.sleep(.01)
        raise RuntimeError('Screenshot state did not load')
    def capture(window,name):
        settle(); assert window.grab().save(str(output/f'r3-{name}.png'))
        print(name,window.width(),window.height(),flush=True)
    backend = DemoBackend(); shell = Shell(backend,backend.me()); shell.resize(1920,1080); shell.show()
    settle(lambda:shell.fleet.snapshot is not None and bool(shell.fleet.activity.hours))
    capture(shell,'fleet-histogram')
    shell.navigate('Review'); review = shell.review_page
    settle(lambda:bool(review.timeline.model.rows) and not review.timeline.thumbnail_runner.busy)
    review.timeline.filters['reviewed'].setCurrentIndex(0)
    settle(lambda:not review.timeline.runner.busy)
    row = next(i for i,e in enumerate(review.timeline.model.rows) if e.id == 101)
    review.timeline.table.setCurrentIndex(review.timeline.model.index(row,0)); review.open_event(101)
    view = review.event_view
    settle(lambda:view.recording is not None and view.recording.id == 101 and not view.evidence_runner.busy and view.player.player.duration()>0)
    view.player.player.setPosition(3000); view.player.player.play()
    settle(lambda:not view.player.canvas.image.isNull() and view.player.canvas.overlay.position_ms >= 3000)
    view.player.player.pause(); capture(shell,'review-1920')
    shell.resize(1366,768); settle(); assert (shell.width(),shell.height()) == (1366,768)
    capture(shell,'review-1366')
    shell.open_palette(); palette = shell.palette_dialog; palette.search.setText('>')
    capture(palette,'palette-commands'); palette.reject()
    shell.navigate('Studio'); studio = shell.screens['Studio']
    settle(lambda:studio.loaded_once)
    studio.filter_table.setCurrentIndex(studio.filter_model.index(0,0)); settle(lambda:not studio.count_runner.busy)
    capture(shell,'studio-filters')
    studio.tabs.setCurrentIndex(1); studio.collection_table.setCurrentIndex(studio.collection_model.index(0,0)); studio.open_collection()
    settle(lambda:bool(studio.grid.model.items) and not studio.grid.runner.busy)
    capture(shell,'collection-grid')
    studio.open_export(); wizard = studio.wizard; wizard.name.setText('entrance_october')
    capture(wizard,'export-1-dataset'); wizard.advance(); capture(wizard,'export-2-split')
    wizard.advance(); settle(lambda:wizard.preview is not None); wizard.check.setChecked(True)
    capture(wizard,'export-3-consent'); wizard.reject()
    studio.tabs.setCurrentIndex(2); studio.export_table.setCurrentIndex(studio.export_model.index(4,0))
    capture(shell,'export-history')
    shell.navigate('Audit'); audit = shell.screens['Audit']; settle(lambda:audit.loaded_once)
    capture(shell,'audit-table'); audit.table.setCurrentIndex(audit.model.index(0,0)); capture(shell,'audit-drawer')
    label_backend = DemoBackend(role='labeler'); labeler = Shell(label_backend,label_backend.me()); labeler.resize(1366,768); labeler.show()
    settle(lambda:labeler.screens['Studio'].loaded_once); capture(labeler,'labeler-studio')
    shell.close(); labeler.close()
    for _ in range(5):
        QThreadPool.globalInstance().waitForDone(); app.processEvents()
    print('Round 3 screenshots complete',flush=True)


if __name__ == '__main__': main()
