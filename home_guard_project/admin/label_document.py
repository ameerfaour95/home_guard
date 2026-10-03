"""Undoable annotation document; all interpolation uses the shared contract."""
from copy import deepcopy
from uuid import uuid4
from PySide6.QtCore import QObject, Signal
from home_guard_project.fleet_contract.tracks import box_at, boxes_at, frame_time, validate_tracks
from .models import Track, Keyframe, AnnotationIn

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'bird', 'cat', 'dog')


class LabelDocument(QObject):
    changed = Signal()
    position_changed = Signal()

    def __init__(self, annotation, duration=None, parent=None):
        super().__init__(parent)
        self.annotation = deepcopy(annotation)
        self.fps = annotation.fps or 25.
        self.frame_count = annotation.frame_count or max(1, round((duration or 1)*self.fps))
        self.duration = duration or self.frame_count / self.fps
        self.frame_size = annotation.frame_size or [640, 360]
        self.frame, self.t_sec = 0, 0.
        self.timestamps = {k.frame: k.t_sec for t in annotation.tracks for k in t.keyframes}
        self.selected, self.current_class = None, CLASSES[0]
        self.tracks = deepcopy(annotation.tracks)
        self.description, self.drop_clip, self.needs_review = annotation.description, annotation.drop_clip, annotation.needs_review
        self.history = [self.snapshot()]
        self.history_index = 0
        self.saved_snapshot = self.snapshot()

    def snapshot(self):
        return deepcopy((self.tracks, self.description, self.drop_clip, self.needs_review))

    @property
    def dirty(self):
        return self.snapshot() != self.saved_snapshot

    @property
    def track(self):
        return next((t for t in self.tracks if t.track_id == self.selected), None)

    def time_for(self, frame):
        return self.timestamps.get(frame, frame_time(frame, self.fps))

    def seek(self, frame, t_sec=None):
        self.frame = max(0, min(self.frame_count-1, frame))
        self.t_sec = self.time_for(self.frame) if t_sec is None else t_sec
        if t_sec is not None:
            self.timestamps[self.frame] = t_sec
        self.position_changed.emit()

    def visible_boxes(self):
        return boxes_at(self.tracks, self.t_sec)

    def checkpoint(self):
        state = self.snapshot()
        if state != self.history[self.history_index]:
            self.history = self.history[:self.history_index+1] + [state]
            self.history_index += 1
            self.changed.emit()

    def undo(self):
        self.restore(self.history_index-1)

    def redo(self):
        self.restore(self.history_index+1)

    def restore(self, index):
        if 0 <= index < len(self.history):
            self.history_index = index
            self.tracks, self.description, self.drop_clip, self.needs_review = deepcopy(self.history[index])
            self.changed.emit()

    def set_text(self, text, drop, review):
        self.description, self.drop_clip, self.needs_review = text, drop, review
        self.checkpoint()

    def keyframe(self, track=None):
        track = track or self.track
        return next((k for k in track.keyframes if k.frame == self.frame), None) if track else None

    def put_box(self, xyxy, track=None, enabled=True):
        """Draw, move and resize share clamping, native-pixel minimum and timestamp rules."""
        x1, y1, x2, y2 = xyxy
        x1, x2 = sorted((max(0., min(1., x1)), max(0., min(1., x2))))
        y1, y2 = sorted((max(0., min(1., y1)), max(0., min(1., y2))))
        if (x2-x1)*self.frame_size[0] < 4-1e-6 or (y2-y1)*self.frame_size[1] < 4-1e-6:
            return False
        if track is None:
            track = Track(uuid4().hex[:8], self.current_class, [])
            self.tracks.append(track)
        k = self.keyframe(track)
        if k:
            k.xyxy, k.t_sec, k.enabled = [x1, y1, x2, y2], self.t_sec, enabled
        else:
            track.keyframes.append(Keyframe(self.frame, self.t_sec, [x1, y1, x2, y2], enabled))
            track.keyframes.sort(key=lambda k: k.t_sec)
        track.source, self.selected = 'human', track.track_id
        self.checkpoint()
        return True

    def toggle_keyframe(self):
        if self.keyframe():
            self.delete_keyframe()
        elif self.track:
            box = box_at(self.track, self.t_sec)
            if box:
                self.put_box(box, self.track)

    def set_enabled(self, enabled=None):
        tr = self.track
        if not tr:
            return
        k = self.keyframe()
        if not k:
            previous = next((k for k in reversed(tr.keyframes) if k.t_sec <= self.t_sec), None)
            if not previous:
                return
            box = box_at(tr, self.t_sec) or previous.xyxy
            # Insert a boundary at the current frame, including when currently hidden.
            k = Keyframe(self.frame, self.t_sec, list(box), previous.enabled)
            tr.keyframes.append(k); tr.keyframes.sort(key=lambda k: k.t_sec)
        k.enabled = not k.enabled if enabled is None else enabled
        tr.source = 'human'
        self.checkpoint()

    def copy_next(self):
        if self.track and self.frame < self.frame_count-1:
            box = box_at(self.track, self.t_sec)
            if box:
                self.seek(self.frame+1)
                self.put_box(box, self.track)

    def delete_keyframe(self):
        if self.keyframe():
            self.track.keyframes.remove(self.keyframe())
            self.track.source = 'human'
            if not self.track.keyframes:
                self.tracks.remove(self.track); self.selected = None
            self.checkpoint()

    def delete_track(self):
        if self.track:
            self.tracks.remove(self.track); self.selected = None; self.checkpoint()

    def change_class(self, name):
        self.current_class = name
        if self.track:
            self.track.label, self.track.source = name, 'human'
            box = box_at(self.track, self.t_sec)
            if box:
                self.put_box(box, self.track)
            else:
                self.checkpoint()

    def accept_all(self):
        for tr in self.tracks:
            tr.source = 'human'
        self.checkpoint()

    def move_keyframe(self, track_id, old_frame, new_frame):
        tr = next(t for t in self.tracks if t.track_id == track_id)
        if any(k.frame == new_frame for k in tr.keyframes) or not 0 <= new_frame < self.frame_count:
            return
        k = next(k for k in tr.keyframes if k.frame == old_frame)
        k.frame, k.t_sec, tr.source = new_frame, self.time_for(new_frame), 'human'
        tr.keyframes.sort(key=lambda k: k.t_sec)
        self.checkpoint()

    def errors(self):
        return validate_tracks(self.tracks, self.duration)

    def request(self, status='edited'):
        return AnnotationIn(self.annotation.version, deepcopy(self.tracks), self.description,
                            self.drop_clip, self.needs_review, status)
