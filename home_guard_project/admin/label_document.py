"""Undoable annotation document; all interpolation uses the shared contract."""
from copy import deepcopy
from uuid import uuid4
from PySide6.QtCore import QObject, Signal
from home_guard_project.fleet_contract.tracks import box_at, boxes_at, frame_time, validate_tracks
from .models import Track, Keyframe, AnnotationIn

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'bird', 'cat', 'dog')
# The box's own entity names (box/entities.py): people P1, P2 and moving vehicles CAR1, numbered per kind.
KIND_PREFIX = {'person': 'P', 'car': 'CAR', 'motorcycle': 'CAR', 'bus': 'CAR', 'truck': 'CAR'}
EPS = 1e-6


def track_names(tracks):
    """track id -> "person P1", "truck CAR2", "dog #1": people and vehicles numbered per kind like the box names
    them, other classes per class; in order of first appearance in the clip. The internal id stays in the data (and
    in tooltips) only."""
    names, counts = {}, {}
    first = lambda t: min((k.t_sec for k in t.keyframes), default=float('inf'))  # noqa: E731
    for tr in sorted(tracks, key=lambda t: (first(t), t.track_id)):
        prefix = KIND_PREFIX.get(tr.label)
        group = prefix or tr.label
        counts[group] = counts.get(group, 0) + 1
        names[tr.track_id] = f'{tr.label} {prefix}{counts[group]}' if prefix else f'{tr.label} #{counts[group]}'
    return names


def visible_spans(track):
    """[(start, end)] times the track's box is shown: from each enabled keyframe to the next one (a box held after
    the last keyframe counts only at that keyframe)."""
    kfs = track.keyframes
    return [(k.t_sec, kfs[i+1].t_sec if i+1 < len(kfs) else k.t_sec + EPS) for i, k in enumerate(kfs) if k.enabled]


def overlap_in_time(a, b):
    return any(s0 < t1 - EPS and t0 < s1 - EPS for s0, s1 in visible_spans(a) for t0, t1 in visible_spans(b))


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
        self.hidden = set()   # track ids hidden from view (H): display only, never saved
        self.preload_source = getattr(annotation, 'preload_source', None)  # 'tracker' / 'yolo': who drew the unchecked boxes
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

    def display_names(self):
        return track_names(self.tracks)

    def display_name(self, track):
        return self.display_names().get(track.track_id, track.label)

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
        """Remove the selected box's whole track at once (Undo brings it back); the next save leaves it out."""
        if self.track:
            self.hidden.discard(self.selected)
            self.tracks.remove(self.track); self.selected = None; self.checkpoint()
            return True
        return False

    def split(self):
        """Cut the selected track at the current frame: it ends here (hidden from this frame) and a new track with
        the same class carries on from this frame's box. Returns the new track, or None."""
        tr = self.track
        box = box_at(tr, self.t_sec) if tr else None
        before = [k for k in tr.keyframes if k.t_sec < self.t_sec - EPS] if tr else []
        if box is None or not before:
            return None
        after = [k for k in tr.keyframes if k.t_sec > self.t_sec + EPS]
        new = Track(uuid4().hex[:8], tr.label, [Keyframe(self.frame, self.t_sec, list(box), True)] + after, 'human')
        tr.keyframes = before + [Keyframe(self.frame, self.t_sec, list(box), False)]
        tr.source = 'human'
        self.tracks.append(new); self.selected = new.track_id
        self.checkpoint()
        return new

    def merge_candidates(self, track=None):
        """Tracks the selected one can merge with: the same kind of object (people with people, vehicles with
        vehicles, else the same class), never visible at the same time; nearest in time first."""
        track = track or self.track
        if not track or not track.keyframes:
            return []
        kind = lambda t: KIND_PREFIX.get(t.label, t.label)  # noqa: E731
        def gap(other):
            a0, a1 = track.keyframes[0].t_sec, track.keyframes[-1].t_sec
            b0, b1 = other.keyframes[0].t_sec, other.keyframes[-1].t_sec
            return max(b0 - a1, a0 - b1, 0.)
        return sorted((t for t in self.tracks if t is not track and t.keyframes and kind(t) == kind(track)
                       and not overlap_in_time(t, track)), key=gap)

    def merge(self, other=None):
        """Join the selected track and `other` (default: the nearest candidate) into one: the one that appears
        first keeps its id and class, so P1 stays P1 across an occlusion. Refused (False) when the two are ever
        visible at the same time."""
        track = self.track
        candidates = self.merge_candidates(track)
        other = other if other is not None else (candidates[0] if candidates else None)
        if not track or other is None or other not in candidates:
            return False
        keep, gone = sorted((track, other), key=lambda t: (t.keyframes[0].t_sec, t.track_id))
        by_frame = {}
        for k in keep.keyframes + gone.keyframes:
            if k.frame not in by_frame or (k.enabled and not by_frame[k.frame].enabled):
                by_frame[k.frame] = k
        keep.keyframes = sorted(by_frame.values(), key=lambda k: k.t_sec)
        keep.source = 'human'
        self.tracks.remove(gone); self.hidden.discard(gone.track_id); self.selected = keep.track_id
        self.checkpoint()
        return True

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

    @property
    def unchecked(self):
        """Preloaded detector tracks nobody has edited yet."""
        return [t for t in self.tracks if t.source in ('yolo', 'suggestion')]

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
