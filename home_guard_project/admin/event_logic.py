"""Pure presentation math. UTC buckets remain unambiguous across DST changes."""
from bisect import bisect_left
from statistics import median
from datetime import timedelta

KINDS = {'alert': 'Alert', 'false_positive': 'Dismissed by AI', 'paused': 'Paused',
         'owner_feedback': 'Owner feedback', 'trigger': 'Collected', 'random': 'Collected', 'unknown': 'Unknown'}
DECISIONS = {'[none]': 'No alert', '[send_message]': 'Message sent', '[call_owner]': 'Call owner'}
VERDICTS = {'real': 'Confirmed', 'false_alarm': 'False alarm', 'real_but_wrong': 'Real, wrong decision'}


def decision(command):
    return DECISIONS.get(command, 'No decision recorded')


def ai_status(status, model=None):
    return {'real': f'Real answer from {model or "an unrecorded model"}',
            'failed': 'AI call failed — no answer', 'fallback': 'Fallback: no model ran',
            'none': 'No AI answer recorded'}.get(status, 'Unknown AI state')


def provenance(status, frames=()):
    if status == 'sampled':
        steps = {b.frame_index-a.frame_index for a, b in zip(frames, frames[1:])}
        return 'Boxes: sampled every 2nd frame' if steps == {2} else 'Boxes: sampled frames'
    return {'captured': 'Boxes: captured', 'recomputed': 'Boxes: recomputed in cloud',
            'none': 'No boxes saved for this clip'}.get(status, 'Unknown boxes state')


def video_rect(width, height, frame_width, frame_height):
    if min(width, height, frame_width, frame_height) <= 0:
        return (0., 0., 0., 0.)
    scale = min(width/frame_width, height/frame_height)
    w, h = frame_width*scale, frame_height*scale
    return ((width-w)/2, (height-h)/2, w, h)


def map_box(xyxy, rect):
    x, y, w, h = rect
    x1, y1, x2, y2 = [max(0., min(1., v)) for v in xyxy]
    return (x+x1*w, y+y1*h, max(0., x2-x1)*w, max(0., y2-y1)*h)


def nearest_frame(frames, position_ms, offset_ms=0, times=None):
    """Positive offset selects later detections; ties choose the earlier frame."""
    if not frames:
        return None
    times = times if times is not None else [f.t_sec for f in frames]
    t = (position_ms+offset_ms)/1000
    i = bisect_left(times, t)
    nearest = frames[0] if i == 0 else frames[-1] if i == len(frames) else frames[i-1] if t-times[i-1] <= times[i]-t else frames[i]
    gaps = [b-a for a, b in zip(times, times[1:]) if b > a]
    tolerance = 1.5 * median(gaps) if gaps else 0
    return nearest if abs(t-nearest.t_sec) <= tolerance else None


def hour_buckets(start, end):
    current = start.replace(minute=0, second=0, microsecond=0)
    result = []
    while current < end:
        result.append(current)
        current += timedelta(hours=1)
    return result


def density(events, start, end, cameras=None):
    hours = hour_buckets(start, end)
    rows = {c: [[0, 0, 0] for _ in hours] for c in sorted(cameras or {e.camera for e in events})}
    for e in events:
        if start <= e.start_utc < end and e.camera in rows:
            i = int((e.start_utc-hours[0]).total_seconds()//3600)
            cell = rows[e.camera][i]
            cell[0] += 1
            cell[1] += e.kind == 'alert'
            cell[2] += 'false_alarm' in e.owner_verdicts
    return hours, rows
