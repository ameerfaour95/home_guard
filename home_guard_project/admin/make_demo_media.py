"""Deterministic synthetic fixtures. Run with uv run --group admin python -m
home_guard_project.admin.make_demo_media. Requires ffmpeg on PATH.
No real people, addresses or camera recordings are used.
"""
import copy
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent / 'demo_data'
FPS, WIDTH, HEIGHT, SECONDS = 12, 640, 360, 6


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False)+'\n', encoding='utf-8')


def make_clips():
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Install ffmpeg and add it to PATH to regenerate the demo clips.')
    media = ROOT/'media'
    media.mkdir(exist_ok=True)
    detections = []
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        for variant, kind in enumerate(('person', 'car', 'dog')):
            frames = []
            source = str(Path(temp)/f'{kind}.avi')
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (WIDTH, HEIGHT))
            if not writer.isOpened():
                raise RuntimeError('OpenCV video encoder unavailable')
            for i in range(FPS*SECONDS):
                frame = np.full((HEIGHT, WIDTH, 3), (38, 46, 52), np.uint8)
                # Restrained architectural scene, explicitly a synthetic camera.
                cv2.rectangle(frame, (0, 0), (640, 85), (55, 67, 72), -1)
                cv2.rectangle(frame, (340, 60), (610, 235), (85, 96, 99), -1)
                cv2.rectangle(frame, (440, 105), (502, 235), (37, 46, 49), -1)
                for x in (365, 530):
                    cv2.rectangle(frame, (x, 104), (x+50, 159), (112, 122, 125), -1)
                    cv2.line(frame, (x+25, 104), (x+25, 159), (64, 78, 85), 2)
                cv2.fillPoly(frame, [np.array([[0, 360], [175, 230], [490, 230], [640, 360]])], (70, 77, 79))
                cv2.rectangle(frame, (14, 175), (130, 245), (56, 83, 69), -1)
                x = 100+int(i*4.5)
                y, w, h = ((180, 28, 80), (230, 110, 44), (254, 44, 28))[variant]
                cv2.rectangle(frame, (x, y), (x+w, y+h), ((161, 180, 184), (138, 145, 155), (120, 157, 168))[variant], -1)
                cv2.putText(frame, 'HOME GUARD  /  SYNTHETIC CAMERA', (18, 26), cv2.FONT_HERSHEY_SIMPLEX, .43, (184, 201, 208), 1, cv2.LINE_AA)
                cv2.putText(frame, f'DEMO   +00:{i//FPS:02}   {kind.upper()}', (18, 340), cv2.FONT_HERSHEY_SIMPLEX, .4, (184, 201, 208), 1, cv2.LINE_AA)
                writer.write(frame)
                if i in (12, 36, 60):
                    cv2.imwrite(str(media/f'{kind}-{i}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
                if i % 2 == 0:
                    frames.append(dict(frame_index=i, t_sec=i/FPS, status='ran', boxes=[dict(
                        cls=(0, 2, 16)[variant], label=kind, conf=.93-variant*.04,
                        xyxy=[x/WIDTH, y/HEIGHT, (x+w)/WIDTH, (y+h)/HEIGHT])]))
            writer.release()
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', source, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '28', '-movflags', '+faststart', '-an', str(media/f'{kind}.mp4')], check=True)
            detections.append(frames)
    assert sum(p.stat().st_size for p in media.glob('*.mp4')) <= 1_000_000
    return detections


def main():
    frames = make_clips()
    customers = json.loads((ROOT/'customers.json').read_text(encoding='utf-8'))
    now = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    items = []
    for i in range(96):
        eid = 101+i
        customer = customers[0 if i % 6 < 4 else (1 if i % 6 == 4 else 2)]
        start = now-timedelta(minutes=5+i*17)
        kind = ('alert', 'false_positive', 'trigger', 'paused', 'owner_feedback', 'random')[i % 6]
        status = ('real', 'failed', 'fallback', 'real', 'none', 'real')[i % 6]
        obj = ('person', 'car', 'dog')[i % 3]
        boxes = ('sampled', 'captured', 'recomputed', 'none')[i % 4]
        summary = ('A person approached the entrance and left a parcel.', 'A vehicle pulled into the driveway.',
                   'An animal crossed the garden.', 'A visitor walked past the front door without stopping.',
                   'A car passed the driveway and continued along the road.', 'A dog walked across the garden toward the trees.')[i % 6]
        verdicts = ['false_alarm'] if i % 5 == 4 else ['true_alert'] if i % 7 == 0 else []
        base = dict(id=eid, site=customer['devices'][0]['site'], customer_id=customer['id'], customer_name=customer['name'],
                    camera=('Front door', 'Driveway', 'Garden')[i % 3], kind=kind, timezone=customer['timezone'],
                    start_utc=start.isoformat(), end_utc=(start+timedelta(seconds=SECONDS)).isoformat(),
                    summary=summary, label=obj, alert_command='[send_message]' if kind == 'alert' else '[none]',
                    detected=[obj], owner_verdicts=verdicts,
                    completeness=dict(video=True, boxes=boxes, ai=status, owner_feedback=bool(verdicts), expired=False, copies=['production', 'training']),
                    reviewed=i % 5 == 0, flagged=i % 11 == 0, thumbnail_url=f'media/{obj}-36.jpg')
        items.append(base)
        artifacts = []
        def artifact(n, role, filename):
            aid = eid*10+n
            path = ROOT/'media'/filename
            artifacts.append(dict(id=aid, role=role, s3_key=f'admin_cache/demo/{filename}', bytes=path.stat().st_size,
                                  available=True, provenance='captured', detail={}))
            return aid
        artifact(0, 'original_video', f'{obj}.mp4')
        if i == 20:
            base['completeness']['expired'] = True
            base['completeness']['video'] = False
            artifacts[0]['available'] = False
        if i == 21:
            base['thumbnail_url'] = None
        input_ids = [artifact(n+1, 'input_frame', f'{obj}-{idx}.jpg') for n, idx in enumerate((12, 36, 60))] if status == 'real' else []
        rawid = None
        parsed = dict(summary=summary, alert_command=base['alert_command'], confidence=.93) if status == 'real' else None
        if status == 'real':
            (ROOT/'media'/f'answer-{eid}.txt').write_text(json.dumps(parsed, indent=2), encoding='utf-8')
            rawid = artifact(4, 'raw_answer', f'answer-{eid}.txt')
        detail = dict(base, clip_start_local=None, duration_sec=float(SECONDS), fps=float(FPS), frame_size=[WIDTH, HEIGHT],
                      alert_reason='A person entered the entrance zone during security hours.' if kind == 'alert' else 'No alert was requested for this event.',
                      dispatch=dict(channel='telegram', sent=kind == 'alert', detail={'delivered': kind == 'alert'}),
                      ai_runs=[dict(id=eid, purpose='guard', status=status, model='gpt-4o' if status in ('real','failed') else None,
                                    prompt_version='guard-v3', prompt='Review the supplied camera frames. Describe the activity and return summary, alert_command and confidence. Alert only for a person entering the entrance zone.',
                                    parsed=parsed, raw_text_artifact_id=rawid, input_frame_artifact_ids=input_ids)],
                      feedback=[dict(id=eid, verdict=v, action='keep', note='Expected delivery.' if v == 'true_alert' else 'Our own car returning home.', raw_text='Owner reply from Telegram.', source='telegram', received_utc=(start+timedelta(minutes=2)).isoformat()) for v in verdicts],
                      artifacts=artifacts, raw_meta=dict(schema_version=1, synthetic=True, event_id=eid, teacher={'model': 'gpt-4o', 'status': status}))
        write(f'event_{eid}.json', detail)
        write(f'detections_{eid}.json', dict(provenance=boxes, model='synthetic-yolo-demo' if boxes != 'none' else None, frames=frames[i % 3] if boxes != 'none' else []))
    write('events.json', dict(items=items, next_cursor=None))
    print(f'Generated {len(items)} events; MP4 total: {sum(p.stat().st_size for p in (ROOT/"media").glob("*.mp4")):,} bytes')


if __name__ == '__main__':
    main()
