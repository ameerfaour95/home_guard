"""Local Cloud + embedded Postgres + moto S3. Never imported by the desktop exe.

Run with uv run --group cloud --group admin --system-certs python -m
home_guard_project.admin.dev_server. Only synthetic fixture data is seeded.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import mimetypes
import os
from pathlib import Path
import secrets
import uuid

DATA = Path(__file__).parent/'demo_data'
BUCKET = 'homeguard-admin-local'


def seed(engine, s3, credentials):
    from sqlalchemy import select, text, delete
    from sqlalchemy.orm import Session
    from home_guard_project.cloud import auth
    from home_guard_project.cloud.models import (Staff, Customer, Device, Camera, Event, Artifact,
        AiRun, Feedback, ReviewState, Collection, CollectionItem, AuditLog, RawRevision)
    now = datetime.now(timezone.utc)
    read = lambda name: json.loads((DATA/name).read_text(encoding='utf-8'))
    reference = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    shift = now-reference
    def stamp(value):
        return datetime.fromisoformat(value)+shift if value else None
    with Session(engine) as session:
        session.execute(text("SELECT setval(pg_get_serial_sequence('artifacts', 'id'), GREATEST(100000, COALESCE((SELECT max(id) FROM artifacts), 0)))"))
        for role, login in credentials.items():
            staff = session.scalar(select(Staff).where(Staff.email == login['email']))
            if staff is None:
                staff = Staff(email=login['email'], name=f'Dev {role.title()}', role=role)
                session.add(staff)
            staff.password_hash = auth.hash_password(login['password'])
            staff.totp_secret, staff.totp_last_counter = login['totp_secret'], None
        session.flush()
        admin = session.scalar(select(Staff).where(Staff.role == 'admin'))
        for source in read('customers.json'):
            customer = session.get(Customer, source['id']) or Customer(id=source['id'])
            for key in ('name', 'timezone', 'consent_live', 'consent_recordings', 'consent_training', 'notes'):
                setattr(customer, key, source[key])
            session.add(customer)
        session.flush()
        devices = {}
        for index, source in enumerate(read('fleet.json')['devices'], 1):
            device = session.get(Device, index) or Device(id=index)
            device.device_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'homeguard-dev-'+source['site']))
            device.site, device.customer_id = source['site'], source['customer_id']
            device.enrolled_at = now-timedelta(days=90)
            device.last_heartbeat_at = stamp(source['last_seen_utc'])
            device.last_heartbeat = dict(site=source['site'], host=source['host'],
                time_utc=device.last_heartbeat_at.isoformat(), mode=source['mode'],
                collector_running=source['collector_running'], disk_free_gb=source['disk_free_gb'],
                stopped=source['stopped'], newest_clip_utc=stamp(source['newest_clip_utc']).isoformat(), cameras={})
            devices[source['site']] = device
            session.add(device)
        session.flush()
        for source in read('events.json')['items']:
            detail = read(f"event_{source['id']}.json")
            device = devices[detail['site']]
            # Camera names in object keys use the cloud's path-safe vocabulary.
            camera = detail['camera'].lower().replace(' ', '_')
            start = stamp(detail['start_utc'])
            stem = f'{camera}_{int(start.timestamp())}_alert'
            event = session.get(Event, detail['id']) or Event(id=detail['id'])
            for key in ('kind', 'summary', 'label', 'alert_command', 'alert_reason', 'detected',
                        'owner_verdicts', 'dispatch', 'duration_sec', 'fps', 'frame_size'):
                setattr(event, key, detail[key])
            event.device_pk, event.site, event.camera, event.stem = device.id, device.site, camera, stem
            event.start_ts, event.end_ts = start.timestamp(), stamp(detail['end_utc']).timestamp()
            event.day, event.created_at, event.updated_at = start.strftime('%Y-%m-%d'), now, now
            event.completeness = dict(detail['completeness'], boxes='sampled')
            session.add(event)
            camera_row = session.scalar(select(Camera).where(Camera.device_pk == device.id, Camera.name == camera))
            if camera_row is None: session.add(Camera(device_pk=device.id, name=camera))
            heartbeat = dict(device.last_heartbeat)
            latest = max(start.isoformat(), heartbeat['cameras'].get(camera, {}).get('newest_clip_utc', ''))
            heartbeat['cameras'] = dict(heartbeat['cameras'], **{camera: dict(newest_clip_utc=latest, clips_waiting=0)})
            device.last_heartbeat = heartbeat
            session.flush()
            # Event times shift on each local run. Drop only this synthetic
            # event's derived fixture pointers, so a fresh in-memory S3 server
            # cannot be asked for labels from a previous run's object keys.
            session.execute(delete(Artifact).where(Artifact.event_id == event.id,
                Artifact.role.in_(['meta', 'yolo_label'])))
            for art in detail['artifacts']:
                local = DATA/'media'/Path(art['s3_key']).name
                key = f'admin_cache/dev/{event.id}/{local.name}'
                if local.exists():
                    s3.client.put_object(Bucket=BUCKET, Key=key, Body=local.read_bytes(),
                        ContentType=mimetypes.guess_type(local)[0] or 'application/octet-stream')
                artifact = session.get(Artifact, art['id']) or Artifact(id=art['id'])
                artifact.event_id, artifact.s3_key = event.id, key
                artifact.role = 'teacher_frame' if art['role'] == 'input_frame' else art['role']
                artifact.available, artifact.provenance = art['available'] and local.exists(), 'box'
                artifact.bytes, artifact.detail = art['bytes'], art['detail']
                artifact.mime = mimetypes.guess_type(local)[0]
                session.add(artifact)
            if detail.get('thumbnail_url'):
                local = DATA/detail['thumbnail_url']
                key = f'admin_cache/dev/{event.id}/thumbnail.jpg'
                s3.client.put_object(Bucket=BUCKET, Key=key, Body=local.read_bytes(), ContentType='image/jpeg')
                thumbnail = session.scalar(select(Artifact).where(Artifact.s3_key == key)) or Artifact(s3_key=key)
                thumbnail.event_id, thumbnail.role = event.id, 'thumbnail'
                thumbnail.available, thumbnail.mime = True, 'image/jpeg'
                session.add(thumbnail)
            # Cloud currently decodes sampled YOLO labels from applied meta revisions.
            # Use the real parser's format, backed by actual synthetic label objects.
            prefix = f'dataset_{device.site}/'
            samples = []
            for frame in read(f'detections_{event.id}.json')['frames']:
                relative = f'yolo/labels/{camera}/{event.day}/{stem}_f{frame["frame_index"]:04d}.txt'
                text = ''
                for box in frame['boxes']:
                    x1, y1, x2, y2 = box['xyxy']
                    text += f'{box["cls"]} {(x1+x2)/2} {(y1+y2)/2} {x2-x1} {y2-y1}\n'
                key = prefix+relative
                response = s3.client.put_object(Bucket=BUCKET, Key=key, Body=text.encode())
                artifact = session.scalar(select(Artifact).where(Artifact.s3_key == key)) or Artifact(s3_key=key)
                artifact.event_id, artifact.role = event.id, 'yolo_label'
                artifact.etag, artifact.available = response['ETag'].strip('"'), True
                session.add(artifact)
                samples.append(dict(frame_index=frame['frame_index'], label_path=relative,
                                    approx_time_offset_sec=frame['t_sec']))
            key = f'{prefix}meta/{camera}/{event.day}/{stem}.meta.json'
            meta = dict(camera_name=camera, fps_estimated=event.fps, yolo_export=dict(exported_frames=samples))
            artifact = session.scalar(select(Artifact).where(Artifact.s3_key == key)) or Artifact(s3_key=key)
            artifact.event_id, artifact.role, artifact.applied_etag = event.id, 'meta', 'dev-v1'
            session.add(artifact)
            if not session.scalar(select(RawRevision.id).where(RawRevision.s3_key == key)):
                session.add(RawRevision(s3_key=key, etag='dev-v1', fetched_at=now, body=meta))
            session.flush()
            for run in detail['ai_runs']:
                row = session.get(AiRun, run['id']) or AiRun(id=run['id'])
                for key in ('purpose', 'status', 'model', 'prompt_version', 'prompt', 'parsed'):
                    setattr(row, key, run[key])
                row.event_id = event.id
                row.raw_artifact_id, row.input_artifact_ids = run['raw_text_artifact_id'], run['input_frame_artifact_ids']
                session.add(row)
            for feedback in detail['feedback']:
                row = session.get(Feedback, feedback['id']) or Feedback(id=feedback['id'])
                for key in ('verdict', 'action', 'note', 'raw_text', 'source'):
                    setattr(row, key, feedback[key])
                row.device_pk, row.event_id = device.id, event.id
                row.received_at = stamp(feedback['received_utc'])
                row.s3_key = f'admin_cache/dev/feedback/{row.id}.json'
                session.add(row)
            session.merge(ReviewState(event_id=event.id, reviewed=detail['reviewed'], flagged=detail['flagged'], by=admin.id, at=now))
        for source in read('studio_collections.json'):
            session.merge(Collection(id=source['id'], name=source['name'], description=source['description'], created_by=admin.id, created_at=now))
        session.flush()
        for cid, ids in read('studio_members.json').items():
            for eid in ids:
                session.merge(CollectionItem(collection_id=int(cid), event_id=eid, added_by=admin.id, added_at=now))
        session.add(AuditLog(ts=now, staff_id=admin.id, staff_name=admin.name, action='dev_seed',
                            target='synthetic-fixtures', reason='Local integration run', detail={'synthetic': True}))
        from home_guard_project.cloud.redact import backfill
        for device in devices.values():
            backfill(session, device, everything=True)
        session.commit()


@contextmanager
def local_service(*, credentials_path=None):
    # This bundled CPython/OpenSSL build cannot use SSLKEYLOGFILE on Windows.
    # It is a debugging sink, not a trust setting; certificate checks stay on.
    os.environ.pop('SSLKEYLOGFILE', None)
    import boto3
    from botocore.config import Config
    from moto.server import ThreadedMotoServer
    import pgserver
    import pyotp
    from home_guard_project.cloud.app import create_app
    from home_guard_project.cloud.settings import Settings
    from home_guard_project.cloud.s3 import S3
    root = Path(os.environ['LOCALAPPDATA'])/'HomeGuardAdmin'/'devdb'
    root.mkdir(parents=True, exist_ok=True)
    postgres = pgserver.get_server(str(root), cleanup_mode='stop')
    moto = ThreadedMotoServer(ip_address='127.0.0.1', port=8601, verbose=False)
    app = None
    try:
        moto.start()
        client = boto3.client('s3', endpoint_url='http://127.0.0.1:8601', region_name='us-east-1',
            aws_access_key_id='local-dev', aws_secret_access_key='local-dev',
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        client.create_bucket(Bucket=BUCKET)
        s3 = S3(client, BUCKET)
        app = create_app(Settings(db_url=postgres.get_uri(), jwt_secret=secrets.token_urlsafe(48), bucket=BUCKET), s3=s3)
        credentials = {role: dict(email=f'{role}@homeguard.local', password=secrets.token_urlsafe(18),
                                  totp_secret=pyotp.random_base32()) for role in ('admin', 'support', 'labeler')}
        seed(app.state.engine, s3, credentials)
        if credentials_path:
            path = Path(credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(credentials), encoding='utf-8')
        for role, login in credentials.items():
            print(f'DEV {role}: email={login["email"]} password={login["password"]} TOTP secret={login["totp_secret"]}', flush=True)
        yield app, credentials
    finally:
        if app: app.state.engine.dispose()
        moto.stop()
        postgres.cleanup()


def main():
    import logging
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials-file', default='build/dev-login.json')
    args = parser.parse_args()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    with local_service(credentials_path=args.credentials_file) as (app, _):
        uvicorn.run(app, host='127.0.0.1', port=8600, access_log=False)


if __name__ == '__main__':
    main()
