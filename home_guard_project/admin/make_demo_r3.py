"""Extend the committed Round 2 fixtures with reproducible Round 3 examples."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    root = Path(__file__).parent/'demo_data'
    now = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    def save(name, value):
        (root/name).write_text(json.dumps(value, separators=(',',':'), ensure_ascii=False)+'\n', encoding='utf-8')
    filters = [('owner_false_alarm', 'Owner said false alarm', 'Find decisions the owner corrected.', {'verdict': 'false_alarm'}),
               ('owner_wrong', 'Real activity, wrong decision', 'Review the moments that needed a different response.', {'verdict': 'real_but_wrong'}),
               ('failed_ai', 'AI call failed', 'Recorded footage with an incomplete AI answer.', {'ai': 'failed'}),
               ('fallback_ai', 'Fallback AI', 'Separate fallback records from model answers.', {'ai': 'fallback'}),
               ('paused', 'Paused-camera footage', 'Review footage retained while alerts were paused.', {'kind': 'paused'}),
               ('flagged', 'Flagged for follow-up', 'Resolve the events your team has set aside.', {'flagged': True})]
    save('studio_filters.json', [dict(key=k, title=t, description=d, builtin=True, query=q) for k,t,d,q in filters])
    members = {'1': list(range(101, 113)), '2': [101, 107, 113, 119, 125, 131], '3': [103, 109, 115, 121]}
    save('studio_members.json', members)
    save('studio_collections.json', [dict(id=i, name=n, description=d, event_count=len(members[str(i)]), created_by=a,
          created_utc=(now-timedelta(days=i)).isoformat()) for i,n,d,a in [
          (1, 'Entrance decisions · October', 'Parcels, visitors and passing traffic at household entrances.', 'Maya Cohen'),
          (2, 'Owner corrections', 'False alarms to compare with the next guard prompt.', 'Amir Levy'),
          (3, 'Garden wildlife', 'Animal crossings in low light and shaded gardens.', 'Maya Cohen')]])
    save('studio_exports.json', [dict(id=i, name=n, version=v, state=state, item_count=count,
         s3_prefix=f's3://homeguard-demo/training_exports/{n}/v{v}/' if state in ('ready','partial') else '',
         manifest_url=None, error=error, created_utc=(now-timedelta(hours=i*3)).isoformat(), created_by='Maya Cohen')
         for i,n,v,state,count,error in [(1,'entrance_october',3,'ready',42,None), (2,'owner_corrections',2,'running',24,None),
         (3,'garden_wildlife',1,'queued',16,None), (4,'entrance_september',2,'partial',38,'4 recordings expired before export.'),
         (5,'night_arrivals',1,'failed',0,'Recording access expired. Create a new export to retry.')]])
    actions = [('recording.access','event:101','Review owner feedback'), ('collection.create','collection:1','Entrance review set'),
               ('export.create','export:1','Train the next guard model'), ('event.review','event:107','Review completed'),
               ('artifact.access','artifact:1011','Review recorded evidence')]
    save('audit.json', dict(items=[dict(id=1000-i, ts=(now-timedelta(minutes=i*17)).isoformat(),
         staff=['Maya Cohen','Amir Levy','Dana Shalev'][i%3], action=actions[i%5][0], customer_id=i%3+1,
         device_id=None, target=actions[i%5][1], reason=actions[i%5][2]) for i in range(62)], next_cursor=None))
    events = json.loads((root/'events.json').read_text(encoding='utf-8'))
    catalog = set()
    for i,event in enumerate(events['items']):
        event['timezone'] = 'Asia/Jerusalem'
        event['summary'] = ('A person approached the entrance and left a parcel.', 'A vehicle pulled into the driveway.',
            'An animal crossed the garden.', 'A visitor walked past the front door without stopping.',
            'A car passed the driveway and continued along the road.', 'A dog walked across the garden toward the trees.')[i%6]
        catalog.add((event['customer_id'],event['camera']))
        path = root/f"event_{event['id']}.json"
        detail = json.loads(path.read_text(encoding='utf-8')); detail.update(timezone=event['timezone'],summary=event['summary'])
        for run in detail['ai_runs']:
            if run.get('parsed'):
                run['parsed']['summary'] = event['summary']
                answer = root/'media'/f"answer-{event['id']}.txt"
                answer.write_text(json.dumps(run['parsed'],indent=2),encoding='utf-8')
                for artifact in detail['artifacts']:
                    if artifact['role'] == 'raw_answer': artifact['bytes'] = answer.stat().st_size
        save(path.name, detail)
    save('events.json', events)
    catalog.add((2,'Front side'))
    save('camera_catalog.json', [dict(customer_id=c,camera=n) for c,n in sorted(catalog)])
    counts = [18,24,31,37,40,35,30,27,22,17,12,7,4,3,4,8,16,29,42,48,43,37,31,25]
    save('fleet_activity.json',dict(bucket='hour', timezone='Asia/Jerusalem',
         starts_utc=[(now-timedelta(hours=24-i)).isoformat() for i in range(24)],
         rows=[dict(camera='Fleet', events=counts, alerts=[0,1,0,2,1,0,1,0,2,0,1,0,0,0,0,0,1,2,0,1,1,1,1,0],
                    false_alarms=[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0])]))
    # A real six-tile sprite exercises the contract's optional artifact metadata.
    import cv2
    import numpy as np
    detail = json.loads((root/'event_101.json').read_text(encoding='utf-8'))
    video = next(a for a in detail['artifacts'] if a['role'] in ('rendition','original_video','clip'))
    capture = cv2.VideoCapture(str(root/'media'/Path(video['s3_key']).name))
    tiles = []
    for second in range(6):
        capture.set(cv2.CAP_PROP_POS_MSEC, second*1000); ok, frame = capture.read()
        if ok:
            tiles.append(cv2.resize(frame,(160,90)))
    capture.release()
    cv2.imwrite(str(root/'media'/'filmstrip_101.jpg'), np.hstack(tiles))
    detail['artifacts'] = [a for a in detail['artifacts'] if a['role'] != 'filmstrip']
    detail['artifacts'].append(dict(id=10199, role='filmstrip', s3_key='admin_cache/filmstrip_101.jpg',
         bytes=(root/'media'/'filmstrip_101.jpg').stat().st_size, available=True, provenance='captured',
         detail=dict(fps=1,tile_w=160,tile_h=90,count=len(tiles))))
    save('event_101.json',detail)
    from dataclasses import asdict
    from .demo_backend import DemoBackend
    backend = DemoBackend()
    save('review_count.json',asdict(backend.review_count()))
    density = asdict(backend.density(from_utc=(now-timedelta(hours=24)).isoformat(),to_utc=now.isoformat()))
    density['starts_utc'] = [stamp.isoformat() for stamp in density['starts_utc']]
    save('events_density.json',density)


if __name__ == '__main__':
    main()
