"""Synthetic local status for safe demonstrations."""
from .strings import tr

def demo_status(names,now,state):
    if state=='live-detections': return recorded_status(names,now)
    data={'updated':now,'cameras':{},'decisions':[]}
    for i,name in enumerate(names):
        data['cameras'][name]={'checked_ts':now if i!=2 else now-20,'ts':now if i==0 else now-30,'objects':[{'label':'person','conf':.71,'box':[.1,.3,.18,.75]},{'label':'car','conf':.92,'box':[.41,.42,.61,.62]}]}
    if state!='ai-empty' and names:
        samples=[(15,'person','demo_ai_person','[send_message]',True,False,False,''),(75,'car','demo_ai_cars','[none]',False,True,False,''),(150,'person','demo_ai_paused','[send_message]',False,False,True,''),(240,'person','demo_ai_urgent','[call_owner]',True,False,False,'')]
        if state=='ai-urgent': samples[0]=(15,'person','demo_ai_urgent','[call_owner]',True,False,False,'')
        if state=='ai-paused': samples[0]=(15,'person','demo_ai_paused','[send_message]',False,False,True,'')
        if state=='ai-training': samples[0]=(15,'car','demo_ai_cars','[none]',False,True,False,'')
        if state=='ai-refused': samples[0]=(15,'person','demo_ai_person','[send_message]',False,False,False,tr('demo_ai_refused'))
        if state=='ai-delivered': samples.append((30,'person','demo_ai_person','[send_message]',False,False,False,tr('demo_ai_refused')))
        for age,label,summary,command,sent,training,muted,error in sorted(samples,key=lambda row:-row[0]):
            data['decisions'].append({'ts':now-age,'camera':names[0 if age==15 else min(1,len(names)-1)],'labels':[label],'summary':tr(summary),'command':command,'sent':sent,'false_positive':training,'muted':muted,'error':error})
    if state in ('paused','ai-conversation','ai-group') and names:
        data['decisions']=[{'ts':now-age,'camera':names[0],'labels':[],'summary':tr('demo_pause_summary'),'command':'[none]','sent':False,'false_positive':False,'muted':True,'error':''} for age in (110,100,90,80,70)]
    if state=='ai-group' and names:
        for decision in data['decisions']: decision['ts']+=65
    data['thinking']={'camera':names[0],'labels':['person'],'ts':now} if state=='ai-thinking' and names else None
    if state=='ai-stale':
        data['updated']=now-30
        for camera in data['cameras'].values():
            camera['checked_ts']-=30
            camera['ts']-=30
        for decision in data['decisions']: decision['ts']-=30
    return data


def recorded_status(names,now):
    """Replay the ai_status.py wire example at wall-clock time; no live capture."""
    import json
    from pathlib import Path
    data=json.loads((Path(__file__).parent/'assets'/'ai_status_recording.json').read_text())
    base=data['updated'];original=data['cameras']['front_door']
    data['updated']=now
    data['cameras']={name:dict(original,ts=now,checked_ts=now) for name in names}
    for decision in data['decisions']:
        decision['ts']+=now-base
        decision['camera']=names[0] if names else 'front_door'
    if data['thinking']:
        data['thinking']['ts']=now
        data['thinking']['camera']=names[0] if names else 'front_door'
    return data
