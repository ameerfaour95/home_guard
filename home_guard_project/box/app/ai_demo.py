"""Synthetic local status for safe demonstrations."""
from .strings import tr

def demo_status(names,now,state):
    data={'updated':now,'cameras':{},'decisions':[]}
    for i,name in enumerate(names):
        data['cameras'][name]={'checked_ts':now if i!=2 else now-20,'ts':now if i==0 else now-30,'objects':[{'label':'person','conf':.71,'box':[.1,.3,.18,.75]},{'label':'car','conf':.92,'box':[.41,.22,.60,.43]}]}
    if state=='ai-stale': data['updated']=now-30
    return data
