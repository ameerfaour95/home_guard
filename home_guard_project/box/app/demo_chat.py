from .strings import tr

def demo_feed(decisions):
    if not decisions: return []
    base=decisions[-1]['ts']-360
    camera=decisions[-1]['camera']
    rows=[{'ts':base,'who':'box','kind':'alert','text':tr('demo_chat_alert'),'camera':camera,'image':'demo_0.jpg','delivered':True},
          {'ts':base+20,'who':'owner','kind':'message','name':'Ameer','text':tr('demo_chat_owner'),'delivered':True},
          {'ts':base+30,'who':'assistant','kind':'answer','text':tr('demo_chat_answer'),'delivered':True},
          {'ts':base+50,'who':'owner','kind':'message','name':'Maya','text':tr('demo_chat_hebrew'),'delivered':True},
          {'ts':base+65,'who':'owner','kind':'button','name':'Ameer','text':tr('demo_chat_button'),'delivered':True},
          {'ts':base+90,'who':'owner','kind':'message','name':'Maya','text':tr('demo_chat_maya'),'delivered':True},
          {'ts':base+100,'who':'assistant','kind':'answer','text':tr('demo_chat_answer2'),'delivered':True}]
    latest=decisions[-1]
    if not latest.get('false_positive') and not latest.get('muted'):
        rows.append({'ts':latest['ts'],'who':'box','kind':'alert','text':latest['summary'],'camera':latest['camera'],'image':'demo_1.jpg','delivered':latest['sent'],'error':latest['error']})
    return rows
