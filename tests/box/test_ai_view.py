import copy,json,unittest
from pathlib import Path
from home_guard_project.box.app.ai_view import decisions,status_note

class AiViewTests(unittest.TestCase):
    def setUp(self): self.data=json.loads((Path(__file__).parent/'fixtures'/'ai_status_sample.json').read_text())
    def test_newest_first_and_one_outcome(self):
        rows=decisions(self.data)
        self.assertEqual(rows[0].outcome(),('muted','No alert - nothing happening (saved for training)'))
        self.assertEqual(rows[1].outcome(),('error','NOT SENT'))
        self.assertIn('not a member',rows[1].error)
        real=copy.deepcopy(self.data['decisions'][0]);real.update(sent=True,command='[call_owner]',false_positive=True,muted=True)
        self.assertEqual(decisions({'decisions':[real]})[0].outcome(),('ok','Urgent alert sent'))
        real.update(sent=False,false_positive=False)
        self.assertEqual(decisions({'decisions':[real]})[0].outcome(),('warning','Not sent - alerts are paused'))
    def test_honest_states_and_limit(self):
        now=self.data['updated']
        self.assertEqual(status_note(self.data,now,False),'')
        self.assertEqual(status_note(self.data,now+16,False),'The AI status is not updating')
        self.assertEqual(status_note(self.data,now+16,True),'Home Guard is stopped')
        self.assertEqual(decisions({}),[])
        self.assertEqual(len(decisions({'decisions':self.data['decisions']*30})),20)
