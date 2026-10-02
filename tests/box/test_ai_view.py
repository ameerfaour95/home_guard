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

class DeliveryBannerTests(unittest.TestCase):
    def setUp(self): self.data=json.loads((Path(__file__).parent/'fixtures'/'ai_status_sample.json').read_text())
    def test_training_and_pause_do_not_hide_an_undelivered_real_alert(self):
        from home_guard_project.box.app.ai_view import undelivered_alert
        refusal=undelivered_alert(self.data)
        self.assertIsNotNone(refusal);self.assertIn('not a member',refusal.error)
        paused=copy.deepcopy(self.data['decisions'][0]);paused.update(ts=self.data['updated'],muted=True)
        self.data['decisions'].append(paused)
        self.assertEqual(undelivered_alert(self.data),refusal)
    def test_newer_delivery_clears_banner_and_later_failure_brings_it_back(self):
        from home_guard_project.box.app.ai_view import undelivered_alert
        sent=copy.deepcopy(self.data['decisions'][0]);sent.update(ts=self.data['updated'],sent=True,error='')
        self.data['decisions'].append(sent)
        self.assertIsNone(undelivered_alert(self.data))
        failed=copy.deepcopy(sent);failed.update(ts=sent['ts']+1,sent=False,error='Telegram refused the alert')
        self.data['decisions'].append(failed)
        self.assertEqual(undelivered_alert(self.data).error,'Telegram refused the alert')
        self.assertIsNone(undelivered_alert({}))
