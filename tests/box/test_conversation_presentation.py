import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from home_guard_project.box.app.timeline import Item,QuietGroup,group_quiet,merge_timeline,text_direction
class ConversationPresentationTests(unittest.TestCase):
    def test_quiet_group_ends_at_messages_or_another_camera(self):
        quiet=lambda ts,camera:Item(ts,'assistant','quiet','Nothing happened',camera=camera)
        rows=[quiet(1,'front'),quiet(2,'front'),Item(3,'owner','message','Hello',name='Maya'),quiet(4,'front'),quiet(5,'garden')]
        groups=group_quiet(rows)
        self.assertEqual([len(g.records) if isinstance(g,QuietGroup) else 0 for g in groups],[2,0,1,1])
    def test_recorded_muted_decisions_are_quiet_not_alerts(self):
        data={'decisions':[{'ts':100,'camera':'front','summary':'Alerts are paused; the AI was not asked.','muted':True},{'ts':110,'camera':'front','summary':'Alerts are paused; the AI was not asked.','muted':True}]}
        rows=merge_timeline(data,[])
        self.assertTrue(all(row.kind=='quiet' and row.muted for row in rows))
        self.assertEqual(len(group_quiet(rows)),1)
    def test_direction_uses_first_strong_character(self):
        self.assertEqual(text_direction('12:00 \u05d4\u05d1\u05d9\u05ea.'),'rtl')
        self.assertEqual(text_direction('\u0647\u0644 \u0627\u0644\u0628\u064a\u062a\u061f'),'rtl')
        self.assertEqual(text_direction('Home Guard'),'ltr')
    def test_expansion_and_names_once_per_run(self):
        from PySide6.QtWidgets import QApplication,QLabel,QPushButton
        from PySide6.QtCore import Qt
        from home_guard_project.box.app.ai_activity_ui import AiActivity
        app=QApplication.instance() or QApplication([])
        panel=AiActivity()
        data={'updated':120,'decisions':[{'ts':100+i,'camera':'front','summary':'Alerts are paused; the AI was not asked.','muted':True} for i in range(3)]}
        feed=[{'ts':110+i,'who':'assistant','kind':'answer','text':'\u05d4\u05d1\u05d9\u05ea.'} for i in range(2)]
        panel.render(data,120,False,feed,count=1)
        names=[label.text() for label in panel.scroll.widget().findChildren(QLabel)]
        self.assertEqual(names.count('AI assistant'),1)
        arrow=panel.scroll.widget().findChild(QPushButton);arrow.setChecked(True)
        self.assertEqual(len(panel.expanded_groups),1)
        message=next(label for label in panel.scroll.widget().findChildren(QLabel) if label.text()=='\u05d4\u05d1\u05d9\u05ea.')
        self.assertEqual(message.layoutDirection(),Qt.LayoutDirection.RightToLeft)
        panel.animation.stop();panel.close()
