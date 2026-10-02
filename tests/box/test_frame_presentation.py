import unittest
from home_guard_project.box.app.status_strip import status_text
from home_guard_project.box.app.theme import PALETTES

def luminance(value):
    rgb=[int(value[i:i+2],16)/255 for i in (1,3,5)]
    rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    return sum(v*w for v,w in zip(rgb,(.2126,.7152,.0722)))
class FramePresentationTests(unittest.TestCase):
    def test_footer_shows_only_time_and_quiet_facts(self):
        self.assertEqual(status_text('2026-10-03 01:30:13',20,937),'Last upload 01:30 \u00b7 20 clips waiting \u00b7 937 GB free')
    def test_body_and_caption_contrast(self):
        for name,palette in PALETTES.items():
            for text in ('text','secondary','muted'):
                for surface in ('bg','surface','raised','bubble'):
                    values=sorted((luminance(palette[text]),luminance(palette[surface])))
                    with self.subTest(theme=name,text=text,surface=surface): self.assertGreaterEqual((values[1]+.05)/(values[0]+.05),4.5)
