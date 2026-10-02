import unittest
from home_guard_project.box.app.engine_backend import Event,OutputParser
from home_guard_project.box.app.setup_details import DetailsModel,readable_line
class SetupDetailsTests(unittest.TestCase):
    def test_recognises_camera_failures_before_decoder_noise(self):
        self.assertEqual(readable_line('[rtsp @ 000002a1371976c0] method DESCRIBE failed: 401 Unauthorized')[0],'error')
        self.assertEqual(readable_line('global cap_ffmpeg_impl.hpp:453 Stream timeout triggered after 30090 ms')[0],'warning')
        self.assertEqual(readable_line('WARNING No working RTSP pattern found for [hidden]:554')[0],'warning')
        self.assertIsNone(readable_line('[h264 @ 000002a1371976c0] decoder noise'))
        self.assertIsNone(readable_line('cap_ffmpeg_impl.hpp:453 internal output'))
        self.assertIsNone(readable_line(''))
    def test_recognises_setup_actions_without_jargon(self):
        for line in ('Trying Hikvision/ISAPI pattern: /Streaming/Channels/101','Updating software with git pull','Copied a file to the box','Running a command on the box'):
            role,text=readable_line(line)
            self.assertTrue(text);self.assertNotIn('RTSP',text)
    def test_groups_deduplicates_and_pins_failure_despite_later_lines(self):
        model=DetailsModel();model.feed(Event('step','update','start'));model.feed(Event('detail',text='Updating software'))
        model.feed(Event('step','update','ok','Finished'));model.feed(Event('step','cameras','start'))
        for _ in range(2): model.feed(Event('detail',text='401 Unauthorized'))
        model.feed(Event('step','cameras','fail','4 of 4 cameras refused this login'))
        for _ in range(100): model.feed(Event('detail',text='Trying stream pattern'))
        self.assertEqual(model.failed,'cameras');self.assertEqual(model.groups['cameras'].status,'fail')
        self.assertEqual(len(model.groups['cameras'].messages),3)
        self.assertEqual(len(model.groups['update'].messages),2)
        self.assertIn('401 Unauthorized',model.technical_log())
    def test_support_log_excludes_rescue_and_redacts_input(self):
        parser=OutputParser(('private-marker',));model=DetailsModel()
        model.feed(parser.parse('password=private-marker rtsp://private-marker@box.example/stream'))
        model.feed(parser.parse('@@rescue hotspot private-marker'))
        self.assertNotIn('private-marker',model.technical_log());self.assertNotIn('@@rescue',model.technical_log())

class AddressRedactionTests(unittest.TestCase):
    def test_bare_camera_address_is_hidden_in_support_log(self):
        from home_guard_project.box.app.engine_backend import OutputParser
        model=DetailsModel()
        model.feed(OutputParser().parse('No working RTSP pattern found for 192.0.2.10:554'))
        self.assertNotIn('192.0.2.10',model.technical_log())
        self.assertTrue(model.groups['connect'].messages)

class FullRunDetailsTests(unittest.TestCase):
    def test_each_step_has_its_own_readable_output(self):
        from pathlib import Path
        from home_guard_project.box.app.engine_backend import OutputParser,ENGINE_STEPS
        model=DetailsModel();parser=OutputParser()
        for line in (Path(__file__).parent/'fixtures'/'setup_readable_full_run.txt').read_text().splitlines(): model.feed(parser.parse(line))
        for step in ENGINE_STEPS:
            with self.subTest(step=step): self.assertTrue(model.groups[step].messages)
        network='\n'.join(text for _,text in model.groups['network'].messages)
        self.assertIn('Ethernet mode',network);self.assertIn('Wrote network.json',network)
        cameras='\n'.join(text for _,text in model.groups['cameras'].messages)
        for sentence in ('2 of 4 cameras refused','3 cameras answer on the recorder','Camera 2 sent a picture','Camera 3 answers','was skipped'): self.assertIn(sentence,cameras)
        self.assertNotIn('192.0.2.',cameras);self.assertNotIn('decoder noise',cameras)
    def test_unknown_line_survives_and_noise_does_not(self):
        self.assertEqual(readable_line('[OK] Saved network.json'),('ok','[OK] Saved network.json'))
        self.assertIsNone(readable_line('[decoder @ deadbeef] noise'))
        self.assertIsNone(readable_line('cap_ffmpeg_impl.hpp internals'))
