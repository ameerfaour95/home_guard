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
        self.assertEqual(len(model.groups['cameras'].messages),2)
        self.assertEqual(len(model.groups['update'].messages),1)
        self.assertIn('401 Unauthorized',model.technical_log())
    def test_support_log_excludes_rescue_and_redacts_input(self):
        parser=OutputParser(('private-marker',));model=DetailsModel()
        model.feed(parser.parse('password=private-marker rtsp://private-marker@box.example/stream'))
        model.feed(parser.parse('@@rescue hotspot private-marker'))
        self.assertNotIn('private-marker',model.technical_log());self.assertNotIn('@@rescue',model.technical_log())
