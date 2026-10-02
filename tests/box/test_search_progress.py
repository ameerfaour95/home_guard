import unittest
from home_guard_project.box.app.search_progress import CameraSearch

class SearchProgressTests(unittest.TestCase):
    def test_long_wait_preserves_latest_recognised_sentence(self):
        search=CameraSearch();search.start(10)
        search.feed('INFO [address hidden]: 3 channel(s) answer (Hikvision/unicast): 2, 3, 4')
        self.assertEqual(search.sentence,'3 cameras answer on the recorder.')
        search.feed('[h264 @ deadbeef] decoder noise');search.feed('cap_ffmpeg_impl.hpp internals');search.feed('')
        search.feed('A line kept only in step details')
        self.assertEqual(search.sentence,'3 cameras answer on the recorder.')
        self.assertEqual(search.elapsed(430),420)
        search.feed('WARNING 2 of 4 device(s) refused this login: [address hidden]')
        self.assertEqual(search.sentence,'2 of 4 cameras refused the login.')
        self.assertEqual(search.role,'error')
    def test_finish_and_retry_reset_the_old_message(self):
        search=CameraSearch();search.start(10);search.feed('401 Unauthorized');search.finish()
        sentence=search.sentence;search.feed('INFO Channel 2: OK (2688x1520)')
        self.assertEqual(search.sentence,sentence)
        search.start(500)
        self.assertNotEqual(search.sentence,sentence);self.assertEqual(search.elapsed(500),0)
        self.assertEqual(search.role,'muted');self.assertTrue(search.running)
