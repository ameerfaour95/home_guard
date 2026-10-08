import os
import tempfile
import unittest

from home_guard_project.box.box_identity import box_id


class BoxIdentityTest(unittest.TestCase):
    def test_made_once_and_kept(self):
        path = os.path.join(tempfile.mkdtemp(), "box_id.txt")
        first = box_id(path)
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        self.assertEqual(box_id(path), first)            # the same on every call, across restarts and renames

    def test_not_created_when_asked_not_to(self):
        path = os.path.join(tempfile.mkdtemp(), "box_id.txt")
        self.assertIsNone(box_id(path, create=False))
        self.assertFalse(os.path.exists(path))

    def test_a_damaged_file_is_replaced(self):
        path = os.path.join(tempfile.mkdtemp(), "box_id.txt")
        with open(path, "w") as f:
            f.write("not an id")
        self.assertRegex(box_id(path), r"^[0-9a-f]{32}$")


if __name__ == "__main__":
    unittest.main()
