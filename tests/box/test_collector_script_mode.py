"""The collector runs as a script (run_collector.sh: python .../data_collection.py).

Then `config` and `zones` are top-level modules, not part of a package, so any
relative import in them fails at startup. These tests import them exactly the
way the script does: in a fresh interpreter whose only extra path entry is the
data_collection directory, from a working directory outside the repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_DC_DIR = Path(__file__).resolve().parents[2] / "home_guard_project" / "data_collection"


def _run(code: str, cwd: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SSLKEYLOGFILE", "AWS_CA_BUNDLE")}
    env["PYTHONPATH"] = str(_DC_DIR)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=300,
    )


class CollectorScriptModeTest(unittest.TestCase):
    def test_config_loads_zones_when_imported_as_a_top_level_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zones_path = os.path.join(tmp, "zones.yaml")
            with open(zones_path, "w", encoding="utf-8") as f:
                f.write("zones:\n  yard: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]\n")
            code = textwrap.dedent(f"""
                import json
                import config
                c = config.load_config(
                    config_path={os.path.join(tmp, 'config.yaml')!r},
                    cameras_path={os.path.join(tmp, 'cameras.yaml')!r},
                    zones_path={zones_path!r},
                )
                print(json.dumps(c.ROI_ZONES))
            """)
            proc = _run(code, tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            zones = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(zones, {"yard": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]})

    def test_config_loads_scene_map_black_areas_next_to_the_zones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zones_path = os.path.join(tmp, "zones.yaml")
            with open(os.path.join(tmp, "scene_maps.yaml"), "w", encoding="utf-8") as f:
                f.write("scene_maps:\n  yard:\n    areas:\n      - {name: w, kind: black, points: "
                        "[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]]}\n      - {name: y, kind: mine, points: "
                        "[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]}\n")
            code = textwrap.dedent(f"""
                import json
                import config
                c = config.load_config(
                    config_path={os.path.join(tmp, 'config.yaml')!r},
                    cameras_path={os.path.join(tmp, 'cameras.yaml')!r},
                    zones_path={zones_path!r},
                )
                print(json.dumps(c.ROI_BLACK))
            """)
            proc = _run(code, tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            black = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(black, {"yard": [[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]]]})

    def test_the_collector_module_imports_the_way_the_script_runs_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run("import data_collection; print('imported')", tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("imported", proc.stdout)


if __name__ == "__main__":
    unittest.main()
