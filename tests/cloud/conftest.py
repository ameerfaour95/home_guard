import shutil
import tempfile

import pytest


@pytest.fixture(scope="session")
def pg_url():
    import pgserver

    root = tempfile.mkdtemp(prefix="hgpg_")
    srv = pgserver.get_server(root, cleanup_mode="stop")
    yield srv.get_uri()  # postgresql://... ; convert driver in db.py
    srv.cleanup()
    shutil.rmtree(root, ignore_errors=True)
