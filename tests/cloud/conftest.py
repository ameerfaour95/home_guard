import shutil
import tempfile

import pytest


@pytest.fixture(scope="session")
def pg_url():
    import pgserver

    root = tempfile.mkdtemp(prefix="hgpg_")
    srv = None
    try:
        srv = pgserver.get_server(root, cleanup_mode="stop")
        yield srv.get_uri()  # postgresql://... ; convert driver in db.py
    finally:
        if srv is not None:
            srv.cleanup()
        shutil.rmtree(root, ignore_errors=True)


def _db_url(base: str, database: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(base).set(database=database).render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def pg_template(pg_url):
    """Run Alembic once into a template database; tests clone it."""
    from sqlalchemy import create_engine, text

    from home_guard_project.cloud.manage import run_migrations

    admin = create_engine(pg_url.replace("postgresql://", "postgresql+psycopg://", 1), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text("DROP DATABASE IF EXISTS hg_template"))
        c.execute(text("CREATE DATABASE hg_template"))
    run_migrations(_db_url(pg_url, "hg_template"))
    yield admin
    admin.dispose()


@pytest.fixture()
def db_engine(pg_url, pg_template):
    """Fresh migrated database per test, cloned from the template."""
    import uuid

    from sqlalchemy import text

    from home_guard_project.cloud.db import make_engine

    name = "t_" + uuid.uuid4().hex[:12]
    with pg_template.connect() as c:
        c.execute(text(f"CREATE DATABASE {name} TEMPLATE hg_template"))
    engine = make_engine(_db_url(pg_url, name))
    yield engine
    engine.dispose()
    with pg_template.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))


@pytest.fixture()
def client(db_engine):
    from fastapi.testclient import TestClient

    from home_guard_project.cloud.app import create_app
    from home_guard_project.cloud.settings import Settings

    url = db_engine.url.render_as_string(hide_password=False)
    app = create_app(Settings.for_tests(url), s3=None, init_db=False)
    with TestClient(app) as c:
        yield c
    app.state.engine.dispose()


@pytest.fixture()
def staff_factory(client):
    """staff_factory(role) -> (staff, password, totp_secret, auth_headers)."""
    import itertools
    import secrets

    import pyotp

    from home_guard_project.cloud import auth
    from home_guard_project.cloud.db import session_scope
    from home_guard_project.cloud.models import Staff

    counter = itertools.count(1)

    def make(role="admin"):
        n = next(counter)
        password = secrets.token_urlsafe(12)
        secret = pyotp.random_base32()
        with session_scope(client.app.state.engine) as s:
            staff = Staff(email=f"{role}{n}@example.com", name=f"{role.title()} {n}", role=role,
                          password_hash=auth.hash_password(password), totp_secret=secret)
            s.add(staff)
            s.flush()
            token = auth.make_access_token(staff, client.app.state.settings)
        return staff, password, secret, {"Authorization": f"Bearer {token}"}

    return make
