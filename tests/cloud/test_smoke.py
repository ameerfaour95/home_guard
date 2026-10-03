from sqlalchemy import create_engine, text


def test_embedded_postgres_select_1(pg_url):
    url = pg_url.replace("postgresql://", "postgresql+psycopg://", 1)
    assert url.startswith("postgresql+psycopg://")
    engine = create_engine(url)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
    engine.dispose()
