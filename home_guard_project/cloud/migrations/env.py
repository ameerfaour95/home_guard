from alembic import context
from sqlalchemy import create_engine

from home_guard_project.cloud.models import Base

target_metadata = Base.metadata


def run_migrations_online() -> None:
    url = context.config.attributes["url"]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
