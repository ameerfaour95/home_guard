import json
import pathlib

from .app import create_app
from .settings import Settings


def build() -> dict:
    app = create_app(Settings.for_tests(db_url="sqlite://"), s3=None, init_db=False)
    return app.openapi()


if __name__ == "__main__":
    out = pathlib.Path("docs/admin/openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
