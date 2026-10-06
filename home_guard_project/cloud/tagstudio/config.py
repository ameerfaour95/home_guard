"""Where the studio's local data lives. Each root: a CLI flag, else an environment variable, else the last default.

- ``HOMEGUARD_DATASET_DIR``: the unified dataset (``annotations/clips.jsonl``, ``vlm/clips.jsonl``,
  ``owner_feedback/``). Default: ``home_guard_data/dataset`` (then the older ``home_guard_dataset``) next to the repo.
- ``HOMEGUARD_EVAL_DIR``: the eval results the teacher reads. The folder itself, or one holding ``results/`` or
  ``eval_set/results/``. Default: ``home_guard_eval``.
- ``HOMEGUARD_STUDIO_EXPORT_DIR``: where exports are written. Default: ``studio_exports`` next to the dataset.
- ``HG_TEACHER_BASE_URL`` / ``HG_TEACHER_MODEL`` / ``HG_TEACHER_API_KEY``: an OpenAI-compatible teacher to ask
  on demand (a self-hosted model). Unset: no teacher model is asked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

# The repo is <workspace>/<repo>; the data folders sit next to it. These are the last defaults only.
_WORKSPACE_GUESSES = (Path(__file__).resolve().parents[4], Path.home() / "Ameer")
DATASET_DEFAULTS = ("home_guard_data/dataset", "home_guard_dataset")
EVAL_DEFAULTS = ("home_guard_eval",)


def _first_existing(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _guesses(names: Sequence[str]) -> list:
    return [base / name for base in _WORKSPACE_GUESSES for name in names]


def results_dir(root: Path) -> Path:
    """The eval results folder inside *root* (``eval_set/results``, ``results``) or *root* itself."""
    for sub in ("eval_set/results", "results"):
        if (root / sub).is_dir():
            return root / sub
    return root


@dataclass(frozen=True)
class StudioPaths:
    dataset: Path
    eval_results: Path
    exports: Path
    teacher_base_url: str = ""
    teacher_model: str = ""
    teacher_api_key: str = ""

    @classmethod
    def resolve(cls, env: Optional[Mapping[str, str]] = None, dataset: Optional[str] = None,
                eval_dir: Optional[str] = None, exports: Optional[str] = None) -> "StudioPaths":
        env = os.environ if env is None else env
        ds = Path(dataset or env.get("HOMEGUARD_DATASET_DIR") or _first_existing(_guesses(DATASET_DEFAULTS)))
        ev = Path(eval_dir or env.get("HOMEGUARD_EVAL_DIR") or _first_existing(_guesses(EVAL_DEFAULTS)))
        out = Path(exports or env.get("HOMEGUARD_STUDIO_EXPORT_DIR") or ds.parent / "studio_exports")
        return cls(dataset=ds, eval_results=results_dir(ev), exports=out,
                   teacher_base_url=str(env.get("HG_TEACHER_BASE_URL") or "").strip(),
                   teacher_model=str(env.get("HG_TEACHER_MODEL") or "").strip(),
                   teacher_api_key=str(env.get("HG_TEACHER_API_KEY") or "").strip())
