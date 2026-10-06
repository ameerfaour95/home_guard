"""Judge the owner's translator: saved English alert summaries, translated by each model side by side.

    # Gemini 3.1 Flash Lite against gpt-6-luna (both on OpenRouter, OPENROUTER_API_KEY in api_key.env):
    python -m home_guard_project.box.eval_translation dataset_box/meta eval_set/results/run.jsonl --limit 50
    python -m home_guard_project.box.eval_translation <inputs> --model openrouter:google/gemini-3.1-flash-lite \\
        --model openrouter:openai/gpt-6-luna --out translation_eval.csv
    python -m home_guard_project.box.eval_translation <inputs> --fake      # no network: checks the setup

Inputs are files or folders: clip ``*.meta.json`` (the ``alert`` the box saved: summary, why, the
model's own summary_owner) and ``eval_prompt`` results ``*.jsonl`` (``ai_summary`` and the raw
answer). Each distinct (summary, why) is translated once per model, through ``messenger.Messenger``
exactly as the box does (same prompt, checks and timeout), so a row whose source is ``fallback`` is
what the owner would have read instead. The CSV (UTF-8 with BOM, opens in Excel) has the English,
the vision model's own Hebrew when it wrote one, every model's translation, latency and cost, and
empty ``owner_pick`` / ``owner_notes`` columns for the owner's verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import messenger, providers

DEFAULT_MODELS = ("openrouter:google/gemini-3.1-flash-lite", "openrouter:openai/gpt-6-luna")


def _row(summary: Any, why: Any = "", summary_owner: Any = "", camera: Any = "", source: str = "") -> Optional[Dict[str, str]]:
    text = str(summary or "").strip()
    if not text:
        return None
    return {"summary": text, "why": str(why or "").strip(), "summary_owner": str(summary_owner or "").strip(),
            "camera": str(camera or ""), "source_file": source}


def _from_record(rec: Any, path: str) -> Optional[Dict[str, str]]:
    if not isinstance(rec, dict):
        return None
    if isinstance(rec.get("alert"), dict):                      # a clip's meta.json
        a = rec["alert"]
        return _row(a.get("summary"), a.get("why"), a.get("summary_owner"), rec.get("camera_name"), path)
    if "ai_summary" in rec:                                     # an eval_prompt results row
        try:
            raw = messenger._parse(rec.get("raw") or "{}")
        except messenger.TranslationError:
            raw = {}
        raw = raw if isinstance(raw, dict) else {}
        return _row(rec.get("ai_summary"), raw.get("why"), raw.get("summary_owner"), rec.get("camera"), path)
    return _row(rec.get("summary"), rec.get("why"), rec.get("summary_owner"), rec.get("camera"), path)


def _files(inputs: Sequence[str]) -> Iterator[str]:
    for item in inputs:
        if os.path.isdir(item):
            for root, _, names in os.walk(item):
                for name in sorted(names):
                    if name.endswith((".json", ".jsonl")):
                        yield os.path.join(root, name)
        else:
            yield item


def load_rows(inputs: Sequence[str], limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Distinct English (summary, why) pairs from *inputs*, in file order, at most *limit*."""
    rows: List[Dict[str, str]] = []
    seen = set()
    for path in _files(inputs):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            records = ([json.loads(line) for line in text.splitlines() if line.strip()]
                       if path.endswith(".jsonl") else [json.loads(text)])
        except (OSError, ValueError) as exc:
            print(f"skipped {path}: {exc}", file=sys.stderr)
            continue
        for rec in records:
            row = _from_record(rec, path)
            if row is None or (row["summary"], row["why"]) in seen:
                continue
            seen.add((row["summary"], row["why"]))
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def parse_model(spec: str) -> Tuple[str, str]:
    """``provider:model`` -> (provider, model); the model id may hold colons (``...:batch``)."""
    provider, sep, model = spec.partition(":")
    if not sep or provider not in providers.PROVIDERS or not model:
        raise SystemExit(f"--model must be provider:model with provider one of {', '.join(sorted(providers.PROVIDERS))}"
                         f" (got {spec!r})")
    return provider, model


class _FakeClient:
    """No network: answers every request with a Hebrew stand-in that keeps the numbers and names."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def create(**kwargs: Any) -> Any:
        src = json.loads(kwargs["messages"][1]["content"])
        answer = {k: (f"[תרגום] {v}" if v else "") for k, v in src.items()}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))],
                               usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


def _messenger(provider: str, model: str, timeout: float, fake: bool) -> messenger.Messenger:
    if fake:
        return messenger.Messenger(_FakeClient(), model, timeout, cache_size=0)
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        from .boxconfig import PROJECT_ROOT  # noqa: PLC0415

        load_dotenv(os.path.join(PROJECT_ROOT, "api_key.env"))
    except Exception:  # noqa: BLE001
        pass
    try:
        client, extra = messenger.build_client(provider, os.environ, timeout, model)
    except providers.ProviderError as exc:
        raise SystemExit(f"{exc}, or use --fake.") from None
    return messenger.Messenger(client, model, timeout, extra, cache_size=0)


def run(rows: List[Dict[str, str]], models: Sequence[str], lang: str, timeout: float, out_path: str,
        fake: bool = False) -> Dict[str, Dict[str, float]]:
    """Translate every row with every model, write the CSV, return per-model totals."""
    columns = ["id", "camera", "summary_en", "why_en", "eye_summary_owner"]
    totals: Dict[str, Dict[str, float]] = {}
    for spec in models:
        provider, model = parse_model(spec)
        tag = providers.model_key(provider, model)
        columns += [f"{tag} summary", f"{tag} why", f"{tag} source", f"{tag} ms", f"{tag} usd"]
        m = _messenger(provider, model, timeout, fake)
        total = totals.setdefault(tag, {"rows": 0, "fallback": 0, "ms": 0.0, "usd": 0.0})
        for row in rows:
            started = time.monotonic()
            told = m.to_owner(row, lang, keep=(row["camera"],))
            ms = (time.monotonic() - started) * 1000
            usd = providers.cost_usd(model, m.last_usage["prompt_tokens"], m.last_usage["completion_tokens"])
            m.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            row.update({f"{tag} summary": told["summary"], f"{tag} why": told["why"],
                        f"{tag} source": told["source"], f"{tag} ms": f"{ms:.0f}",
                        f"{tag} usd": "" if usd is None else f"{usd:.6f}"})
            total["rows"] += 1
            total["fallback"] += told["source"] == "fallback"
            total["ms"] += ms
            total["usd"] += usd or 0.0
    columns += ["owner_pick", "owner_notes"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({**row, "id": i, "summary_en": row["summary"], "why_en": row["why"],
                             "eye_summary_owner": row["summary_owner"], "owner_pick": "", "owner_notes": ""})
    return totals


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Translate saved English alert summaries with each model, side by side.")
    parser.add_argument("inputs", nargs="+", help="Clip meta.json files, eval_prompt results .jsonl, or folders of them.")
    parser.add_argument("--model", action="append", default=None,
                        help=f"provider:model, repeatable (default: {' and '.join(DEFAULT_MODELS)}).")
    parser.add_argument("--lang", default="he", choices=sorted(messenger.LANGUAGE_NAMES))
    parser.add_argument("--limit", type=int, default=50, help="At most N distinct summaries (default 50).")
    parser.add_argument("--timeout", type=float, default=messenger.DEFAULT_TIMEOUT_SEC,
                        help="Seconds per call, as on the box (a slower answer counts as a fallback).")
    parser.add_argument("--out", default="translation_eval.csv")
    parser.add_argument("--fake", action="store_true", help="No network: a stand-in translator, for checking the setup.")
    args = parser.parse_args(argv)

    rows = load_rows(args.inputs, args.limit)
    if not rows:
        print("No English summaries found in the inputs.", file=sys.stderr)
        return 1
    totals = run(rows, args.model or list(DEFAULT_MODELS), args.lang, args.timeout, args.out, fake=args.fake)
    for tag, t in totals.items():
        n = max(int(t["rows"]), 1)
        print(f"{tag}: {int(t['rows'])} rows, {int(t['fallback'])} fell back, "
              f"{t['ms'] / n:.0f} ms average, ${t['usd']:.4f} total")
    print(f"Wrote {args.out}: fill owner_pick (which model reads best) and owner_notes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
