"""Every text the box sends a model as an instruction lives in this folder, one file per prompt.

- ``<area>_<name>.system_prompt``: a system message.
- ``<area>_<name>.prompt``: an instruction sent in a user message (the vision prompts, follow-up rules, fragments the
  code puts together).

The code decides WHICH files go into a prompt and in what order; the files hold the words. A value the code fills in
is written ``{{name}}`` in the file and passed to :func:`render`; a missing or an unused value is an error, so a
renamed placeholder cannot silently drop text. Everything else in a file (JSON examples with single braces, quotes)
is sent as it is.

A file holds the text exactly; the one newline that ends the file is not part of the prompt. Files are read with
universal newlines, so a checkout with CRLF endings sends the same text (``.gitattributes`` keeps them LF anyway).

Changing the words of a prompt the box runs is a prompt change: it goes through ``box/eval_prompt.py`` first.
"""
from __future__ import annotations

import functools
import os
import re
from typing import Any

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUFFIXES = (".prompt", ".system_prompt")

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def path(name: str, /) -> str:
    """The file of prompt *name* (``"eye_legacy.prompt"``)."""
    if not name.endswith(SUFFIXES) or os.path.basename(name) != name:
        raise ValueError(f"not a prompt file name: {name!r}")
    return os.path.join(PROMPTS_DIR, name)


@functools.lru_cache(maxsize=None)
def load(name: str, /) -> str:
    """The text of prompt *name*, without the newline that ends the file."""
    with open(path(name), encoding="utf-8") as f:
        text = f.read()
    return text[:-1] if text.endswith("\n") else text


def placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER.findall(text))


def fill(text: str, /, **values: Any) -> str:
    """*text* with every ``{{name}}`` replaced by ``str(values[name])``; every value must be used."""
    wanted = placeholders(text)
    missing = wanted - values.keys()
    unused = values.keys() - wanted
    if missing or unused:
        raise KeyError(f"prompt placeholders: missing {sorted(missing)}, unused {sorted(unused)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def render(name: str, /, **values: Any) -> str:
    """Prompt *name* with its ``{{placeholders}}`` filled in."""
    return fill(load(name), **values)


def names() -> list[str]:
    """Every prompt file in the folder, sorted."""
    return sorted(n for n in os.listdir(PROMPTS_DIR) if n.endswith(SUFFIXES))
