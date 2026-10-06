"""The laptop's view of a box's folders, over SSH: commands for the box's Python, and where its files are.

The code folder is the one fixed place (INSTALL_DIR). Where the config, data and logs are, the box
says itself (``python -m home_guard_project.box paths --json``, paths.py): C:\\ProgramData\\HomeGuard on
a migrated box, the code folder on an old one. A box whose software is too old to answer keeps
everything in the code folder, the old layout.
"""
import json
import ntpath

from .. import paths

INSTALL_DIR = r'C:\home_guard'
PYTHON = r'.venv\Scripts\python.exe'


def box_command(args, install_dir=INSTALL_DIR):
    """A cmd.exe line that runs the box's Python with *args* in its code folder."""
    return rf'cd /d {install_dir} && {PYTHON} {args}'


PATHS_COMMAND = box_command('-m home_guard_project.box paths --json')


def legacy_paths(install_dir=INSTALL_DIR):
    """The places of a box that cannot say: all in the code folder."""
    return paths.legacy(install_dir, 'the box did not answer paths --json').as_dict()


def parse_paths(output, returncode, install_dir=INSTALL_DIR):
    """The box's answer to PATHS_COMMAND; the old layout when it has none (old software, an error)."""
    if not returncode:
        start = (output or '').find('{')
        try:
            data = json.loads(output[start:]) if start >= 0 else None
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get('logs_dir'), str):
            return data
    return legacy_paths(install_dir)


def type_command(directory, name):
    """A cmd.exe line printing the box's file *name* in *directory* (a path from the box's answer)."""
    path = ntpath.join(directory, name)
    if '"' in path:
        raise ValueError('Quotes are not allowed in a box path')
    return f'type "{path}"'
