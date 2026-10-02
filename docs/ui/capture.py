"""Regenerate the review gallery offline, with no credentials or camera input."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("screenshots")
CASES = []
for state in (
    "mixed",
    "live",
    "offline",
    "stopped",
    "hidden",
    "empty",
    "error",
    "loading",
    "inference",
    "quiet",
):
    CASES.append(("box-" + state, ["--demo", "--state", state]))
for count in (1, 4, 9):
    CASES.append(
        (
            "box-" + str(count) + "-cameras",
            [
                "--demo",
                "--cameras",
                str(count),
                "--state",
                "live" if count < 9 else "mixed",
            ],
        )
    )
CASES.append(("box-details", ["--demo", "--details"]))
for page in (
    "address",
    "network",
    "house",
    "cameras",
    "progress",
    "summary",
    "failure",
    "validation",
):
    CASES.append(("setup-" + page, ["--setup", "--demo", "--page", page]))
CASES.extend(
    [
        ("setup-wifi", ["--setup", "--demo", "--page", "network", "--wifi"]),
        ("setup-alerts", ["--setup", "--demo", "--page", "house", "--alerts"]),
        (
            "setup-cameras-later",
            ["--setup", "--demo", "--page", "cameras", "--skip-cameras"],
        ),
        (
            "setup-summary-cameras-later",
            ["--setup", "--demo", "--page", "summary", "--skip-cameras"],
        ),
    ]
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for size in ("1366x768", "1920x1080"):
        for name, flags in CASES:
            path = OUT / (name + "-" + size + ".png")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "home_guard_project.box.app",
                    *flags,
                    "--size",
                    size,
                    "--screenshot",
                    str(path),
                ],
                cwd=ROOT,
                check=True,
            )
            print(path.name, flush=True)


if __name__ == "__main__":
    main()
