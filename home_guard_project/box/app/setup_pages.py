"""Stable names for setup navigation and failed-step ownership."""
from enum import IntEnum


class Page(IntEnum):
    ADDRESS = 0
    NETWORK = 1
    HOUSE = 2
    OWNER = 3
    CAMERAS = 4
    PROGRESS = 5
    SUMMARY = 6
    REVIEW = 7
    CAMERA_CHECK = 8
