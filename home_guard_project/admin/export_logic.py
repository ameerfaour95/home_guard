"""Local form validation. Export eligibility always comes from the backend."""
import math
import re


def validate_export(name, formats, split):
    if not re.fullmatch(r'[a-z0-9_-]+', name):
        return 'Use lowercase letters, numbers, underscores or hyphens for the export name.'
    if not formats or not set(formats) <= {'yolo','vlm_jsonl','clips'}:
        return 'Choose at least one export format.'
    if set(split) != {'train','val','test'} or any(not math.isfinite(v) or not 0 <= v <= 1 for v in split.values()) or not math.isclose(sum(split.values()),1,abs_tol=1e-8):
        return 'Train, validation and test must total 100%.'
    return ''
