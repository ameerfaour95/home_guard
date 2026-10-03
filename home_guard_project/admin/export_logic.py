"""Pure export validation and consent partitioning shared by UI and demo service."""
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


def consent_summary(events, consent, include_fallback=False):
    result = dict(included=[], excluded=[], unknown=[])
    for event in events:
        if event.id not in consent:
            result['unknown'].append(event.id)
        elif not consent[event.id]:
            result['excluded'].append((event.id, event.camera, 'No training consent'))
        elif event.completeness.ai == 'fallback' and not include_fallback:
            result['excluded'].append((event.id, event.camera, 'Fallback AI excluded'))
        else:
            result['included'].append(event.id)
    return result


