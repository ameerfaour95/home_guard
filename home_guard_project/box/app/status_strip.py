"""Quiet operational facts shown in the footer and Settings."""
import re
from .strings import tr

def status_text(upload,clips,disk,error=False):
    if error: return tr('unknown')
    match=re.search(r'\b(\d{2}:\d{2})(?::\d{2})?\b',upload or '')
    stamp=match[1] if match else upload or tr('never')
    return tr('quiet_status',time=stamp,clips=clips,disk=f'{disk:g}')
