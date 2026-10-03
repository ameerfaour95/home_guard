"""Local diagnostics without wire payloads, credentials or media URLs."""
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import sys
import traceback


class RedactionFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if record.exc_info:
            # Stack locations are useful; exception messages can contain arbitrary
            # request bodies, credentials and signed URLs. Never persist them.
            message += '\n' + ''.join(f'{frame.filename}:{frame.lineno} in {frame.name}\n'
                                      for frame in traceback.extract_tb(record.exc_info[2]))
            message += type(record.exc_info[1]).__name__
        message = re.sub(r'https?://\S+', '[URL REDACTED]', message)
        message = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', message)
        message = re.sub(r'''(?ix)(["']?(?:password|totp|code|access_token|refresh_token|token)["']?\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,}]+)''', r'\1[REDACTED]', message)
        record.msg, record.args, record.exc_info, record.exc_text = message, (), None, None
        return True


def setup_logging():
    logger = logging.getLogger('homeguard.admin')
    if not logger.handlers:
        path = Path(os.environ.get('LOCALAPPDATA', Path.home()))/'HomeGuardAdmin'/'logs'
        path.mkdir(parents=True, exist_ok=True)
        # Active file plus four backups: at most five 1 MiB files.
        handler = RotatingFileHandler(path/'admin.log', maxBytes=1024*1024, backupCount=4, encoding='utf-8')
        handler.addFilter(RedactionFilter())
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def install_exception_hook(show_error):
    def report(kind, value, tb):
        setup_logging().error('Unhandled UI exception', exc_info=(kind, value, tb))
        show_error('Something went wrong. Close this view and try again. Details were saved to the local admin log.')
    sys.excepthook = report
