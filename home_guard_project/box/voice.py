"""The owner's voice messages, as text: fetch the recording from Telegram and transcribe it.

Used for the explanation of an alert ("Other…"): the owner may speak instead of
typing. No network at import; the caller injects both steps, so tests never reach
Telegram or the transcription service.

    download(token, file_id)        the recording's bytes and file name (getFile, then the file URL)
    make_transcriber(env, model)    a function (audio, file name, language) -> text, or None without a key
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from . import telegram_notify

log = logging.getLogger("box.voice")

_FILE_URL = "https://api.telegram.org/file/bot{token}/{path}"
MAX_VOICE_BYTES = 20 * 1024 * 1024   # the most a bot may download (Telegram's getFile limit)
DEFAULT_MODEL = "gpt-4o-mini-transcribe"

Transcriber = Callable[[bytes, str, str], str]


def download(token: str, file_id: str, post: Callable[..., Dict[str, Any]] = telegram_notify._http_post,
             opener: Callable[..., Any] = urllib.request.urlopen, timeout: float = 30.0) -> Tuple[bytes, str]:
    """The bytes and file name of the Telegram file *file_id*. Raises on any failure (the caller says so)."""
    resp = post(token, "getFile", {"file_id": file_id}, timeout=timeout)
    path = str(((resp or {}).get("result") or {}).get("file_path") or "")
    if not (resp or {}).get("ok") or not path:
        raise OSError(f"Telegram gave no file for the voice message: {(resp or {}).get('description')}")
    with opener(_FILE_URL.format(token=token, path=path), timeout=timeout) as f:
        data = f.read(MAX_VOICE_BYTES + 1)
    if len(data) > MAX_VOICE_BYTES:
        raise OSError("the voice message is too long")
    return data, os.path.basename(path) or "voice.ogg"


def make_transcriber(env: Mapping[str, str], model: str = DEFAULT_MODEL) -> Optional[Transcriber]:
    """A transcriber on the OpenAI API (the OS trust store, like the chat model), or None without a key."""
    key = env.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        import ssl  # noqa: PLC0415

        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=key, http_client=httpx.Client(verify=ssl.create_default_context()))
    except Exception as exc:  # noqa: BLE001 - without it, a voice answer is asked for in writing
        log.warning("Voice messages cannot be transcribed: %s", exc)
        return None

    def transcribe(audio: bytes, name: str, language: str = "") -> str:
        kwargs: Dict[str, Any] = {"model": model, "file": (name or "voice.ogg", audio)}
        if language:
            kwargs["language"] = language        # ISO 639-1: the box language ("he", "en", "ar")
        result = client.audio.transcriptions.create(**kwargs)
        return " ".join(str(getattr(result, "text", "") or "").split())

    return transcribe
