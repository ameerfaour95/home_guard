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
FALLBACK_MODEL = "google/gemini-3.1-flash-lite"     # OpenRouter, audio in

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


def make_transcriber(env: Mapping[str, str], model: str = DEFAULT_MODEL,
                     fallback_model: str = FALLBACK_MODEL) -> Optional[Transcriber]:
    """A transcriber on the OpenAI API (the OS trust store, like the chat model), then OpenRouter's audio model when
    OpenAI fails or has no key; None without either key.

    2026-10-11: the OpenAI account had no credit ("insufficient_quota"), so every voice message failed; OpenRouter
    (the key the box's AI runs on) takes Telegram's OGG as it is (gemini flash lite, about $0.0001 a message)."""
    first = _openai_transcriber(env, model)
    second = _openrouter_transcriber(env, fallback_model)
    if first is None:
        return second
    if second is None:
        return first

    def transcribe(audio: bytes, name: str, language: str = "") -> str:
        try:
            heard = first(audio, name, language)
            if heard:
                return heard
        except Exception as exc:  # noqa: BLE001 - the second service answers
            log.warning("Voice: OpenAI transcription failed (%s); trying %s", str(exc)[:160], fallback_model)
        return second(audio, name, language)

    return transcribe


def _openai_transcriber(env: Mapping[str, str], model: str) -> Optional[Transcriber]:
    key = env.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        import ssl  # noqa: PLC0415

        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=key, http_client=httpx.Client(verify=ssl.create_default_context()), max_retries=0)
    except Exception as exc:  # noqa: BLE001 - without it, a voice answer is asked for in writing
        log.warning("Voice messages cannot be transcribed by OpenAI: %s", exc)
        return None

    def transcribe(audio: bytes, name: str, language: str = "") -> str:
        kwargs: Dict[str, Any] = {"model": model, "file": (name or "voice.ogg", audio)}
        if language:
            kwargs["language"] = language        # ISO 639-1: the box language ("he", "en", "ar")
        from . import usage_ledger  # noqa: PLC0415

        result = usage_ledger.call("transcription", lambda: client.audio.transcriptions.create(**kwargs),
                                   client=client, provider="openai", model=model)
        return " ".join(str(getattr(result, "text", "") or "").split())

    return transcribe


_FORMATS = {".oga": "ogg", ".ogg": "ogg", ".opus": "ogg", ".mp3": "mp3", ".wav": "wav", ".m4a": "m4a"}


def _openrouter_transcriber(env: Mapping[str, str], model: str) -> Optional[Transcriber]:
    key = env.get("OPENROUTER_API_KEY", "")
    if not key or not model:
        return None
    try:
        import ssl  # noqa: PLC0415

        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=40.0, max_retries=0,
                        http_client=httpx.Client(verify=ssl.create_default_context()))
    except Exception as exc:  # noqa: BLE001
        log.warning("Voice messages cannot be transcribed by OpenRouter: %s", exc)
        return None
    from ..prompts import render  # noqa: PLC0415

    def transcribe(audio: bytes, name: str, language: str = "") -> str:
        import base64  # noqa: PLC0415

        fmt = _FORMATS.get(os.path.splitext(name or "")[1].lower(), "ogg")
        content = [{"type": "text", "text": render("voice_transcribe.prompt", language=language or "the speaker's")},
                   {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode("ascii"),
                                                            "format": fmt}}]
        from . import usage_ledger  # noqa: PLC0415

        resp = usage_ledger.call("transcription", lambda: client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}]),
            client=client, provider="openrouter", model=model)
        return " ".join(str(resp.choices[0].message.content or "").split())

    return transcribe
