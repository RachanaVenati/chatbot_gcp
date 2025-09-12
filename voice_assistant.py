import io
import time
import requests
from typing import Tuple, Optional

# ============
# Speech-to-Text (OpenAI Whisper REST)
# ============

def transcribe_audio_bytes(
    audio_bytes: bytes,
    api_key: str,
    model: str = "whisper-1",
    prompt: Optional[str] = None,
    language: Optional[str] = None,  # e.g., "en"; None lets Whisper auto-detect
    timeout: int = 60,
) -> str:
    """
    Send raw audio bytes (WAV/PCM) to OpenAI Whisper for transcription.

    Returns the transcribed text string or raises a RuntimeError on failure.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for transcription.")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        # Whisper accepts many formats; audio_recorder_streamlit returns a WAV-like byte stream
        "file": ("speech.wav", io.BytesIO(audio_bytes), "audio/wav"),
        "model": (None, model),
    }
    if prompt:
        files["prompt"] = (None, prompt)
    if language:
        files["language"] = (None, language)

    try:
        resp = requests.post(url, headers=headers, files=files, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Whisper REST returns { text: "..." }
        text = data.get("text", "").strip()
        return text
    except requests.RequestException as e:
        raise RuntimeError(f"Transcription request failed: {e}") from e
    except Exception as e:
        # Provide server's response for easier debugging
        try:
            bad = resp.text  # type: ignore
        except Exception:
            bad = ""
        raise RuntimeError(f"Bad transcription response: {bad}") from e


# ============
# Text-to-Speech (gTTS)
# ============

from gtts import gTTS

def synthesize_speech_mp3(
    text: str,
    lang: str = "en",
    slow: bool = False,
) -> Tuple[bytes, str]:
    """
    Generate speech audio (MP3) bytes from text using gTTS.
    Returns (audio_bytes, mime_type='audio/mpeg').
    """
    if not text or not text.strip():
        return b"", "audio/mpeg"

    tts = gTTS(text=text, lang=lang, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read(), "audio/mpeg"


# ============
# Simple rate limiter
# ============

class SimpleRateLimiter:
    def __init__(self, min_interval_sec: float = 0.75):
        self.min_interval = min_interval_sec
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()
