"""Слова песни с таймкодами: Demucs отделяет голос, faster-whisper расшифровывает по словам."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def separate_vocals(wav: str, work: Path) -> str:
    out = work / "demucs"
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(out), wav],
                   check=True)
    return str(next(out.rglob("vocals.wav")))


def transcribe(audio: str, language: str | None = None, model: str = "large-v3") -> list[dict]:
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = m.transcribe(audio, language=language, word_timestamps=True, vad_filter=True,
                                  condition_on_previous_text=False, beam_size=5)
    lines = []
    for s in segments:
        words = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()} for w in (s.words or [])]
        if words:
            lines.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})
    return lines


def analyze(wav: str, work: Path, language: str | None = None) -> dict:
    try:
        vocals = separate_vocals(wav, work)
    except Exception as e:  # без Demucs тоже работает, просто хуже
        print(f"  demucs не сработал ({e}), распознаю по общей дорожке")
        vocals = wav
    return {"language": language, "lines": transcribe(vocals, language)}


def chord_line(words: list[dict], chord_at) -> tuple[str, str]:
    """Две строки: аккорды над тем словом, на котором они меняются, и сами слова."""
    top, bottom = "", ""
    prev = None
    for i, w in enumerate(words):
        c = chord_at(w["start"])
        token = w["word"] + " "
        if c != prev or i == 0:
            label = c if c != "N" else ""
            if len(top) > len(bottom):
                bottom += " " * (len(top) - len(bottom))
            top += " " * (len(bottom) - len(top)) + label + " "
            prev = c
        bottom += token
    return top.rstrip(), bottom.rstrip()
