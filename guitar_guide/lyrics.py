"""Слова песни с таймкодами: Demucs отделяет голос, faster-whisper расшифровывает по словам."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def separate(wav: str, work: Path) -> dict:
    """{'vocals': голос, 'no_vocals': всё остальное, то есть гитара}."""
    out = work / "demucs"
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(out), wav],
                   check=True)
    return {"vocals": str(next(out.rglob("vocals.wav"))), "no_vocals": str(next(out.rglob("no_vocals.wav")))}


# Фразы, которые Whisper выдумывает на музыке и тишине
_HALLUCINATIONS = ("suscr", "gracias", "subt", "amara.org", "thank you", "продолжение следует", "редактор субтитров")


def vocal_regions(vocals: str, min_len: float = 0.8, gap: float = 0.7) -> list[tuple[float, float]]:
    """Где поют: энергия вокальной дорожки выше порога, короткие паузы склеиваем."""
    import librosa
    import numpy as np

    y, sr = librosa.load(vocals, sr=16000, mono=True)
    hop = 320
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    thr = max(0.12 * np.percentile(rms, 95), 1e-4)
    on = rms > thr
    t = np.arange(len(rms)) * hop / sr
    regs, start = [], None
    for ti, v in zip(t, on):
        if v and start is None:
            start = ti
        elif not v and start is not None:
            regs.append([start, ti])
            start = None
    if start is not None:
        regs.append([start, t[-1]])
    merged = []
    for r in regs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(round(float(a), 2), round(float(b), 2)) for a, b in merged if b - a >= min_len]


def transcribe(audio: str, language: str | None = None, model: str = "large-v3", prompt: str | None = None) -> dict:
    import librosa
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device="cpu", compute_type="int8")
    y, sr = librosa.load(audio, sr=16000, mono=True)
    regions = vocal_regions(audio)
    lines = []
    for a, b in regions:
        chunk = y[max(0, int((a - 0.3) * sr)): int((b + 0.3) * sr)]
        off = max(0.0, a - 0.3)
        segs, _ = m.transcribe(chunk, language=language, word_timestamps=True, vad_filter=False,
                               condition_on_previous_text=False, beam_size=5, initial_prompt=prompt,
                               temperature=(0.0, 0.2, 0.4), no_speech_threshold=0.7)
        got = False
        for s in segs:
            text = s.text.strip()
            if not text or any(h in text.lower() for h in _HALLUCINATIONS):
                continue
            words = [{"start": round(off + w.start, 2), "end": round(off + w.end, 2), "word": w.word.strip(),
                      "p": round(w.probability, 2)} for w in (s.words or []) if w.word.strip()]
            if not words:
                continue
            got = True
            lines.append({"start": words[0]["start"], "end": words[-1]["end"], "text": text, "words": words,
                          "logprob": round(s.avg_logprob, 2)})
        if not got:
            lines.append({"start": a, "end": b, "text": "", "words": [], "logprob": None})
    return {"regions": regions, "lines": lines}


def analyze(vocals: str, language: str | None = None, prompt: str | None = None) -> dict:
    res = transcribe(vocals, language, prompt=prompt)
    res["language"] = language
    return res


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
