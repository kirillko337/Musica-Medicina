"""Сверка: что услышал звуковой анализ и что увидел Claude на видео."""
from __future__ import annotations

from collections import Counter

from . import theory


def capo_from_vision(over: dict | None, vis: list[dict]) -> int | None:
    votes = Counter()
    if over and over.get("capo_fret", -1) >= 0:
        votes[over["capo_fret"]] += 2
    for v in vis:
        r = v.get("vision") or {}
        if r.get("fretboard_visible") and r.get("capo_fret", -1) >= 0:
            votes[r["capo_fret"]] += 1
    return votes.most_common(1)[0][0] if votes else None


def chords(vis: list[dict], capo: int) -> dict:
    """Сравниваем формы: звучащий аккорд из звука переводим в форму с учётом каподастра."""
    rows, score = [], Counter()
    for v in vis:
        r = v.get("vision") or {}
        audio_shape = theory.transpose(v["audio"], -capo)
        shape = r.get("shape", "unknown")
        conf = float(r.get("confidence", 0) or 0)
        if "error" in r or not r.get("fretboard_visible") or shape == "unknown" or conf < 0.4:
            verdict = "не видно"
        else:
            verdict = {"exact": "совпало", "triad": "почти (тот же трезвучный)", "root": "тот же корень",
                       "no": "расхождение"}[theory.same(audio_shape, shape)]
        score[verdict] += 1
        rows.append({"start": v["start"], "end": v["end"], "audio_shape": audio_shape,
                     "vision_shape": shape, "vision_conf": conf, "verdict": verdict,
                     "frets": r.get("frets")})
    seen = sum(n for k, n in score.items() if k != "не видно")
    match = score["совпало"] + score["почти (тот же трезвучный)"]
    return {"rows": rows, "counts": dict(score), "agreement": round(match / seen, 3) if seen else None}


def strums(onsets: list[dict], flow: list[dict]) -> dict:
    by_t = {round(f["t"], 3): f for f in flow}
    rule_vs_video, audio_vs_video, n = 0, 0, 0
    for o in onsets:
        f = by_t.get(round(o["t"], 3))
        if not f or f["dir_video"] == "?":
            continue
        o["dir_video"] = f["dir_video"]
        n += 1
        rule_vs_video += o.get("dir_rule") == f["dir_video"]
        audio_vs_video += o.get("dir_audio") == f["dir_video"]
    return {
        "checked": n,
        "rule_vs_video": round(rule_vs_video / n, 3) if n else None,
        "audio_vs_video": round(audio_vs_video / n, 3) if n else None,
    }


def consensus(segments: list[dict], cmp_rows: list[dict], capo: int) -> list[dict]:
    """Итоговые формы для гайда: уверенное зрение перекрывает звук, иначе берём звук."""
    by_start = {r["start"]: r for r in cmp_rows}
    out = []
    for s in segments:
        shape = theory.transpose(s["chord"], -capo) if s["chord"] != "N" else "N"
        src = "звук"
        r = by_start.get(s["start"])
        if r and r["verdict"] == "расхождение" and r["vision_conf"] >= 0.8 and theory.parse(r["vision_shape"]):
            shape, src = r["vision_shape"], "видео"
        out.append({**s, "shape": shape, "source": src})
    return out
