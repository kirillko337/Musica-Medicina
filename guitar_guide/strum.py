"""Бой по звуку: удары, привязка к сетке, направление (вниз/вверх), повторяющийся рисунок."""
from __future__ import annotations

from collections import Counter

import librosa
import numpy as np

HOP = 512


def onsets(y: np.ndarray, sr: int) -> list[dict]:
    y_perc = librosa.effects.percussive(y, margin=2.0)
    oenv = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP, aggregate=np.median)
    frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP, backtrack=False,
                                        delta=0.07, wait=3)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    med = float(np.median(oenv[frames])) if len(frames) else 1.0
    return [{"t": round(float(t), 3), "strength": round(float(oenv[f] / (med + 1e-9)), 2)}
            for t, f in zip(times, frames)]


def _bass_ratio(y: np.ndarray, sr: int, t: float) -> float:
    """Сколько новой энергии пришло в бас (70–150 Гц, 6-я и 5-я струны) относительно середины.
    Удар вниз почти всегда цепляет басовые струны, удар вверх — в основном верхние три-четыре."""
    nfft, hop = 4096, 128
    a = max(0, int((t - 0.12) * sr))
    seg = y[a: int((t + 0.12) * sr)]
    if len(seg) < nfft // 2:
        return float("nan")
    S = np.abs(librosa.stft(seg, n_fft=nfft, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    tt = np.arange(S.shape[1]) * hop / sr + a / sr
    before = S[:, (tt > t - 0.06) & (tt < t - 0.02)]
    after = S[:, (tt > t + 0.03) & (tt < t + 0.07)]
    if before.size == 0 or after.size == 0:
        return float("nan")
    d = np.maximum(0, after.mean(1) - before.mean(1)) ** 2
    return float(d[(f >= 70) & (f < 150)].sum() / (d[(f >= 150) & (f < 1000)].sum() + 1e-12))


def acoustic_directions(y: np.ndarray, sr: int, onset_list: list[dict]) -> None:
    """Вниз/вверх по звуку: бас + громкость, порог — делим удары песни на две группы (2-means).
    Это слабый признак (на синтетике ~80%), поэтому в гайде он только подтверждает «маятник»."""
    if len(onset_list) < 6:
        for o in onset_list:
            o["dir_audio"], o["dir_audio_conf"] = "?", 0.0
        return
    br = np.array([_bass_ratio(y, sr, o["t"]) for o in onset_list])
    lb = np.log10(np.nan_to_num(br, nan=np.nanmedian(br)) + 1e-6)
    ls = np.log10(np.array([o["strength"] for o in onset_list]) + 1e-3)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    score = z(lb) + 0.5 * z(ls)
    thr = float(np.median(score))
    for _ in range(20):
        hi, lo = score[score > thr], score[score <= thr]
        if not len(hi) or not len(lo):
            break
        thr = (hi.mean() + lo.mean()) / 2
    sd = score.std() + 1e-9
    for o, sc, raw in zip(onset_list, score, br):
        if np.isnan(raw):
            o["dir_audio"], o["dir_audio_conf"] = "?", 0.0
            continue
        o["dir_audio"] = "down" if sc > thr else "up"
        o["dir_audio_conf"] = round(float(min(1.0, abs(sc - thr) / sd)), 2)


def grid(onset_list: list[dict], beat_times: list[float], subdiv: int) -> None:
    """Проставляет каждому удару номер доли и позицию внутри неё (0..subdiv-1)."""
    bt = np.asarray(beat_times)
    period = float(np.median(np.diff(bt)))
    for o in onset_list:
        t = o["t"]
        i = int(np.searchsorted(bt, t, side="right") - 1)
        if i < 0:
            o["beat"], o["pos"] = None, None
            continue
        start = bt[i]
        length = bt[i + 1] - start if i + 1 < len(bt) else period
        x = (t - start) / length * subdiv
        pos = int(round(x))
        if abs(x - pos) > 0.3 or pos > subdiv:  # слишком далеко от сетки
            o["beat"], o["pos"] = None, None
            continue
        if pos == subdiv:
            i, pos = i + 1, 0
        o["beat"], o["pos"] = i, pos


def choose_subdivision(onset_list: list[dict], beat_times: list[float]) -> int:
    probe = [dict(o) for o in onset_list]
    grid(probe, beat_times, 4)
    placed = [o for o in probe if o["pos"] is not None]
    if not placed:
        return 2
    odd = sum(1 for o in placed if o["pos"] % 2 == 1)
    return 4 if odd / len(placed) > 0.2 else 2


def pattern(onset_list: list[dict], n_beats: int, phase: int, subdiv: int, beats_per_bar: int = 4) -> dict:
    slots = beats_per_bar * subdiv
    bars: dict[int, list[dict | None]] = {}
    for o in onset_list:
        if o.get("beat") is None:
            continue
        b = o["beat"] - phase
        if b < 0:
            continue
        bar, beat_in_bar = divmod(b, beats_per_bar)
        bars.setdefault(bar, [None] * slots)[beat_in_bar * subdiv + o["pos"]] = o

    vectors = {bar: tuple(int(s is not None) for s in v) for bar, v in bars.items() if sum(s is not None for s in v) >= 2}
    if not vectors:
        return {"subdiv": subdiv, "slots": [], "bars_matching": 0, "bars_total": 0, "variants": []}
    counts = Counter(vectors.values())
    (main, n_main), *rest = counts.most_common(3)

    # направление: «маятник» — на долю рука идёт вниз, между долями вверх;
    # акустика и видео потом проверяют это предположение
    accents = []
    for i in range(slots):
        st = [bars[b][i]["strength"] for b, v in vectors.items() if v == main and bars[b][i]]
        accents.append(bool(st) and float(np.median(st)) > 1.35)
    out = []
    for i, hit in enumerate(main):
        rule = "down" if i % 2 == 0 else "up"
        out.append({"hit": bool(hit), "dir": rule if hit else None, "accent": accents[i] and bool(hit)})
    return {
        "subdiv": subdiv,
        "slots": out,
        "bars_matching": n_main,
        "bars_total": len(vectors),
        "variants": [{"slots": list(v), "bars": n} for v, n in rest],
    }


def profile(y: np.ndarray, sr: int, beat_times: list[float], phase: int, beats_per_bar: int = 4,
            hit: float = 0.3, accent: float = 0.8) -> dict:
    """Рисунок боя по средней силе удара на каждой из 16 шестнадцатых такта.

    Пороговый детектор ударов теряет слабые удары, а «бой» как раз и состоит из сильных
    и слабых. Поэтому берём огибающую атак и усредняем её по всем тактам песни:
    случайные удары гасятся, повторяющийся рисунок остаётся. Сила 1.0 — удар на счёт."""
    hop = 128
    yp = librosa.effects.percussive(y, margin=2.0)
    oe = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)
    ot = librosa.times_like(oe, sr=sr, hop_length=hop)
    bt = np.asarray(beat_times)

    def at(t, w=0.03):
        m = (ot > t - w) & (ot < t + w)
        return float(oe[m].max()) if m.any() else 0.0

    rows = []
    for i in range(phase, len(bt) - beats_per_bar, beats_per_bar):
        # равномерная сетка на весь такт: трекер долей дёргает отдельные доли на синкопах
        b0, b1 = bt[i], bt[i + beats_per_bar]
        n = beats_per_bar * 4
        rows.append([at(b0 + j * (b1 - b0) / n) for j in range(n)])
    if len(rows) < 2:
        return {"subdiv": 4, "slots": [], "bars_matching": 0, "bars_total": 0, "variants": []}
    R = np.array(rows)
    loud = R.max(axis=1) > 0.3 * np.median(R.max(axis=1))  # такты, где вообще играют
    R = R[loud] / (np.median(R[loud][:, ::4]) + 1e-9)
    med = np.median(R, axis=0)
    hits = med > hit
    per_bar = (R > hit)
    matching = int((per_bar == hits).all(axis=1).sum())
    # нечётные шестнадцатые пустые → обычные восьмые
    sub = 4 if hits[1::2].any() else 2
    idx = range(0, len(med), 4 // sub)
    slots = []
    for i in idx:
        h = bool(hits[i])
        # маятник на выбранной сетке: вниз на чётных позициях, вверх на нечётных
        slots.append({"hit": h, "dir": ("down" if (i // (4 // sub)) % 2 == 0 else "up") if h else None,
                      "accent": bool(h and med[i] >= accent), "strength": round(float(med[i]), 2)})
    return {"subdiv": sub, "slots": slots, "bars_matching": matching, "bars_total": int(len(R)),
            "variants": [], "profile16": [round(float(v), 2) for v in med]}


def analyze(y: np.ndarray, sr: int, beat_times: list[float], phase: int) -> dict:
    ons = [o for o in onsets(y, sr) if o["strength"] >= 0.4]  # отсекаем звон и шум
    subdiv = choose_subdivision(ons, beat_times)
    grid(ons, beat_times, subdiv)
    acoustic_directions(y, sr, ons)
    for o in ons:
        if o.get("pos") is not None:
            o["dir_rule"] = "down" if o["pos"] % 2 == 0 else "up"
    pat = profile(y, sr, beat_times, phase)
    for o in ons:  # направление по маятнику на той сетке, которую выбрал профиль
        if o.get("pos") is not None and pat["subdiv"] != subdiv:
            o["dir_rule"] = None

    # насколько акустика согласна с «маятником»
    checked = [o for o in ons if o.get("dir_rule") and o["dir_audio"] != "?"]
    agree = sum(o["dir_audio"] == o["dir_rule"] for o in checked)
    pat["audio_agreement"] = round(agree / len(checked), 3) if checked else None
    pat["audio_checked"] = len(checked)
    return {"onsets": ons, "pattern": pat}


def render(pat: dict, beats_per_bar: int = 4) -> str:
    if not pat["slots"]:
        return "(рисунок боя не найден)"
    sub = pat["subdiv"]
    names = {2: ["{n}", "и"], 4: ["{n}", "е", "и", "а"]}[sub]
    count, arrows, acc = [], [], []
    for i, s in enumerate(pat["slots"]):
        beat, k = divmod(i, sub)
        count.append(names[k].format(n=beat + 1))
        arrows.append({"down": "↓", "up": "↑", None: "·"}[s["dir"]])
        acc.append(">" if s["accent"] else " ")
    w = 3
    line = lambda xs: "".join(x.ljust(w) for x in xs).rstrip()
    rows = ["Счёт:   " + line(count), "Бой:    " + line(arrows)]
    if any(a == ">" for a in acc):
        rows.append("Акцент: " + line(acc))
    return "\n".join(rows)


def short(pat: dict) -> str:
    """Запись вида «Д ДВ ВДВ» (Д — вниз, В — вверх, пробел — пропуск)."""
    if not pat["slots"]:
        return ""
    sub = pat["subdiv"]
    out = []
    for i, s in enumerate(pat["slots"]):
        if i and i % sub == 0:
            out.append(" ")
        out.append({"down": "Д", "up": "В", None: "-"}[s["dir"]])
    return "".join(out)
