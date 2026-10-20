"""Перебор по звуку: ноты (basic-pitch) → струны по аппликатуре аккорда → повторяющийся рисунок.

Бой и перебор различаются тем, сколько струн звучит разом. При бое на одну атаку приходится
3–6 струн, при переборе 1–2. Поэтому сначала распознаём отдельные ноты, затем решаем по каждому
такту, что там: бой или перебор. Рисунок перебора усредняем по всем тактам одного аккорда:
случайные ноты и ошибки распознавания гасятся, повторяющаяся схема остаётся.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import theory

OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # струны 6→1: E2 A2 D3 G3 B3 E4
FINGER = {1: "a", 2: "m", 3: "i"}  # классическая расстановка: басы — большой палец (p)


def transcribe(wav: str) -> list[dict]:
    """Ноты basic-pitch: [{'t', 'end', 'midi', 'amp'}]. Модель ONNX лежит в самом пакете."""
    import os

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    model = os.path.join(os.path.dirname(ICASSP_2022_MODEL_PATH), "nmp.onnx")
    if not os.path.exists(model):
        model = ICASSP_2022_MODEL_PATH
    _, _, notes = predict(wav, model, onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60,
                          minimum_frequency=70, maximum_frequency=1400)
    return sorted(({"t": round(float(s), 3), "end": round(float(e), 3), "midi": int(m), "amp": round(float(a), 3)}
                   for s, e, m, a, *_ in notes), key=lambda n: n["t"])


def drop_retriggers(notes: list[dict], y: np.ndarray, sr: int, min_rise_db: float = 3.0) -> list[dict]:
    """basic-pitch на каждом щипке заново «включает» все уже звенящие струны: прошлая нота D3
    заканчивается ровно там, где начинается новая D3, хотя щипнули только D4. Настоящий повторный
    щипок даёт скачок громкости на своей высоте, звон — нет. Такие продолжения выбрасываем."""
    import librosa

    hop = 128
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(OPEN_MIDI[0]),
                                                   n_bins=60, filter_scale=2)), ref=np.max)
    last_end: dict[int, float] = {}
    out = []
    for n in sorted(notes, key=lambda n: n["t"]):
        m, t = n["midi"], n["t"]
        cont = last_end.get(m, -1) >= t - 0.05
        last_end[m] = max(last_end.get(m, -1), n["end"])
        k = m - OPEN_MIDI[0]
        if cont and 0 <= k < C.shape[0]:
            f = int(t * sr / hop)
            pre, post = C[k, max(0, f - 8):max(1, f - 2)], C[k, f + 2:f + 10]
            if pre.size and post.size and post.mean() - pre.mean() < min_rise_db:
                continue
        out.append(n)
    return out


def events(notes: list[dict], min_amp: float = 0.3, together: float = 0.07) -> list[dict]:
    """Склеивает одновременные ноты в атаки и выбрасывает обертоны.

    Обертон — нота на октаву, дуодециму или две октавы выше другой, начавшейся в тот же момент
    и не тише её. Без этого открытая 4-я струна D3 выглядит как «D3 + D4 + A4», то есть как три струны."""
    ns = [n for n in notes if n["amp"] >= min_amp and n["midi"] >= OPEN_MIDI[0]]
    keep = [n for n in ns if not any(abs(m["t"] - n["t"]) < 0.06 and n["midi"] - m["midi"] in (12, 19, 24)
                                     and m["amp"] >= 0.7 * n["amp"] for m in ns)]
    out: list[dict] = []
    for n in keep:
        if out and n["t"] - out[-1]["t"] < together:
            out[-1]["midis"].add(n["midi"])
            out[-1]["amp"] = max(out[-1]["amp"], n["amp"])
        else:
            out.append({"t": n["t"], "midis": {n["midi"]}, "amp": n["amp"]})
    return out


def strings_for(midis: set[int], frets: list[int], capo: int) -> list[int]:
    """Номера струн (1 — тонкая) для звучащих нот при данной аппликатуре.
    Нота, которой нет в аппликатуре, ищется на ближайшей струне в пределах 4 ладов от формы."""
    sound = {6 - i: OPEN_MIDI[i] + capo + f for i, f in enumerate(frets) if f >= 0}
    res = set()
    for m in midis:
        exact = [s for s, p in sound.items() if p == m]
        if exact:
            res.add(exact[0])
            continue
        near = [(abs(OPEN_MIDI[6 - s] + capo + max(frets[6 - s], 0) - m), s) for s in range(1, 7)
                if 0 <= m - OPEN_MIDI[6 - s] - capo <= 12]
        if near:
            d, s = min(near)
            if d <= 4:
                res.add(s)
    return sorted(res, reverse=True)


def _shape_at(segs: list[dict], a: float, b: float) -> str:
    dur = Counter()
    for s in segs:
        ov = min(b, s["end"]) - max(a, s["start"])
        if ov > 0 and s.get("shape", s.get("chord")) not in (None, "N"):
            dur[s.get("shape", s["chord"])] += ov
    return dur.most_common(1)[0][0] if dur else "N"


def half_tempo(beats: list[float], phase: int, beats_per_bar: int, notes: list[dict],
               segs: list[dict], y: np.ndarray | None = None, sr: int | None = None,
               max_off: float = 0.15) -> tuple[list[float], int] | None:
    """Трекер долей на ровном переборе часто считает долей каждую восьмую: темп выходит вдвое больше,
    а такт режется пополам. Признак: на полудолях почти нет атак, а темп выше 100.
    Тогда берём каждую вторую долю; начало такта — та чётность, на которую чаще приходятся смены аккорда."""
    bt = np.asarray(beats, float)
    if len(bt) < 16:
        return None
    tempo = 60.0 / float(np.median(np.diff(bt)))
    ev = events(drop_retriggers(notes, y, sr) if y is not None else notes, min_amp=0.4)
    if tempo < 100 or len(ev) < 16:
        return None
    period = float(np.median(np.diff(bt)))
    ts = np.array([e["t"] for e in ev])
    i = np.clip(np.searchsorted(bt, ts) - 1, 0, len(bt) - 2)
    frac = (ts - bt[i]) / period
    off_beat = float(np.mean((frac > 0.3) & (frac < 0.7)))
    if off_beat > max_off:
        return None
    changes = [s["start"] for s in segs[1:] if s.get("shape", s.get("chord")) not in (None, "N")]
    best = None
    for parity in (0, 1):
        b2 = bt[parity::2]
        score = sum(np.min(np.abs(b2 - c)) < 0.12 for c in changes) if changes else 0
        if best is None or score > best[0]:
            best = (score, parity, b2)
    _, parity, b2 = best
    # сильная доля: прежнее начало такта, приведённое к новой сетке
    first_down = bt[phase] if phase < len(bt) else bt[0]
    j = int(np.argmin(np.abs(b2 - first_down)))
    return [round(float(t), 3) for t in b2], j % beats_per_bar


def _grid_offset(ev: list[dict], beats: np.ndarray, step_div: int = 2) -> float:
    """Систематический сдвиг сетки долей относительно атак. Трекер ставит долю на пик огибающей,
    а basic-pitch — на начало ноты; разница бывает до 0,1 с и сбивает позиции в такте."""
    if len(beats) < 3 or not ev:
        return 0.0
    period = float(np.median(np.diff(beats))) / step_div
    ts = np.array([e["t"] for e in ev])
    i = np.clip(np.searchsorted(beats, ts) - 1, 0, len(beats) - 2)
    ph = ((ts - beats[i]) / period) % 1.0
    ang = np.angle(np.exp(2j * np.pi * ph).mean())
    return float(ang / (2 * np.pi) * period)


def analyze(notes: list[dict], beats: list[float], phase: int, segs: list[dict], capo: int,
            beats_per_bar: int = 4, y: np.ndarray | None = None, sr: int | None = None,
            shapes: dict[str, list[int]] | None = None) -> dict:
    """Стиль каждого такта (перебор/бой/тихо), части с одинаковым стилем и рисунок перебора по аккордам."""
    bt = np.asarray(beats, dtype=float)
    raw = events(notes)  # стиль такта — по всем атакам: в ударе по аккорду звенят и старые струны
    ev = events(drop_retriggers(notes, y, sr)) if y is not None else raw
    off = _grid_offset(ev, bt)
    n16 = beats_per_bar * 4
    bars = []
    for i in range(phase, len(bt) - beats_per_bar, beats_per_bar):
        b0, b1 = bt[i] + off, bt[i + beats_per_bar] + off
        bb = bt[i:i + beats_per_bar + 1] + off  # позиции считаем от своей доли: внутри такта темп плавает
        step = (b1 - b0) / n16

        def slot_of(t, bb=bb):
            j = int(np.clip(np.searchsorted(bb, t + 0.05) - 1, 0, beats_per_bar - 1))
            x = (t - bb[j]) / (bb[j + 1] - bb[j]) * 4
            k = int(round(x))
            return None if abs(x - k) > 0.35 or not 0 <= j * 4 + k < n16 else j * 4 + k

        inside = lambda e, b0=b0, b1=b1, step=step: b0 - step / 2 <= e["t"] < b1 - step / 2
        shape = _shape_at(segs, b0, b1)
        if shapes and shape in shapes:  # аппликатура со схемы на экране — так играет сам автор
            frets = shapes[shape]
        else:
            sh = theory.shape_for(shape) if shape != "N" else None
            frets = sh["frets"] if sh else None
        slots: list[list[int] | None] = [None] * n16
        for e in filter(inside, ev):
            k = slot_of(e["t"])
            if k is None:
                continue
            strs = strings_for(e["midis"], frets, capo) if frets else []
            if strs:
                slots[k] = sorted(set((slots[k] or []) + strs), reverse=True)
        sizes = [len(e["midis"]) for e in filter(inside, ev)]
        # тип атаки на каждой позиции: Б — бас большим пальцем, А — аккорд (3+ струны), в — верхняя струна
        kinds: list[str | None] = [None] * n16
        for e in filter(inside, raw):
            k = slot_of(e["t"])
            if k is None:
                continue
            strs = strings_for(e["midis"], frets, capo) if frets else []
            kind = "А" if len(e["midis"]) >= 3 else "Б" if strs and min(strs) >= 4 else "в"
            if kinds[k] != "А":
                kinds[k] = kind
        bars.append({"t": round(float(b0), 2), "shape": shape, "sizes": sizes, "slots": slots, "kinds": kinds,
                     "bass": _bass_string(frets) if frets else None})

    smooth_styles(bars)
    secs = sections(bars)
    # рисунок перебора — отдельно для каждой части от 4 тактов: во вступлении и в концовке он бывает разный
    for sec in secs:
        if sec["style"] == "перебор" and sec["bars"] >= 4:
            inside_sec = [b for b in bars if b["style"] == "перебор" and sec["start"] <= b["t"] <= sec["end"]]
            sec["patterns"] = patterns(inside_sec, n16)
    return {"grid_offset": round(off, 3), "bars": bars, "sections": secs}


def strike_kinds(bars: list[dict], ranges: list[tuple[float, float]], n16: int = 16) -> list[dict | None]:
    """Для частей с боем: что чаще звучит на каждой шестнадцатой — бас, аккорд или верхняя струна."""
    bs = [b for b in bars if any(a - 0.1 <= b["t"] <= z + 0.1 for a, z in ranges)]
    out: list[dict | None] = []
    for k in range(n16):
        c = Counter(b["kinds"][k] for b in bs if b["kinds"][k])
        if not c:
            out.append(None)
            continue
        kind, n = c.most_common(1)[0]
        out.append({"kind": kind, "share": round(n / len(bs), 2)})
    return out


def smooth_styles(bars: list[dict], radius: int = 2) -> None:
    """Стиль такта по соседям: число ударов по 3+ струнам, медиана по пяти тактам.
    В части с боем попадаются такты с одним ударом и басами, во вступлении — случайный щипок аккорда;
    по одному такту их не отличить, по окрестности — легко."""
    counts = [sum(s >= 3 for s in b["sizes"]) for b in bars]
    for i, b in enumerate(bars):
        if len(b["sizes"]) < 3:
            b["style"] = "тихо"
            continue
        win = [c for j, c in enumerate(counts[max(0, i - radius):i + radius + 1])
               if len(bars[max(0, i - radius) + j]["sizes"]) >= 3]
        med = float(np.median(win)) if win else counts[i]
        b["style"] = "бой" if med >= 1.5 else "перебор" if med < 1 else "смешанно"


def classify(sizes: list[int]) -> str:
    """Бой — в такте минимум два удара по 3+ струнам разом (между ними бывают одиночные басы).
    Перебор — почти все атаки по одной-две струны."""
    if len(sizes) < 3:
        return "тихо"
    chords = sum(s >= 3 for s in sizes)
    if chords >= 2:
        return "бой"
    return "перебор" if chords == 0 or len(sizes) >= 6 else "смешанно"


def _bass_string(frets: list[int]) -> int:
    for i, f in enumerate(frets):
        if f >= 0:
            return 6 - i
    return 4


def sections(bars: list[dict], min_bars: int = 4) -> list[dict]:
    """Соседние такты одного стиля → части. Одиночный такт другого стиля считаем вариацией."""
    out: list[dict] = []
    for b in bars:
        st = b["style"] if b["style"] != "смешанно" else (out[-1]["style"] if out else "смешанно")
        if out and out[-1]["style"] == st:
            out[-1]["end"] = b["t"]
            out[-1]["bars"] += 1
        else:
            out.append({"style": st, "start": b["t"], "end": b["t"], "bars": 1})
    merged: list[dict] = []
    for s in out:
        if merged and s["bars"] < min_bars:
            merged[-1]["end"], merged[-1]["bars"] = s["end"], merged[-1]["bars"] + s["bars"]
        elif merged and merged[-1]["style"] == s["style"]:
            merged[-1]["end"], merged[-1]["bars"] = s["end"], merged[-1]["bars"] + s["bars"]
        else:
            merged.append(dict(s))
    return merged


def _cells(bs: list[dict], n16: int) -> list[dict | None]:
    cells = []
    for k in range(n16):
        c = Counter()
        for b in bs:
            for st in b["slots"][k] or []:
                c[st] += 1
        share = {st: n / len(bs) for st, n in c.items()}
        # щипок «бас + струна» записываем двумя струнами, только если обе звучат в большинстве тактов:
        # иначе звон предыдущей ноты, который распознаватель иногда видит как новую атаку, засоряет схему
        strs = sorted((st for st, v in share.items() if v >= 0.5), reverse=True)
        if not strs and share:
            st, v = max(share.items(), key=lambda kv: kv[1])
            strs = [st] if v >= 0.35 else []
        prev = next((c["strings"] for c in reversed(cells) if c), [])
        if len(strs) > 1:
            # струна с прошлой позиции, скорее всего, ещё звенит; в щипок её берём, только если она есть почти всегда
            strs = [x for x in strs if x not in prev or share[x] >= 0.8] or strs[-1:]
        cells.append({"strings": strs, "share": round(min(share[x] for x in strs), 2)} if strs else None)
    return cells


def _agreement(b: dict, cells: list[dict | None]) -> float:
    """Доля позиций рисунка, где в такте звучит та же верхняя струна."""
    want = [(k, c["strings"][-1]) for k, c in enumerate(cells) if c]
    if not want:
        return 0.0
    return sum(bool(b["slots"][k]) and min(b["slots"][k]) == st for k, st in want) / len(want)


def patterns(bars: list[dict], n16: int = 16, min_bars: int = 2) -> dict:
    """Для каждой басовой струны: на каждой позиции такта самые частые струны и доля тактов, где они там звучат.

    Два прохода: черновой рисунок по всем тактам, затем заново только по тактам, которые совпали с ним
    хотя бы наполовину. Так разговор перед песней, сбивки и концовка не портят схему."""
    # схема перебора зависит не от аккорда, а от струны, на которой бас: «бас на 5-й: 5-4-3-2»
    by: dict[int, list[dict]] = {}
    for b in bars:
        if b["shape"] != "N" and b["bass"]:
            by.setdefault(b["bass"], []).append(b)
    res = {}
    for bass, bs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(bs) < min_bars:
            continue
        cells = _cells(bs, n16)
        core = [b for b in bs if _agreement(b, cells) >= 0.5]
        if len(core) >= min_bars:
            cells = _cells(core, n16)
        # нечётные шестнадцатые почти пустые → восьмые
        odd = sum(1 for k in range(1, n16, 2) if cells[k])
        sub = 4 if odd >= 2 else 2
        step = 4 // sub
        slots = []
        for k in range(0, n16, step):
            c = cells[k]
            if c:
                c = dict(c, string=c["strings"][-1], finger="+".join(finger(x, bass) for x in c["strings"]))
            slots.append(c)
        grid_cells = [cells[k] if k % step == 0 else None for k in range(n16)]
        match = sum(_agreement(b, grid_cells) >= 0.75 for b in bs)
        filled = [c for c in slots if c]
        chords = [c for c, _ in Counter(b["shape"] for b in bs).most_common()]
        res[bass] = {"bass": bass, "chords": chords, "subdiv": sub, "slots": slots, "bars": len(bs),
                     "support": round(float(np.mean([c["share"] for c in filled])) if filled else 0.0, 2),
                     "bars_matching": int(match)}
    return res


def finger(string: int, bass: int | None) -> str:
    """p — басовые струны (от баса аккорда до 4-й), дальше i-m-a по 3-2-1-й."""
    if string >= 4 or (bass is not None and string >= bass):
        return "p"
    return FINGER[string]


def render(pat: dict, beats_per_bar: int = 4) -> str:
    sub = pat["subdiv"]
    names = {2: ["{n}", "и"], 4: ["{n}", "е", "и", "а"]}[sub]
    count, strs, fings = [], [], []
    for i, c in enumerate(pat["slots"]):
        beat, k = divmod(i, sub)
        count.append(names[k].format(n=beat + 1))
        strs.append("+".join(map(str, c["strings"])) if c else "·")
        fings.append(c["finger"] if c else " ")
    w = 4
    line = lambda xs: "".join(x.ljust(w) for x in xs).rstrip()
    return "\n".join(["Счёт:    " + line(count), "Струна:  " + line(strs), "Палец:   " + line(fings)])


def short(pat: dict) -> str:
    """«4-3-2-3 4-3-2-3»: струны по порядку, «·» — пауза."""
    sub = pat["subdiv"]
    per_half = sub * 2
    groups = []
    for i in range(0, len(pat["slots"]), per_half):
        groups.append("-".join("+".join(map(str, c["strings"])) if c else "·" for c in pat["slots"][i:i + per_half]))
    return " ".join(groups)


def tab(pat: dict, chord: str, frets: list[int] | None = None) -> str:
    """Один такт рисунка табулатурой на аппликатуре аккорда (лады относительно каподастра)."""
    sh = {"frets": frets} if frets else theory.shape_for(chord)
    if not sh:
        return ""
    names = ["e", "B", "G", "D", "A", "E"]  # 1→6
    w = 3
    rows = {s: names[s - 1] + "|" for s in range(1, 7)}
    for c in pat["slots"]:
        for s in range(1, 7):
            f = sh["frets"][6 - s]
            cell = str(f) if c and s in c["strings"] and f >= 0 else "-"
            rows[s] += cell.ljust(w, "-")
    return "\n".join(rows[s] + "|" for s in range(1, 7))
