"""Ноты и перебор.

Два независимых распознавателя нот по чистой гитаре:
  1) Basic Pitch (нейросеть Spotify, полифоническая транскрипция, ONNX);
  2) спектральный поток по каждому полутону (CQT) — простой детектор начала ноты.
Нота считается подтверждённой, если оба нашли её с разницей не больше 50 мс.

Затем каждая нота привязывается к струне по форме аккорда, которую держит левая рука
(с учётом каподастра), раскладывается по сетке такта, и собираются:
  - фигура перебора для каждого аккорда (какая струна на какой позиции такта и как часто);
  - полная табулатура по тактам с временем в ролике.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import librosa
import numpy as np

from . import theory

OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # индекс 0 = 6-я струна (E2) … 5 = 1-я (E4)
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]


# ───────────────────────── распознавание ─────────────────────────

_MODEL = None


def _bp_model():
    global _MODEL
    if _MODEL is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        onnx = os.path.join(os.path.dirname(ICASSP_2022_MODEL_PATH), "nmp.onnx")
        _MODEL = Model(onnx if os.path.exists(onnx) else ICASSP_2022_MODEL_PATH)
    return _MODEL


def transcribe(wav: str, onset_threshold: float = 0.5) -> list[dict]:
    """Basic Pitch → [{t, end, midi, amp, onset_p, frame_p}]. Модель ONNX лежит внутри пакета.

    melodia_trick выключен: он достраивает ноты из продолжающегося звука без начала ноты
    и на гитаре даёт разрывы звучащих нот и обертоны (проверено на эталоне с сэмплами)."""
    from basic_pitch.inference import predict

    out_model, _, raw = predict(wav, _bp_model(), onset_threshold=onset_threshold, frame_threshold=0.3,
                                minimum_note_length=60, minimum_frequency=75, maximum_frequency=1100,
                                melodia_trick=False)
    on, fr = out_model["onset"], out_model["note"]
    fps = 22050 / 256
    out = []
    for s, e, p, a, _ in raw:
        k = int(p) - 21
        i = int(round(float(s) * fps))
        j = max(i + 1, int(round(float(e) * fps)))
        onset_p = float(on[max(0, i - 2):i + 3, k].max()) if 0 <= k < on.shape[1] else 0.0
        frame_p = float(fr[i:j, k].mean()) if 0 <= k < fr.shape[1] and j > i else 0.0
        out.append({"t": float(s), "end": float(e), "midi": int(p), "amp": float(a),
                    "onset_p": round(onset_p, 3), "frame_p": round(frame_p, 3)})
    out.extend(_repluck_candidates(out, on, fr, fps))
    return sorted(out, key=lambda n: (n["t"], n["midi"]))


def _repluck_candidates(notes: list[dict], on: np.ndarray, fr: np.ndarray, fps: float,
                        min_gap: float = 0.12, min_peak: float = 0.15) -> list[dict]:
    """Повторный щипок звучащей ноты: Basic Pitch не начинает новую ноту, пока та же высота
    звучит, хотя в его выходе виден пик уверенности начала. Такие пики — кандидаты; настоящие
    они или нет, решает фильтр нот."""
    from scipy.signal import find_peaks
    extra = []
    for n in list(notes):
        k = n["midi"] - 21
        if not 0 <= k < on.shape[1] or n["end"] - n["t"] < 2 * min_gap:
            continue
        a, b = int((n["t"] + min_gap) * fps), int((n["end"] - 0.03) * fps)
        if b <= a:
            continue
        pk, props = find_peaks(on[a:b, k], height=min_peak, distance=int(min_gap * fps))
        for i, h in zip(pk, props["peak_heights"]):
            t = (a + i) / fps
            extra.append({"t": float(t), "end": float(n["end"]), "midi": n["midi"], "amp": float(fr[a + i, k]),
                          "onset_p": round(float(h), 3), "frame_p": round(float(fr[a + i:b, k].mean()), 3),
                          "repluck": True})
    return extra


def flux_notes(y: np.ndarray, sr: int, lo: int = 40, hi: int = 84, hop: int = 128) -> list[dict]:
    """Независимый детектор: рост энергии в каждом полутоне (CQT, 3 бина на полутон)."""
    n_bins = (hi - lo + 1) * 3
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(lo) * 2 ** (-1 / 36),
                           n_bins=n_bins, bins_per_octave=36, filter_scale=1.5))
    C = C.reshape(hi - lo + 1, 3, -1).max(axis=1)
    L = np.log1p(30 * C / (C.max() + 1e-9))
    rise = np.maximum(0, np.diff(L, axis=1, prepend=L[:, :1]))
    rise = np.apply_along_axis(lambda r: np.convolve(r, np.ones(5), "same"), 1, rise)
    # подавляем обертоны: рост в полутоне засчитываем, только если он не меньше 60% роста
    # на октаву/дуодециму ниже в тот же момент (иначе это гармоника нижней ноты)
    out = []
    from scipy.signal import find_peaks
    thr = np.percentile(rise[rise > 0], 90) if (rise > 0).any() else 1.0
    for k in range(rise.shape[0]):
        pk, _ = find_peaks(rise[k], height=thr * 0.6, distance=int(0.09 * sr / hop))
        for p in pk:
            h = rise[k, p]
            lower = [rise[k - d, max(0, p - 2):p + 3].max() for d in (12, 19, 24) if k - d >= 0]
            if lower and h < 0.6 * max(lower):
                continue
            out.append({"t": float(p * hop / sr), "midi": lo + k, "amp": float(h)})
    return sorted(out, key=lambda n: (n["t"], n["midi"]))


def cross_check(a: list[dict], b: list[dict], tol: float = 0.05) -> dict:
    """Помечает ноты из `a`, которые нашёл и второй метод (та же высота, разница ≤ tol)."""
    by_pitch = defaultdict(list)
    for n in b:
        by_pitch[n["midi"]].append(n["t"])
    for k in by_pitch:
        by_pitch[k] = np.array(sorted(by_pitch[k]))
    hit = 0
    for n in a:
        ts = by_pitch.get(n["midi"])
        ok = ts is not None and len(ts) and np.min(np.abs(ts - n["t"])) <= tol
        n["confirmed"] = bool(ok)
        hit += ok
    used = sum(1 for n in b if any(abs(n["t"] - m["t"]) <= tol and n["midi"] == m["midi"] for m in a))
    return {"a": len(a), "b": len(b), "a_confirmed": hit,
            "a_share": round(hit / max(1, len(a)), 3), "b_share": round(used / max(1, len(b)), 3)}


# ───────────────────────── струны ─────────────────────────

def _shape_frets(shape_name: str) -> list[int] | None:
    sh = theory.shape_for(shape_name)
    return sh["frets"] if sh else None


def seg_at(t: float, segs: list[dict]) -> dict | None:
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s
    return None


def assign_strings(notes: list[dict], segs: list[dict], capo: int, lead: float = 0.12) -> None:
    """Струна и лад для каждой ноты по форме аккорда.

    Порядок: нота лежит в форме текущего аккорда → в форме соседнего (нота чуть раньше или позже
    смены: пальцы уже переставлены или ещё не сняты) → открытая струна, которая в форме играется
    (вариант аккорда, например G с открытой 1-й) → ближайший к форме лад (нота вне аккорда).
    """
    for n in notes:
        # каждый проход начинаем с высоты, которую дал распознаватель (прошлый проход мог
        # «исправить» октаву под другой аккорд)
        if "midi_raw" in n:
            n["midi"] = n.pop("midi_raw")
            n.pop("octave_fixed", None)
        cands = []
        for dt in (0.0, lead, -lead):
            s = seg_at(n["t"] + dt, segs)
            if s and s.get("shape") not in (None, "N") and s not in cands:
                cands.append(s)
        n.update(string=None, fret=None, in_shape=False, variant=False, shape=cands[0]["shape"] if cands else None)
        shapes = [(s["shape"], _shape_frets(s["shape"])) for s in cands]
        shapes = [(nm, fr) for nm, fr in shapes if fr]
        done = False
        for nm, frets in shapes:
            for i, f in enumerate(frets):
                if f >= 0 and OPEN_MIDI[i] + capo + f == n["midi"]:
                    n.update(string=6 - i, fret=f, in_shape=True, shape=nm)
                    done = True
                    break
            if done:
                break
        if done:
            continue
        for nm, frets in shapes:
            for i in range(6):
                if OPEN_MIDI[i] + capo == n["midi"] and frets[i] >= 0:
                    n.update(string=6 - i, fret=0, in_shape=True, variant=True, shape=nm)
                    done = True
                    break
            if done:
                break
        if done or not shapes:
            continue
        # ошибка октавы распознавателя: на октаву выше/ниже нота ложится в форму
        for nm, frets in shapes:
            for d in (12, -12):
                for i, f in enumerate(frets):
                    if f >= 0 and OPEN_MIDI[i] + capo + f == n["midi"] + d:
                        n.update(string=6 - i, fret=f, in_shape=True, shape=nm, octave_fixed=d,
                                 midi_raw=n["midi"], midi=n["midi"] + d)
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if done:
            continue
        nm, frets = shapes[0]
        used = [f for f in frets if f > 0] or [1]
        center = float(np.mean(used))
        best = None
        for i in range(6):
            f = n["midi"] - OPEN_MIDI[i] - capo
            if 0 <= f <= 15:
                cost = abs(f - center) + (0.5 if frets[i] < 0 else 0)
                if best is None or cost < best[0]:
                    best = (cost, i, f)
        if best:
            n.update(string=6 - best[1], fret=best[2], in_shape=False, shape=nm)


def mark_octave_ghosts(notes: list[dict], tol: float = 0.035) -> int:
    """Нота ровно на октаву выше одновременной басовой — часто обертон, а не сыгранная нота.
    Помечаем `ghost`, если её не подтвердил второй метод (у него подавление обертонов)."""
    cnt = 0
    for n in notes:
        n["ghost"] = False
        for m in notes:
            if m is n or abs(m["t"] - n["t"]) > tol:
                continue
            if (n["midi"] - m["midi"] in (12, 19, 24, 28, 31) and m["amp"] >= 0.7 * n["amp"]
                    and (not n.get("confirmed", False) or not n.get("in_shape", True))):
                n["ghost"] = True
                cnt += 1
                break
    return cnt


# ───────────────────────── сетка ─────────────────────────

def grid_offset(notes: list[dict], beats: np.ndarray) -> float:
    """Медианный сдвиг нот относительно долей (beat_track часто отстаёт на 20–40 мс)."""
    period = float(np.median(np.diff(beats)))
    d = []
    for n in notes:
        i = np.searchsorted(beats, n["t"])
        for j in (i - 1, i):
            if 0 <= j < len(beats) and abs(n["t"] - beats[j]) < 0.25 * period:
                d.append(n["t"] - beats[j])
    return float(np.median(d)) if d else 0.0


def subdivision(notes: list[dict], beats: np.ndarray) -> int:
    """Сколько позиций на долю: самое маленькое k, при котором ноты ложатся на сетку."""
    pos = np.interp([n["t"] for n in notes], beats, np.arange(len(beats)))
    best = 1
    for k in (1, 2, 3, 4):
        err = np.abs(pos * k - np.round(pos * k)) / k  # в долях доли
        share = float(np.mean(err < 0.16 / max(1, k / 2)))
        if share >= 0.85:
            return k
        best = k
    return best


def meter(beat_chords: list[str], notes: list[dict], beats: np.ndarray, per_beat: int = 1) -> tuple[int, int]:
    """Долей в такте и фаза первой доли.

    Два признака:
      - фигура перебора по струнам повторяется каждый такт, какой бы ни был аккорд
        (струны уже привязаны к форме, так что фигура от аккорда не зависит);
      - смены аккордов стоят на первой доле.
    Счёт = похожесть фигуры на себя через такт + выравнивание смен аккордов сверх случайного;
    при равенстве — более короткий такт (фигура с периодом L совпадает и через 2L), а если все
    смены аккордов стоят только на каждом втором таком такте — удваиваем."""
    beat = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    pos = np.interp([n["t"] for n in notes], beats, np.arange(len(beats))) * per_beat
    Q = int(len(beats) * per_beat) + 2
    S = np.zeros((Q, 6), bool)
    for n, q in zip(notes, pos):
        if n.get("string") and 0 <= round(q) < Q:
            S[int(round(q)), n["string"] - 1] = True
    changes = [i for i in range(1, len(beat_chords)) if beat_chords[i] != beat_chords[i - 1]]
    use_ch = len(changes) >= 3
    bass = np.zeros(len(beats))
    for n in notes:
        if n.get("string") and n["string"] >= 4:
            bass[int(np.argmin(np.abs(beats - n["t"])))] += n["amp"]
    cand = {}
    for nb in (2, 3, 4, 6, 8, 12):
        dur = nb * beat
        if not 0.7 <= dur <= 5.0:
            continue
        L = nb * per_beat
        inter = (S[:-L] & S[L:]).sum()
        union = (S[:-L] | S[L:]).sum()
        sim = inter / union if union else 0.0
        best_ph = max(range(nb), key=lambda ph: (
            np.mean([(c - ph) % nb == 0 for c in changes]) if use_ch else 0.0,
            bass[ph::nb].mean() / (bass.mean() + 1e-9) if bass.any() else 0.0))
        ch = np.mean([(c - best_ph) % nb == 0 for c in changes]) if use_ch else 1.0
        cand[nb] = {"sim": float(sim), "ch": float(ch), "ph": best_ph}
    if not cand:
        return 4, 0
    # выравнивание смен аккордов сверх случайного: в такте из nb долей случайная смена попадает
    # на первую долю с вероятностью 1/nb, поэтому короткие такты «выравниваются» даром
    for nb, v in cand.items():
        v["excess"] = (v["ch"] - 1 / nb) / (1 - 1 / nb) if use_ch else 0.0
        v["score"] = v["sim"] + 0.5 * v["excess"]
    best = max(v["score"] for v in cand.values())
    nb = min(k for k, v in cand.items() if v["score"] >= best - 0.03)
    ok = cand
    # все смены аккордов приходятся только на каждый второй «такт» — показываем такт вдвое длиннее,
    # чтобы в такте был один аккорд (3/4 с аккордом на два такта → 6/8)
    while use_ch and 2 * nb in cand and cand[2 * nb]["ch"] >= cand[nb]["ch"] - 0.02:
        nb = 2 * nb
    return nb, ok.get(nb, cand[nb])["ph"]


def place(notes: list[dict], beats: np.ndarray, per_beat: int, beats_per_bar: int, phase: int) -> None:
    """Номер такта и позиция в такте (0 … beats_per_bar*per_beat-1) для каждой ноты."""
    pos = np.interp([n["t"] for n in notes], beats, np.arange(len(beats)))
    slots = beats_per_bar * per_beat
    for n, p in zip(notes, pos):
        q = int(round((p - phase) * per_beat))
        n["bar"] = q // slots
        n["slot"] = q % slots
        n["slot_err"] = float(abs((p - phase) * per_beat - round((p - phase) * per_beat)))


def bar_chords(notes: list[dict], beats, per_beat, beats_per_bar, phase, segs) -> dict[int, str]:
    out = {}
    nbar = int(np.ceil((len(beats) - phase) / beats_per_bar)) + 1
    for b in range(-1, nbar):
        i0 = phase + b * beats_per_bar
        i1 = i0 + beats_per_bar
        if i0 < 0 or i0 >= len(beats):
            continue
        t0 = beats[i0]
        t1 = beats[min(i1, len(beats) - 1)] if i1 < len(beats) else t0 + (beats[-1] - beats[-2]) * beats_per_bar
        votes = Counter()
        for s in segs:
            ov = min(t1, s["end"]) - max(t0, s["start"])
            if ov > 0:
                votes[s.get("shape")] += ov
        if votes:
            out[b] = votes.most_common(1)[0][0]
    return out


def chords_from_notes(notes: list[dict], beats, beats_per_bar: int, phase: int, capo: int) -> list[dict]:
    """Аккорд каждого такта по распознанным нотам (третий, независимый от хромы способ).
    Вес ноты = громкость × длительность. Счёт = совпадение нот с аккордом + доля нот,
    которые физически ложатся в его аппликатуру на грифе."""
    labels, T = theory.templates(("", "m", "7", "m7"))
    out = []
    nbar = (len(beats) - phase) // beats_per_bar + 1
    for b in range(-1, nbar):
        i0 = phase + b * beats_per_bar
        if i0 < 0 or i0 >= len(beats):
            continue
        q = 0.25 * float(np.median(np.diff(beats)))  # живые ноты приходят чуть раньше доли
        t0 = beats[i0] - q
        i1 = i0 + beats_per_bar
        t1 = (beats[i1] if i1 < len(beats) else beats[i0] + float(np.median(np.diff(beats))) * beats_per_bar) - q
        pc = np.zeros(12)
        low = None
        bassw: dict[int, float] = {}
        for n in notes:
            ov = min(t1, n.get("end", n["t"] + 0.3)) - max(t0, n["t"])
            if ov <= 0 or not (t0 <= n["t"] < t1):
                continue
            w = n["amp"] * min(ov, 1.0)
            pc[n["midi"] % 12] += w
            if n["midi"] < 55:
                bassw[n["midi"]] = bassw.get(n["midi"], 0.0) + w
        onsets = sum(1 for n in notes if t0 <= n["t"] < t1)
        if pc.sum() == 0 or onsets < 3:
            continue  # гитара не играет (звенит последний аккорд, пауза) — аккорд не выводим
        v = pc / np.linalg.norm(pc)
        sc = T @ v
        # гитарная проверка: какая доля звучащих нот физически ложится в аппликатуру кандидата
        # (открытая струна, которая в форме играется, засчитывается как вариант с весом 0.7)
        bar_notes = [(n["midi"], n["amp"] * min(1.0, n.get("end", n["t"] + 0.3) - n["t"]))
                     for n in notes if t0 <= n["t"] < t1]
        # бас весит втрое: его струну пальцы физически либо зажимают, либо нет
        mult = [3.0 if m < 52 else 1.0 for m, _ in bar_notes]
        tot = sum(w * k for (_, w), k in zip(bar_notes, mult)) or 1.0
        for i, lab in enumerate(labels):
            sh = theory.shape_for(theory.transpose(lab, -capo))
            if not sh:
                continue
            fr = sh["frets"]
            exact = {OPEN_MIDI[k] + capo + f for k, f in enumerate(fr) if f >= 0}
            var = {OPEN_MIDI[k] + capo for k, f in enumerate(fr) if f >= 0}
            muted_open = {OPEN_MIDI[k] + capo for k, f in enumerate(fr) if f < 0}
            fit = 0.0
            for (m, w), k in zip(bar_notes, mult):
                if m in exact:
                    fit += w * k
                elif m in var:
                    fit += 0.7 * w * k  # открытая струна, которая в форме играется
                elif m in muted_open:
                    fit += 0.4 * w * k  # задели открытую струну, которую форма глушит
            sc[i] += 0.5 * fit / tot
            # с каподастром и без гитаристы почти всегда держат открытые формы, баррэ — реже
            if sh.get("barre"):
                sc[i] -= 0.06
        # септаккорд только если заметно лучше трезвучия
        best = int(np.argmax(sc))
        lab = labels[best]
        if lab.endswith("7"):
            tri = lab[:-1]
            if tri in labels and sc[labels.index(tri)] > sc[best] - 0.04:
                lab = tri
        r0 = theory.parse(lab)[0]
        thirds = pc[(r0 + 3) % 12] + pc[(r0 + 4) % 12]
        out.append({"start": float(t0), "end": float(t1), "chord": lab, "bar": b,
                    "score": round(float(sc.max()), 3), "no_third": bool(thirds < 0.08 * pc.sum())})
    # без терции (квинта, «пустой» аккорд) берём лад из остальной песни
    qual = Counter()
    for o in out:
        if not o["no_third"]:
            qual[theory.parse(o["chord"])] += 1
    for o in out:
        if o["no_third"]:
            root = theory.parse(o["chord"])[0]
            same_root = [(k, v) for k, v in qual.items() if k[0] == root]
            if same_root:
                o["chord"] = theory.name(*max(same_root, key=lambda kv: kv[1])[0])
        o["shape"] = theory.transpose(o["chord"], -capo)
    return out


def compare_chords(note_bars: list[dict], segs: list[dict]) -> dict:
    """Совпадение аккордов по нотам и по хроме (по времени)."""
    agree, total, diffs = 0.0, 0.0, []
    for b in note_bars:
        d = b["end"] - b["start"]
        votes = Counter()
        for s in segs:
            ov = min(b["end"], s["end"]) - max(b["start"], s["start"])
            if ov > 0:
                votes[s["chord"]] += ov
        if not votes:
            continue
        other = votes.most_common(1)[0][0]
        same = theory.same(other, b["chord"])
        total += d
        if same in ("exact", "triad"):
            agree += d
        else:
            diffs.append({"bar": b["bar"], "t": round(b["start"], 2), "notes": b["chord"], "chroma": other})
    return {"share": round(agree / max(total, 1e-9), 3), "diffs": diffs}


# ───────────────────────── фигура и табы ─────────────────────────

def is_picking(notes: list[dict], tol: float = 0.06) -> dict:
    """Перебор или бой: в бою за 60 мс звучат 4+ струны (удар по 6 струнам длится 30–60 мс),
    в переборе 1–2."""
    ts = sorted(n["t"] for n in notes)
    groups, cur = [], []
    for t in ts:
        if cur and t - cur[0] > tol:
            groups.append(len(cur))
            cur = []
        cur.append(t)
    if cur:
        groups.append(len(cur))
    g = np.array(groups) if groups else np.array([0])
    big = float(np.mean(g >= 4))
    return {"picking": big < 0.3, "share_4plus": round(big, 3), "mean_group": round(float(g.mean()), 2)}


def patterns(notes: list[dict], bar_chord: dict[int, str], slots: int, min_share: float = 0.5) -> dict:
    """Для каждого аккорда: на какой позиции какая струна и в какой доле тактов."""
    bars_of = defaultdict(set)
    for b, ch in bar_chord.items():
        bars_of[ch].add(b)
    res = {}
    for ch, bars in bars_of.items():
        if not ch or ch == "N":
            continue
        cnt = defaultdict(set)
        frets = defaultdict(Counter)
        for n in notes:
            if (n.get("bar") in bars and n.get("string") and not n.get("ghost") and n["slot_err"] < 0.3
                    and n.get("in_shape") and not n.get("dropped")):
                cnt[(n["string"], n["slot"])].add(n["bar"])
                frets[(n["string"], n["slot"])][n["fret"]] += 1
        nb = len(bars)
        need = 1.0 if nb < 3 else min_share
        share = {k: len(v) / nb for k, v in cnt.items()}
        # лад — самый частый на этой позиции (в начале такта бывают «хвосты» прошлого аккорда)
        keep = {k: frets[k].most_common(1)[0][0] for k, s in share.items() if s >= need - 1e-9}
        res[ch] = {"bars": nb, "share": {f"{k[0]}:{k[1]}": round(v, 2) for k, v in share.items()},
                   "tab": {f"{k[0]}:{k[1]}": v for k, v in keep.items()}, "need": need}
    return res


def _tab_lines(cells: dict, slots: int, per_beat: int) -> list[str]:
    """cells: {(string, slot): fret} или {"string:slot": fret}. 6 строк табулатуры (e … E)."""
    cells = {tuple(map(int, k.split(":"))) if isinstance(k, str) else k: v for k, v in cells.items()}
    lines = []
    for s in range(1, 7):
        row = []
        for i in range(slots):
            f = cells.get((s, i))
            row.append("-" if f is None else str(f))
        w = max(len(x) for x in row)
        lines.append(f"{STRING_NAMES[6 - s]}|-" + "-".join(x.rjust(w, "-") for x in row) + "-|")
    return lines


def count_header(slots: int, per_beat: int) -> str:
    marks = []
    for i in range(slots):
        if i % per_beat == 0:
            marks.append(str(i // per_beat + 1))
        else:
            marks.append({2: "и", 3: "и", 4: "е"}.get(per_beat, "·") if per_beat != 4 or i % 2 else "и")
    return "   " + " ".join(marks)


def full_tab(notes: list[dict], beats, per_beat, beats_per_bar, phase, bar_chord: dict) -> list[dict]:
    slots = per_beat * beats_per_bar
    bars = defaultdict(dict)
    for n in notes:
        if n.get("string") and not n.get("ghost") and n["slot_err"] < 0.3:
            bars[n["bar"]][(n["string"], n["slot"])] = n["fret"]
    out = []
    for b in sorted(set(bars) | set(bar_chord)):
        i0 = phase + b * beats_per_bar
        if i0 < 0 or i0 >= len(beats):
            continue
        out.append({"bar": b, "t": float(beats[i0]), "chord": bar_chord.get(b, "?"),
                    "lines": _tab_lines(bars.get(b, {}), slots, per_beat)})
    return out


def refine_with_pattern(notes, rejected, bchord, slots, beats, per_beat, bpb, phase, note_bars, capo,
                        thr: float, rescue_min: float = 0.005, keep_factor: float = 1.6) -> dict:
    """Второй проход с учётом повторяемости фигуры.

    Если на позиции, где в большинстве тактов этого аккорда звучит струна, нота не найдена,
    ищем среди отброшенных кандидатов ровно эту струну и лад на этой позиции. Совпадение
    струна+лад+позиция с фигурой — сильное свидетельство, поэтому порог низкий (rescue_min):
    фильтр без контекста часто гасит вторую ноту щипка вместе с басом. Ноту вне фигуры оставляем, только если уверенность ≥ keep_factor·порога
    (или фильтра нет). Ничего не придумываем: решение только по реально найденным кандидатам."""
    pat = patterns(notes, bchord, slots)
    cells = {ch: {tuple(map(int, k.split(":"))) for k in p["tab"]} for ch, p in pat.items()}
    have = defaultdict(set)
    for n in notes:
        if n.get("string"):
            have[n["bar"]].add((n["string"], n["slot"]))
    assign_strings(rejected, note_bars, capo)
    place(rejected, beats, per_beat, bpb, phase)
    rescued = []
    for n in sorted(rejected, key=lambda n: -(n.get("p_real") or 0)):
        ch = bchord.get(n.get("bar"))
        key = (n.get("string"), n.get("slot"))
        if (ch in cells and key in cells[ch] and key not in have[n["bar"]] and n.get("in_shape")
                and n["slot_err"] < 0.3 and (n.get("p_real") or 0) >= rescue_min):
            n["rescued"] = True
            n["ghost"] = False
            have[n["bar"]].add(key)
            rescued.append(n)
    # ошибка октавы на найденной ноте: клетка фигуры пустая, а на той же позиции нота ровно
    # на октаву выше/ниже, которой в фигуре нет — это та же струна
    for n in notes:
        ch = bchord.get(n.get("bar"))
        key = (n.get("string"), n.get("slot"))
        if ch not in cells or key in cells[ch] or n.get("octave_fixed"):
            continue
        frets = _shape_frets(ch)
        if not frets:
            continue
        for (s_, slot) in cells[ch]:
            if slot != n["slot"] or (s_, slot) in have[n["bar"]]:
                continue
            f = frets[6 - s_]
            want = OPEN_MIDI[6 - s_] + capo + f
            if f >= 0 and abs(want - n["midi"]) == 12:
                have[n["bar"]].discard(key)
                n.update(octave_fixed=want - n["midi"], midi_raw=n["midi"], midi=want, string=s_, fret=f)
                have[n["bar"]].add((s_, slot))
                break
    dropped = 0
    for n in notes:
        ch = bchord.get(n.get("bar"))
        key = (n.get("string"), n.get("slot"))
        if ch in cells and pat[ch]["bars"] >= 3 and key not in cells[ch] and n.get("p_real") is not None \
                and n["p_real"] < min(0.95, keep_factor * thr):
            n["dropped"] = True
            dropped += 1
    return {"rescued": rescued, "dropped": dropped}


def write_midi(notes: list[dict], path: str) -> None:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24, name="guitar")
    for n in notes:
        end = n.get("end") or n["t"] + 0.2
        inst.notes.append(pretty_midi.Note(velocity=int(40 + 80 * min(1, n["amp"])), pitch=n["midi"],
                                           start=n["t"], end=max(end, n["t"] + 0.05)))
    pm.instruments.append(inst)
    pm.write(path)


def analyze(wav: str, res: dict, segs: list[dict], capo: int) -> dict:
    """Полный разбор нот: распознавание двумя методами, сверка, сетка, аккорды по нотам,
    струны, фигуры, табы."""
    from . import notefilter

    y, sr = librosa.load(wav, sr=22050, mono=True)
    raw = transcribe(wav)
    nf = notefilter.apply(raw, y, sr)
    bp = [n for n in raw if n.get("real", True)]
    fx = flux_notes(y, sr)
    check = cross_check(bp, fx)
    beats = np.array(res["beats"], dtype=float)
    off = grid_offset(bp, beats)
    beats = beats + off
    per_beat = subdivision(bp, beats)
    # первая привязка к струнам — по аккордам хромы, чтобы найти бас для определения такта
    assign_strings(bp, segs, capo)
    bpb, phase = meter(res.get("beat_chords") or [], bp, beats, per_beat)
    note_bars = chords_from_notes(bp, beats, bpb, phase, capo)
    vs_chroma = compare_chords(note_bars, segs)
    # окончательная привязка — по аккордам, найденным по самим нотам
    assign_strings(bp, note_bars, capo)
    if nf.get("model"):
        for n in bp:
            n["ghost"] = False  # обертоны уже отсеял обученный фильтр
        ghosts = len(raw) - len(bp)
    else:
        ghosts = mark_octave_ghosts(bp)
    place(bp, beats, per_beat, bpb, phase)
    bchord = {b["bar"]: b["shape"] for b in note_bars}
    slots = per_beat * bpb
    rejected = [n for n in raw if not n.get("real", True)]
    refine = refine_with_pattern(bp, rejected, bchord, slots, beats, per_beat, bpb, phase, note_bars, capo,
                                 nf.get("threshold") or 0.5)
    bp = [n for n in bp if not n.get("dropped")] + refine["rescued"]
    bp.sort(key=lambda n: (n["t"], n["midi"]))
    return {
        "notes": bp, "raw_count": len(raw), "filter": nf, "rejected": [n for n in raw if not n.get("real", True)],
        "refine": {"rescued": len(refine["rescued"]), "dropped": refine["dropped"]},
        "flux_count": len(fx), "cross_check": check, "ghosts": ghosts,
        "grid_offset": round(off, 3), "per_beat": per_beat, "beats_per_bar": bpb, "phase": phase,
        "slots": slots, "note_bars": note_bars, "chords_vs_chroma": vs_chroma,
        "kind": is_picking(raw), "patterns": patterns(bp, bchord, slots),
        "full_tab": full_tab(bp, beats, per_beat, bpb, phase, bchord), "beats_shifted": beats.tolist(),
    }


def apply_video_chords(result: dict, checks: list[dict], capo: int, min_share: float = 0.6,
                       rescued: list[dict] | None = None) -> list[dict]:
    """Исправляет аккорд такта, если группа тактов с той же формой левой руки на видео
    уверенно называет другой аккорд, и добавляет ноты, возвращённые по движению пальцев.
    Возвращает список исправлений аккордов."""
    if rescued:
        ids = {id(n) for n in result["notes"]}
        result["notes"].extend(n for n in rescued if id(n) not in ids)
        result["notes"].sort(key=lambda n: (n["t"], n["midi"]))
        result["rejected"] = [n for n in result["rejected"] if not n.get("rescued_video")]
        result["video_rescued"] = len(rescued)
    fixes = []
    for b, c in zip(result["note_bars"], checks):
        b["video"] = c.get("video")
        if c.get("video") and c.get("group_share") and c["group_share"] >= min_share and c["video"] != b["chord"]:
            fixes.append({"bar": b["bar"], "t": round(b["start"], 2), "audio": b["chord"], "video": c["video"]})
            b["chord_audio"] = b["chord"]
            b["chord"] = c["video"]
            b["shape"] = theory.transpose(c["video"], -capo)
            b["source"] = "видео"
    if fixes or rescued:
        notes = result["notes"]
        assign_strings(notes, result["note_bars"], capo)
        if not (result.get("filter") or {}).get("model"):
            mark_octave_ghosts(notes)
        beats = np.array(result["beats_shifted"])
        place(notes, beats, result["per_beat"], result["beats_per_bar"], result["phase"])
        bchord = {b["bar"]: b["shape"] for b in result["note_bars"]}
        result["patterns"] = patterns(notes, bchord, result["slots"])
        result["full_tab"] = full_tab(notes, beats, result["per_beat"], result["beats_per_bar"], result["phase"], bchord)
    result["video_fixes"] = fixes
    return fixes


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def report_md(r: dict, hands_res: dict | None = None, capo: int = 0) -> str:
    """Раздел гайда: перебор, фигуры по аккордам, полная табулатура, как проверено."""
    from .hands import FINGER_RU
    L = []
    kind = r["kind"]
    L.append("## Ноты и перебор\n")
    if kind["picking"]:
        L.append(f"Правая рука играет **перебором**: в среднем {kind['mean_group']} ноты звучат одновременно, "
                 f"аккорд из 4+ струн — только в {kind['share_4plus'] * 100:.0f}% моментов.\n")
    else:
        L.append(f"Правая рука играет **боем**: в {kind['share_4plus'] * 100:.0f}% моментов вместе звучат 4+ струны.\n")
    L.append(f"Сетка: {r['beats_per_bar']} долей в такте, по {r['per_beat']} позиции на долю "
             f"({r['slots']} позиций в такте). Лады считаются от каподастра.\n")
    L.append("### Фигура по аккордам\n")
    L.append("Нота попадает в фигуру, если звучит на этой позиции хотя бы в половине тактов с этим аккордом "
             "(если таких тактов меньше трёх — во всех).\n")
    L.append("```")
    for ch, p in sorted(r["patterns"].items(), key=lambda kv: -kv[1]["bars"]):
        L.append(f"{ch}  ({p['bars']} такт.)")
        L.append(count_header(r["slots"], r["per_beat"]))
        L.extend(_tab_lines(p["tab"], r["slots"], r["per_beat"]))
        L.append("")
    L.append("```\n")
    L.append("### Полная табулатура по тактам\n")
    L.append("Слева время начала такта в ролике и аккорд.\n")
    L.append("```")
    L.append(count_header(r["slots"], r["per_beat"]))
    for b in r["full_tab"]:
        L.append(f"{_mmss(b['t'])} {b['chord']}")
        L.extend(b["lines"])
        L.append("")
    L.append("```\n")
    L.append("### Как проверено\n")
    L.append("| Что | Метод | Результат |")
    L.append("|---|---|---|")
    f = r.get("filter") or {}
    L.append(f"| Ноты | Basic Pitch (нейросеть Spotify) | найдено {r['raw_count']}" +
             (f", фильтр артефактов оставил {f['kept']}" if f.get("model") else "") +
             (f", второй проход по фигуре: +{r['refine']['rescued']} / −{r['refine']['dropped']}" if r.get("refine") else "")
             + " |")
    cc = r["cross_check"]
    L.append(f"| Ноты, второй метод | рост энергии по каждому полутону (CQT) | {cc['a_share'] * 100:.0f}% нот "
             f"подтверждены с точностью 50 мс |")
    vc = r["chords_vs_chroma"]
    L.append(f"| Аккорды | по нотам такта и аппликатуре vs по хроме | совпало {vc['share'] * 100:.0f}% времени" +
             (", расхождения: " + ", ".join(f"{_mmss(d['t'])} {d['notes']}/{d['chroma']}" for d in vc["diffs"][:6])
              if vc["diffs"] else "") + " |")
    if hands_res:
        checks = hands_res["bar_checks"]
        known = [c for c in checks if c.get("agree") is not None]
        agree = sum(1 for c in known if c["agree"])
        L.append(f"| Аккорды, видео | форма левой руки (MediaPipe), группы тактов | {agree} из {len(known)} тактов "
                 f"совпали с аккордом по нотам |")
        for fx in r.get("video_fixes", []):
            L.append(f"| Исправлено по видео | такт {_mmss(fx['t'])} | по звуку {fx['audio']}, по руке {fx['video']} |")
        if r.get("video_rescued"):
            L.append(f"| Ноты, видео | отброшенная фильтром нота, на которой палец этой струны сгибается как на "
                     f"принятых | возвращено {r['video_rescued']} |")
        L.append(f"| Синхронизация | сдвиг видео относительно звука по связи «струна ↔ палец» | "
                 f"{hands_res['lag'] * 1000:+.0f} мс |")
        L.append("")
        L.append("**Какими пальцами играется каждая струна** (по видео, число нот):\n")
        L.append("| Струна | " + " | ".join(FINGER_RU[f] for f in ("thumb", "index", "middle", "ring")) + " | не видно |")
        L.append("|---|---|---|---|---|---|")
        for s_, c in sorted(hands_res["fingers"].items(), key=lambda kv: -int(kv[0])):
            tot = sum(c.values())
            cells = [f"{c.get(f, 0) * 100 / tot:.0f}%" for f in ("thumb", "index", "middle", "ring", "?")]
            L.append(f"| {s_}-я | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def write_csv(r: dict, path: str) -> None:
    """Каждая нота: время, такт, позиция, нота, струна, лад, аккорд, палец по видео и все оценки."""
    import csv

    from .hands import FINGER_RU
    per = r["per_beat"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["время_с", "такт", "счёт", "нота", "струна", "лад", "аккорд", "палец_видео", "сила_сгиба",
                    "уверенность_фильтра", "второй_метод", "в_форме_аккорда"])
        for n in sorted(r["notes"], key=lambda n: (n["t"], -(n.get("string") or 0))):
            slot = n.get("slot", 0)
            cnt = f"{slot // per + 1}" + ("" if slot % per == 0 else "и")
            w.writerow([f"{n['t']:.3f}", n.get("bar", ""), cnt, librosa.midi_to_note(n["midi"], unicode=False),
                        n.get("string") or "", "" if n.get("fret") is None else n["fret"], n.get("shape") or "",
                        FINGER_RU.get(n.get("finger"), "") if n.get("finger") else "не видно",
                        n.get("finger_z", ""), n.get("p_real", ""), "да" if n.get("confirmed") else "нет",
                        "да" if n.get("in_shape") else "нет"])
