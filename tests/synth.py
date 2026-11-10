"""Синтетическая «песня» на гитаре (Karplus–Strong) с известными аккордами и боем — для проверки."""
from __future__ import annotations

import numpy as np

from guitar_guide import theory

SR = 22050
OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4


def pluck(freq: float, dur: float, rng: np.random.Generator, bright: float = 0.5) -> np.ndarray:
    n = int(dur * SR)
    period = int(SR / freq)
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    decay = 0.996
    for i in range(n):
        out[i] = buf[i % period]
        nxt = buf[(i + 1) % period]
        buf[i % period] = decay * (bright * buf[i % period] + (1 - bright) * nxt)
    return out


def strum(frets: list[int], direction: str, dur: float, rng, gain: float = 1.0, spread_ms: float = 9.0) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    strings = [i for i, f in enumerate(frets) if f >= 0]
    if direction == "up":
        strings = strings[::-1][:4]  # вверх обычно цепляют верхние струны
        gain *= 0.75
    for k, s in enumerate(strings):
        midi = OPEN_MIDI[s] + frets[s]
        off = int(k * spread_ms / 1000 * SR)
        tone = pluck(440 * 2 ** ((midi - 69) / 12), dur, rng)
        out[off:] += tone[: n - off] * gain
    return out


def song(progression=("Am", "F", "C", "G"), pattern="D-DU-UDU", bpm=96, loops=4, seed=0, capo=0):
    """pattern — 8 восьмых или 16 шестнадцатых на такт: D вниз, U вверх, - пропуск."""
    rng = np.random.default_rng(seed)
    eighth = 60 / bpm * 4 / len(pattern)
    n = len(pattern)
    total = int((len(progression) * loops * n + n // 2) * eighth * SR)
    y = np.zeros(total)
    events = []
    t0 = 0.3
    for loop in range(loops):
        for ci, chord in enumerate(progression):
            shape = theory.shape_for(theory.transpose(chord, -capo))["frets"]
            frets = [f if f < 0 else f + capo for f in shape]
            for k, ch in enumerate(pattern):
                if ch == "-":
                    continue
                t = t0 + ((loop * len(progression) + ci) * n + k) * eighth
                d = "down" if ch == "D" else "up"
                g = 1.0 if k % (n // 4) == 0 else 0.6
                s = strum(frets, d, 60 / bpm * 1.25, rng, gain=g)
                fade = int(0.04 * SR)  # без затухания обрыв звука даёт ложную «атаку»
                s[-fade:] *= np.linspace(1, 0, fade)
                a = int(t * SR)
                y[a: a + len(s)] += s[: max(0, total - a)]
                events.append({"t": t, "dir": d, "chord": chord})
    y /= np.abs(y).max() + 1e-9
    return y.astype(np.float32), events


# ───────────── перебор ─────────────
# фигура: позиция (половина доли) → список струн (6 = басовая струна аккорда, 'b' — бас формы)
PICK_PATTERN = {0: ["b"], 3: [3], 4: [2], 6: ["b", 2], 7: [3], 9: [3], 10: [2]}


def bass_string(frets: list[int]) -> int:
    """Номер басовой струны (6…1) — самая низкая играемая струна формы."""
    for i, f in enumerate(frets):
        if f >= 0:
            return 6 - i
    return 6


def picking_song(progression=("Am", "Am", "G", "Am", "Dm", "Am", "G", "Am"), beat_bpm=137, beats_per_bar=6,
                 per_beat=2, pattern=None, seed=0, capo=0, jitter_ms=8.0):
    """Перебор: известная нота на каждую позицию. Возвращает звук и список нот
    [{t, midi, string, fret, chord}] — эталон для проверки распознавания."""
    pattern = pattern or PICK_PATTERN
    rng = np.random.default_rng(seed)
    beat = 60 / beat_bpm
    slot = beat / per_beat
    bar_len = beat * beats_per_bar
    total = int((len(progression) * bar_len + 2.0) * SR)
    y = np.zeros(total)
    truth = []
    t0 = 0.4
    for bi, chord in enumerate(progression):
        shape = theory.shape_for(theory.transpose(chord, -capo))["frets"]
        bstr = bass_string(shape)
        for pos, strings in pattern.items():
            for s in strings:
                st = bstr if s == "b" else s
                f = shape[6 - st]
                if f < 0:
                    continue
                midi = OPEN_MIDI[6 - st] + f + capo
                t = t0 + bi * bar_len + pos * slot + rng.normal(0, jitter_ms / 1000)
                gain = 1.0 if s == "b" else 0.55
                tone = pluck(440 * 2 ** ((midi - 69) / 12), 1.2, rng, bright=0.6) * gain
                fade = int(0.05 * SR)
                tone[-fade:] *= np.linspace(1, 0, fade)
                a = int(t * SR)
                y[a:a + len(tone)] += tone[: max(0, total - a)]
                truth.append({"t": float(t), "midi": midi, "string": st, "fret": f, "chord": chord,
                              "bar": bi, "slot": pos})
    y /= np.abs(y).max() + 1e-9
    return y.astype(np.float32), sorted(truth, key=lambda n: (n["t"], n["midi"]))


SF2_PATHS = ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2"]


def sf2_available() -> str | None:
    import os
    import shutil
    if not shutil.which("fluidsynth"):
        return None
    return next((p for p in SF2_PATHS if os.path.exists(p)), None)


def render_sf2(truth: list[dict], wav_path: str, program: int = 24, seed: int = 0) -> str:
    """Рендер нот сэмплами настоящей гитары (General MIDI, 24 — нейлоновая).
    Нота звучит, пока ту же струну не дёрнут снова (как на гитаре) или 1.6 с."""
    import os
    import subprocess
    import tempfile

    import pretty_midi
    rng = np.random.default_rng(seed)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    by_string = {}
    for n in sorted(truth, key=lambda n: n["t"]):
        by_string.setdefault(n["string"], []).append(n)
    for s, lst in by_string.items():
        for i, n in enumerate(lst):
            end = min(n["t"] + 1.6, lst[i + 1]["t"] - 0.01) if i + 1 < len(lst) else n["t"] + 1.6
            vel = int(np.clip((96 if s >= 4 else 72) + rng.normal(0, 8), 40, 127))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=n["midi"], start=n["t"], end=end))
    pm.instruments.append(inst)
    with tempfile.TemporaryDirectory() as d:
        mid = os.path.join(d, "x.mid")
        pm.write(mid)
        subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-r", "22050", "-F", wav_path, sf2_available(), mid],
                       check=True, capture_output=True)
    return wav_path


CHORD_POOL = ["Am", "C", "G", "Em", "Dm", "F", "D", "E", "A"]


def random_picking(seed: int):
    """Случайная песня перебором: аккорды, размер, темп, фигура, каподастр — для обучения
    и проверки фильтра нот. Возвращает (truth, meta)."""
    rng = np.random.default_rng(seed)
    beats_per_bar = int(rng.choice([3, 4, 6]))
    per_beat = int(rng.choice([1, 2])) if beats_per_bar != 6 else 2
    slots = beats_per_bar * per_beat
    bpm = float(rng.uniform(60, 150)) * (2 if per_beat == 1 and rng.random() < 0.5 else 1)
    bpm = min(bpm, 170)
    capo = int(rng.choice([0, 0, 0, 1, 2, 3, 5]))
    pool = list(rng.choice(CHORD_POOL, size=4, replace=False))
    prog = [pool[i] for i in rng.integers(0, 4, size=8)]
    pattern = {0: ["b"]}
    for s in range(1, slots):
        r = rng.random()
        if r < 0.15:
            continue
        if s == slots // 2 and r < 0.6:
            pattern[s] = ["b"] + ([int(rng.choice([1, 2]))] if rng.random() < 0.4 else [])
        else:
            pattern[s] = [int(rng.choice([1, 2, 3, 3, 4]))]
            if rng.random() < 0.12:
                pattern[s].append(int(rng.choice([1, 2])))
    pattern = {k: sorted(set(v), key=str) for k, v in pattern.items()}
    _, truth = picking_song(tuple(prog), beat_bpm=bpm, beats_per_bar=beats_per_bar, per_beat=per_beat,
                            pattern=pattern, seed=seed, capo=capo, jitter_ms=float(rng.uniform(3, 15)))
    # без дублей одной струны в одной позиции
    seen, uniq = set(), []
    for n in truth:
        k = (round(n["t"], 3), n["string"])
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return uniq, {"prog": prog, "beats_per_bar": beats_per_bar, "per_beat": per_beat, "bpm": bpm,
                  "capo": capo, "pattern": pattern}
