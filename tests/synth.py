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


def arpeggio(progression=("Dm", "Am"), patterns=None, bpm=72, loops=4, seed=1):
    """Перебор: на каждую восьмую одна струна. patterns — {аккорд: [струны 1..6 на 8 восьмых]}."""
    rng = np.random.default_rng(seed)
    patterns = patterns or {}
    eighth = 60 / bpm / 2
    total = int((len(progression) * loops * 8 + 6) * eighth * SR)
    y = np.zeros(total)
    t0 = 0.3
    for loop in range(loops):
        for ci, chord in enumerate(progression):
            frets = theory.shape_for(chord)["frets"]
            for k, string in enumerate(patterns[chord]):
                f = frets[6 - string]
                midi = OPEN_MIDI[6 - string] + f
                tone = pluck(440 * 2 ** ((midi - 69) / 12), eighth * 3, rng, bright=0.6)
                tone *= 1.0 if k % 4 == 0 else 0.7
                fade = int(0.04 * SR)
                tone[-fade:] *= np.linspace(1, 0, fade)
                a = int((t0 + ((loop * len(progression) + ci) * 8 + k) * eighth) * SR)
                y[a: a + len(tone)] += tone[: max(0, total - a)]
    y /= np.abs(y).max() + 1e-9
    return y.astype(np.float32)
