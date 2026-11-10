"""Пальцы и форма руки на синтетических траекториях с известным ответом."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import hands  # noqa: E402

FPS = 25.0
STRING_FINGER = {5: "thumb", 4: "thumb", 3: "index", 2: "middle", 1: "ring"}


def fake_signals(notes, lag, seconds, rng):
    """Скорость сгиба: шум + провал у пальца, который щипнул струну (через lag после звука)."""
    T = int(seconds * FPS)
    sig = {f: rng.normal(0, 0.4, T) for f in hands.ORDER}
    for n in notes:
        f = STRING_FINGER[n["string"]]
        i = int(round((n["t"] + lag) * FPS))
        for d, v in ((-1, -2.0), (0, -6.0), (1, -2.5)):
            if 0 <= i + d < T:
                sig[f][i + d] += v
        # безымянный тянется за средним, но слабее
        if f == "middle" and 0 <= i < T:
            sig["ring"][i] += -2.5
    return sig


def make_notes(rng, bars=16, slot=0.22):
    pattern = [(0, 5), (3, 3), (4, 2), (6, 5), (6, 2), (7, 3), (9, 3), (10, 2), (11, 1)]
    out = []
    for b in range(bars):
        for pos, s in pattern:
            out.append({"t": 0.5 + (b * 12 + pos) * slot + rng.normal(0, 0.008), "string": s, "midi": 40 + 5 * (6 - s)})
    return out


def test_lag_and_fingers():
    rng = np.random.default_rng(1)
    notes = make_notes(rng)
    lag = 0.12
    sig = fake_signals(notes, lag, notes[-1]["t"] + 1, rng)
    got, _ = hands.estimate_lag(sig, notes, FPS)
    assert abs(got - lag) <= 0.02, got
    hands.fingers_for_notes(notes, sig, FPS, got)
    ok = sum(n["finger"] == STRING_FINGER[n["string"]] for n in notes)
    assert ok / len(notes) >= 0.95, ok / len(notes)


def test_chord_groups_fix_mislabel():
    rng = np.random.default_rng(2)
    base = {"Am": rng.normal(0, 1, 40), "G": rng.normal(0, 1, 40), "Dm": rng.normal(0, 1, 40)}
    truth = ["Am", "Am", "G", "Am", "Dm", "Am", "G", "Am", "Am", "Am", "G", "Am", "Dm", "Am", "G", "Am"]
    prints = [base[c] + rng.normal(0, 0.1, 40) for c in truth]
    labels = list(truth)
    labels[2] = "Em"  # звук ошибся в одном такте
    res = hands.check_bars(prints, labels)
    assert res[2]["video"] == "G"
    assert all(r["video"] == t for r, t in zip(res, truth))


def test_rescue_by_video():
    rng = np.random.default_rng(3)
    notes = make_notes(rng, bars=8)
    sig = fake_signals(notes, 0.0, notes[-1]["t"] + 1, rng)
    hands.fingers_for_notes(notes, sig, FPS, 0.0)
    # кандидат, на котором средний палец реально щипнул (провал), и пустой кандидат
    real = {"t": 0.5 + 50 * 0.22 + 0.11, "string": 2, "in_shape": True, "midi": 60}
    i = int(round(real["t"] * FPS))
    sig["middle"][i] += -6.0
    empty = {"t": 0.5 + 26 * 0.22 + 0.11, "string": 2, "in_shape": True, "midi": 60}
    got = hands.rescue_by_video(notes, [real, empty], sig, FPS, 0.0)
    assert real in got and empty not in got
