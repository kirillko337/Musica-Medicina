"""Фильтр нот: настоящая сыгранная нота или артефакт распознавания.

Basic Pitch на гитаре находит все сыгранные ноты, но добавляет лишние:
  - разрыв звучащей ноты в момент, когда щипнули другую струну;
  - обертоны (октава, дуодецима) басовой ноты;
  - короткие отзвуки.
Градиентный бустинг на деревьях (признаки ниже) обучен на синтетических песнях, отрендеренных
сэмплами настоящей гитары (scripts/train_note_filter.py). Коэффициенты — в models/note_filter.json.
"""
from __future__ import annotations

import json
import os

import librosa
import numpy as np

HARM = (12, 19, 24, 28, 31)
FEATURES = ["amp", "log_dur", "onset_p", "frame_p", "rise_f", "rise_h", "split", "coincide", "harm_sim",
            "harm_ratio", "harm_sounding", "short", "midi", "split_x_rise", "harm_x_onset", "group",
            "amp_rel", "onset_rel", "end_split", "recent_same", "pc_context", "gap_prev_same", "rise_sub",
            "repluck"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "note_filter.json")


def _rise_fn(y: np.ndarray, sr: int, hop: int = 128):
    lo = 36
    n = 60
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(lo) * 2 ** (-1 / 36), n_bins=n * 3,
                           bins_per_octave=36, filter_scale=1.5)).reshape(n, 3, -1).max(1)
    D = librosa.amplitude_to_db(C, ref=C.max() + 1e-9)
    t = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)

    def rise(tt: float, m: int, h: int = 0) -> float:
        k = m - lo + h
        if not 0 <= k < n:
            return 0.0
        a = np.searchsorted(t, [tt - 0.06, tt - 0.015, tt - 0.01, tt + 0.05])
        if a[1] <= a[0] or a[3] <= a[2]:
            return 0.0
        return float(D[k, a[2]:a[3]].max() - D[k, a[0]:a[1]].max())
    return rise


def features(notes: list[dict], y: np.ndarray, sr: int) -> np.ndarray:
    rise = _rise_fn(y, sr)
    ts = np.array([n["t"] for n in notes])
    X = []
    for i, q in enumerate(notes):
        near = np.flatnonzero(np.abs(ts - q["t"]) < 0.035)
        sim = [notes[j] for j in near if j != i]
        prev = [r for r in notes if r is not q and r["midi"] == q["midi"] and r["t"] < q["t"] - 0.02
                and abs(r["end"] - q["t"]) < 0.03]
        coincide = sum(1 for r in sim if r["midi"] != q["midi"])
        hs = [r for r in sim if q["midi"] - r["midi"] in HARM]
        harm_ratio = q["amp"] / max(r["amp"] for r in hs) if hs else 0.0
        sounding = [r for r in notes if r["t"] < q["t"] - 0.035 and r["end"] > q["t"] + 0.02
                    and q["midi"] - r["midi"] in HARM and r["amp"] >= q["amp"]]
        dur = q["end"] - q["t"]
        rf, rh = rise(q["t"], q["midi"]), rise(q["t"], q["midi"], 12)
        split = float(bool(prev) and coincide > 0)
        amp_rel = q["amp"] / max([q["amp"]] + [r["amp"] for r in sim])
        onset_rel = q.get("onset_p", 0.5) / max([q.get("onset_p", 0.5)] + [r.get("onset_p", 0.5) for r in sim] + [1e-3])
        end_split = float(any(r["midi"] == q["midi"] and abs(r["t"] - q["end"]) < 0.03 for r in notes if r is not q))
        recent = [r for r in notes if r["midi"] == q["midi"] and 0.02 < q["t"] - r["t"] < 0.6]
        gap_prev = min([q["t"] - r["end"] for r in recent] + [1.0])
        ctx = [r for r in notes if r is not q and abs(r["t"] - q["t"]) < 1.2]
        tot = sum(r["amp"] for r in ctx) or 1.0
        pc_ctx = sum(r["amp"] for r in ctx if r["midi"] % 12 == q["midi"] % 12) / tot
        rise_sub = max(rise(q["t"], q["midi"], -12), rise(q["t"], q["midi"], -19))
        X.append([q["amp"], np.log(max(dur, 0.02)), q.get("onset_p", 0.5), q.get("frame_p", 0.5), rf, rh,
                  split, min(coincide, 3), float(bool(hs)), harm_ratio, float(bool(sounding)), float(dur < 0.16),
                  (q["midi"] - 60) / 12, split * rf, float(bool(hs)) * q.get("onset_p", 0.5), len(near),
                  amp_rel, onset_rel, end_split, len(recent), pc_ctx, gap_prev, rise_sub,
                  float(q.get("repluck", False))])
    return np.array(X, dtype=float)


def load_model(path: str = MODEL_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _tree_raw(x: np.ndarray, tree: dict) -> float:
    i = 0
    f, thr, left, right, leaf, val, miss = (tree[k] for k in ("f", "thr", "l", "r", "leaf", "v", "ml"))
    while not leaf[i]:
        xv = x[f[i]]
        if np.isnan(xv):
            i = left[i] if miss[i] else right[i]
        else:
            i = left[i] if xv <= thr[i] else right[i]
    return val[i]


def proba(X: np.ndarray, model: dict) -> np.ndarray:
    if model.get("kind") == "trees":
        raw = np.array([model["baseline"] + sum(_tree_raw(x, t) for t in model["trees"]) for x in X])
        return 1 / (1 + np.exp(-raw))
    Z = (X - np.array(model["mean"])) / np.array(model["scale"])
    return 1 / (1 + np.exp(-(Z @ np.array(model["coef"]) + model["intercept"])))


def apply(notes: list[dict], y: np.ndarray, sr: int, model: dict | None = None) -> dict:
    """Помечает у каждой ноты p_real и real. Ноты не удаляет — решение видно в analysis.json."""
    model = model or load_model()
    if not model or not notes:
        for n in notes:
            n["p_real"], n["real"] = None, True
        return {"model": None}
    p = proba(features(notes, y, sr), model)
    thr = model.get("threshold", 0.5)
    for n, v in zip(notes, p):
        n["p_real"] = round(float(v), 3)
        n["real"] = bool(v >= thr)
    return {"model": os.path.basename(MODEL_PATH), "threshold": thr, "kept": int(sum(p >= thr)), "total": len(notes)}
