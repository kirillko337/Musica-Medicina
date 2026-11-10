"""Обучение фильтра нот на синтетике с сэмплами настоящей гитары.

python scripts/train_note_filter.py --songs 60 --work .work/train

Нужны fluidsynth и звуковой банк FluidR3_GM (apt: fluidsynth fluid-soundfont-gm).
Каждая песня: случайные аккорды, размер, темп, каподастр и фигура перебора (tests/synth.random_picking),
рендер нейлоновой (24) или стальной (25) гитарой. Ноты Basic Pitch размечаются по эталону:
настоящая = есть нота эталона той же высоты в пределах 50 мс.
Проверка — на отложенных песнях (каждая 4-я), в отчёт идут полнота и точность до и после фильтра.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import notefilter, notes  # noqa: E402
from tests.synth import random_picking, render_sf2  # noqa: E402


ONSET = 0.5  # порог Basic Pitch (0.35 проверен: полнота та же, лишних больше)


def label(pred: list[dict], truth: list[dict], tol: float = 0.05) -> list[int]:
    used = set()
    lab = [0] * len(pred)
    for t in truth:
        best = None
        for i, p in enumerate(pred):
            if i in used or p["midi"] != t["midi"]:
                continue
            d = abs(p["t"] - t["t"])
            if d <= tol and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1])
            lab[best[1]] = 1
    return lab


def build(n_songs: int, work: Path) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    cache = work / f"data_{n_songs}_on{ONSET}_rp.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    data = []
    for seed in range(n_songs):
        truth, meta = random_picking(1000 + seed)
        wav = str(work / f"song{seed}.wav")
        if not os.path.exists(wav):
            render_sf2(truth, wav, program=24 if seed % 3 else 25, seed=seed)
        pred = notes.transcribe(wav, onset_threshold=ONSET)
        data.append({"seed": seed, "wav": wav, "pred": pred, "y": np.array(label(pred, truth)),
                     "n_truth": len(truth), "meta": meta})
        print(f"песня {seed}: эталон {len(truth)}, найдено {len(pred)}, настоящих {int(sum(data[-1]['y']))}", flush=True)
    pickle.dump(data, open(cache, "wb"))
    return data


def add_features(data: list[dict]) -> None:
    """Признаки считаются отдельно от рендера: их можно менять без повторного распознавания."""
    for d in data:
        y, sr = librosa.load(d["wav"], sr=22050)
        d["X"] = notefilter.features(d["pred"], y, sr)


def export_trees(clf) -> dict:
    trees = []
    for it in clf._predictors:
        nd = it[0].nodes
        trees.append({"f": nd["feature_idx"].tolist(), "thr": nd["num_threshold"].tolist(),
                      "l": nd["left"].tolist(), "r": nd["right"].tolist(), "leaf": nd["is_leaf"].astype(int).tolist(),
                      "v": nd["value"].tolist(), "ml": nd["missing_go_to_left"].astype(int).tolist()})
    return {"kind": "trees", "baseline": float(np.ravel(clf._baseline_prediction)[0]), "trees": trees}


def evaluate(part: list[dict], model: dict | None) -> dict:
    tp = fp = tot = 0
    for d in part:
        keep = np.ones(len(d["y"]), bool) if model is None else notefilter.proba(d["X"], model) >= model["threshold"]
        tp += int((keep & (d["y"] == 1)).sum())
        fp += int((keep & (d["y"] == 0)).sum())
        tot += d["n_truth"]
    return {"recall": round(tp / tot, 4), "precision": round(tp / max(1, tp + fp), 4), "missed": tot - tp, "extra": fp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=60)
    ap.add_argument("--work", default=".work/train")
    ap.add_argument("--min-recall", type=float, default=0.985, help="порог выбирается так, чтобы на обучении полнота была не ниже")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    data = build(args.songs, Path(args.work))
    add_features(data)
    test = [d for d in data if d["seed"] % 4 == 3]
    train = [d for d in data if d["seed"] % 4 != 3]
    X = np.vstack([d["X"] for d in train])
    y = np.concatenate([d["y"] for d in train])
    clf = HistGradientBoostingClassifier(max_depth=6, max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                         random_state=0).fit(X, y)
    model = export_trees(clf)
    model["features"] = notefilter.FEATURES
    p = notefilter.proba(X, model)
    assert np.allclose(p, clf.predict_proba(X)[:, 1], atol=1e-6), "экспорт деревьев не совпал с sklearn"
    # порог: максимальная точность при полноте ≥ min_recall на кросс-проверке по песням обучения
    from sklearn.model_selection import GroupKFold, cross_val_predict
    groups = np.concatenate([[d["seed"]] * len(d["y"]) for d in train])
    pcv = cross_val_predict(HistGradientBoostingClassifier(max_depth=6, max_iter=400, learning_rate=0.05,
                                                           l2_regularization=1.0, random_state=0),
                            X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    best = 0.05
    for thr in np.arange(0.05, 0.9, 0.01):
        rec = ((pcv >= thr) & (y == 1)).sum() / max(1, (y == 1).sum())
        if rec >= args.min_recall:
            best = float(thr)
    model["threshold"] = round(best, 2)
    report = {"train_songs": len(train), "test_songs": len(test),
              "test_before": evaluate(test, None), "test_after": evaluate(test, model),
              "train_after": evaluate(train, model)}
    model["report"] = report
    os.makedirs(os.path.dirname(notefilter.MODEL_PATH), exist_ok=True)
    with open(notefilter.MODEL_PATH, "w") as f:
        json.dump(model, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
