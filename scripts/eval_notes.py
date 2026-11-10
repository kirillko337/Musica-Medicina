"""Проверка всего конвейера нот на отложенных синтетических песнях (seed % 4 == 3).

python scripts/eval_notes.py --songs 160 --work .work/train
Песни и звук те же, что в scripts/train_note_filter.py; модель фильтра их не видела.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import audio, compare, notes, theory  # noqa: E402
from tests.synth import random_picking, render_sf2  # noqa: E402
from scripts.train_note_filter import label  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=160)
    ap.add_argument("--work", default=".work/train")
    a = ap.parse_args()
    tot = tp = fp = sf_ok = 0
    rows = []
    for seed in range(3, a.songs, 4):
        truth, meta = random_picking(1000 + seed)
        wav = os.path.join(a.work, f"song{seed}.wav")
        if not os.path.exists(wav):
            render_sf2(truth, wav, program=24 if seed % 3 else 25, seed=seed)
        res, _, _ = audio.analyze(wav)
        capo = meta["capo"]  # каподастр известен из видео/подсказки; здесь проверяем ноты, а не его поиск
        segs = compare.consensus(res["segments"], [], capo)
        out = notes.analyze(wav, res, segs, capo)
        pred = out["notes"]
        lab = label(pred, truth)
        m = sum(lab)
        # струна и лад у совпавших
        ok = 0
        used = set()
        for t in truth:
            for i, p in enumerate(pred):
                if i not in used and p["midi"] == t["midi"] and abs(p["t"] - t["t"]) <= 0.05:
                    used.add(i)
                    ok += p.get("string") == t["string"] and p.get("fret") == t["fret"]
                    break
        import numpy as np
        beat = float(np.median(np.diff(out["beats_shifted"])))
        bar_det = out["beats_per_bar"] * beat
        bar_true = meta["beats_per_bar"] * 60 / meta["bpm"]
        slots_ok = out["slots"] == meta["beats_per_bar"] * meta["per_beat"]
        bar_ok = abs(bar_det - bar_true) / bar_true < 0.06 and slots_ok
        wrong = []
        used2 = set()
        for t in truth:
            for i, p in enumerate(pred):
                if i not in used2 and p["midi"] == t["midi"] and abs(p["t"] - t["t"]) <= 0.05:
                    used2.add(i)
                    if not (p.get("string") == t["string"] and p.get("fret") == t["fret"]):
                        wrong.append((t["chord"], t["string"], t["fret"], p.get("string"), p.get("fret"), p.get("shape")))
                    break
        tot += len(truth)
        tp += m
        fp += len(pred) - m
        sf_ok += ok
        rows.append({"seed": seed, "truth": len(truth), "found": len(pred), "match": m, "string_ok": ok,
                     "meter": f"{out['beats_per_bar']}x{out['per_beat']}", "meter_true": f"{meta['beats_per_bar']}x{meta['per_beat']}",
                     "bar_ok": bool(bar_ok), "bar_s": [round(bar_det, 2), round(bar_true, 2)], "capo": capo,
                     "wrong": wrong[:6]})
        print(rows[-1], flush=True)
    meter_ok = sum(r["bar_ok"] for r in rows)
    rep = {"songs": len(rows), "recall": round(tp / tot, 4), "precision": round(tp / max(1, tp + fp), 4),
           "string_fret_ok": round(sf_ok / max(1, tp), 4), "meter_ok": f"{meter_ok}/{len(rows)}"}
    print(json.dumps(rep, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
