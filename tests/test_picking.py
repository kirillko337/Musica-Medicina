"""Перебор на синтетике: известны каждая нота, струна, лад, такт и позиция."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import audio, compare, notes, theory  # noqa: E402
from tests.synth import PICK_PATTERN, SR, picking_song, render_sf2, sf2_available, song  # noqa: E402

pytest.importorskip("basic_pitch")
pytest.importorskip("onnxruntime")
if not sf2_available():
    pytest.skip("нужны fluidsynth и FluidR3_GM: эталон рендерится сэмплами настоящей гитары",
                allow_module_level=True)


def match(pred, truth, tol=0.05):
    """Пары предсказание↔эталон с той же высотой и разницей ≤ tol."""
    used, pairs = set(), []
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
            pairs.append((pred[best[1]], t))
    return pairs


def run(tmp: Path, progression, capo=0, seed=0):
    _, truth = picking_song(progression, capo=capo, seed=seed)
    wav = tmp / "pick.wav"
    render_sf2(truth, str(wav), seed=seed)
    res, _, _ = audio.analyze(str(wav))
    capo_opts = theory.best_capo([s["chord"] for s in res["segments"]], [s["end"] - s["start"] for s in res["segments"]])
    segs = compare.consensus(res["segments"], [], capo)
    out = notes.analyze(str(wav), res, segs, capo)
    return out, truth, capo_opts


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    prog = ("Am", "Am", "G", "Am", "Dm", "Am", "G", "Am", "Am", "Am", "G", "Am", "Dm", "Am", "G", "Am")
    return run(tmp_path_factory.mktemp("pick"), prog) + (prog,)


def test_notes_found(result):
    out, truth, _, _ = result
    real = [n for n in out["notes"] if not n.get("ghost")]
    pairs = match(real, truth)
    recall = len(pairs) / len(truth)
    precision = len(pairs) / max(1, len(real))
    print(f"нот: эталон {len(truth)}, найдено {len(real)}, совпало {len(pairs)}; "
          f"полнота {recall:.3f}, точность {precision:.3f}")
    # на отложенных синтетических песнях (scripts/train_note_filter.py) фильтр даёт ~96% / ~90%;
    # здесь фигура повторяется, и второй проход по фигуре поднимает обе цифры
    assert recall >= 0.95
    assert precision >= 0.95


def test_strings_and_frets(result):
    out, truth, _, _ = result
    pairs = match([n for n in out["notes"] if not n.get("ghost")], truth)
    ok = sum(p["string"] == t["string"] and p["fret"] == t["fret"] for p, t in pairs)
    print(f"струна и лад верно: {ok}/{len(pairs)}")
    assert ok / len(pairs) >= 0.97


def test_meter_and_grid(result):
    out, truth, _, _ = result
    assert out["kind"]["picking"]
    assert out["beats_per_bar"] * out["per_beat"] == 12, (out["beats_per_bar"], out["per_beat"])


def test_chords_by_notes(result):
    out, truth, _, prog = result
    bars = [b for b in out["note_bars"] if 0 <= b["bar"] < len(prog)]
    print("аккорды по нотам:", [(b["chord"], b["no_third"]) for b in bars])
    # без терции (в фигуре Dm нет 1-й струны: только ре и ля) мажор от минора по звуку не отличить —
    # такой такт должен быть честно помечен no_third, а корень обязан совпасть
    for b, want in zip(bars, prog):
        if b["no_third"]:
            assert theory.parse(b["chord"])[0] == theory.parse(want)[0], (b, want)
        else:
            assert theory.triad(b["chord"]) == theory.triad(want), (b, want)


def test_pattern_tab(result):
    out, truth, _, _ = result
    pat = out["patterns"]["Am"]["tab"]
    want = {}
    shape = theory.shape_for("Am")["frets"]
    for pos, strings in PICK_PATTERN.items():
        for s in strings:
            st = 5 if s == "b" else s
            want[f"{st}:{pos}"] = shape[6 - st]
    print("фигура Am:", pat)
    assert pat == want


def test_strum_is_not_picking(tmp_path):
    y, _ = song(("Am", "C"), "D-DU-UDU", bpm=96, loops=2)
    wav = tmp_path / "strum.wav"
    sf.write(wav, y, SR)
    found = notes.transcribe(str(wav))
    assert not notes.is_picking(found)["picking"]
