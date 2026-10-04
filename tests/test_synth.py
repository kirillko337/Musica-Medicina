"""Проверка звукового анализа на синтетике: известные аккорды, темп и бой."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import audio, strum, theory  # noqa: E402
from tests.synth import SR, song  # noqa: E402


def run(tmp: Path, progression, pattern, bpm, capo=0):
    y, events = song(progression, pattern, bpm=bpm, loops=4, capo=capo)
    wav = tmp / "synth.wav"
    sf.write(wav, y, SR)
    res, yy, sr = audio.analyze(str(wav))
    st = strum.analyze(yy, sr, res["beats"], res["downbeat_phase"])

    # точность аккордов по ударам
    hits = 0
    for e in events:
        seg = next((s for s in res["segments"] if s["start"] <= e["t"] + 0.05 < s["end"]), None)
        hits += bool(seg and theory.triad(seg["chord"]) == theory.triad(e["chord"]))
    chord_acc = hits / len(events)

    # акустическое направление удара
    det = [o for o in st["onsets"] if o["dir_audio"] != "?"]
    ok = 0
    for o in det:
        e = min(events, key=lambda e: abs(e["t"] - o["t"]))
        ok += abs(e["t"] - o["t"]) < 0.04 and e["dir"] == o["dir_audio"]
    dir_acc = ok / max(1, len(det))
    return res, st, chord_acc, dir_acc


def test_am_f_c_g(tmp_path):
    res, st, chord_acc, dir_acc = run(tmp_path, ("Am", "F", "C", "G"), "D-DU-UDU", 96)
    print(res["tempo"], res["key"], chord_acc, dir_acc, strum.short(st["pattern"]))
    assert 85 <= res["tempo"] <= 107
    assert chord_acc > 0.85
    assert strum.short(st["pattern"]).replace(" ", "") == "Д-ДВ-ВДВ"
    assert dir_acc > 0.5  # акустическое направление — слабый признак, главное проверяет видео


def test_capo_suggestion(tmp_path):
    # звучит A E F#m D — это G D Em C с каподастром на 2-м ладу
    res, st, chord_acc, _ = run(tmp_path, ("A", "E", "F#m", "D"), "D-D-DUDU", 110)
    assert chord_acc > 0.8
    assert strum.short(st["pattern"]).replace(" ", "") == "Д-Д-ДВДВ"
    weights = [s["end"] - s["start"] for s in res["segments"]]
    best = theory.best_capo([s["chord"] for s in res["segments"]], weights)[0]
    assert best["capo"] == 2 and best["easy_share"] > 0.95


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for prog, pat, bpm in ((("Am", "F", "C", "G"), "D-DU-UDU", 96), (("A", "E", "F#m", "D"), "D-D-DUDU", 110)):
            res, st, ca, da = run(Path(d), prog, pat, bpm)
            print(prog, pat, "→ tempo", res["tempo"], "key", res["key"], "chords", round(ca, 2),
                  "dir", round(da, 2), "pattern", strum.short(st["pattern"]),
                  [s["chord"] for s in res["segments"]][:10])
            print(strum.render(st["pattern"]))


def test_sanjuanito_16ths(tmp_path):
    # бой из La Anaconda: 1 · и а | 2 · и · — «вниз, вниз-вверх, вниз, вниз»
    res, st, chord_acc, _ = run(tmp_path, ("Am", "Em"), "D-DUD-D-D-DUD-D-", 108)
    assert chord_acc > 0.85
    assert st["pattern"]["subdiv"] == 4
    assert strum.short(st["pattern"]).replace(" ", "") == "Д-ДВД-Д-Д-ДВД-Д-"
