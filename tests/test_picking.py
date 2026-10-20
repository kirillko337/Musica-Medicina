"""Перебор на синтетике: известная схема струн должна восстановиться по нотам."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import audio, picking  # noqa: E402
from tests.synth import SR, arpeggio  # noqa: E402

pytest.importorskip("basic_pitch")


def test_arpeggio_strings(tmp_path):
    pats = {"Dm": [4, 3, 2, 3, 4, 3, 2, 3], "Am": [5, 3, 2, 3, 4, 3, 2, 3]}
    y = arpeggio(("Dm", "Am"), pats, bpm=72, loops=4)
    wav = tmp_path / "arp.wav"
    sf.write(wav, y, SR)
    res, yy, sr = audio.analyze(str(wav), vocab=["Dm", "Am"])
    segs = [dict(s, shape=s["chord"]) for s in res["segments"]]
    notes = picking.transcribe(str(wav))
    beats, phase = res["beats"], res["downbeat_phase"]
    half = picking.half_tempo(beats, phase, res["beats_per_bar"], notes, segs, yy, sr)
    if half:  # трекер принял восьмые за доли — так и должно чиниться
        beats, phase = half
    assert 60 <= 60 / (beats[5] - beats[4]) <= 85
    pk = picking.analyze(notes, beats, phase, segs, 0, res["beats_per_bar"], yy, sr)
    styles = {s["style"] for s in pk["sections"]}
    assert "перебор" in styles and "бой" not in styles
    sec = next(s for s in pk["sections"] if s.get("patterns"))
    got = {b: picking.short(p).replace(" ", "-") for b, p in sec["patterns"].items()}
    print(got)
    assert got[4] == "4-3-2-3-4-3-2-3"
    assert got[5].startswith("5-3-2-3")
