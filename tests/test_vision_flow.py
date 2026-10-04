"""Оптический поток: «рука» (светлое пятно) ходит маятником, проверяем направление в моменты ударов."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitar_guide import vision  # noqa: E402


def make_video(path: Path, beat: float = 0.6, seconds: float = 6.0, fps: int = 30) -> list[tuple[float, str]]:
    w, h = 640, 360
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        t = i / fps
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(img, (150, 120), (500, 240), (60, 90, 140), -1)  # «гитара»
        # вниз на долю, вверх между долями: y = A·sin(2πt/beat), скорость >0 при t = k·beat
        y = int(180 + 70 * np.sin(2 * np.pi * t / beat))
        cv2.ellipse(img, (330, y), (40, 28), 0, 0, 360, (200, 200, 230), -1)
        vw.write(img)
    vw.release()
    ev = []
    k = 0
    while k * beat / 2 < seconds - 0.5:
        ev.append((round(k * beat / 2, 3), "down" if k % 2 == 0 else "up"))
        k += 1
    return [e for e in ev if e[0] > 0.2]


def test_flow_direction(tmp_path):
    video = tmp_path / "v.mp4"
    events = make_video(video)
    res = vision.strum_flow(str(video), [0.4, 0.1, 0.25, 0.8], [t for t, _ in events])
    got = {r["t"]: r["dir_video"] for r in res}
    ok = sum(got[t] == d for t, d in events)
    print(ok, len(events))
    assert ok / len(events) > 0.9


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_flow_direction(Path(d))
