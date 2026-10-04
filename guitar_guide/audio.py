"""Распознавание по звуку: темп, доли, тональность, аккорды."""
from __future__ import annotations

import librosa
import numpy as np

from . import theory

SR = 22050
HOP = 512

# Профили Крумхансла для определения тональности
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def beats(y_perc: np.ndarray, sr: int, bpm: float | None = None) -> tuple[float, np.ndarray]:
    oenv = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=HOP, start_bpm=bpm or 100, bpm=bpm, tightness=120
    )
    tempo = float(np.atleast_1d(tempo)[0])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    # гитарный бой почти всегда 60–160 BPM; поправляем ошибку «в два раза»
    if bpm is None and tempo > 160 and len(times) > 8:
        times, tempo = times[::2], tempo / 2
    elif bpm is None and tempo < 60 and len(times) > 2:
        mids = (times[:-1] + times[1:]) / 2
        times, tempo = np.sort(np.concatenate([times, mids])), tempo * 2
    times = _extend(times, librosa.get_duration(y=y_perc, sr=sr))
    return tempo, times


def _extend(times: np.ndarray, duration: float) -> np.ndarray:
    """beat_track часто пропускает начало и конец записи — достраиваем сетку с тем же шагом."""
    if len(times) < 2:
        return times
    period = float(np.median(np.diff(times)))
    head = np.arange(times[0] - period, -0.02, -period)[::-1]
    tail = np.arange(times[-1] + period, duration - 0.5 * period, period)
    return np.concatenate([head, times, tail])


def key(chroma: np.ndarray) -> str:
    prof = chroma.mean(axis=1)
    best, best_s = "C", -9.0
    for r in range(12):
        for mode, tpl in (("", _MAJOR), ("m", _MINOR)):
            s = np.corrcoef(prof, np.roll(tpl, r))[0, 1]
            if s > best_s:
                best, best_s = theory.name(r, mode), s
    return best


def chords(y_harm: np.ndarray, sr: int, beat_times: np.ndarray, sevenths: bool = False,
           p_stay: float = 0.85) -> dict:
    tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning, bins_per_octave=36)
    bass = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning,
                                      fmin=librosa.note_to_hz("E1"), n_octaves=2, bins_per_octave=36)
    rms = librosa.feature.rms(y=y_harm, hop_length=HOP)[0]
    n = chroma.shape[1]

    bf = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    bf = librosa.util.fix_frames(bf, x_min=0, x_max=n)
    c_sync = librosa.util.sync(chroma, bf, aggregate=np.median)
    b_sync = librosa.util.sync(bass, bf, aggregate=np.median)
    r_sync = librosa.util.sync(rms[np.newaxis, :], bf, aggregate=np.mean)[0]
    # bf = [0, доли..., n]; столбец t — отрезок от bf[t] до bf[t+1]
    seg_times = librosa.frames_to_time(bf, sr=sr, hop_length=HOP)
    pickup = int(bf[0] < librosa.time_to_frames(beat_times[0], sr=sr, hop_length=HOP))

    labels, tpl = theory.templates(("", "m", "7", "m7") if sevenths else ("", "m"))
    cn = c_sync / (np.linalg.norm(c_sync, axis=0, keepdims=True) + 1e-9)
    sim = tpl @ cn  # косинусная близость, (chords, beats)
    bn = b_sync / (b_sync.max(axis=0, keepdims=True) + 1e-9)
    roots = np.array([theory.parse(l)[0] for l in labels])
    sim = sim + 0.15 * bn[roots, :]
    if sevenths:  # септаккорды берём только при явном перевесе
        sim[24:, :] -= 0.04

    # состояние «нет аккорда» для тишины
    quiet = r_sync < 0.1 * (np.percentile(r_sync, 90) + 1e-9)
    n_sim = np.where(quiet, 1.2, 0.3)
    sim = np.vstack([sim, n_sim])
    labels = labels + ["N"]

    prob = np.exp(12 * (sim - sim.max(axis=0, keepdims=True)))
    prob /= prob.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(len(labels), p_stay)
    path = librosa.sequence.viterbi(prob, trans)

    beat_labels = [labels[i] for i in path]
    beat_conf = [float(prob[i, t]) for t, i in enumerate(path)]

    segments = []
    for t, lab in enumerate(beat_labels):
        start = float(seg_times[t])
        end = float(seg_times[t + 1])
        if segments and segments[-1]["chord"] == lab:
            segments[-1]["end"] = end
            segments[-1]["_c"].append(beat_conf[t])
        else:
            segments.append({"start": start, "end": end, "chord": lab, "_c": [beat_conf[t]]})
    for s in segments:
        s["conf"] = round(float(np.mean(s.pop("_c"))), 3)
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)

    return {
        "tuning_cents": round(tuning * 100, 1),
        "key": key(chroma),
        "grid_times": [round(float(t), 3) for t in seg_times],
        "grid_chords": beat_labels,
        "grid_pickup": pickup,
        "segments": segments,
    }


def downbeat_phase(grid_chords: list[str], beats_per_bar: int = 4) -> int:
    """Сильная доля: на какую фазу чаще всего приходится смена аккорда."""
    counts = np.zeros(beats_per_bar)
    for i in range(1, len(grid_chords)):
        if grid_chords[i] != grid_chords[i - 1]:
            counts[i % beats_per_bar] += 1
    return int(np.argmax(counts)) if counts.sum() else 0


def analyze(path: str, bpm: float | None = None, sevenths: bool = False) -> tuple[dict, np.ndarray, int]:
    y, sr = load(path)
    y_harm, y_perc = librosa.effects.hpss(y)
    tempo, beat_times = beats(y_perc, sr, bpm)
    if len(beat_times) < 4:
        raise RuntimeError("Не удалось найти доли — запись слишком короткая или без ритма")
    res = chords(y_harm, sr, beat_times, sevenths=sevenths)
    res["tempo"] = round(tempo, 1)
    res["beats"] = [round(float(t), 3) for t in beat_times]
    res["duration"] = round(len(y) / sr, 2)
    # аккорды по долям (без затакта до первой доли)
    res["beat_chords"] = res["grid_chords"][res["grid_pickup"]:][: len(beat_times)]
    res["downbeat_phase"] = downbeat_phase(res["beat_chords"])
    return res, y, sr
