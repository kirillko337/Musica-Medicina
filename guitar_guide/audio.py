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


def beats_nn(path: str) -> dict | None:
    """Доли и сильные доли нейросетью: beat_this (CPJKU, 2024), если нет — madmom (RNN + DBN).
    librosa ищет только доли и на живой игре с плавающим темпом часто ошибается в 1,5–2 раза;
    эти модели дают ещё и начало такта, то есть размер и сетку для боя и перебора."""
    try:
        from beat_this.inference import File2Beats

        b, d = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(path)
        src = "beat_this"
    except Exception as e:  # нет пакета или весов
        print(f"  beat_this недоступен ({type(e).__name__}), пробую madmom")
        try:
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

            act = RNNDownBeatProcessor()(path)
            bb = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100, min_bpm=50, max_bpm=190)(act)
            b, d = bb[:, 0], bb[bb[:, 1] == 1, 0]
            src = "madmom"
        except Exception as e2:
            print(f"  madmom недоступен ({type(e2).__name__}), доли считает librosa")
            return None
    b, d = np.asarray(b, float), np.asarray(d, float)
    if len(b) < 8:
        return None
    # размер: сколько долей между соседними сильными
    idx = np.searchsorted(b, d - 0.05)
    per = np.diff(idx)
    per = per[(per >= 2) & (per <= 7)]
    bpb = int(np.bincount(per).argmax()) if len(per) else 4
    first = int(np.searchsorted(b, d[0] - 0.05)) if len(d) else 0
    return {"beats": b, "downbeats": d, "beats_per_bar": bpb, "phase": first % bpb, "source": src}


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
           p_stay: float = 0.85, vocab: list[str] | None = None) -> dict:
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
    vocab_info = None
    if vocab:
        sim, vocab_info = _apply_vocab(sim, labels, vocab, r_sync)

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

    song_key = key(chroma)
    beat_labels = resolve_fifths(beat_labels, c_sync, song_key)

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
        "key": song_key,
        "grid_times": [round(float(t), 3) for t in seg_times],
        "grid_chords": beat_labels,
        "grid_pickup": pickup,
        "segments": segments,
        "vocab": vocab_info,
    }


def _apply_vocab(sim: np.ndarray, labels: list[str], vocab: list[str], r_sync: np.ndarray,
                 min_fit: float = 0.9) -> tuple[np.ndarray, dict | None]:
    """Оставляет только аккорды из словаря (подписи на экране). Подписи бывают формами при каподастре,
    поэтому пробуем сдвиги 0–7 и берём тот, что лучше объясняет звук. Если даже лучший сдвиг
    объясняет звук заметно хуже, чем полный набор шаблонов, словарь не про эту песню — игнорируем."""
    loud = r_sync >= np.percentile(r_sync, 30)
    best_all = sim[:, loud].max(axis=0).mean()
    idx = {l: i for i, l in enumerate(labels)}
    best = None
    for shift in range(0, 8):
        allowed = set()
        for v in vocab:
            p = theory.parse(v)
            if p is None:
                continue
            lab = theory.name(p[0] + shift, p[1])
            if lab not in idx:  # септаккорд без шаблонов септаккордов — берём трезвучие
                lab = theory.name(p[0] + shift, "m" if p[1].startswith("m") else "")
            if lab in idx:
                allowed.add(idx[lab])
        if len(allowed) < 2:
            continue
        fit = sim[sorted(allowed)][:, loud].max(axis=0).mean() / (best_all + 1e-9)
        if best is None or fit > best[0]:
            best = (fit, shift, allowed)
    if best is None or best[0] < min_fit:
        return sim, {"used": False, "fit": round(float(best[0]), 3) if best else None}
    fit, shift, allowed = best
    out = sim.copy()
    mask = np.ones(len(labels), bool)
    mask[list(allowed)] = False
    out[mask] -= 1.0
    return out, {"used": True, "shift": shift, "fit": round(float(fit), 3),
                 "chords": [labels[i] for i in sorted(allowed)]}


# Диатонические аккорды: ступень от тоники → качество (натуральный мажор и минор + доминанта)
_DIATONIC = {"": {0: "", 2: "m", 4: "m", 5: "", 7: "", 9: "m"},
             "m": {0: "m", 3: "", 5: "m", 7: "", 8: "", 10: ""}}


def resolve_fifths(labels: list[str], c_sync: np.ndarray, song_key: str, min_third: float = 0.3) -> list[str]:
    """Аккорд без терции (звучат только тоника и квинта, как D-A-D) не бывает ни мажором, ни минором:
    шаблоны выбирают наугад, обычно мажор. Такой участок называем по тональности: в ре миноре D5 → Dm.
    Проверено на He Yama Yo: в интро Dm играется без 1-й струны, и все детекторы писали D."""
    kp = theory.parse(song_key)
    if kp is None:
        return labels
    tonic, mode = kp
    out = list(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        p = theory.parse(labels[i])
        if p is not None:
            root, q = p
            c = c_sync[:, i:j].mean(axis=1)
            third = max(c[(root + 3) % 12], c[(root + 4) % 12])
            frame = max(c[root], c[(root + 7) % 12]) + 1e-9
            deg = (root - tonic) % 12
            want = _DIATONIC[mode if mode in _DIATONIC else ""].get(deg)
            if third / frame < min_third and want is not None:
                newq = want + ("7" if q.endswith("7") and want == "" else "")
                out[i:j] = [theory.name(root, newq)] * (j - i)
        i = j
    return out


def halfbars(y_harm: np.ndarray, sr: int, beat_times: np.ndarray, phase: int) -> list[dict]:
    """Аккорд на каждые две доли без сглаживания: лучший кандидат, второй и отрыв между ними.
    Нужен, чтобы увидеть короткие проходящие аккорды, которые Витерби склеивает."""
    C = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, bins_per_octave=36)
    B = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, bins_per_octave=36,
                                   fmin=librosa.note_to_hz("E1"), n_octaves=2)
    ft = librosa.times_like(C, sr=sr, hop_length=HOP)
    labels, tpl = theory.templates(("", "m", "7", "m7"))
    roots = [theory.parse(l)[0] for l in labels]
    out = []
    for i in range(phase, len(beat_times) - 1, 2):
        a, b = beat_times[i], beat_times[min(i + 2, len(beat_times) - 1)]
        m = (ft >= a) & (ft < b)
        if not m.any():
            continue
        c = np.median(C[:, m], 1)
        c /= np.linalg.norm(c) + 1e-9
        bs = np.median(B[:, m], 1)
        bs /= bs.max() + 1e-9
        sc = tpl @ c + 0.15 * bs[roots]
        sc[24:] -= 0.04
        o = np.argsort(-sc)[:2]
        out.append({"t": round(float(a), 2), "chord": labels[o[0]], "second": labels[o[1]],
                    "margin": round(float(sc[o[0]] - sc[o[1]]), 3)})
    return out


def tempo_curve(beat_times: np.ndarray, window: float = 30.0) -> list[dict]:
    """Темп по полуминутам: живые записи часто разгоняются к концу."""
    bt = np.asarray(beat_times)
    out = []
    for a in np.arange(0, bt[-1], window):
        m = (bt >= a) & (bt < a + window)
        if m.sum() > 4:
            out.append({"t": round(float(a)), "bpm": round(60.0 / float(np.median(np.diff(bt[m]))))})
    return out


def downbeat_phase(grid_chords: list[str], beats_per_bar: int = 4) -> int:
    """Сильная доля: на какую фазу чаще всего приходится смена аккорда."""
    counts = np.zeros(beats_per_bar)
    for i in range(1, len(grid_chords)):
        if grid_chords[i] != grid_chords[i - 1]:
            counts[i % beats_per_bar] += 1
    return int(np.argmax(counts)) if counts.sum() else 0


def analyze(path: str, bpm: float | None = None, sevenths: bool = False, nn: bool = True,
            vocab: list[str] | None = None) -> tuple[dict, np.ndarray, int]:
    y, sr = load(path)
    y_harm, y_perc = librosa.effects.hpss(y)
    net = beats_nn(path) if nn and bpm is None else None
    if net:
        beat_times = net["beats"]
        tempo = 60.0 / float(np.median(np.diff(beat_times)))
        n_head = int(np.sum(_extend(beat_times, len(y) / sr) < beat_times[0] - 1e-6))
        beat_times = _extend(beat_times, len(y) / sr)
        net["phase"] = (net["phase"] + n_head) % net["beats_per_bar"]
    else:
        tempo, beat_times = beats(y_perc, sr, bpm)
    if len(beat_times) < 4:
        raise RuntimeError("Не удалось найти доли — запись слишком короткая или без ритма")
    if vocab and any(theory.parse(v) and theory.parse(v)[1].endswith("7") for v in vocab):
        sevenths = True  # на экране есть септаккорды — нужны их шаблоны
    res = chords(y_harm, sr, beat_times, sevenths=sevenths, vocab=vocab)
    res["tempo"] = round(tempo, 1)
    res["beats"] = [round(float(t), 3) for t in beat_times]
    res["duration"] = round(len(y) / sr, 2)
    # аккорды по долям (без затакта до первой доли)
    res["beat_chords"] = res["grid_chords"][res["grid_pickup"]:][: len(beat_times)]
    if net:
        res["downbeat_phase"], res["beats_per_bar"], res["beat_source"] = net["phase"], net["beats_per_bar"], net["source"]
        res["downbeats"] = [round(float(t), 3) for t in net["downbeats"]]
    else:
        res["downbeat_phase"], res["beats_per_bar"], res["beat_source"] = downbeat_phase(res["beat_chords"]), 4, "librosa"
    res["tempo_curve"] = tempo_curve(beat_times)
    res["halfbars"] = halfbars(y_harm, sr, beat_times, res["downbeat_phase"])
    return res, y, sr
