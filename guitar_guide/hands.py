"""Руки на видео (MediaPipe Hands): какой палец играет каждую ноту и какую форму держит левая рука.

Правая рука: для каждого пальца считаем, насколько кончик приближается к основанию (сгиб).
Щипок струны — резкий сгиб. По всем нотам подбираем сдвиг видео относительно звука,
затем для каждой ноты находим палец, который сгибается в этот момент.

Левая рука: 21 точка кисти, нормированные на положение и размер ладони, — «отпечаток» формы.
Для каждого такта берём медианный отпечаток и сравниваем с отпечатками тактов, где аккорд
по звуку надёжен. Так видео независимо подтверждает или исправляет аккорд такта.
"""
from __future__ import annotations

import os
import urllib.request
from collections import Counter, defaultdict

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")
FINGERS = {"thumb": (4, 2), "index": (8, 5), "middle": (12, 9), "ring": (16, 13)}
FINGER_RU = {"thumb": "большой (p)", "index": "указательный (i)", "middle": "средний (m)", "ring": "безымянный (a)"}
ORDER = ["thumb", "index", "middle", "ring"]  # от басовых струн к тонким


def available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except Exception:
        return False


def model_path() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "guitar_guide")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "hand_landmarker.task")
    if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def track(video: str, max_side: int = 1280) -> dict:
    """Точки обеих рук в каждом кадре. Слот 0 — рука левее в кадре, 1 — правее
    (у гитариста руки не перекрещиваются, так что порядок по x устойчив)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opt = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path()), num_hands=2,
                                       running_mode=vision.RunningMode.VIDEO, min_hand_detection_confidence=0.3,
                                       min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
    det = vision.HandLandmarker.create_from_options(opt)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        sc = min(1.0, max_side / max(h, w))
        if sc < 1:
            fr = cv2.resize(fr, (int(w * sc), int(h * sc)))
            h, w = fr.shape[:2]
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
                                 int(i * 1000 / fps))
        row = np.full((2, 21, 3), np.nan)
        found = [np.array([[p.x * w, p.y * h, p.z * w] for p in lm]) for lm in r.hand_landmarks]
        found.sort(key=lambda a: a[0, 0])
        if len(found) == 2:
            row[0], row[1] = found
        elif len(found) == 1 and rows:
            prev = rows[-1]
            d = [np.nanmean(np.abs(prev[k, 0, :2] - found[0][0, :2])) if not np.isnan(prev[k, 0, 0]) else 1e9
                 for k in (0, 1)]
            row[int(np.argmin(d))] = found[0]
        rows.append(row)
        i += 1
    cap.release()
    try:
        det.close()
    except Exception:
        pass
    return {"fps": float(fps), "hands": np.array(rows), "size": (w, h) if rows else (0, 0)}


def _fill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    bad = np.isnan(x)
    if bad.all():
        return x
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def _points(hand: np.ndarray) -> np.ndarray:
    P = hand[..., :2].copy()
    for j in range(21):
        for c in range(2):
            P[:, j, c] = _fill(P[:, j, c])
    return P


def finger_signals(hand: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Скорость сгиба каждого пальца (отрицательная = кончик идёт к ладони), в размерах ладони в секунду."""
    P = _points(hand)
    size = np.linalg.norm(P[:, 9] - P[:, 0], axis=1)
    size = np.where(size > 1, size, np.nanmedian(size))
    out = {}
    for f, (tip, base) in FINGERS.items():
        ref = 5 if f == "thumb" else base  # большой меряем до основания указательного
        d = np.linalg.norm(P[:, tip] - P[:, ref], axis=1) / size
        out[f] = np.gradient(d) * fps
    return out


def _response(sig: dict, times: np.ndarray, fps: float, lag: float) -> dict[str, float]:
    t = np.arange(len(next(iter(sig.values())))) / fps
    return {f: float(np.mean(np.interp(times + lag, t, v))) for f, v in sig.items()}


def picking_hand(tr: dict, note_times: list[float]) -> int:
    """Рука, пальцы которой сильнее всего сгибаются в моменты нот."""
    best, score = 0, -1e9
    times = np.array(note_times)
    for k in (0, 1):
        if np.mean(~np.isnan(tr["hands"][:, k, 0, 0])) < 0.3:
            continue
        sig = finger_signals(tr["hands"][:, k], tr["fps"])
        s = max(-min(_response(sig, times, tr["fps"], lag).values()) for lag in np.arange(-0.2, 0.21, 0.04))
        if s > score:
            best, score = k, s
    return best


def _mutual_info(pairs: list[tuple]) -> float:
    if not pairs:
        return 0.0
    n = len(pairs)
    pxy = Counter(pairs)
    px = Counter(p[0] for p in pairs)
    py = Counter(p[1] for p in pairs)
    return float(sum(c / n * np.log((c / n) / (px[x] / n * py[y] / n)) for (x, y), c in pxy.items()))


def estimate_lag(sig: dict, notes: list[dict], fps: float) -> tuple[float, float]:
    """Сдвиг видео относительно звука.

    Ноты в переборе идут почти периодично, поэтому «где пальцы сильнее сгибаются» даёт ложные
    максимумы через ноту. Берём другой признак: при верном сдвиге каждая струна играется
    своим пальцем, то есть взаимная информация «струна ↔ палец» максимальна."""
    single = [n for n in notes if n.get("string") and not n.get("ghost")]
    tt = np.arange(len(next(iter(sig.values())))) / fps
    zs = {}
    for f, v in sig.items():
        mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-6
        zs[f] = v / mad
    rows = []
    for lag in np.arange(-0.3, 0.301, 0.01):
        tmp = [dict(n) for n in single]
        fingers_for_notes(tmp, sig, fps, float(lag))
        pairs = [(n["string"], n["finger"]) for n in tmp if n.get("finger")]
        mi = _mutual_info(pairs) * len(pairs) / max(1, len(tmp))
        # глубина сгиба точно в момент ноты (интерполяция между кадрами, без окна)
        depth = float(np.mean([-np.interp(n["t"] + lag, tt, zs[n["finger"]]) for n in tmp if n.get("finger")])) \
            if pairs else 0.0
        rows.append((round(float(lag), 2), mi, depth))
    best_mi = max(r[1] for r in rows)
    # при почти равной связи струна↔палец берём сдвиг, где сгибы на нотах глубже всего
    lag, mi, _ = max((r for r in rows if r[1] >= 0.97 * best_mi), key=lambda r: r[2])
    return lag, mi


def fingers_for_notes(notes: list[dict], sig: dict, fps: float, lag: float, tol: float = 0.035) -> None:
    """Палец для каждой ноты. Одновременные ноты (щипок) раздаём активным пальцам по порядку:
    большой — самой низкой струне, дальше указательный, средний, безымянный."""
    t = np.arange(len(next(iter(sig.values())))) / fps
    z = {}
    for f, v in sig.items():
        mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-6
        z[f] = v / mad
    win = max(1, int(round(0.05 * fps)))

    def strength(f, tt, raw=False):
        i = int(round((tt + lag) * fps))
        a, b = max(0, i - win), min(len(t), i + win + 1)
        src = sig[f] if raw else z[f]
        return float(-np.min(src[a:b])) if b > a else 0.0

    order = sorted(range(len(notes)), key=lambda i: notes[i]["t"])
    groups, cur = [], []
    for i in order:
        if cur and notes[i]["t"] - notes[cur[0]]["t"] > tol:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    for g in groups:
        tt = float(np.mean([notes[i]["t"] for i in g]))
        st = {f: strength(f, tt) for f in ORDER}
        raw = {f: strength(f, tt, raw=True) for f in ORDER}
        # большой против остальных — по нормированной силе (у него другая геометрия);
        # среди указательного, среднего и безымянного — по реальной скорости в размерах ладони:
        # безымянный связан со средним и при нормировке «раздувается»
        fing = max(("index", "middle", "ring"), key=lambda f: raw[f])
        cand = {"thumb": st["thumb"], fing: st[fing]}
        active = [f for f in ORDER if f in cand and cand[f] >= 1.5]
        others = sorted((f for f in ("index", "middle", "ring") if f != fing and st[f] >= 1.5
                         and raw[f] >= 0.6 * raw[fing]), key=ORDER.index)
        g_sorted = sorted(g, key=lambda i: -(notes[i].get("string") or 0))  # от 6-й к 1-й
        if not active:
            for i in g:
                notes[i]["finger"], notes[i]["finger_z"] = None, round(max(st.values()), 2)
            continue
        if len(g_sorted) == 1:
            f = max(active, key=lambda f: cand[f])
            notes[g_sorted[0]]["finger"], notes[g_sorted[0]]["finger_z"] = f, round(st[f], 2)
            continue
        # несколько нот (щипок): по порядку струн и пальцев
        act = sorted(set(active) | set(others), key=ORDER.index)
        for k, i in enumerate(g_sorted):
            f = act[min(k, len(act) - 1)]
            notes[i]["finger"], notes[i]["finger_z"] = f, round(st[f], 2)


def rescue_by_video(notes: list[dict], rejected: list[dict], sig: dict, fps: float, lag: float,
                    min_share: float = 0.5) -> list[dict]:
    """Возвращает отброшенные фильтром ноты, если палец, который обычно играет эту струну,
    в этот момент сгибается не слабее, чем на принятых нотах этой струны (медиана)."""
    z = {}
    for f, v in sig.items():
        mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-6
        z[f] = v / mad
    win = max(1, int(round(0.05 * fps)))

    def zat(f, t):
        i = int(round((t + lag) * fps))
        a, b = max(0, i - win), min(len(z[f]), i + win + 1)
        return float(-np.min(z[f][a:b])) if b > a else 0.0
    by_string = defaultdict(Counter)
    for n in notes:
        if n.get("string") and n.get("finger"):
            by_string[n["string"]][n["finger"]] += 1
    owner, level = {}, {}
    for s_, c in by_string.items():
        f, k = c.most_common(1)[0]
        if k / sum(c.values()) >= min_share:
            owner[s_] = f
            level[s_] = float(np.median([zat(f, n["t"]) for n in notes if n.get("string") == s_]))
    rescued = []
    taken = [(n["midi"], n["t"]) for n in notes]
    for n in rejected:
        s_ = n.get("string")
        if not s_ or not n.get("in_shape") or s_ not in owner:
            continue
        if any(m == n["midi"] and abs(t - n["t"]) < 0.06 for m, t in taken):
            continue  # эта нота уже есть среди принятых

        v = zat(owner[s_], n["t"])
        if v >= max(2.0, level[s_]):
            n.update(real=True, rescued_video=True, finger=owner[s_], finger_z=round(v, 2), ghost=False)
            rescued.append(n)
            taken.append((n["midi"], n["t"]))
    return rescued


def finger_table(notes: list[dict]) -> dict:
    """Струна → какими пальцами её играли (по видео)."""
    tab = defaultdict(Counter)
    for n in notes:
        if n.get("string") and not n.get("ghost"):
            tab[n["string"]][n.get("finger") or "?"] += 1
    return {s: dict(c) for s, c in sorted(tab.items())}


# ───────────────────────── левая рука ─────────────────────────

def shape_print(hand: np.ndarray) -> np.ndarray:
    """(T,21,3) → (T,40): точки в системе ладони (запястье в нуле, запястье→средний палец по оси x)."""
    P = _points(hand)
    out = np.zeros((len(P), 40))
    for i, p in enumerate(P):
        q = p - p[0]
        ax = q[9]
        n = np.linalg.norm(ax) + 1e-9
        c, s = ax[0] / n, ax[1] / n
        R = np.array([[c, s], [-s, c]])
        out[i] = (q[1:] @ R.T / n).ravel()
    return out


def bar_shapes(tr: dict, hand_idx: int, bars: list[dict], trim: float = 0.2) -> list[np.ndarray | None]:
    """Медианный отпечаток левой руки на такт (без краёв такта, где пальцы переставляются)."""
    fps = tr["fps"]
    H = tr["hands"][:, hand_idx]
    ok = ~np.isnan(H[:, 0, 0])
    F = shape_print(H)
    out = []
    for b in bars:
        d = b["end"] - b["start"]
        a, z = int((b["start"] + trim * d) * fps), int((b["end"] - trim * d) * fps)
        idx = [i for i in range(max(0, a), min(len(F), z)) if ok[i]]
        out.append(np.median(F[idx], axis=0) if len(idx) >= 3 else None)
    return out


def check_bars(prints: list, labels: list[str], conf: list[float] | None = None) -> list[dict]:
    """Группы тактов по форме левой руки (без меток), затем имя группы — по большинству аккордов
    по звуку в ней. Так одиночная ошибка звука не становится эталоном."""
    idx = [i for i, p in enumerate(prints) if p is not None]
    res = [{"video": None, "group": None, "agree": None} for _ in prints]
    if len(idx) < 3:
        return res
    D = np.array([[np.linalg.norm(prints[i] - prints[j]) for j in idx] for i in idx])
    # порог от данных: форма руки в тактах с одним аккордом почти совпадает (ближайший сосед
    # близко), а разные аккорды заметно дальше
    nn = np.sort(D + np.eye(len(idx)) * 1e9, axis=1)[:, 0]
    tau = 2.5 * float(np.median(nn))
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if D[a, b] < tau:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for k, i in enumerate(idx):
        groups[find(k)].append(i)
    for gid, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        for i in members:
            votes = Counter(labels[j] for j in members if j != i)
            name = votes.most_common(1)[0][0] if votes else None
            share = votes.most_common(1)[0][1] / sum(votes.values()) if votes else None
            res[i] = {"video": name, "group": gid, "group_size": len(members),
                      "agree": None if name is None else name == labels[i],
                      "group_share": None if share is None else round(share, 2)}
    return res


def analyze(video: str, notes: list[dict], bars: list[dict], rejected: list[dict] | None = None) -> dict:
    """Руки: сдвиг видео, палец на каждую ноту, проверка аккордов по левой руке."""
    tr = track(video)
    times = [n["t"] for n in notes if not n.get("ghost")]
    pk = picking_hand(tr, times)
    fr = 1 - pk
    sig = finger_signals(tr["hands"][:, pk], tr["fps"])
    lag, mi = estimate_lag(sig, [n for n in notes if not n.get("ghost")], tr["fps"])
    fingers_for_notes(notes, sig, tr["fps"], lag)
    rescued = rescue_by_video(notes, rejected or [], sig, tr["fps"], lag)
    prints = bar_shapes(tr, fr, bars)
    checks = check_bars(prints, [b["chord"] for b in bars])
    seen = {k: float(np.mean(~np.isnan(tr["hands"][:, k, 0, 0]))) for k in (0, 1)}
    return {"lag": lag, "lag_mi": round(mi, 3), "picking_hand": pk, "seen_pick": round(seen[pk], 3), "seen_fret": round(seen[fr], 3),
            "fingers": finger_table(notes), "bar_checks": checks, "fps": tr["fps"], "rescued": rescued}
