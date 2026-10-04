"""Аккорды: названия, шаблоны для распознавания, транспонирование, аппликатуры."""
from __future__ import annotations

import re

import numpy as np

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
_ALIASES = {"Db": "C#", "D#": "Eb", "Gb": "F#", "Ab": "G#", "A#": "Bb", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

QUALITIES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
}

# Открытые аппликатуры, струны 6→1, -1 = не играть.
OPEN_SHAPES = {
    "C": [-1, 3, 2, 0, 1, 0],
    "C7": [-1, 3, 2, 3, 1, 0],
    "D": [-1, -1, 0, 2, 3, 2],
    "Dm": [-1, -1, 0, 2, 3, 1],
    "D7": [-1, -1, 0, 2, 1, 2],
    "Dm7": [-1, -1, 0, 2, 1, 1],
    "E": [0, 2, 2, 1, 0, 0],
    "Em": [0, 2, 2, 0, 0, 0],
    "E7": [0, 2, 0, 1, 0, 0],
    "Em7": [0, 2, 2, 0, 3, 0],
    "G": [3, 2, 0, 0, 0, 3],
    "G7": [3, 2, 0, 0, 0, 1],
    "A": [-1, 0, 2, 2, 2, 0],
    "Am": [-1, 0, 2, 2, 1, 0],
    "A7": [-1, 0, 2, 0, 2, 0],
    "Am7": [-1, 0, 2, 0, 1, 0],
    "B7": [-1, 2, 1, 2, 0, 2],
}
EASY = set(OPEN_SHAPES)

_E_SHAPE = {"": [0, 2, 2, 1, 0, 0], "m": [0, 2, 2, 0, 0, 0], "7": [0, 2, 0, 1, 0, 0], "m7": [0, 2, 0, 0, 0, 0]}
_A_SHAPE = {"": [None, 0, 2, 2, 2, 0], "m": [None, 0, 2, 2, 1, 0], "7": [None, 0, 2, 0, 2, 0], "m7": [None, 0, 2, 0, 1, 0]}

_CHORD_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*(maj7|m7|min7|m|min|7|maj|dim|aug|sus2|sus4|add9|6|9)?", re.I)


def parse(name: str) -> tuple[int, str] | None:
    """'Am7' -> (9, 'm7'). Неизвестные качества сводим к ближайшим из QUALITIES."""
    if not name or name.upper() in {"N", "NC", "N.C."}:
        return None
    m = _CHORD_RE.match(name)
    if not m:
        return None
    note = m.group(1).upper() + (m.group(2) or "")
    note = _ALIASES.get(note, note)
    if note not in NOTES:
        return None
    q = (m.group(3) or "")
    ql = q.lower()
    if q in ("m", "min") or ql == "min":
        qual = "m"
    elif q == "m7" or ql == "min7":
        qual = "m7"
    elif ql == "7" or ql == "9":
        qual = "7"
    else:  # maj, maj7, sus, add9, 6 и т.п. — для сравнения считаем мажором
        qual = ""
    return NOTES.index(note), qual


def name(root: int, qual: str) -> str:
    return NOTES[root % 12] + qual


def transpose(chord: str, semitones: int) -> str:
    p = parse(chord)
    if p is None:
        return chord
    return name(p[0] + semitones, p[1])


def triad(chord: str) -> str:
    p = parse(chord)
    if p is None:
        return "N"
    return name(p[0], "m" if p[1].startswith("m") else "")


def same(a: str, b: str) -> str:
    """exact / triad / root / no — насколько совпадают два аккорда."""
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return "exact" if pa == pb else "no"
    if pa == pb:
        return "exact"
    if triad(a) == triad(b):
        return "triad"
    if pa[0] == pb[0]:
        return "root"
    return "no"


def templates(qualities=("", "m")) -> tuple[list[str], np.ndarray]:
    labels, rows = [], []
    for q in qualities:
        for r in range(12):
            v = np.zeros(12)
            for i, iv in enumerate(QUALITIES[q]):
                v[(r + iv) % 12] = 1.0 if i < 3 else 0.7
            labels.append(name(r, q))
            rows.append(v / np.linalg.norm(v))
    return labels, np.array(rows)


def shape_for(chord: str) -> dict | None:
    """Аппликатура: открытая, если есть, иначе баррэ от 6-й или 5-й струны."""
    p = parse(chord)
    if p is None:
        return None
    key = name(*p)
    if key in OPEN_SHAPES:
        return {"chord": key, "frets": OPEN_SHAPES[key], "barre": None}
    root, q = p
    e_fret = (root - 4) % 12 or 12
    a_fret = (root - 9) % 12 or 12
    if e_fret <= a_fret:
        frets = [f + e_fret for f in _E_SHAPE[q]]
        return {"chord": key, "frets": frets, "barre": e_fret}
    frets = [-1 if f is None else f + a_fret for f in _A_SHAPE[q]]
    return {"chord": key, "frets": frets, "barre": a_fret}


def frets_str(frets: list[int]) -> str:
    out = []
    for f in frets:
        out.append("x" if f < 0 else (str(f) if f < 10 else f"({f})"))
    return "".join(out)


def best_capo(chords: list[str], weights: list[float] | None = None, max_capo: int = 7) -> list[dict]:
    """Варианты каподастра: сколько звучащих аккордов ложится в открытые формы."""
    weights = weights or [1.0] * len(chords)
    total = sum(w for c, w in zip(chords, weights) if parse(c)) or 1.0
    res = []
    for capo in range(0, max_capo + 1):
        shapes = [transpose(c, -capo) for c in chords]
        easy = sum(w for s, w in zip(shapes, weights) if parse(s) and name(*parse(s)) in EASY)
        res.append({"capo": capo, "easy_share": easy / total})
    # лучше больше открытых, при равенстве — каподастр пониже
    res.sort(key=lambda r: (-round(r["easy_share"], 3), r["capo"]))
    return res


def chord_svg(chord: str, frets: list[int]) -> str:
    """Небольшая SVG-схема аккорда (струны 6→1 слева направо)."""
    played = [f for f in frets if f > 0]
    base = 1
    if played and max(played) > 4:
        base = min(played)
    n_rows = 4 if not played else max(4, max(played) - base + 1)
    w, h, left, top, sx, sy = 120, 40 + n_rows * 22 + 16, 22, 40, 16, 22
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w/2}" y="16" text-anchor="middle" font-size="15" font-weight="bold">{chord}</text>',
    ]
    for i in range(6):
        x = left + i * sx
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + n_rows * sy}" stroke="#333" stroke-width="1"/>')
    for r in range(n_rows + 1):
        y = top + r * sy
        sw = 4 if (r == 0 and base == 1) else 1
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + 5 * sx}" y2="{y}" stroke="#333" stroke-width="{sw}"/>')
    if base > 1:
        parts.append(f'<text x="{left + 5 * sx + 6}" y="{top + sy * 0.7}" font-size="11">{base}</text>')
    for i, f in enumerate(frets):
        x = left + i * sx
        if f < 0:
            parts.append(f'<text x="{x}" y="{top - 6}" text-anchor="middle" font-size="12">×</text>')
        elif f == 0:
            parts.append(f'<circle cx="{x}" cy="{top - 10}" r="4" fill="none" stroke="#333"/>')
        else:
            y = top + (f - base + 0.5) * sy
            parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#222"/>')
    parts.append("</svg>")
    return "\n".join(parts)
