"""Распознавание по видео.

1. Claude смотрит на кадры: каподастр, где правая рука, какую форму аккорда держит левая.
2. OpenCV считает оптический поток в зоне правой руки: рука идёт вниз или вверх в момент удара.
"""
from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

MODEL = os.environ.get("GUIDE_MODEL", "claude-opus-5-5")

_OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "guitar_visible": {"type": "boolean"},
        "capo_fret": {"type": "integer", "description": "0 если каподастра нет, -1 если не видно"},
        "mirrored": {"type": "boolean", "description": "true, если картинка похожа на зеркальное селфи"},
        "left_handed_player": {"type": "boolean"},
        "strum_hand_box": {
            "type": "array", "items": {"type": "number"},
            "description": "[x, y, w, h] зоны, где правая (бьющая) рука проходит по струнам, доли кадра 0..1",
        },
        "notes": {"type": "string"},
    },
    "required": ["guitar_visible", "capo_fret", "mirrored", "left_handed_player", "strum_hand_box", "notes"],
    "additionalProperties": False,
}

_CHORD_SCHEMA = {
    "type": "object",
    "properties": {
        "fretboard_visible": {"type": "boolean"},
        "shape": {"type": "string", "description": "Название формы аккорда относительно каподастра, например Am, G, F#m. 'unknown', если не разобрать"},
        "frets": {
            "type": "array", "items": {"type": "integer"},
            "description": "Лады от 6-й струны к 1-й относительно каподастра: -1 не играется, -2 не видно, 0 открытая",
        },
        "capo_fret": {"type": "integer"},
        "confidence": {"type": "number", "description": "0..1"},
        "notes": {"type": "string"},
    },
    "required": ["fretboard_visible", "shape", "frets", "capo_fret", "confidence", "notes"],
    "additionalProperties": False,
}

SYSTEM = (
    "Ты гитарист-преподаватель и разбираешь видео, где человек играет на гитаре. "
    "Тебе дают кадры из ролика. Смотри на руки и гриф, а не на подписи или текст на экране. "
    "Аккорд называй как форму, которую держит левая рука относительно каподастра "
    "(с каподастром на 2-м ладу и формой Am пиши shape='Am', capo_fret=2). "
    "Учитывай, что селфи-видео бывает зеркальным: тогда басовые струны визуально с другой стороны. "
    "Если по кадру нельзя уверенно сказать, ставь низкую confidence и пиши 'unknown' — "
    "угадывать не нужно, твой ответ будут сравнивать с распознаванием по звуку."
)


def _client():
    import anthropic
    return anthropic.Anthropic()


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def frame_at(video: str, t: float, max_side: int = 1280) -> np.ndarray | None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        frame = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return frame


def _img_block(frame: np.ndarray) -> dict:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode()}}


def _ask(content: list[dict], schema: dict, effort: str = "high") -> dict | None:
    client = _client()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(text) if text else None


def overview(video: str, duration: float) -> dict | None:
    times = [duration * k for k in (0.2, 0.45, 0.7)]
    frames = [f for f in (frame_at(video, t) for t in times) if f is not None]
    if not frames:
        return None
    content = [_img_block(f) for f in frames] + [{
        "type": "text",
        "text": "Три кадра из одного ролика. Определи: видна ли гитара, есть ли каподастр и на каком ладу, "
                "зеркальное ли видео, левша ли исполнитель, и прямоугольник, внутри которого правая рука "
                "бьёт по струнам (над розеткой/звукоснимателем, с запасом на размах руки).",
    }]
    return _ask(content, _OVERVIEW_SCHEMA)


def chord_at(video: str, seg: dict, capo_hint: int | None, audio_shape: str | None = None) -> dict | None:
    """Форма аккорда в сегменте. Звуковую версию модели НЕ показываем, чтобы сравнение было честным."""
    length = seg["end"] - seg["start"]
    ts = [seg["start"] + length * k for k in ((0.35, 0.7) if length > 1.2 else (0.5,))]
    frames = [f for f in (frame_at(video, t) for t in ts) if f is not None]
    if not frames:
        return None
    hint = f"Каподастр, по-видимому, на {capo_hint}-м ладу. " if capo_hint and capo_hint > 0 else ""
    content = [_img_block(f) for f in frames] + [{
        "type": "text",
        "text": f"{hint}Кадры сняты внутри одного аккорда ({seg['start']:.1f}–{seg['end']:.1f} с). "
                "Какую форму аккорда держит левая рука? Перечисли лады по струнам, если видно.",
    }]
    return _ask(content, _CHORD_SCHEMA)


def chords(video: str, segments: list[dict], capo_hint: int | None, max_calls: int = 24, workers: int = 4) -> list[dict]:
    cand = [s for s in segments if s["chord"] != "N" and s["end"] - s["start"] >= 0.6]
    if len(cand) > max_calls:
        idx = np.linspace(0, len(cand) - 1, max_calls).round().astype(int)
        cand = [cand[i] for i in sorted(set(idx))]

    def one(seg):
        try:
            r = chord_at(video, seg, capo_hint)
        except Exception as e:  # сеть, лимиты — не валим весь разбор
            r = {"error": str(e)[:200]}
        return {"start": seg["start"], "end": seg["end"], "audio": seg["chord"], "vision": r}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, cand))


def strum_flow(video: str, box: list[float], onset_times: list[float]) -> list[dict]:
    """Вертикальный оптический поток в зоне правой руки. Плюс — рука идёт вниз (к полу)."""
    if not box or len(box) != 4 or not onset_times:
        return []
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x, y, w, h = box
    x0, y0 = int(max(0, x) * W), int(max(0, y) * H)
    x1, y1 = int(min(1, x + w) * W), int(min(1, y + h) * H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        cap.release()
        return []
    scale = 160 / max(1, x1 - x0)

    t_end = max(onset_times) + 0.5
    prev, vy, ts = None, [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t > t_end:
            break
        roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, roi, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            moving = mag > np.percentile(mag, 80)  # берём только движущееся — руку, а не гитару
            vy.append(float(flow[..., 1][moving].mean()) if moving.any() else 0.0)
            ts.append(t - 0.5 / fps)
        prev = roi
    cap.release()
    if len(vy) < 5:
        return []
    vy, ts = np.array(vy), np.array(ts)
    full = float(np.percentile(np.abs(vy), 90)) + 1e-6  # «полный размах» руки в этом ролике
    out = []
    for t in onset_times:
        m = (ts >= t - 0.07) & (ts <= t + 0.03)
        if not m.any():
            out.append({"t": t, "dir_video": "?", "conf": 0.0})
            continue
        v = float(vy[m].mean())
        conf = min(1.0, abs(v) / full)
        d = "?" if conf < 0.2 else ("down" if v > 0 else "up")
        out.append({"t": t, "dir_video": d, "conf": round(conf, 2)})
    return out
