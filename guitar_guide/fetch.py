"""Скачивание ролика (yt-dlp) или подготовка локального файла."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.lower(), flags=re.U).strip("-")
    return s[:60] or "song"


def download(url: str, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--no-playlist", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
           "--merge-output-format", "mp4", "-o", str(work / "video.%(ext)s"),
           "--write-info-json", "--no-progress"]
    cookies = os.environ.get("YT_COOKIES_FILE")
    if cookies and Path(cookies).exists():
        cmd += ["--cookies", cookies]
    subprocess.run(cmd + [url], check=True)
    info = json.loads((work / "video.info.json").read_text())
    video = next(p for p in work.iterdir() if p.name.startswith("video.") and p.suffix in (".mp4", ".mkv", ".webm"))
    return {"video": str(video), "title": info.get("title"), "id": info.get("id"), "url": url,
            "channel": info.get("channel") or info.get("uploader")}


def local(path: str) -> dict:
    p = Path(path)
    is_video = p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
    return {"video": str(p) if is_video else None, "audio_src": str(p), "title": p.stem, "id": slug(p.stem), "url": None}


def extract_audio(src: str, work: Path) -> str:
    out = work / "audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "22050", str(out)],
                   check=True)
    return str(out)
