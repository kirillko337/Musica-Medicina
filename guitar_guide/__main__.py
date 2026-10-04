"""python -m guitar_guide <ссылка на YouTube или файл> [--out guides]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio, compare, fetch, report, strum, theory, vision


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Разбор песни на гитаре: аккорды, бой, сверка со зрением")
    ap.add_argument("source", help="ссылка на YouTube или путь к видео/аудио")
    ap.add_argument("--out", default="guides")
    ap.add_argument("--work", default=".work")
    ap.add_argument("--title")
    ap.add_argument("--bpm", type=float, help="задать темп вручную, если автоматика ошиблась вдвое")
    ap.add_argument("--sevenths", action="store_true", help="искать септаккорды (7, m7)")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--max-vision-calls", type=int, default=24)
    ap.add_argument("--lyrics", action="store_true", help="распознать слова (Demucs + Whisper, нужен интернет)")
    ap.add_argument("--lang", help="язык песни для Whisper: es, ru, en… (по умолчанию определит сам)")
    args = ap.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.source.startswith(("http://", "https://")):
        meta = fetch.download(args.source, work)
        audio_src = meta["video"]
    else:
        meta = fetch.local(args.source)
        audio_src = meta["audio_src"]
    if args.title:
        meta["title"] = args.title
    wav = fetch.extract_audio(audio_src, work)

    print("▶ звук: темп, доли, аккорды…", flush=True)
    res, y, sr = audio.analyze(wav, bpm=args.bpm, sevenths=args.sevenths)
    print(f"  темп {res['tempo']} BPM, тональность {res['key']}, сегментов {len(res['segments'])}")

    print("▶ звук: бой…", flush=True)
    st = strum.analyze(y, sr, res["beats"], res["downbeat_phase"])
    print(f"  рисунок: {strum.short(st['pattern'])}")

    weights = [s["end"] - s["start"] for s in res["segments"]]
    capo_opts = theory.best_capo([s["chord"] for s in res["segments"]], weights)
    capo_audio = capo_opts[0]["capo"]
    capo, capo_src = capo_audio, "подобран по звуку под открытые аккорды"

    vision_info, vis_rows = None, []
    if not args.no_vision and meta.get("video") and vision.available():
        print("▶ видео: обзор кадра…", flush=True)
        over = vision.overview(meta["video"], res["duration"]) or {}
        print(f"  {json.dumps(over, ensure_ascii=False)}")
        print("▶ видео: аккорды по кадрам…", flush=True)
        vis_rows = vision.chords(meta["video"], res["segments"], over.get("capo_fret"), args.max_vision_calls)
        capo_video = compare.capo_from_vision(over, vis_rows)
        if capo_video is not None:
            capo, capo_src = capo_video, "виден на видео"
        print("▶ видео: движение правой руки…", flush=True)
        flow = vision.strum_flow(meta["video"], over.get("strum_hand_box"), [o["t"] for o in st["onsets"]])
        vision_info = {
            "overview": over,
            "capo_audio": capo_audio,
            "capo_video": capo_video if capo_video is not None else "не видно",
            "chords": compare.chords(vis_rows, capo),
            "strums": compare.strums(st["onsets"], flow),
        }
    elif not args.no_vision:
        print("▶ видео пропущено: нет файла видео или ANTHROPIC_API_KEY")

    words = None
    if args.lyrics:
        from . import lyrics
        print("▶ слова: отделяю голос и расшифровываю…", flush=True)
        words = lyrics.analyze(wav, work, args.lang)
        print(f"  строк: {len(words['lines'])}")

    segs = compare.consensus(res["segments"], vision_info["chords"]["rows"] if vision_info else [], capo)
    name = fetch.slug(meta.get("title") or meta.get("id") or "song")
    out_dir = Path(args.out) / name
    path = report.write(out_dir, meta, res, st, segs, capo, capo_src, vision_info, words)

    dump = {"meta": {k: v for k, v in meta.items() if k not in ("video", "audio_src")}, "audio": res,
            "strum": st, "capo": capo, "capo_options": capo_opts[:3], "segments": segs,
            "vision": vision_info, "vision_raw": vis_rows, "lyrics": words}
    (out_dir / "analysis.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✔ гайд: {path}")
    gh_out = Path(__import__("os").environ.get("GITHUB_OUTPUT", "/dev/null"))
    with open(gh_out, "a") as f:
        f.write(f"guide_dir={out_dir}\nguide_md={path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
