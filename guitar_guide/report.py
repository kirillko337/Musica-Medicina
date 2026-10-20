"""Сборка гайда в Markdown."""
from __future__ import annotations

from collections import Counter
from pathlib import Path


from . import strum, theory


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def bars(beats: list[float], segs: list[dict], phase: int, per_bar: int = 4) -> list[dict]:
    def shape_at(t):
        for s in segs:
            if s["start"] <= t < s["end"]:
                return s["shape"]
        return "N"

    out = []
    for i in range(phase, len(beats) - per_bar + 1, per_bar):
        names = [shape_at(beats[j] + 0.02) for j in range(i, i + per_bar)]
        uniq = [n for k, n in enumerate(names) if k == 0 or n != names[k - 1]]
        out.append({"t": beats[i], "chords": uniq})
    return out


def main_loop(bar_list: list[dict], size: int = 4) -> tuple[list[str], int] | None:
    keys = [" ".join(b["chords"]) for b in bar_list]
    cnt = Counter(tuple(keys[i:i + size]) for i in range(0, len(keys) - size + 1))
    cnt = Counter({k: v for k, v in cnt.items() if "N" not in k})
    if not cnt:
        return None
    loop, n = cnt.most_common(1)[0]
    return (list(loop), n) if n >= 2 else None


def _tempo_line(res: dict) -> str:
    curve = [c["bpm"] for c in res.get("tempo_curve") or []]
    if len(curve) >= 2 and (max(curve) - min(curve)) / min(curve) > 0.08:
        return f"~{curve[0]} → {curve[-1]} BPM, по ходу песни ускоряется"
    return f"~{res['tempo']:.0f} BPM"


def _picking_md(pick: dict, shapes: dict, capo: int) -> list[str]:
    from . import picking

    L = []
    for sec in pick["sections"]:
        if sec["style"] != "перебор" or not sec.get("patterns"):
            continue
        L.append(f"### {_mmss(sec['start'])}–{_mmss(sec['end'])}\n")
        shown = 0
        for bass, pat in sec["patterns"].items():
            share = pat["bars_matching"] / max(1, pat["bars"])
            chords = ", ".join(pat["chords"])
            if share < 0.3:
                L.append(f"- Бас на {bass}-й струне ({chords}): схема не повторяется "
                         f"(совпала в {pat['bars_matching']} из {pat['bars']} тактов), здесь играют свободно.\n")
                continue
            shown += 1
            ch = pat["chords"][0]
            L.append(f"**Бас на {bass}-й струне** ({chords}): **{picking.short(pat)}**\n")
            L.append("```")
            L.append(picking.render(pat))
            L.append("```\n")
            t = picking.tab(pat, ch, shapes.get(ch))
            if t:
                L.append(f"Табулатура на {ch}" + (" (лады от каподастра)" if capo else "") + ":\n")
                L.append("```\n" + t + "\n```\n")
            L.append(f"Схема совпала в {pat['bars_matching']} из {pat['bars']} тактов.\n")
        if shown:
            L.append("p — большой палец (басы), i — указательный, m — средний, a — безымянный. "
                     "Цифры — номера струн, 1 — самая тонкая. «4+3» — две струны щипком одновременно.\n")
    return L


def write(out_dir: Path, meta: dict, res: dict, st: dict, segs: list[dict], capo: int, capo_src: str,
          vision_info: dict | None, words: dict | None = None, pick: dict | None = None,
          onscreen: dict | None = None, shapes: dict | None = None) -> Path:
    shapes = shapes or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    cdir = out_dir / "chords"
    cdir.mkdir(exist_ok=True)
    bpb = res.get("beats_per_bar", 4)

    bl = bars(res["beats"], segs, res["downbeat_phase"], bpb)
    dur = Counter()
    for s in segs:
        if s["shape"] != "N":
            dur[s["shape"]] += s["end"] - s["start"]
    used = [c for c, d in dur.most_common() if d >= 1.0] or [c for c, _ in dur.most_common()]
    secs = (pick or {}).get("sections") or []
    has_pick = any(s["style"] == "перебор" and s.get("patterns") for s in secs)
    has_strum = any(s["style"] == "бой" for s in secs) or not secs
    pat = st["pattern"]

    def frets_of(c):
        if c in shapes:
            return {"frets": shapes[c], "barre": None}
        return theory.shape_for(c)

    L = []
    L.append(f"# {meta.get('title') or 'Разбор песни'}\n")
    if meta.get("url"):
        L.append(f"Видео: {meta['url']}\n")

    L.append("## Коротко\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Каподастр | {'нет' if capo == 0 else f'**{capo}-й лад**'} ({capo_src}) |")
    L.append(f"| Аккорды | {' '.join(f'**{c}**' for c in used)} |")
    L.append(f"| Размер и темп | {bpb}/4, {_tempo_line(res)} |")
    if secs:
        parts = "; ".join(f"{_mmss(s['start'])}–{_mmss(s['end'])} {s['style']}" for s in secs if s["style"] != "тихо")
        L.append(f"| Как играть | {parts} |")
    elif pat["slots"]:
        L.append(f"| Бой | {strum.short(pat)} |")
    L.append(f"| Тональность (звучит) | {res['key']} |")
    if abs(res["tuning_cents"]) > 20:
        L.append(f"| Строй | отклонение {res['tuning_cents']:+.0f} центов, подстройтесь под запись |")
    L.append(f"| Длительность | {_mmss(res['duration'])} |\n")

    L.append("## Аккорды\n")
    src = "Аппликатуры взяты со схем в самом ролике. " if shapes else ""
    L.append(f"{src}Формы указаны с учётом каподастра, так, как их держит левая рука.\n")
    imgs, rows = [], []
    for c in used:
        sh = frets_of(c)
        if not sh:
            continue
        fn = c.replace("#", "s") + ".svg"
        (cdir / fn).write_text(theory.chord_svg(c, sh["frets"]), encoding="utf-8")
        imgs.append(f'<img src="chords/{fn}" alt="{c} {theory.frets_str(sh["frets"])}" width="110">')
        barre = f", баррэ на {sh['barre']}-м" if sh.get("barre") else ""
        easy = ""
        if sh.get("barre") and c in theory.SIMPLE:
            fr, note = theory.SIMPLE[c]
            easy = f"`{theory.frets_str(fr)}` — {note}"
        rows.append(f"| **{c}** | `{theory.frets_str(sh['frets'])}`{barre} | {easy} | {dur[c]:.0f} с |")
    L.append(" ".join(imgs) + "\n")
    L.append("| Аккорд | Лады (6→1) | Проще для начала | Сколько звучит |\n|---|---|---|---|")
    L.extend(rows)
    L.append("")

    if has_pick:
        L.append("## Перебор\n")
        L.extend(_picking_md(pick, shapes, capo))

    if has_strum:
        L.append("## Бой\n")
        if pat["slots"]:
            if pat.get("fingerstyle"):
                L.append(f"Запись: **{strum.short(pat)}** (Б — бас большим пальцем, Д — удар по аккорду вниз, "
                         "«-» — пауза)\n")
            else:
                L.append(f"Запись: **{strum.short(pat)}** (Д — вниз, В — вверх, «-» — рука проходит мимо струн)\n")
            L.append("```")
            L.append(strum.render(pat))
            L.append("```\n")
            L.append(f"Этот рисунок совпал в {pat['bars_matching']} из {pat['bars_total']} тактов.\n")
            if pat.get("fingerstyle"):
                L.append("Это игра пальцами: большой палец играет бас на счёт (Б), а между басами пальцы бьют "
                         "по аккорду вниз (↓). Медиатор не нужен.\n")
            else:
                L.append("Рука двигается как маятник без остановок: вниз на счёт, вверх на «и». "
                         "Там, где точка, рука всё равно делает движение, просто не задевает струны.\n")
        else:
            L.append("Чёткий повторяющийся бой не нашёлся: возможно, здесь свободный ритм.\n")

    loop = main_loop(bl)
    L.append("## Последовательность\n")
    if loop:
        L.append(f"Основной круг (×{loop[1]}): **| " + " | ".join(loop[0]) + " |**\n")
    L.append(f"По тактам ({bpb}/4, в строке 4 такта):\n")
    L.append("```")
    for i in range(0, len(bl), 4):
        row = bl[i:i + 4]
        L.append(f"{_mmss(row[0]['t']):>5}  | " + " | ".join(" ".join(b["chords"]).replace("N", "—").ljust(7) for b in row) + " |")
    L.append("```\n")

    if words and words.get("lines"):
        from .lyrics import chord_line

        def chord_at(t):
            for s in segs:
                if s["start"] <= t < s["end"]:
                    return s["shape"]
            return "N"

        L.append("## Текст с аккордами\n")
        L.append("Аккорд стоит над словом, на котором его нужно сменить. Слева время в ролике. "
                 "Текст распознан автоматически: в редких словах возможны ошибки.\n")
        L.append("```")
        for ln in words["lines"]:
            if not ln["words"]:
                L.append(" " * 7 + chord_at(ln["start"]))
                L.append(f"{_mmss(ln['start']):>5}  (поют {ln['end'] - ln['start']:.0f} с, слова не разобраны)")
                L.append("")
                continue
            top, bottom = chord_line(ln["words"], chord_at)
            pad = " " * 7
            L.append(pad + top)
            L.append(f"{_mmss(ln['start']):>5}  " + bottom)
            L.append("")
        L.append("```\n")

    L.append("## Слух против зрения\n")
    if vision_info:
        ca = vision_info["chords"]
        sa = vision_info["strums"]
        L.append("| Что сравнивали | Результат |\n|---|---|")
        L.append(f"| Каподастр | по звуку предложен {vision_info['capo_audio']}, на видео {vision_info['capo_video']} |")
        if ca["agreement"] is not None:
            L.append(f"| Аккорды (где гриф виден) | совпадение {ca['agreement']*100:.0f}% |")
        if sa["checked"]:
            L.append(f"| Направление ударов: «маятник» vs видео | {sa['rule_vs_video']*100:.0f}% из {sa['checked']} ударов |")
            L.append(f"| Направление ударов: акустика vs видео | {sa['audio_vs_video']*100:.0f}% |")
        L.append("")
        L.append("<details><summary>Сравнение по фрагментам</summary>\n")
        L.append("| Время | По звуку | На видео | Уверенность видео | Итог |\n|---|---|---|---|---|")
        for r in ca["rows"]:
            L.append(f"| {_mmss(r['start'])}–{_mmss(r['end'])} | {r['audio_shape']} | {r['vision_shape']} | "
                     f"{r['vision_conf']:.2f} | {r['verdict']} |")
        L.append("\n</details>\n")
        L.append("Где звук и видео разошлись, а видео было уверенным (≥0.8), в гайд пошла версия с видео.\n")
    else:
        L.append("Видеоанализ не запускался (нет `ANTHROPIC_API_KEY` или видео). Аккорды и бой — только по звуку.\n")

    L.append("## Как выучить\n")
    first = used[:4]
    slow = max(50, int(res["tempo"] * 0.7))
    L.append(f"1. Аккорды {', '.join(first)}: поставьте каждый, проверьте, что звенят все струны.")
    if len(first) >= 2:
        L.append(f"2. Смена за минуту: переставляйте {first[0]} ↔ {first[1]} 60 секунд и считайте, сколько раз "
                 "успели. Завтра попробуйте побить свой рекорд. Так же с остальными парами.")
    step = 3
    if has_pick:
        L.append(f"{step}. Перебор на одном аккорде без смен, метроном {slow} BPM, на каждый щелчок одна струна.")
        step += 1
    if has_strum and pat["slots"]:
        L.append(f"{step}. Бой «{strum.short(pat)}» на заглушённых струнах, потом на одном аккорде, метроном {slow} BPM.")
        step += 1
    L.append(f"{step}. Круг аккордов с рисунком на {slow} BPM, затем поднимайте до {res['tempo']:.0f}.")
    L.append(f"{step + 1}. Играйте вместе с видео, по частям: сначала одна часть, потом переходы между ними.\n")

    tools = [f"доли — {res.get('beat_source', 'librosa')}", "аккорды — хрома + Витерби"]
    if res.get("vocab") and res["vocab"].get("used"):
        tools.append("словарь аккордов — подписи на экране")
    if pick:
        tools.append("ноты — basic-pitch")
    if vision_info:
        tools.append("видео — Claude и OpenCV")
    L.append("---\n_Разбор собран автоматически: " + "; ".join(tools) + ". Проверяйте на слух, особенно редкие аккорды._")

    path = out_dir / "README.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
