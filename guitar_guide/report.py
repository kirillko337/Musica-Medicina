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


def write(out_dir: Path, meta: dict, res: dict, st: dict, segs: list[dict], capo: int, capo_src: str,
          vision_info: dict | None, words: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cdir = out_dir / "chords"
    cdir.mkdir(exist_ok=True)

    bl = bars(res["beats"], segs, res["downbeat_phase"])
    shapes = [s["shape"] for s in segs if s["shape"] != "N"]
    dur = Counter()
    for s in segs:
        if s["shape"] != "N":
            dur[s["shape"]] += s["end"] - s["start"]
    used = [c for c, d in dur.most_common() if d >= 1.0] or [c for c, _ in dur.most_common()]

    L = []
    L.append(f"# {meta.get('title') or 'Разбор песни'}\n")
    if meta.get("url"):
        L.append(f"Видео: {meta['url']}\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Темп | ~{res['tempo']:.0f} BPM |")
    L.append(f"| Тональность (звучит) | {res['key']} |")
    L.append(f"| Каподастр | {'нет' if capo == 0 else f'{capo}-й лад'} ({capo_src}) |")
    if abs(res["tuning_cents"]) > 20:
        L.append(f"| Строй | отклонение {res['tuning_cents']:+.0f} центов — подстройтесь под запись |")
    L.append(f"| Длительность | {_mmss(res['duration'])} |\n")

    L.append("## Аккорды\n")
    L.append("Формы указаны с учётом каподастра — так, как их держит левая рука.\n")
    imgs = []
    for c in used:
        sh = theory.shape_for(c)
        if not sh:
            continue
        fn = c.replace("#", "s") + ".svg"
        (cdir / fn).write_text(theory.chord_svg(c, sh["frets"]), encoding="utf-8")
        imgs.append(f'<img src="chords/{fn}" alt="{c} {theory.frets_str(sh["frets"])}" width="110">')
    L.append(" ".join(imgs) + "\n")
    L.append("| Аккорд | Лады (6→1) | Сколько звучит |\n|---|---|---|")
    for c in used:
        sh = theory.shape_for(c)
        if sh:
            barre = f", баррэ на {sh['barre']}-м" if sh["barre"] else ""
            L.append(f"| **{c}** | `{theory.frets_str(sh['frets'])}`{barre} | {dur[c]:.0f} с |")
    L.append("")

    loop = main_loop(bl)
    L.append("## Последовательность\n")
    if loop:
        L.append(f"Основной квадрат (×{loop[1]}): **| " + " | ".join(loop[0]) + " |**\n")
    L.append("По тактам (4/4, в строке 4 такта):\n")
    L.append("```")
    for i in range(0, len(bl), 4):
        row = bl[i:i + 4]
        L.append(f"{_mmss(row[0]['t']):>5}  | " + " | ".join(" ".join(b["chords"]).replace("N", "—").ljust(7) for b in row) + " |")
    L.append("```\n")

    pat = st["pattern"]
    L.append("## Бой\n")
    if pat["slots"]:
        L.append(f"Запись: **{strum.short(pat)}** (Д — вниз, В — вверх, «-» — рука проходит мимо струн)\n")
        L.append("```")
        L.append(strum.render(pat))
        L.append("```\n")
        line = f"Этот рисунок совпал в {pat['bars_matching']} из {pat['bars_total']} тактов."
        if pat["variants"] and pat["bars_matching"] < pat["bars_total"]:
            line += " В остальных — вариации, обычно на сменах частей."
        L.append(line + "\n")
        L.append("Рука двигается как маятник без остановок: вниз на счёт, вверх на «и». "
                 "Там, где точка, рука всё равно делает движение, просто не задевает струны.\n")
    else:
        L.append("Чёткий повторяющийся бой не нашёлся — возможно, здесь перебор или свободный ритм.\n")

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
    L.append(f"1. Поставьте аккорды {', '.join(first)} и переставляйте их по кругу без ритма, пока смена не станет "
             "занимать меньше доли.")
    if pat["slots"]:
        L.append(f"2. Бой «{strum.short(pat)}» сначала на одном заглушённом аккорде, под метроном "
                 f"{max(50, int(res['tempo'] * 0.7))} BPM.")
    L.append(f"3. Соедините: квадрат + бой на медленном темпе, затем поднимайте до {res['tempo']:.0f} BPM.")
    L.append("4. Играйте вместе с видео — на сменах частей слушайте, где бой меняется.\n")

    L.append("---\n_Разбор собран автоматически: звук — librosa (хрома + Витерби, онсеты), "
             "видео — Claude (формы аккордов, каподастр) и OpenCV (движение правой руки). "
             "Проверяйте на слух, особенно редкие аккорды._")

    path = out_dir / "README.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
