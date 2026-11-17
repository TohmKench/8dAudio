"""Command line for sleepgen.

  python -m sleepgen themes
  python -m sleepgen recipe --theme gentle_rain --seed 7 -o rain.json
  python -m sleepgen render rain.json -o rain.m4a --hours 10
  python -m sleepgen random -o out/video.m4a --hours 10 [--theme ocean] [--library sounds/]
  python -m sleepgen previews -d previews --seconds 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import render_with_sidecar
from .recipes import THEMES, random_recipe


def _length(args) -> float:
    if args.seconds:
        return float(args.seconds)
    return float(args.hours) * 3600


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sleepgen", description="Long, calm soundscapes for sleep videos.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("themes", help="list themes")

    rp = sub.add_parser("recipe", help="write a random recipe to JSON")
    rp.add_argument("--theme")
    rp.add_argument("--seed", type=int)
    rp.add_argument("--library", help="folder of library sounds, one subfolder per category")
    rp.add_argument("-o", "--out", required=True)

    def add_len(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--hours", type=float, default=10.0)
        g.add_argument("--seconds", type=float)
        sp.add_argument("--bitrate", default="256k")

    rn = sub.add_parser("render", help="render a recipe JSON")
    rn.add_argument("recipe")
    rn.add_argument("-o", "--out", required=True)
    add_len(rn)

    rr = sub.add_parser("random", help="make a random recipe and render it")
    rr.add_argument("--theme")
    rr.add_argument("--seed", type=int)
    rr.add_argument("--library")
    rr.add_argument("-o", "--out", required=True)
    add_len(rr)

    pv = sub.add_parser("previews", help="render a short clip of every theme")
    pv.add_argument("-d", "--dir", default="previews")
    pv.add_argument("--seconds", type=float, default=90)
    pv.add_argument("--seed", type=int, default=1)
    pv.add_argument("--library")

    args = p.parse_args(argv)

    if args.cmd == "themes":
        print("\n".join(sorted(THEMES)))
        return 0

    if args.cmd == "recipe":
        recipe = random_recipe(args.seed, args.theme, args.library)
        with open(args.out, "w") as f:
            json.dump(recipe, f, indent=2)
        print(f"{recipe['meta']['theme']} (seed {recipe['seed']}) -> {args.out}")
        return 0

    if args.cmd in ("render", "random"):
        if args.cmd == "render":
            with open(args.recipe) as f:
                recipe = json.load(f)
        else:
            recipe = random_recipe(args.seed, args.theme, args.library)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        print(f"Rendering {recipe['meta']['theme']} (seed {recipe['seed']}), {_length(args) / 3600:.2f} h",
              file=sys.stderr)
        report = render_with_sidecar(recipe, args.out, _length(args), bitrate=args.bitrate)
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "previews":
        os.makedirs(args.dir, exist_ok=True)
        for name in sorted(THEMES):
            recipe = random_recipe(args.seed, name, args.library)
            recipe["master"]["fade_in_s"] = 3
            recipe["master"]["fade_out_s"] = 3
            out = os.path.join(args.dir, f"{name}.mp3")
            report = render_with_sidecar(recipe, out, args.seconds, bitrate="192k", progress=False)
            print(f"{name:15s} {report['render_seconds']:5.1f}s  peak {report['peak_dbfs']} dBFS  -> {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
