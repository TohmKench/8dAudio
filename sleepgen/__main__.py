"""Command line for sleepgen.

  python -m sleepgen themes
  python -m sleepgen recipe --theme rain --seed 7 -o rain.json
  python -m sleepgen render rain.json -o rain.m4a --hours 10
  python -m sleepgen random -o out/video.m4a --hours 10 [--theme ocean] [--library sounds/]
  python -m sleepgen random --theme rain --secondary wind --accent crickets -o out/v.m4a
  python -m sleepgen previews -d previews --seconds 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import render_with_sidecar
from .recipes import ACCENTS, BASES, SECONDARIES, available_bases, random_recipe


def _length(args) -> float:
    if args.seconds:
        return float(args.seconds)
    return float(args.hours) * 3600


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sleepgen", description="Long, calm soundscapes for sleep videos.")
    p.add_argument("--device", help="auto (default), gpu, cuda, rocm, directml or cpu")
    p.add_argument("--quality", type=int,
                   help="individually synthesised raindrops per second (default 16000 on GPU, 2500 on CPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("themes", help="list bases, secondaries and accents")

    def add_tiers(sp):
        sp.add_argument("--theme", help="base sound (random if omitted)")
        sp.add_argument("--secondary", action="append", choices=sorted(SECONDARIES),
                        help="force a secondary (repeatable); default: 0-2 at random")
        sp.add_argument("--no-secondary", action="store_true")
        sp.add_argument("--accent", action="append", choices=sorted(ACCENTS), help="force an accent")
        sp.add_argument("--no-accent", action="store_true")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--library", help="folder of library sounds, one subfolder per category")

    rp = sub.add_parser("recipe", help="write a random recipe to JSON")
    add_tiers(rp)
    rp.add_argument("-o", "--out", required=True)

    def add_len(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--hours", type=float, default=10.0)
        g.add_argument("--seconds", type=float)
        sp.add_argument("--bitrate", default="384k")

    rn = sub.add_parser("render", help="render a recipe JSON")
    rn.add_argument("recipe")
    rn.add_argument("-o", "--out", required=True)
    add_len(rn)

    rr = sub.add_parser("random", help="make a random recipe and render it")
    add_tiers(rr)
    rr.add_argument("-o", "--out", required=True)
    add_len(rr)

    pv = sub.add_parser("previews", help="render a short clip of every theme")
    pv.add_argument("-d", "--dir", default="previews")
    pv.add_argument("--seconds", type=float, default=90)
    pv.add_argument("--seed", type=int, default=1)
    pv.add_argument("--library")
    pv.add_argument("--rain", action="store_true", help="one clip of rain in every environment instead")
    pv.add_argument("--rate", type=float, help="with --rain: rainfall in mm/h (default: random per seed)")

    bn = sub.add_parser("bench", help="time rain rendering on this machine and project a 10 hour render")
    bn.add_argument("--seconds", type=float, default=20)
    bn.add_argument("--environment", action="append")

    mt = sub.add_parser("match", help="tune a base sound to match a reference recording")
    mt.add_argument("reference")
    mt.add_argument("--base", default="rain", choices=["rain", "ocean"])
    mt.add_argument("-o", "--out", default="matched")
    mt.add_argument("--evals", type=int, default=70)

    args = p.parse_args(argv)
    if args.quality:
        os.environ["SLEEPGEN_QUALITY"] = str(args.quality)
    if args.device:
        from . import gpu
        gpu.set_device(args.device)

    if args.cmd == "bench":
        import time
        import numpy as np
        from . import gpu
        from .dsp import SR
        from .scene import ENVIRONMENTS, RainScene
        print(f"device: {gpu.describe()}   drops/s budget: {gpu.default_budget()}")
        worst = 0.0
        for env in args.environment or list(ENVIRONMENTS):
            t = time.time()
            s = RainScene(np.random.default_rng(1), environment=env, rain_mm_h=6.0)
            t_init = time.time() - t
            t = time.time()
            for _ in range(int(args.seconds // 10)):
                s(SR * 10)
            spent = time.time() - t
            per_hour = spent / (args.seconds // 10 * 10) * 3600
            worst = max(worst, per_hour)
            print(f"{env:9s} {args.seconds / spent:6.1f}x realtime   10 h would take {per_hour * 10 / 60:6.1f} min"
                  f"   (setup {t_init:.1f}s)")
        print(f"slowest environment: 10 h in about {worst * 10 / 60:.0f} min")
        return 0

    if args.cmd == "match":
        from .match import run
        report = run(args.reference, args.base, args.out, args.evals)
        print(json.dumps({k: report[k] for k in ("similarity_settings_only", "similarity_with_eq")}, indent=2))
        return 0

    def make(a):
        sec = [] if a.no_secondary else a.secondary
        acc = [] if a.no_accent else a.accent
        return random_recipe(a.seed, a.theme, a.library, secondaries=sec, accents=acc)

    if args.cmd == "themes":
        for name, spec in sorted(BASES.items()):
            needs = f"  (needs library: {', '.join(spec['needs'])})" if spec.get("needs") else ""
            print(f"{name}{needs}\n  secondaries: {', '.join(spec['secondaries']) or '-'}"
                  f"\n  accents:     {', '.join(spec['accents']) or '-'}")
        return 0

    if args.cmd == "recipe":
        recipe = make(args)
        with open(args.out, "w") as f:
            json.dump(recipe, f, indent=2)
        print(f"{recipe['meta']['theme']} (seed {recipe['seed']}) -> {args.out}")
        return 0

    if args.cmd in ("render", "random"):
        if args.cmd == "render":
            with open(args.recipe) as f:
                recipe = json.load(f)
        else:
            recipe = make(args)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        print(f"Rendering {recipe['meta']['headline']} (seed {recipe['seed']}), {_length(args) / 3600:.2f} h",
              file=sys.stderr)
        report = render_with_sidecar(recipe, args.out, _length(args), bitrate=args.bitrate)
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "previews" and args.rain:
        from .scene import ENVIRONMENTS
        os.makedirs(args.dir, exist_ok=True)
        for env in ENVIRONMENTS:
            recipe = random_recipe(args.seed, "rain", args.library, secondaries=[], accents=[])
            recipe["layers"][0]["params"]["environment"] = env
            if args.rate:
                recipe["layers"][0]["params"]["rain_mm_h"] = args.rate
            recipe["meta"]["headline"] = ENVIRONMENTS[env]["title"]
            recipe["master"].update(fade_in_s=3, fade_out_s=3)
            out = os.path.join(args.dir, f"rain-{env}.mp3")
            report = render_with_sidecar(recipe, out, args.seconds, bitrate="256k", progress=False)
            print(f"{env:9s} {report['render_seconds']:6.1f}s  -> {out}")
        return 0

    if args.cmd == "previews":
        os.makedirs(args.dir, exist_ok=True)
        for name in available_bases(args.library):
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
