"""Command line for sleepgen.

  python -m sleepgen catalog                                   (bases, modifiers, secondaries, tertiaries, scenes)
  python -m sleepgen recipe --base rain --seed 7 -o rain.json
  python -m sleepgen render rain.json -o rain.m4a --hours 10
  python -m sleepgen random -o out/video.m4a --hours 10 [--base ocean] [--library sounds/]
  python -m sleepgen random --base rain --modifier window --strength gentle --secondary wind \
         --tertiary crickets -o out/v.m4a
  python -m sleepgen random --scene rainy_study_window -o out/study.m4a --hours 8
  python -m sleepgen previews -d previews --seconds 90
  python -m sleepgen previews --scenes -d previews             (one clip of every scene that is built)
  python -m sleepgen previews --rain --rate 4 -d previews      (rain in every environment)
  python -m sleepgen previews --fire -d previews               (fire in every setting)
  python -m sleepgen bench                                     (GPU speed check, synthesis only)
  python -m sleepgen bench --full flac --seconds 300           (whole pipeline incl. encoding)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import render_with_sidecar
from .recipes import (BASES, SCENES, SECONDARIES, TERTIARIES, available_bases, random_recipe, scene_recipe,
                      scene_status, strengths_for)


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

    sub.add_parser("catalog", aliases=["themes"], help="list bases, modifiers, secondaries, tertiaries and scenes")

    def add_tiers(sp):
        sp.add_argument("--base", "--theme", dest="base", help="base sound (random if omitted)")
        sp.add_argument("--modifier", help="the base's place or kind, e.g. window, campfire, brown")
        sp.add_argument("--strength", help="e.g. gentle, steady, heavy (see `catalog`)")
        sp.add_argument("--secondary", action="append", choices=sorted(SECONDARIES),
                        help="force a secondary (repeatable); default: 0-2 at random")
        sp.add_argument("--no-secondary", action="store_true")
        sp.add_argument("--tertiary", action="append", choices=sorted(TERTIARIES),
                        help="force a tertiary (repeatable); default: 0-2 at random")
        sp.add_argument("--no-tertiary", action="store_true")
        sp.add_argument("--study", action="store_true",
                        help="a study video: tertiaries keep going all video instead of stopping after ~1 h")
        sp.add_argument("--scene", choices=sorted(SCENES), help="a scene preset (fixes all four tiers)")
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
    pv.add_argument("--fire", action="store_true", help="one clip of fire in every setting instead")
    pv.add_argument("--intensity", type=float, help="with --fire: 0.5 gentle .. 1.3 roaring (default: random)")
    pv.add_argument("--scenes", action="store_true", help="one clip of every scene that is built instead")

    bn = sub.add_parser("bench", help="time rain/fire rendering on this machine and project a 10 hour render")
    bn.add_argument("--seconds", type=float, default=20)
    bn.add_argument("--environment", action="append", help="rain environments or fire settings (default: all)")
    bn.add_argument("--full", choices=["flac", "m4a"],
                    help="time the whole pipeline (loudness, master chain, encoding) to a temporary file")

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
        import tempfile
        import time
        import numpy as np
        from . import gpu
        from .dsp import SR
        from .fire import FIRE_ENVIRONMENTS, FireScene
        from .scene import ENVIRONMENTS, RainScene
        print(f"device: {gpu.describe()}   drops/s budget: {gpu.default_budget()}"
              + (f"   whole pipeline to .{args.full}" if args.full else "   synthesis only"))
        worst = 0.0
        for env in args.environment or list(ENVIRONMENTS) + list(FIRE_ENVIRONMENTS):
            fire = env in FIRE_ENVIRONMENTS
            if args.full:
                recipe = random_recipe(1, "fire" if fire else "rain", None, secondaries=[], tertiaries=[],
                                       modifier=env)
                if not fire:
                    recipe["layers"][0]["params"]["rain_mm_h"] = 6.0
                with tempfile.TemporaryDirectory() as tmp:
                    from .engine import render
                    rep = render(recipe, os.path.join(tmp, f"bench.{args.full}"), args.seconds, progress=False)
                    # The loudness check (120 s, once per video) is left out: it is 0.3% of 10 h.
                    spent, t_init = rep["render_seconds"] - rep["loudness_check_seconds"], 0.0
            else:
                t = time.time()
                s = FireScene(np.random.default_rng(1), environment=env) if fire else \
                    RainScene(np.random.default_rng(1), environment=env, rain_mm_h=6.0)
                s(SR * 10)  # warm up (first-use GPU set-up is a one-off per render, not per hour)
                t_init = time.time() - t
                t = time.time()
                for _ in range(int(args.seconds // 10)):
                    s(SR * 10)
                spent = time.time() - t
            secs = args.seconds if args.full else args.seconds // 10 * 10
            per_hour = spent / secs * 3600
            worst = max(worst, per_hour)
            print(f"{env:10s} {secs / spent:6.1f}x realtime   10 h would take {per_hour * 10 / 60:6.1f} min"
                  + ("" if args.full else f"   (setup {t_init:.1f}s)"), flush=True)
        print(f"slowest: 10 h in about {worst * 10 / 60:.0f} min")
        return 0

    if args.cmd == "match":
        from .match import run
        report = run(args.reference, args.base, args.out, args.evals)
        print(json.dumps({k: report[k] for k in ("similarity_settings_only", "similarity_with_eq")}, indent=2))
        return 0

    def make(a):
        if a.scene:
            return scene_recipe(a.scene, a.seed, a.library)
        sec = [] if a.no_secondary else a.secondary
        ter = [] if a.no_tertiary else a.tertiary
        return random_recipe(a.seed, a.base, a.library, secondaries=sec, tertiaries=ter, modifier=a.modifier,
                             strength=a.strength, video_type="study" if a.study else "sleep")

    if args.cmd in ("catalog", "themes"):
        print("BASES  (modifiers: strengths)")
        for name, spec in sorted(BASES.items()):
            needs = f"  (needs library: {', '.join(spec['needs'])})" if spec.get("needs") else ""
            mods = ", ".join(f"{m}: {'/'.join(strengths_for(name, m))}" for m in spec["modifiers"])
            print(f"  {name}{needs}\n    modifiers:   {mods}"
                  f"\n    secondaries: {', '.join(spec['secondaries']) or '-'}"
                  f"\n    tertiaries:  {', '.join(spec['tertiaries']) or '-'}")
        print("SECONDARIES\n  " + ", ".join(sorted(SECONDARIES)))
        print("TERTIARIES\n  " + ", ".join(f"{n} ({'/'.join(sorted(t['types']))})"
                                          for n, t in sorted(TERTIARIES.items())))
        print("SCENES")
        for name, s in SCENES.items():
            missing = scene_status(name, getattr(args, "library", None))
            state = "ready" if not missing else "needs " + ", ".join(missing)
            print(f"  {name:20s} {s['type']:5s}  {s['title']}  [{state}]")
        return 0

    if args.cmd == "recipe":
        recipe = make(args)
        with open(args.out, "w") as f:
            json.dump(recipe, f, indent=2)
        print(f"{recipe['meta']['combo']} (seed {recipe['seed']}) -> {args.out}")
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
            recipe = random_recipe(args.seed, "rain", args.library, secondaries=[], tertiaries=[], modifier=env)
            if args.rate:
                recipe["layers"][0]["params"]["rain_mm_h"] = args.rate
            recipe["meta"]["headline"] = ENVIRONMENTS[env]["title"]
            recipe["master"].update(fade_in_s=3, fade_out_s=3)
            out = os.path.join(args.dir, f"rain-{env}.mp3")
            report = render_with_sidecar(recipe, out, args.seconds, bitrate="256k", progress=False)
            print(f"{env:9s} {report['render_seconds']:6.1f}s  -> {out}")
        return 0

    if args.cmd == "previews" and args.fire:
        from .fire import FIRE_ENVIRONMENTS
        os.makedirs(args.dir, exist_ok=True)
        for env in FIRE_ENVIRONMENTS:
            recipe = random_recipe(args.seed, "fire", args.library, secondaries=[], tertiaries=[], modifier=env)
            if args.intensity:
                recipe["layers"][0]["params"]["intensity"] = args.intensity
            recipe["meta"]["headline"] = recipe["meta"]["base_title"] = FIRE_ENVIRONMENTS[env]["title"]
            recipe["master"].update(fade_in_s=3, fade_out_s=3)
            out = os.path.join(args.dir, f"fire-{env}.mp3")
            report = render_with_sidecar(recipe, out, args.seconds, bitrate="256k", progress=False)
            print(f"{env:10s} {report['render_seconds']:6.1f}s  peak {report['peak_dbfs']} dBFS  -> {out}")
        return 0

    if args.cmd == "previews" and args.scenes:
        from .recipes import available_scenes
        os.makedirs(args.dir, exist_ok=True)
        for name in available_scenes(args.library):
            recipe = scene_recipe(name, args.seed, args.library)
            recipe["master"].update(fade_in_s=3, fade_out_s=3)
            out = os.path.join(args.dir, f"scene-{name}.mp3")
            report = render_with_sidecar(recipe, out, args.seconds, bitrate="256k", progress=False)
            print(f"{name:20s} {report['render_seconds']:6.1f}s  peak {report['peak_dbfs']} dBFS  -> {out}")
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
