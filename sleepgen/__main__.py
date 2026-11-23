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
  python -m sleepgen refs fetch page_turns --query "page turn"  (CC0 reference recordings, Freesound)
  python -m sleepgen tune page_turns references/page_turns     (search until it sounds like the real ones)
  python -m sleepgen listen runs/page_turns                    (A/B page: pick, never describe)
  python -m sleepgen tune page_turns references/page_turns --resume   (learn from the picks, go again)
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


def _tune(args) -> int:
    """`tune`: score against the references, search, and write the listening set. With --resume,
    first learn from the listening page's picks (new weights, start from the preferred version)."""
    import shutil

    from . import eventmatch as em
    from .listen import load_picks
    out = args.out or os.path.join("runs", args.target)
    print(f"measuring the references in {args.references} ...", flush=True)
    ref = em.Reference.from_folder(args.references)
    print(f"  {len(ref.events)} events; room tone {ref.snr_db:.0f} dB under them", flush=True)
    weights, start, round_no = None, None, 1
    if args.resume:
        best_path = os.path.join(out, "best.json")
        trials_path = os.path.join(out, "listen", "trials.json")
        picks = load_picks(out)
        if not (os.path.isfile(best_path) and os.path.isfile(trials_path) and picks):
            print("nothing to resume from: run `tune` and then `listen` first", file=sys.stderr)
            return 2
        with open(best_path) as fh:
            best = json.load(fh)
        with open(trials_path) as fh:
            trials = json.load(fh)
        v = em.verdict(picks)
        weights = em.reweight(best.get("weights", {}), picks, trials["devs"])
        prefer = max((k for k in trials["params"] if k != "real"), key=lambda k: v["rates"].get(k, 0.0))
        start = trials["params"][prefer]
        round_no = int(trials.get("round", 1)) + 1
        archive = os.path.join(out, f"listen-{round_no - 1}")
        shutil.rmtree(archive, ignore_errors=True)
        shutil.copytree(os.path.join(out, "listen"), archive)
        os.remove(os.path.join(out, "listen", "picks.json"))
        print(f"from your picks: real preferred {', '.join(f'{k} {r:.0%}' for k, r in v['real_beats'].items())}; "
              f"starting from {prefer}; weights " + " ".join(f"{g} {w:.2f}" for g, w in weights.items()))
        if v["done"]:
            print("(the real one no longer stood out: this round is a polish)")
    listener = None if args.no_listener else em.try_listener(args.target)
    if listener is not None:
        ref.listener_good = listener.judge(ref.events)["good"]
        print(f"  automatic listener hears the target in the real ones {ref.listener_good:.0%} of the time")
    top = em.tune(args.target, ref, out, rounds=args.rounds, population=args.population, n_events=args.events,
                  start=start, weights=weights, listener=listener)
    em.make_listening(out, args.target, top, ref, round_no=round_no)
    print(f"\nbest: score {top[0][1]['total']:.3f}, tell-apart {top[0][1]['tell_apart']:.2f} "
          f"(0.5 = a classifier can't tell). Settings in {os.path.join(out, 'best.json')}")
    print(f"now listen: python -m sleepgen listen {out}")
    return 0


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

    ds = sub.add_parser("describe", help="print the YouTube title, description and tags for a recipe "
                                         "(refuses a recipe whose image owner isn't credited)")
    ds.add_argument("recipe", help="recipe JSON, or a render's .json sidecar")
    ds.add_argument("--hours", type=float, default=10.0)

    mt = sub.add_parser("match", help="tune a base sound to match a reference recording")
    mt.add_argument("reference")
    mt.add_argument("--base", default="rain", choices=["rain", "ocean"])
    mt.add_argument("-o", "--out", default="matched")
    mt.add_argument("--evals", type=int, default=70)

    rf = sub.add_parser("refs", help="reference recordings for `tune`: fetch CC0 ones, or check a folder")
    rf.add_argument("action", choices=["fetch", "check"])
    rf.add_argument("target", help="fetch: what they are for (e.g. page_turns); check: the folder")
    rf.add_argument("--query", help='Freesound search, e.g. "page turn"')
    rf.add_argument("-n", type=int, default=10, help="how many recordings to fetch")
    rf.add_argument("-d", "--dir", help="where to put them (default references/<target>)")
    rf.add_argument("--key", help="Freesound API key (default: FREESOUND_API_KEY or .env)")

    tu = sub.add_parser("tune", help="tune a generated sound until it can't be told from real recordings")
    tu.add_argument("target", choices=["page_turns"])
    tu.add_argument("references", help="folder of real recordings of the sound")
    tu.add_argument("-o", "--out", help="run folder (default runs/<target>)")
    tu.add_argument("--rounds", type=int, default=30)
    tu.add_argument("--population", type=int, default=24)
    tu.add_argument("--events", type=int, default=10, help="generated events per variation")
    tu.add_argument("--no-listener", action="store_true", help="don't use the audio recognition model")
    tu.add_argument("--resume", action="store_true",
                    help="learn from the picks on the listening page and run another round from there")

    ls = sub.add_parser("listen", help="the A/B listening page for a `tune` run (on this computer only)")
    ls.add_argument("run", help="run folder from `tune`")
    ls.add_argument("--port", type=int, default=8765)
    ls.add_argument("--no-browser", action="store_true")

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

    if args.cmd == "describe":
        from .describe import MissingCredit, video_metadata
        with open(args.recipe) as f:
            data = json.load(f)
        try:
            meta = video_metadata(data.get("recipe", data), args.hours)
        except MissingCredit as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "match":
        from .match import run
        report = run(args.reference, args.base, args.out, args.evals)
        print(json.dumps({k: report[k] for k in ("similarity_settings_only", "similarity_with_eq")}, indent=2))
        return 0

    if args.cmd == "refs":
        from . import refs
        if args.action == "fetch":
            if not args.query:
                p.error("refs fetch needs --query")
            out = args.dir or os.path.join("references", args.target)
            got = refs.fetch(args.query, out, args.n, args.key)
            print(f"{len(got)} CC0 recordings in {out} (credits in sources.json)")
        else:
            refs.check(args.target)
        return 0

    if args.cmd == "tune":
        return _tune(args)

    if args.cmd == "listen":
        from .listen import serve
        picks = serve(args.run, args.port, not args.no_browser)
        print(f"{len(picks)} picks saved in {os.path.join(args.run, 'listen', 'picks.json')}")
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
