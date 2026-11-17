"""Random recipe generator, built in three tiers.

  Base       1, always on, full level. What the video *is*: rain, ocean, fan...
  Secondary  0-2, always on, 6-14 dB under the base: wind, a noise bed, binaural tones...
  Accent     0-1, short events minutes apart: crickets, distant thunder, an owl...
             Accents only play in the first 45-60 minutes, then fade out, so nothing
             can startle someone who is already asleep.

Each base lists the secondaries and accents that suit it. Settings are picked at random
within ranges that sound good, so every video is a genuinely different mix (channels that
upload near-identical long videos get flagged by YouTube as "repetitive content").

Some elements need real recordings from the sound library (`--library`, one subfolder
per category). They are only offered when that folder has files in it.

A recipe is plain JSON. Its `meta` block names the tiers, which step 2 uses for titles
and the weekly report uses to see which combinations people stay with longest.
"""

from __future__ import annotations

import glob
import os

import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
BANDS = {"delta": (0.5, 3.5), "theta": (4.0, 7.0)}
SOLFEGGIO = [174, 285, 396, 432, 528, 639]


def _r(rng, lo, hi, nd=2):
    return round(float(rng.uniform(lo, hi)), nd)


def _drift(rng, depth=(1.0, 3.0)):
    return {"depth_db": _r(rng, *depth), "period_s": _r(rng, 60, 240, 0)}


def _pan(rng, mode=None, width=(0.3, 0.6), period=(40, 120)):
    mode = mode or rng.choice(["sweep", "circle", "none"], p=[0.35, 0.35, 0.3])
    return {"mode": str(mode), "width": _r(rng, *width), "period_s": _r(rng, *period, 0)}


def library_files(library, categories) -> list[str]:
    if not library:
        return []
    files = []
    for cat in categories:
        for path in glob.glob(os.path.join(library, cat, "**", "*"), recursive=True):
            if path.lower().endswith(AUDIO_EXTS):
                files.append(path)
    return sorted(files)


def _sample(rng, ctx, categories, level=0.0):
    path = str(rng.choice(library_files(ctx["library"], categories)))
    ctx["sources"].append(os.path.relpath(path, ctx["library"]))
    return {"type": "sample", "level_db": level, "params": {"path": path}, "drift": _drift(rng, (1.0, 2.5))}


# --------------------------------------------------------------------------- bases
# Each returns (layers, label). Layers are at 0 dB.

def base_rain(rng, ctx):
    intensity = _r(rng, 0.2, 0.95)
    ctx["rain_intensity"] = intensity
    layer = {"type": "rain", "level_db": 0.0,
             "params": {"intensity": intensity, "closeness": _r(rng, 0.2, 0.8),
                        "gusts": _r(rng, 0.1, 0.5), "gust_period": _r(rng, 45, 140, 0)},
             "drift": _drift(rng, (0.5, 2.0))}
    label = "Gentle Rain" if intensity < 0.45 else ("Heavy Rain" if intensity > 0.75 else "Rain")
    if rng.random() < 0.25:
        layer["pan"] = _pan(rng, "circle", (0.5, 0.75), (30, 70))
        ctx["features"].append("8D Audio")
    else:
        layer["pan"] = _pan(rng, width=(0.15, 0.4))
    layers = [layer]
    if library_files(ctx["library"], ["rain", "rain_roof", "rain_window"]) and rng.random() < 0.7:
        layers.append(_sample(rng, ctx, ["rain", "rain_roof", "rain_window"], _r(rng, -6, 0, 1)))
    return layers, label


def base_ocean(rng, ctx):
    layers = [{"type": "ocean", "level_db": 0.0,
               "params": {"min_period": _r(rng, 6, 9, 1), "max_period": _r(rng, 10, 15, 1),
                          "brightness": _r(rng, 0.2, 0.7), "calm": _r(rng, 0.2, 0.45)},
               "pan": _pan(rng, width=(0.15, 0.35), period=(90, 200))}]
    if library_files(ctx["library"], ["ocean"]) and rng.random() < 0.7:
        layers.append(_sample(rng, ctx, ["ocean"], _r(rng, -6, 0, 1)))
    return layers, "Ocean Waves"


def base_brown_noise(rng, ctx):
    lp = int(rng.uniform(250, 700))
    return [{"type": "noise", "level_db": 0.0, "params": {"color": "brown", "lowpass_hz": lp},
             "drift": _drift(rng, (0.5, 2.0)), "pan": _pan(rng, width=(0.15, 0.35), period=(120, 300))}], \
        ("Deep Brown Noise" if lp < 450 else "Brown Noise")


def base_pink_noise(rng, ctx):
    return [{"type": "noise", "level_db": 0.0,
             "params": {"color": "pink", "lowpass_hz": int(rng.uniform(1800, 4500))},
             "drift": _drift(rng, (0.5, 1.5))}], "Pink Noise"


def base_fan(rng, ctx):
    return [{"type": "fan", "level_db": 0.0,
             "params": {"size": _r(rng, 0.2, 0.9), "hum": _r(rng, 0.0, 0.06, 3),
                        "flutter": _r(rng, 0.0, 0.05, 3)}}], "Fan Noise"


def base_drone(rng, ctx):
    f = int(rng.choice(SOLFEGGIO))
    ctx["drone_hz"] = f
    return [{"type": "drone", "level_db": 0.0, "params": {"freq_hz": f, "warmth": _r(rng, 0.2, 0.8)},
             "pan": _pan(rng, width=(0.2, 0.5), period=(60, 160)), "drift": _drift(rng)}], f"{f} Hz"


def base_stream(rng, ctx):
    return [_sample(rng, ctx, ["stream"])], "Gentle Stream"


def base_fireplace(rng, ctx):
    return [_sample(rng, ctx, ["fire"])], "Crackling Fireplace"


# ---------------------------------------------------------------------- secondaries
# Each returns (layer, label). `level` is dB under the base.

def sec_wind(rng, ctx, level):
    if library_files(ctx["library"], ["wind"]) and rng.random() < 0.5:
        return _sample(rng, ctx, ["wind"], level), "Distant Wind"
    return {"type": "wind", "level_db": level,
            "params": {"strength": _r(rng, 0.3, 0.8), "whistle": _r(rng, 0.0, 0.4),
                       "gust_period": _r(rng, 14, 35, 0)},
            "pan": _pan(rng, "sweep", (0.2, 0.5), (60, 180))}, "Distant Wind"


def sec_noise_bed(rng, ctx, level):
    color = str(rng.choice(["brown", "pink"], p=[0.75, 0.25]))
    lp = int(rng.uniform(350, 900)) if color == "brown" else int(rng.uniform(1500, 3500))
    return {"type": "noise", "level_db": level, "params": {"color": color, "lowpass_hz": lp},
            "drift": _drift(rng)}, f"{color.title()} Noise"


def sec_binaural(rng, ctx, level):
    band = str(rng.choice(list(BANDS), p=[0.75, 0.25]))
    beat = _r(rng, *BANDS[band], 1)
    ctx["features"].append("Binaural Beats")
    # Tones need to be a little more present than other secondaries to be felt.
    return {"type": "binaural", "level_db": min(level + 2, -8),
            "params": {"carrier_hz": int(rng.uniform(110, 220)), "beat_hz": beat}}, f"{band.title()} Waves {beat:g} Hz"


def sec_drone(rng, ctx, level):
    f = int(rng.choice(SOLFEGGIO))
    return {"type": "drone", "level_db": level, "params": {"freq_hz": f, "warmth": _r(rng, 0.3, 0.8)},
            "pan": _pan(rng, width=(0.2, 0.5), period=(60, 160)), "drift": _drift(rng)}, f"{f} Hz"


def sec_light_rain(rng, ctx, level):
    return {"type": "rain", "level_db": level,
            "params": {"intensity": _r(rng, 0.2, 0.5), "closeness": _r(rng, 0.1, 0.4), "gusts": _r(rng, 0.1, 0.3)},
            "drift": _drift(rng, (0.5, 2.0))}, "Soft Rain"


def sec_fire(rng, ctx, level):
    return _sample(rng, ctx, ["fire"], level), "Crackling Fire"


def sec_stream(rng, ctx, level):
    return _sample(rng, ctx, ["stream"], level), "Stream"


# -------------------------------------------------------------------------- accents
# Each returns (layer, label). They get a `gate` so they come and go, and stop early.

def _gate(rng, on, off, fade=3.0):
    return {"on_s": list(on), "off_s": list(off), "fade_s": fade, "start_s": _r(rng, 20, 90, 0)}


def acc_thunder(rng, ctx, level):
    return {"type": "thunder", "level_db": level + 4,
            "params": {"min_gap": _r(rng, 90, 180, 0), "max_gap": _r(rng, 300, 600, 0),
                       "darkness": _r(rng, 0.3, 0.8)}, "gate": {}}, "Distant Thunder"


def acc_crickets(rng, ctx, level):
    if library_files(ctx["library"], ["crickets"]) and rng.random() < 0.5:
        layer = _sample(rng, ctx, ["crickets"], level)
    else:
        layer = {"type": "crickets", "level_db": level,
                 "params": {"count": int(rng.integers(2, 6)), "distance": _r(rng, 0.4, 0.9)}}
    layer["gate"] = _gate(rng, (20, 60), (30, 120))
    return layer, "Crickets"


def _acc_library(category, label):
    def build(rng, ctx, level):
        layer = _sample(rng, ctx, [category], level)
        layer["gate"] = _gate(rng, (5, 14), (60, 240), fade=1.5)
        return layer, label
    return build


# --------------------------------------------------------------------------- tables

BASES = {
    "rain": dict(build=base_rain, secondaries=["wind", "noise_bed", "binaural", "drone", "fire"],
                 accents=["thunder", "crickets", "owl", "frogs", "chimes"]),
    "ocean": dict(build=base_ocean, secondaries=["wind", "noise_bed", "binaural", "drone"],
                  accents=["crickets", "chimes"]),
    "brown_noise": dict(build=base_brown_noise, secondaries=["binaural", "drone", "light_rain"], accents=[]),
    "pink_noise": dict(build=base_pink_noise, secondaries=["binaural"], accents=[]),
    "fan": dict(build=base_fan, secondaries=["light_rain", "binaural"], accents=[]),
    "drone": dict(build=base_drone, secondaries=["noise_bed", "light_rain", "wind", "stream"], accents=["chimes"]),
    "stream": dict(build=base_stream, secondaries=["wind", "noise_bed", "light_rain"],
                   accents=["crickets", "frogs", "owl"], needs=["stream"]),
    "fireplace": dict(build=base_fireplace, secondaries=["light_rain", "wind"],
                      accents=["crickets", "owl"], needs=["fire"]),
}

SECONDARIES = {
    "wind": dict(build=sec_wind),
    "noise_bed": dict(build=sec_noise_bed),
    "binaural": dict(build=sec_binaural),
    "drone": dict(build=sec_drone),
    "light_rain": dict(build=sec_light_rain),
    "fire": dict(build=sec_fire, needs=["fire"]),
    "stream": dict(build=sec_stream, needs=["stream"]),
}

ACCENTS = {
    "thunder": dict(build=acc_thunder),
    "crickets": dict(build=acc_crickets),
    "owl": dict(build=_acc_library("owl", "Owl"), needs=["owl"]),
    "frogs": dict(build=_acc_library("frogs", "Frogs"), needs=["frogs"]),
    "chimes": dict(build=_acc_library("chimes", "Wind Chimes"), needs=["chimes"]),
}

# Kept for the CLI: a "theme" is now a base.
THEMES = BASES


def _available(table, names, library):
    return [n for n in names if all(library_files(library, [c]) for c in table[n].get("needs", []))]


def available_bases(library=None) -> list[str]:
    return sorted(_available(BASES, list(BASES), library))


# Which secondary reads best in a title when there are two.
_TITLE_ORDER = ["Distant Wind", "Crackling Fire", "Stream", "Soft Rain", "Brown Noise", "Pink Noise"]


def _headline(base_label, sec_labels, acc_labels):
    named = sorted((l for l in sec_labels if l in _TITLE_ORDER), key=_TITLE_ORDER.index)
    extras = named[:1] + acc_labels[:1]
    if not extras:
        return base_label
    return f"{base_label} with {' and '.join(extras)}"


def random_recipe(seed: int | None = None, theme: str | None = None, library: str | None = None,
                  weights: dict[str, float] | None = None, secondaries: list[str] | None = None,
                  accents: list[str] | None = None) -> dict:
    """Make a fresh recipe.

    theme        base to use (random if None)
    secondaries  force these secondaries (otherwise 0-2 picked at random from the base's list)
    accents      force these accents (otherwise 0-1 picked at random)
    weights      favour some bases, e.g. from the weekly report: {"rain": 3, "fan": 0.5}
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    rng = np.random.default_rng(seed)
    bases = available_bases(library)
    if theme is None:
        w = np.array([float((weights or {}).get(n, 1.0)) for n in bases])
        theme = str(rng.choice(bases, p=w / w.sum()))
    if theme not in BASES:
        raise ValueError(f"unknown base {theme!r}; choose from {', '.join(sorted(BASES))}")
    if theme not in bases:
        raise ValueError(f"base {theme!r} needs library recordings in: {', '.join(BASES[theme]['needs'])}")
    spec = BASES[theme]
    ctx = {"library": library, "sources": [], "features": []}

    layers, base_label = spec["build"](rng, ctx)

    # Secondaries
    pool = _available(SECONDARIES, spec["secondaries"], library)
    if secondaries is None:
        k = min(len(pool), int(rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])))
        secondaries = [str(s) for s in rng.choice(pool, size=k, replace=False)] if k else []
    sec_labels = []
    for name in secondaries:
        layer, label = SECONDARIES[name]["build"](rng, ctx, _r(rng, -14, -6, 1))
        layers.append(layer)
        sec_labels.append(label)

    # Accents: thunder only suits proper rain.
    pool = _available(ACCENTS, spec["accents"], library)
    if ctx.get("rain_intensity", 1.0) < 0.5:
        pool = [a for a in pool if a != "thunder"]
    if accents is None:
        accents = [str(rng.choice(pool))] if pool and rng.random() < 0.5 else []
    acc_labels = []
    stop_at = _r(rng, 2700, 3600, 0)  # 45-60 minutes in
    for name in accents:
        layer, label = ACCENTS[name]["build"](rng, ctx, _r(rng, -14, -8, 1))
        layer.setdefault("gate", {})
        layer["gate"].update({"until_s": stop_at, "until_fade_s": 300})
        layers.append(layer)
        acc_labels.append(label)

    meta = {
        "theme": theme,
        "base": theme,
        "base_label": base_label,
        "secondaries": list(secondaries),
        "secondary_labels": sec_labels,
        "accents": list(accents),
        "accent_labels": acc_labels,
        "combo": "+".join([theme, *sorted(secondaries), *sorted(accents)]),
        "headline": _headline(base_label, sec_labels, acc_labels),
        "features": sorted(set(ctx["features"])) + [l for l in sec_labels if "Waves" in l],
        "binaural": any(l["type"] == "binaural" for l in layers),
        "eight_d": any((l.get("pan") or {}).get("mode") == "circle" for l in layers),
        "accents_until_s": stop_at if accents else None,
    }
    if ctx["sources"]:
        meta["sources"] = ctx["sources"]
    master = {"lowpass_hz": int(rng.uniform(7000, 12000)), "fade_in_s": 20, "fade_out_s": 45}
    return {"version": 2, "seed": seed, "master": master, "layers": layers, "meta": meta}
