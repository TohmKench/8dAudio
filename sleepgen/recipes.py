"""Recipes, built in four tiers.

  Base           1, always on, full level. What the video *is*: rain, fire, ocean, noise...
  Base modifier  Where and how the base sounds: its place or kind (on a window, in a forest, a
                 fireplace, brown noise) and its strength (gentle, steady, heavy).
  Secondary      0-2, always on, 6-14 dB under the base: a candle, wind, soft rain, binaural tones...
  Tertiary       0-2, occasional events: page turns, pencil writing, crickets, distant thunder...

Every video is either for **sleep** or for **study**. In sleep videos the tertiaries play only in
the first 45-60 minutes and then fade out, so nothing can startle someone who is already asleep.
Study videos are played by people who are awake, so their tertiaries carry on all video (still
quiet and never sudden).

A **scene** (SCENES) fixes all four tiers and gives the video its own title, e.g. "Studying by a
Rainy Window" = rain · on a window, gentle · with a candle · with page turns and pencil writing.
Random recipes pick each tier within what suits the base and modifier (no fireplace in a tent),
with settings drawn from ranges that sound good, so every video is a different mix (YouTube flags
near-identical long uploads as "repetitive content").

Some parts need real recordings from the sound library (`--library`, one subfolder per category).
They are only offered when that folder has files in it.

A recipe is plain JSON. Its `meta` block names every tier; titles are built from it, and the
weekly report groups videos by `meta.combo` (base/modifier + secondaries + tertiaries).
"""

from __future__ import annotations

import glob
import os

import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
BANDS = {"delta": (0.5, 3.5), "theta": (4.0, 7.0)}
SOLFEGGIO = [174, 285, 396, 432, 528, 639]
VIDEO_TYPES = ("sleep", "study")

# Do tertiaries keep going all video? Sleep: no, they fade out 45-60 min in. Study: yes, because
# people are awake (suggested to Thomas in the catalog doc, 2026-11-20; flip here if he disagrees).
TERTIARIES_ALL_VIDEO = {"sleep": False, "study": True}


def _r(rng, lo, hi, nd=2):
    return round(float(rng.uniform(lo, hi)), nd)


def _pick(rng, weights: dict):
    names = list(weights)
    w = np.array([float(weights[n]) for n in names])
    return str(rng.choice(names, p=w / w.sum()))


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
# Each takes the chosen modifier and strength and returns (layers, title). Layers are at 0 dB.
# The title names base and modifier together, e.g. "Gentle Rain on the Window".

RAIN_RATES = {"gentle": (1.0, 2.5), "steady": (2.5, 7.0), "heavy": (7.0, 16.0)}  # mm/h
RAIN_WORDS = {"gentle": "Gentle ", "steady": "", "heavy": "Heavy "}


def base_rain(rng, ctx, modifier, strength):
    """Physically modelled rain falling on a real place (scene.py)."""
    from .scene import ENVIRONMENTS

    rate = _r(rng, *RAIN_RATES[strength], 1)
    ctx["rain_intensity"] = min(1.0, rate / 8.0)
    ctx["features"].append("3D Audio")
    layer = {"type": "rainscene", "level_db": 0.0,
             "params": {"environment": modifier, "rain_mm_h": rate, "variability": _r(rng, 0.2, 0.4)},
             "send": _r(rng, 0.0, 0.08)}  # the scene has its own room/space
    layers = [layer]
    cats = {"roof": ["rain_roof"], "window": ["rain_window"], "lake": ["rain"], "forest": ["rain"]}.get(modifier, [])
    if library_files(ctx["library"], cats) and rng.random() < 0.5:
        layers.append(_sample(rng, ctx, cats, _r(rng, -12, -6, 1)))
    return layers, RAIN_WORDS[strength] + ENVIRONMENTS[modifier]["title"]


FIRE_INTENSITY = {"gentle": (0.5, 0.7), "steady": (0.7, 1.05), "roaring": (1.05, 1.35)}
FIRE_WORDS = {"gentle": "Gentle", "steady": "", "roaring": "Roaring"}
FIRE_WOODS = {"softwood": 0.3, "mixed": 0.4, "hardwood": 0.3}


def base_fire(rng, ctx, modifier, strength):
    """A physically modelled wood fire in a setting (fire.py). The wood is part of the modifier."""
    from .fire import FIRE_ENVIRONMENTS

    spec = FIRE_ENVIRONMENTS[modifier]
    wood = _pick(rng, FIRE_WOODS)
    damp = _r(rng, 0.0, 0.5)
    if "wood" in spec:  # a candle always burns its wick
        wood, damp = spec["wood"], round(damp * 0.3, 2)
    ctx["detail"]["wood"] = wood
    ctx["features"].append("3D Audio")
    layer = {"type": "firescene", "level_db": 0.0, "send": _r(rng, 0.0, 0.06),
             "params": {"environment": modifier, "intensity": _r(rng, *FIRE_INTENSITY[strength]),
                        "wood": wood, "dampness": damp, "variability": _r(rng, 0.15, 0.3)}}
    layers = [layer]
    if library_files(ctx["library"], ["fire"]) and rng.random() < 0.4 and "wood" not in spec:
        layers.append(_sample(rng, ctx, ["fire"], _r(rng, -12, -6, 1)))
    word = FIRE_WORDS[strength]
    return layers, f"{word} {spec['name']}" if word and "wood" not in spec else spec["title"]


def base_ocean(rng, ctx, modifier, strength):
    calm = {"calm": (0.5, 0.6), "rolling": (0.4, 0.5)}[strength]
    layers = [{"type": "ocean", "level_db": 0.0,
               "params": {"min_period": _r(rng, 6, 9, 1), "max_period": _r(rng, 10, 15, 1),
                          "brightness": _r(rng, 0.2, 0.6), "calm": _r(rng, *calm)},
               "pan": _pan(rng, width=(0.15, 0.35), period=(90, 200))}]
    if library_files(ctx["library"], ["ocean"]) and rng.random() < 0.7:
        layers.append(_sample(rng, ctx, ["ocean"], _r(rng, -6, 0, 1)))
    return layers, "Calm Ocean Waves" if strength == "calm" else "Ocean Waves"


NOISE_LOWPASS = {("brown", "deep"): (250, 450), ("brown", "soft"): (450, 700),
                 ("pink", "soft"): (1800, 3000), ("pink", "bright"): (3000, 4500)}
NOISE_TITLES = {("brown", "deep"): "Deep Brown Noise", ("brown", "soft"): "Brown Noise",
                ("pink", "soft"): "Soft Pink Noise", ("pink", "bright"): "Pink Noise"}


def base_noise(rng, ctx, modifier, strength):
    lp = int(rng.uniform(*NOISE_LOWPASS[modifier, strength]))
    layer = {"type": "noise", "level_db": 0.0, "params": {"color": modifier, "lowpass_hz": lp},
             "drift": _drift(rng, (0.5, 2.0) if modifier == "brown" else (0.5, 1.5))}
    if modifier == "brown":
        layer["pan"] = _pan(rng, width=(0.15, 0.35), period=(120, 300))
    return [layer], NOISE_TITLES[modifier, strength]


def base_fan(rng, ctx, modifier, strength):
    size = {"low": (0.55, 0.9), "high": (0.2, 0.55)}[strength]  # bigger = slower, darker air
    return [{"type": "fan", "level_db": 0.0,
             "params": {"size": _r(rng, *size), "hum": _r(rng, 0.0, 0.06, 3),
                        "flutter": _r(rng, 0.0, 0.05, 3)}}], "Soft Fan Noise" if strength == "low" else "Fan Noise"


def base_drone(rng, ctx, modifier, strength):
    f = int(modifier.removesuffix("hz"))
    warmth = {"warm": (0.5, 0.8), "pure": (0.2, 0.5)}[strength]
    ctx["drone_hz"] = f
    return [{"type": "drone", "level_db": 0.0, "params": {"freq_hz": f, "warmth": _r(rng, *warmth)},
             "pan": _pan(rng, width=(0.2, 0.5), period=(60, 160)), "drift": _drift(rng)}], f"{f} Hz"


def base_stream(rng, ctx, modifier, strength):
    return [_sample(rng, ctx, ["stream"])], "Gentle Stream"


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
    env = str(rng.choice(["window", "roof"]))
    return {"type": "rainscene", "level_db": level,
            "params": {"environment": env, "rain_mm_h": _r(rng, 1.0, 3.0, 1), "variability": 0.25}}, "Soft Rain"


def sec_fire(rng, ctx, level):
    """A fire in the room while it rains outside."""
    if library_files(ctx["library"], ["fire"]) and rng.random() < 0.3:
        return _sample(rng, ctx, ["fire"], level), "Crackling Fire"
    return {"type": "firescene", "level_db": level,
            "params": {"environment": str(rng.choice(["fireplace", "wood_stove"])),
                       "intensity": _r(rng, 0.55, 0.9), "wood": str(rng.choice(["mixed", "hardwood"])),
                       "dampness": _r(rng, 0.0, 0.2), "variability": 0.2}}, "Crackling Fire"


def sec_stream(rng, ctx, level):
    return _sample(rng, ctx, ["stream"], level), "Stream"


def sec_candle(rng, ctx, level):
    """A wood-wick candle on the desk, softly crackling (the fire engine's candle setting)."""
    # Its own level range: a few dB under the base, so the soft crackles come through.
    return {"type": "firescene", "level_db": _r(rng, -9, -5, 1), "send": _r(rng, 0.05, 0.15),  # close by
            "params": {"environment": "candle", "intensity": _r(rng, 0.7, 0.9), "wood": "wick",
                       "dampness": _r(rng, 0.0, 0.15), "variability": _r(rng, 0.15, 0.25)}}, "Candlelight"


# -------------------------------------------------------------------------- tertiaries
# Each returns (layer, label). Night sounds get a `gate` so they come and go; in sleep videos
# every tertiary also stops 45-60 minutes in (added by `_build`).

def _gate(rng, on, off, fade=3.0):
    return {"on_s": list(on), "off_s": list(off), "fade_s": fade, "start_s": _r(rng, 20, 90, 0)}


def ter_thunder(rng, ctx, level):
    return {"type": "thunder", "level_db": level + 4,
            "params": {"min_gap": _r(rng, 90, 180, 0), "max_gap": _r(rng, 300, 600, 0),
                       "darkness": _r(rng, 0.3, 0.8)}, "gate": {}}, "Distant Thunder"


def ter_crickets(rng, ctx, level):
    if library_files(ctx["library"], ["crickets"]) and rng.random() < 0.5:
        layer = _sample(rng, ctx, ["crickets"], level)
    else:
        layer = {"type": "crickets", "level_db": level,
                 "params": {"count": int(rng.integers(2, 6)), "distance": _r(rng, 0.6, 0.95)}}
    layer["gate"] = _gate(rng, (20, 60), (30, 120))
    return layer, "Crickets"


def ter_page_turns(rng, ctx, level):
    """Someone reading at the desk (desk.py). Close by: little reverb, all direct sound, and
    louder than far-off night sounds (they are near you, and paper hides in the rain's hiss)."""
    return {"type": "page_turns", "level_db": _r(rng, -6, -3, 1), "send": _r(rng, 0.12, 0.25), "dry": 1.0,
            "params": {"paper": _pick(rng, {"book": 0.6, "notebook": 0.4}), "mean_gap": _r(rng, 30, 75, 0),
                       "flurry": _r(rng, 0.05, 0.15)}, "gate": {}}, "Page Turning"


def ter_pencil(rng, ctx, level):
    """Someone writing in pencil at the desk, in spells (desk.py)."""
    return {"type": "pencil", "level_db": _r(rng, -7, -4, 1), "send": _r(rng, 0.1, 0.2), "dry": 1.0,
            "params": {"activity": _r(rng, 0.2, 0.4), "surface": _pick(rng, {"pad": 0.6, "desk": 0.4}),
                       "pressure": _r(rng, 0.55, 0.9)}, "gate": {}}, "Pencil Writing"


def _ter_library(category, label):
    def build(rng, ctx, level):
        layer = _sample(rng, ctx, [category], level)
        layer["gate"] = _gate(rng, (5, 14), (60, 240), fade=1.5)
        return layer, label
    return build


# --------------------------------------------------------------------------- tables
# A base lists its modifiers (place or kind -> how often a random recipe picks it), its strengths,
# and the secondaries and tertiaries that suit it. `indoor` names the modifiers that are inside a
# room (None: the base has no place, like noise). Indoor-only parts (a fire in the hearth, a candle,
# a desk) are only offered there; `outdoor_excludes` / `indoor_excludes` drop parts that don't fit.

BASES = {
    "rain": dict(build=base_rain, label="Rain",
                 modifiers={"roof": 1.0, "window": 1.0, "tin_roof": 0.8, "forest": 1.0, "lake": 0.8, "tent": 0.8},
                 indoor={"roof", "window", "tin_roof"}, outdoor_excludes={"fire"},
                 strengths={"gentle": 0.35, "steady": 0.4, "heavy": 0.25},
                 secondaries=["wind", "noise_bed", "binaural", "drone", "fire", "candle"],
                 tertiaries=["thunder", "crickets", "owl", "frogs", "chimes", "page_turns", "pencil"]),
    "fire": dict(build=base_fire, label="Fire",
                 modifiers={"fireplace": 1.2, "campfire": 1.0, "wood_stove": 0.8, "bonfire": 0.6, "candle": 0.5},
                 indoor={"fireplace", "wood_stove", "candle"},
                 outdoor_excludes={"light_rain"},  # nobody sits by a campfire in the rain
                 indoor_excludes={"crickets", "owl", "frogs"},  # not heard over a fire indoors
                 strengths={"gentle": 0.3, "steady": 0.5, "roaring": 0.2},
                 strengths_for={"candle": ["gentle", "steady"]},
                 secondaries=["light_rain", "wind", "noise_bed", "binaural"],
                 tertiaries=["crickets", "owl", "page_turns", "pencil"]),
    "ocean": dict(build=base_ocean, label="Ocean", modifiers={"beach": 1.0}, indoor=set(),
                  strengths={"calm": 0.5, "rolling": 0.5},
                  secondaries=["wind", "noise_bed", "binaural", "drone"], tertiaries=["crickets", "chimes"]),
    "noise": dict(build=base_noise, label="Noise", modifiers={"brown": 1.0, "pink": 0.6}, indoor=None,
                  strengths={"deep": 0.4, "soft": 0.4, "bright": 0.2},
                  strengths_for={"brown": ["deep", "soft"], "pink": ["soft", "bright"]},
                  secondaries=["binaural", "drone", "light_rain"], tertiaries=["page_turns", "pencil"]),
    "fan": dict(build=base_fan, label="Fan", modifiers={"box_fan": 1.0}, indoor=None,
                strengths={"low": 0.5, "high": 0.5},
                secondaries=["light_rain", "binaural"], tertiaries=[]),
    "drone": dict(build=base_drone, label="Frequency", modifiers={f"{f}hz": 1.0 for f in SOLFEGGIO}, indoor=None,
                  strengths={"warm": 0.5, "pure": 0.5},
                  secondaries=["noise_bed", "light_rain", "wind", "stream"], tertiaries=["chimes"]),
    "stream": dict(build=base_stream, label="Stream", modifiers={"brook": 1.0}, indoor=set(),
                   strengths={"steady": 1.0},
                   secondaries=["wind", "noise_bed", "light_rain"],
                   tertiaries=["crickets", "frogs", "owl"], needs=["stream"]),
}

SECONDARIES = {
    "wind": dict(build=sec_wind),
    "noise_bed": dict(build=sec_noise_bed),
    "binaural": dict(build=sec_binaural),
    "drone": dict(build=sec_drone),
    "light_rain": dict(build=sec_light_rain),
    "fire": dict(build=sec_fire, indoor_only=True),
    "stream": dict(build=sec_stream, needs=["stream"]),
    "candle": dict(build=sec_candle, indoor_only=True),
}

# `types`: which videos a tertiary belongs in. Night sounds suit both; desk sounds only study.
TERTIARIES = {
    "thunder": dict(build=ter_thunder, types={"sleep", "study"}),
    "crickets": dict(build=ter_crickets, types={"sleep", "study"}),
    "owl": dict(build=_ter_library("owl", "Owl"), needs=["owl"], types={"sleep", "study"}),
    "frogs": dict(build=_ter_library("frogs", "Frogs"), needs=["frogs"], types={"sleep", "study"}),
    "chimes": dict(build=_ter_library("chimes", "Wind Chimes"), needs=["chimes"], types={"sleep", "study"}),
    "page_turns": dict(build=ter_page_turns, types={"study"}, indoor_only=True),
    "pencil": dict(build=ter_pencil, types={"study"}, indoor_only=True),
}

# Scenes: fixed recipes across the four tiers, with their own title. A part ending in "?" is
# used when it is available (e.g. a library recording) and skipped otherwise. A scene whose
# required parts are not built yet is listed by `scene_status` but cannot be rendered.
SCENES = {
    "rainy_study_window": dict(title="Studying by a Rainy Window", type="study",
                               base="rain", modifier="window", strength="gentle",
                               secondaries=["candle"], tertiaries=["page_turns", "pencil"]),
    "late_night_library": dict(title="Late-Night Library", type="study",
                               base="rain", modifier="window", strength="steady",
                               secondaries=["room_hum", "clock"], tertiaries=["page_turns", "mug"]),
    "cabin_storm": dict(title="Cosy Cabin in a Storm", type="sleep",
                        base="rain", modifier="roof", strength="heavy",
                        secondaries=["fire"], tertiaries=["thunder", "timber_creaks"]),
    "rainy_fireside": dict(title="Fireside on a Rainy Night", type="sleep",
                           base="fire", modifier="fireplace", strength="gentle",
                           secondaries=["light_rain"], tertiaries=[]),  # logs settle inside the fire
    "campfire_stars": dict(title="Campfire Under the Stars", type="sleep",
                           base="fire", modifier="campfire", strength="steady",
                           secondaries=["wind"], tertiaries=["crickets", "owl?"]),
    "beach_night": dict(title="Beach at Night", type="sleep",
                        base="ocean", modifier="beach", strength="calm",
                        secondaries=["wind"], tertiaries=["crickets"]),
    "blizzard_cabin": dict(title="Blizzard Cabin", type="sleep",
                           base="wind", modifier="blizzard", strength="steady",
                           secondaries=["fire", "kettle"], tertiaries=["timber_creaks"]),
    "sleeper_train": dict(title="Sleeper Train in the Rain", type="sleep",
                          base="night_train", modifier="sleeper_cabin", strength="steady",
                          secondaries=["light_rain"], tertiaries=["train_passing"]),
}

# Kept for older callers: a "theme" is a base.
THEMES = BASES


def _usable(table, name, library) -> bool:
    return name in table and all(library_files(library, [c]) for c in table[name].get("needs", []))


def _available(table, names, library):
    return [n for n in names if _usable(table, n, library)]


def available_bases(library=None) -> list[str]:
    return sorted(_available(BASES, list(BASES), library))


def strengths_for(base: str, modifier: str) -> list[str]:
    spec = BASES[base]
    return list(spec.get("strengths_for", {}).get(modifier, spec["strengths"]))


def scene_status(name: str, library=None) -> list[str]:
    """What a scene still lacks (empty = it can be rendered)."""
    s = SCENES[name]
    missing = []
    if not _usable(BASES, s["base"], library):
        missing.append(f"base {s['base']}")
    elif s["modifier"] not in BASES[s["base"]]["modifiers"]:
        missing.append(f"modifier {s['base']}/{s['modifier']}")
    for tier, table in (("secondaries", SECONDARIES), ("tertiaries", TERTIARIES)):
        for part in s[tier]:
            if not part.endswith("?") and not _usable(table, part, library):
                missing.append(part)
    return missing


def available_scenes(library=None) -> list[str]:
    return [n for n in SCENES if not scene_status(n, library)]


# Which secondary reads best in a title when there are two.
_TITLE_ORDER = ["Candlelight", "Distant Wind", "Crackling Fire", "Stream", "Soft Rain", "Brown Noise", "Pink Noise"]


def _headline(base_title, sec_labels, ter_labels):
    named = sorted((l for l in sec_labels if l in _TITLE_ORDER), key=_TITLE_ORDER.index)
    extras = named[:1] + ter_labels[:1]
    if not extras:
        return base_title
    return f"{base_title} with {' and '.join(extras)}"


def random_recipe(seed: int | None = None, base: str | None = None, library: str | None = None,
                  weights: dict[str, float] | None = None, secondaries: list[str] | None = None,
                  tertiaries: list[str] | None = None, modifier: str | None = None,
                  strength: str | None = None, video_type: str = "sleep") -> dict:
    """Make a fresh recipe. Anything left as None is picked at random.

    base         rain, fire, ocean... (see BASES)
    modifier     the base's place or kind, e.g. "window" for rain, "campfire" for fire
    strength     e.g. "gentle", "steady", "heavy" (see the base's `strengths`)
    secondaries  force these secondaries (otherwise 0-2 picked from what suits the base)
    tertiaries   force these tertiaries (otherwise 0-2 picked from what suits the base and video type)
    video_type   "sleep" (tertiaries stop 45-60 min in) or "study" (they carry on)
    weights      favour some bases, e.g. from the weekly report: {"rain": 3, "fan": 0.5}
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    rng = np.random.default_rng(seed)
    if video_type not in VIDEO_TYPES:
        raise ValueError(f"unknown video type {video_type!r}; choose from {', '.join(VIDEO_TYPES)}")
    bases = available_bases(library)
    if base is None:
        w = np.array([float((weights or {}).get(n, 1.0)) for n in bases])
        base = str(rng.choice(bases, p=w / w.sum()))
    if base not in BASES:
        raise ValueError(f"unknown base {base!r}; choose from {', '.join(sorted(BASES))}")
    if base not in bases:
        raise ValueError(f"base {base!r} needs library recordings in: {', '.join(BASES[base]['needs'])}")
    spec = BASES[base]
    if modifier is None:
        modifier = _pick(rng, spec["modifiers"])
    if modifier not in spec["modifiers"]:
        raise ValueError(f"{base} has no modifier {modifier!r}; choose from {', '.join(spec['modifiers'])}")
    allowed = strengths_for(base, modifier)
    if strength is None:
        strength = _pick(rng, {s: spec["strengths"][s] for s in allowed})
    if strength not in allowed:
        raise ValueError(f"{base}/{modifier} has no strength {strength!r}; choose from {', '.join(allowed)}")
    return _build(rng, seed, base, modifier, strength, library, secondaries, tertiaries, video_type)


def scene_recipe(name: str, seed: int | None = None, library: str | None = None) -> dict:
    """The recipe for a scene preset (SCENES), with a fresh random take on its settings."""
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    missing = scene_status(name, library)
    if missing:
        raise ValueError(f"scene {name!r} is not built yet; it still needs: {', '.join(missing)}")
    s = SCENES[name]
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    parts = {}
    for tier, table in (("secondaries", SECONDARIES), ("tertiaries", TERTIARIES)):
        parts[tier] = [p.rstrip("?") for p in s[tier]
                       if not p.endswith("?") or _usable(table, p.rstrip("?"), library)]
    recipe = _build(np.random.default_rng(seed), seed, s["base"], s["modifier"], s["strength"], library,
                    parts["secondaries"], parts["tertiaries"], s["type"])
    recipe["meta"]["scene"] = name
    recipe["meta"]["headline"] = s["title"]
    return recipe


def _build(rng, seed, base, modifier, strength, library, secondaries, tertiaries, video_type) -> dict:
    spec = BASES[base]
    indoor = spec["indoor"] is None or modifier in spec["indoor"]  # None: no place, e.g. noise
    excluded = spec.get("indoor_excludes" if indoor else "outdoor_excludes", set())
    ctx = {"library": library, "sources": [], "features": [], "detail": {}}

    layers, base_title = spec["build"](rng, ctx, modifier, strength)
    for l in layers:  # the base is close and mostly direct, with a touch of the room
        l.setdefault("send", _r(rng, 0.12, 0.25))

    def fits(table, name):
        part = table[name]
        if name in excluded or (part.get("indoor_only") and not indoor):
            return False
        return video_type in part.get("types", VIDEO_TYPES)

    # Secondaries
    if secondaries is None:
        pool = [s for s in _available(SECONDARIES, spec["secondaries"], library) if fits(SECONDARIES, s)]
        k = min(len(pool), int(rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])))
        secondaries = [str(s) for s in rng.choice(pool, size=k, replace=False)] if k else []
    sec_labels = []
    for name in secondaries:
        if name not in SECONDARIES:
            raise ValueError(f"unknown secondary {name!r}; choose from {', '.join(sorted(SECONDARIES))}")
        layer, label = SECONDARIES[name]["build"](rng, ctx, _r(rng, -14, -6, 1))
        if layer["type"] != "binaural":  # tones stay dry so the beat survives
            layer.setdefault("send", _r(rng, 0.25, 0.45))
        layers.append(layer)
        sec_labels.append(label)

    # Tertiaries: thunder only suits proper rain.
    if tertiaries is None:
        pool = [t for t in _available(TERTIARIES, spec["tertiaries"], library) if fits(TERTIARIES, t)]
        if ctx.get("rain_intensity", 1.0) < 0.5:
            pool = [t for t in pool if t != "thunder"]
        k = min(len(pool), int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])))
        tertiaries = [str(t) for t in rng.choice(pool, size=k, replace=False)] if k else []
    all_video = TERTIARIES_ALL_VIDEO[video_type]
    stop_at = None if all_video or not tertiaries else _r(rng, 2700, 3600, 0)  # 45-60 minutes in
    ter_labels = []
    for name in tertiaries:
        if name not in TERTIARIES:
            raise ValueError(f"unknown tertiary {name!r}; choose from {', '.join(sorted(TERTIARIES))}")
        # Night sounds are quiet and far away: mostly reverb, little direct sound. Builders can
        # place their sound closer (a page turned at your own desk) by setting send/dry themselves.
        layer, label = TERTIARIES[name]["build"](rng, ctx, _r(rng, -20, -14, 1))
        layer.setdefault("send", _r(rng, 0.6, 0.85))
        layer.setdefault("dry", _r(rng, 0.3, 0.5))
        layer.setdefault("gate", {})
        if stop_at:
            layer["gate"].update({"until_s": stop_at, "until_fade_s": 300})
        layers.append(layer)
        ter_labels.append(label)

    meta = {
        "video_type": video_type,
        "scene": None,
        "base": base,
        "base_label": spec["label"],
        "modifier": modifier,
        "strength": strength,
        "modifier_detail": ctx["detail"],
        "base_title": base_title,
        "secondaries": list(secondaries),
        "secondary_labels": sec_labels,
        "tertiaries": list(tertiaries),
        "tertiary_labels": ter_labels,
        "tertiaries_until_s": stop_at,
        "combo": "+".join([f"{base}/{modifier}", *sorted(secondaries), *sorted(tertiaries)]),
        "headline": _headline(base_title, sec_labels, ter_labels),
        "features": sorted(set(ctx["features"])) + [l for l in sec_labels if "Waves" in l],
        "binaural": any(l["type"] == "binaural" for l in layers),
        "eight_d": any((l.get("pan") or {}).get("mode") == "circle" for l in layers),
    }
    if ctx["sources"]:
        meta["sources"] = ctx["sources"]
    # Physically modelled rain and fire carry real high-frequency detail; don't dull them much.
    lp = (13000, 16000) if base in ("rain", "fire") else (7000, 12000)
    master = {"lowpass_hz": int(rng.uniform(*lp)), "fade_in_s": 20, "fade_out_s": 45,
              "reverb": {"rt60": _r(rng, 1.6, 3.2), "predelay_ms": _r(rng, 15, 40, 0),
                         "damping": _r(rng, 0.4, 0.8), "level": 1.0}}
    return {"version": 3, "seed": seed, "master": master, "layers": layers, "meta": meta}
