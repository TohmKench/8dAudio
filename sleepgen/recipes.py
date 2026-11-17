"""Random recipe generator.

Each theme picks its own settings at random from ranges that sound good, so
every video is a genuinely different mix. That matters for YouTube: channels
that upload near-identical long videos get flagged as "repetitive content".

A recipe is plain JSON, so it can be saved, edited by hand and re-rendered.
It carries a `meta` block that step 2 uses for titles and descriptions and
the weekly report uses to see which sounds perform best.
"""

from __future__ import annotations

import glob
import os

import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")

# Brainwave bands used for binaural beats (Hz).
BANDS = {"delta": (0.5, 3.5), "theta": (4.0, 7.0)}
SOLFEGGIO = [174, 285, 396, 432, 528, 639]


def _r(rng, lo, hi, nd=2):
    return round(float(rng.uniform(lo, hi)), nd)


def _drift(rng, depth=(1.0, 3.0)):
    return {"depth_db": _r(rng, *depth), "period_s": _r(rng, 60, 240, 0)}


def _pan(rng, mode=None, width=(0.3, 0.6), period=(40, 120)):
    mode = mode or rng.choice(["sweep", "circle", "none"], p=[0.35, 0.35, 0.3])
    return {"mode": str(mode), "width": _r(rng, *width), "period_s": _r(rng, *period, 0)}


def _master(rng, lowpass=(7000, 12000)):
    return {"lowpass_hz": int(rng.uniform(*lowpass)), "fade_in_s": 20, "fade_out_s": 45}


def _noise_bed(rng, level=(-14, -8)):
    color = str(rng.choice(["brown", "pink"], p=[0.7, 0.3]))
    lp = int(rng.uniform(350, 900)) if color == "brown" else int(rng.uniform(1500, 3500))
    return {"type": "noise", "level_db": _r(rng, *level, 1),
            "params": {"color": color, "lowpass_hz": lp}, "drift": _drift(rng)}, f"{color.title()} Noise"


def _binaural(rng, band=None, level=(-20, -14)):
    band = band or str(rng.choice(list(BANDS)))
    beat = _r(rng, *BANDS[band], 1)
    carrier = int(rng.uniform(110, 220))
    layer = {"type": "binaural", "level_db": _r(rng, *level, 1),
             "params": {"carrier_hz": carrier, "beat_hz": beat}}
    return layer, f"{band.title()} Waves {beat:g} Hz"


def _rain(rng, intensity, closeness=(0.2, 0.7)):
    return {"type": "rain", "level_db": 0.0,
            "params": {"intensity": _r(rng, *intensity), "closeness": _r(rng, *closeness),
                       "gusts": _r(rng, 0.1, 0.5), "gust_period": _r(rng, 45, 140, 0)},
            "drift": _drift(rng, (0.5, 2.0))}


def theme_gentle_rain(rng):
    layers, features = [_rain(rng, (0.2, 0.5))], []
    layers[0]["pan"] = _pan(rng, width=(0.2, 0.45))
    if rng.random() < 0.5:
        bed, name = _noise_bed(rng, (-16, -10))
        layers.append(bed)
        features.append(name)
    return layers, {"headline": "Gentle Rain", "features": features}


def theme_heavy_rain(rng):
    bed, name = _noise_bed(rng, (-12, -7))
    return [_rain(rng, (0.65, 0.95), (0.4, 0.9)), bed], {"headline": "Heavy Rain", "features": [name]}


def theme_thunderstorm(rng):
    thunder = {"type": "thunder", "level_db": _r(rng, -6, -2, 1),
               "params": {"min_gap": _r(rng, 90, 180, 0), "max_gap": _r(rng, 300, 600, 0),
                          "darkness": _r(rng, 0.3, 0.8)}}
    return [_rain(rng, (0.55, 0.9)), thunder], {"headline": "Rain and Distant Thunder", "features": []}


def theme_rain_8d(rng):
    rain = _rain(rng, (0.35, 0.75))
    rain["pan"] = _pan(rng, "circle", (0.5, 0.75), (30, 70))
    return [rain], {"headline": "8D Rain", "features": ["8D Audio"]}


def theme_rain_binaural(rng):
    bb, name = _binaural(rng, "delta")
    return [_rain(rng, (0.3, 0.7)), bb], {"headline": "Rain", "features": [name, "Binaural Beats"]}


def theme_ocean(rng):
    layers = [{"type": "ocean", "level_db": 0.0,
               "params": {"min_period": _r(rng, 6, 9, 1), "max_period": _r(rng, 10, 15, 1),
                          "brightness": _r(rng, 0.2, 0.7), "calm": _r(rng, 0.2, 0.45)},
               "pan": _pan(rng, width=(0.15, 0.35), period=(90, 200))}]
    features = []
    if rng.random() < 0.35:
        bb, name = _binaural(rng)
        layers.append(bb)
        features.append(name)
    return layers, {"headline": "Ocean Waves", "features": features}


def theme_brown_noise(rng):
    lp = int(rng.uniform(250, 700))
    layer = {"type": "noise", "level_db": 0.0, "params": {"color": "brown", "lowpass_hz": lp},
             "drift": _drift(rng, (0.5, 2.0)), "pan": _pan(rng, width=(0.15, 0.35), period=(120, 300))}
    headline = "Deep Brown Noise" if lp < 450 else "Brown Noise"
    return [layer], {"headline": headline, "features": []}


def theme_pink_noise(rng):
    layer = {"type": "noise", "level_db": 0.0,
             "params": {"color": "pink", "lowpass_hz": int(rng.uniform(1800, 4500))},
             "drift": _drift(rng, (0.5, 1.5))}
    return [layer], {"headline": "Pink Noise", "features": []}


def theme_fan(rng):
    layer = {"type": "fan", "level_db": 0.0,
             "params": {"size": _r(rng, 0.2, 0.9), "hum": _r(rng, 0.0, 0.06, 3), "flutter": _r(rng, 0.0, 0.05, 3)}}
    return [layer], {"headline": "Fan Noise", "features": []}


def theme_delta_waves(rng):
    bed, name = _noise_bed(rng, (0, 0))
    bb, bname = _binaural(rng, "delta", (-14, -9))
    return [bed, bb], {"headline": bname, "features": [name, "Binaural Beats"]}


def theme_solfeggio(rng):
    f = int(rng.choice(SOLFEGGIO))
    drone = {"type": "drone", "level_db": 0.0, "params": {"freq_hz": f, "warmth": _r(rng, 0.2, 0.8)},
             "pan": _pan(rng, width=(0.2, 0.5), period=(60, 160)), "drift": _drift(rng)}
    layers, features = [drone], []
    roll = rng.random()
    if roll < 0.4:
        rain = _rain(rng, (0.2, 0.5))
        rain["level_db"] = _r(rng, -9, -4, 1)
        layers.append(rain)
        features.append("Rain")
    elif roll < 0.8:
        bed, name = _noise_bed(rng, (-12, -6))
        layers.append(bed)
        features.append(name)
    return layers, {"headline": f"{f} Hz", "features": features}


THEMES = {
    "gentle_rain": theme_gentle_rain,
    "heavy_rain": theme_heavy_rain,
    "thunderstorm": theme_thunderstorm,
    "rain_8d": theme_rain_8d,
    "rain_binaural": theme_rain_binaural,
    "ocean": theme_ocean,
    "brown_noise": theme_brown_noise,
    "pink_noise": theme_pink_noise,
    "fan": theme_fan,
    "delta_waves": theme_delta_waves,
    "solfeggio": theme_solfeggio,
}

# Library folder names a theme can borrow a real recording from.
THEME_LIBRARY = {
    "gentle_rain": ["rain", "rain_roof", "rain_window"],
    "heavy_rain": ["rain", "rain_roof"],
    "thunderstorm": ["rain", "thunder"],
    "rain_8d": ["rain"],
    "rain_binaural": ["rain", "rain_window"],
    "ocean": ["ocean"],
    "solfeggio": ["stream", "wind"],
}


def library_files(library: str | None, categories) -> list[str]:
    if not library:
        return []
    files = []
    for cat in categories:
        for path in glob.glob(os.path.join(library, cat, "**", "*"), recursive=True):
            if path.lower().endswith(AUDIO_EXTS):
                files.append(path)
    return sorted(files)


def random_recipe(seed: int | None = None, theme: str | None = None, library: str | None = None,
                  weights: dict[str, float] | None = None) -> dict:
    """Make a fresh recipe. `weights` lets the weekly report favour themes that perform well."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    rng = np.random.default_rng(seed)
    names = sorted(THEMES)
    if theme is None:
        w = np.array([float((weights or {}).get(n, 1.0)) for n in names])
        theme = str(rng.choice(names, p=w / w.sum()))
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; choose from {', '.join(names)}")
    layers, meta = THEMES[theme](rng)

    # Blend in a real recording when the library has one for this theme.
    files = library_files(library, THEME_LIBRARY.get(theme, []))
    if files and rng.random() < 0.7:
        path = str(rng.choice(files))
        layers.append({"type": "sample", "level_db": _r(rng, -6, 0, 1), "params": {"path": path},
                       "drift": _drift(rng, (1.0, 2.5))})
        meta.setdefault("sources", []).append(os.path.relpath(path, library))

    meta["theme"] = theme
    meta["binaural"] = any(l["type"] == "binaural" for l in layers)
    meta["eight_d"] = any((l.get("pan") or {}).get("mode") == "circle" for l in layers)
    return {"version": 1, "seed": seed, "master": _master(rng), "layers": layers, "meta": meta}
