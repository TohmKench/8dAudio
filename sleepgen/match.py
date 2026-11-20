"""Match a synthetic layer to a reference recording.

Measures a reference clip (tone balance, drop density and sharpness, how the level
moves), searches the layer's settings for the closest match, then adds a matching
EQ for the remaining tone difference. Used as a measuring stick only: none of the
reference audio ends up in the output.

  python -m sleepgen match reference.mp3 --base rain -o matched/
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import signal

from .dsp import SR
from .engine import Mix, render
from .layers import decode_audio

# Third-octave band centres used for the tone comparison.
BANDS = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
         3150, 4000, 5000, 6300, 8000, 10000, 12500]

# Search spaces: name -> (low, high, log-scale?)
SPACES = {
    "rain": {
        "intensity": (0.1, 1.0, False),
        "closeness": (0.0, 1.0, False),
        "plinks": (0.0, 0.25, False),
        "wash_hz": (400, 5000, True),
        "wash_mix": (0.05, 1.5, True),
        "patter_rate": (80, 9000, True),
        "tap_lo": (600, 4000, True),
        "tap_span": (1.2, 4.0, False),
        "tone_hz": (2500, 14000, True),
    },
    "ocean": {
        "min_period": (4.0, 10.0, False),
        "max_period": (8.0, 18.0, False),
        "brightness": (0.0, 1.0, False),
        "calm": (0.1, 0.8, False),
    },
}


# ------------------------------------------------------------------------ measuring

def _mono(x):
    m = x.mean(axis=1) if x.ndim == 2 else x
    m = m - m.mean()
    return m / (np.sqrt(np.mean(m ** 2)) + 1e-12)


def _frame_db(x, frame):
    n = len(x) // frame
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    return 20 * np.log10(e + 1e-9)


def measure(x: np.ndarray) -> dict:
    """Describe a clip in numbers that track how rain/sea textures are heard."""
    if len(x) > SR * 8:  # skip any fade at the edges
        x = x[SR * 2: -SR * 2]
    m = _mono(x)
    f, p = signal.welch(m, SR, nperseg=8192)
    total = np.sum(p)
    bands = []
    for c in BANDS:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        sel = (f >= lo) & (f < hi)
        bands.append(10 * np.log10(np.sum(p[sel]) / total + 1e-12))
    hp = signal.sosfilt(signal.butter(4, [1000, 8000], "bandpass", fs=SR, output="sos"), m)
    fine = _frame_db(hp, SR // 200)  # 5 ms frames
    base = signal.medfilt(fine, 41)
    peaks, _ = signal.find_peaks(fine - base, height=4.0, distance=2)
    slow = _frame_db(m, SR // 10)  # 100 ms frames
    slow = slow - np.convolve(slow, np.ones(30) / 30, mode="same")  # ignore drifts slower than ~3 s
    slow = slow[15:-15]
    return {
        "bands_db": [round(float(b), 2) for b in bands],
        "drops_per_s": round(len(peaks) / (len(m) / SR), 1),
        "grain_db": round(float(np.std(fine - base)), 2),
        "crest_db": round(float(20 * np.log10(np.percentile(np.abs(m), 99.95))), 2),
        "swell_db": round(float(np.std(slow)), 2),
    }


def compare(ref: dict, ours: dict) -> dict:
    """Per-measure similarity in percent (100 = identical) and an overall score."""
    tone_err = float(np.mean(np.abs(np.array(ref["bands_db"]) - np.array(ours["bands_db"]))))
    parts = {
        "tone": 100 * math.exp(-tone_err / 6.0),
        "drop density": 100 * math.exp(-abs(math.log((ours["drops_per_s"] + 1) / (ref["drops_per_s"] + 1))) / 0.5),
        "graininess": 100 * math.exp(-abs(ours["grain_db"] - ref["grain_db"]) / 2.0),
        "sharpness": 100 * math.exp(-abs(ours["crest_db"] - ref["crest_db"]) / 3.0),
        "level movement": 100 * math.exp(-abs(ours["swell_db"] - ref["swell_db"]) / 2.0),
    }
    weights = {"tone": 3, "drop density": 2, "graininess": 2, "sharpness": 1, "level movement": 1}
    overall = sum(parts[k] * w for k, w in weights.items()) / sum(weights.values())
    return {"overall": round(overall, 1), **{k: round(v, 1) for k, v in parts.items()},
            "tone_error_db": round(tone_err, 2)}


# ------------------------------------------------------------------------ searching

def _params(base: str, v: dict) -> dict:
    if base != "rain":
        p = dict(v)
        if p.get("max_period", 0) < p.get("min_period", 0) + 1:
            p["max_period"] = p["min_period"] + 1
        return p
    p = {k: v[k] for k in ("intensity", "closeness", "plinks", "wash_hz", "wash_mix", "patter_rate", "tone_hz")}
    p["tap_hz"] = [v["tap_lo"], min(v["tap_lo"] * v["tap_span"], 20000)]
    p["gusts"] = 0.2
    return p


def _recipe(base: str, params: dict, eq=None, seed=1) -> dict:
    master = {"reverb": {"rt60": 1.8, "predelay_ms": 20, "damping": 0.6}, "leveler": False}
    if eq:
        master["eq"] = eq
    return {"seed": seed, "master": master,
            "layers": [{"type": base, "level_db": 0.0, "params": params, "send": 0.18}],
            "meta": {"base": base, "headline": f"matched {base}"}}


def _synth(recipe: dict, seconds: float) -> np.ndarray:
    mix = Mix(recipe)
    mix.raw(SR * 2)  # let the reverb fill in
    return np.concatenate([mix.raw(SR * 5) for _ in range(int(seconds // 5))])


def _sample_point(space, rng):
    v = {}
    for k, (lo, hi, log) in space.items():
        v[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if log else float(rng.uniform(lo, hi))
    return v


def _nudge(space, v, rng, scale):
    out = {}
    for k, (lo, hi, log) in space.items():
        if log:
            out[k] = float(np.clip(math.exp(math.log(v[k]) + rng.normal(0, scale * math.log(hi / lo))), lo, hi))
        else:
            out[k] = float(np.clip(v[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return out


def match_eq(ref: dict, ours: dict, limit_db: float = 9.0):
    """EQ points that move our tone balance onto the reference's."""
    diff = np.array(ref["bands_db"]) - np.array(ours["bands_db"])
    diff = np.convolve(np.pad(diff, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")  # smooth
    diff -= np.median(diff)
    return [[float(c), round(float(np.clip(d, -limit_db, limit_db)), 2)] for c, d in zip(BANDS, diff)]


def run(reference: str, base: str = "rain", out_dir: str = "matched", evals: int = 70,
        seconds: float = 12.0, seed: int = 0, rounds: int = 3, refine: int = 15, log=print) -> dict:
    if base not in SPACES:
        raise ValueError(f"matching supports: {', '.join(SPACES)}")
    os.makedirs(out_dir, exist_ok=True)
    ref_audio = decode_audio(reference, max_seconds=240)
    ref = measure(ref_audio)
    log(f"reference: {json.dumps({k: v for k, v in ref.items() if k != 'bands_db'})}")
    space = SPACES[base]
    rng = np.random.default_rng(seed)

    def score(v, eq=None):
        ours = measure(_synth(_recipe(base, _params(base, v), eq), seconds))
        return compare(ref, ours)["overall"], ours

    best_v, best_s, best_m = None, -1.0, None
    explore = max(10, evals // 2)
    for i in range(evals):
        if i < explore or best_v is None:
            v = _sample_point(space, rng)
        else:
            v = _nudge(space, best_v, rng, 0.25 * (1 - (i - explore) / max(evals - explore, 1)) + 0.03)
        s, m = score(v)
        if s > best_s:
            best_v, best_s, best_m = v, s, m
            log(f"  [{i + 1}/{evals}] new best {s:.1f}")
    before = compare(ref, best_m)

    # Refine: add a matching EQ, re-tune the settings with it in place, repeat.
    eq = None
    for rnd in range(rounds):
        step = match_eq(ref, best_m)
        if eq is None:
            eq = step
        else:
            eq = [[f, round(float(np.clip(g + 0.7 * d, -12, 12)), 2)] for (f, g), (_, d) in zip(eq, step)]
        best_s, best_m = score(best_v, eq)
        for i in range(refine):
            v = _nudge(space, best_v, rng, 0.08 * (1 - i / refine) + 0.02)
            s, m = score(v, eq)
            if s > best_s:
                best_v, best_s, best_m = v, s, m
        log(f"  refine round {rnd + 1}: {best_s:.1f}")
    params = _params(base, best_v)
    after_m = measure(_synth(_recipe(base, params, eq), seconds * 3))
    after = compare(ref, after_m)
    log(f"settings only: {before['overall']}  after refining with matching EQ: {after['overall']}")

    recipe = _recipe(base, params, eq)
    recipe["master"].pop("leveler", None)
    with open(os.path.join(out_dir, f"matched-{base}.json"), "w") as f:
        json.dump(recipe, f, indent=2)
    # Side-by-side clips at the same loudness.
    clip = min(60.0, len(ref_audio) / SR)
    render(recipe, os.path.join(out_dir, f"ours-{base}.mp3"), 60, bitrate="256k", progress=False)
    _loudness_matched_copy(reference, os.path.join(out_dir, f"reference-{base}.mp3"), clip)
    report = {"reference": os.path.basename(reference), "base": base, "similarity_settings_only": before,
              "similarity_with_eq": after, "reference_measures": ref, "our_measures": after_m,
              "params": params, "eq": eq}
    with open(os.path.join(out_dir, f"report-{base}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _loudness_matched_copy(src, dst, seconds):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-t", str(seconds),
                    "-af", "loudnorm=I=-26:TP=-2:LRA=11", "-ar", str(SR), "-b:a", "256k", dst], check=True)
