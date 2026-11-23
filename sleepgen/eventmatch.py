"""Tune a short generated sound (a page turn, for now) until it can't be told from real
recordings of the same thing, without anyone having to describe what they hear.

  python -m sleepgen refs fetch page_turns --query "page turn"      (CC0 only, from Freesound)
  python -m sleepgen tune page_turns references/page_turns -o runs/page_turns
  python -m sleepgen listen runs/page_turns                        (the A/B page)
  python -m sleepgen tune page_turns references/page_turns -o runs/page_turns --resume

1. References  a few real recordings, cut into single events (`segment`). They are only a
               measuring stick: none of their audio goes into a video.
2. Measures    every event becomes numbers in six groups, chosen so each failure heard so far
               shows up in one of them (`features`):
                 tone     third-octave balance           (too bright, too dull)
                 shape    length, attack, decay, outline (a "bang" starts too fast and ends too soon)
                 texture  how many tiny transients and how fast the level flickers
                          ("fabric ripping" is a dense, even buzz of them)
                 ring     narrow peaks in the spectrum   ("metal scraping" rings)
                 bang     crest factor and the biggest sudden jump in level
                 colour   where the brightness sits and how it moves through the event
3. Score       how far the generated events sit from the real ones, each measure scaled by how
               much the real recordings vary among themselves (a symmetric nearest-neighbour
               distance, so the generated events must also cover the real ones' variety). With
               an audio recognition model installed (CLAP, `pip install transformers`), it also
               asks the model what it hears; the known failures are labels it must not pick.
4. Loop        an evolution strategy over the generator's settings (`SPACES`): 24 variations a
               round, keep the best 6, breed the next round around them, narrowing as it goes.
5. Listening   the best 3 and a real recording, in pairs: "A is better", "B is better" or
               "Can't tell" (`listen.py`). `--resume` reads those picks, shifts the score's
               weights toward the groups that gave the generated ones away, and goes on from
               the preferred one. A sound is done when the real one stops winning.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy import signal

from .dsp import SR

GROUPS = ("tone", "shape", "texture", "ring", "bang", "colour")

# Third-octave bands for tone (stops at 12.5 kHz: Freesound previews are 128 kbps mp3).
BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
         4000, 5000, 6300, 8000, 10000, 12500]
ENV_POINTS = 12

# The smallest spread a measure is given, so a handful of references can't make it razor sharp
# (in the measure's own units: dB, log seconds, octaves ...).
_FLOORS = {"tone": 2.0, "length": 0.2, "attack": 0.35, "decay": 0.35, "peak_pos": 0.1, "humps": 0.6,
           "env": 3.0, "micro_rate": 0.3, "kurtosis": 0.25, "mod_lo": 2.0, "mod_mid": 2.0, "mod_hi": 2.0,
           "ring_max": 2.0, "ring_p95": 1.0, "flatness": 0.08, "crest": 1.5, "jump": 2.0,
           "centroid": 0.2, "centroid_slope": 0.25, "centroid_std": 0.1}


# ----------------------------------------------------------------------------- cutting

def _mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _frames_db(x, hop):
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    e = np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)
    return 10 * np.log10(e + 1e-20)


_HP80 = signal.butter(2, 80, "highpass", fs=SR, output="sos")


def segment(x, min_gap=0.5, pad=0.04, min_len=0.08, max_len=4.0, snr_db=20.0, bounds=False):
    """Cut a recording into single events (one page turn each): stretches that rise well
    above the recording's own background and are separated by at least `min_gap` of it.
    With bounds=True, returns (start, end) sample pairs (without the padding) instead."""
    m = signal.sosfiltfilt(_HP80, _mono(x))
    hop = SR // 100
    env = _frames_db(m, hop)
    floor = np.percentile(env, 10)
    thr = max(floor + 8.0, env.max() - 45.0)
    on = env > thr
    # Bridge short dips, so one turn with a quiet moment stays one event.
    gap = int(min_gap * 100)
    idx = np.flatnonzero(on)
    events = []
    if len(idx) == 0:
        return events
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    for a, b in runs:
        length = (b - a + 1) / 100
        if length < min_len or length > max_len or env[a:b + 1].max() < floor + snr_db:
            continue
        if bounds:
            events.append((a * hop, (b + 1) * hop))
            continue
        lo = max(0, a * hop - int(pad * SR))
        hi = min(len(m), (b + 1) * hop + int(pad * SR))
        events.append(m[lo:hi])
    return events


def room_noise(x, min_s=0.15):
    """The quiet stretches of a recording (its room tone), joined end to end."""
    m = signal.sosfiltfilt(_HP80, _mono(x))
    hop = SR // 100
    env = _frames_db(m, hop)
    quiet = env < np.percentile(env, 10) + 3.0
    parts, run = [], []
    for i, q in enumerate(quiet):
        if q:
            run.append(i)
        elif run:
            if len(run) * hop >= min_s * SR:
                parts.append(m[(run[0] + 2) * hop:(run[-1] - 1) * hop])
            run = []
    if run and len(run) * hop >= min_s * SR:
        parts.append(m[(run[0] + 2) * hop:(run[-1] - 1) * hop])
    return np.concatenate(parts) if parts else np.zeros(0)


def sounding_rms(x):
    """RMS over the part of a clip within 20 dB of its loudest 10 ms."""
    e = _frames_db(_mono(x), SR // 100)
    loud = e[e > e.max() - 20]
    return float(np.sqrt(np.mean(10 ** (loud / 10))))


def write_wav(path, x, peak=0.89):
    """16-bit WAV, mono or stereo, scaled down only if it would clip."""
    from scipy.io import wavfile
    x = np.asarray(x, dtype=np.float64)
    g = min(1.0, peak / (np.max(np.abs(x)) + 1e-12))
    wavfile.write(path, SR, (x * g * 32767).astype(np.int16))


# ---------------------------------------------------------------------------- measures

def _band_db(f, p, bands):
    total = np.sum(p) + 1e-30
    out = []
    for c in bands:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(np.sum(p[sel]) / total + 1e-12))
    return np.array(out)


def features(ev) -> dict:
    """Describe one event (mono or stereo, any level) in numbers grouped as in GROUPS."""
    x = signal.sosfiltfilt(_HP80, _mono(ev)) if len(ev) > 30 else _mono(ev)
    hop = SR // 200  # 5 ms
    env = _frames_db(x, hop)
    env -= env.max()
    sounding = np.flatnonzero(env > -20)
    first, last = int(sounding[0]), int(sounding[-1])
    span = x[first * hop:(last + 1) * hop]
    span = span / (np.sqrt(np.mean(span ** 2)) + 1e-20)
    f = {}

    # tone: the spectrum over the part that sounds
    nper = min(2048, 1 << int(np.log2(max(256, len(span)))))
    fr, p = signal.welch(span, SR, nperseg=nper)
    f["tone"] = _band_db(fr, p, BANDS)

    # shape
    above3 = np.flatnonzero(env > -3)
    f["length"] = math.log((last - first + 1) * hop / SR)
    f["attack"] = math.log(1 + (above3[0] - first) * hop / SR * 1000)
    f["decay"] = math.log(1 + (last - above3[-1]) * hop / SR * 1000)
    f["peak_pos"] = (int(np.argmax(env)) - first) / max(1, last - first)
    sm = np.convolve(env, np.ones(4) / 4, mode="same")
    pk, _ = signal.find_peaks(sm[first:last + 1], prominence=6)
    f["humps"] = float(len(pk))
    outline = np.clip(env[first:last + 1], -30, 0)
    f["env"] = np.interp(np.linspace(0, len(outline) - 1, ENV_POINTS), np.arange(len(outline)), outline)

    # texture: tiny transients above 2 kHz, and how fast the level flickers
    hp = signal.sosfilt(signal.butter(2, 2000, "highpass", fs=SR, output="sos"), span)
    e1 = _frames_db(hp, SR // 1000)
    base = signal.medfilt(e1, 31) if len(e1) >= 31 else np.full_like(e1, np.median(e1))
    micro, _ = signal.find_peaks(e1 - base, height=6, distance=2)
    f["micro_rate"] = math.log(1 + len(micro) / (len(span) / SR))
    k = np.mean(hp ** 4) / (np.mean(hp ** 2) ** 2 + 1e-30)
    f["kurtosis"] = math.log(max(k, 1e-3))
    a = np.sqrt(10 ** (_frames_db(span, SR // 1000) / 10))  # 1 kHz envelope
    slow = np.convolve(a, np.ones(31) / 31, mode="same") + 1e-12
    flick = a / slow - 1
    if len(flick) >= 64:
        mf, mp = signal.welch(flick, 1000, nperseg=min(256, len(flick)))
        tot = np.sum(mp[(mf >= 4) & (mf < 250)]) + 1e-20
        for name, lo, hi in (("mod_lo", 4, 16), ("mod_mid", 16, 64), ("mod_hi", 64, 250)):
            f[name] = 10 * np.log10(np.sum(mp[(mf >= lo) & (mf < hi)]) / tot + 1e-6)
    else:
        f["mod_lo"] = f["mod_mid"] = f["mod_hi"] = -5.0

    # ring: narrow peaks standing out of the spectrum
    fr2, p2 = signal.welch(span, SR, nperseg=min(4096, nper * 2) if len(span) >= 4096 else nper)
    sel = (fr2 >= 300) & (fr2 <= 10000)
    pd = 10 * np.log10(p2[sel] + 1e-30)
    ratio = pd - signal.medfilt(pd, 31 if len(pd) > 31 else max(1, (len(pd) // 2) * 2 - 1))
    f["ring_max"] = float(min(ratio.max(), 30.0))
    f["ring_p95"] = float(np.percentile(ratio, 95))
    f["flatness"] = float(np.exp(np.mean(np.log(p2[sel] + 1e-30))) / (np.mean(p2[sel]) + 1e-30))

    # bang: a sudden peak
    f["crest"] = 20 * math.log10(np.max(np.abs(span)) + 1e-12)
    e10 = _frames_db(span, SR // 100)
    before = np.array([e10[max(0, i - 5):i].max() if i else -120.0 for i in range(len(e10))])
    jumps = (e10 - before)[1:]
    f["jump"] = float(np.clip(jumps.max(), 0, 40)) if len(jumps) else 0.0

    # colour: spectral centroid (octaves above 1 kHz) and its movement
    fw, tw, S = signal.stft(span, SR, nperseg=1024, noverlap=512)
    P = np.abs(S) ** 2
    w = P.sum(axis=0)
    keep = w > w.max() * 0.01
    cen = np.log2((fw[:, None] * P).sum(axis=0)[keep] / (w[keep] + 1e-30) / 1000 + 1e-6)
    f["centroid"] = float(np.average(cen, weights=w[keep]))
    f["centroid_std"] = float(np.sqrt(np.average((cen - f["centroid"]) ** 2, weights=w[keep])))
    tpos = np.linspace(0, 1, len(cen)) if len(cen) > 1 else np.zeros(1)
    f["centroid_slope"] = float(np.polyfit(tpos, cen, 1, w=np.sqrt(w[keep]))[0]) if len(cen) > 2 else 0.0
    return f


GROUP_OF = {"tone": "tone", "length": "shape", "attack": "shape", "decay": "shape", "peak_pos": "shape",
            "humps": "shape", "env": "shape", "micro_rate": "texture", "kurtosis": "texture",
            "mod_lo": "texture", "mod_mid": "texture", "mod_hi": "texture", "ring_max": "ring",
            "ring_p95": "ring", "flatness": "ring", "crest": "bang", "jump": "bang", "centroid": "colour",
            "centroid_std": "colour", "centroid_slope": "colour"}


def _vector(f):
    """Flatten a features dict to (values, group index, floor) arrays in a fixed order."""
    vals, grp, flo = [], [], []
    for name, g in GROUP_OF.items():
        v = np.atleast_1d(np.asarray(f[name], dtype=float))
        vals.append(v)
        grp += [GROUPS.index(g)] * len(v)
        flo += [_FLOORS[name]] * len(v)
    return np.concatenate(vals), np.array(grp), np.array(flo)


# ------------------------------------------------------------------------------ score

class Reference:
    """The real recordings' events, measured once."""

    def __init__(self, events, names=None, clips=None, room=None, snr_db=60.0):
        if len(events) < 2:
            raise ValueError("need at least 2 reference events to score against")
        self.events = events
        self.names = names or [f"ref{i}" for i in range(len(events))]
        self.clips = clips or events  # the same events with more of their room around them
        self.room = room if room is not None and len(room) > SR // 2 else None
        self.snr_db = float(snr_db)  # how far the room tone sits under the events
        rows = [_vector(features(e)) for e in events]
        self.F = np.stack([r[0] for r in rows])
        self.group = rows[0][1]
        floor = rows[0][2]
        self.scale = np.sqrt(self.F.std(axis=0) ** 2 + floor ** 2)
        self.listener_good = None  # set by the listener: how much it hears the target in real ones

    @classmethod
    def from_folder(cls, folder, max_events=40, seed=0):
        """Every event in every recording in `folder` (not its subfolders)."""
        from .layers import decode_audio
        events, names, clips, rooms, snrs = [], [], [], [], []
        exts = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff", ".aif", ".opus")
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(exts):
                continue
            x = signal.sosfiltfilt(_HP80, _mono(decode_audio(os.path.join(folder, fn), max_seconds=600)))
            room = room_noise(x)
            noise = np.sqrt(np.mean(room ** 2)) if len(room) else 0.0
            for i, (a, b) in enumerate(segment(x, bounds=True)):
                ev = x[max(0, a - int(0.04 * SR)):b + int(0.04 * SR)]
                events.append(ev)
                clips.append(x[max(0, a - int(0.3 * SR)):b + int(0.3 * SR)])
                names.append(f"{fn}#{i + 1}")
                if noise > 0:
                    snrs.append(20 * np.log10(sounding_rms(ev) / noise))
            if len(room):
                rooms.append(room / (np.sqrt(np.mean(room ** 2)) + 1e-20))
        if len(events) > max_events:  # many from one long file shouldn't swamp the rest
            keep = np.sort(np.random.default_rng(seed).choice(len(events), max_events, replace=False))
            events, names, clips = [events[i] for i in keep], [names[i] for i in keep], [clips[i] for i in keep]
        return cls(events, names, clips, np.concatenate(rooms) if rooms else None,
                   float(np.median(snrs)) if snrs else 60.0)

    def distances(self, F, weights):
        """Per-group mean squared z between every row of F and every reference: (n, refs, groups)."""
        z2 = ((F[:, None, :] - self.F[None, :, :]) / self.scale) ** 2
        out = np.zeros(F.shape[:1] + (len(self.F), len(GROUPS)))
        for g in range(len(GROUPS)):
            out[..., g] = z2[..., self.group == g].mean(axis=-1)
        return out

    def score(self, events, weights=None) -> dict:
        """How far `events` are from the real ones: 0 is identical, ~1 is as different as the
        real ones are among themselves. Also how a simple classifier would tell them apart."""
        w = np.ones(len(GROUPS)) if weights is None else np.asarray([weights.get(g, 1.0) for g in GROUPS])
        F = np.stack([_vector(features(e))[0] for e in events])
        D = self.distances(F, w)  # (n, refs, groups)
        tot = (D * w).sum(axis=-1) / w.sum()
        gen_nn = tot.argmin(axis=1)
        ref_nn = tot.argmin(axis=0)
        # Symmetric: every generated event near some real one, and every real one near some
        # generated one (so a single safe-sounding turn repeated can't win).
        groups = {}
        for g, name in enumerate(GROUPS):
            a = D[np.arange(len(F)), gen_nn, g].mean()
            b = D[ref_nn, np.arange(len(self.F)), g].mean()
            groups[name] = float(np.sqrt(0.5 * (a + b)))
        dist = float(np.sqrt(0.5 * (tot.min(axis=1).mean() + tot.min(axis=0).mean())))
        return {"distance": dist, "groups": groups, "tell_apart": self.tell_apart(F, w), "F": F}

    def tell_apart(self, F, w):
        """Leave-one-out 1-nearest-neighbour accuracy at telling generated from real (balanced):
        0.5 means a classifier can't tell them apart on these measures, 1.0 means always."""
        n = min(len(F), len(self.F))
        X = np.concatenate([F[:n], self.F[:n]]) / self.scale
        y = np.r_[np.zeros(n), np.ones(n)]
        gw = np.array([w[g] / np.sum(self.group == g) for g in self.group])
        d = (((X[:, None, :] - X[None, :, :]) ** 2) * gw).sum(axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(np.mean(y[d.argmin(axis=1)] == y))


# --------------------------------------------------------------------------- listener

TARGETS = {
    "page_turns": dict(
        question="Which sounds more like a real page turn?",
        good=["a page of a book being turned", "turning the pages of a paper book", "paper rustling"],
        bad=["a loud bang", "metal scraping", "fabric ripping", "static noise", "wind blowing",
             "a plastic bag crinkling"]),
}


def _unit_emb(out):
    """An embedding scaled to length 1 (transformers 5 wraps it in an output object)."""
    t = out if hasattr(out, "norm") else out.pooler_output
    return t / t.norm(dim=-1, keepdim=True)


class Listener:
    """An audio recognition model (CLAP) asked which description fits a clip best. Optional:
    needs `pip install transformers` and downloads the model (~600 MB) on first use."""

    def __init__(self, target, model="laion/clap-htsat-unfused"):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.labels = TARGETS[target]["good"] + TARGETS[target]["bad"]
        self.n_good = len(TARGETS[target]["good"])
        self.proc = ClapProcessor.from_pretrained(model)
        self.model = ClapModel.from_pretrained(model).eval()
        with torch.no_grad():
            t = self.proc(text=self.labels, return_tensors="pt", padding=True)
            self.text = _unit_emb(self.model.get_text_features(**t))

    def hear(self, events) -> np.ndarray:
        """Probabilities over self.labels for each event: (n, labels)."""
        torch = self.torch
        out = []
        for ev in events:
            x = _mono(ev).astype(np.float32)
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
            with torch.no_grad():
                try:  # the keyword was renamed in newer transformers
                    a = self.proc(audio=[x], sampling_rate=SR, return_tensors="pt")
                except (TypeError, ValueError):
                    a = self.proc(audios=[x], sampling_rate=SR, return_tensors="pt")
                e = _unit_emb(self.model.get_audio_features(**a))
                logits = (e @ self.text.T) * self.model.logit_scale_a.exp()
                out.append(torch.softmax(logits, dim=-1)[0].numpy())
        return np.stack(out)

    def judge(self, events, ref_good=None) -> dict:
        P = self.hear(events)
        good = P[:, : self.n_good].sum(axis=1)
        top = [self.labels[i] for i in P.argmax(axis=1)]
        bad_tops = [t for t in top if self.labels.index(t) >= self.n_good]
        heard = {}
        for t in top:
            heard[t] = heard.get(t, 0) + 1
        penalty = max(0.0, (ref_good if ref_good is not None else 0.8) - float(good.mean()))
        return {"good": float(good.mean()), "heard": heard, "fails": len(bad_tops) / len(top),
                "penalty": penalty}


def try_listener(target, log=print):
    try:
        return Listener(target)
    except Exception as e:  # noqa: BLE001 - optional extra: report and carry on without it
        log(f"(no automatic listener: {type(e).__name__}: {str(e)[:120]})")
        return None


# ------------------------------------------------------------------------- generators

def _page_turns(params, n, seed, per_book=2):
    """n single turns, a couple from each of several books (each has its own paper grain and
    resonance, as each real recording is of a different book)."""
    from .desk import PageTurns
    out = []
    for i in range(math.ceil(n / per_book)):
        pt = PageTurns(np.random.default_rng([seed, i]), flurry=0.0, azimuth=0.0, shape=params)
        out += [pt.turn() for _ in range(per_book)]
    return out[:n]


def cut_like_references(events, seed=0):
    """Put generated events through the same cutting as the recordings (with a faint room
    noise under them), so both sides are measured the same way: a hand smoothing the page a
    moment after the turn becomes its own event in both, for instance."""
    rng = np.random.default_rng(seed)
    out = []
    for ev in events:
        x = _mono(ev)
        level = np.sqrt(np.mean(x ** 2)) + 1e-12
        pad = np.zeros(int(0.6 * SR))
        y = np.concatenate([pad, x, pad])
        y = y + rng.standard_normal(len(y)) * level * 10 ** (-55 / 20)
        out += segment(y)
    return out or [_mono(e) for e in events]


GENERATORS = {"page_turns": _page_turns}

# The settings searched, as (low, high, log scale?). Wide on purpose: the loop may find that a
# page turn wants far fewer crackles and much more air than anything tried by ear.
SPACES = {
    "page_turns": {
        "rustle": (20.0, 900.0, True),
        "colour_lo": (500.0, 4000.0, True),
        "colour_hi": (1500.0, 14000.0, True),
        "crackle_ms": (0.2, 4.0, True),
        "crackle_hp": (150.0, 1500.0, True),
        "sizes_alpha": (1.3, 5.0, False),
        "speed_exp": (0.3, 2.0, False),
        "swing": (0.15, 0.9, True),
        "rustle_level": (0.01, 0.4, True),
        "air": (0.01, 0.8, True),
        "air_lo": (80.0, 1200.0, True),
        "air_hi": (1000.0, 10000.0, True),
        "fff_level": (0.003, 0.2, True),
        "fff_decay": (0.02, 0.3, True),
        "land_rustle": (0.0, 0.3, False),
        "flap_level": (0.0, 0.3, False),
        "flap_hz": (150.0, 2000.0, True),
        "settle_lp": (1000.0, 9000.0, True),
    },
}


def defaults(target):
    if target == "page_turns":
        from .desk import PAGE_SHAPE, PAPERS
        d = {**PAGE_SHAPE, **PAPERS["book"]}
        return {k: float(d[k]) for k in SPACES[target]}
    raise ValueError(f"unknown target {target!r}")


def _to_unit(params, space):
    u = []
    for k, (lo, hi, lg) in space.items():
        v = float(np.clip(params[k], lo, hi))
        u.append((math.log(v / lo) / math.log(hi / lo)) if lg else (v - lo) / (hi - lo))
    return np.array(u)


def _from_unit(u, space):
    out = {}
    for (k, (lo, hi, lg)), x in zip(space.items(), np.clip(u, 0, 1)):
        out[k] = float(lo * (hi / lo) ** x if lg else lo + (hi - lo) * x)
    return out


# ------------------------------------------------------------------------------- loop

def render(target, params, n=10, seed=1):
    """n generated events for these settings, cut the way the references are."""
    return cut_like_references(GENERATORS[target](params, n, seed), seed)


def evaluate(target, params, ref, weights=None, n=10, seed=1, listener=None):
    events = render(target, params, n, seed)
    s = ref.score(events, weights)
    s.pop("F")
    s["total"] = s["distance"]
    if listener is not None:
        j = listener.judge(events, ref.listener_good)
        s["listener"] = j
        s["total"] += 2.0 * j["penalty"] + 1.0 * j["fails"]
    return s


class SepCMA:
    """Separable CMA-ES (Ros & Hansen 2008): an evolution strategy that learns how far to step
    in each setting, and shrinks or grows its steps by how consistently it is making progress.
    Works in the unit cube; points outside it are evaluated at the nearest edge plus a penalty."""

    def __init__(self, x0, sigma=0.25, popsize=24, seed=1):
        n = len(x0)
        self.n, self.lam = n, popsize
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff) * (n + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
                       * (n + 2) / 3)
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.asarray(x0, dtype=float).copy()
        self.sigma = sigma
        self.C = np.ones(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.x = self.m + self.sigma * self.z * np.sqrt(self.C)
        return self.x

    def tell(self, fitness):
        order = np.argsort(fitness)[: self.mu]
        y = (self.x[order] - self.m) / self.sigma
        yw = self.w @ y
        self.m = self.m + self.sigma * yw
        D = np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / D
        self.gen += 1
        hs = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) < (1.4 + 2 / (self.n + 1)) * self.chi
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w @ (y ** 2)))
        self.sigma *= math.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chi - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 0.5))


def tune(target, ref, out_dir, rounds=30, population=24, n_events=10, seed=1, start=None,
         weights=None, listener=None, log=print):
    """Search the target's settings for the ones that score closest to the references. Writes
    history.jsonl and best.json in out_dir and returns the top 3 distinct candidates (best first)."""
    os.makedirs(out_dir, exist_ok=True)
    space = SPACES[target]
    seen = {}
    hist_path = os.path.join(out_dir, "history.jsonl")

    def run(u):
        uc = np.clip(u, 0, 1)
        key = tuple(np.round(uc, 4))
        if key not in seen:
            p = _from_unit(uc, space)
            seen[key] = [uc, p, evaluate(target, p, ref, weights, n_events, seed, None)]
        return seen[key]

    def hear(item):  # only the front-runners: the model is slow on a CPU
        uc, p, s = item
        if listener is not None and "listener" not in s:
            j = listener.judge(render(target, p, n_events, seed), ref.listener_good)
            s["listener"] = j
            s["total"] = s["distance"] + 2.0 * j["penalty"] + 1.0 * j["fails"]

    es = SepCMA(_to_unit(start or defaults(target), space), popsize=population, seed=seed)
    t0 = time.time()
    for r in range(rounds):
        X = es.ask()
        items = [run(u) for u in X]
        for i in np.argsort([it[2]["total"] for it in items])[:3]:
            hear(items[i])
        fit = [it[2]["total"] + 2.0 * float(np.sum((u - np.clip(u, 0, 1)) ** 2)) for u, it in zip(X, items)]
        es.tell(np.array(fit))
        best = min(seen.values(), key=lambda it: it[2]["total"])
        s = best[2]
        row = {"round": r + 1, "total": s["total"], "distance": s["distance"], "tell_apart": s["tell_apart"],
               "groups": s["groups"], "params": best[1], "step": es.sigma, "seconds": round(time.time() - t0, 1)}
        if "listener" in s:
            row["listener"] = s["listener"]
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"round {r + 1:2d}/{rounds}: best {s['total']:.3f} (distance {s['distance']:.3f}, "
            f"tell-apart {s['tell_apart']:.2f}, step {es.sigma:.3f}) "
            + " ".join(f"{g} {v:.2f}" for g, v in s["groups"].items())
            + (f"  heard {s['listener']['heard']}" if "listener" in s else ""))
    ranked = sorted(seen.values(), key=lambda it: it[2]["total"])
    for it in ranked[:8]:
        hear(it)
    ranked = sorted(seen.values(), key=lambda it: it[2]["total"])
    top = _distinct(ranked, 3)
    with open(os.path.join(out_dir, "best.json"), "w") as fh:
        json.dump({"target": target, "weights": weights or {g: 1.0 for g in GROUPS},
                   "references": ref.names,
                   "top": [{"params": p, "score": _clean(s)} for _, p, s in top]}, fh, indent=1)
    return [(p, s) for _, p, s in top]


def _distinct(ranked, k, min_gap=0.08):
    """The best k that differ from each other (not three near-copies of the winner)."""
    out = []
    for item in ranked:
        if all(np.sqrt(np.mean((item[0] - o[0]) ** 2)) >= min_gap for o in out):
            out.append(item)
        if len(out) == k:
            break
    return out


def _clean(s):
    return {k: v for k, v in s.items() if k != "F"}


# --------------------------------------------------------------- learning from picks

def reweight(weights, picks, devs, rate=0.3):
    """Shift the group weights toward what gave the generated sounds away in the picks.

    picks   [{"a": item, "b": item, "choice": "a"|"b"|"same"}], items like "c1" or "real"
    devs    {item: {group: distance}} (the real recording is 0 in every group)
    For every decided pair, the groups where the loser sits further from the real recordings
    than the winner get more weight."""
    w = {g: float(weights.get(g, 1.0)) for g in GROUPS}
    zero = {g: 0.0 for g in GROUPS}
    for pk in picks:
        if pk.get("choice") not in ("a", "b"):
            continue
        win, lose = (pk["a"], pk["b"]) if pk["choice"] == "a" else (pk["b"], pk["a"])
        dw, dl = devs.get(win, zero), devs.get(lose, zero)
        delta = np.array([dl[g] - dw[g] for g in GROUPS])
        scale = np.mean(np.abs(delta)) + 1e-9
        for g, d in zip(GROUPS, delta):
            w[g] *= math.exp(rate * d / scale)
    m = np.mean(list(w.values()))
    return {g: float(np.clip(v / m, 0.25, 4.0)) for g, v in w.items()}


def verdict(picks):
    """Win rate of each item over its pairs (a tie counts half), and whether the real recording
    still stands out: done when it wins no more than 60% against the best generated one."""
    games = {}
    for pk in picks:
        a, b, c = pk["a"], pk["b"], pk.get("choice")
        sa = 1.0 if c == "a" else 0.0 if c == "b" else 0.5
        for item, s in ((a, sa), (b, 1 - sa)):
            g = games.setdefault(item, [0.0, 0])
            g[0] += s
            g[1] += 1
    rates = {k: v[0] / v[1] for k, v in games.items() if v[1]}
    vs_real = {}
    for pk in picks:
        if "real" in (pk["a"], pk["b"]):
            other = pk["b"] if pk["a"] == "real" else pk["a"]
            c = pk.get("choice")
            real_won = 1.0 if (c == "a") == (pk["a"] == "real") and c in ("a", "b") else 0.5 if c not in ("a", "b") \
                else 0.0
            g = vs_real.setdefault(other, [0.0, 0])
            g[0] += real_won
            g[1] += 1
    real_vs = {k: v[0] / v[1] for k, v in vs_real.items()}
    best = min(real_vs, key=real_vs.get) if real_vs else None
    return {"rates": rates, "real_beats": real_vs, "best": best,
            "done": bool(best is not None and real_vs[best] <= 0.6)}


# ------------------------------------------------------------------------ listening set

ITEMS = ("c1", "c2", "c3", "real")
LISTEN_LEVEL_DB = -26.0  # every clip is played at the same loudness, so level gives nothing away


def _clip(ev, ref, rng):
    """A generated event with the references' own room tone under and around it (0.3 s either
    side), so silence vs room noise can't give the generated ones away."""
    x = _mono(ev)
    y = np.concatenate([np.zeros(int(0.3 * SR)), x, np.zeros(int(0.3 * SR))])
    if ref.room is not None:
        start = int(rng.integers(0, max(1, len(ref.room) - len(y))))
        room = np.resize(ref.room[start:start + len(y)], len(y))
        y = y + room * sounding_rms(x) * 10 ** (-ref.snr_db / 20)
    return y


def _level(y):
    y = _mono(y) * 10 ** (LISTEN_LEVEL_DB / 20) / (sounding_rms(y) + 1e-12)
    f = min(int(0.03 * SR), len(y) // 4)
    if f:
        ramp = np.linspace(0, 1, f)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y


def make_listening(run_dir, target, top, ref, pairs_each=2, seed=777, round_no=1):
    """Clips and trials for the A/B page: the best 3 and a real recording, every pair twice
    (different clips each time), in random order and sides."""
    from itertools import combinations
    out = os.path.join(run_dir, "listen")
    clips_dir = os.path.join(out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    for fn in os.listdir(clips_dir):
        if fn.endswith(".wav"):
            os.remove(os.path.join(clips_dir, fn))
    rng = np.random.default_rng(seed + round_no)
    need = 3 * pairs_each
    names = iter(f"{int(i):06x}.wav" for i in rng.choice(16 ** 6, 4 * need, replace=False))  # names give nothing away
    files = {}
    for k, (params, _) in zip(ITEMS, top):
        evs = render(target, params, need + 2, seed + round_no)[:need]
        files[k] = []
        for ev in evs:
            fn = next(names)
            write_wav(os.path.join(clips_dir, fn), _level(_clip(ev, ref, rng)))
            files[k].append(fn)
    pick = rng.permutation(len(ref.clips))[:need]
    files["real"] = []
    for j in pick:
        fn = next(names)
        write_wav(os.path.join(clips_dir, fn), _level(ref.clips[j]))
        files["real"].append(fn)
    used = {k: 0 for k in files}
    trials = []
    present = [k for k in ITEMS if files.get(k)]
    for _ in range(pairs_each):
        for a, b in combinations(present, 2):
            if rng.random() < 0.5:
                a, b = b, a
            fa = files[a][used[a] % len(files[a])]
            fb = files[b][used[b] % len(files[b])]
            used[a] += 1
            used[b] += 1
            trials.append({"a": a, "b": b, "a_file": fa, "b_file": fb})
    order = rng.permutation(len(trials))
    trials = [dict(trials[i], id=n + 1) for n, i in enumerate(order)]
    devs = {k: s["groups"] for k, (_, s) in zip(ITEMS, top)}
    devs["real"] = {g: 0.0 for g in GROUPS}
    with open(os.path.join(out, "trials.json"), "w") as fh:
        json.dump({"target": target, "round": round_no, "question": TARGETS[target]["question"],
                   "trials": trials, "devs": devs,
                   "params": {k: p for k, (p, _) in zip(ITEMS, top)}}, fh, indent=1)
    return trials
