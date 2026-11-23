"""Desk sounds for study scenes: pages turning and a pencil writing, close by on the desk.

Both are built from what happens physically, not from recordings:

  Friction  Anything sliding on paper (a pencil tip, a fingertip, a page settling on the
            others) is heard as the paper's surface scraped at the speed of the slide. The
            surface is a fixed random profile along the paper (fibres and tooth, a few to a
            few hundred micrometres apart) that is read out at the sliding speed, so a faster
            stroke sounds brighter and louder, just as a real pencil does. A hard graphite
            tip catches every grain of the paper; a soft fingertip smooths over them.
  Body      The paper pad and the desk under it add a little low, woody resonance.
  Pages     A turn is a rustle: a soft finger slide at the corner, then a swell of dry
            crackles (a few crisp ones over many tiny ones, like crumpled paper) and a
            "fwoosh" of air as the page flexes over, rising and falling with its speed, and it
            settles onto the other pages on a cushion of air with a soft "fff", never a slap. Now and then a hand smooths the
            new page flat, or a few pages are flipped through.
  Writing   Handwriting is strokes of 50-250 ms, each with a smooth rise and fall in speed,
            run together into words. The pencil lifts between words (and for the odd dot or
            cross) and lands again with a faint tick. Writing comes in spells with pauses
            for thinking between them.

Each sound sits on the desk in front of the listener, a little to one side (a small time
and level difference between the ears); the shared room reverb is added by the engine.
The layers are silent between events, and every event is scaled to about unit RMS while it
sounds (like thunder), so `level_db` sets how loud a page turn or a written word is.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import SR, PeakLimiter
from .layers import Layer, child

GRID_UM = 0.5  # the paper texture is sampled every 0.5 micrometres


def _texture(rng, n, k_lo, k_hi, spikes=0.0, spike_um=5.0):
    """A paper surface profile along a line (unit RMS), sampled every GRID_UM.

    k_lo, k_hi   band of surface detail, in cycles per micrometre (1/size of the features)
    spikes       how much of it is sharp tooth peaks (what makes graphite sound scratchy)"""
    k = np.fft.rfftfreq(n, d=GRID_UM)
    shape = (k / k_lo) ** 2 / (1 + (k / k_lo) ** 2) / np.sqrt(1 + (k / k_hi) ** 4)
    g = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n)
    g /= g.std()
    if spikes:
        imp = np.zeros(n)
        idx = rng.integers(0, n, int(n * GRID_UM / 15.0))  # a tooth peak every ~15 um
        size = np.minimum(rng.lognormal(0, 0.4, len(idx)), 3.0)  # no freak peaks: they would click
        np.add.at(imp, idx, size * rng.choice([-1.0, 1.0], len(idx)))
        w = np.hanning(max(3, int(spike_um / GRID_UM)))
        sp = signal.fftconvolve(imp, w, mode="same")
        sp -= signal.fftconvolve(sp, np.ones(401) / 401, mode="same")  # no slow drift
        g = np.sqrt(1 - spikes) * g + np.sqrt(spikes) * sp / sp.std()
    return g / g.std()


def _bell(n):
    """A smooth rise and fall over n samples (the speed profile of a natural movement)."""
    return np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2


def _smooth_noise(rng, n, hz):
    """Slow random wobble around 0 (unit RMS), changing at about `hz`."""
    k = max(2, int(n * hz / SR) + 2)
    return np.interp(np.arange(n), np.linspace(0, n, k), rng.standard_normal(k))


def _modes(rng, n, f, tau, amp, attack=0.0):
    """A sum of damped sinusoids (a tiny resonant click or knock), n samples long."""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for fi, ti, ai in zip(f, tau, amp):
        y += ai * np.sin(2 * np.pi * fi * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / ti)
    if attack:
        y *= 1 - np.exp(-t / attack)
    return y


def _place(x, azimuth_deg):
    """Mono to stereo for a sound on the desk in front, `azimuth_deg` to the right (negative:
    left): the far ear hears it a fraction of a millisecond later, a little quieter and duller."""
    th = np.radians(abs(azimuth_deg))
    lag = int(round(0.0875 / 343.0 * (th + np.sin(th)) * SR))  # Woodworth's head model
    shade = 0.6 * np.sin(th)
    b, a = signal.butter(1, 2500, fs=SR)
    far = (1 - shade) * x + shade * signal.lfilter(b, a, x)
    far = np.concatenate([np.zeros(lag), far])[: len(x)] * (1 - 0.2 * np.sin(th))
    near = x * (1 + 0.2 * np.sin(th))
    return np.stack([far, near] if azimuth_deg >= 0 else [near, far], axis=1)


def _unit(ev):
    """Scale an event so it is unit RMS over the part that is clearly sounding."""
    win = SR // 100
    m = len(ev) // win * win
    ms = np.mean(ev[:m] ** 2, axis=1).reshape(-1, win).mean(axis=1) if m else np.array([np.mean(ev ** 2)])
    loud = ms[ms > ms.max() * 0.01]  # within 20 dB of the loudest 10 ms
    return ev / np.sqrt(max(loud.mean(), 1e-20))


class _Events(Layer):
    """A layer that is silent except for events, each made whole by `_event` and played
    `_gap` samples after the previous one ends. The same events whatever the chunk size."""

    continuous = False

    def __init__(self, rng, first_s):
        self.rng = rng
        self.pending = np.zeros((0, 2))
        self.until_next = int(SR * first_s)

    def _event(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def _gap(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, n):
        out = np.zeros((n, 2))
        used = min(n, len(self.pending))
        out[:used] = self.pending[:used]
        self.pending = self.pending[used:]
        cursor = 0
        while cursor + self.until_next < n:
            cursor += self.until_next
            ev = self._event()
            take = min(len(ev), n - cursor)
            out[cursor:cursor + take] += ev[:take]
            rest = ev[take:]
            if len(rest):
                if len(rest) > len(self.pending):
                    self.pending = np.pad(self.pending, ((0, len(rest) - len(self.pending)), (0, 0)))
                self.pending[:len(rest)] += rest
            self.until_next = len(ev) + self._gap()
        self.until_next -= n - cursor
        return out


class _Desk:
    """What the desk sounds share: paper textures, the body resonance, and friction."""

    def __init__(self, rng, body_lo=180.0, body_hi=1500.0):
        self.tex_rng = child(rng)
        # Each texture is made the first time it is used (a page turn never needs graphite).
        self._tex_rngs = {name: child(rng) for name in ("graphite", "paper", "finger")}
        self._tex = {}
        # Paper and wood are heavily damped: every mode dies within a few cycles (Q 2-6). Lightly
        # damped modes ring like metal (v2 had Q up to 50 and sounded like "metal scraping").
        k = 8
        f = np.exp(rng.uniform(np.log(body_lo), np.log(body_hi), k))
        q = rng.uniform(2.0, 6.0, k)
        body = _modes(rng, int(0.06 * SR), f, q / (np.pi * f), rng.normal(0, 1, k), attack=2e-4)
        self.body = body / np.sqrt(np.sum(body ** 2))

    _TEXTURES = {  # cycles per micrometre (low, high), sharp tooth share, tooth width (um)
        "graphite": (1 / 100.0, 1 / 10.0, 0.6, 5.0),
        "paper": (1 / 300.0, 1 / 15.0, 0.25, 8.0),
        "finger": (1 / 400.0, 1 / 25.0, 0.0, 5.0),
    }

    def _texture(self, name):
        if name not in self._tex:
            lo, hi, spikes, um = self._TEXTURES[name]
            n = 1 << 20  # about half a metre of paper
            self._tex[name] = _texture(self._tex_rngs[name], n, lo, hi, spikes=spikes, spike_um=um)
        return self._tex[name]

    tex_graphite = property(lambda self: self._texture("graphite"))
    tex_paper = property(lambda self: self._texture("paper"))
    tex_finger = property(lambda self: self._texture("finger"))

    def friction(self, tex, v, force):
        """Sliding at speed v (m/s, per sample) with contact `force` (0..1): the texture read
        out along the path, louder for a faster slide."""
        travel = np.sum(v) / SR * 1e6 / GRID_UM + 2
        start = self.tex_rng.uniform(0, max(1.0, len(tex) - travel - 1))
        pos = start + np.cumsum(v) / SR * 1e6 / GRID_UM
        x = np.interp(pos, np.arange(len(tex)), tex)
        return x * force * np.sqrt(np.maximum(v, 0) / 0.05)

    def with_body(self, x, mix):
        return x + mix * signal.fftconvolve(x, self.body)[: len(x)]


# ------------------------------------------------------------------------------- pages

PAPERS = {
    # rustle: crackles per second at the height of the swing (stiffer paper crackles more);
    # colour_lo/hi: how high the crackles reach (Hz); settle_lp: how soft the landing is
    "book": dict(rustle=450.0, colour_lo=1800.0, colour_hi=5000.0, air=0.15, settle_lp=3000.0),
    "notebook": dict(rustle=600.0, colour_lo=2500.0, colour_hi=7000.0, air=0.12, settle_lp=4250.0),
}

# Everything else about the sound of a turn. `PageTurns(shape=...)` overrides any of these (or
# of the paper's), which is how `sleepgen tune` searches for settings that sound like real pages.
PAGE_SHAPE = dict(
    rustle_level=0.12,  # crackles while the page swings over
    crackle_ms=0.9,     # how long one crackle lasts (each between 1/3 and 5/3 of this)
    crackle_hp=600.0,   # crackles have nothing below this
    sizes_alpha=2.5,    # crackle sizes are Pareto: smaller = more crisp ones standing out
    speed_exp=0.75,     # a slow page crackles more softly (size goes with speed to this power)
    swing=0.5,          # seconds the page takes to swing over (each turn 0.7-1.3x this)
    air_lo=250.0,       # the "fwoosh" of air moved by the page: its band
    air_hi=3000.0,
    fff_level=0.03,     # the page settling onto a cushion of air
    fff_attack=0.0225,
    fff_decay=0.075,
    land_rustle=0.08,   # a last few crackles as it settles
    flap_level=0.0,     # a soft low "thup" as the page lands (paper onto paper)
    flap_hz=500.0,
    finger_level=0.02,  # a fingertip sliding over the corner, and the hand smoothing the page
    smooth_chance=0.3,
    body_mix=0.1,       # the book under it: a little low resonance
    limit_db=18.0,      # no crackle more than this over the turn's level
)


class PageTurns(_Events):
    """Someone reading at the desk turns a page now and then.

    A page turn is a rustle, not a knock: the page flexes all the way over, so the sound is a
    soft swell of tiny crinkles and paper brushing paper that rises and falls with the page's
    speed, and the page settles onto a cushion of air at the end rather than slapping down.

    paper      book (soft, heavy pages) or notebook (crisper)
    mean_gap   seconds between turns, on average (reading speed)
    flurry     chance that a turn becomes a quick flip through a few pages
    azimuth    where the book lies, degrees to the right of straight ahead
    shape      overrides for PAGE_SHAPE / PAPERS settings (see `sleepgen tune`)
    """

    def __init__(self, rng, paper="book", mean_gap=45.0, flurry=0.1, azimuth=None, shape=None):
        if paper not in PAPERS:
            raise ValueError(f"unknown paper {paper!r}; choose from {', '.join(PAPERS)}")
        unknown = set(shape or {}) - set(PAGE_SHAPE) - set(PAPERS[paper])
        if unknown:
            raise ValueError(f"unknown page-turn settings: {', '.join(sorted(unknown))}")
        super().__init__(child(rng), first_s=rng.uniform(4, min(30.0, mean_gap)))
        self.p = p = {**PAGE_SHAPE, **PAPERS[paper], **{k: float(v) for k, v in (shape or {}).items()}}
        self.mean_gap = float(mean_gap)
        self.flurry = float(flurry)
        self.azimuth = float(rng.uniform(-20, 20) if azimuth is None else azimuth)
        self.desk = _Desk(rng, 120.0, 900.0)
        # Each crackle is a tiny dry burst, a fraction of a millisecond to a few milliseconds of
        # noise, rolled off gently (first order, non-ringing) above a corner that varies from
        # crackle to crackle: paper crackles are dry, never tuned, and never a bare click.
        lo = max(p["colour_lo"], p["crackle_hp"] * 1.2)
        hi = min(max(p["colour_hi"], lo * 1.3), 0.45 * SR)
        self.size_ms = float(np.mean(self._sizes(np.random.default_rng(0), 200000) ** 2))
        self.kernels = []
        krng = child(rng)
        c = p["crackle_ms"] * 1e-3
        L = int(max(0.006, 6 * c) * SR)
        tt = np.arange(L) / SR
        for fc in np.geomspace(lo, hi, 16):
            sos = signal.butter(1, [p["crackle_hp"], fc], "bandpass", fs=SR, output="sos")
            burst = krng.standard_normal(L) * (1 - np.exp(-tt / 1.5e-4)) * np.exp(-tt / krng.uniform(c / 3, c * 5 / 3))
            k = signal.sosfilt(sos, burst)
            self.kernels.append(k / np.sqrt(np.sum(k ** 2)))

    def _gap(self):
        # Reading is fairly regular: a gamma spread around the mean, never back to back.
        return int(SR * max(6.0, self.rng.gamma(3.0, self.mean_gap / 3.0)))

    def _sizes(self, rng, k):
        # Pareto, capped: no freak cracks
        return np.minimum((1 - rng.random(k)) ** (-1 / self.p["sizes_alpha"]), 2.5)

    def _rustle(self, rng, speed):
        """Crackles of the flexing page, as many as its speed (0..1 per sample) calls for. Like
        crumpled paper, a few crisp ones stand out over many tiny ones (heavy-tailed sizes)."""
        n = len(speed)
        lam = self.p["rustle"] * np.sqrt(speed) / SR
        hits = rng.random(n) < lam
        idx = np.flatnonzero(hits)
        # A slowly moving page crackles less often and more softly than a fast one; mostly more
        # softly, so a slow start isn't a few lone ticks.
        amp = self._sizes(rng, len(idx)) * speed[idx] ** self.p["speed_exp"] * rng.choice([-1.0, 1.0], len(idx))
        out = np.zeros(n + len(self.kernels[0]) - 1)
        band = rng.integers(0, len(self.kernels), len(idx))
        for b, k in enumerate(self.kernels):
            sel = band == b
            if not sel.any():
                continue
            imp = np.zeros(n)
            imp[idx[sel]] = amp[sel]
            out += signal.fftconvolve(imp, k)[: len(out)]
        # Unit RMS at full speed, whatever the paper.
        return out[:n] / np.sqrt(self.p["rustle"] / SR * self.size_ms)

    def _turn(self):
        rng, p, d = self.rng, self.p, self.desk
        y = np.zeros(int(2.5 * SR))

        def add(at, x):
            i = int(at * SR)
            x = x[: max(0, len(y) - i)]
            y[i:i + len(x)] += x

        # 1. A finger slides softly over the corner and lifts it.
        n = int(rng.uniform(0.08, 0.16) * SR)
        v = rng.uniform(0.04, 0.09) * _bell(n) * (1 + 0.15 * _smooth_noise(rng, n, 40))
        add(0.0, p["finger_level"] * d.friction(d.tex_finger, v, np.sqrt(_bell(n))))
        t = n / SR * 0.6
        # 2. The page bends up and swings over: a swell of crackles and a "fwoosh" of air,
        # rising and falling with its speed.
        swing = min(p["swing"] * rng.uniform(0.7, 1.3), 1.6)
        n = int(swing * SR)
        u = np.arange(n) / n
        speed = np.sin(np.pi * u ** rng.uniform(0.7, 1.0)) ** 2 * (1 + 0.25 * _smooth_noise(rng, n, 12)).clip(0.3)
        add(t, p["rustle_level"] * self._rustle(rng, speed))
        air_lo = min(p["air_lo"], 0.4 * SR)
        sos = signal.butter(1, [air_lo, min(max(p["air_hi"], air_lo * 1.5), 0.45 * SR)], "bandpass", fs=SR,
                            output="sos")
        add(t, p["air"] * signal.sosfilt(sos, rng.standard_normal(n)) * speed)
        # 3. It settles onto the other pages on a cushion of air: a soft "fff", not a slap.
        t += swing * rng.uniform(0.6, 0.75)  # it lands while still moving
        n = int(0.4 * SR)
        tt = np.arange(n) / SR
        env = (1 - np.exp(-tt / (p["fff_attack"] * rng.uniform(0.67, 1.33)))) \
            * np.exp(-tt / (p["fff_decay"] * rng.uniform(0.67, 1.33)))
        lp = min(max(p["settle_lp"] * rng.uniform(0.85, 1.15), 700.0), 0.45 * SR)
        sos = signal.butter(2, [500, lp], "bandpass", fs=SR, output="sos")
        fff = signal.sosfilt(sos, rng.standard_normal(n)) * env
        add(t, p["fff_level"] * fff / (np.sqrt(np.mean(fff[: int(0.15 * SR)] ** 2)) + 1e-12))
        add(t, p["land_rustle"] * self._rustle(rng, np.exp(-tt / 0.04) * 0.5))  # a last few crackles
        if p["flap_level"] > 0:  # the page meeting the others: a soft, low, damped "thup"
            m = int(0.12 * SR)
            tm = np.arange(m) / SR
            sos = signal.butter(2, [80, min(p["flap_hz"], 0.4 * SR)], "bandpass", fs=SR, output="sos")
            thup = signal.sosfilt(sos, rng.standard_normal(m)) * (1 - np.exp(-tm / 0.004)) * np.exp(-tm / 0.025)
            add(t, p["flap_level"] * thup / (np.sqrt(np.mean(thup[: int(0.05 * SR)] ** 2)) + 1e-12))
        end = t + 0.3
        # 4. Sometimes a hand smooths the new page flat.
        if rng.random() < p["smooth_chance"]:
            t2 = t + rng.uniform(0.3, 0.7)
            m = int(rng.uniform(0.3, 0.6) * SR)
            v = rng.uniform(0.04, 0.08) * _bell(m) * (1 + 0.1 * _smooth_noise(rng, m, 15))
            add(t2, p["finger_level"] * 1.5 * d.friction(d.tex_finger, v, _bell(m) ** 0.5))
            end = t2 + m / SR
        return y[: int(min(end + 0.1, 2.5) * SR)]

    def turn(self) -> np.ndarray:
        """One page turn on its own, mono, dry (no placement), unit RMS while it sounds. Used by
        `sleepgen tune` to compare against recordings of real page turns."""
        y = self.desk.with_body(self._turn(), self.p["body_mix"])
        y = _unit(y[:, None])[:, 0]
        return self._limit(y[:, None])[:, 0]

    def _limit(self, y):
        # The odd big crackle is eased down (smoothly, over a millisecond or two) so it never
        # pokes out as a tick.
        lim = PeakLimiter(10 ** (self.p["limit_db"] / 20), window_ms=1.0, channels=y.shape[1])
        return lim(np.concatenate([y, np.zeros((2 * lim.W, y.shape[1]))]))[2 * lim.W:]

    def _event(self):
        rng = self.rng
        turns = [self._turn()]
        if rng.random() < self.flurry:  # flipping through to find a page
            turns += [self._turn() for _ in range(int(rng.integers(1, 4)))]
        starts = np.cumsum([0] + [int(rng.uniform(0.15, 0.5) * SR) + len(x) for x in turns[:-1]])
        y = np.zeros(starts[-1] + len(turns[-1]))
        for s, x in zip(starts, turns):
            y[s:s + len(x)] += x
        y = self.desk.with_body(y, self.p["body_mix"])  # paper on paper is mostly airborne
        y = _unit(_place(y, self.azimuth + rng.uniform(-3, 3)))
        return self._limit(y)


# ------------------------------------------------------------------------------ pencil

SURFACES = {
    # a pad of paper softens and darkens the scratch; one sheet on a hard desk is crisper
    "pad": dict(lp=7000.0, body=0.45),
    "desk": dict(lp=11000.0, body=0.25),
}


class Pencil(_Events):
    """Someone writing with a pencil at the desk, in spells with thinking pauses between.

    activity   share of the time spent writing (0.1 .. 0.6)
    surface    pad (softer, darker) or desk (one sheet on a hard desk: crisper)
    pressure   how hard they press (0.5 light .. 1.0 firm): firmer is a little brighter
    azimuth    where the hand is, degrees to the right of straight ahead
    """

    def __init__(self, rng, activity=0.3, surface="pad", pressure=0.75, azimuth=None):
        if surface not in SURFACES:
            raise ValueError(f"unknown surface {surface!r}; choose from {', '.join(SURFACES)}")
        super().__init__(child(rng), first_s=rng.uniform(3, 20))
        self.activity = float(np.clip(activity, 0.05, 0.9))
        self.surface = SURFACES[surface]
        self.pressure = float(np.clip(pressure, 0.3, 1.2))
        self.azimuth = float(rng.uniform(10, 30) if azimuth is None else azimuth)
        self.desk = _Desk(rng)
        self.spell_mean = 15.0  # seconds of writing at a time, on average
        b, a = signal.butter(2, 400, "highpass", fs=SR)
        self.hp = (b, a)
        self.lp = signal.butter(1, self.surface["lp"], fs=SR)

    def _gap(self):
        mean = self.spell_mean * (1 - self.activity) / self.activity
        return int(SR * max(3.0, self.rng.gamma(2.0, mean / 2.0)))

    def _word(self, rng):
        """Speed (m/s) and contact (0..1) for one word, plus where the pencil lands."""
        n_strokes = int(rng.integers(2, 8) * rng.uniform(1.5, 2.5))
        dur = np.clip(rng.lognormal(np.log(0.11), 0.3, n_strokes), 0.05, 0.25)
        peak = np.clip(rng.lognormal(np.log(0.045), 0.35, n_strokes), 0.015, 0.12)
        starts, t, lands = [], 0.0, [0.0]
        for i in range(n_strokes):
            starts.append(t)
            step = dur[i] * rng.uniform(0.55, 0.85)  # strokes overlap: the pencil keeps moving
            if i < n_strokes - 1 and rng.random() < 0.15:  # lifted for a dot or a cross
                step = dur[i] + rng.uniform(0.04, 0.12)
                lands.append(t + step)
            t += step
        n = int((starts[-1] + dur[-1]) * SR) + 1
        v = np.zeros(n)
        contact = np.zeros(n)
        for s, d_, pk in zip(starts, dur, peak):
            i, m = int(s * SR), int(d_ * SR)
            v[i:i + m] += pk * _bell(m)[: n - i]
            contact[i:i + m] = 1.0
        ramp = int(0.006 * SR)  # the tip presses in and lets go over a few milliseconds
        contact = np.convolve(contact, np.ones(ramp) / ramp, mode="same")
        return v, contact, lands

    def _event(self):
        rng, d = self.rng, self.desk
        spell = max(2.0, rng.gamma(2.0, self.spell_mean / 2.0))
        parts, ticks, t = [], [], 0.0
        while t < spell:
            v, contact, lands = self._word(rng)
            i0 = int(t * SR)
            parts.append((i0, v, contact))
            ticks += [i0 + int(x * SR) for x in lands]
            t += len(v) / SR + rng.uniform(0.15, 0.45)  # lift and move on to the next word
            if rng.random() < 0.12:
                t += rng.uniform(0.6, 1.4)  # a new line, or a moment's thought
        n = int(t * SR) + int(0.2 * SR)
        v, contact = np.zeros(n), np.zeros(n)
        for i0, vv, cc in parts:
            v[i0:i0 + len(vv)] += vv
            contact[i0:i0 + len(cc)] = np.maximum(contact[i0:i0 + len(cc)], cc)
        force = contact * self.pressure * (1 + 0.15 * _smooth_noise(rng, n, 6))
        x = d.friction(d.tex_graphite, v * (0.85 + 0.3 * self.pressure), force)
        for i in ticks:  # the tip landing: a faint, tiny tick
            k = 3
            f = np.exp(rng.uniform(np.log(1500), np.log(5000), k))
            click = _modes(rng, int(0.006 * SR), f, rng.uniform(0.0003, 0.001, k), rng.normal(0, 1, k) / np.sqrt(k))
            seg = click * rng.uniform(0.1, 0.25) * self.pressure
            x[i:i + len(seg)] += seg[: n - i]
        x = signal.lfilter(*self.hp, x)
        x = signal.lfilter(*self.lp, x)
        x = d.with_body(x, self.surface["body"])
        return _unit(_place(x, self.azimuth + rng.uniform(-4, 4)))
