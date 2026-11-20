"""Desk sounds for study scenes: pages turning and a pencil writing, close by on the desk.

Both are built from what happens physically, not from recordings:

  Friction  Anything sliding on paper (a pencil tip, a fingertip, a page settling on the
            others) is heard as the paper's surface scraped at the speed of the slide. The
            surface is a fixed random profile along the paper (fibres and tooth, a few to a
            few hundred micrometres apart) that is read out at the sliding speed, so a faster
            stroke sounds brighter and louder, just as a real pencil does. A hard graphite
            tip catches every grain of the paper; a soft fingertip smooths over them.
  Body      The paper pad and the desk under it add a little low, woody resonance.
  Pages     A turn is a soft finger slide at the corner, a few small crinkles as the page
            bends, a breath of air as it swings over, and a soft flap as it lands on the
            other pages, which it then slides on for a moment. Now and then a hand smooths
            the new page flat, or a few pages are flipped through.
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

from .dsp import SR
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
        n = 1 << 20  # about half a metre of paper
        self.tex_graphite = _texture(child(rng), n, 1 / 100.0, 1 / 10.0, spikes=0.6)
        self.tex_paper = _texture(child(rng), n, 1 / 300.0, 1 / 15.0, spikes=0.25, spike_um=8.0)
        self.tex_finger = _texture(child(rng), n, 1 / 400.0, 1 / 25.0)
        k = 8
        f = np.exp(rng.uniform(np.log(body_lo), np.log(body_hi), k))
        body = _modes(rng, int(0.12 * SR), f, rng.uniform(0.006, 0.03, k), rng.normal(0, 1, k), attack=2e-4)
        self.body = body / np.sqrt(np.sum(body ** 2))

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
    # flap brightness, the stack's low knock, how many crinkles a page makes as it bends
    "book": dict(flap_lp=(2000, 3500), thump_hz=(110, 210), crinkles=1.5, air=0.18),
    "notebook": dict(flap_lp=(3500, 6000), thump_hz=(160, 320), crinkles=3.0, air=0.22),
}


class PageTurns(_Events):
    """Someone reading at the desk turns a page now and then.

    paper      book (soft, heavy pages) or notebook (crisper)
    mean_gap   seconds between turns, on average (reading speed)
    flurry     chance that a turn becomes a quick flip through a few pages
    azimuth    where the book lies, degrees to the right of straight ahead
    """

    def __init__(self, rng, paper="book", mean_gap=45.0, flurry=0.1, azimuth=None):
        if paper not in PAPERS:
            raise ValueError(f"unknown paper {paper!r}; choose from {', '.join(PAPERS)}")
        super().__init__(child(rng), first_s=rng.uniform(4, min(30.0, mean_gap)))
        self.p = PAPERS[paper]
        self.mean_gap = float(mean_gap)
        self.flurry = float(flurry)
        self.azimuth = float(rng.uniform(-20, 20) if azimuth is None else azimuth)
        self.desk = _Desk(rng, 120.0, 900.0)

    def _gap(self):
        # Reading is fairly regular: a gamma spread around the mean, never back to back.
        return int(SR * max(6.0, self.rng.gamma(3.0, self.mean_gap / 3.0)))

    def _turn(self):
        rng, p, d = self.rng, self.p, self.desk
        y = np.zeros(int(2.2 * SR))

        def add(at, x):
            i = int(at * SR)
            x = x[: len(y) - i]
            y[i:i + len(x)] += x

        # 1. A finger slides over the corner and lifts it.
        n = int(rng.uniform(0.07, 0.16) * SR)
        v = rng.uniform(0.06, 0.15) * _bell(n) * (1 + 0.15 * _smooth_noise(rng, n, 40))
        add(0.0, 0.35 * d.friction(d.tex_finger, v, np.sqrt(_bell(n))))
        t = n / SR * 0.6
        # 2. The page bends: a few small crinkles.
        bend = rng.uniform(0.12, 0.3)
        for _ in range(rng.poisson(p["crinkles"])):
            k = 3
            f = np.exp(rng.uniform(np.log(1800), np.log(7000), k))
            q = rng.uniform(2.0, 4.0, k)
            click = _modes(rng, int(0.01 * SR), f, q / (np.pi * f), rng.normal(0, 1, k) / np.sqrt(k), 3e-5)
            add(t + rng.uniform(0, bend), click * rng.lognormal(np.log(0.15), 0.4))
        # 3. It swings over: a breath of air, and paper brushing paper as it leaves the stack.
        t += bend * 0.5
        swing = rng.uniform(0.25, 0.45)
        n = int(swing * SR)
        sos = signal.butter(2, [250, 2000], "bandpass", fs=SR, output="sos")
        add(t, p["air"] * signal.sosfilt(sos, rng.standard_normal(n)) * _bell(n) ** 2 * 2.5)
        m = int(rng.uniform(0.04, 0.08) * SR)
        add(t, 0.25 * d.friction(d.tex_paper, rng.uniform(0.08, 0.14) * _bell(m), _bell(m)))
        # 4. It lands on the other pages: a soft flap with a low knock from the book,
        t += swing * rng.uniform(0.85, 1.0)
        n = int(0.25 * SR)
        tt = np.arange(n) / SR
        env = (1 - np.exp(-tt / rng.uniform(0.002, 0.004))) * np.exp(-tt / rng.uniform(0.025, 0.06))
        sos = signal.butter(2, [200, rng.uniform(*p["flap_lp"])], "bandpass", fs=SR, output="sos")
        flap = signal.sosfilt(sos, rng.standard_normal(n)) * env
        f0 = rng.uniform(*p["thump_hz"])
        knock = _modes(rng, n, [f0], [rng.uniform(0.012, 0.025)], [1.0], attack=0.003)
        add(t, flap / (np.sqrt(np.mean(flap ** 2)) + 1e-12) * 0.25 + 0.06 * knock)
        # 5. and slides on them for a moment as it settles.
        m = int(rng.uniform(0.06, 0.15) * SR)
        add(t + 0.01, 0.2 * d.friction(d.tex_paper, rng.uniform(0.02, 0.05) * _bell(m), np.sqrt(_bell(m))))
        end = t + 0.3
        # 6. Sometimes a hand smooths the new page flat.
        if rng.random() < 0.3:
            t2 = t + rng.uniform(0.25, 0.6)
            m = int(rng.uniform(0.25, 0.5) * SR)
            v = rng.uniform(0.05, 0.1) * _bell(m) * (1 + 0.1 * _smooth_noise(rng, m, 15))
            add(t2, 0.25 * d.friction(d.tex_finger, v, _bell(m) ** 0.5))
            end = t2 + m / SR
        return y[: int(min(end + 0.1, 2.2) * SR)]

    def _event(self):
        rng = self.rng
        turns = [self._turn()]
        if rng.random() < self.flurry:  # flipping through to find a page
            turns += [self._turn() for _ in range(int(rng.integers(1, 4)))]
        starts = np.cumsum([0] + [int(rng.uniform(0.15, 0.5) * SR) + len(x) for x in turns[:-1]])
        y = np.zeros(starts[-1] + len(turns[-1]))
        for s, x in zip(starts, turns):
            y[s:s + len(x)] += x
        y = self.desk.with_body(y, 0.25)
        return _unit(_place(y, self.azimuth + rng.uniform(-3, 3)))


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
