"""Rain scenes: rain falling on real surfaces around a listener, one drop at a time.

A scene is a set of *patches* (a roof above you, a window to one side, a lake in
front, a forest canopy overhead). Each second, every patch receives the number and
sizes of drops that real rain at the current rainfall rate would deliver to its area.
Each drop is synthesised individually on the GPU from its own physical properties
(see surfaces.py) and placed in 3D: its distance sets level and air absorption, and
its direction sets the delay and head-shadow at each ear, so it is heard where it lands.

There are far more drops than any machine could render one by one (a lake gets
millions per second), so the loudest ones nearby are synthesised individually, up to
a budget, and every other drop is folded into a *diffuse field*: noise whose spectrum
and level are measured from a Monte Carlo sample of exactly those drops. Near and far
therefore always match in tone, and a bigger budget (a faster GPU) simply means more
of the scene is made of distinct, individual drops.

Rainfall rate and wind wander slowly over the night (rainphys.Wander), and the drop
size spread follows the rate, so heavier spells sound deeper as well as louder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sp_fft
from scipy import signal

from . import gpu
from . import rainphys as P
from .dsp import SR, Reverb, rms
from .layers import Layer, child
from .surfaces import SHORT_L, Modes, make_surface

CTRL = SR // 10  # rain-rate / wind resolution: 0.1 s
UNIT = SR        # drops are generated in fixed 1 s units, so output never depends on chunk size


# --------------------------------------------------------------------------- rendering

def render_events(buf, start, delays, amps, modes: Modes, L: int):
    """Synthesise drops into `buf` (torch tensor, shape (2, N)).

    start  (E,) onset in samples relative to buf[0] (fractional is fine)
    delays (2, E) extra delay per ear in samples (interaural time difference)
    amps   (2, E, M) per-ear amplitude of every mode
    """
    k = modes.n_short
    if 0 < k < modes.f.shape[1] and L > SHORT_L:
        cols = slice(0, k), slice(k, None)
        for c, length in zip(cols, (SHORT_L, L)):
            sub = Modes(modes.f[:, c], modes.tau[:, c], modes.amp[:, c], modes.ph[:, c], modes.chirp[:, c],
                        modes.attack)
            _render(buf, start, delays, amps[:, :, c], sub, length)
    else:
        _render(buf, start, delays, amps, modes, L)


def _render(buf, start, delays, amps, modes: Modes, L: int):
    torch = gpu._torch()
    E, M = modes.f.shape
    if E == 0:
        return
    dev = buf.device
    n = torch.arange(L, device=dev, dtype=torch.float32)
    taper = torch.clamp((L - n) / (0.12 * L), max=1.0)
    nl = n.long()
    bs = max(1, gpu.batch_elements() // (M * L))

    def T(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=torch.float32)

    # The interaural delay is applied as a whole-sample offset per ear (<= 21 us error),
    # so each drop's modes are computed once and shared by both ears.
    d_int = np.round(delays).astype(np.int64)
    for s in range(0, E, bs):
        e = slice(s, s + bs)
        f, tau, ph, ch, att = T(modes.f[e]), T(modes.tau[e]), T(modes.ph[e]), T(modes.chirp[e]), T(modes.attack[e])
        t0 = start[e]
        i0 = np.floor(t0)
        tt = ((n[None, :] - T(t0 - i0)[:, None]) / SR).clamp_min(0)  # time since impact
        x = tt[:, None, :]
        g = torch.exp(-x / tau[:, :, None]) * torch.sin(2 * np.pi * f[:, :, None] * (x + 0.5 * ch[:, :, None] * x * x)
                                                       + ph[:, :, None])
        env = (1 - torch.exp(-tt / att[:, None])) * taper
        base = torch.as_tensor(i0.astype(np.int64), device=dev)
        for ear in (0, 1):
            y = torch.einsum("bm,bml->bl", T(amps[ear][e]), g) * env
            idx = (base + torch.as_tensor(d_int[ear][e], device=dev))[:, None] + nl[None, :]
            buf[ear].index_add_(0, idx.reshape(-1), y.reshape(-1))


# ------------------------------------------------------------------------------ patches

@dataclass
class Patch:
    name: str
    surface: str
    center: tuple
    size: tuple                     # box to scatter drops in; a 0 makes it a plane
    weight: float = 1.0             # share of the individually-rendered drop budget
    individual: bool = True         # False: too far away to pick out drops, diffuse only
    gain_db: float = 0.0
    transmission: tuple | None = None  # (cutoff Hz, order, gain): ceiling, wall or glass in the way
    rate_scale: float = 1.0         # shelter (e.g. under a canopy)
    wind_driven: bool = False       # vertical surfaces only catch wind-blown rain
    fall_height: float | None = None  # short falls (drips) land slower
    fixed_d: tuple | None = None    # drips: diameter range (mm) instead of the rain's spread
    fixed_rate: float | None = None  # drips: drops/s at the reference rainfall rate

    @property
    def area(self):
        sx, sy, sz = self.size
        if sy == 0:
            return sx * sz
        if sx == 0:
            return sy * sz
        return sx * sy

    def positions(self, rng, k):
        c = np.asarray(self.center, float)
        s = np.asarray(self.size, float)
        u = rng.random((k, 3))
        pos = c + (u - 0.5) * s
        axes = np.argsort(-s)[:2]  # the two largest dimensions map to the surface's (u, v)
        return pos, u[:, axes]

    def path_gain(self, f):
        g = 10 ** (self.gain_db / 20)
        if self.transmission:
            fc, order, tg = self.transmission
            g = g * tg / np.sqrt(1 + (f / fc) ** (2 * order))
        return g


# ------------------------------------------------------------------------- environments

def _side(rng):
    return float(rng.choice([-1.0, 1.0]))


def env_roof(rng):
    """In bed in a loft room right under a tiled roof: close, softened patter through
    the roof (tiles, felt and boards: about 6 dB per octave above 1-2 kHz), a window
    to one side, a gutter dripping into a drain outside."""
    h = rng.uniform(2.6, 3.6)
    side = _side(rng)
    return dict(
        indoor=True,
        patches=[
            Patch("roof", "tile", (0, 0.5, h), (10, 9, 0), weight=0.85,
                  transmission=(rng.uniform(1000, 1800), 1, 1.0)),
            Patch("window", "glass", (side * 2.2, 1.0, 0.3), (0, 1.2, 1.3), weight=0.15,
                  wind_driven=True, gain_db=-4),
            Patch("gutter", "water_drip", (side * 3.6, rng.uniform(-1, 3), -1.2), (0.5, 0.5, 0),
                  fixed_d=(3.5, 5.5), fixed_rate=rng.uniform(0.6, 2.0), fall_height=3.0,
                  transmission=(1400, 1, 0.3), gain_db=-3),
            Patch("garden", "ground", (0, 12, -1.5), (40, 25, 0), individual=False,
                  transmission=(900, 2, 0.18)),
        ],
        room=dict(rt60=rng.uniform(0.35, 0.55), predelay_ms=4, damping=0.7, wet=0.22),
    )


def env_tin_roof(rng):
    """In a small cabin or shed under a corrugated metal roof: bright, ringing drumming."""
    h = rng.uniform(2.2, 2.8)
    side = _side(rng)
    drip_surface = "water_drip" if rng.random() < 0.5 else "ground"
    return dict(
        indoor=True,
        patches=[
            Patch("roof", "metal", (0, 0, h), (6, 7, 0), weight=0.9),
            Patch("eaves", drip_surface, (side * 3.3, 0, -1.3), (0.2, 6, 0), fixed_d=(3.0, 5.0),
                  fixed_rate=rng.uniform(3, 8), fall_height=2.5, transmission=(2500, 1, 0.5), gain_db=-4),
            Patch("outside", "ground", (0, 10, -1.5), (40, 30, 0), individual=False,
                  transmission=(2200, 1, 0.4)),
        ],
        room=dict(rt60=rng.uniform(0.45, 0.7), predelay_ms=3, damping=0.6, wet=0.25),
    )


def env_window(rng):
    """Sitting a metre or two from a window: crisp ticks on the glass, drops on the
    sill outside, and the street muffled behind the pane."""
    x = rng.uniform(-0.8, 0.8)
    y = rng.uniform(1.0, 2.2)
    return dict(
        indoor=True,
        patches=[
            Patch("glass", "glass", (x, y, 0.3), (1.4, 0, 1.4), weight=0.75, wind_driven=True),
            Patch("sill", "stone", (x, y + 0.12, -0.45), (1.5, 0.22, 0), weight=0.25,
                  transmission=(3000, 1, 0.35)),
            Patch("street", "ground", (0, 15, -1.5), (50, 30, 0), individual=False,
                  transmission=(1600, 2, 0.15)),
            Patch("rooftops", "tile", (0, 20, 6), (50, 30, 0), individual=False,
                  transmission=(1600, 2, 0.08)),
        ],
        room=dict(rt60=rng.uniform(0.4, 0.6), predelay_ms=5, damping=0.65, wet=0.2),
    )


def env_lake(rng):
    """On the shore of a lake: drops on the water in front, the shore underfoot and
    the treeline behind."""
    return dict(
        indoor=False,
        patches=[
            Patch("lake", "water", (0, 13, -1.0), (30, 22, 0), weight=0.75),
            Patch("shore", "ground", (0, -1, -1.2), (8, 5, 0), weight=0.15),
            Patch("treeline", "leaf", (0, -9, 6), (30, 6, 8), weight=0.1),
            Patch("far_lake", "water", (0, 70, -1.0), (140, 90, 0), individual=False),
        ],
        room=dict(rt60=rng.uniform(0.7, 1.1), predelay_ms=30, damping=0.8, wet=0.08),
    )


def env_forest(rng):
    """Under trees: rain on the canopy high above, heavy drops falling through onto
    the leaves around you and the forest floor."""
    return dict(
        indoor=False,
        patches=[
            Patch("canopy", "leaf", (0, 0, 9), (30, 30, 10), weight=0.4),
            Patch("understorey", "leaf", (0, 0, 1.0), (8, 8, 1.6), weight=0.35, rate_scale=0.4),
            Patch("throughfall", "leaf", (0, 0, 1.0), (10, 10, 1.6), fixed_d=(2.5, 5.0),
                  fixed_rate=rng.uniform(10, 25), fall_height=6.0),
            Patch("floor", "ground", (0, 0, -1.5), (14, 14, 0), weight=0.25, rate_scale=0.35),
            Patch("floor_drips", "ground", (0, 0, -1.5), (14, 14, 0), fixed_d=(3.0, 5.5),
                  fixed_rate=rng.uniform(3, 8), fall_height=8.0),
            Patch("far_canopy", "leaf", (0, 0, 9), (80, 80, 10), individual=False),
        ],
        room=dict(rt60=rng.uniform(1.4, 2.0), predelay_ms=25, damping=0.85, wet=0.3),
    )


def env_tent(rng):
    """Lying in a small ridge tent: the fly sheet sloping down on both sides close to
    your ears, the odd fat drop from the trees, wet grass and woods outside."""
    # Each side of the fly is a sloped sheet from the ridge (~0.9 m up) down to the
    # ground beside you; the box spans that slope, and rain lands on its plan area.
    w = rng.uniform(0.6, 0.8)
    return dict(
        indoor=True,
        patches=[
            Patch("fly_left", "fabric", (-w / 2, 0, 0.35), (w, 2.4, 0.8), weight=0.425),
            Patch("fly_right", "fabric", (w / 2, 0, 0.35), (w, 2.4, 0.8), weight=0.425),
            Patch("tree_drips", "fabric", (0, 0, 0.35), (2 * w, 2.4, 0.8), fixed_d=(3.5, 5.5),
                  fixed_rate=rng.uniform(0.3, 1.5), fall_height=5.0),
            Patch("grass", "ground", (0, 0, -0.25), (10, 10, 0), weight=0.15,
                  transmission=(3000, 1, 0.6)),
            Patch("woods", "leaf", (0, 0, 8), (60, 60, 8), individual=False,
                  transmission=(3500, 1, 0.6)),
        ],
        room=dict(rt60=0.15, predelay_ms=1, damping=0.5, wet=0.04),
    )


ENVIRONMENTS = {
    "roof": dict(build=env_roof, title="Rain on the Roof", wind=(0.5, 3.0)),
    "tin_roof": dict(build=env_tin_roof, title="Rain on a Tin Roof", wind=(0.5, 3.0)),
    "window": dict(build=env_window, title="Rain on the Window", wind=(2.0, 5.0), bed_db=-4.0),
    "lake": dict(build=env_lake, title="Rain on a Lake", wind=(0.5, 3.0), bed_db=0.0),
    "forest": dict(build=env_forest, title="Rain in the Forest", wind=(0.5, 2.5), bed_db=0.0),
    "tent": dict(build=env_tent, title="Rain on a Tent", wind=(0.5, 2.5)),
}


# --------------------------------------------------------------------------- diffuse

class _FIR:
    """Streaming FIR (overlap-add) on stereo numpy blocks."""

    def __init__(self, h):
        self.h = h
        self.tail = np.zeros((len(h) - 1, 2))
        self._spec = {}

    def __call__(self, x):
        n = len(x)
        size = sp_fft.next_fast_len(n + len(self.h) - 1)
        if size not in self._spec:
            self._spec = {size: sp_fft.rfft(self.h, size)[:, None]}
        y = sp_fft.irfft(sp_fft.rfft(x, size, axis=0) * self._spec[size], size, axis=0)[: n + len(self.tail)]
        y[: len(self.tail)] += self.tail
        self.tail = y[n:].copy()
        return y[:n]


# ----------------------------------------------------------------------------- scene

class RainScene(Layer):
    """Rain falling on an environment, heard from a listener's position.

    environment  roof, tin_roof, window, lake, forest, tent
    rain_mm_h    average rainfall rate (light 1-2.5, moderate 2.5-8, heavy 8+)
    wind_ms      average wind (m/s); matters most for windows
    variability  how much the rain wanders between lighter and heavier spells
    budget       individually-synthesised drops per second (default depends on device)
    diffuse_db   adjust the far/small-drop field relative to its physical level
    bed_db       ceiling for the diffuse field relative to the individual drops, so the
                 distinct drops always lead and the background never turns into static
    """

    pannable = False

    def __init__(self, rng, environment="roof", rain_mm_h=4.0, wind_ms=None, variability=0.3,
                 budget=None, diffuse_db=0.0, bed_db=None, calibrate=True):
        if environment not in ENVIRONMENTS:
            raise ValueError(f"unknown environment {environment!r}; choose from {', '.join(ENVIRONMENTS)}")
        torch = gpu._torch()
        self.torch = torch
        self.dev = gpu.device()
        spec = ENVIRONMENTS[environment]
        bed_db = spec.get("bed_db", -2.0) if bed_db is None else bed_db
        self.params = dict(environment=environment, rain_mm_h=rain_mm_h, wind_ms=wind_ms,
                           variability=variability, budget=budget, diffuse_db=diffuse_db, bed_db=bed_db)
        env_rng = child(rng)
        env = spec["build"](env_rng)
        self.patches: list[Patch] = env["patches"]
        self.rate_ref = float(rain_mm_h)
        wind_ms = float(wind_ms if wind_ms is not None else env_rng.uniform(*spec["wind"]))
        self.wind_ref = wind_ms
        self.budget = int(budget or gpu.default_budget())
        self.rain = P.Wander(child(rng), rain_mm_h, variability, timescale_s=240, swell=variability,
                             lo=0.25, hi=3.5)
        self.wind = P.Wander(child(rng), wind_ms, 0.4, timescale_s=20, swell=0.2, lo=0.1, hi=3.0)
        self.ev_rng = child(rng)
        self.noise_rng = child(rng)
        # Surfaces see the patch as (u, v) = its two largest dimensions, as Patch.positions maps them.
        self.surfaces = [make_surface(p.surface, child(rng), tuple(sorted(p.size, reverse=True)[:2]))
                         for p in self.patches]
        weights = sum(p.weight for p in self.patches if p.individual and p.fixed_rate is None)
        self.share = [p.weight / weights * self.budget if (p.individual and p.fixed_rate is None) else 0.0
                      for p in self.patches]
        self.Lmax = max(s.L for s in self.surfaces)
        self.room = None
        if env.get("room"):
            r = env["room"]
            self.room = Reverb(child(rng), rt60=r["rt60"], predelay_ms=r["predelay_ms"], damping=r["damping"])
            self.room_wet = r["wet"]

        self.pending = [[] for _ in self.patches]
        self.gen_until = 0  # samples generated so far (multiple of UNIT)
        self.ctrl_rate = np.zeros(0)
        self.ctrl_start = 0  # control-block index of ctrl_rate[0]
        self.pos = 0
        self.tail = torch.zeros(2, self.Lmax + 256, device=self.dev)

        self.diffuse = self._diffuse_filter(child(rng), 1.0)
        self.diffuse_gain = 10 ** (diffuse_db / 20)
        self.gain = 1.0
        if calibrate:
            # Measure the individual drops on a twin (so this scene's own stream is untouched),
            # then hold the diffuse bed at least `bed_db` under them.
            twin = RainScene(np.random.default_rng(int(rng.integers(2**31))), calibrate=False, **self.params)
            twin.diffuse, twin.room = None, None
            ind = rms(twin(SR * 4))
            diff = 0.0
            if self.diffuse is not None:
                phys = float(np.sqrt(np.sum(self.diffuse.h ** 2)))
                self.diffuse_gain = min(self.diffuse_gain, 10 ** (bed_db / 20) * ind / max(phys, 1e-12))
                diff = phys * self.diffuse_gain
            self.gain = 1.0 / max(np.sqrt(ind ** 2 + diff ** 2), 1e-9)

    # -- drop statistics ------------------------------------------------------------
    def _wind_factor(self, wind):
        return wind / 5.0  # horizontal / vertical speed of a typical drop

    def _dmin(self, i, rate, wind):
        p = self.patches[i]
        area = p.area * p.rate_scale * (self._wind_factor(wind) if p.wind_driven else 1.0)
        return P.dmin_for_budget(rate, area, self.share[i]) if self.share[i] else P.D_MAX

    def _expected(self, i, rate, wind, d_lo, d_hi):
        """Drops per second on patch i with diameters in [d_lo, d_hi)."""
        p = self.patches[i]
        if p.fixed_rate is not None:
            return p.fixed_rate * (rate / self.rate_ref) ** 0.8
        grid, tail = P.cumulative_flux_above(rate)
        n = np.interp(d_lo, grid, tail) - np.interp(d_hi, grid, tail)
        area = p.area * p.rate_scale * (self._wind_factor(wind) if p.wind_driven else 1.0)
        return max(n, 0.0) * area

    def _drops(self, rng, i, k, rate, d_lo, d_hi):
        p = self.patches[i]
        if p.fixed_d:
            d = rng.uniform(*p.fixed_d, k)
        else:
            d = P.sample_diameters(rng, k, rate, d_lo, d_hi)
        v = P.fall_velocity(d, p.fall_height)
        pos, uv = p.positions(rng, k)
        return d, v, pos, uv

    def _voice(self, rng, i, d, v, pos, uv):
        """Modes and per-ear amplitudes/delays for drops on patch i."""
        p = self.patches[i]
        m = self.surfaces[i].voice(rng, d, v, uv)
        r, dl, dr, th_l, th_r = P.ear_geometry(pos)
        common = m.amp * P.air_absorption(m.f, r[:, None]) / r[:, None] * p.path_gain(m.f)
        amps = np.stack([common * P.head_shadow(m.f, th_l[:, None]), common * P.head_shadow(m.f, th_r[:, None])])
        return m, np.stack([dl, dr]), amps

    # -- generation in fixed 1 s units ------------------------------------------------
    def _generate_unit(self):
        u0 = self.gen_until
        nctrl = UNIT // CTRL
        rates = self.rain.blocks(nctrl)
        winds = self.wind.blocks(nctrl)
        self.ctrl_rate = np.concatenate([self.ctrl_rate, rates])
        rng = self.ev_rng
        for i, p in enumerate(self.patches):
            if not p.individual:
                continue
            times, ds, vs, poss, uvs = [], [], [], [], []
            for b in range(nctrl):
                rate, wind = rates[b], winds[b]
                if p.fixed_rate is not None:
                    lo, hi = 0.0, 0.0
                    lam = self._expected(i, rate, wind, 0, 0)
                else:
                    lo, hi = self._dmin(i, rate, wind), P.D_MAX
                    lam = self._expected(i, rate, wind, lo, hi)
                k = rng.poisson(lam * CTRL / SR)
                if k == 0:
                    continue
                d, v, pos, uv = self._drops(rng, i, k, rate, lo, hi)
                times.append(u0 + b * CTRL + rng.random(k) * CTRL)
                ds.append(d), vs.append(v), poss.append(pos), uvs.append(uv)
            if times:
                d, v, pos, uv = (np.concatenate(a) for a in (ds, vs, poss, uvs))
                m, delays, amps = self._voice(rng, i, d, v, pos, uv)
                self.pending[i].append((np.concatenate(times), m, delays, amps))
        self.gen_until += UNIT

    def _take(self, i, end):
        """Pop pending drops on patch i that start before `end`."""
        keep, out = [], []
        for t, m, dl, am in self.pending[i]:
            sel = t < end
            if sel.all():
                out.append((t, m, dl, am))
            elif sel.any():
                out.append((t[sel], _sub(m, sel), dl[:, sel], am[:, sel]))
                keep.append((t[~sel], _sub(m, ~sel), dl[:, ~sel], am[:, ~sel]))
            else:
                keep.append((t, m, dl, am))
        self.pending[i] = keep
        if not out:
            return None
        t = np.concatenate([o[0] for o in out])
        m = Modes(*(np.concatenate([getattr(o[1], k) for o in out]) for k in ("f", "tau", "amp", "ph", "chirp")),
                  attack=np.concatenate([o[1].attack for o in out]), n_short=out[0][1].n_short)
        return t, m, np.concatenate([o[2] for o in out], axis=1), np.concatenate([o[3] for o in out], axis=1)

    # -- diffuse field ------------------------------------------------------------------
    def _diffuse_filter(self, rng, level):
        """Measure the spectrum and level of every drop we don't render individually."""
        torch = self.torch
        seconds = 2.0
        n = int(SR * seconds)
        total = np.zeros(4097)
        f = None
        for i, p in enumerate(self.patches):
            if p.fixed_rate is not None:
                continue
            if p.individual:
                lo, hi = P.D_MIN, self._dmin(i, self.rate_ref, self.wind_ref)
            else:
                lo, hi = P.D_MIN, P.D_MAX
            lam = self._expected(i, self.rate_ref, self.wind_ref, lo, hi)
            if lam <= 0 or hi <= lo:
                continue
            k = 1200
            d, v, pos, uv = self._drops(rng, i, k, self.rate_ref, lo, hi)
            m, delays, amps = self._voice(rng, i, d, v, pos, uv)
            buf = torch.zeros(2, n + self.Lmax + 256, device=self.dev)
            render_events(buf, rng.random(k) * n, delays, amps, m, self.surfaces[i].L)
            mono = buf.mean(0).cpu().numpy()[: n].astype(np.float64)
            f, psd = signal.welch(mono, SR, nperseg=8192)
            total += psd * (lam * seconds / k)
        if f is None or total.sum() <= 0:
            return None
        # Smooth on a log-frequency grid, then build a filter that turns unit white
        # noise into exactly this spectrum (white noise has one-sided PSD 2/SR).
        grid = np.geomspace(20, SR / 2 - 1, 160)
        band = np.array([total[(f >= g / 1.12) & (f < g * 1.12)].mean() if np.any((f >= g / 1.12) & (f < g * 1.12))
                         else np.interp(g, f, total) for g in grid])
        mag = np.sqrt(band * SR / 2) * level
        freqs = np.concatenate([[0], grid, [SR / 2]])
        gains = np.concatenate([[mag[0]], mag, [mag[-1] * 0.1]])
        return _FIR(signal.firwin2(2047, freqs, gains, fs=SR))

    # -- main ----------------------------------------------------------------------------
    def __call__(self, n):
        torch = self.torch
        start, end = self.pos, self.pos + n
        # One control block of look-ahead, so the rain-rate curve for the last samples of
        # this chunk is interpolated the same way whatever the chunk size.
        while self.gen_until < end + CTRL:
            self._generate_unit()
        buf = torch.zeros(2, n + self.Lmax + 256, device=self.dev)
        buf[:, : self.tail.shape[1]] += self.tail
        for i, p in enumerate(self.patches):
            got = self._take(i, end)
            if got is None:
                continue
            t, m, delays, amps = got
            render_events(buf, t - start, delays, amps, m, self.surfaces[i].L)
        self.tail = buf[:, n:].clone()
        out = buf[:, :n].T.cpu().numpy().astype(np.float64)

        if self.diffuse is not None:
            c0 = start // CTRL - self.ctrl_start
            idx = np.clip((np.arange(n) + start % CTRL) / CTRL + c0 - 0.5, 0, len(self.ctrl_rate) - 1)
            rate = np.interp(idx, np.arange(len(self.ctrl_rate)), self.ctrl_rate)
            env = np.sqrt(rate / self.rate_ref)
            out += self.diffuse(self.noise_rng.standard_normal((n, 2))) * (env * self.diffuse_gain)[:, None]
        # Forget control values we no longer need.
        drop = max(0, end // CTRL - 2 - self.ctrl_start)
        if drop:
            self.ctrl_rate = self.ctrl_rate[drop:]
            self.ctrl_start += drop
        if self.room is not None:
            out = out + self.room_wet * self.room(out)
        self.pos = end
        return out * self.gain


def _sub(m: Modes, sel):
    return Modes(m.f[sel], m.tau[sel], m.amp[sel], m.ph[sel], m.chirp[sel], m.attack[sel], m.n_short)
