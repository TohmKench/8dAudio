"""What a single raindrop sounds like when it lands on different things.

Every impact is described as a handful of decaying resonances ("modes"): a few very
short ones for the click of contact, plus longer ones for whatever the surface does
next (a bubble ringing in water, a tile or a leaf vibrating, a metal sheet pinging).
Each drop gets its own frequencies, decay times and strengths, worked out from its
size, speed and where on the surface it lands, so no two drops are identical.

Plates and membranes (metal roofing, window glass, tent fabric) use real vibration
patterns: each sheet has a fixed set of modes, and how strongly a drop excites each
one depends on where it hits (the mode shape at that point), just like a real sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rainphys as P


@dataclass
class Modes:
    f: np.ndarray       # (E, M) Hz
    tau: np.ndarray     # (E, M) s, decay time constant
    amp: np.ndarray     # (E, M)
    ph: np.ndarray      # (E, M) starting phase
    chirp: np.ndarray   # (E, M) 1/s, pitch rises as f(t) = f (1 + chirp t)
    attack: np.ndarray  # (E,) s
    n_short: int = 0    # the first n_short modes die within a few ms (contact clicks)

    @staticmethod
    def stack(parts):
        # Short-lived click modes first, so they can be rendered with a short window.
        n_short = parts[0].f.shape[1] if parts[0].n_short else 0
        return Modes(*(np.concatenate([getattr(p, k) for p in parts], axis=1)
                       for k in ("f", "tau", "amp", "ph", "chirp")), attack=parts[0].attack, n_short=n_short)


def _loguni(rng, lo, hi, shape):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape))


def _q_tau(f, q, cap=None):
    tau = q / (np.pi * f)
    return np.minimum(tau, cap) if cap else tau


def _attack(d, v):
    return np.clip(0.35 * (d * 1e-3) / np.maximum(v, 0.5), 2e-5, 3e-4)


def clicks(rng, d, v, k, f_lo, f_hi, tau_lo, tau_hi, gain):
    """The contact transient: a short burst made of very short-lived random modes."""
    E = len(d)
    f = _loguni(rng, f_lo, f_hi, (E, k))
    tau = rng.uniform(tau_lo, tau_hi, (E, k))
    amp = rng.normal(0, 1, (E, k)) / np.sqrt(k)
    amp *= gain * P.impact_strength(d, v)[:, None] * P.force_weight(f, d, v)
    return Modes(f, tau, amp, rng.uniform(0, 2 * np.pi, (E, k)), np.zeros((E, k)), _attack(d, v), n_short=k)


def rings(rng, d, v, k, f_lo, f_hi, q_lo, q_hi, gain, cap=None):
    E = len(d)
    f = _loguni(rng, f_lo, f_hi, (E, k))
    tau = _q_tau(f, rng.uniform(q_lo, q_hi, (E, k)), cap)
    amp = rng.normal(0, 1, (E, k)) / np.sqrt(k)
    amp *= gain * P.impact_strength(d, v)[:, None] * P.force_weight(f, d, v)
    return Modes(f, tau, amp, rng.uniform(0, 2 * np.pi, (E, k)), np.zeros((E, k)), _attack(d, v))


SHORT_L = 256  # samples rendered for the contact clicks (they are gone within ~5 ms)
FLOOR_DB = 40.0  # every mode is at least this far down by the end of its window


class Surface:
    L = 1024  # samples rendered per drop (long enough for the slowest mode to die away)

    def __init__(self, rng, size=(1.0, 1.0)):
        self.size = size

    def modes(self, rng, d, v, uv) -> Modes:  # pragma: no cover - interface
        raise NotImplementedError

    def voice(self, rng, d, v, uv) -> Modes:
        """modes(), with every decay short enough to have faded out (-40 dB) inside the
        window each drop is rendered in, so no ring is ever cut off audibly."""
        m = self.modes(rng, d, v, uv)
        k = FLOOR_DB / 8.686  # time constants to fall FLOOR_DB
        tau = np.minimum(m.tau, self.L / (k * P.SR))
        if m.n_short:
            tau[:, : m.n_short] = np.minimum(tau[:, : m.n_short], SHORT_L / (k * P.SR))
        m.tau = tau
        return m


class Water(Surface):
    """Lake, pond or puddle. Small drops (0.8-1.1 mm) trap a tiny bubble that rings
    at 13-18 kHz (the "fizz" of rain on water); big drops sometimes trap a larger,
    lower bubble. Medium drops rarely make a bubble at all."""

    L = 640

    def __init__(self, rng, size=(1.0, 1.0), drip=False):
        super().__init__(rng, size)
        self.drip = drip
        if drip:
            self.L = 2400  # a drip into a puddle rings longer ("plonk")

    def modes(self, rng, d, v, uv):
        E = len(d)
        c = clicks(rng, d, v, 5, 900, 12000, 0.05e-3, 0.22e-3, 0.5)
        if self.drip:
            p = np.full(E, 0.85)
            a = rng.uniform(0.9, 3.5, E)
            q = rng.uniform(1.6, 2.6, E)
        else:
            p = np.where(d < 0.8, 0.0, np.where(d < 1.15, 0.8, np.where(d < 2.2, 0.05, 0.4)))
            a = np.where(d < 1.15, rng.uniform(0.19, 0.26, E), rng.uniform(0.4, 2.2, E))
            q = rng.uniform(0.9, 1.3, E)
        has = rng.random(E) < p
        f0 = P.minnaert_hz(a)
        damp = P.bubble_damping(f0)
        tau = q / damp
        chirp = rng.uniform(0.04, 0.1, E) * damp  # the slight upward 'blip' of a real bubble
        amp = has * 1.1 * P.impact_strength(d, v) * rng.uniform(0.5, 1.0, E) * (a / 1.0) ** 0.25
        b = Modes(f0[:, None], tau[:, None], amp[:, None], rng.uniform(0, 2 * np.pi, (E, 1)),
                  chirp[:, None], c.attack)
        return Modes.stack([c, b])


class Leaf(Surface):
    """A leaf: a light, floppy plate. A soft tap with a short, woody resonance."""

    L = 1408

    def modes(self, rng, d, v, uv):
        return Modes.stack([clicks(rng, d, v, 4, 1000, 10000, 0.08e-3, 0.35e-3, 0.6),
                            rings(rng, d, v, 3, 450, 3000, 3, 8, 0.55)])


class Ground(Surface):
    """Soil, grass or leaf litter: dull, short, low."""

    L = 768

    def modes(self, rng, d, v, uv):
        return Modes.stack([clicks(rng, d, v, 5, 250, 3500, 0.15e-3, 0.7e-3, 0.7),
                            rings(rng, d, v, 1, 200, 800, 1.5, 3, 0.25)])


class Tile(Surface):
    """Clay or concrete roof tiles (or a stone sill): hard and heavy. A crisp tick,
    a brief stony ring, and a faint low knock through the roof structure."""

    L = 2304

    def modes(self, rng, d, v, uv):
        return Modes.stack([clicks(rng, d, v, 4, 1500, 14000, 0.05e-3, 0.25e-3, 0.9),
                            rings(rng, d, v, 3, 600, 5500, 10, 25, 0.35),
                            rings(rng, d, v, 1, 90, 220, 1.5, 2.0, 0.3)])


PLATE_MATERIALS = {
    # f11: lowest mode; band: modes kept; q: quality factor range; cap: longest decay (s)
    "metal": dict(membrane=False, f11=(18, 35), band=(250, 7000), q=(60, 180), cap=0.035,
                  n_modes=48, m=12, gain=0.55, clicks=(3, 2000, 14000, 0.04e-3, 0.15e-3, 0.6),
                  boom=(2, 45, 110, 4, 8, 0.35), panel=(0.9, 2.4), L=7680),
    "glass": dict(membrane=False, f11=(14, 24), band=(250, 4500), q=(12, 35), cap=0.006,
                  n_modes=90, m=12, gain=0.08, clicks=(5, 2500, 14000, 0.04e-3, 0.2e-3, 1.0),
                  boom=None, panel=(1.0, 1.25), L=2880),
    "fabric": dict(membrane=True, f11=(30, 50), band=(70, 1100), q=(4, 12), cap=0.018,
                   n_modes=30, m=8, gain=0.6, clicks=(4, 700, 5000, 0.12e-3, 0.5e-3, 0.8),
                   boom=None, panel=(1.1, 2.3), L=3840),
}


class Plate(Surface):
    """Metal roofing, a window pane or tent fabric, split into panels.

    Each panel gets its own fixed set of vibration modes (so each sheet has a voice).
    A drop landing at (u, v) on a panel excites mode (m, n) in proportion to
    sin(m pi u) sin(n pi v): hits near the middle ring the low modes, hits near
    an edge bring out the higher ones.
    """

    def __init__(self, rng, size=(1.0, 1.0), material="metal"):
        super().__init__(rng, size)
        spec = PLATE_MATERIALS[material]
        self.spec = spec
        self.L = spec["L"]
        pa, pb = spec["panel"]
        # A grid of sheets/panes: e.g. a 6 x 7 m metal roof is ~20 separate sheets,
        # each with its own slightly different set of modes.
        self.nu = max(1, int(round(size[0] / pb)))
        self.nv = max(1, int(round(size[1] / pa)))
        self.n_panels = self.nu * self.nv
        self.panels = []
        for _ in range(self.n_panels):
            a = pa * rng.uniform(0.85, 1.15)
            b = pb * rng.uniform(0.85, 1.15)
            f11 = rng.uniform(*spec["f11"])
            mm, nn = np.meshgrid(np.arange(1, 80), np.arange(1, 80), indexing="ij")
            g = (mm / a) ** 2 + (nn / b) ** 2
            g11 = (1 / a) ** 2 + (1 / b) ** 2
            f = f11 * (np.sqrt(g / g11) if spec["membrane"] else g / g11)
            lo, hi = spec["band"]
            ok = np.flatnonzero((f.ravel() >= lo) & (f.ravel() <= hi))
            if len(ok) > spec["n_modes"]:
                # Keep a spread across the band (log-spaced picks), not just the lowest ones.
                order = ok[np.argsort(f.ravel()[ok])]
                picks = np.unique(np.round(np.geomspace(1, len(order), spec["n_modes"])).astype(int) - 1)
                ok = order[picks]
            self.panels.append(dict(f=f.ravel()[ok], m=mm.ravel()[ok], n=nn.ravel()[ok],
                                    q=rng.uniform(*spec["q"], len(ok))))
        k = max(len(p["f"]) for p in self.panels)
        self.k = k
        self.pf = np.ones((self.n_panels, k))
        self.pm = np.ones((self.n_panels, k))
        self.pn = np.ones((self.n_panels, k))
        self.pq = np.ones((self.n_panels, k))
        self.valid = np.zeros((self.n_panels, k), dtype=bool)
        for i, p in enumerate(self.panels):
            j = len(p["f"])
            self.pf[i, :j], self.pm[i, :j], self.pn[i, :j], self.pq[i, :j] = p["f"], p["m"], p["n"], p["q"]
            self.valid[i, :j] = True

    def modes(self, rng, d, v, uv):
        spec = self.spec
        E = len(d)
        u = np.clip(uv[:, 0], 0, 0.9999) * self.nu
        w_ = np.clip(uv[:, 1], 0, 0.9999) * self.nv
        iu, iv = u.astype(int), w_.astype(int)
        panel = iu * self.nv + iv
        pu, pv = u - iu, w_ - iv
        f = self.pf[panel]
        shape = np.sin(self.pm[panel] * np.pi * pu[:, None]) * np.sin(self.pn[panel] * np.pi * pv[:, None])
        w = P.force_weight(f, d, v)
        score = np.abs(shape) * w * rng.uniform(0.6, 1.4, f.shape) * self.valid[panel]
        m = min(spec["m"], self.k)
        top = np.argpartition(-score, m - 1, axis=1)[:, :m]
        rows = np.arange(E)[:, None]
        f_sel = f[rows, top]
        tau = _q_tau(f_sel, self.pq[panel][rows, top], spec["cap"])
        amp = (shape * w)[rows, top] * self.valid[panel][rows, top]
        amp *= spec["gain"] * P.impact_strength(d, v)[:, None] / np.sqrt(m) * 2.0
        parts = [clicks(rng, d, v, *spec["clicks"]),
                 Modes(f_sel, tau, amp, rng.uniform(0, 2 * np.pi, (E, m)), np.zeros((E, m)), _attack(d, v))]
        if spec["boom"]:
            parts.append(rings(rng, d, v, *spec["boom"]))
        return Modes.stack(parts)


def make_surface(kind: str, rng, size=(1.0, 1.0)) -> Surface:
    if kind == "water":
        return Water(rng, size)
    if kind == "water_drip":
        return Water(rng, size, drip=True)
    if kind == "leaf":
        return Leaf(rng, size)
    if kind == "ground":
        return Ground(rng, size)
    if kind in ("tile", "stone"):
        return Tile(rng, size)
    if kind in PLATE_MATERIALS:
        return Plate(rng, size, kind)
    raise ValueError(f"unknown surface {kind!r}")
