"""Fire: a wood fire heard from where you are, built from what a fire physically does.

A burning fire makes five kinds of sound, and each is modelled separately:

  Flame roar   Turbulent combustion: a soft, low, broadband rush. Flames "puff" at a
               rate set by the size of the fire (about 1.5 / sqrt(diameter) Hz, the
               pool-fire puffing law), so a bonfire breathes slowly and a stove quickly,
               and they flicker at random on top of that.
  Crackles     Pockets of sap and water flash to steam and burst, and drying wood
               cracks. Each crackle is an individual event placed in 3D inside the fire
               bed: a dry, sharp snap, then a dense sputter of tiny grains that dies away
               within 10-30 ms (a short "krsh"). Nothing rings with a pitch, which would
               sound like bubbling liquid. They come in clusters (one crack sets off a
               few more within milliseconds), their sizes follow a heavy-tailed law
               (gently evened out for sleep), and busy and quiet spells come and go.
  Ember ticks  A light sprinkle of tiny ticks from embers and charring surfaces, each
               one distinct, following the flames.
  Steam hiss   Escaping gas and steam: a faint breathy hiss, stronger for damp wood, with
               the odd soft "singing" jet that glides in pitch and fades.
  Settling     Every few minutes the burnt logs shift: a soft cluster of ticks and a low
               thump, and the fire flares up a little afterwards.

The crackles lead. The roar, ticks and hiss form a soft, quiet bed under them (a real
fire close by is mostly heard as its cracks and pops, not as a rushing noise). A gentle
look-ahead limiter keeps every peak within `crest_db` of the fire's recent level, so the
pops stay clear but never jolt a sleeper awake.

Fire size and intensity wander slowly over the night, so it burns brighter and lower
in long, natural spells. Wood type and dampness change the balance (pine pops and
hisses, oak is steadier and deeper), and each setting (campfire, fireplace, wood stove,
bonfire) has its own distance, room and walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import gpu
from . import rainphys as P
from .dsp import SR, PeakLimiter, Reverb
from .layers import Layer, child
from .scene import _FIR, _sub, render_events
from .surfaces import SHORT_L, Modes

CTRL = SR // 100  # control resolution: 10 ms (flicker reaches ~15 Hz)
UNIT = SR         # events are generated in fixed 1 s units, so output never depends on chunk size
NC = UNIT // CTRL

# --------------------------------------------------------------------------- settings

WOODS = {
    # crackle rate (per s at full intensity), cluster size, size-law exponent (lower = more
    # big pops), brightness, sap whistles per minute at full dampness
    "softwood": dict(rate=(6.0, 10.0), cascade=2.0, alpha=2.1, bright=1.1, whistle=0.8, hiss=1.0),
    "mixed": dict(rate=(4.0, 7.0), cascade=1.5, alpha=2.3, bright=1.0, whistle=0.4, hiss=0.8),
    "hardwood": dict(rate=(2.2, 4.5), cascade=1.1, alpha=2.6, bright=0.9, whistle=0.15, hiss=0.6),
}


@dataclass
class FirePlace:
    title: str
    center: tuple          # metres, relative to the head (x right, y forward, z up)
    size: tuple            # fire bed box (the flames and embers that make the sound)
    diameter: float        # fire base diameter (m): sets the puffing rate
    roar_hz: float         # where the roar's spectrum peaks
    coherence: float       # 1 = small point source, 0 = wide (both ears hear different parts)
    mix_db: dict           # bed: roar / fizz / hiss power (dB, roar = 0); crackle: crackle power over the bed (dB)
    rate_scale: float = 1.0
    transmission: tuple | None = None  # (cutoff Hz, order, gain): stove walls / glass
    room: dict | None = None
    ticks: bool = False    # a cast iron stove ticks as it expands


def _side(rng):
    return float(rng.choice([-1.0, 1.0]))


def env_campfire(rng):
    """Lying beside a campfire on open ground, a couple of metres from the flames."""
    return FirePlace(
        title="Crackling Campfire",
        center=(_side(rng) * rng.uniform(0.0, 0.6), rng.uniform(1.6, 2.4), -0.6),
        size=(0.7, 0.7, 0.45), diameter=rng.uniform(0.5, 0.8), roar_hz=rng.uniform(170, 250),
        coherence=0.45, mix_db=dict(roar=0.0, fizz=-11.0, hiss=-20.0, crackle=1.0),
        room=dict(rt60=0.3, predelay_ms=12, damping=0.6, wet=0.05),  # trees and ground nearby
    )


def env_fireplace(rng):
    """An open fireplace across a living room: brick hearth, chimney draw, soft room."""
    return FirePlace(
        title="Crackling Fireplace",
        center=(_side(rng) * rng.uniform(0.4, 1.4), rng.uniform(2.5, 3.8), -0.45),
        size=(0.6, 0.35, 0.35), diameter=rng.uniform(0.35, 0.55), roar_hz=rng.uniform(130, 200),
        coherence=0.65, mix_db=dict(roar=0.0, fizz=-12.0, hiss=-21.0, crackle=1.0),
        transmission=(rng.uniform(5000, 8000), 1, 1.0),  # the hearth opening shades the top end
        room=dict(rt60=rng.uniform(0.45, 0.7), predelay_ms=8, damping=0.65, wet=0.22),
    )


def env_wood_stove(rng):
    """A cast iron wood stove with a glass door: muffled crackles, a steady draw, ticking iron."""
    return FirePlace(
        title="Wood Stove Fire",
        center=(_side(rng) * rng.uniform(0.6, 1.5), rng.uniform(2.0, 3.0), -0.4),
        size=(0.35, 0.25, 0.25), diameter=rng.uniform(0.25, 0.35), roar_hz=rng.uniform(110, 170),
        coherence=0.75, mix_db=dict(roar=0.0, fizz=-15.0, hiss=-24.0, crackle=1.0),
        rate_scale=0.8, transmission=(rng.uniform(1300, 2200), 1, 1.0),
        room=dict(rt60=rng.uniform(0.4, 0.6), predelay_ms=6, damping=0.7, wet=0.2), ticks=True,
    )


def env_bonfire(rng):
    """A big bonfire at night, a few metres off: deep, slow roar and busy crackling."""
    return FirePlace(
        title="Bonfire at Night",
        center=(_side(rng) * rng.uniform(0.0, 1.0), rng.uniform(4.5, 7.0), -0.2),
        size=(2.0, 2.0, 1.4), diameter=rng.uniform(1.5, 2.5), roar_hz=rng.uniform(70, 120),
        coherence=0.35, mix_db=dict(roar=0.0, fizz=-10.0, hiss=-19.0, crackle=1.0),
        rate_scale=2.5, room=dict(rt60=0.5, predelay_ms=20, damping=0.6, wet=0.08),
    )


FIRE_ENVIRONMENTS = {
    "campfire": dict(build=env_campfire, title="Crackling Campfire", name="Campfire", outdoor=True),
    "fireplace": dict(build=env_fireplace, title="Crackling Fireplace", name="Fireplace", outdoor=False),
    "wood_stove": dict(build=env_wood_stove, title="Wood Stove Fire", name="Wood Stove", outdoor=False),
    "bonfire": dict(build=env_bonfire, title="Bonfire at Night", name="Bonfire", outdoor=True),
}


# ---------------------------------------------------------------------------- voices

def _loguni(rng, lo, hi, shape):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape))


def _clamp(m: Modes, L: int) -> Modes:
    """Every mode fades by at least 40 dB inside its render window (no cut-off rings)."""
    k = 40.0 / 8.686
    m.tau = np.minimum(m.tau, L / (k * SR))
    if m.n_short:
        m.tau[:, : m.n_short] = np.minimum(m.tau[:, : m.n_short], SHORT_L / (k * SR))
    return m


def _stack(parts, attack):
    n_short = parts[0].f.shape[1]
    return Modes(*(np.concatenate([getattr(p, k) for p in parts], axis=1) for k in ("f", "tau", "amp", "ph", "chirp")),
                 attack=attack, n_short=n_short)


def _modes(rng, f, tau, amp):
    E, K = f.shape
    return Modes(f, tau, amp, rng.uniform(0, 2 * np.pi, (E, K)), np.zeros((E, K)), np.zeros(E))


POP_L = 1024


def crack_colour(rng, a, bright):
    """Each crack has its own colour: the centre of its band (Hz). Bigger cracks a little lower."""
    return _loguni(rng, 1800, 6500, len(a)) * np.clip(a, 0.2, 6.0) ** -0.1 * bright


def pop_modes(rng, a, bright, fc=None):
    """The snap that starts a crackle: a cluster of very short modes across the treble
    plus a few broad (low-Q) resonances around the crack's colour `fc` that give it a dry,
    woody bite. `a` is the size (1 = a typical crack).

    Nothing in it rings with a clear pitch: a pitched ring around 1 kHz, repeated in a
    sputter, is how bubbling liquid sounds (it made v2's crackles sound like wet globs)."""
    E = len(a)
    if fc is None:
        fc = crack_colour(rng, a, bright)
    k = 6
    f = _loguni(rng, 1500, 13000, (E, k)) * (fc[:, None] / 3500) ** 0.3
    tau = rng.uniform(0.06e-3, 0.3e-3, (E, k))
    amp = rng.normal(0, 0.5, (E, k)) / np.sqrt(k) * a[:, None]
    clicks = _modes(rng, np.minimum(f, 16000), tau, amp)
    kb = 4
    fr = fc[:, None] * _loguni(rng, 0.7, 1.5, (E, kb))
    q = rng.uniform(2.0, 4.5, (E, kb))
    ramp = rng.choice([-1.0, 1.0], (E, kb)) * rng.uniform(0.5, 1.0, (E, kb)) * 0.6 * a[:, None]
    bite = _modes(rng, np.minimum(fr, 16000), q / (np.pi * fr), ramp)
    attack = rng.uniform(1.5e-5, 6e-5, E)
    return _clamp(_stack([clicks, bite], attack), POP_L)


def grain_modes(rng, a, fc):
    """One grain of a crack's sputter: a tiny, heavily damped tick in the crack's colour.
    Dozens of them in a few milliseconds make the noisy "krsh" of a crackle."""
    E = len(a)
    k = 3
    f = fc[:, None] * _loguni(rng, 0.55, 1.8, (E, k))
    q = rng.uniform(1.2, 2.5, (E, k))
    amp = rng.normal(0, 1, (E, k)) / np.sqrt(k) * a[:, None]
    m = _modes(rng, np.minimum(f, 16000), q / (np.pi * f), amp)
    m.attack = rng.uniform(1e-5, 4e-5, E)
    m.n_short = k
    return _clamp(m, SHORT_L)


def fizz_modes(rng, a):
    """An ember tick: a tiny, very bright click."""
    E = len(a)
    f = _loguni(rng, 2200, 14000, (E, 3))
    tau = rng.uniform(0.03e-3, 0.15e-3, (E, 3))
    amp = rng.normal(0, 1, (E, 3)) / np.sqrt(3) * a[:, None]
    m = _modes(rng, f, tau, amp)
    m.attack = rng.uniform(1e-5, 4e-5, E)
    m.n_short = 3
    return _clamp(m, SHORT_L)


THUMP_L = 4096


def thump_modes(rng, a):
    """Logs settling: a soft, low knock."""
    E = len(a)
    f = _loguni(rng, 45, 120, (E, 2))
    q = rng.uniform(1.0, 2.5, (E, 2))
    amp = rng.normal(0, 1, (E, 2)) * a[:, None]
    m = _modes(rng, f, q / (np.pi * f), amp)
    m.attack = rng.uniform(2e-3, 6e-3, E)  # soft, padded onset
    return _clamp(m, THUMP_L)


def tick_modes(rng, a):
    """Cast iron ticking as it heats and cools: a small, bright metallic ping."""
    E = len(a)
    f = _loguni(rng, 1500, 6000, (E, 3))
    q = rng.uniform(25, 70, (E, 3))
    amp = rng.normal(0, 1, (E, 3)) / np.sqrt(3) * a[:, None]
    m = _modes(rng, f, q / (np.pi * f), amp)
    m.attack = rng.uniform(5e-5, 2e-4, E)
    return _clamp(m, THUMP_L)


class _Queue:
    """Events waiting to be rendered (times are absolute sample positions)."""

    def __init__(self, L):
        self.L = L
        self.items = []

    def push(self, t, m, delays, amps):
        if len(t):
            self.items.append((t, m, delays, amps))

    def take(self, end):
        keep, out = [], []
        for t, m, dl, am in self.items:
            sel = t < end
            if sel.all():
                out.append((t, m, dl, am))
            elif sel.any():
                out.append((t[sel], _sub(m, sel), dl[:, sel], am[:, sel]))
                keep.append((t[~sel], _sub(m, ~sel), dl[:, ~sel], am[:, ~sel]))
            else:
                keep.append((t, m, dl, am))
        self.items = keep
        if not out:
            return None
        t = np.concatenate([o[0] for o in out])
        m = Modes(*(np.concatenate([getattr(o[1], k) for o in out]) for k in ("f", "tau", "amp", "ph", "chirp")),
                  attack=np.concatenate([o[1].attack for o in out]), n_short=out[0][1].n_short)
        return t, m, np.concatenate([o[2] for o in out], axis=1), np.concatenate([o[3] for o in out], axis=1)


class _OU:
    """Ornstein-Uhlenbeck process at control rate (unit variance), streamed with state."""

    def __init__(self, rng, timescale_s):
        a = np.exp(-(CTRL / SR) / timescale_s)
        self.b, self.a = [np.sqrt(1 - a * a)], [1.0, -a]
        self.zi = np.array([rng.normal() * a])
        self.rng = rng

    def __call__(self, n):
        y, self.zi = signal.lfilter(self.b, self.a, self.rng.standard_normal(n), zi=self.zi)
        return y


class _Puff:
    """Quasi-periodic flame puffing: noise through a resonator at the puffing frequency."""

    def __init__(self, rng, hz, q=2.5):
        fs = SR / CTRL
        self.b, self.a = signal.iirpeak(hz, q, fs=fs)
        self.zi = signal.lfilter_zi(self.b, self.a) * 0.0
        probe = signal.lfilter(self.b, self.a, np.random.default_rng(0).standard_normal(int(fs * 400)))
        self.g = 1.0 / np.std(probe[int(fs * 20):])
        self.rng = rng

    def __call__(self, n):
        y, self.zi = signal.lfilter(self.b, self.a, self.rng.standard_normal(n), zi=self.zi)
        return y * self.g


def _spectrum_fir(points, path=None, taps=2047):
    """FIR for a (freq Hz, amplitude) curve, times an optional path gain function."""
    f = np.array([p[0] for p in points], float)
    g = np.array([p[1] for p in points], float)
    if path is not None:
        g = g * path(f)
    freqs = np.concatenate([[0], f, [SR / 2]])
    gains = np.concatenate([[0.0], g, [g[-1] * 0.1]])
    return signal.firwin2(taps, freqs, gains, fs=SR)


# ----------------------------------------------------------------------------- scene

class FireScene(Layer):
    """A wood fire in a setting, heard from the listener's position.

    environment  campfire, fireplace, wood_stove, bonfire
    intensity    how big and lively the fire is (0.4 low embers .. 1.4 roaring)
    wood         softwood (pops, sap), mixed, hardwood (steady, deep)
    dampness     0 dry .. 1 damp (more steam hiss, sizzle and singing)
    variability  how far the fire wanders between brighter and lower spells
    budget       individually rendered ember ticks per second (default depends on device)
    crest_db     loudest the fire may ever peak above its average level (sleep safety)
    """

    pannable = False

    def __init__(self, rng, environment="fireplace", intensity=0.8, wood="mixed", dampness=0.2,
                 variability=0.25, budget=None, crest_db=20.0):
        if environment not in FIRE_ENVIRONMENTS:
            raise ValueError(f"unknown fire setting {environment!r}; choose from {', '.join(FIRE_ENVIRONMENTS)}")
        if wood not in WOODS:
            raise ValueError(f"unknown wood {wood!r}; choose from {', '.join(WOODS)}")
        self.torch = gpu._torch()
        self.dev = gpu.device()
        env_rng = child(rng)
        self.place = place = FIRE_ENVIRONMENTS[environment]["build"](env_rng)
        self.wood = w = WOODS[wood]
        self.params = dict(environment=environment, intensity=intensity, wood=wood, dampness=dampness,
                           variability=variability, budget=budget, crest_db=crest_db)
        self.mean_i = float(intensity)
        self.damp = float(np.clip(dampness, 0, 1))
        self.crackle_rate = env_rng.uniform(*w["rate"]) * place.rate_scale
        self.alpha = w["alpha"]
        self.cascade = w["cascade"]
        self.bright = w["bright"]
        self.whistle_per_s = w["whistle"] * self.damp / 60.0
        self.settle_gap = env_rng.uniform(150, 420)  # s between log settles, on average
        self.tick_gap = 25.0  # s between the stove's iron ticks, on average
        # Ember ticks per second at intensity 1: a sprinkle of distinct ticks, not a wash.
        self.fizz_phys = 80.0 * place.rate_scale ** 0.5 * (1 + 1.5 * self.damp)
        budget = int(budget or gpu.default_budget())
        self.fizz_rate = min(self.fizz_phys, 0.5 * budget)
        self.fizz_amp = np.sqrt(self.fizz_phys / self.fizz_rate)  # fewer ticks, same energy
        self.pop_cap = 2.5  # biggest pop, relative to a typical one (soft limit)
        self.crest_db = crest_db
        self.limiter = PeakLimiter(10 ** (crest_db / 20), relative_s=2.0)  # output is ~unit RMS
        self.gain = None  # set by _calibrate
        self.mute = set()  # parts to leave out (for listening to one part alone)

        # Slow and fast control processes.
        self.inten = P.Wander(child(rng), self.mean_i, variability, timescale_s=600, swell=variability * 0.8,
                              lo=0.4, hi=1.8, dt=CTRL / SR)
        self.flicker = _OU(child(rng), env_rng.uniform(0.25, 0.6))
        self.activity = _OU(child(rng), env_rng.uniform(3.0, 7.0))
        self.hiss_wander = _OU(child(rng), env_rng.uniform(4.0, 12.0))
        self.puff = _Puff(child(rng), 1.5 / np.sqrt(place.diameter) * env_rng.uniform(0.9, 1.1))
        self.puff_depth = env_rng.uniform(0.1, 0.2)  # gentle: deep, fast swells sound like gusting wind
        self.flicker_depth = env_rng.uniform(0.1, 0.2)
        self.flare = 0.0

        self.ev_rng = child(rng)
        self.noise_rng = child(rng)
        self.whistle_rng = child(rng)
        self.q_pop, self.q_grain, self.q_fizz = _Queue(POP_L), _Queue(SHORT_L), _Queue(SHORT_L)
        self.q_low, self.q_tick = _Queue(THUMP_L), _Queue(THUMP_L)  # one voice per queue (mode counts differ)
        self.queues = [self.q_pop, self.q_grain, self.q_fizz, self.q_low, self.q_tick]
        self.Lmax = THUMP_L
        self.tail = self.torch.zeros(2, self.Lmax + 256, device=self.dev)
        self.ctrl = np.zeros((0, 3))  # roar envelope, hiss envelope, intensity
        self.ctrl_start = 0
        self.gen_until = 0
        self.pos = 0
        self.whistles = []

        # Steady parts: roar and hiss as shaped noise, heard through the same path.
        c = np.asarray(place.center, float)
        r, _, _, th_l, th_r = P.ear_geometry(c[None])
        self.r0 = float(r[0])

        def path(f):
            g = P.air_absorption(f, self.r0) / self.r0
            if place.transmission:
                fc, order, tg = place.transmission
                g = g * tg / np.sqrt(1 + (f / fc) ** (2 * order))
            return g

        self.path = path
        fp = place.roar_hz
        grid = np.geomspace(20, 20000, 80)
        roar = (grid / fp) ** 2 / (1 + (grid / fp) ** 2) / (1 + (grid / fp) ** 2.0)  # rise, peak, -12 dB/oct
        self.roar_fir = _FIR(_spectrum_fir(list(zip(grid, roar)), path))
        hiss = (grid / 3500.0) ** 2 / (1 + (grid / 1800.0) ** 4) ** 0.5 / (1 + (grid / 7000.0) ** 3)
        self.hiss_fir = _FIR(_spectrum_fir(list(zip(grid, hiss)), path))
        # Interaural level difference for the steady parts, from the fire's direction.
        self.roar_ild = np.array([P.head_shadow(500, th_l[0]), P.head_shadow(500, th_r[0])])
        self.hiss_ild = np.array([P.head_shadow(3500, th_l[0]), P.head_shadow(3500, th_r[0])])
        self.roar_ild /= np.sqrt(np.mean(self.roar_ild ** 2))
        self.hiss_ild /= np.sqrt(np.mean(self.hiss_ild ** 2))

        self.room = None
        if place.room:
            rm = place.room
            self.room = Reverb(child(rng), rt60=rm["rt60"], predelay_ms=rm["predelay_ms"], damping=rm["damping"])
            self.room_wet = rm["wet"]
        self._calibrate(child(rng))

    # -- events ----------------------------------------------------------------------
    def _positions(self, rng, k):
        c, s = np.asarray(self.place.center, float), np.asarray(self.place.size, float)
        return c + (rng.random((k, 3)) - 0.5) * s

    def _spatial(self, m, pos, through=True):
        r, dl, dr, th_l, th_r = P.ear_geometry(pos)
        common = m.amp * P.air_absorption(m.f, r[:, None]) / r[:, None]
        if self.place.transmission and through:
            fc, order, tg = self.place.transmission
            common = common * tg / np.sqrt(1 + (m.f / fc) ** (2 * order))
        amps = np.stack([common * P.head_shadow(m.f, th_l[:, None]), common * P.head_shadow(m.f, th_r[:, None])])
        return np.stack([dl, dr]), amps

    def _pop_sizes(self, rng, k):
        a = (1 - rng.random(k)) ** (-1 / self.alpha) / 2 ** (1 / self.alpha)  # Pareto, median 1
        # Heard sizes are gentler than the physics (a fire close by has some loud pops; for
        # sleep they are evened out, so the crackle texture carries rather than a few bangs).
        a = a ** 0.5
        return self.pop_cap * np.tanh(a / self.pop_cap)

    def _pops(self, rng, times, sizes):
        """Primary cracks plus the little cascade each sets off (more cracks within tens
        of milliseconds). Each crack is a dry snap followed by a dense sputter of tiny
        grains in the crack's own colour that dies away within about 10-30 ms: a short
        burst of crackling noise, the way real crackles sound. It also carries more sound
        for the same peak than a bare click, so crackles stay clear under the sleep-safety
        limit. Returns the snaps (times, sizes, colours) and the grains (the same)."""
        n_sec = rng.poisson(self.cascade, len(times))
        par = np.repeat(np.arange(len(times)), n_sec)
        t2 = times[par] + rng.exponential(0.008 * SR, len(par))
        a2 = sizes[par] * rng.uniform(0.25, 0.65, len(par))
        t, a = np.concatenate([times, t2]), np.concatenate([sizes, a2])
        fc = crack_colour(rng, a, self.bright)
        # Bigger cracks sputter on for longer.
        n_g = rng.poisson(18.0 + 12.0 * np.minimum(a, 3.0))
        par = np.repeat(np.arange(len(t)), n_g)
        gap = rng.exponential(0.00045 * SR, len(par))
        cs = np.cumsum(gap)
        first = np.cumsum(n_g) - n_g
        off = cs - np.repeat(np.concatenate([[0.0], cs])[first], n_g)
        tau = rng.uniform(0.004, 0.012, len(t)) * SR
        ag = a[par] * rng.uniform(0.35, 0.8, len(par)) * np.exp(-off / tau[par])
        return (t, a * 0.9, fc), (t[par] + off, ag, fc[par])

    def _emit(self, q, rng, times, m, through=True):
        if len(times) == 0:
            return
        delays, amps = self._spatial(m, self._positions(rng, len(times)), through)
        q.push(times, m, delays, amps)

    def _crackles(self, rng, times, sizes):
        (t, a, fc), (tg, ag, fcg) = self._pops(rng, times, sizes)
        self._emit(self.q_pop, rng, t, pop_modes(rng, a, self.bright, fc))
        self._emit(self.q_grain, rng, tg, grain_modes(rng, ag, fcg))

    def _generate_unit(self):
        rng = self.ev_rng
        u0 = self.gen_until
        dt = CTRL / SR
        # A log settle this second? It flares the fire up for a while afterwards.
        settle_at = None
        if rng.random() < 1.0 / self.settle_gap:
            settle_at = int(rng.integers(NC))
        inten = self.inten.blocks(NC)
        flare = np.empty(NC)
        for b in range(NC):
            if b == settle_at:
                self.flare += rng.uniform(0.15, 0.35)
            self.flare *= np.exp(-dt / 25.0)
            flare[b] = self.flare
        inten = inten * (1 + flare)
        flick = np.exp(self.flicker_depth * self.flicker(NC) - self.flicker_depth ** 2)
        puff = np.maximum(0.15, 1 + self.puff_depth * self.puff(NC)) / np.sqrt(1 + self.puff_depth ** 2)
        roar_env = (inten / self.mean_i) ** 0.8 * flick * puff
        act = np.exp(0.6 * self.activity(NC) - 0.18)
        hiss_env = (inten / self.mean_i) * np.exp(0.4 * self.hiss_wander(NC) - 0.08)
        self.ctrl = np.concatenate([self.ctrl, np.stack([roar_env, hiss_env, inten], axis=1)])

        starts = u0 + np.arange(NC) * CTRL
        # Crackles: a doubly stochastic Poisson process (busy and quiet spells), flames
        # flaring up bring more of them.
        lam = self.crackle_rate * (inten / self.mean_i) ** 1.3 * act * flick ** 0.5 * dt
        k = rng.poisson(lam)
        t = np.repeat(starts, k) + rng.random(k.sum()) * CTRL
        self._crackles(rng, t, self._pop_sizes(rng, len(t)))
        # Ember fizz follows the flames closely.
        k = rng.poisson(self.fizz_rate * (inten / self.mean_i) * flick * dt)
        t = np.repeat(starts, k) + rng.random(k.sum()) * CTRL
        a = rng.lognormal(0, 0.7, len(t)) * self.fizz_amp
        self._emit(self.q_fizz, rng, t, fizz_modes(rng, a))
        # Settling logs: a soft cluster of ticks over about a second and a low knock.
        if settle_at is not None:
            t0 = u0 + settle_at * CTRL
            n = int(rng.integers(15, 50))
            dur = rng.uniform(0.4, 1.4) * SR
            t = t0 + dur * rng.beta(1.5, 3.0, n)
            self._emit(self.q_pop, rng, t, pop_modes(rng, rng.uniform(0.1, 0.5, n), self.bright))
            self._emit(self.q_low, rng, np.array([t0 + rng.uniform(0, 0.1) * SR]), thump_modes(rng, np.array([0.9])))
        if self.place.ticks:
            k = rng.poisson(1 / self.tick_gap)
            if k:
                # On the outside of the stove, so not muffled by its walls.
                self._emit(self.q_tick, rng, u0 + rng.random(k) * UNIT, tick_modes(rng, rng.uniform(0.2, 0.5, k)),
                           through=False)
        # Sap and steam "singing": rare, soft, gliding, fades in and out.
        if rng.random() < self.whistle_per_s:
            self._new_whistle(u0 + rng.random() * UNIT)
        self.gen_until += UNIT

    def _new_whistle(self, t0):
        rng = self.whistle_rng
        dur = rng.uniform(2.0, 8.0)
        n = int(dur * SR)
        f0 = rng.uniform(1800, 5000)
        glide = rng.uniform(-0.25, 0.25)
        # Narrowband noise: a smooth random envelope times a gliding carrier.
        bw = rng.uniform(20, 80)
        sos = signal.butter(2, bw, "lowpass", fs=SR, output="sos")
        mod = signal.sosfilt(sos, rng.standard_normal((n, 2)), axis=0)
        mod /= np.sqrt(np.mean(mod ** 2)) + 1e-12
        tt = np.arange(n) / SR
        carrier = np.sin(2 * np.pi * f0 * (tt + 0.5 * glide * tt * tt / dur))
        fade = np.minimum(1, np.minimum(tt, dur - tt) / rng.uniform(0.6, 1.8)) ** 2
        g = self.path(np.array([f0]))[0] * rng.uniform(0.25, 0.6)
        self.whistles.append((int(t0), mod * (carrier * fade)[:, None] * g))

    # -- calibration -------------------------------------------------------------------
    def _calibrate(self, rng):
        """Set each part's gain from its expected power at the average intensity, then
        mix them at the setting's balance: the bed (roar, ember ticks, hiss) by `mix_db`
        (roar = 0 dB), and the crackles `crackle` dB above the whole bed."""
        def energy(m, amps):  # expected energy per event (both ears, sum of squares)
            return float(np.mean(np.sum(amps ** 2 * (m.tau * SR / 4)[None], axis=(0, 2))))

        n = 3000
        (t, a, fc), (tg, ag, fcg) = self._pops(rng, np.zeros(n), self._pop_sizes(rng, n))
        m_pop, m_grain = pop_modes(rng, a, self.bright, fc), grain_modes(rng, ag, fcg)
        e_crack = sum(len(x) * energy(m, self._spatial(m, self._positions(rng, len(x)))[1])
                      for x, m in ((a, m_pop), (ag, m_grain))) / n  # per primary crack
        m_fizz = fizz_modes(rng, rng.lognormal(0, 0.7, n) * self.fizz_amp)
        _, a_fizz = self._spatial(m_fizz, self._positions(rng, n))
        mix = dict(self.place.mix_db)
        mix["hiss"] = mix["hiss"] + 10 * self.damp * self.wood["hiss"]
        mix["crackle"] = mix["crackle"] + 10 * np.log10(sum(10 ** (mix[k] / 10) for k in ("roar", "hiss", "fizz")))
        p = {
            "roar": float(np.sum(self.roar_fir.h ** 2)),
            "hiss": float(np.sum(self.hiss_fir.h ** 2)),
            "fizz": self.fizz_rate * energy(m_fizz, a_fizz) / SR / 2,
            "crackle": self.crackle_rate * e_crack / SR / 2,
        }
        self.g = {k: 10 ** (mix[k] / 20) / np.sqrt(max(p[k], 1e-30)) for k in p}
        wet = self.room_wet ** 2 if self.room is not None else 0.0
        self.gain = 1.0 / np.sqrt(sum(10 ** (mix[k] / 10) for k in p) * (1 + wet))

    # -- main ----------------------------------------------------------------------------
    def _env(self, col, start, n):
        idx = (start + np.arange(n)) / CTRL - self.ctrl_start
        return np.interp(idx, np.arange(len(self.ctrl)), self.ctrl[:, col])

    def __call__(self, n):
        torch = self.torch
        start, end = self.pos, self.pos + n
        while self.gen_until < end + UNIT:  # look ahead: cascades and envelopes cross units
            self._generate_unit()
        buf = torch.zeros(2, n + self.Lmax + 256, device=self.dev)
        buf[:, : self.tail.shape[1]] += self.tail
        g_ = {k: (0.0 if k in self.mute else v) for k, v in self.g.items()}
        for q, g in ((self.q_pop, g_["crackle"]), (self.q_grain, g_["crackle"]), (self.q_fizz, g_["fizz"]),
                     (self.q_low, g_["crackle"]), (self.q_tick, g_["crackle"])):
            got = q.take(end)
            if got is not None:
                t, m, delays, amps = got
                render_events(buf, t - start, delays, amps * g, m, q.L)
        while self.gen_until < end + n + UNIT:  # prepare the next chunk while the GPU works
            self._generate_unit()
        self.tail = buf[:, n:].clone()
        out = buf[:, :n].T.cpu().numpy().astype(np.float64)

        # One draw per sample row, so the noise is the same whatever the chunk size.
        z = self.noise_rng.standard_normal((n, 6))
        rho = self.place.coherence
        roar = self.roar_fir(np.sqrt(rho) * z[:, :1] + np.sqrt(1 - rho) * z[:, 1:3])
        hiss = self.hiss_fir(np.sqrt(0.3) * z[:, 3:4] + np.sqrt(0.7) * z[:, 4:6])
        out += roar * self.roar_ild * (self._env(0, start, n) * g_["roar"])[:, None]
        out += hiss * self.hiss_ild * (self._env(1, start, n) * g_["hiss"])[:, None]
        if self.whistles:
            live = []
            for t0, wav in self.whistles:
                a, b = max(start, t0), min(end, t0 + len(wav))
                if a < b:
                    out[a - start:b - start] += wav[a - t0:b - t0] * g_["hiss"]
                if t0 + len(wav) > end:
                    live.append((t0, wav))
            self.whistles = live

        drop = max(0, (start // CTRL) - 2 - self.ctrl_start)
        if drop:
            self.ctrl = self.ctrl[drop:]
            self.ctrl_start += drop
        if self.room is not None:
            out = out + self.room_wet * self.room(out)
        self.pos = end
        return self.limiter(out * self.gain)
