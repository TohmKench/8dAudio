"""Rain scenes: rain falling on real surfaces around a listener, one drop at a time.

A scene is a set of *patches* (a roof above you, a window to one side, a lake in
front, a forest canopy overhead). Each second, every patch receives the number and
sizes of drops that real rain at the current rainfall rate would deliver to its area.
Each drop is modelled individually from its own physical properties (see surfaces.py)
and placed in 3D: its distance sets level and air absorption, and its direction sets the
delay and head-shadow at each ear, so it is heard where it lands.

Rendering is built for speed on the GPU without changing what a drop is. Each surface's
resonances are drawn from large pools rendered once (DropPool), and a drop strikes its
own few members, each with its own strength per ear. Sheets (metal, glass, fabric) are
banks of fixed per-panel resonators (PlateBank). Long rings are rendered as FFT filter
banks (FilterBank): a drop places one impulse per resonance it excites, so the cost
depends on the number of resonances rather than the number of drops.

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
from scipy import signal

from . import gpu
from . import rainphys as P
from .dsp import SR, Convolver, Reverb, rms
from .layers import Layer, child
from .surfaces import FLOOR_DB, SHORT_L, SIZE_BINS, Modes, Plate, _attack, make_surface, size_bin

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
        return gpu.upload(a, torch.float32)

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
        base = gpu.upload(i0.astype(np.int64))
        for ear in (0, 1):
            y = torch.einsum("bm,bml->bl", T(amps[ear][e]), g) * env
            idx = (base + gpu.upload(d_int[ear][e]))[:, None] + nl[None, :]
            buf[ear].index_add_(0, idx.reshape(-1), y.reshape(-1))


class FilterBank:
    """Drops rendered as impulses into a bank of fixed responses, by FFT convolution.

    Copying each drop's whole response into the soundtrack costs one addition per sample
    per drop, and on the GPU those additions are the limit (about 3 billion per second,
    whatever the layout). Here a drop only places one impulse for each resonance it
    excites. Each resonance's impulse train is then convolved with its response in blocks,
    with FFTs, so the cost depends on how many resonances there are, not on how many drops
    hit them: more drops cost almost nothing extra.

    waves   (R, Lw) tensor: each resonance's response
    gains   optional (P, 2, K) fixed gain per ear for R = P x K resonances in P groups of K
            (a sheet's panels, heard from each panel's centre), with `delays` (2, P): each
            group's whole-sample delay per ear. Without them, every impulse carries its own
            per-ear amplitude and position.
    """

    def __init__(self, waves, dev, gains=None, delays=None):
        torch = gpu._torch()
        R, Lw = waves.shape
        self.R = R
        dmax = 0 if delays is None else int(np.max(delays))
        self.nfft = gpu.fft_len(max(8192, 6 * (Lw + dmax)))
        self.Nb = self.nfft - Lw - dmax  # impulse positions covered per block
        self.W = torch.fft.rfft(waves, n=self.nfft, dim=1)  # (R, F)
        if gains is None:
            self.C = 2
        else:
            self.C = 1
            F = self.W.shape[1]
            self.P, _, self.K = gains.shape
            self.G = torch.as_tensor(np.asarray(gains, np.float32), device=dev).to(self.W.dtype)
            w = torch.arange(F, device=dev, dtype=torch.float32) * (2 * np.pi / self.nfft)
            d = torch.as_tensor(np.asarray(delays, np.float32).T, device=dev)  # (P, 2)
            self.phase = torch.polar(torch.ones(self.P, 2, F, device=dev), -d[:, :, None] * w[None, None])

    @property
    def span(self):
        """Samples past the last impulse position that a render can write into."""
        return self.nfft

    def render(self, buf, pos, member, amp):
        """pos (C, Q) whole-sample positions in `buf` (2, N); member (Q,); amp (C, Q)."""
        self.prepare(pos, member, amp)(buf)

    def prepare(self, pos, member, amp):
        """Sort and upload the impulses now; returns the job that renders them into a buffer
        (so all uploads can happen before any GPU work is queued, see RainScene.__call__)."""
        torch = gpu._torch()
        if pos.shape[1] == 0:
            return lambda buf: None
        C, R, nfft, Nb = self.C, self.R, self.nfft, self.Nb
        blk = pos // Nb
        flat = (np.arange(C)[:, None] * R + member[None]) * nfft + (pos - blk * Nb)
        blk, flat, a = blk.ravel(), flat.ravel(), amp.ravel()
        order = np.argsort(blk.astype(np.int16), kind="stable")  # small ints: a fast radix sort
        blk, flat, a = blk[order], flat[order], a[order]
        bounds = np.searchsorted(blk, np.arange(blk[-1] + 2))
        flat_t = gpu.upload(flat)
        a_t = gpu.upload(a, torch.float32)
        F = nfft // 2 + 1
        rc = max(1, (1 << 23) // (2 * F))  # resonances per multiply-and-sum step (bounded temporaries)
        blocks = [(int(b), int(bounds[b]), int(bounds[b + 1])) for b in np.flatnonzero(np.diff(bounds))]

        def run(buf):
            dev = buf.device
            x = torch.zeros(C * R * nfft, device=dev)
            for b, lo, hi in blocks:
                x.zero_()
                x.index_add_(0, flat_t[lo:hi], a_t[lo:hi])
                X = torch.fft.rfft(x.view(C, R, nfft), dim=2)
                if C == 1:  # per panel: weight each mode per ear, then delay each panel per ear
                    Z = torch.bmm(self.G, (X[0] * self.W).view(self.P, self.K, F))  # (P, 2, F)
                    Y = (Z * self.phase).sum(0)
                else:
                    Y = torch.zeros(2, F, dtype=X.dtype, device=dev)
                    for r0 in range(0, R, rc):
                        Y += (X[:, r0:r0 + rc] * self.W[r0:r0 + rc]).sum(1)
                s = b * Nb
                buf[:, s:s + nfft] += torch.fft.irfft(Y, nfft, dim=1)
        return run


def _wave(dev, f, tau, L, ph, chirp=None, att=None):
    """Decaying resonances (R,) rendered to (R, L), with the end tapered to zero."""
    torch = gpu._torch()

    def T(a):
        return torch.as_tensor(np.asarray(a, np.float32), device=dev)[:, None]

    n = torch.arange(L, device=dev, dtype=torch.float32)
    x = (n / SR)[None]
    arg = x if chirp is None else x + 0.5 * T(chirp) * x * x
    w = torch.exp(-x / T(tau)) * torch.sin(2 * np.pi * T(f) * arg + T(ph))
    w = w * torch.clamp((L - n) / (0.12 * L), max=1.0)
    if att is not None:
        w = w * (1 - torch.exp(-x / T(att)))
    return w


class PlateBank:
    """A panelled sheet (metal roofing, glass, tent fabric) as a bank of resonators.

    A sheet's vibration modes are fixed, so each panel's modes (and its low boom) are
    resonators heard from the panel's centre (a ringing sheet radiates from its whole
    surface, not from the point that was hit), with the right delay, head shadow,
    distance and air loss at each ear. A drop strikes the modes its landing spot and size
    excite (precomputed per panel, spot and size class), each with its own strength.
    """

    def __init__(self, patch, surf: Plate, dev, rng):
        s, c = np.asarray(patch.size, float), np.asarray(patch.center, float)
        axes = np.argsort(-s)[:2]
        centres = []
        for p in range(surf.n_panels):
            iu, iv = divmod(p, surf.nv)
            uvw = np.full(3, 0.5)
            uvw[axes[0]], uvw[axes[1]] = (iu + 0.5) / surf.nu, (iv + 0.5) / surf.nv
            centres.append(c + (uvw - 0.5) * s)
        v_bins = P.fall_velocity(SIZE_BINS, patch.fall_height)
        self.idx, self.amp = surf.bank_modes(v_bins, rng)  # (Pn, S, B, m)
        f, tau, ok = surf.resonators()  # (Pn, K)
        Pn, K = f.shape
        self.K = K
        self.n_spots, self.n_sizes = self.idx.shape[1], self.idx.shape[2]
        r, dl, dr, th_l, th_r = P.ear_geometry(np.array(centres))
        common = P.air_absorption(f, r[:, None]) / r[:, None] * patch.path_gain(f) * ok
        gains = np.stack([common * P.head_shadow(f, th_l[:, None]), common * P.head_shadow(f, th_r[:, None])], axis=1)
        delays = np.round(np.stack([dl, dr])).astype(int)  # (2, Pn)
        d_typ = np.mean(patch.fixed_d) if patch.fixed_d else 1.5
        att = float(_attack(np.array([d_typ]), P.fall_velocity(np.array([d_typ]), patch.fall_height))[0])
        waves = _wave(dev, f.ravel(), tau.ravel(), surf.L, rng.uniform(0, 2 * np.pi, Pn * K),
                      att=np.full(Pn * K, att))
        self.bank = FilterBank(waves, dev, gains, delays)

    def strikes(self, panel, spot, size, strength):
        """Resonators (E, m) each drop strikes and the strength of each strike."""
        idx = self.idx[panel, spot, size] + (panel * self.K)[:, None]
        return idx, self.amp[panel, spot, size] * strength[:, None]

    def prepare(self, start, idx, amp):
        E, m = idx.shape
        pos = np.repeat(np.round(start).astype(np.int64), m)[None]
        return self.bank.prepare(pos, idx.ravel(), amp.ravel()[None])


class DropPool:
    """Pre-rendered resonances for one surface's per-drop sound (surfaces.Burst, Ring, Bubble).

    Each part of the sound gets a pool of single decaying resonances drawn from the same
    distribution the drops would draw from, rendered once. A drop picks its own members
    of each pool and gives each a strength per ear (from its frequency, the drop's size and
    speed, distance, air loss and head shadow, exactly as a freshly synthesised drop
    would). The parts are grouped by how long they ring: the clicks (SHORT_L samples) and
    everything else (the surface's L). Each group is rendered whichever way is cheaper on
    this device for the expected number of drops: as weighted copies of the members
    (cheap for short clicks; keeps each drop's own attack), or as an FFT filter bank.
    """

    DIRECT_SIZE = 2048  # members per part when copied per drop (more variety costs nothing)

    def __init__(self, surf, dev, rng, drops_per_s, att):
        self.parts = surf.parts
        self.info = [None] * len(surf.parts)
        self.groups = []  # (part indices, member offsets, waves (J, L), FilterBank or None)
        kf = FLOOR_DB / 8.686
        for short in (True, False):
            ids = [i for i, p in enumerate(surf.parts) if p.short == short]
            if not ids:
                continue
            L = SHORT_L if short else surf.L
            # Estimated cost per second of audio, each way.
            n_fft = sum(surf.parts[i].members for i in ids)
            nfft = gpu.fft_len(max(8192, 6 * L))
            fft_cost = 2 * n_fft * SR * nfft / (nfft - L) / gpu.fft_rate()
            direct_cost = drops_per_s * 2 * L / gpu.scatter_rate()
            use_fft = fft_cost < direct_cost
            for i in ids:
                self.info[i] = surf.parts[i].pool(rng, surf.parts[i].members if use_fft else self.DIRECT_SIZE)
            pools = [self.info[i] for i in ids]
            offsets = np.cumsum([0] + [len(q["f"]) for q in pools[:-1]])
            f = np.concatenate([q["f"] for q in pools])
            tau = np.minimum(np.concatenate([q["tau"] for q in pools]), L / (kf * SR))
            chirp = np.concatenate([q.get("chirp", np.zeros(len(q["f"]))) for q in pools])
            ph = rng.uniform(0, 2 * np.pi, len(f))
            if use_fft:  # the attack is baked in (it barely matters for these longer rings)
                waves = _wave(dev, f, tau, L, ph, chirp, att=np.full(len(f), att))
                self.groups.append((ids, offsets, waves, FilterBank(waves, dev)))
            else:
                self.groups.append((ids, offsets, _wave(dev, f, tau, L, ph, chirp), None))

    @property
    def span(self):
        return max(fb.span if fb is not None else w.shape[1] for _, _, w, fb in self.groups)

    def pick(self, rng, d, v):
        """For drops (d, v): per group, pool members (E, k), amplitudes (E, k) and their
        frequencies (E, k)."""
        out = []
        for ids, offsets, _, _ in self.groups:
            idx, amp, f = [], [], []
            for i, off in zip(ids, offsets):
                j, a = self.parts[i].pick(rng, d, v, self.info[i])
                idx.append(j + off), amp.append(a), f.append(self.info[i]["f"][j])
            out.append((np.concatenate(idx, axis=1), np.concatenate(amp, axis=1), np.concatenate(f, axis=1)))
        return out

    def prepare(self, start, delays, att, picks):
        """start (E,), delays (2, E) samples, att (E,) s; picks: per group (idx (E, k), amps (2, E, k)).
        Returns the jobs that render them (see FilterBank.prepare)."""
        jobs = []
        for (_, _, wave, fb), (idx, amps) in zip(self.groups, picks):
            if fb is None:
                jobs.append(prepare_pool(start, delays, att, idx, amps, wave))
            else:
                E, k = idx.shape
                pos = np.repeat(np.round(start).astype(np.int64)[None] + np.round(delays).astype(np.int64), k, axis=1)
                jobs.append(fb.prepare(pos, idx.ravel(), amps.reshape(2, -1)))
        return jobs


def prepare_pool(start, delays, att, idx, amps, wave):
    """Upload drops made of pool members; returns the job that adds them into a buffer (2, N).

    start (E,) onset in samples relative to buf[0]; delays (2, E) per-ear delay in samples;
    att (E,) attack time (s); idx (E, k) members of `wave` (J, L); amps (2, E, k).
    """
    torch = gpu._torch()
    E, k = idx.shape
    if E == 0:
        return lambda buf: None
    L = wave.shape[1]
    A = min(L, 96)  # the attack (<= 0.3 ms) is over well within 96 samples
    i0 = np.round(start).astype(np.int64)
    d_int = np.round(delays).astype(np.int64)
    bs = max(1, gpu.batch_elements() // (k * L + 2 * L))
    idx_t = gpu.upload(idx)
    a_t = gpu.upload(amps.transpose(1, 0, 2), torch.float32)  # (E, 2, k)
    att_t = gpu.upload(att, torch.float32)
    pos_t = gpu.upload(i0[None] + d_int)  # (2, E)

    def run(buf):
        dev = buf.device
        na = torch.arange(A, device=dev, dtype=torch.float32) / SR
        nl = torch.arange(L, device=dev)
        for s in range(0, E, bs):
            e = slice(s, s + bs)
            y = torch.bmm(a_t[e], wave[idx_t[e]])  # (b, 2, k) @ (b, k, L) -> (b, 2, L)
            y[:, :, :A] *= (1 - torch.exp(-na[None] / att_t[e][:, None]))[:, None, :]
            for ear in (0, 1):
                buf[ear].index_add_(0, (pos_t[ear, e][:, None] + nl[None]).reshape(-1), y[:, ear].reshape(-1))
    return run


def render_pool(buf, start, delays, att, idx, amps, wave):
    """Add drops made of pool members into `buf` (2, N) (see prepare_pool)."""
    prepare_pool(start, delays, att, idx, amps, wave)(buf)


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

class _FIR(Convolver):
    """Streaming FIR (overlap-add) on stereo numpy blocks."""


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
        # Every drop is made from pre-rendered pools of resonances (DropPool), and panelled
        # sheets ring from a precomputed bank of whole responses (PlateBank).
        bank_rng = child(rng)
        self.banks = [PlateBank(p, s, self.dev, bank_rng) if isinstance(s, Plate) and p.individual else None
                      for p, s in zip(self.patches, self.surfaces)]
        pool_rng = child(rng)
        self.pools = []
        for i, (p, s) in enumerate(zip(self.patches, self.surfaces)):
            if not p.individual:
                self.pools.append(None)
                continue
            rate = p.fixed_rate if p.fixed_rate is not None else self.share[i]
            d_typ = np.mean(p.fixed_d) if p.fixed_d else 1.5
            att = float(_attack(np.array([d_typ]), P.fall_velocity(np.array([d_typ]), p.fall_height))[0])
            self.pools.append(DropPool(s, self.dev, pool_rng, rate, att))
        # Room past the end of each chunk for what drops near its end write (carried over).
        self.Lmax = max([s.L for s in self.surfaces] + [x.span + 64 for x in self.pools if x is not None]
                        + [b.bank.span + 64 for b in self.banks if b is not None])
        self.room = None
        if env.get("room"):
            r = env["room"]
            self.room = Reverb(child(rng), rt60=r["rt60"], predelay_ms=r["predelay_ms"], damping=r["damping"])
            self.room_wet = r["wet"]

        self.pending = [[] for _ in self.patches]
        self.bank_pending = [[] for _ in self.patches]
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
        m = self.surfaces[i].voice(rng, d, v, uv)
        delays, amps = self._spatial(i, m, pos)
        return m, delays, amps

    def _spatial(self, i, m, pos):
        geo = P.ear_geometry(pos)
        return np.stack(geo[1:3]), self._ear_amps(i, m.amp, m.f, geo)

    def _ear_amps(self, i, amp, f, geo):
        """Per-ear amplitudes (2, E, k) of modes (E, k) heard from drops at `geo`."""
        r, _, _, th_l, th_r = geo
        common = amp * P.air_absorption(f, r[:, None]) / r[:, None] * self.patches[i].path_gain(f)
        return np.stack([common * P.head_shadow(f, th_l[:, None]), common * P.head_shadow(f, th_r[:, None])])

    @staticmethod
    def _take_from(pending, end):
        """Pop the drops in `pending` (tuples of arrays, events on axis 0) that start before `end`."""
        keep, out = [], []
        for item in pending:
            sel = item[0] < end
            if sel.all():
                out.append(item)
            elif sel.any():
                out.append(tuple(a[sel] for a in item))
                keep.append(tuple(a[~sel] for a in item))
            else:
                keep.append(item)
        pending[:] = keep
        if not out:
            return None
        return tuple(np.concatenate([o[j] for o in out]) for j in range(len(out[0])))

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
                t = np.concatenate(times)
                geo = P.ear_geometry(pos)
                item = [t, _attack(d, v), np.stack(geo[1:3], axis=1)]  # events on axis 0
                for idx, amp, f in self.pools[i].pick(rng, d, v):
                    item += [idx, self._ear_amps(i, amp, f, geo).transpose(1, 0, 2)]
                self.pending[i].append(tuple(item))
                if self.banks[i] is not None:
                    panel, spot = self.surfaces[i].locate(uv)
                    idx, amp = self.banks[i].strikes(panel, spot, size_bin(d), P.impact_strength(d, v))
                    self.bank_pending[i].append((t, idx, amp))
        self.gen_until += UNIT

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
        # Upload everything first: a copy to the GPU waits for all work queued before it, so
        # uploading between kernels would make the CPU and GPU take turns.
        jobs = []
        for i, p in enumerate(self.patches):
            got = self._take_from(self.pending[i], end)
            if got is not None:
                t, att, delays = got[:3]
                picks = [(got[3 + 2 * g], got[4 + 2 * g].transpose(1, 0, 2)) for g in range(len(self.pools[i].groups))]
                jobs += self.pools[i].prepare(t - start, delays.T, att, picks)
            if self.banks[i] is not None:
                got = self._take_from(self.bank_pending[i], end)
                if got is not None:
                    t, idx, amp = got
                    jobs.append(self.banks[i].prepare(t - start, idx, amp))
        buf = torch.zeros(2, n + self.Lmax + 256, device=self.dev)
        buf[:, : self.tail.shape[1]] += self.tail
        for job in jobs:
            job(buf)
        # The GPU works through what was just queued; meanwhile prepare the next chunk's
        # drops (assuming the same size), which is what the CPU would otherwise do later.
        while self.gen_until < end + n + CTRL:
            self._generate_unit()
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
