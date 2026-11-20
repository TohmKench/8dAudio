"""The physics of falling rain, used to drive the synthesis.

Everything starts from a rainfall rate in mm/h (light ~1-2.5, moderate ~2.5-8,
heavy ~8-50). From that we get how many drops of each size hit each square metre
per second, how fast they are travelling, and so how hard each one lands.

  Drop sizes:       Marshall-Palmer distribution, N(D) = N0 exp(-L D), L = 4.1 R^-0.21
  Fall speed:       Atlas et al. terminal velocity, v(D) = 9.65 - 10.3 exp(-0.6 D)
  Short falls:      drag-limited approach to terminal speed (drips from leaves, eaves)
  Bubble pitch:     Minnaert resonance, f = 3.26 / radius (m)
  Bubble damping:   van den Doel, d = 0.13 f + 0.0072 f^1.5
  Ears:             Woodworth interaural delay + Brown-Duda head-shadow filter
"""

from __future__ import annotations

import numpy as np

SR = 48_000
G = 9.81
N0 = 8000.0  # drops per m^3 per mm of diameter
D_MIN, D_MAX = 0.3, 6.0  # mm; smaller drops are inaudible, bigger ones break up
HEAD_RADIUS = 0.0875  # m
FORCE_K = 2.5  # impact brightness: force spectrum corner = FORCE_K / contact time
SPEED_OF_SOUND = 343.0

_GRID = np.linspace(D_MIN, D_MAX, 600)


def mp_lambda(rate_mm_h: float) -> float:
    return 4.1 * max(rate_mm_h, 0.05) ** -0.21


def terminal_velocity(d_mm):
    return np.clip(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, dtype=float)), 0.3, None)


def fall_velocity(d_mm, height_m=None):
    """Impact speed after falling `height_m` (None = from the cloud, i.e. terminal)."""
    vt = terminal_velocity(d_mm)
    if height_m is None:
        return vt
    return vt * np.sqrt(1 - np.exp(-2 * G * np.asarray(height_m, dtype=float) / vt ** 2))


def flux_density(d_mm, rate_mm_h):
    """Drops per m^2 per second per mm of diameter, hitting a horizontal surface."""
    return N0 * np.exp(-mp_lambda(rate_mm_h) * d_mm) * terminal_velocity(d_mm)


def cumulative_flux_above(rate_mm_h):
    """(grid, F) where F[i] = drops/m^2/s with diameter >= grid[i]."""
    dens = flux_density(_GRID, rate_mm_h)
    step = _GRID[1] - _GRID[0]
    tail = np.cumsum(dens[::-1])[::-1] * step
    return _GRID, tail


def dmin_for_budget(rate_mm_h, area_m2, events_per_s):
    """Smallest diameter we can synthesise one by one within the budget.

    Drops below it still count: they go into the diffuse 'remainder' field.
    """
    grid, tail = cumulative_flux_above(rate_mm_h)
    total = tail * area_m2
    if total[0] <= events_per_s:
        return D_MIN
    i = np.searchsorted(-total, -events_per_s)  # total is decreasing
    return float(grid[min(i, len(grid) - 1)])


def sample_diameters(rng, n, rate_mm_h, lo=D_MIN, hi=D_MAX):
    """Diameters of drops that land (flux-weighted), between lo and hi mm."""
    if n == 0:
        return np.zeros(0)
    g = np.linspace(lo, hi, 256)
    dens = flux_density(g, rate_mm_h)
    cdf = np.cumsum(dens)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-12)
    return np.interp(rng.random(n), cdf, g)


def impact_strength(d_mm, v):
    """Relative sound pressure of an impact: sqrt of kinetic energy (~ D^1.5 v)."""
    return (np.asarray(d_mm) / 2.0) ** 1.5 * np.asarray(v) / 6.5


def force_weight(f, d_mm, v):
    """Impact force spectrum for modes `f` (E,K) of drops `d_mm`, `v` (E,).

    Bigger, slower drops push for longer, so they sound duller.
    """
    tc = (np.asarray(d_mm) * 1e-3) / np.maximum(np.asarray(v), 0.5)  # contact time
    fc = FORCE_K / tc  # the initial pressure spike is much sharper than the whole contact
    return 1.0 / np.sqrt(1.0 + (np.asarray(f) / fc[:, None]) ** 2)


def minnaert_hz(radius_mm):
    return 3.26 / (np.asarray(radius_mm) * 1e-3)


def bubble_damping(f_hz):
    f = np.asarray(f_hz)
    return 0.13 * f + 0.0072 * f ** 1.5


def air_absorption(f_hz, distance_m):
    """Amplitude factor for high-frequency loss in air (~0.16 dB/m at 8 kHz)."""
    db = 0.16 * (np.asarray(f_hz) / 8000.0) ** 1.8 * np.asarray(distance_m)
    return 10 ** (-db / 20)


def ear_geometry(pos):
    """For positions (E,3) in metres (x right, y forward, z up) relative to the head:
    distance, left/right delays in samples, and the angle from each ear's axis."""
    pos = np.asarray(pos, dtype=float)
    r = np.maximum(np.linalg.norm(pos, axis=1), 0.25)
    sx = np.clip(pos[:, 0] / r, -1, 1)
    lateral = np.arcsin(sx)
    itd = HEAD_RADIUS / SPEED_OF_SOUND * (np.sin(lateral) + lateral)  # + = right ear first
    delay_l = np.maximum(itd, 0) * SR
    delay_r = np.maximum(-itd, 0) * SR
    theta_r = np.arccos(sx)  # 0 = straight at the right ear
    theta_l = np.arccos(-sx)
    return r, delay_l, delay_r, theta_l, theta_r


def head_shadow(f_hz, theta):
    """Brown-Duda head shadow magnitude for angle `theta` from the ear axis."""
    w = 2 * np.pi * np.asarray(f_hz)
    w0 = SPEED_OF_SOUND / HEAD_RADIUS
    alpha = 1.05 + 0.95 * np.cos(np.asarray(theta) * (180.0 / 150.0))
    x = w / (2 * w0)
    return np.sqrt(1 + (alpha * x) ** 2) / np.sqrt(1 + x ** 2)


class Wander:
    """A slowly wandering value (log-normal Ornstein-Uhlenbeck plus a long swell).

    Used for rainfall rate and wind, so a 10 hour night drifts naturally between
    lighter and heavier spells instead of sitting at one level.
    """

    def __init__(self, rng, mean, variability=0.3, timescale_s=240.0, swell=0.25,
                 swell_period_s=(1800, 5400), lo=0.3, hi=3.0, dt=0.1):
        self.rng = rng
        self.mean, self.var, self.dt = float(mean), float(variability), dt
        self.a = np.exp(-dt / timescale_s)
        self.b = variability * np.sqrt(1 - self.a ** 2)
        self.x = rng.normal(0, variability)
        self.swell = swell
        self.swell_period = rng.uniform(*swell_period_s)
        self.swell_phase = rng.uniform(0, 2 * np.pi)
        self.lo, self.hi = lo, hi
        self.k = 0

    def blocks(self, n: int) -> np.ndarray:
        noise = self.rng.standard_normal(n)
        out = np.empty(n)
        x = self.x
        for i in range(n):
            x = self.a * x + self.b * noise[i]
            out[i] = x
        self.x = x
        t = (self.k + np.arange(n)) * self.dt
        self.k += n
        log_mult = out + self.swell * np.sin(2 * np.pi * t / self.swell_period + self.swell_phase)
        return self.mean * np.clip(np.exp(log_mult), self.lo, self.hi)
