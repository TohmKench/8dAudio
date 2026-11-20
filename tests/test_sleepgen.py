import json
import os
import subprocess

import numpy as np
import pytest

from sleepgen import engine, layers
from sleepgen.dsp import SR, loudness_lufs
from sleepgen.recipes import BASES, available_bases, random_recipe


def _render_raw(recipe, seconds, chunk_s):
    mix = engine.Mix(recipe)
    n, chunk = int(seconds * SR), int(chunk_s * SR)
    out = []
    while n > 0:
        k = min(chunk, n)
        out.append(mix.raw(k))
        n -= k
    return np.concatenate(out)


@pytest.mark.parametrize("spec", [
    {"type": "noise", "params": {"color": "brown", "lowpass_hz": 500}, "pan": {"mode": "circle"}},
    {"type": "noise", "params": {"color": "pink"}, "drift": {"depth_db": 2}},
    {"type": "fan", "params": {}},
    {"type": "binaural", "params": {"carrier_hz": 150, "beat_hz": 2}},
    {"type": "drone", "params": {"freq_hz": 528}, "pan": {"mode": "sweep"}},
])
def test_chunk_size_does_not_change_output(spec):
    """Same audio whether rendered in 1 s or 3.7 s chunks: no seams at chunk edges."""
    recipe = {"seed": 5, "layers": [spec]}
    a = _render_raw(recipe, 8, 1.0)
    b = _render_raw(recipe, 8, 3.7)
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("theme", available_bases())
def test_every_theme_renders_to_target(theme, tmp_path):
    recipe = random_recipe(11, theme)
    json.dumps(recipe)  # recipes must be plain JSON
    recipe["master"].update(fade_in_s=3, fade_out_s=3)
    out = tmp_path / f"{theme}.wav"
    report = engine.render(recipe, str(out), 40, progress=False)
    assert report["peak_dbfs"] <= -0.9
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-ss", "10", "-t", "25",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)
    if theme != "ocean":  # waves swing a lot over 25 s
        assert abs(loudness_lufs(x) - engine.DEFAULT_TARGET_LUFS) < 2.0


def test_same_seed_same_recipe():
    assert random_recipe(42) == random_recipe(42)
    assert random_recipe(42) != random_recipe(43)


def test_binaural_keeps_ears_separate():
    recipe = {"seed": 1, "layers": [{"type": "binaural", "params": {"carrier_hz": 200, "beat_hz": 4},
                                     "pan": {"mode": "circle", "width": 1}}]}
    x = _render_raw(recipe, 2, 1)
    spec_l = np.abs(np.fft.rfft(x[:, 0]))
    spec_r = np.abs(np.fft.rfft(x[:, 1]))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    assert freqs[spec_l.argmax()] == pytest.approx(198, abs=1)
    assert freqs[spec_r.argmax()] == pytest.approx(202, abs=1)


def test_thunder_is_mostly_silent_with_events():
    th = layers.Thunder(np.random.default_rng(3), min_gap=20, max_gap=40)
    x = np.concatenate([th(SR * 10) for _ in range(12)])
    active = np.abs(x).max(axis=1) > 0.01
    assert 0.05 < active.mean() < 0.8


def test_sample_layer_loops_without_gaps(tmp_path):
    src = tmp_path / "rain" / "clip.wav"
    src.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=8:c=pink:a=0.3",
                    "-ac", "2", str(src)], check=True)
    layer = layers.Sample(np.random.default_rng(0), str(src), min_segment=3, max_segment=5, crossfade=1)
    x = np.concatenate([layer(SR) for _ in range(30)])
    win = SR // 10
    level = np.sqrt(np.mean(x[: len(x) // win * win].reshape(-1, win, 2) ** 2, axis=(1, 2)))
    assert level.min() > 0.4 * np.median(level)  # no dropouts at the joins

    picked = [random_recipe(seed, "rain", library=str(tmp_path)) for seed in range(10)]
    assert any(l["type"] == "sample" for r in picked for l in r["layers"])


def test_sidecar_written(tmp_path):
    out = tmp_path / "v.m4a"
    engine.render_with_sidecar(random_recipe(2, "fan"), str(out), 5, progress=False)
    side = json.loads((tmp_path / "v.json").read_text())
    assert side["recipe"]["meta"]["theme"] == "fan"
    assert os.path.getsize(out) > 10_000


def _lib(tmp_path, *cats):
    for c in cats:
        d = tmp_path / c
        d.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=6:c=brown:a=0.2",
                        "-ac", "2", str(d / "a.wav")], check=True)
    return str(tmp_path)


def test_library_only_parts_need_the_library(tmp_path):
    assert "stream" not in available_bases()
    with pytest.raises(ValueError):
        random_recipe(1, "stream")
    lib = _lib(tmp_path, "stream", "owl")
    assert "stream" in available_bases(lib)
    r = random_recipe(1, "stream", library=lib, accents=["owl"])
    assert r["meta"]["headline"].endswith("Owl")
    assert r["meta"]["sources"]


def test_tiers_and_headline():
    r = random_recipe(4, "rain", secondaries=["noise_bed", "wind"], accents=["crickets"])
    m = r["meta"]
    assert m["secondaries"] == ["noise_bed", "wind"] and m["accents"] == ["crickets"]
    assert m["headline"].endswith("with Distant Wind and Crickets")
    assert m["combo"] == "rain+noise_bed+wind+crickets"
    levels = [l["level_db"] for l in r["layers"]]
    assert levels[0] == 0 and all(-22 <= v <= -6 for v in levels[1:])
    assert r["layers"][-1]["send"] > 0.5 and r["layers"][-1]["dry"] < 0.6  # accents sit far away


def test_bases_only_offer_known_parts():
    from sleepgen.recipes import ACCENTS, SECONDARIES
    for spec in BASES.values():
        assert set(spec["secondaries"]) <= set(SECONDARIES)
        assert set(spec["accents"]) <= set(ACCENTS)


def test_accents_come_and_go_then_stop():
    spec = {"type": "crickets", "params": {}, "gate": {"on_s": [5, 5], "off_s": [10, 10], "start_s": 2,
                                                       "until_s": 60, "until_fade_s": 10}}
    x = _render_raw({"seed": 1, "layers": [spec]}, 80, 1)
    level = np.sqrt(np.mean(x.reshape(80, SR, 2) ** 2, axis=(1, 2)))
    assert level[:2].max() == 0                 # silent until start_s
    assert level[2:7].min() > 0                 # first burst
    assert level[8:16].max() == 0               # gap
    assert level[60:].max() == 0                # stopped for the rest of the night


def test_wind_and_crickets_are_unit_level():
    for kind in (layers.Wind, layers.Crickets):
        x = kind(np.random.default_rng(2))(SR * 20)
        assert 0.5 < np.sqrt(np.mean(x ** 2)) < 2


def test_reverb_has_no_seams_between_chunks():
    recipe = {"seed": 3, "master": {"reverb": {"rt60": 2.0}},
              "layers": [{"type": "noise", "params": {"color": "pink"}, "send": 0.5},
                         {"type": "crickets", "params": {}, "send": 0.8, "dry": 0.4}]}
    a = _render_raw(recipe, 8, 1.0)
    b = _render_raw(recipe, 8, 3.7)
    from sleepgen import gpu
    # On a GPU the convolution runs in float32 (rounding differs with the block size).
    np.testing.assert_allclose(a, b, atol=2e-5 if gpu.is_gpu() else 1e-8)


def test_leveler_eases_loud_parts_down_only():
    from sleepgen.dsp import Leveler
    rng = np.random.default_rng(0)
    quiet = rng.standard_normal((SR * 10, 2)) * 0.01
    loud = rng.standard_normal((SR * 10, 2)) * 0.2
    lev = Leveler(-26)
    out_q = lev(quiet)
    np.testing.assert_allclose(out_q, quiet)             # never turns quiet audio up or down
    out_l = lev(loud)
    assert np.std(out_l[-SR:]) < 0.6 * np.std(loud)       # loud stretch is eased down


def test_match_measures_and_eq():
    from sleepgen.dsp import MatchEQ
    from sleepgen.match import compare, match_eq, measure
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SR * 12, 2))
    m = measure(x)
    assert compare(m, m)["overall"] == 100.0
    flat = MatchEQ([[100, 0], [1000, 0], [10000, 0]])
    y = np.concatenate([flat(x[i:i + SR]) for i in range(0, len(x), SR)])
    d = flat.delay
    np.testing.assert_allclose(y[d:], x[:-d], atol=2e-3)
    assert all(abs(g) < 1e-9 for _, g in match_eq(m, m))


# -- rain scene (physical, per-drop) ---------------------------------------------------

from sleepgen import rainphys as P  # noqa: E402
from sleepgen.dsp import rms  # noqa: E402
from sleepgen.scene import ENVIRONMENTS, RainScene  # noqa: E402
from sleepgen.surfaces import PLATE_MATERIALS, SHORT_L, make_surface  # noqa: E402


def _scene_out(env, chunk_s, seconds=3.0, seed=3, **kw):
    s = RainScene(np.random.default_rng(seed), environment=env, rain_mm_h=5.0, budget=600, **kw)
    n, chunk, out = int(seconds * SR), int(chunk_s * SR), []
    while n > 0:
        k = min(chunk, n)
        out.append(s(k))
        n -= k
    return np.concatenate(out)


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_rain_scene_is_chunk_invariant_and_level(env):
    """Each environment renders, has no seams at chunk edges, and comes out near unit level."""
    a = _scene_out(env, 1.0)
    b = _scene_out(env, 0.37)
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a, b, atol=2e-4)
    assert 0.3 < rms(a) < 3.0
    # Stereo, not mono: the ears hear different drops at different times.
    assert np.corrcoef(a[:, 0], a[:, 1])[0, 1] < 0.95


def test_rain_scene_is_deterministic():
    # GPU sums aren't bit-identical run to run (summing order), hence the small tolerance.
    np.testing.assert_allclose(_scene_out("window", 1.0, 2.0, seed=9), _scene_out("window", 1.0, 2.0, seed=9),
                               atol=1e-4)
    assert not np.allclose(_scene_out("window", 1.0, 2.0, seed=9), _scene_out("window", 1.0, 2.0, seed=10))


def test_rain_scene_bed_stays_under_the_drops():
    """The diffuse field never outweighs the individually heard drops (that is what reads as static)."""
    for env in ("forest", "roof"):
        s = RainScene(np.random.default_rng(1), environment=env, rain_mm_h=6.0, budget=600, bed_db=-6.0)
        if s.diffuse is None:
            continue
        twin = RainScene(np.random.default_rng(1), environment=env, rain_mm_h=6.0, budget=600, calibrate=False)
        twin.diffuse, twin.room = None, None
        ind = rms(twin(SR * 3))
        bed = float(np.sqrt(np.sum(s.diffuse.h ** 2))) * s.diffuse_gain
        assert bed <= 10 ** (-6 / 20) * ind * 1.6


@pytest.mark.parametrize("kind", ["water", "water_drip", "leaf", "ground", "tile", *PLATE_MATERIALS])
def test_every_surface_gives_sane_modes(kind):
    rng = np.random.default_rng(4)
    surf = make_surface(kind, rng, (3.0, 2.0))
    d = P.sample_diameters(rng, 50, 5.0, P.D_MIN, P.D_MAX)
    v = P.fall_velocity(d, 10.0)
    uv = rng.random((50, 2))
    m = surf.voice(rng, d, v, uv)
    f, tau, amp = m.f, m.tau, m.amp
    assert f.shape == tau.shape == amp.shape and f.shape[0] == 50
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(amp)) and np.all(np.isfinite(tau))
    assert np.all(f > 0) and np.all(f < SR / 2)
    assert np.all(tau > 0)
    # Every mode has died away (-40 dB) inside the window rendered for each drop.
    assert np.all(tau * 4.6 * SR <= surf.L * 1.001)
    assert np.all(tau[:, : m.n_short] * 4.6 * SR <= SHORT_L * 1.001)


def test_rain_physics_basics():
    # Bigger drops fall faster and hit harder; heavier rain has more big drops.
    v = P.terminal_velocity(np.array([0.5, 1.0, 2.0, 4.0]))
    assert np.all(np.diff(v) > 0) and 8.0 < v[-1] < 10.0
    assert P.mp_lambda(10.0) < P.mp_lambda(1.0)
    # A drip from a 2 m branch is slower than terminal velocity.
    assert P.fall_velocity(np.array([4.0]), 2.0)[0] < P.terminal_velocity(np.array([4.0]))[0]
    assert P.minnaert_hz(1.0) == pytest.approx(3260, rel=0.01)  # 1 mm bubble radius


# -- fire (physical) --------------------------------------------------------------------

from sleepgen.fire import FIRE_ENVIRONMENTS, FireScene  # noqa: E402


def _fire_out(env, chunk_s, seconds=4.0, seed=3, **kw):
    s = FireScene(np.random.default_rng(seed), environment=env, budget=600, **kw)
    n, chunk, out = int(seconds * SR), int(chunk_s * SR), []
    while n > 0:
        k = min(chunk, n)
        out.append(s(k))
        n -= k
    return np.concatenate(out)


@pytest.mark.parametrize("env", list(FIRE_ENVIRONMENTS))
def test_fire_is_chunk_invariant_level_and_sleep_safe(env):
    a = _fire_out(env, 1.0, 6.0)
    b = _fire_out(env, 0.37, 6.0)
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a, b, atol=2e-4)
    assert 0.3 < rms(a) < 3.0
    assert np.corrcoef(a[:, 0], a[:, 1])[0, 1] < 0.9
    # No crackle may jump far out of the bed. Sample peaks stay within crest_db (20 dB) of
    # the last couple of seconds' level (a 6 s window can sit a few dB under that)...
    assert 20 * np.log10(np.abs(a).max() / rms(a)) < 24.0
    # ...and what the ear hears as loudness (10 ms) never jumps far over the average.
    ms = np.mean(a[: len(a) // 480 * 480] ** 2, axis=1).reshape(-1, 480).mean(axis=1)
    assert 10 * np.log10(ms.max() / np.mean(a ** 2)) < 13.0
    # ...but they are there: strong crackles stand well clear of the bed.
    assert 20 * np.log10(np.percentile(np.abs(a), 99.99) / rms(a)) > 10


def test_fire_crackles_lead_a_quiet_bed():
    """Heard as cracks and pops over a soft bed, not as static or wind: the crackles are
    louder than the roar, and the ember ticks and hiss stay far down."""
    for env in ("campfire", "fireplace"):
        parts = {}
        for keep in ("crackle", "roar", "fizz", "hiss", None):
            s = FireScene(np.random.default_rng(2), environment=env, budget=600)
            if keep:  # one part alone, unlimited (the limiter follows the whole fire's level)
                s.mute = {k for k in s.g if k != keep}
                s.limiter.threshold = np.inf
            parts[keep] = loudness_lufs(s(int(12 * SR)))
        whole = parts.pop(None)
        assert parts["crackle"] > parts["roar"] + 1.0
        assert parts["crackle"] > whole - 4
        assert parts["fizz"] < whole - 10 and parts["hiss"] < whole - 12


def test_fire_cracks_have_no_pitched_ring():
    """A pitched ring in the low treble, repeated through a crack's sputter, is how bubbling
    liquid sounds: fire v2 had one and was heard as "wet globs". Below 4 kHz every mode of
    a crack is broad (low Q) and gone within a couple of milliseconds."""
    from sleepgen.fire import crack_colour, grain_modes, pop_modes

    rng = np.random.default_rng(0)
    for bright in (0.9, 1.1):
        a = rng.uniform(0.1, 2.5, 3000)
        fc = crack_colour(rng, a, bright)
        for m in (pop_modes(rng, a, bright, fc), grain_modes(rng, a, fc), pop_modes(rng, a, bright)):
            low = (m.f < 4000) & (m.amp != 0)
            assert np.all(np.pi * m.f[low] * m.tau[low] < 5.0)
            assert np.all(m.tau[low] < 1.5e-3)


def test_peak_limiter_holds_its_limit_and_ignores_chunking():
    from sleepgen.dsp import PeakLimiter

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, (SR * 5, 2)) * np.linspace(1, 0.3, SR * 5)[:, None]
    for t in rng.integers(0, SR * 5 - 300, 60):
        x[t:t + 200] += rng.normal(0, 1, (200, 2)) * np.exp(-np.arange(200) / 40)[:, None] * rng.uniform(1, 8)

    def run(chunk, **kw):
        lim, out = PeakLimiter(4.0, **kw), []
        for i in range(0, len(x), chunk):
            out.append(lim(x[i:i + chunk]))
        return np.concatenate(out)

    from scipy.ndimage import maximum_filter1d

    d = 2 * PeakLimiter(4.0).W  # the output is delayed by this many samples
    calm = maximum_filter1d(np.abs(x).max(axis=1), 4 * d + 1)[:-d] < 1.0  # no peak anywhere near
    assert calm.sum() > SR // 10
    for kw in ({}, {"relative_s": 1.0}):
        a = run(SR, **kw)
        np.testing.assert_allclose(a, run(7777, **kw), atol=1e-12)
        assert np.abs(a).max() <= 4.0 + 1e-9
        # Away from the peaks nothing changes.
        np.testing.assert_allclose(a[d:][calm], x[:-d][calm], atol=1e-12)
    # Following the recent level, it holds a lull's pops lower than the fixed limit does.
    assert np.abs(run(SR, relative_s=1.0)[-SR:]).max() < np.abs(run(SR)[-SR:]).max()


def test_fire_is_deterministic_and_woods_differ():
    np.testing.assert_allclose(_fire_out("campfire", 1.0, 2.0, seed=4), _fire_out("campfire", 1.0, 2.0, seed=4),
                               atol=1e-4)
    soft = FireScene(np.random.default_rng(1), environment="campfire", wood="softwood", budget=600)
    hard = FireScene(np.random.default_rng(1), environment="campfire", wood="hardwood", budget=600)
    assert soft.crackle_rate > hard.crackle_rate


def test_fire_recipes_fit_the_setting():
    for seed in range(40):
        r = random_recipe(seed, "fire")
        env = r["layers"][0]["params"]["environment"]
        assert r["layers"][0]["type"] == "firescene"
        if FIRE_ENVIRONMENTS[env]["outdoor"]:
            assert "light_rain" not in r["meta"]["secondaries"]
        else:
            assert not set(r["meta"]["accents"]) & {"crickets", "owl", "frogs"}
        r = random_recipe(seed, "rain")
        if r["layers"][0]["params"]["environment"] in ("lake", "forest", "tent"):
            assert "fire" not in r["meta"]["secondaries"]


def test_filter_bank_matches_direct_copies():
    """The FFT filter bank gives the same sum as copying each impulse's response in place,
    across block edges, for per-impulse ears (pools) and fixed per-panel ears (sheets)."""
    import torch
    from sleepgen import gpu
    from sleepgen.scene import FilterBank
    rng = np.random.default_rng(0)
    dev = gpu.device()
    P_, K, L, N, Q = 3, 4, 700, 60000, 400
    waves = rng.standard_normal((P_ * K, L)).astype(np.float32)
    wt = torch.as_tensor(waves, device=dev)
    member = rng.integers(0, P_ * K, Q)
    tol = 2e-4 if gpu.is_gpu() else 2e-5

    # Per-impulse ears.
    pos = rng.integers(0, N, (2, Q))
    amp = rng.standard_normal((2, Q))
    buf = torch.zeros(2, N + FilterBank(wt, dev).span + 64, device=dev)
    FilterBank(wt, dev).render(buf, pos, member, amp)
    ref = np.zeros((2, buf.shape[1]))
    for e in range(2):
        for q in range(Q):
            ref[e, pos[e, q]:pos[e, q] + L] += amp[e, q] * waves[member[q]]
    np.testing.assert_allclose(buf.cpu().numpy(), ref, atol=tol * np.abs(ref).max())

    # Fixed gain per ear for each resonance and a delay per ear for each group of K.
    gains = rng.uniform(0.2, 1.0, (P_, 2, K))
    delays = rng.integers(0, 30, (2, P_))
    fb = FilterBank(wt, dev, gains, delays)
    p1 = rng.integers(0, N, (1, Q))
    a1 = rng.standard_normal((1, Q))
    buf = torch.zeros(2, N + fb.span + 64, device=dev)
    fb.render(buf, p1, member, a1)
    ref = np.zeros((2, buf.shape[1]))
    for q in range(Q):
        g, k = divmod(member[q], K)
        for e in range(2):
            s = p1[0, q] + delays[e, g]
            ref[e, s:s + L] += a1[0, q] * gains[g, e, k] * waves[member[q]]
    np.testing.assert_allclose(buf.cpu().numpy(), ref, atol=tol * np.abs(ref).max())


def test_rain_pool_paths_agree():
    """Drops copied from the pool and drops rendered through the filter bank sound the same
    (same level and spectrum per band), so the faster path can be chosen freely."""
    from scipy import signal
    from sleepgen import gpu
    from sleepgen.scene import RainScene

    def bands(force_fft):
        old = gpu.scatter_rate
        gpu.scatter_rate = (lambda: 1.0) if force_fft else (lambda: 1e30)
        try:
            s = RainScene(np.random.default_rng(4), environment="forest", rain_mm_h=4, budget=1500, calibrate=False)
        finally:
            gpu.scatter_rate = old
        s.diffuse = None
        s.room = None
        x = s(SR * 6)
        f, p = signal.welch(x.mean(1), SR, nperseg=4096)
        return np.array([10 * np.log10(p[(f > b / 1.41) & (f < b * 1.41)].mean()) for b in (250, 1000, 4000)])

    np.testing.assert_allclose(bands(True), bands(False), atol=1.5)
