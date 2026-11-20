# 8dAudio / sleepgen

`sleepgen` makes long (6 to 12 hour), calm soundscapes for a sleep channel: rain, thunder, ocean,
fan, wind, crickets, brown/pink noise, binaural beats and soft drones, with slow 8D-style movement.
Every video gets a randomly generated recipe, so no two uploads are the same mix.

## How a mix is built: three tiers

| Tier | How many | Level | Examples |
|---|---|---|---|
| **Base** | 1, always on | 0 dB | rain, fire, ocean, brown/pink noise, fan, solfeggio drone, stream* |
| **Secondary** | 0 to 2, always on | 6 to 14 dB under | wind, noise bed, binaural beats, drone, soft rain, fire, stream* |
| **Accent** | 0 to 1, comes and goes | 8 to 14 dB under | crickets, distant thunder, owl*, frogs*, wind chimes* |

\* needs recordings in the sound library.

Each base lists the secondaries and accents that suit it (`python -m sleepgen themes`). Accents play
only in the first 45 to 60 minutes and then fade out, so nothing can startle someone who is already
asleep. Accents are night sounds only: no daytime birdsong. The title headline is built from the
tiers, e.g. "Gentle Rain with Distant Wind and Crickets".

Needs Python 3.10+, `ffmpeg` on the PATH, and `pip install -r requirements.txt`. Rain is synthesised
with PyTorch and is meant to run on a GPU (see "Running on the GPU" below).

## Quick start

```bash
python -m sleepgen themes                                  # list themes
python -m sleepgen previews -d previews --seconds 90       # a short clip of every theme
python -m sleepgen previews --rain -d previews             # rain in every environment
python -m sleepgen previews --fire -d previews             # fire in every setting
python -m sleepgen bench                                   # how fast this machine renders rain
python -m sleepgen random -o out/video1.m4a --hours 10     # random theme, 10 hours
python -m sleepgen random --theme rain --secondary wind --accent crickets -o out/v.m4a --hours 8
python -m sleepgen recipe --theme ocean -o ocean.json      # save a recipe, edit it by hand...
python -m sleepgen render ocean.json -o out/ocean.m4a --hours 12   # ...then render it
```

Each render writes `<name>.json` beside the audio with the full recipe and a render report. The
recipe's `meta` block (theme, headline, features) feeds video titles and the weekly analytics report.

## How it keeps things right for sleep

- **Steady, quiet volume.** Every mix is measured and set to -26 LUFS, a slow leveler eases down
  anything that swells more than 3 dB above that (big waves, thunder), and a soft limiter sits at
  -1 dBFS. YouTube only ever turns loud videos down, never quiet ones up.
- **One shared space.** All layers feed a convolution reverb with a natural, darkening tail. The
  base is mostly direct, secondaries sit further back, and accents are mostly reverb, so a cricket
  sounds like it's out in the garden, not in your ear.
- **No surprises.** Rain gets lighter and heavier only over minutes, thunder is rare, low and
  distant, and slow level drift keeps it alive without anything sudden.
- **No audible loop.** Everything is generated live, and library recordings are played as random
  stretches with crossfades between them.
- **8D movement** (`pan.mode`: `sweep` or `circle`) is gentle by default (width 0.3 to 0.75) so it
  stays comfortable for someone lying on one ear. Binaural beats are never panned.
- **Quality over speed.** Rain is built from up to 16,000 individually modelled drops per second on a
  GPU (see below), each with its own size, speed, landing spot and sound. Audio streams to ffmpeg in
  10 second chunks, so memory stays flat. Output is 384 kbps AAC (`.m4a`, YouTube's recommended rate)
  or lossless FLAC (`.flac`, 16 bit), which encodes about ten times faster.

## Rain: physical scenes

Rain is not a noise preset. `rainscene` places a listener in an environment and simulates the rain
falling on everything around them, drop by drop:

| Environment | What you hear |
|---|---|
| `roof` | A loft room right under a tiled roof (softened through it), a side window, a gutter drip, the garden outside |
| `tin_roof` | Corrugated metal sheets ringing above you, drips off the eaves |
| `window` | Wind-driven rain on glass right beside you, a stone sill, the street and rooftops beyond |
| `lake` | Rain on open water (tiny bubbles "fizz"), the shore, the treeline across the lake |
| `forest` | Leaves of the canopy and understorey, heavy drips falling from the branches, the forest floor |
| `tent` | Rain on taut fabric overhead, drips from the trees, wet grass, the woods around |

What is modelled:

- **Drops.** Sizes follow the Marshall-Palmer distribution for the rain rate, and each falls at its
  terminal speed (drips from branches and eaves are slower and bigger). Heavier rain means more drops
  *and* bigger drops, as in real rain.
- **Impacts.** Each drop is a set of decaying resonances worked out from its size and speed: a short
  click of contact, plus whatever the surface does next. Water traps a small bubble that rings
  (Minnaert frequency) with a rising chirp; leaves and tiles give a short woody or stony ring.
  Metal sheets, window panes and tent fabric are split into panels, each with its own vibration modes,
  and a drop excites each mode by how its shape moves at the landing point.
- **Fast without shortcuts in the sound.** Every surface's resonances come from large pools rendered
  once, and each drop strikes its own few members with its own strengths per ear (thousands of members
  per band, closer together than the ear can tell apart). A sheet's modes are fixed, so each panel is a
  bank of resonators heard from its centre (a ringing sheet radiates from its whole surface). Long rings
  are rendered as FFT filter banks: a drop only places one impulse per resonance it strikes, and each
  resonance's impulses are convolved with its response in blocks. The cost then depends on how many
  resonances there are, not on how many drops hit them. Copying every drop's ring into the soundtrack
  was the limit before (the GPU tops out at about 3 billion additions a second).
- **Space.** Every drop has a position in 3D. Its sound reaches each ear with the right delay, level,
  head shadow, distance loss and air absorption, and through a ceiling, wall or glass where there is
  one. All of it shares a small room or outdoor reverb. Listen on headphones.
- **The rest of the rain.** The thousands of drops too far away or too small to render one by one are
  added as a matching background whose spectrum is measured from the same drop models, kept under the
  close drops (`bed_db`) so the scene never turns into static.
- **Slow changes.** Rain rate and wind wander over minutes, with lighter and heavier spells across the
  night. Nothing jumps.

Recipe parameters: `environment`, `rain_mm_h` (light 1 to 2.5, moderate 2.5 to 7, heavy 7+), `wind_ms`,
`variability`, `bed_db`, `diffuse_db`. The code is in `sleepgen/scene.py` (environments and rendering),
`sleepgen/surfaces.py` (what a drop sounds like on each surface) and `sleepgen/rainphys.py` (the physics).

## Fire: physical, crackle by crackle

`firescene` builds a wood fire from what a fire physically does, in four settings:

| Setting | What you hear |
|---|---|
| `campfire` | Lying a couple of metres from a campfire on open ground |
| `fireplace` | An open fireplace across a living room: brick hearth, chimney draw, soft room |
| `wood_stove` | A cast iron stove with a glass door: muffled crackles, a steady draw, ticking iron |
| `bonfire` | A big bonfire a few metres off: deep, slow roar and busy crackling |

- **Flame roar.** Turbulent combustion as a soft, low rush. Flames "puff" at a rate set by the fire's
  size (about 1.5 / sqrt(diameter) Hz), so a bonfire breathes slowly and a stove quickly, with random
  flicker on top.
- **Crackles.** Each crack of drying wood or burst of sap is an event placed in 3D inside the fire bed:
  a rough burst of micro-fractures over a few milliseconds, often with a short woody ring. They come in
  clusters, with a heavy-tailed spread of sizes and busy and quiet spells.
- **Ember fizz.** Thousands of tiny ticks a second, rendered one by one (the GPU budget sets how many).
- **Steam hiss.** A breathy hiss that grows with damp wood, and now and then a soft, gliding "singing" jet.
- **Settling logs.** Every few minutes, a soft cluster of ticks and a low knock, and the fire flares up a
  little for a while afterwards.
- **Sleep safety.** A strong crackle peaks about 12 dB above the fire's average level, and none can
  peak more than 19 dB above it (`crest_db`).

Parameters: `environment`, `intensity` (0.5 gentle to 1.35 roaring), `wood` (`softwood` pops and hisses,
`mixed`, `hardwood` is steadier and deeper), `dampness` (0 to 1), `variability`, `crest_db`. Fire sets
its own exclusions in recipes: no rain around a campfire, no crickets indoors. Under rain on a roof or
window, fire can also be a secondary ("Rain on the Window with Crackling Fire").

## Running on the GPU

`--device` picks where synthesis runs: `auto` (default) tries a CUDA-style GPU, then DirectML, then
the CPU. `--quality` sets how many drops per second are modelled individually (default 16,000 on a GPU,
2,500 on a CPU). More is more detail; the rest still comes from the matching background.

On Windows with an AMD Radeon RX 9000 card (e.g. the RX 9060 XT), either:

1. **AMD's PyTorch for Windows (ROCm), preferred.** Install it with AMD's "PyTorch on Windows" guide for
   your Adrenalin driver version. The card then shows up to PyTorch as a CUDA device, and
   `python -m sleepgen bench` prints `device: GPU: AMD Radeon RX 9060 XT`.
2. **DirectML, fallback.** `pip install torch-directml` (it installs the PyTorch version it needs).
   Works with any DirectX 12 GPU, usually slower than ROCm.

Check it with `python -m sleepgen bench`, which renders a little rain in every environment and fire in
every setting, and projects how long a 10 hour video takes. `bench --full flac --seconds 300` times the
whole pipeline (master chain and encoding too). Force a device with `--device cuda` / `directml` / `cpu`
(or `SLEEPGEN_DEVICE`).

What runs where: drops, sheets and fire crackles are synthesised on the GPU, and so are the long
convolutions (room and master reverbs, the rain's background bed, the fire's roar and hiss). The CPU
prepares the next chunk's drops while the GPU renders the current one, and ffmpeg encodes on its own
CPU core from a queue, so encoding and rendering run side by side. On a 6-core CPU two renders at once
use the machine best.

## Sound library (optional)

Point `--library` at a folder with one subfolder per category and sleepgen will blend real
recordings into matching themes:

```
sounds/
  rain/  rain_roof/  rain_window/  ocean/  stream/  fire/  wind/      (blended into bases and secondaries)
  crickets/  owl/  frogs/  chimes/                                   (accents)
```

Only use recordings you may use commercially: CC0 / public domain (e.g. Freesound filtered to
"Creative Commons 0") or Pixabay's licence. Keep a note of each file's source. A recording should be
at least 30 seconds long and free of voices, cars, birds and other sharp sounds.

## Recipe format

```json
{
  "seed": 7,
  "master": {"target_lufs": -26, "lowpass_hz": 9000, "fade_in_s": 20, "fade_out_s": 45,
             "reverb": {"rt60": 2.2, "predelay_ms": 25, "damping": 0.6}},
  "layers": [
    {"type": "rainscene", "level_db": 0, "send": 0.05,
     "params": {"environment": "window", "rain_mm_h": 4.0, "variability": 0.3}},
    {"type": "wind", "level_db": -10, "params": {"strength": 0.5}, "send": 0.35},
    {"type": "crickets", "level_db": -18, "params": {"count": 3}, "send": 0.75, "dry": 0.4,
     "gate": {"on_s": [20, 60], "off_s": [30, 120], "start_s": 40, "until_s": 3000, "until_fade_s": 300}}
  ],
  "meta": {"base": "rain", "secondaries": ["wind"], "accents": ["crickets"],
           "headline": "Rain with Distant Wind and Crickets"}
}
```

Layer types: `rainscene`, `firescene`, `noise`, `rain` (older, simpler rain), `thunder`, `ocean`, `fan`, `wind`, `crickets`, `binaural`, `drone`, `sample`.
`send` (0 to 1) feeds the shared reverb and `dry` (0 to 1) lowers the direct sound to push a layer
further away. `gate` makes any layer come and go (`on_s` / `off_s` ranges) and stop for good at `until_s`.
See `sleepgen/layers.py` for each one's parameters.

## Matching a reference recording

`python -m sleepgen match reference.mp3 --base rain` tunes a base sound's settings and a matching EQ
until its tone, drop density, graininess and swell are as close as possible to the reference, and
writes the recipe, both clips and a similarity report to `matched/`.

## Tests

```bash
pip install pytest && python -m pytest -q
```

## Older 8D web app

The earlier React + Node/XAMPP 8D audio app is not in this repository.
