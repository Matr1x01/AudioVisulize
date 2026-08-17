# Audio Visualizer — how the whole project works

A from-the-ground-up walkthrough of what this program is, how audio gets from
your speakers onto the screen, and how every piece of the rendering engine
fits together. Written to be read top to bottom; later sections assume the
earlier ones.

For a narrower log of one specific round of changes (bloom pass, vectorscope,
palettes, a bass-band bug fix), see
[`visualization-quality-upgrades.md`](visualization-quality-upgrades.md).
This document instead covers the whole program as it stands.

## Contents

1. [What this is](#1-what-this-is)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Audio capture & analysis — `src/lib.rs`](#3-audio-capture--analysis--srclibrs)
4. [The rendering engine core — `src/render.rs`](#4-the-rendering-engine-core--srcrenderrs)
5. [GPU setup & the frame lifecycle — `src/main.rs`](#5-gpu-setup--the-frame-lifecycle--srcmainrs)
6. [Bloom post-process — `src/bloom.rs`](#6-bloom-post-process--srcbloomrs)
7. [The visualizer system — `src/visuals.rs`](#7-the-visualizer-system--srcvisualsrs)
8. [The debug HUD — `src/hud.rs`](#8-the-debug-hud--srchudrs)
9. [Main event loop, start to finish](#9-main-event-loop-start-to-finish)
10. [Testing strategy](#10-testing-strategy)
11. [Runtime configuration & keybindings](#11-runtime-configuration--keybindings)
12. [Dependencies](#12-dependencies)
13. [File map](#13-file-map)

---

## 1. What this is

A real-time **system-audio visualizer**: it captures whatever is playing
through your speakers (not a microphone — the actual output mix), runs it
through an FFT, and draws one of six full-window visualizations that react
to it, rendered with the GPU via [`wgpu`](https://wgpu.rs/). It's a single
Rust binary (`audio_processor`) with no UI framework — a bare `winit` window,
one draw call per frame, and a bitmap-font debug overlay for diagnostics.

Run it, and system audio starts animating a chosen visual immediately; `Space`
or the arrow keys cycle between six styles, number keys jump directly to one,
and a handful of function keys expose frame-timing diagnostics.

## 2. Architecture at a glance

```
┌─────────────── audio thread (cpal capture callback) ───────────────┐
│  speakers → PulseAudio/PipeWire monitor → cpal stream → ring buffer│
└──────────────────────────────┬───────────────────────────────────┘
                                │ Mutex<Vec<f32>>
┌───────────────────────────── analysis thread ──────────────────────┐
│  ~60 Hz: drain ring buffer → Hann window → FFT (realfft) →         │
│  magnitude → dB → normalize 0..1 → bass energy → AudioMetrics      │
└──────────────────────────────┬───────────────────────────────────┘
                                │ Mutex<AudioMetrics>
┌───────────────────────────── render thread (winit event loop) ─────┐
│  snapshot metrics → log-band grouping (update_bands) → FrameData → │
│  visualizer.draw() → MeshBuilder (triangles) → HUD overlay →       │
│  GPU: scene pass → bloom (threshold/blur/composite) → present      │
└─────────────────────────────────────────────────────────────────┘
```

Three threads, two handoffs, both guarded by a plain `std::sync::Mutex`
rather than a lock-free structure — deliberately: the code measures how long
each side blocks on that lock (`AudioProbe`, the HUD's `LOCK` and `VSYNC
BLOCK` figures) instead of assuming a fancier structure would make contention
a non-issue.

**Two logical halves, split across a library and a binary:**

- **`audio_processor`** (`src/lib.rs`) — pure audio: device capture, FFT,
  band grouping. No `wgpu`, no rendering types. Depends on `cpal` and
  `realfft` only.
- **The binary** (`src/main.rs` + `render.rs`, `visuals.rs`, `hud.rs`,
  `bloom.rs`) — everything graphics. Depends on `wgpu` and `winit`, and pulls
  in `audio_processor` as a library the way any consumer would.

This split exists so the audio pipeline's correctness (band math, FFT
windowing, the attack/release envelope) can, in principle, be reasoned about
and tested without a GPU or a display attached — `cargo test` runs `lib.rs`'s
unit-test target with zero rendering dependencies pulled in.

## 3. Audio capture & analysis — `src/lib.rs`

### Getting *system* audio, not a microphone

`cpal`'s ALSA backend can't enumerate PulseAudio/PipeWire monitor sources —
it only sees generic plugin names (`"pulse"`, `"default"`) and raw `hw:`
device nodes, none of which carry the system output mix. `resolve_monitor_source()`
works around this by shelling out to `pactl get-default-sink`, appending
`.monitor` to the sink name, and exporting that as the `PULSE_SOURCE`
environment variable *before* any audio device opens — the ALSA→PulseAudio
plugin reads that variable when it initializes and routes accordingly.
`AUDIO_SOURCE=<name>.monitor` overrides this explicitly (e.g. for capturing
a specific app's sink-monitor instead of the whole desktop mix). If neither
resolves, capture silently falls back to the default input device — usually
an actual microphone — with a warning printed to stderr.

### The capture callback (real-time thread)

`cpal::Stream`'s callback runs on an OS real-time-priority audio thread. Its
job is deliberately tiny: append the incoming interleaved PCM samples to a
`Mutex<Vec<f32>>` ring buffer, trim it back down to `FFT_SIZE * channels`
samples from the front if it's grown past that, and update a handful of
lock-free atomic counters (`AudioProbe`) for instrumentation — call count,
longest callback duration, longest time spent *waiting* for the ring
buffer's mutex. That last one is the interesting one: if the analysis or
render thread ever holds that mutex too long, this is where it would show up
as audio glitching, and the probe exists specifically to make that
measurable instead of just audible.

`AudioProbe`'s fields are atomics rather than being folded into
`AudioMetrics` for a specific reason: the whole point is to measure how long
the real-time thread blocks acquiring a lock, so the measurement itself must
not require taking one.

### The analysis thread (~60 Hz)

A second thread wakes every 16 ms, and — if the ring buffer holds a full
`FFT_SIZE` window — does the actual signal processing:

1. **De-interleave** stereo PCM into separate left/right buffers, and also
   into `waveform` (mono-summed, `(l+r)*0.5`) and, separately,
   `left_waveform`/`right_waveform` (per-channel, for the stereo Vectorscope
   visual — see §7).
2. **Hann-window** each channel (`0.5 * (1 - cos(2π·i/(N-1)))`) before the
   FFT, to suppress spectral leakage from the window boundary — the raw
   (unwindowed) `waveform` copy is kept separately because time-domain
   visuals need the untapered signal; windowing would show up as the trace
   visibly collapsing to zero at both edges of the screen.
3. **FFT** via `realfft` (a real-input specialization of `rustfft` that's
   roughly 2x faster than a general complex FFT for real signals), producing
   `NUM_BINS = FFT_SIZE / 2 = 512` complex bins per channel.
4. **Magnitude → dB → normalized 0..1.** Linear magnitude reads almost
   perceptually flat-lined — nearly every bar would sit near zero — so
   magnitude is converted to dB and linearly mapped against a
   `DB_FLOOR = -90.0` noise floor: `(1.0 - db / DB_FLOOR).clamp(0, 1)`.
5. **Bass energy**: the mean normalized level across bins in the 20–250 Hz
   range, averaged over both channels — this drives the background tint and
   several visuals' bass-reactive motion (see §7).
6. Publish the whole result into `Arc<Mutex<AudioMetrics>>`, along with
   instrumentation: a `seq` counter bumped only when genuinely new audio
   arrived (so the renderer can tell "recomputed the same window" from
   "actually new spectrum"), `produced_at` (an `Instant`, for measuring
   spectrum *age* at draw time — the metric that actually exposes
   producer/consumer clock drift, where comparing raw *rates* would hide it
   since 60.0 Hz vs. 61.0 Hz both look "healthy"), and `hop_frames` (how many
   *actual* input frames advanced since the last pass — not a configured
   value, but whatever the wall clock and callback cadence produced,
   because that's precisely the kind of drift this instrumentation exists to
   catch).

### Log-spaced band grouping

The FFT produces 512 linearly-spaced bins, but the display only ever shows
`NUM_BANDS = 64` bars/points, spaced **logarithmically** in frequency —
because pitch is perceived logarithmically (each octave "sounds like" equal
width), and a linear mapping would cram every bass note into the first two
columns while spending most of the display on near-silent treble.

`log_band_edges(sample_rate)` computes each band's `(lo, hi)` span as
**continuous, fractional** bin positions (`BandEdge = (f32, f32)`), not
rounded integers. This matters specifically at the bass end: with a
1024-point FFT, one bin covers ~43–47 Hz (`bin_hz = sample_rate / FFT_SIZE`),
but an octave-wide log band down there can be narrower than that — so
integer rounding would make dozens of consecutive bands resolve to the exact
same one or two bins, and therefore read back bit-for-bit identical values
(they'd all move in perfect lockstep). `update_bands()` samples each band's
continuous span at four evenly-spaced points, **linearly interpolating** the
magnitude spectrum between adjacent bins at each fractional position, and
keeps the peak of those samples — "peak, not mean" is preserved from the
original design (a single strong tone shouldn't be averaged away by quiet
neighbors sharing its band), while the interpolation gives every band a
value that depends on exactly where within a shared bin it falls, instead of
every such band reading the identical rounded-off number.

### Attack/release smoothing

Each band's displayed level follows a fast-attack/slow-release envelope, in
**wall-clock time** rather than a fixed per-frame ratio:

```rust
let tau = if peak > levels[i] { BAND_ATTACK_TAU } else { BAND_RELEASE_TAU };
let alpha = 1.0 - (-dt / tau.max(1e-4)).exp();
levels[i] += (peak - levels[i]) * alpha;
```

`BAND_ATTACK_TAU = 0.05s`, `BAND_RELEASE_TAU = 0.22s` — reacts quickly to a
transient (about 63% of the way there within 3 frames at 60 Hz) without
snapping in a single frame, which is what keeps the display from reading as
jittery noise while still tracking beats crisply. This being a proper
exponential time-constant (not "decay 20% per frame") is what makes the
motion look identical whether the display is running at 30, 60, or 144 Hz —
`main.rs` computes a real `dt` between redraws and passes it in every frame.

## 4. The rendering engine core — `src/render.rs`

This module is the drawing toolkit every visualizer is built from: one
`MeshBuilder` accumulating indexed triangles, painter's-algorithm ordering
(no depth buffer — later geometry always paints over earlier geometry), and
a handful of primitives (strokes, filled outlines, gradient fills, splines)
that all visuals compose from.

### Antialiasing model: feathered fringes, not MSAA

MSAA alone can't make this look smooth — at 4x it resolves to four coverage
levels (two bits of edge gradient), which is visibly stepped on the
near-horizontal thin strokes this app is mostly made of. Instead, **every**
filled edge — stroke sides, round caps, filled-outline perimeters — carries
its own **1-pixel alpha-feathered fringe**: the outline is extruded outward
by exactly one pixel to vertices whose alpha is zero, and the GPU's linear
interpolation across that fringe produces a smooth analytic coverage ramp.
This is what actually removes the jaggies; MSAA (when enabled at all) only
cleans up whatever's left. Measured on Ridge Bed, the heaviest visual, this
is why the app defaults to **1x MSAA (effectively off)**: 4x cost roughly
double the GPU time (7.2 ms vs 3.9 ms mean) and dropped frames the 1x
configuration didn't, for a side-by-side-indistinguishable result. Override
with the `MSAA=1|2|4|8` environment variable.

### Color model: premultiplied alpha, and additive glow for free

Vertex colors are stored **premultiplied** (`rgb *= alpha`), matching the
pipeline's `PREMULTIPLIED_ALPHA_BLENDING` blend state, and the fragment
shader (`shader.wgsl`) passes them through completely untouched. Two
consequences fall out of this for free:

- A feathered fringe vertex is exactly `[0, 0, 0, 0]` — transparent black —
  which contributes nothing whether the blend mode composites "over" or
  additively. One CLEAR constant works everywhere.
- A vertex with `rgb > 0` and `alpha == 0` composites as `dst + src`, i.e.
  **additive**, under the same premultiplied-alpha blend state, with no
  second pipeline and no blend-state switch. This is the entire mechanism
  behind every glow effect in the app (`glow_stroke`/`glow_stroke_with`):
  stack the same polyline several times at increasing width and *decreasing*
  alpha, each pass additive, and they accumulate into a soft halo instead of
  occluding one another the way ordinary "over" blending would.

Callers everywhere else in the codebase pass ordinary straight-alpha colors
(`[r, g, b, a]`); the premultiply conversion happens once, at the point each
primitive is emitted.

### Splines: monotone cubic (Fritsch–Carlson), not Catmull–Rom

`resample_monotone()` turns a sparse polyline (one point per band, 64 total)
into a smooth curve by fitting a **monotone cubic Hermite spline** through
it. This was chosen specifically over an unconstrained cubic (Catmull–Rom)
because an unconstrained cubic *overshoots*: interpolating between a tall
band and its silent neighbor produces a curve that dips below the baseline
between them — visible as the silhouette punching through its own floor.
Fritsch–Carlson tangent-limiting guarantees the curve never overshoots past
the data on any span where the data itself is monotone, which removes that
artifact entirely. Every band-driven visual (Ridge Bed, Spectrum Area) routes
its raw per-band points through this before drawing.

### Primitives

- **`stroke`/`stroke_with`** — a thick antialiased polyline with miter joins
  (truncated past a length limit, since a miter's length grows to infinity
  as an angle closes up) and round caps. Consecutive segments *share* their
  offset vertices at each joint rather than overlapping, which is what
  avoids the "beading" artifact overlapping alpha-blended joints produce.
  `stroke_with` takes a closure so color can vary continuously along the
  polyline's length — this is how every visual's height-driven gradient
  actually gets onto a curve.
- **`fill_outline`/`round_rect`** — feathered convex fills; a rounded
  rectangle is built as a fan of arc points around a convex outline and fed
  through `fill_outline`.
- **`fill_under`** — fills between a curve and a horizontal baseline as one
  connected triangle strip, sharing vertices between adjacent spans so
  alpha-blended edges don't double-blend into visible seams. Deliberately
  **not** feathered — every caller either strokes the same curve directly
  over this fill's top edge, or butts it against another identically-colored
  fill, so a ramp there would be invisible at best and a seam at worst.
- **`ring`** — a closed stroke approximating a circle, with segment count
  derived from the radius in pixels so it never facets visibly nor
  over-tessellates a small ring.
- **`MIN_STROKE_PX = 1.25`** — sub-pixel stroke requests are widened to this
  floor and their alpha scaled down by the same factor, holding total
  emitted light constant. Below roughly one pixel, a stroke's rasterized
  coverage varies with its sub-pixel position, so a thin moving line would
  otherwise shimmer and intermittently vanish.

All widths, radii and feather amounts are specified in **pixels**, not
normalized device coordinates — `MeshBuilder` holds the current framebuffer
size specifically so this conversion happens in one place; specifying
these in NDC directly is what let thin strokes fall below a pixel on short
windows and start shimmering, before this design.

## 5. GPU setup & the frame lifecycle — `src/main.rs`

`GpuState` owns every `wgpu` resource: the surface, device, queue, the one
render pipeline the scene draws with, growable vertex/index buffers, the
optional MSAA target, the `Bloom` post-process chain (§6), and an optional
GPU timer.

### Setup

- Requests a `HighPerformance` adapter, and only requires
  `wgpu::Features::TIMESTAMP_QUERY` if the adapter actually advertises it —
  not universally available, so GPU-side frame timing degrades gracefully to
  "unavailable" in the HUD rather than failing device creation.
- Picks an **sRGB** surface format explicitly (falling back to whatever's
  first if none is sRGB) — colors are decoded from sRGB hex literals into
  linear space on the CPU (`srgb_hex`) specifically because the surface will
  re-encode on store, and skipping either half of that round trip produces
  either washed-out or oversaturated output.
- One render pipeline, `PrimitiveTopology::TriangleList`, no depth buffer,
  `PREMULTIPLIED_ALPHA_BLENDING`, vertex layout matching `Vertex` (`position:
  vec2, color: vec4`) exactly.
- Vertex/index buffers start at fixed capacities (64K vertices, 256K
  indices) and **grow on demand** (`ensure_capacity`, doubling to the next
  power of two) rather than being sized for a worst case up front — spline
  resampling makes per-frame vertex counts data-dependent, so this is
  load-bearing, not defensive padding.

### `GpuTimer` — async GPU timestamp readback

wgpu timestamp queries are asynchronous: you write two timestamps into a
query set during a render pass, resolve them into a buffer, then map that
buffer back to the CPU — and blocking on that map would stall the very
pipeline you're trying to measure. `GpuTimer` handles this with a one-query-
in-flight state machine: `resolve_into()` queues the resolve+copy (skipped
if a previous read hasn't landed yet), `map()` kicks off an async map whose
callback only flips an `AtomicBool` (never touches the mapped memory itself,
since the callback may run on a different thread than the render loop),
and `collect()` polls that flag and only then reads the timestamps back —
called at the *start* of the next frame's `render()`, so it's always
draining the *previous* frame's numbers, never blocking on the current one.

### `render()` — one frame, end to end

1. Collect the previous frame's GPU timing (non-blocking, see above).
2. Grow vertex/index buffers if this frame's mesh needs more room.
3. `surface.get_current_texture()` — this is the vsync block on a Fifo
   swapchain, and it's timed **separately** from everything else
   (`wait_ms`), because folding it into "CPU frame time" would make every
   frame read as ~16.6 ms regardless of how much actual work happened.
4. Upload the frame's vertex/index buffers.
5. **Scene render pass** — draws into the bloom module's offscreen
   `scene_view` (§6), never directly into the swapchain. With MSAA enabled,
   the pass targets the multisampled texture and resolves into `scene_view`;
   without it, straight into `scene_view`.
6. Resolve GPU timestamps for *this* pass into the timer's buffer.
7. `self.bloom.render(...)` — runs the four-pass bloom chain, whose last
   pass is the only one that writes the actual swapchain image (§6).
8. Submit the single command buffer, `frame.present()`, and kick off the
   async GPU-timer map for next frame to collect.

`RenderTiming` separates `wait_ms` (vsync), `submit_ms` (buffer upload +
encode + submit), and `gpu_ms` (actual GPU execution, when available) —
three genuinely different costs that would otherwise all get lumped into one
misleading "frame time" number.

## 6. Bloom post-process — `src/bloom.rs`

Every glow in the visuals themselves is faked **per-stroke**
(`glow_stroke`/`glow_stroke_with` in §4): the same line drawn several times,
wider and dimmer each time, additively blended into a halo around *that one
curve*. It's cheap and reads well for a single line, but it can't do what a
real bloom does — let a bright *shape* raise the brightness of pixels around
it, including across separate, unconnected shapes (a bright bar and its
neighbor never brighten each other this way).

`Bloom` adds a genuine screen-space post-process pass on top, run every
frame in the same command encoder immediately after the scene pass:

```
scene draw ──► scene_view (offscreen, full res)
                    │
                    ├─► threshold ──► bright_view (offscreen, HALF res)
                    │                      │
                    │                      ├─► blur H ──► blur_a (half res)
                    │                      │                  │
                    │                      │                  ├─► blur V ──► blur_b (half res)
                    │                      │                                       │
                    └──────────────────────┴───────────────────────────────────────┴─► composite ──► swapchain
```

1. **Threshold** — subtracts a luminance threshold (`0.55`) from the
   full-resolution scene and clamps negative results to zero, writing into a
   **half-resolution** target. Sampling the full-res source into a half-res
   render target gets free bilinear downsampling from the texture sampler —
   no separate downsample pass needed.
2. **Blur horizontal**, then **blur vertical** — a 9-tap separable Gaussian,
   the same fragment shader run twice with a different uniform buffer
   (`params.dir` = a horizontal or vertical texel step). Separable means
   O(2n) texture samples instead of O(n²) for the same effective radius.
3. **Composite** — samples the full-res `scene_view` and the half-res
   blurred result (upscaled for free by the sampler's linear filtering),
   adds `bloom * intensity` on top, clamps to `[0, 1]` (the target is an
   8-bit UNORM surface, not float — values past 1.0 would wrap, not clip),
   and writes the **actual swapchain image**. This is the only pass in the
   whole chain that touches the real presented frame.

Every pass draws a single **fullscreen triangle** (3 vertices, no vertex
buffer, UVs computed from `vertex_index` in the vertex shader) — the classic
oversized-triangle trick, which avoids the diagonal seam a two-triangle quad
would need to rasterize exactly, at the cost of some harmless overdraw
outside the viewport.

A few implementation choices worth knowing if touching this code:

- **Two separate `.wgsl` files**, not one with more entry points
  (`bloom_sample.wgsl` for threshold/blur, `bloom_composite.wgsl` for the
  composite). WGSL validates resource bindings per shader module across
  every declared global, not just per reachable entry point — one file
  reusing `@group(0) @binding(0)` for two differently-typed resources in
  different sections is exactly the kind of layout collision this split
  avoids.
- **Four separate uniform buffers**, not one reused across passes. All the
  `queue.write_buffer()` calls for a frame's parameters happen before the
  single `queue.submit()`, and writes land before the submitted command
  buffer executes — not interleaved with the individual render passes
  inside it. Reusing one buffer would mean every pass reads whichever value
  was written *last*, not the one intended for it.
- **Only `TextureView`s are stored**, never the parent `Texture` handles —
  same pattern `main.rs`'s own MSAA target already uses. A view holds its
  own reference to the underlying GPU resource, so the Rust-side texture
  handle doesn't need to be kept alive once a view exists.
- **Resize rebuilds the whole struct** (`*self = Self::new(...)`) rather
  than patching individual textures — resizes are rare relative to frames,
  so one build path beats maintaining a separate incremental-rebuild path
  that could drift out of sync with it.
- Weight arrays for the blur kernel must be declared as a function-scope
  `var`, not a module-scope `const` — this naga/wgpu version rejects
  indexing a `const` array with a non-compile-time-constant (i.e. loop
  variable) index, a validation error that only surfaces when the shader is
  actually loaded by a real `wgpu::Device`, not at `cargo check` time.

`BLOOM_INTENSITY=<float>` overrides how strongly the glow is added back
(default `0.85`), read once at startup — same override pattern as `MSAA=`.

## 7. The visualizer system — `src/visuals.rs`

### The `Visualizer` trait

```rust
pub trait Visualizer {
    fn name(&self) -> &'static str;
    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder);
}
```

Implementations own whatever cross-frame state they need — history buffers,
peak holds, rotation angles, reusable point buffers — because they're
constructed once (`visuals::all()`) and reused every frame, so `draw` itself
never allocates (every visual pulls its scratch `Vec`s out with
`std::mem::take` at the top of `draw` and puts them back at the bottom).
Adding a new visual is exactly three steps: write the struct, implement the
trait, add one line to `all()` — nothing in `main.rs` needs to change, which
is how the Vectorscope was added without touching the event loop or the
dispatch logic at all.

### The shared color grammar

Every visual colors itself through exactly one function, `ramp(height_t,
freq_t)`, keyed **primarily on height** and only **secondarily** on
frequency:

- `height_t` (0 = resting/baseline, 1 = tallest thing on screen) selects
  position along a six-stop gradient, and therefore luminance *and* alpha
  together.
- `freq_t` (0 = lowest band, 1 = highest) only tilts red against blue by a
  small fixed fraction (`HUE_SHIFT = 0.12`), just enough to separate bass
  from treble without fracturing the image into rainbow stripes.

This ordering is deliberate: coloring by frequency first is what turns a
spectrum display into rainbow stripes where every bar looks identical
regardless of how loud it is. The hue tilt is normalized so it can only ever
*attenuate* a channel, never push it past its stop value — an unnormalized
tilt would saturate red at the ramp's hot end and then, climbing further,
only lower green/blue, making luminance start *falling* right at the top —
a dark band exactly where the ramp is designed to avoid one.

**Three palettes** back this ramp (`PALETTE_AURORA`/`PALETTE_SUNSET`/
`PALETTE_GLACIAL`), selected by a global atomic index and cycled with the
**P** key; all three follow the same construction rule — the top stop is a
warm/near-white color rather than a saturated hue, since a saturated top
stop is *darker* than an off-white one and produces the dark-band artifact
described above.

### `AutoRange` — auto-ranging contrast stretch

Raw normalized band levels arrive on a 90 dB scale mapped to 0..1, but real
music only ever occupies a narrow slice near the top of that range —
measured on real material, the median band sits at 0.71 and the loudest at
0.83, an 11 dB spread. Drawing that directly is why an un-stretched display
reads as every bar pinned at peak: the real differences exist, they're just
squeezed into roughly a tenth of the available height.

`AutoRange` tracks the loudest band with its own fast-attack (0.15s)
/slow-release (2.5s) envelope, and stretches a fixed-width window
(`RANGE_SPAN_DB = 25.0`, chosen from that same measured distribution) below
that tracked ceiling to fill 0..1. Tracking the *peak* rather than a fixed
absolute window is what keeps the picture stable as playback volume
changes — turning music down moves the window down with it instead of
dimming the whole display; a floor on the ceiling (`.clamp(RANGE_SPAN, 1.0)`)
stops silence, where the peak is zero, from stretching up to full scale.
Every band-driven visual owns one `AutoRange` instance and calls `update()`
once per frame before applying it.

### `Clock` — frame-rate-independent time deltas

A tiny helper every visual uses instead of counting frames: `tick(time)`
returns the wall-clock delta since the previous call, clamped to
`DT_CEILING = 0.1s` so an alt-tab stall can't inject a single enormous jump
that teleports an animation. Every animated quantity in every visual
integrates against this `dt`, which is what makes motion look identical at
30, 60, or 144 Hz — verified directly by dedicated tests (§10).

### The six visualizers

| Visual | What it draws | Key technique |
|---|---|---|
| **Bars** | Classic mirrored spectrum bars with falling peak-hold markers | Each bar is a thick vertical *stroke* (not a rectangle) so round caps give the Material end-shape and feathering comes free; peak markers snap up instantly and sink at a fixed rate/second |
| **Ridge Bed** | Joy-Division-style stacked spectrum history scrolling into a 3D-ish terrain | Each history row is an opaque fill that paints over the row behind it (occlusion), with only a shallow strip filled — not the whole row — since every row behind sits at a strictly higher baseline anyway; spectral tilt and idle "breathing" keep silence from flatlining |
| **Spectrum Area** | One filled silhouette with a glowing crest and a faded mirrored reflection | The vertical gradient is built from horizontal *slabs* (`fill_under` gives one color per column, so a smooth vertical ramp needs stacking), with run-length encoding so a flat span of the silhouette costs two vertices instead of one per column |
| **Radial Burst** | Spectrum wrapped as spokes around a bass-driven rotating ring | Spokes mirrored left/right across the vertical axis so bass sits at top and treble sweeps down both sides symmetrically |
| **Oscilloscope** | Triggered time-domain waveform trace | Rising zero-crossing trigger keeps the trace anchored instead of sliding sideways every frame; bucket-extreme decimation (not averaging) so transient peaks survive; `tanh` soft saturation instead of hard clipping |
| **Vectorscope** | Stereo X/Y ("Lissajous") trace, broadcast-scope style | Points plotted in **mid/side** rotation (`X = L−R`, `Y = L+R`) rather than raw L/R, so fully mono material collapses to a vertical line — the orientation a real vectorscope rests at — instead of a 45° diagonal |

All six route their line/fill geometry through the same `stroke_with`/
`fill_under`/`glow_stroke_with` primitives from §4, and color every vertex
through the one shared `ramp()` — there is no per-visual color logic outside
of *which* height/frequency values get fed into it.

## 8. The debug HUD — `src/hud.rs`

Purely diagnostic scaffolding — not shipping UI — toggled with **F1**, reset
with **F2**, dumped to stdout with **F3**. It draws with the same
`MeshBuilder` the visuals use, so it costs one more chunk of the single draw
call and needs no separate pipeline.

**Why these particular metrics.** Mean frame time is *not* the interesting
number in this app — the GPU sits mostly idle and the CPU has headroom to
spare. What actually matters is the **phase relationship** between two
unlocked clocks: the analysis thread producing spectra roughly every 16 ms,
and the swapchain presenting frames roughly every 16.6 ms. Comparing their
*rates* hides a real problem (60.0 Hz vs. 61.0 Hz both look healthy);
comparing spectrum **age** at draw time exposes it as a sawtooth that wraps
once per beat period between the two clocks. So the HUD's headline numbers
are `AGE` and `DUP` (duplicate-spectrum frames), not `CPU`/`GPU`.

- **Bitmap font**: a hand-authored 5×7 glyph table (ASCII 32–95, lowercase
  folded to uppercase), rendered by merging horizontal runs of lit pixels
  into single rectangles rather than one quad per pixel — brings a typical
  glyph from ~35 quads down to ~10, keeping the overlay's own vertex count
  from swamping the scene-geometry count it exists to report.
- **`Ring`**: a fixed-capacity circular buffer (240 samples = 4s at 60 Hz)
  with order-statistic queries (percentile, min/max/mean), preallocated at
  construction so the HUD — which exists partly to complain about per-frame
  allocation — doesn't itself allocate per frame.
- **Panel layout**: frame timing (CPU/GPU/submit/vsync breakdown) →
  presentation (FPS, dropped-frame count, a strip chart of present-to-present
  deltas) → audio→render handoff (hop rate vs. render rate, the "beat"
  frequency between them, spectrum age, a strip chart that turns into a
  visible sawtooth exactly when the two clocks drift) → geometry & settings
  (vertex/triangle counts, MSAA sample count, current visual and palette
  name).
- **`dump()`** prints a fuller plain-text summary including the audio
  thread's probe counters (too verbose for the on-screen panel) — triggered
  by F3, or automatically on an interval via `HUD_AUTODUMP=<seconds>` for
  scripted/headless profiling runs.

## 9. Main event loop, start to finish

`main()` in `src/main.rs`:

1. `spawn_audio_engine()` — starts capture + analysis, returns the shared
   `AudioMetrics` mutex, the `AudioProbe`, the negotiated sample rate, and
   the `cpal::Stream` handle (which must be kept alive for the duration of
   the program — dropping it stops capture, even though nothing reads from
   it directly).
2. `log_band_edges(sample_rate)` computed once, since it only depends on the
   negotiated sample rate.
3. A `winit` window + `GpuState` (§5) are created; `ControlFlow::Poll` keeps
   the loop running continuously rather than waiting for OS events, since
   this is a real-time animation, not a static UI.
4. On every `WindowEvent::RedrawRequested`:
   - Lock `AudioMetrics`, clone out this frame's spectra/waveforms/bass/rms
     (the lock is held only long enough to clone, not for the rest of the
     frame — this is what the `LOCK` timing in the HUD measures), and
     compute spectrum age (`now - produced_at`).
   - `update_bands()` twice (left and right channels), with a `dt` tracked
     the same way `visuals::Clock` tracks it, then average into `avg_levels`.
   - Compute the background clear color — the Material dark surface
     (`#121212`) mixed a small, deliberately-capped amount toward Deep
     Purple 900 by bass energy (capped because the target hue is saturated
     and the eye is meant to read low-alpha glow *against* this background,
     which stops working if the mix pushes it too far toward "purple
     background" instead of "dark surface with a bass lift").
   - Build `FrameData`, call the current visualizer's `draw()` into the
     shared `MeshBuilder`, then `hud.draw()` appends its own geometry to the
     same mesh (scene vertex/triangle counts are captured *before* this, so
     the HUD doesn't inflate the numbers it reports about itself).
   - `gpu.render(...)` uploads and submits (§5, §6); on `SurfaceError::Lost`
     or `Outdated` the surface is reconfigured at the current size rather
     than treated as fatal; `OutOfMemory` exits the process.
   - Record this frame's timings into the HUD's rolling stats.
5. Keyboard handling: `Space`/`→` and `←` cycle visualizers, `1`–`9` jump
   directly (digit key codes are contiguous in `winit`'s `KeyCode` enum, so
   this is one array lookup rather than nine match arms), `F1`–`F3` control
   the HUD, `F11` toggles borderless fullscreen, `P` cycles the color
   palette, `Escape` exits.

## 10. Testing strategy

Two kinds of tests, in two different files, doing different jobs:

- **`src/visuals.rs`** (`mod tests`) — mostly a **fuzz harness**
  (`exercise()`) that drives every registered visualizer (via `visuals::all()`,
  so a newly-added visual is automatically covered with no extra wiring)
  through a dozen frames of synthetic signal, checking invariants a compiler
  can't: no non-finite (`NaN`/`inf`) vertex coordinates or colors leaking
  into the vertex buffer (which would render as invisible or
  screen-filling garbage rather than crashing), no index buffer entry
  pointing past the end of the vertex buffer, no partial triangles. It's run
  against normal signal, silence, out-of-range levels (>1.0 and negative,
  since smoothing filters can technically overshoot), and extreme aspect
  ratios (0.15 and 3.0). Separate targeted tests check the shared `ramp()`
  function's invariants (never non-finite, luminance/alpha never decrease
  with height, base is dim and tip is bright), `RidgeBed`'s height-bound
  guarantees, `AutoRange`'s contrast-stretch behavior against a measured
  real-world band distribution, and that several time-driven behaviors
  (`AutoRange`, peak-marker decay) produce frame-rate-independent results by
  literally comparing simulated runs at 30/60/144 fps against each other.
  A separate `every_palette_is_monotone_in_luminance_and_alpha` test checks
  the *raw stop tables* directly (not through `ramp()`/the shared global
  palette index) specifically to avoid a subtle flakiness trap: Rust's test
  harness runs tests in parallel threads within one process by default, so
  a test that mutated the global palette selector while another test asserted
  against `ramp()`'s output would be a real, intermittent race.
- **`src/hud.rs`** (`mod tests`) — narrower unit tests for the ring buffer's
  ordering after wraparound, percentile correctness, duplicate-spectrum and
  dropped-frame counting logic, and that the bitmap font's glyph table
  doesn't overflow its 5-pixel width for any character actually used.
- **`mod bench`** in `visuals.rs` — not a real benchmark (timing here is
  printed, not asserted, since it's far too machine-dependent to gate CI on)
  but a **budget assertion**: every visualizer's per-frame vertex/triangle
  count is checked against a fixed ceiling (`VERTEX_BUDGET = 48_000`,
  `TRIANGLE_BUDGET = 64_000`), because spline-subdivision tuning is easy to
  accidentally blow up by 10x without noticing, and that cost lands directly
  on the per-frame GPU buffer upload.

`cargo test` runs all of the above with zero GPU/display dependency —
`MeshBuilder` and every visualizer operate purely on CPU-side geometry data,
so the entire visual-correctness test suite runs the same in CI as on a
desktop. What it **cannot** catch: WGSL shader validation errors, which
`wgpu`/`naga` only surface when a real `wgpu::Device` actually loads a
shader — one such bug (dynamic indexing into a module-scope `const` array,
rejected by this naga version) was only found by actually launching the
binary against a real GPU driver, not by `cargo check`/`cargo build`. See
`visualization-quality-upgrades.md` §6 for the full story.

## 11. Runtime configuration & keybindings

### Environment variables

| Variable | Effect |
|---|---|
| `AUDIO_SOURCE=<name>.monitor` | Override which PulseAudio/PipeWire source to capture instead of auto-resolving the default sink's monitor |
| `MSAA=1\|2\|4\|8` | Override the MSAA sample count (default 1, effectively off — see §4) |
| `START_VISUAL=<1-9>` | Start on a specific visualizer instead of the first, for scripted/profiling runs without keyboard input |
| `HUD_AUTODUMP=<seconds>` | Print the HUD summary on an interval instead of requiring an F3 keypress, for headless/scripted capture |
| `BLOOM_INTENSITY=<float>` | Override how strongly the bloom glow is added back over the scene (default `0.85`) |

### Keybindings

| Key | Action |
|---|---|
| `Space` / `→` | Next visualizer |
| `←` | Previous visualizer |
| `1`–`9` | Jump to visualizer N |
| `F1` | Toggle debug HUD |
| `F2` | Reset HUD statistics |
| `F3` | Dump HUD summary to stdout |
| `F11` | Toggle fullscreen |
| `P` | Cycle color palette |
| `Esc` | Quit |

## 12. Dependencies

| Crate | Role |
|---|---|
| `wgpu` (0.19) | Cross-backend GPU API (Vulkan/Metal/DX12/GL) — the entire rendering layer |
| `winit` (0.29) | Cross-platform window + event loop |
| `cpal` (0.15) | Cross-platform audio capture |
| `realfft` (3.3) + `rustfft` (6.2) | Real-input-specialized FFT (`realfft` wraps `rustfft`, roughly 2x faster than a general complex FFT for real signals) |
| `bytemuck` (1.x, `derive`) | Safe `struct → &[u8]` casting for GPU buffer uploads (`Vertex`, bloom's `Params` uniform) without `unsafe` |
| `pollster` (0.3) | Blocks on the one-time async `wgpu` adapter/device setup at startup, so `main()` doesn't need to be an async runtime |

## 13. File map

| File | Role |
|---|---|
| `src/lib.rs` | `audio_processor` library: device capture, FFT analysis thread, log-band grouping, attack/release smoothing |
| `src/main.rs` | Binary entry point: `winit` event loop, `GpuState` (device/pipeline/buffers), per-frame orchestration, keybindings |
| `src/render.rs` | `MeshBuilder` drawing primitives, antialiasing/color model, splines, the `Visualizer` trait, `FrameData` |
| `src/visuals.rs` | The six visualizers, the shared color ramp + palette system, `AutoRange`, `Clock` |
| `src/hud.rs` | Debug overlay: bitmap font, rolling stats, frame-timing/audio-handoff/geometry panels |
| `src/bloom.rs` | Offscreen bloom post-process: textures, pipelines, bind groups, the 4-pass chain |
| `src/shader.wgsl` | The scene pipeline's vertex/fragment shaders (trivial passthrough — all the interesting work already happened on the CPU side building the mesh) |
| `src/bloom_sample.wgsl` | Fullscreen-triangle vertex shader + threshold/blur fragment shaders |
| `src/bloom_composite.wgsl` | Fullscreen-triangle vertex shader + composite fragment shader |
| `Cargo.toml` | Single-crate manifest (library + binary share one `Cargo.toml`, package name `audio_processor`) |
