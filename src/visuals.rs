//! The visualization styles.
//!
//! # Adding a new visual
//!
//! 1. Write a struct holding whatever cross-frame state it needs, including
//!    reusable point buffers so `draw` never allocates.
//! 2. `impl Visualizer` — push geometry via the `MeshBuilder` helpers.
//! 3. Add one line to [`all`].
//!
//! That's the whole contract; nothing in `main.rs` needs to change.
//!
//! # Shared color grammar
//!
//! Every visual colors itself through one function, [`ramp`], keyed primarily
//! on **height** and only secondarily on frequency. Height drives position
//! along the ramp, and therefore luminance and alpha together; frequency only
//! tilts warm against cool. That ordering is deliberate — coloring by
//! frequency first is what turns a spectrum display into rainbow stripes, in
//! which every bar looks identical no matter how loud it is.

use crate::render::{mix, resample_monotone, rgba, srgb_hex, FrameData, MeshBuilder, Visualizer};
use audio_processor::NUM_BANDS;
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::sync::OnceLock;

/// All visualizers, in cycle order.
///
/// To add a new visual: implement [`Visualizer`] below and add one line here.
pub fn all() -> Vec<Box<dyn Visualizer>> {
    vec![
        Box::new(Bars::new()),
        Box::new(RidgeBed::new()),
        Box::new(SpectrumArea::new()),
        Box::new(RadialBurst::new()),
        Box::new(Oscilloscope::new()),
    ]
}

// ===========================================================================
// Tuning constants
// ===========================================================================

// -- shared ramp ------------------------------------------------------------

/// Ramp stops: (position, sRGB hex, alpha). Decoded to linear on first use.
///
/// The top stop is a *warm white*, not an orange. An orange heat accent would
/// be darker than the ice white below it, putting a dark band at the very top
/// of the ramp — the single most common reason a gradient reads badly. Warm
/// white keeps red at full while ice white is red-deficient, so luminance
/// still rises into the accent (0.866 -> 0.893) and it reads as white-hot.
const RAMP_STOPS: [(f32, u32, f32); 6] = [
    (0.00, 0x2A1258, 0.35), // deep violet, dim base
    (0.22, 0x5E35B1, 0.50), // Deep Purple 600
    (0.45, 0x2979FF, 0.68), // Blue A400, electric blue
    (0.68, 0x1DE9B6, 0.82), // Teal A400
    (0.88, 0xDCF3FF, 0.95), // ice white
    (1.00, 0xFFF0E6, 0.98), // heat accent, warm white
];

/// Peak red/blue tilt applied by `freq_t`, as a fraction. Small on purpose:
/// enough to separate bass from treble, not enough to fracture into rainbow.
const HUE_SHIFT: f32 = 0.12;

/// Material dark-theme surface (#121212) and the tint bass pushes it toward.
/// Read by `main.rs` for the clear color; not part of the ramp.
pub const SURFACE_HEX: u32 = 0x121212;
pub const SURFACE_BASS_HEX: u32 = 0x311B92; // Deep Purple 900

/// Largest frame delta any time-driven filter will integrate, in seconds.
/// Caps how far an alt-tab stall can teleport an animation.
const DT_CEILING: f32 = 0.1;

// -- Bars -------------------------------------------------------------------

/// NDC half-height of a full-scale (level 1.0) bar.
const MAX_BAR_H: f32 = 0.90;
/// Gradient samples per bar half. The bar is drawn as a stroke, so these are
/// polyline points and the GPU interpolates colors between them.
const BAR_SLICES: usize = 12;
/// Fraction of the band pitch a bar occupies; the rest is the gap.
const BAR_FILL: f32 = 0.82;
/// Peak marker fall rate, in level units per second (was 0.006 per frame).
const PEAK_FALL_PER_SEC: f32 = 0.36;
/// Peak marker thickness in pixels.
const PEAK_MARKER_PX: f32 = 3.0;
/// Below this level a bar is shorter than it is wide, where a round-capped
/// stroke would render as a blob; those are drawn as a flat pill instead.
const BAR_MIN_LEVEL: f32 = 0.004;
/// Resting half-height in pixels, so silence shows a dim violet baseline
/// rather than nothing at all.
const BAR_REST_PX: f32 = 1.5;

// -- Ridge Bed --------------------------------------------------------------

/// Baseline of the topmost (oldest) row.
///
/// Lowered from 0.80 because `AMP` is now large enough that a peak on the top
/// row would otherwise leave the screen: 0.80 + 0.62 = 1.42. Large amplitude
/// and a high baseline are mutually exclusive.
const RIDGE_TOP: f32 = 0.32;
/// Total NDC span the row baselines are spread over. Spacing = SPAN / ROWS.
const RIDGE_SPAN: f32 = 1.20;
/// Full-scale ridge height in NDC. Overlap ratio against row spacing is ~15:1,
/// which is what makes rows occlude each other into a terrain rather than
/// reading as a stack of separate lines.
const AMP: f32 = 0.62;
/// Floor on ridge height, so silence still shows a soft undulation.
const MIN_RIDGE: f32 = 0.012;
/// Hard ceiling on ridge height, enforced regardless of normalization.
const MAX_RIDGE: f32 = 0.62;
/// Highest NDC y a peak may reach. `RIDGE_TOP + MAX_RIDGE` must stay under it.
const SAFE_TOP: f32 = 0.96;
/// Response curve exponent. Below 1.0 lifts low and mid levels while leaving
/// peaks headroom; linear magnitude always reads flat on a spectrum display.
const RESPONSE_EXP: f32 = 0.65;
/// Width of the auto-ranging contrast window, in dB. See [`AutoRange`].
///
/// Measured on real material the band levels run: min 0.20, median 0.71, p90
/// 0.81, max 0.83 — a median of -26 dB and a peak of -15 dB. Only ~11 dB
/// separates a typical band from the loudest one, so a window much wider than
/// this cannot spread them apart and everything reads as pinned at peak.
const RANGE_SPAN_DB: f32 = 25.0;
/// Seconds for the window's ceiling to rise to a new peak, and to fall back.
/// Fast up so transients are not clipped, slow down so the image does not
/// pump between loud and quiet passages.
const RANGE_ATTACK: f32 = 0.15;
const RANGE_RELEASE: f32 = 2.5;
/// Spectral tilt in dB per octave, applied before height mapping so the treble
/// end of a ridge is not a flat line.
///
/// Applied as a *multiplicative* gain on the normalized level, not an additive
/// offset. An additive tilt raises the noise floor along with the signal, and
/// the per-row normalizer below then scales that floor straight back up to
/// full height — silence would render as a permanent ramp climbing to the
/// right. Multiplying leaves zero at zero.
const TILT_DB_PER_OCTAVE: f32 = 2.2;
/// Octaves spanned by the display range (20 Hz to 20 kHz).
const SPECTRUM_OCTAVES: f32 = 9.97;
/// dB range the band levels are normalized over; mirrors `DB_FLOOR` in lib.rs.
const DB_RANGE: f32 = 90.0;
/// Amplitude of the idle breathing undulation, in normalized level units.
const BREATH_LEVEL: f32 = 0.055;
/// The two slow rates the breathing is built from, in Hz.
const BREATH_HZ_A: f32 = 0.07;
const BREATH_HZ_B: f32 = 0.11;
/// Phase advance per band, so the breathing travels along the row.
const BREATH_BAND_PHASE: f32 = 0.19;
/// Seconds between history rows (was every 3rd frame).
const ROW_INTERVAL: f32 = 0.05;
/// Alpha multiplier for the oldest row, scaling to 1.0 at the newest.
const ROW_ALPHA_FAR: f32 = 0.45;
/// Stroke width in pixels for the oldest and the newest row.
const RIDGE_WIDTH_FAR_PX: f32 = 1.3;
const RIDGE_WIDTH_NEAR_PX: f32 = 2.6;
/// Horizontal squeeze applied to the oldest row, hinting at depth.
const RIDGE_SQUEEZE_FAR: f32 = 0.72;
/// How far below its own baseline a row's occluder extends, in row spacings.
///
/// Every row behind this one has a *higher* baseline, so its ridge line can
/// never fall below `baseline + spacing`; reaching one spacing under our own
/// baseline therefore hides everything this row is supposed to hide. Filling
/// all the way down instead — the obvious thing to do — costs about 16x the
/// fill rate for pixels that were already the right color.
const RIDGE_OCCLUDE_SPACINGS: f32 = 2.0;

// -- Spectrum Area ----------------------------------------------------------

/// Baseline of the silhouette, and the NDC height of a full-scale peak.
const AREA_BASE: f32 = -0.75;
const AREA_HEIGHT: f32 = 1.50;
/// Target height of one gradient slab, in pixels. `fill_under` gives one color
/// per column, so a vertical ramp has to be stacked out of horizontal slabs;
/// at ~4 px each the steps fall below the visible banding threshold.
const AREA_SLAB_PX: f32 = 4.0;
/// Clamp on the slab count. The upper bound is what keeps the worst case
/// (a silhouette that crosses every slab at every column) inside the vertex
/// budget; run-length encoding keeps the typical case far below it.
const AREA_BANDS_MIN: usize = 12;
const AREA_BANDS_MAX: usize = 60;
/// Slabs for the mirrored reflection. Fewer, because it is faint.
const AREA_REFLECT_BANDS: usize = 8;
/// Spline subdivisions per band for the area silhouette.
const AREA_SUBDIV: usize = 4;
/// Alpha multiplier for the mirrored reflection under the baseline.
const AREA_REFLECT_ALPHA: f32 = 0.28;
/// How far the reflection is squashed relative to the silhouette.
const AREA_REFLECT_SQUASH: f32 = 0.35;

// -- Radial Burst -----------------------------------------------------------

/// Hub radius at silence, and how much bass inflates it.
const RADIAL_INNER: f32 = 0.20;
const RADIAL_INNER_BASS: f32 = 0.10;
/// NDC length of a full-scale spoke.
const RADIAL_SPOKE_MAX: f32 = 0.62;
/// Gradient samples along each spoke, hub to tip.
const RADIAL_SLICES: usize = 6;
/// Rotation rate in radians per second, at rest and per unit of bass.
const RADIAL_ROT_PER_SEC: f32 = 0.09;
const RADIAL_ROT_BASS_PER_SEC: f32 = 0.24;

// -- Oscilloscope -----------------------------------------------------------

/// Screen-space resolution of the trace.
const OSC_POINTS: usize = 320;
/// Vertical clamp, and therefore the amplitude mapping to the top of the ramp.
const OSC_MAX_Y: f32 = 0.92;
/// Additive glow layers under the core stroke.
const OSC_GLOW_LAYERS: u32 = 2;

// ===========================================================================
// Shared height-driven ramp
// ===========================================================================

fn ramp_stops() -> &'static [(f32, [f32; 3], f32); 6] {
    static STOPS: OnceLock<[(f32, [f32; 3], f32); 6]> = OnceLock::new();
    STOPS.get_or_init(|| {
        let mut out = [(0.0, [0.0; 3], 0.0); 6];
        for (i, &(pos, hex, alpha)) in RAMP_STOPS.iter().enumerate() {
            out[i] = (pos, srgb_hex(hex), alpha);
        }
        out
    })
}

/// Shared color for every visual.
///
/// `height_t`: 0.0 at rest/baseline, 1.0 at the tallest thing on screen.
/// `freq_t`:   0.0 lowest band, 1.0 highest band.
///
/// Height dominates: it selects the position along the ramp, and therefore
/// luminance and alpha together. Frequency only tilts red against blue by
/// [`HUE_SHIFT`], separating bass from treble without breaking the image into
/// stripes.
///
/// Both inputs are clamped and non-finite inputs fall back to a defined value,
/// so this always returns finite components in 0..1.
pub fn ramp(height_t: f32, freq_t: f32) -> [f32; 4] {
    let h = if height_t.is_finite() {
        height_t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let f = if freq_t.is_finite() {
        freq_t.clamp(0.0, 1.0)
    } else {
        0.5
    };

    let stops = ramp_stops();
    let mut seg = stops.len() - 2;
    for i in 0..stops.len() - 1 {
        if h <= stops[i + 1].0 {
            seg = i;
            break;
        }
    }
    let (p0, c0, a0) = stops[seg];
    let (p1, c1, a1) = stops[seg + 1];
    let t = ((h - p0) / (p1 - p0).max(1e-6)).clamp(0.0, 1.0);

    // Interpolated in linear space. Doing this on sRGB-encoded values is what
    // produces the muddy brown-grey midpoint between two saturated colors.
    let rgb = mix(c0, c1, t);
    let alpha = a0 + (a1 - a0) * t;

    // The tilt is normalized so neither multiplier can exceed 1.0: a channel
    // is only ever attenuated, never pushed past its stop value.
    //
    // This matters more than it looks. The heat stop has red at full scale, so
    // an un-normalized tilt saturates it; once red is pinned at 1.0, climbing
    // further up the ramp only lowers green and blue and luminance starts
    // *falling* — a dark band at the very top, exactly the artifact the ramp
    // exists to avoid. Attenuating instead costs a uniform 1/(1+HUE_SHIFT)
    // of brightness, which is a flat scale and so cannot reorder anything.
    let warm = (f - 0.5) * 2.0 * HUE_SHIFT;
    let norm = 1.0 / (1.0 + HUE_SHIFT);
    let tilted = [
        (rgb[0] * (1.0 + warm) * norm).clamp(0.0, 1.0),
        (rgb[1] * norm).clamp(0.0, 1.0),
        (rgb[2] * (1.0 - warm) * norm).clamp(0.0, 1.0),
    ];
    rgba(tilted, alpha.clamp(0.0, 1.0))
}

/// Rec. 709 luminance of a linear-space color. Used by the ramp tests.
#[cfg(test)]
fn luminance(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Normalized-level width of the contrast window.
const RANGE_SPAN: f32 = RANGE_SPAN_DB / DB_RANGE;

/// Auto-ranging contrast stretch, shared by every band-driven visual.
///
/// The band levels arrive on a 90 dB scale mapped into 0..1, but music only
/// ever occupies a narrow slice near the top of it: measured, the median band
/// sits at 0.71 and the loudest at 0.83. Drawing that directly is why bars and
/// ridges all sit at what looks like peak level — the differences are real,
/// they are just squeezed into a tenth of the available height.
///
/// This tracks the loudest band with a fast-attack, slow-release envelope and
/// stretches the [`RANGE_SPAN_DB`] window *below* that peak to fill 0..1.
/// Tracking the peak rather than a fixed window is what keeps the picture
/// stable when playback volume changes: turning the music down moves the
/// window down with it instead of dimming everything.
///
/// The ceiling is floored at the window width so `lo` can never go negative —
/// otherwise silence, where the peak is zero, would stretch up to full scale.
struct AutoRange {
    hi: f32,
}

impl AutoRange {
    fn new() -> Self {
        Self { hi: RANGE_SPAN }
    }

    fn update(&mut self, bands: &[f32], dt: f32) {
        let peak = bands
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max)
            .clamp(0.0, 1.0);
        let tau = if peak > self.hi {
            RANGE_ATTACK
        } else {
            RANGE_RELEASE
        };
        // Time-constant form, so the envelope behaves the same at any frame rate.
        let alpha = 1.0 - (-dt / tau.max(1e-4)).exp();
        self.hi += (peak - self.hi) * alpha;
        self.hi = self.hi.clamp(RANGE_SPAN, 1.0);
    }

    fn apply(&self, level: f32) -> f32 {
        if !level.is_finite() {
            return 0.0;
        }
        ((level - (self.hi - RANGE_SPAN)) / RANGE_SPAN).clamp(0.0, 1.0)
    }
}

/// Frame-delta source. Every animated quantity integrates against this rather
/// than counting frames, so motion is identical at 30, 60 and 144 fps.
#[derive(Default)]
struct Clock {
    last: Option<f32>,
}

impl Clock {
    fn tick(&mut self, time: f32) -> f32 {
        let time = if time.is_finite() { time } else { 0.0 };
        let dt = match self.last {
            Some(prev) => (time - prev).clamp(0.0, DT_CEILING),
            None => 0.0,
        };
        self.last = Some(time);
        dt
    }
}

/// Emit one horizontal slab of a vertical gradient fill.
///
/// `curve` is clipped into `lo..hi` and filled to `baseline`. Only the columns
/// where the clipped silhouette actually *changes* get a vertex — a flat run
/// needs nothing but its two ends. That decouples the slab count from the
/// horizontal resolution: without it the two multiply, and a gradient fine
/// enough to hide banding costs tens of thousands of vertices per frame.
///
/// `height_t` is constant across the slab (it *is* the vertical position);
/// frequency is recovered from each column's x, because run-length encoding
/// breaks the usual "parameter along the polyline" mapping.
fn gradient_slab(
    mesh: &mut MeshBuilder,
    curve: &[[f32; 2]],
    scratch: &mut Vec<[f32; 2]>,
    lo: f32,
    hi: f32,
    baseline: f32,
    height_t: f32,
    alpha_scale: f32,
) {
    let n = curve.len();
    scratch.clear();
    for i in 0..n {
        let y = curve[i][1].clamp(lo, hi);
        let flat_left = i > 0 && (curve[i - 1][1].clamp(lo, hi) - y).abs() <= 1e-7;
        let flat_right = i + 1 < n && (curve[i + 1][1].clamp(lo, hi) - y).abs() <= 1e-7;
        if !(flat_left && flat_right) {
            scratch.push([curve[i][0], y]);
        }
    }
    if scratch.len() < 2 {
        return;
    }
    let pts: &[[f32; 2]] = scratch;
    let last = (pts.len() - 1) as f32;
    mesh.fill_under(pts, baseline, |t| {
        let idx = ((t * last).round() as usize).min(pts.len() - 1);
        let freq_t = (pts[idx][0] + 1.0) * 0.5;
        let mut c = ramp(height_t, freq_t);
        c[3] *= alpha_scale;
        c
    });
}

/// Stacked additive passes under a core stroke, with a per-length color.
///
/// `MeshBuilder::glow_stroke` takes a single flat color; this is the same idea
/// driven by the shared ramp instead. Wide layers are additive so they
/// accumulate into a halo rather than occluding one another.
fn glow_stroke_with(
    mesh: &mut MeshBuilder,
    pts: &[[f32; 2]],
    width_px: f32,
    layers: u32,
    color_at: &impl Fn(f32) -> [f32; 4],
) {
    for layer in (1..layers.max(1) + 1).rev() {
        let spread = 1.0 + layer as f32 * 2.4;
        let fade = 1.0 / (1.0 + layer as f32 * 3.0);
        mesh.stroke_with(pts, width_px * spread, true, |t| {
            let c = color_at(t);
            [c[0], c[1], c[2], c[3] * fade]
        });
    }
    mesh.stroke_with(pts, width_px, false, color_at);
}

// ---------------------------------------------------------------------------
// Bars — mirrored spectrum bars with falling peak markers.
// ---------------------------------------------------------------------------

pub struct Bars {
    peaks: Vec<f32>,
    clock: Clock,
    range: AutoRange,
    column: Vec<[f32; 2]>,
}

impl Bars {
    pub fn new() -> Self {
        Self {
            peaks: vec![0.0; NUM_BANDS],
            clock: Clock::default(),
            range: AutoRange::new(),
            column: Vec::with_capacity(BAR_SLICES * 2 + 1),
        }
    }
}

impl Visualizer for Bars {
    fn name(&self) -> &'static str {
        "Bars"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        let dt = self.clock.tick(frame.time);
        self.range.update(frame.bands, dt);
        let pitch = 2.0 / NUM_BANDS as f32;
        let aspect = mesh.aspect();
        let px = mesh.px();

        let mut column = std::mem::take(&mut self.column);

        for i in 0..NUM_BANDS {
            // Contrast-stretched, not raw: the raw levels only span 0.71..0.83
            // for typical material, which is why every bar looked pinned at
            // peak. This also lets a loud band actually reach the hot end of
            // the ramp, since ramp position tracks the stretched level.
            let level = self.range.apply(frame.bands[i]);
            // Never fully collapses: at silence every bar sits at the resting
            // height showing the dim violet base of the ramp.
            let h = (level * MAX_BAR_H).max(BAR_REST_PX * px);
            let x0 = -1.0 + i as f32 * pitch;
            let x1 = x0 + pitch * BAR_FILL;
            let cx = (x0 + x1) * 0.5;
            let freq_t = i as f32 / (NUM_BANDS - 1) as f32;
            let bar_px = mesh.to_px((x1 - x0) * aspect);

            // The gradient is sampled at *absolute* NDC height, not as a
            // fraction of this bar's own height. That is what makes a quiet bar
            // show only the cool end of the ramp while a loud one sweeps the
            // whole thing; normalizing per bar makes every bar look identical
            // regardless of level.
            let color_at = |t: f32| {
                let y = -h + t * 2.0 * h;
                ramp((y.abs() / MAX_BAR_H).clamp(0.0, 1.0), freq_t)
            };

            if level > BAR_MIN_LEVEL {
                column.clear();
                let n = BAR_SLICES * 2;
                for k in 0..=n {
                    column.push([cx, -h + 2.0 * h * k as f32 / n as f32]);
                }
                // A bar is a thick vertical stroke: round caps give the
                // Material end shape, the feathered sides come free, and `t`
                // along the stroke is exactly the vertical gradient axis.
                mesh.stroke_with(&column, bar_px, false, color_at);
            } else {
                // Shorter than it is wide, where a round-capped stroke would
                // render as a disc. Flat pill instead, at the tip color.
                mesh.round_rect(x0, -h, x1, h, bar_px * 0.5, color_at(1.0));
            }

            // Peak marker: snaps up instantly, then sinks at a fixed rate per
            // second, so its fall looks the same at any refresh rate.
            self.peaks[i] = (self.peaks[i].max(level) - PEAK_FALL_PER_SEC * dt).clamp(0.0, 1.0);
            let p = self.peaks[i];
            if p > level + 0.01 {
                let py = p * MAX_BAR_H;
                let thick = PEAK_MARKER_PX * px;
                // Colored from the same ramp at its own height, so there is no
                // seam between bar and cap — the marker reads as the next step
                // up the gradient rather than a separate white object.
                let c = ramp(p, freq_t);
                mesh.round_rect(x0, py, x1, py + thick, thick * 0.5, c);
                mesh.round_rect(x0, -py - thick, x1, -py, thick * 0.5, c);
            }
        }

        self.column = column;
    }
}

// ---------------------------------------------------------------------------
// Ridge Bed — stacked spectrum history, Joy Division style.
// ---------------------------------------------------------------------------

pub struct RidgeBed {
    /// Rows hold *shaped* levels in 0..1 — tilt, normalization, response curve
    /// and breathing are all baked in when the row is pushed, so redrawing a
    /// row costs only geometry and the terrain scrolls rigidly instead of
    /// rippling in place.
    history: VecDeque<Vec<f32>>,
    range: AutoRange,
    since_row: f32,
    clock: Clock,
    ctrl: Vec<[f32; 2]>,
    curve: Vec<[f32; 2]>,
    scratch: Vec<f32>,
}

impl RidgeBed {
    const ROWS: usize = 30;

    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(Self::ROWS),
            range: AutoRange::new(),
            since_row: 0.0,
            clock: Clock::default(),
            ctrl: Vec::with_capacity(NUM_BANDS),
            curve: Vec::with_capacity(NUM_BANDS * 5),
            scratch: vec![0.0; NUM_BANDS],
        }
    }

    /// Fractional gain applied to band `i`: 0.0 at the bottom of the spectrum,
    /// rising with frequency, used as `level * (1 + lift)`.
    ///
    /// The bands are log-spaced, so band index is proportional to octave and a
    /// constant dB-per-octave tilt is linear in the index. Without it the
    /// treble end of every ridge is a flat line.
    fn tilt_lift(i: usize) -> f32 {
        let octave = i as f32 / (NUM_BANDS - 1).max(1) as f32 * SPECTRUM_OCTAVES;
        (TILT_DB_PER_OCTAVE * octave / DB_RANGE).clamp(0.0, 1.0)
    }

    /// Slow idle undulation so silence breathes instead of flatlining.
    /// Two incommensurate rates read as noise rather than an obvious sine.
    fn breath(i: usize, time: f32) -> f32 {
        let phase = i as f32 * BREATH_BAND_PHASE;
        let a = (time * BREATH_HZ_A * TAU + phase).sin();
        let b = (time * BREATH_HZ_B * TAU + phase * 1.7).sin();
        ((a * 0.6 + b * 0.4) * 0.5 + 0.5) * BREATH_LEVEL
    }

    /// Turn raw band levels into shaped 0..1 heights, into `self.scratch`.
    ///
    /// Order matters: tilt first, so the contrast window sees the tilted
    /// spectrum and is not driven purely by bass; then stretch; then the
    /// response curve; then breathing on top, so it survives silence.
    fn shape_row(&mut self, bands: &[f32], dt: f32, time: f32) {
        for i in 0..NUM_BANDS {
            let raw = bands.get(i).copied().unwrap_or(0.0);
            let raw = if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.scratch[i] = (raw * (1.0 + Self::tilt_lift(i))).clamp(0.0, 1.0);
        }

        self.range.update(&self.scratch, dt);

        for i in 0..NUM_BANDS {
            let e = self.range.apply(self.scratch[i]);
            let shaped = e.powf(RESPONSE_EXP) + Self::breath(i, time);
            self.scratch[i] = shaped.clamp(0.0, 1.0);
        }
    }

    /// Map a shaped 0..1 level to a ridge height in NDC.
    ///
    /// Guaranteed to land in `MIN_RIDGE..=MAX_RIDGE` for every input, finite or
    /// not — this is the invariant that keeps peaks from punching out of the
    /// top of the frame, and it is a pure NDC-y quantity so it holds at any
    /// aspect ratio.
    pub fn ridge_height(shaped: f32) -> f32 {
        let s = if shaped.is_finite() {
            shaped.clamp(0.0, 1.0)
        } else {
            0.0
        };
        (MIN_RIDGE + (AMP - MIN_RIDGE) * smoothstep(s)).clamp(MIN_RIDGE, MAX_RIDGE)
    }
}

impl Visualizer for RidgeBed {
    fn name(&self) -> &'static str {
        "Ridge Bed"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        let dt = self.clock.tick(frame.time);
        self.since_row += dt;
        // Time-driven rather than frame-counted, so the terrain scrolls at the
        // same speed regardless of refresh rate. The first frame always pushes,
        // so the display is never empty.
        if self.since_row >= ROW_INTERVAL || self.history.is_empty() {
            self.since_row = 0.0;
            self.shape_row(frame.bands, dt, frame.time);
            // Recycle the retired row's allocation instead of freeing it.
            let mut row = if self.history.len() >= Self::ROWS {
                self.history.pop_front().unwrap()
            } else {
                Vec::with_capacity(NUM_BANDS)
            };
            row.clear();
            row.extend_from_slice(&self.scratch);
            self.history.push_back(row);
        }

        let rows = self.history.len();
        if rows == 0 {
            return;
        }

        let spacing = RIDGE_SPAN / Self::ROWS as f32;
        let opaque_bg = [
            frame.background[0],
            frame.background[1],
            frame.background[2],
            1.0,
        ];

        let mut ctrl = std::mem::take(&mut self.ctrl);
        let mut curve = std::mem::take(&mut self.curve);

        // Oldest first: each newer row's opaque fill paints over the row behind
        // it, which is what produces the layered occlusion of the original.
        for (row, levels) in self.history.iter().enumerate() {
            let age = row as f32 / (rows.max(2) - 1) as f32; // 0 = oldest, 1 = newest
            let baseline = RIDGE_TOP - row as f32 * spacing;
            let squeeze = RIDGE_SQUEEZE_FAR + (1.0 - RIDGE_SQUEEZE_FAR) * age;

            ctrl.clear();
            for (b, &shaped) in levels.iter().enumerate().take(NUM_BANDS) {
                let x = (-1.0 + 2.0 * b as f32 / (NUM_BANDS - 1) as f32) * squeeze;
                // Enforced here, not just asserted in a test: the margin holds
                // even if RIDGE_TOP or AMP are retuned to a combination that
                // would otherwise push the top row off screen.
                ctrl.push([x, (baseline + Self::ridge_height(shaped)).min(SAFE_TOP)]);
            }

            // Distant rows are narrower on screen and carry thinner strokes, so
            // they need fewer subdivisions to sit below the faceting threshold.
            let subdiv = 2 + (age * 2.0).round() as usize;
            resample_monotone(&ctrl, subdiv, &mut curve);
            if curve.len() < 2 {
                continue;
            }

            // See RIDGE_OCCLUDE_SPACINGS: rows behind this one all sit at
            // higher baselines, so a shallow fill hides them just as well as a
            // full-height one at a fraction of the fill rate.
            mesh.fill_under(&curve, baseline - spacing * RIDGE_OCCLUDE_SPACINGS, |_| {
                opaque_bg
            });

            let row_alpha = ROW_ALPHA_FAR + (1.0 - ROW_ALPHA_FAR) * age;
            let width_px = RIDGE_WIDTH_FAR_PX + (RIDGE_WIDTH_NEAR_PX - RIDGE_WIDTH_FAR_PX) * age;
            let last = (curve.len() - 1) as f32;

            // Per-vertex color by the ridge's height above its own baseline,
            // recovered by indexing the curve from the stroke parameter.
            let color_at = |t: f32| {
                let idx = ((t * last).round() as usize).min(curve.len() - 1);
                let h = ((curve[idx][1] - baseline) / AMP).clamp(0.0, 1.0);
                let mut c = ramp(h, t);
                c[3] *= row_alpha;
                c
            };

            if age > 0.92 {
                glow_stroke_with(mesh, &curve, width_px, 2, &color_at);
            } else {
                mesh.stroke_with(&curve, width_px, false, color_at);
            }
        }

        self.ctrl = ctrl;
        self.curve = curve;
    }
}

// ---------------------------------------------------------------------------
// Spectrum Area — one filled silhouette with a glowing crest.
// ---------------------------------------------------------------------------

pub struct SpectrumArea {
    clock: Clock,
    range: AutoRange,
    ctrl: Vec<[f32; 2]>,
    curve: Vec<[f32; 2]>,
    slab: Vec<[f32; 2]>,
}

impl SpectrumArea {
    pub fn new() -> Self {
        Self {
            clock: Clock::default(),
            range: AutoRange::new(),
            ctrl: Vec::with_capacity(NUM_BANDS),
            curve: Vec::with_capacity(NUM_BANDS * AREA_SUBDIV + 1),
            slab: Vec::with_capacity(NUM_BANDS * AREA_SUBDIV + 1),
        }
    }
}

impl Visualizer for SpectrumArea {
    fn name(&self) -> &'static str {
        "Spectrum Area"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        let dt = self.clock.tick(frame.time);
        self.range.update(frame.bands, dt);
        let mut ctrl = std::mem::take(&mut self.ctrl);
        let mut curve = std::mem::take(&mut self.curve);
        let mut slab = std::mem::take(&mut self.slab);

        ctrl.clear();
        for b in 0..NUM_BANDS {
            let x = -1.0 + 2.0 * b as f32 / (NUM_BANDS - 1) as f32;
            ctrl.push([x, AREA_BASE + self.range.apply(frame.bands[b]) * AREA_HEIGHT]);
        }
        resample_monotone(&ctrl, AREA_SUBDIV, &mut curve);

        if curve.len() >= 2 {
            // Slab count follows the framebuffer, so the gradient is stepped
            // at a constant ~4 px regardless of window size. Slabs abut
            // exactly, so there is no seam between them.
            let bands = (mesh.to_px(AREA_HEIGHT) / AREA_SLAB_PX).round() as usize;
            let bands = bands.clamp(AREA_BANDS_MIN, AREA_BANDS_MAX);
            let slab_h = AREA_HEIGHT / bands as f32;
            for k in 0..bands {
                let lo = AREA_BASE + k as f32 * slab_h;
                let height_t = ((k as f32 + 0.5) / bands as f32).clamp(0.0, 1.0);
                gradient_slab(mesh, &curve, &mut slab, lo, lo + slab_h, lo, height_t, 1.0);
            }

            // Mirrored reflection under the baseline, squashed and faded.
            // Also slabbed: coloring it per column instead would tie the color
            // to each column's height, which stripes vertically wherever
            // neighbouring bands differ.
            let refl_h = AREA_HEIGHT * AREA_REFLECT_SQUASH;
            let refl_slab = refl_h / AREA_REFLECT_BANDS as f32;
            slab.clear();
            let mirrored: Vec<[f32; 2]> = curve
                .iter()
                .map(|p| [p[0], AREA_BASE - (p[1] - AREA_BASE) * AREA_REFLECT_SQUASH])
                .collect();
            for k in 0..AREA_REFLECT_BANDS {
                let hi = AREA_BASE - k as f32 * refl_slab;
                let lo = hi - refl_slab;
                // Depth below the baseline maps back to the silhouette height
                // it mirrors, so the reflection fades along the same ramp.
                let height_t = ((k as f32 + 0.5) / AREA_REFLECT_BANDS as f32).clamp(0.0, 1.0);
                gradient_slab(
                    mesh,
                    &mirrored,
                    &mut slab,
                    lo,
                    hi,
                    hi,
                    height_t,
                    AREA_REFLECT_ALPHA,
                );
            }

            let last = (curve.len() - 1) as f32;
            let crest = |t: f32| {
                let idx = ((t * last).round() as usize).min(curve.len() - 1);
                let h = ((curve[idx][1] - AREA_BASE) / AREA_HEIGHT).clamp(0.0, 1.0);
                ramp(h, t)
            };
            glow_stroke_with(mesh, &curve, 2.0, 2, &crest);
        }

        self.ctrl = ctrl;
        self.curve = curve;
        self.slab = slab;
    }
}

// ---------------------------------------------------------------------------
// Radial Burst — spectrum wrapped around a bass-driven ring.
// ---------------------------------------------------------------------------

pub struct RadialBurst {
    rotation: f32,
    clock: Clock,
    range: AutoRange,
    spoke: Vec<[f32; 2]>,
}

impl RadialBurst {
    pub fn new() -> Self {
        Self {
            rotation: 0.0,
            clock: Clock::default(),
            range: AutoRange::new(),
            spoke: Vec::with_capacity(RADIAL_SLICES + 1),
        }
    }
}

impl Visualizer for RadialBurst {
    fn name(&self) -> &'static str {
        "Radial Burst"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        let dt = self.clock.tick(frame.time);
        self.range.update(frame.bands, dt);
        let bass = frame.bass.clamp(0.0, 1.0);
        // Radians per second, not per frame.
        self.rotation += (RADIAL_ROT_PER_SEC + bass * RADIAL_ROT_BASS_PER_SEC) * dt;

        let center = [0.0, 0.0];
        let inner = RADIAL_INNER + bass * RADIAL_INNER_BASS;
        let pitch_px = mesh.to_px(TAU * inner / NUM_BANDS as f32);
        let spoke_px = (pitch_px * 0.72).clamp(2.0, 14.0);

        let mut spoke = std::mem::take(&mut self.spoke);

        // Spokes are mirrored across the vertical axis so the ring reads as
        // symmetric: bass at top, treble sweeping down both sides.
        for i in 0..NUM_BANDS {
            let level = self.range.apply(frame.bands[i]);
            let reach = level * RADIAL_SPOKE_MAX;
            let t_band = i as f32 / (NUM_BANDS - 1) as f32;

            for dir in [1.0f32, -1.0] {
                let a = self.rotation + dir * (t_band * TAU * 0.5);
                spoke.clear();
                for k in 0..=RADIAL_SLICES {
                    let f = k as f32 / RADIAL_SLICES as f32;
                    spoke.push(mesh.polar(center, inner + reach * f, a));
                }
                // Radial gradient, hub to tip. `height_t` is the absolute
                // radial extent, so a quiet spoke never reaches the hot end.
                mesh.stroke_with(&spoke, spoke_px, false, |t| {
                    ramp((t * reach / RADIAL_SPOKE_MAX).clamp(0.0, 1.0), t_band)
                });
            }
        }

        // Hub ring, riding the same ramp so it pulses along the shared
        // gradient rather than being a separate white outline.
        mesh.ring(center, inner - 0.02, 2.0, ramp(0.45 + bass * 0.35, 0.5));

        self.spoke = spoke;
    }
}

// ---------------------------------------------------------------------------
// Oscilloscope — triggered time-domain trace.
// ---------------------------------------------------------------------------

pub struct Oscilloscope {
    pts: Vec<[f32; 2]>,
}

impl Oscilloscope {
    pub fn new() -> Self {
        Self {
            pts: Vec::with_capacity(OSC_POINTS),
        }
    }

    /// Find a rising zero crossing to anchor the trace.
    ///
    /// Without this the waveform slides sideways every frame, since the capture
    /// window has no relationship to the signal's period.
    fn trigger(wave: &[f32], span: usize) -> usize {
        let search = wave.len().saturating_sub(span);
        for i in 1..search.max(1) {
            if wave[i - 1] <= 0.0 && wave[i] > 0.0 {
                return i;
            }
        }
        0
    }
}

impl Visualizer for Oscilloscope {
    fn name(&self) -> &'static str {
        "Oscilloscope"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        if frame.waveform.len() < 64 {
            return;
        }

        let span = (frame.waveform.len() * 3 / 4).min(768);
        let start = Self::trigger(frame.waveform, span);
        let slice = &frame.waveform[start..start + span];

        // Gain rides the signal level so quiet passages stay visible.
        let gain = 1.6 / frame.rms.max(0.02).sqrt().max(0.25);

        // Decimate to display resolution, taking the extreme of each bucket so
        // transient peaks survive rather than being averaged into the noise.
        let buckets = OSC_POINTS.min(span);
        let mut pts = std::mem::take(&mut self.pts);
        pts.clear();
        for k in 0..buckets {
            let lo = k * span / buckets;
            let hi = ((k + 1) * span / buckets).max(lo + 1).min(span);
            let mut extreme = 0.0f32;
            for &s in &slice[lo..hi] {
                if s.abs() > extreme.abs() {
                    extreme = s;
                }
            }
            let x = -1.0 + 2.0 * k as f32 / (buckets - 1).max(1) as f32;
            // Soft saturation rather than a hard clamp: through a hard limiter
            // a loud passage flat-tops into solid blocks, while tanh
            // compresses into the same range and keeps the shape readable.
            pts.push([x, (extreme * gain).tanh() * OSC_MAX_Y]);
        }

        // Center line, at the resting end of the ramp.
        let hair = mesh.px();
        mesh.round_rect(-1.0, -hair, 1.0, hair, 0.0, ramp(0.0, 0.5));

        if pts.len() >= 2 {
            let last = (pts.len() - 1) as f32;
            // Colored by |amplitude|: violet through the quiet crossings, ice
            // white at the extremes. `freq_t` is held neutral — horizontal
            // position here is time, not frequency.
            let color_at = |t: f32| {
                let idx = ((t * last).round() as usize).min(pts.len() - 1);
                ramp((pts[idx][1].abs() / OSC_MAX_Y).clamp(0.0, 1.0), 0.5)
            };
            glow_stroke_with(mesh, &pts, 2.0, OSC_GLOW_LAYERS, &color_at);
        }

        self.pts = pts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_processor::FFT_SIZE;

    fn frame_at<'a>(bands: &'a [f32], waveform: &'a [f32], time: f32) -> FrameData<'a> {
        FrameData {
            bands,
            left_bands: bands,
            right_bands: bands,
            waveform,
            bass: 0.5,
            rms: 0.2,
            time,
            background: [0.02, 0.02, 0.05],
        }
    }

    /// Drive every visualizer over a run of frames and check what it emits.
    ///
    /// Guards the things a compile can't: out-of-range slicing, NaN/inf
    /// coordinates leaking into the vertex buffer (which render as invisible or
    /// screen-filling garbage rather than an error), and index buffer entries
    /// pointing past the end of the vertex buffer.
    fn exercise(signal: impl Fn(usize, usize) -> f32, width_px: f32, height_px: f32) {
        let mut mesh = MeshBuilder::new();

        for mut vis in all() {
            let mut produced = 0usize;

            for frame_idx in 0..12 {
                let bands: Vec<f32> = (0..NUM_BANDS).map(|b| signal(frame_idx, b)).collect();
                let waveform: Vec<f32> = (0..FFT_SIZE)
                    .map(|i| (i as f32 * 0.05 + frame_idx as f32).sin() * 0.5)
                    .collect();

                mesh.begin(width_px, height_px);
                vis.draw(
                    &frame_at(&bands, &waveform, frame_idx as f32 / 60.0),
                    &mut mesh,
                );

                for v in mesh.vertices() {
                    assert!(
                        v.position.iter().all(|c| c.is_finite())
                            && v.color.iter().all(|c| c.is_finite()),
                        "{} emitted a non-finite vertex: {:?} {:?}",
                        vis.name(),
                        v.position,
                        v.color
                    );
                }

                let n = mesh.vertices().len() as u32;
                for &i in mesh.indices() {
                    assert!(
                        i < n,
                        "{} emitted index {} past {} vertices",
                        vis.name(),
                        i,
                        n
                    );
                }
                assert_eq!(
                    mesh.indices().len() % 3,
                    0,
                    "{} emitted a partial triangle",
                    vis.name()
                );

                produced += mesh.vertices().len();
            }

            assert!(produced > 0, "{} never emitted any geometry", vis.name());
        }
    }

    #[test]
    fn visualizers_handle_normal_audio() {
        exercise(
            |f, b| (b as f32 * 0.2 + f as f32 * 0.3).sin() * 0.5 + 0.5,
            1600.0,
            900.0,
        );
    }

    #[test]
    fn visualizers_handle_silence() {
        exercise(|_, _| 0.0, 1600.0, 900.0);
    }

    /// Levels are clamped downstream, but a visual shouldn't blow up if the
    /// smoothing ever hands it something outside 0..1.
    #[test]
    fn visualizers_handle_out_of_range_levels() {
        exercise(|_, b| if b % 2 == 0 { 1.8 } else { -0.4 }, 900.0, 900.0);
    }

    /// Aspect 0.15 (tall and narrow) and 3.0 (wide and short). Stroke widths
    /// are pixel-derived, so neither should distort them.
    #[test]
    fn visualizers_handle_extreme_aspect() {
        exercise(|_, b| b as f32 / NUM_BANDS as f32, 150.0, 1000.0);
        exercise(|_, b| b as f32 / NUM_BANDS as f32, 3000.0, 1000.0);
    }

    // -- ramp --------------------------------------------------------------

    #[test]
    fn ramp_is_finite_and_in_range_over_the_whole_domain() {
        let probes = [
            f32::NEG_INFINITY,
            -1e9,
            -0.5,
            0.0,
            0.37,
            0.88,
            1.0,
            1.7,
            1e9,
            f32::INFINITY,
            f32::NAN,
        ];
        for &h in &probes {
            for &f in &probes {
                let c = ramp(h, f);
                assert!(
                    c.iter().all(|v| v.is_finite()),
                    "ramp({h}, {f}) produced a non-finite component: {c:?}"
                );
                assert!(
                    c.iter().all(|v| (0.0..=1.0).contains(v)),
                    "ramp({h}, {f}) escaped 0..1: {c:?}"
                );
            }
        }
    }

    /// A dark band in the middle of a gradient is the single most common reason
    /// one reads badly, so luminance must never decrease with height.
    #[test]
    fn ramp_is_monotone_in_luminance_and_alpha() {
        for &f in &[-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5] {
            let mut prev_lum = f32::NEG_INFINITY;
            let mut prev_alpha = f32::NEG_INFINITY;
            for step in 0..=400 {
                // Deliberately sweeps outside 0..1 at both ends.
                let h = -0.5 + step as f32 * (2.0 / 400.0);
                let c = ramp(h, f);
                let lum = luminance(c);
                assert!(
                    lum >= prev_lum - 1e-5,
                    "luminance fell from {prev_lum} to {lum} at height {h}, freq {f}"
                );
                assert!(
                    c[3] >= prev_alpha - 1e-5,
                    "alpha fell from {prev_alpha} to {} at height {h}, freq {f}",
                    c[3]
                );
                prev_lum = lum;
                prev_alpha = c[3];
            }
        }
    }

    #[test]
    fn ramp_base_is_dim_and_tip_is_bright() {
        let base = ramp(0.0, 0.5);
        let tip = ramp(1.0, 0.5);
        assert!(base[3] < 0.4, "base alpha {} should be dim", base[3]);
        assert!(tip[3] > 0.9, "tip alpha {} should be near opaque", tip[3]);
        assert!(
            luminance(tip) > luminance(base) * 20.0,
            "tip should be far brighter than base"
        );
    }

    /// Frequency must only modulate. If it dominated, two bars at the same
    /// height but different bands would read as unrelated colors.
    #[test]
    fn frequency_only_modulates_the_ramp() {
        for step in 0..=20 {
            let h = step as f32 / 20.0;
            let cool = luminance(ramp(h, 0.0));
            let warm = luminance(ramp(h, 1.0));
            let mid = luminance(ramp(h, 0.5));
            assert!(
                (warm - cool).abs() <= 0.02 + mid * 0.30,
                "freq shifted luminance by {} at height {h}; it should only tilt hue",
                (warm - cool).abs()
            );
        }
    }

    // -- ridge dynamics ----------------------------------------------------

    #[test]
    fn ridge_height_stays_within_bounds_for_every_input() {
        let mut probes: Vec<f32> = (0..=300).map(|i| -1.0 + i as f32 / 100.0).collect();
        probes.extend_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e9, -1e9]);

        for &s in &probes {
            let h = RidgeBed::ridge_height(s);
            assert!(h.is_finite(), "ridge_height({s}) was not finite");
            assert!(
                (MIN_RIDGE..=MAX_RIDGE).contains(&h),
                "ridge_height({s}) = {h} escaped {MIN_RIDGE}..={MAX_RIDGE}"
            );
        }
        // Silence still breathes; full scale still reaches the ceiling.
        assert_eq!(RidgeBed::ridge_height(0.0), MIN_RIDGE);
        assert!((RidgeBed::ridge_height(1.0) - MAX_RIDGE).abs() < 1e-6);
    }

    #[test]
    fn ridge_layout_leaves_headroom_at_the_top() {
        assert!(
            RIDGE_TOP + MAX_RIDGE <= SAFE_TOP,
            "a peak on the top row reaches {}, past the {SAFE_TOP} safe margin",
            RIDGE_TOP + MAX_RIDGE
        );
        // Rows must overlap or the bed reads as a stack of separate lines.
        let spacing = RIDGE_SPAN / RidgeBed::ROWS as f32;
        assert!(
            AMP / spacing > 5.0,
            "overlap ratio {} is too low for the layered look",
            AMP / spacing
        );
    }

    /// The margin must hold at any aspect, since ridge height is a pure NDC-y
    /// quantity and must not pick up an aspect dependency.
    #[test]
    fn ridge_peaks_never_escape_the_safe_margin() {
        // Round caps and the feathered fringe extend a couple of pixels past
        // the polyline vertices; at 620 px tall that is well under 0.02 NDC.
        const TOLERANCE: f32 = 0.02;

        for (w, h) in [(150.0, 1000.0), (3000.0, 1000.0), (1100.0, 620.0)] {
            for signal in [0.0f32, 0.5, 1.8] {
                let mut vis = RidgeBed::new();
                let mut mesh = MeshBuilder::new();
                let bands = vec![signal; NUM_BANDS];
                let waveform = vec![0.0f32; FFT_SIZE];

                for step in 0..90 {
                    mesh.begin(w, h);
                    vis.draw(&frame_at(&bands, &waveform, step as f32 * 0.05), &mut mesh);

                    for v in mesh.vertices() {
                        assert!(
                            v.position[1] <= SAFE_TOP + TOLERANCE,
                            "ridge vertex reached y={} at {w}x{h} with level {signal}",
                            v.position[1]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shaped_levels_stay_normalized_for_silence_and_clipping() {
        for signal in [0.0f32, 0.25, 1.0, 1.8, -0.4] {
            let mut vis = RidgeBed::new();
            let bands = vec![signal; NUM_BANDS];
            for step in 0..60 {
                vis.shape_row(&bands, 0.05, step as f32 * 0.05);
                for (i, &v) in vis.scratch.iter().enumerate() {
                    assert!(
                        v.is_finite() && (0.0..=1.0).contains(&v),
                        "band {i} shaped to {v} for input {signal}"
                    );
                }
            }
            // The normalization floor is what stops silence from being scaled
            // up into full-height noise.
            if signal <= 0.0 {
                let peak = vis.scratch.iter().fold(0.0f32, |a, &b| a.max(b));
                assert!(
                    peak < 0.85,
                    "silence normalized up to {peak}; the floor is not holding"
                );
            }
        }
    }

    /// The distribution measured on real material through the HUD:
    /// min 0.20, median 0.71, p90 0.81, max 0.83. Drawn raw, that is a tenth
    /// of the display range and everything reads as pinned at peak.
    fn measured_bands() -> Vec<f32> {
        (0..NUM_BANDS)
            .map(|i| match i {
                0..=5 => 0.20 + i as f32 * 0.02,
                6..=31 => 0.55 + (i - 6) as f32 * 0.006,
                _ => 0.71 + ((i - 32) as f32 / 31.0) * 0.12,
            })
            .collect()
    }

    fn settle(range: &mut AutoRange, bands: &[f32], seconds: f32) {
        let dt = 1.0 / 60.0;
        for _ in 0..((seconds / dt) as usize) {
            range.update(bands, dt);
        }
    }

    #[test]
    fn auto_range_spreads_the_measured_distribution() {
        let bands = measured_bands();
        let mut range = AutoRange::new();
        settle(&mut range, &bands, 5.0);

        let loudest = range.apply(0.83);
        let median = range.apply(0.71);
        let quiet = range.apply(0.20);

        assert!(loudest > 0.9, "loudest band only reached {loudest}");
        assert!(
            (0.35..0.75).contains(&median),
            "median band landed at {median}; it should sit near the middle, \
             not bunched against the top"
        );
        assert_eq!(quiet, 0.0, "the dead bottom of the range should clamp away");
        // The whole point: the spread must be far wider than the raw 0.12.
        assert!(
            loudest - median > 0.3,
            "loudest and median differ by only {}",
            loudest - median
        );
    }

    #[test]
    fn auto_range_keeps_silence_dark() {
        let mut range = AutoRange::new();
        settle(&mut range, &vec![0.0; NUM_BANDS], 10.0);
        assert_eq!(
            range.apply(0.0),
            0.0,
            "silence must not be stretched up to full scale"
        );
    }

    /// The envelope is a time-constant filter, so where it settles must not
    /// depend on how many frames it took to get there.
    #[test]
    fn auto_range_is_frame_rate_independent() {
        let bands = measured_bands();
        let settled = |fps: f32| {
            let mut range = AutoRange::new();
            let dt = 1.0 / fps;
            for _ in 0..(fps as usize * 3) {
                range.update(&bands, dt);
            }
            range.apply(0.71)
        };
        let (a, b, c) = (settled(30.0), settled(60.0), settled(144.0));
        assert!(
            (a - b).abs() < 0.01 && (b - c).abs() < 0.01,
            "auto-range diverged across frame rates: 30={a} 60={b} 144={c}"
        );
    }

    /// The spectral tilt must actually lift the treble end, or the right-hand
    /// side of every ridge stays a flat line.
    #[test]
    fn spectral_tilt_lifts_the_treble_end() {
        let low = RidgeBed::tilt_lift(0);
        let high = RidgeBed::tilt_lift(NUM_BANDS - 1);
        assert!(low.abs() < 1e-6, "lowest band should not be tilted");
        assert!(
            high > 0.15 && high < 0.4,
            "top band lift {high} is outside the useful range"
        );
        for i in 1..NUM_BANDS {
            assert!(
                RidgeBed::tilt_lift(i) >= RidgeBed::tilt_lift(i - 1),
                "tilt must rise monotonically with frequency"
            );
        }
    }

    /// Motion is integrated against `frame.time`, so the same elapsed time must
    /// produce the same state at any frame rate.
    #[test]
    fn peak_decay_is_frame_rate_independent() {
        let settled = |fps: f32| {
            let mut vis = Bars::new();
            let mut mesh = MeshBuilder::new();
            let waveform = vec![0.0f32; FFT_SIZE];
            let steps = fps as usize; // one second of frames
            for step in 0..=steps {
                // A single loud frame, then silence for the rest of the second.
                let level = if step == 0 { 1.0 } else { 0.0 };
                let bands = vec![level; NUM_BANDS];
                mesh.begin(1100.0, 620.0);
                vis.draw(&frame_at(&bands, &waveform, step as f32 / fps), &mut mesh);
            }
            vis.peaks[0]
        };

        let a = settled(30.0);
        let b = settled(60.0);
        let c = settled(144.0);
        assert!(
            (a - b).abs() < 0.02 && (b - c).abs() < 0.02,
            "peak decay diverged across frame rates: 30={a} 60={b} 144={c}"
        );
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    /// Rough per-frame CPU cost of mesh construction, printed with
    /// `cargo test --release -- --nocapture bench`.
    ///
    /// Not a precise benchmark; it exists to catch order-of-magnitude
    /// regressions in tessellation cost, which is easy to blow up by raising a
    /// subdivision count without noticing.
    #[test]
    fn geometry_build_cost() {
        let bands: Vec<f32> = (0..NUM_BANDS)
            .map(|b| ((b as f32) * 0.37).sin() * 0.5 + 0.5)
            .collect();
        let waveform: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();

        let make = |t: f32| FrameData {
            bands: &bands,
            left_bands: &bands,
            right_bands: &bands,
            waveform: &waveform,
            bass: 0.4,
            rms: 0.2,
            time: t,
            background: [0.02, 0.02, 0.05],
        };

        for mut vis in all() {
            let mut mesh = MeshBuilder::new();
            // Warm up cross-frame state (history rings, reusable buffers).
            for i in 0..128 {
                mesh.begin(1100.0, 620.0);
                vis.draw(&make(i as f32 / 60.0), &mut mesh);
            }

            let iters = 200;
            let start = Instant::now();
            for i in 0..iters {
                mesh.begin(1100.0, 620.0);
                vis.draw(&make((128 + i) as f32 / 60.0), &mut mesh);
            }
            let per_frame = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let verts = mesh.vertices().len();
            let tris = mesh.indices().len() / 3;
            println!(
                "{:<15} {:>8.3} ms/frame  {:>7} verts  {:>7} tris  {:>6.0} KB/frame",
                vis.name(),
                per_frame,
                verts,
                tris,
                (verts as f32 * 24.0 + mesh.indices().len() as f32 * 4.0) / 1024.0
            );

            // Budget. Spline subdivision makes these counts easy to inflate by
            // a factor of ten without noticing, and the cost lands on the
            // per-frame buffer upload rather than on tessellation — Ridge Bed
            // at 2.2 MB/frame spent 6.6 ms in `write_buffer` alone.
            //
            // Timing is printed but not asserted: it is far too machine- and
            // load-dependent to gate on. Geometry counts are deterministic.
            assert!(
                verts <= VERTEX_BUDGET,
                "{} emits {verts} vertices, over the {VERTEX_BUDGET} budget",
                vis.name()
            );
            assert!(
                tris <= TRIANGLE_BUDGET,
                "{} emits {tris} triangles, over the {TRIANGLE_BUDGET} budget",
                vis.name()
            );
        }
    }

    /// Per-frame ceilings, chosen to keep the vertex+index upload comfortably
    /// under ~1.5 MB/frame (~90 MB/s at 60 Hz).
    const VERTEX_BUDGET: usize = 48_000;
    const TRIANGLE_BUDGET: usize = 64_000;
}
