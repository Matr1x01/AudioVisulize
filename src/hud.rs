//! Debug HUD — frame pacing, the audio→render handoff, and geometry counters.
//!
//! Everything here is diagnostic scaffolding, not shipping code. It draws with
//! the same `MeshBuilder` the visuals use so it costs one extra chunk of the
//! single draw call and needs no new pipeline.
//!
//! # Why these metrics
//!
//! Mean frame time is not the interesting number in this app — the GPU is
//! nearly idle and the CPU has ~14 ms of headroom. What matters is the *phase*
//! relationship between two unlocked clocks: the analysis thread producing
//! spectra and the swapchain presenting frames. Comparing their **rates** hides
//! the problem (60.0 vs 61.0 Hz looks healthy). Comparing spectrum **age** at
//! draw time exposes it as a sawtooth that wraps once per beat period.
//!
//! So the headline metrics are `AGE` and `DUP`, not `CPU` and `GPU`.

use crate::render::MeshBuilder;
use audio_processor::AudioProbe;
use std::fmt::Write as _;
use std::sync::atomic::Ordering;
use std::time::Instant;

const FONT_W: usize = 5;
const FONT_H: usize = 7;
/// One blank column/row between cells so glyphs don't touch.
const CELL_W: usize = FONT_W + 1;
const CELL_H: usize = FONT_H + 2;

/// How many frames of history the rolling stats and strip charts keep.
/// 240 frames = 4 s at 60 Hz, long enough to contain several beat periods.
const HISTORY: usize = 240;

// ---------------------------------------------------------------------------
// Bitmap font — ASCII 32..=95 (space through underscore), 5x7.
// Lowercase folds to uppercase. Bit 4 (0b10000) is the leftmost pixel.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const GLYPHS: [[u8; FONT_H]; 64] = [
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00], // ' '
    [0x04,0x04,0x04,0x04,0x04,0x00,0x04], // !
    [0x0A,0x0A,0x00,0x00,0x00,0x00,0x00], // "
    [0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A], // #
    [0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04], // $
    [0x19,0x1A,0x02,0x04,0x08,0x0B,0x13], // %
    [0x08,0x14,0x14,0x08,0x15,0x12,0x0D], // &
    [0x04,0x04,0x00,0x00,0x00,0x00,0x00], // '
    [0x02,0x04,0x08,0x08,0x08,0x04,0x02], // (
    [0x08,0x04,0x02,0x02,0x02,0x04,0x08], // )
    [0x00,0x15,0x0E,0x1F,0x0E,0x15,0x00], // *
    [0x00,0x04,0x04,0x1F,0x04,0x04,0x00], // +
    [0x00,0x00,0x00,0x00,0x0C,0x0C,0x18], // ,
    [0x00,0x00,0x00,0x1F,0x00,0x00,0x00], // -
    [0x00,0x00,0x00,0x00,0x00,0x0C,0x0C], // .
    [0x01,0x02,0x02,0x04,0x08,0x08,0x10], // /
    [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E], // 0
    [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E], // 1
    [0x0E,0x11,0x01,0x02,0x04,0x08,0x1F], // 2
    [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E], // 3
    [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02], // 4
    [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E], // 5
    [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E], // 6
    [0x1F,0x01,0x02,0x04,0x08,0x08,0x08], // 7
    [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E], // 8
    [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C], // 9
    [0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00], // :
    [0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x18], // ;
    [0x02,0x04,0x08,0x10,0x08,0x04,0x02], // <
    [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00], // =
    [0x08,0x04,0x02,0x01,0x02,0x04,0x08], // >
    [0x0E,0x11,0x01,0x02,0x04,0x00,0x04], // ?
    [0x0E,0x11,0x17,0x15,0x17,0x10,0x0E], // @
    [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11], // A
    [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E], // B
    [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E], // C
    [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C], // D
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F], // E
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10], // F
    [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F], // G
    [0x11,0x11,0x11,0x1F,0x11,0x11,0x11], // H
    [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E], // I
    [0x07,0x02,0x02,0x02,0x02,0x12,0x0C], // J
    [0x11,0x12,0x14,0x18,0x14,0x12,0x11], // K
    [0x10,0x10,0x10,0x10,0x10,0x10,0x1F], // L
    [0x11,0x1B,0x15,0x15,0x11,0x11,0x11], // M
    [0x11,0x11,0x19,0x15,0x13,0x11,0x11], // N
    [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E], // O
    [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10], // P
    [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D], // Q
    [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11], // R
    [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E], // S
    [0x1F,0x04,0x04,0x04,0x04,0x04,0x04], // T
    [0x11,0x11,0x11,0x11,0x11,0x11,0x0E], // U
    [0x11,0x11,0x11,0x11,0x11,0x0A,0x04], // V
    [0x11,0x11,0x11,0x15,0x15,0x1B,0x11], // W
    [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11], // X
    [0x11,0x11,0x0A,0x04,0x04,0x04,0x04], // Y
    [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F], // Z
    [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E], // [
    [0x10,0x08,0x08,0x04,0x02,0x02,0x01], // \
    [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E], // ]
    [0x04,0x0A,0x11,0x00,0x00,0x00,0x00], // ^
    [0x00,0x00,0x00,0x00,0x00,0x00,0x1F], // _
];

fn glyph(c: char) -> &'static [u8; FONT_H] {
    let upper = if c.is_ascii_lowercase() {
        (c as u8) - 32
    } else {
        c as u8
    };
    let idx = (upper as usize).wrapping_sub(32);
    GLYPHS.get(idx).unwrap_or(&GLYPHS[0])
}

// ---------------------------------------------------------------------------
// Rolling window
// ---------------------------------------------------------------------------

/// Fixed-capacity circular buffer of f32 samples with order-statistic queries.
///
/// Preallocated at construction: the HUD is where we complain about per-frame
/// allocation, so it does not allocate per frame either.
struct Ring {
    buf: Vec<f32>,
    idx: usize,
    len: usize,
}

impl Ring {
    fn new() -> Self {
        Self {
            buf: vec![0.0; HISTORY],
            idx: 0,
            len: 0,
        }
    }

    fn push(&mut self, v: f32) {
        self.buf[self.idx] = v;
        self.idx = (self.idx + 1) % HISTORY;
        self.len = (self.len + 1).min(HISTORY);
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Samples oldest-first.
    fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let start = if self.len == HISTORY { self.idx } else { 0 };
        (0..self.len).map(move |i| self.buf[(start + i) % HISTORY])
    }

    fn mean(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.iter().sum::<f32>() / self.len as f32
    }

    fn min(&self) -> f32 {
        self.iter().fold(f32::INFINITY, f32::min)
    }

    fn max(&self) -> f32 {
        self.iter().fold(f32::NEG_INFINITY, f32::max)
    }

    /// `p` in 0..1. Sorts into the caller's scratch buffer to avoid allocating.
    fn percentile(&self, p: f32, scratch: &mut Vec<f32>) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        scratch.clear();
        scratch.extend(self.iter());
        scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((scratch.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
        scratch[i]
    }
}

// ---------------------------------------------------------------------------
// Per-frame sample
// ---------------------------------------------------------------------------

/// Everything the HUD records about one frame. Filled in by `main`.
#[derive(Default)]
pub struct FrameSample {
    /// Building the mesh: audio snapshot + visualizer draw. Excludes vsync block.
    pub cpu_build_ms: f32,
    /// The `mesh.vertices().to_vec()` copy, measured on its own.
    pub cpu_copy_ms: f32,
    /// Encode + submit. Excludes vsync block.
    pub cpu_submit_ms: f32,
    /// Time blocked inside `get_current_texture()` — this is vsync, not work.
    pub wait_ms: f32,
    /// Time the render thread blocked on the metrics mutex.
    pub lock_wait_ms: f32,
    /// Wall clock between successive `present()` calls.
    pub present_dt_ms: f32,
    /// Age of the spectrum this frame drew, at the moment it was snapshotted.
    pub spectrum_age_ms: f32,
    /// Monotonic id of the spectrum drawn; equal to last frame's means a dup.
    pub spectrum_seq: u64,
    /// Input frames the capture callback delivered between the last two
    /// analysis passes — the actual, unquantized FFT hop.
    pub hop_frames: u64,
    pub scene_verts: usize,
    /// Index count, which is what actually determines the triangle count now
    /// that the mesh is drawn indexed.
    pub scene_indices: usize,
    /// Recorded per frame so a dump can never be misattributed to the wrong
    /// visual after a mid-run switch.
    pub visualizer: &'static str,
}

/// Settings the HUD reports but does not own.
pub struct StaticInfo {
    pub sample_count: u32,
    pub fft_size: usize,
    pub num_bands: usize,
    pub present_mode: &'static str,
    pub available_modes: String,
    pub gpu_timing: bool,
    /// Spline subdivisions per band. 1 = straight segments between band points.
    pub subdivisions: u32,
    /// Whether strokes carry an alpha-feathered fringe.
    pub feathered: bool,
    pub visualizer: &'static str,
    pub palette: &'static str,
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

pub struct Hud {
    pub enabled: bool,

    present_dt: Ring,
    cpu_total: Ring,
    cpu_copy: Ring,
    submit: Ring,
    wait: Ring,
    lock_wait: Ring,
    gpu: Ring,
    age: Ring,
    hop: Ring,
    band_min: Ring,
    band_med: Ring,
    band_max: Ring,
    band_p90: Ring,

    frames: u64,
    dup_spectra: u64,
    dropped_frames: u64,
    last_seq: u64,
    new_spectra: u64,
    scene_verts: usize,
    scene_indices: usize,
    hud_verts: usize,
    hud_indices: usize,
    visual: &'static str,

    started: Instant,
    scratch: Vec<f32>,
    text: String,
}

impl Hud {
    pub fn new() -> Self {
        Self {
            enabled: false,
            present_dt: Ring::new(),
            cpu_total: Ring::new(),
            cpu_copy: Ring::new(),
            submit: Ring::new(),
            wait: Ring::new(),
            lock_wait: Ring::new(),
            gpu: Ring::new(),
            age: Ring::new(),
            hop: Ring::new(),
            band_min: Ring::new(),
            band_med: Ring::new(),
            band_max: Ring::new(),
            band_p90: Ring::new(),
            frames: 0,
            dup_spectra: 0,
            dropped_frames: 0,
            last_seq: u64::MAX,
            new_spectra: 0,
            scene_verts: 0,
            scene_indices: 0,
            hud_verts: 0,
            hud_indices: 0,
            visual: "?",
            started: Instant::now(),
            scratch: Vec::with_capacity(HISTORY),
            text: String::with_capacity(64),
        }
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    pub fn reset(&mut self) {
        let enabled = self.enabled;
        *self = Self::new();
        self.enabled = enabled;
    }

    /// Record one frame. Called unconditionally so the stats are already warm
    /// when the HUD is first shown.
    pub fn record(&mut self, s: &FrameSample) {
        self.frames += 1;
        self.scene_verts = s.scene_verts;
        self.scene_indices = s.scene_indices;
        if !s.visualizer.is_empty() {
            self.visual = s.visualizer;
        }

        self.cpu_total
            .push(s.cpu_build_ms + s.cpu_copy_ms + s.cpu_submit_ms);
        self.cpu_copy.push(s.cpu_copy_ms);
        self.submit.push(s.cpu_submit_ms);
        self.wait.push(s.wait_ms);
        self.lock_wait.push(s.lock_wait_ms);
        self.age.push(s.spectrum_age_ms);

        // present_dt is zero on the very first frame; don't pollute the stats.
        if s.present_dt_ms > 0.0 {
            self.present_dt.push(s.present_dt_ms);
        }

        if s.spectrum_seq == self.last_seq {
            // The analysis thread produced nothing new since the previous
            // frame, so this frame redraws the same spectrum. On a Fifo
            // swapchain this is invisible in the frame-time graph but is
            // exactly what reads as a stutter.
            self.dup_spectra += 1;
        } else {
            self.new_spectra += 1;
            self.hop.push(s.hop_frames as f32);
            self.last_seq = s.spectrum_seq;
        }
    }

    /// Estimated display refresh period, in ms.
    ///
    /// Taken as the median present-to-present delta rather than the mean: a
    /// handful of long frames would drag the mean and make every subsequent
    /// dropped-frame test too lenient.
    fn refresh_ms(&mut self) -> f32 {
        if self.present_dt.is_empty() {
            return 16.667;
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        let m = self.present_dt.percentile(0.5, &mut scratch);
        self.scratch = scratch;
        if m > 0.1 {
            m
        } else {
            16.667
        }
    }

    /// Frames the display showed that we did not produce, counted over the
    /// rolling window: any present gap of ~2x the refresh period or longer.
    fn count_dropped(&mut self) -> u64 {
        let period = self.refresh_ms();
        let mut dropped = 0u64;
        for dt in self.present_dt.iter() {
            let slots = (dt / period).round() as i64;
            if slots > 1 {
                dropped += (slots - 1) as u64;
            }
        }
        self.dropped_frames = dropped;
        dropped
    }

    /// Record the distribution of this frame's band levels.
    ///
    /// Dynamic range is the thing this diagnoses: if the minimum never
    /// approaches zero, the bottom of the display range is dead and every bar
    /// sits bunched near the top no matter what the visuals do.
    pub fn note_bands(&mut self, bands: &[f32], scratch: &mut Vec<f32>) {
        if bands.is_empty() {
            return;
        }
        scratch.clear();
        scratch.extend(bands.iter().copied().filter(|v| v.is_finite()));
        if scratch.is_empty() {
            return;
        }
        scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = scratch.len();
        self.band_min.push(scratch[0]);
        self.band_med.push(scratch[n / 2]);
        self.band_p90.push(scratch[(n * 9 / 10).min(n - 1)]);
        self.band_max.push(scratch[n - 1]);
    }

    pub fn push_gpu_ms(&mut self, ms: f32) {
        self.gpu.push(ms);
    }

    /// Set before `draw` so the panel reports this frame's count rather than
    /// the previous one's.
    pub fn note_scene_geometry(&mut self, verts: usize, indices: usize) {
        self.scene_verts = verts;
        self.scene_indices = indices;
    }

    /// Print a plain-text summary, including the audio-thread probes that are
    /// too verbose for the on-screen panel.
    pub fn dump(&mut self, probe: &AudioProbe) {
        let mut scratch = std::mem::take(&mut self.scratch);
        let refresh = self.refresh_ms();
        let dropped = self.count_dropped();
        let elapsed = self.started.elapsed().as_secs_f32().max(1e-3);
        let hop_rate = self.new_spectra as f32 / elapsed;
        let frame_rate = self.frames as f32 / elapsed;
        let beat = (hop_rate - frame_rate).abs();

        println!("\n=== HUD SUMMARY  {}  ({elapsed:.1} s) ===", self.visual);
        println!(
            "cpu        mean {:6.2} ms  p99 {:6.2} ms   (vertex copy {:.3} ms)",
            self.cpu_total.mean(),
            self.cpu_total.percentile(0.99, &mut scratch),
            self.cpu_copy.mean()
        );
        if self.gpu.is_empty() {
            println!("gpu        unavailable (no TIMESTAMP_QUERY)");
        } else {
            println!(
                "gpu        mean {:6.2} ms  max {:6.2} ms",
                self.gpu.mean(),
                self.gpu.max()
            );
        }
        println!(
            "present    mean {:6.2} ms  1% low {:6.2} ms  refresh {:.2} ms",
            self.present_dt.mean(),
            self.present_dt.percentile(0.99, &mut scratch),
            refresh
        );
        println!(
            "submit     {:6.2} ms  (buffer upload + encode)",
            self.submit.mean()
        );
        println!(
            "vsync block {:5.2} ms  render lock wait {:.3} ms  dropped {}",
            self.wait.mean(),
            self.lock_wait.mean(),
            dropped
        );
        println!(
            "rates      hop {:.2} Hz vs render {:.2} Hz  ->  beat {:.3} Hz (period {:.2} s)",
            hop_rate,
            frame_rate,
            beat,
            if beat > 1e-3 { 1.0 / beat } else { f32::INFINITY }
        );
        println!(
            "duplicates {} of {} frames ({:.2}%) drew a spectrum already drawn",
            self.dup_spectra,
            self.frames,
            self.dup_spectra as f32 / self.frames.max(1) as f32 * 100.0
        );
        println!(
            "age        min {:.1}  mean {:.1}  max {:.1} ms",
            self.age.min(),
            self.age.mean(),
            self.age.max()
        );
        println!(
            "hop        mean {:.0} smp  range {:.0}..{:.0}  overlap {:.1}%",
            self.hop.mean(),
            self.hop.min(),
            self.hop.max(),
            (1.0 - self.hop.mean() / 1024.0) * 100.0
        );
        println!(
            "audio cb   {} calls  max {} us  max lock wait {} us",
            probe.callbacks.load(Ordering::Relaxed),
            probe.callback_max_us.load(Ordering::Relaxed),
            probe.callback_lock_max_us.load(Ordering::Relaxed),
        );
        let passes = probe.analysis_passes.load(Ordering::Relaxed).max(1);
        let stale = probe.stale_analyses.load(Ordering::Relaxed);
        let gapped = probe.dropped_windows.load(Ordering::Relaxed);
        println!(
            "analysis   {passes} passes: {stale} stale ({:.0}%), {gapped} with zero overlap ({:.0}%)",
            stale as f32 / passes as f32 * 100.0,
            gapped as f32 / passes as f32 * 100.0,
        );
        println!(
            "           distinct spectra {:.1} Hz vs {} analysis passes/s",
            self.new_spectra as f32 / elapsed,
            (passes as f32 / elapsed).round() as i32,
        );
        println!(
            "bands      min {:.3}  median {:.3}  p90 {:.3}  max {:.3}   (usable span {:.3})",
            self.band_min.mean(),
            self.band_med.mean(),
            self.band_p90.mean(),
            self.band_max.mean(),
            self.band_max.mean() - self.band_min.mean(),
        );
        println!(
            "geometry   scene {} verts / {} tris   hud {} verts   upload {:.0} KB/frame\n",
            self.scene_verts,
            self.scene_indices / 3,
            self.hud_verts,
            ((self.scene_verts + self.hud_verts) as f32 * 24.0
                + (self.scene_indices + self.hud_indices) as f32 * 4.0)
                / 1024.0
        );
        self.scratch = scratch;
    }

    // -- drawing ------------------------------------------------------------

    /// Pixel-space rect -> NDC. Snapping to integer pixels keeps the glyphs
    /// crisp; a half-pixel offset would let MSAA soften every stem.
    fn px_rect(
        mesh: &mut MeshBuilder,
        fb: (f32, f32),
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    ) {
        let x0 = (x.round() / fb.0) * 2.0 - 1.0;
        let x1 = ((x + w).round() / fb.0) * 2.0 - 1.0;
        let y0 = 1.0 - ((y + h).round() / fb.1) * 2.0;
        let y1 = 1.0 - (y.round() / fb.1) * 2.0;
        mesh.rect(x0, y0, x1, y1, color);
    }

    /// Draw a string, merging horizontal runs of lit pixels into single rects.
    ///
    /// A naive one-rect-per-pixel emitter costs ~35 quads per glyph; merging
    /// runs brings a typical glyph down to ~10, which keeps the overlay's own
    /// vertex count from swamping the scene count it is trying to report.
    fn text_at(
        mesh: &mut MeshBuilder,
        fb: (f32, f32),
        x: f32,
        y: f32,
        scale: f32,
        s: &str,
        color: [f32; 4],
    ) {
        for (ci, c) in s.chars().enumerate() {
            let g = glyph(c);
            let gx = x + ci as f32 * CELL_W as f32 * scale;
            for (row, bits) in g.iter().enumerate() {
                let mut col = 0usize;
                while col < FONT_W {
                    if bits & (0x10 >> col) == 0 {
                        col += 1;
                        continue;
                    }
                    let start = col;
                    while col < FONT_W && bits & (0x10 >> col) != 0 {
                        col += 1;
                    }
                    Self::px_rect(
                        mesh,
                        fb,
                        gx + start as f32 * scale,
                        y + row as f32 * scale,
                        (col - start) as f32 * scale,
                        scale,
                        color,
                    );
                }
            }
        }
    }

    /// Strip chart of the last `HISTORY` samples, newest at the right.
    ///
    /// `warn` draws a horizontal reference line — for present deltas that is
    /// the refresh period, for spectrum age it is one analysis period.
    fn strip(
        mesh: &mut MeshBuilder,
        fb: (f32, f32),
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        ring: &Ring,
        scale_max: f32,
        color: [f32; 4],
        warn: Option<f32>,
    ) {
        Self::px_rect(mesh, fb, x, y, w, h, [0.10, 0.11, 0.16, 0.85]);

        if let Some(v) = warn {
            let ly = y + h - (v / scale_max).clamp(0.0, 1.0) * h;
            Self::px_rect(mesh, fb, x, ly, w, 1.0, [0.9, 0.75, 0.2, 0.55]);
        }

        if ring.is_empty() {
            return;
        }
        let n = ring.len.min(HISTORY);
        let bar_w = (w / n as f32).max(1.0);
        for (i, v) in ring.iter().enumerate() {
            let bh = ((v / scale_max).clamp(0.0, 1.0) * h).max(1.0);
            Self::px_rect(
                mesh,
                fb,
                x + i as f32 * (w / n as f32),
                y + h - bh,
                bar_w,
                bh,
                color,
            );
        }
    }

    pub fn draw(&mut self, mesh: &mut MeshBuilder, fb_w: u32, fb_h: u32, info: &StaticInfo) {
        if !self.enabled {
            self.hud_verts = 0;
            self.hud_indices = 0;
            return;
        }
        let before = mesh.vertices().len();
        let before_idx = mesh.indices().len();
        let fb = (fb_w.max(1) as f32, fb_h.max(1) as f32);

        // Scale the overlay with the framebuffer so it stays legible when the
        // window is small, but never below 1 px per font pixel.
        let scale = (fb.1 / 620.0).clamp(1.0, 3.0).floor();
        let pad = 8.0 * scale;
        let line_h = CELL_H as f32 * scale;
        let panel_w = 46.0 * CELL_W as f32 * scale + pad * 2.0;
        let panel_h = 27.0 * line_h + pad * 3.0;

        Self::px_rect(mesh, fb, pad, pad, panel_w, panel_h, [0.0, 0.0, 0.0, 0.78]);
        Self::px_rect(mesh, fb, pad, pad, panel_w, 1.0, [0.4, 0.9, 1.0, 0.5]);

        let x = pad * 2.0;
        let mut y = pad * 2.0;

        let white = [1.0, 1.0, 1.0, 0.95];
        let dim = [0.62, 0.68, 0.78, 0.9];
        let good = [0.45, 0.95, 0.6, 0.95];
        let bad = [1.0, 0.42, 0.38, 1.0];
        let hot = [1.0, 0.78, 0.3, 1.0];

        let refresh = self.refresh_ms();
        let dropped = self.count_dropped();

        let mut scratch = std::mem::take(&mut self.scratch);

        macro_rules! line {
            ($color:expr, $($arg:tt)*) => {{
                self.text.clear();
                let _ = write!(self.text, $($arg)*);
                Self::text_at(mesh, fb, x, y, scale, &self.text, $color);
                y += line_h;
            }};
        }

        line!(white, "AUDIO VISUALIZER HUD    F1 HIDE  F2 RESET");
        line!(dim, "VISUAL {}  PALETTE {}", info.visualizer, info.palette);
        y += line_h * 0.4;

        // -- frame timing ---------------------------------------------------
        line!(hot, "-- FRAME TIMING ---------------------------");
        let cpu_mean = self.cpu_total.mean();
        let cpu_p99 = self.cpu_total.percentile(0.99, &mut scratch);
        line!(
            if cpu_p99 > refresh { bad } else { good },
            "CPU     {:>6.2} MS   P99 {:>6.2}  BUDGET {:>5.2}",
            cpu_mean,
            cpu_p99,
            refresh
        );
        if info.gpu_timing && !self.gpu.is_empty() {
            line!(
                good,
                "GPU     {:>6.2} MS   MAX {:>6.2}",
                self.gpu.mean(),
                self.gpu.max()
            );
        } else {
            line!(dim, "GPU        N/A     (NO TIMESTAMP QUERY)");
        }
        line!(
            dim,
            "  OF WHICH VERTEX COPY {:>6.3} MS",
            self.cpu_copy.mean()
        );
        line!(
            if self.submit.mean() > 3.0 { bad } else { dim },
            "SUBMIT  {:>6.2} MS   UPLOAD+ENCODE",
            self.submit.mean()
        );
        line!(dim, "VSYNC BLOCK {:>6.2} MS  LOCK {:>6.3} MS", self.wait.mean(), self.lock_wait.mean());

        // -- presentation ---------------------------------------------------
        y += line_h * 0.4;
        line!(hot, "-- PRESENTATION ---------------------------");
        let p_mean = self.present_dt.mean();
        let p_low = self.present_dt.percentile(0.99, &mut scratch);
        line!(
            white,
            "PRESENT {:>6.2} MS   1% LOW {:>6.2} MS",
            p_mean,
            p_low
        );
        line!(
            white,
            "FPS     {:>6.1}      1% LOW {:>6.1}",
            if p_mean > 0.0 { 1000.0 / p_mean } else { 0.0 },
            if p_low > 0.0 { 1000.0 / p_low } else { 0.0 }
        );
        line!(
            if dropped > 0 { bad } else { good },
            "REFRESH {:>6.2} MS   DROPPED {:>4}/{:<4}",
            refresh,
            dropped,
            self.present_dt.len
        );
        line!(dim, "MODE {} AVAIL {}", info.present_mode, info.available_modes);

        Self::strip(
            mesh,
            fb,
            x,
            y,
            panel_w - pad * 2.0,
            line_h * 2.0,
            &self.present_dt,
            refresh * 2.5,
            [0.45, 0.85, 1.0, 0.95],
            Some(refresh),
        );
        y += line_h * 2.6;

        // -- audio handoff: the headline ------------------------------------
        line!(hot, "-- AUDIO -> RENDER HANDOFF ----------------");
        let elapsed = self.started.elapsed().as_secs_f32().max(1e-3);
        let hop_rate = self.new_spectra as f32 / elapsed;
        let frame_rate = self.frames as f32 / elapsed;
        line!(
            white,
            "HOP RATE {:>6.2} HZ  RENDER {:>6.2} HZ",
            hop_rate,
            frame_rate
        );
        // The beat period is what turns two healthy-looking rates into a
        // once-a-second visible hitch: 1 / |f_audio - f_render|.
        let beat = (hop_rate - frame_rate).abs();
        line!(
            if beat > 0.05 { bad } else { good },
            "BEAT {:>6.2} HZ  PERIOD {:>7.2} S",
            beat,
            if beat > 1e-3 { 1.0 / beat } else { 999.99 }
        );
        let dup_pct = if self.frames > 0 {
            self.dup_spectra as f32 / self.frames as f32 * 100.0
        } else {
            0.0
        };
        line!(
            if self.dup_spectra > 0 { bad } else { good },
            "DUP SPECTRA {:>7} ({:>5.2}% OF FRAMES)",
            self.dup_spectra,
            dup_pct
        );
        line!(
            white,
            "AGE MIN {:>5.1} AVG {:>5.1} MAX {:>5.1} MS",
            self.age.min(),
            self.age.mean(),
            self.age.max()
        );
        let hop_mean = self.hop.mean();
        let overlap = (1.0 - hop_mean / info.fft_size as f32) * 100.0;
        line!(
            if self.hop.max() - self.hop.min() > 64.0 { bad } else { good },
            "HOP {:>5.0} SMP ({:>4.0}..{:>4.0}) OVERLAP {:>4.1}%",
            hop_mean,
            self.hop.min(),
            self.hop.max(),
            overlap
        );

        // The sawtooth in this strip IS the judder. A correct pipeline shows a
        // flat line here; a free-running producer shows a ramp that wraps once
        // per beat period.
        Self::strip(
            mesh,
            fb,
            x,
            y,
            panel_w - pad * 2.0,
            line_h * 2.0,
            &self.age,
            (self.age.max() * 1.2).max(20.0),
            [1.0, 0.55, 0.35, 0.95],
            Some(refresh),
        );
        y += line_h * 2.6;

        // -- geometry -------------------------------------------------------
        line!(hot, "-- GEOMETRY & SETTINGS --------------------");
        line!(
            white,
            "SCENE VERTS {:>7}  TRIS {:>7}",
            self.scene_verts,
            self.scene_indices / 3
        );
        line!(
            dim,
            "HUD  VERTS  {:>7}  UPLOAD {:>5.0} KB/F",
            self.hud_verts,
            ((self.scene_verts + self.hud_verts) as f32 * 24.0
                + (self.scene_indices + self.hud_indices) as f32 * 4.0)
                / 1024.0
        );
        line!(
            dim,
            "MSAA {}X  SUBDIV {}  FEATHER {}",
            info.sample_count,
            info.subdivisions,
            if info.feathered { "ON" } else { "OFF" }
        );
        line!(dim, "FFT {}  BANDS {}", info.fft_size, info.num_bands);
        let _ = y; // last row: the macro's cursor advance is intentionally unused

        self.scratch = scratch;
        self.hud_verts = mesh.vertices().len() - before;
        self.hud_indices = mesh.indices().len() - before_idx;
    }
}

impl Default for Hud {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_orders_oldest_first_after_wrap() {
        let mut r = Ring::new();
        for i in 0..HISTORY + 10 {
            r.push(i as f32);
        }
        let v: Vec<f32> = r.iter().collect();
        assert_eq!(v.len(), HISTORY);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[HISTORY - 1], (HISTORY + 9) as f32);
    }

    #[test]
    fn percentile_matches_sorted_position() {
        let mut r = Ring::new();
        let mut scratch = Vec::new();
        for i in 0..100 {
            r.push(i as f32);
        }
        assert_eq!(r.percentile(0.0, &mut scratch), 0.0);
        assert_eq!(r.percentile(1.0, &mut scratch), 99.0);
        assert_eq!(r.percentile(0.5, &mut scratch), 50.0);
    }

    #[test]
    fn duplicate_spectra_are_counted_only_on_repeat_seq() {
        let mut hud = Hud::new();
        for seq in [1u64, 1, 2, 2, 2, 3] {
            hud.record(&FrameSample {
                spectrum_seq: seq,
                present_dt_ms: 16.6,
                ..Default::default()
            });
        }
        // 6 frames carrying 3 distinct spectra => 3 duplicates.
        assert_eq!(hud.frames, 6);
        assert_eq!(hud.dup_spectra, 3);
        assert_eq!(hud.new_spectra, 3);
    }

    #[test]
    fn dropped_frames_counted_from_present_gaps() {
        let mut hud = Hud::new();
        for _ in 0..50 {
            hud.record(&FrameSample {
                present_dt_ms: 16.667,
                spectrum_seq: 0,
                ..Default::default()
            });
        }
        assert_eq!(hud.count_dropped(), 0);
        // One 3-refresh gap = two frames the display showed that we missed.
        hud.record(&FrameSample {
            present_dt_ms: 50.0,
            spectrum_seq: 1,
            ..Default::default()
        });
        assert_eq!(hud.count_dropped(), 2);
    }

    #[test]
    fn glyphs_cover_the_printable_range_used_by_the_hud() {
        for c in ' '..='_' {
            let g = glyph(c);
            assert!(g.iter().all(|row| row & 0xE0 == 0), "glyph {c:?} overflows 5 columns");
        }
        // Lowercase folds onto uppercase rather than rendering blank.
        assert_eq!(glyph('a'), glyph('A'));
    }
}
