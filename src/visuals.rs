//! The visualization styles.
//!
//! # Adding a new visual
//!
//! 1. Write a struct holding whatever cross-frame state it needs.
//! 2. `impl Visualizer` — push triangles via the `MeshBuilder` helpers.
//! 3. Add one line to [`all`].
//!
//! That's the whole contract; nothing in `main.rs` needs to change.

use crate::render::{FrameData, MeshBuilder, Visualizer};
use audio_processor::NUM_BANDS;
use std::collections::VecDeque;
use std::f32::consts::TAU;

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

/// Cyan -> green -> yellow -> red across the frequency axis, carried over from
/// the original terminal renderer.
fn band_color(t: f32, alpha: f32) -> [f32; 4] {
    if t < 0.5 {
        let k = t * 2.0;
        [k, 1.0, 1.0 - k * 0.8, alpha]
    } else {
        let k = (t - 0.5) * 2.0;
        [1.0, 1.0 - k, 0.0, alpha]
    }
}

// ---------------------------------------------------------------------------
// Bars — mirrored spectrum bars with falling peak markers.
// ---------------------------------------------------------------------------

pub struct Bars {
    peaks: Vec<f32>,
}

impl Bars {
    pub fn new() -> Self {
        Self {
            peaks: vec![0.0; NUM_BANDS],
        }
    }
}

impl Visualizer for Bars {
    fn name(&self) -> &'static str {
        "Bars"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        let bar_width = 2.0 / NUM_BANDS as f32;

        for i in 0..NUM_BANDS {
            let level = frame.bands[i].clamp(0.0, 1.0);
            let h = level * 0.9;
            let x0 = -1.0 + i as f32 * bar_width;
            let x1 = x0 + bar_width * 0.82; // gap between bars
            let t = i as f32 / (NUM_BANDS - 1) as f32;

            mesh.rect(x0, -h, x1, h, band_color(t, 0.92));

            // Peak marker: snaps up instantly, sinks slowly, so transients stay
            // readable after the bar itself has dropped away.
            self.peaks[i] = self.peaks[i].max(level) - 0.006;
            self.peaks[i] = self.peaks[i].clamp(0.0, 1.0);
            let p = self.peaks[i] * 0.9;
            mesh.rect(x0, p, x1, p + 0.012, [1.0, 1.0, 1.0, 0.85]);
            mesh.rect(x0, -p - 0.012, x1, -p, [1.0, 1.0, 1.0, 0.85]);
        }
    }
}

// ---------------------------------------------------------------------------
// Ridge Bed — stacked spectrum history, Joy Division style.
// ---------------------------------------------------------------------------

pub struct RidgeBed {
    history: VecDeque<Vec<f32>>,
    frame_counter: u32,
}

impl RidgeBed {
    const ROWS: usize = 30;
    /// Push a history row every N frames so rows differ visibly instead of
    /// looking like duplicates of a 60fps-smoothed spectrum.
    const PUSH_EVERY: u32 = 3;

    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(Self::ROWS),
            frame_counter: 0,
        }
    }
}

impl Visualizer for RidgeBed {
    fn name(&self) -> &'static str {
        "Ridge Bed"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        if self.frame_counter % Self::PUSH_EVERY == 0 {
            if self.history.len() == Self::ROWS {
                self.history.pop_front();
            }
            self.history.push_back(frame.bands.to_vec());
        }

        let rows = self.history.len();
        if rows == 0 {
            return;
        }

        const TOP: f32 = 0.80;
        const SPAN: f32 = 1.55;
        const AMP: f32 = 0.30;
        const FLOOR: f32 = -1.6;

        let spacing = SPAN / Self::ROWS as f32;
        let opaque_bg = [frame.background[0], frame.background[1], frame.background[2], 1.0];

        // Oldest first: each newer row's opaque fill paints over the row behind
        // it, which is what produces the layered occlusion of the original.
        for (row, levels) in self.history.iter().enumerate() {
            let age = row as f32 / (rows.max(2) - 1) as f32; // 0 = oldest, 1 = newest
            let baseline = TOP - row as f32 * spacing;
            // Older rows sit slightly narrower, hinting at depth.
            let squeeze = 0.72 + 0.28 * age;

            let pts: Vec<[f32; 2]> = (0..NUM_BANDS)
                .map(|b| {
                    let x = (-1.0 + 2.0 * b as f32 / (NUM_BANDS - 1) as f32) * squeeze;
                    [x, baseline + levels[b].clamp(0.0, 1.0) * AMP]
                })
                .collect();

            mesh.fill_under(&pts, FLOOR, opaque_bg);

            // Cool blue in the distance, warming to white at the front.
            let color = [
                0.35 + 0.65 * age,
                0.55 + 0.45 * age,
                1.0,
                0.35 + 0.65 * age,
            ];
            let thickness = 0.004 + 0.004 * age;
            if age > 0.92 {
                mesh.glow_polyline(&pts, thickness, color, 3);
            } else {
                mesh.polyline(&pts, thickness, color);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spectrum Area — one filled silhouette with a glowing crest.
// ---------------------------------------------------------------------------

pub struct SpectrumArea;

impl SpectrumArea {
    pub fn new() -> Self {
        Self
    }
}

impl Visualizer for SpectrumArea {
    fn name(&self) -> &'static str {
        "Spectrum Area"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        const BASE: f32 = -0.75;

        let pts: Vec<[f32; 2]> = (0..NUM_BANDS)
            .map(|b| {
                let x = -1.0 + 2.0 * b as f32 / (NUM_BANDS - 1) as f32;
                [x, BASE + frame.bands[b].clamp(0.0, 1.0) * 1.5]
            })
            .collect();

        // Body of the silhouette, tinted by frequency.
        for (i, w) in pts.windows(2).enumerate() {
            let t = i as f32 / (NUM_BANDS - 1) as f32;
            let c = band_color(t, 0.30);
            mesh.quad(w[0], w[1], [w[1][0], BASE], [w[0][0], BASE], c);
        }

        // Mirrored reflection under the baseline, squashed and faded.
        let reflection: Vec<[f32; 2]> = pts
            .iter()
            .map(|p| [p[0], BASE - (p[1] - BASE) * 0.35])
            .collect();
        for (i, w) in reflection.windows(2).enumerate() {
            let t = i as f32 / (NUM_BANDS - 1) as f32;
            let c = band_color(t, 0.10);
            mesh.quad(w[0], w[1], [w[1][0], BASE], [w[0][0], BASE], c);
        }

        mesh.glow_polyline(&pts, 0.008, [1.0, 1.0, 1.0, 0.95], 3);
        mesh.polyline_with(&pts, 0.004, |t| band_color(t, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Radial Burst — spectrum wrapped around a bass-driven ring.
// ---------------------------------------------------------------------------

pub struct RadialBurst {
    rotation: f32,
}

impl RadialBurst {
    pub fn new() -> Self {
        Self { rotation: 0.0 }
    }
}

impl Visualizer for RadialBurst {
    fn name(&self) -> &'static str {
        "Radial Burst"
    }

    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder) {
        self.rotation += 0.0015 + frame.bass * 0.004;

        let center = [0.0, 0.0];
        let inner = 0.20 + frame.bass * 0.10;
        let half_step = TAU / NUM_BANDS as f32 * 0.36;

        // Spokes are mirrored across the vertical axis so the ring reads as
        // symmetric: bass at top, treble sweeping down both sides.
        for i in 0..NUM_BANDS {
            let level = frame.bands[i].clamp(0.0, 1.0);
            let outer = inner + level * 0.62;
            let t = i as f32 / (NUM_BANDS - 1) as f32;
            let color = band_color(t, 0.9);

            for dir in [1.0f32, -1.0] {
                let a = self.rotation + dir * (t * TAU * 0.5);
                let (a0, a1) = (a - half_step, a + half_step);
                mesh.quad(
                    mesh.polar(center, inner, a0),
                    mesh.polar(center, outer, a0),
                    mesh.polar(center, outer, a1),
                    mesh.polar(center, inner, a1),
                    color,
                );
            }
        }

        let pulse = 0.55 + frame.bass * 0.45;
        mesh.ring(center, inner - 0.02, 0.006, [1.0, 1.0, 1.0, pulse], 96);
    }
}

// ---------------------------------------------------------------------------
// Oscilloscope — triggered time-domain trace.
// ---------------------------------------------------------------------------

pub struct Oscilloscope;

impl Oscilloscope {
    pub fn new() -> Self {
        Self
    }

    /// Find a rising zero crossing to anchor the trace.
    ///
    /// Without this the waveform slides sideways every frame, since the capture
    /// window has no relationship to the signal's period.
    fn trigger(wave: &[f32], span: usize) -> usize {
        let search = wave.len().saturating_sub(span);
        for i in 1..search {
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

        let pts: Vec<[f32; 2]> = slice
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let x = -1.0 + 2.0 * i as f32 / (span - 1) as f32;
                [x, (s * gain).clamp(-0.92, 0.92)]
            })
            .collect();

        mesh.rect(-1.0, -0.0015, 1.0, 0.0015, [1.0, 1.0, 1.0, 0.12]);

        let hot = frame.bass.clamp(0.0, 1.0);
        let color = [0.35 + 0.65 * hot, 1.0 - 0.35 * hot, 0.85, 0.95];
        mesh.glow_polyline(&pts, 0.006, color, 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_processor::FFT_SIZE;

    /// Drive every visualizer over a run of frames and check what it emits.
    ///
    /// Guards the things a compile can't: out-of-range slicing, and NaN/inf
    /// coordinates leaking into the vertex buffer (which render as invisible or
    /// screen-filling garbage rather than an error).
    fn exercise(signal: impl Fn(usize, usize) -> f32, aspect: f32) {
        let mut mesh = MeshBuilder::new();

        for mut vis in all() {
            let mut produced = 0usize;

            for frame_idx in 0..12 {
                let bands: Vec<f32> = (0..NUM_BANDS).map(|b| signal(frame_idx, b)).collect();
                let waveform: Vec<f32> = (0..FFT_SIZE)
                    .map(|i| (i as f32 * 0.05 + frame_idx as f32).sin() * 0.5)
                    .collect();

                let frame = FrameData {
                    bands: &bands,
                    left_bands: &bands,
                    right_bands: &bands,
                    waveform: &waveform,
                    bass: 0.5,
                    rms: 0.2,
                    time: frame_idx as f32 / 60.0,
                    background: [0.02, 0.02, 0.05],
                };

                mesh.begin(aspect);
                vis.draw(&frame, &mut mesh);

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
                produced += mesh.vertices().len();
            }

            assert!(produced > 0, "{} never emitted any geometry", vis.name());
        }
    }

    #[test]
    fn visualizers_handle_normal_audio() {
        exercise(|f, b| (b as f32 * 0.2 + f as f32 * 0.3).sin() * 0.5 + 0.5, 16.0 / 9.0);
    }

    #[test]
    fn visualizers_handle_silence() {
        exercise(|_, _| 0.0, 16.0 / 9.0);
    }

    /// Levels are clamped downstream, but a visual shouldn't blow up if the
    /// smoothing ever hands it something outside 0..1.
    #[test]
    fn visualizers_handle_out_of_range_levels() {
        exercise(|_, b| if b % 2 == 0 { 1.8 } else { -0.4 }, 1.0);
    }

    #[test]
    fn visualizers_handle_extreme_aspect() {
        exercise(|_, b| b as f32 / NUM_BANDS as f32, 0.15);
    }
}
